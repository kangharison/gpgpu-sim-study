/*
 * [한국어 설명] SM(Streaming Multiprocessor) 파이프라인 전체 구현 (shader.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim에서 GPU의 SM(Streaming Multiprocessor, NVIDIA 용어)을 모델링하는
 * 핵심 구현 파일이다. "shader core"라는 GPGPU-Sim 내부 명칭으로 SM을 지칭하며,
 * shader_core_ctx 클래스가 하나의 SM 전체 상태(파이프라인 레지스터, warp 배열, 스코어보드,
 * 스케줄러, 캐시 등)를 보유한다. 타이밍 모델(gpgpu-sim/) 계층에서 사이클-레벨
 * 시뮬레이션의 핵심 루프를 담당한다. 사이클마다 shader_core_ctx::cycle()이 호출되어
 * Writeback → Execute → Read Operands → Issue → Decode → Fetch 역방향 순으로
 * 파이프라인 스테이지를 진행시킨다. warp 스케줄러(LRR/GTO/Two-Level), 오퍼랜드 컬렉터,
 * 메모리 파이프라인(ldst_unit), 배리어/동기화 모두 이 파일에서 구현된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *     → simt_core_cluster::core_cycle() [이 파일]
 *         → shader_core_ctx::cycle() [이 파일] — SM별 1 사이클
 *             ├─ writeback()   : EX_WB 스테이지 — 결과를 레지스터 파일에 씀, 스코어보드 해제
 *             ├─ execute()     : 각 FU(sp_unit/sfu/ldst_unit 등)의 cycle() 호출
 *             ├─ read_operands(): opndcoll_rfu_t::step() — 레지스터 뱅크 충돌 해소
 *             ├─ issue()       : 스케줄러 cycle() — warp 선택 및 파이프라인 레지스터에 발행
 *             └─ decode()/fetch(): I-cache 접근 및 I-buffer 채움
 *   simt_core_cluster::icnt_cycle() : NoC 응답 처리
 *   simt_core_cluster::issue_block2core() : CTA를 SM에 할당
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — 단일 스레드 시뮬레이터 루프.
 * GPU 병렬성은 시뮬레이션 모델로만 표현되며, 실제 병렬 스레드가 아님.
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   shader.h           — shader_core_ctx, scheduler_unit, ldst_unit, opndcoll_rfu_t 선언
 *   gpu-sim.{cc,h}     — gpgpu_sim 최상위 시뮬레이터, gpu_sim_cycle 카운터
 *   gpu-cache.{cc,h}   — l1_cache, read_only_cache, tex_cache (L1I/L1D/L1C/L1T)
 *   mem_fetch.{cc,h}   — mem_fetch 패킷 (메모리 요청 객체)
 *   scoreboard.{cc,h}  — RAW 해저드 감지 Scoreboard 클래스
 *   addrdec.{cc,h}     — 메모리 주소 디코딩 (서브파티션 ID 계산)
 *   icnt_wrapper.{cc,h}— NoC(intersim2) 푸시/팝 인터페이스
 *   abstract_hardware_model.{cc,h} — warp_inst_t, simt_stack, kernel_info_t, core_t
 *   cuda-sim/          — PTX 기능 시뮬레이션 (execute_warp_inst_t, ptx_fetch_inst)
 *
 * 데이터 흐름:
 *   fetch() → m_inst_fetch_buffer → decode() → warp.ibuffer → issue() →
 *   m_pipeline_reg[ID_OC_*] → opndcoll (read_operands) →
 *   m_pipeline_reg[OC_EX_*] → execute() / FU.cycle() →
 *   m_pipeline_reg[EX_WB]   → writeback()
 *
 *   메모리 요청: ldst_unit → m_icnt → NoC → DRAM → NoC → icnt_cycle() →
 *   m_response_fifo → accept_ldst_unit_response() → ldst_unit.fill()
 *
 * === 주요 함수/구조체 요약 ===
 *   shader_core_ctx::cycle()       — SM 1-사이클 전체 파이프라인 구동 (핵심 진입점)
 *   shader_core_ctx::fetch()       — I-cache 접근, warp 라운드로빈 순환, I-miss 처리
 *   shader_core_ctx::decode()      — I-buffer에 최대 2개 명령어 디코딩 적재
 *   shader_core_ctx::issue()       — 모든 스케줄러 cycle() 호출 (공정 라운드로빈)
 *   scheduler_unit::cycle()        — warp 정렬 후 scoreboard 통과 명령어 발행
 *   shader_core_ctx::execute()     — 모든 FU cycle() 호출, result bus 예약
 *   shader_core_ctx::writeback()   — EX_WB에서 결과 커밋, 스코어보드/파이프라인 카운터 해제
 *   ldst_unit::cycle()             — 메모리 파이프라인 (shared/const/tex/global)
 *   ldst_unit::memory_cycle()      — L1D 또는 NoC(bypass)로 메모리 요청 전달
 *   opndcoll_rfu_t::allocate_reads()— wavefront 알고리즘으로 레지스터 뱅크 충돌 해소
 *   barrier_set_t::warp_reaches_barrier() — __syncthreads() 구현 핵심
 *   simt_core_cluster::core_cycle()— 클러스터 내 모든 SM의 cycle() 순차 호출
 *   simt_core_cluster::icnt_cycle()— NoC 응답 FIFO 처리, SM에 메모리 응답 전달
 */
// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh, Vijay Kandiah, Nikos
// Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The University of
// British Columbia, Northwestern University, Purdue University All rights
// reserved.
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

#include "shader.h"          // [한국어] SM 파이프라인 전체 선언: shader_core_ctx, scheduler_unit, ldst_unit, opndcoll_rfu_t 등
#include <float.h>           // [한국어] FLT_MAX 등 부동소수점 한계값 — 스케줄러 비교 로직에 사용
#include <limits.h>          // [한국어] INT_MAX 등 정수 한계값
#include <string.h>          // [한국어] snprintf, memset 등 C 문자열/메모리 함수
#include "../../libcuda/gpgpu_context.h" // [한국어] gpgpu_context — PTX 기능 시뮬레이션 전역 컨텍스트 (ptx_fetch_inst 등 접근)
#include "../cuda-sim/cuda-sim.h"        // [한국어] PTX 기능 시뮬레이션 인터페이스 — execute_warp_inst_t, ptx_thread_info
#include "../cuda-sim/ptx-stats.h"       // [한국어] PTX 레벨 통계 카운터 (명령어 타입별 카운트)
#include "../cuda-sim/ptx_sim.h"         // [한국어] ptx_thread_info — 각 스레드의 레지스터/PC 상태 (기능 모델)
#include "../statwrapper.h"              // [한국어] 통계 수집 래퍼 함수
#include "addrdec.h"         // [한국어] 메모리 주소 디코딩: 선형 주소 → 채널/뱅크/행/열/서브파티션 분해
#include "dram.h"            // [한국어] DRAM 타이밍 모델 — 메모리 요청 최종 수신자
#include "gpu-misc.h"        // [한국어] LOGB2, gs_min2 등 유틸리티 함수
#include "gpu-sim.h"         // [한국어] gpgpu_sim 최상위 시뮬레이터 — gpu_sim_cycle, kernel 관리
#include "icnt_wrapper.h"    // [한국어] NoC(intersim2) 인터페이스: icnt_push, icnt_pop, icnt_has_buffer
#include "mem_fetch.h"       // [한국어] mem_fetch 메모리 요청 패킷 — SM ↔ 메모리 계층 통신 단위
#include "mem_latency_stat.h"// [한국어] 메모리 레이턴시 통계 (memlatstat_read_done)
#include "shader_trace.h"    // [한국어] SCHED_DPRINTF, SHADER_DPRINTF 등 디버그 트레이스 매크로
#include "stat-tool.h"       // [한국어] shader_cache_access_log, shader_CTA_count_unlog 등 통계 도구
#include "traffic_breakdown.h" // [한국어] NoC 트래픽 세분화 통계 (m_outgoing_traffic_stats)
#include "visualizer.h"      // [한국어] AerialVision 가시화 도구 지원 (gzprintf 기반 시각화)

// [한국어] MSHR(Miss Status Holding Register) 응답을 writeback보다 먼저 처리하는 정책.
// 1로 설정 시 응답 FIFO에서 MSHR 응답이 writeback 요청보다 우선 처리됨.
#define PRIORITIZE_MSHR_OVER_WB 1
// [한국어] 두 값 중 더 큰 값을 반환하는 매크로. 인라인 함수 대신 매크로로 정의해 타입 독립성 확보.
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
// [한국어] 두 값 중 더 작은 값을 반환하는 매크로. swl_scheduler의 warp 수 제한에 사용.
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/*
 * [한국어]
 * shader_core_mem_fetch_allocator::alloc (간단 오버로드) — 간단한 메모리 요청 패킷 생성
 *
 * @addr:     접근할 메모리 주소
 * @type:     접근 유형 (INST_ACC_R, GLOBAL_ACC_R 등 — mem_access_type enum)
 * @size:     접근 바이트 수
 * @wr:       쓰기 여부 (true=write, false=read)
 * @cycle:    현재 시뮬레이션 사이클 (mem_fetch 타임스탬프 기록용)
 * @streamID: 이 요청이 속한 CUDA 스트림 ID (다중 스트림 지원)
 * @return:   생성된 mem_fetch 포인터 (호출자가 소유권을 가짐)
 *
 * I-cache 미스 처리 등 warp_id가 필요 없는 단순 메모리 요청을 생성한다.
 * active_mask/byte_mask/sector_mask 없는 단순 버전으로, warp_id를 -1로 설정.
 * 호출 체인: shader_core_ctx::fetch() → [이 함수] (I-cache miss 시 mf 직접 생성)
 *            단, I-cache는 현재 이 함수를 직접 쓰지 않고 new mem_fetch()를 직접 호출.
 */
mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, unsigned size, bool wr,
    unsigned long long cycle, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx); // [한국어] mem_access_t 생성: 주소/크기/방향 캡슐화
  mem_fetch *mf = new mem_fetch(
      access, NULL, streamID, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1, // [한국어] warp_id=-1: 특정 warp에 귀속되지 않는 요청
      m_core_id, m_cluster_id, m_memory_config, cycle);                      // [한국어] 이 SM의 core_id와 cluster_id로 경로 식별
  return mf; // [한국어] 호출자가 적절히 delete하거나 m_icnt->push()로 소유권 이전해야 함
}

/*
 * [한국어]
 * shader_core_mem_fetch_allocator::alloc (전체 오버로드) — warp별 세밀한 메모리 요청 패킷 생성
 *
 * @addr:        접근 주소
 * @type:        접근 유형 (GLOBAL_ACC_R, LOCAL_ACC_W 등)
 * @active_mask: 이 요청에 참여하는 스레드 마스크 (32비트, warp 내 활성 스레드)
 * @byte_mask:   캐시 라인 내 접근된 바이트 마스크 (sector cache 지원용)
 * @sector_mask: 캐시 라인 내 접근된 섹터 마스크 (sector-assoc MSHR용)
 * @size:        접근 바이트 수
 * @wr:          쓰기 여부
 * @cycle:       현재 사이클
 * @wid:         요청을 발생시킨 warp ID
 * @sid:         shader core ID (SM ID)
 * @tpc:         TPC(Texture Processing Cluster) ID
 * @original_mf: write-allocate 시 원본 mf (NULL이면 최초 요청)
 * @streamID:    CUDA 스트림 ID
 * @return:      생성된 mem_fetch 포인터
 *
 * 글로벌/로컬 메모리 접근 시 coalescing 후 생성되는 실제 메모리 요청 패킷을 만든다.
 * active_mask와 sector_mask는 L2 캐시/DRAM이 어떤 바이트가 실제로 필요한지 파악할 때 사용.
 * 호출 체인: ldst_unit::memory_cycle() → m_mf_allocator->alloc() → [이 함수] → icnt_push()
 */
mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask, // [한국어] 전체 마스크 정보를 가진 mem_access_t 생성 — sector-associative MSHR 지원
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, streamID, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid, // [한국어] 쓰기면 데이터 포함 패킷 크기, 읽기면 컨트롤 헤더만
      m_core_id, m_cluster_id, m_memory_config, cycle, original_mf);           // [한국어] original_mf: write-allocate의 경우 writeback 체인 추적
  return mf;
}
/////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * shader_core_ctx::get_regs_written — 명령어가 쓰는 목적 레지스터 목록 반환
 *
 * @fvt:    조사할 명령어 (inst_t 참조 — warp_inst_t의 부모 클래스)
 * @return: 이 명령어가 쓰는 레지스터 번호 목록 (최대 MAX_REG_OPERANDS개)
 *
 * 레지스터 파일 writeback 시 어느 뱅크에 쓸지 판단하거나, 오퍼랜드 컬렉터가
 * writeback 시 뱅크 충돌을 처리할 때 사용된다.
 * arch_reg.dst[op] 배열에서 유효 레지스터(>=0)만 수집한다.
 * -1은 해당 슬롯에 목적 레지스터가 없음을 의미.
 *
 * 호출 체인: opndcoll_rfu_t::writeback() → [이 함수] → 뱅크 할당
 */
std::list<unsigned> shader_core_ctx::get_regs_written(const inst_t &fvt) const {
  std::list<unsigned> result; // [한국어] 반환할 쓰기 레지스터 번호 목록
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) { // [한국어] MAX_REG_OPERANDS: 최대 목적 레지스터 수 (보통 1~2개, 술어 레지스터 포함 시 2개)
    int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    // [한국어] reg_num >= 0이면 유효 레지스터: -1은 슬롯 미사용을 의미
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num); // [한국어] 유효한 목적 레지스터 번호를 결과에 추가
  }
  return result;
}

/*
 * [한국어]
 * exec_shader_core_ctx::create_shd_warp — warp 배열 초기화
 *
 * @return: void
 *
 * 이 SM이 수용할 수 있는 최대 warp 수(max_warps_per_shader)만큼 shd_warp_t 객체를
 * 동적 할당하여 m_warp 배열에 채운다. shd_warp_t는 warp의 하드웨어 상태
 * (IBuffer, PC, 완료된 스레드 수, 배리어 대기 여부 등)를 관리한다.
 * SM 생성자에서 호출되며, 이후 init_warps()에서 각 warp에 실제 CTA가 할당된다.
 *
 * 호출 체인: exec_shader_core_ctx 생성자 → [이 함수]
 */
void exec_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader); // [한국어] max_warps_per_shader: gpgpusim.config의 -gpgpu_shader_core_pipeline 설정에서 결정 (보통 48~64)
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new shd_warp_t(this, m_config->warp_size); // [한국어] 각 warp_id k에 대해 shd_warp_t 할당. warp_size는 보통 32.
  }
}

/*
 * [한국어]
 * shader_core_ctx::create_front_pipeline — 프론트엔드 파이프라인 초기화
 *
 * @return: void
 *
 * SM의 프론트엔드 파이프라인(Fetch→Decode→Issue 경로)에 필요한 모든 자료구조를 생성한다:
 *   1. 파이프라인 레지스터 배열(m_pipeline_reg) 구성:
 *      - 표준 스테이지: IF→ID, ID→OC(SP/DP/SFU/INT/TC/MEM), OC→EX, EX→WB
 *      - 특수 유닛(specialized_unit)의 ID_OC와 OC_EX 레지스터 추가
 *   2. sub_core_model일 경우 스케줄러 수와 파이프라인 레지스터 폭 일치 검증
 *   3. m_threadState 배열 초기화 (스레드 상태 추적용)
 *   4. 메모리 인터페이스 생성 (SST/perfect_mem/일반 모드)
 *   5. L1I(명령어 캐시) 생성
 *
 * sub_core_model: 각 스케줄러가 별도의 파이프라인 슬롯을 가지는 Volta+ 아키텍처 모델.
 * 이 모드에서는 스케줄러 수 == 파이프라인 레지스터 폭이어야 한다.
 *
 * 호출 체인: shader_core_ctx 생성자 (create_front_pipeline) → [이 함수]
 */
void shader_core_ctx::create_front_pipeline() {
  // pipeline_stages is the sum of normal pipeline stages and specialized_unit
  // stages * 2 (for ID and EX)
  // [한국어] 총 파이프라인 스테이지 수 = 표준 스테이지 + 특수 유닛 × 2(ID_OC + OC_EX)
  unsigned total_pipeline_stages =
      N_PIPELINE_STAGES + m_config->m_specialized_unit.size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages); // [한국어] vector 재할당 방지 — 포인터 안정성 보장
  // [한국어] 표준 파이프라인 스테이지 등록: IF_ID, ID_OC_SP, ID_OC_SFU, ..., EX_WB 등
  for (int j = 0; j < N_PIPELINE_STAGES; j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->pipe_widths[j], pipeline_stage_name_decode[j])); // [한국어] pipe_widths[j]: 해당 스테이지의 병렬 슬롯 수 (보통 1, sub_core_model에서는 스케줄러 수)
  }
  // [한국어] 특수 유닛(specialized_unit) ID→OC 파이프라인 레지스터 추가 (텐서코어 등 사용자 정의 유닛)
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1; // [한국어] 나중에 issue_warp()가 올바른 파이프라인 레지스터를 찾기 위해 인덱스 저장
    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]); // [한국어] 스케줄러가 특수 유닛으로 dispatch할 포트 목록에 추가
  }
  // [한국어] 특수 유닛 OC→EX 파이프라인 레지스터 추가
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1; // [한국어] 오퍼랜드 컬렉터가 특수 유닛 실행 스테이지로 dispatch하는 포트 인덱스 저장
  }

  if (m_config->sub_core_model) {
    // in subcore model, each scheduler should has its own issue register, so
    // ensure num scheduler = reg width
    // [한국어] sub_core_model: Volta+처럼 각 warp 스케줄러가 독립적인 파이프라인 슬롯을 가질 때.
    // 스케줄러 수 == 각 파이프라인 레지스터의 슬롯 수여야 함 — 불일치 시 assert 실패.
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SP].get_size()); // [한국어] SP(단정밀도 부동소수점) 파이프라인 레지스터 폭 검증
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SFU].get_size()); // [한국어] SFU(특수 함수 유닛) 파이프라인 레지스터 폭 검증
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_MEM].get_size()); // [한국어] MEM(LD/ST 유닛) 파이프라인 레지스터 폭 검증
    if (m_config->gpgpu_tensor_core_avail) // [한국어] 텐서코어 구성 시에만 검증 — Volta+ A100 등
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_TENSOR_CORE].get_size());
    if (m_config->gpgpu_num_dp_units > 0) // [한국어] 배정밀도(DP) 유닛 존재 시 검증
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_DP].get_size());
    if (m_config->gpgpu_num_int_units > 0) // [한국어] 정수 연산 유닛 존재 시 검증 (Volta+의 INT 파이프라인)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_INT].get_size());
    for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
      if (m_config->m_specialized_unit[j].num_units > 0) // [한국어] 특수 유닛 수 > 0일 때만 검증
        assert(m_config->gpgpu_num_sched_per_core ==
               m_config->m_specialized_unit[j].id_oc_spec_reg_width);
    }
  }

  // [한국어] m_threadState: SM 내 모든 스레드의 상태(CTA 소속, 활성 여부, 명령어 카운트)를 추적하는 배열.
  // calloc으로 할당하여 0으로 초기화.
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);

  m_not_completed = 0;       // [한국어] 아직 완료되지 않은 스레드 수 초기화 — 0이면 SM이 비어 있음
  m_active_threads.reset();  // [한국어] 활성 스레드 비트마스크 전체 클리어
  m_n_active_cta = 0;        // [한국어] 현재 SM에서 실행 중인 CTA 수 초기화
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0; // [한국어] CTA별 남은 활성 스레드 수 초기화
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;               // [한국어] 기능 시뮬레이션 스레드 포인터 초기화 (실제 할당은 CTA 배치 시)
    m_threadState[i].m_cta_id = -1;  // [한국어] -1: 아직 CTA에 할당되지 않음
    m_threadState[i].m_active = false; // [한국어] 활성 상태 아님 초기화
  }

  // m_icnt = new shader_memory_interface(this,cluster);
  // [한국어] 메모리 인터페이스 생성: SST 통합 모드, 완벽 메모리 모드, 일반 NoC 모드 중 선택
  if (m_memory_config->SST_mode) {
    // [한국어] SST_mode: Structural Simulation Toolkit 연동 — 실제 메모리 계층과 co-simulation
    m_icnt = new sst_memory_interface(
        this, static_cast<sst_simt_core_cluster *>(m_cluster));
  } else if (m_config->gpgpu_perfect_mem) {
    // [한국어] perfect_mem: 이상적 메모리 (레이턴시 0) — 메모리 계층 제거 후 연산 성능만 측정할 때
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    // [한국어] 일반 모드: intersim2 NoC를 통해 메모리 계층과 통신
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  // [한국어] mem_fetch 팩토리 객체 생성 — SM ID와 TPC ID를 캡슐화하여 각 mf에 소속 SM 정보 자동 부여
  m_mem_fetch_allocator =
      new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);

  // fetch
  m_last_warp_fetched = 0; // [한국어] 라운드로빈 fetch 시작 warp_id 초기화 — 첫 사이클에 warp 0부터 탐색

#define STRSIZE 1024   // [한국어] 캐시 이름 문자열 최대 길이
  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sid); // [한국어] L1I 캐시 이름: "L1I_000" 형식 — 통계/디버그 출력에 사용
  // [한국어] L1I(명령어 캐시) 생성: read_only_cache (명령어는 수정되지 않으므로 읽기 전용).
  // m_L1I->access()가 fetch()에서 호출되어 I-cache hit/miss를 결정.
  m_L1I = new read_only_cache(name, m_config->m_L1I_config, m_sid,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);
}

/*
 * [한국어]
 * shader_core_ctx::create_schedulers — warp 스케줄러 및 스코어보드 초기화
 *
 * @return: void
 *
 * gpgpusim.config의 -gpgpu_scheduler 옵션에서 스케줄러 종류를 파싱하고,
 * gpgpu_num_sched_per_core 수만큼 스케줄러 인스턴스를 생성한다.
 * 지원 스케줄러:
 *   lrr         — LRR(Loose Round Robin): 모든 warp를 순환하며 발행 가능한 첫 warp 선택
 *   gto         — GTO(Greedy-Then-Oldest): 직전 발행 warp를 먼저 시도, 실패하면 가장 오래된 warp
 *   two_level_active — 2레벨 스케줄러: 활성 풀/대기 풀 분리, 긴 레이턴시 연산 warp를 대기로 강등
 *   rrr         — RRR(Round Robin Restart): 성공적으로 발행 시 다음 warp로 이동
 *   oldest_first — 가장 오래된 dynamic_warp_id를 가진 warp 우선
 *   warp_limiting — GTO 기반이지만 활성 warp 수를 제한
 *
 * 각 스케줄러는 gpgpu_num_sched_per_core 수로 생성되며(보통 2~4개),
 * m_warp를 스케줄러에 균등 분배한다. 이는 각 스케줄러가 독립적인 warp 집합을 담당.
 * Scoreboard는 모든 스케줄러가 공유 (warp당 하나의 scoreboard).
 *
 * 호출 체인: shader_core_ctx 생성 → create_schedulers() → [이 함수]
 */
void shader_core_ctx::create_schedulers() {
  // [한국어] Scoreboard 생성: warp당 RAW(Read-After-Write) 해저드 추적.
  // reserveRegisters() / releaseRegisters() / checkCollision() 로 사용됨.
  m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);

  // scedulers
  // must currently occur after all inputs have been initialized.
  // [한국어] gpgpusim.config의 -gpgpu_scheduler 옵션 문자열 파싱. 예: "gto", "lrr", "two_level_active 3 3"
  std::string sched_config = m_config->gpgpu_scheduler_string;
  // [한국어] 문자열에서 스케줄러 종류를 감지 — 순서 중요: two_level_active를 gto보다 먼저 검사
  const concrete_scheduler scheduler =
      sched_config.find("lrr") != std::string::npos ? CONCRETE_SCHEDULER_LRR
      : sched_config.find("two_level_active") != std::string::npos
          ? CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE
      : sched_config.find("gto") != std::string::npos ? CONCRETE_SCHEDULER_GTO
      : sched_config.find("rrr") != std::string::npos ? CONCRETE_SCHEDULER_RRR
      : sched_config.find("old") != std::string::npos
          ? CONCRETE_SCHEDULER_OLDEST_FIRST
      : sched_config.find("warp_limiting") != std::string::npos
          ? CONCRETE_SCHEDULER_WARP_LIMITING
          : NUM_CONCRETE_SCHEDULERS;
  assert(scheduler != NUM_CONCRETE_SCHEDULERS); // [한국어] 알 수 없는 스케줄러 문자열이면 시뮬레이션 중단

  // [한국어] gpgpu_num_sched_per_core 수만큼 스케줄러를 생성 (보통 2~4).
  // 각 스케줄러는 ID(i)를 가지며, sub_core_model에서는 해당 ID의 파이프라인 슬롯만 사용.
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
    switch (scheduler) {
      case CONCRETE_SCHEDULER_LRR: // [한국어] LRR: Loose Round Robin — 모든 supervised warp 순환, 발행 가능한 첫 warp 선택
        schedulers.push_back(new lrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE: // [한국어] Two-Level: 활성 warp풀(즉각 발행 가능)과 대기 풀 분리, 긴 레이턴시 대기 warp는 대기 풀로 강등
        schedulers.push_back(new two_level_active_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      case CONCRETE_SCHEDULER_GTO: // [한국어] GTO: Greedy-Then-Oldest — 직전 성공 warp 유지, 실패 시 가장 오래된(dynamic_warp_id 최소) warp 선택
        schedulers.push_back(new gto_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_RRR: // [한국어] RRR: Round Robin Restart — 한 사이클에 한 warp만 고려, 성공 시 다음으로 이동
        schedulers.push_back(new rrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_OLDEST_FIRST: // [한국어] Oldest First: dynamic_warp_id 기준 가장 오래된 warp를 항상 먼저 선택
        schedulers.push_back(new oldest_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_WARP_LIMITING: // [한국어] Warp Limiting: GTO 기반이지만 활성 warp 수를 m_num_warps_to_limit으로 제한 — 리소스 경쟁 억제
        schedulers.push_back(new swl_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      default:
        abort(); // [한국어] 알 수 없는 스케줄러 타입 — 도달 불가 (위에서 assert로 차단됨)
    };
  }

  // [한국어] SM의 모든 warp를 스케줄러에 라운드로빈으로 분배.
  // 예: 4개 스케줄러에 48 warp → 스케줄러 0: 0,4,8,..., 스케줄러 1: 1,5,9,...
  for (unsigned i = 0; i < m_warp.size(); i++) {
    // distribute i's evenly though schedulers;
    schedulers[i % m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(
        i); // [한국어] 스케줄러에 자신이 관리할 warp_id를 등록
  }
  // [한국어] 모든 warp 등록이 끝났음을 스케줄러에 통보 — 내부 순환 이터레이터를 초기화
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; ++i) {
    schedulers[i]->done_adding_supervised_warps();
  }
}

/*
 * [한국어]
 * shader_core_ctx::create_exec_pipeline — 실행 파이프라인(오퍼랜드 컬렉터, FU 목록) 초기화
 *
 * @return: void
 *
 * SM의 실행 백엔드를 구성한다:
 *   1. 오퍼랜드 컬렉터(opndcoll_rfu_t) 구성:
 *      - CU(Collector Unit) 집합 등록: SP/DP/SFU/TC/INT/MEM/GEN
 *      - 포트(입력 파이프라인 레지스터 → 출력 파이프라인 레지스터 → CU 집합) 연결
 *      - enable_specialized_operand_collector: 유닛 타입별 전용 CU 집합 사용 여부
 *   2. 함수 유닛(FU) 목록 구성:
 *      - sp_unit: 단정밀도 부동소수점 ALU
 *      - dp_unit: 배정밀도 부동소수점 ALU
 *      - int_unit: 정수 연산 유닛 (Volta+ 전용)
 *      - sfu: 특수 함수 유닛 (sin/cos/sqrt/rcp 등)
 *      - tensor_core: Tensor Core (행렬 연산)
 *      - specialized_unit: 사용자 정의 유닛
 *      - ldst_unit: 로드/스토어 유닛 (메모리 접근 전담)
 *   3. result_bus 초기화: EX_WB 스테이지 폭만큼 비트셋 생성 (ALU 레이턴시 충돌 방지)
 *
 * 호출 체인: shader_core_ctx 생성자 → create_exec_pipeline()
 */
void shader_core_ctx::create_exec_pipeline() {
  // op collector configuration
  // [한국어] 오퍼랜드 컬렉터 CU 집합 ID. 각 FU 타입이 사용하는 CU 집합을 구분.
  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };

  opndcoll_rfu_t::port_vector_t in_ports;   // [한국어] 입력 포트 벡터: ID→OC 파이프라인 레지스터 포인터 목록
  opndcoll_rfu_t::port_vector_t out_ports;  // [한국어] 출력 포트 벡터: OC→EX 파이프라인 레지스터 포인터 목록
  opndcoll_rfu_t::uint_vector_t cu_sets;    // [한국어] 이 포트가 사용하는 CU 집합 ID 목록

  // configure generic collectors
  // [한국어] GEN_CUS: 모든 유닛 타입이 공유하는 범용 CU 집합. 모든 명령어 타입을 처리 가능.
  // gpgpu_operand_collector_num_units_gen: 범용 CU 수 (gpgpusim.config에서 설정)
  m_operand_collector.add_cu_set(
      GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen,
      m_config->gpgpu_operand_collector_num_out_ports_gen);

  // [한국어] 범용 포트 구성: gpgpu_operand_collector_num_in_ports_gen 수만큼 포트를 생성.
  // 각 포트는 (SP, SFU, MEM, 선택적으로 TC/DP/INT/특수 유닛)의 ID_OC→OC_EX 쌍을 묶음.
  // 하나의 포트에 여러 파이프라인 레지스터를 연결하면 오퍼랜드 컬렉터가 한 번에 여러 스테이지를 서비스.
  for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen;
       i++) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);    // [한국어] SP(단정밀도) ID→OC 입력
    in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);   // [한국어] SFU ID→OC 입력
    in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);   // [한국어] MEM(LD/ST) ID→OC 입력
    out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);   // [한국어] SP OC→EX 출력
    out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);  // [한국어] SFU OC→EX 출력
    out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);  // [한국어] MEM OC→EX 출력
    if (m_config->gpgpu_tensor_core_avail) { // [한국어] 텐서코어 구성 시에만 TC 포트 추가
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
    }
    if (m_config->gpgpu_num_dp_units > 0) { // [한국어] DP 유닛 존재 시 DP 포트 추가 (Fermi는 DP를 SFU에서 처리하므로 0)
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
    }
    if (m_config->gpgpu_num_int_units > 0) { // [한국어] 정수 유닛 존재 시 (Volta+ 아키텍처)
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
    }
    if (m_config->m_specialized_unit.size() > 0) { // [한국어] 사용자 정의 특수 유닛 포트 추가
      for (unsigned j = 0; j < m_config->m_specialized_unit.size(); ++j) {
        in_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].ID_OC_SPEC_ID]);
        out_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].OC_EX_SPEC_ID]);
      }
    }
    cu_sets.push_back((unsigned)GEN_CUS); // [한국어] 이 포트는 범용 CU 집합을 사용
    m_operand_collector.add_port(in_ports, out_ports, cu_sets); // [한국어] 오퍼랜드 컬렉터에 포트 등록
    in_ports.clear(), out_ports.clear(), cu_sets.clear(); // [한국어] 다음 포트를 위해 벡터 초기화
  }

  // [한국어] enable_specialized_operand_collector: 유닛 타입별 전용 CU 집합 사용 여부.
  // 활성화 시 SP/DP/SFU/TC/INT/MEM 각각 전용 CU를 가져 레지스터 뱅크 충돌 처리가 더 세밀해짐.
  // 비활성화 시 모든 유닛이 GEN_CUS 범용 CU 집합을 공유.
  if (m_config->enable_specialized_operand_collector) {
    m_operand_collector.add_cu_set(
        SP_CUS, m_config->gpgpu_operand_collector_num_units_sp, // [한국어] SP 전용 CU 집합 등록
        m_config->gpgpu_operand_collector_num_out_ports_sp);
    m_operand_collector.add_cu_set(
        DP_CUS, m_config->gpgpu_operand_collector_num_units_dp, // [한국어] DP 전용 CU 집합
        m_config->gpgpu_operand_collector_num_out_ports_dp);
    m_operand_collector.add_cu_set(
        TENSOR_CORE_CUS,
        m_config->gpgpu_operand_collector_num_units_tensor_core, // [한국어] 텐서코어 전용 CU 집합
        m_config->gpgpu_operand_collector_num_out_ports_tensor_core);
    m_operand_collector.add_cu_set(
        SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu, // [한국어] SFU 전용 CU 집합
        m_config->gpgpu_operand_collector_num_out_ports_sfu);
    m_operand_collector.add_cu_set(
        MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem, // [한국어] MEM(LD/ST) 전용 CU 집합
        m_config->gpgpu_operand_collector_num_out_ports_mem);
    m_operand_collector.add_cu_set(
        INT_CUS, m_config->gpgpu_operand_collector_num_units_int, // [한국어] INT(정수) 전용 CU 집합
        m_config->gpgpu_operand_collector_num_out_ports_int);

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_dp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
      cu_sets.push_back((unsigned)DP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
      cu_sets.push_back((unsigned)SFU_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0;
         i < m_config->gpgpu_operand_collector_num_in_ports_tensor_core; i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
      cu_sets.push_back((unsigned)TENSOR_CORE_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
      cu_sets.push_back((unsigned)MEM_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_int;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
      cu_sets.push_back((unsigned)INT_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
  }

  // [한국어] 오퍼랜드 컬렉터 초기화: 뱅크 수와 SM 포인터 설정.
  // m_arbiter가 이 시점에서 num_banks × num_collectors 크기의 요청 행렬을 초기화.
  m_operand_collector.init(m_config->gpgpu_num_reg_banks, this);

  // [한국어] 총 함수 유닛 수 계산: SP + DP + SFU + TC + INT + 특수 유닛들 + 1(ldst_unit)
  m_num_function_units =
      m_config->gpgpu_num_sp_units + m_config->gpgpu_num_dp_units +
      m_config->gpgpu_num_sfu_units + m_config->gpgpu_num_tensor_core_units +
      m_config->gpgpu_num_int_units + m_config->m_specialized_unit_num +
      1;  // sp_unit, sfu, dp, tensor, int, ldst_unit

  // [한국어] SP(단정밀도 FP) 유닛 생성. 각 유닛은 EX_WB 레지스터셋으로 결과를 전달.
  // dispatch_port: 스케줄러가 발행하는 ID_OC 레지스터 인덱스
  // issue_port: 오퍼랜드 컬렉터가 실행 유닛으로 전달하는 OC_EX 레지스터 인덱스
  for (unsigned k = 0; k < m_config->gpgpu_num_sp_units; k++) {
    m_fu.push_back(new sp_unit(&m_pipeline_reg[EX_WB], m_config, this, k)); // [한국어] sp_unit: FADD/FMUL/FFMA 등 단정밀도 연산 담당
    m_dispatch_port.push_back(ID_OC_SP);  // [한국어] 스케줄러 → 이 유닛의 입력 포트
    m_issue_port.push_back(OC_EX_SP);     // [한국어] 오퍼랜드 컬렉터 → 실행 유닛 포트
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_dp_units; k++) {
    m_fu.push_back(new dp_unit(&m_pipeline_reg[EX_WB], m_config, this, k)); // [한국어] dp_unit: 배정밀도(64-bit) 연산 (Fermi에서는 SFU 담당, Pascal+에서는 DP 유닛 전용)
    m_dispatch_port.push_back(ID_OC_DP);
    m_issue_port.push_back(OC_EX_DP);
  }
  for (unsigned k = 0; k < m_config->gpgpu_num_int_units; k++) {
    m_fu.push_back(new int_unit(&m_pipeline_reg[EX_WB], m_config, this, k)); // [한국어] int_unit: 정수 ALU (IADD/IMUL/ISCADD). Volta+ 아키텍처에서 SP와 분리됨.
    m_dispatch_port.push_back(ID_OC_INT);
    m_issue_port.push_back(OC_EX_INT);
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
    m_fu.push_back(new sfu(&m_pipeline_reg[EX_WB], m_config, this, k)); // [한국어] sfu: 특수 함수(sin/cos/rcp/sqrt/ex2/lg2) 및 DP 연산(Fermi). 긴 레이턴시(보통 4~20 사이클).
    m_dispatch_port.push_back(ID_OC_SFU);
    m_issue_port.push_back(OC_EX_SFU);
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_tensor_core_units; k++) {
    m_fu.push_back(new tensor_core(&m_pipeline_reg[EX_WB], m_config, this, k)); // [한국어] tensor_core: Volta+의 WMMA/HMMA 행렬 연산 유닛. 매우 높은 처리량.
    m_dispatch_port.push_back(ID_OC_TENSOR_CORE);
    m_issue_port.push_back(OC_EX_TENSOR_CORE);
  }

  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    for (unsigned k = 0; k < m_config->m_specialized_unit[j].num_units; k++) {
      // [한국어] specialized_unit: 연구자가 gpgpusim.config로 커스텀 정의한 유닛.
      // SPEC_UNIT_START_ID + j로 유닛 타입 구분.
      m_fu.push_back(new specialized_unit(
          &m_pipeline_reg[EX_WB], m_config, this, SPEC_UNIT_START_ID + j,
          m_config->m_specialized_unit[j].name,
          m_config->m_specialized_unit[j].latency, k));
      m_dispatch_port.push_back(m_config->m_specialized_unit[j].ID_OC_SPEC_ID);
      m_issue_port.push_back(m_config->m_specialized_unit[j].OC_EX_SPEC_ID);
    }
  }

  // [한국어] ldst_unit: 로드/스토어 유닛 — 공유 메모리, 상수 캐시, 텍스처 캐시, L1D, NoC 접근 담당.
  // m_fu에 마지막으로 추가되며, execute()에서 다른 FU와 동일하게 cycle()이 호출됨.
  m_ldst_unit = new ldst_unit(m_icnt, m_mem_fetch_allocator, this,
                              &m_operand_collector, m_scoreboard, m_config,
                              m_memory_config, m_stats, m_sid, m_tpc, m_gpu);
  m_fu.push_back(m_ldst_unit);       // [한국어] m_fu 목록에 ldst_unit 추가
  m_dispatch_port.push_back(ID_OC_MEM); // [한국어] MEM 명령어는 ID_OC_MEM을 통해 발행
  m_issue_port.push_back(OC_EX_MEM);    // [한국어] OC_EX_MEM을 통해 ldst_unit에 전달

  // [한국어] FU 수, dispatch/issue 포트 수가 일치하는지 검증
  assert(m_num_function_units == m_fu.size() and
         m_fu.size() == m_dispatch_port.size() and
         m_fu.size() == m_issue_port.size());

  // there are as many result buses as the width of the EX_WB stage
  // [한국어] result bus: 각 ALU FU가 완료 시 EX_WB 스테이지에 결과를 올리는 버스.
  // 같은 사이클에 여러 FU가 완료되면 result bus 충돌이 발생할 수 있으므로
  // MAX_ALU_LATENCY 크기의 비트셋으로 예약 상태를 추적.
  num_result_bus = m_config->pipe_widths[EX_WB]; // [한국어] EX_WB 스테이지 폭 = result bus 수
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>()); // [한국어] 각 result bus를 비트셋으로 표현 — 비트 k가 set이면 k 사이클 후 해당 버스 사용 예정
  }
}

/*
 * [한국어]
 * shader_core_ctx::shader_core_ctx — SM 컨텍스트 생성자
 *
 * @gpu:       최상위 gpgpu_sim 객체 포인터 (gpu_sim_cycle 접근, 커널 관리)
 * @cluster:   이 SM이 속한 simt_core_cluster 포인터
 * @shader_id: 전역 SM ID (m_sid). gpu-sim에서 SM을 고유하게 식별.
 * @tpc_id:    TPC(Texture Processing Cluster) ID (m_tpc). NoC 주소 지정에 사용.
 * @config:    SM 설정 (shader_core_config — gpgpusim.config에서 읽은 파라미터)
 * @mem_config: 메모리 시스템 설정 (memory_config)
 * @stats:     통계 수집 객체 (shader_core_stats)
 * @return:    (생성자)
 *
 * core_t 기반 클래스를 초기화하고(warp_size, n_thread_per_shader),
 * barrier_set_t(CTA 배리어 관리), m_active_warps, m_dynamic_warp_id 등 멤버 초기화.
 * AccelWattch 전력 시뮬레이션이 활성화된 경우 scaling_coeffs(전력 스케일링 계수)를 가져옴.
 * 동시 커널(concurrent kernels on SM) 지원을 위한 리소스 점유 변수도 초기화.
 *
 * 호출 체인: exec_shader_core_ctx 생성자 → shader_core_ctx 생성자 → [이 함수]
 *            exec_simt_core_cluster::create_shader_core_ctx() → exec_shader_core_ctx 생성자
 */
shader_core_ctx::shader_core_ctx(class gpgpu_sim *gpu,
                                 class simt_core_cluster *cluster,
                                 unsigned shader_id, unsigned tpc_id,
                                 const shader_core_config *config,
                                 const memory_config *mem_config,
                                 shader_core_stats *stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader), // [한국어] core_t 기반 초기화: GPU 포인터, warp 크기(32), 스레드 수(보통 2048)
      m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size), // [한국어] barrier_set_t 초기화: __syncthreads() 구현을 위한 warp 배리어 추적 구조
      m_active_warps(0),    // [한국어] 현재 활성(실행 중) warp 수 초기화 — 점유율 계산에 사용
      m_dynamic_warp_id(0) { // [한국어] 동적 warp ID 카운터: 새 CTA가 배치될 때마다 증가, 스케줄러 oldest-first 정책의 기준
  m_cluster = cluster;      // [한국어] 소속 클러스터 포인터 저장
  m_config = config;        // [한국어] SM 설정 포인터 저장 (파이프라인 폭, 캐시 설정 등)
  m_memory_config = mem_config; // [한국어] 메모리 시스템 설정 저장
  m_stats = stats;          // [한국어] 통계 수집 객체 저장
  // unsigned warp_size = config->warp_size;
  Issue_Prio = 0; // [한국어] 스케줄러 라운드로빈 우선순위 초기화: 사이클마다 증가하며 어느 스케줄러가 먼저 실행될지 결정

  m_sid = shader_id;  // [한국어] 전역 SM ID 저장 (통계, 캐시 이름, NoC 주소 등에 사용)
  m_tpc = tpc_id;     // [한국어] TPC ID 저장 (메모리 파티션 주소 계산에 사용)

  // [한국어] AccelWattch 전력 시뮬레이션 활성화 시 연산 유형별 전력 스케일링 계수 로드
  if (get_gpu()->get_config().g_power_simulation_enabled) {
    scaling_coeffs = get_gpu()->get_scaling_coeffs();
  }

  m_last_inst_gpu_sim_cycle = 0;     // [한국어] 마지막 명령어 완료 사이클 타임스탬프 초기화
  m_last_inst_gpu_tot_sim_cycle = 0; // [한국어] 누적 사이클 포함 타임스탬프 초기화

  // Jin: for concurrent kernels on a SM
  // [한국어] 동시 커널 지원: 하나의 SM에서 여러 커널의 CTA가 공존할 때 리소스 점유 추적
  m_occupied_n_threads = 0;        // [한국어] 현재 점유된 스레드 수
  m_occupied_shmem = 0;            // [한국어] 현재 점유된 공유 메모리 바이트
  m_occupied_regs = 0;             // [한국어] 현재 점유된 레지스터 수
  m_occupied_ctas = 0;             // [한국어] 현재 점유된 CTA 수
  m_occupied_hwtid.reset();        // [한국어] 점유된 하드웨어 스레드 ID 비트마스크 초기화
  m_occupied_cta_to_hwtid.clear(); // [한국어] CTA ID → 하드웨어 스레드 ID 매핑 맵 초기화
}

/*
 * [한국어]
 * shader_core_ctx::reinit — SM 상태 부분/전체 재초기화
 *
 * @start_thread:       재초기화할 스레드 범위 시작 (포함)
 * @end_thread:         재초기화할 스레드 범위 끝 (미포함)
 * @reset_not_completed: true이면 전역 완료 카운터도 초기화 (SM 전체 재시작 시)
 * @return:             void
 *
 * 커널 실행 종료 후 새로운 커널을 로드하거나, SM을 초기화할 때 호출된다.
 * reset_not_completed=true이면 SM 전체를 초기화(새 커널 시작).
 * reset_not_completed=false이면 특정 스레드 범위만 초기화(CTA 부분 재사용).
 * 각 warp의 scoreboard, SIMT 스택, IBuffer 상태를 모두 리셋.
 *
 * 호출 체인: simt_core_cluster::reinit() → shader_core_ctx::reinit()
 *            또는 issue_block2core() 완료 후 새 CTA 배치 시
 */
void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread,
                             bool reset_not_completed) {
  if (reset_not_completed) { // [한국어] SM 전체 재시작: 전역 카운터 및 점유 정보 모두 초기화
    m_not_completed = 0;        // [한국어] 완료되지 않은 스레드 수 0으로 리셋
    m_active_threads.reset();   // [한국어] 활성 스레드 비트마스크 전체 클리어

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;        // [한국어] 점유 스레드 수 초기화
    m_occupied_shmem = 0;            // [한국어] 점유 공유 메모리 초기화
    m_occupied_regs = 0;             // [한국어] 점유 레지스터 초기화
    m_occupied_ctas = 0;             // [한국어] 점유 CTA 수 초기화
    m_occupied_hwtid.reset();        // [한국어] 점유 하드웨어 스레드 비트마스크 초기화
    m_occupied_cta_to_hwtid.clear(); // [한국어] CTA→하드웨어 스레드 매핑 초기화
    m_active_warps = 0;              // [한국어] 활성 warp 수 초기화 (점유율 계산 기준)
  }
  // [한국어] 지정된 스레드 범위의 명령어 카운트 및 CTA 소속 초기화
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;    // [한국어] 이 스레드가 실행한 명령어 수 초기화
    m_threadState[i].m_cta_id = -1; // [한국어] CTA 소속 해제 (-1: 미할당)
  }
  // [한국어] 지정된 스레드 범위에 해당하는 warp들의 상태 리셋
  for (unsigned i = start_thread / m_config->warp_size;
       i < end_thread / m_config->warp_size; ++i) {
    m_warp[i]->reset();       // [한국어] warp 상태 초기화: IBuffer, 완료 카운터, store 요청 등
    m_simt_stack[i]->reset(); // [한국어] SIMT 스택 초기화: 분기 상태(active mask, PC) 클리어
  }
}

/*
 * [한국어]
 * shader_core_ctx::init_warps — CTA 배치 후 warp 초기화
 *
 * @cta_id:       SM 내부 CTA 슬롯 ID (0 ~ max_cta_per_core-1)
 * @start_thread: 이 CTA의 첫 번째 하드웨어 스레드 ID
 * @end_thread:   이 CTA의 마지막+1 하드웨어 스레드 ID
 * @ctaid:        전역 CTA ID (체크포인트 재시작 시 CTA 범위 판별에 사용)
 * @cta_size:     CTA 내 스레드 수
 * @kernel:       이 CTA가 속한 커널 정보 (시작 PC, 스트림 ID 등)
 * @return:       void
 *
 * 새로운 CTA가 이 SM에 배치될 때 호출된다. POST_DOMINATOR 모델에서:
 *   1. CTA 내 각 warp에 대해 활성 스레드 마스크를 계산
 *   2. SIMT 스택을 시작 PC로 초기화 (launch())
 *   3. 체크포인트 재시작 옵션이 있으면 파일에서 SIMT 스택 상태 복원
 *   4. shd_warp_t::init()으로 warp 하드웨어 상태 초기화
 *   5. m_not_completed와 m_active_warps 카운터 증가
 *
 * CTA 경계가 warp에 딱 맞지 않으면(cta_size % warp_size != 0) 마지막 warp는 일부 스레드만 활성화.
 *
 * 호출 체인: shader_core_ctx::issue_block2core() → [이 함수]
 */
void shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_pc(start_thread); // [한국어] 첫 번째 스레드의 현재 PC를 시작 PC로 사용 (PTX 기능 시뮬레이터에서 가져옴)
  unsigned kernel_id = kernel.get_uid(); // [한국어] 커널 고유 ID — 체크포인트 재시작 시 대상 커널 식별
  if (m_config->model == POST_DOMINATOR) { // [한국어] POST_DOMINATOR: 현재 GPGPU-Sim이 지원하는 유일한 SIMT 분기 모델
    unsigned start_warp = start_thread / m_config->warp_size; // [한국어] 첫 warp ID: 시작 스레드를 warp 크기로 나눔
    unsigned warp_per_cta = cta_size / m_config->warp_size;   // [한국어] 이 CTA의 총 warp 수 (체크포인트 파일명 생성에 사용)
    // [한국어] 마지막 warp ID 계산: end_thread가 warp 경계에 정확히 맞지 않으면 ceil
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);
    for (unsigned i = start_warp; i < end_warp; ++i) { // [한국어] CTA 내 각 warp 순서대로 초기화
      unsigned n_active = 0;       // [한국어] 이 warp에서 실제로 활성화된 스레드 수
      simt_mask_t active_threads;  // [한국어] 이 warp의 활성 스레드 비트마스크 (32비트)
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        unsigned hwtid = i * m_config->warp_size + t; // [한국어] 이 warp의 t번째 스레드의 하드웨어 스레드 ID
        if (hwtid < end_thread) { // [한국어] end_thread 미만인 스레드만 활성화 (CTA 경계 처리)
          n_active++;
          assert(!m_active_threads.test(hwtid)); // [한국어] 동일 스레드가 이미 활성화되어 있으면 안 됨 (중복 배치 방지)
          m_active_threads.set(hwtid);  // [한국어] SM 전역 활성 스레드 비트마스크에 등록
          active_threads.set(t);        // [한국어] 이 warp 내 로컬 활성 마스크에도 등록
        }
      }
      m_simt_stack[i]->launch(start_pc, active_threads); // [한국어] SIMT 스택 초기화: start_pc와 전체 활성 마스크로 시작. 이후 분기가 발생하면 스택에 push됨.

      // [한국어] 체크포인트 재시작: resume_option=1이고 이 CTA가 재시작 범위에 있으면 SIMT 스택 상태를 파일에서 복원
      if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
          ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt",
                 i % warp_per_cta, ctaid); // [한국어] warp별 SIMT 체크포인트 파일명 생성
        unsigned pc, rpc;
        m_simt_stack[i]->resume(fname); // [한국어] 체크포인트 파일에서 SIMT 스택 상태 복원
        m_simt_stack[i]->get_pdom_stack_top_info(&pc, &rpc); // [한국어] 복원된 스택의 현재 PC와 reconvergence PC 가져옴
        for (unsigned t = 0; t < m_config->warp_size; t++) {
          if (m_thread != NULL) {
            m_thread[i * m_config->warp_size + t]->set_npc(pc); // [한국어] 기능 시뮬레이션 스레드의 PC를 복원된 값으로 설정
            m_thread[i * m_config->warp_size + t]->update_pc(); // [한국어] PTX 스레드 상태에 PC 업데이트 반영
          }
        }
        start_pc = pc; // [한국어] 시작 PC를 체크포인트에서 복원한 값으로 교체
      }

      // [한국어] shd_warp_t 초기화: 시작 PC, CTA ID, warp ID, 활성 마스크, 동적 warp ID, 스트림 ID 설정
      m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id,
                      kernel.get_streamID());
      ++m_dynamic_warp_id;     // [한국어] 동적 warp ID 증가 — GTO/Oldest 스케줄러의 순위 결정 기준
      m_not_completed += n_active; // [한국어] SM 전체 미완료 스레드 수에 이 warp의 활성 스레드 추가
      ++m_active_warps;            // [한국어] SM 활성 warp 수 증가 — 점유율 계산에 사용
    }
  }
}

/*
 * [한국어]
 * shader_core_ctx::next_pc — 스레드의 현재 PC 반환 (기능 시뮬레이션 레이어)
 *
 * @tid:    조회할 스레드의 하드웨어 스레드 ID (-1이면 유효하지 않음)
 * @return: 해당 스레드의 현재 PC. tid=-1이거나 스레드가 NULL이면 -1 반환.
 *
 * 기능 시뮬레이션(cuda-sim) 레이어의 ptx_thread_info에서 현재 PC를 가져온다.
 * 이 PC는 이전 사이클에 shader_decode()에서 이미 다음 명령어 PC로 업데이트되어 있음.
 * init_warps()에서 warp 시작 PC를 결정할 때 사용.
 *
 * 호출 체인: shader_core_ctx::init_warps() → [이 함수] → ptx_thread_info::get_pc()
 */
// return the next pc of a thread
address_type shader_core_ctx::next_pc(int tid) const {
  if (tid == -1) return -1; // [한국어] 유효하지 않은 tid — 호출자가 -1을 체크하도록 -1 반환
  ptx_thread_info *the_thread = m_thread[tid]; // [한국어] 기능 시뮬레이션 스레드 객체 조회
  if (the_thread == NULL) return -1; // [한국어] 아직 초기화되지 않은 스레드 (trace-driven 모드에서 발생 가능)
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
                   // [한국어] ptx_thread_info::get_pc()가 반환하는 PC는 이미 "다음 실행할" 명령어 주소임
}

/*
 * [한국어]
 * gpgpu_sim::get_pdom_stack_top_info — 최상위 레벨에서 SIMT 스택 상단 정보 조회
 *
 * @sid: 조회할 SM의 전역 ID
 * @tid: 조회할 스레드의 하드웨어 스레드 ID
 * @pc:  [출력] 현재 SIMT 스택 상단의 PC
 * @rpc: [출력] Reconvergence PC (PDOM 스택에서 재합류 지점)
 * @return: void
 *
 * gpgpu_sim → cluster → SM → SIMT 스택 순으로 위임하는 래퍼 체인의 최상위.
 * 체크포인트/디버그 목적으로 외부에서 특정 SM의 SIMT 상태를 조회할 때 사용.
 *
 * 호출 체인: 외부 → [이 함수] → cluster::get_pdom_stack_top_info() → shader_core_ctx::get_pdom_stack_top_info()
 */
void gpgpu_sim::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                        unsigned *pc, unsigned *rpc) {
  unsigned cluster_id = m_shader_config->sid_to_cluster(sid); // [한국어] SM ID → 클러스터 ID 변환 (sid_to_cluster: sid / n_simt_cores_per_cluster)
  m_cluster[cluster_id]->get_pdom_stack_top_info(sid, tid, pc, rpc); // [한국어] 클러스터로 위임
}

/*
 * [한국어]
 * shader_core_ctx::get_pdom_stack_top_info — SM 내 SIMT 스택 상단 정보 조회
 *
 * @tid: 조회할 스레드의 하드웨어 스레드 ID
 * @pc:  [출력] 현재 warp의 SIMT 스택 상단 PC
 * @rpc: [출력] Reconvergence PC (이 분기에서 모든 스레드가 재합류하는 지점)
 * @return: void
 *
 * tid를 warp_id로 변환하고 해당 warp의 SIMT 스택에서 top 정보를 가져온다.
 * PDOM(Post-Dominator) 스택은 분기 시 활성 마스크와 재합류 PC를 스택에 저장.
 *
 * 호출 체인: gpgpu_sim::get_pdom_stack_top_info() → cluster → [이 함수] → simt_stack::get_pdom_stack_top_info()
 */
void shader_core_ctx::get_pdom_stack_top_info(unsigned tid, unsigned *pc,
                                              unsigned *rpc) const {
  unsigned warp_id = tid / m_config->warp_size; // [한국어] 스레드 ID → warp ID 변환 (32개 스레드 = 1 warp)
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc); // [한국어] 해당 warp의 SIMT 스택 상단에서 PC와 rPC 조회
}

/*
 * [한국어]
 * shader_core_ctx::get_current_occupancy — 현재 SM의 warp 점유율 계산
 *
 * @active: [입출력] 누적 활성 warp 수 (이 SM의 m_active_warps를 더함)
 * @total:  [입출력] 누적 총 warp 슬롯 수 (이 SM의 m_warp.size()를 더함)
 * @return: 현재까지 누적된 active/total 비율 (점유율). 이 SM이 비활성이면 0.
 *
 * nvprof의 achieved_occupancy와 일치하도록, 활성 warp가 있는 SM만 계산에 포함.
 * 비활성 SM(m_active_warps==0)은 0을 반환하여 분모에 영향을 미치지 않음.
 *
 * 호출 체인: simt_core_cluster::get_current_occupancy() → [이 함수]
 */
float shader_core_ctx::get_current_occupancy(unsigned long long &active,
                                             unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) { // [한국어] 활성 warp가 있는 SM만 점유율 계산에 포함 (nvprof 기준)
    total += m_warp.size();    // [한국어] 이 SM의 최대 warp 슬롯 수를 총합에 추가
    active += m_active_warps;  // [한국어] 현재 활성 warp 수를 활성 합산에 추가
    return float(active) / float(total); // [한국어] 현재까지의 누적 점유율 반환
  } else {
    return 0; // [한국어] 비활성 SM은 0 반환 (분모에 추가하지 않음)
  }
}

/*
 * [한국어]
 * shader_core_stats::print — SM 통계를 파일에 출력
 *
 * @fout: 출력 대상 파일 포인터 (보통 stdout 또는 results 파일)
 * @return: void
 *
 * 시뮬레이션 종료 후 gpgpusim.config 분석에 필요한 SM 레벨 통계를 출력한다:
 * - 총 스레드/warp 명령어 카운트
 * - 공유 메모리/글로벌/로컬/텍스처/상수 메모리 접근 수
 * - 스톨 분류 (뱅크 충돌, 코얼레싱, 데이터 포트 스톨)
 * - warp 점유율 분포 (divergence, scoreboard 대기, 파이프라인 스톨)
 * - 단일/이중 발행 횟수
 *
 * 호출 체인: gpgpu_sim::print_stats() → shader_core_stats::print()
 */
void shader_core_stats::print(FILE *fout) const {
  unsigned long long thread_icount_uarch = 0; // [한국어] 모든 SM의 스레드 명령어 카운트 누적 (clock-gated 모드에서는 활성 스레드 수만 카운트)
  unsigned long long warp_icount_uarch = 0;   // [한국어] 모든 SM의 warp 명령어 카운트 누적 (warp 단위 — 비활성 스레드 포함)

  for (unsigned i = 0; i < m_config->num_shader(); i++) { // [한국어] 모든 SM의 통계 합산
    thread_icount_uarch += m_num_sim_insn[i];  // [한국어] SM i의 시뮬레이션 명령어 수 누적
    warp_icount_uarch += m_num_sim_winsn[i];   // [한국어] SM i의 warp 명령어 수 누적
  }
  fprintf(fout, "gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch); // [한국어] 총 스레드 명령어 카운트 출력
  fprintf(fout, "gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);       // [한국어] 총 warp 명령어 카운트 출력

  fprintf(fout, "gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem);
  fprintf(fout, "gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
  fprintf(fout, "gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
  fprintf(fout, "gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
  fprintf(fout, "gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
  fprintf(fout, "gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
  fprintf(fout, "gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

  fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
  fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
  fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
  fprintf(fout, "gpgpu_n_sstarr_insn = %d\n", gpgpu_n_sstarr_insn);
  fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
  fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
  fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

  fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
  fprintf(fout, "gpgpu_n_l1cache_bkconflict = %d\n",
          gpgpu_n_l1cache_bkconflict);

  fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
          gpgpu_n_intrawarp_mshr_merge);
  fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

  fprintf(fout, "gpgpu_stall_shd_mem[c_mem][resource_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
  // fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][DATA_PORT_STALL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][DATA_PORT_STALL]);
  fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n",
          gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][resource_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]);  // coalescing stall
                                                            // at data cache
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                     [COAL_STALL]);  // coalescing stall + bank
                                                     // conflict at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[G_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[G_MEM_ST][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                         [DATA_PORT_STALL]);  // data port stall
                                                              // at data cache
  // fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

  fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n",
          gpu_reg_bank_conflict_stalls);

  fprintf(fout, "Warp Occupancy Distribution:\n");
  fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
  fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
  fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
  for (unsigned i = 3; i < m_config->warp_size + 3; i++)
    fprintf(fout, "\tW%d:%d", i - 2, shader_cycle_distro[i]);
  fprintf(fout, "\n");
  fprintf(fout, "single_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, single_issue_nums[i]);
  fprintf(fout, "\n");
  fprintf(fout, "dual_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, dual_issue_nums[i]);
  fprintf(fout, "\n");

  m_outgoing_traffic_stats->print(fout);
  m_incoming_traffic_stats->print(fout);
}

/*
 * [한국어]
 * shader_core_stats::event_warp_issued — warp 발행 이벤트 통계 기록
 *
 * @s_id:            SM ID (통계 배열의 인덱스)
 * @warp_id:         발행된 warp의 슬롯 ID (0 ~ max_warps_per_shader-1)
 * @num_issued:      이번 사이클에 이 warp에서 발행된 명령어 수 (보통 1, 듀얼 발행 시 2)
 * @dynamic_warp_id: 이 warp의 동적 ID (CTA 배치 순서로 부여, 시뮬레이션 전체에서 단조 증가)
 * @return:          void
 *
 * 두 종류의 발행 분포 히스토그램을 갱신한다:
 * - m_shader_warp_slot_issue_distro: warp 슬롯(하드웨어 warp ID)별 발행 횟수
 *   → 특정 슬롯에 편중이 있는지 파악 (스케줄러 불균형 분석)
 * - m_shader_dynamic_warp_issue_distro: 동적 warp ID별 발행 횟수
 *   → 어떤 커널/CTA의 warp가 더 많이 발행되는지 파악
 *
 * 호출 체인: scheduler_unit::do_on_warp_issued() → [이 함수]
 */
void shader_core_stats::event_warp_issued(unsigned s_id, unsigned warp_id,
                                          unsigned num_issued,
                                          unsigned dynamic_warp_id) {
  assert(warp_id <= m_config->max_warps_per_shader); // [한국어] warp_id 범위 검증 (0 ~ max_warps_per_shader 이내)
  for (unsigned i = 0; i < num_issued; ++i) { // [한국어] 듀얼 발행 시 2번 카운트 (num_issued만큼 반복)
    if (m_shader_dynamic_warp_issue_distro[s_id].size() <= dynamic_warp_id) {
      m_shader_dynamic_warp_issue_distro[s_id].resize(dynamic_warp_id + 1); // [한국어] 동적 warp ID가 현재 크기를 초과하면 벡터 확장
    }
    ++m_shader_dynamic_warp_issue_distro[s_id][dynamic_warp_id]; // [한국어] 이 동적 warp ID의 발행 카운트 증가
    if (m_shader_warp_slot_issue_distro[s_id].size() <= warp_id) {
      m_shader_warp_slot_issue_distro[s_id].resize(warp_id + 1); // [한국어] warp 슬롯 ID가 현재 크기를 초과하면 벡터 확장
    }
    ++m_shader_warp_slot_issue_distro[s_id][warp_id]; // [한국어] 이 warp 슬롯의 발행 카운트 증가
  }
}

void shader_core_stats::visualizer_print(gzFile visualizer_file) {
  // warp divergence breakdown
  gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
  unsigned int total = 0;
  unsigned int cf =
      (m_config->gpgpu_warpdistro_shader == -1) ? m_config->num_shader() : 1;
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf);
  for (unsigned i = 0; i < m_config->warp_size + 3; i++) {
    if (i >= 3) {
      total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
      if (((i - 3) % (m_config->warp_size / 8)) ==
          ((m_config->warp_size / 8) - 1)) {
        gzprintf(visualizer_file, " %d", total / cf);
        total = 0;
      }
    }
    last_shader_cycle_distro[i] = shader_cycle_distro[i];
  }
  gzprintf(visualizer_file, "\n");

  gzprintf(visualizer_file, "ctas_completed: %d\n", ctas_completed);
  ctas_completed = 0;
  // warp issue breakdown
  unsigned sid = m_config->gpgpu_warp_issue_shader;
  unsigned count = 0;
  unsigned warp_id_issued_sum = 0;
  gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
  if (m_shader_warp_slot_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_warp_slot_issue_distro[sid].begin();
         iter != m_shader_warp_slot_issue_distro[sid].end(); iter++, count++) {
      unsigned diff = count < m_last_shader_warp_slot_issue_distro.size()
                          ? *iter - m_last_shader_warp_slot_issue_distro[count]
                          : *iter;
      gzprintf(visualizer_file, " %d", diff);
      warp_id_issued_sum += diff;
    }
    m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[sid];
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

#define DYNAMIC_WARP_PRINT_RESOLUTION 32
  unsigned total_issued_this_resolution = 0;
  unsigned dynamic_id_issued_sum = 0;
  count = 0;
  gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
  if (m_shader_dynamic_warp_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_dynamic_warp_issue_distro[sid].begin();
         iter != m_shader_dynamic_warp_issue_distro[sid].end();
         iter++, count++) {
      unsigned diff =
          count < m_last_shader_dynamic_warp_issue_distro.size()
              ? *iter - m_last_shader_dynamic_warp_issue_distro[count]
              : *iter;
      total_issued_this_resolution += diff;
      if ((count + 1) % DYNAMIC_WARP_PRINT_RESOLUTION == 0) {
        gzprintf(visualizer_file, " %d", total_issued_this_resolution);
        dynamic_id_issued_sum += total_issued_this_resolution;
        total_issued_this_resolution = 0;
      }
    }
    if (count % DYNAMIC_WARP_PRINT_RESOLUTION != 0) {
      gzprintf(visualizer_file, " %d", total_issued_this_resolution);
      dynamic_id_issued_sum += total_issued_this_resolution;
    }
    m_last_shader_dynamic_warp_issue_distro =
        m_shader_dynamic_warp_issue_distro[sid];
    assert(warp_id_issued_sum == dynamic_id_issued_sum);
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

  // overall cache miss rates
  gzprintf(visualizer_file, "gpgpu_n_l1cache_bkconflict: %d\n",
           gpgpu_n_l1cache_bkconflict);
  gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n",
           gpgpu_n_shmem_bkconflict);

  // instruction count per shader core
  gzprintf(visualizer_file, "shaderinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_insn[i]);
  gzprintf(visualizer_file, "\n");
  // warp instruction count per shader core
  gzprintf(visualizer_file, "shaderwarpinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i]);
  gzprintf(visualizer_file, "\n");
  // warp divergence per shader core
  gzprintf(visualizer_file, "shaderwarpdiv: ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_n_diverge[i]);
  gzprintf(visualizer_file, "\n");
}

// [한국어] PROGRAM_MEM_START: 명령어 메모리 주소 공간의 시작 오프셋.
// PTX 가상 주소 공간에서 명령어는 0xF0000000부터 배치되어 다른 메모리 공간(글로벌/공유/로컬 등)과 겹치지 않음.
// fetch()에서 PTX PC를 NoC 주소로 변환할 때 이 오프셋을 더함: ppc = pc + PROGRAM_MEM_START
#define PROGRAM_MEM_START                                      \
  0xF0000000 /* should be distinct from other memory spaces... \
                check ptx_ir.h to verify this does not overlap \
                other memory spaces */

/*
 * [한국어]
 * exec_shader_core_ctx::get_next_inst — 기능 시뮬레이션에서 명령어 가져오기
 *
 * @warp_id: 명령어를 가져올 warp ID (사용되지 않음 — ptx_fetch_inst는 PC만 사용)
 * @pc:      가져올 명령어의 PTX PC
 * @return:  해당 PC의 warp_inst_t 포인터 (기능 모델이 소유, 복사 금지)
 *
 * PTX 기능 시뮬레이터(gpgpu_ctx->ptx_fetch_inst)에서 명령어 객체를 가져온다.
 * 이 명령어 객체는 decode()에서 I-buffer에 채워진다.
 * exec 버전(실제 실행 모드)에서만 사용되며, trace-driven 모드는 별도 구현.
 *
 * 호출 체인: shader_core_ctx::decode() → exec_shader_core_ctx::get_next_inst() → ptx_fetch_inst()
 */
const warp_inst_t *exec_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                       address_type pc) {
  // read the inst from the functional model
  return m_gpu->gpgpu_ctx->ptx_fetch_inst(pc); // [한국어] PTX IR에서 해당 PC의 명령어 객체 조회 — PTX 파서가 빌드한 정적 명령어 테이블에서 읽음
}

/*
 * [한국어]
 * exec_shader_core_ctx::get_pdom_stack_top_info — warp의 SIMT 스택 상단 PC/rPC 조회
 *
 * @warp_id: 조회할 warp ID
 * @pI:      현재 명령어 포인터 (미사용, 인터페이스 일관성을 위해 존재)
 * @pc:      [출력] SIMT 스택 상단의 현재 PC
 * @rpc:     [출력] Reconvergence PC (재합류 지점)
 * @return:  void
 *
 * 스케줄러가 ibuffer의 명령어 PC와 SIMT 스택의 실제 PC를 비교할 때 사용.
 * 불일치 시 제어 해저드(control hazard)로 ibuffer를 플러시.
 *
 * 호출 체인: scheduler_unit::cycle() → m_shader->get_pdom_stack_top_info() → [이 함수]
 */
void exec_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                   const warp_inst_t *pI,
                                                   unsigned *pc,
                                                   unsigned *rpc) {
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc); // [한국어] 해당 warp의 SIMT 스택 상단에서 PC와 reconvergence PC 조회
}

/*
 * [한국어]
 * exec_shader_core_ctx::get_active_mask — warp의 현재 활성 스레드 마스크 반환
 *
 * @warp_id: 조회할 warp ID
 * @pI:      현재 명령어 포인터 (미사용)
 * @return:  SIMT 스택 상단의 active_mask 참조 (어떤 스레드가 이 명령어를 실행할지)
 *
 * 스케줄러가 issue_warp()를 호출할 때 활성 마스크를 전달하기 위해 사용.
 * SIMT 스택의 활성 마스크는 분기 진입 시 변경되고, 재합류 시 복원됨.
 *
 * 호출 체인: scheduler_unit::cycle() → m_shader->get_active_mask() → [이 함수]
 */
const active_mask_t &exec_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  return m_simt_stack[warp_id]->get_active_mask(); // [한국어] SIMT 스택 상단의 활성 마스크 반환 — 어느 스레드가 이 명령어를 실행할지 결정
}

/*
 * [한국어]
 * shader_core_ctx::decode — I-buffer 채움 스테이지 (Fetch → Decode)
 *
 * @return: void
 *
 * m_inst_fetch_buffer에 유효한 명령어 패킷이 있으면, 최대 2개의 명령어를
 * 디코딩하여 해당 warp의 I-buffer(ibuffer)에 적재한다.
 * GPGPU-Sim에서 "decode"는 PTX 명령어를 warp_inst_t 형태로 가져와 I-buffer에 넣는 것을 의미.
 * 실제 디코딩(명령어 해석)은 이미 PTX 파서가 완료했으므로, 여기서는 I-buffer 로딩만 수행.
 *
 * 동작 단계:
 *   1. m_inst_fetch_buffer.m_valid가 true이면 진행
 *   2. PC에서 pI1 = 첫 번째 명령어 조회 (get_next_inst)
 *   3. ibuffer[0]에 pI1 적재, inst_in_pipeline 카운터 증가
 *   4. pI1의 operand 타입으로 AccelWattch 통계 업데이트
 *   5. pI1.isize만큼 앞의 PC에서 pI2 = 두 번째 명령어 조회 시도
 *   6. pI2가 존재하면 ibuffer[1]에 적재
 *   7. m_inst_fetch_buffer를 무효화 (다음 fetch를 허용)
 *
 * 사이클마다 최대 2개 명령어를 디코딩할 수 있어 warp 전진 속도를 높임.
 *
 * 호출 체인: shader_core_ctx::cycle() → decode() → get_next_inst() → ptx_fetch_inst()
 */
void shader_core_ctx::decode() {
  if (m_inst_fetch_buffer.m_valid) { // [한국어] fetch()가 I-cache HIT 또는 이전 사이클 미스 복귀 후 유효 패킷을 채워두었을 때만 진행
    // decode 1 or 2 instructions and place them into ibuffer
    address_type pc = m_inst_fetch_buffer.m_pc; // [한국어] 디코딩할 첫 번째 명령어 PC
    const warp_inst_t *pI1 = get_next_inst(m_inst_fetch_buffer.m_warp_id, pc); // [한국어] PTX 기능 모델에서 첫 번째 명령어 객체 가져오기
    m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(0, pI1); // [한국어] I-buffer 슬롯 0에 명령어 적재
    m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline(); // [한국어] 파이프라인 내 명령어 수 증가 (writeback 완료 시 감소)
    if (pI1) { // [한국어] 유효한 명령어가 있는 경우 (NULL이면 warp 종료 혹은 NOP)
      m_stats->m_num_decoded_insn[m_sid]++; // [한국어] 디코딩된 명령어 수 통계 증가
      if ((pI1->oprnd_type == INT_OP) ||
          (pI1->oprnd_type == UN_OP)) {  // these counters get added up in mcPat
                                         // to compute scheduler power
        // [한국어] INT_OP/UN_OP: 정수/단항 연산 — AccelWattch/McPAT 스케줄러 전력 계산용
        m_stats->m_num_INTdecoded_insn[m_sid]++;
      } else if (pI1->oprnd_type == FP_OP) { // [한국어] FP_OP: 부동소수점 연산
        m_stats->m_num_FPdecoded_insn[m_sid]++;
      }
      // [한국어] 두 번째 명령어 시도: pI1의 크기(isize) 이후 주소에서 조회
      const warp_inst_t *pI2 =
          get_next_inst(m_inst_fetch_buffer.m_warp_id, pc + pI1->isize); // [한국어] pI1.isize: PTX 명령어 크기 (보통 8바이트). 다음 명령어 PC = pc + isize.
      if (pI2) { // [한국어] 두 번째 명령어도 유효한 경우 I-buffer에 추가 적재
        m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(1, pI2); // [한국어] I-buffer 슬롯 1에 두 번째 명령어 적재
        m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline(); // [한국어] 두 번째 명령어도 파이프라인 진입 카운트
        m_stats->m_num_decoded_insn[m_sid]++; // [한국어] 두 번째 명령어 통계 증가
        if ((pI1->oprnd_type == INT_OP) ||
            (pI1->oprnd_type == UN_OP)) {  // these counters get added up in
                                           // mcPat to compute scheduler power
          m_stats->m_num_INTdecoded_insn[m_sid]++; // [한국어] 정수 명령어 카운터 증가 (주의: pI2가 아닌 pI1의 타입을 잘못 체크하는 버그가 있음)
        } else if (pI2->oprnd_type == FP_OP) { // [한국어] 두 번째 명령어가 FP 연산인 경우
          m_stats->m_num_FPdecoded_insn[m_sid]++;
        }
      }
    }
    m_inst_fetch_buffer.m_valid = false; // [한국어] fetch buffer 무효화 — 다음 사이클에 fetch()가 새 패킷을 채울 수 있도록
  }
}

/*
 * [한국어]
 * shader_core_ctx::fetch — 명령어 캐시(L1I) 접근 및 I-buffer 채우기 스테이지
 *
 * @return: void
 *
 * SM 파이프라인의 첫 번째 스테이지. 매 사이클 호출되어:
 * 1. m_inst_fetch_buffer가 유효하지 않은 경우에만 동작
 * 2. L1I 캐시에서 이전 miss에 대한 응답이 도착했는지 확인 (access_ready)
 *    - 도착했으면: miss_pending 해제, fetch buffer 채움
 * 3. 응답이 없으면: 라운드로빈으로 warp 순회하여 fetch 가능한 warp 탐색:
 *    조건: functional_done=false && imiss_pending=false && ibuffer_empty=true
 *    - 탐색 성공 시: L1I->access() 호출
 *      * HIT: 즉시 fetch buffer 채움 (0 사이클 레이턴시)
 *      * MISS: imiss_pending 설정, 다음 사이클에 응답 대기
 *      * RESERVATION_FAIL: MSHR이 꽉 찬 경우, mf 삭제하고 다음 사이클 재시도
 * 4. warp가 hardware_done이면 스레드 종료 처리 (register_cta_thread_exit)
 * 5. m_L1I->cycle() 호출 (캐시 내부 레이턴시 큐 진행)
 *
 * perfect_inst_const_cache: L1I를 무한 용량/0 레이턴시로 가정할 때 status=HIT 강제.
 * m_last_warp_fetched: 라운드로빈 시작점 — 매번 다음 warp부터 탐색하여 공정성 보장.
 *
 * 호출 체인: shader_core_ctx::cycle() → fetch() → m_L1I->access()
 *            L1I miss → NoC → DRAM → accept_fetch_response() → L1I::fill()
 */
void shader_core_ctx::fetch() {
  if (!m_inst_fetch_buffer.m_valid) { // [한국어] fetch buffer가 비어있어야 새로운 fetch를 시도 — decode()가 소비하면 무효화됨
    if (m_L1I->access_ready()) { // [한국어] L1I 캐시에서 이전 miss 요청에 대한 응답이 준비되었는지 확인
      mem_fetch *mf = m_L1I->next_access(); // [한국어] L1I에서 완료된 fetch 응답(mem_fetch) 가져오기
      m_warp[mf->get_wid()]->clear_imiss_pending(); // [한국어] I-cache miss 대기 상태 해제 — 이 warp가 다시 fetch 가능해짐
      m_inst_fetch_buffer =
          ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
                          mf->get_access_size(), mf->get_wid()); // [한국어] fetch buffer를 warp PC, 접근 크기, warp ID로 채움
      assert(m_warp[mf->get_wid()]->get_pc() ==
             (mf->get_addr() -
              PROGRAM_MEM_START));  // Verify that we got the instruction we
                                    // were expecting.
                                    // [한국어] 검증: 응답으로 받은 주소(NoC 주소 - 오프셋)가 warp의 현재 PC와 일치하는지 확인
      m_inst_fetch_buffer.m_valid = true; // [한국어] fetch buffer 유효 플래그 설정 — decode()가 다음 사이클에 읽을 수 있음
      m_warp[mf->get_wid()]->set_last_fetch(m_gpu->gpu_sim_cycle); // [한국어] 마지막 fetch 사이클 기록 (디버그/통계용)
      delete mf; // [한국어] 처리 완료된 mem_fetch 해제 — I-cache hit 응답은 데이터를 fetch buffer로 옮겼으므로 불필요
    } else {
      // find an active warp with space in instruction buffer that is not
      // already waiting on a cache miss and get next 1-2 instructions from
      // i-cache...
      // [한국어] 라운드로빈 방식으로 fetch 가능한 warp 탐색.
      // m_last_warp_fetched 다음 warp부터 시작하여 한 바퀴 순환.
      for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        unsigned warp_id =
            (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader; // [한국어] 라운드로빈: 마지막으로 fetch한 warp 다음부터 순환

        // this code checks if this warp has finished executing and can be
        // reclaimed
        // [한국어] warp가 완전히 완료되었는지 확인: hardware_done=true(모든 스레드 기능적 완료 + store 완료 + 파이프라인 비어있음)
        // 이고 스코어보드에 pending write가 없고 아직 done_exit 처리가 안 된 경우
        if (m_warp[warp_id]->hardware_done() &&
            !m_scoreboard->pendingWrites(warp_id) &&
            !m_warp[warp_id]->done_exit()) {
          bool did_exit = false; // [한국어] 실제로 종료 처리된 스레드가 있는지 추적
          for (unsigned t = 0; t < m_config->warp_size; t++) {
            unsigned tid = warp_id * m_config->warp_size + t; // [한국어] warp 내 t번째 스레드의 전역 스레드 ID
            if (m_threadState[tid].m_active == true) { // [한국어] 아직 활성 상태인 스레드만 종료 처리
              m_threadState[tid].m_active = false; // [한국어] 스레드 비활성화
              unsigned cta_id = m_warp[warp_id]->get_cta_id(); // [한국어] 이 스레드가 속한 CTA ID
              if (m_thread[tid] == NULL) { // [한국어] 기능 시뮬레이션 스레드가 없는 경우 (trace-driven 모드)
                register_cta_thread_exit(cta_id,
                                         m_warp[warp_id]->get_kernel_info()); // [한국어] CTA 스레드 종료 등록 (kernel_info_t로 커널 식별)
              } else {
                register_cta_thread_exit(cta_id,
                                         &(m_thread[tid]->get_kernel())); // [한국어] execution-driven 모드: ptx_thread_info에서 커널 정보 가져와 종료 등록
              }
              m_not_completed -= 1;      // [한국어] SM 전체 미완료 스레드 수 감소
              m_active_threads.reset(tid); // [한국어] SM 활성 스레드 비트마스크에서 해제
              did_exit = true;
            }
          }
          if (did_exit) m_warp[warp_id]->set_done_exit(); // [한국어] 이 warp를 "완전 종료" 상태로 마킹 — 이후 fetch에서 무시
          --m_active_warps;               // [한국어] SM 활성 warp 수 감소
          assert(m_active_warps >= 0);    // [한국어] 활성 warp가 음수가 되면 로직 오류
        }

        // this code fetches instructions from the i-cache or generates memory
        // [한국어] fetch 가능 조건: 기능적으로 완료되지 않았고, I-cache miss 대기 중이 아니고, I-buffer가 비어있음
        if (!m_warp[warp_id]->functional_done() &&
            !m_warp[warp_id]->imiss_pending() &&
            m_warp[warp_id]->ibuffer_empty()) {
          address_type pc;
          pc = m_warp[warp_id]->get_pc(); // [한국어] 이 warp의 현재 PC (다음에 실행할 명령어 주소)
          address_type ppc = pc + PROGRAM_MEM_START; // [한국어] NoC 주소로 변환: PTX PC + PROGRAM_MEM_START 오프셋
          unsigned nbytes = 16; // [한국어] 기본 fetch 크기: 16바이트 (PTX 명령어 2개 분량, 명령어당 최대 8바이트)
          // [한국어] 캐시 라인 경계 처리: PC가 캐시 라인 끝 부분에 걸쳐있으면 fetch 크기를 줄여 라인 경계를 넘지 않도록
          unsigned offset_in_block =
              pc & (m_config->m_L1I_config.get_line_sz() - 1); // [한국어] 캐시 라인 내 PC 오프셋 계산 (라인 크기는 2의 거듭제곱 가정)
          if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
            nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block); // [한국어] 라인 경계까지만 fetch하여 불필요한 크로스-라인 접근 방지

          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          // [한국어] I-cache 접근용 mem_fetch 생성: INST_ACC_R 타입으로 명령어 읽기 요청
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
          mem_fetch *mf = new mem_fetch(
              acc, NULL, m_warp[warp_id]->get_kernel_info()->get_streamID(),
              READ_PACKET_SIZE, warp_id, m_sid, m_tpc, m_memory_config,
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle); // [한국어] warp_id, SM ID, TPC ID를 포함하여 응답이 올바른 warp에 전달되도록 함
          std::list<cache_event> events; // [한국어] 캐시 접근 이벤트 목록 (eviction, write 등 기록)
          enum cache_request_status status; // [한국어] 캐시 응답 상태: HIT, MISS, RESERVATION_FAIL
          if (m_config->perfect_inst_const_cache) { // [한국어] 이상적 I-cache 모드: 무조건 HIT 처리
            status = HIT;
            shader_cache_access_log(m_sid, INSTRUCTION, 0); // [한국어] I-cache 접근 통계 기록
          } else
            // [한국어] 실제 L1I 캐시 접근: 캐시 히트/미스/예약 실패 판정
            status = m_L1I->access(
                (new_addr_type)ppc, mf,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);

          if (status == MISS) {
            // [한국어] I-cache MISS: mf를 L1I 내부에 보관하고(MSHR에 등록) 응답을 기다림
            m_last_warp_fetched = warp_id;            // [한국어] 마지막으로 fetch 시도한 warp 갱신
            m_warp[warp_id]->set_imiss_pending();     // [한국어] 이 warp의 I-miss pending 플래그 설정 — 응답 전까지 fetch에서 다시 선택 안 됨
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle); // [한국어] 마지막 fetch 시간 기록
          } else if (status == HIT) {
            // [한국어] I-cache HIT: 즉시 fetch buffer 채움 (다음 사이클에 decode 가능)
            m_last_warp_fetched = warp_id;
            m_inst_fetch_buffer = ifetch_buffer_t(pc, nbytes, warp_id); // [한국어] fetch buffer를 PTX PC, 크기, warp ID로 초기화
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
            delete mf; // [한국어] HIT의 경우 캐시가 데이터를 제공했으므로 mf 해제
          } else {
            // [한국어] RESERVATION_FAIL: MSHR이 꽉 차서 이 miss를 처리할 수 없음.
            // mf를 해제하고 다음 사이클에 다시 시도. m_last_warp_fetched는 갱신하여 다른 warp 시도 가능.
            m_last_warp_fetched = warp_id;
            assert(status == RESERVATION_FAIL); // [한국어] HIT, MISS, RESERVATION_FAIL 이외의 상태는 없어야 함
            delete mf; // [한국어] RESERVATION_FAIL 시 mf 해제 (캐시에 저장되지 않음)
          }
          break; // [한국어] 이 사이클에 하나의 warp만 fetch — break로 루프 종료
        }
      }
    }
  }

  m_L1I->cycle(); // [한국어] L1I 캐시 내부 사이클 진행: miss queue에서 응답 대기, fill pipeline 진행
}

/*
 * [한국어]
 * exec_shader_core_ctx::func_exec_inst — warp 명령어 기능 시뮬레이션 실행
 *
 * @inst: 실행할 warp 명령어 (warp_inst_t 참조)
 * @return: void
 *
 * issue_warp()에서 호출되어 명령어를 기능적으로 실행(레지스터 값 업데이트, PC 진행)한다.
 * 메모리 명령어(load/store)의 경우 generate_mem_accesses()로 실제 메모리 접근 목록을 생성.
 * 이 함수는 execution-driven 모드 전용 — trace-driven 모드는 별도 구현.
 *
 * 동작:
 *   1. execute_warp_inst_t(inst): 모든 활성 스레드에 대해 PTX 명령어를 기능적으로 실행
 *      (레지스터 파일 업데이트, 술어 레지스터 설정, PC 업데이트 등)
 *   2. 메모리 명령어면 generate_mem_accesses()로 coalesced 접근 목록 생성
 *      이 목록(inst.m_accessq)이 ldst_unit에서 순서대로 처리됨
 *
 * 호출 체인: shader_core_ctx::issue_warp() → func_exec_inst() → execute_warp_inst_t() [cuda-sim]
 */
void exec_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  execute_warp_inst_t(inst); // [한국어] cuda-sim/instructions.cc의 PTX 명령어 시맨틱 실행 — 레지스터 값과 PC를 기능적으로 업데이트
  if (inst.is_load() || inst.is_store()) { // [한국어] 메모리 명령어인 경우 접근 목록 생성 필요
    inst.generate_mem_accesses(); // [한국어] 32개 스레드의 주소를 코얼레싱하여 실제 메모리 요청 목록(m_accessq) 생성
    // inst.print_m_accessq();    // [한국어] 디버그용 접근 큐 출력 (주석 처리됨)
  }
}

/*
 * [한국어]
 * shader_core_ctx::issue_warp — warp 명령어를 파이프라인 레지스터에 발행
 *
 * @pipe_reg_set: 명령어가 들어갈 파이프라인 레지스터 집합 (ID_OC_SP, ID_OC_SFU 등)
 * @next_inst:    발행할 명령어 포인터 (warp I-buffer에서 가져온 정적 명령어)
 * @active_mask:  이 명령어를 실행할 스레드 마스크 (SIMT 스택에서 가져옴)
 * @warp_id:      발행하는 warp ID
 * @sch_id:       이 발행을 수행하는 스케줄러 ID (sub_core_model에서 슬롯 선택)
 * @return:       void
 *
 * 스케줄러가 발행 가능한 warp와 명령어를 결정한 후 호출된다.
 * 동작:
 *   1. pipe_reg_set에서 여유 레지스터 슬롯 확보
 *   2. I-buffer에서 명령어 소비 (ibuffer_free)
 *   3. 정적 명령어 정보를 파이프라인 레지스터에 복사
 *   4. 동적 정보 설정: active_mask, warp_id, 사이클, dynamic_warp_id
 *   5. 기능 시뮬레이션 실행 (func_exec_inst)
 *   6. LDGSTS(비동기 global→shared 복사) 버퍼 관리
 *   7. 배리어/메모리 배리어 처리
 *   8. SIMT 스택 업데이트 (분기 시 스택 push/pop)
 *   9. 스코어보드에 목적 레지스터 예약
 *   10. warp의 다음 PC 업데이트
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → func_exec_inst() → updateSIMTStack() → m_scoreboard->reserveRegisters()
 */
void shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                 const warp_inst_t *next_inst,
                                 const active_mask_t &active_mask,
                                 unsigned warp_id, unsigned sch_id) {
  warp_inst_t **pipe_reg =
      pipe_reg_set.get_free(m_config->sub_core_model, sch_id); // [한국어] sub_core_model: sch_id에 해당하는 슬롯만 탐색. 일반 모드: 임의의 여유 슬롯.
  assert(pipe_reg); // [한국어] 여유 슬롯이 없으면 assert — 스케줄러가 has_free 확인 후 호출하므로 정상적으론 실패 안 함

  m_warp[warp_id]->ibuffer_free(); // [한국어] I-buffer에서 이 명령어 소비 — 스케줄러가 다음 명령어를 읽을 수 있도록
  assert(next_inst->valid()); // [한국어] 발행할 명령어가 유효한지 확인
  **pipe_reg = *next_inst;  // static instruction information
                            // [한국어] 정적 명령어 정보(opcode, operands, latency 등)를 파이프라인 레지스터에 복사
  (*pipe_reg)->issue(
      active_mask, warp_id, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
      m_warp[warp_id]->get_dynamic_warp_id(), sch_id,
      m_warp[warp_id]->get_streamID());  // dynamic instruction information
      // [한국어] 동적 정보 설정: 활성 마스크, warp ID, 발행 사이클, 동적 warp ID, 스케줄러 ID, 스트림 ID
  m_stats->shader_cycle_distro[2 + (*pipe_reg)->active_count()]++; // [한국어] warp 점유율 분포 통계: active_count()개 스레드가 활성인 warp 발행 카운트
  func_exec_inst(**pipe_reg); // [한국어] 기능 시뮬레이션 실행: 레지스터 업데이트, 메모리 접근 목록 생성

  // Add LDGSTS instructions into a buffer
  // [한국어] LDGSTS(비동기 global→shared 로드): Ampere+ 아키텍처의 비동기 메모리 복사 명령어.
  // m_ldgdepbar_buf에 그룹별로 추적하여 DEPBAR 명령어가 의존하는 LDGSTS 완료를 감시.
  unsigned int ldgdepbar_id = m_warp[warp_id]->m_ldgdepbar_id; // [한국어] 현재 LDGDEPBAR 그룹 ID (LDGDEPBAR 명령어마다 증가)
  if (next_inst->m_is_ldgsts) { // [한국어] LDGSTS 명령어인 경우 추적 버퍼에 추가
    if (m_warp[warp_id]->m_ldgdepbar_buf.size() == ldgdepbar_id + 1) {
      m_warp[warp_id]->m_ldgdepbar_buf[ldgdepbar_id].push_back(*next_inst); // [한국어] 기존 그룹에 이 LDGSTS 추가
    } else {
      assert(m_warp[warp_id]->m_ldgdepbar_buf.size() < ldgdepbar_id + 1); // [한국어] 버퍼 크기가 예상보다 작아야 함 (새 그룹을 만들어야 하는 상황)
      std::vector<warp_inst_t> l;
      l.push_back(*next_inst);
      m_warp[warp_id]->m_ldgdepbar_buf.push_back(l); // [한국어] 새 그룹 생성 후 이 LDGSTS 추가
    }
    // If the mask of the instruction is all 0, then the address is also 0,
    // so that there's no need to check through the writeback
    // [한국어] 활성 마스크가 모두 0이면 실제 메모리 접근이 없으므로 pc=-1로 마킹하여 완료로 간주
    if (next_inst->get_active_mask() == 0) {
      (m_warp[warp_id]->m_ldgdepbar_buf.back()).back().pc = -1;
    }
  }

  if (next_inst->op == BARRIER_OP) {
    // [한국어] __syncthreads() 또는 명시적 barrier.sync — warp가 배리어에 도달했음을 기록
    m_warp[warp_id]->store_info_of_last_inst_at_barrier(*pipe_reg); // [한국어] 배리어 instruction 정보 저장 (barrier reduction에서 복원용)
    m_barriers.warp_reaches_barrier(m_warp[warp_id]->get_cta_id(), warp_id,
                                    const_cast<warp_inst_t *>(next_inst)); // [한국어] barrier_set_t에 이 warp가 배리어에 도달했음을 알림 — 모든 warp 도달 시 해제

  } else if (next_inst->op == MEMORY_BARRIER_OP) {
    // [한국어] membar / fence 명령어: 이 warp의 모든 메모리 요청이 완료될 때까지 진행 차단
    m_warp[warp_id]->set_membar(); // [한국어] m_membar 플래그 설정 — warp_waiting_at_mem_barrier()에서 확인
  } else if (next_inst->m_is_ldgdepbar) {  // Add for LDGDEPBAR
    // [한국어] LDGDEPBAR: 새로운 비동기 복사 그룹 시작 마커 — m_ldgdepbar_id 증가
    m_warp[warp_id]->m_ldgdepbar_id++;
    // If there are no added LDGSTS, insert an empty vector
    // [한국어] 이 그룹에 LDGSTS가 없으면 빈 벡터를 삽입하여 그룹 인덱스 일관성 유지
    if (m_warp[warp_id]->m_ldgdepbar_buf.size() != ldgdepbar_id + 1) {
      assert(m_warp[warp_id]->m_ldgdepbar_buf.size() < ldgdepbar_id + 1);
      std::vector<warp_inst_t> l;
      m_warp[warp_id]->m_ldgdepbar_buf.push_back(l); // [한국어] 빈 그룹 삽입
    }
  } else if (next_inst->m_is_depbar) {  // Add for DEPBAR
    // Set to true immediately when a DEPBAR instruction is met
    // [한국어] DEPBAR: 특정 그룹의 LDGSTS가 모두 완료될 때까지 warp를 대기시키는 명령어
    m_warp[warp_id]->m_waiting_ldgsts = true; // [한국어] 즉시 대기 상태로 설정 — warp::waiting()에서 확인
    m_warp[warp_id]->m_depbar_group =
        next_inst->m_depbar_group_no;  // set in trace_driven.cc
        // [한국어] 이 DEPBAR가 모니터링하는 그룹 수 (몇 개의 이전 그룹을 기다릴지)

    // Record the last group that's possbily being monitored by this DEPBAR
    // instr
    // [한국어] 이 DEPBAR가 감시할 수 있는 최대 그룹 ID (현재 ldgdepbar_id - 1)
    m_warp[warp_id]->m_depbar_start_id = m_warp[warp_id]->m_ldgdepbar_id - 1;

    // Record the last group that's actually being monitored by this DEPBAR
    // instr
    // [한국어] 실제로 감시하는 마지막 그룹 ID: ldgdepbar_id - depbar_group
    unsigned int end_group =
        m_warp[warp_id]->m_ldgdepbar_id - m_warp[warp_id]->m_depbar_group;

    // Check for the case that the LDGSTSs monitored have finished when
    // encountering the DEPBAR instruction
    // [한국어] DEPBAR 발행 시점에 이미 모든 감시 대상 LDGSTS가 완료된 경우를 확인
    bool done_flag = true;
    for (int i = 0; i < end_group; i++) { // [한국어] 감시 그룹 범위 내 모든 LDGSTS 확인
      for (int j = 0; j < m_warp[warp_id]->m_ldgdepbar_buf[i].size(); j++) {
        if (m_warp[warp_id]->m_ldgdepbar_buf[i][j].pc != -1) { // [한국어] pc != -1이면 아직 완료되지 않은 LDGSTS 존재
          done_flag = false;
          goto UpdateDEPBAR; // [한국어] 완료되지 않은 LDGSTS 발견 — 조기 종료
        }
      }
    }

  UpdateDEPBAR:
    if (done_flag) { // [한국어] 모든 LDGSTS가 이미 완료된 경우 즉시 대기 해제
      if (m_warp[warp_id]->m_waiting_ldgsts) {
        m_warp[warp_id]->m_waiting_ldgsts = false; // [한국어] 대기 플래그 해제 — 이 warp가 바로 계속 실행 가능
      }
    }
  }

  updateSIMTStack(warp_id, *pipe_reg); // [한국어] SIMT 스택 업데이트: 분기 명령어면 분기 마스크를 push, 재합류 지점이면 pop

  m_scoreboard->reserveRegisters(*pipe_reg); // [한국어] 목적 레지스터를 스코어보드에 예약 — 완료 전까지 다른 명령어가 이 레지스터를 읽으려 하면 RAW 해저드 감지
  m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->isize); // [한국어] warp의 다음 PC를 현재 명령어 PC + 크기로 업데이트 (분기 명령어의 경우 func_exec_inst에서 이미 변경됨)
}

/*
 * [한국어]
 * shader_core_ctx::issue — 모든 스케줄러를 라운드로빈으로 호출하는 발행 스테이지
 *
 * @return: void
 *
 * 매 사이클 호출되어 SM에 연결된 모든 scheduler_unit들을 공정하게 실행한다.
 * Issue_Prio를 사용한 라운드로빈 방식으로 어떤 스케줄러도 매 사이클 먼저 기회를 독점하지 않도록 보장.
 * 각 스케줄러의 cycle()이 실제 warp 선택 및 issue_warp() 호출을 담당.
 * gpgpu_num_sched_per_core 개수만큼의 스케줄러가 있으며, 각각 독립적으로 warp를 관리.
 *
 * 호출 체인: shader_core_ctx::cycle() → issue() → scheduler_unit::cycle() → issue_warp()
 */
void shader_core_ctx::issue() {
  // Ensure fair round robin issu between schedulers
  // [한국어] 스케줄러 간 공정한 라운드로빈 발행 순서 보장 — 매 사이클 시작 스케줄러를 변경
  unsigned j;
  for (unsigned i = 0; i < schedulers.size(); i++) {
    j = (Issue_Prio + i) % schedulers.size(); // [한국어] Issue_Prio를 시작점으로 하는 라운드로빈 인덱스
    schedulers[j]->cycle(); // [한국어] j번 스케줄러의 warp 선택 및 발행 사이클 실행
  }
  Issue_Prio = (Issue_Prio + 1) % schedulers.size(); // [한국어] 다음 사이클의 우선 스케줄러 순환 (1씩 증가)

  // really is issue;
  // for (unsigned i = 0; i < schedulers.size(); i++) {
  //    schedulers[i]->cycle();
  //}
}

/*
 * [한국어]
 * scheduler_unit::warp — 특정 warp_id의 shd_warp_t 참조 반환
 *
 * @i:      warp ID
 * @return: 해당 warp의 shd_warp_t 참조
 *
 * m_warp 포인터 배열에서 i번째 warp 객체를 참조로 반환하는 단순 접근자.
 * 스케줄러가 warp 상태를 확인할 때(done_exit, waiting, ibuffer 등) 사용.
 *
 * 호출 체인: scheduler_unit::order_rrr(), cycle() 등 내부에서 직접 호출
 */
shd_warp_t &scheduler_unit::warp(int i) { return *((*m_warp)[i]); } // [한국어] (*m_warp)[i]: m_warp는 shd_warp_t* 벡터 포인터 — 이중 역참조로 i번째 warp 반환

/*
 * [한국어]
 * scheduler_unit::order_lrr — Loose Round Robin 방식으로 warp 우선순위 목록 작성
 *
 * @result_list:            [출력] 우선순위 순서로 채워진 warp 목록
 * @input_list:             정렬할 warp 목록 (m_supervised_warps)
 * @last_issued_from_input: 직전 사이클에 마지막으로 발행한 warp 이터레이터
 * @num_warps_to_add:       result_list에 추가할 warp 수
 *
 * Loose Round Robin(LRR): last_issued_from_input 다음부터 순환하여 num_warps_to_add개를 result_list에 채운다.
 * 발행하지 못한 warp도 계속 RR 순환에 포함되어, 어떤 warp도 영원히 굶지 않음(starvation-free).
 * "Loose"는 발행 여부와 관계없이 매 사이클 다음 warp로 넘어가는 특성을 의미.
 *
 * 호출 체인: lrr_scheduler::order_warps() → [이 함수]
 */

/**
 * A general function to order things in a Loose Round Robin way. The simplist
 * use of this function would be to implement a loose RR scheduler between all
 * the warps assigned to this core. A more sophisticated usage would be to order
 * a set of "fetch groups" in a RR fashion. In the first case, the templated
 * class variable would be a simple unsigned int representing the warp_id.  In
 * the 2lvl case, T could be a struct or a list representing a set of warp_ids.
 * @param result_list: The resultant list the caller wants returned.  This list
 * is cleared and then populated in a loose round robin way
 * @param input_list: The list of things that should be put into the
 * result_list. For a simple scheduler this can simply be the m_supervised_warps
 * list.
 * @param last_issued_from_input:  An iterator pointing the last member in the
 * input_list that issued. Since this function orders in a RR fashion, the
 * object pointed to by this iterator will be last in the prioritization list
 * @param num_warps_to_add: The number of warps you want the scheudler to pick
 * between this cycle. Normally, this will be all the warps availible on the
 * core, i.e. m_supervised_warps.size(). However, a more sophisticated scheduler
 * may wish to limit this number. If the number if < m_supervised_warps.size(),
 * then only the warps with highest RR priority will be placed in the
 * result_list.
 */
template <class T>
void scheduler_unit::order_lrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  assert(num_warps_to_add <= input_list.size()); // [한국어] 요청 수가 전체 warp 수를 초과하면 안 됨
  result_list.clear(); // [한국어] 이전 사이클의 우선순위 목록 초기화
  typename std::vector<T>::const_iterator iter =
      (last_issued_from_input == input_list.end()) ? input_list.begin()
                                                   : last_issued_from_input + 1;
  // [한국어] 시작 이터레이터: 이전 발행 warp의 다음부터 (end이면 처음부터 = 초기화 상태)

  for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
    if (iter == input_list.end()) {
      iter = input_list.begin(); // [한국어] 리스트 끝 도달 시 처음으로 순환 (라운드로빈)
    }
    result_list.push_back(*iter); // [한국어] 라운드로빈 순서로 result_list에 추가
  }
}

/*
 * [한국어]
 * scheduler_unit::order_rrr — Rigid Round Robin 방식으로 warp 우선순위 목록 작성
 *
 * @result_list:            [출력] 우선순위 순서로 채워진 warp 목록 (최대 1개)
 * @input_list:             전체 warp 목록
 * @last_issued_from_input: 직전에 발행한 warp 이터레이터
 * @num_warps_to_add:       추가할 warp 수 (보통 1)
 *
 * Rigid Round Robin(RRR): 현재 turn의 warp가 발행(또는 done/waiting)했을 때만 다음 warp로 넘어감.
 * "Rigid"는 발행 성공 여부에 따라 turn이 변경되는 특성 — 발행 실패 시 같은 warp를 계속 시도.
 * m_current_turn_warp: 현재 turn인 warp ID를 추적.
 *
 * 호출 체인: rrr_scheduler::order_warps() → [이 함수]
 */
template <class T>
void scheduler_unit::order_rrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  result_list.clear(); // [한국어] 이전 사이클의 우선순위 목록 초기화

  if (m_num_issued_last_cycle > 0 || warp(m_current_turn_warp).done_exit() ||
      warp(m_current_turn_warp).waiting()) {
    // [한국어] 이전 사이클에 발행이 있었거나, 현재 turn warp가 종료/대기 중이면 다음 warp로 전환
    std::vector<shd_warp_t *>::const_iterator iter =
        (last_issued_from_input == input_list.end())
            ? input_list.begin()
            : last_issued_from_input + 1;
    for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
      if (iter == input_list.end()) {
        iter = input_list.begin(); // [한국어] 순환 처리
      }
      unsigned warp_id = (*iter)->get_warp_id();
      if (!(*iter)->done_exit() && !(*iter)->waiting()) {
        result_list.push_back(*iter); // [한국어] 종료/대기 상태가 아닌 첫 warp를 result_list에 추가
        m_current_turn_warp = warp_id; // [한국어] 이 warp로 현재 turn 변경
        break; // [한국어] RRR은 한 번에 하나의 warp만 선택
      }
    }
  } else {
    result_list.push_back(&warp(m_current_turn_warp)); // [한국어] 이전 사이클에 발행 없었고 turn warp가 유효하면 같은 warp 재선택
  }
}

/*
 * [한국어]
 * scheduler_unit::order_by_priority — 우선순위 함수 기반으로 warp 순서 결정
 *
 * @result_list:            [출력] 우선순위 순서로 채워진 warp 목록
 * @input_list:             정렬할 warp 목록
 * @last_issued_from_input: 직전에 발행한 warp 이터레이터 (GREEDY 모드에서 사용)
 * @num_warps_to_add:       result_list에 추가할 warp 수
 * @ordering:               정렬 방식 enum (ORDERING_GREEDY_THEN_PRIORITY_FUNC | ORDERED_PRIORITY_FUNC_ONLY)
 * @priority_func:          두 warp를 비교하는 함수 포인터 (예: oldest dynamic_warp_id)
 *
 * GTO(Greedy-Then-Oldest) 모드: 마지막 발행 warp를 먼저 배치하고 나머지를 우선순위 정렬.
 * PRIORITY_ONLY 모드: 전체를 우선순위 함수로 정렬하여 배치.
 * gto_scheduler와 oldest_scheduler에서 사용.
 *
 * 호출 체인: gto_scheduler::order_warps() / oldest_scheduler::order_warps() → [이 함수]
 */

/**
 * A general function to order things in an priority-based way.
 * The core usage of the function is similar to order_lrr.
 * The explanation of the additional parameters (beyond order_lrr) explains the
 * further extensions.
 * @param ordering: An enum that determines how the age function will be treated
 * in prioritization see the definition of OrderingType.
 * @param priority_function: This function is used to sort the input_list.  It
 * is passed to stl::sort as the sorting fucntion. So, if you wanted to sort a
 * list of integer warp_ids with the oldest warps having the most priority, then
 * the priority_function would compare the age of the two warps.
 */
template <class T>
void scheduler_unit::order_by_priority(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add, OrderingType ordering,
    bool (*priority_func)(T lhs, T rhs)) {
  assert(num_warps_to_add <= input_list.size()); // [한국어] 요청 수 범위 검증
  result_list.clear(); // [한국어] 이전 우선순위 목록 초기화
  typename std::vector<T> temp = input_list; // [한국어] 입력 리스트 복사 (정렬을 위해)

  if (ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering) {
    // [한국어] GTO 방식: 현재 greedy warp(직전 발행)를 최우선으로 배치하고 나머지를 우선순위 정렬
    T greedy_value = *last_issued_from_input; // [한국어] 직전에 발행한 warp (greedy 선택)
    result_list.push_back(greedy_value); // [한국어] greedy warp를 result_list 첫 번째에 배치

    std::sort(temp.begin(), temp.end(), priority_func); // [한국어] 나머지를 우선순위 함수로 정렬 (예: dynamic_warp_id 오름차순)
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      if (*iter != greedy_value) {
        result_list.push_back(*iter); // [한국어] greedy warp 중복 제외하고 나머지 추가
      }
    }
  } else if (ORDERED_PRIORITY_FUNC_ONLY == ordering) {
    // [한국어] 순수 우선순위 정렬 방식 (OLDEST 스케줄러 등에서 사용)
    std::sort(temp.begin(), temp.end(), priority_func); // [한국어] 전체를 우선순위 함수로 정렬
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      result_list.push_back(*iter); // [한국어] 정렬된 순서로 num_warps_to_add개 추가
    }
  } else {
    fprintf(stderr, "Unknown ordering - %d\n", ordering); // [한국어] 알 수 없는 정렬 방식 — 프로그래밍 오류
    abort();
  }
}

/*
 * [한국어]
 * scheduler_unit::cycle — warp 스케줄러의 핵심 사이클 함수 (발행 로직)
 *
 * @return: void
 *
 * 매 사이클 shader_core_ctx::issue()에서 호출되어 이 스케줄러가 담당하는 warp 중
 * 발행 가능한 warp를 선택하고 적절한 실행 유닛으로 명령어를 발행한다.
 *
 * 동작 단계:
 *   1. order_warps() — 스케줄러 정책(LRR/GTO/RRR 등)에 따라 warp 우선순위 목록 작성
 *   2. 우선순위 순서로 warp 순회하여 발행 가능 여부 검사:
 *      - waiting() 체크: 배리어/membar/LDGSTS 대기 중이면 건너뜀
 *      - ibuffer_empty() 체크: 명령어 없으면 건너뜀
 *      - 제어 해저드 체크: SIMT 스택 PC와 ibuffer PC 불일치 시 ibuffer flush
 *      - 스코어보드 충돌 체크: RAW 해저드 있으면 건너뜀
 *   3. 명령어 타입에 따라 해당 실행 유닛의 파이프라인 레지스터로 issue_warp() 호출:
 *      - LOAD/STORE/MEM_BARRIER → m_mem_out (ldst_unit)
 *      - SP_OP/ALU_OP (INT unit 없으면) → m_sp_out (sp_unit)
 *      - INT_OP (Volta+ INT unit 있으면) → m_int_out (int_unit)
 *      - DP_OP (DP unit 있으면) → m_dp_out, 없으면 → m_sfu_out
 *      - SFU_OP/ALU_SFU_OP → m_sfu_out (sfu)
 *      - TENSOR_CORE_OP → m_tensor_core_out
 *      - SPEC_UNIT_START_ID 이상 → m_spec_cores_out[spec_id]
 *   4. 발행 성공 시 m_last_supervised_issued 갱신, 통계 업데이트, 루프 탈출
 *   5. 발행 실패 이유를 shader_cycle_distro에 기록
 *
 * dual issue: gpgpu_max_insn_issue_per_warp > 1이면 같은 warp에서 최대 2개 발행 가능.
 *             gpgpu_dual_issue_diff_exec_units: Maxwell/Pascal처럼 서로 다른 유닛에만 듀얼 발행 허용.
 *
 * 호출 체인: shader_core_ctx::issue() → [이 함수] → issue_warp()
 */
void scheduler_unit::cycle() {
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  // [한국어] valid_inst: 제어 해저드 없이 유효한 명령어가 있었는지 (ibuffer에 정상 명령어 존재)
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  // [한국어] ready_inst: 스코어보드를 통과한 (RAW 해저드 없는) 명령어가 있었는지
  bool issued_inst = false;  // of these we issued one
  // [한국어] issued_inst: 실제로 실행 유닛으로 발행에 성공했는지

  order_warps(); // [한국어] 스케줄러 정책에 따라 m_next_cycle_prioritized_warps 우선순위 목록 갱신
  for (std::vector<shd_warp_t *>::const_iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end(); iter++) {
    // Don't consider warps that are not yet valid
    // [한국어] NULL이거나 이미 done_exit(완전 종료)된 warp는 건너뜀
    if ((*iter) == NULL || (*iter)->done_exit()) {
      continue;
    }
    SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n",
                  (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    unsigned warp_id = (*iter)->get_warp_id(); // [한국어] 현재 검사 중인 warp의 하드웨어 슬롯 ID
    unsigned checked = 0;  // [한국어] 이번 사이클에 이 warp에서 검사한 명령어 수
    unsigned issued = 0;   // [한국어] 이번 사이클에 이 warp에서 실제로 발행한 명령어 수
    exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE; // [한국어] 직전에 발행한 명령어의 실행 유닛 타입 (dual issue 시 유닛 중복 방지용)
    unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp; // [한국어] 한 warp에서 한 사이클에 최대 발행 가능 명령어 수 (보통 1, dual issue 시 2)
    bool diff_exec_units =
        m_shader->m_config
            ->gpgpu_dual_issue_diff_exec_units;  // In tis mode, we only allow
                                                 // dual issue to diff execution
                                                 // units (as in Maxwell and
                                                 // Pascal)
    // [한국어] diff_exec_units: true면 두 명령어가 서로 다른 실행 유닛으로만 dual issue 허용 (Maxwell/Pascal 모델)

    if (warp(warp_id).ibuffer_empty())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    if (warp(warp_id).waiting())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for "
          "barrier\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    // [한국어] warp 발행 내부 루프: 한 warp에서 최대 max_issue개 명령어를 연속 발행하려 시도
    // checked: 검사한 수, issued: 발행 성공한 수. checked <= issued 조건은 발행 실패 시 루프 탈출.
    while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() &&
           (checked < max_issue) && (checked <= issued) &&
           (issued < max_issue)) {
      const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst(); // [한국어] I-buffer에서 다음 발행할 명령어 포인터 가져오기 (소비 X)
      // Jin: handle cdp latency;
      // [한국어] CDP(CUDA Dynamic Parallelism) 레이턴시 처리: cudaLaunchDevice 명령어 발행 후 latency 사이클 대기
      if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
        assert(warp(warp_id).m_cdp_dummy); // [한국어] CDP latency 대기 중에는 m_cdp_dummy=true 상태여야 함
        warp(warp_id).m_cdp_latency--; // [한국어] 레이턴시 카운트다운
        break; // [한국어] 이 사이클에는 더 이상 발행 불가
      }

      bool valid = warp(warp_id).ibuffer_next_valid(); // [한국어] I-buffer의 다음 슬롯이 유효한지 확인
      bool warp_inst_issued = false; // [한국어] 이 명령어가 발행에 성공했는지 추적
      unsigned pc, rpc;
      m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc); // [한국어] SIMT 스택 상단의 실제 PC와 reconvergence PC 조회
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
          m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc)
              .c_str());
      if (pI) {
        assert(valid); // [한국어] 명령어 포인터가 있으면 valid flag도 true여야 함
        if (pc != pI->pc) {
          SCHED_DPRINTF(
              "Warp (warp_id %u, dynamic_warp_id %u) control hazard "
              "instruction flush\n",
              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          // control hazard
          // [한국어] 제어 해저드: SIMT 스택의 실제 PC와 I-buffer의 명령어 PC가 다름.
          // 분기 결과로 SIMT 스택 PC가 바뀐 후 ibuffer가 아직 갱신 안 된 상태.
          warp(warp_id).set_next_pc(pc); // [한국어] warp의 다음 PC를 SIMT 스택의 PC로 수정
          warp(warp_id).ibuffer_flush(); // [한국어] 잘못된 명령어가 있는 I-buffer 플러시 — 다음 사이클에 올바른 PC로 fetch 재시도
        } else {
          valid_inst = true; // [한국어] 유효한 명령어 존재 확인 (제어 해저드 없음)
          if (!m_scoreboard->checkCollision(warp_id, pI)) {
            // [한국어] 스코어보드 통과: 이 명령어의 소스 레지스터에 대해 pending write가 없음 (RAW 해저드 없음)
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
            ready_inst = true; // [한국어] 스코어보드 통과한 명령어 존재

            const active_mask_t &active_mask =
                m_shader->get_active_mask(warp_id, pI); // [한국어] SIMT 스택에서 현재 활성 스레드 마스크 가져오기

            assert(warp(warp_id).inst_in_pipeline()); // [한국어] 파이프라인 내 명령어 수가 0보다 커야 함 (decode에서 증가)

            if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) ||
                (pI->op == MEMORY_BARRIER_OP) ||
                (pI->op == TENSOR_CORE_LOAD_OP) ||
                (pI->op == TENSOR_CORE_STORE_OP)) {
              // [한국어] 메모리 명령어(load/store/membar): ldst_unit(m_mem_out)으로 발행
              if (m_mem_out->has_free(m_shader->m_config->sub_core_model,
                                      m_id) &&
                  (!diff_exec_units ||
                   previous_issued_inst_exec_type != exec_unit_type_t::MEM)) {
                // [한국어] m_mem_out에 여유 슬롯이 있고, diff_exec_units 모드면 직전 발행이 MEM이 아닌 경우에만 발행
                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id,
                                     m_id); // [한국어] ldst_unit 파이프라인 레지스터로 명령어 발행
                issued++;
                issued_inst = true;
                warp_inst_issued = true;
                previous_issued_inst_exec_type = exec_unit_type_t::MEM; // [한국어] 이 발행의 유닛 타입 기록
              }
            } else {
              // This code need to be refactored
              // [한국어] ALU/SFU/DP/Tensor/Specialized 명령어: 타입에 따라 적절한 실행 유닛으로 발행
              if (pI->op != TENSOR_CORE_OP && pI->op != SFU_OP &&
                  pI->op != DP_OP && !(pI->op >= SPEC_UNIT_START_ID)) {
                // [한국어] SP(단정밀도FP)/INT/ALU 명령어: SP unit 또는 INT unit으로 발행
                bool execute_on_SP = false;
                bool execute_on_INT = false;

                bool sp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sp_units > 0) &&
                    m_sp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id); // [한국어] SP unit 파이프라인에 여유 슬롯이 있는지
                bool int_pipe_avail =
                    (m_shader->m_config->gpgpu_num_int_units > 0) &&
                    m_int_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id); // [한국어] INT unit 파이프라인에 여유 슬롯이 있는지 (Volta+ 전용)

                // if INT unit pipline exist, then execute ALU and INT
                // operations on INT unit and SP-FPU on SP unit (like in Volta)
                // if INT unit pipline does not exist, then execute all ALU, INT
                // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                // [한국어] INT unit이 존재하고(Volta+), SP 명령어가 아니고, INT pipe에 여유가 있으면 INT unit 사용
                if (m_shader->m_config->gpgpu_num_int_units > 0 &&
                    int_pipe_avail && pI->op != SP_OP &&
                    !(diff_exec_units &&
                      previous_issued_inst_exec_type == exec_unit_type_t::INT))
                  execute_on_INT = true;
                else if (sp_pipe_avail &&
                         (m_shader->m_config->gpgpu_num_int_units == 0 ||
                          (m_shader->m_config->gpgpu_num_int_units > 0 &&
                           pI->op == SP_OP)) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::SP))
                  execute_on_SP = true; // [한국어] INT unit 없거나(Fermi/Pascal) SP 전용 명령어면 SP unit 사용

                if (execute_on_INT || execute_on_SP) {
                  // Jin: special for CDP api
                  // [한국어] CDP(CUDA Dynamic Parallelism) 명령어 특수 처리:
                  // cudaLaunchDevice 호출은 m_cdp_latency 사이클 동안 파이프라인을 점유
                  if (pI->m_is_cdp && !warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0); // [한국어] 새 CDP 호출이면 latency가 0이어야 함
                    if (pI->m_is_cdp == 1)
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1]; // [한국어] cudaLaunchDeviceV2 이전 API의 latency
                    else  // cudaLaunchDeviceV2 and cudaGetParameterBufferV2
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1] +
                          m_shader->m_config->gpgpu_ctx->func_sim
                                  ->cdp_latency[pI->m_is_cdp] *
                              active_mask.count(); // [한국어] cudaLaunchDeviceV2: 활성 스레드 수에 비례하는 latency 추가
                    warp(warp_id).m_cdp_dummy = true; // [한국어] dummy 플래그 설정: latency 소진 후 실제 발행을 위한 두 번째 사이클 표시
                    break; // [한국어] 이 사이클에 발행하지 않고 latency 대기
                  } else if (pI->m_is_cdp && warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0); // [한국어] latency 소진 후 dummy 상태 해제
                    warp(warp_id).m_cdp_dummy = false; // [한국어] dummy 플래그 해제: 실제 발행 진행
                  }
                }

                if (execute_on_SP) {
                  m_shader->issue_warp(*m_sp_out, pI, active_mask, warp_id,
                                       m_id); // [한국어] SP unit으로 발행: 단정밀도 FP 및 ALU 명령어
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SP;
                } else if (execute_on_INT) {
                  m_shader->issue_warp(*m_int_out, pI, active_mask, warp_id,
                                       m_id); // [한국어] INT unit으로 발행: Volta+ 정수 연산 전용 파이프라인
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::INT;
                }
              } else if ((m_shader->m_config->gpgpu_num_dp_units > 0) &&
                         (pI->op == DP_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::DP)) {
                // [한국어] DP unit이 별도 존재하고(Volta+), 배정밀도 명령어면 DP unit으로 발행
                bool dp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_dp_units > 0) &&
                    m_dp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id); // [한국어] DP unit 파이프라인 여유 슬롯 확인

                if (dp_pipe_avail) {
                  m_shader->issue_warp(*m_dp_out, pI, active_mask, warp_id,
                                       m_id); // [한국어] DP unit으로 배정밀도 FP 명령어 발행
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::DP;
                }
              }  // If the DP units = 0 (like in Fermi archi), then execute DP
                 // inst on SFU unit
                 // [한국어] DP unit 없거나(Fermi) SFU 명령어면 SFU unit으로 발행
              else if (((m_shader->m_config->gpgpu_num_dp_units == 0 &&
                         pI->op == DP_OP) ||
                        (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) &&
                       !(diff_exec_units && previous_issued_inst_exec_type ==
                                                exec_unit_type_t::SFU)) {
                bool sfu_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sfu_units > 0) &&
                    m_sfu_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id); // [한국어] SFU 파이프라인 여유 슬롯 확인

                if (sfu_pipe_avail) {
                  m_shader->issue_warp(*m_sfu_out, pI, active_mask, warp_id,
                                       m_id); // [한국어] SFU unit으로 발행: sin/cos/sqrt/rcp 등 초월함수 및 DP_OP(Fermi)
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SFU;
                }
              } else if ((pI->op == TENSOR_CORE_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::TENSOR)) {
                // [한국어] Tensor Core 명령어(wmma, mma 등): m_tensor_core_out으로 발행
                bool tensor_core_pipe_avail =
                    (m_shader->m_config->gpgpu_num_tensor_core_units > 0) &&
                    m_tensor_core_out->has_free(
                        m_shader->m_config->sub_core_model, m_id); // [한국어] Tensor Core 파이프라인 여유 슬롯 확인

                if (tensor_core_pipe_avail) {
                  m_shader->issue_warp(*m_tensor_core_out, pI, active_mask,
                                       warp_id, m_id); // [한국어] Tensor Core unit으로 행렬 곱 명령어 발행
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;
                }
              } else if ((pI->op >= SPEC_UNIT_START_ID) &&
                         !(diff_exec_units &&
                           previous_issued_inst_exec_type ==
                               exec_unit_type_t::SPECIALIZED)) {
                // [한국어] 특수화 유닛(specialized unit) 명령어: op에서 SPEC_UNIT_START_ID를 빼서 유닛 인덱스 계산
                unsigned spec_id = pI->op - SPEC_UNIT_START_ID; // [한국어] 특수 유닛 ID (0부터 시작)
                assert(spec_id < m_shader->m_config->m_specialized_unit.size()); // [한국어] 유효한 특수 유닛 ID인지 확인
                register_set *spec_reg_set = m_spec_cores_out[spec_id]; // [한국어] 해당 특수 유닛의 파이프라인 레지스터 집합
                bool spec_pipe_avail =
                    (m_shader->m_config->m_specialized_unit[spec_id].num_units >
                     0) &&
                    spec_reg_set->has_free(m_shader->m_config->sub_core_model,
                                           m_id); // [한국어] 해당 특수 유닛 파이프라인 여유 슬롯 확인

                if (spec_pipe_avail) {
                  m_shader->issue_warp(*spec_reg_set, pI, active_mask, warp_id,
                                       m_id); // [한국어] 특수화 유닛으로 명령어 발행
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type =
                      exec_unit_type_t::SPECIALIZED;
                }
              }

            }  // end of else
          } else {
            // [한국어] 스코어보드 충돌: RAW 해저드로 인해 이 명령어 발행 불가 (소스 레지스터에 pending write 존재)
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          }
        }
      } else if (valid) {
        // this case can happen after a return instruction in diverged warp
        // [한국어] pI=NULL(I-buffer 비어있음)이지만 valid=true인 경우: diverged warp에서 return 이후 발생 가능.
        // SIMT 스택 PC로 warp PC를 강제 수정하고 ibuffer 플러시.
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp "
            "flush\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
        warp(warp_id).set_next_pc(pc); // [한국어] SIMT 스택 PC로 다음 PC 수정
        warp(warp_id).ibuffer_flush(); // [한국어] ibuffer 플러시하여 올바른 PC에서 다시 fetch
      }
      if (warp_inst_issued) {
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), issued);
        do_on_warp_issued(warp_id, issued, iter); // [한국어] 발행 완료 후처리: 통계 업데이트, ibuffer 전진, 스케줄러별 후처리
      }
      checked++; // [한국어] 검사한 명령어 수 증가 (dual issue 루프 제어)
    }
    if (issued) {
      // This might be a bit inefficient, but we need to maintain
      // two ordered list for proper scheduler execution.
      // We could remove the need for this loop by associating a
      // supervised_is index with each entry in the
      // m_next_cycle_prioritized_warps vector. For now, just run through until
      // you find the right warp_id
      // [한국어] 발행 성공 시: m_last_supervised_issued를 발행된 warp의 이터레이터로 갱신
      // m_supervised_warps와 m_next_cycle_prioritized_warps가 별도로 유지되어 선형 탐색 필요
      for (std::vector<shd_warp_t *>::const_iterator supervised_iter =
               m_supervised_warps.begin();
           supervised_iter != m_supervised_warps.end(); ++supervised_iter) {
        if (*iter == *supervised_iter) {
          m_last_supervised_issued = supervised_iter; // [한국어] 마지막으로 발행한 warp의 supervised 이터레이터 갱신 (LRR 다음 사이클 시작점)
        }
      }
      m_num_issued_last_cycle = issued; // [한국어] 이번 사이클 발행 수 기록 (RRR 스케줄러에서 turn 전환 판단)
      if (issued == 1)
        m_stats->single_issue_nums[m_id]++; // [한국어] 단일 발행 통계 (dual issue 분석용)
      else if (issued > 1)
        m_stats->dual_issue_nums[m_id]++;   // [한국어] 듀얼 발행 통계
      else
        abort();  // issued should be > 0

      break; // [한국어] 이번 사이클에 한 warp에서 발행 성공 — 더 이상 다른 warp 탐색 불필요
    }
  }

  // issue stall statistics:
  // [한국어] 발행 실패 원인 분류 통계 (AerialVision 및 성능 분석에서 사용)
  if (!valid_inst)
    m_stats->shader_cycle_distro[0]++;  // idle or control hazard
    // [한국어] 인덱스 0: 모든 warp가 ibuffer 비어있거나 제어 해저드 — 유휴 사이클
  else if (!ready_inst)
    m_stats->shader_cycle_distro[1]++;  // waiting for RAW hazards (possibly due
                                        // to memory)
    // [한국어] 인덱스 1: 유효한 명령어가 있으나 스코어보드 충돌(RAW 해저드) — 메모리 지연 가능성 높음
  else if (!issued_inst)
    m_stats->shader_cycle_distro[2]++;  // pipeline stalled
    // [한국어] 인덱스 2: 스코어보드 통과했으나 실행 유닛 파이프라인 꽉 참(structural hazard)
}

/*
 * [한국어]
 * scheduler_unit::do_on_warp_issued — warp 발행 완료 후처리
 *
 * @warp_id:         발행한 warp ID
 * @num_issued:      이번 사이클에 발행한 명령어 수 (1 또는 2)
 * @prioritized_iter: m_next_cycle_prioritized_warps에서 이 warp의 이터레이터
 * @return:          void
 *
 * 통계 기록(event_warp_issued)과 I-buffer 전진(ibuffer_step)을 수행.
 * 서브클래스(two_level_active_scheduler)에서 오버라이드하여 추가 처리 가능.
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수]
 */
void scheduler_unit::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<shd_warp_t *>::const_iterator &prioritized_iter) {
  m_stats->event_warp_issued(m_shader->get_sid(), warp_id, num_issued,
                             warp(warp_id).get_dynamic_warp_id()); // [한국어] 발행 이벤트를 통계에 기록 (warp 슬롯별/동적 warp별 발행 분포 히스토그램)
  warp(warp_id).ibuffer_step(); // [한국어] I-buffer 포인터를 다음 슬롯으로 이동 (발행된 명령어 소비 완료)
}

/*
 * [한국어]
 * scheduler_unit::sort_warps_by_oldest_dynamic_id — warp 비교 함수 (오래된 warp 우선)
 *
 * @lhs: 왼쪽 warp 포인터
 * @rhs: 오른쪽 warp 포인터
 * @return: lhs가 rhs보다 높은 우선순위이면 true
 *
 * dynamic_warp_id가 작을수록 오래된 warp (CTA 배치 순서로 단조 증가).
 * done_exit/waiting 상태의 warp는 항상 낮은 우선순위로 밀려남.
 * GTO 및 Oldest 스케줄러의 우선순위 정렬 기준으로 사용.
 *
 * 호출 체인: gto_scheduler::order_warps() / oldest_scheduler::order_warps() → order_by_priority() → [이 함수]
 */
bool scheduler_unit::sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs,
                                                     shd_warp_t *rhs) {
  if (rhs && lhs) {
    if (lhs->done_exit() || lhs->waiting()) {
      return false; // [한국어] lhs가 종료/대기 중이면 rhs를 우선하는 방향으로 비교 (false)
    } else if (rhs->done_exit() || rhs->waiting()) {
      return true;  // [한국어] rhs가 종료/대기 중이면 lhs가 우선 (true)
    } else {
      return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id(); // [한국어] 동적 warp ID가 작을수록(오래된 warp) 우선
    }
  } else {
    return lhs < rhs; // [한국어] NULL 포인터 처리: 포인터 값 비교 (실제로는 발생하지 않아야 함)
  }
}

/*
 * [한국어]
 * lrr_scheduler::order_warps — LRR 방식으로 m_next_cycle_prioritized_warps 갱신
 *
 * 감독하는 모든 warp를 LRR 순서로 정렬.
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → order_lrr()
 */
void lrr_scheduler::order_warps() {
  order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size()); // [한국어] 마지막 발행 warp 다음부터 시작하는 LRR 순서로 전체 supervised warp 정렬
}

/*
 * [한국어]
 * rrr_scheduler::order_warps — RRR 방식으로 m_next_cycle_prioritized_warps 갱신
 *
 * 발행 성공 시에만 다음 warp로 turn을 넘기는 Rigid RR 방식.
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → order_rrr()
 */
void rrr_scheduler::order_warps() {
  order_rrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size()); // [한국어] 현재 turn의 warp가 발행/완료/대기인 경우에만 다음 warp로 전환
}

/*
 * [한국어]
 * gto_scheduler::order_warps — GTO(Greedy-Then-Oldest) 방식으로 우선순위 목록 갱신
 *
 * 직전 발행 warp를 최우선으로 하고, 나머지를 오래된 순으로 정렬.
 * 메모리 지연이 없는 warp를 계속 실행하여 처리량 극대화.
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → order_by_priority()
 */
void gto_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id); // [한국어] 직전 greedy warp 우선 + 나머지 oldest dynamic_warp_id 순
}

/*
 * [한국어]
 * oldest_scheduler::order_warps — Oldest 방식으로 우선순위 목록 갱신
 *
 * 전체 warp를 동적 warp ID 오름차순으로 정렬 (순수 oldest-first).
 * GTO와 달리 greedy 우선 없이 항상 가장 오래된 warp가 우선.
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → order_by_priority()
 */
void oldest_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERED_PRIORITY_FUNC_ONLY,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id); // [한국어] 전체 warp를 오래된 순(dynamic_warp_id 오름차순)으로 정렬
}

/*
 * [한국어]
 * two_level_active_scheduler::do_on_warp_issued — 2레벨 스케줄러의 발행 후처리
 *
 * @warp_id:         발행한 warp ID
 * @num_issued:      발행한 명령어 수
 * @prioritized_iter: 발행한 warp의 이터레이터
 * @return:          void
 *
 * 기반 클래스 do_on_warp_issued를 호출한 후, active 레벨 목록을 내부 우선순위 정책으로 재정렬.
 * 현재는 m_inner_level_prioritization == LRR만 지원 (발행한 warp 다음부터 재정렬).
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수]
 */
void two_level_active_scheduler::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<shd_warp_t *>::const_iterator &prioritized_iter) {
  scheduler_unit::do_on_warp_issued(warp_id, num_issued, prioritized_iter); // [한국어] 기반 클래스 처리: 통계 기록, ibuffer_step
  if (SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization) {
    std::vector<shd_warp_t *> new_active;
    order_lrr(new_active, m_next_cycle_prioritized_warps, prioritized_iter,
              m_next_cycle_prioritized_warps.size()); // [한국어] 발행한 warp 다음부터 LRR로 active 목록 재정렬
    m_next_cycle_prioritized_warps = new_active; // [한국어] 재정렬된 목록을 active 목록으로 교체
  } else {
    fprintf(stderr, "Unimplemented m_inner_level_prioritization: %d\n",
            m_inner_level_prioritization);
    abort();
  }
}

/*
 * [한국어]
 * two_level_active_scheduler::order_warps — 2레벨 스케줄러 우선순위 목록 갱신
 *
 * 두 단계로 warp를 관리:
 *   1. Active 레벨 (m_next_cycle_prioritized_warps): 즉시 발행 가능한 warp 집합 (최대 m_max_active_warps)
 *   2. Pending 레벨 (m_pending_warps): 배리어/long-latency op 대기 중인 warp 집합
 *
 * 매 사이클:
 *   - Active 레벨에서 waiting 상태 warp를 Pending으로 demote
 *   - Pending 레벨에서 대기 해제된 warp를 Active로 promote
 *
 * m_outer_level_prioritization으로 Pending→Active 승격 순서를 결정 (현재 SRR만 지원).
 * m_inner_level_prioritization으로 Active 내 발행 순서를 결정 (LRR만 지원).
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수]
 */
void two_level_active_scheduler::order_warps() {
  // Move waiting warps to m_pending_warps
  // [한국어] 현재 active 집합에서 waiting 상태의 warp를 pending 큐로 강등(demote)
  unsigned num_demoted = 0;
  for (std::vector<shd_warp_t *>::iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end();) {
    bool waiting = (*iter)->waiting(); // [한국어] 배리어/membar 대기 여부 확인
    for (int i = 0; i < MAX_INPUT_VALUES; i++) {
      const warp_inst_t *inst = (*iter)->ibuffer_next_inst();
      // Is the instruction waiting on a long operation?
      // [한국어] 다음 명령어의 소스 레지스터가 long-latency op(메모리 load 등)에 의존하는지 확인
      if (inst && inst->in[i] > 0 &&
          this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in[i])) {
        waiting = true; // [한국어] long op 의존 시 대기 상태로 간주 — pending으로 강등
      }
    }

    if (waiting) {
      m_pending_warps.push_back(*iter); // [한국어] pending 큐로 이동
      iter = m_next_cycle_prioritized_warps.erase(iter); // [한국어] active 집합에서 제거하고 다음 이터레이터 반환
      SCHED_DPRINTF("DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                    (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
      ++num_demoted;
    } else {
      ++iter;
    }
  }

  // If there is space in m_next_cycle_prioritized_warps, promote the next
  // m_pending_warps
  unsigned num_promoted = 0;
  if (SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization) {
    while (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
      m_pending_warps.pop_front();
      SCHED_DPRINTF(
          "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
          (m_next_cycle_prioritized_warps.back())->get_warp_id(),
          (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id());
      ++num_promoted;
    }
  } else {
    fprintf(stderr, "Unimplemented m_outer_level_prioritization: %d\n",
            m_outer_level_prioritization);
    abort();
  }
  assert(num_promoted == num_demoted); // [한국어] demote된 수만큼만 promote해야 active 크기 일정 유지
}

/*
 * [한국어]
 * swl_scheduler::swl_scheduler — Warp Limiting 스케줄러 생성자
 *
 * @config_string: "warp_limiting:<prioritization>:<num_warps_to_limit>" 형식의 설정 문자열
 *                 gpgpusim.config의 -gpgpu_sched 옵션에서 전달
 *
 * SWL(Swapping Warp Limiting) 스케줄러: 한 사이클에 고려하는 warp 수를 제한하여
 * 실제 HW의 warp 점유율(occupancy) 영향을 시뮬레이션. 현재는 GTO 우선순위만 지원.
 *
 * 호출 체인: shader_core_ctx::create_schedulers() → [이 생성자]
 */
swl_scheduler::swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_string)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out, mem_out,
                     id) {
  unsigned m_prioritization_readin;
  int ret = sscanf(config_string, "warp_limiting:%d:%d",
                   &m_prioritization_readin, &m_num_warps_to_limit); // [한국어] config 문자열에서 우선순위 타입과 warp 제한 수 파싱
  assert(2 == ret); // [한국어] 정확히 2개 파라미터가 파싱되어야 함
  m_prioritization = (scheduler_prioritization_type)m_prioritization_readin; // [한국어] enum으로 형변환
  // Currently only GTO is implemented
  assert(m_prioritization == SCHEDULER_PRIORITIZATION_GTO); // [한국어] 현재는 GTO만 구현됨
  assert(m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader); // [한국어] 제한 warp 수가 최대 warp 수를 초과하면 안 됨
}

/*
 * [한국어]
 * swl_scheduler::order_warps — Warp Limiting + GTO 방식으로 우선순위 목록 갱신
 *
 * m_supervised_warps 중 m_num_warps_to_limit개만 고려하여 GTO 정렬.
 * warp 점유율을 제한함으로써 레지스터/공유메모리 압력을 조절하는 효과.
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수] → order_by_priority()
 */
void swl_scheduler::order_warps() {
  if (SCHEDULER_PRIORITIZATION_GTO == m_prioritization) {
    order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                      m_last_supervised_issued,
                      MIN(m_num_warps_to_limit, m_supervised_warps.size()), // [한국어] 최대 m_num_warps_to_limit개 warp만 고려 (warp limiting의 핵심)
                      ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                      scheduler_unit::sort_warps_by_oldest_dynamic_id); // [한국어] GTO 정렬: 제한된 warp 집합 내에서 greedy-then-oldest
  } else {
    fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
    abort();
  }
}

/*
 * [한국어]
 * shader_core_ctx::read_operands — 오퍼랜드 콜렉터 스텝 (Read Operands 스테이지)
 *
 * @return: void
 *
 * 매 사이클 레지스터 파일 포트 처리량(reg_file_port_throughput)만큼 오퍼랜드 콜렉터를 전진.
 * 오퍼랜드 콜렉터(opndcoll_rfu_t)는 레지스터 파일 뱅크 충돌을 처리하고,
 * 준비된 collector unit(CU)을 실행 유닛(OC_EX 레지스터)으로 dispatch.
 *
 * 호출 체인: shader_core_ctx::cycle() → read_operands() → m_operand_collector.step()
 */
void shader_core_ctx::read_operands() {
  for (unsigned int i = 0; i < m_config->reg_file_port_throughput; ++i)
    m_operand_collector.step(); // [한국어] 오퍼랜드 콜렉터 1스텝 전진: bank 할당, 오퍼랜드 읽기, dispatch
}

/*
 * [한국어]
 * coalesced_segment — 메모리 주소가 속하는 세그먼트 번호 계산
 *
 * @addr:                    메모리 주소
 * @segment_size_lg2bytes:   세그먼트 크기의 log2 (예: 128B 세그먼트 → 7)
 * @return:                  addr이 속하는 세그먼트 번호 (addr >> log2_size)
 *
 * 여러 스레드의 메모리 주소가 같은 세그먼트에 속하는지 판별할 때 사용.
 * abstract_hardware_model.cc의 coalescing 로직에서도 동일한 계산을 사용.
 *
 * 호출 체인: ldst_unit::memory_cycle() 등 메모리 coalescing 과정에서 사용
 */
address_type coalesced_segment(address_type addr,
                               unsigned segment_size_lg2bytes) {
  return (addr >> segment_size_lg2bytes); // [한국어] 주소를 세그먼트 크기로 나눈 몫 = 세그먼트 번호
}

/*
 * [한국어]
 * shader_core_ctx::translate_local_memaddr — 로컬 메모리 주소를 타이밍 시뮬레이션용 선형 주소로 변환
 *
 * @localaddr:          스레드 기준의 로컬 메모리 주소 (기능 시뮬레이션에서 사용)
 * @tid:                전역 스레드 ID (SM 내 상대 ID가 아님)
 * @num_shader:         전체 SM 수 (GPU 전체에서 고유한 주소 공간 생성용)
 * @datasize:           접근 데이터 크기 (바이트)
 * @translated_addrs:   [출력] 변환된 선형 주소 배열 (최대 MAX_ACCESSES_PER_INSN_PER_THREAD)
 * @return:             생성된 접근 수 (datasize / 4, 또는 sub-4B이면 1)
 *
 * 기능 시뮬레이션에서는 각 스레드가 독립적인 로컬 메모리 공간을 가지지만,
 * 타이밍 시뮬레이션에서는 이를 단일 공유 주소 공간으로 매핑해야 코얼레싱 분석이 가능.
 *
 * gpgpu_local_mem_map=1 (새 방식): 같은 CTA 내 스레드가 연속한 주소를 가지도록 배치,
 * 이후 SM별, CTA별로 분산 — 다른 SM의 같은 tid가 인접하여 메모리 뱅크 분산 효과.
 *
 * gpgpu_local_mem_map=0 (레거시): 같은 주소의 로컬 변수가 모든 스레드에서 연속 배치.
 *
 * 호출 체인: ldst_unit::memory_cycle() → [이 함수] → translated_addrs로 mem_access_t 생성
 */
// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned shader_core_ctx::translate_local_memaddr(
    address_type localaddr, unsigned tid, unsigned num_shader,
    unsigned datasize, new_addr_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0; // [한국어] 이 tid의 로컬 메모리 선형 기준 주소
  unsigned max_concurrent_threads = 0; // [한국어] 전체 GPU에서 동시에 실행 가능한 최대 스레드 수 (주소 공간 크기 결정)
  if (m_config->gpgpu_local_mem_map) {
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to contiguous
    // addresses, then distribute across memory space by CTAs from successive
    // shader cores first, then by successive CTA in same shader core
    // [한국어] 새 매핑: 같은 CTA 내 스레드가 연속 주소 → SM별, CTA별로 분산
    thread_base =
        4 * (kernel_padded_threads_per_cta *
                 (m_sid + num_shader * (tid / kernel_padded_threads_per_cta)) +
             tid % kernel_padded_threads_per_cta);
    // [한국어] 공식: 4 * (pad * (SM_id + num_SM * cta_id_in_SM) + thread_in_cta)
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader; // [한국어] 전체 GPU 동시 스레드 수
  } else {
    // legacy mapping that maps the same address in the local memory space of
    // all threads to a single contiguous address region
    // [한국어] 레거시 매핑: 같은 주소의 로컬 변수가 모든 스레드에서 연속 배치 (단순하지만 뱅크 충돌 가능)
    thread_base = 4 * (m_config->n_thread_per_shader * m_sid + tid); // [한국어] SM 내 스레드 순서에 따른 선형 기준 주소
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads); // [한국어] 주소가 유효한 범위 내에 있는지 확인

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4; // [한국어] 4B 단위 접근 수: 8B → 2, 16B → 4, 128B → 32
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(
        localaddr % 4 ==
        0);  // Address must be 4B aligned - required if accessing 4B per
             // request, otherwise access will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i; // [한국어] i번째 4B 워드의 워드 인덱스
      address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    thread_base + LOCAL_GENERIC_START; // [한국어] 전체 GPU 선형 주소: 워드*전체스레드*4B + 스레드기준 + LOCAL_GENERIC_START 오프셋
      translated_addrs[i] = linear_address; // [한국어] i번째 접근의 변환된 주소
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4; // [한국어] 이 주소가 속한 4B 워드 인덱스
    address_type local_word_offset = localaddr % 4; // [한국어] 워드 내 바이트 오프셋 (0~3)
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B chunk
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  local_word_offset + thread_base +
                                  LOCAL_GENERIC_START; // [한국어] sub-4B 접근: 워드 내 오프셋도 포함하여 주소 계산
    translated_addrs[0] = linear_address;
  }
  return num_accesses; // [한국어] 생성된 4B 단위 접근 수 반환 (caller가 이 수만큼 mem_access_t 생성)
}

/////////////////////////////////////////////////////////////////////////////////////////
/*
 * [한국어]
 * shader_core_ctx::test_res_bus — result bus 예약 가능 여부 확인
 *
 * @latency: 이 명령어의 실행 레이턴시 (사이클 수, 비트 인덱스로 사용)
 * @return:  사용 가능한 result bus 인덱스 (0 이상), 모두 사용 중이면 -1
 *
 * result_bus는 MAX_ALU_LATENCY 크기의 bitset으로, 각 비트가 해당 레이턴시 후에
 * writeback이 있는지를 나타냄. latency 위치의 비트가 0이어야 예약 가능.
 * 여러 개의 result bus(num_result_bus)가 있어 동시 writeback 처리 가능.
 *
 * 호출 체인: shader_core_ctx::execute() → [이 함수]
 */
int shader_core_ctx::test_res_bus(int latency) {
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) { // [한국어] m_result_bus[i]의 latency 비트가 0이면 이 버스 사용 가능
      return i; // [한국어] 사용 가능한 result bus 인덱스 반환
    }
  }
  return -1; // [한국어] 모든 result bus가 이 레이턴시 후에 이미 사용 예정 — structural hazard
}

/*
 * [한국어]
 * shader_core_ctx::execute — 실행 유닛 스테이지 (Execute / Writeback 파이프라인)
 *
 * @return: void
 *
 * 매 사이클 호출되어:
 *   1. result_bus 비트셋 오른쪽 시프트 (1사이클 경과 — writeback 타이머 진행)
 *   2. 각 함수 유닛(m_fu[n])의 cycle() 호출 (내부 파이프라인 진행, clock_multiplier만큼 반복)
 *   3. 각 함수 유닛의 active_lanes_in_pipeline() 호출 (AccelWattch 전력 통계 업데이트)
 *   4. 각 함수 유닛의 issue_port에서 ready 명령어가 있으면 해당 유닛으로 이동:
 *      - stallable FU (ldst_unit): result bus 예약 없이 바로 issue (자체 writeback 관리)
 *      - non-stallable FU (ALU/SFU 등): result bus에 latency 위치 예약 후 issue
 *
 * result_bus.set(latency): latency 사이클 후에 writeback이 있다는 표시.
 * MAX_ALU_LATENCY: result_bus의 최대 추적 레이턴시 (비트셋 크기).
 *
 * 호출 체인: shader_core_ctx::cycle() → execute() → m_fu[n]->issue()
 */
void shader_core_ctx::execute() {
  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1; // [한국어] 모든 result bus의 비트를 1씩 오른쪽 시프트: 1사이클 경과로 타이머 진행
  }
  for (unsigned n = 0; n < m_num_function_units; n++) {
    unsigned multiplier = m_fu[n]->clock_multiplier(); // [한국어] 이 FU의 클럭 배율 (ldst_unit은 더 느린 메모리 클럭 사용)
    for (unsigned c = 0; c < multiplier; c++) m_fu[n]->cycle(); // [한국어] FU 내부 사이클 진행 (multiplier만큼 반복)
    m_fu[n]->active_lanes_in_pipeline(); // [한국어] 이 FU에서 실행 중인 활성 레인 수 업데이트 (AccelWattch 전력 모델)
    unsigned issue_port = m_issue_port[n]; // [한국어] 이 FU가 명령어를 받는 파이프라인 레지스터 인덱스 (ID_OC_* 단계)
    register_set &issue_inst = m_pipeline_reg[issue_port]; // [한국어] 이 FU의 입력 파이프라인 레지스터 집합
    unsigned reg_id;
    bool partition_issue =
        m_config->sub_core_model && m_fu[n]->is_issue_partitioned(); // [한국어] sub_core_model: 스케줄러별로 별도 슬롯 할당
    if (partition_issue) {
      reg_id = m_fu[n]->get_issue_reg_id(); // [한국어] sub_core_model에서 이 FU가 담당하는 스케줄러 슬롯 ID
    }
    warp_inst_t **ready_reg = issue_inst.get_ready(partition_issue, reg_id); // [한국어] 이 FU에서 처리 가능한 준비된 명령어 포인터
    if (issue_inst.has_ready(partition_issue, reg_id) &&
        m_fu[n]->can_issue(**ready_reg)) { // [한국어] 준비된 명령어가 있고 이 FU가 받을 수 있는지 확인
      bool schedule_wb_now = !m_fu[n]->stallable(); // [한국어] stallable=false(ALU/SFU): result bus 예약 필요. true(ldst_unit): 자체 writeback 관리
      int resbus = -1;
      if (schedule_wb_now &&
          (resbus = test_res_bus((*ready_reg)->latency)) != -1) {
        // [한국어] non-stallable FU이고 result bus 예약 가능: issue 진행
        assert((*ready_reg)->latency < MAX_ALU_LATENCY); // [한국어] 레이턴시가 result_bus 크기 이내인지 확인
        m_result_bus[resbus]->set((*ready_reg)->latency); // [한국어] result bus에 latency 위치 비트 설정 (이 사이클 후 latency 사이클에 writeback 예약)
        m_fu[n]->issue(issue_inst); // [한국어] FU로 명령어 이동 (issue_inst에서 FU 내부 파이프라인으로)
      } else if (!schedule_wb_now) {
        m_fu[n]->issue(issue_inst); // [한국어] stallable FU(ldst_unit): result bus 예약 없이 바로 issue
      } else {
        // stall issue (cannot reserve result bus)
        // [한국어] result bus가 꽉 참: 이 사이클에 issue 불가 (structural hazard — execute stage stall)
      }
    }
  }
}

/*
 * [한국어]
 * ldst_unit::print_cache_stats — L1D 캐시 통계를 파일에 출력
 *
 * @fp:            출력 파일 포인터
 * @dl1_accesses:  [입출력] L1D 전체 접근 수 누적
 * @dl1_misses:    [입출력] L1D 미스 수 누적
 * @return:        void
 *
 * 호출 체인: shader_core_ctx::print_cache_stats() → [이 함수] → m_L1D->print()
 */
void ldst_unit::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                  unsigned &dl1_misses) {
  if (m_L1D) {
    m_L1D->print(fp, dl1_accesses, dl1_misses); // [한국어] L1D 데이터 캐시 통계 출력 (m_L1D가 없으면 no-op)
  }
}

/*
 * [한국어]
 * ldst_unit::get_cache_stats — L1D/L1C/L1T 캐시 통계를 cache_stats 객체에 누적
 *
 * @cs:   [입출력] 통계를 누적할 cache_stats 객체
 * @return: void
 *
 * 호출 체인: shader_core_ctx::get_cache_stats() → [이 함수]
 */
void ldst_unit::get_cache_stats(cache_stats &cs) {
  // Adds stats to 'cs' from each cache
  if (m_L1D) cs += m_L1D->get_stats(); // [한국어] L1D(데이터) 캐시 통계 누적
  if (m_L1C) cs += m_L1C->get_stats(); // [한국어] L1C(상수) 캐시 통계 누적
  if (m_L1T) cs += m_L1T->get_stats(); // [한국어] L1T(텍스처) 캐시 통계 누적
}

/*
 * [한국어]
 * ldst_unit::get_L1D_sub_stats / get_L1C_sub_stats / get_L1T_sub_stats
 * — 개별 캐시의 세부 통계(hit/miss/MSHR 등)를 cache_sub_stats 구조체에 반환
 *
 * @css:    [입출력] 세부 통계를 채울 cache_sub_stats 구조체
 * @return: void
 *
 * 호출 체인: shader_core_ctx::get_L1D/L1C/L1T_sub_stats() → [이 함수]
 */
void ldst_unit::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1D) m_L1D->get_sub_stats(css); // [한국어] L1D 세부 통계 반환 (없으면 no-op)
}
void ldst_unit::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1C) m_L1C->get_sub_stats(css); // [한국어] L1C(상수 캐시) 세부 통계 반환
}
void ldst_unit::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1T) m_L1T->get_sub_stats(css); // [한국어] L1T(텍스처 캐시) 세부 통계 반환
}

/*
 * [한국어]
 * shader_core_ctx::unset_depbar — LDGSTS 완료 시 DEPBAR 대기 상태 해제
 *
 * @inst: writeback 완료된 LDGSTS 명령어
 * @return: void
 *
 * LDGSTS가 writeback(메모리에서 공유 메모리로 데이터 복사 완료)될 때 호출.
 * m_ldgdepbar_buf에서 해당 LDGSTS의 pc를 -1로 마킹하여 완료 표시.
 * 감시 중인 그룹의 모든 LDGSTS가 완료되었으면 m_waiting_ldgsts=false로 해제하여
 * DEPBAR 이후의 명령어 발행을 허용.
 *
 * 로직:
 *   1. pc와 주소(get_addr(0))가 일치하는 LDGSTS 버퍼 항목을 찾아 pc=-1 마킹 (DoneWB)
 *      (같은 PC의 LDGSTS가 여러 개일 수 있으므로 주소로 구분)
 *   2. end_group 이내의 모든 LDGSTS가 완료됐는지 확인 (UpdateDEPBAR)
 *   3. 모두 완료됐으면 m_waiting_ldgsts=false 해제
 *
 * 호출 체인: ldst_unit::process_cache_access() → m_core->unset_depbar(inst)
 */
// Add this function to unset depbar
void shader_core_ctx::unset_depbar(const warp_inst_t &inst) {
  bool done_flag = true; // [한국어] 감시 그룹의 모든 LDGSTS가 완료됐는지 추적
  // [한국어] end_group: DEPBAR가 실제로 감시하는 마지막 그룹까지의 범위 계산
  // m_depbar_start_id가 0이면 전체 버퍼를 검사, 아니면 start_id - group + 1까지만 검사
  unsigned int end_group = m_warp[inst.warp_id()]->m_depbar_start_id == 0
                               ? m_warp[inst.warp_id()]->m_ldgdepbar_buf.size()
                               : (m_warp[inst.warp_id()]->m_depbar_start_id -
                                  m_warp[inst.warp_id()]->m_depbar_group + 1);

  if (inst.m_is_ldgsts) { // [한국어] writeback된 명령어가 LDGSTS인 경우에만 처리
    for (int i = 0; i < m_warp[inst.warp_id()]->m_ldgdepbar_buf.size(); i++) { // [한국어] 모든 그룹 순회
      for (int j = 0; j < m_warp[inst.warp_id()]->m_ldgdepbar_buf[i].size();
           j++) {
        if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc == inst.pc) {
          // Handle the case that same pc results in multiple LDGSTS
          // instructions
          // [한국어] 같은 PC에서 여러 LDGSTS가 발행될 수 있으므로 주소로도 구분
          if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].get_addr(0) ==
              inst.get_addr(0)) {
            m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc = -1; // [한국어] 완료 마킹: pc=-1로 설정
            goto DoneWB; // [한국어] 일치 항목을 찾았으므로 조기 종료
          }
        }
      }
    }

  DoneWB:
    // [한국어] 감시 그룹 범위 내 모든 LDGSTS가 완료됐는지 재확인
    for (int i = 0; i < end_group; i++) {
      for (int j = 0; j < m_warp[inst.warp_id()]->m_ldgdepbar_buf[i].size();
           j++) {
        if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc != -1) {
          done_flag = false; // [한국어] 아직 완료 안 된 LDGSTS가 있음
          goto UpdateDEPBAR; // [한국어] 조기 종료: 아직 해제 불가
        }
      }
    }

  UpdateDEPBAR:
    if (done_flag) { // [한국어] 모든 LDGSTS 완료 → DEPBAR 대기 해제
      if (m_warp[inst.warp_id()]->m_waiting_ldgsts) {
        m_warp[inst.warp_id()]->m_waiting_ldgsts = false; // [한국어] 대기 플래그 해제 — 이 warp의 DEPBAR 이후 명령어 발행 허용
      }
    }
  }
}

/*
 * [한국어]
 * shader_core_ctx::warp_inst_complete — 완료된 warp 명령어 통계 기록
 *
 * @inst: 완료된 warp 명령어 (writeback 단계에서 호출)
 * @return: void
 *
 * writeback 단계에서 명령어가 완료될 때 호출되어 각종 통계를 업데이트.
 * AccelWattch 전력 추정을 위한 파이프라인별 완료 카운터와 IPC 계산용 시뮬레이션 명령어 수를 기록.
 * gpu_clock_gated_lanes: true이면 실제 활성 스레드 수만 카운트 (비활성 레인은 클럭 게이팅).
 *
 * 호출 체인: shader_core_ctx::writeback() → [이 함수]
 */
void shader_core_ctx::warp_inst_complete(const warp_inst_t &inst) {
#if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu \n",
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc,  m_gpu->gpu_tot_sim_cycle +  m_gpu->gpu_sim_cycle);
#endif
  if (inst.op_pipe == SP__OP)
    m_stats->m_num_sp_committed[m_sid]++; // [한국어] SP unit 완료 카운터 (AccelWattch 전력 모델용)
  else if (inst.op_pipe == SFU__OP)
    m_stats->m_num_sfu_committed[m_sid]++; // [한국어] SFU unit 완료 카운터
  else if (inst.op_pipe == MEM__OP)
    m_stats->m_num_mem_committed[m_sid]++; // [한국어] MEM unit 완료 카운터

  if (m_config->gpgpu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sid] += m_config->warp_size; // [한국어] 클럭 게이팅 없음: warp_size(32)개 스레드 모두 카운트
  else
    m_stats->m_num_sim_insn[m_sid] += inst.active_count(); // [한국어] 클럭 게이팅: 실제 활성 스레드 수만 카운트 (발산 warp 분석 가능)

  m_stats->m_num_sim_winsn[m_sid]++; // [한국어] warp 명령어 완료 수 증가 (IPC 계산용)
  m_gpu->gpu_sim_insn += inst.active_count(); // [한국어] GPU 전체 시뮬레이션 명령어 수 증가
  inst.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle); // [한국어] 명령어 완료 사이클 기록 (trace/디버그용)
}

/*
 * [한국어]
 * shader_core_ctx::writeback — Writeback 스테이지: 레지스터 파일에 결과 쓰기
 *
 * @return: void
 *
 * 매 사이클 호출되어 EX_WB 파이프라인 레지스터에 있는 완료된 명령어를 처리:
 *   1. 파이프라인 duty cycle 통계 계산
 *   2. EX_WB에서 준비된 명령어를 가져와 루프:
 *      a. m_operand_collector.writeback(): 오퍼랜드 콜렉터의 대기 할당 해제
 *      b. m_scoreboard->releaseRegisters(): 스코어보드에서 목적 레지스터 예약 해제 (다른 명령어 read 허용)
 *      c. m_warp->dec_inst_in_pipeline(): 파이프라인 내 명령어 수 감소
 *      d. warp_inst_complete(): 통계 업데이트
 *      e. pipe_reg->clear(): EX_WB 레지스터 초기화 (다음 명령어 수용)
 *   3. 스코어보드 해제로 인해 대기 중이던 warp가 다음 사이클에 발행 가능해짐
 *
 * 오퍼랜드 콜렉터 writeback이 스톨을 일으킬 수 있지만, 파이프라인이 unstallable이므로
 * 반환값을 무시하여 스톨 없이 진행.
 *
 * 호출 체인: shader_core_ctx::cycle() → writeback() → m_scoreboard->releaseRegisters()
 */
void shader_core_ctx::writeback() {
  // [한국어] 파이프라인 duty cycle: 실제 완료된 스레드 명령어 수 / 이론적 최대 처리량
  unsigned max_committed_thread_instructions =
      m_config->warp_size *
      (m_config->pipe_widths[EX_WB]);  // from the functional units
      // [한국어] 최대 처리량: warp_size × EX_WB 파이프 너비(병렬 writeback 포트 수)
  m_stats->m_pipeline_duty_cycle[m_sid] =
      ((float)(m_stats->m_num_sim_insn[m_sid] -
               m_stats->m_last_num_sim_insn[m_sid])) /
      max_committed_thread_instructions; // [한국어] 이번 사이클의 duty cycle 갱신 (0~1, 1이 최대 활용)

  m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid]; // [한국어] 다음 사이클 duty cycle 계산용 이전 값 갱신
  m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];

  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready(); // [한국어] EX_WB 파이프라인 레지스터에서 완료 준비된 명령어 포인터 가져오기
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
     * The operand collector writeback can generally generate a stall
     * However, here, the pipelines should be un-stallable. This is
     * guaranteed because this is the first time the writeback function
     * is called after the operand collector's step function, which
     * resets the allocations. There is one case which could result in
     * the writeback function returning false (stall), which is when
     * an instruction tries to modify two registers (GPR and predicate)
     * To handle this case, we ignore the return value (thus allowing
     * no stalling).
     */

    m_operand_collector.writeback(*pipe_reg); // [한국어] 오퍼랜드 콜렉터에 결과 쓰기 알림 (할당 해제, 대기 CU 해제)
    unsigned warp_id = pipe_reg->warp_id(); // [한국어] 완료된 명령어의 warp ID
    m_scoreboard->releaseRegisters(pipe_reg); // [한국어] 스코어보드에서 목적 레지스터 예약 해제 — 이후 이 레지스터를 소스로 하는 명령어 발행 가능
    m_warp[warp_id]->dec_inst_in_pipeline(); // [한국어] 파이프라인 내 명령어 수 감소 (decode에서 증가, writeback에서 감소)
    warp_inst_complete(*pipe_reg); // [한국어] 완료 통계 기록 (op_pipe별 카운터, IPC 통계)
    m_gpu->gpu_sim_insn_last_update_sid = m_sid; // [한국어] 마지막으로 명령어를 완료한 SM ID 기록
    m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle; // [한국어] 마지막 명령어 완료 사이클 기록
    m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle; // [한국어] 이 SM에서 마지막 명령어 완료 사이클
    m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle; // [한국어] 누적 사이클 포함 마지막 명령어 완료 시점
    pipe_reg->clear(); // [한국어] EX_WB 파이프라인 레지스터 초기화 — 다음 명령어 수용 준비
    preg = m_pipeline_reg[EX_WB].get_ready(); // [한국어] 다음 완료 준비된 명령어 조회 (EX_WB에 여러 슬롯이 있을 수 있음)
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}

/*
 * [한국어]
 * ldst_unit::shared_cycle — 공유 메모리 접근 처리 (Shared Memory Cycle)
 *
 * @inst:      처리할 warp 명령어 (shared_space load/store)
 * @rc_fail:   [출력] 실패 원인 (BK_CONF: 뱅크 충돌, NO_RC_FAIL: 성공)
 * @fail_type: [출력] 실패 메모리 유형 (S_MEM: 공유 메모리)
 * @return:    true = 이 사이클에 완료 (공유 메모리가 아닌 명령어 포함), false = 뱅크 충돌로 스톨
 *
 * 공유 메모리(shared memory) 접근을 처리하며, 뱅크 충돌을 시뮬레이션.
 * shared_space가 아닌 명령어는 즉시 true 반환 (다른 경로로 처리됨).
 * dispatch_delay(): 뱅크 충돌이 있으면 1 사이클 지연을 카운트다운하고 스톨 여부를 반환.
 * 실제 GPU에서 공유 메모리는 32개 뱅크(warp_size)로 구성되어, 같은 사이클에 같은 뱅크 접근 시 직렬화.
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수]
 */
bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                             mem_stage_access_type &fail_type) {
  if (inst.space.get_type() != shared_space) return true; // [한국어] 공유 메모리 명령어가 아니면 즉시 완료로 반환

  if (inst.active_count() == 0) return true; // [한국어] 활성 스레드가 없으면 no-op — 발산 warp에서 발생 가능

  if (inst.has_dispatch_delay()) {
    m_stats->gpgpu_n_shmem_bank_access[m_sid]++; // [한국어] 공유 메모리 뱅크 접근 카운터 증가
  }

  bool stall = inst.dispatch_delay(); // [한국어] 뱅크 충돌 레이턴시 처리: 충돌이 남아있으면 true(스톨), 해소되면 false(진행)
  if (stall) {
    fail_type = S_MEM; // [한국어] 실패 원인: 공유 메모리 뱅크 충돌
    rc_fail = BK_CONF; // [한국어] BK_CONF: Bank Conflict — 같은 뱅크에 여러 스레드가 동시 접근
    m_stats->gpgpu_n_shmem_bkconflict++; // [한국어] 공유 메모리 뱅크 충돌 통계 증가
  } else
    rc_fail = NO_RC_FAIL; // [한국어] 충돌 없음, 정상 완료
  return !stall; // [한국어] 스톨이면 false(이 사이클 미완료), 아니면 true(완료)
}

/*
 * [한국어]
 * ldst_unit::process_cache_access — 캐시 접근 결과 처리
 *
 * @cache:   접근한 캐시 (L1D/L1C/L1T)
 * @address: 접근 주소
 * @inst:    요청한 warp 명령어
 * @events:  캐시 접근으로 발생한 이벤트 목록 (write sent, read sent 등)
 * @mf:      이 접근용 mem_fetch 패킷
 * @status:  캐시 접근 결과 (HIT/MISS/HIT_RESERVED/RESERVATION_FAIL)
 * @return:  mem_stage_stall_type (NO_RC_FAIL/BK_CONF/COAL_STALL)
 *
 * 캐시 접근 후의 상태를 처리하는 공통 로직:
 *   - HIT: accessq에서 항목 제거, load면 pending_writes 감소. LDGSTS면 depbar 확인.
 *     write_sent 없으면 mf 즉시 해제.
 *   - RESERVATION_FAIL: MSHR/캐시 예약 불가 — BK_CONF 반환, mf 해제.
 *   - MISS/HIT_RESERVED: accessq에서 항목 제거하고 mf는 캐시 내부에서 관리.
 *   - accessq에 더 남아있으면 COAL_STALL 반환하여 다음 사이클에 계속 처리.
 *
 * 호출 체인: ldst_unit::process_memory_access_queue() / process_memory_access_queue_l1cache() → [이 함수]
 */
mem_stage_stall_type ldst_unit::process_cache_access(
    cache_t *cache, new_addr_type address, warp_inst_t &inst,
    std::list<cache_event> &events, mem_fetch *mf,
    enum cache_request_status status) {
  mem_stage_stall_type result = NO_RC_FAIL; // [한국어] 기본 반환값: 실패 없음
  bool write_sent = was_write_sent(events); // [한국어] 이 접근으로 캐시가 NoC로 쓰기 요청을 보냈는지
  bool read_sent = was_read_sent(events);   // [한국어] 이 접근으로 캐시가 NoC로 읽기 요청을 보냈는지
  if (write_sent) {
    // [한국어] 쓰기 요청이 발생한 경우: store_req 카운터 증가 (store_ack()에서 감소, warp 완료 판단에 사용)
    unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                           ? (mf->get_data_size() / SECTOR_SIZE)
                           : 1; // [한국어] SECTOR_ASSOC MSHR이면 섹터 단위로 ack 예상, 아니면 1개
    for (unsigned i = 0; i < inc_ack; ++i)
      m_core->inc_store_req(inst.warp_id()); // [한국어] 이 warp의 미완료 store 요청 수 증가
  }
  if (status == HIT) {
    // [한국어] 캐시 HIT: 즉시 데이터 제공됨
    assert(!read_sent); // [한국어] HIT이면 NoC로 읽기 요청을 보내지 않아야 함
    inst.accessq_pop_back(); // [한국어] 처리 완료한 접근 항목을 accessq에서 제거
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]]--; // [한국어] 이 load의 목적 레지스터에 대한 pending write 수 감소

      // release LDGSTS
      // [한국어] LDGSTS(비동기 global→shared 로드) 완료 처리: pending_ldgsts 카운터 감소
      if (inst.m_is_ldgsts) {
        m_pending_ldgsts[inst.warp_id()][inst.pc][inst.get_addr(0)]--;
        if (m_pending_ldgsts[inst.warp_id()][inst.pc][inst.get_addr(0)] == 0) {
          m_core->unset_depbar(inst); // [한국어] 이 LDGSTS 주소의 모든 접근이 완료됐으면 depbar 해제 시도
        }
      }
    }
    if (!write_sent) delete mf; // [한국어] HIT이고 write_sent도 없으면 mf 즉시 해제 (캐시가 소유권을 가져가지 않음)
  } else if (status == RESERVATION_FAIL) {
    // [한국어] RESERVATION_FAIL: MSHR/캐시 예약 슬롯 없음 — 다음 사이클에 재시도
    result = BK_CONF; // [한국어] 구조적 충돌로 인한 실패 (뱅크 충돌이 아닌 예약 실패)
    assert(!read_sent); // [한국어] 예약 실패 시 NoC 요청이 없어야 함
    assert(!write_sent);
    delete mf; // [한국어] 예약 실패이므로 mf 해제 (캐시에 저장되지 않음)
  } else {
    assert(status == MISS || status == HIT_RESERVED); // [한국어] MISS 또는 이미 같은 주소에 MSHR 항목이 있는 HIT_RESERVED
    // inst.clear_active( access.get_warp_mask() ); // threads in mf writeback
    // when mf returns
    inst.accessq_pop_back(); // [한국어] MISS/HIT_RESERVED: accessq에서 제거 (mf는 캐시/MSHR이 관리, 응답 시 fill로 돌아옴)
  }
  if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL; // [한국어] 아직 처리할 접근이 더 있으면 COAL_STALL: 다음 사이클에 계속 처리
  return result;
}

/*
 * [한국어]
 * ldst_unit::process_memory_access_queue — 일반 캐시에 대한 메모리 접근 큐 처리
 *
 * @cache: 접근할 캐시 (L1D/L1C/L1T)
 * @inst:  처리할 warp 명령어
 * @return: mem_stage_stall_type
 *
 * inst.accessq(coalesced 접근 목록)에서 하나를 꺼내 캐시에 접근.
 * m_mf_allocator로 mem_fetch를 생성하고 cache->access()를 호출한 후
 * process_cache_access()로 결과 처리.
 * L1D 레이턴시 큐를 사용하지 않는 캐시(L1C, L1T)에서 사용.
 *
 * 호출 체인: ldst_unit::constant_cycle() / texture_cycle() → [이 함수]
 */
mem_stage_stall_type ldst_unit::process_memory_access_queue(cache_t *cache,
                                                            warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (inst.accessq_empty()) return result; // [한국어] 처리할 접근이 없으면 즉시 반환

  if (!cache->data_port_free()) return DATA_PORT_STALL; // [한국어] 캐시 데이터 포트가 사용 중이면 DATA_PORT_STALL 반환

  // const mem_access_t &access = inst.accessq_back();
  // [한국어] accessq의 맨 뒤(다음 처리할) 접근에 대한 mem_fetch 생성
  mem_fetch *mf = m_mf_allocator->alloc(
      inst, inst.accessq_back(),
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] 현재 시뮬레이션 사이클 전달 (mem_fetch 타임스탬프)
  std::list<cache_event> events; // [한국어] 캐시 접근으로 발생한 이벤트 기록 (write_sent, read_sent 등)
  enum cache_request_status status = cache->access(
      mf->get_addr(), mf,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
      events); // [한국어] 캐시에 접근하여 HIT/MISS/RESERVATION_FAIL 판정
  return process_cache_access(cache, mf->get_addr(), inst, events, mf, status); // [한국어] 캐시 접근 결과에 따라 상태 처리
}

/*
 * [한국어]
 * ldst_unit::process_memory_access_queue_l1cache — L1D 레이턴시 큐를 사용한 메모리 접근 처리
 *
 * @cache: L1D 캐시 (l1_cache 타입)
 * @inst:  처리할 warp 명령어
 * @return: mem_stage_stall_type
 *
 * L1D 캐시는 뱅크별 레이턴시(l1_latency)를 가진 l1_latency_queue를 통해 접근.
 * l1_latency > 0 모드: 먼저 l1_latency_queue에 삽입하고, L1_latency_queue_cycle()에서 실제 캐시에 접근.
 *   - 사이클당 최대 l1_banks개의 요청을 l1_latency_queue의 마지막 슬롯에 삽입
 *   - 뱅크가 꽉 차면 BK_CONF 반환
 * l1_latency = 0 모드: 즉시 캐시에 접근 (process_memory_access_queue와 동일).
 *
 * 호출 체인: ldst_unit::memory_cycle() → [이 함수] → L1_latency_queue_cycle() → m_L1D->access()
 */
mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
    l1_cache *cache, warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (inst.accessq_empty()) return result; // [한국어] 처리할 접근 없으면 즉시 반환

  if (m_config->m_L1D_config.l1_latency > 0) {
    // [한국어] L1D 레이턴시 큐 모드: 요청을 큐의 마지막 슬롯(l1_latency-1)에 먼저 삽입
    for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks;
         j++) {  // We can handle at max l1_banks reqs per cycle
      // [한국어] 사이클당 최대 l1_banks개 요청 처리 (뱅크별로 1개씩)

      if (inst.accessq_empty()) return result; // [한국어] 모든 접근 처리 완료

      mem_fetch *mf =
          m_mf_allocator->alloc(inst, inst.accessq_back(),
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] 이 접근에 대한 mem_fetch 생성
      unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr()); // [한국어] 주소로 L1D 뱅크 ID 결정
      assert(bank_id < m_config->m_L1D_config.l1_banks); // [한국어] 유효한 뱅크 ID인지 확인

      if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
          NULL) {
        // [한국어] 이 뱅크의 레이턴시 큐 마지막 슬롯이 비어있음 — 삽입 가능
        l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf; // [한국어] 레이턴시 큐 마지막 슬롯에 삽입 (l1_latency 사이클 후 캐시에 접근)

        if (mf->get_inst().is_store()) {
          // [한국어] store인 경우: 레이턴시 큐에 들어가는 시점에 store_req 증가 (나중에 store_ack로 감소)
          unsigned inc_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf->get_data_size() / SECTOR_SIZE)
                  : 1;

          for (unsigned i = 0; i < inc_ack; ++i)
            m_core->inc_store_req(inst.warp_id()); // [한국어] warp의 미완료 store 요청 수 증가
        }

        inst.accessq_pop_back(); // [한국어] 이 접근을 accessq에서 제거 (레이턴시 큐가 이제 소유)
      } else {
        // [한국어] 이 뱅크의 레이턴시 큐가 꽉 참 — L1D 뱅크 충돌
        result = BK_CONF;
        m_stats->gpgpu_n_l1cache_bkconflict++; // [한국어] L1D 뱅크 충돌 통계 증가
        delete mf; // [한국어] 큐에 삽입 못했으므로 mf 해제
        break;  // do not try again, just break from the loop and try the next
                // cycle
                // [한국어] 이 사이클에 이 뱅크는 더 이상 처리 불가 — 다음 사이클 재시도
      }
    }
    if (!inst.accessq_empty() && result != BK_CONF) result = COAL_STALL; // [한국어] BK_CONF 없이 아직 처리할 접근이 더 있으면 COAL_STALL

    return result;
  } else {
    // [한국어] l1_latency = 0: 레이턴시 큐 없이 즉시 L1D 캐시에 접근 (l1_cache::access 직접 호출)
    mem_fetch *mf =
        m_mf_allocator->alloc(inst, inst.accessq_back(),
                              m_core->get_gpu()->gpu_sim_cycle +
                                  m_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(
        mf->get_addr(), mf,
        m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
        events);
    return process_cache_access(cache, mf->get_addr(), inst, events, mf,
                                status);
  }
}

/*
 * [한국어]
 * ldst_unit::L1_latency_queue_cycle — L1D 레이턴시 큐 사이클 진행
 *
 * @return: void
 *
 * l1_latency_queue[bank][stage] 배열을 1단계씩 앞으로 이동시키는 파이프라인 시프트.
 * 큐의 맨 앞(슬롯 0)에 도달한 요청은 실제 L1D 캐시에 접근하여 결과 처리:
 *   - HIT: pending_writes 감소, LDGSTS이면 depbar 해제, store면 store_ack
 *   - RESERVATION_FAIL: 이 사이클은 포기, 다음 사이클에 슬롯 0에 남아 재시도
 *   - MISS/HIT_RESERVED: 슬롯 0 해제, write-allocate 정책에 따라 store_ack 처리
 *
 * 매 사이클 모든 뱅크를 순회하여 파이프라인 진행 (레이턴시 큐 시프트).
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수]
 */
void ldst_unit::L1_latency_queue_cycle() {
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) { // [한국어] 모든 L1D 뱅크 순회
    if ((l1_latency_queue[j][0]) != NULL) {
      // [한국어] 슬롯 0에 요청이 있음 — 이 사이클에 실제 L1D 캐시에 접근
      mem_fetch *mf_next = l1_latency_queue[j][0]; // [한국어] 레이턴시 큐 맨 앞 요청 (l1_latency 사이클 대기 완료)
      std::list<cache_event> events;
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle,
                        events); // [한국어] 실제 L1D 캐시 접근: HIT/MISS/RESERVATION_FAIL 판정

      bool write_sent = was_write_sent(events); // [한국어] 캐시가 NoC로 쓰기 요청을 보냈는지
      bool read_sent = was_read_sent(events);   // [한국어] 캐시가 NoC로 읽기 요청을 보냈는지

      if (status == HIT) {
        assert(!read_sent); // [한국어] HIT이면 NoC 읽기 요청 없어야 함
        l1_latency_queue[j][0] = NULL; // [한국어] 처리 완료 — 슬롯 0 해제
        if (mf_next->get_inst().is_load()) {
          for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
            if (mf_next->get_inst().out[r] > 0) {
              assert(m_pending_writes[mf_next->get_inst().warp_id()]
                                     [mf_next->get_inst().out[r]] > 0); // [한국어] pending_writes가 0이 되어선 안 됨 (잘못된 writeback)
              unsigned still_pending =
                  --m_pending_writes[mf_next->get_inst().warp_id()]
                                    [mf_next->get_inst().out[r]]; // [한국어] 이 출력 레지스터에 대한 pending write 수 감소
              if (!still_pending) {
                // [한국어] 이 레지스터의 모든 접근이 완료됨 — 스코어보드에서 레지스터 해제
                m_pending_writes[mf_next->get_inst().warp_id()].erase(
                    mf_next->get_inst().out[r]); // [한국어] pending_writes 맵에서 제거
                m_scoreboard->releaseRegister(mf_next->get_inst().warp_id(),
                                              mf_next->get_inst().out[r]); // [한국어] 스코어보드에서 레지스터 해제 — 이 레지스터를 소스로 하는 명령어 발행 허용
                m_core->warp_inst_complete(mf_next->get_inst()); // [한국어] load 명령어 완료 통계 기록
              }
            }

          // release LDGSTS
          // [한국어] LDGSTS HIT 완료: pending_ldgsts 카운터 감소, 0이 되면 depbar 해제
          if (mf_next->get_inst().m_is_ldgsts) {
            m_pending_ldgsts[mf_next->get_inst().warp_id()]
                            [mf_next->get_inst().pc]
                            [mf_next->get_inst().get_addr(0)]--;
            if (m_pending_ldgsts[mf_next->get_inst().warp_id()]
                                [mf_next->get_inst().pc]
                                [mf_next->get_inst().get_addr(0)] == 0) {
              m_core->unset_depbar(mf_next->get_inst()); // [한국어] 이 LDGSTS 주소의 모든 접근 완료 → DEPBAR 대기 해제 시도
            }
          }
        }

        // For write hit in WB policy
        // [한국어] WB(Write-Back) 정책에서 L1D write HIT: NoC 전송 없이 캐시에만 쓰여진 경우
        if (mf_next->get_inst().is_store() && !write_sent) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1; // [한국어] SECTOR_ASSOC이면 섹터 단위 ack, 아니면 1

          mf_next->set_reply(); // [한국어] mf를 reply로 마킹 (store_ack에서 이 mf를 참조)

          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next); // [한국어] store 완료 통보 — warp의 store_req 카운터 감소
        }

        if (!write_sent) delete mf_next; // [한국어] NoC write_sent 없으면 mf 해제

      } else if (status == RESERVATION_FAIL) {
        // [한국어] RESERVATION_FAIL: L1D MSHR/뱅크 예약 실패 — 슬롯 0에 mf를 남겨두고 다음 사이클 재시도
        assert(!read_sent); // [한국어] 예약 실패 시 NoC 요청 없어야 함
        assert(!write_sent);
        // [한국어] l1_latency_queue[j][0]은 NULL로 만들지 않음 — 다음 사이클에 다시 시도
      } else {
        // [한국어] MISS 또는 HIT_RESERVED: 슬롯 0 해제, mf는 MSHR이 관리
        assert(status == MISS || status == HIT_RESERVED);
        l1_latency_queue[j][0] = NULL; // [한국어] 슬롯 0 해제 — mf는 캐시/MSHR이 소유
        if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
            mf_next->get_inst().is_store() &&
            (m_config->m_L1D_config.get_write_allocate_policy() ==
                 FETCH_ON_WRITE ||
             m_config->m_L1D_config.get_write_allocate_policy() ==
                 LAZY_FETCH_ON_READ) &&
            !was_writeallocate_sent(events)) {
          // [한국어] WB 정책 + write-allocate 방식에서 write miss 시 즉시 store_ack 처리
          // (FETCH_ON_WRITE/LAZY_FETCH_ON_READ이고 write-allocate 요청이 아직 안 보내진 경우)
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;
          mf_next->set_reply(); // [한국어] store 완료로 마킹
          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next); // [한국어] store 완료 처리
          if (!write_sent && !read_sent) delete mf_next; // [한국어] NoC 요청이 없으면 mf 해제
        }
      }
    }

    // [한국어] 레이턴시 큐 시프트: 각 슬롯을 한 단계씩 앞으로 이동 (NULL인 슬롯을 앞으로 전진)
    for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
         ++stage)
      if (l1_latency_queue[j][stage] == NULL) {
        l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1]; // [한국어] stage+1의 요청을 stage로 이동
        l1_latency_queue[j][stage + 1] = NULL; // [한국어] stage+1 슬롯 비움
      }
  }
}

/*
 * [한국어]
 * ldst_unit::constant_cycle — 상수 메모리 접근 처리 (Constant Memory Cycle)
 *
 * @inst:      처리할 warp 명령어 (const_space 또는 param_space_kernel)
 * @rc_fail:   [출력] 실패 원인
 * @fail_type: [출력] 실패 메모리 유형 (C_MEM: 상수 메모리)
 * @return:    true = accessq 비어있음(완료 또는 해당 없음), false = 아직 처리 중
 *
 * 상수 메모리(constant memory)와 커널 파라미터(param_space_kernel) 접근 처리.
 * perfect_inst_const_cache: 이상적 상수 캐시 모드 — 즉시 모든 접근 완료 (0 레이턴시).
 * 실제 모드: L1C(상수 캐시)에 process_memory_access_queue()로 접근.
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수] → process_memory_access_queue(m_L1C, inst)
 */
bool ldst_unit::constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                               mem_stage_access_type &fail_type) {
  if (inst.empty() || ((inst.space.get_type() != const_space) &&
                       (inst.space.get_type() != param_space_kernel)))
    return true; // [한국어] 상수 메모리 명령어가 아니면 즉시 완료로 처리

  if (inst.active_count() == 0) return true; // [한국어] 활성 스레드 없으면 no-op

  mem_stage_stall_type fail;
  if (m_config->perfect_inst_const_cache) {
    // [한국어] 이상적 상수 캐시: 레이턴시 없이 즉시 완료. 모든 접근을 accessq에서 제거.
    fail = NO_RC_FAIL;
    unsigned access_count = inst.accessq_count(); // [한국어] 처리할 접근 수 (pending_writes 감소에 사용)
    while (inst.accessq_count() > 0) inst.accessq_pop_back(); // [한국어] 모든 접근 즉시 소비
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0)
          m_pending_writes[inst.warp_id()][inst.out[r]] -= access_count; // [한국어] load인 경우 pending_writes를 접근 수만큼 감소
    }
  } else {
    fail = process_memory_access_queue(m_L1C, inst); // [한국어] L1C(상수 캐시) 접근
  }

  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = C_MEM; // [한국어] 실패 원인: 상수 메모리 접근 문제
    if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
      m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
                                             // conflict, but this maintains
                                             // previous behavior.
      // [한국어] 상수 메모리 포트 충돌/coalescing 스톨 통계 (COAL_STALL도 여기 포함)
    }
  }
  return inst.accessq_empty();  // done if empty.
  // [한국어] accessq가 비었으면 완료(true), 아직 처리할 접근이 있으면 false
}

/*
 * [한국어]
 * ldst_unit::texture_cycle — 텍스처 메모리 접근 처리 (Texture Memory Cycle)
 *
 * @inst:      처리할 warp 명령어 (tex_space)
 * @rc_fail:   [출력] 실패 원인
 * @fail_type: [출력] 실패 메모리 유형 (T_MEM: 텍스처 메모리)
 * @return:    true = 완료 또는 해당 없음, false = 아직 처리 중
 *
 * 텍스처 메모리(tex_space) 접근을 L1T(텍스처 캐시)로 처리.
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수] → process_memory_access_queue(m_L1T, inst)
 */
bool ldst_unit::texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                              mem_stage_access_type &fail_type) {
  if (inst.empty() || inst.space.get_type() != tex_space) return true; // [한국어] 텍스처 명령어가 아니면 즉시 완료
  if (inst.active_count() == 0) return true; // [한국어] 활성 스레드 없으면 no-op
  mem_stage_stall_type fail = process_memory_access_queue(m_L1T, inst); // [한국어] L1T(텍스처 캐시) 접근
  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = T_MEM; // [한국어] 실패 원인: 텍스처 메모리 접근 문제
  }
  return inst.accessq_empty();  // done if empty.
  // [한국어] 모든 접근이 처리됐으면 true
}

/*
 * [한국어]
 * ldst_unit::memory_cycle — 글로벌/로컬 메모리 접근 처리 (Global/Local Memory Cycle)
 *
 * @inst:        처리할 warp 명령어 (global_space / local_space / param_space_local)
 * @stall_reason: [출력] 스톨 원인 (ICNT_RC_FAIL/BK_CONF/COAL_STALL 등)
 * @access_type: [출력] 접근 유형 (G_MEM_LD/G_MEM_ST/L_MEM_LD/L_MEM_ST)
 * @return:      true = accessq 비어있음(완료), false = 아직 처리 중
 *
 * 글로벌/로컬 메모리 접근의 핵심 처리 함수:
 *   - bypassL1D 결정: cache_op=GLOBAL, L1D 없음, gmem_skip_L1D 설정 시 L1D 우회
 *   - bypass 시: m_icnt(NoC)로 직접 mem_fetch push. NoC 가득 찼으면 ICNT_RC_FAIL.
 *     SST_mode: SST 인터페이스의 full() 체크 (mf 타입 포함).
 *   - L1D 사용 시: process_memory_access_queue_l1cache()로 L1D 접근 (레이턴시 큐 포함).
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수] → m_icnt->push(mf) 또는 process_memory_access_queue_l1cache()
 */
bool ldst_unit::memory_cycle(warp_inst_t &inst,
                             mem_stage_stall_type &stall_reason,
                             mem_stage_access_type &access_type) {
  if (inst.empty() || ((inst.space.get_type() != global_space) &&
                       (inst.space.get_type() != local_space) &&
                       (inst.space.get_type() != param_space_local)))
    return true; // [한국어] 글로벌/로컬 메모리 명령어가 아니면 즉시 완료

  if (inst.active_count() == 0) return true; // [한국어] 활성 스레드 없으면 no-op
  if (inst.accessq_empty()) return true;     // [한국어] 처리할 접근 없으면 완료

  mem_stage_stall_type stall_cond = NO_RC_FAIL;
  const mem_access_t &access = inst.accessq_back(); // [한국어] accessq에서 다음 처리할 접근 정보

  bool bypassL1D = false; // [한국어] L1D 캐시를 우회하고 NoC로 직접 보낼지 결정
  if (CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL)) {
    bypassL1D = true; // [한국어] cache_op=GLOBAL(캐시 무력화 명령어) 또는 L1D 없음
  } else if (inst.space.is_global()) {  // global memory access
    // skip L1 cache if the option is enabled
    if (m_core->get_config()->gmem_skip_L1D && (CACHE_L1 != inst.cache_op))
      bypassL1D = true; // [한국어] gmem_skip_L1D 설정이고 CACHE_L1 명시가 아닌 경우 L1D 우회
  }
  if (bypassL1D) {
    // bypass L1 cache
    // [한국어] L1D 우회: NoC(intersim2)로 직접 mem_fetch 전송
    unsigned control_size =
        inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE; // [한국어] store이면 WRITE_PACKET_SIZE, load이면 READ_PACKET_SIZE
    unsigned size = access.get_size() + control_size; // [한국어] 총 패킷 크기: 데이터 크기 + 제어 헤더 크기
    // printf("Interconnect:Addr: %x, size=%d\n",access.get_addr(),size);
    if (m_memory_config->SST_mode &&
        (static_cast<sst_memory_interface *>(m_icnt)->full(
            size, inst.is_store() || inst.isatomic(), access.get_type()))) {
      // SST need mf type here
      // Cast it to sst_memory_interface pointer first as this full() method
      // is not a virtual method in parent class
      // [한국어] SST(SystemC Simulation Toolkit) 모드: SST 인터페이스가 가득 찼는지 mf 타입과 함께 확인
      stall_cond = ICNT_RC_FAIL; // [한국어] NoC 큐 가득 참: ICNT_RC_FAIL로 스톨
    } else if (!m_memory_config->SST_mode &&
               (m_icnt->full(size, inst.is_store() || inst.isatomic()))) {
      stall_cond = ICNT_RC_FAIL; // [한국어] 일반 모드: NoC 큐 가득 참으로 스톨
    } else {
      // [한국어] NoC 큐에 여유 있음: mem_fetch를 생성하여 NoC로 push
      mem_fetch *mf =
          m_mf_allocator->alloc(inst, access,
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] NoC 전송용 mem_fetch 생성
      m_icnt->push(mf); // [한국어] NoC로 mem_fetch 전송 (intersim2가 메모리 컨트롤러로 라우팅)
      inst.accessq_pop_back(); // [한국어] 이 접근을 accessq에서 제거
      // inst.clear_active( access.get_warp_mask() );
      if (inst.is_load()) {
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
          if (inst.out[r] > 0)
            assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0); // [한국어] load이면 pending_writes가 예약되어 있어야 함 (scoreboard가 reserveRegisters에서 설정)
      } else if (inst.is_store())
        m_core->inc_store_req(inst.warp_id()); // [한국어] store이면 미완료 요청 수 증가 (NoC 응답이 올 때 감소)
    }
  } else {
    // [한국어] L1D 사용: 레이턴시 큐를 통해 L1D 캐시에 접근
    assert(CACHE_UNDEFINED != inst.cache_op); // [한국어] 캐시 정책이 정의되어 있어야 함
    stall_cond = process_memory_access_queue_l1cache(m_L1D, inst); // [한국어] L1D 레이턴시 큐에 접근 요청 삽입
  }
  if (!inst.accessq_empty() && stall_cond == NO_RC_FAIL)
    stall_cond = COAL_STALL; // [한국어] 처리할 접근이 더 있고 실패도 없으면 COAL_STALL (다음 사이클에 계속)
  if (stall_cond != NO_RC_FAIL) {
    stall_reason = stall_cond; // [한국어] 스톨 원인 출력
    bool iswrite = inst.is_store();
    if (inst.space.is_local())
      access_type = (iswrite) ? L_MEM_ST : L_MEM_LD; // [한국어] 로컬 메모리 store/load
    else
      access_type = (iswrite) ? G_MEM_ST : G_MEM_LD; // [한국어] 글로벌 메모리 store/load
  }
  return inst.accessq_empty(); // [한국어] accessq 비었으면 완료(true), 아직 있으면 false
}

/*
 * [한국어]
 * ldst_unit::response_buffer_full — ldst_unit 응답 FIFO가 가득 찼는지 확인
 *
 * @return: true = 응답 FIFO가 최대 크기에 도달 (더 이상 응답 수신 불가), false = 여유 있음
 *
 * m_response_fifo: NoC에서 돌아온 load/store 응답(mem_fetch)을 임시 저장하는 큐.
 * simt_core_cluster::icnt_cycle()에서 accept_ldst_unit_response() 호출 전 이 함수로 공간 확인.
 *
 * 호출 체인: simt_core_cluster::icnt_cycle() → [이 함수]
 */
bool ldst_unit::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size; // [한국어] 현재 FIFO 크기가 최대 크기 이상이면 가득 찼음
}

/*
 * [한국어]
 * ldst_unit::fill — NoC 응답 mem_fetch를 응답 FIFO에 삽입
 *
 * @mf:     NoC에서 도착한 메모리 응답 mem_fetch (load 데이터 또는 store ack)
 * @return: void
 *
 * simt_core_cluster::icnt_cycle()에서 NoC로부터 응답을 수신하면 호출.
 * mf의 상태를 IN_SHADER_LDST_RESPONSE_FIFO로 변경하고 m_response_fifo에 enqueue.
 * ldst_unit::cycle()에서 m_response_fifo에서 dequeue하여 처리.
 *
 * 호출 체인: simt_core_cluster::icnt_cycle() → shader_core_ctx::accept_ldst_unit_response() → [이 함수]
 */
void ldst_unit::fill(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] mf 상태를 IN_SHADER_LDST_RESPONSE_FIFO로 변경 (타임스탬프 포함)
  m_response_fifo.push_back(mf); // [한국어] 응답 FIFO에 삽입 — ldst_unit::cycle()에서 순서대로 처리
}

/*
 * [한국어]
 * ldst_unit::flush — L1D 데이터 캐시 플러시
 *
 * @return: void
 *
 * 커널 종료 또는 명시적 cache flush 명령 시 L1D를 비움.
 * WB(Write-Back) 정책이면 dirty 라인을 하위 캐시(L2)로 기록.
 *
 * 호출 체인: shader_core_ctx::cache_flush() → [이 함수]
 */
void ldst_unit::flush() {
  // Flush L1D cache
  m_L1D->flush(); // [한국어] L1D 캐시 플러시: dirty 라인을 모두 하위 계층으로 writeback하고 무효화
}

/*
 * [한국어]
 * ldst_unit::invalidate — L1D 캐시 무효화 (flush 없이 모든 라인 무효화)
 *
 * @return: void
 *
 * 커널 간 L1D 캐시 무효화 (dirty 라인 버림). 커넥트런시 문제로 캐시를 깨끗하게 비울 때 사용.
 *
 * 호출 체인: shader_core_ctx::cache_invalidate() → [이 함수]
 */
void ldst_unit::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate(); // [한국어] L1D 캐시 무효화: 모든 라인을 invalid 상태로 만듦 (dirty 라인 하위 계층으로 writeback 안 함)
}

/*
 * [한국어]
 * simd_function_unit::simd_function_unit — SIMD 기능 유닛 기반 클래스 생성자
 *
 * @config: 셰이더 코어 설정 (레이턴시, 파이프라인 너비 등)
 *
 * 모든 SIMD 실행 유닛(SP/DP/SFU/INT/ldst_unit 등)의 공통 기반 클래스.
 * m_dispatch_reg: 스케줄러에서 issue_warp()가 쓴 파이프라인 레지스터에서 이 FU로 오는 중간 dispatch 레지스터.
 *
 * 호출 체인: pipelined_simd_unit 생성자 → [이 기반 생성자]
 */
simd_function_unit::simd_function_unit(const shader_core_config *config) {
  m_config = config; // [한국어] 셰이더 코어 설정 포인터 저장
  m_dispatch_reg = new warp_inst_t(config); // [한국어] dispatch 레지스터 초기화 (issue_warp → FU 내부로 명령어 전달 버퍼)
}

/*
 * [한국어]
 * simd_function_unit::issue — FU로 명령어 dispatch
 *
 * @source_reg: 이 FU의 입력 파이프라인 레지스터 집합 (OC_EX_* 단계)
 * @return:     void
 *
 * source_reg에서 ready 명령어를 m_dispatch_reg로 이동하고,
 * occupied 비트셋에 latency 위치를 set하여 이 FU가 해당 사이클까지 사용 중임을 표시.
 * sub_core_model이면 m_issue_reg_id로 특정 슬롯의 명령어만 가져옴.
 *
 * 호출 체인: pipelined_simd_unit::issue() → [이 함수] (템플릿 메서드 패턴)
 */
void simd_function_unit::issue(register_set &source_reg) {
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned(); // [한국어] sub_core_model 여부 확인
  source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
                         m_dispatch_reg); // [한국어] source_reg에서 m_dispatch_reg로 명령어 이동
  occupied.set(m_dispatch_reg->latency); // [한국어] occupied 비트셋에 latency 위치 set: 이 FU가 latency 사이클 후까지 사용됨을 표시
}

/*
 * [한국어]
 * sfu::sfu — SFU(Special Function Unit) 생성자
 *
 * @result_port:   실행 결과를 쓸 EX_WB 파이프라인 레지스터 집합
 * @config:        셰이더 코어 설정
 * @core:          이 FU가 속한 SM
 * @issue_reg_id:  sub_core_model에서 이 FU가 담당하는 스케줄러 슬롯 ID
 *
 * SFU: sin/cos/sqrt/rcp 등 초월함수 실행 유닛. max_sfu_latency 깊이의 파이프라인.
 * Fermi에서 DP_OP도 SFU에서 처리.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
sfu::sfu(register_set *result_port, const shader_core_config *config,
         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_sfu_latency, core,
                          issue_reg_id) { // [한국어] max_sfu_latency 깊이의 파이프라인으로 초기화
  m_name = "SFU"; // [한국어] 파이프라인 출력 시 표시될 유닛 이름
}

/*
 * [한국어]
 * tensor_core::tensor_core — Tensor Core 유닛 생성자
 *
 * @result_port:   실행 결과를 쓸 EX_WB 파이프라인 레지스터 집합
 * @config:        셰이더 코어 설정
 * @core:          이 FU가 속한 SM
 * @issue_reg_id:  sub_core_model에서 이 FU가 담당하는 스케줄러 슬롯 ID
 *
 * Tensor Core: wmma/mma(행렬 곱) 실행 유닛. max_tensor_core_latency 깊이의 파이프라인.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
tensor_core::tensor_core(register_set *result_port,
                         const shader_core_config *config,
                         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_tensor_core_latency,
                          core, issue_reg_id) { // [한국어] max_tensor_core_latency 깊이의 파이프라인으로 초기화
  m_name = "TENSOR_CORE"; // [한국어] 파이프라인 출력 시 표시될 유닛 이름
}

/*
 * [한국어]
 * sfu::issue — SFU로 명령어 발행
 *
 * @source_reg: 입력 파이프라인 레지스터 집합 (OC_EX_SFU)
 * @return:     void
 *
 * op_pipe를 SFU__OP로 설정하고 SFU 통계를 업데이트한 후 파이프라인으로 이동.
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->issue() → [이 함수]
 */
void sfu::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id); // [한국어] 준비된 명령어 포인터 조회
  // m_core->incexecstat((*ready_reg));

  (*ready_reg)->op_pipe = SFU__OP; // [한국어] 이 명령어의 실행 파이프 타입을 SFU로 설정 (writeback에서 통계 분류에 사용)
  m_core->incsfu_stat(m_core->get_config()->warp_size, (*ready_reg)->latency); // [한국어] SFU 실행 통계 증가 (AccelWattch 전력 모델용)
  pipelined_simd_unit::issue(source_reg); // [한국어] 기반 클래스 issue: dispatch_reg로 명령어 이동, occupied 비트 설정
}

/*
 * [한국어]
 * tensor_core::issue — Tensor Core로 명령어 발행
 *
 * @source_reg: 입력 파이프라인 레지스터 집합 (OC_EX_TC)
 * @return:     void
 *
 * op_pipe를 TENSOR_CORE__OP로 설정하고 SFU 통계(Tensor Core 실행도 SFU 계열) 업데이트.
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->issue() → [이 함수]
 */
void tensor_core::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id); // [한국어] 준비된 명령어 포인터 조회
  // m_core->incexecstat((*ready_reg));

  (*ready_reg)->op_pipe = TENSOR_CORE__OP; // [한국어] 파이프 타입을 TENSOR_CORE로 설정
  m_core->incsfu_stat(m_core->get_config()->warp_size, (*ready_reg)->latency); // [한국어] Tensor Core 통계 증가 (SFU 계열로 집계)
  pipelined_simd_unit::issue(source_reg); // [한국어] 기반 클래스 issue 호출
}

/*
 * [한국어]
 * pipelined_simd_unit::get_active_lanes_in_pipeline — 파이프라인 내 활성 레인 수 반환
 *
 * @return: 모든 파이프라인 스테이지의 활성 스레드 수 합산 (AccelWattch 전력 계산용)
 *
 * g_power_simulation_enabled일 때만 동작: 각 스테이지의 명령어 active_mask를 OR하여 카운트.
 * 이 값이 AccelWattch 전력 모델에서 FU 동적 전력 계산에 사용됨.
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->active_lanes_in_pipeline() → [이 함수]
 */
unsigned pipelined_simd_unit::get_active_lanes_in_pipeline() {
  active_mask_t active_lanes;
  active_lanes.reset(); // [한국어] 활성 레인 마스크 초기화
  if (m_core->get_gpu()->get_config().g_power_simulation_enabled) {
    // [한국어] 전력 시뮬레이션 활성화 시에만 계산 (성능 오버헤드 방지)
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (!m_pipeline_reg[stage]->empty())
        active_lanes |= m_pipeline_reg[stage]->get_active_mask(); // [한국어] 이 스테이지의 활성 스레드 마스크를 OR로 누적
    }
  }
  return active_lanes.count(); // [한국어] 전체 파이프라인에서 활성 스레드 레인 수 반환
}

/*
 * [한국어]
 * active_lanes_in_pipeline 가상 함수 구현들 — 유닛별 AccelWattch 통계 업데이트
 *
 * 각 유닛은 전력 모델에 필요한 다른 카운터를 증가시킴:
 * - ldst_unit: FU 메모리 활성 레인 수만 업데이트
 * - sp_unit: SP 활성 레인 + FU 활성 레인 + FU 메모리 활성 레인
 * - dp_unit: FU 활성 레인 + FU 메모리 활성 레인 (SP 카운터 제외)
 * - int_unit: SP와 동일 (SP 카운터 공유)
 * - sfu: SFU 전용 카운터 + FU 카운터
 * - tensor_core: SFU 카운터 공유 (wmma는 SFU와 유사한 전력 모델)
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->active_lanes_in_pipeline() → [이 함수들]
 */
void ldst_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline(); // [한국어] 파이프라인 내 활성 레인 수 계산
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incfumemactivelanes_stat(active_count); // [한국어] ldst_unit: 메모리 FU 활성 레인 통계만 업데이트
}

void sp_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count);    // [한국어] SP unit 전용 활성 레인 통계
  m_core->incfuactivelanes_stat(active_count);    // [한국어] 전체 FU 활성 레인 통계
  m_core->incfumemactivelanes_stat(active_count); // [한국어] FU 메모리(SP 포함) 활성 레인 통계
}
void dp_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  // m_core->incspactivelanes_stat(active_count); // [한국어] DP unit은 SP 카운터를 증가시키지 않음
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
void specialized_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count); // [한국어] 특수화 유닛: SP 계열로 집계
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void int_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count); // [한국어] INT unit: SP 카운터 공유 (INT와 SP가 같은 전력 모델 사용)
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
void sfu::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incsfuactivelanes_stat(active_count); // [한국어] SFU 전용 활성 레인 통계
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void tensor_core::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incsfuactivelanes_stat(active_count); // [한국어] Tensor Core: SFU 통계 공유
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

/*
 * [한국어]
 * sp_unit::sp_unit — SP(Single-Precision FP) 실행 유닛 생성자
 *
 * max_sp_latency 깊이의 파이프라인으로 단정밀도 FP 명령어 처리.
 * Fermi/Pascal에서는 ALU/INT도 여기서 처리 (INT unit 없는 경우).
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
sp_unit::sp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_sp_latency, core,
                          issue_reg_id) { // [한국어] max_sp_latency 깊이의 파이프라인
  m_name = "SP ";
}

/*
 * [한국어]
 * specialized_unit::specialized_unit — 특수화 유닛 생성자
 *
 * @supported_op:  이 유닛이 처리하는 op 코드 (SPEC_UNIT_START_ID 이상)
 * @unit_name:     유닛 이름 (gpgpusim.config에서 지정)
 * @latency:       파이프라인 깊이 (사이클)
 *
 * 사용자 정의 특수화 유닛. m_specialized_unit 설정으로 여러 개 생성 가능.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
specialized_unit::specialized_unit(register_set *result_port,
                                   const shader_core_config *config,
                                   shader_core_ctx *core, int supported_op,
                                   char *unit_name, unsigned latency,
                                   unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, latency, core, issue_reg_id) {
  m_name = unit_name;         // [한국어] 설정 파일에서 지정된 유닛 이름
  m_supported_op = supported_op; // [한국어] 이 유닛이 담당하는 op 코드 (scheduler가 발행 대상 판별에 사용)
}

/*
 * [한국어]
 * dp_unit::dp_unit — DP(Double-Precision FP) 실행 유닛 생성자
 *
 * max_dp_latency 깊이의 파이프라인으로 배정밀도 FP 명령어 처리.
 * Volta+ 아키텍처에서 별도 DP unit 존재; Fermi에서는 SFU가 담당.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
dp_unit::dp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_dp_latency, core,
                          issue_reg_id) { // [한국어] max_dp_latency 깊이의 파이프라인
  m_name = "DP ";
}

/*
 * [한국어]
 * int_unit::int_unit — INT(정수) 실행 유닛 생성자
 *
 * max_int_latency 깊이의 파이프라인으로 정수 ALU 명령어 처리.
 * Volta+ 아키텍처에서 SP와 별도 INT 파이프라인 존재.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
int_unit::int_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_int_latency, core,
                          issue_reg_id) { // [한국어] max_int_latency 깊이의 파이프라인
  m_name = "INT ";
}

/*
 * [한국어]
 * sp_unit::issue / dp_unit::issue / specialized_unit::issue / int_unit::issue
 * — 각 유닛으로 명령어 발행 (op_pipe 설정 + 통계 업데이트)
 *
 * @source_reg: 입력 파이프라인 레지스터 집합 (OC_EX_* 단계)
 * @return:     void
 *
 * 공통 패턴: op_pipe 필드를 유닛별 타입으로 설정, 실행 통계 증가, pipelined_simd_unit::issue() 호출.
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->issue() → [이 함수들]
 */
void sp_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id); // [한국어] 준비된 명령어 포인터 조회
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = SP__OP; // [한국어] SP 파이프 타입으로 설정 (writeback 통계 분류용)
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency); // [한국어] SP 실행 통계 증가
  pipelined_simd_unit::issue(source_reg); // [한국어] 공통 issue 로직: dispatch_reg로 이동, occupied 설정
}

void dp_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = DP__OP; // [한국어] DP 파이프 타입으로 설정
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency); // [한국어] SP 통계에 포함 (DP도 SP 계열로 집계)
  pipelined_simd_unit::issue(source_reg);
}

void specialized_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = SPECIALIZED__OP; // [한국어] 특수화 유닛 파이프 타입으로 설정
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void int_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = INTP__OP; // [한국어] INT 파이프 타입으로 설정
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency); // [한국어] SP 통계에 포함 (INT도 SP 계열로 집계)
  pipelined_simd_unit::issue(source_reg);
}

/*
 * [한국어]
 * pipelined_simd_unit::pipelined_simd_unit — 파이프라인형 SIMD 실행 유닛 생성자
 *
 * @result_port:   실행 결과를 쓸 EX_WB 파이프라인 레지스터 집합
 * @config:        셰이더 코어 설정
 * @max_latency:   파이프라인 깊이 (=실행 레이턴시, 사이클 수)
 * @core:          이 FU가 속한 SM
 * @issue_reg_id:  sub_core_model에서 이 FU가 담당하는 스케줄러 슬롯 ID
 *
 * max_latency 깊이의 파이프라인 레지스터 배열을 동적으로 할당.
 * 각 스테이지는 독립적인 warp_inst_t를 가짐.
 *
 * 호출 체인: sp_unit/sfu/dp_unit/int_unit/tensor_core 생성자 → [이 기반 생성자]
 */
pipelined_simd_unit::pipelined_simd_unit(register_set *result_port,
                                         const shader_core_config *config,
                                         unsigned max_latency,
                                         shader_core_ctx *core,
                                         unsigned issue_reg_id)
    : simd_function_unit(config) {
  m_result_port = result_port;        // [한국어] 실행 결과를 writeback으로 보낼 EX_WB 레지스터 집합
  m_pipeline_depth = max_latency;     // [한국어] 파이프라인 깊이 = 실행 레이턴시 (사이클)
  m_pipeline_reg = new warp_inst_t *[m_pipeline_depth]; // [한국어] 파이프라인 레지스터 배열 할당
  for (unsigned i = 0; i < m_pipeline_depth; i++)
    m_pipeline_reg[i] = new warp_inst_t(config); // [한국어] 각 스테이지의 warp_inst_t 초기화
  m_core = core;                      // [한국어] 이 FU가 속한 SM 포인터
  m_issue_reg_id = issue_reg_id;      // [한국어] sub_core_model에서 이 FU의 스케줄러 슬롯 ID
  active_insts_in_pipeline = 0;       // [한국어] 현재 파이프라인 내 활성 명령어 수 (cycle() 시 업데이트)
}

/*
 * [한국어]
 * pipelined_simd_unit::cycle — 파이프라인 사이클 진행
 *
 * @return: void
 *
 * 매 사이클 호출되어 파이프라인을 한 단계씩 전진:
 *   1. m_pipeline_reg[0]이 비어있지 않으면: m_result_port(EX_WB)로 이동 (완료)
 *   2. 활성 명령어가 있으면: 각 스테이지를 stage+1 → stage로 시프트
 *   3. m_dispatch_reg에 새 명령어가 있으면: dispatch_delay 체크 후 start_stage에 삽입
 *      start_stage = latency - initiation_interval: 파이프라인 시작 위치 (파이프라인 투명도 지원)
 *   4. occupied 비트셋을 오른쪽 시프트 (1사이클 경과)
 *
 * 호출 체인: shader_core_ctx::execute() → m_fu[n]->cycle() → [이 함수]
 */
void pipelined_simd_unit::cycle() {
  if (!m_pipeline_reg[0]->empty()) {
    m_result_port->move_in(m_pipeline_reg[0]); // [한국어] 파이프라인 맨 앞 명령어를 EX_WB 파이프라인 레지스터로 이동 (완료)
    assert(active_insts_in_pipeline > 0); // [한국어] 활성 명령어가 없는데 stage 0이 비어있지 않으면 오류
    active_insts_in_pipeline--;           // [한국어] 파이프라인 내 활성 명령어 수 감소
  }
  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]); // [한국어] 파이프라인 시프트: stage+1의 명령어를 stage로 이동
  }
  if (!m_dispatch_reg->empty()) {
    if (!m_dispatch_reg->dispatch_delay()) { // [한국어] dispatch_delay 체크: 발행 지연이 없으면 파이프라인 삽입
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initiation_interval; // [한국어] start_stage: 파이프라인 시작 위치 (throughput=1/initiation_interval 지원)
      if (m_pipeline_reg[start_stage]->empty()) {
        move_warp(m_pipeline_reg[start_stage], m_dispatch_reg); // [한국어] dispatch_reg에서 start_stage로 명령어 이동
        active_insts_in_pipeline++;  // [한국어] 파이프라인 내 활성 명령어 수 증가
      }
    }
  }
  occupied >>= 1; // [한국어] occupied 비트셋 오른쪽 시프트: 1사이클 경과로 점유 타이머 진행
}

/*
 * [한국어]
 * pipelined_simd_unit::issue — 파이프라인 FU로 명령어 발행
 *
 * @source_reg: 입력 파이프라인 레지스터 집합 (OC_EX_* 단계)
 * @return:     void
 *
 * incexecstat()으로 실행 통계를 업데이트하고 기반 클래스 issue()를 호출.
 *
 * 호출 체인: sp_unit::issue() / sfu::issue() 등 → [이 함수] → simd_function_unit::issue()
 */
void pipelined_simd_unit::issue(register_set &source_reg) {
  // move_warp(m_dispatch_reg,source_reg);
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned(); // [한국어] sub_core_model 여부
  warp_inst_t **ready_reg =
      source_reg.get_ready(partition_issue, m_issue_reg_id); // [한국어] 준비된 명령어 포인터 조회
  m_core->incexecstat((*ready_reg)); // [한국어] 이 명령어의 실행 통계 업데이트 (op 타입별 카운터)
  // source_reg.move_out_to(m_dispatch_reg);
  simd_function_unit::issue(source_reg); // [한국어] 기반 클래스 issue: source_reg → dispatch_reg 이동, occupied 설정
}

/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/

/*
 * [한국어]
 * ldst_unit::init — ldst_unit 공통 초기화 (두 생성자가 공유하는 로직)
 *
 * @icnt:               NoC 인터페이스 포인터 (mem_fetch 전송용)
 * @mf_allocator:       mem_fetch 할당자 (SM 설정 포함)
 * @core:               이 ldst_unit이 속한 SM
 * @operand_collector:  오퍼랜드 콜렉터 (writeback 시 레지스터 파일 업데이트)
 * @scoreboard:         스코어보드 (레지스터 의존성 추적)
 * @config:             셰이더 코어 설정
 * @mem_config:         메모리 설정 (SST 모드, 캐시 설정 등)
 * @stats:              통계 객체
 * @sid:                SM ID
 * @tpc:                TPC(Texture Processing Cluster) ID
 *
 * L1T(텍스처 캐시)와 L1C(상수 캐시)를 생성하고 포인터들을 초기화.
 * L1D는 이후 생성자에서 설정 (disabled이면 NULL로 유지).
 *
 * 호출 체인: ldst_unit::ldst_unit() → [이 함수]
 */
void ldst_unit::init(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc) {
  m_memory_config = mem_config;        // [한국어] 메모리 설정 (SST 모드, 캐시 파라미터 등)
  m_icnt = icnt;                       // [한국어] NoC 인터페이스: mem_fetch를 메모리 시스템으로 전송
  m_mf_allocator = mf_allocator;       // [한국어] mem_fetch 할당자: 메모리 접근 패킷 생성
  m_core = core;                       // [한국어] 이 ldst_unit이 속한 SM 포인터
  m_operand_collector = operand_collector; // [한국어] 오퍼랜드 콜렉터: writeback 시 레지스터 파일 업데이트
  m_scoreboard = scoreboard;           // [한국어] 스코어보드: 레지스터 의존성 추적 (releaseRegister)
  m_stats = stats;                     // [한국어] 통계 객체: 메모리 접근 카운터
  m_sid = sid;                         // [한국어] SM ID (캐시 이름 생성에 사용)
  m_tpc = tpc;                         // [한국어] TPC ID
#define STRSIZE 1024
  char L1T_name[STRSIZE];
  char L1C_name[STRSIZE];
  snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid); // [한국어] 텍스처 캐시 이름 생성 (예: "L1T_000")
  snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid); // [한국어] 상수 캐시 이름 생성 (예: "L1C_000")
  m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
                        get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
                        IN_SHADER_L1T_ROB); // [한국어] L1T(텍스처 캐시): ROB(Reorder Buffer) 지원하는 특수 캐시
  m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
                              get_shader_constant_cache_id(), icnt,
                              IN_L1C_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu); // [한국어] L1C(상수 캐시): 읽기 전용 캐시
  m_L1D = NULL;    // [한국어] L1D는 생성자에서 설정 (disabled이면 NULL 유지)
  m_mem_rc = NO_RC_FAIL; // [한국어] 메모리 RC 실패 상태 초기화
  m_num_writeback_clients =
      5;  // = shared memory, global/local (uncached), L1D, L1T, L1C
      // [한국어] writeback 클라이언트 5개: 공유메모리, 글로벌/로컬(비캐시), L1D, L1T, L1C
  m_writeback_arb = 0;   // [한국어] writeback 라운드로빈 중재자 초기화 (0번 클라이언트부터 시작)
  m_next_global = NULL;  // [한국어] 다음 글로벌 메모리 writeback mf 포인터 초기화
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
}

/*
 * [한국어]
 * ldst_unit::ldst_unit (첫 번째 생성자) — 일반 ldst_unit 생성 (L1D 포함)
 *
 * @gpu:    gpgpu_sim 포인터 (L1D 생성에 필요)
 *
 * smem_latency 깊이의 파이프라인으로 초기화 (공유 메모리용).
 * m_L1D_config.disabled()가 아니면 L1D 캐시와 l1_latency_queue 생성.
 * l1_latency_queue: l1_banks × l1_latency 크기의 2D 배열.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 생성자]
 */
ldst_unit::ldst_unit(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc, gpgpu_sim *gpu)
    : pipelined_simd_unit(NULL, config, config->smem_latency, core, 0), // [한국어] NULL result_port: ldst_unit은 자체 writeback 로직을 가짐 (EX_WB 대신 m_next_wb 사용)
      m_next_wb(config), // [한국어] 다음에 writeback할 명령어 초기화
      m_gpu(gpu) {
  assert(config->smem_latency > 1); // [한국어] 공유 메모리 레이턴시는 최소 2 이상이어야 함
  init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
       mem_config, stats, sid, tpc); // [한국어] 공통 초기화 (L1T, L1C 생성)
  if (!m_config->m_L1D_config.disabled()) {
    // [한국어] L1D가 활성화된 경우 데이터 캐시 생성
    char L1D_name[STRSIZE];
    snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid); // [한국어] 데이터 캐시 이름 생성
    m_L1D = new l1_cache(L1D_name, m_config->m_L1D_config, m_sid,
                         get_shader_normal_cache_id(), m_icnt, m_mf_allocator,
                         IN_L1D_MISS_QUEUE, core->get_gpu(), L1_GPU_CACHE); // [한국어] L1D: 일반 캐시 (읽기/쓰기 지원, WB 또는 WT 정책)

    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks); // [한국어] l1_latency_queue를 l1_banks 크기로 확장
    assert(m_config->m_L1D_config.l1_latency > 0); // [한국어] 레이턴시는 양수여야 함

    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
                                 (mem_fetch *)NULL); // [한국어] 각 뱅크의 레이턴시 큐를 l1_latency 크기로 초기화 (NULL 패딩)
  }
  m_name = "MEM "; // [한국어] 파이프라인 출력 시 표시될 유닛 이름
}

/*
 * [한국어]
 * ldst_unit::ldst_unit (두 번째 생성자) — 외부 L1D 캐시를 받는 생성자 (SST 모드 등)
 *
 * @new_l1d_cache: 외부에서 생성된 L1D 캐시 포인터 (SST 시뮬레이션에서 사용)
 *
 * 파이프라인 깊이를 3으로 고정 (최소 레이턴시).
 * 외부 L1D 캐시를 m_L1D에 직접 할당.
 *
 * 호출 체인: sst_simt_core_cluster 등에서 사용
 */
ldst_unit::ldst_unit(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc, l1_cache *new_l1d_cache)
    : pipelined_simd_unit(NULL, config, 3, core, 0), // [한국어] 파이프라인 깊이 3 고정 (SST 모드 최소값)
      m_L1D(new_l1d_cache),  // [한국어] 외부에서 생성된 L1D 캐시 직접 할당
      m_next_wb(config) {
  init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
       mem_config, stats, sid, tpc); // [한국어] 공통 초기화 (L1T, L1C 생성)
}

/*
 * [한국어]
 * ldst_unit::issue — ldst_unit으로 메모리 명령어 발행
 *
 * @reg_set: 입력 파이프라인 레지스터 집합 (ID_OC_MEM 단계)
 * @return:  void
 *
 * pending_writes와 pending_ldgsts를 설정하고 파이프라인으로 이동:
 *   1. load (비공유 메모리)이면: 목적 레지스터별 pending_writes를 n_accesses만큼 증가
 *      - n_accesses: coalescing 결과 생성된 메모리 접근 수 (accessq_count)
 *      - LDGSTS이면 pending_ldgsts도 증가
 *   2. op_pipe를 MEM__OP로 설정
 *   3. 통계 수집 (mem_instruction_stats, incmem_stat)
 *   4. pipelined_simd_unit::issue()로 파이프라인 이동
 *
 * pending_writes는 writeback()에서 각 메모리 접근이 완료될 때 감소하며,
 * 0이 되면 스코어보드에서 레지스터를 해제하여 RAW 해저드를 제거.
 *
 * 호출 체인: scheduler_unit::cycle() → m_shader->issue_warp(*m_mem_out, ...) → ldst_unit::issue()
 */
void ldst_unit::issue(register_set &reg_set) {
  warp_inst_t *inst = *(reg_set.get_ready()); // [한국어] 파이프라인 레지스터에서 발행할 명령어 포인터

  // record how many pending register writes/memory accesses there are for this
  // instruction
  // [한국어] load 명령어의 메모리 접근 수를 pending_writes에 기록:
  // 각 접근이 완료될 때마다 감소하여 0이 되면 스코어보드 해제
  assert(inst->empty() == false); // [한국어] 발행할 명령어가 비어있으면 안 됨
  if (inst->is_load() and inst->space.get_type() != shared_space) {
    // [한국어] 공유 메모리 제외한 load: 접근 수만큼 pending_writes 설정
    unsigned warp_id = inst->warp_id();
    unsigned n_accesses = inst->accessq_count(); // [한국어] 이 명령어의 coalesced 메모리 접근 수
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      unsigned reg_id = inst->out[r]; // [한국어] r번째 출력 레지스터 ID
      if (reg_id > 0) {
        m_pending_writes[warp_id][reg_id] += n_accesses; // [한국어] 이 레지스터의 pending 접근 수 증가
      }
    }
    if (inst->m_is_ldgsts) {
      m_pending_ldgsts[warp_id][inst->pc][inst->get_addr(0)] += n_accesses; // [한국어] LDGSTS 추적: pc와 첫 번째 주소로 구분하여 카운터 증가
    }
  }

  inst->op_pipe = MEM__OP; // [한국어] 파이프 타입을 MEM으로 설정 (warp_inst_complete에서 MEM 카운터 증가)
  // stat collection
  m_core->mem_instruction_stats(*inst); // [한국어] 메모리 명령어 타입별 통계 기록
  m_core->incmem_stat(m_core->get_config()->warp_size, 1); // [한국어] 메모리 명령어 카운터 증가 (warp_size개 스레드, 1 명령어)
  pipelined_simd_unit::issue(reg_set); // [한국어] 파이프라인으로 이동 (dispatch_reg에 배치)
}

/*
 * [한국어]
 * ldst_unit::writeback — 메모리 완료 응답을 레지스터 파일로 writeback
 *
 * @return: void
 *
 * 두 단계로 동작:
 *   1단계 — m_next_wb가 비어있지 않으면 오퍼랜드 콜렉터로 writeback 시도:
 *     - operand_collector->writeback()이 수락하면 pending_writes 감소
 *     - 비공유 메모리: pending_writes[warp_id][reg_id]를 감소하여 0이 되면 스코어보드 해제
 *     - 공유 메모리: 한 번에 스코어보드 해제
 *     - LDGSTS: pending_ldgsts 감소, 0이 되면 warp_inst_complete + unset_depbar
 *   2단계 — 5개 writeback 클라이언트를 라운드로빈 방식으로 순회하여 m_next_wb에 채움:
 *     - 클라이언트 0: 공유 메모리 (m_pipeline_reg[0])
 *     - 클라이언트 1: 텍스처 캐시 (m_L1T)
 *     - 클라이언트 2: 상수 캐시 (m_L1C)
 *     - 클라이언트 3: 글로벌/로컬 비캐시 (m_next_global)
 *     - 클라이언트 4: L1D 데이터 캐시 (m_L1D)
 *
 * m_writeback_arb는 마지막으로 서비스된 클라이언트+1을 기억하여 공정성 보장.
 *
 * 호출 체인: ldst_unit::cycle() → [이 함수] / shader_core_ctx::writeback()와는 별도
 */
void ldst_unit::writeback() {
  // process next instruction that is going to writeback
  if (!m_next_wb.empty()) {
    // [한국어] m_next_wb에 완료된 메모리 명령어가 있으면 오퍼랜드 콜렉터로 writeback 시도
    if (m_operand_collector->writeback(m_next_wb)) {
      // [한국어] 오퍼랜드 콜렉터가 writeback 수락: pending 레지스터 처리
      bool insn_completed = false; // [한국어] 명령어 전체 완료 여부 플래그
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        if (m_next_wb.out[r] > 0) {
          // [한국어] 출력 레지스터가 있는 경우
          if (m_next_wb.space.get_type() != shared_space) {
            // [한국어] 비공유 메모리 load: pending_writes 감소
            assert(m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0);
            unsigned still_pending =
                --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]]; // [한국어] 이 레지스터의 남은 pending 접근 수 감소
            if (!still_pending) {
              // [한국어] 모든 접근이 완료: 스코어보드에서 레지스터 해제
              m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]); // [한국어] pending_writes 맵에서 항목 제거 (메모리 절약)
              m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                            m_next_wb.out[r]); // [한국어] 스코어보드 해제: 이 레지스터를 기다리는 다음 명령어가 발행 가능
              insn_completed = true;
            }
          } else {  // shared
            // [한국어] 공유 메모리 load: 단일 접근이므로 즉시 스코어보드 해제
            m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                          m_next_wb.out[r]);
            insn_completed = true;
          }
        } else if (m_next_wb.m_is_ldgsts) {  // for LDGSTS instructions where no
                                             // output register is used
          // [한국어] LDGSTS: 출력 레지스터 없이 공유 메모리에 직접 쓰는 비동기 명령어
          m_pending_ldgsts[m_next_wb.warp_id()][m_next_wb.pc]
                          [m_next_wb.get_addr(0)]--; // [한국어] 이 PC+주소의 pending LDGSTS 카운터 감소
          if (m_pending_ldgsts[m_next_wb.warp_id()][m_next_wb.pc]
                              [m_next_wb.get_addr(0)] == 0) {
            insn_completed = true; // [한국어] 해당 LDGSTS 그룹의 모든 접근 완료
          }
          break; // [한국어] LDGSTS는 출력 레지스터가 없으므로 루프 종료
        }
      }
      if (insn_completed) {
        m_core->warp_inst_complete(m_next_wb); // [한국어] 명령어 완료 통보: 통계 카운터 업데이트, CTA 종료 여부 확인
        if (m_next_wb.m_is_ldgsts) {
          m_core->unset_depbar(m_next_wb); // [한국어] LDGSTS 완료 시 DEPBAR 의존성 카운터 감소 (LDGDEPBAR 그룹 완료 신호)
        }
      }

      m_next_wb.clear(); // [한국어] writeback 버퍼 비우기: 다음 사이클에 새로운 클라이언트 서비스 가능
      m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle; // [한국어] 마지막 writeback 사이클 기록 (유틸리티 통계)
      m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
    }
  }

  // [한국어] 2단계: 5개 writeback 클라이언트를 라운드로빈으로 순회하여 m_next_wb에 채움
  unsigned serviced_client = -1; // [한국어] 이번 사이클에 서비스된 클라이언트 ID (-1 = 없음)
  for (unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients);
       c++) {
    // [한국어] m_writeback_arb부터 시작하는 라운드로빈: 공정성 보장
    unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;
    switch (next_client) {
      case 0:  // shared memory
        // [한국어] 클라이언트 0: 공유 메모리 — 파이프라인 stage 0의 완료된 명령어
        if (!m_pipeline_reg[0]->empty()) {
          m_next_wb = *m_pipeline_reg[0]; // [한국어] 공유 메모리 완료 명령어를 writeback 버퍼로 복사
          if (m_next_wb.isatomic()) {
            m_next_wb.do_atomic(); // [한국어] 원자적 연산 실행 (공유 메모리 atomicAdd 등)
            m_core->decrement_atomic_count(m_next_wb.warp_id(),
                                           m_next_wb.active_count()); // [한국어] 진행 중인 원자적 연산 카운터 감소
          }
          m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id()); // [한국어] 파이프라인 내 명령어 수 감소 (CTA 완료 추적용)
          m_pipeline_reg[0]->clear(); // [한국어] 파이프라인 레지스터 비우기
          serviced_client = next_client;
        }
        break;
      case 1:  // texture response
        // [한국어] 클라이언트 1: 텍스처 캐시 — L1T에서 완료된 텍스처 접근
        if (m_L1T->access_ready()) {
          mem_fetch *mf = m_L1T->next_access(); // [한국어] L1T 완료 큐에서 mem_fetch 꺼내기
          m_next_wb = mf->get_inst(); // [한국어] mem_fetch에서 원본 명령어 참조 추출
          delete mf; // [한국어] mem_fetch 패킷 해제
          serviced_client = next_client;
        }
        break;
      case 2:  // const cache response
        // [한국어] 클라이언트 2: 상수 캐시 — L1C에서 완료된 상수 접근
        if (m_L1C->access_ready()) {
          mem_fetch *mf = m_L1C->next_access(); // [한국어] L1C 완료 큐에서 mem_fetch 꺼내기
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 3:  // global/local
        // [한국어] 클라이언트 3: 글로벌/로컬 비캐시 — bypassL1D 경로로 온 응답 (m_next_global)
        if (m_next_global) {
          m_next_wb = m_next_global->get_inst(); // [한국어] 글로벌 메모리 응답에서 명령어 추출
          if (m_next_global->isatomic()) {
            // [한국어] 글로벌 원자적 연산 완료: 카운터 감소
            m_core->decrement_atomic_count(
                m_next_global->get_wid(),
                m_next_global->get_access_warp_mask().count());
          }
          delete m_next_global; // [한국어] mem_fetch 해제
          m_next_global = NULL; // [한국어] 다음 사이클에 새 응답 수신 가능
          serviced_client = next_client;
        }
        break;
      case 4:
        // [한국어] 클라이언트 4: L1D 데이터 캐시 — L1D에서 완료된 캐시 접근 (hit 또는 miss 후 fill)
        if (m_L1D && m_L1D->access_ready()) {
          mem_fetch *mf = m_L1D->next_access(); // [한국어] L1D 완료 큐에서 mem_fetch 꺼내기
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      default:
        abort(); // [한국어] 예상치 못한 클라이언트 번호: 프로그래밍 오류
    }
  }
  // update arbitration priority only if:
  // 1. the writeback buffer was available
  // 2. a client was serviced
  // [한국어] 이번 사이클에 서비스된 클라이언트가 있으면 다음 사이클의 시작 클라이언트를 업데이트
  if (serviced_client != (unsigned)-1) {
    m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients; // [한국어] 라운드로빈: 마지막 서비스 클라이언트의 다음 번호로 이동
  }
}

/*
 * [한국어]
 * ldst_unit::clock_multiplier — ldst_unit의 클럭 배율 반환
 *
 * @return: 메모리 유닛의 클럭 배율 (정수)
 *
 * 메모리 유닛은 레지스터 파일의 다중 읽기 포트를 모델링하기 위해
 * 여러 사이클 동안 동작한다. mem_unit_ports가 설정되면 그 값을 사용하고,
 * 그렇지 않으면 mem_warp_parts를 사용한다.
 *
 * 호출 체인: shader_core_ctx::cycle() → ldst_unit::clock_multiplier()
 */
unsigned ldst_unit::clock_multiplier() const {
  // to model multiple read port, we give multiple cycles for the memory units
  if (m_config->mem_unit_ports)
    return m_config->mem_unit_ports; // [한국어] 설정된 메모리 유닛 포트 수 사용 (다중 읽기 포트 모델링)
  else
    return m_config->mem_warp_parts; // [한국어] warp를 나누는 파트 수 사용 (부분 warp 처리 모델링)
}
/*
void ldst_unit::issue( register_set &reg_set )
{
        warp_inst_t* inst = *(reg_set.get_ready());
   // stat collection
   m_core->mem_instruction_stats(*inst);

   // record how many pending register writes/memory accesses there are for this
instruction assert(inst->empty() == false); if (inst->is_load() and
inst->space.get_type() != shared_space) { unsigned warp_id = inst->warp_id();
      unsigned n_accesses = inst->accessq_count();
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
         unsigned reg_id = inst->out[r];
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses;
         }
      }
   }

   pipelined_simd_unit::issue(reg_set);
}
*/
/*
 * [한국어]
 * ldst_unit::cycle — 메모리 유닛 한 사이클 처리 (ldst_unit의 최상위 사이클 함수)
 *
 * @return: void
 *
 * 매 사이클 다음 단계를 순서대로 처리:
 *   1. writeback(): 완료된 메모리 응답을 레지스터 파일로 writeback
 *   2. 파이프라인 버블 이동: stage[n]이 비었고 stage[n+1]이 차있으면 앞으로 이동
 *   3. m_response_fifo 처리: NoC에서 도착한 응답을 캐시로 fill하거나 writeback 준비:
 *      - TEXTURE_ACC_R: L1T에 fill
 *      - CONST_ACC_R: L1C에 fill
 *      - WRITE_ACK / perfect mem write: store_ack() 호출
 *      - 나머지 load 응답: bypassL1D 여부에 따라 m_next_global 또는 L1D fill
 *   4. 캐시 cycle: L1T, L1C, L1D (+ l1_latency_queue_cycle)
 *   5. dispatch_reg 처리: shared_cycle/constant_cycle/texture_cycle/memory_cycle 호출
 *      - 하나라도 stall이면 통계 기록 후 early return
 *      - load 완료: 공유 메모리는 파이프라인으로, 글로벌/로컬은 pending 확인 후 클리어
 *      - store 완료: store_req_count 감소
 *
 * 이 함수는 SM의 cycle()에서 clock_multiplier()만큼 반복 호출된다.
 *
 * 호출 체인: shader_core_ctx::cycle() → ldst_unit::cycle() (clock_multiplier번 반복)
 */
void ldst_unit::cycle() {
  writeback(); // [한국어] 1단계: 완료된 메모리 응답을 레지스터 파일로 writeback

  // [한국어] 2단계: 파이프라인 버블 이동 (공유 메모리 파이프라인 전진)
  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
    if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty())
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]); // [한국어] 빈 stage로 명령어 이동 (파이프라인 전진)

  // [한국어] 3단계: m_response_fifo에서 NoC 응답 처리
  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front(); // [한국어] 응답 FIFO의 첫 번째 mem_fetch (L2/DRAM에서 도착)
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      // [한국어] 텍스처 읽기 응답: L1T 캐시에 fill
      if (m_L1T->fill_port_free()) {
        m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] L1T에 캐시 라인 fill (fill 포트가 여유 있을 때만)
        m_response_fifo.pop_front(); // [한국어] FIFO에서 처리된 항목 제거
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      // [한국어] 상수 읽기 응답: L1C 캐시에 fill
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
                       m_core->get_gpu()->gpu_sim_cycle +
                           m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] 상태 기록 (타이밍 통계)
        m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] L1C에 캐시 라인 fill
        m_response_fifo.pop_front();
      }
    } else {
      // [한국어] 글로벌/로컬/공유 메모리 응답
      if (mf->get_type() == WRITE_ACK ||
          ((m_config->gpgpu_perfect_mem || m_memory_config->SST_mode) &&
           mf->get_is_write())) {
        // SST memory is handled by SST mem hierarchy
        // Perfect mem
        // [한국어] write ACK 또는 perfect mem / SST 모드의 쓰기 응답: store_ack 처리
        m_core->store_ack(mf); // [한국어] store_req_count 감소, warp unblock 확인
        m_response_fifo.pop_front();
        delete mf; // [한국어] mem_fetch 패킷 해제
      } else {
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only
        // [한국어] load miss 응답: 캐시 bypass 여부 판단

        bool bypassL1D = false;
        if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
          bypassL1D = true; // [한국어] CACHE_GLOBAL cache_op (streaming) 또는 L1D 비활성화 → bypass
        } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                   mf->get_access_type() ==
                       GLOBAL_ACC_W) {  // global memory access
          if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true; // [한국어] gmem_skip_L1D 설정: 모든 글로벌 접근이 L1D를 우회
        }
        if (bypassL1D) {
          // [한국어] L1D bypass: m_next_global에 임시 저장 (writeback 클라이언트 3이 처리)
          if (m_next_global == NULL) {
            mf->set_status(IN_SHADER_FETCHED,
                           m_core->get_gpu()->gpu_sim_cycle +
                               m_core->get_gpu()->gpu_tot_sim_cycle); // [한국어] 상태 업데이트
            m_response_fifo.pop_front();
            m_next_global = mf; // [한국어] 다음 writeback 사이클에서 클라이언트 3이 처리
          }
        } else {
          // [한국어] L1D에 fill: 캐시 라인을 L1D에 적재하여 다음 접근 시 hit
          if (m_L1D->fill_port_free()) {
            m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }

  // [한국어] 4단계: 캐시 사이클 처리 (내부 파이프라인 전진)
  m_L1T->cycle(); // [한국어] 텍스처 캐시 사이클: miss 큐 처리, fill 파이프라인 전진
  m_L1C->cycle(); // [한국어] 상수 캐시 사이클
  if (m_L1D) {
    m_L1D->cycle(); // [한국어] 데이터 캐시 사이클
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle(); // [한국어] L1D 레이턴시 큐 사이클: 뱅크별 레이턴시 파이프라인 처리
  }

  // [한국어] 5단계: dispatch_reg(발행된 명령어)를 실제 메모리 서브시스템에 접근시도
  warp_inst_t &pipe_reg = *m_dispatch_reg; // [한국어] 현재 처리할 메모리 명령어 (issue()에서 dispatch_reg로 이동된 것)
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL; // [한국어] stall 타입 초기화
  mem_stage_access_type type; // [한국어] stall 통계를 위한 메모리 접근 타입
  bool done = true; // [한국어] 모든 메모리 접근이 처리되었는지 여부
  done &= shared_cycle(pipe_reg, rc_fail, type);   // [한국어] 공유 메모리 접근 처리
  done &= constant_cycle(pipe_reg, rc_fail, type); // [한국어] 상수 캐시 접근 처리
  done &= texture_cycle(pipe_reg, rc_fail, type);  // [한국어] 텍스처 캐시 접근 처리
  done &= memory_cycle(pipe_reg, rc_fail, type);   // [한국어] 글로벌/로컬 메모리 접근 처리 (L1D 또는 NoC 경유)
  m_mem_rc = rc_fail; // [한국어] stall 타입 저장 (display_pipeline에서 디버그 출력에 사용)

  if (!done) {  // log stall types and return
    // [한국어] 어느 하나라도 stall: 통계 기록 후 early return
    assert(rc_fail != NO_RC_FAIL);
    m_stats->gpgpu_n_stall_shd_mem++; // [한국어] 공유 메모리 stall 전체 카운터
    m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++; // [한국어] stall 타입별 세분 통계
    return;
  }

  if (!pipe_reg.empty()) {
    unsigned warp_id = pipe_reg.warp_id();
    if (pipe_reg.is_load()) {
      if (pipe_reg.space.get_type() == shared_space) {
        // [한국어] 공유 메모리 load: 레이턴시 파이프라인의 끝 stage로 이동
        if (m_pipeline_reg[m_config->smem_latency - 1]->empty()) {
          // new shared memory request
          move_warp(m_pipeline_reg[m_config->smem_latency - 1], m_dispatch_reg); // [한국어] 공유 메모리 파이프라인의 마지막 stage에 배치
          m_dispatch_reg->clear();
        }
      } else {
        // if( pipe_reg.active_count() > 0 ) {
        //    if( !m_operand_collector->writeback(pipe_reg) )
        //        return;
        //}

        // [한국어] 글로벌/로컬 load: pending_writes가 0인지 확인 후 dispatch_reg 클리어
        bool pending_requests = false;
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
          unsigned reg_id = pipe_reg.out[r];
          if (reg_id > 0) {
            if (m_pending_writes[warp_id].find(reg_id) !=
                m_pending_writes[warp_id].end()) {
              if (m_pending_writes[warp_id][reg_id] > 0) {
                pending_requests = true; // [한국어] 아직 완료되지 않은 메모리 접근이 남아있음
                break;
              } else {
                // this instruction is done already
                m_pending_writes[warp_id].erase(reg_id);
              }
            }
          }
        }
        if (!pending_requests) {
          // [한국어] 모든 pending 접근이 완료된 경우: 명령어 완료 처리
          m_core->warp_inst_complete(*m_dispatch_reg); // [한국어] warp_inst_complete: 통계 갱신, 커널 완료 확인
          m_scoreboard->releaseRegisters(m_dispatch_reg); // [한국어] 스코어보드에서 모든 출력 레지스터 해제

          // release LDGSTS
          if (m_dispatch_reg->m_is_ldgsts) {
            // m_pending_ldgsts[m_dispatch_reg->warp_id()][m_dispatch_reg->pc][m_dispatch_reg->get_addr(0)]--;
            if (m_pending_ldgsts[m_dispatch_reg->warp_id()][m_dispatch_reg->pc]
                                [m_dispatch_reg->get_addr(0)] == 0) {
              m_core->unset_depbar(*m_dispatch_reg); // [한국어] LDGSTS 그룹 완료: DEPBAR 의존성 카운터 감소
            }
          }
        }
        m_core->dec_inst_in_pipeline(warp_id); // [한국어] 파이프라인 내 명령어 수 감소 (CTA 종료 판단에 사용)
        m_dispatch_reg->clear(); // [한국어] dispatch_reg 초기화: 다음 사이클에 새 명령어 수신 가능
      }
    } else {
      // stores exit pipeline here
      // [한국어] store 명령어: 메모리에 전송 완료 → 파이프라인에서 제거
      m_core->dec_inst_in_pipeline(warp_id);
      m_core->warp_inst_complete(*m_dispatch_reg); // [한국어] store 완료 통보 (WRITE_ACK는 별도로 store_ack에서 처리)
      m_dispatch_reg->clear();
    }
  }
}

/*
 * [한국어]
 * shader_core_ctx::register_cta_thread_exit — CTA 내 스레드 1개 완료 등록
 *
 * @cta_num: 완료된 스레드가 속한 CTA(협동 스레드 배열) 번호
 * @kernel:  실행 중인 커널 정보
 * @return:  void
 *
 * 스레드가 종료될 때마다 호출되어 m_cta_status[cta_num]을 감소시킨다.
 * m_cta_status가 0이 되면 해당 CTA의 모든 스레드가 완료:
 *   - CTA 통계 업데이트 (ctas_completed, inc_completed_cta)
 *   - 배리어 해제 (deallocate_barrier)
 *   - SM 자원 해제 (release_shader_resource_1block)
 *   - 커널 running 카운터 감소, 모든 CTA가 완료되면 set_kernel_done 호출
 *
 * warp_inst_complete → register_cta_thread_exit 경로로 호출된다.
 *
 * 호출 체인: warp_inst_complete() → [이 함수] → set_kernel_done()
 */
void shader_core_ctx::register_cta_thread_exit(unsigned cta_num,
                                               kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0); // [한국어] CTA 내 남은 스레드 수가 0보다 커야 함
  m_cta_status[cta_num]--;  // [한국어] 이 CTA 내 완료된 스레드 수 감소
  if (!m_cta_status[cta_num]) {
    // [한국어] 이 CTA의 모든 스레드가 완료: CTA 종료 처리
    // Increment the completed CTAs
    m_stats->ctas_completed++;  // [한국어] 완료된 CTA 통계 카운터 증가
    m_gpu->inc_completed_cta(); // [한국어] GPU 전체 완료 CTA 수 증가
    m_n_active_cta--;           // [한국어] 이 SM의 활성 CTA 수 감소
    m_barriers.deallocate_barrier(cta_num); // [한국어] 이 CTA에 할당된 배리어 슬롯 해제 (__syncthreads 리소스 반환)
    shader_CTA_count_unlog(m_sid, 1); // [한국어] 파워 통계: CTA count unlog (전력 모델에 반영)

    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
        m_n_active_cta);

    if (m_n_active_cta == 0) {
      // [한국어] 이 SM의 모든 CTA가 완료: SM이 비어있음
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        // [한국어] 동시 커널 실행 시: 현재 완료된 커널이 SM의 active 커널이 아닐 수 있음
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
      m_kernel = NULL; // [한국어] SM에 할당된 커널 해제 (새로운 커널을 받을 수 있는 상태)
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel); // [한국어] 이 CTA가 점유하던 SM 자원 해제 (레지스터, 공유 메모리 등)
    kernel->dec_running(); // [한국어] 커널의 running CTA 카운터 감소
    if (!m_gpu->kernel_more_cta_left(kernel)) {
      // [한국어] 이 커널에 더 이상 발행할 CTA가 없음: 커널 종료 여부 확인
      if (!kernel->running()) {
        // [한국어] 모든 CTA가 완료되어 커널도 완료됨
        SHADER_DPRINTF(LIVENESS,
                       "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                       "finished on shader %u.\n",
                       kernel->get_uid(), kernel->name().c_str(), m_sid);

        if (m_kernel == kernel) m_kernel = NULL;
        m_gpu->set_kernel_done(kernel); // [한국어] 커널 완료 통보: 시뮬레이터가 다음 커널로 진행
      }
    }
  }
}

/*
 * [한국어]
 * gpgpu_sim::shader_print_runtime_stat — SM 런타임 통계 출력 (현재 비활성)
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * SM별 명령어 실행 수, 스레드 수, 분기 발생 수 등을 출력하는 디버그 함수.
 * 현재는 전체 내용이 주석 처리되어 있어 아무것도 출력하지 않는다.
 *
 * 호출 체인: gpgpu_sim::gpu_print_stat() → [이 함수]
 */
void gpgpu_sim::shader_print_runtime_stat(FILE *fout) {
  /*
 fprintf(fout, "SHD_INSN: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_THDS: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_not_completed());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_DIVG: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
 fprintf(fout, "\n");

 fprintf(fout, "THD_INSN: ");
 for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
    fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
 fprintf(fout, "\n");
 */
}

/*
 * [한국어]
 * gpgpu_sim::shader_print_scheduler_stat — warp 스케줄러 통계 출력
 *
 * @fout:               출력 파일 포인터
 * @print_dynamic_info: true이면 dynamic warp ID 기반, false이면 warp ID 기반 분포 출력
 * @return: void
 *
 * 특정 SM(gpgpu_warp_issue_shader로 지정)의 warp ID별 발행 분포를 출력.
 * 각 warp가 얼마나 자주 발행되었는지의 히스토그램을 출력한다.
 *
 * 호출 체인: gpgpu_sim::gpu_print_stat() → [이 함수]
 */
void gpgpu_sim::shader_print_scheduler_stat(FILE *fout,
                                            bool print_dynamic_info) const {
  fprintf(fout, "ctas_completed %d, ", m_shader_stats->ctas_completed); // [한국어] 완료된 CTA 수 출력
  // Print out the stats from the sampling shader core
  const unsigned scheduler_sampling_core =
      m_shader_config->gpgpu_warp_issue_shader; // [한국어] 통계 샘플링 SM ID (gpgpusim.config의 gpgpu_warp_issue_shader)
#define STR_SIZE 55
  char name_buff[STR_SIZE];
  name_buff[STR_SIZE - 1] = '\0';
  const std::vector<unsigned> &distro =
      print_dynamic_info
          ? m_shader_stats->get_dynamic_warp_issue()[scheduler_sampling_core]
          : m_shader_stats->get_warp_slot_issue()[scheduler_sampling_core]; // [한국어] dynamic_info 여부에 따라 다른 발행 분포 통계 선택
  if (print_dynamic_info) {
    snprintf(name_buff, STR_SIZE - 1, "dynamic_warp_id"); // [한국어] 동적 warp ID 기반 분포
  } else {
    snprintf(name_buff, STR_SIZE - 1, "warp_id"); // [한국어] 정적 warp ID 기반 분포
  }
  fprintf(fout, "Shader %d %s issue ditsribution:\n", scheduler_sampling_core,
          name_buff);
  const unsigned num_warp_ids = distro.size();
  // First print out the warp ids
  fprintf(fout, "%s:\n", name_buff);
  for (unsigned warp_id = 0; warp_id < num_warp_ids; ++warp_id) {
    fprintf(fout, "%d, ", warp_id); // [한국어] warp ID 목록 출력
  }

  fprintf(fout, "\ndistro:\n");
  // Then print out the distribution of instuctions issued
  for (std::vector<unsigned>::const_iterator iter = distro.begin();
       iter != distro.end(); iter++) {
    fprintf(fout, "%d, ", *iter); // [한국어] 각 warp ID의 발행 횟수 출력
  }
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * gpgpu_sim::shader_print_cache_stats — 모든 SM의 L1 캐시 통계 출력
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * L1I, L1D, L1C, L1T 캐시의 accesses, misses, miss_rate, pending_hits,
 * reservation_fails를 집계하여 출력. 비활성화된 캐시는 출력 생략.
 *
 * 호출 체인: gpgpu_sim::gpu_print_stat() → [이 함수]
 */
void gpgpu_sim::shader_print_cache_stats(FILE *fout) const {
  // L1I
  struct cache_sub_stats total_css;
  struct cache_sub_stats css;

  if (!m_shader_config->m_L1I_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "\n========= Core cache stats =========\n");
    fprintf(fout, "L1I_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1I_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1I_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1I_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1I_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1I_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1D
  if (!m_shader_config->m_L1D_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1D_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->get_L1D_sub_stats(css);

      fprintf(stdout,
              "\tL1D_cache_core[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, css.accesses, css.misses,
              (double)css.misses / (double)css.accesses, css.pending_hits,
              css.res_fails);

      total_css += css;
    }
    fprintf(fout, "\tL1D_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1D_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1D_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1D_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
    total_css.print_port_stats(fout, "\tL1D_cache");
  }

  // L1C
  if (!m_shader_config->m_L1C_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1C_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1C_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1C_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1C_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1C_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1C_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1T
  if (!m_shader_config->m_L1T_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1T_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1T_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1T_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1T_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1T_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1T_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }
}

/*
 * [한국어]
 * gpgpu_sim::shader_print_l1_miss_stat — L1D miss 통계 집계 출력
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * 모든 클러스터의 L1D 캐시 miss/access 수를 집계하여 전체 miss rate 출력.
 *
 * 호출 체인: gpgpu_sim::gpu_print_stat() → [이 함수]
 */
void gpgpu_sim::shader_print_l1_miss_stat(FILE *fout) const {
  unsigned total_d1_misses = 0, total_d1_accesses = 0;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
    unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
    m_cluster[i]->print_cache_stats(fout, cluster_d1_accesses,
                                    custer_d1_misses); // [한국어] 클러스터 i의 캐시 통계 집계
    total_d1_misses += custer_d1_misses;
    total_d1_accesses += cluster_d1_accesses;
  }
  fprintf(fout, "total_dl1_misses=%d\n", total_d1_misses);
  fprintf(fout, "total_dl1_accesses=%d\n", total_d1_accesses);
  fprintf(fout, "total_dl1_miss_rate= %f\n",
          (float)total_d1_misses / (float)total_d1_accesses);
  /*
  fprintf(fout, "THD_INSN_AC: ");
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i)); fprintf(fout, "\n"); fprintf(fout,
  "T_L1_Acc: "); //l1 access per thread for (unsigned i=0;
  i<m_shader_config->n_thread_per_shader; i++) fprintf(fout, "%d ",
  m_sc[0]->get_thread_n_l1_access_ac(i)); fprintf(fout, "\n");

  //per warp
  int temp =0;
  fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp=0;
  fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i) ); if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp =0;
  fprintf(fout, "W_L1_Acc: "); //l1 access per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_access_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  */
}

/*
 * [한국어]
 * warp_inst_t::print — warp 명령어 정보 출력 (디버그용)
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * PC, warp ID, active mask (비트맵), PTX 명령어 텍스트를 출력.
 * 비어있으면 "bubble" 출력.
 *
 * 호출 체인: display_pipeline(), ldst_unit::print() 등 디버그 경로
 */
void warp_inst_t::print(FILE *fout) const {
  if (empty()) {
    fprintf(fout, "bubble\n"); // [한국어] 빈 파이프라인 슬롯은 "bubble"로 표시
    return;
  } else
    fprintf(fout, "0x%04llx ", pc); // [한국어] 명령어 PC 주소 출력
  fprintf(fout, "w%02d[", m_warp_id); // [한국어] warp ID 출력
  for (unsigned j = 0; j < m_config->warp_size; j++)
    fprintf(fout, "%c", (active(j) ? '1' : '0')); // [한국어] active mask 비트맵 출력 ('1'=활성, '0'=비활성)
  fprintf(fout, "]: ");
  m_config->gpgpu_ctx->func_sim->ptx_print_insn(pc, fout); // [한국어] PTX IR에서 이 PC의 명령어 텍스트 출력
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * shader_core_ctx::incexecstat — 명령어 타입별 전력 통계 카운터 증가 (AccelWattch)
 *
 * @inst: 발행된 warp 명령어 포인터
 * @return: void
 *
 * sp_op 필드에 따라 정수/FP/DP/SFU/텍스처 연산 카운터를 증가시킨다.
 * AccelWattch 전력 모델이 활성화된 경우에만 동작 (g_power_simulation_enabled).
 * scaling_coeffs는 연산 타입별 전력 스케일링 계수 (마이크로벤치마킹 결과).
 *
 * 호출 체인: execute() → issue_warp 완료 후 → [이 함수]
 */
void shader_core_ctx::incexecstat(warp_inst_t *&inst) {
  // Latency numbers for next operations are used to scale the power values
  // for special operations, according observations from microbenchmarking
  // TODO: put these numbers in the xml configuration
  if (get_gpu()->get_config().g_power_simulation_enabled) {
    switch (inst->sp_op) {
      case INT__OP:
        incialu_stat(inst->active_count(), scaling_coeffs->int_coeff);
        break;
      case INT_MUL_OP:
        incimul_stat(inst->active_count(), scaling_coeffs->int_mul_coeff);
        break;
      case INT_MUL24_OP:
        incimul24_stat(inst->active_count(), scaling_coeffs->int_mul24_coeff);
        break;
      case INT_MUL32_OP:
        incimul32_stat(inst->active_count(), scaling_coeffs->int_mul32_coeff);
        break;
      case INT_DIV_OP:
        incidiv_stat(inst->active_count(), scaling_coeffs->int_div_coeff);
        break;
      case FP__OP:
        incfpalu_stat(inst->active_count(), scaling_coeffs->fp_coeff);
        break;
      case FP_MUL_OP:
        incfpmul_stat(inst->active_count(), scaling_coeffs->fp_mul_coeff);
        break;
      case FP_DIV_OP:
        incfpdiv_stat(inst->active_count(), scaling_coeffs->fp_div_coeff);
        break;
      case DP___OP:
        incdpalu_stat(inst->active_count(), scaling_coeffs->dp_coeff);
        break;
      case DP_MUL_OP:
        incdpmul_stat(inst->active_count(), scaling_coeffs->dp_mul_coeff);
        break;
      case DP_DIV_OP:
        incdpdiv_stat(inst->active_count(), scaling_coeffs->dp_div_coeff);
        break;
      case FP_SQRT_OP:
        incsqrt_stat(inst->active_count(), scaling_coeffs->sqrt_coeff);
        break;
      case FP_LG_OP:
        inclog_stat(inst->active_count(), scaling_coeffs->log_coeff);
        break;
      case FP_SIN_OP:
        incsin_stat(inst->active_count(), scaling_coeffs->sin_coeff);
        break;
      case FP_EXP_OP:
        incexp_stat(inst->active_count(), scaling_coeffs->exp_coeff);
        break;
      case TENSOR__OP:
        inctensor_stat(inst->active_count(), scaling_coeffs->tensor_coeff);
        break;
      case TEX__OP:
        inctex_stat(inst->active_count(), scaling_coeffs->tex_coeff);
        break;
      default:
        break;
    }
    if (inst->const_cache_operand)  // warp has const address space load as one
                                    // operand
      inc_const_accesses(1); // [한국어] 상수 캐시 접근 카운터 증가 (상수 오퍼랜드 사용 시)
  }
}

/*
 * [한국어]
 * shader_core_ctx::print_stage — 특정 파이프라인 스테이지 내용 출력
 *
 * @stage: 파이프라인 스테이지 인덱스 (0=IF/ID, 1=OC, ... )
 * @fout:  출력 파일 포인터
 * @return: void
 *
 * 호출 체인: display_pipeline() → [이 함수]
 */
void shader_core_ctx::print_stage(unsigned int stage, FILE *fout) const {
  m_pipeline_reg[stage].print(fout); // [한국어] 파이프라인 레지스터 집합의 내용 출력
  // m_pipeline_reg[stage].print(fout);
}

/*
 * [한국어]
 * shader_core_ctx::display_simt_state — SIMT 스택 상태 출력 (디버그용)
 *
 * @fout: 출력 파일 포인터
 * @mask: 출력 마스크 비트 (4=SIMT 상태, 8=완료된 스레드 사이클 포함)
 * @return: void
 *
 * mask & 4가 설정되고 POST_DOMINATOR 모델인 경우 각 warp의 SIMT 스택을 출력.
 * mask & 8이 설정되면 완료된 스레드의 donecycle도 포함.
 *
 * 호출 체인: display_pipeline() → [이 함수]
 */
void shader_core_ctx::display_simt_state(FILE *fout, int mask) const {
  if ((mask & 4) && m_config->model == POST_DOMINATOR) {
    fprintf(fout, "per warp SIMT control-flow state:\n");
    unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
    for (unsigned i = 0; i < n; i++) {
      unsigned nactive = 0;
      for (unsigned j = 0; j < m_config->warp_size; j++) {
        unsigned tid = i * m_config->warp_size + j;
        int done = ptx_thread_done(tid);
        nactive += (ptx_thread_done(tid) ? 0 : 1);
        if (done && (mask & 8)) {
          unsigned done_cycle = m_thread[tid]->donecycle();
          if (done_cycle) {
            printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle);
          }
        }
      }
      if (nactive == 0) {
        continue;
      }
      m_simt_stack[i]->print(fout);
    }
    fprintf(fout, "\n");
  }
}

/*
 * [한국어]
 * ldst_unit::print — ldst_unit 전체 상태 출력 (디버그용)
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * dispatch_reg(현재 처리 중인 명령어), stall 조건, writeback 버퍼(m_next_wb),
 * 마지막 writeback 사이클, pending_writes 맵, L1C/L1T/L1D 상태, 응답 FIFO를 출력.
 *
 * 호출 체인: shader_core_ctx::display_pipeline() → [이 함수]
 */
void ldst_unit::print(FILE *fout) const {
  fprintf(fout, "LD/ST unit  = ");
  m_dispatch_reg->print(fout); // [한국어] 현재 dispatch_reg의 명령어 출력
  if (m_mem_rc != NO_RC_FAIL) {
    // [한국어] stall 조건이 있으면 그 종류 출력
    fprintf(fout, "              LD/ST stall condition: ");
    switch (m_mem_rc) {
      case BK_CONF:
        fprintf(fout, "BK_CONF"); // [한국어] 뱅크 충돌
        break;
      case MSHR_RC_FAIL:
        fprintf(fout, "MSHR_RC_FAIL"); // [한국어] MSHR 예약 실패
        break;
      case ICNT_RC_FAIL:
        fprintf(fout, "ICNT_RC_FAIL"); // [한국어] NoC 버퍼 가득참
        break;
      case COAL_STALL:
        fprintf(fout, "COAL_STALL"); // [한국어] 메모리 coalescing 대기
        break;
      case WB_ICNT_RC_FAIL:
        fprintf(fout, "WB_ICNT_RC_FAIL"); // [한국어] writeback 시 NoC 버퍼 실패
        break;
      case WB_CACHE_RSRV_FAIL:
        fprintf(fout, "WB_CACHE_RSRV_FAIL"); // [한국어] writeback 시 캐시 예약 실패
        break;
      case N_MEM_STAGE_STALL_TYPE:
        fprintf(fout, "N_MEM_STAGE_STALL_TYPE");
        break;
      default:
        abort();
    }
    fprintf(fout, "\n");
  }
  fprintf(fout, "LD/ST wb    = ");
  m_next_wb.print(fout); // [한국어] 다음 writeback할 명령어 출력
  fprintf(
      fout,
      "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);
  fprintf(fout, "Pending register writes:\n");
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/> >::const_iterator
      w;
  for (w = m_pending_writes.begin(); w != m_pending_writes.end(); w++) {
    unsigned warp_id = w->first;
    const std::map<unsigned /*regnum*/, unsigned /*count*/> &warp_info =
        w->second;
    if (warp_info.empty()) continue;
    fprintf(fout, "  w%2u : ", warp_id); // [한국어] warp ID 출력
    std::map<unsigned /*regnum*/, unsigned /*count*/>::const_iterator r;
    for (r = warp_info.begin(); r != warp_info.end(); ++r) {
      fprintf(fout, "  %u(%u)", r->first, r->second); // [한국어] 레지스터 번호(pending 접근 수) 출력
    }
    fprintf(fout, "\n");
  }
  m_L1C->display_state(fout); // [한국어] L1 상수 캐시 상태 출력
  m_L1T->display_state(fout); // [한국어] L1 텍스처 캐시 상태 출력
  if (!m_config->m_L1D_config.disabled()) m_L1D->display_state(fout); // [한국어] L1 데이터 캐시 상태 출력 (활성화된 경우)
  fprintf(fout, "LD/ST response FIFO (occupancy = %zu):\n",
          m_response_fifo.size()); // [한국어] 응답 FIFO 점유율 출력
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout); // [한국어] 각 mem_fetch 패킷 정보 출력
  }
}

/*
 * [한국어]
 * shader_core_ctx::display_pipeline — SM 파이프라인 전체 상태 출력 (디버그용)
 *
 * @fout:      출력 파일 포인터
 * @print_mem: 메모리 유닛 상세 출력 여부
 * @mask:      출력 마스크 (SIMT 상태, 완료 스레드 등)
 * @return:    void
 *
 * SM의 현재 cycle, L1I 상태, IF/ID 버퍼, I-buffer, SIMT 스택, 스코어보드,
 * 오퍼랜드 콜렉터, 모든 파이프라인 레지스터, FU 상태, result_bus, 활성 스레드를 출력.
 * 시뮬레이터 디버깅, 단계별 추적 분석에 사용.
 *
 * 호출 체인: gpgpu_sim::cycle() 디버그 경로 → [이 함수]
 */
void shader_core_ctx::display_pipeline(FILE *fout, int print_mem,
                                       int mask) const {
  fprintf(fout, "=================================================\n");
  fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid,
          m_gpu->gpu_tot_sim_cycle, m_gpu->gpu_sim_cycle, m_not_completed);
  fprintf(fout, "=================================================\n");

  dump_warp_state(fout); // [한국어] 각 warp의 상태(대기, 실행, 완료) 출력
  fprintf(fout, "\n");

  m_L1I->display_state(fout); // [한국어] L1 명령어 캐시 상태 출력

  fprintf(fout, "IF/ID       = ");
  if (!m_inst_fetch_buffer.m_valid)
    fprintf(fout, "bubble\n");
  else {
    fprintf(fout, "w%2u : pc = 0x%llx, nbytes = %u\n",
            m_inst_fetch_buffer.m_warp_id, m_inst_fetch_buffer.m_pc,
            m_inst_fetch_buffer.m_nbytes);
  }
  fprintf(fout, "\nibuffer status:\n");
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (!m_warp[i]->ibuffer_empty()) m_warp[i]->print_ibuffer(fout);
  }
  fprintf(fout, "\n");
  display_simt_state(fout, mask);
  fprintf(fout, "-------------------------- Scoreboard\n");
  m_scoreboard->printContents();
  /*
     fprintf(fout,"ID/OC (SP)  = ");
     print_stage(ID_OC_SP, fout);
     fprintf(fout,"ID/OC (SFU) = ");
     print_stage(ID_OC_SFU, fout);
     fprintf(fout,"ID/OC (MEM) = ");
     print_stage(ID_OC_MEM, fout);
  */
  fprintf(fout, "-------------------------- OP COL\n");
  m_operand_collector.dump(fout);
  /* fprintf(fout, "OC/EX (SP)  = ");
     print_stage(OC_EX_SP, fout);
     fprintf(fout, "OC/EX (SFU) = ");
     print_stage(OC_EX_SFU, fout);
     fprintf(fout, "OC/EX (MEM) = ");
     print_stage(OC_EX_MEM, fout);
  */
  fprintf(fout, "-------------------------- Pipe Regs\n");

  for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
    fprintf(fout, "--- %s ---\n", pipeline_stage_name_decode[i]);
    print_stage(i, fout);
    fprintf(fout, "\n");
  }

  fprintf(fout, "-------------------------- Fu\n");
  for (unsigned n = 0; n < m_num_function_units; n++) {
    m_fu[n]->print(fout);
    fprintf(fout, "---------------\n");
  }
  fprintf(fout, "-------------------------- other:\n");

  for (unsigned i = 0; i < num_result_bus; i++) {
    std::string bits = m_result_bus[i]->to_string();
    fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str());
  }
  fprintf(fout, "EX/WB      = ");
  print_stage(EX_WB, fout);
  fprintf(fout, "\n");
  fprintf(
      fout,
      "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);

  if (m_active_threads.count() <= 2 * m_config->warp_size) {
    fprintf(fout, "Active Threads : ");
    unsigned last_warp_id = -1;
    for (unsigned tid = 0; tid < m_active_threads.size(); tid++) {
      unsigned warp_id = tid / m_config->warp_size;
      if (m_active_threads.test(tid)) {
        if (warp_id != last_warp_id) {
          fprintf(fout, "\n  warp %u : ", warp_id);
          last_warp_id = warp_id;
        }
        fprintf(fout, "%u ", tid);
      }
    }
  }
}

/*
 * [한국어]
 * shader_core_config::max_cta — SM에서 동시에 실행 가능한 최대 CTA 수 계산
 *
 * @k: 실행할 커널 정보 (스레드 수, 레지스터, 공유 메모리 등)
 * @return: 이 SM에서 동시 실행 가능한 최대 CTA 수
 *
 * 실제 GPU의 CTA occupancy 계산 로직을 모방. 다음 4가지 제약 중 최솟값:
 *   1. 스레드 수 제한: n_thread_per_shader / padded_cta_size
 *   2. 공유 메모리 제한: gpgpu_shmem_size / kernel_smem
 *   3. 레지스터 제한: gpgpu_shader_registers / (cta_size * regs_per_thread)
 *   4. CTA 슬롯 제한: max_cta_per_core
 *
 * adaptive_cache_config가 활성화된 경우 (Volta/Ampere unified L1/shmem):
 *   총 공유 메모리 사용량에 따라 L1D 연관도(associativity)를 동적으로 재설정.
 *
 * 호출 체인: gpgpu_sim::issue_block2core() → [이 함수] (몇 개의 CTA를 할당할지 결정)
 */
unsigned int shader_core_config::max_cta(const kernel_info_t &k) const {
  unsigned threads_per_cta = k.threads_per_cta(); // [한국어] CTA당 스레드 수
  const class function_info *kernel = k.entry();  // [한국어] 커널 함수 정보 (PTX)
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size); // [한국어] warp_size(32)의 배수로 올림: 실제 GPU와 동일한 스레드 슬롯 소비

  // Limit by n_threads/shader
  unsigned int result_thread = n_thread_per_shader / padded_cta_size; // [한국어] SM의 최대 스레드 수 대비 CTA 크기로 나눈 값

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel); // [한국어] 커널의 레지스터/공유 메모리 사용량 정보

  // Limit by shmem/shader
  unsigned int result_shmem = (unsigned)-1; // [한국어] 초기값: 무한 (제약 없음)
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem; // [한국어] 공유 메모리 제한: SM 공유 메모리 총량 / CTA당 공유 메모리

  // Limit by register count, rounded up to multiple of 4.
  unsigned int result_regs = (unsigned)-1; // [한국어] 초기값: 무한 (제약 없음)
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3)); // [한국어] 레지스터 제한: 스레드당 레지스터는 4의 배수로 올림 (GPU HW 정렬)

  // Limit by CTA
  unsigned int result_cta = max_cta_per_core; // [한국어] SM당 최대 CTA 수 하드 제한

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem); // [한국어] 공유 메모리 제한 적용
  result = gs_min2(result, result_regs);  // [한국어] 레지스터 제한 적용
  result = gs_min2(result, result_cta);   // [한국어] CTA 슬롯 제한 적용

  static const struct gpgpu_ptx_sim_info *last_kinfo = NULL;
  if (last_kinfo !=
      kernel_info) {  // Only print out stats if kernel_info struct changes
    // [한국어] 커널이 바뀔 때만 occupancy 제한 이유 출력 (중복 출력 방지)
    last_kinfo = kernel_info;
    printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
    if (result == result_thread) printf(" threads"); // [한국어] 스레드 수 병목
    if (result == result_shmem) printf(" shmem");   // [한국어] 공유 메모리 병목
    if (result == result_regs) printf(" regs");     // [한국어] 레지스터 병목
    if (result == result_cta) printf(" cta_limit"); // [한국어] CTA 슬롯 병목
    printf("\n");
  }

  // gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep
  // all cores busy
  // [한국어] 전체 CTA 수가 SM 수 × result보다 작으면 균등 분배를 위해 줄임
  if (k.num_blocks() < result * num_shader()) {
    result = k.num_blocks() / num_shader();
    if (k.num_blocks() % num_shader()) result++; // [한국어] 나머지가 있으면 올림 (일부 SM에 1개 더 할당)
  }

  assert(result <= MAX_CTA_PER_SHADER); // [한국어] 시뮬레이터 최대 CTA 상수 초과 방지
  if (result < 1) {
    // [한국어] 자원이 부족하여 CTA를 하나도 실행할 수 없음: 오류 출력
    printf(
        "GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader "
        "has.\n");
    if (gpgpu_ignore_resources_limitation) {
      // [한국어] gpgpu_ignore_resources_limitation 설정 시 강제로 1 CTA 반환 (연구 목적)
      printf(
          "GPGPU-Sim uArch: gpgpu_ignore_resources_limitation is set, ignore "
          "the ERROR!\n");
      return 1;
    }
    abort();
  }

  if (adaptive_cache_config && !k.cache_config_set) {
    // For more info about adaptive cache, see
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
    // [한국어] Volta/Ampere adaptive cache config: 공유 메모리 사용량에 따라 L1D 연관도 동적 재설정
    unsigned total_shmem = kernel_info->smem * result; // [한국어] 이 SM의 총 공유 메모리 사용량
    assert(total_shmem >= 0 && total_shmem <= shmem_opt_list.back());

    // Unified cache config is in KB. Converting to B
    unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024; // [한국어] 통합 캐시 크기 (L1D + shmem 합계)

    bool l1d_configured = false;
    unsigned max_assoc = m_L1D_config.get_max_assoc(); // [한국어] L1D의 최대 연관도

    for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
         it < shmem_opt_list.end(); it++) {
      if (total_shmem <= *it) {
        // [한국어] shmem_opt_list에서 총 공유 메모리를 수용 가능한 가장 작은 옵션 선택
        float l1_ratio = 1 - ((float)*(it) / total_unified); // [한국어] L1D에 할당할 비율
        // make sure the ratio is between 0 and 1
        assert(0 <= l1_ratio && l1_ratio <= 1);
        // round to nearest instead of round down
        m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f); // [한국어] L1D 연관도 재설정 (반올림)
        l1d_configured = true;
        break;
      }
    }

    assert(l1d_configured && "no shared memory option found");

    if (m_L1D_config.is_streaming()) {
      // for streaming cache, if the whole memory is allocated
      // to the L1 cache, then make the allocation to be on_MISS
      // otherwise, make it ON_FILL to eliminate line allocation fails
      // i.e. MSHR throughput is the same, independent on the L1 cache
      // size/associativity
      // [한국어] streaming 캐시 정책: shmem이 0이면 ON_MISS(L1 full 활용), 아니면 ON_FILL
      if (total_shmem == 0) {
        m_L1D_config.set_allocation_policy(ON_MISS);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
      } else {
        m_L1D_config.set_allocation_policy(ON_FILL);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
      }
    }
    printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
           m_L1D_config.get_total_size_inKB()); // [한국어] 재설정된 L1D 크기 출력

    k.cache_config_set = true; // [한국어] 이 커널의 캐시 설정 완료 표시 (중복 설정 방지)
  }

  return result; // [한국어] 4가지 제약 중 최솟값: 이 SM에서 동시 실행 가능한 CTA 수
}

/*
 * [한국어]
 * shader_core_config::set_pipeline_latency — 파이프라인 최대 레이턴시 계산 및 설정
 *
 * @return: void
 *
 * gpgpusim.config의 opcode_latency_* 문자열을 파싱하여 연산 타입별 레이턴시를 설정:
 *   - int_latency[6]: INT 연산 [ADD/SUB, MAX/MIN, MUL, MAD, DIV, SHFL]
 *   - fp_latency[5]: FP(SP) 연산 [ADD/SUB, MAX/MIN, MUL, MAD, DIV]
 *   - dp_latency[5]: DP 연산 [ADD/SUB, MAX/MIN, MUL, MAD, DIV]
 *   - sfu_latency: SFU(특수함수) 연산
 *   - tensor_latency: 텐서 코어 연산
 *
 * MAX 레이턴시는 result_bus 할당(occupied bitset)과 파이프라인 깊이 결정에 사용.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 함수]
 */
void shader_core_config::set_pipeline_latency() {
  // calculate the max latency  based on the input

  unsigned int_latency[6]; // [한국어] INT 연산 레이턴시: [ADD/SUB, MAX/MIN, MUL, MAD, DIV, SHFL]
  unsigned fp_latency[5];  // [한국어] FP(단정밀도) 연산 레이턴시: [ADD/SUB, MAX/MIN, MUL, MAD, DIV]
  unsigned dp_latency[5];  // [한국어] DP(배정밀도) 연산 레이턴시
  unsigned sfu_latency;    // [한국어] SFU(sqrt, log, sin, exp 등) 레이턴시
  unsigned tensor_latency; // [한국어] 텐서 코어(HMMA) 레이턴시

  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]); // [한국어] INT 레이턴시 문자열 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]); // [한국어] FP 레이턴시 문자열 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]); // [한국어] DP 레이턴시 문자열 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);       // [한국어] SFU 레이턴시 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency); // [한국어] 텐서 코어 레이턴시 파싱

  // all div operation are executed on sfu
  // assume that the max latency are dp div or normal sfu_latency
  max_sfu_latency = std::max(dp_latency[4], sfu_latency); // [한국어] SFU 최대 레이턴시: DP div와 일반 SFU 중 최대값 (DIV도 SFU에서 실행)
  // assume that the max operation has the max latency
  max_sp_latency = fp_latency[1];                                // [한국어] SP 최대 레이턴시: FP MAX/MIN 레이턴시 (가장 긴 FP 연산 대표)
  max_int_latency = std::max(int_latency[1], int_latency[5]);   // [한국어] INT 최대 레이턴시: MAX/MIN과 SHFL 중 최대값
  max_dp_latency = dp_latency[1];                                // [한국어] DP 최대 레이턴시: DP MAX/MIN 레이턴시
  max_tensor_core_latency = tensor_latency;                      // [한국어] 텐서 코어 레이턴시
}

/*
 * [한국어]
 * shader_core_ctx::cycle — SM(Streaming Multiprocessor) 한 사이클 처리 (THE 핵심 함수)
 *
 * @return: void
 *
 * 이 함수는 GPGPU-Sim에서 가장 중요한 함수이며, GPU 사이클 시뮬레이션의 핵심이다.
 * gpgpu_sim::cycle()이 매 사이클마다 각 SM에 대해 이 함수를 호출한다.
 *
 * SM 파이프라인 스테이지를 역순(Writeback → Front)으로 처리하여
 * 구조적 해저드(structural hazard)와 데이터 해저드(RAW)를 올바르게 모델링:
 *
 *   1. writeback()     — EX_WB 파이프라인 레지스터에서 완료된 명령어를 처리
 *                        스코어보드 해제, 레지스터 파일 업데이트
 *   2. execute()       — 모든 FU(실행 유닛)의 한 사이클 실행:
 *                        pipelined_simd_unit::cycle(), ldst_unit::cycle() 포함
 *   3. read_operands() — 오퍼랜드 콜렉터 한 사이클: 레지스터 파일 읽기, 콜렉터 채우기
 *   4. issue()         — 각 warp 스케줄러(schedulers[])가 cycle()을 호출하여
 *                        I-buffer에서 준비된 명령어를 FU로 발행
 *   5. decode()+fetch() — inst_fetch_throughput번 반복:
 *                        decode(): I-cache 응답을 I-buffer로 디코딩
 *                        fetch(): 새 명령어 캐시 요청 발행
 *
 * 파이프라인 단계 순서 (역순 처리로 구현):
 *   FETCH → DECODE → ISSUE → READ_OPERANDS → EXECUTE → WRITEBACK
 *   (시뮬레이터는 WB→EX→OC→IS→DC→FC 순으로 처리 — 버블이 앞으로 전파되게)
 *
 * 실행 컨텍스트: 단일 시뮬레이터 스레드 (또는 멀티코어 시뮬레이션에서는 SM별 스레드)
 *
 * 호출 체인: gpgpu_sim::cycle() → simt_core_cluster::core_cycle() → [이 함수]
 */
void shader_core_ctx::cycle() {
  if (!isactive() && get_not_completed() == 0) return; // [한국어] SM이 비활성이고 완료되지 않은 스레드도 없으면 조기 종료

  m_stats->shader_cycles[m_sid]++; // [한국어] 이 SM의 사이클 카운터 증가 (통계용)
  writeback();     // [한국어] 1단계: EX_WB → 레지스터 파일 (완료된 명령어 처리, 스코어보드 해제)
  execute();       // [한국어] 2단계: 모든 FU 사이클 (파이프라인 전진, ldst 응답 처리)
  read_operands(); // [한국어] 3단계: 오퍼랜드 콜렉터 사이클 (레지스터 파일 읽기, 콜렉터 채우기)
  issue();         // [한국어] 4단계: warp 스케줄러 사이클 (I-buffer → FU 발행)
  for (unsigned int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    // [한국어] 5단계: inst_fetch_throughput번 decode + fetch 반복 (슈퍼스칼라 fetch 모델링)
    decode(); // [한국어] I-cache 응답을 I-buffer로 디코딩
    fetch();  // [한국어] 새 명령어 캐시 요청 발행 (warp PC → L1I 접근)
  }
}

// Flushes all content of the cache to memory

/*
 * [한국어]
 * shader_core_ctx::cache_flush — SM의 L1 캐시 전체 flush (writeback)
 *
 * @return: void
 *
 * 커널 종료 또는 cudaDeviceSynchronize() 시 L1 캐시의 dirty 라인을 L2로 flush.
 * ldst_unit::flush()를 호출하여 L1D, L1T, L1C를 모두 플러시.
 *
 * 호출 체인: gpgpu_sim::cycle() 커널 완료 경로 → [이 함수]
 */
void shader_core_ctx::cache_flush() { m_ldst_unit->flush(); }

/*
 * [한국어]
 * shader_core_ctx::cache_invalidate — SM의 L1 캐시 전체 무효화
 *
 * @return: void
 *
 * 캐시 일관성을 위해 L1 캐시의 모든 라인을 무효화 (dirty 라인 포함).
 * ldst_unit::invalidate()를 호출.
 *
 * 호출 체인: 캐시 일관성 요청 경로
 */
void shader_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }

// modifiers
/*
 * [한국어]
 * opndcoll_rfu_t::arbiter_t::allocate_reads — wavefront 알고리즘으로 레지스터 뱅크 읽기 중재
 *
 * @return: 이번 사이클에 읽기가 허가된 op_t 리스트 (뱅크별 최대 1개)
 *
 * 레지스터 파일 뱅크 충돌 해결 알고리즘:
 * - 각 뱅크(input)는 하나의 오퍼랜드 콜렉터(output)에 데이터를 공급
 * - 여러 콜렉터가 같은 뱅크를 요청하면 하나만 허가 → 나머지는 대기
 * - 주석: 같은 콜렉터가 여러 뱅크를 같은 사이클에 읽는 것은 허용됨
 *
 * wavefront 알고리즘 (BookSim NoC 시뮬레이터에서 채용):
 *   - 요청 행렬 _request[input][output]: 뱅크 i에서 콜렉터 j로의 요청
 *   - _pri(우선 대각선)에서 시작하여 _square개의 대각선을 순회
 *   - 대각선을 따라 미매칭(inmatch=-1)된 입력을 먼저 매칭
 *   - 쓰기(writeback)는 최우선: _inmatch[i]=0으로 미리 선점
 *   - 매 사이클 _pri를 +1 이동하여 라운드로빈 공정성 보장
 *
 * 이 알고리즘은 O(max_banks²) 복잡도이며, 최악 시 일부 콜렉터가 여러 사이클 대기.
 *
 * 호출 체인: opndcoll_rfu_t::read_operands() → arbiter_t::allocate_reads() → [이 함수]
 */
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() {
  std::list<op_t>
      result;  // a list of registers that (a) are in different register banks,
               // (b) do not go to the same operand collector
  // [한국어] 이번 사이클에 허가된 읽기 목록: (a) 서로 다른 뱅크에서, (b) 다른 콜렉터로

  int input;  // [한국어] 뱅크 인덱스 (레지스터 파일 뱅크)
  int output; // [한국어] 콜렉터 인덱스 (오퍼랜드 콜렉터)
  int _inputs = m_num_banks;     // [한국어] 레지스터 파일 뱅크 수 (=행렬 행)
  int _outputs = m_num_collectors; // [한국어] 오퍼랜드 콜렉터 수 (=행렬 열)
  int _square = (_inputs > _outputs) ? _inputs : _outputs; // [한국어] 정방행렬 크기 (wavefront 순회용)
  assert(_square > 0);
  int _pri = (int)m_last_cu; // [한국어] 우선 대각선 시작점 (라운드로빈 상태)

  // Clear matching
  for (int i = 0; i < _inputs; ++i) _inmatch[i] = -1;   // [한국어] 입력(뱅크) 매칭 초기화 (-1=미매칭)
  for (int j = 0; j < _outputs; ++j) _outmatch[j] = -1; // [한국어] 출력(콜렉터) 매칭 초기화

  for (unsigned i = 0; i < m_num_banks; i++) {
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);
      _request[i][j] = 0; // [한국어] 요청 행렬 초기화
    }
    if (!m_queue[i].empty()) {
      // [한국어] 뱅크 i의 읽기 큐가 비어있지 않으면: 해당 콜렉터로의 요청 등록
      const op_t &op = m_queue[i].front();
      int oc_id = op.get_oc_id(); // [한국어] 이 오퍼랜드를 기다리는 콜렉터 ID
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      _request[i][oc_id] = 1; // [한국어] 요청 행렬에 등록: 뱅크 i → 콜렉터 oc_id
    }
    if (m_allocated_bank[i].is_write()) {
      assert(i < (unsigned)_inputs);
      _inmatch[i] = 0;  // write gets priority
      // [한국어] writeback(쓰기)이 이 뱅크를 사용 중: 미리 매칭하여 읽기 요청 차단
    }
  }

  ///// wavefront allocator from booksim... --->
  // [한국어] wavefront 알고리즘: BookSim NoC 시뮬레이터의 중재 알고리즘 적용

  // Loop through diagonals of request matrix
  // printf("####\n");

  for (int p = 0; p < _square; ++p) {
    // [한국어] _square개의 대각선을 순회 (각 대각선은 하나의 완전 매칭 기회)
    output = (_pri + p) % _outputs; // [한국어] 이번 대각선의 시작 출력(콜렉터) 인덱스

    // Step through the current diagonal
    for (input = 0; input < _inputs; ++input) {
      // [한국어] 현재 대각선을 따라 입력(뱅크)을 순회하며 매칭 시도
      assert(input < _inputs);
      assert(output < _outputs);
      if ((output < _outputs) && (_inmatch[input] == -1) &&
          //( _outmatch[output] == -1 ) &&   //allow OC to read multiple reg
          // banks at the same cycle
          // [한국어] 콜렉터 매칭(_outmatch)은 확인하지 않음: 한 콜렉터가 여러 뱅크를 같은 사이클에 읽는 것 허용
          (_request[input][output] /*.label != -1*/)) {
        // Grant!
        // [한국어] 매칭 성공: 뱅크 input을 콜렉터 output에 할당
        _inmatch[input] = output; // [한국어] 뱅크 input은 콜렉터 output에 할당됨
        _outmatch[output] = input; // [한국어] 콜렉터 output은 뱅크 input에서 읽음
        // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
        // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
        // (m_queue[input].front()).get_wid(),
        // (m_queue[input].front()).get_reg());
      }

      output = (output + 1) % _outputs; // [한국어] 다음 출력(대각선 이동)
    }
  }

  // Round-robin the priority diagonal
  _pri = (_pri + 1) % _outputs; // [한국어] 다음 사이클의 우선 대각선 이동 (라운드로빈 공정성)

  /// <--- end code from booksim

  m_last_cu = _pri; // [한국어] 상태 저장: 다음 사이클에 사용할 우선 콜렉터
  for (unsigned i = 0; i < m_num_banks; i++) {
    if (_inmatch[i] != -1) {
      // [한국어] 이 뱅크가 매칭된 경우: 읽기 허가
      if (!m_allocated_bank[i].is_write()) {
        // [한국어] writeback이 아닌 읽기 요청만 결과에 추가
        unsigned bank = (unsigned)i;
        op_t &op = m_queue[bank].front(); // [한국어] 이 뱅크의 읽기 큐에서 다음 오퍼랜드
        result.push_back(op); // [한국어] 결과 목록에 추가
        m_queue[bank].pop_front(); // [한국어] 큐에서 제거 (이번 사이클에 서비스됨)
      }
    }
  }

  return result; // [한국어] 이번 사이클에 허가된 읽기 오퍼랜드 목록
}

/*
 * [한국어]
 * barrier_set_t::barrier_set_t — __syncthreads 배리어 관리 구조체 생성자
 *
 * @shader:               이 배리어 세트를 소유하는 SM
 * @max_warps_per_core:   SM당 최대 warp 수
 * @max_cta_per_core:     SM당 최대 CTA 수
 * @max_barriers_per_cta: CTA당 최대 배리어 수 (CUDA 13 이상에서 named barriers 지원)
 * @warp_size:            warp 크기 (32)
 *
 * __syncthreads() 구현의 핵심: 같은 CTA 내의 모든 warp가 배리어에 도달할 때까지
 * 각 warp를 blocking 상태로 유지한다.
 *
 * m_warp_active: 현재 SM에서 실행 중인 warp 비트셋
 * m_warp_at_barrier: 배리어에서 대기 중인 warp 비트셋
 * m_bar_id_to_warps: 각 배리어 ID별로 도달한 warp 비트셋
 *
 * 호출 체인: shader_core_ctx 생성자 → [이 생성자]
 */
barrier_set_t::barrier_set_t(shader_core_ctx *shader,
                             unsigned max_warps_per_core,
                             unsigned max_cta_per_core,
                             unsigned max_barriers_per_cta,
                             unsigned warp_size) {
  m_max_warps_per_core = max_warps_per_core; // [한국어] SM당 최대 warp 수
  m_max_cta_per_core = max_cta_per_core;     // [한국어] SM당 최대 CTA 수
  m_max_barriers_per_cta = max_barriers_per_cta; // [한국어] CTA당 최대 배리어 수
  m_warp_size = warp_size;                   // [한국어] warp 크기 (32)
  m_shader = shader;                         // [한국어] 소유 SM 포인터
  if (max_warps_per_core > WARP_PER_CTA_MAX) {
    // [한국어] warp 비트셋(warp_set_t)의 최대 크기를 초과하면 오류
    printf(
        "ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or "
        "warps per cta in gpgpusim.config\n",
        WARP_PER_CTA_MAX, max_warps_per_core);
    exit(1);
  }
  if (max_barriers_per_cta > MAX_BARRIERS_PER_CTA) {
    // [한국어] 배리어 배열 크기 초과 시 오류
    printf(
        "ERROR ** increase MAX_BARRIERS_PER_CTA in abstract_hardware_model.h "
        "from %u to >= %u or barriers per cta in gpgpusim.config\n",
        MAX_BARRIERS_PER_CTA, max_barriers_per_cta);
    exit(1);
  }
  m_warp_active.reset();     // [한국어] 초기에는 모든 warp 비활성
  m_warp_at_barrier.reset(); // [한국어] 초기에는 배리어에 대기 중인 warp 없음
  for (unsigned i = 0; i < max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i].reset(); // [한국어] 각 배리어 ID별 warp 비트셋 초기화
  }
}

/*
 * [한국어]
 * barrier_set_t::allocate_barrier — CTA 할당 시 배리어 슬롯 예약
 *
 * @cta_id: 할당할 CTA ID
 * @warps:  이 CTA에 속하는 warp 비트셋
 * @return: void
 *
 * CTA가 SM에 할당될 때 호출. m_cta_to_warps에 CTA→warp 매핑 등록,
 * m_warp_active에 이 CTA의 warp들을 설정.
 *
 * 호출 체인: shader_core_ctx::set_max_cta() / init_warps() → [이 함수]
 */
// during cta allocation
void barrier_set_t::allocate_barrier(unsigned cta_id, warp_set_t warps) {
  assert(cta_id < m_max_cta_per_core); // [한국어] 유효한 CTA ID 확인
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  assert(w == m_cta_to_warps.end());  // cta should not already be active or
                                      // allocated barrier resources
  // [한국어] 이미 활성화된 CTA가 중복 할당되지 않도록 확인
  m_cta_to_warps[cta_id] = warps; // [한국어] CTA ID → warp 비트셋 매핑 등록
  assert(m_cta_to_warps.size() <=
         m_max_cta_per_core);  // catch cta's that were not properly deallocated
  // [한국어] 제대로 deallocate되지 않은 CTA 감지

  m_warp_active |= warps;         // [한국어] 이 CTA의 warp들을 활성 상태로 설정
  m_warp_at_barrier &= ~warps;    // [한국어] 배리어 대기 상태 초기화 (새 CTA는 배리어에서 시작 안 함)
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i] &= ~warps; // [한국어] 각 배리어 ID의 warp 비트셋에서 이 CTA의 warp 제거
  }
}

/*
 * [한국어]
 * barrier_set_t::deallocate_barrier — CTA 완료 시 배리어 슬롯 해제
 *
 * @cta_id: 해제할 CTA ID
 * @return: void
 *
 * CTA가 완료되어 SM에서 제거될 때 호출.
 * 아직 배리어에 대기 중인 warp나 활성 warp가 있으면 assert 실패.
 *
 * 호출 체인: register_cta_thread_exit() → [이 함수]
 */
// during cta deallocation
void barrier_set_t::deallocate_barrier(unsigned cta_id) {
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  if (w == m_cta_to_warps.end()) return; // [한국어] 이미 deallocated되었거나 등록되지 않은 CTA
  warp_set_t warps = w->second;
  warp_set_t at_barrier = warps & m_warp_at_barrier;
  assert(at_barrier.any() == false);  // no warps stuck at barrier
  // [한국어] 아직 배리어에 갇힌 warp가 있으면 CTA가 올바르게 완료되지 않음
  warp_set_t active = warps & m_warp_active;
  assert(active.any() == false);  // no warps in CTA still running
  // [한국어] 아직 실행 중인 warp가 있으면 CTA가 올바르게 완료되지 않음
  m_warp_active &= ~warps;     // [한국어] 이 CTA의 warp들을 활성 비트셋에서 제거
  m_warp_at_barrier &= ~warps; // [한국어] 배리어 대기 비트셋에서도 제거

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps & m_bar_id_to_warps[i];
    assert(at_a_specific_barrier.any() == false);  // no warps stuck at barrier
    m_bar_id_to_warps[i] &= ~warps; // [한국어] 각 배리어 ID의 warp 비트셋에서 제거
  }
  m_cta_to_warps.erase(w); // [한국어] CTA→warp 매핑 삭제
}

/*
 * [한국어]
 * barrier_set_t::warp_reaches_barrier — warp가 __syncthreads 배리어에 도달
 *
 * @cta_id:  도달한 warp의 CTA ID
 * @warp_id: 배리어에 도달한 warp ID
 * @inst:    배리어 명령어 (bar_type, bar_id, bar_count 포함)
 * @return:  void
 *
 * warp가 BARRIER_OP 명령어를 실행할 때 호출. 다음을 처리:
 *   1. 이 warp를 bar_id 배리어에 등록 (m_bar_id_to_warps[bar_id].set(warp_id))
 *   2. SYNC/RED 타입이면 m_warp_at_barrier에도 설정 → 스케줄러가 이 warp를 blocking
 *   3. bar_count==-1: 모든 활성 warp가 도달하면 배리어 해제
 *   4. bar_count!=-1: 지정한 스레드 수가 도달하면 조기 해제 (partial barrier)
 *   5. RED 타입이면 broadcast_barrier_reduction() 호출
 *
 * 호출 체인: scheduler_unit::cycle() → 배리어 처리 → [이 함수]
 */
// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                                         warp_inst_t *inst) {
  barrier_type bar_type = inst->bar_type; // [한국어] 배리어 타입: SYNC(__syncthreads), RED(reduction), ARRIVE(비차단)
  unsigned bar_id = inst->bar_id;         // [한국어] 배리어 ID (named barriers: 0~MAX_BARRIERS_PER_CTA-1)
  unsigned bar_count = inst->bar_count;   // [한국어] 배리어 해제에 필요한 스레드 수 (-1=모든 활성 warp)
  assert(bar_id != (unsigned)-1);         // [한국어] 유효한 bar_id 확인
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);

  if (w == m_cta_to_warps.end()) {  // cta is active
    // [한국어] 오류: 이 CTA가 배리어 세트에 등록되지 않음
    printf(
        "ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n",
        cta_id, m_shader->get_gpu()->gpu_tot_sim_cycle,
        m_shader->get_gpu()->gpu_sim_cycle);
    dump();
    abort();
  }
  assert(w->second.test(warp_id) == true);  // warp is in cta
  // [한국어] 이 warp가 해당 CTA에 속하는지 확인

  m_bar_id_to_warps[bar_id].set(warp_id); // [한국어] 이 배리어에 도달한 warp 등록
  if (bar_type == SYNC || bar_type == RED) {
    m_warp_at_barrier.set(warp_id); // [한국어] SYNC/RED: 이 warp를 배리어 대기 상태로 설정 (스케줄러가 skip)
  }
  warp_set_t warps_in_cta = w->second; // [한국어] 이 CTA에 속하는 모든 warp 비트셋
  warp_set_t at_barrier = warps_in_cta & m_bar_id_to_warps[bar_id]; // [한국어] 이 배리어에 도달한 CTA 내 warp
  warp_set_t active = warps_in_cta & m_warp_active; // [한국어] 현재 활성화된 CTA 내 warp
  if (bar_count == (unsigned)-1) {
    // [한국어] 일반 __syncthreads: 모든 활성 warp가 도달해야 해제
    if (at_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      // [한국어] 모든 활성 warp가 배리어에 도달: 배리어 해제
      m_bar_id_to_warps[bar_id] &= ~at_barrier; // [한국어] 이 배리어의 warp 비트셋 초기화
      m_warp_at_barrier &= ~at_barrier;          // [한국어] 배리어 대기 상태 해제
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier); // [한국어] reduction barrier: 결과 브로드캐스트
      }
    }
  } else {
    // TODO: check on the hardware if the count should include warp that exited
    // [한국어] partial barrier: bar_count 스레드가 도달하면 조기 해제
    if ((at_barrier.count() * m_warp_size) == bar_count) {
      // required number of warps have reached barrier, so release waiting
      // warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier);
      }
    }
  }
}

/*
 * [한국어]
 * barrier_set_t::warp_exit — warp 종료 시 배리어 상태 업데이트
 *
 * @warp_id: 종료되는 warp ID
 * @return:  void
 *
 * warp가 종료될 때 m_warp_active에서 제거 후, 이 warp의 종료로 인해
 * 배리어 해제 조건(at_barrier == active)이 충족되면 배리어를 해제.
 * 이를 통해 마지막 warp가 일찍 종료되더라도 나머지 warp가 영원히 블로킹되지 않음.
 *
 * 호출 체인: shader_core_ctx::warp_exit() → barrier_set_t::warp_exit()
 */
// warp reaches exit
void barrier_set_t::warp_exit(unsigned warp_id) {
  // caller needs to verify all threads in warp are done, e.g., by checking PDOM
  // stack to see it has only one entry during exit_impl()
  // [한국어] 호출자는 이 warp의 모든 스레드가 완료되었음을 보장해야 함
  m_warp_active.reset(warp_id); // [한국어] 이 warp를 활성 비트셋에서 제거

  // test for barrier release
  // [한국어] 이 warp가 속한 CTA 찾기
  cta_to_warp_t::iterator w = m_cta_to_warps.begin();
  for (; w != m_cta_to_warps.end(); ++w) {
    if (w->second.test(warp_id) == true) break; // [한국어] 이 warp가 속한 CTA 발견
  }
  warp_set_t warps_in_cta = w->second; // [한국어] 이 CTA의 모든 warp
  warp_set_t active = warps_in_cta & m_warp_active; // [한국어] 이 CTA에서 아직 활성화된 warp

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps_in_cta & m_bar_id_to_warps[i]; // [한국어] 배리어 i에 도달한 CTA 내 warp
    if (at_a_specific_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      // [한국어] 남은 활성 warp가 모두 배리어에 있으면 해제 (종료된 warp 제외)
      m_bar_id_to_warps[i] &= ~at_a_specific_barrier; // [한국어] 배리어 warp 비트셋 초기화
      m_warp_at_barrier &= ~at_a_specific_barrier;     // [한국어] 배리어 대기 상태 해제
    }
  }
}

/*
 * [한국어]
 * barrier_set_t::warp_waiting_at_barrier — warp가 배리어 대기 중인지 확인
 *
 * @warp_id: 확인할 warp ID
 * @return: true이면 이 warp가 __syncthreads에서 대기 중
 *
 * 호출 체인: scheduler_unit::cycle() → shd_warp_t::waiting() → [이 함수]
 */
// assertions
bool barrier_set_t::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_warp_at_barrier.test(warp_id); // [한국어] 배리어 대기 비트셋에서 이 warp 비트 확인
}

/*
 * [한국어]
 * barrier_set_t::dump — 배리어 세트 상태 출력 (디버그용)
 *
 * @return: void
 *
 * m_cta_to_warps, m_warp_active, m_warp_at_barrier, m_bar_id_to_warps를 출력.
 *
 * 호출 체인: warp_reaches_barrier() 오류 경로 → [이 함수]
 */
void barrier_set_t::dump() {
  printf("barrier set information\n");
  printf("  m_max_cta_per_core = %u\n", m_max_cta_per_core);
  printf("  m_max_warps_per_core = %u\n", m_max_warps_per_core);
  printf(" m_max_barriers_per_cta =%u\n", m_max_barriers_per_cta);
  printf("  cta_to_warps:\n");

  cta_to_warp_t::const_iterator i;
  for (i = m_cta_to_warps.begin(); i != m_cta_to_warps.end(); i++) {
    unsigned cta_id = i->first;
    warp_set_t warps = i->second;
    printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str()); // [한국어] CTA ID와 warp 비트셋 출력
  }
  printf("  warp_active: %s\n", m_warp_active.to_string().c_str()); // [한국어] 활성 warp 비트셋
  printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str()); // [한국어] 배리어 대기 warp 비트셋
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t warps_reached_barrier = m_bar_id_to_warps[i];
    printf("  warp_at_barrier %u: %s\n", i,
           warps_reached_barrier.to_string().c_str()); // [한국어] 각 배리어 ID별 도달 warp 비트셋
  }
  fflush(stdout);
}

/*
 * [한국어]
 * shader_core_ctx::warp_exit — warp 종료 처리
 *
 * @warp_id: 종료되는 warp ID
 * @return:  void
 *
 * warp의 모든 스레드가 완료되었는지 확인 후 barrier_set_t::warp_exit()를 호출.
 * 배리어 상태 업데이트와 CTA 종료 조건 확인이 연쇄적으로 이루어짐.
 *
 * 호출 체인: functional_done(), register_cta_thread_exit() 경로
 */
void shader_core_ctx::warp_exit(unsigned warp_id) {
  bool done = true;
  for (unsigned i = warp_id * get_config()->warp_size;
       i < (warp_id + 1) * get_config()->warp_size; i++) {
    //		if(this->m_thread[i]->m_functional_model_thread_state &&
    // this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
    // done = false;
    //		}

    if (m_thread[i] && !m_thread[i]->is_done()) done = false; // [한국어] 이 warp의 스레드 중 아직 실행 중인 것이 있으면 done=false
  }
  // if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
  // if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
  if (done) m_barriers.warp_exit(warp_id); // [한국어] 모든 스레드 완료 시 배리어 세트에 종료 통보
}

/*
 * [한국어]
 * shader_core_ctx::check_if_non_released_reduction_barrier — 비해제 reduction 배리어 검사
 *
 * @inst:   검사할 warp 명령어
 * @return: true이면 이 warp의 파이프라인에 미해제 RED 배리어가 있음
 *
 * warp 발행 시 특수 케이스 감지: RED 배리어가 파이프라인에 있으면서 아직 해제되지 않은 경우.
 * 이 경우 warp는 발행을 중단해야 함.
 *
 * 호출 체인: scheduler_unit::cycle() → [이 함수]
 */
bool shader_core_ctx::check_if_non_released_reduction_barrier(
    warp_inst_t &inst) {
  unsigned warp_id = inst.warp_id();
  bool bar_red_op = (inst.op == BARRIER_OP) && (inst.bar_type == RED); // [한국어] RED(reduction) 배리어 명령어인지 확인
  bool non_released_barrier_reduction = false;
  bool warp_stucked_at_barrier = warp_waiting_at_barrier(warp_id); // [한국어] 이 warp가 배리어에 대기 중인지
  bool single_inst_in_pipeline =
      (m_warp[warp_id]->num_issued_inst_in_pipeline() == 1); // [한국어] 파이프라인에 명령어가 정확히 1개(이 배리어)인지
  non_released_barrier_reduction =
      single_inst_in_pipeline and warp_stucked_at_barrier and bar_red_op; // [한국어] 세 조건 모두 만족: 비해제 RED 배리어
  printf("non_released_barrier_reduction=%u\n", non_released_barrier_reduction);
  return non_released_barrier_reduction;
}

/*
 * [한국어]
 * shader_core_ctx::warp_waiting_at_barrier — warp가 __syncthreads 대기 중인지 확인
 *
 * @warp_id: 확인할 warp ID
 * @return:  true이면 대기 중
 *
 * 호출 체인: scheduler_unit::cycle() → shd_warp_t::waiting() → [이 함수]
 */
bool shader_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id); // [한국어] barrier_set_t에 위임
}

/*
 * [한국어]
 * shader_core_ctx::warp_waiting_at_mem_barrier — warp가 메모리 배리어 대기 중인지 확인
 *
 * @warp_id: 확인할 warp ID
 * @return:  true이면 메모리 fence 대기 중
 *
 * membar 플래그가 설정된 warp는 모든 pending 쓰기가 완료될 때까지 발행 불가.
 * pendingWrites가 없으면 membar 해제하고 선택적으로 L1 캐시 무효화.
 *
 * 호출 체인: scheduler_unit::cycle() → shd_warp_t::waiting() → [이 함수]
 */
bool shader_core_ctx::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_warp[warp_id]->get_membar()) return false; // [한국어] membar 플래그가 없으면 즉시 반환
  if (!m_scoreboard->pendingWrites(warp_id)) {
    // [한국어] 모든 pending 쓰기가 완료: membar 해제
    m_warp[warp_id]->clear_membar(); // [한국어] membar 플래그 클리어
    if (m_gpu->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
      cache_invalidate(); // [한국어] L1 캐시 무효화: 메모리 일관성 보장 (NVIDIA 문서 요구사항)
      // TO DO: you need to stall the SM for 5k cycles.
    }
    return false; // [한국어] 메모리 배리어 해제됨: 발행 가능
  }
  return true; // [한국어] 아직 pending 쓰기가 있음: 발행 불가
}

/*
 * [한국어]
 * shader_core_ctx::set_max_cta — 커널 실행 전 SM의 최대 CTA 및 로컬 메모리 레이아웃 설정
 *
 * @kernel: 실행할 커널 정보
 * @return: void
 *
 * 로컬 메모리 주소 매핑을 위해 kernel_max_cta_per_shader와
 * kernel_padded_threads_per_cta를 계산하여 저장.
 *
 * 호출 체인: gpgpu_sim::issue_block2core() → [이 함수]
 */
void shader_core_ctx::set_max_cta(const kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  kernel_max_cta_per_shader = m_config->max_cta(kernel); // [한국어] 이 SM에서 동시 실행 가능한 최대 CTA 수
  unsigned int gpu_cta_size = kernel.threads_per_cta();  // [한국어] CTA당 스레드 수
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size; // [한국어] warp_size 배수로 올림: 로컬 메모리 주소 계산의 정렬 기준
}

/*
 * [한국어]
 * shader_core_ctx::decrement_atomic_count — 원자적 연산 완료 카운터 감소
 *
 * @wid: warp ID
 * @n:   완료된 원자적 연산 수
 * @return: void
 *
 * 원자적 연산(atomicAdd 등)이 완료될 때 호출하여 pending 카운터를 감소.
 * 카운터가 0이 되면 이 warp는 원자적 연산 대기에서 해제됨.
 *
 * 호출 체인: ldst_unit::writeback() / ldst_unit::cycle() → [이 함수]
 */
void shader_core_ctx::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_warp[wid]->get_n_atomic() >= n); // [한국어] 음수가 되지 않도록 검증
  m_warp[wid]->dec_n_atomic(n); // [한국어] pending 원자적 연산 카운터 감소
}

/*
 * [한국어]
 * shader_core_ctx::broadcast_barrier_reduction — RED 배리어 완료 시 결과 브로드캐스트
 *
 * @cta_id: 배리어가 완료된 CTA ID
 * @bar_id: 배리어 ID
 * @warps:  이 배리어에 참여한 warp 비트셋
 * @return: void
 *
 * RED(reduction) 배리어가 완료되면 각 warp에 저장된 배리어 명령어에
 * broadcast_barrier_reduction()을 호출하여 reduction 결과를 전파.
 *
 * 호출 체인: barrier_set_t::warp_reaches_barrier() / warp_exit() → [이 함수]
 */
void shader_core_ctx::broadcast_barrier_reduction(unsigned cta_id,
                                                  unsigned bar_id,
                                                  warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      // [한국어] 이 배리어에 참여한 warp에 대해 reduction 결과 브로드캐스트
      const warp_inst_t *inst =
          m_warp[i]->restore_info_of_last_inst_at_barrier(); // [한국어] 배리어에서 대기 중이던 마지막 명령어 복구
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask()); // [한국어] 활성 마스크를 사용하여 reduction 결과 브로드캐스트
    }
  }
}

/*
 * [한국어]
 * shader_core_ctx::fetch_unit_response_buffer_full — I-cache 응답 버퍼 포화 여부
 *
 * @return: false (항상 공간 있음 — L1I는 내부 FIFO 사용)
 *
 * NoC의 백프레셔 제어를 위한 인터페이스.
 * 현재 구현은 항상 false를 반환 (I-cache는 항상 응답을 받을 수 있음).
 */
bool shader_core_ctx::fetch_unit_response_buffer_full() const { return false; }

/*
 * [한국어]
 * shader_core_ctx::accept_fetch_response — I-cache miss 응답 수신 (NoC → L1I)
 *
 * @mf: L2에서 도착한 I-cache miss 응답 mem_fetch
 * @return: void
 *
 * NoC를 통해 L2에서 도착한 명령어 캐시 응답을 L1I에 fill.
 *
 * 호출 체인: simt_core_cluster::icnt_cycle() → [이 함수]
 */
void shader_core_ctx::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle); // [한국어] 상태를 IN_SHADER_FETCHED로 설정 (타이밍 추적)
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle); // [한국어] L1I에 캐시 라인 fill
}

/*
 * [한국어]
 * shader_core_ctx::ldst_unit_response_buffer_full — ldst 응답 버퍼 포화 여부
 *
 * @return: true이면 ldst_unit의 응답 FIFO가 가득 참
 *
 * NoC 백프레셔 제어: 가득 찬 경우 NoC가 이 SM으로 추가 응답을 보내지 않음.
 */
bool shader_core_ctx::ldst_unit_response_buffer_full() const {
  return m_ldst_unit->response_buffer_full(); // [한국어] ldst_unit의 m_response_fifo 포화 여부 확인
}

/*
 * [한국어]
 * shader_core_ctx::accept_ldst_unit_response — 메모리 응답 수신 (NoC → ldst_unit)
 *
 * @mf: L2/DRAM에서 도착한 데이터 응답 mem_fetch
 * @return: void
 *
 * NoC를 통해 도착한 메모리 데이터 응답을 ldst_unit의 response_fifo에 추가.
 *
 * 호출 체인: simt_core_cluster::icnt_cycle() → [이 함수]
 */
void shader_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit->fill(mf); // [한국어] ldst_unit의 m_response_fifo에 mem_fetch 추가
}

/*
 * [한국어]
 * shader_core_ctx::store_ack — 쓰기 완료 ACK 처리
 *
 * @mf: WRITE_ACK 타입의 mem_fetch (L2에서 store 완료 통보)
 * @return: void
 *
 * 글로벌 메모리 store가 L2에서 완료됐을 때 호출.
 * warp의 store_req 카운터를 감소시켜 membar 대기 해제 가능.
 *
 * 호출 체인: ldst_unit::cycle() WRITE_ACK 경로 → [이 함수]
 */
void shader_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         ((m_config->gpgpu_perfect_mem || m_memory_config->SST_mode) &&
          mf->get_is_write())); // [한국어] WRITE_ACK 또는 perfect_mem/SST 쓰기임을 확인
  unsigned warp_id = mf->get_wid();
  m_warp[warp_id]->dec_store_req(); // [한국어] 이 warp의 pending store 카운터 감소 (membar 해제 조건)
}

/*
 * [한국어]
 * shader_core_ctx::print_cache_stats — L1 캐시 통계 출력
 *
 * 호출 체인: simt_core_cluster::print_cache_stats() → [이 함수]
 */
void shader_core_ctx::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                        unsigned &dl1_misses) {
  m_ldst_unit->print_cache_stats(fp, dl1_accesses, dl1_misses); // [한국어] ldst_unit의 캐시 통계 출력 위임
}

/*
 * [한국어]
 * shader_core_ctx::get_cache_stats — 캐시 통계 집계
 *
 * @cs: 통계를 누적할 cache_stats 객체 (in/out)
 *
 * L1I, L1D, L1C, L1T 통계를 모두 cs에 누적.
 */
void shader_core_ctx::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I->get_stats();          // Get L1I stats  // [한국어] L1 명령어 캐시 통계 누적
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats  // [한국어] 데이터/상수/텍스처 캐시 통계 누적
}

/*
 * [한국어]
 * shader_core_ctx::get_L1I_sub_stats — L1I 서브 통계 조회
 */
void shader_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1I) m_L1I->get_sub_stats(css); // [한국어] L1I 캐시의 서브 통계(accesses/misses/pending_hits/res_fails)
}

/*
 * [한국어]
 * shader_core_ctx::get_L1D_sub_stats — L1D 서브 통계 조회
 */
void shader_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css); // [한국어] L1D 서브 통계 조회 ldst_unit에 위임
}
void shader_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1C_sub_stats(css); // [한국어] L1C(상수 캐시) 서브 통계 조회
}
void shader_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1T_sub_stats(css); // [한국어] L1T(텍스처 캐시) 서브 통계 조회
}

/*
 * [한국어]
 * shader_core_ctx::get_icnt_power_stats — NoC 전력 통계 집계
 *
 * @n_simt_to_mem: SM→메모리 방향 전송 수 누적 (out)
 * @n_mem_to_simt: 메모리→SM 방향 전송 수 누적 (out)
 * @return: void
 *
 * AccelWattch 전력 모델에서 NoC 전력 계산에 사용.
 *
 * 호출 체인: gpgpu_sim::get_icnt_power_stats() → [이 함수]
 */
void shader_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
                                           long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sid]; // [한국어] 이 SM에서 메모리로 보낸 전송 수 누적
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sid]; // [한국어] 메모리에서 이 SM으로 받은 전송 수 누적
}

/*
 * [한국어]
 * shd_warp_t::get_kernel_info — 이 warp가 실행 중인 커널 정보 반환
 *
 * @return: kernel_info_t 포인터 (SM→커널 정보)
 *
 * 커널 완료 판단, 로컬 메모리 주소 계산 등에 사용.
 */
kernel_info_t *shd_warp_t::get_kernel_info() const {
  return m_shader->get_kernel_info(); // [한국어] SM의 커널 정보를 반환
}

/*
 * [한국어]
 * shd_warp_t::functional_done — warp의 기능적 실행 완료 여부
 *
 * @return: true이면 이 warp의 모든 스레드가 완료(n_completed == warp_size)
 *
 * 기능 시뮬레이션(PTX 실행) 완료를 나타냄. 타이밍 모델(hardware_done)과 별개.
 */
bool shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size; // [한국어] 완료된 스레드 수가 warp 크기와 같으면 기능적으로 완료
}

/*
 * [한국어]
 * shd_warp_t::hardware_done — warp의 하드웨어 완료 여부
 *
 * @return: true이면 기능적 완료 + store 완료 + 파이프라인 비어있음
 *
 * 이 warp가 완전히 완료되어 SM 자원을 반환할 준비가 되었음을 의미.
 */
bool shd_warp_t::hardware_done() const {
  return functional_done() && stores_done() && !inst_in_pipeline(); // [한국어] 기능 완료 + 모든 store ACK 수신 + 파이프라인 명령어 없음
}

/*
 * [한국어]
 * shd_warp_t::waiting — warp가 발행 불가 상태인지 확인
 *
 * @return: true이면 이 warp는 스케줄러에서 건너뜀
 *
 * 다음 중 하나라도 해당되면 waiting:
 *   1. functional_done: 이미 모든 스레드 완료 (새 커널 대기 중)
 *   2. warp_waiting_at_barrier: __syncthreads에서 다른 warp 대기 중
 *   3. warp_waiting_at_mem_barrier: 메모리 fence(membar) 대기 중
 *   4. m_n_atomic > 0: 원자적 연산이 DRAM에서 완료되기를 기다리는 중
 *      (타이밍 모델 정확성보다는 기능 모델의 레지스터 읽기 오류 방지 목적)
 *   5. m_waiting_ldgsts: LDGSTS 비동기 복사 완료 대기 (DEPBAR)
 *
 * 호출 체인: scheduler_unit::cycle() → warp(warp_id).waiting()
 */
bool shd_warp_t::waiting() {
  if (functional_done()) {
    // waiting to be initialized with a kernel
    return true; // [한국어] 모든 스레드 완료: 새 CTA/커널 할당 대기
  } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
    // waiting for other warps in CTA to reach barrier
    return true; // [한국어] __syncthreads 대기: 다른 warp가 배리어에 도달하기를 기다림
  } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
    // waiting for memory barrier
    return true; // [한국어] membar 대기: 모든 pending 쓰기가 완료되어야 함
  } else if (m_n_atomic > 0) {
    // waiting for atomic operation to complete at memory:
    // this stall is not required for accurate timing model, but rather we
    // stall here since if a call/return instruction occurs in the meantime
    // the functional execution of the atomic when it hits DRAM can cause
    // the wrong register to be read.
    // [한국어] 원자적 연산 대기: DRAM에서 결과가 돌아올 때까지 stall
    // (타이밍 정확도보다 기능 모델의 레지스터 읽기 오류 방지가 목적)
    return true;
  } else if (m_waiting_ldgsts) {  // Waiting for LDGSTS to finish
    return true; // [한국어] LDGSTS(비동기 global→shared 복사) 완료 대기 (DEPBAR 명령어로 설정)
  }
  return false; // [한국어] 모든 조건 미해당: 이 warp는 발행 가능
}

/*
 * [한국어]
 * shd_warp_t::print — warp 상태 출력 (디버그용)
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * warp ID, 다음 PC, 완료 플래그(f/s/i/e), 완료 스레드 수, 파이프라인 명령어 수,
 * 미완료 store 수, 원자적 연산 수, 각 스레드 완료 비트맵, 마지막 fetch 사이클을 출력.
 *
 * 호출 체인: display_pipeline() → dump_warp_state() → [이 함수]
 */
void shd_warp_t::print(FILE *fout) const {
  if (!done_exit()) {
    fprintf(fout,
            "w%02u npc: 0x%04llx, done:%c%c%c%c:%2u i:%u s:%u a:%u (done: ",
            m_warp_id, m_next_pc, (functional_done() ? 'f' : ' '),
            (stores_done() ? 's' : ' '), (inst_in_pipeline() ? ' ' : 'i'),
            (done_exit() ? 'e' : ' '), n_completed, m_inst_in_pipeline,
            m_stores_outstanding, m_n_atomic);
    // [한국어] 플래그: f=기능적 완료, s=store 완료, i(공백)=파이프라인 명령어 없음, e=exit 완료
    for (unsigned i = m_warp_id * m_warp_size;
         i < (m_warp_id + 1) * m_warp_size; i++) {
      if (m_shader->ptx_thread_done(i))
        fprintf(fout, "1"); // [한국어] 완료된 스레드
      else
        fprintf(fout, "0"); // [한국어] 아직 실행 중인 스레드
      if ((((i + 1) % 4) == 0) && (i + 1) < (m_warp_id + 1) * m_warp_size)
        fprintf(fout, ","); // [한국어] 4개마다 콤마로 구분
    }
    fprintf(fout, ") ");
    fprintf(fout, " active=%s", m_active_threads.to_string().c_str()); // [한국어] 활성 스레드 비트맵
    fprintf(fout, " last fetched @ %5llu", m_last_fetch); // [한국어] 마지막 fetch 사이클
    if (m_imiss_pending) fprintf(fout, " i-miss pending"); // [한국어] I-cache miss 대기 중
    fprintf(fout, "\n");
  }
}

/*
 * [한국어]
 * shd_warp_t::print_ibuffer — I-buffer(명령어 버퍼) 내용 출력 (디버그용)
 *
 * @fout: 출력 파일 포인터
 * @return: void
 *
 * I-buffer의 각 슬롯에 들어있는 명령어를 출력.
 * 비어있으면 "<empty>", 유효하지 않으면 "<invalid instruction>" 출력.
 *
 * 호출 체인: display_pipeline() → [이 함수]
 */
void shd_warp_t::print_ibuffer(FILE *fout) const {
  fprintf(fout, "  ibuffer[%2u] : ", m_warp_id);
  for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
    const inst_t *inst = m_ibuffer[i].m_inst;
    if (inst)
      inst->print_insn(fout); // [한국어] 유효한 명령어 텍스트 출력
    else if (m_ibuffer[i].m_valid)
      fprintf(fout, " <invalid instruction> "); // [한국어] 유효 비트는 있지만 명령어 포인터가 없는 경우
    else
      fprintf(fout, " <empty> "); // [한국어] 완전히 빈 슬롯
  }
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * opndcoll_rfu_t::add_cu_set — 콜렉터 유닛 세트 추가
 *
 * @set_id:      콜렉터 세트 ID (SP/SFU/MEM 등 FU 타입 구분)
 * @num_cu:      이 세트에 포함할 콜렉터 유닛 수
 * @num_dispatch: 이 세트에 전용 dispatch 유닛 수
 * @return: void
 *
 * 각 FU 타입별로 콜렉터 세트를 생성. reserve()가 필수:
 * vector가 재할당되면 m_cu의 포인터가 무효화되기 때문.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 함수]
 */
void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu,
                                unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  // [한국어] reserve 필수: vector 재할당 시 m_cu의 포인터가 무효화되는 것을 방지
  for (unsigned i = 0; i < num_cu; i++) {
    m_cus[set_id].push_back(collector_unit_t()); // [한국어] 빈 콜렉터 유닛 추가
    m_cu.push_back(&m_cus[set_id].back());       // [한국어] 전체 콜렉터 포인터 배열에 등록
  }
  // for now each collector set gets dedicated dispatch units.
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id])); // [한국어] 이 세트 전용 dispatch 유닛 추가
  }
}

/*
 * [한국어]
 * opndcoll_rfu_t::add_port — 오퍼랜드 콜렉터 입출력 포트 추가
 *
 * @input:   ID_OC 단계의 입력 파이프라인 레지스터 집합
 * @output:  OC_EX 단계의 출력 파이프라인 레지스터 집합
 * @cu_sets: 이 포트가 서비스하는 콜렉터 세트 ID 목록
 * @return:  void
 *
 * 포트는 특정 FU 타입(SP/SFU/MEM 등)의 파이프라인 레지스터와 콜렉터 세트를 연결.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 함수]
 */
void opndcoll_rfu_t::add_port(port_vector_t &input, port_vector_t &output,
                              uint_vector_t cu_sets) {
  // m_num_ports++;
  // ...
  m_in_ports.push_back(input_port_t(input, output, cu_sets)); // [한국어] 입출력 포트 쌍과 연결된 콜렉터 세트를 m_in_ports에 추가
}

/*
 * [한국어]
 * opndcoll_rfu_t::init — 오퍼랜드 콜렉터 초기화
 *
 * @num_banks: 레지스터 파일 뱅크 수
 * @shader:    이 콜렉터가 속한 SM
 * @return:    void
 *
 * 중재자(arbiter) 초기화, sub_core_model 설정, 각 콜렉터 유닛 및 dispatch 유닛 초기화.
 * sub_core_model이면 뱅크를 warp 스케줄러 수로 균등 분할하여 각 콜렉터에 할당.
 *
 * 호출 체인: shader_core_ctx::create_exec_pipeline() → [이 함수]
 */
void opndcoll_rfu_t::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter.init(m_cu.size(), num_banks); // [한국어] wavefront 중재자 초기화: 콜렉터 수×뱅크 수 행렬 준비
  // for( unsigned n=0; n<m_num_ports;n++ )
  //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
  m_num_banks = num_banks;
  m_warp_size = shader->get_config()->warp_size; // [한국어] warp 크기 (레지스터 뱅크 계산에 사용)

  sub_core_model = shader->get_config()->sub_core_model; // [한국어] Volta+ sub_core_model 여부
  m_num_warp_scheds = shader->get_config()->gpgpu_num_sched_per_core; // [한국어] SM당 warp 스케줄러 수
  unsigned reg_id = 0;
  if (sub_core_model) {
    // [한국어] sub_core_model: 뱅크와 콜렉터를 스케줄러 수로 균등 분할 검증
    assert(num_banks % shader->get_config()->gpgpu_num_sched_per_core == 0);
    assert(m_num_warp_scheds <= m_cu.size() &&
           m_cu.size() % m_num_warp_scheds == 0);
  }
  m_num_banks_per_sched =
      num_banks / shader->get_config()->gpgpu_num_sched_per_core; // [한국어] 스케줄러당 레지스터 파일 뱅크 수

  for (unsigned j = 0; j < m_cu.size(); j++) {
    if (sub_core_model) {
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds; // [한국어] 스케줄러당 콜렉터 수
      reg_id = j / cusPerSched; // [한국어] 이 콜렉터가 속하는 스케줄러 ID
    }
    m_cu[j]->init(j, num_banks, shader->get_config(), this, sub_core_model,
                  reg_id, m_num_banks_per_sched); // [한국어] 각 콜렉터 유닛 초기화
  }
  for (unsigned j = 0; j < m_dispatch_units.size(); j++) {
    m_dispatch_units[j].init(sub_core_model, m_num_warp_scheds); // [한국어] dispatch 유닛 초기화
  }
  m_initialized = true; // [한국어] 초기화 완료 플래그
}

/*
 * [한국어]
 * register_bank — 레지스터 번호와 warp ID로 레지스터 파일 뱅크 계산
 *
 * @regnum:         레지스터 번호 (0~255)
 * @wid:            warp ID
 * @num_banks:      전체 레지스터 파일 뱅크 수
 * @sub_core_model: Volta+ sub_core_model 여부
 * @banks_per_sched: 스케줄러당 뱅크 수 (sub_core_model에서만 사용)
 * @sched_id:       이 명령어를 발행한 warp 스케줄러 ID
 * @return:         레지스터 파일 뱅크 번호
 *
 * bank = (regnum + wid) % num_banks: warp와 레지스터 번호를 더해 뱅크를 분산.
 * sub_core_model에서는 스케줄러별로 뱅크를 파티셔닝하여 충돌 최소화.
 *
 * 호출 체인: opndcoll_rfu_t::allocate_reads() / writeback() → [이 함수]
 */
unsigned register_bank(int regnum, int wid, unsigned num_banks,
                       bool sub_core_model, unsigned banks_per_sched,
                       unsigned sched_id) {
  int bank = regnum;
  bank += wid; // [한국어] regnum + wid 합산으로 뱅크 분산 (연속 warp의 같은 레지스터가 다른 뱅크에 매핑)
  if (sub_core_model) {
    // [한국어] sub_core_model: 스케줄러별로 뱅크 파티셔닝
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched); // [한국어] 스케줄러 ID의 파티션 내에서 뱅크 선택
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks; // [한국어] 일반 모드: 전체 뱅크에 균등 분산
}

/*
 * [한국어]
 * opndcoll_rfu_t::writeback — 실행 완료 결과를 레지스터 파일 뱅크에 쓰기 예약
 *
 * @inst: 완료된 warp 명령어 (목적 레지스터 정보 포함)
 * @return: true이면 성공, false이면 뱅크 충돌로 실패 (다음 사이클 재시도)
 *
 * 완료된 명령어의 목적 레지스터를 레지스터 파일에 쓰는 과정:
 *   1. 각 목적 레지스터의 뱅크 번호 계산
 *   2. 해당 뱅크가 idle이면 중재자에 쓰기 예약 (allocate_bank_for_write)
 *   3. 뱅크가 사용 중이면 false 반환 (stall — writeback 재시도)
 *   4. 성공하면 clock_gated_reg_file 설정에 따라 쓰기 카운터 업데이트
 *
 * 호출 체인: shader_core_ctx::writeback() / ldst_unit::writeback() → [이 함수]
 */
bool opndcoll_rfu_t::writeback(warp_inst_t &inst) {
  assert(!inst.empty()); // [한국어] 비어있는 명령어에 writeback 시도 방지

  std::list<unsigned> regs = m_shader->get_regs_written(inst); // [한국어] 이 명령어가 쓰는 레지스터 목록 (실제 쓰기 수 계산용)
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    // [한국어] 목적 레지스터 번호 (-1이면 유효하지 않음)
    if (reg_num >= 0) {                   // valid register
      // [한국어] 유효한 목적 레지스터: 해당 뱅크가 idle인지 확인
      unsigned bank =
          register_bank(reg_num, inst.warp_id(), m_num_banks, sub_core_model,
                        m_num_banks_per_sched, inst.get_schd_id()); // [한국어] 레지스터 뱅크 번호 계산
      if (m_arbiter.bank_idle(bank)) {
        // [한국어] 뱅크가 idle: 쓰기 예약
        m_arbiter.allocate_bank_for_write(
            bank, op_t(&inst, reg_num, m_num_banks, sub_core_model,
                       m_num_banks_per_sched, inst.get_schd_id())); // [한국어] 이 뱅크에 쓰기 op 할당
        inst.arch_reg.dst[op] = -1; // [한국어] 처리 완료 표시 (다음 op에서 중복 처리 방지)
      } else {
        return false; // [한국어] 뱅크 충돌: writeback 실패 → 다음 사이클 재시도
      }
    }
  }
  for (unsigned i = 0; i < (unsigned)regs.size(); i++) {
    // [한국어] 레지스터 파일 쓰기 카운터 업데이트 (AccelWattch 전력 모델)
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      // [한국어] clock gating: 활성 스레드 그룹별로 쓰기 카운터 계산
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (inst.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group; // [한국어] 이 그룹에 활성 스레드가 하나라도 있으면 그룹 전체를 카운트
            break;
          }
        }
      }
      m_shader->incregfile_writes(active_count); // [한국어] 실제 쓰기 레인 수로 카운터 증가
    } else {
      m_shader->incregfile_writes(
          m_shader->get_config()->warp_size);  // inst.active_count());
      // [한국어] clock gating 없음: warp_size 전체를 쓰기 카운트
    }
  }
  return true; // [한국어] 모든 목적 레지스터 쓰기 예약 성공
}

/*
 * [한국어]
 * opndcoll_rfu_t::dispatch_ready_cu — 오퍼랜드 수집 완료된 콜렉터 유닛을 OC_EX로 dispatch
 *
 * @return: void
 *
 * 모든 dispatch 유닛을 순회하여 준비된 콜렉터 유닛을 찾아 OC_EX 파이프라인 레지스터로 이동.
 * clock_gated_reg_file이 활성화된 경우 ialu/const 읽기 카운터도 업데이트.
 *
 * 호출 체인: shader_core_ctx::read_operands() → [이 함수]
 */
void opndcoll_rfu_t::dispatch_ready_cu() {
  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    dispatch_unit_t &du = m_dispatch_units[p];
    collector_unit_t *cu = du.find_ready(); // [한국어] 이 dispatch 유닛에서 준비된(m_not_ready.none()) 콜렉터 탐색
    if (cu) {
      // [한국어] 준비된 콜렉터를 dispatch: 비레지스터 오퍼랜드 카운터 업데이트 후 OC_EX로 이동
      for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
           i++) {
        // [한국어] 레지스터가 아닌 오퍼랜드(상수, 즉각값 등)에 대한 카운터 업데이트
        if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
          unsigned active_count = 0;
          for (unsigned i = 0; i < m_shader->get_config()->warp_size;
               i = i + m_shader->get_config()->n_regfile_gating_group) {
            for (unsigned j = 0;
                 j < m_shader->get_config()->n_regfile_gating_group; j++) {
              if (cu->get_active_mask().test(i + j)) {
                active_count += m_shader->get_config()->n_regfile_gating_group;
                break;
              }
            }
          }
          m_shader->incnon_rf_operands(active_count); // [한국어] 비레지스터 오퍼랜드 카운터 (clock gating 적용)
        } else {
          m_shader->incnon_rf_operands(
              m_shader->get_config()->warp_size);  // cu->get_active_count());
          // [한국어] 비레지스터 오퍼랜드 카운터 (전체 warp_size)
        }
      }
      cu->dispatch(); // [한국어] 모든 오퍼랜드 수집 완료: OC_EX 파이프라인 레지스터로 이동
    }
  }
}

/*
 * [한국어]
 * opndcoll_rfu_t::allocate_cu — 입력 파이프라인 레지스터를 빈 콜렉터 유닛에 할당
 *
 * @port_num: 처리할 입력 포트 번호
 * @return:   void
 *
 * ID_OC 단계의 파이프라인 레지스터에 ready 명령어가 있으면 빈 콜렉터 유닛 탐색:
 *   1. sub_core_model이면 발행한 스케줄러 ID에 해당하는 콜렉터 범위만 탐색
 *   2. 빈 콜렉터 유닛에 allocate() 후 중재자에 읽기 요청 등록
 *
 * 오퍼랜드 읽기의 첫 단계: ID_OC → 콜렉터 유닛 할당 → 레지스터 파일 읽기 대기
 *
 * 호출 체인: shader_core_ctx::read_operands() → [이 함수]
 */
void opndcoll_rfu_t::allocate_cu(unsigned port_num) {
  input_port_t &inp = m_in_ports[port_num]; // [한국어] 이 포트의 입력/출력 파이프라인 레지스터 집합
  for (unsigned i = 0; i < inp.m_in.size(); i++) {
    if ((*inp.m_in[i]).has_ready()) {
      // [한국어] 이 입력 파이프라인 레지스터에 ready 명령어가 있음: 빈 콜렉터 탐색
      // find a free cu
      for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]]; // [한국어] 이 포트가 서비스하는 콜렉터 세트
        bool allocated = false;
        unsigned cuLowerBound = 0;
        unsigned cuUpperBound = cu_set.size();
        unsigned schd_id;
        if (sub_core_model) {
          // Sub core model only allocates on the subset of CUs assigned to the
          // scheduler that issued
          // [한국어] sub_core_model: 발행한 스케줄러에 할당된 콜렉터 부분 집합만 탐색
          unsigned reg_id = (*inp.m_in[i]).get_ready_reg_id(); // [한국어] ready 파이프라인 레지스터의 schd_id 추출
          schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
          assert(cu_set.size() % m_num_warp_scheds == 0 &&
                 cu_set.size() >= m_num_warp_scheds);
          unsigned cusPerSched = cu_set.size() / m_num_warp_scheds; // [한국어] 스케줄러당 콜렉터 수
          cuLowerBound = schd_id * cusPerSched; // [한국어] 이 스케줄러의 콜렉터 시작 인덱스
          cuUpperBound = cuLowerBound + cusPerSched; // [한국어] 이 스케줄러의 콜렉터 끝 인덱스
          assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        }
        for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
          if (cu_set[k].is_free()) {
            // [한국어] 빈 콜렉터 유닛 발견: 이 포트의 ready 명령어 할당
            collector_unit_t *cu = &cu_set[k];
            allocated = cu->allocate(inp.m_in[i], inp.m_out[i]); // [한국어] 명령어를 콜렉터에 이동하고 소스 레지스터 m_not_ready 설정
            m_arbiter.add_read_requests(cu); // [한국어] 중재자에 읽기 요청 등록 (wavefront 알고리즘이 처리)
            break;
          }
        }
        if (allocated) break;  // cu has been allocated, no need to search more.
        // [한국어] 할당 성공: 더 이상 다른 콜렉터 세트를 탐색할 필요 없음
      }
      // break;  // can only service a single input, if it failed it will fail
      // for
      // others.
    }
  }
}

/*
 * [한국어]
 * opndcoll_rfu_t::allocate_reads — 중재자의 읽기 결과를 콜렉터 유닛에 전달
 *
 * @return: void
 *
 * wavefront 중재자(allocate_reads)의 결과를 받아 각 콜렉터 유닛의 오퍼랜드를 수집:
 *   1. arbiter.allocate_reads()로 뱅크 충돌 없는 읽기 목록 획득
 *   2. 각 허가된 오퍼랜드에 대해 allocate_for_read로 중재자 상태 업데이트
 *   3. 콜렉터 유닛에 collect_operand()로 오퍼랜드 수집 완료 통보 (m_not_ready 비트 클리어)
 *   4. incregfile_reads로 레지스터 파일 읽기 카운터 업데이트
 *
 * 모든 오퍼랜드가 수집되면 (m_not_ready.none()) 콜렉터 유닛이 dispatch 가능 상태.
 *
 * 호출 체인: shader_core_ctx::read_operands() → [이 함수]
 */
void opndcoll_rfu_t::allocate_reads() {
  // process read requests that do not have conflicts
  std::list<op_t> allocated = m_arbiter.allocate_reads(); // [한국어] wavefront 중재자로 이번 사이클 허가된 읽기 요청 목록
  std::map<unsigned, op_t> read_ops; // [한국어] bank_id → op_t 매핑 (뱅크별 최대 1개 읽기)
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg(); // [한국어] 읽을 레지스터 번호
    unsigned wid = rr.get_wid(); // [한국어] warp ID
    unsigned bank = register_bank(reg, wid, m_num_banks, sub_core_model,
                                  m_num_banks_per_sched, rr.get_sid()); // [한국어] 이 레지스터의 뱅크 번호 재계산
    m_arbiter.allocate_for_read(bank, rr); // [한국어] 중재자에 이 뱅크가 이번 사이클에 읽기로 할당됨을 알림
    read_ops[bank] = rr; // [한국어] 뱅크별 읽기 오퍼랜드 저장
  }
  std::map<unsigned, op_t>::iterator r;
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    // [한국어] 각 허가된 읽기 오퍼랜드를 해당 콜렉터 유닛에 전달
    op_t &op = r->second;
    unsigned cu = op.get_oc_id(); // [한국어] 이 오퍼랜드가 속한 콜렉터 유닛 ID
    unsigned operand = op.get_operand(); // [한국어] 이 콜렉터 유닛의 몇 번째 오퍼랜드인지
    m_cu[cu]->collect_operand(operand); // [한국어] m_not_ready의 operand 비트 클리어: 이 오퍼랜드 수집 완료
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      // [한국어] clock gating 적용: 활성 스레드 그룹별로 읽기 카운터 계산
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (op.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group;
            break;
          }
        }
      }
      m_shader->incregfile_reads(active_count); // [한국어] 레지스터 파일 읽기 카운터 증가 (AccelWattch 전력)
    } else {
      m_shader->incregfile_reads(
          m_shader->get_config()->warp_size);  // op.get_active_count());
      // [한국어] clock gating 없음: warp_size 전체를 읽기 카운트
    }
  }
}

/*
 * [한국어]
 * opndcoll_rfu_t::collector_unit_t::ready — 콜렉터 유닛이 dispatch 가능한지 확인
 *
 * @return: true이면 dispatch 가능 (할당됨 + 모든 오퍼랜드 수집 + OC_EX 출력 공간 있음)
 *
 * 세 조건 모두 만족해야 dispatch 가능:
 *   1. !m_free: 콜렉터가 명령어를 보유 중
 *   2. m_not_ready.none(): 모든 소스 오퍼랜드 수집 완료
 *   3. output_register.has_free: OC_EX 파이프라인 레지스터에 빈 슬롯 있음
 */
bool opndcoll_rfu_t::collector_unit_t::ready() const {
  return (!m_free) && m_not_ready.none() &&
         (*m_output_register).has_free(m_sub_core_model, m_reg_id); // [한국어] sub_core_model이면 해당 스케줄러 슬롯만 확인
}

/*
 * [한국어]
 * opndcoll_rfu_t::collector_unit_t::dump — 콜렉터 유닛 상태 출력 (디버그용)
 *
 * @fp:     출력 파일
 * @shader: SM 포인터
 * @return: void
 *
 * 비어있으면 "<free>", 아니면 warp 명령어와 아직 수집되지 않은 오퍼랜드 목록 출력.
 */
void opndcoll_rfu_t::collector_unit_t::dump(
    FILE *fp, const shader_core_ctx *shader) const {
  if (m_free) {
    fprintf(fp, "    <free>\n"); // [한국어] 콜렉터가 비어있음
  } else {
    m_warp->print(fp); // [한국어] 현재 처리 중인 warp 명령어 출력
    for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      if (m_not_ready.test(i)) {
        std::string r = m_src_op[i].get_reg_string();
        fprintf(fp, "    '%s' not ready\n", r.c_str()); // [한국어] 아직 수집되지 않은 오퍼랜드 레지스터 이름 출력
      }
    }
  }
}

/*
 * [한국어]
 * opndcoll_rfu_t::collector_unit_t::init — 콜렉터 유닛 초기화
 *
 * @n:             콜렉터 유닛 ID
 * @num_banks:     레지스터 파일 뱅크 수
 * @config:        셰이더 코어 설정
 * @rfu:           소유 opndcoll_rfu_t
 * @sub_core_model: sub_core_model 여부
 * @reg_id:        이 콜렉터의 스케줄러 ID (sub_core_model에서 사용)
 * @banks_per_sched: 스케줄러당 뱅크 수
 * @return: void
 *
 * 호출 체인: opndcoll_rfu_t::init() → [이 함수]
 */
void opndcoll_rfu_t::collector_unit_t::init(unsigned n, unsigned num_banks,
                                            const core_config *config,
                                            opndcoll_rfu_t *rfu,
                                            bool sub_core_model,
                                            unsigned reg_id,
                                            unsigned banks_per_sched) {
  m_rfu = rfu;              // [한국어] 소유 opndcoll_rfu_t 포인터
  m_cuid = n;               // [한국어] 콜렉터 유닛 ID
  m_num_banks = num_banks;  // [한국어] 레지스터 파일 뱅크 수
  assert(m_warp == NULL);
  m_warp = new warp_inst_t(config); // [한국어] 명령어를 저장할 warp_inst_t 할당
  m_sub_core_model = sub_core_model;
  m_reg_id = reg_id;        // [한국어] 이 콜렉터의 스케줄러 ID (dispatch 시 올바른 출력 슬롯 선택)
  m_num_banks_per_sched = banks_per_sched;
}

/*
 * [한국어]
 * opndcoll_rfu_t::collector_unit_t::allocate — 입력 파이프라인 레지스터를 이 콜렉터에 할당
 *
 * @pipeline_reg_set: ID_OC 단계의 입력 파이프라인 레지스터 집합
 * @output_reg_set:   OC_EX 단계의 출력 파이프라인 레지스터 집합
 * @return:           true이면 성공, false이면 실패
 *
 * ID_OC의 ready 명령어를 이 콜렉터로 이동하고 소스 레지스터 정보를 분석:
 *   - 각 소스 레지스터를 m_src_op에 등록하고 m_not_ready 비트 설정
 *   - 중복 레지스터는 하나만 등록 (뱅크 중복 예약 방지)
 *   - pipeline_reg_set에서 명령어를 m_warp로 이동
 *
 * 호출 체인: opndcoll_rfu_t::allocate_cu() → [이 함수]
 */
bool opndcoll_rfu_t::collector_unit_t::allocate(register_set *pipeline_reg_set,
                                                register_set *output_reg_set) {
  assert(m_free);          // [한국어] 빈 콜렉터에만 할당 가능
  assert(m_not_ready.none()); // [한국어] 이전 오퍼랜드가 완전히 수집된 상태여야 함
  m_free = false;          // [한국어] 이 콜렉터는 이제 사용 중
  m_output_register = output_reg_set; // [한국어] OC_EX 파이프라인 레지스터 참조 저장
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready(); // [한국어] ID_OC의 ready 명령어 포인터 획득
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    m_warp_id = (*pipeline_reg)->warp_id();
    std::vector<int> prev_regs;  // remove duplicate regs within same instr
    // [한국어] 중복 레지스터 추적 (같은 레지스터를 두 오퍼랜드가 참조하면 한 번만 읽기 요청)
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op];  // this math needs to match that used in
                                   // function_info::ptx_decode_inst
      // [한국어] 소스 레지스터 번호 (음수면 유효하지 않음)
      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num) new_reg = false; // [한국어] 이미 등록된 레지스터: 중복 건너뜀
      }
      if (reg_num >= 0 && new_reg) {  // valid register
        prev_regs.push_back(reg_num);
        m_src_op[op] =
            op_t(this, op, reg_num, m_num_banks, m_sub_core_model,
                 m_num_banks_per_sched, (*pipeline_reg)->get_schd_id()); // [한국어] 소스 오퍼랜드 정보 등록
        m_not_ready.set(op); // [한국어] 이 오퍼랜드는 아직 수집되지 않음 (레지스터 파일 읽기 대기)
      } else
        m_src_op[op] = op_t(); // [한국어] 유효하지 않은 오퍼랜드: 빈 op_t
    }
    // move_warp(m_warp,*pipeline_reg);
    pipeline_reg_set->move_out_to(m_warp); // [한국어] ID_OC에서 이 콜렉터로 명령어 이동
    return true;
  }
  return false; // [한국어] ready 명령어가 없음 (예상치 못한 케이스)
}

/*
 * [한국어]
 * opndcoll_rfu_t::collector_unit_t::dispatch — 모든 오퍼랜드 수집 완료: OC_EX로 dispatch
 *
 * @return: void
 *
 * m_not_ready가 모두 0이 된 후 호출. OC_EX 파이프라인 레지스터로 명령어를 이동하고
 * 이 콜렉터 유닛을 free 상태로 복원.
 *
 * 호출 체인: opndcoll_rfu_t::dispatch_ready_cu() → [이 함수]
 */
void opndcoll_rfu_t::collector_unit_t::dispatch() {
  assert(m_not_ready.none()); // [한국어] 모든 오퍼랜드가 수집된 상태여야 함
  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp); // [한국어] OC_EX 파이프라인 레지스터로 명령어 이동
  m_free = true;             // [한국어] 이 콜렉터 유닛을 free 상태로 복원
  m_output_register = NULL;  // [한국어] 출력 레지스터 참조 해제
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) m_src_op[i].reset(); // [한국어] 소스 오퍼랜드 정보 초기화 (다음 명령어를 위해)
}

/*
 * [한국어]
 * exec_simt_core_cluster::create_shader_core_ctx — execution-driven 모드의 SM 생성
 *
 * @return: void
 *
 * 클러스터 내 각 SM을 exec_shader_core_ctx 인스턴스로 생성하고 m_core 배열에 저장.
 * exec_shader_core_ctx는 PTX 기능 시뮬레이션을 직접 실행하는 execution-driven 모드.
 *
 * 호출 체인: simt_core_cluster 생성자 → [이 함수]
 */
void exec_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster]; // [한국어] 클러스터 내 SM 포인터 배열 할당
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id); // [한국어] (클러스터 내 인덱스, 클러스터 ID) → 전역 SM ID 변환
    m_core[i] = new exec_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                         m_config, m_mem_config, m_stats); // [한국어] execution-driven SM 객체 생성
    m_core_sim_order.push_back(i); // [한국어] 사이클 실행 순서 목록에 추가 (라운드로빈 순환 가능)
  }
}

/*
 * [한국어]
 * simt_core_cluster::simt_core_cluster — SIMT 코어 클러스터 생성자
 *
 * @gpu:        최상위 GPU 시뮬레이터
 * @cluster_id: 이 클러스터의 ID
 * @config:     셰이더 코어 설정
 * @mem_config: 메모리 설정
 * @stats:      통계 객체
 * @mstats:     메모리 통계 객체
 *
 * TPC(Texture Processing Cluster)에 해당하는 simt_core_cluster를 초기화.
 * m_cta_issue_next_core를 마지막 SM부터 시작하여 첫 번째 CTA가 SM 0에서 실행되도록.
 *
 * 호출 체인: gpgpu_sim 생성자 → [이 생성자]
 */
simt_core_cluster::simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                                     const shader_core_config *config,
                                     const memory_config *mem_config,
                                     shader_core_stats *stats,
                                     class memory_stats_t *mstats) {
  m_config = config;
  m_cta_issue_next_core = m_config->n_simt_cores_per_cluster -
                          1;  // this causes first launch to use hw cta 0
  // [한국어] 마지막 SM부터 시작: 첫 번째 issue_block2core() 호출 시 +1 후 modulo로 SM 0 선택
  m_cluster_id = cluster_id;
  m_gpu = gpu;
  m_stats = stats;
  m_memory_stats = mstats;
  m_mem_config = mem_config;
}

/*
 * [한국어]
 * simt_core_cluster::core_cycle — 클러스터 내 모든 SM의 한 사이클 처리
 *
 * @return: void
 *
 * m_core_sim_order 순서로 각 SM의 cycle()을 호출.
 * simt_core_sim_order == 1이면 리스트를 회전시켜 라운드로빈 순서로 처리
 * (시뮬레이션 편향 방지 목적).
 *
 * 호출 체인: gpgpu_sim::cycle() → [이 함수]
 */
void simt_core_cluster::core_cycle() {
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    m_core[*it]->cycle(); // [한국어] 각 SM의 한 사이클 처리
  }

  if (m_config->simt_core_sim_order == 1) {
    // [한국어] 라운드로빈 순서: 리스트의 첫 번째 원소를 끝으로 이동 (공정성 보장)
    m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order,
                            m_core_sim_order.begin());
  }
}

/*
 * [한국어]
 * simt_core_cluster::reinit — 클러스터 내 모든 SM 재초기화
 *
 * 시뮬레이션 리셋 또는 새 커널 시작 전 SM 상태 초기화.
 */
void simt_core_cluster::reinit() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->reinit(0, m_config->n_thread_per_shader, true); // [한국어] 각 SM 재초기화: 스레드 0부터 n_thread_per_shader까지
}

/*
 * [한국어]
 * simt_core_cluster::max_cta — 이 클러스터 전체의 최대 CTA 수 반환
 *
 * 클러스터 내 SM 수 × 각 SM의 max_cta.
 */
unsigned simt_core_cluster::max_cta(const kernel_info_t &kernel) {
  return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel); // [한국어] 클러스터 전체 최대 CTA = SM당 최대 × SM 수
}

/*
 * [한국어]
 * simt_core_cluster::get_not_completed — 클러스터 내 완료되지 않은 스레드 수 합산
 */
unsigned simt_core_cluster::get_not_completed() const {
  unsigned not_completed = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    not_completed += m_core[i]->get_not_completed(); // [한국어] 각 SM의 미완료 스레드 수 합산
  return not_completed;
}

/*
 * [한국어]
 * simt_core_cluster::print_not_completed — 각 SM의 미완료 스레드 수 출력
 */
void simt_core_cluster::print_not_completed(FILE *fp) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned not_completed = m_core[i]->get_not_completed();
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id); // [한국어] 클러스터 내 인덱스 → 전역 SM ID
    fprintf(fp, "%u(%u) ", sid, not_completed);
  }
}

/*
 * [한국어]
 * simt_core_cluster::get_current_occupancy — 클러스터의 평균 SM occupancy 반환
 *
 * @active: 활성 슬롯 수 누적 (out)
 * @total:  전체 슬롯 수 누적 (out)
 * @return: 클러스터 내 SM 평균 occupancy (0.0~1.0)
 */
float simt_core_cluster::get_current_occupancy(
    unsigned long long &active, unsigned long long &total) const {
  float aggregate = 0.f;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    aggregate += m_core[i]->get_current_occupancy(active, total); // [한국어] 각 SM의 occupancy 합산
  }
  return aggregate / m_config->n_simt_cores_per_cluster; // [한국어] 평균 반환
}

/*
 * [한국어]
 * simt_core_cluster::get_n_active_cta — 클러스터 내 활성 CTA 수 합산
 */
unsigned simt_core_cluster::get_n_active_cta() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->get_n_active_cta(); // [한국어] 각 SM의 활성 CTA 수 합산
  return n;
}

/*
 * [한국어]
 * simt_core_cluster::get_n_active_sms — 클러스터 내 활성 SM 수 반환
 */
unsigned simt_core_cluster::get_n_active_sms() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->isactive(); // [한국어] 각 SM의 활성 여부 합산 (0 또는 1)
  return n;
}

/*
 * [한국어]
 * simt_core_cluster::issue_block2core — 클러스터 내 SM 중 하나에 CTA 발행
 *
 * @return: 발행된 블록 수 (0 또는 1)
 *
 * m_cta_issue_next_core + 1부터 시작하는 라운드로빈으로 SM을 탐색:
 *   1. 동시 커널 모드(gpgpu_concurrent_kernel_sm): GPU에서 최신 커널 선택
 *   2. 일반 모드: 현재 SM의 커널 유지, 더 이상 CTA가 없고 완료된 경우에만 새 커널 선택
 * 커널에 CTA가 남아있고 SM이 수용 가능하면 issue_block2core() 호출.
 *
 * 호출 체인: gpgpu_sim::issue_block2core() → [이 함수] → shader_core_ctx::issue_block2core()
 */
unsigned simt_core_cluster::issue_block2core() {
  unsigned num_blocks_issued = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster; // [한국어] m_cta_issue_next_core+1부터 시작하는 라운드로빈

    kernel_info_t *kernel;
    // Jin: fetch kernel according to concurrent kernel setting
    if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
      // always select latest issued kernel
      kernel_info_t *k = m_gpu->select_kernel(); // [한국어] 동시 커널 모드: GPU의 실행 대기 큐에서 커널 선택
      kernel = k;
    } else {
      // first select core kernel, if no more cta, get a new kernel
      // only when core completes
      kernel = m_core[core]->get_kernel(); // [한국어] 이 SM의 현재 커널
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        // wait till current kernel finishes
        if (m_core[core]->get_not_completed() == 0) {
          // [한국어] 이 SM의 모든 스레드 완료 + 현재 커널에 더 이상 CTA 없음: 새 커널 선택
          kernel_info_t *k = m_gpu->select_kernel();
          if (k) m_core[core]->set_kernel(k); // [한국어] 새 커널 할당
          kernel = k;
        }
      }
    }

    if (m_gpu->kernel_more_cta_left(kernel) &&
        //            (m_core[core]->get_n_active_cta() <
        //            m_config->max_cta(*kernel)) ) {
        m_core[core]->can_issue_1block(*kernel)) {
      // [한국어] 이 SM이 CTA를 수용할 수 있으면 발행
      m_core[core]->issue_block2core(*kernel); // [한국어] SM에 CTA 발행: warp 초기화, 스레드 생성, 자원 할당
      num_blocks_issued++;
      m_cta_issue_next_core = core; // [한국어] 다음 사이클의 시작 SM 업데이트
      break; // [한국어] 이번 사이클에 최대 1개 CTA 발행
    }
  }
  return num_blocks_issued;
}

/*
 * [한국어]
 * simt_core_cluster::cache_flush — 클러스터 내 모든 SM의 캐시 flush
 */
void simt_core_cluster::cache_flush() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_flush(); // [한국어] 각 SM의 L1 캐시 flush (dirty 라인 → L2 writeback)
}

/*
 * [한국어]
 * simt_core_cluster::cache_invalidate — 클러스터 내 모든 SM의 캐시 무효화
 */
void simt_core_cluster::cache_invalidate() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_invalidate(); // [한국어] 각 SM의 L1 캐시 무효화 (메모리 일관성)
}

/*
 * [한국어]
 * simt_core_cluster::icnt_injection_buffer_full — NoC 주입 버퍼 포화 여부 확인
 *
 * @size:  요청 패킷 크기 (바이트)
 * @write: true이면 쓰기 요청 (데이터 포함), false이면 읽기 요청 (메타데이터만)
 * @return: true이면 NoC 버퍼가 가득 참 → 이 SM은 새 요청 발행 불가
 *
 * 읽기 요청은 READ_PACKET_SIZE(메타데이터 크기)로 감소시켜 확인.
 */
bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write) {
  unsigned request_size = size;
  if (!write) request_size = READ_PACKET_SIZE; // [한국어] 읽기 요청: 제어 헤더 크기만 확인 (데이터 없음)
  return !::icnt_has_buffer(m_cluster_id, request_size); // [한국어] intersim2에 이 클러스터에서 request_size 패킷을 주입할 공간이 있는지 확인
}

bool sst_simt_core_cluster::SST_injection_buffer_full(unsigned size, bool write,
                                                      mem_access_type type) {
  switch (type) {
    case CONST_ACC_R:
    case INST_ACC_R: {
      return response_queue_full();
      break;
    }
    default: {
      return ::is_SST_buffer_full(m_cluster_id);
      break;
    }
  }
}

/*
 * [한국어]
 * simt_core_cluster::icnt_inject_request_packet — mem_fetch를 NoC(intersim2)에 주입
 *
 * @mf: 주입할 메모리 요청 패킷
 * @return: void
 *
 * SM에서 발생한 메모리 요청을 NoC를 통해 메모리 파티션으로 전송:
 *   1. 통계 업데이트 (update_icnt_stats)
 *   2. 패킷 크기 결정: 쓰기/원자적은 데이터 포함(mf->size()), 읽기는 제어 헤더만(ctrl_size)
 *   3. 상태를 IN_ICNT_TO_MEM으로 설정
 *   4. icnt_push로 intersim2에 패킷 주입 (목적지: sub_partition_id → mem2device 변환)
 *
 * 호출 체인: ldst_unit::memory_cycle() → simt_core_cluster::icnt_inject_request_packet()
 */
void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {
  // Update stats based on mf type
  update_icnt_stats(mf); // [한국어] 메모리 접근 타입별 통계 카운터 업데이트

  // The packet size varies depending on the type of request:
  // - For write request and atomic request, the packet contains the data
  // - For read request (i.e. not write nor atomic), the packet only has control
  // metadata
  unsigned int packet_size = mf->size(); // [한국어] 기본값: 전체 패킷 크기 (데이터 + 헤더)
  if (!mf->get_is_write() && !mf->isatomic()) {
    packet_size = mf->get_ctrl_size(); // [한국어] 읽기 요청: 제어 헤더만 전송 (데이터는 응답에 포함)
  }
  m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size); // [한국어] 송신 트래픽 통계 기록
  unsigned destination = mf->get_sub_partition_id(); // [한국어] 목적 메모리 서브 파티션 ID (주소 디코딩 결과)
  mf->set_status(IN_ICNT_TO_MEM,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle); // [한국어] 상태: NoC를 통해 메모리로 이동 중
  if (!mf->get_is_write() && !mf->isatomic())
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->get_ctrl_size()); // [한국어] 읽기 요청: 제어 헤더 크기로 NoC에 주입
  else
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->size()); // [한국어] 쓰기/원자적: 전체 패킷(데이터 포함) 크기로 NoC에 주입
}

/*
 * [한국어]
 * simt_core_cluster::update_icnt_stats — NoC 트래픽 통계 업데이트
 *
 * @mf: 통계를 기록할 mem_fetch
 * @return: void
 *
 * 메모리 접근 타입별로 전역 통계 카운터를 증가.
 * 읽기/쓰기 총 수, CONST/TEXTURE/GLOBAL/LOCAL/INST 별 세분 통계.
 *
 * 호출 체인: icnt_inject_request_packet() → [이 함수]
 */
void simt_core_cluster::update_icnt_stats(class mem_fetch *mf) {
  // stats
  if (mf->get_is_write())
    m_stats->made_write_mfs++; // [한국어] 쓰기 요청 총 수
  else
    m_stats->made_read_mfs++;  // [한국어] 읽기 요청 총 수
  switch (mf->get_access_type()) {
    case CONST_ACC_R:
      m_stats->gpgpu_n_mem_const++;
      break;
    case TEXTURE_ACC_R:
      m_stats->gpgpu_n_mem_texture++;
      break;
    case GLOBAL_ACC_R:
      m_stats->gpgpu_n_mem_read_global++;
      break;
    // case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++;
    // printf("read_global%d\n",m_stats->gpgpu_n_mem_read_global); break;
    case GLOBAL_ACC_W:
      m_stats->gpgpu_n_mem_write_global++;
      break;
    case LOCAL_ACC_R:
      m_stats->gpgpu_n_mem_read_local++;
      break;
    case LOCAL_ACC_W:
      m_stats->gpgpu_n_mem_write_local++;
      break;
    case INST_ACC_R:
      m_stats->gpgpu_n_mem_read_inst++;
      break;
    case L1_WRBK_ACC:
      m_stats->gpgpu_n_mem_write_global++;
      break;
    case L2_WRBK_ACC:
      m_stats->gpgpu_n_mem_l2_writeback++;
      break;
    case L1_WR_ALLOC_R:
      m_stats->gpgpu_n_mem_l1_write_allocate++;
      break;
    case L2_WR_ALLOC_R:
      m_stats->gpgpu_n_mem_l2_write_allocate++;
      break;
    default:
      assert(0);
  }
}

/*
 * [한국어]
 * sst_simt_core_cluster::icnt_inject_request_packet_to_SST — SST 메모리 계층으로 요청 전송
 *
 * @mf: 전송할 메모리 요청 패킷
 * @return: void
 *
 * SST(Structural Simulation Toolkit) 모드에서 사용하는 메모리 요청 전송 경로:
 *   - CONST_ACC_R / INST_ACC_R: perfect memory로 처리 (즉시 response_fifo에 추가)
 *   - 그 외 읽기: send_read_request_SST로 SST 메모리 시스템에 전달
 *   - 그 외 쓰기/원자적: send_write_request_SST로 전달
 *
 * 호출 체인: ldst_unit::memory_cycle() (SST 모드) → [이 함수]
 */
void sst_simt_core_cluster::icnt_inject_request_packet_to_SST(
    class mem_fetch *mf) {
  // Update stats
  update_icnt_stats(mf); // [한국어] 통계 업데이트

  // The packet size varies depending on the type of request:
  // - For write request and atomic request, the packet contains the data
  // - For read request (i.e. not write nor atomic), the packet only has control
  // metadata
  unsigned int packet_size = mf->size();
  if (!mf->get_is_write() && !mf->isatomic()) {
    packet_size = mf->get_ctrl_size(); // [한국어] 읽기: 제어 헤더만
  }
  m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size); // [한국어] 트래픽 통계
  mf->set_status(IN_ICNT_TO_MEM,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle); // [한국어] 상태 업데이트
  switch (mf->get_access_type()) {
    case CONST_ACC_R:
    case INST_ACC_R: {
      push_response_fifo(mf); // [한국어] 상수/명령어 접근: SST perfect memory로 즉시 응답
      break;
    }
    default: {
      if (!mf->get_is_write() && !mf->isatomic())
        ::send_read_request_SST(m_cluster_id, mf->get_addr(),
                                mf->get_data_size(), (void *)mf); // [한국어] SST에 읽기 요청 전송
      else
        ::send_write_request_SST(m_cluster_id, mf->get_addr(),
                                 mf->get_data_size(), (void *)mf); // [한국어] SST에 쓰기 요청 전송

      break;
    }
  }
}

/*
 * [한국어]
 * simt_core_cluster::icnt_cycle — NoC 응답 처리 사이클 (메모리→SM 방향)
 *
 * @return: void
 *
 * 두 단계로 처리:
 *   1단계 — m_response_fifo에서 SM으로 응답 전달:
 *     - INST_ACC_R: accept_fetch_response → L1I에 fill
 *     - 그 외: accept_ldst_unit_response → ldst_unit response_fifo에 추가
 *     - 각 SM의 버퍼가 가득 차면 대기 (백프레셔)
 *   2단계 — icnt_pop으로 NoC에서 새 응답 수신:
 *     - ejection buffer(m_response_fifo)에 여유가 있으면 NoC에서 꺼내 추가
 *     - 쓰기 ACK: 제어 메타데이터만, 읽기 응답: 데이터 포함 크기로 트래픽 기록
 *
 * 호출 체인: gpgpu_sim::cycle() → [이 함수]
 */
void simt_core_cluster::icnt_cycle() {
  if (!m_response_fifo.empty()) {
    // [한국어] 1단계: 응답 FIFO에서 SM으로 응답 전달
    mem_fetch *mf = m_response_fifo.front(); // [한국어] 가장 먼저 도착한 응답
    unsigned cid = m_config->sid_to_cid(mf->get_sid()); // [한국어] SM ID → 클러스터 내 인덱스 변환
    if (mf->get_access_type() == INST_ACC_R) {
      // instruction fetch response
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        // [한국어] I-cache 응답 버퍼에 여유: L1I로 전달
        m_response_fifo.pop_front();
        m_core[cid]->accept_fetch_response(mf); // [한국어] L1I에 캐시 라인 fill
      }
    } else {
      // data response
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        // [한국어] ldst 응답 버퍼에 여유: 데이터 캐시/메모리 응답으로 전달
        m_response_fifo.pop_front();
        m_memory_stats->memlatstat_read_done(mf); // [한국어] 메모리 레이턴시 통계 완료 기록
        m_core[cid]->accept_ldst_unit_response(mf); // [한국어] ldst_unit의 response_fifo에 추가
      }
    }
  }
  // [한국어] 2단계: NoC에서 새 응답 수신 (ejection buffer 여유가 있을 때만)
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id); // [한국어] intersim2에서 이 클러스터로 도착한 패킷 꺼내기
    if (!mf) return; // [한국어] 도착한 패킷 없음: 종료
    assert(mf->get_tpc() == m_cluster_id); // [한국어] 목적 클러스터가 맞는지 확인
    assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK); // [한국어] 응답 타입 검증

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size(); // [한국어] 쓰기 ACK=제어헤더, 읽기 응답=데이터 포함
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size); // [한국어] 수신 트래픽 통계
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle); // [한국어] 상태: 클러스터 ejection buffer에서 SM으로 이동 대기
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    m_response_fifo.push_back(mf); // [한국어] 응답 FIFO에 추가 (다음 사이클에 SM으로 전달)
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false); // [한국어] NoC→SM 방향 플릿 수 통계 업데이트
  }
}

/*
 * [한국어]
 * sst_simt_core_cluster::icnt_cycle_SST — SST 모드의 응답 처리 사이클
 *
 * @return: void
 *
 * simt_core_cluster::icnt_cycle의 SST 버전:
 *   - response_fifo → SM 전달 로직은 동일
 *   - NoC 대신 SST_pop_mem_reply로 SST 메모리에서 응답 수신
 *   - 원자적 연산 응답이 도착하면 즉시 do_atomic() 실행 (현재 검증 필요)
 *
 * 호출 체인: gpgpu_sim::cycle() (SST 모드) → [이 함수]
 */
void sst_simt_core_cluster::icnt_cycle_SST() {
  if (!m_response_fifo.empty()) {
    // [한국어] 1단계: 응답 FIFO에서 SM으로 응답 전달 (icnt_cycle과 동일)
    mem_fetch *mf = m_response_fifo.front();
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    if (mf->get_access_type() == INST_ACC_R) {
      // instruction fetch response
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_core[cid]->accept_fetch_response(mf); // [한국어] L1I에 fill
      }
    } else {
      // data response
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_memory_stats->memlatstat_read_done(mf);
        m_core[cid]->accept_ldst_unit_response(mf); // [한국어] ldst_unit response_fifo에 추가
      }
    }
  }

  // pop from SST buffers
  // [한국어] 2단계: SST 메모리에서 응답 수신
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    mem_fetch *mf = (mem_fetch *)(static_cast<sst_gpgpu_sim *>(get_gpu())
                                      ->SST_pop_mem_reply(m_cluster_id)); // [한국어] SST 메모리에서 이 클러스터의 응답 꺼내기
    if (!mf) return;
    assert(mf->get_tpc() == m_cluster_id);

    // do atomic here
    // For now, we execute atomic when the mem reply comes back
    // This needs to be validated
    // [한국어] 원자적 연산: 메모리 응답이 돌아올 때 실행 (현재 구현 - 검증 필요)
    if (mf && mf->isatomic()) mf->do_atomic(); // [한국어] 원자적 연산 실행 (atomicAdd 등)

    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size); // [한국어] 수신 트래픽 통계
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    m_response_fifo.push_back(mf); // [한국어] 응답 FIFO에 추가
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false); // [한국어] 플릿 수 통계
  }
}

/*
 * [한국어]
 * simt_core_cluster::get_pdom_stack_top_info — 특정 스레드의 SIMT 스택 상단 정보 조회
 *
 * 클러스터 레벨 래퍼: sid → cid 변환 후 SM에 위임.
 */
void simt_core_cluster::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                                unsigned *pc,
                                                unsigned *rpc) const {
  unsigned cid = m_config->sid_to_cid(sid); // [한국어] SM ID → 클러스터 내 인덱스
  m_core[cid]->get_pdom_stack_top_info(tid, pc, rpc); // [한국어] SM의 PDOM 스택 상단에서 pc, rpc 조회
}

/*
 * [한국어]
 * simt_core_cluster::display_pipeline — 특정 SM의 파이프라인 상태 출력 (디버그)
 *
 * SM의 파이프라인 출력 후 클러스터 레벨의 응답 FIFO 상태도 출력.
 */
void simt_core_cluster::display_pipeline(unsigned sid, FILE *fout,
                                         int print_mem, int mask) {
  m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout, print_mem, mask); // [한국어] SM 파이프라인 출력

  fprintf(fout, "\n");
  fprintf(fout, "Cluster %u pipeline state\n", m_cluster_id);
  fprintf(fout, "Response FIFO (occupancy = %zu):\n", m_response_fifo.size()); // [한국어] 클러스터 응답 FIFO 점유율
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout); // [한국어] 각 응답 패킷 정보 출력
  }
}

/*
 * [한국어]
 * simt_core_cluster::print_cache_stats — 클러스터 내 모든 SM의 L1 캐시 통계 출력
 */
void simt_core_cluster::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                          unsigned &dl1_misses) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->print_cache_stats(fp, dl1_accesses, dl1_misses); // [한국어] 각 SM의 L1 캐시 통계 누적
  }
}

/*
 * [한국어]
 * simt_core_cluster::get_icnt_stats — 클러스터 내 NoC 트래픽 통계 집계
 *
 * @n_simt_to_mem: SM→메모리 방향 통계 (out)
 * @n_mem_to_simt: 메모리→SM 방향 통계 (out)
 */
void simt_core_cluster::get_icnt_stats(long &n_simt_to_mem,
                                       long &n_mem_to_simt) const {
  long simt_to_mem = 0;
  long mem_to_simt = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt); // [한국어] 각 SM의 NoC 통계 누적
  }
  n_simt_to_mem = simt_to_mem;
  n_mem_to_simt = mem_to_simt;
}

/*
 * [한국어]
 * simt_core_cluster::get_cache_stats — 클러스터 내 모든 캐시 통계 집계
 */
void simt_core_cluster::get_cache_stats(cache_stats &cs) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_cache_stats(cs); // [한국어] 각 SM의 L1I/L1D/L1C/L1T 통계 누적
  }
}

/*
 * [한국어]
 * simt_core_cluster::get_L1I/L1D/L1C/L1T_sub_stats — 클러스터 레벨 캐시 서브 통계 집계
 *
 * 각 SM의 캐시 서브 통계를 클러스터 전체 합산으로 집계.
 */
void simt_core_cluster::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1I_sub_stats(temp_css);
    total_css += temp_css; // [한국어] L1I 서브 통계 합산
  }
  css = total_css;
}
void simt_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1D_sub_stats(temp_css);
    total_css += temp_css; // [한국어] L1D 서브 통계 합산
  }
  css = total_css;
}
void simt_core_cluster::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1C_sub_stats(temp_css);
    total_css += temp_css; // [한국어] L1C(상수 캐시) 서브 통계 합산
  }
  css = total_css;
}
void simt_core_cluster::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1T_sub_stats(temp_css);
    total_css += temp_css; // [한국어] L1T(텍스처 캐시) 서브 통계 합산
  }
  css = total_css;
}

/*
 * [한국어]
 * exec_shader_core_ctx::checkExecutionStatusAndUpdate — 기능 시뮬레이션 명령어 실행 후 상태 업데이트
 *
 * @inst: 방금 기능 시뮬레이션이 완료된 warp 명령어 (warp_inst_t)
 * @t:    warp 내 스레드 인덱스 (0~warp_size-1)
 * @tid:  SM 전체 스레드 ID (warp_id * warp_size + t)
 * @return: void
 *
 * func_exec_inst() 내부에서 각 활성 스레드에 대해 호출된다.
 * 명령어 종류에 따라 세 가지 상태를 업데이트한다:
 *
 *  1) 원자적 명령어(isatomic): m_n_atomic 카운터 증가 → 메모리 응답 도착 때까지 warp 블로킹
 *  2) 로컬 메모리 접근(is_local): PTX의 로컬 주소 공간을 글로벌 주소 공간의 스레드별 전용 슬롯으로 변환.
 *     GPGPU-Sim은 로컬 메모리를 글로벌 DRAM에 flat하게 매핑하며, 스레드 ID와 SM 수에 따라 오프셋을 계산.
 *  3) 스레드 완료(ptx_thread_done): 해당 스레드를 warp의 완료 비트에 set하고 I-buffer 플러시.
 *
 * 마지막으로 PC 히스토그램(cflog)을 갱신 — 제어 흐름 로그(cflog) 분석 도구에 현재 PC를 기록.
 *
 * 실행 컨텍스트: execution-driven 모드, func_exec_inst() 루프 내 각 스레드마다 호출.
 *
 * 호출 체인:
 *   exec_shader_core_ctx::func_exec_inst() → [이 함수] (각 활성 스레드에 대해)
 */
void exec_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                         unsigned t,
                                                         unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();
  // [한국어] 원자적 연산(atomic)이면 m_n_atomic 카운터 증가.
  // 이 카운터가 0보다 크면 shd_warp_t::waiting()에서 이 warp를 스케줄 불가 상태로 표시.
  // 원자적 연산은 글로벌 메모리 응답이 올 때 dec_n_atomic()으로 감소한다.

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    // [한국어] PTX 로컬 메모리 공간의 주소를 실제 글로벌 DRAM 주소로 변환.
    // GPGPU-Sim은 각 스레드의 로컬 메모리를 글로벌 주소 공간에 interleaved layout으로 매핑한다.
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD]; // [한국어] 변환된 실제 주소 배열 (최대 접근 수만큼)
    unsigned num_addrs; // [한국어] 실제로 생성된 주소 개수 (코얼레싱 전 스레드별 1개)
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        // [한국어] 총 SM 수 = 클러스터 수 × SM/클러스터 — 로컬 메모리 인터리빙 stride 계산에 사용
        inst.data_size, (new_addr_type *)localaddrs);
    // [한국어] translate_local_memaddr: PTX 로컬 주소 → DRAM flat 주소 변환 (cuda-sim/ptx_ir.cc 정의)
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    // [한국어] 변환된 주소를 inst의 해당 스레드 슬롯에 다시 저장 → 이후 메모리 코얼레싱에 사용
  }
  if (ptx_thread_done(tid)) {
    // [한국어] PTX 기능 시뮬레이션에서 이 스레드가 최종 명령어까지 실행을 마쳤으면
    m_warp[inst.warp_id()]->set_completed(t);
    // [한국어] warp의 완료 비트셋에 스레드 t를 set — 모든 스레드가 완료되면 warp 자체가 완료로 간주
    m_warp[inst.warp_id()]->ibuffer_flush();
    // [한국어] I-buffer에 남아있는 인스트럭션을 플러시 — 이미 완료된 스레드가 포함된 warp의 불필요 명령어 제거
  }

  // PC-Histogram Update
  // [한국어] 제어 흐름 로그(cflog) 갱신: 각 활성 스레드의 현재 PC를 기록.
  // cflog_update_thread_pc는 PC 히스토그램 통계를 수집 — warp divergence 분석, 핫스팟 프로파일링에 활용.
  unsigned warp_id = inst.warp_id(); // [한국어] 이 명령어를 실행한 warp ID
  unsigned pc = inst.pc; // [한국어] 현재 명령어의 PC (프로그램 카운터)
  for (unsigned t = 0; t < m_config->warp_size; t++) {
    // [한국어] warp 내 전체 스레드 순회 (비활성 스레드는 제외)
    if (inst.active(t)) {
      // [한국어] active mask 확인: SIMT 분기로 비활성화된 스레드는 PC 업데이트에서 제외
      int tid = warp_id * m_config->warp_size + t; // [한국어] SM 전체 스레드 ID = warp_id * 32 + t
      cflog_update_thread_pc(m_sid, tid, pc); // [한국어] sid별 제어 흐름 로그에 이 스레드의 PC 기록
    }
  }
}
