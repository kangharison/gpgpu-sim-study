/*
 * [한국어 설명] L2 캐시·메모리 파티션 구현 (l2cache.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 L2 캐시 계층과 DRAM 메모리 파티션 전체를 구현한다.
 * GPU의 각 DRAM 채널(memory_partition_unit)은 여러 개의 L2 슬라이스
 * (memory_sub_partition)를 포함하며, 이 파일이 두 클래스의 생성자·소멸자·사이클
 * 단위 동작을 모두 정의한다. L2 캐시 접근(HIT/MISS/RESERVATION_FAIL)과 ROP 지연 큐,
 * DRAM 크레딧 기반 중재(arbitration_metadata), simple_dram_model 및 실제 DRAM
 * 타이밍 모델(dram_cycle) 경로를 함께 관리한다. 또한 ICNT→L2→DRAM→L2→ICNT로
 * 이어지는 4단계 FIFO 파이프라인을 구동한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 타이밍 시뮬레이션 계층(gpgpu-sim/)의 메모리 서브시스템 최하단에 위치한다.
 * 실행 흐름:
 *   SM(shader.cc) → ICNT(intersim2) → memory_sub_partition::push()
 *     → ROP 큐 → icnt_L2_queue → L2cache::access()
 *     → L2_dram_queue → memory_partition_unit::dram_cycle() / simple_dram_model_cycle()
 *     → dram_t(dram.cc) → dram_L2_queue → L2cache::fill()
 *     → L2_icnt_queue → ICNT → SM
 * gpu-sim.cc의 gpgpu_sim::cycle()이 매 사이클마다 cache_cycle()과 dram_cycle()을
 * 호출함으로써 이 파일의 로직이 구동된다. 실행 컨텍스트: 단일 스레드 시뮬레이션 루프
 * (호스트 유저스페이스, cycle-by-cycle).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - dram.cc/h: DRAM 타이밍 모델(dram_t); dram_cycle()에서 직접 호출
 *   - gpu-cache.cc/h: l2_cache 클래스; L2 접근/충전/내보내기 API
 *   - mem_fetch.cc/h: 메모리 요청 패킷; 모든 큐 원소가 mem_fetch* 타입
 *   - mem_latency_stat.cc/h: 지연 통계(memlatstat_icnt2mem_pop 등)
 *   - shader.cc/h: SM에서 내보내는 메모리 요청의 원산지
 *   - gpu-sim.cc/h: gpgpu_sim::print_dram_stats(), cycle() 루프
 *   - abstract_hardware_model.h: active_mask_t, mem_access_t 타입 정의
 * 공유 자료구조:
 *   - mem_fetch: ICNT·L2·DRAM 경계를 가로지르는 핵심 패킷
 *   - memory_config: 파티션 수, 큐 크기, L2 설정 등 시뮬레이션 파라미터
 *   - memory_stats_t: 지연 통계 집계 객체
 *
 * === 주요 함수/구조체 요약 ===
 * partition_mf_allocator::alloc()    - L2 내부에서 writeback/fill용 mem_fetch 생성
 * memory_partition_unit::dram_cycle()     - DRAM 리턴큐·사이클·L2→DRAM 중재 통합 수행
 * memory_partition_unit::simple_dram_model_cycle() - 고정 지연 DRAM 대체 경로
 * memory_sub_partition::cache_cycle() - L2 fill 응답, DRAM→L2, L2 접근, ROP 큐 처리
 * memory_sub_partition::push()       - ICNT에서 도착한 요청을 ROP 큐 또는 텍스처 큐에 투입
 * memory_sub_partition::pop()        - L2_icnt_queue에서 완료 응답을 꺼내 SM에 반환
 * arbitration_metadata               - sub-partition 간 DRAM 크레딧 공유 중재 구조체
 */
// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
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

/* [한국어] C 표준 입출력: fprintf, gzprintf 등 파일 기반 출력에 사용 */
#include <stdio.h>
/* [한국어] C 표준 유틸리티: assert, malloc/free 등 메모리 관리에 사용 */
#include <stdlib.h>
/* [한국어] C 문자열 처리: snprintf(L2c_name 포맷), sscanf(큐 설정 파싱)에 사용 */
#include <string.h>

/* [한국어] STL 리스트: m_dram_latency_queue(dram_delay_t 링크드 리스트) 자료형 */
#include <list>
/* [한국어] STL 집합: m_request_tracker(미완료 mem_fetch* 추적용 set)에 사용 */
#include <set>

/* [한국어] GPU 워프·스레드 블록 추상 모델: mem_access_t, active_mask_t,
 *         mem_access_byte_mask_t, mem_access_sector_mask_t 등 기본 타입 정의 */
#include "../abstract_hardware_model.h"
/* [한국어] 설정 파일 파서: memory_config의 옵션을 gpgpusim.config에서 읽어옴 */
#include "../option_parser.h"
/* [한국어] 통계 래퍼 (gzip 압축 visualizer 출력): gzFile, gzprintf 등 제공 */
#include "../statwrapper.h"
/* [한국어] DRAM 타이밍 모델(dram_t): dram_cycle()에서 m_dram->cycle()·push()·return_queue_top() 호출 */
#include "dram.h"
/* [한국어] L2 캐시 클래스(l2_cache): access(), fill(), cycle(), flush() 등 캐시 API */
#include "gpu-cache.h"
/* [한국어] 최상위 시뮬레이션 루프(gpgpu_sim): gpu_sim_cycle, print_dram_stats() 등 */
#include "gpu-sim.h"
/* [한국어] 히스토그램 유틸리티: 지연 분포 기록에 사용 */
#include "histogram.h"
/* [한국어] 이 파일에서 구현하는 memory_partition_unit·memory_sub_partition 선언 헤더 */
#include "l2cache.h"
/* [한국어] L2 캐시 트레이스 매크로(MEMPART_DPRINTF, MEM_SUBPART_DPRINTF): 조건부 디버그 출력 */
#include "l2cache_trace.h"
/* [한국어] 메모리 요청 패킷(mem_fetch): 모든 큐를 흐르는 핵심 자료구조 */
#include "mem_fetch.h"
/* [한국어] 메모리 지연 통계(memory_stats_t): memlatstat_icnt2mem_pop() 등 지연 기록 API */
#include "mem_latency_stat.h"
/* [한국어] SM(Shader Core) 헤더: 요청 경로의 출처 식별자(sid, tpc 등) 타입 정의 */
#include "shader.h"

/*
 * [한국어]
 * partition_mf_allocator::alloc (단순 overload) — 쓰기 전용 writeback mem_fetch 생성
 *
 * @addr:     새로 생성할 mem_fetch의 대상 주소 (L2 evict 대상 라인 주소)
 * @type:     메모리 접근 유형 (L2_WRBK_ACC 등 writeback 전용 타입)
 * @size:     패킷 데이터 크기 (바이트)
 * @wr:       반드시 true여야 함 — 이 overload는 쓰기 전용 (assert(wr)로 보장)
 * @cycle:    생성 시각 (gpu_sim_cycle + gpu_tot_sim_cycle)
 * @streamID: 이 요청이 속한 CUDA 스트림 ID
 * @return:   새로 heap-할당된 mem_fetch*; 호출자(l2_cache write evict 경로)가 소유권을 가짐
 *
 * L2 캐시가 write-back 정책으로 dirty 라인을 축출(evict)할 때, 실제 메모리 요청 패킷이
 * 없으므로 이 팩토리가 합성 writeback mem_fetch를 만들어준다. active_mask 없이
 * 주소·타입·크기만으로 mem_access_t를 구성하는 단순 경로이다.
 * warp ID, shader ID, TPC 자리에 -1을 넘겨 "내부 생성"임을 표시한다.
 *
 * 호출 체인:
 *   l2_cache (gpu-cache.cc) → [이 함수] → new mem_fetch(...)
 */
mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr); /* [한국어] 이 overload는 쓰기(writeback)만 허용 — 읽기로 호출되면 버그 */
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  /* [한국어] mem_access_t 생성: active_mask·byte_mask 없는 단순 주소+타입+크기 조합.
   *         gpgpu_ctx는 PTX 컨텍스트 포인터(mem_access_t 내부 초기화에 필요) */
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  /* [한국어] mem_fetch 동적 할당:
   *   - NULL: 상위 인스트럭션 없음 (내부 생성 writeback이므로)
   *   - WRITE_PACKET_SIZE: 쓰기 패킷 고정 크기
   *   - -1, -1, -1: wid·sid·tpc 미지정 (L2 내부 생성) */
  return mf; /* [한국어] 호출자(l2_cache)에게 소유권 이전 */
}

/*
 * [한국어]
 * partition_mf_allocator::alloc (섹터 overload) — breakdown_request_to_sector_requests용 섹터 단위 mem_fetch 생성
 *
 * @addr:        섹터 시작 주소 (원본 mf 주소 + SECTOR_SIZE * i 오프셋 적용 후 값)
 * @type:        원본 요청의 접근 유형(mem_access_type) 그대로 상속
 * @active_mask: 원본 mf의 warp active mask (어떤 레인이 이 요청에 참여하는지)
 * @byte_mask:   원본 byte_mask & 해당 섹터 범위 mask (이 섹터에 해당하는 바이트만 선택)
 * @sector_mask: 이 섹터 인덱스 i만 set한 단일 비트 bitset<SECTOR_CHUNCK_SIZE>
 * @size:        SECTOR_SIZE (32바이트 고정)
 * @wr:          원본 요청의 쓰기 여부
 * @cycle:       생성 시각 (gpu_tot_sim_cycle + gpu_sim_cycle)
 * @wid:         원본 warp ID
 * @sid:         원본 shader(SM) ID
 * @tpc:         원본 TPC ID
 * @original_mf: 이 섹터 패킷을 낳은 부모 mem_fetch 포인터 (응답 시 부모 추적에 사용)
 * @streamID:    원본 CUDA 스트림 ID
 * @return:      새로 할당된 섹터 단위 mem_fetch*; breakdown_request_to_sector_requests가 소유
 *
 * SECTOR 캐시 타입에서 128바이트(MAX_MEMORY_ACCESS_SIZE) 요청을 32바이트(SECTOR_SIZE) 단위로
 * 분해할 때 각 섹터마다 이 함수가 호출된다. 읽기/쓰기 여부에 따라 패킷 크기를
 * READ_PACKET_SIZE 또는 WRITE_PACKET_SIZE로 결정하며, original_mf 링크를 통해
 * 섹터 완료 시 부모 요청 상태를 업데이트할 수 있다.
 *
 * 호출 체인:
 *   memory_sub_partition::breakdown_request_to_sector_requests() → [이 함수] → new mem_fetch(...)
 */
mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  /* [한국어] mem_access_t 구성: 주소·타입·크기에 active_mask·byte_mask·sector_mask를
   *         모두 포함하는 완전한 접근 기술자. SECTOR 캐시가 섹터 마스크를 사용해
   *         어떤 32바이트 슬롯이 유효한지 판단한다 */
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  /* [한국어] 섹터 mem_fetch 동적 할당:
   *   - wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE: 쓰기/읽기에 따라 패킷 헤더 크기 구분
   *   - wid/sid/tpc: 원본 warp·SM·TPC ID 그대로 전달 (통계 집계에 필요)
   *   - original_mf: 부모 연결 — 섹터 응답 집계 시 부모 mf 완료 처리에 사용 */
  return mf; /* [한국어] 섹터 mem_fetch 반환; breakdown 벡터에 push_back됨 */
}
/*
 * [한국어]
 * memory_partition_unit::memory_partition_unit - DRAM 채널 파티션 유닛 생성자
 *
 * @partition_id: 이 파티션의 전역 고유 ID (0 ~ m_n_mem-1)
 * @config:       메모리 서브시스템 설정 포인터 (파티션 수, 큐 크기 등 포함)
 * @stats:        전역 메모리 지연 통계 객체 (모든 sub_partition이 공유)
 * @gpu:          최상위 시뮬레이터 포인터 (현재 사이클 조회 등에 사용)
 * @return:       없음 (생성자)
 *
 * 하나의 DRAM 채널에 대응하는 파티션 유닛을 초기화한다. 이 파티션은
 * 1개의 dram_t 인스턴스(실제 DRAM 타이밍)와 m_n_sub_partition_per_memory_channel개의
 * memory_sub_partition(L2 슬라이스 + FIFO 큐 묶음)으로 구성된다.
 * sub_partition의 전역 ID는 partition_id * n_sub_partition + local_index로 계산되어
 * 전체 시스템에서 고유하게 식별된다.
 * arbitration_metadata는 이니셜라이저 리스트에서 config 기반으로 크레딧 한도를 설정한다.
 *
 * 호출 체인:
 *   gpgpu_sim 생성자(gpu-sim.cc) → [이 함수] → new dram_t, new memory_sub_partition
 */
memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),       /* [한국어] 이 DRAM 채널의 전역 파티션 인덱스 */
      m_config(config),         /* [한국어] 메모리 설정 포인터 — 파티션 내 모든 참조에 사용 */
      m_stats(stats),           /* [한국어] 전역 지연 통계 공유 포인터 */
      m_arbitration_metadata(config), /* [한국어] 크레딧 기반 중재 메타데이터 초기화 */
      m_gpu(gpu) {              /* [한국어] 최상위 시뮬레이터 참조 */
  m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);
  /* [한국어] DRAM 타이밍 모델 생성: m_id로 이 채널을 식별하며,
   *         m_stats를 통해 DRAM 접근 통계를 기록하고,
   *         this(memory_partition_unit*)로 완료 콜백 경로를 연결 */

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  /* [한국어] sub_partition 포인터 배열 동적 할당:
   *         크기 = m_n_sub_partition_per_memory_channel (통상 2 또는 4) */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    /* [한국어] 전역 sub_partition ID 계산:
     *         파티션 0의 sub-partition 0·1, 파티션 1의 sub-partition 2·3, ...
     *         이 ID로 dram_cycle()에서 return 목적지를 역산 */
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
    /* [한국어] 각 L2 슬라이스 생성: FIFO 큐 4개, L2 캐시 인스턴스, ROP 큐 포함 */
  }
}

/*
 * [한국어]
 * memory_partition_unit::handle_memcpy_to_gpu - CUDA memcpy 엔진의 L2 태그 강제 업데이트 처리
 *
 * @addr:              복사 대상 GPU 주소 (호스트→디바이스 memcpy 목적지)
 * @global_subpart_id: 이 주소가 매핑된 전역 sub_partition ID
 * @mask:              유효 섹터를 나타내는 비트마스크 (어떤 32바이트 슬롯이 복사됐는지)
 * @return:            없음
 *
 * cuMemcpy 같은 비동기 복사 엔진이 GPU 메모리를 직접 갱신할 때, L2 캐시 내의 해당
 * 라인 태그를 강제로 유효(valid) 상태로 만들어야 한다. 그렇지 않으면 이후 SM이
 * 같은 주소를 읽을 때 outdated 캐시를 보게 된다. 이 함수는 전역 ID를 로컬 ID로
 * 변환한 후 sub_partition의 force_l2_tag_update()로 위임한다.
 * 실행 컨텍스트: 시뮬레이터 제어 경로 (사이클 루프 외부에서 호출될 수 있음).
 *
 * 호출 체인:
 *   gpgpu_sim (memcpy 처리 경로) → [이 함수] → m_sub_partition[p]->force_l2_tag_update()
 */
void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  /* [한국어] 전역 sub_partition ID를 이 파티션 내 로컬 인덱스(0-based)로 변환 */
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  /* [한국어] 섹터 마스크를 이진 문자열로 변환 — 디버그 출력용 */
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  /* [한국어] 조건부 디버그 로그: MEMPART_DPRINTF는 l2cache_trace.h에서 매크로 정의됨 */
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
  /* [한국어] 해당 L2 슬라이스에 태그 강제 유효화 요청:
   *         현재 사이클을 타임스탬프로 전달하여 fill 이벤트로 처리 */
}

/*
 * [한국어]
 * memory_partition_unit::~memory_partition_unit - DRAM 채널 파티션 유닛 소멸자
 *
 * 시뮬레이션 종료 시 이 파티션에 속한 모든 동적 자원을 해제한다.
 * dram_t, 각 memory_sub_partition 인스턴스, 그리고 서브파티션 포인터 배열 자체를 삭제한다.
 * 호출 순서: sub_partition들 먼저 삭제 → 포인터 배열 삭제 → dram_t 삭제.
 */
memory_partition_unit::~memory_partition_unit() {
  delete m_dram; /* [한국어] DRAM 타이밍 모델 해제 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p]; /* [한국어] 각 L2 슬라이스(FIFO 큐·L2 캐시·ROP 큐 포함) 해제 */
  }
  delete[] m_sub_partition; /* [한국어] 서브파티션 포인터 배열 자체 해제 */
}

/*
 * [한국어]
 * arbitration_metadata::arbitration_metadata - DRAM 요청 중재 메타데이터 초기화
 *
 * @config: 메모리 설정 포인터 (sub_partition 수, DRAM 스케줄러 큐 크기 등 참조)
 * @return: 없음 (생성자)
 *
 * sub_partition 간의 DRAM 접근을 공정하게 중재하기 위한 크레딧 시스템을 초기화한다.
 * 각 sub_partition은 독점 크레딧(private: 한도 1)을 가지며, 이를 다 쓰면
 * 공유 크레딧 풀(shared)을 빌린다. 공유 한도는 FRFCFS 스케줄러 큐와 리턴 큐의
 * 총 용량에서 private 전용분을 제외하여 결정된다. 큐 크기가 0이면 무제한 허용.
 *
 * 크레딧 한도 계산:
 *   shared_credit_limit = frfcfs_queue_size + return_queue_size - (n_sub_partitions - 1)
 *   (쓰기 전용 큐 활성 시 += write_queue_size)
 *
 * 호출 체인:
 *   memory_partition_unit 생성자 이니셜라이저 리스트 → [이 함수]
 */
memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
    /* [한국어] m_last_borrower 초기값 = 마지막 sub_partition 인덱스:
     *         라운드로빈 시작 시 0번부터 탐색하기 위한 초기 오프셋 */
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
    /* [한국어] private_credit 벡터: n_sub_partition 크기로 초기화, 모두 0 */
      m_shared_credit(0) { /* [한국어] 공유 크레딧 사용량 0으로 초기화 */
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  /* [한국어] 각 sub_partition의 독점 크레딧 한도 = 1:
   *         기아(starvation) 방지를 위해 최소 1개는 독점 보장 */
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  /* [한국어] 공유 크레딧 한도 계산:
   *         총 DRAM 큐 용량 - private 전용분(n-1개) = 공유 가능한 슬롯 수.
   *         이 값을 초과하는 요청은 DRAM이 수용 불가하므로 발행 차단 */
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  /* [한국어] 분리 쓰기 큐 활성 시 쓰기 큐 크기만큼 공유 한도 추가:
   *         FRFCFS 분리 읽기/쓰기 스케줄러를 사용할 때만 적용 */
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
    /* [한국어] 큐 크기가 0(무제한)이면 공유 크레딧 한도도 0(무제한)으로 설정:
     *         has_credits()에서 m_shared_credit_limit == 0이면 항상 허용 */
  }
  assert(m_shared_credit_limit >= 0);
  /* [한국어] 크레딧 한도 음수 방지 검증 — 설정값 오류 조기 탐지 */
}

/*
 * [한국어]
 * arbitration_metadata::has_credits - 특정 sub_partition의 DRAM 발행 가능 여부 조회
 *
 * @inner_sub_partition_id: 이 파티션 내의 로컬 sub_partition 인덱스
 * @return: true = 크레딧 여유 있음 (DRAM 요청 발행 가능),
 *          false = 크레딧 고갈 (DRAM 큐가 이미 이 파티션 요청으로 포화)
 *
 * 우선 private 크레딧을 확인하고, private가 한도(1)에 달했으면 공유 풀을 확인한다.
 * 공유 한도가 0이면 무제한(항상 허용)이다. can_issue_to_dram()에서 호출된다.
 *
 * 호출 체인:
 *   memory_partition_unit::can_issue_to_dram() → [이 함수]
 */
bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id; /* [한국어] 로컬 sub_partition 인덱스 별칭 */
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true; /* [한국어] private 크레딧 미소진 → 즉시 발행 허용 */
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true; /* [한국어] shared_credit_limit==0(무제한) 또는 공유 크레딧 여유 있음 */
  } else {
    return false; /* [한국어] private·shared 모두 한도 도달 → 발행 차단 */
  }
}

/*
 * [한국어]
 * arbitration_metadata::borrow_credit - DRAM 발행 시 크레딧 소비 기록
 *
 * @inner_sub_partition_id: 크레딧을 소비하는 sub_partition의 로컬 인덱스
 * @return: 없음
 *
 * sub_partition이 DRAM에 요청을 발행할 때 호출되어 크레딧을 차감한다.
 * private 크레딧이 한도 미만이면 private에서 소비, 그렇지 않으면 shared에서 소비.
 * 둘 다 고갈된 상태에서 호출하면 assert로 버그를 탐지한다.
 * m_last_borrower를 갱신하여 다음 사이클의 라운드로빈 시작점을 결정한다.
 *
 * 호출 체인:
 *   simple_dram_model_cycle() / dram_cycle() → [이 함수] (L2→DRAM 발행 시)
 */
void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id; /* [한국어] 로컬 인덱스 별칭 */
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1; /* [한국어] private 크레딧 소비: private 슬롯 우선 사용 */
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1; /* [한국어] shared 크레딧 소비: 공유 풀에서 슬롯 점유 */
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
    /* [한국어] has_credits() 없이 borrow_credit() 호출 = 로직 버그 */
  }
  m_last_borrower = spid;
  /* [한국어] 마지막 발행 서브파티션 갱신 — 다음 사이클 라운드로빈의 시작점이 됨 */
}

/*
 * [한국어]
 * arbitration_metadata::return_credit - DRAM 응답 수신 시 크레딧 반환
 *
 * @inner_sub_partition_id: 응답을 받은 sub_partition의 로컬 인덱스
 * @return: 없음
 *
 * DRAM 완료 패킷이 sub_partition의 dram_L2_queue로 돌아올 때 호출되어 크레딧을 반환한다.
 * private 크레딧을 먼저 복원하고, private가 0이면 shared 크레딧을 반환한다.
 * shared가 음수가 되면 assert로 과반환 버그를 탐지한다.
 *
 * 호출 체인:
 *   simple_dram_model_cycle() / dram_cycle() → [이 함수] (DRAM→L2 리턴 시)
 *   memory_partition_unit::set_done() → [이 함수] (writeback 완료 시)
 */
void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id; /* [한국어] 로컬 인덱스 별칭 */
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1; /* [한국어] private 크레딧 반환 우선 */
  } else {
    m_shared_credit -= 1; /* [한국어] private가 0이면 공유 풀로 반환 */
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
  /* [한국어] shared_credit 음수 방지: 크레딧보다 더 많이 반환하면 버그 */
}

/*
 * [한국어]
 * arbitration_metadata::print - 크레딧 상태를 파일에 출력 (디버그용)
 *
 * @fp: 출력 대상 파일 포인터 (통상 stdout 또는 시뮬레이터 로그 파일)
 * @return: 없음
 *
 * 각 sub_partition의 private 크레딧 사용량과 한도, 그리고 공유 크레딧 상태를 출력.
 * memory_partition_unit::print()에서 호출되어 전체 파티션 상태 덤프의 일부로 사용.
 */
void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = "); /* [한국어] private 크레딧 헤더 출력 */
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]); /* [한국어] 각 sub_partition별 크레딧 사용량 */
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  /* [한국어] private 크레딧 한도 (항상 1) */
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
  /* [한국어] 현재 공유 크레딧 사용량과 한도 출력 */
}

/*
 * [한국어]
 * memory_partition_unit::busy - 이 DRAM 파티션에 미완료 요청이 있는지 조회
 *
 * @return: true = 하나 이상의 sub_partition에서 아직 완료되지 않은 요청 존재,
 *          false = 모든 sub_partition의 request_tracker가 비어 있음
 *
 * 커널 완료 여부 판단 시 gpgpu_sim이 모든 메모리 파티션의 busy() 상태를 확인한다.
 * 하나라도 busy이면 커널이 아직 종료되지 않은 것으로 간주한다.
 *
 * 호출 체인:
 *   gpgpu_sim::active() → [이 함수] → m_sub_partition[p]->busy()
 */
bool memory_partition_unit::busy() const {
  bool busy = false; /* [한국어] 기본 상태: 유휴 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true; /* [한국어] 하나라도 미완료 요청이 있으면 busy로 판정 */
    }
  }
  return busy;
}

/*
 * [한국어]
 * memory_partition_unit::cache_cycle - 이 파티션의 모든 L2 슬라이스를 1사이클 진행
 *
 * @cycle: 현재 시뮬레이션 사이클 번호 (ROP 지연 큐의 ready_cycle 비교에 사용)
 * @return: 없음
 *
 * gpgpu_sim::cycle()이 매 사이클마다 호출하며, 각 sub_partition의 cache_cycle()을
 * 순서대로 실행한다. L2 fill 응답, DRAM→L2 큐 처리, L2 접근, ROP 큐 등 전체
 * L2-side 로직을 구동한다. DRAM 타이밍 모델은 별도로 dram_cycle()에서 처리된다.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → [이 함수] → m_sub_partition[p]->cache_cycle()
 */
void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle); /* [한국어] 각 L2 슬라이스 1사이클 진행 */
  }
}

/*
 * [한국어]
 * memory_partition_unit::visualizer_print - AerialVision 시각화 데이터 출력
 *
 * @visualizer_file: gzip 압축 visualizer 출력 파일 핸들
 * @return: 없음
 *
 * AerialVision GPU 성능 가시화 도구를 위해 이 파티션의 DRAM 통계와 각 L2 슬라이스의
 * per-window 통계를 gzip 파일에 기록한다. gpgpu_sim이 일정 사이클 주기마다 호출한다.
 *
 * 호출 체인:
 *   gpgpu_sim::visualizer_print() → [이 함수] → m_dram/m_sub_partition visualizer_print()
 */
void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  /* [한국어] DRAM 접근 통계(활성화, 사전충전, 읽기/쓰기 명령 수 등)를 visualizer에 기록 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
    /* [한국어] 각 L2 슬라이스의 read_hit/miss, write_hit/miss 등 per-window 통계 기록 */
  }
}

/*
 * [한국어]
 * memory_partition_unit::can_issue_to_dram - 특정 sub_partition이 DRAM에 요청 발행 가능한지 판단
 *
 * @inner_sub_partition_id: 발행 여부를 확인할 sub_partition의 로컬 인덱스
 * @return: true = DRAM 발행 가능 (크레딧 있고 dram_L2 큐에 여유 있음),
 *          false = 발행 불가 (크레딧 고갈 또는 dram_L2 큐 포화)
 *
 * DRAM 발행 가능 조건을 두 가지로 확인한다:
 * 1) 크레딧 보유: DRAM 큐에 더 밀어 넣을 공간이 예약되어 있는가
 * 2) dram_L2_queue 미포화: DRAM 완료 응답을 받을 공간이 있는가
 * simple_dram_model_cycle()과 dram_cycle()의 L2→DRAM 발행 루프에서 매 사이클 호출된다.
 *
 * 호출 체인:
 *   simple_dram_model_cycle() / dram_cycle() → [이 함수] → has_credits(), dram_L2_queue_full()
 */
// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id; /* [한국어] 로컬 인덱스 별칭 */
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  /* [한국어] dram_L2_queue 포화 여부: 리턴 큐가 가득 차면 추가 발행 불가 */
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);
  /* [한국어] 크레딧 보유 여부: private 또는 shared 크레딧이 남아 있으면 true */

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');
  /* [한국어] 디버그 로그: 발행 판단 근거 출력 (MEMPART_DPRINTF = 조건부 fprintf) */

  return (has_dram_resource && !sub_partition_contention);
  /* [한국어] 크레딧 있고(has_dram_resource) AND dram_L2 큐 여유 있을 때(not contention)만 true */
}

/*
 * [한국어]
 * memory_partition_unit::global_sub_partition_id_to_local_id - 전역 sub_partition ID를 로컬 인덱스로 변환
 *
 * @global_sub_partition_id: 시뮬레이터 전체에서 고유한 sub_partition 인덱스
 * @return: 이 파티션 내의 로컬 인덱스 (0 ~ m_n_sub_partition_per_memory_channel-1)
 *
 * 전역 ID = partition_id * n_sub_per_channel + local_id이므로,
 * local_id = global_id - partition_id * n_sub_per_channel으로 역산한다.
 * DRAM 리턴 패킷의 목적지를 이 파티션 내 배열 인덱스로 매핑할 때 사용한다.
 *
 * 호출 체인:
 *   dram_cycle() / simple_dram_model_cycle() / handle_memcpy_to_gpu() → [이 함수]
 */
int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
  /* [한국어] global - (partition_id * n_sub) = 이 파티션 내 로컬 인덱스 */
}

/*
 * [한국어]
 * memory_partition_unit::simple_dram_model_cycle - 단순 고정 지연 DRAM 모델 1사이클 진행
 *
 * @return: 없음
 *
 * gpgpusim.config에서 -gpgpu_dram_model simple 옵션 선택 시 실제 FRFCFS 스케줄러 대신
 * 이 함수가 사용된다. m_dram_latency_queue(dram_delay_t 링크드 리스트)에 고정 지연
 * (m_config->dram_latency 사이클)으로 요청을 집어넣고, ready_cycle에 도달하면 꺼내어
 * 해당 sub_partition의 dram_L2_queue로 반환한다. 실제 DRAM 타이밍 없이 지연만 모델링.
 *
 * 두 단계로 동작:
 * 1단계 (DRAM→L2): m_dram_latency_queue 앞쪽 원소가 준비됐으면 dram_L2_queue로 이동.
 *                   writeback(L1/L2_WRBK_ACC)이면 set_done 후 삭제.
 * 2단계 (L2→DRAM): 라운드로빈으로 sub_partition을 탐색, 발행 가능하면 L2_dram_queue에서
 *                   꺼내 dram_latency_queue에 삽입. 1사이클 1요청 제한(break).
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → [이 함수] (simple DRAM 모드 시)
 */
void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    /* [한국어] 큐 앞쪽 요청이 준비 사이클에 도달했는지 확인:
     *         gpu_sim_cycle + gpu_tot_sim_cycle = 절대 사이클 번호 */
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    /* [한국어] 준비된 DRAM 응답 패킷 참조 */
    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      /* [한국어] 일반 읽기/쓰기 요청 경로 (writeback이 아닌 경우) */
      mf_return->set_reply();
      /* [한국어] mem_fetch를 '응답 패킷'으로 변환: 방향이 DRAM→SM으로 바뀜 */

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      /* [한국어] 이 요청이 원래 발행된 sub_partition의 전역 ID 조회 */
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      /* [한국어] 전역 ID → 이 파티션 내 로컬 인덱스 변환 */
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      /* [한국어] 변환 결과 일관성 검증 */
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        /* [한국어] 목적지 sub_partition의 dram_L2_queue에 여유가 있을 때만 처리 */
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          /* [한국어] L1 writeback: L2에 넣지 않고 완료 처리 후 삭제 */
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          /* [한국어] dram_L2_queue에 응답 삽입: 다음 cache_cycle()에서 L2로 fill됨 */
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          /* [한국어] mem_fetch 상태 갱신: 지연 통계를 위한 큐 진입 타임스탬프 기록 */
          m_arbitration_metadata.return_credit(dest_spid);
          /* [한국어] DRAM 응답이 왔으므로 이전에 빌린 크레딧을 반환 */
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
        /* [한국어] 처리 완료한 원소를 지연 큐에서 제거 */
      }

    } else {
      /* [한국어] L1/L2 writeback 요청이 DRAM 완료: L2에 돌려보내지 않고 폐기 */
      this->set_done(mf_return);
      /* [한국어] 파티션 레벨 set_done: writeback의 크레딧도 반환 */
      delete mf_return; /* [한국어] writeback 패킷 메모리 해제 */
      m_dram_latency_queue.pop_front(); /* [한국어] 지연 큐에서 제거 */
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  /* [한국어] 마지막으로 발행했던 sub_partition 인덱스: 라운드로빈 시작점 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    /* [한국어] 라운드로빈: last_borrower+1부터 시작하여 모든 파티션을 순환 */
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      /* [한국어] 해당 sub_partition의 L2→DRAM 큐에 요청이 있고, 발행 가능한 경우 */
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      /* [한국어] 발행할 요청을 큐 앞에서 참조 (아직 pop하지 않음) */
      if (m_dram->full(mf->is_write())) break;
      /* [한국어] simple_dram에서도 dram_t 포화 여부 확인 (통계 목적으로 dram_t 유지) */

      m_sub_partition[spid]->L2_dram_queue_pop();
      /* [한국어] L2→DRAM 큐에서 실제로 꺼냄 */
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf; /* [한국어] 지연 큐 원소에 요청 포인터 저장 */
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      /* [한국어] 준비 시각 = 현재 사이클 + dram_latency(gpgpusim.config 설정값) */
      m_dram_latency_queue.push_back(d);
      /* [한국어] 고정 지연 큐 뒤에 삽입: FIFO 순서로 처리됨 */
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      /* [한국어] 지연 통계용 상태 갱신: DRAM 지연 큐 진입 타임스탬프 */
      m_arbitration_metadata.borrow_credit(spid);
      /* [한국어] 이 sub_partition의 크레딧 소비: 응답이 돌아올 때 return_credit */
      break;  // the DRAM should only accept one request per cycle
      /* [한국어] 1사이클 1요청 제한: break로 라운드로빈 루프 종료 */
    }
  }
  //}
}

/*
 * [한국어]
 * memory_partition_unit::dram_cycle - 실제 DRAM 타이밍 모델 1사이클 통합 진행
 *
 * @return: 없음
 *
 * FRFCFS(First-Ready First-Come-First-Served) 스케줄러를 포함한 dram_t 타이밍 모델을
 * 사용하는 정상 경로이다. 한 사이클에 네 가지 작업을 순서대로 수행한다:
 *
 * 1) DRAM 리턴큐 → dram_L2_queue: m_dram->return_queue_top()에서 완료 응답을 꺼내
 *    목적지 sub_partition에 전달. writeback이면 set_done+삭제.
 * 2) DRAM 1사이클 진행: m_dram->cycle() — 내부 FRFCFS 스케줄러, bank 상태 머신 갱신.
 *    dram_log(SAMPLELOG)로 파형 기록.
 * 3) L2_dram_queue → m_dram_latency_queue: 라운드로빈으로 sub_partition을 선택,
 *    발행 가능하면 고정 지연 큐에 삽입 (L2→DRAM 경계 지연 모델링).
 * 4) m_dram_latency_queue → m_dram->push(): 준비된 요청이 있고 DRAM이 수용 가능하면
 *    실제 DRAM 스케줄러에 투입.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → [이 함수] → m_dram->cycle(), m_dram->push(), return_queue_top/pop()
 */
void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  mem_fetch *mf_return = m_dram->return_queue_top();
  /* [한국어] DRAM 스케줄러가 완료한 응답 패킷을 리턴큐 앞에서 참조 */
  if (mf_return) {
    /* [한국어] 리턴큐에 완료 응답이 있는 경우 */
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    /* [한국어] 이 응답을 받아야 할 sub_partition의 전역 ID */
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    /* [한국어] 전역 ID → 로컬 인덱스 변환 */
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    /* [한국어] 변환 일관성 검증 */
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      /* [한국어] dram_L2_queue에 여유가 있을 때만 응답 처리 (없으면 다음 사이클로 대기) */
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        /* [한국어] L1 writeback 완료: L2에 전달 불필요, 즉시 완료 처리 */
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return; /* [한국어] L1 writeback 패킷 해제 */
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        /* [한국어] 일반 응답을 dram_L2_queue에 삽입: cache_cycle()에서 L2 fill 처리 */
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] 큐 진입 타임스탬프 기록 (지연 통계용) */
        m_arbitration_metadata.return_credit(dest_spid);
        /* [한국어] 크레딧 반환: 이 응답으로 DRAM 큐 슬롯 1개 해제됨 */
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
      /* [한국어] DRAM 리턴큐에서 처리 완료한 원소 제거 */
    }
  } else {
    m_dram->return_queue_pop();
    /* [한국어] NULL이지만 pop은 호출: dram_t 내부 상태 진전을 위해 필요 */
  }

  m_dram->cycle();
  /* [한국어] DRAM 타이밍 모델 1사이클 진행:
   *         FRFCFS 스케줄러가 bank 상태에 따라 ACT/RD/WR/PRE 명령을 발행 */
  m_dram->dram_log(SAMPLELOG);
  /* [한국어] DRAM 파형 로그 기록: SAMPLELOG 모드로 주기적 샘플링 */

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  /* [한국어] 직전 사이클에 마지막으로 발행한 sub_partition: 라운드로빈 출발점 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    /* [한국어] last_borrower+1부터 순환하는 라운드로빈 인덱스 */
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      /* [한국어] 발행 대기 요청이 있고 크레딧·dram_L2 큐 여유가 있을 때 */
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      /* [한국어] 발행할 요청 참조 (peek) */
      if (m_dram->full(mf->is_write())) break;
      /* [한국어] DRAM 스케줄러 큐가 포화이면 발행 포기 */

      m_sub_partition[spid]->L2_dram_queue_pop();
      /* [한국어] L2→DRAM 큐에서 실제 꺼냄 */
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf; /* [한국어] 지연 큐 원소에 요청 저장 */
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      /* [한국어] L2→DRAM 경계 지연: 현재 사이클 + dram_latency 후 실제 DRAM에 투입 */
      m_dram_latency_queue.push_back(d);
      /* [한국어] 고정 지연 큐 뒤에 삽입 */
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      /* [한국어] 상태/타임스탬프 갱신 */
      m_arbitration_metadata.borrow_credit(spid);
      /* [한국어] 크레딧 소비: 응답이 돌아오면 return_credit */
      break;  // the DRAM should only accept one request per cycle
      /* [한국어] 1사이클 1발행 제한 */
    }
  }
  //}

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    /* [한국어] 지연 큐 앞쪽 원소가 준비됐고 DRAM이 수용 가능할 때 DRAM에 투입 */
    mem_fetch *mf = m_dram_latency_queue.front().req;
    /* [한국어] 준비된 요청 참조 */
    m_dram_latency_queue.pop_front();
    /* [한국어] 지연 큐에서 제거 */
    m_dram->push(mf);
    /* [한국어] FRFCFS 스케줄러에 요청 투입: 이후 dram_t가 bank 상태에 따라 스케줄링 */
  }
}

/*
 * [한국어]
 * memory_partition_unit::set_done - 파티션 수준에서 요청 완료 처리
 *
 * @mf: 완료 처리할 mem_fetch 포인터
 * @return: 없음
 *
 * writeback(L1_WRBK_ACC / L2_WRBK_ACC) 유형의 요청이 DRAM에서 완료될 때 크레딧을 반환하고,
 * 해당 sub_partition의 request_tracker에서 제거한다. simple_dram_model_cycle()에서
 * writeback 폐기 경로에 사용된다.
 *
 * 호출 체인:
 *   simple_dram_model_cycle() / memory_sub_partition::pop() → [이 함수]
 *   → m_arbitration_metadata.return_credit(), m_sub_partition[spid]->set_done()
 */
void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  /* [한국어] 이 요청이 속한 sub_partition의 전역 ID */
  int spid = global_sub_partition_id_to_local_id(global_spid);
  /* [한국어] 전역 ID → 로컬 인덱스 변환 */
  assert(m_sub_partition[spid]->get_id() == global_spid);
  /* [한국어] 변환 일관성 검증 */
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    /* [한국어] writeback 유형: DRAM에 데이터가 쓰여지는 것이므로 크레딧 반환 필요 */
    m_arbitration_metadata.return_credit(spid);
    /* [한국어] writeback이 DRAM 큐에서 완료됐으므로 점유 크레딧 해제 */
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
  /* [한국어] request_tracker에서 이 요청 제거: busy() 상태 갱신 */
}

/*
 * [한국어]
 * memory_partition_unit::set_dram_power_stats - AccelWattch 전력 통계 수집용 DRAM 카운터 조회
 *
 * @n_cmd, @n_activity, @n_nop: DRAM 명령/활성/NOP 카운터 (출력 파라미터)
 * @n_act, @n_pre, @n_rd, @n_wr, @n_wr_WB: 활성화/프리차지/읽기/쓰기 카운터 (출력)
 * @n_req: 총 DRAM 요청 수 (출력)
 * @return: 없음 (모두 레퍼런스 출력 파라미터)
 *
 * power_stat_t::save_stats()에서 AccelWattch 전력 계산에 사용할 DRAM 통계를 수집할 때
 * 이 함수가 호출되며, 실제 수집은 dram_t::set_dram_power_stats()에 위임한다.
 *
 * 호출 체인:
 *   power_stat_t::save_stats() / gpgpu_sim::print_dram_stats() → [이 함수] → m_dram->set_dram_power_stats()
 */
void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
  /* [한국어] DRAM 타이밍 모델에서 누적 카운터를 읽어 출력 파라미터에 저장 */
}

/*
 * [한국어]
 * memory_partition_unit::print - 이 파티션의 전체 상태를 파일에 덤프 (디버그용)
 *
 * @fp: 출력 대상 파일 포인터 (통상 stdout)
 * @return: 없음
 *
 * 시뮬레이션 중 메모리 시스템 상태를 진단할 때 gpgpu_sim::print_GPU_stats()에서 호출된다.
 * 각 sub_partition의 미완료 요청 목록, dram_latency_queue 내용, DRAM 내부 상태를 출력한다.
 *
 * 호출 체인:
 *   gpgpu_sim::print_GPU_stats() → [이 함수] → sub_partition->print(), m_dram->print()
 */
void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  /* [한국어] 이 파티션의 ID 헤더 출력 */
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
    /* [한국어] 각 L2 슬라이스의 미완료 요청 목록과 L2 캐시 상태 출력 */
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  /* [한국어] L2→DRAM 경계 고정 지연 큐의 원소 수 출력 */
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    /* [한국어] 지연 큐를 순회하며 각 원소의 준비 사이클과 요청 정보 출력 */
    mem_fetch *mf = mf_dlq->req; /* [한국어] 이 지연 슬롯의 요청 패킷 */
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    /* [한국어] 이 요청이 DRAM에 투입될 사이클 번호 출력 */
    if (mf)
      mf->print(fp); /* [한국어] 요청 상세 정보 출력 */
    else
      fprintf(fp, " <NULL mem_fetch?>\n"); /* [한국어] NULL 패킷 이상 감지 */
  }
  m_dram->print(fp);
  /* [한국어] DRAM 타이밍 모델(bank 상태, 스케줄러 큐, 전력 통계 등) 출력 */
}

/*
 * [한국어]
 * memory_sub_partition::memory_sub_partition - L2 캐시 슬라이스 + FIFO 큐 묶음 생성자
 *
 * @sub_partition_id: 이 슬라이스의 전역 고유 ID (0 ~ m_n_mem_sub_partition-1)
 * @config:           메모리 설정 포인터 (큐 크기, L2 캐시 설정 등)
 * @stats:            전역 메모리 지연 통계 공유 포인터
 * @gpu:              최상위 시뮬레이터 포인터 (현재 사이클, memcpy 오프셋 등)
 * @return:           없음 (생성자)
 *
 * 하나의 L2 캐시 슬라이스(bank)와 이를 둘러싼 네 개의 FIFO 파이프라인 큐,
 * ROP 지연 큐, L2interface 어댑터, mem_fetch 팩토리를 초기화한다.
 * 큐 크기는 gpgpusim.config의 -gpgpu_L2_queue_config "icnt_L2:L2_dram:dram_L2:L2_icnt"
 * 형식 문자열에서 파싱된다.
 * L2 캐시가 비활성(disabled)이면 m_L2cache 생성을 건너뛴다.
 *
 * 호출 체인:
 *   memory_partition_unit 생성자 → [이 함수] → new l2_cache, new fifo_pipeline×4
 */
memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;    /* [한국어] 전역 L2 슬라이스 ID 저장 */
  m_config = config;          /* [한국어] 메모리 설정 포인터 저장 */
  m_stats = stats;            /* [한국어] 전역 통계 포인터 저장 */
  m_gpu = gpu;                /* [한국어] 시뮬레이터 포인터 저장 */
  m_memcpy_cycle_offset = 0;  /* [한국어] memcpy 사이클 오프셋 초기값 0: force_l2_tag_update 시 사용 */

  assert(m_id < m_config->m_n_mem_sub_partition);
  /* [한국어] ID가 설정된 최대 슬라이스 수를 초과하지 않는지 검증 */

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  /* [한국어] L2 캐시 인스턴스 이름 생성: "L2_bank_000", "L2_bank_001" 등
   *         통계 출력 시 슬라이스를 식별하는 데 사용 */
  m_L2interface = new L2interface(this);
  /* [한국어] L2interface 어댑터 생성: l2_cache가 미스 패킷을 m_L2_dram_queue에 넣을 때 사용 */
  m_mf_allocator = new partition_mf_allocator(config);
  /* [한국어] mem_fetch 팩토리 생성: L2 evict/writeback 시 합성 패킷 생성에 사용 */

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE);
  /* [한국어] L2 캐시 활성 시 l2_cache 인스턴스 생성:
   *   - -1, -1: shader ID·core ID (파티션 수준이므로 미해당)
   *   - m_L2interface: 미스 패킷을 L2_dram_queue로 보내는 어댑터
   *   - IN_PARTITION_L2_MISS_QUEUE: 미스 상태 코드 (통계용)
   *   - L2_GPU_CACHE: L2 캐시 타입 식별자 */

  unsigned int icnt_L2;   /* [한국어] ICNT→L2 큐 최대 크기 */
  unsigned int L2_dram;   /* [한국어] L2→DRAM 큐 최대 크기 */
  unsigned int dram_L2;   /* [한국어] DRAM→L2 큐 최대 크기 */
  unsigned int L2_icnt;   /* [한국어] L2→ICNT 큐 최대 크기 */
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  /* [한국어] gpgpusim.config의 -gpgpu_L2_queue_config 값을 파싱:
   *         형식: "icnt_L2:L2_dram:dram_L2:L2_icnt" (예: "8:4:4:8") */
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  /* [한국어] ICNT→L2 입력 큐: ROP 완료 후 요청이 진입, L2 접근 전 단계 */
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  /* [한국어] L2 미스 출력 큐: L2 미스 패킷이 DRAM으로 내려가는 경로 */
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  /* [한국어] DRAM 완료 응답 수신 큐: DRAM→L2 fill 또는 icnt 반환 대기 */
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  /* [한국어] L2→ICNT 출력 큐: L2 HIT 응답 또는 DRAM 완료 응답이 SM으로 나가는 경로 */
  wb_addr = -1;
  /* [한국어] writeback 주소 초기화: -1은 유효한 writeback 없음을 의미 */
}

/*
 * [한국어]
 * memory_sub_partition::~memory_sub_partition - L2 슬라이스 소멸자
 *
 * 이 슬라이스에 속한 FIFO 큐 4개, L2 캐시 인스턴스, L2interface 어댑터를 해제한다.
 * m_mf_allocator는 여기서 삭제하지 않음에 주의 (생성자에서 생성하지만 소멸자에서
 * 명시적으로 삭제되지 않는 메모리 누수 가능성 — 시뮬레이터 종료 시 OS가 회수).
 */
memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;   /* [한국어] ICNT→L2 입력 큐 해제 */
  delete m_L2_dram_queue;   /* [한국어] L2→DRAM 미스 큐 해제 */
  delete m_dram_L2_queue;   /* [한국어] DRAM→L2 응답 큐 해제 */
  delete m_L2_icnt_queue;   /* [한국어] L2→ICNT 출력 큐 해제 */
  delete m_L2cache;         /* [한국어] L2 캐시 인스턴스 해제 (disabled이면 nullptr 해제 → 안전) */
  delete m_L2interface;     /* [한국어] L2interface 어댑터 해제 */
}

/*
 * [한국어]
 * memory_sub_partition::cache_cycle - L2 슬라이스의 1사이클 전체 파이프라인 진행
 *
 * @cycle: 현재 시뮬레이션 사이클 번호 (ROP 큐의 ready_cycle 비교에 사용)
 * @return: 없음
 *
 * 이 함수는 한 L2 슬라이스의 모든 큐와 캐시를 1사이클 진행시키는 핵심 루프이다.
 * 순서가 중요하며, 다음 5단계로 동작한다:
 *
 * 1단계) L2 fill 응답 처리:
 *   이전 사이클의 L2 miss가 DRAM에서 돌아와 fill이 완료된 경우, m_L2cache->next_access()로
 *   응답을 꺼내 L2_icnt_queue로 보낸다. write-allocate 읽기(L2_WR_ALLOC_R)는 상위로
 *   전달하지 않고 원본 쓰기 mf를 대신 전달(FETCH_ON_WRITE 정책).
 *
 * 2단계) DRAM→L2 fill 또는 SM 직접 반환:
 *   dram_L2_queue의 응답을 처리한다. L2가 해당 라인을 기다리면 fill() 호출,
 *   그렇지 않으면 (텍스처 아닌 경우 등) L2_icnt_queue로 직접 전달.
 *
 * 3단계) L2 내부 사이클 진행:
 *   m_L2cache->cycle() — L2 미스 큐에서 새 MSHR 슬롯을 배정하고 L2_dram_queue에 삽입.
 *
 * 4단계) 새 요청 L2 접근:
 *   icnt_L2_queue 앞쪽 요청을 L2에 접근시킨다. HIT이면 즉시 응답, MISS면 L2가
 *   내부적으로 MSHR 등록 후 L2_dram_queue로 내려보냄(3단계에서 처리됨).
 *   RESERVATION_FAIL이면 pop하지 않고 다음 사이클에 재시도.
 *   write-alloc 정책(FETCH_ON_WRITE/LAZY_FETCH_ON_READ)에 따라 쓰기 응답 처리 분기.
 *
 * 5단계) ROP 지연 큐 처리:
 *   쓰기 요청은 push() 시점에 m_rop 큐로 진입하며, ready_cycle 도달 시 icnt_L2_queue로 이동.
 *   텍스처 요청은 ROP를 거치지 않고 push() 때 바로 icnt_L2_queue로 진입.
 *
 * 실행 컨텍스트: 단일 스레드 시뮬레이션 루프(사이클 당 1회 호출).
 *
 * 호출 체인:
 *   memory_partition_unit::cache_cycle() → [이 함수]
 *     → m_L2cache->access/fill/cycle/next_access
 */
void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  /* [한국어] ===== 1단계: 이전 DRAM 응답으로 인한 L2 fill 완료 패킷 처리 ===== */
  if (!m_config->m_L2_config.disabled()) {
    /* [한국어] L2 캐시 활성 시에만 fill 응답 처리 */
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      /* [한국어] access_ready(): L2 내부에서 처리 완료된 요청이 있음
       *         L2_icnt_queue에 여유가 있을 때만 꺼냄 (큐 포화 시 대기) */
      mem_fetch *mf = m_L2cache->next_access();
      /* [한국어] L2 파이프라인 완료 큐에서 응답 꺼냄 */
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        /* [한국어] L2_WR_ALLOC_R: write-allocate 정책에서 L2 miss 후 DRAM에서 읽어온
         *         데이터 로드 요청 — 이것은 상위 캐시로 올라가서는 안 됨 */
        mf->set_reply();
        /* [한국어] mem_fetch를 요청→응답으로 변환 */
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] L2→ICNT 큐 진입 타임스탬프 기록 */
        m_L2_icnt_queue->push(mf);
        /* [한국어] L2→ICNT 출력 큐에 삽입: ICNT가 SM으로 전달 */
      } else {
        /* [한국어] L2_WR_ALLOC_R 응답: write-allocate 읽기 완료 */
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          /* [한국어] FETCH_ON_WRITE 정책: write-allocate 읽기 완료 시 원본 쓰기 mf를 SM에 응답 */
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          /* [한국어] 이 write-allocate 읽기를 유발한 원본 쓰기 요청 복원 */
          assert(original_wr_mf);
          /* [한국어] 원본 쓰기 mf가 반드시 존재해야 함 */
          original_wr_mf->set_reply();
          /* [한국어] 원본 쓰기 요청을 응답 패킷으로 변환 */
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
          /* [한국어] 원본 쓰기 mf를 SM에 응답으로 전달 */
        }
        m_request_tracker.erase(mf);
        /* [한국어] write-allocate 읽기 mf는 request_tracker에서 제거 */
        delete mf;
        /* [한국어] write-allocate 읽기 mf 해제 (원본 쓰기 mf는 위에서 전달됨) */
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  /* [한국어] ===== 2단계: DRAM 완료 응답 → L2 fill 또는 SM 직접 반환 ===== */
  if (!m_dram_L2_queue->empty()) {
    /* [한국어] DRAM에서 돌아온 응답이 있는 경우 */
    mem_fetch *mf = m_dram_L2_queue->top();
    /* [한국어] dram_L2_queue 앞쪽 요청 참조 (peek) */
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      /* [한국어] L2가 이 주소의 fill을 기다리고 있으면 (MSHR에 등록된 경우) fill 수행 */
      if (m_L2cache->fill_port_free()) {
        /* [한국어] fill 포트가 비어 있을 때만 fill 가능 (한 사이클에 1 fill 제한) */
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] fill 큐 진입 타임스탬프 기록 */
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        /* [한국어] L2 캐시에 데이터 충전: MSHR을 만족시키고 라인을 유효화
         *         m_memcpy_cycle_offset: force_l2_tag_update 시 사이클 정렬 보정 */
        m_dram_L2_queue->pop();
        /* [한국어] dram_L2_queue에서 처리 완료 원소 제거 */
      }
    } else if (!m_L2_icnt_queue->full()) {
      /* [한국어] L2가 이 주소를 기다리지 않거나(L2 disabled, texture bypass 등) L2_icnt 여유 있음 */
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      /* [한국어] 쓰기 ACK 패킷만 상태 갱신 (읽기는 이미 set_reply로 변환됨) */
      m_L2_icnt_queue->push(mf);
      /* [한국어] L2 bypass: DRAM 응답을 L2 fill 없이 직접 ICNT로 전달 */
      m_dram_L2_queue->pop();
      /* [한국어] dram_L2_queue에서 제거 */
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  /* [한국어] ===== 3단계: L2 캐시 내부 1사이클 진행 ===== */
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();
  /* [한국어] L2 내부 사이클: 이전 사이클에 등록된 MSHR miss 패킷들을
   *         m_L2_dram_queue(L2interface를 통해)로 내보냄 */

  // new L2 texture accesses and/or non-texture accesses
  /* [한국어] ===== 4단계: icnt_L2_queue의 새 요청을 L2에 접근 ===== */
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    /* [한국어] L2_dram_queue에 여유가 있고 처리할 요청이 있을 때
     *         (L2_dram_queue full = DRAM 큐 역압(back-pressure), 새 접근 중단) */
    mem_fetch *mf = m_icnt_L2_queue->top();
    /* [한국어] icnt_L2_queue 앞쪽 요청 참조 (peek) */
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      /* [한국어] L2 활성 AND (L2가 텍스처 전용이면 텍스처 요청만, 그렇지 않으면 모든 요청) */
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      /* [한국어] 출력 큐(L2_icnt) 포화 여부: HIT 응답을 받을 공간이 없으면 접근 중단 */
      bool port_free = m_L2cache->data_port_free();
      /* [한국어] L2 데이터 포트 여유: 한 사이클에 1개 접근만 허용 */
      if (!output_full && port_free) {
        /* [한국어] 출력 큐 여유 있고 L2 포트 사용 가능할 때만 L2 접근 */
        std::list<cache_event> events;
        /* [한국어] cache_event 목록: L2 접근 중 발생한 이벤트(write-sent, read-sent 등)를 수집 */
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        /* [한국어] L2 캐시 접근 수행:
         *   반환값: HIT / MISS / RESERVATION_FAIL
         *   HIT: 데이터 즉시 반환 가능
         *   MISS: MSHR 등록, DRAM으로 읽기 요청 발행
         *   RESERVATION_FAIL: MSHR 고갈 또는 다른 충돌 — 이 사이클에 처리 불가 */
        bool write_sent = was_write_sent(events);
        /* [한국어] 이번 접근에서 DRAM으로 쓰기(writeback/evict)가 발행됐는지 */
        bool read_sent = was_read_sent(events);
        /* [한국어] 이번 접근에서 DRAM으로 읽기(miss fill)가 발행됐는지 */
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);
        /* [한국어] 디버그 로그: L2 접근 결과 출력 */

        if (status == HIT) {
          /* [한국어] L2 캐시 히트: 데이터가 L2에 있어 DRAM 접근 불필요 */
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            /* [한국어] HIT + write_sent==false: 순수 히트 응답 경로
             *         write_sent==false가 아니면 write-allocate hit (L2에 쓰기만 했음) */
            if (mf->get_access_type() == L1_WRBK_ACC) {
              /* [한국어] L1 writeback HIT: L1이 dirty를 L2에 써넣는 것이므로 SM에 응답 불필요 */
              m_request_tracker.erase(mf);
              delete mf; /* [한국어] L1 writeback 패킷 폐기 */
            } else {
              mf->set_reply();
              /* [한국어] 읽기/일반 요청 히트: 응답 패킷으로 변환 */
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
              /* [한국어] 히트 응답을 L2→ICNT 큐에 삽입: SM이 수신 예정 */
            }
            m_icnt_L2_queue->pop();
            /* [한국어] 처리 완료 — icnt_L2_queue에서 제거 */
          } else {
            assert(write_sent);
            /* [한국어] HIT + write_sent==true: L2에 쓰기를 했으므로(write-allocate evict 등)
             *         SM 응답 없이 요청만 완료 */
            m_icnt_L2_queue->pop();
            /* [한국어] 요청 완료 — icnt_L2_queue에서 제거 */
          }
        } else if (status != RESERVATION_FAIL) {
          /* [한국어] MISS 경로: L2에 없음 → DRAM 요청 발행됨 */
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            /* [한국어] 쓰기 MISS + write-allocate 정책(FETCH_ON_WRITE/LAZY_FETCH_ON_READ) +
             *         이번에 writeallocate 읽기가 발행되지 않은 경우:
             *         write-allocate 읽기 없이 쓰기가 처리된 경우 응답 처리 */
            if (mf->get_access_type() == L1_WRBK_ACC) {
              /* [한국어] L1 writeback miss: L2에 없으므로 DRAM으로 쓰기 발행, SM 응답 불필요 */
              m_request_tracker.erase(mf);
              delete mf; /* [한국어] L1 writeback 패킷 폐기 */
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              /* [한국어] write-back 정책 + 일반 쓰기 MISS: write-allocate 없이 즉시 응답 */
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
              /* [한국어] 쓰기 응답을 SM에 반환 (DRAM으로는 별도로 발행됨) */
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
          /* [한국어] MISS 수리됨 — icnt_L2_queue에서 제거 (MSHR이 나머지 처리) */
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
          /* [한국어] RESERVATION_FAIL: MSHR 슬롯 고갈 또는 캐시 뱅크 충돌
           *         icnt_L2_queue에서 제거하지 않음 → 다음 사이클에 재시도 */
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      /* [한국어] L2 비활성 또는 텍스처 전용 L2에 비텍스처 요청이 온 경우:
       *         L2를 경유하지 않고 바로 DRAM으로 내려보냄 */
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      /* [한국어] L2 bypass → L2_dram_queue 진입 타임스탬프 기록 */
      m_L2_dram_queue->push(mf);
      /* [한국어] L2 없이 직접 DRAM 큐로 삽입 */
      m_icnt_L2_queue->pop();
      /* [한국어] icnt_L2_queue에서 제거 */
    }
  }

  // ROP delay queue
  /* [한국어] ===== 5단계: ROP 지연 큐 처리 ===== */
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    /* [한국어] ROP 큐 앞쪽 원소가 준비됐고(cycle >= ready_cycle) icnt_L2에 여유 있으면 처리 */
    mem_fetch *mf = m_rop.front().req;
    /* [한국어] ROP 지연이 완료된 요청 참조 */
    m_rop.pop();
    /* [한국어] ROP 큐에서 제거 */
    m_icnt_L2_queue->push(mf);
    /* [한국어] icnt_L2_queue로 이동: 다음 사이클의 4단계에서 L2 접근 처리됨 */
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    /* [한국어] icnt_L2_queue 진입 타임스탬프 기록 */
  }
}

/*
 * [한국어]
 * memory_sub_partition::full (단순 버전) - icnt_L2_queue 포화 여부 반환
 *
 * @return: true = icnt_L2_queue가 가득 참, false = 여유 있음
 *
 * memory_partition_unit::can_issue_to_dram()과 외부 ICNT 레이어가
 * 이 슬라이스에 새 요청을 넣을 수 있는지 확인할 때 사용한다.
 */
bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

/*
 * [한국어]
 * memory_sub_partition::full (크기 버전) - icnt_L2_queue의 size개 이상 여유 여부 반환
 *
 * @size: 확인할 필요 슬롯 수
 * @return: true = size개를 수용할 공간 없음(가득 찼거나 부족), false = 여유 있음
 *
 * ICNT에서 여러 패킷을 한 번에 투입하기 전에 충분한 슬롯이 있는지 확인할 때 사용.
 */
bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
  /* [한국어] is_avilable_size(size): size개 이상의 빈 슬롯이 없으면 true (오타 'avilable'은 원본 코드) */
}

/*
 * [한국어]
 * memory_sub_partition::L2_dram_queue_empty - L2→DRAM 미스 큐 비어 있는지 조회
 *
 * @return: true = L2 미스 없음, false = DRAM 발행 대기 중인 요청 존재
 *
 * memory_partition_unit의 dram_cycle()/simple_dram_model_cycle()에서 발행 가능한
 * 요청이 있는지 확인하는 용도.
 */
bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

/*
 * [한국어]
 * memory_sub_partition::L2_dram_queue_top - L2→DRAM 미스 큐 앞쪽 요청 참조 (peek)
 *
 * @return: L2_dram_queue 앞쪽 mem_fetch* (pop하지 않음)
 *
 * DRAM 발행 전 요청 내용을 확인(쓰기 여부 등)하기 위해 peek 사용.
 */
class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

/*
 * [한국어]
 * memory_sub_partition::L2_dram_queue_pop - L2→DRAM 미스 큐 앞쪽 원소 제거
 *
 * DRAM으로 발행을 결정한 후 큐에서 실제로 꺼낼 때 호출.
 */
void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

/*
 * [한국어]
 * memory_sub_partition::dram_L2_queue_full - DRAM→L2 응답 수신 큐 포화 여부 조회
 *
 * @return: true = dram_L2_queue가 가득 참 (DRAM 발행 차단 조건 중 하나)
 *
 * can_issue_to_dram()에서 sub_partition_contention 조건으로 확인.
 * dram_L2_queue가 가득 차면 응답 수신 불가이므로 새 발행을 금지.
 */
bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

/*
 * [한국어]
 * memory_sub_partition::dram_L2_queue_push - DRAM 완료 응답을 dram_L2_queue에 삽입
 *
 * @mf: DRAM에서 돌아온 완료 응답 패킷
 *
 * memory_partition_unit::dram_cycle()/simple_dram_model_cycle()이 DRAM 리턴큐에서
 * 완료 응답을 꺼낸 후 목적지 sub_partition의 dram_L2_queue에 넣을 때 호출.
 */
void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf); /* [한국어] DRAM 완료 응답을 수신 큐에 삽입 */
}

/*
 * [한국어]
 * memory_sub_partition::print_cache_stat - L2 캐시 접근 통계 출력 (stdout)
 *
 * @accesses: [출력] 총 L2 접근 수
 * @misses:   [출력] 총 L2 미스 수
 * @return:   없음
 *
 * 시뮬레이션 결과 요약 출력 시 gpgpu_sim이 호출하여 각 슬라이스의 적중률을 집계한다.
 */
void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout; /* [한국어] stdout으로 출력 */
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
  /* [한국어] L2 캐시 활성 시에만 접근/미스 카운터 출력 */
}

/*
 * [한국어]
 * memory_sub_partition::print - 이 슬라이스의 미완료 요청과 L2 상태 덤프 (디버그용)
 *
 * @fp: 출력 대상 파일 포인터
 * @return: 없음
 *
 * m_request_tracker에 남아 있는 미완료 요청 목록과 L2 캐시 내부 상태(WAY 내용 등)를
 * 출력한다. memory_partition_unit::print()에서 슬라이스별로 호출된다.
 */
void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    /* [한국어] 미완료 요청이 있는 경우만 출력 (빈 슬라이스는 건너뜀) */
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    /* [한국어] 슬라이스 ID와 헤더 출력 */
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      /* [한국어] request_tracker를 순회: set이므로 중복 없고 정렬됨 */
      mem_fetch *mf = *r; /* [한국어] 현재 원소의 mem_fetch 포인터 */
      if (mf)
        mf->print(fp); /* [한국어] 요청 상세 정보 출력 */
      else
        fprintf(fp, " <NULL mem_fetch?>\n"); /* [한국어] NULL 포인터 이상 탐지 */
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
  /* [한국어] L2 캐시 내부 상태(라인별 유효/태그/더티 비트 등) 출력 */
}

/*
 * [한국어]
 * memory_stats_t::visualizer_print - AerialVision용 L2 통계 gzip 파일에 출력
 *
 * @visualizer_file: gzip 압축 AerialVision 출력 파일 핸들
 * @return: 없음
 *
 * 주기적으로 호출되어 현재 샘플링 윈도우의 L2 읽기/쓰기 히트·미스 카운터와
 * mem_fetch 평균 지연을 gzip 파일에 기록한다. 출력 후 per-window 카운터를 리셋하여
 * 다음 윈도우가 독립적으로 집계되도록 한다.
 *
 * 호출 체인:
 *   gpgpu_sim::visualizer_print() → memory_sub_partition::visualizer_print()
 *     → [이 함수] (memory_stats_t::visualizer_print)
 */
void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  /* [한국어] L2 쓰기 미스 카운터 출력 (이번 윈도우 누적) */
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  /* [한국어] L2 쓰기 히트 카운터 출력 */
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  /* [한국어] L2 읽기 미스 카운터 출력 */
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  /* [한국어] L2 읽기 히트 카운터 출력 */
  clear_L2_stats_pw();
  /* [한국어] 출력 완료 후 per-window 카운터 초기화 — 다음 윈도우 독립 집계 */

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
  /* [한국어] 완료된 mem_fetch가 하나라도 있으면 평균 지연 출력:
   *         mf_total_lat / num_mfs = 평균 지연 사이클 */
}

/*
 * [한국어]
 * memory_stats_t::clear_L2_stats_pw - L2 per-window 히트·미스 카운터 초기화
 *
 * @return: 없음
 *
 * visualizer_print()와 memory_sub_partition::visualizer_print()에서 윈도우 출력 후
 * 카운터를 리셋하여 다음 샘플링 윈도우가 독립적으로 집계되도록 한다.
 *
 * 호출 체인:
 *   memory_stats_t::visualizer_print() / memory_sub_partition::visualizer_print() → [이 함수]
 */
void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0; /* [한국어] L2 쓰기 미스 윈도우 카운터 초기화 */
  L2_write_hit = 0;  /* [한국어] L2 쓰기 히트 윈도우 카운터 초기화 */
  L2_read_miss = 0;  /* [한국어] L2 읽기 미스 윈도우 카운터 초기화 */
  L2_read_hit = 0;   /* [한국어] L2 읽기 히트 윈도우 카운터 초기화 */
}

/*
 * [한국어]
 * gpgpu_sim::print_dram_stats - 전체 DRAM 채널의 명령 통계를 집계하여 출력
 *
 * @fout: 출력 대상 파일 포인터 (통상 시뮬레이터 결과 파일)
 * @return: 없음
 *
 * 모든 DRAM 채널(m_n_mem개)의 명령 카운터를 합산하여 전체 시뮬레이션 동안의
 * DRAM 활동을 요약 출력한다. AccelWattch 전력 분석과 시뮬레이션 결과 검증에 활용된다.
 * 출력 항목: 총 읽기, 쓰기(wr + wr_WB), 활성화(ACT), 명령 수, NOP, 사전충전(PRE), 요청 수.
 *
 * 호출 체인:
 *   gpgpu_sim::print_GPU_stats() → [이 함수] → m_memory_partition_unit[i]->set_dram_power_stats()
 */
void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;       /* [한국어] 채널별 총 DRAM 명령 수 (임시 변수) */
  unsigned activity = 0;  /* [한국어] 채널별 활성 사이클 수 */
  unsigned nop = 0;       /* [한국어] 채널별 NOP 명령 수 */
  unsigned act = 0;       /* [한국어] 채널별 ACT(활성화) 명령 수 */
  unsigned pre = 0;       /* [한국어] 채널별 PRE(사전충전) 명령 수 */
  unsigned rd = 0;        /* [한국어] 채널별 읽기 명령 수 */
  unsigned wr = 0;        /* [한국어] 채널별 쓰기 명령 수 */
  unsigned wr_WB = 0;     /* [한국어] 채널별 writeback 쓰기 명령 수 */
  unsigned req = 0;       /* [한국어] 채널별 총 요청 수 */
  unsigned tot_cmd = 0;   /* [한국어] 전체 채널 합산 명령 수 */
  unsigned tot_nop = 0;   /* [한국어] 전체 채널 합산 NOP */
  unsigned tot_act = 0;   /* [한국어] 전체 채널 합산 ACT */
  unsigned tot_pre = 0;   /* [한국어] 전체 채널 합산 PRE */
  unsigned tot_rd = 0;    /* [한국어] 전체 채널 합산 읽기 */
  unsigned tot_wr = 0;    /* [한국어] 전체 채널 합산 쓰기(wr + wr_WB 합산) */
  unsigned tot_req = 0;   /* [한국어] 전체 채널 합산 요청 수 */

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    /* [한국어] 모든 DRAM 채널(파티션)을 순회하며 통계 수집 */
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    /* [한국어] i번 채널의 DRAM 명령 카운터를 출력 파라미터로 읽어옴 */
    tot_cmd += cmd;   /* [한국어] 전체 명령 수 누적 */
    tot_nop += nop;   /* [한국어] 전체 NOP 수 누적 */
    tot_act += act;   /* [한국어] 전체 ACT 수 누적 */
    tot_pre += pre;   /* [한국어] 전체 PRE 수 누적 */
    tot_rd += rd;     /* [한국어] 전체 읽기 수 누적 */
    tot_wr += wr + wr_WB; /* [한국어] 일반 쓰기 + writeback 쓰기를 합산 */
    tot_req += req;   /* [한국어] 전체 요청 수 누적 */
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  /* [한국어] 전체 DRAM 읽기 명령 수 출력 */
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  /* [한국어] 전체 DRAM 쓰기 명령 수 출력 (wr + wr_WB 합산) */
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  /* [한국어] 전체 DRAM ACT(행 활성화) 명령 수 출력 */
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  /* [한국어] 전체 DRAM 명령 총 수 출력 */
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  /* [한국어] 전체 DRAM NOP(유휴) 사이클 수 출력 */
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  /* [한국어] 전체 DRAM PRE(사전충전) 명령 수 출력 */
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
  /* [한국어] 전체 DRAM 요청 수 출력 */
}

/*
 * [한국어]
 * memory_sub_partition::flushL2 - L2 캐시 슬라이스의 모든 dirty 라인을 flush
 *
 * @return: 0 (TODO: flush된 데이터를 main memory에 쓰는 기능 미구현)
 *
 * CUDA cuMemsetD32/커널 완료 후 L2를 클린 상태로 만들 때 호출된다.
 * 현재는 단순히 L2 캐시를 flush만 하며, flush된 dirty 데이터를 DRAM으로 실제로
 * 내려보내는 기능은 TODO로 남아 있다(주석 참고).
 *
 * 호출 체인:
 *   gpgpu_sim::gpu_flush() → [이 함수] → m_L2cache->flush()
 */
unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush(); /* [한국어] L2 캐시의 모든 dirty 라인을 무효화(clean 처리) */
  }
  return 0;  // TODO: write the flushed data to the main memory
  /* [한국어] 반환값 0: 실제 flush 효과 없음. DRAM 반영 구현 필요(TODO) */
}

/*
 * [한국어]
 * memory_sub_partition::invalidateL2 - L2 캐시 슬라이스의 모든 라인을 무효화
 *
 * @return: 0
 *
 * flush와 달리 dirty 데이터를 DRAM으로 내보내지 않고 모든 캐시 라인을 invalid 상태로
 * 만든다. 시뮬레이션 체크포인팅이나 캐시 동작 실험을 위한 강제 초기화에 사용.
 *
 * 호출 체인:
 *   시뮬레이터 제어 코드 → [이 함수] → m_L2cache->invalidate()
 */
unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate(); /* [한국어] 모든 캐시 라인 invalid화 (dirty 데이터 버림) */
  }
  return 0; /* [한국어] 반환값 0: 고정 */
}

/*
 * [한국어]
 * memory_sub_partition::busy - 이 슬라이스에 미완료 요청이 있는지 조회
 *
 * @return: true = request_tracker에 아직 완료되지 않은 요청 존재,
 *          false = 모든 요청 완료
 *
 * request_tracker는 push()에서 요청 삽입, pop()/set_done()에서 제거되므로
 * 비어 있으면 이 슬라이스가 완전히 유휴 상태임을 의미한다.
 */
bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }
/* [한국어] request_tracker가 비어 있으면 false(유휴), 아니면 true(처리 중) */

/*
 * [한국어]
 * memory_sub_partition::breakdown_request_to_sector_requests - 큰 요청을 SECTOR 단위로 분해
 *
 * @mf: 분해할 원본 mem_fetch (최대 128바이트 요청)
 * @return: 섹터(32바이트) 단위로 분해된 mem_fetch 포인터 벡터
 *
 * SECTOR 캐시 타입을 사용할 때 128바이트 요청을 32바이트 섹터 단위로 나누어
 * 각 섹터마다 독립적인 mem_fetch를 생성한다. SECTOR 캐시는 라인을 4개의 32바이트
 * 섹터로 분리하여 관리하므로, 각 섹터의 히트/미스를 독립적으로 판단할 수 있다.
 *
 * 4가지 분기:
 * 1) 이미 SECTOR_SIZE(32B) + sector_mask.count()==1: 그대로 반환 (이미 섹터 단위)
 * 2) MAX_MEMORY_ACCESS_SIZE(128B): 4개 섹터로 분해, 주소 += SECTOR_SIZE*i
 * 3) 64B + sector_mask all/none: 상수 캐시(constant cache) 경우,
 *    주소 정렬에 따라 섹터 0~1 또는 2~3 사용
 * 4) 기타: sector_mask에서 set된 비트에 해당하는 섹터만 생성
 *
 * 호출 체인:
 *   memory_sub_partition::push() → [이 함수] → m_mf_allocator->alloc()
 */
std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  /* [한국어] 분해된 섹터 mem_fetch 포인터를 담을 결과 벡터 */
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  /* [한국어] 원본 요청의 섹터 마스크: 어떤 32바이트 슬롯이 유효한지 나타냄 */
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    /* [한국어] 이미 단일 섹터(32B) 단위 요청: 분해 없이 그대로 사용 */
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    /* [한국어] 최대 크기(128B) 요청: 4개의 32B 섹터로 완전 분해 */
    // break down every sector
    mem_access_byte_mask_t mask;
    /* [한국어] 각 섹터의 byte_mask를 만들기 위한 임시 마스크 */
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      /* [한국어] SECTOR_CHUNCK_SIZE = 4: 섹터 인덱스 0~3 순회 */
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
        /* [한국어] 섹터 i의 바이트 범위[i*32 ~ (i+1)*32)를 mask에 설정 */
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());
      /* [한국어] 섹터 i용 mem_fetch 생성:
       *   - 주소: 원본 주소 + i*32 (각 섹터의 시작 주소)
       *   - byte_mask & mask: 이 섹터에 해당하는 바이트만 선택
       *   - set(i): 단일 비트 sector_mask (섹터 i만 유효)
       *   - mf: 부모 포인터 (섹터 완료 시 부모 추적용) */

      result.push_back(n_mf);
      /* [한국어] 결과 벡터에 추가 */
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    /* [한국어] 상수 캐시(constant cache) 특수 케이스:
     *         64B 요청이고 sector_mask가 전체 설정 또는 전체 미설정인 경우
     *         주소 정렬에 따라 하위 2개(0,1) 또는 상위 2개(2,3) 섹터 사용 */
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0; /* [한국어] 128B 경계에 정렬된 경우: 섹터 0,1 사용 */
    else
      start = 2; /* [한국어] 정렬 안 된 경우: 섹터 2,3 사용 */
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      /* [한국어] 2개의 인접 섹터(start, start+1)를 생성 */
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k); /* [한국어] 섹터 i의 바이트 범위를 mask에 설정 */
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());
      /* [한국어] 상수 캐시 섹터 mem_fetch 생성:
       *         주소 오프셋 없음 (같은 주소에서 섹터 인덱스로 구분) */

      result.push_back(n_mf);
    }
  } else {
    /* [한국어] 일반 케이스: sector_mask에서 set된 비트의 섹터만 생성 */
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        /* [한국어] 섹터 i가 유효한 경우만 mem_fetch 생성 */
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k); /* [한국어] 섹터 i 바이트 범위 마스크 */
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());
        /* [한국어] sector_mask가 지정한 섹터만 생성: 부분 접근 요청 처리 */

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  /* [한국어] 섹터가 하나도 생성되지 않으면 로직 버그: assert 발동 */
  return result; /* [한국어] 분해된 섹터 mem_fetch 벡터 반환 */
}

/*
 * [한국어]
 * memory_sub_partition::push - ICNT에서 도착한 메모리 요청을 이 슬라이스에 투입
 *
 * @m_req: ICNT에서 수신한 mem_fetch 패킷 포인터 (NULL이면 무시)
 * @cycle: 도착 사이클 번호 (ROP ready_cycle 계산 기준)
 * @return: 없음
 *
 * SM이 생성한 메모리 요청이 ICNT를 통과한 후 이 슬라이스에 도착하는 진입점이다.
 * 세 가지 작업을 수행한다:
 * 1) 지연 통계 기록: memlatstat_icnt2mem_pop()로 ICNT→MEM 지연 측정
 * 2) SECTOR 캐시 분해: SECTOR 타입이면 breakdown_request_to_sector_requests()로 32B 단위 분해
 * 3) 큐 라우팅:
 *    - 텍스처 요청: ROP 없이 icnt_L2_queue로 직접 진입 (읽기 전용이므로 ROP 불필요)
 *    - 비텍스처 요청: rop_latency 사이클 지연 후 m_rop 큐로 진입
 *
 * ROP(Render Output Unit) 지연은 GPU의 write path 파이프라인 단계를 모델링한다.
 *
 * 호출 체인:
 *   gpgpu_sim::interconnect_inject_memory_pop() → [이 함수]
 *     → breakdown_request_to_sector_requests(), m_rop.push(), m_icnt_L2_queue->push()
 */
void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    /* [한국어] NULL이 아닌 유효한 요청만 처리 */
    m_stats->memlatstat_icnt2mem_pop(m_req);
    /* [한국어] ICNT→메모리 지연 기록: 패킷이 ICNT에 진입한 시각부터 여기까지의 지연 측정 */
    std::vector<mem_fetch *> reqs;
    /* [한국어] 처리할 요청 목록 (SECTOR 분해 후 1~4개) */
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
      /* [한국어] SECTOR 캐시: 원본 요청을 32B 섹터 단위로 분해 */
    else
      reqs.push_back(m_req);
      /* [한국어] 일반 캐시: 원본 요청 그대로 사용 */

    for (unsigned i = 0; i < reqs.size(); ++i) {
      /* [한국어] 분해된 각 섹터 요청(또는 원본 요청)을 큐에 투입 */
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      /* [한국어] request_tracker에 등록: busy()와 set_done()에서 사용 */
      if (req->istexture()) {
        /* [한국어] 텍스처 요청: ROP 없이 바로 icnt_L2_queue 진입 */
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] icnt_L2_queue 진입 타임스탬프 기록 */
      } else {
        /* [한국어] 비텍스처 요청: ROP 지연 큐를 거쳐 icnt_L2_queue로 이동 */
        rop_delay_t r;
        r.req = req; /* [한국어] ROP 슬롯에 요청 포인터 저장 */
        r.ready_cycle = cycle + m_config->rop_latency;
        /* [한국어] ROP 완료 사이클 = 도착 사이클 + rop_latency (gpgpusim.config 설정) */
        m_rop.push(r);
        /* [한국어] ROP 지연 큐에 삽입: cache_cycle()의 5단계에서 처리됨 */
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] ROP 큐 진입 타임스탬프 기록 */
      }
    }
  }
}

/*
 * [한국어]
 * memory_sub_partition::pop - L2→ICNT 출력 큐에서 완료 응답 꺼내기
 *
 * @return: SM에 전달할 완료 mem_fetch*(writeback이면 NULL 반환),
 *          큐가 비어 있으면 NULL
 *
 * ICNT 레이어가 이 슬라이스에서 완료 응답을 가져갈 때 호출된다. 세 가지 처리:
 * 1) request_tracker에서 제거 (busy 상태 갱신)
 * 2) atomic 연산이면 do_atomic() 수행 (원자 연산 완료 처리)
 * 3) writeback(L2_WRBK_ACC/L1_WRBK_ACC)이면 SM에 반환하지 않고 삭제 후 NULL 반환
 *
 * 호출 체인:
 *   gpgpu_sim::interconnect_inject_memory_pop() / SM 메모리 완료 경로 → [이 함수]
 */
mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  /* [한국어] L2→ICNT 출력 큐에서 완료 응답 꺼냄 */
  m_request_tracker.erase(mf);
  /* [한국어] request_tracker에서 제거: 이 요청이 완료됐음을 기록 */
  if (mf && mf->isatomic()) mf->do_atomic();
  /* [한국어] atomic 연산(atomicCAS 등) 완료 시 do_atomic() 수행:
   *         메모리 갱신을 실제로 적용하는 기능 시뮬레이션 */
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    /* [한국어] writeback 응답: SM에 전달할 필요 없음 — 삭제하고 NULL 반환 */
    delete mf; /* [한국어] writeback 패킷 해제 */
    mf = NULL; /* [한국어] NULL 반환으로 호출자에게 "전달할 응답 없음" 통보 */
  }
  return mf; /* [한국어] 일반 완료 응답 반환 (NULL이면 writeback이었음) */
}

/*
 * [한국어]
 * memory_sub_partition::top - L2→ICNT 출력 큐 앞쪽 원소 peek (pop 안함)
 *
 * @return: L2_icnt_queue 앞쪽 mem_fetch* (writeback이면 즉시 처리 후 NULL 반환)
 *
 * ICNT 레이어가 응답을 꺼내기 전에 내용을 확인하거나, writeback을 미리 처리할 때 사용.
 * writeback을 발견하면 pop+erase+delete를 수행하고 NULL을 반환한다.
 *
 * 호출 체인:
 *   gpgpu_sim 또는 ICNT 경계 코드 → [이 함수]
 */
mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  /* [한국어] L2_icnt_queue 앞쪽 참조 (아직 pop하지 않음) */
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    /* [한국어] writeback이 큐 앞에 있으면 즉시 처리하고 NULL 반환 */
    m_L2_icnt_queue->pop();  /* [한국어] 큐에서 제거 */
    m_request_tracker.erase(mf); /* [한국어] request_tracker에서 제거 */
    delete mf; /* [한국어] writeback 패킷 해제 */
    mf = NULL; /* [한국어] NULL 반환 신호 */
  }
  return mf; /* [한국어] 일반 완료 응답 반환 (NULL이면 writeback이었거나 큐가 비었음) */
}

/*
 * [한국어]
 * memory_sub_partition::set_done - request_tracker에서 요청 제거 (소유권 없이 완료 처리)
 *
 * @mf: 완료 처리할 mem_fetch 포인터
 * @return: 없음
 *
 * pop()이나 top()과 달리 패킷을 delete하지 않고 추적만 해제한다.
 * memory_partition_unit::set_done()이 writeback 크레딧 반환 후 이 함수를 호출하며,
 * L2cache의 writeback evict가 발행될 때처럼 소유권이 다른 경로에 있는 경우에 사용.
 *
 * 호출 체인:
 *   memory_partition_unit::set_done() → [이 함수]
 */
void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
  /* [한국어] request_tracker에서 이 패킷의 추적만 제거: delete는 호출자 책임 */
}

/*
 * [한국어]
 * memory_sub_partition::accumulate_L2cache_stats - 이 슬라이스의 L2 통계를 누적 합산
 *
 * @l2_stats: [입출력] 전체 L2 통계 집계 객체; 이 슬라이스 통계가 += 로 추가됨
 * @return: 없음
 *
 * 시뮬레이션 결과 집계 시 gpgpu_sim이 모든 슬라이스의 통계를 합산하는 데 사용.
 * operator+=로 cache_stats를 병합하므로 L2 슬라이스 수에 상관없이 전체 통계 집계 가능.
 *
 * 호출 체인:
 *   gpgpu_sim::print_GPU_stats() → [이 함수] → m_L2cache->get_stats()
 */
void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
    /* [한국어] 이 슬라이스의 캐시 통계(히트/미스/접근 수 등)를 전달받은 집계 객체에 합산 */
  }
}

/*
 * [한국어]
 * memory_sub_partition::get_L2cache_sub_stats - 이 슬라이스의 L2 서브통계 조회
 *
 * @css: [출력] 단일 슬라이스의 read_hit/miss, write_hit/miss 등 요약 통계
 * @return: 없음
 *
 * AccelWattch 전력 계산과 최종 통계 출력 시 슬라이스별 세부 통계를 가져올 때 사용.
 *
 * 호출 체인:
 *   power_stat_t::save_stats() / 통계 출력 경로 → [이 함수] → m_L2cache->get_sub_stats()
 */
void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
    /* [한국어] L2 캐시에서 누적 서브통계를 css에 저장 */
  }
}

/*
 * [한국어]
 * memory_sub_partition::get_L2cache_sub_stats_pw - per-window L2 통계 조회
 *
 * @css: [출력] 현재 샘플링 윈도우의 read_hit/miss, write_hit/miss 통계
 * @return: 없음
 *
 * visualizer_print()에서 호출하여 이번 윈도우의 L2 히트·미스를 수집한 후
 * memory_stats_t의 전역 카운터에 누적한다.
 *
 * 호출 체인:
 *   memory_sub_partition::visualizer_print() → [이 함수] → m_L2cache->get_sub_stats_pw()
 */
void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
    /* [한국어] L2 캐시에서 per-window 통계를 css에 저장 */
  }
}

/*
 * [한국어]
 * memory_sub_partition::clear_L2cache_stats_pw - L2 캐시 per-window 통계 초기화
 *
 * @return: 없음
 *
 * visualizer_print()에서 윈도우 통계를 수집한 후 다음 윈도우를 위해 캐시 내부의
 * per-window 카운터를 초기화한다.
 *
 * 호출 체인:
 *   memory_sub_partition::visualizer_print() → [이 함수] → m_L2cache->clear_pw()
 */
void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
    /* [한국어] L2 캐시의 per-window 히트·미스 카운터 초기화 */
  }
}

/*
 * [한국어]
 * memory_sub_partition::visualizer_print - AerialVision L2 슬라이스 통계 수집 및 초기화
 *
 * @visualizer_file: gzip 압축 AerialVision 출력 파일 핸들 (현재 미사용)
 * @return: 없음
 *
 * 이 슬라이스의 per-window L2 캐시 통계를 memory_stats_t의 전역 카운터에 누적한다.
 * 그 후 슬라이스 내 per-window 카운터를 초기화하여 다음 윈도우가 독립적으로 집계되도록 한다.
 * 실제 gzip 파일 출력은 memory_stats_t::visualizer_print()에서 처리한다.
 *
 * 호출 체인:
 *   memory_partition_unit::visualizer_print() → [이 함수]
 *     → get_L2cache_sub_stats_pw(), m_stats 카운터 누적, clear_L2cache_stats_pw()
 */
void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  /* [한국어] 이번 윈도우의 슬라이스별 L2 히트·미스 통계를 담을 임시 구조체 */
  get_L2cache_sub_stats_pw(temp_sub_stats);
  /* [한국어] 이 슬라이스의 현재 윈도우 통계를 읽어옴 */

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  /* [한국어] 이번 윈도우 읽기 미스를 전역 통계에 누적 */
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  /* [한국어] 이번 윈도우 쓰기 미스를 전역 통계에 누적 */
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  /* [한국어] 이번 윈도우 읽기 히트를 전역 통계에 누적 */
  m_stats->L2_write_hit += temp_sub_stats.write_hits;
  /* [한국어] 이번 윈도우 쓰기 히트를 전역 통계에 누적 */

  clear_L2cache_stats_pw();
  /* [한국어] 슬라이스 내 per-window 카운터 초기화: 다음 윈도우 독립 집계를 위해 */
}
