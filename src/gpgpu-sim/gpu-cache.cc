// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
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

/*
 * [한국어 설명] GPU 캐시 계층 구현 (gpu-cache.cc)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 GPU 내부 캐시 계층(L1D, L1 Read-Only, Texture, L2)의 전체 동작을
 * 구현하는 핵심 타이밍 모델 파일이다. 각 사이클마다 캐시에 도달하는 mem_fetch
 * 요청을 처리하고, 태그 배열(tag_array) 탐색, MSHR(Miss Status Holding Register)
 * 관리, 쓰기 정책(write-back/write-through/write-evict/write-allocate) 처리,
 * 대역폭 관리, 통계 집계까지 담당한다. 섹터 캐시(sector cache)와 일반 라인
 * 캐시(normal line cache) 양쪽을 모두 지원하며, 캐시 설정은 gpgpusim.config의
 * -gpgpu_cache:dl1, -gpgpu_cache:dl2 등의 옵션으로 제어된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 타이밍 시뮬레이션(gpgpu-sim/) 계층의 메모리 서브시스템 핵심에 위치한다.
 * SM(shader core)의 메모리 파이프라인에서 발생한 메모리 요청이 mem_fetch 객체로
 * 패키징되어 이 파일의 캐시 access() 함수로 전달된다.
 *
 * 호출 체인 (요청 방향):
 *   shader.cc (ldst_unit::cycle) → l1_cache::access / l2_cache::access
 *     → data_cache::access → process_tag_probe
 *       → tag_array::probe / tag_array::access / tag_array::fill
 *       → send_read_request / send_write_request → mem_fetch → ICNT(intersim2)
 *
 * 호출 체인 (응답 방향):
 *   하위 메모리(DRAM/L2) → baseline_cache::fill
 *     → tag_array::fill → mshr_table::mark_ready
 *
 * 실행 컨텍스트: 사이클-레벨 타이밍 시뮬레이션의 메모리 파이프라인 단계에서
 * 매 GPU 사이클마다 단일 시뮬레이션 스레드로 실행된다(재진입 없음).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - gpu-cache.h: 모든 캐시 클래스 선언(cache_config, tag_array, mshr_table,
 *                  baseline_cache, data_cache, l1_cache, l2_cache, tex_cache,
 *                  read_only_cache, cache_stats, bandwidth_management)
 *   - mem_fetch.h: 메모리 요청 패킷 (주소, 접근 타입, 섹터/바이트 마스크 등)
 *   - gpu-sim.h: 전역 GPU 시뮬레이터 상태 (aggregated_l1_stats, gpu_sim_cycle 등)
 *   - hashing.h: bitwise_hash_function, ipoly_hash_function (세트 인덱싱 해시)
 *   - stat-tool.h: shader_cache_access_log (L1 접근 로그 헬퍼)
 *
 * 이 모듈에 의존하는 모듈:
 *   - shader.cc: ldst_unit이 l1_cache::access() 및 fill()을 호출
 *   - gpu-sim.cc: L2 캐시 사이클, 통계 집계를 위해 l2_cache::access() 및
 *                 baseline_cache::fill() 호출
 *   - mem_sub_partition.cc: L2 캐시 접근 및 fill 제어
 *
 * 데이터 흐름:
 *   mem_fetch (SM 발원) → [cache access] → HIT(데이터 반환) 또는
 *   MISS(m_miss_queue에 삽입 → ICNT 전송 → DRAM 응답 → fill() → MSHR 해소)
 *
 * 공유 자료구조:
 *   - tag_array::m_lines[]: 캐시 라인 배열 (cache_block_t* 또는 sector_cache_block*)
 *   - mshr_table::m_data: MSHR 엔트리 맵 (block_addr → 대기 중인 mem_fetch 목록)
 *   - baseline_cache::m_miss_queue: 하위 메모리로 전송 대기 중인 mem_fetch 큐
 *   - baseline_cache::m_extra_mf_fields: fill 응답 시 캐시 인덱스 복원용 맵
 *
 * === 주요 함수/구조체 요약 ===
 * tag_array::probe()            - 태그 배열 탐색; HIT/MISS/RESERVATION_FAIL 등 판단
 * tag_array::access()           - 태그 접근 + LRU 갱신 + ON_MISS 정책 시 라인 할당
 * tag_array::fill()             - 하위 메모리에서 데이터 복귀 시 태그 배열에 채우기
 * cache_config::hash_function() - 5가지 세트 인덱싱 해시(LINEAR/FERMI/XOR/IPOLY/CUSTOM)
 * baseline_cache::fill()        - 하위 메모리 응답 처리, MSHR mark_ready, 섹터 집계
 * baseline_cache::send_read_request() - MSHR probe/merge 후 miss_queue에 read 요청 등록
 * data_cache::process_tag_probe()     - probe 결과에 따라 4가지 쓰기/읽기 정책 함수 디스패치
 * data_cache::wr_hit_wb()        - write-back 정책: 캐시만 MODIFIED로 표시
 * data_cache::wr_hit_wt()        - write-through 정책: 캐시 + 하위 메모리에 즉시 쓰기
 * data_cache::wr_miss_wa_naive() - write-allocate: write + read 요청 동시 생성(3.x 방식)
 * data_cache::wr_miss_wa_fetch_on_write() - 전체 라인 쓰기 시 fetch 불필요, 부분 시 fetch
 * data_cache::wr_miss_wa_lazy_fetch_on_read() - 쓰기 시 MODIFIED, 읽기 시에만 fetch
 * tex_cache::access()           - fragment FIFO + ROB으로 out-of-order 응답을 in-order 정렬
 * cache_stats::inc_stats()      - (접근 타입, 결과 상태) 쌍의 통계 카운터 증가
 */

#include "gpu-cache.h"       /* [한국어] 캐시 클래스 계층 전체 선언 — tag_array, mshr_table, data_cache, tex_cache 등 */
#include <assert.h>          /* [한국어] 런타임 단언(assertion) — 캐시 인덱스 범위 등 불변 조건 검증에 사용 */
#include "gpu-sim.h"         /* [한국어] 전역 GPU 시뮬레이터 상태 접근 — aggregated_l1/l2_stats, gpu_sim_cycle */
#include "hashing.h"         /* [한국어] 세트 인덱싱용 해시 함수 선언 — bitwise_hash_function, ipoly_hash_function */
#include "stat-tool.h"       /* [한국어] 캐시 접근 로그 헬퍼 — shader_cache_access_log(core_id, type_id, is_miss) */

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

/*
 * [한국어]
 * cache_request_status_str - 캐시 요청 결과 상태(enum)를 디버그용 문자열로 변환
 *
 * @status: 변환할 cache_request_status enum 값
 *          (HIT, HIT_RESERVED, MISS, RESERVATION_FAIL, SECTOR_MISS, MSHR_HIT)
 * @return: 해당 상태를 나타내는 정적 문자열 포인터 (해제 불필요)
 *
 * 시뮬레이션 로그 출력, cache_stats::print_stats() 등에서 열(column) 레이블로 사용된다.
 * assert로 enum 값 범위를 검증하므로, 새 상태 추가 시 static 배열도 동시에 갱신해야 한다.
 *
 * 호출 체인:
 *   cache_stats::print_stats() → [이 함수] (문자열 변환)
 */
const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
      "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
      "SECTOR_MISS", "MSHR_HIT"};
  /* [한국어] HIT: 태그 배열에서 유효한 라인 탐색 성공
   * HIT_RESERVED: 라인이 이미 RESERVED(하위 메모리 대기 중) — 파이프라인 재시도 대상
   * MISS: 태그 불일치 또는 INVALID 라인 — 하위 메모리 요청 필요
   * RESERVATION_FAIL: 모든 라인이 RESERVED 상태 — 이 사이클에 처리 불가
   * SECTOR_MISS: 태그는 일치하나 요청 섹터가 INVALID — 섹터 단위 부분 미스
   * MSHR_HIT: 동일 블록 요청이 이미 MSHR에 대기 중 — merge 처리 */

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS); /* [한국어] 배열 크기와 enum 개수 일치 여부 컴파일-타임/런타임 검증 */
  assert(status < NUM_CACHE_REQUEST_STATUS); /* [한국어] 유효 범위 내 status인지 검증 */

  return static_cache_request_status_str[status]; /* [한국어] enum 값을 배열 인덱스로 직접 사용하여 문자열 반환 */
}

/*
 * [한국어]
 * cache_fail_status_str - 캐시 예약 실패 이유(enum)를 디버그용 문자열로 변환
 *
 * @status: 변환할 cache_reservation_fail_reason enum 값
 *          (LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL,
 *           MSHR_MERGE_ENRTY_FAIL, MSHR_RW_PENDING)
 * @return: 해당 실패 이유를 나타내는 정적 문자열 포인터
 *
 * RESERVATION_FAIL 반환 시 구체적인 원인을 로그로 남기기 위해 사용된다.
 * cache_stats::print_fail_stats() 에서 실패 이유별 통계 출력에 활용된다.
 *
 * 실패 원인 의미:
 *   LINE_ALLOC_FAIL      - 세트 내 모든 라인이 RESERVED 상태 (라인 부족)
 *   MISS_QUEUE_FULL      - 하위 메모리로의 발송 큐가 가득 참
 *   MSHR_ENRTY_FAIL      - MSHR 신규 엔트리 공간 부족 (새 블록 추가 불가)
 *   MSHR_MERGE_ENRTY_FAIL - 기존 MSHR 엔트리에 merge 슬롯 부족
 *   MSHR_RW_PENDING      - 쓰기 요청 후 읽기가 대기 중 — 값 오염 방지를 위해 차단
 *
 * 호출 체인:
 *   cache_stats::print_fail_stats() → [이 함수] (실패 원인 문자열 변환)
 */
const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
      "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
      "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};
  /* [한국어] 각 실패 유형에 대응하는 문자열 배열 — enum 순서와 정확히 일치해야 함 */

  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS); /* [한국어] 배열 크기와 enum 개수 일치 검증 */
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS); /* [한국어] 유효 범위 내 status인지 검증 */

  return static_cache_reservation_fail_reason_str[status]; /* [한국어] 인덱스 직접 접근으로 해당 문자열 반환 */
}

/*
 * [한국어]
 * l1d_cache_config::set_bank - L1 데이터 캐시의 뱅크 인덱스 계산
 *
 * @addr: 접근 대상 메모리 주소 (new_addr_type = uint64_t)
 * @return: 0 ~ (l1_banks - 1) 범위의 뱅크 인덱스
 *
 * Volta 아키텍처에서는 섹터 캐시(sector cache) 구조를 사용하며, 뱅크는 섹터
 * 인터리빙(sector interleaving) 방식으로 선택된다. 즉, 연속된 섹터들이 서로
 * 다른 뱅크에 분산되어 동일 사이클에 여러 섹터를 병렬로 접근할 수 있다.
 * 일반 라인 캐시의 경우 라인 인터리빙(line interleaving) 방식을 사용한다.
 * gpgpusim.config의 -gpgpu_l1_banks, -gpgpu_l1_banks_hashing_function 옵션으로 제어.
 *
 * 호출 체인:
 *   ldst_unit (shader.cc) → l1d_cache::access() → [이 함수] (뱅크 충돌 검사)
 */
unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  return cache_config::hash_function(addr, l1_banks,         /* [한국어] 총 뱅크 수 (gpgpusim.config의 l1_banks) */
                                     l1_banks_byte_interleaving_log2, /* [한국어] 섹터/라인 크기의 log2 (인터리빙 단위) */
                                     l1_banks_log2,           /* [한국어] 뱅크 수의 log2 (해시 출력 비트 수) */
                                     l1_banks_hashing_function); /* [한국어] 사용할 해시 함수 종류 (LINEAR/FERMI 등) */
}

/*
 * [한국어]
 * cache_config::set_index - 주소에서 캐시 세트 인덱스 계산
 *
 * @addr: 접근 대상 메모리 주소
 * @return: 0 ~ (m_nset - 1) 범위의 세트 인덱스
 *
 * 캐시 세트 인덱스는 캐시의 조직(세트 수, 라인 크기)과 설정된 해시 함수에 따라
 * 결정된다. 기본적으로 LINEAR_SET_FUNCTION이며, 캐시 충돌을 줄이기 위해
 * FERMI_HASH_SET_FUNCTION이나 BITWISE_XORING_FUNCTION 등을 선택할 수 있다.
 * 이 함수는 l2_cache_config에서 override되어 파티션 주소를 기준으로 재계산된다.
 *
 * 호출 체인:
 *   tag_array::probe() → [이 함수] → cache_config::hash_function()
 */
unsigned cache_config::set_index(new_addr_type addr) const {
  return cache_config::hash_function(addr, m_nset,          /* [한국어] 캐시 세트 수 (예: 32, 64) */
                                     m_line_sz_log2,         /* [한국어] 라인(또는 섹터) 크기의 log2 */
                                     m_nset_log2,            /* [한국어] 세트 수의 log2 */
                                     m_set_index_function);  /* [한국어] 세트 인덱싱 해시 함수 종류 */
}

/*
 * [한국어]
 * cache_config::hash_function - 5가지 세트 인덱싱 해시 함수 구현 (캐시 세트/뱅크 인덱스 결정)
 *
 * @addr: 접근 대상 메모리 주소 (64비트)
 * @m_nset: 캐시의 총 세트 수 (예: 32, 64)
 * @m_line_sz_log2: 라인(또는 섹터) 크기의 log2 — 하위 오프셋 비트를 제거하기 위한 시프트 값
 * @m_nset_log2: 세트 수의 log2 — LINEAR 해시에서 마스크 비트 수 결정
 * @m_index_function: 사용할 해시 함수 종류 (LINEAR_SET_FUNCTION 등의 enum)
 * @return: 0 ~ (m_nset - 1) 범위의 세트 인덱스
 *
 * GPU 캐시의 세트 인덱싱 방식을 선택적으로 구성할 수 있도록 5가지 해시 방식을 지원한다.
 * 단순 LINEAR 방식은 캐시 충돌(set thrashing)을 일으키기 쉬우므로, GPU의 규칙적인
 * 스트라이드 접근 패턴에서는 FERMI_HASH_SET_FUNCTION 같은 XOR 기반 해시가 유효하다.
 * 설정 파일의 -gpgpu_cache:dl1, -gpgpu_cache:dl2 등에서 'S' 파라미터로 지정.
 *
 * 지원 해시 함수:
 *   LINEAR_SET_FUNCTION    - addr[line_sz_log2 + nset_log2 - 1 : line_sz_log2] (단순 마스크)
 *   FERMI_HASH_SET_FUNCTION - XOR 기반 Fermi 해시 (HPCA 2014 논문 기반, 32/64 세트 전용)
 *   BITWISE_XORING_FUNCTION - 상위 비트와 하위 인덱스를 XOR (hashing.h의 bitwise_hash_function)
 *   HASH_IPOLY_FUNCTION    - iPoly 기반 해시 (hashing.h의 ipoly_hash_function)
 *   CUSTOM_SET_FUNCTION    - 사용자 정의 (현재 미구현)
 *
 * 호출 체인:
 *   cache_config::set_index() → [이 함수]
 *   l1d_cache_config::set_bank() → [이 함수]
 */
unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
  unsigned set_index = 0; /* [한국어] 계산된 세트 인덱스를 저장할 변수 — 기본값 0으로 초기화 */

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      /* [한국어] FERMI_HASH_SET_FUNCTION: Fermi GPU의 실제 세트 인덱싱을 모델링한 XOR 기반 해시.
       * 주소의 하위 비트(라인 내 인덱스)와 상위 비트(페이지 내 위치)를 XOR하여
       * 연속 스트라이드 접근에서 동일 세트로 집중되는 충돌을 줄인다. */
      unsigned lower_xor = 0; /* [한국어] XOR 연산의 하위 피연산자 — 라인 태그 하위 5비트 */
      unsigned upper_xor = 0; /* [한국어] XOR 연산의 상위 피연산자 — 상위 주소 비트들에서 추출 */

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F; /* [한국어] 라인 주소(오프셋 제거 후)의 하위 5비트 추출 */

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        /* [한국어] 0xE000 = 0b1110_0000_0000_0000 — 비트 13~15 추출 후 최하위로 이동 */
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        /* [한국어] 0x20000 = 비트 17 추출 후 비트 3 위치로 이동 */
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19
        /* [한국어] 0x80000 = 비트 19 추출 후 비트 4 위치로 이동 — 5비트 upper_xor 완성 */

        set_index = (lower_xor ^ upper_xor); /* [한국어] lower와 upper를 XOR하여 5비트 세트 인덱스 생성 */

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;
        /* [한국어] 64세트(48KB 캐시) 시 비트 12를 최상위 비트(bit 5)로 추가해 6비트 인덱스 완성 */

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0); /* [한국어] FERMI_HASH는 32 또는 64 세트에서만 유효 — 다른 세트 수는 지원 안 함 */
      }
      break;
    }

    case BITWISE_XORING_FUNCTION: {
      /* [한국어] BITWISE_XORING_FUNCTION: 상위 주소 비트와 기본 LINEAR 인덱스를
       * bitwise XOR 하여 세트 분산 효과를 높임 (hashing.h의 bitwise_hash_function 사용) */
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2); /* [한국어] 세트 인덱스 비트를 넘어서는 상위 비트들 추출 */
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1); /* [한국어] LINEAR 방식의 기본 세트 인덱스 계산 */
      set_index = bitwise_hash_function(higher_bits, index, m_nset); /* [한국어] 상위 비트와 LINEAR 인덱스를 XOR 조합 */
      break;
    }
    case HASH_IPOLY_FUNCTION: {
      /* [한국어] HASH_IPOLY_FUNCTION: iPoly(Irreducible Polynomial) 기반 해시.
       * GF(2^n) 위의 다항식 연산을 이용해 더 균등한 분산을 달성 (hashing.h의 ipoly_hash_function 사용) */
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2); /* [한국어] 세트 인덱스 상위 비트 추출 */
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1); /* [한국어] 기본 세트 인덱스 */
      set_index = ipoly_hash_function(higher_bits, index, m_nset); /* [한국어] iPoly 해시 적용 */
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      /* [한국어] 사용자 정의 해시 함수 — 현재 미구현. 연구 목적으로 직접 코드를 채워넣는 용도 */
      break;
    }

    case LINEAR_SET_FUNCTION: {
      /* [한국어] LINEAR_SET_FUNCTION: 가장 단순한 방식 — 주소에서 라인 오프셋 제거 후 하위 nset_log2 비트 추출.
       * 스트라이드 패턴에서 충돌이 많지만 구현이 단순하고 분석이 용이하다 */
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1); /* [한국어] [line_sz_log2 + nset_log2 - 1 : line_sz_log2] 비트 추출 */
      break;
    }

    default: {
      assert("\nUndefined set index function.\n" && 0); /* [한국어] 정의되지 않은 해시 함수 — 시뮬레이터 설정 오류 */
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");
  /* [한국어] 최종 세트 인덱스가 유효 범위(0 ~ m_nset-1) 내에 있는지 검증
   * 범위 초과 시 CUSTOM_SET_FUNCTION 미구현 또는 해시 함수 버그 가능성 */

  return set_index; /* [한국어] 계산된 세트 인덱스 반환 — 호출자(tag_array::probe)가 way 탐색에 사용 */
}

/*
 * [한국어]
 * l2_cache_config::init - L2 캐시 설정 초기화 및 주소 매핑 객체 연결
 *
 * @address_mapping: linear → raw 주소 변환기 포인터 (파티션 주소 계산용)
 *                   NULL이면 단순 linear 주소를 그대로 사용
 *
 * 상위 캐시 설정(cache_config::init)을 수행하고 L2에 특화된 주소 매핑 객체를
 * 저장한다. L2는 GPU의 여러 메모리 파티션(sub-partition)에 걸쳐 분산되어 있기
 * 때문에, 세트 인덱스 계산 시 파티션 비트를 제거해 set camping(특정 파티션의
 * 세트에 집중)을 방지한다.
 *
 * 호출 체인:
 *   gpu-sim.cc (전역 초기화) → [이 함수] → cache_config::init()
 */
void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone); /* [한국어] 설정 문자열 파싱 (세트 수, 결합도, 라인 크기 등) */
  m_address_mapping = address_mapping; /* [한국어] 파티션 주소 계산에 사용할 linear→raw 변환기 저장 */
}

/*
 * [한국어]
 * l2_cache_config::set_index - L2 캐시 세트 인덱스 계산 (파티션 주소 기반)
 *
 * @addr: 원본 선형 메모리 주소
 * @return: 파티션 비트를 제거한 주소 기반의 세트 인덱스 (0 ~ m_nset-1)
 *
 * L2 캐시는 여러 메모리 파티션에 걸쳐 분산 배치된다. address_mapping이 설정된
 * 경우 파티션 구분 비트를 제거한 파티션 주소(partition_address)를 기준으로 세트
 * 인덱스를 계산한다. 이렇게 하면 특정 파티션의 동일 세트에 요청이 몰리는
 * set camping 현상을 완화할 수 있다. address_mapping이 없으면 원본 주소를 사용.
 *
 * 호출 체인:
 *   tag_array::probe() → cache_config::set_index() → [이 함수] (L2에서 override)
 */
unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr; /* [한국어] 최초에는 원본 주소를 그대로 사용 */

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    /* [한국어] address_mapping이 설정된 경우 — 파티션 구분 비트를 제거한 파티션 주소로 변환
     * 이렇게 하면 파티션이 달라도 동일한 세트 인덱스를 얻어 L2 전체 세트를 고르게 활용 */
    part_addr = m_address_mapping->partition_address(addr); /* [한국어] 파티션 비트 제거 후의 주소 획득 */
  }

  return cache_config::set_index(part_addr); /* [한국어] 파티션 주소를 기준으로 세트 인덱스 계산 */
}

/*
 * [한국어]
 * tag_array::~tag_array - 태그 배열 소멸자: 캐시 라인 메모리 해제
 *
 * 생성자에서 동적 할당한 cache_block_t 객체들과 포인터 배열을 해제한다.
 * 커널 전환 또는 시뮬레이션 종료 시 호출된다. get_max_num_lines()는
 * 커널별로 가변적인 캐시 크기 변화를 수용하기 위해 최대 라인 수를 반환한다.
 *
 * 호출 체인:
 *   baseline_cache 소멸자 → [이 함수] → 각 cache_block_t 소멸자
 */
tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines(); /* [한국어] 최대 캐시 라인 수 (커널간 크기 변화 수용용) */
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i]; /* [한국어] 각 캐시 라인 객체 해제 (line_cache_block 또는 sector_cache_block) */
  delete[] m_lines; /* [한국어] 포인터 배열 자체 해제 */
}

/*
 * [한국어]
 * tag_array::tag_array (new_lines 버전) - 외부에서 할당된 캐시 라인 배열로 태그 배열 초기화
 *
 * @config: 캐시 설정 참조 (세트 수, 결합도, 라인 크기, 교체 정책 등)
 * @core_id: 이 태그 배열이 속한 SM(shader core) ID — 통계 로그용
 * @type_id: 캐시 타입 ID (L1D, L1T 등) — 통계 로그 구분용
 * @new_lines: 외부에서 미리 할당된 cache_block_t* 배열 포인터
 *
 * 이 생성자는 커널 전환 시 캐시 크기가 변경되어도 기존 메모리를 재사용하기 위해
 * 외부에서 할당된 라인 배열을 직접 받는 방식을 사용한다.
 *
 * 호출 체인:
 *   baseline_cache 생성자 → [이 함수] → tag_array::init()
 */
tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) { /* [한국어] 외부 라인 배열 직접 사용 — 소유권은 이 태그 배열로 이전 */
  init(core_id, type_id); /* [한국어] 통계 카운터 및 메타 정보 초기화 */
}

/*
 * [한국어]
 * tag_array::update_cache_parameters - 런타임에 캐시 설정 파라미터 갱신
 *
 * @config: 새 캐시 설정
 *
 * 커널 전환 시 캐시 크기나 결합도가 동적으로 변경될 때 호출된다.
 * CUDA의 shared memory / L1 cache 크기 조정(cudaDeviceSetCacheConfig 등)에 대응.
 *
 * 호출 체인:
 *   baseline_cache (커널 전환) → [이 함수]
 */
void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config; /* [한국어] 새 설정으로 캐시 파라미터 갱신 (세트 수, 결합도, 라인 크기 등) */
}

/*
 * [한국어]
 * tag_array::tag_array (일반 버전) - 새 캐시 라인 배열을 동적 할당하여 태그 배열 초기화
 *
 * @config: 캐시 설정 참조
 * @core_id: SM ID (통계 로그용)
 * @type_id: 캐시 타입 ID (통계 로그 구분용)
 *
 * config.m_cache_type에 따라 NORMAL(line_cache_block) 또는 SECTOR(sector_cache_block)
 * 타입의 캐시 블록 객체를 동적 할당한다. 섹터 캐시는 하나의 캐시 라인을 여러
 * 섹터(SECTOR_CHUNCK_SIZE개)로 나누어 독립적인 유효/수정 비트를 관리한다.
 *
 * 호출 체인:
 *   baseline_cache 생성자 → [이 함수] → tag_array::init()
 */
tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines(); /* [한국어] 최대 라인 수 계산 (세트 수 × 결합도, 커널간 최대값) */
  m_lines = new cache_block_t *[cache_lines_num]; /* [한국어] 캐시 블록 포인터 배열 동적 할당 */
  if (config.m_cache_type == NORMAL) {
    /* [한국어] 일반 라인 캐시: 각 엔트리가 전체 캐시 라인을 단일 상태로 관리 */
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block(); /* [한국어] 라인 단위 캐시 블록 생성 (INVALID 상태로 초기화) */
  } else if (config.m_cache_type == SECTOR) {
    /* [한국어] 섹터 캐시: 각 엔트리가 SECTOR_CHUNCK_SIZE개 섹터를 독립 상태 비트로 관리.
     * Volta 이후 GPU에서 사용되며, 섹터별 partial hit(SECTOR_MISS) 처리가 가능 */
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block(); /* [한국어] 섹터 단위 캐시 블록 생성 */
  } else
    assert(0); /* [한국어] NORMAL/SECTOR 외의 알 수 없는 캐시 타입 — 설정 오류 */

  init(core_id, type_id); /* [한국어] 통계 카운터(m_access, m_miss 등) 및 메타 정보 초기화 */
}

/*
 * [한국어]
 * tag_array::init - 태그 배열 통계 카운터 및 메타 정보 초기화
 *
 * @core_id: 이 태그 배열이 속한 SM ID — shader_cache_access_log() 호출 시 사용
 * @type_id: 캐시 타입 ID (L1D=0, Texture=1 등) — 통계 구분용
 *
 * 생성자에서 호출되며, 커널 전환 시 통계 리셋 목적으로도 재호출 가능하다.
 * snapshot 카운터는 AerialVision 시각화 도구의 윈도우 단위 미스율 계산에 사용된다.
 * is_used는 캐시가 실제로 접근된 적 있는지를 나타내며, flush/invalidate 시
 * 불필요한 순회를 방지하는 조기 종료 조건으로 활용된다.
 *
 * 호출 체인:
 *   tag_array 생성자 → [이 함수]
 */
void tag_array::init(int core_id, int type_id) {
  m_access = 0;        /* [한국어] 전체 캐시 접근 횟수 카운터 (HIT + MISS + RESERVATION_FAIL 포함) */
  m_miss = 0;          /* [한국어] 완전 미스(MISS) 횟수 — 태그 불일치 또는 라인 INVALID */
  m_pending_hit = 0;   /* [한국어] HIT_RESERVED 발생 횟수 — RESERVED 상태 라인에 대한 접근 */
  m_res_fail = 0;      /* [한국어] RESERVATION_FAIL 횟수 — 모든 라인이 RESERVED 상태일 때 */
  m_sector_miss = 0;   /* [한국어] SECTOR_MISS 횟수 — 태그 일치하나 해당 섹터만 INVALID */
  // initialize snapshot counters for visualizer
  /* [한국어] AerialVision 시각화 도구의 윈도우 단위 미스율 계산을 위한 스냅샷 카운터 초기화 */
  m_prev_snapshot_access = 0;       /* [한국어] 이전 윈도우 끝의 접근 수 — 현재 윈도우 접근 수 = m_access - 이 값 */
  m_prev_snapshot_miss = 0;         /* [한국어] 이전 윈도우 끝의 미스 수 */
  m_prev_snapshot_pending_hit = 0;  /* [한국어] 이전 윈도우 끝의 pending hit 수 */
  m_core_id = core_id; /* [한국어] SM(shader core) ID 저장 — shader_cache_access_log() 인자로 사용 */
  m_type_id = type_id; /* [한국어] 캐시 타입 ID 저장 — 통계 로그 구분용 */
  is_used = false;     /* [한국어] 캐시 사용 여부 플래그 — false이면 flush/invalidate를 조기 종료 */
  m_dirty = 0;         /* [한국어] MODIFIED 상태 라인(더티 라인) 수 — wr_percent 계산의 분자로 사용 */
}

/*
 * [한국어]
 * tag_array::add_pending_line - pending_lines 테이블에 블록 주소 등록
 *
 * @mf: 등록할 메모리 요청 패킷 포인터 (NULL 불가)
 *
 * 하위 메모리로 발송된 read 요청의 블록 주소를 pending_lines 맵에 기록한다.
 * pending_lines는 동일 블록에 대한 중복 요청을 감지하고 MSHR merge를 돕기 위한
 * 보조 자료구조이다. 이미 등록된 주소는 중복 등록하지 않는다.
 *
 * 호출 체인:
 *   baseline_cache::send_read_request() → [이 함수]
 */
void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf); /* [한국어] NULL 포인터 방어 — mf는 항상 유효한 요청이어야 함 */
  new_addr_type addr = m_config.block_addr(mf->get_addr()); /* [한국어] 바이트 주소를 블록(라인) 정렬 주소로 변환 */
  line_table::const_iterator i = pending_lines.find(addr); /* [한국어] 이미 등록된 주소인지 탐색 */
  if (i == pending_lines.end()) {
    /* [한국어] 신규 주소인 경우에만 등록 — 동일 블록의 중복 요청은 MSHR merge로 처리되므로 여기서는 첫 등록만 */
    pending_lines[addr] = mf->get_inst().get_uid(); /* [한국어] 블록 주소 → 명령어 UID 매핑 저장 */
  }
}

/*
 * [한국어]
 * tag_array::remove_pending_line - pending_lines 테이블에서 블록 주소 제거
 *
 * @mf: 제거할 메모리 요청 패킷 포인터 (NULL 불가)
 *
 * fill() 완료 후 해당 블록 주소를 pending_lines에서 삭제한다.
 * 이미 삭제된 주소나 등록되지 않은 주소에 대해서는 아무 동작도 하지 않는다.
 *
 * 호출 체인:
 *   baseline_cache::fill() 완료 후 → [이 함수]
 */
void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf); /* [한국어] NULL 포인터 방어 */
  new_addr_type addr = m_config.block_addr(mf->get_addr()); /* [한국어] 블록 정렬 주소 계산 */
  line_table::const_iterator i = pending_lines.find(addr); /* [한국어] 해당 주소가 pending_lines에 존재하는지 탐색 */
  if (i != pending_lines.end()) {
    pending_lines.erase(addr); /* [한국어] 해당 블록 주소 엔트리 삭제 — fill 완료로 pending 상태 해소 */
  }
}

/*
 * [한국어]
 * tag_array::probe (mem_fetch 버전) - mem_fetch에서 섹터 마스크를 추출하여 probe 위임
 *
 * @addr: 탐색할 메모리 주소 (블록 정렬 주소)
 * @idx: [출력] 찾은 캐시 라인 인덱스 (교체 후보 또는 히트 라인)
 * @mf: 메모리 요청 패킷 — 섹터 마스크와 쓰기 여부 추출에 사용
 * @is_write: 쓰기 요청 여부
 * @probe_mode: true이면 실제 상태 변경 없이 탐색만 수행 (data_cache::access에서 사용)
 * @return: HIT / HIT_RESERVED / MISS / RESERVATION_FAIL / SECTOR_MISS
 *
 * mem_fetch에서 섹터 마스크를 추출하여 mask 버전의 probe()로 위임하는 래퍼 함수.
 *
 * 호출 체인:
 *   tag_array::access() → [이 함수] → tag_array::probe(mask 버전)
 *   data_cache::access() → [이 함수] → tag_array::probe(mask 버전)
 */
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask(); /* [한국어] 요청이 접근하는 섹터 비트마스크 추출 */
  return probe(addr, idx, mask, is_write, probe_mode, mf); /* [한국어] 실제 탐색은 mask 버전 probe에 위임 */
}

/*
 * [한국어]
 * tag_array::probe (mask 버전) - 태그 배열에서 주소 탐색 및 교체 후보 선택
 *
 * @addr: 탐색할 메모리 주소 (블록 정렬 주소)
 * @idx: [출력] 찾은 캐시 라인 인덱스 (HIT 시 해당 way 인덱스, MISS 시 교체 후보 인덱스)
 * @mask: 접근하는 섹터들의 비트마스크 (섹터 캐시에서 섹터 단위 상태 확인)
 * @is_write: 쓰기 요청 여부 — MODIFIED 라인 HIT 판정에 영향
 * @probe_mode: true이면 LRU 등 상태 변경 없이 탐색만 (통계 기록 목적)
 * @mf: 메모리 요청 패킷 (NULL 가능) — 현재는 직접 사용 안 함
 * @return: HIT / HIT_RESERVED / MISS / RESERVATION_FAIL / SECTOR_MISS
 *
 * 태그 배열의 핵심 탐색 함수. 동작 과정:
 * 1. 세트 인덱스와 태그를 주소에서 계산
 * 2. 해당 세트의 모든 way를 순회하며 태그 일치 여부 확인
 * 3. 태그 일치 시 섹터/라인 상태(RESERVED/VALID/MODIFIED/INVALID)에 따라 결과 결정:
 *    - RESERVED → HIT_RESERVED (해당 섹터가 이미 하위 메모리에서 대기 중)
 *    - VALID → HIT
 *    - MODIFIED + readable → HIT, MODIFIED + !readable → SECTOR_MISS
 *    - 다른 섹터는 VALID이나 해당 섹터만 INVALID → SECTOR_MISS
 * 4. 태그 불일치 시 교체 후보(invalid_line 또는 LRU/FIFO valid_line) 추적
 *    - m_wr_percent 임계값 미만일 때만 MODIFIED 라인을 교체 후보로 허용
 *    (더티 라인이 적으면 클린 라인만 교체 — writeback 트래픽 제어)
 * 5. 모든 라인이 RESERVED이면 RESERVATION_FAIL (ON_MISS 정책에서만 가능)
 * 6. 교체 후보를 idx에 설정 후 MISS 반환
 *
 * 호출 체인:
 *   tag_array::access() → [이 함수]
 *   data_cache::access() → [이 함수] (probe_mode=true)
 *   baseline_cache::fill() → tag_array::fill() → [이 함수]
 */
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  // assert( m_config.m_write_policy == READ_ONLY );
  unsigned set_index = m_config.set_index(addr); /* [한국어] 주소에서 세트 인덱스 계산 (hash_function 경유) */
  new_addr_type tag = m_config.tag(addr);        /* [한국어] 주소에서 태그 비트 추출 (세트 인덱스 + 라인 오프셋 비트 제거) */

  unsigned invalid_line = (unsigned)-1;          /* [한국어] 발견된 INVALID 라인의 인덱스 — (unsigned)-1은 "미발견" 센티넬 */
  unsigned valid_line = (unsigned)-1;            /* [한국어] 발견된 교체 후보 VALID/MODIFIED 라인의 인덱스 */
  unsigned long long valid_timestamp = (unsigned)-1; /* [한국어] LRU/FIFO 교체 후보의 타임스탬프 — 가장 오래된 라인 추적 */

  bool all_reserved = true; /* [한국어] 세트 내 모든 라인이 RESERVED인지 여부 — false이면 교체 가능한 라인 존재 */
  // check for hit or pending hit
  for (unsigned way = 0; way < m_config.m_assoc; way++) { /* [한국어] 세트 내 모든 way 순회 (결합도만큼 반복) */
    unsigned index = set_index * m_config.m_assoc + way; /* [한국어] 태그 배열 내 절대 인덱스 = 세트 인덱스 × 결합도 + way */
    cache_block_t *line = m_lines[index]; /* [한국어] 해당 캐시 라인 포인터 획득 */
    if (line->m_tag == tag) {
      /* [한국어] 태그 일치 — 이 라인에 요청 주소가 매핑될 가능성 있음 */
      if (line->get_status(mask) == RESERVED) {
        /* [한국어] 요청 섹터가 RESERVED 상태: 이미 하위 메모리로 요청이 나가 있고 응답 대기 중
         * 이 요청은 MSHR에 merge되어야 함 */
        idx = index; /* [한국어] 해당 라인 인덱스를 출력으로 전달 */
        return HIT_RESERVED; /* [한국어] 캐시 히트이지만 데이터 아직 미도착 — 파이프라인 재시도 */
      } else if (line->get_status(mask) == VALID) {
        /* [한국어] 요청 섹터가 VALID 상태: 데이터가 캐시에 존재하고 읽기 가능 */
        idx = index;
        return HIT; /* [한국어] 완전한 캐시 히트 */
      } else if (line->get_status(mask) == MODIFIED) {
        /* [한국어] 요청 섹터가 MODIFIED(더티) 상태: 쓰기는 항상 HIT, 읽기는 readable 여부에 따라 결정 */
        if ((!is_write && line->is_readable(mask)) || is_write) {
          /* [한국어] 읽기이면서 readable(byte mask가 충분히 채워진) 상태이거나 쓰기 요청이면 HIT */
          idx = index;
          return HIT;
        } else {
          /* [한국어] 읽기이지만 아직 해당 섹터의 모든 바이트가 채워지지 않은 경우 — 부분 채움 상태
           * SECTOR_MISS를 반환해 나머지 데이터를 하위 메모리에서 fetch하도록 유도 */
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        /* [한국어] 라인의 다른 섹터는 VALID이나 요청 섹터만 INVALID — SECTOR_MISS
         * 태그는 일치하지만 해당 섹터의 데이터가 없는 섹터 캐시 특유의 상태 */
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID); /* [한국어] 라인 전체가 INVALID — 태그 일치해도 유효하지 않음 */
      }
    }
    if (!line->is_reserved_line()) {
      /* [한국어] RESERVED 상태가 아닌 라인 — 교체 후보 가능성 있음 */
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      /* [한국어] 전체 캐시에서 더티(MODIFIED) 라인 비율(%) 계산
       * m_dirty는 MODIFIED 상태 라인 수; m_wr_percent(설정 파라미터)와 비교해
       * 더티 라인 비율이 낮을 때는 클린 라인만 교체해 writeback 트래픽 억제 */
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty cachelines
      // reach the limit.
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        /* [한국어] 클린 라인이거나, 더티 비율이 m_wr_percent 임계값 이상이면 교체 후보 허용
         * 이 조건이 false이면 MODIFIED 라인은 교체 후보에서 제외 (writeback 트래픽 제어) */
        all_reserved = false; /* [한국어] 교체 가능한 라인이 존재함을 표시 — RESERVATION_FAIL 방지 */
        if (line->is_invalid_line()) {
          invalid_line = index; /* [한국어] INVALID 라인 발견 — 가장 우선되는 교체 후보로 기록 */
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            /* [한국어] LRU(Least Recently Used) 교체 정책: 가장 오래된 접근 시간을 가진 라인 선택 */
            if (line->get_last_access_time() < valid_timestamp) {
              valid_timestamp = line->get_last_access_time(); /* [한국어] 현재까지 발견된 가장 오래된 접근 시간 갱신 */
              valid_line = index; /* [한국어] 현재 LRU 교체 후보 인덱스 갱신 */
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            /* [한국어] FIFO(First In, First Out) 교체 정책: 가장 오래된 할당 시간을 가진 라인 선택 */
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time(); /* [한국어] 가장 오래된 할당 시간 갱신 */
              valid_line = index; /* [한국어] 현재 FIFO 교체 후보 인덱스 갱신 */
            }
          }
        }
      }
    }
  }
  if (all_reserved) {
    /* [한국어] 세트 내 모든 라인이 RESERVED 상태 — 교체 가능한 라인이 없음
     * ON_MISS 정책에서만 발생 가능 (ON_FILL은 RESERVED 상태를 만들지 않음) */
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  if (invalid_line != (unsigned)-1) {
    idx = invalid_line; /* [한국어] INVALID 라인이 있으면 최우선 교체 후보로 선택 — writeback 불필요 */
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line; /* [한국어] INVALID 라인 없으면 LRU/FIFO가 선택한 VALID 라인을 교체 후보로 선택 */
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable
              /* [한국어] all_reserved=false인데 교체 후보가 없는 모순 상태 — 논리 오류 */

  return MISS; /* [한국어] 태그 불일치(진짜 미스) — idx에 교체 후보 인덱스를 설정하고 MISS 반환 */
}

/*
 * [한국어]
 * tag_array::access (writeback 없는 버전) - 태그 접근 래퍼 (writeback 정보 필요 없는 경우용)
 *
 * @addr: 접근 대상 블록 주소
 * @time: 현재 시뮬레이션 사이클 — LRU 타임스탬프 갱신에 사용
 * @idx: [출력] 접근한 캐시 라인 인덱스
 * @mf: 메모리 요청 패킷
 * @return: cache_request_status (HIT/MISS/HIT_RESERVED/RESERVATION_FAIL/SECTOR_MISS)
 *
 * writeback 정보가 필요 없는 읽기 전용 캐시(read_only_cache) 등에서 사용하는 간소화 버전.
 * 내부적으로 wb=false로 호출하며, MISS 시 writeback이 발생하면 assert로 오류 처리.
 *
 * 호출 체인:
 *   read_only_cache::access() → [이 함수] → tag_array::access(wb 버전)
 *   baseline_cache::send_read_request() → [이 함수] (read_only=true 경우)
 */
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;        /* [한국어] writeback 발생 여부 — 이 버전에서는 발생하면 안 됨 */
  evicted_block_info evicted; /* [한국어] evict된 라인 정보 — 이 버전에서는 사용하지 않음 */
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf); /* [한국어] 실제 접근 처리 위임 */
  assert(!wb); /* [한국어] writeback이 발생해서는 안 됨 — 읽기 전용 또는 LRU만 갱신하는 경우 */
  return result;
}

/*
 * [한국어]
 * tag_array::access (wb 버전) - 태그 배열 접근 + LRU 갱신 + ON_MISS 시 라인 할당
 *
 * @addr: 접근 대상 블록 주소
 * @time: 현재 시뮬레이션 사이클 — LRU 타임스탬프 업데이트에 사용
 * @idx: [출력] 접근한 캐시 라인 인덱스 (probe가 결정)
 * @wb: [출력] writeback이 필요한 경우 true로 설정됨 (MODIFIED 라인 교체 시)
 * @evicted: [출력] evict된 라인의 블록 주소, 수정 크기, byte/sector 마스크 정보
 * @mf: 메모리 요청 패킷 (접근 타입, 섹터 마스크, 쓰기 여부 등)
 * @return: cache_request_status (통계 및 다음 동작 결정에 사용)
 *
 * tag_array의 핵심 접근 함수. 동작 단계:
 * 1. m_access 카운터 증가, is_used=true, AerialVision 로그 기록
 * 2. probe()로 태그 탐색 — HIT/MISS/HIT_RESERVED/RESERVATION_FAIL/SECTOR_MISS 결정
 * 3. 결과별 처리:
 *    - HIT_RESERVED: m_pending_hit++ (이후 HIT 처리로 fall-through)
 *    - HIT: LRU 타임스탬프 갱신 (set_last_access_time)
 *    - MISS: m_miss++, ON_MISS 정책이면 교체 후보 라인 즉시 allocate.
 *            교체 대상이 MODIFIED이면 wb=true로 설정 + evicted 정보 저장, m_dirty--
 *    - SECTOR_MISS: m_sector_miss++, ON_MISS이면 해당 섹터만 allocate_sector
 *    - RESERVATION_FAIL: m_res_fail++ (할당 없음)
 *
 * ON_MISS vs ON_FILL:
 *   ON_MISS: 미스 탐지 즉시 라인 할당(RESERVED 상태) — 이 함수에서 처리
 *   ON_FILL: 하위 메모리 응답이 돌아왔을 때 fill()에서 할당 — 이 함수에서는 할당 안 함
 *
 * 호출 체인:
 *   data_cache::wr_hit_wb/wt() → [이 함수] (LRU 갱신용)
 *   data_cache::rd_hit_base() → [이 함수] (LRU 갱신용)
 *   baseline_cache::send_read_request() → [이 함수] (MISS 시 라인 할당)
 */
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  m_access++;   /* [한국어] 전체 접근 횟수 증가 — 미스율 계산의 분모 */
  is_used = true; /* [한국어] 캐시가 실제로 사용됨을 표시 — flush/invalidate 최적화에 활용 */
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  /* [한국어] AerialVision 시각화 도구에 캐시 접근 이벤트 기록 (is_miss=0 → 접근) */
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write()); /* [한국어] 태그 탐색 실행 — 결과와 교체 후보 idx 결정 */
  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++; /* [한국어] HIT_RESERVED 횟수 증가 — RESERVED 상태 라인 접근 */
    case HIT:
      /* [한국어] HIT_RESERVED도 여기로 fall-through — LRU 타임스탬프 갱신 공통 처리 */
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask()); /* [한국어] LRU를 위한 마지막 접근 시간 갱신 */
      break;
    case MISS:
      m_miss++;   /* [한국어] 완전 미스 횟수 증가 */
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      /* [한국어] AerialVision에 미스 이벤트 기록 (is_miss=1) */
      if (m_config.m_alloc_policy == ON_MISS) {
        /* [한국어] ON_MISS 정책: 미스 탐지 즉시 교체 후보 라인 할당 (RESERVED 상태로 변경) */
        if (m_lines[idx]->is_modified_line()) {
          /* [한국어] 교체 대상 라인이 MODIFIED(더티) — writeback 필요 */
          wb = true;  /* [한국어] 호출자에게 writeback 필요 신호 전달 */
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,    /* [한국어] evict된 라인의 블록 주소 */
                           m_lines[idx]->get_modified_size(),  /* [한국어] writeback 대상 수정 데이터 크기 */
                           m_lines[idx]->get_dirty_byte_mask(),  /* [한국어] 수정된 바이트 위치 비트마스크 */
                           m_lines[idx]->get_dirty_sector_mask()); /* [한국어] 수정된 섹터 비트마스크 */
          m_dirty--;  /* [한국어] MODIFIED 라인이 evict되므로 더티 카운터 감소 */
        }
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
        /* [한국어] 새 라인 할당: 태그, 블록 주소, 할당 시간, 섹터 마스크 설정 → 상태 RESERVED로 변경 */
      }
      break;
    case SECTOR_MISS:
      /* [한국어] SECTOR_MISS는 섹터 캐시에서만 발생 — 태그 일치하나 요청 섹터가 INVALID */
      assert(m_config.m_cache_type == SECTOR); /* [한국어] 섹터 캐시 타입이 아닌데 SECTOR_MISS는 불가 */
      m_sector_miss++;  /* [한국어] 섹터 미스 횟수 증가 */
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      /* [한국어] AerialVision에 미스 이벤트 기록 */
      if (m_config.m_alloc_policy == ON_MISS) {
        /* [한국어] ON_MISS: 섹터 미스 즉시 해당 섹터만 RESERVED로 할당 */
        bool before = m_lines[idx]->is_modified_line(); /* [한국어] 섹터 할당 전 라인의 MODIFIED 여부 저장 */
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        /* [한국어] sector_cache_block으로 다운캐스팅하여 해당 섹터만 RESERVED 상태로 할당 */
        if (before && !m_lines[idx]->is_modified_line()) {
          /* [한국어] 섹터 할당으로 인해 라인의 MODIFIED 상태가 해소된 경우 더티 카운터 감소 */
          m_dirty--;
        }
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;   /* [한국어] 예약 실패 횟수 증가 — 이 사이클에 처리 불가 */
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      /* [한국어] AerialVision에 미스(실패) 이벤트 기록 */
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status); /* [한국어] 알 수 없는 상태 — 에러 메시지 출력 후 abort */
      abort();
  }
  return status; /* [한국어] 최종 결과 반환 — 호출자(data_cache::process_tag_probe 등)가 후속 처리에 사용 */
}

/*
 * [한국어]
 * tag_array::fill (mem_fetch 버전) - mem_fetch의 섹터/바이트 마스크로 fill 위임
 *
 * @addr: fill 대상 블록 주소
 * @time: 현재 시뮬레이션 사이클
 * @mf: 하위 메모리에서 돌아온 요청 패킷 (섹터/바이트 마스크 포함)
 * @is_write: 쓰기 응답 여부 (MODIFIED 상태로 fill 여부 결정)
 *
 * mem_fetch에서 마스크를 추출하여 mask 버전 fill()로 위임하는 래퍼.
 * ON_FILL 정책에서 사용되며, baseline_cache::fill()이 이를 호출한다.
 *
 * 호출 체인:
 *   baseline_cache::fill() → [이 함수] → tag_array::fill(mask 버전)
 */
void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(), /* [한국어] mem_fetch에서 마스크 추출 후 위임 */
       is_write);
}

/*
 * [한국어]
 * tag_array::fill (mask 버전) - ON_FILL 정책에서 하위 메모리 응답으로 태그 배열 채우기
 *
 * @addr: fill 대상 블록 주소
 * @time: 현재 시뮬레이션 사이클 — 할당 및 접근 타임스탬프 갱신
 * @mask: fill 대상 섹터들의 비트마스크
 * @byte_mask: fill 대상 바이트들의 비트마스크 (dirty byte 추적용)
 * @is_write: true이면 MODIFIED 상태로 fill
 *
 * ON_FILL 정책: 하위 메모리 응답 도착 시점에 라인을 할당하고 데이터를 채운다.
 * 동작 단계:
 * 1. probe()로 교체 후보 라인 탐색
 * 2. RESERVATION_FAIL이면 즉시 반환 (교체 불가 — 희귀 상황)
 * 3. MISS이면 전체 라인 allocate, SECTOR_MISS이면 해당 섹터만 allocate_sector
 * 4. allocate 과정에서 MODIFIED 해소 시 m_dirty 감소
 * 5. fill()로 라인에 데이터 채우기 — 쓰기이면 MODIFIED 상태 설정
 * 6. fill 결과 MODIFIED 상태가 되면 m_dirty 증가
 *
 * 호출 체인:
 *   tag_array::fill(mem_fetch 버전) → [이 함수]
 */
void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx; /* [한국어] probe가 선택한 교체 후보 또는 히트 라인의 인덱스 */
  enum cache_request_status status = probe(addr, idx, mask, is_write); /* [한국어] 태그 탐색 — ON_FILL에서는 아직 라인이 없으므로 MISS 또는 SECTOR_MISS 예상 */

  if (status == RESERVATION_FAIL) {
    /* [한국어] 교체 가능한 라인이 없는 희귀 상황 — 이 fill은 버려짐 (상위에서 재시도 필요) */
    return;
  }

  bool before = m_lines[idx]->is_modified_line(); /* [한국어] allocate 전 라인의 MODIFIED 상태 저장 — allocate로 인한 상태 변화 감지용 */
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    /* [한국어] 완전 미스: 전체 라인을 새로 할당 (ON_FILL 정책) */
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                           mask); /* [한국어] 태그, 블록 주소, 시간, 섹터 마스크로 라인 초기화 */
  } else if (status == SECTOR_MISS) {
    /* [한국어] 섹터 미스: 라인은 이미 존재하나 해당 섹터만 INVALID — 섹터만 추가 할당 */
    assert(m_config.m_cache_type == SECTOR); /* [한국어] SECTOR_MISS는 섹터 캐시 타입에서만 가능 */
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask); /* [한국어] 해당 섹터만 RESERVED로 할당 */
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    /* [한국어] allocate로 인해 MODIFIED 상태가 해소된 경우(클린 라인으로 덮어씀) 더티 카운터 감소 */
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line(); /* [한국어] fill 전의 MODIFIED 상태 재확인 — fill 후 dirty 증가 감지용 */
  m_lines[idx]->fill(time, mask, byte_mask); /* [한국어] 실제 데이터 채우기 — is_write이면 MODIFIED 상태로 설정 */
  if (m_lines[idx]->is_modified_line() && !before) {
    /* [한국어] fill 결과 MODIFIED 상태가 되었으면(write fill) 더티 카운터 증가 */
    m_dirty++;
  }
}

/*
 * [한국어]
 * tag_array::fill (index 버전) - ON_MISS 정책에서 MSHR 응답으로 기존 예약 라인 채우기
 *
 * @index: 미리 할당된(RESERVED 상태) 캐시 라인의 절대 인덱스
 * @time: 현재 시뮬레이션 사이클
 * @mf: 하위 메모리에서 돌아온 요청 패킷 (섹터/바이트 마스크)
 *
 * ON_MISS 정책 전용: access() 시점에 이미 라인이 RESERVED로 할당되었으므로,
 * 하위 메모리 응답이 돌아오면 해당 인덱스의 라인에 직접 데이터를 채운다.
 * probe()를 통한 교체 후보 탐색 없이 index를 직접 사용하므로 더 빠르다.
 * MODIFIED 상태가 새로 생기면 m_dirty 카운터를 증가시킨다.
 *
 * 호출 체인:
 *   baseline_cache::fill() → [이 함수] (ON_MISS 정책 분기)
 */
void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  assert(m_config.m_alloc_policy == ON_MISS); /* [한국어] ON_MISS 정책에서만 index 직접 사용 가능 */
  bool before = m_lines[index]->is_modified_line(); /* [한국어] fill 전 MODIFIED 상태 저장 — 더티 카운터 변화 감지용 */
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask()); /* [한국어] 지정 인덱스 라인에 섹터/바이트 마스크 기반으로 데이터 채우기 */
  if (m_lines[index]->is_modified_line() && !before) {
    /* [한국어] fill 결과 새로 MODIFIED 상태가 된 경우(write fill) 더티 카운터 증가 */
    m_dirty++;
  }
}

// TODO: we need write back the flushed data to the upper level
/*
 * [한국어]
 * tag_array::flush - MODIFIED(더티) 라인만 선택적으로 무효화 (flush)
 *
 * is_used 플래그가 false이면 조기 종료하여 불필요한 순회를 방지한다.
 * 더티 라인을 상위 레벨로 writeback하는 기능은 TODO로 남아있다.
 * flush는 커널 종료 또는 캐시 사이즈 변경 시 더티 상태를 정리하기 위해 사용된다.
 * SECTOR_CHUNCK_SIZE개의 섹터를 개별적으로 INVALID로 설정하여 섹터 캐시도 지원한다.
 *
 * 주의: 현재 구현은 더티 데이터를 버리므로 write-back이 필요한 캐시에서는 데이터 손실 가능.
 *
 * 호출 체인:
 *   baseline_cache (커널 전환 또는 명시적 flush) → [이 함수]
 */
void tag_array::flush() {
  if (!is_used) return; /* [한국어] 한 번도 접근되지 않은 캐시는 순회 불필요 — 조기 종료 */

  for (unsigned i = 0; i < m_config.get_num_lines(); i++) /* [한국어] 모든 캐시 라인 순회 */
    if (m_lines[i]->is_modified_line()) {
      /* [한국어] MODIFIED(더티) 라인만 대상으로 — VALID/RESERVED/INVALID 라인은 이미 클린이므로 건너뜀 */
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        /* [한국어] 모든 섹터를 INVALID로 설정 — 섹터별로 개별 처리하여 섹터 캐시도 지원 */
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j)); /* [한국어] j번 섹터 하나만 선택하는 마스크 생성 후 INVALID 설정 */
      }
    }

  m_dirty = 0;     /* [한국어] 더티 카운터 초기화 — 모든 더티 라인이 무효화됨 */
  is_used = false; /* [한국어] 캐시 사용 플래그 초기화 — 다음 flush/invalidate 시 조기 종료 허용 */
}

/*
 * [한국어]
 * tag_array::invalidate - 전체 캐시 라인 무효화 (클린/더티 구분 없이)
 *
 * flush()와 달리 MODIFIED 여부를 확인하지 않고 모든 라인의 모든 섹터를 INVALID로 설정.
 * 컨텍스트 스위치, 커널 간 캐시 일관성 유지, 또는 특정 메모리 영역 무효화에 사용된다.
 * 더티 데이터 writeback 없이 강제 무효화이므로 캐시 일관성이 상위에서 이미 보장된
 * 경우에만 안전하게 사용할 수 있다.
 *
 * 호출 체인:
 *   baseline_cache (커널 종료 또는 명시적 invalidate) → [이 함수]
 */
void tag_array::invalidate() {
  if (!is_used) return; /* [한국어] 사용된 적 없는 캐시는 조기 종료 */

  for (unsigned i = 0; i < m_config.get_num_lines(); i++) /* [한국어] 모든 캐시 라인 순회 */
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)    /* [한국어] 모든 섹터 순회 */
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j)); /* [한국어] 해당 섹터를 INVALID로 강제 설정 */

  m_dirty = 0;     /* [한국어] 더티 카운터 초기화 */
  is_used = false; /* [한국어] 사용 플래그 초기화 */
}

/*
 * [한국어]
 * tag_array::windowed_miss_rate - 현재 윈도우 내 미스율 계산 (AerialVision 시각화용)
 *
 * @return: 현재 윈도우(이전 스냅샷 이후) 동안의 미스율 (0.0 ~ 1.0)
 *
 * new_window() 호출 이후부터 현재까지의 접근 수와 미스 수를 사용해 미스율을 계산한다.
 * 전체 누적 미스율이 아니라 특정 시간 구간의 미스율을 제공하여 캐시 동작을
 * 시간적으로 분석할 수 있게 한다. AerialVision 실시간 가시화 도구에서 사용된다.
 * 접근 수가 0이면 0.0 반환 (나눗셈 오류 방지).
 *
 * 호출 체인:
 *   AerialVision 시각화 루프 → [이 함수]
 */
float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access; /* [한국어] 현재 윈도우 내 접근 수 = 현재 접근 수 - 이전 스냅샷 접근 수 */
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss; /* [한국어] 현재 윈도우 내 미스 수 (MISS + SECTOR_MISS 합계) */
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f; /* [한국어] 미스율 초기값 0 — 접근이 없을 때 반환될 기본값 */
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  /* [한국어] 접근이 있을 때만 미스율 계산 — n_miss에 이미 m_sector_miss가 포함되므로 중복 가산 주의 (버그 가능성) */
  return missrate;
}

/*
 * [한국어]
 * tag_array::new_window - 미스율 계산 윈도우 갱신 (스냅샷 카운터 업데이트)
 *
 * 현재 누적 카운터 값을 스냅샷으로 저장하여 다음 windowed_miss_rate() 호출 시
 * 이 시점 이후의 접근만을 대상으로 미스율을 계산하도록 기준점을 갱신한다.
 * AerialVision의 주기적 통계 수집 루프에서 매 주기 시작 시 호출된다.
 *
 * 호출 체인:
 *   AerialVision 시각화 루프 (주기 시작) → [이 함수]
 */
void tag_array::new_window() {
  m_prev_snapshot_access = m_access;            /* [한국어] 현재 접근 수를 스냅샷으로 저장 */
  m_prev_snapshot_miss = m_miss;                /* [한국어] 현재 미스 수를 스냅샷으로 저장 (이 줄은 아래 줄에 덮어씌워짐 — 불필요) */
  m_prev_snapshot_miss = m_miss + m_sector_miss; /* [한국어] MISS + SECTOR_MISS 합계를 스냅샷으로 저장 */
  m_prev_snapshot_pending_hit = m_pending_hit;  /* [한국어] 현재 pending hit 수를 스냅샷으로 저장 */
}

/*
 * [한국어]
 * tag_array::print - 태그 배열 통계를 파일 스트림에 출력
 *
 * @stream: 출력 대상 파일 스트림
 * @total_access: [입출력] 누적 접근 수 — 이 함수 호출 후 m_access만큼 증가
 * @total_misses: [입출력] 누적 미스 수 — 이 함수 호출 후 (m_miss+m_sector_miss)만큼 증가
 *
 * 시뮬레이션 결과 리포트에서 각 캐시의 접근/미스/섹터미스/펜딩히트 통계를 출력한다.
 * baseline_cache::print()가 호출하며 SM 단위 통계 집계에 사용된다.
 *
 * 호출 체인:
 *   baseline_cache::print() → [이 함수]
 */
void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream); /* [한국어] 캐시 설정(세트 수, 결합도, 라인 크기, 정책 등) 출력 */
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  /* [한국어] 접근 수, 순수 미스, 섹터 미스, 전체 미스(합), 미스율, 펜딩 히트, 펜딩 히트율 출력 */
  total_misses += (m_miss + m_sector_miss); /* [한국어] 상위 집계 변수에 이 태그 배열의 미스 수 누적 */
  total_access += m_access;                /* [한국어] 상위 집계 변수에 이 태그 배열의 접근 수 누적 */
}

/*
 * [한국어]
 * tag_array::get_stats - 태그 배열 통계를 변수로 반환 (프로그래밍 방식 집계용)
 *
 * @total_access: [출력] 전체 접근 수
 * @total_misses: [출력] 전체 미스 수 (MISS + SECTOR_MISS)
 * @total_hit_res: [출력] HIT_RESERVED 횟수 (pending hit)
 * @total_res_fail: [출력] RESERVATION_FAIL 횟수
 *
 * 출력 리포트 대신 다른 코드에서 통계 값을 직접 읽어야 할 때 사용하는 접근자.
 * gpu-sim.cc의 통계 집계나 Accel-Sim 프레임워크와의 연동에서 활용된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (통계 집계) → [이 함수]
 */
void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;                 /* [한국어] 전체 접근 횟수 반환 */
  total_misses = (m_miss + m_sector_miss); /* [한국어] MISS와 SECTOR_MISS 합계 반환 */
  total_hit_res = m_pending_hit;           /* [한국어] HIT_RESERVED(pending hit) 횟수 반환 */
  total_res_fail = m_res_fail;             /* [한국어] RESERVATION_FAIL 횟수 반환 */
}

/*
 * [한국어]
 * was_write_sent - 이벤트 목록에 WRITE_REQUEST_SENT 이벤트가 있는지 확인
 *
 * @events: 캐시 접근 중 발생한 이벤트 목록 (write/read/writeback 등)
 * @return: WRITE_REQUEST_SENT 이벤트가 하나 이상 있으면 true
 *
 * cache access 결과 목록에서 실제로 하위 메모리로 쓰기 요청이 발송되었는지 확인한다.
 * bandwidth_management::use_data_port() 등에서 이 결과를 토대로 포트 사용량을 결정한다.
 *
 * 호출 체인:
 *   bandwidth_management::use_data_port() → [이 함수]
 */
bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true; /* [한국어] WRITE_REQUEST_SENT 이벤트 발견 — 쓰기 요청이 발송됨 */
  }
  return false; /* [한국어] 목록에 없으면 false 반환 */
}

/*
 * [한국어]
 * was_writeback_sent - 이벤트 목록에 WRITE_BACK_REQUEST_SENT 이벤트가 있는지 확인
 *
 * @events: 캐시 접근 중 발생한 이벤트 목록
 * @wb_event: [출력] 발견된 writeback 이벤트 (evicted 블록 정보 포함)
 * @return: WRITE_BACK_REQUEST_SENT 이벤트가 있으면 true
 *
 * 더티 라인 교체 시 발생하는 writeback 이벤트를 확인하고, 해당 이벤트의
 * evicted 블록 정보(수정 크기 등)를 wb_event로 전달한다.
 * bandwidth_management에서 데이터 포트 사용량 계산에 활용된다.
 *
 * 호출 체인:
 *   bandwidth_management::use_data_port() → [이 함수]
 */
bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e; /* [한국어] writeback 이벤트 복사 — evicted 블록의 수정 크기 정보를 호출자에게 전달 */
      return true;
    }
  }
  return false;
}

/*
 * [한국어]
 * was_read_sent - 이벤트 목록에 READ_REQUEST_SENT 이벤트가 있는지 확인
 *
 * @events: 캐시 접근 중 발생한 이벤트 목록
 * @return: READ_REQUEST_SENT 이벤트가 하나 이상 있으면 true
 *
 * 캐시 미스 또는 write-allocate 처리 중 하위 메모리로 읽기 요청이 발송되었는지 확인한다.
 * 상위 모듈에서 결과 처리 분기에 사용된다.
 *
 * 호출 체인:
 *   shader.cc (ldst_unit) → [이 함수] (캐시 이벤트 결과 분석)
 */
bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true; /* [한국어] READ_REQUEST_SENT 이벤트 발견 */
  }
  return false;
}

/*
 * [한국어]
 * was_writeallocate_sent - 이벤트 목록에 WRITE_ALLOCATE_SENT 이벤트가 있는지 확인
 *
 * @events: 캐시 접근 중 발생한 이벤트 목록
 * @return: WRITE_ALLOCATE_SENT 이벤트가 하나 이상 있으면 true
 *
 * write-allocate 처리 중 allocate 요청이 발송되었는지 확인한다.
 * wr_miss_wa_naive/fetch_on_write/lazy_fetch_on_read 함수들이 이 이벤트를 생성한다.
 *
 * 호출 체인:
 *   shader.cc (ldst_unit) → [이 함수] (write-allocate 이벤트 분석)
 */
bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true; /* [한국어] WRITE_ALLOCATE_SENT 이벤트 발견 */
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/
/* [한국어] MSHR(Miss Status Holding Register) — 캐시 미스 추적 및 병합 구조
 * GPU에서는 여러 warp가 동시에 동일 캐시 라인을 요청할 수 있다. MSHR은 이미
 * 하위 메모리로 발송된 요청과 동일한 블록을 대상으로 하는 후속 요청들을
 * 하나의 엔트리에 모아(merge) 중복 메모리 요청을 방지한다.
 * 설정: -gpgpu_cache:dl1의 MSHR 엔트리 수와 최대 merge 수로 제어된다. */

/*
 * [한국어]
 * mshr_table::probe - 해당 블록 주소에 대한 MSHR 엔트리 존재 여부 확인
 *
 * @block_addr: 탐색할 블록 주소 (캐시 라인 정렬 주소)
 * @return: 이미 MSHR에 동일 블록 주소의 엔트리가 있으면 true (= MSHR hit)
 *
 * MSHR hit이면 새로운 메모리 요청을 기존 엔트리에 merge할 수 있다.
 * baseline_cache::send_read_request()에서 mshr_hit 여부 판단에 사용.
 *
 * 호출 체인:
 *   baseline_cache::send_read_request() → [이 함수]
 */
/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator a = m_data.find(block_addr); /* [한국어] hash map에서 블록 주소 탐색 */
  return a != m_data.end(); /* [한국어] 엔트리가 존재하면 true (MSHR hit) */
}

/*
 * [한국어]
 * mshr_table::full - MSHR 엔트리 추가 가능 여부 확인 (공간 부족 여부)
 *
 * @block_addr: 추가하려는 블록 주소
 * @return: 더 이상 이 주소를 수용할 수 없으면 true (full)
 *
 * 두 가지 경우에 true 반환:
 * 1. 기존 엔트리가 있고 merge 한도(m_max_merged)에 도달한 경우
 * 2. 기존 엔트리가 없고 MSHR 전체 엔트리 수가 한도(m_num_entries)에 도달한 경우
 *
 * 이 함수가 true를 반환하면 캐시는 RESERVATION_FAIL을 반환한다.
 *
 * 호출 체인:
 *   baseline_cache::send_read_request() → [이 함수]
 *   data_cache::wr_miss_wa_naive() → [이 함수]
 */
/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator i = m_data.find(block_addr); /* [한국어] 기존 엔트리 탐색 */
  if (i != m_data.end())
    return i->second.m_list.size() >= m_max_merged; /* [한국어] 기존 엔트리의 merge 목록이 한도(m_max_merged) 이상이면 full */
  else
    return m_data.size() >= m_num_entries; /* [한국어] 새 엔트리 추가 시 전체 MSHR 엔트리 수가 한도(m_num_entries) 이상이면 full */
}

/*
 * [한국어]
 * mshr_table::add - MSHR에 새 요청 추가 또는 기존 엔트리에 merge
 *
 * @block_addr: 대상 블록 주소
 * @mf: 추가할 메모리 요청 패킷
 *
 * 동일 블록 주소의 엔트리가 이미 있으면 해당 엔트리의 m_list에 mf를 append(merge).
 * 없으면 operator[]에 의해 새 엔트리가 자동 생성된 후 추가된다.
 * 원자 연산(atomic)이면 has_atomic 플래그를 설정하여 fill 시 특수 처리를 유도한다.
 * add() 후 엔트리 수와 merge 수가 한도 내인지 assert로 검증한다.
 *
 * 호출 체인:
 *   baseline_cache::send_read_request() → [이 함수]
 */
/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  m_data[block_addr].m_list.push_back(mf); /* [한국어] 엔트리에 mem_fetch 추가 — operator[]로 없으면 신규 생성 */
  assert(m_data.size() <= m_num_entries);  /* [한국어] 전체 MSHR 엔트리 수 한도 검증 */
  assert(m_data[block_addr].m_list.size() <= m_max_merged); /* [한국어] 엔트리별 merge 한도 검증 */
  // indicate that this MSHR entry contains an atomic operation
  if (mf->isatomic()) {
    /* [한국어] 원자 연산 포함 여부 기록 — fill() 완료 시 해당 라인을 MODIFIED로 강제 설정하기 위해 사용 */
    m_data[block_addr].m_has_atomic = true;
  }
}

/*
 * [한국어]
 * mshr_table::is_read_after_write_pending - RAW(Read-After-Write) 해저드 검사
 *
 * @block_addr: 검사할 블록 주소
 * @return: 해당 엔트리에 쓰기 뒤 읽기(RAW) 패턴이 있으면 true
 *
 * MSHR 엔트리의 대기 요청 목록에서 쓰기 요청 뒤에 읽기 요청이 있는지 확인한다.
 * 이 경우 새 쓰기 요청을 merge하면 첫 번째 쓰기 값이 두 번째 쓰기에 덮어씌워져
 * 대기 중인 읽기가 잘못된 값을 읽을 수 있으므로 RESERVATION_FAIL을 반환해야 한다.
 *
 * 호출 체인:
 *   data_cache::wr_miss_wa_fetch_on_write() → [이 함수] (RAW 방지)
 */
/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list; /* [한국어] 대기 중인 요청 목록 복사 */
  bool write_found = false; /* [한국어] 쓰기 요청 발견 플래그 */
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;  /* [한국어] 쓰기 요청 발견 — 이후 읽기 요청이 오면 RAW 해저드 */
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;         /* [한국어] 쓰기 뒤 읽기 패턴 발견 — RAW 해저드 존재 */
  }

  return false; /* [한국어] RAW 패턴 없음 — 새 쓰기 merge 허용 */
}

/*
 * [한국어]
 * mshr_table::mark_ready - 하위 메모리 응답 도착 시 MSHR 엔트리를 응답 대기 상태로 전환
 *
 * @block_addr: 응답이 도착한 블록 주소
 * @has_atomic: [출력] 이 엔트리에 원자 연산이 포함되었는지 여부
 *
 * 하위 메모리(DRAM/L2)에서 데이터가 돌아오면 해당 블록 주소를 m_current_response
 * 큐에 추가하여 next_access()에서 처리할 수 있게 한다.
 * busy() 상태(이미 처리 중인 응답이 있음)에서는 호출하면 안 된다.
 * has_atomic을 반환하여 fill() 후 원자 연산 특수 처리를 지시한다.
 *
 * 호출 체인:
 *   baseline_cache::fill() → [이 함수]
 */
/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  assert(!busy()); /* [한국어] 이미 처리 중인 응답이 있으면 호출 불가 — 단일 응답 처리 제약 */
  table::iterator a = m_data.find(block_addr); /* [한국어] 해당 블록 주소의 MSHR 엔트리 탐색 */
  assert(a != m_data.end()); /* [한국어] 엔트리가 반드시 존재해야 함 — fill()은 MSHR에 등록된 요청에만 호출 */
  m_current_response.push_back(block_addr); /* [한국어] 응답 처리 큐에 블록 주소 추가 — next_access()가 순서대로 처리 */
  has_atomic = a->second.m_has_atomic; /* [한국어] 이 엔트리에 원자 연산이 있는지 반환 — fill() 후 MODIFIED 강제 설정 여부 결정 */
  assert(m_current_response.size() <= m_data.size()); /* [한국어] 응답 큐 크기가 MSHR 엔트리 수를 초과하지 않음을 검증 */
}

/*
 * [한국어]
 * mshr_table::next_access - 응답 대기 중인 MSHR 엔트리에서 다음 요청 반환
 *
 * @return: m_current_response 큐 앞의 블록에 대기 중인 첫 번째 mem_fetch 포인터
 *
 * mark_ready()로 준비된 엔트리의 대기 목록에서 요청을 하나씩 꺼낸다.
 * 대기 목록이 빈 경우 MSHR 엔트리와 m_current_response에서도 제거한다.
 * 이 함수가 반환한 mem_fetch는 상위 캐시 또는 SM의 replay 경로로 전달된다.
 *
 * 호출 체인:
 *   shader.cc (ldst_unit 또는 mem_sub_partition) → [이 함수] → SM으로 결과 전달
 */
/// Returns next ready access
mem_fetch *mshr_table::next_access() {
  assert(access_ready()); /* [한국어] access_ready() = m_current_response가 비어있지 않음 */
  new_addr_type block_addr = m_current_response.front(); /* [한국어] 현재 처리 중인 블록 주소 */
  assert(!m_data[block_addr].m_list.empty()); /* [한국어] 대기 목록에 처리할 요청이 반드시 있어야 함 */
  mem_fetch *result = m_data[block_addr].m_list.front(); /* [한국어] 대기 목록의 첫 번째 요청 선택 */
  m_data[block_addr].m_list.pop_front(); /* [한국어] 선택된 요청을 목록에서 제거 */
  if (m_data[block_addr].m_list.empty()) {
    // release entry
    /* [한국어] 대기 목록이 빈 경우 MSHR 엔트리 완전 해제 */
    m_data.erase(block_addr);           /* [한국어] MSHR 해시맵에서 엔트리 삭제 — 슬롯 반환 */
    m_current_response.pop_front();     /* [한국어] 응답 처리 큐에서도 제거 */
  }
  return result; /* [한국어] SM의 replay 또는 결과 처리 경로로 전달될 mem_fetch 반환 */
}

/*
 * [한국어]
 * mshr_table::display - MSHR 내용 디버그 출력
 *
 * @fp: 출력 대상 파일 스트림
 *
 * 시뮬레이션 디버깅 시 MSHR 상태를 사람이 읽을 수 있는 형태로 출력한다.
 * 각 엔트리의 블록 주소(태그), atomic 여부, 대기 요청 수, 첫 번째 요청 정보를 표시한다.
 *
 * 호출 체인:
 *   baseline_cache::display_state() → [이 함수]
 */
void mshr_table::display(FILE *fp) const {
  fprintf(fp, "MSHR contents\n"); /* [한국어] MSHR 내용 출력 헤더 */
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    unsigned block_addr = e->first; /* [한국어] 현재 엔트리의 블록 주소 */
    fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
            e->second.m_has_atomic, e->second.m_list.size());
    /* [한국어] 블록 주소(16진), atomic 포함 여부, 병합된 요청 수 출력 */
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front(); /* [한국어] 대기 목록의 첫 번째 요청 포인터 */
      fprintf(fp, "%p :", mf); /* [한국어] 요청 포인터 주소 출력 */
      mf->print(fp); /* [한국어] mem_fetch 상세 정보(주소, 타입, 크기 등) 출력 */
    } else {
      fprintf(fp, " no memory requests???\n"); /* [한국어] 목록이 비어있는 비정상 상태 경고 */
    }
  }
}
/***************************************************************** Caches
 * *****************************************************************/
/* [한국어] cache_stats: 캐시 접근 통계 관리 클래스
 * CUDA 스트림 ID별로 (접근 타입 × 결과 상태) 2차원 카운터를 관리한다.
 * 전체 통계(m_stats)와 윈도우별 통계(m_stats_pw), 실패 통계(m_fail_stats)를 분리 관리한다.
 * 포트 사용률(data port, fill port)도 추적하여 대역폭 병목 분석을 지원한다. */

/*
 * [한국어]
 * cache_stats::cache_stats - 캐시 통계 구조 생성자 초기화
 *
 * 포트 사용 사이클 카운터를 0으로 초기화한다. m_stats/m_stats_pw/m_fail_stats는
 * std::map으로, 기본 생성 시 자동으로 빈 상태가 된다.
 *
 * 호출 체인:
 *   baseline_cache 생성자 → [이 함수]
 */
cache_stats::cache_stats() {
  m_cache_port_available_cycles = 0; /* [한국어] 전체 사용 가능 캐시 포트 사이클 수 초기화 */
  m_cache_data_port_busy_cycles = 0; /* [한국어] 데이터 포트가 실제 사용된 사이클 수 초기화 */
  m_cache_fill_port_busy_cycles = 0; /* [한국어] fill 포트가 실제 사용된 사이클 수 초기화 */
}

/*
 * [한국어]
 * cache_stats::clear - 모든 캐시 통계를 초기화 (커널 간 초기화용)
 *
 * m_stats, m_stats_pw, m_fail_stats 맵을 모두 clear하고 포트 사이클 카운터도 0으로 리셋한다.
 * 커널 전환 또는 시뮬레이션 초기화 시 호출된다.
 *
 * 호출 체인:
 *   baseline_cache (커널 초기화) → [이 함수]
 */
void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  m_stats.clear();      /* [한국어] 전체 누적 통계 맵 초기화 */
  m_stats_pw.clear();   /* [한국어] 윈도우별 통계 맵 초기화 */
  m_fail_stats.clear(); /* [한국어] 실패 유형별 통계 맵 초기화 */

  m_cache_port_available_cycles = 0; /* [한국어] 포트 가용 사이클 초기화 */
  m_cache_data_port_busy_cycles = 0; /* [한국어] 데이터 포트 사용 사이클 초기화 */
  m_cache_fill_port_busy_cycles = 0; /* [한국어] fill 포트 사용 사이클 초기화 */
}

/*
 * [한국어]
 * cache_stats::clear_pw - 윈도우별 통계만 초기화 (전체 통계는 유지)
 *
 * m_stats_pw만 clear하여 다음 통계 윈도우를 위한 준비를 한다.
 * AerialVision의 주기적 통계 수집에서 윈도우 경계마다 호출된다.
 *
 * 호출 체인:
 *   AerialVision (윈도우 주기 시작) → [이 함수]
 */
void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  m_stats_pw.clear(); /* [한국어] 윈도우별 통계만 초기화 — m_stats(전체 누적)는 유지 */
}

/*
 * [한국어]
 * cache_stats::inc_stats - (접근 타입, 결과 상태) 쌍의 전체 누적 통계 증가
 *
 * @access_type: 메모리 접근 타입 (GLOBAL_ACC_R, GLOBAL_ACC_W, CONST_ACC_R 등)
 * @access_outcome: 캐시 요청 결과 (HIT, MISS, HIT_RESERVED 등)
 * @streamID: CUDA 스트림 ID — 스트림별 통계 분리를 위해 사용
 *
 * 스트림 ID가 처음 등장하면 새 2D 통계 배열을 생성한다.
 * (접근 타입, 결과 상태) 쌍에 해당하는 카운터를 1 증가시킨다.
 *
 * 호출 체인:
 *   data_cache::access() → [이 함수]
 *   read_only_cache::access() → [이 함수]
 */
void cache_stats::inc_stats(int access_type, int access_outcome,
                            unsigned long long streamID) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome"); /* [한국어] 유효하지 않은 타입/결과 조합 — 설정 오류 */

  if (m_stats.find(streamID) == m_stats.end()) {
    /* [한국어] 이 스트림 ID의 통계 엔트리가 없는 경우 새로 생성 */
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE); /* [한국어] 행: 접근 타입 수만큼 */
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0); /* [한국어] 열: 캐시 결과 상태 수만큼, 0으로 초기화 */
    }
    m_stats.insert(std::pair<unsigned long long,
                             std::vector<std::vector<unsigned long long>>>(
        streamID, new_val)); /* [한국어] 새 스트림 ID 엔트리를 맵에 삽입 */
  }
  m_stats.at(streamID)[access_type][access_outcome]++; /* [한국어] 해당 (접근 타입, 결과 상태) 카운터 1 증가 */
}

/*
 * [한국어]
 * cache_stats::inc_stats_pw - (접근 타입, 결과 상태) 쌍의 윈도우별 통계 증가
 *
 * @access_type: 메모리 접근 타입
 * @access_outcome: 캐시 요청 결과
 * @streamID: CUDA 스트림 ID
 *
 * inc_stats()와 동일한 동작이나 m_stats_pw(윈도우별 통계)를 갱신한다.
 * clear_pw() 호출로 주기적으로 초기화되어 구간별 통계를 제공한다.
 *
 * 호출 체인:
 *   data_cache::access() → [이 함수]
 */
void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               unsigned long long streamID) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome"); /* [한국어] 유효하지 않은 조합 방어 */

  if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
    /* [한국어] 윈도우 통계에 이 스트림 ID 엔트리가 없으면 새로 생성 */
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE); /* [한국어] 접근 타입 수만큼 행 생성 */
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0); /* [한국어] 결과 상태 수만큼 열 생성, 0으로 초기화 */
    }
    m_stats_pw.insert(std::pair<unsigned long long,
                                std::vector<std::vector<unsigned long long>>>(
        streamID, new_val)); /* [한국어] 윈도우 통계 맵에 새 엔트리 삽입 */
  }
  m_stats_pw.at(streamID)[access_type][access_outcome]++; /* [한국어] 윈도우별 통계 카운터 1 증가 */
}

/*
 * [한국어]
 * cache_stats::inc_fail_stats - 캐시 예약 실패 유형별 통계 증가
 *
 * @access_type: 실패한 접근의 메모리 접근 타입
 * @fail_outcome: 실패 원인 (LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL 등)
 * @streamID: CUDA 스트림 ID
 *
 * RESERVATION_FAIL 발생 시 구체적인 실패 원인을 m_fail_stats에 기록한다.
 * 스트림 ID별로 분리 관리하며, 최초 등장 시 새 엔트리를 생성한다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() / send_read_request() 등 → [이 함수]
 */
void cache_stats::inc_fail_stats(int access_type, int fail_outcome,
                                 unsigned long long streamID) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail"); /* [한국어] 유효하지 않은 타입/실패원인 조합 방어 */

  if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
    /* [한국어] 실패 통계에 이 스트림 ID 엔트리가 없으면 새로 생성 */
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE); /* [한국어] 접근 타입 수만큼 행 생성 */
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0); /* [한국어] 실패 유형 수만큼 열 생성, 0으로 초기화 */
    }
    m_fail_stats.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
        streamID, new_val)); /* [한국어] 실패 통계 맵에 새 엔트리 삽입 */
  }
  m_fail_stats.at(streamID)[access_type][fail_outcome]++; /* [한국어] 해당 (접근 타입, 실패 원인) 카운터 1 증가 */
}

/*
 * [한국어]
 * cache_stats::select_stats_status - probe 결과와 access 결과 중 통계 기록에 사용할 상태 선택
 *
 * @probe: tag_array::probe()의 반환값 (실제 태그 탐색 결과)
 * @access: 실제 access 처리 후의 최종 결과
 * @return: 통계 기록에 사용할 cache_request_status
 *
 * probe와 access 결과가 다를 수 있는 경우:
 * - HIT_RESERVED + 정상 처리: SM에서는 MISS처럼 재시도되지만 캐시 통계는 HIT_RESERVED로 기록
 * - SECTOR_MISS + MISS로 처리된 경우: probe의 SECTOR_MISS를 기록하여 섹터 미스 통계 보존
 * 그 외에는 access 결과를 그대로 사용한다.
 *
 * 호출 체인:
 *   data_cache::access() → [이 함수] → cache_stats::inc_stats()
 */
enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe; /* [한국어] HIT_RESERVED이고 실제 처리 성공했으면 HIT_RESERVED로 기록 (SM에서의 MISS 처리와 구별) */
  else if (probe == SECTOR_MISS && access == MISS)
    return probe; /* [한국어] probe에서 SECTOR_MISS였으나 MISS로 처리된 경우 SECTOR_MISS 유지 */
  else
    return access; /* [한국어] 그 외에는 최종 access 결과를 통계 기록에 사용 */
}

/*
 * [한국어]
 * cache_stats::operator() (비상수 버전) - 통계 값 읽기/쓰기 접근자
 *
 * @access_type: 접근 타입 인덱스
 * @access_outcome: 결과 상태 인덱스
 * @fail_outcome: true이면 m_fail_stats, false이면 m_stats 접근
 * @streamID: CUDA 스트림 ID
 * @return: 해당 카운터에 대한 참조 (수정 가능)
 *
 * operator() 오버로딩으로 함수처럼 호출하여 카운터를 직접 읽거나 수정할 수 있다.
 * 별도 read/write 멤버 함수 없이 하나의 인터페이스로 접근 가능하다.
 *
 * 호출 체인:
 *   cache_stats::operator+() 내부 누적 연산 → [이 함수]
 */
unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            unsigned long long streamID) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome"); /* [한국어] 유효하지 않은 실패 통계 접근 방어 */

    return m_fail_stats.at(streamID)[access_type][access_outcome]; /* [한국어] 실패 통계 카운터 참조 반환 */
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome"); /* [한국어] 유효하지 않은 일반 통계 접근 방어 */

    return m_stats.at(streamID)[access_type][access_outcome]; /* [한국어] 일반 통계 카운터 참조 반환 */
  }
}

/*
 * [한국어]
 * cache_stats::operator() (상수 버전) - 통계 값 읽기 전용 접근자
 *
 * @access_type, @access_outcome, @fail_outcome, @streamID: 비상수 버전과 동일
 * @return: 해당 카운터 값 (복사본, 수정 불가)
 *
 * const 맥락(읽기 전용 cache_stats 참조)에서 호출되는 버전.
 * cache_stats::operator+() 의 cs 인자(const 참조)에서 사용된다.
 *
 * 호출 체인:
 *   cache_stats::operator+() 내부 누적 → [이 함수]
 */
unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           unsigned long long streamID) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome"); /* [한국어] 유효하지 않은 접근 방어 */

    return m_fail_stats.at(streamID)[access_type][access_outcome]; /* [한국어] 실패 통계 카운터 값 반환 */
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome"); /* [한국어] 유효하지 않은 접근 방어 */

    return m_stats.at(streamID)[access_type][access_outcome]; /* [한국어] 일반 통계 카운터 값 반환 */
  }
}

/*
 * [한국어]
 * cache_stats::operator+ - 두 cache_stats를 합산하여 새 통계 객체 반환
 *
 * @cs: 더할 다른 cache_stats 객체 (상수 참조)
 * @return: 합산된 새 cache_stats 객체
 *
 * 여러 SM의 캐시 통계를 하나로 집계할 때 사용한다. 스트림 ID별로 처리하며,
 * 한쪽에만 있는 스트림은 그대로 복사하고, 양쪽에 있는 스트림은 타입/상태별로 누적한다.
 * m_stats, m_stats_pw, m_fail_stats를 모두 합산한다.
 *
 * 호출 체인:
 *   gpu-sim.cc (전체 통계 집계) → [이 함수]
 */
cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  cache_stats ret; /* [한국어] 합산 결과를 담을 새 cache_stats 객체 */
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    /* [한국어] this의 m_stats를 ret에 복사 (기존 스트림 ID별 통계) */
    unsigned long long streamID = iter->first;
    ret.m_stats.insert(std::pair<unsigned long long,
                                 std::vector<std::vector<unsigned long long>>>(
        streamID, m_stats.at(streamID))); /* [한국어] this의 해당 스트림 통계를 ret에 복사 */
  }
  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    /* [한국어] this의 m_stats_pw를 ret에 복사 */
    unsigned long long streamID = iter->first;
    ret.m_stats_pw.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_stats_pw.at(streamID))); /* [한국어] this의 윈도우 통계를 ret에 복사 */
  }
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    /* [한국어] this의 m_fail_stats를 ret에 복사 */
    unsigned long long streamID = iter->first;
    ret.m_fail_stats.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_fail_stats.at(streamID))); /* [한국어] this의 실패 통계를 ret에 복사 */
  }
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    /* [한국어] cs(인자)의 m_stats를 ret에 누적 */
    unsigned long long streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      /* [한국어] ret에 없는 스트림 ID이면 cs의 값을 그대로 복사 */
      ret.m_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats.at(streamID)));
    } else {
      /* [한국어] ret에 이미 있는 스트림 ID이면 타입/상태별로 누적 */
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID); /* [한국어] cs의 해당 카운터를 ret에 누적 */
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    /* [한국어] cs의 윈도우 통계를 ret에 누적 */
    unsigned long long streamID = iter->first;
    if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
      ret.m_stats_pw.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats_pw.at(streamID))); /* [한국어] 신규 스트림이면 그대로 복사 */
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID); /* [한국어] 기존 스트림이면 누적 */
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    /* [한국어] cs의 실패 통계를 ret에 누적 */
    unsigned long long streamID = iter->first;
    if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
      ret.m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID))); /* [한국어] 신규 스트림이면 그대로 복사 */
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          ret.m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID); /* [한국어] 실패 통계 누적 (fail_outcome=true) */
        }
      }
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles; /* [한국어] 포트 가용 사이클 합산 */
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles; /* [한국어] 데이터 포트 사용 사이클 합산 */
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles; /* [한국어] fill 포트 사용 사이클 합산 */
  return ret; /* [한국어] 합산된 통계 객체 반환 */
}

/*
 * [한국어]
 * cache_stats::operator+= - 이 통계 객체에 다른 통계를 누적 (제자리 합산)
 *
 * @cs: 누적할 cache_stats 객체
 * @return: 누적 후 this 참조
 *
 * operator+와 동일한 논리이나 새 객체를 생성하지 않고 this에 직접 누적한다.
 * 반복적인 통계 집계 루프에서 메모리 할당 없이 효율적으로 사용 가능하다.
 *
 * 호출 체인:
 *   gpu-sim.cc (SM별 통계 누적) → [이 함수]
 */
cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    /* [한국어] cs.m_stats의 각 스트림 엔트리를 순회하며 this->m_stats에 누적
     * m_stats는 streamID → [접근타입][상태] 형태의 중첩 맵/벡터 구조 */
    unsigned long long streamID = iter->first; /* [한국어] 현재 순회 중인 CUDA 스트림 ID 추출 */
    if (m_stats.find(streamID) == m_stats.end()) {
      /* [한국어] this->m_stats에 해당 streamID가 없으면: cs의 엔트리를 통째로 복사하여 삽입
       * 새 스트림이 처음 등장하는 경우로, 기존 값이 없으므로 덮어쓰기 없이 직접 insert */
      m_stats.insert(std::pair<unsigned long long,
                               std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats.at(streamID)));
    } else {
      /* [한국어] 이미 this->m_stats에 해당 streamID가 존재하면: (타입, 상태) 쌍별로 누적 */
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        /* [한국어] 모든 메모리 접근 타입(GLOBAL_ACC_R, GLOBAL_ACC_W, L1_ACC_R 등) 순회 */
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          /* [한국어] 모든 캐시 요청 결과(HIT, MISS, SECTOR_MISS, HIT_RESERVED, RESERVATION_FAIL) 순회 */
          m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
          /* [한국어] cs의 해당 (streamID, type, status) 카운터를 this에 누적
           * cs(type, status, false, streamID): fail_stat=false → m_stats에서 값 조회 */
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    /* [한국어] 윈도우별 통계(m_stats_pw)를 동일한 방식으로 누적
     * m_stats_pw는 주기적 성능 윈도우(AerialVision 등)에서 리셋되는 통계 */
    unsigned long long streamID = iter->first; /* [한국어] 현재 순회 중인 스트림 ID */
    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
      /* [한국어] 해당 streamID가 없으면 cs의 윈도우 통계를 통째로 삽입 */
      m_stats_pw.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats_pw.at(streamID)));
    } else {
      /* [한국어] 이미 존재하면 (타입, 상태) 쌍별로 누적 */
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        /* [한국어] 모든 메모리 접근 타입 순회 */
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          /* [한국어] 모든 캐시 요청 결과 순회 */
          m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
          /* [한국어] 윈도우 통계 누적 — fail_stat=false이므로 m_stats_pw 기준 조회 */
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    /* [한국어] 실패 원인별 통계(m_fail_stats)를 누적
     * m_fail_stats는 LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL 등 예약 실패 원인 추적 */
    unsigned long long streamID = iter->first; /* [한국어] 현재 순회 중인 스트림 ID */
    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
      /* [한국어] 해당 streamID가 없으면 cs의 실패 통계를 통째로 삽입 */
      m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID)));
    } else {
      /* [한국어] 이미 존재하면 (타입, 실패 원인) 쌍별로 누적 */
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        /* [한국어] 모든 메모리 접근 타입 순회 */
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          /* [한국어] 모든 예약 실패 원인(LINE_ALLOC_FAIL, MISS_QUEUE_FULL 등) 순회 */
          m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID);
          /* [한국어] 실패 통계 누적 — fail_stat=true이므로 m_fail_stats에서 값 조회 */
        }
      }
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  /* [한국어] 포트 가용 사이클 수 누적 — 포트 사용률 계산의 분모 */
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  /* [한국어] 데이터 포트 사용 사이클 수 누적 — data_port_util 계산에 사용 */
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  /* [한국어] fill 포트 사용 사이클 수 누적 — fill_port_util 계산에 사용 */
  return *this; /* [한국어] 복합 대입 연산자 관례에 따라 *this를 반환하여 체인 연산 가능 */
}

/*
 * [한국어]
 * cache_stats::print_stats - 캐시 접근 통계를 파일로 출력
 *
 * @fout: 출력 대상 파일 스트림
 * @streamID: 출력할 CUDA 스트림 ID (-1이면 모든 스트림 출력)
 * @cache_name: 출력 레이블 (기본값 "Cache_stats")
 *
 * 형식: "<cache_name>[<접근_타입>][<결과_상태>] = <카운터_값>"
 * 비-0 값만 출력하지 않고 모든 (타입, 상태) 조합을 출력하며 타입별 TOTAL_ACCESS도 출력한다.
 * RESERVATION_FAIL과 MSHR_HIT은 TOTAL_ACCESS 집계에서 제외된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (시뮬레이션 결과 리포트) → [이 함수]
 */
void cache_stats::print_stats(FILE *fout, unsigned long long streamID,
                              const char *cache_name) const {
  ///
  /// For a given CUDA stream, print out each non-zero cache statistic for every
  /// memory access type and status "cache_name" defaults to "Cache_stats" when
  /// no argument is provided, otherwise the provided name is used. The printed
  /// format is
  /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  /// Specify streamID to be -1 to print every stream.

  std::vector<unsigned> total_access;
  std::string m_cache_name = cache_name;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    total_access.clear();
    total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_request_status_str((enum cache_request_status)status),
                m_stats.at(streamid)[type][status]);

        if (status != RESERVATION_FAIL && status != MSHR_HIT)
          // MSHR_HIT is a special type of SECTOR_MISS
          // so its already included in the SECTOR_MISS
          total_access[type] += m_stats.at(streamid)[type][status];
      }
    }
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      if (total_access[type] > 0)
        fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
                total_access[type]);
    }
  }
}

/*
 * [한국어]
 * cache_stats::print_fail_stats - 캐시 예약 실패 통계를 파일로 출력
 *
 * @fout: 출력 대상 파일 스트림
 * @streamID: 출력할 CUDA 스트림 ID (-1이면 모든 스트림 출력)
 * @cache_name: 출력 레이블 접두어 (기본값 "Cache_stats")
 *
 * m_fail_stats를 순회하여 (접근 타입, 실패 원인) 쌍별로 0이 아닌 카운터를 출력한다.
 * 실패 원인은 LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL, MSHR_MERGE_ENRTY_FAIL,
 * MSHR_RW_PENDING 등 cache_reservation_fail_reason enum 값에 해당한다.
 * print_stats()와 달리 HIT/MISS 카운터가 아니라 스톨(stall) 원인 분류 데이터를 출력한다.
 * 시뮬레이션 종료 시 gpu-sim.cc의 성능 리포트 단계에서 호출된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (성능 통계 리포트) → [이 함수]
 */
void cache_stats::print_fail_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_name) const {
  std::string m_cache_name = cache_name; /* [한국어] C 문자열을 std::string으로 복사 — fprintf 포맷 편의 */
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    /* [한국어] m_fail_stats의 모든 스트림 엔트리를 순회 */
    unsigned long long streamid = iter->first; /* [한국어] 현재 순회 중인 스트림 ID */
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    /* [한국어] streamID가 지정된 경우(-1이 아닌 경우) 일치하는 스트림만 출력, 나머지 건너뜀 */
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      /* [한국어] 모든 메모리 접근 타입(GLOBAL_ACC_R, GLOBAL_ACC_W 등) 순회 */
      for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS;
           ++fail) {
        /* [한국어] 모든 예약 실패 원인(LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL 등) 순회 */
        if (m_fail_stats.at(streamid)[type][fail] > 0) {
          /* [한국어] 0이 아닌 카운터만 출력하여 리포트 간결성 유지 */
          fprintf(
              fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_fail_status_str((enum cache_reservation_fail_reason)fail),
              m_fail_stats.at(streamid)[type][fail]);
          /* [한국어] 형식: "<cache_name>[<접근타입>][<실패원인>] = <카운터값>"
           * cache_fail_status_str(): 실패 원인 enum을 사람이 읽을 수 있는 문자열로 변환 */
        }
      }
    }
  }
}

/*
 * [한국어]
 * cache_sub_stats::print_port_stats - 캐시 데이터/fill 포트 사용률을 파일로 출력
 *
 * @fout: 출력 대상 파일 스트림
 * @cache_name: 출력 레이블 접두어
 *
 * 캐시에는 두 가지 포트가 있다:
 *   - 데이터 포트(data port): CPU 측(SM) 요청이 캐시에 접근하는 포트
 *   - fill 포트(fill port): 하위 메모리(L2/DRAM)에서 데이터가 캐시를 채우는 포트
 * 사용률 = 해당_포트_사용_사이클 / 포트_가용_사이클 (0~1.0 범위)
 * port_available_cycles가 0이면 나누기 금지이므로 0.0으로 초기화된 채 출력한다.
 * 시뮬레이션 종료 시 gpu-sim.cc의 성능 리포트 단계에서 호출된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (성능 통계 리포트) → get_sub_stats() → [이 함수]
 */
void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
  float data_port_util = 0.0f; /* [한국어] 데이터 포트 사용률 초기값 — port_available_cycles가 0일 때 안전한 기본값 */
  if (port_available_cycles > 0) {
    /* [한국어] 가용 사이클이 1 이상인 경우에만 나눗셈 수행 (0 나누기 방지) */
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
    /* [한국어] 데이터 포트 사용률 계산: SM 접근이 포트를 점유한 사이클 비율 */
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  /* [한국어] 데이터 포트 사용률 출력, 소수점 3자리까지 표시 */
  float fill_port_util = 0.0f; /* [한국어] fill 포트 사용률 초기값 */
  if (port_available_cycles > 0) {
    /* [한국어] 가용 사이클이 1 이상인 경우에만 나눗셈 수행 */
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
    /* [한국어] fill 포트 사용률 계산: 하위 메모리가 캐시 라인을 채우는 포트 점유 비율 */
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
  /* [한국어] fill 포트 사용률 출력, 소수점 3자리까지 표시 */
}

/*
 * [한국어]
 * cache_stats::get_stats - 지정된 (접근타입, 상태) 배열 조합의 통계 합산
 *
 * @access_type: 합산할 mem_access_type 배열 (예: {GLOBAL_ACC_R, GLOBAL_ACC_W})
 * @num_access_type: access_type 배열의 원소 수
 * @access_status: 합산할 cache_request_status 배열 (예: {HIT, MISS})
 * @num_access_status: access_status 배열의 원소 수
 * @return: 지정된 모든 (타입, 상태) 쌍의 카운터를 모든 스트림에 걸쳐 합산한 값
 *
 * 모든 CUDA 스트림에 걸쳐 지정된 (접근타입, 상태) 쌍의 카운터를 합산한다.
 * 예를 들어 전체 글로벌 읽기 히트 수를 구하려면 access_type={GLOBAL_ACC_R},
 * access_status={HIT}로 호출한다.
 * check_valid()로 유효성을 검사하고, 무효한 인자에는 assert로 패닉한다.
 * gpu-sim.cc의 성능 지표 계산 및 Accel-Sim 통계 수집에서 사용된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (성능 카운터 집계) → [이 함수] → check_valid()
 */
unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0; /* [한국어] 합산 누적 변수 초기화 */
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    /* [한국어] 모든 CUDA 스트림 엔트리를 순회하여 스트림 구분 없이 전체 합산 */
    unsigned long long streamID = iter->first; /* [한국어] 현재 순회 중인 스트림 ID */
    for (unsigned type = 0; type < num_access_type; ++type) {
      /* [한국어] 호출자가 지정한 접근 타입 배열 순회 */
      for (unsigned status = 0; status < num_access_status; ++status) {
        /* [한국어] 호출자가 지정한 요청 상태 배열 순회 */
        if (!check_valid((int)access_type[type], (int)access_status[status]))
          assert(0 && "Unknown cache access type or access outcome");
        /* [한국어] 무효한 (타입, 상태) 조합이면 즉시 패닉 — 호출자 버그 감지 */
        total += m_stats.at(streamID)[access_type[type]][access_status[status]];
        /* [한국어] 해당 (스트림, 접근타입, 상태) 카운터를 누적 합산 */
      }
    }
  }
  return total; /* [한국어] 모든 스트림에 걸친 지정 (타입, 상태) 카운터의 총합 반환 */
}

/*
 * [한국어]
 * cache_stats::get_sub_stats - 통합 서브 통계 구조체 채우기 (접근/미스/포트 통계)
 *
 * @css: [출력] 채울 cache_sub_stats 구조체
 *
 * 모든 스트림의 통계를 합산하여 단순화된 서브 통계(accesses, misses,
 * pending_hits, res_fails, 포트 사용률)를 제공한다.
 * RESERVATION_FAIL은 accesses에 포함되지 않는다(실제 캐시 접근이 아님).
 * gpu-sim.cc의 성능 리포트 및 Accel-Sim 프레임워크에서 사용된다.
 *
 * 호출 체인:
 *   gpu-sim.cc (성능 통계 수집) → [이 함수]
 */
void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css; /* [한국어] 임시 서브 통계 구조체 — 모든 스트림 합산 후 css에 대입 */
  t_css.clear(); /* [한국어] 0으로 초기화 */

  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats.at(streamID)[type][status];

        if (status == MISS || status == SECTOR_MISS)
          t_css.misses += m_stats.at(streamID)[type][status];

        if (status == HIT_RESERVED)
          t_css.pending_hits += m_stats.at(streamID)[type][status];

        if (status == RESERVATION_FAIL)
          t_css.res_fails += m_stats.at(streamID)[type][status];
      }
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

/*
 * [한국어]
 * cache_stats::get_sub_stats_pw - 윈도우별 서브 통계 구조체 채우기
 *
 * @css: [출력] 채울 cache_sub_stats_pw 구조체 (읽기/쓰기 구분된 윈도우 통계)
 *
 * m_stats_pw(윈도우별 통계)를 기반으로 읽기/쓰기 히트/미스/pending_hit/res_fail을
 * 접근 타입별로 분리하여 채운다. AerialVision의 주기적 성능 가시화에 사용된다.
 *
 * 호출 체인:
 *   AerialVision 또는 Accel-Sim (윈도우 통계 수집) → [이 함수]
 */
void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats_pw t_css; /* [한국어] 임시 윈도우별 서브 통계 구조체 */
  t_css.clear(); /* [한국어] 0으로 초기화 */

  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    /* [한국어] m_stats_pw의 모든 스트림 엔트리를 순회하여 윈도우 통계 합산 */
    unsigned long long streamID = iter->first; /* [한국어] 현재 순회 중인 스트림 ID */
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      /* [한국어] 모든 메모리 접근 타입 순회 (GLOBAL_ACC_R, GLOBAL_ACC_W, CONST_ACC_R 등) */
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        /* [한국어] 모든 캐시 요청 결과 상태 순회 */
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats_pw.at(streamID)[type][status];
        /* [한국어] 실제 캐시 접근(HIT/MISS/SECTOR_MISS/HIT_RESERVED)의 총 횟수 누적
         * RESERVATION_FAIL은 실제 접근이 아닌 스톨이므로 accesses에서 제외 */

        if (status == HIT) {
          /* [한국어] 히트 상태인 경우: 읽기/쓰기 접근 타입별로 분리하여 누적 */
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            /* [한국어] 읽기 접근(글로벌/상수/명령어 캐시 읽기)의 히트 카운트 누적 */
            t_css.read_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            /* [한국어] 쓰기 접근(글로벌 쓰기)의 히트 카운트 누적 */
            t_css.write_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == MISS || status == SECTOR_MISS) {
          /* [한국어] 미스 상태(전체 라인 미스 또는 섹터 미스)인 경우: 읽기/쓰기별 분리
           * SECTOR_MISS: 섹터 기반 캐시에서 라인은 할당됐지만 해당 섹터가 없는 경우 */
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            /* [한국어] 읽기 접근의 미스 카운트 누적 */
            t_css.read_misses += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            /* [한국어] 쓰기 접근의 미스 카운트 누적 */
            t_css.write_misses += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == HIT_RESERVED) {
          /* [한국어] HIT_RESERVED: 이미 MSHR에 대기 중인 라인에 hit — 실제 데이터 아직 미도착
           * pending_hit로 분류하여 실제 미스와 구분 */
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            /* [한국어] 읽기 접근의 pending_hit(MSHR 대기 히트) 카운트 누적 */
            t_css.read_pending_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            /* [한국어] 쓰기 접근의 pending_hit 카운트 누적 */
            t_css.write_pending_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == RESERVATION_FAIL) {
          /* [한국어] RESERVATION_FAIL: MSHR/미스 큐/라인 할당 실패로 이 사이클에 처리 불가
           * 읽기/쓰기별로 분리하여 병목 원인 분석에 사용 */
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            /* [한국어] 읽기 접근의 예약 실패 카운트 누적 */
            t_css.read_res_fails += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            /* [한국어] 쓰기 접근의 예약 실패 카운트 누적 */
            t_css.write_res_fails += m_stats_pw.at(streamID)[type][status];
          }
        }
      }
    }
  }

  css = t_css; /* [한국어] 모든 스트림 합산이 완료된 t_css를 출력 파라미터 css에 복사 */
}

/*
 * [한국어]
 * cache_stats::check_valid - (접근 타입, 요청 상태) 쌍의 유효성 검사
 *
 * @type: 검사할 메모리 접근 타입 (mem_access_type enum의 정수 값)
 * @status: 검사할 캐시 요청 상태 (cache_request_status enum의 정수 값)
 * @return: 유효한 경우 true, 범위 초과이면 false
 *
 * type이 [0, NUM_MEM_ACCESS_TYPE) 범위에 속하고
 * status가 [0, NUM_CACHE_REQUEST_STATUS) 범위에 속하는지 확인한다.
 * get_stats()에서 호출자가 전달한 배열 원소의 범위를 사전 검증할 때 사용된다.
 * 검사 실패 시 get_stats()는 assert(0)으로 패닉하여 무효 접근을 조기에 탐지한다.
 *
 * 호출 체인:
 *   cache_stats::get_stats() → [이 함수]
 */
bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

/*
 * [한국어]
 * cache_stats::check_fail_valid - (접근 타입, 예약 실패 원인) 쌍의 유효성 검사
 *
 * @type: 검사할 메모리 접근 타입 (mem_access_type enum의 정수 값)
 * @fail: 검사할 예약 실패 원인 (cache_reservation_fail_reason enum의 정수 값)
 * @return: 유효한 경우 true, 범위 초과이면 false
 *
 * type이 [0, NUM_MEM_ACCESS_TYPE) 범위에 속하고
 * fail이 [0, NUM_CACHE_RESERVATION_FAIL_STATUS) 범위에 속하는지 확인한다.
 * 예약 실패 원인에는 LINE_ALLOC_FAIL, MISS_QUEUE_FULL, MSHR_ENRTY_FAIL,
 * MSHR_MERGE_ENRTY_FAIL, MSHR_RW_PENDING 등이 있다.
 * check_valid()와 달리 status 대신 fail_reason 인덱스를 검사하는 점이 다르다.
 *
 * 호출 체인:
 *   inc_fail_stats() / 실패 통계 접근 코드 → [이 함수]
 */
bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

/*
 * [한국어]
 * cache_stats::sample_cache_port_utility - 캐시 포트 사용률 샘플링 (매 사이클 호출)
 *
 * @data_port_busy: 이 사이클에 데이터 포트가 사용 중인지
 * @fill_port_busy: 이 사이클에 fill 포트가 사용 중인지
 *
 * 매 사이클 포트 가용/사용 상태를 누적하여 포트 사용률 계산의 기반을 만든다.
 * 최종 사용률 = busy_cycles / available_cycles. baseline_cache::cycle()에서 호출.
 *
 * 호출 체인:
 *   baseline_cache::cycle() → [이 함수]
 */
void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1; /* [한국어] 이 사이클에 포트가 가용 상태였음을 기록 */
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1; /* [한국어] 데이터 포트가 사용 중이면 카운터 증가 */
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1; /* [한국어] fill 포트가 사용 중이면 카운터 증가 */
  }
}

/*
 * [한국어]
 * baseline_cache::bandwidth_management::bandwidth_management - 대역폭 관리 생성자
 *
 * @config: 캐시 설정 참조 — 포트 폭(m_data_port_width) 접근에 사용
 *
 * 데이터 포트와 fill 포트의 점유 사이클 카운터를 0으로 초기화한다.
 * 각 포트는 사용 시 점유 사이클을 증가시키고, replenish_port_bandwidth()로 매 사이클 1씩 감소된다.
 *
 * 호출 체인:
 *   baseline_cache 생성자 → [이 함수]
 */
baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0; /* [한국어] 데이터 포트 점유 사이클 초기화 */
  m_fill_port_occupied_cycles = 0; /* [한국어] fill 포트 점유 사이클 초기화 */
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
/*
 * [한국어]
 * bandwidth_management::use_data_port - 접근 결과에 따라 데이터 포트 점유 사이클 계산
 *
 * @mf: 메모리 요청 패킷 (데이터 크기 확인용)
 * @outcome: 캐시 접근 결과
 * @events: 접근 중 발생한 이벤트 목록 (writeback 여부 확인용)
 *
 * HIT 시에는 요청 데이터 크기를 포트 폭으로 나누어 점유 사이클 계산.
 * MISS/HIT_RESERVED 시에는 writeback이 있는 경우에만 수정된 크기만큼 점유.
 * SECTOR_MISS/RESERVATION_FAIL은 포트 대역폭을 소비하지 않는다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수]
 */
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
  unsigned data_size = mf->get_data_size();     /* [한국어] 요청 데이터 크기 (바이트) */
  unsigned port_width = m_config.m_data_port_width; /* [한국어] 데이터 포트 폭 (바이트/사이클) */
  switch (outcome) {
    case HIT: {
      /* [한국어] HIT: 캐시에서 데이터를 읽어 SM으로 전달 — 데이터 크기에 따른 포트 점유 */
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      /* [한국어] 올림(ceil) 계산 — 데이터가 포트 폭의 배수가 아닐 때 마지막 부분 사이클 포함 */
      m_data_port_occupied_cycles += data_cycles; /* [한국어] 데이터 포트 점유 사이클 증가 */
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      /* [한국어] MISS/HIT_RESERVED: 데이터 자체는 아직 없음, writeback이 있을 때만 포트 사용
       * 섹터 캐시에서는 수정된 섹터만 writeback하므로 수정 크기만큼만 포트 점유 */
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        /* [한국어] writeback 이벤트가 발생한 경우 수정된 데이터 크기만큼 포트 점유 */
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        /* [한국어] evicted 블록의 수정 크기를 포트 폭으로 나누어 필요 사이클 계산 */
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      /* [한국어] SECTOR_MISS/RESERVATION_FAIL: 실제 데이터 이동 없음 — 포트 대역폭 소비 없음 */
      break;
    default:
      assert(0); /* [한국어] 알 수 없는 결과 상태 — 논리 오류 */
      break;
  }
}

/// use the fill port
/*
 * [한국어]
 * bandwidth_management::use_fill_port - fill 포트 점유 사이클 계산
 *
 * @mf: fill 응답 패킷 (현재 미사용, 전체 원자 크기로 고정 계산)
 *
 * 하위 메모리에서 데이터가 돌아와 캐시 라인을 채울 때 fill 포트 점유 사이클을 계산한다.
 * 현재는 전체 원자 크기(get_atom_sz)를 기준으로 계산하며, 섹터 단위 부분 fill은 미지원.
 *
 * 호출 체인:
 *   baseline_cache::fill() → [이 함수]
 */
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  /* [한국어] fill 포트 점유 사이클 = 원자 크기 / 포트 폭 — 전체 라인을 채우는 데 필요한 사이클 */
  m_fill_port_occupied_cycles += fill_cycles; /* [한국어] fill 포트 점유 사이클 증가 */
}

/// called every cache cycle to free up the ports
/*
 * [한국어]
 * bandwidth_management::replenish_port_bandwidth - 매 사이클 포트 점유 사이클 감소
 *
 * 데이터 포트와 fill 포트의 점유 사이클을 각각 1씩 감소시켜 포트가 점차 해방되게 한다.
 * 이 함수가 0에서 1을 빼지 않도록 보호하는 assert가 있다.
 * baseline_cache::cycle()에서 매 사이클 마지막에 호출된다.
 *
 * 호출 체인:
 *   baseline_cache::cycle() → [이 함수]
 */
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1; /* [한국어] 데이터 포트 점유 사이클 1 감소 */
  }
  assert(m_data_port_occupied_cycles >= 0); /* [한국어] 음수가 되면 안 됨 — 논리 오류 방어 */

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1; /* [한국어] fill 포트 점유 사이클 1 감소 */
  }
  assert(m_fill_port_occupied_cycles >= 0); /* [한국어] 음수가 되면 안 됨 — 논리 오류 방어 */
}

/// query for data port availability
/*
 * [한국어]
 * bandwidth_management::data_port_free - 데이터 포트 사용 가능 여부 반환
 *
 * @return: 데이터 포트가 비어있으면 true (점유 사이클 = 0)
 *
 * cache access 전에 데이터 포트 가용 여부를 확인하는 데 사용된다.
 * 포트가 busy이면 해당 사이클에 접근을 차단하여 실제 하드웨어 대역폭 제약을 모델링한다.
 *
 * 호출 체인:
 *   baseline_cache::cycle() → [이 함수]
 */
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0); /* [한국어] 점유 사이클이 0이면 포트 사용 가능 */
}

/// query for fill port availability
/*
 * [한국어]
 * bandwidth_management::fill_port_free - fill 포트 사용 가능 여부 반환
 *
 * @return: fill 포트가 비어있으면 true
 *
 * baseline_cache::cycle()에서 fill 포트 사용률 측정 목적으로 호출된다.
 *
 * 호출 체인:
 *   baseline_cache::cycle() → [이 함수]
 */
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0); /* [한국어] fill 포트 점유 사이클이 0이면 포트 사용 가능 */
}

/// Sends next request to lower level of memory
/*
 * [한국어]
 * baseline_cache::cycle - 매 GPU 사이클 호출되는 캐시 타이밍 업데이트 함수
 *
 * miss_queue의 첫 번째 요청을 하위 메모리 인터페이스(m_memport)로 전송 시도하고,
 * 포트 사용률을 샘플링하고, 포트 점유 사이클을 1씩 감소시킨다.
 * 이 함수는 GPU 코어 사이클마다 shader.cc의 메모리 파이프라인에서 호출된다.
 *
 * 동작 단계:
 * 1. miss_queue가 비어있지 않으면 첫 번째 mem_fetch를 확인
 * 2. m_memport(하위 메모리 인터페이스)가 가득 차지 않았으면 push하여 전송
 * 3. 현재 포트 상태(busy 여부) 샘플링
 * 4. replenish_port_bandwidth()로 포트 점유 사이클 감소
 *
 * 호출 체인:
 *   shader.cc ldst_unit 또는 mem_sub_partition::cycle() → [이 함수]
 */
void baseline_cache::cycle() {
  if (!m_miss_queue.empty()) {
    /* [한국어] miss_queue에 대기 중인 요청이 있으면 하위 메모리로 발송 시도 */
    mem_fetch *mf = m_miss_queue.front(); /* [한국어] 큐의 첫 번째 요청 확인 (발송 대기 순위가 가장 높음) */
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      /* [한국어] 하위 메모리 인터페이스(ICNT 또는 L2 포트)가 가득 차지 않은 경우에만 발송 */
      m_miss_queue.pop_front(); /* [한국어] 큐에서 제거 */
      m_memport->push(mf);      /* [한국어] 하위 메모리 포트에 push — ICNT를 통해 전달됨 */
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free(); /* [한국어] 데이터 포트 사용 중 여부 확인 */
  bool fill_port_busy = !m_bandwidth_management.fill_port_free(); /* [한국어] fill 포트 사용 중 여부 확인 */
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy); /* [한국어] 이 사이클의 포트 상태 통계 기록 */
  m_bandwidth_management.replenish_port_bandwidth(); /* [한국어] 포트 점유 사이클 1씩 감소 — 다음 사이클 가용 여부 업데이트 */
}

/// Interface for response from lower memory level (model bandwidth restictions
/// in caller)
/*
 * [한국어]
 * baseline_cache::fill - 하위 메모리에서 데이터 응답 도착 시 캐시 채우기
 *
 * @mf: 하위 메모리(L2/DRAM)에서 돌아온 메모리 요청 패킷
 * @time: 현재 시뮬레이션 사이클
 *
 * 미스로 인해 하위 메모리로 발송했던 요청이 완료되면 이 함수가 호출된다.
 * 동작 단계:
 * 1. SECTOR_ASSOC MSHR이면 섹터별 부분 응답 처리:
 *    - pending_read 감소. 아직 남아있으면 delete 후 return (대기)
 *    - 마지막 응답이면 original_mf로 전환하여 처리 계속
 * 2. m_extra_mf_fields에서 캐시 인덱스와 원래 주소 복원
 * 3. ON_MISS이면 tag_array::fill(index)로 미리 할당된 라인 채우기
 *    ON_FILL이면 tag_array::fill(addr)로 새 라인 할당 후 채우기
 * 4. MSHR::mark_ready()로 이 블록을 기다리던 요청들 처리 가능 상태로 전환
 * 5. 원자 연산이면 라인을 MODIFIED로 표시 (원자 실행 결과가 캐시에 기록되므로)
 * 6. m_extra_mf_fields에서 엔트리 삭제, fill 포트 사용 기록
 *
 * 호출 체인:
 *   mem_sub_partition (DRAM 응답) → [이 함수] → tag_array::fill() + mshr_table::mark_ready()
 */
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    /* [한국어] SECTOR_ASSOC MSHR: 하나의 캐시 라인을 여러 섹터로 나누어 독립 요청을 발송하는 구조
     * 섹터별 응답이 개별적으로 도착하므로 모든 섹터 응답이 모일 때까지 대기 */
    assert(mf->get_original_mf()); /* [한국어] 섹터 요청은 반드시 원본 요청(original_mf)이 있어야 함 */
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf()); /* [한국어] 원본 요청의 extra 필드 탐색 */
    assert(e != m_extra_mf_fields.end()); /* [한국어] 등록된 요청이어야 함 */
    e->second.pending_read--;  /* [한국어] 대기 중인 섹터 응답 수 감소 */

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      /* [한국어] 아직 응답을 기다리는 섹터가 남아있음 — 이 섹터 응답은 삭제 후 대기 */
      delete mf; /* [한국어] 부분 응답 패킷 삭제 */
      return;    /* [한국어] 나머지 섹터 응답이 도착할 때까지 대기 */
    } else {
      /* [한국어] 마지막 섹터 응답 도착 — 원본 요청으로 전환하여 이후 처리 수행 */
      mem_fetch *temp = mf;                    /* [한국어] 현재 섹터 응답 패킷 임시 저장 */
      mf = mf->get_original_mf();              /* [한국어] 원본 요청 패킷으로 전환 */
      delete temp;                             /* [한국어] 섹터 응답 패킷 삭제 */
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); /* [한국어] extra 필드 탐색 — 캐시 인덱스, 원래 주소 등 저장 */
  assert(e != m_extra_mf_fields.end()); /* [한국어] send_read_request 시 등록된 엔트리가 반드시 있어야 함 */
  assert(e->second.m_valid); /* [한국어] 유효한 extra 필드 엔트리인지 검증 */
  mf->set_data_size(e->second.m_data_size); /* [한국어] 원래 데이터 크기 복원 (send_read_request에서 atom_sz로 변경되었음) */
  mf->set_addr(e->second.m_addr);           /* [한국어] 원래 주소 복원 (send_read_request에서 mshr_addr로 변경되었음) */
  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf); /* [한국어] ON_MISS: 미리 할당된(RESERVED) 라인에 데이터 채우기 */
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write()); /* [한국어] ON_FILL: 이 시점에 라인을 새로 할당하고 채우기 */
  } else
    abort(); /* [한국어] 알 수 없는 alloc 정책 — 설정 오류 */
  bool has_atomic = false; /* [한국어] 원자 연산 포함 여부 — mark_ready에서 설정됨 */
  m_mshrs.mark_ready(e->second.m_block_addr, has_atomic); /* [한국어] 이 블록을 기다리던 MSHR 엔트리를 처리 가능 상태로 전환 */
  if (has_atomic) {
    /* [한국어] 원자 연산 포함 엔트리: fill 후 라인을 MODIFIED로 강제 설정
     * 원자 연산(atomicAdd 등)은 읽어서 수정하므로 결과가 캐시에 더티 상태로 존재 */
    assert(m_config.m_alloc_policy == ON_MISS); /* [한국어] 원자 연산은 ON_MISS에서만 지원 */
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty(); /* [한국어] 새로 MODIFIED 상태가 되면 더티 카운터 증가 */
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as dirty for
                                                      // atomic operation
    /* [한국어] 원자 연산 결과를 캐시에 반영 — MODIFIED로 표시하여 나중에 writeback 필요하게 함 */
    block->set_byte_mask(mf); /* [한국어] 원자 연산이 수정한 바이트 위치를 byte mask에 기록 */
  }
  m_extra_mf_fields.erase(mf);      /* [한국어] fill 완료 후 extra_mf_fields 엔트리 삭제 — 메모리 해방 */
  m_bandwidth_management.use_fill_port(mf); /* [한국어] fill 포트 사용 기록 — 대역폭 모델 업데이트 */
}

/// Checks if mf is waiting to be filled by lower memory level
/*
 * [한국어]
 * baseline_cache::waiting_for_fill - 이 mem_fetch가 하위 메모리 응답을 기다리는지 확인
 *
 * @mf: 확인할 메모리 요청 패킷
 * @return: m_extra_mf_fields에 등록된 경우 true (= 하위 메모리로 발송 후 응답 대기 중)
 *
 * 상위 모듈(shader.cc)에서 특정 mem_fetch가 이 캐시에서 처리 중인지 확인할 때 사용한다.
 * fill()이 호출되면 erase되므로, fill 완료 후에는 false를 반환한다.
 *
 * 호출 체인:
 *   shader.cc 또는 mem_sub_partition → [이 함수]
 */
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); /* [한국어] extra_mf_fields 맵에서 탐색 */
  return e != m_extra_mf_fields.end(); /* [한국어] 엔트리가 있으면 true — 아직 fill 대기 중 */
}

/*
 * [한국어]
 * baseline_cache::print - 캐시 통계 출력
 *
 * @fp: 출력 대상 파일 스트림
 * @accesses: [입출력] 누적 접근 수
 * @misses: [입출력] 누적 미스 수
 *
 * 캐시 이름과 함께 tag_array의 통계를 출력한다. 시뮬레이션 결과 리포트에서 사용.
 *
 * 호출 체인:
 *   gpu-sim.cc (결과 리포트) → [이 함수] → tag_array::print()
 */
void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str()); /* [한국어] 캐시 이름 출력 (예: "L1D_SM0") */
  m_tag_array->print(fp, accesses, misses);    /* [한국어] 태그 배열 통계(접근/미스 등) 출력 및 누적 */
}

/*
 * [한국어]
 * baseline_cache::display_state - 캐시 현재 상태 디버그 출력
 *
 * @fp: 출력 대상 파일 스트림
 *
 * 캐시 이름과 MSHR 내용을 출력한다. 시뮬레이션 디버깅 시 캐시 상태 스냅샷 확인용.
 *
 * 호출 체인:
 *   shader.cc 또는 gpu-sim.cc (디버그 출력) → [이 함수] → mshr_table::display()
 */
void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str()); /* [한국어] 캐시 이름 출력 */
  m_mshrs.display(fp);                         /* [한국어] MSHR 현재 내용 출력 */
  fprintf(fp, "\n");
}

/*
 * [한국어]
 * baseline_cache::inc_aggregated_stats - L1 또는 L2 전역 집계 통계에 접근 결과 기록
 *
 * @status: probe 결과 상태
 * @cache_status: 최종 캐시 접근 결과 상태
 * @mf: 메모리 요청 패킷 (스트림 ID, 접근 타입 추출용)
 * @level: 이 통계를 어느 캐시 레벨에 기록할지 (L1_GPU_CACHE 또는 L2_GPU_CACHE)
 *
 * GPU 전역 aggregated 통계(gpu-sim.cc의 aggregated_l1_stats/l2_stats)에 이 요청의
 * 접근 결과를 기록한다. 개별 SM 통계와 별도로 전체 GPU 차원의 통계를 집계하기 위해 사용.
 *
 * 호출 체인:
 *   l1_cache::access() 또는 l2_cache::access() → [이 함수]
 */
void baseline_cache::inc_aggregated_stats(cache_request_status status,
                                          cache_request_status cache_status,
                                          mem_fetch *mf,
                                          enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    /* [한국어] L1 캐시 집계 통계에 기록 — probe/access 결과를 select_stats_status로 정규화 */
    m_gpu->aggregated_l1_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    /* [한국어] L2 캐시 집계 통계에 기록 */
    m_gpu->aggregated_l2_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/*
 * [한국어]
 * baseline_cache::inc_aggregated_fail_stats - L1 또는 L2 전역 집계 실패 통계에 기록
 *
 * @status, @cache_status, @mf, @level: inc_aggregated_stats와 동일
 *
 * RESERVATION_FAIL 등 실패 유형별 집계 통계를 GPU 전역 차원에서 기록한다.
 * 개별 캐시의 fail_stats와 별도로 전체 GPU 실패 패턴 분석에 사용된다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수]
 */
void baseline_cache::inc_aggregated_fail_stats(
    cache_request_status status, cache_request_status cache_status,
    mem_fetch *mf, enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    /* [한국어] L1 집계 실패 통계에 기록 */
    m_gpu->aggregated_l1_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    /* [한국어] L2 집계 실패 통계에 기록 */
    m_gpu->aggregated_l2_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/*
 * [한국어]
 * baseline_cache::inc_aggregated_stats_pw - L1 또는 L2 전역 윈도우별 집계 통계에 기록
 *
 * @status, @cache_status, @mf, @level: inc_aggregated_stats와 동일
 *
 * 윈도우별 집계 통계(m_stats_pw)를 GPU 전역에서 기록한다. AerialVision 주기적 가시화용.
 *
 * 호출 체인:
 *   data_cache::access() → [이 함수]
 */
void baseline_cache::inc_aggregated_stats_pw(cache_request_status status,
                                             cache_request_status cache_status,
                                             mem_fetch *mf,
                                             enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    /* [한국어] L1 윈도우 집계 통계에 기록 */
    m_gpu->aggregated_l1_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    /* [한국어] L2 윈도우 집계 통계에 기록 */
    m_gpu->aggregated_l2_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/// Read miss handler without writeback
/*
 * [한국어]
 * baseline_cache::send_read_request (wb 없는 버전) - writeback 정보 불필요한 read 요청 발송
 *
 * @addr: 접근 주소
 * @block_addr: 블록 정렬 주소
 * @cache_index: tag_array::probe()가 선택한 교체 후보 라인 인덱스
 * @mf: 발송할 메모리 요청 패킷
 * @time: 현재 시뮬레이션 사이클
 * @do_miss: [출력] 요청이 성공적으로 처리되었으면 true
 * @events: [출력] 발생한 캐시 이벤트 목록
 * @read_only: true이면 read_only_cache용 (writeback 없음)
 * @wa: write-allocate 여부 (true이면 READ_REQUEST_SENT 이벤트 생성 안 함)
 *
 * writeback 정보가 필요 없는 경우를 위한 간소화 버전. 내부적으로 wb 버전에 위임.
 *
 * 호출 체인:
 *   read_only_cache::access() → [이 함수] → send_read_request(wb 버전)
 */
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;         /* [한국어] writeback 발생 여부 — 이 버전에서는 사용하지 않음 */
  evicted_block_info e;    /* [한국어] evicted 블록 정보 — 이 버전에서는 사용하지 않음 */
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa); /* [한국어] wb 버전으로 위임 */
}

/// Read miss handler. Check MSHR hit or MSHR available
/*
 * [한국어]
 * baseline_cache::send_read_request (wb 버전) - 읽기 미스 처리: MSHR 등록 및 miss_queue 발송
 *
 * @addr: 접근 주소
 * @block_addr: 블록 정렬 주소
 * @cache_index: 교체 후보 라인 인덱스
 * @mf: 발송할 메모리 요청 패킷
 * @time: 현재 사이클
 * @do_miss: [출력] 성공적으로 처리되었으면 true
 * @wb: [출력] writeback이 필요하면 true (MODIFIED 라인 교체 시)
 * @evicted: [출력] evict된 라인 정보
 * @events: [출력] 발생한 캐시 이벤트 목록
 * @read_only: read_only_cache 여부
 * @wa: write-allocate 여부
 *
 * 읽기 미스 처리의 핵심 함수. 동작 단계:
 * 1. MSHR 주소(mshr_addr) 계산: 섹터 캐시에서는 섹터 정렬 주소 사용
 * 2. MSHR hit + avail: tag_array access 후 MSHR merge (새 메모리 요청 불필요)
 *    → MSHR_HIT 통계 기록, do_miss=true
 * 3. MSHR miss + avail + miss_queue 여유: tag_array access + MSHR 등록 + miss_queue 추가
 *    → extra_mf_fields에 복원 정보 저장, mf의 주소/크기를 mshr/atom_sz로 변경
 *    → READ_REQUEST_SENT 이벤트 추가 (wa=false인 경우), do_miss=true
 * 4. MSHR hit + !avail → MSHR_MERGE_ENRTY_FAIL 통계 기록
 * 5. MSHR miss + !avail → MSHR_ENRTY_FAIL 통계 기록
 *
 * 호출 체인:
 *   data_cache::rd_miss_base() → [이 함수]
 *   data_cache::wr_miss_wa_naive() → [이 함수]
 */
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr()); /* [한국어] MSHR용 주소 계산 — 섹터 캐시에서는 섹터 정렬 주소 */
  bool mshr_hit = m_mshrs.probe(mshr_addr);   /* [한국어] 동일 블록에 대한 MSHR 엔트리 존재 여부 확인 */
  bool mshr_avail = !m_mshrs.full(mshr_addr); /* [한국어] MSHR에 추가 공간(merge 슬롯 또는 신규 엔트리) 여부 확인 */
  if (mshr_hit && mshr_avail) {
    /* [한국어] MSHR hit + avail: 동일 블록의 이전 요청이 이미 하위 메모리로 발송됨
     * 새 메모리 요청 없이 기존 요청에 merge하면 됨 */
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf); /* [한국어] read_only: wb 없이 LRU만 갱신 */
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf); /* [한국어] 일반 캐시: wb/evicted 정보 포함하여 tag 접근 */

    m_mshrs.add(mshr_addr, mf);                                          /* [한국어] 기존 MSHR 엔트리에 이 요청 merge */
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID()); /* [한국어] MSHR_HIT 통계 기록 */
    do_miss = true; /* [한국어] miss 처리됨 표시 — 호출자가 이 결과를 MISS로 처리 */

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    /* [한국어] 완전한 신규 미스: MSHR에 새 엔트리 추가 + miss_queue에 메모리 요청 발송 */
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf); /* [한국어] read_only: wb 없이 접근 */
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf); /* [한국어] ON_MISS 시 이 시점에 RESERVED 할당 */

    m_mshrs.add(mshr_addr, mf); /* [한국어] 새 MSHR 엔트리 생성 및 이 요청 등록 */
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    /* [한국어] fill() 시 원래 주소/크기/인덱스 복원을 위한 보조 정보 저장 */
    mf->set_data_size(m_config.get_atom_sz()); /* [한국어] mf의 데이터 크기를 원자 크기(전체 라인/섹터)로 변경 */
    mf->set_addr(mshr_addr);       /* [한국어] mf의 주소를 MSHR 주소(섹터 정렬)로 변경 — 하위 메모리 요청용 */
    m_miss_queue.push_back(mf);    /* [한국어] miss_queue에 추가 — cycle()에서 하위 메모리로 전송 */
    mf->set_status(m_miss_queue_status, time); /* [한국어] mf 상태를 miss_queue 상태로 업데이트 */
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));
    /* [한국어] write-allocate가 아닌 일반 읽기 미스이면 READ_REQUEST_SENT 이벤트 기록 */

    do_miss = true; /* [한국어] miss 처리됨 표시 */
  } else if (mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                           mf->get_streamID()); /* [한국어] MSHR 엔트리는 있지만 merge 슬롯 부족 — 실패 통계 기록 */
  else if (!mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                           mf->get_streamID()); /* [한국어] 신규 MSHR 엔트리 공간 부족 — 실패 통계 기록 */
  else
    assert(0); /* [한국어] 논리적으로 도달 불가능한 경우 — 위 4가지 조합이 전부 */
}

/// Sends write request to lower level memory (write or writeback)
/*
 * [한국어]
 * data_cache::send_write_request - 쓰기 또는 writeback 요청을 miss_queue에 추가
 *
 * @mf: 발송할 쓰기 요청 패킷
 * @request: 발생한 캐시 이벤트 (WRITE_REQUEST_SENT 또는 WRITE_BACK_REQUEST_SENT)
 * @time: 현재 시뮬레이션 사이클
 * @events: [출력] 이벤트 기록 목록
 *
 * 쓰기 요청(write-through 또는 writeback)을 events에 기록하고 miss_queue에 추가한다.
 * miss_queue에 들어가면 cycle()에서 하위 메모리 인터페이스로 전송된다.
 *
 * 호출 체인:
 *   data_cache::wr_hit_wt/we(), wr_miss_*(), rd_miss_base() → [이 함수]
 */
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);         /* [한국어] 쓰기 이벤트 타입(WRITE_REQUEST_SENT 등) 기록 */
  m_miss_queue.push_back(mf);        /* [한국어] miss_queue에 추가 — cycle()에서 하위 메모리로 전송 */
  mf->set_status(m_miss_queue_status, time); /* [한국어] mf 상태를 miss_queue 대기 상태로 갱신 */
}

/*
 * [한국어]
 * data_cache::update_m_readable - 섹터 캐시의 readable 상태 업데이트
 *
 * @mf: 방금 쓰기를 완료한 메모리 요청 패킷 (접근 섹터/바이트 마스크 포함)
 * @cache_index: 업데이트할 캐시 라인 인덱스
 *
 * lazy_fetch_on_read 정책에서 부분 쓰기 후 해당 섹터의 모든 바이트가 채워졌는지 확인하여
 * readable 상태를 갱신한다. 섹터 내 모든 바이트가 dirty(written)이면 readable=true로 설정.
 * 이로써 나중에 같은 섹터를 읽을 때 하위 메모리 fetch 없이 캐시에서 바로 반환 가능하다.
 *
 * 호출 체인:
 *   data_cache::wr_hit_wb/wt/we(), wr_miss_wa_lazy_fetch_on_read() → [이 함수]
 */
void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 업데이트할 캐시 블록 획득 */
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    /* [한국어] 요청이 접근한 섹터들을 순회 */
    if (mf->get_access_sector_mask().test(i)) {
      /* [한국어] i번 섹터에 접근한 경우 해당 섹터의 모든 바이트가 채워졌는지 확인 */
      bool all_set = true; /* [한국어] 섹터 내 모든 바이트가 dirty인지 여부 */
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        /* [한국어] 섹터 범위(i*SECTOR_SIZE ~ (i+1)*SECTOR_SIZE) 내 모든 바이트 확인 */
        if (!block->get_dirty_byte_mask().test(k)) {
          /* [한국어] dirty로 표시되지 않은 바이트가 있으면 아직 incomplete — readable 불가 */
          all_set = false;
          break;
        }
      }
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
      /* [한국어] 섹터의 모든 바이트가 dirty이면 readable=true — 이후 읽기 시 fetch 불필요 */
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
/*
 * [한국어]
 * data_cache::wr_hit_wb - write-back 정책의 캐시 히트 쓰기 처리
 *
 * @addr: 쓰기 대상 주소
 * @cache_index: tag_array::probe()가 찾은 히트 라인 인덱스
 * @mf: 쓰기 요청 패킷
 * @time: 현재 사이클 (LRU 갱신용)
 * @events: [출력] 이벤트 목록 (이 함수에서는 추가 없음)
 * @status: probe 결과 (HIT)
 * @return: HIT
 *
 * write-back 정책: 캐시에만 쓰고 MODIFIED로 표시. 하위 메모리에는 즉시 기록하지 않음.
 * LRU 갱신 → MODIFIED 표시 → byte_mask 업데이트 → readable 상태 갱신.
 * 나중에 이 라인이 evict될 때 writeback이 발생한다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (wr_hit 함수 포인터)
 */
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  /* [한국어] LRU 타임스탬프 갱신 — 이 접근으로 라인이 최근 사용됨으로 표시 */
  cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 캐시 블록 포인터 획득 */
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty(); /* [한국어] 처음 MODIFIED 상태가 되는 경우 더티 카운터 증가 */
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask()); /* [한국어] 해당 섹터를 MODIFIED(더티)로 표시 */
  block->set_byte_mask(mf);  /* [한국어] 쓰기가 발생한 바이트 위치를 byte mask에 기록 */
  update_m_readable(mf, cache_index); /* [한국어] 섹터의 모든 바이트가 채워졌으면 readable=true로 갱신 */

  return HIT; /* [한국어] 쓰기 히트 완료 */
}

/// Write-through hit: Directly send request to lower level memory
/*
 * [한국어]
 * data_cache::wr_hit_wt - write-through 정책의 캐시 히트 쓰기 처리
 *
 * @addr, @cache_index, @mf, @time, @events, @status: wr_hit_wb와 동일
 * @return: HIT 또는 RESERVATION_FAIL (miss_queue 가득 찬 경우)
 *
 * write-through 정책: 캐시에 쓰는 동시에 하위 메모리에도 즉시 쓰기 요청을 발송한다.
 * miss_queue가 가득 차면 이 사이클에 처리 불가 → RESERVATION_FAIL 반환.
 * MODIFIED 표시 + byte_mask + readable 갱신 후 하위 메모리로 WRITE_REQUEST_SENT 발송.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (wr_hit 함수 포인터)
 */
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    /* [한국어] write-through는 하위 메모리 쓰기가 필수이므로 miss_queue에 공간이 있어야 함 */
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID()); /* [한국어] 실패 통계 기록 */
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  /* [한국어] LRU 타임스탬프 갱신 */
  cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 캐시 블록 포인터 획득 */
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty(); /* [한국어] 처음 MODIFIED 상태가 되는 경우 더티 카운터 증가 */
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask()); /* [한국어] 해당 섹터 MODIFIED 표시 */
  block->set_byte_mask(mf);  /* [한국어] 쓰기 바이트 위치 기록 */
  update_m_readable(mf, cache_index); /* [한국어] readable 상태 갱신 */

  // generate a write-through
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  /* [한국어] 하위 메모리에도 동시에 쓰기 요청 발송 — write-through 핵심 동작 */

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
/*
 * [한국어]
 * data_cache::wr_hit_we - write-evict 정책의 캐시 히트 쓰기 처리
 *
 * @addr, @cache_index, @mf, @time, @events, @status: wr_hit_wb와 동일
 * @return: HIT 또는 RESERVATION_FAIL
 *
 * write-evict 정책: 쓰기 시 캐시 라인을 evict하고 하위 메모리에만 기록.
 * GPU의 global memory 쓰기에서 사용 — 같은 데이터를 다시 읽을 가능성이 낮아 캐시에 보관 불필요.
 * 쓰기 요청을 하위 메모리로 발송 후 해당 섹터를 INVALID로 설정.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (wr_hit 함수 포인터, global write-evict)
 */
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    /* [한국어] 하위 메모리 쓰기 요청을 위한 miss_queue 공간 필요 */
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] evict할 캐시 블록 포인터 획득 */
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  /* [한국어] 하위 메모리에 쓰기 요청 발송 — 데이터가 하위 메모리에 영구 기록됨 */

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());
  /* [한국어] 해당 섹터를 INVALID로 설정 — write-evict의 핵심: 캐시에서 제거 */

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
/*
 * [한국어]
 * data_cache::wr_hit_global_we_local_wb - 글로벌 쓰기는 evict, 로컬 쓰기는 write-back
 *
 * @addr, @cache_index, @mf, @time, @events, @status: wr_hit_wb와 동일
 * @return: HIT 또는 RESERVATION_FAIL
 *
 * private 캐시(L1D)에서 글로벌 메모리 쓰기(GLOBAL_ACC_W)와 로컬 메모리 쓰기를 구분한다.
 * 글로벌 쓰기는 다른 SM과 공유되므로 write-evict로 일관성 유지 (L2에 기록).
 * 로컬(shared/private) 쓰기는 같은 SM에서만 접근하므로 write-back으로 성능 최적화.
 * Fermi 이후 NVIDIA GPU의 private L1 캐시 쓰기 정책을 모델링한다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (wr_hit 함수 포인터)
 */
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  /* [한국어] 글로벌 메모리 쓰기이면 evict=true, 로컬/공유 메모리 쓰기이면 evict=false */
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
    /* [한국어] 글로벌 쓰기: write-evict — 하위 메모리에 기록하고 캐시에서 삭제 */
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
    /* [한국어] 로컬 쓰기: write-back — 캐시에만 쓰고 나중에 evict 시 writeback */
}

/****** Write-miss functions (Set by config file) ******/
/* [한국어] 쓰기 미스 처리 함수들 — 설정 파일의 캐시 정책에 따라 m_wr_miss 함수 포인터로 선택됨
 * 정책 종류: naive(구 3.x 방식), fetch_on_write, lazy_fetch_on_read, no_write_allocate */

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
/*
 * [한국어]
 * data_cache::wr_miss_wa_naive - 쓰기 미스 시 write-allocate 처리 (구 GPGPU-Sim 3.x 방식)
 *
 * @addr: 쓰기 요청 주소
 * @cache_index: 태그 배열에서 할당된 캐시 라인 인덱스
 * @mf: 원본 쓰기 요청 mem_fetch 패킷
 * @time: 현재 사이클
 * @events: 이 사이클에 발생한 캐시 이벤트 리스트 (상위로 전달됨)
 * @status: tag_array::access()가 반환한 상태 (MISS 또는 HIT_RESERVED)
 * @return: MISS (성공적으로 처리됨) 또는 RESERVATION_FAIL (이 사이클에 처리 불가)
 *
 * write-allocate 정책의 naive 구현: 쓰기 미스 발생 시
 *   1. miss_queue에 최대 2개 추가 슬롯(쓰기 요청 + writeback)이 확보되는지, MSHR이 가용한지 확인
 *   2. 원본 쓰기 요청을 하위 메모리로 전송 (WRITE_REQUEST_SENT)
 *   3. write-allocate용 읽기 요청(n_mf)을 새로 생성하여 send_read_request()로 전송
 *   4. 교체된 블록이 MODIFIED 상태이고 write-through가 아니면 writeback 전송 (WRITE_BACK_REQUEST_SENT)
 * fetch_on_write와 달리 쓰기 데이터를 캐시에 직접 반영하지 않고 읽기 요청만 보내는 단순 방식이다.
 * 자원 부족(miss_queue full, MSHR full) 시 RESERVATION_FAIL을 반환하여 warp를 스톨시킨다.
 *
 * 호출 체인:
 *   data_cache::send_write_request() / data_cache::access() → m_wr_miss 함수 포인터 → [이 함수]
 *     → send_write_request() (원본 쓰기)
 *     → send_read_request() (write-allocate 읽기)
 *     → send_write_request() (writeback, 조건부)
 */
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 캐시 라인 정렬 주소 추출 */
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr()); /* [한국어] MSHR 룩업용 주소 (섹터 캐시는 섹터 단위, 일반은 블록 단위) */

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit = m_mshrs.probe(mshr_addr); /* [한국어] 해당 주소가 MSHR에 이미 존재하는지 확인 (merge 가능 여부) */
  bool mshr_avail = !m_mshrs.full(mshr_addr); /* [한국어] MSHR에 새 엔트리를 추가할 공간이 있는지 확인 */
  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    /* [한국어] 자원 부족 조건 검사:
     * miss_queue_full(2): 최악의 경우 2개 추가 슬롯(쓰기+writeback) 필요
     * MSHR 조건: (hit이지만 merge 불가) OR (hit 없이 새 MSHR 엔트리 공간도 없음) */
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
      /* [한국어] miss 큐가 가득 찬 경우 실패 원인 통계 기록 */
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                             mf->get_streamID());
      /* [한국어] MSHR에 같은 주소가 있지만 merge 엔트리 한도 초과 — MSHR merge 실패 통계 */
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                             mf->get_streamID());
      /* [한국어] MSHR에 주소도 없고 새 엔트리 공간도 없음 — MSHR 신규 할당 실패 통계 */
    else
      assert(0); /* [한국어] 위 세 경우 외의 상황은 로직 버그 — 즉시 패닉 */

    return RESERVATION_FAIL; /* [한국어] 자원 부족으로 이 사이클에 처리 불가, warp 스톨 */
  }

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  /* [한국어] 원본 쓰기 요청을 하위 메모리(L2 또는 DRAM)로 전송
   * WRITE_REQUEST_SENT 이벤트를 events에 추가하여 상위 파이프라인에 알림 */
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);
  /* [한국어] write-allocate 읽기를 위한 새 메모리 접근 디스크립터 생성
   * m_wr_alloc_type: 설정된 write-allocate 접근 타입 (예: GLOBAL_ACC_R)
   * get_atom_sz(): 원자 접근 크기 — 캐시 라인 또는 섹터 크기
   * false: 쓰기가 아닌 읽기로 수행 (캐시 라인 fetch) */

  mem_fetch *n_mf = new mem_fetch(
      *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
  /* [한국어] write-allocate 읽기용 새 mem_fetch 패킷 생성
   * 원본 mf의 wid/sid/tpc 등 컨텍스트 정보를 상속하여 SM 추적 가능하게 함
   * NULL: 상위 inst 없음 (통계 목적의 요청) */

  bool do_miss = false; /* [한국어] send_read_request()가 실제로 미스 큐에 추가했는지 여부 */
  bool wb = false;       /* [한국어] 교체로 인해 writeback이 필요한지 여부 */
  evicted_block_info evicted; /* [한국어] 교체된 블록의 주소/바이트마스크/크기 정보 */

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);
  /* [한국어] write-allocate 읽기 요청을 MSHR 및 miss_queue에 추가
   * do_miss: 미스 처리가 실제로 이루어졌으면 true
   * wb: 교체된 블록이 dirty이면 true (writeback 필요)
   * false: read_only 아님, true: write_allocate 요청임 */

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));
  /* [한국어] WRITE_ALLOCATE_SENT 이벤트를 기록하여 상위 파이프라인(shader)이 인지하게 함 */

  if (do_miss) {
    /* [한국어] send_read_request()가 실제로 미스 처리를 수행한 경우 */
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      /* [한국어] 교체된 블록이 dirty이고 write-through가 아닌 경우: writeback 전송 필요
       * write-through라면 이미 하위 메모리에 최신 데이터가 있으므로 writeback 불필요 */
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      /* [한국어] naive 방식에서 writeback은 완전 MISS일 때만 발생 — 안전성 검증 */
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      /* [한국어] 교체된 dirty 블록을 하위 메모리에 쓰기 위한 writeback mem_fetch 생성
       * evicted.m_block_addr: 교체된 블록의 주소
       * m_wrbk_type: writeback 접근 타입 (예: L1_WRBK_ACC)
       * evicted.m_modified_size: dirty 바이트 수 — 실제 변경된 부분만 기록 */
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      /* [한국어] 고급 L2 해싱 사용 시 chip ID가 잘못 설정될 수 있으므로 원본 mf의 chip으로 보정 */
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      /* [한국어] 올바른 sub_partition(메모리 파티션)으로 설정하여 정확한 경로로 라우팅 */
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
      /* [한국어] writeback 요청을 miss_queue에 추가하고 WRITE_BACK_REQUEST_SENT 이벤트 기록 */
    }
    return MISS; /* [한국어] write-allocate 처리 성공 — 상위에 MISS 상태 반환 */
  }

  return RESERVATION_FAIL; /* [한국어] send_read_request()가 처리 못한 경우 — 자원 부족 */
}

/*
 * [한국어]
 * data_cache::wr_miss_wa_fetch_on_write - 쓰기 미스 시 fetch-on-write write-allocate 처리
 *
 * @addr: 쓰기 요청 주소
 * @cache_index: 태그 배열에서 할당된 캐시 라인 인덱스
 * @mf: 원본 쓰기 요청 mem_fetch 패킷
 * @time: 현재 사이클
 * @events: 이 사이클에 발생한 캐시 이벤트 리스트
 * @status: tag_array::access()가 반환한 상태 (MISS 또는 HIT_RESERVED)
 * @return: MISS (성공) 또는 RESERVATION_FAIL (자원 부족)
 *
 * 쓰기가 전체 캐시 라인/섹터를 커버하는지(byte_mask.count() == atom_sz)에 따라 두 경로로 분기:
 *
 * [전체 라인 쓰기 경로]
 *   - 하위 메모리에서 fetch 불필요 — 직접 tag_array::access()로 라인 할당
 *   - 블록을 MODIFIED로 마킹하고 byte_mask 기록
 *   - HIT_RESERVED이면 set_ignore_on_fill(true): 나중에 도착하는 fill이 우리 쓰기를 덮지 않도록
 *   - 교체된 블록이 MODIFIED이고 write-through 아니면 writeback 전송
 *
 * [부분 라인 쓰기 경로]
 *   - MSHR과 miss_queue 가용성 확인 (1개 슬롯 필요)
 *   - Write-Read-Write 해저드 방지: MSHR에 이미 read_after_write pending이면 거부
 *   - write-allocate 읽기 요청(n_mf) 생성 및 send_read_request()로 전송
 *   - set_modified_on_fill(true): fill 도착 시 자동으로 MODIFIED 마킹
 *   - set_byte_mask_on_fill(true): fill 도착 시 byte_mask 갱신
 *   - 교체된 블록이 MODIFIED이면 writeback 전송
 *
 * 호출 체인:
 *   data_cache::access() → m_wr_miss 함수 포인터 → [이 함수]
 *     → m_tag_array->access() (전체 라인 경로)
 *     → send_read_request() (부분 라인 경로)
 *     → send_write_request() (writeback, 조건부)
 */
enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 캐시 라인 정렬 주소 */
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr()); /* [한국어] MSHR 룩업용 주소 */

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    /* [한국어] 쓰기가 전체 캐시 라인/섹터를 커버하는 경우
     * byte_mask.count()가 atom_sz(섹터 크기)와 같으면 모든 바이트를 덮어쓰므로
     * 하위 메모리에서 fetch 없이 바로 MODIFIED로 마킹 가능 */
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
      /* [한국어] writeback 가능성을 위한 miss_queue 슬롯 0개 추가 필요 — 기본 가용성 확인 */
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
      return RESERVATION_FAIL;  // cannot handle request this cycle
      /* [한국어] miss_queue 가득 찬 경우 이 사이클 처리 불가 */
    }

    bool wb = false;            /* [한국어] 교체로 인한 writeback 필요 여부 */
    evicted_block_info evicted; /* [한국어] 교체된 블록 정보 */

    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    /* [한국어] 태그 배열에 접근하여 캐시 라인 할당 — 함수 파라미터 status를 지역 변수로 가림 */
    assert(status != HIT); /* [한국어] 쓰기 미스 경로이므로 HIT은 불가 — 안전성 검증 */
    cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 할당된 캐시 블록 포인터 획득 */
    if (!block->is_modified_line()) {
      /* [한국어] 이전에 MODIFIED 상태가 아니었다면 dirty 카운터 증가
       * is_modified_line()이 false = 이 섹터 중 어떤 섹터도 MODIFIED 아님 */
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    /* [한국어] 해당 섹터를 MODIFIED 상태로 마킹 — 나중에 evict 시 writeback 필요함을 표시 */
    block->set_byte_mask(mf);
    /* [한국어] mf의 byte_mask를 블록에 기록 — 어떤 바이트가 실제로 쓰였는지 추적 */
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    /* [한국어] HIT_RESERVED: 이 라인에 대한 읽기 요청이 이미 진행 중인 경우
     * ignore_on_fill=true: fill(데이터 도착)이 와도 이 섹터는 우리 쓰기 값으로 덮지 않도록 보호 */

    if (status != RESERVATION_FAIL) {
      /* [한국어] 라인 할당이 성공한 경우 — 교체 블록 writeback 처리 */
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        /* [한국어] 교체된 블록이 dirty이고 write-through 정책이 아니면 writeback 필요 */
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());
        /* [한국어] 교체된 dirty 블록의 writeback mem_fetch 생성 */
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        /* [한국어] 고급 L2 해싱 시 chip ID 보정 */
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        /* [한국어] 올바른 메모리 파티션으로 설정 */
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
        /* [한국어] writeback 요청을 miss_queue에 추가 및 이벤트 기록 */
      }
      return MISS; /* [한국어] 전체 라인 쓰기 성공 — MISS 반환 (write-allocate 완료) */
    }
    return RESERVATION_FAIL; /* [한국어] 라인 할당 실패 (RESERVATION_FAIL) — warp 스톨 */
  } else {
    /* [한국어] 부분 라인 쓰기 경로: 쓰기 데이터가 전체 섹터를 커버하지 않는 경우
     * 나머지 바이트를 하위 메모리에서 fetch해야 하므로 읽기 요청이 필요 */
    bool mshr_hit = m_mshrs.probe(mshr_addr); /* [한국어] MSHR에 같은 주소 이미 존재 여부 */
    bool mshr_avail = !m_mshrs.full(mshr_addr); /* [한국어] MSHR에 추가 엔트리 공간 여부 */
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      /* [한국어] 자원 부족 검사: miss_queue 1개 슬롯 필요 + MSHR 가용성 확인 */
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        /* [한국어] miss 큐 가득 참 — 실패 통계 기록 */
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                               mf->get_streamID());
        /* [한국어] MSHR merge 한도 초과 — 실패 통계 기록 */
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                               mf->get_streamID());
        /* [한국어] MSHR 신규 할당 불가 — 실패 통계 기록 */
      else
        assert(0); /* [한국어] 예상치 못한 경우 — 로직 버그 패닉 */

      return RESERVATION_FAIL; /* [한국어] 자원 부족 — warp 스톨 */
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      /* [한국어] Write-Read-Write 해저드 방지:
       * 같은 주소에 대해 쓰기 요청 후 읽기가 MSHR에 pending 중인데 또 쓰기가 오면
       * 첫 번째 쓰기 값이 두 번째 쓰기로 덮어씌워져 읽기 결과가 잘못됨 → 거부 */
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING,
                             mf->get_streamID());
      /* [한국어] MSHR_RW_PENDING 실패 통계 기록 */
      return RESERVATION_FAIL; /* [한국어] 해저드 회피를 위해 이 사이클 처리 거부 */
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);
    /* [한국어] write-allocate 읽기 디스크립터 생성 — 부분 쓰기를 완성하기 위한 fetch */

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);
    /* [한국어] write-allocate 읽기용 새 mem_fetch 생성
     * 마지막 인자 mf: 원본 쓰기 요청을 상위 mf로 연결하여 MSHR merge 시 추적 가능 */

    new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 캐시 라인 정렬 주소 (else 블록 내 재선언) */
    bool do_miss = false; /* [한국어] 실제 미스 처리 수행 여부 */
    bool wb = false;       /* [한국어] writeback 필요 여부 */
    evicted_block_info evicted; /* [한국어] 교체된 블록 정보 */
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);
    /* [한국어] write-allocate 읽기 요청을 MSHR 및 miss_queue에 추가 */

    cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 할당된 캐시 블록 포인터 */
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    /* [한국어] fill 도착 시 자동으로 MODIFIED 상태로 마킹 — 우리의 쓰기가 반영됨을 표시 */
    block->set_byte_mask_on_fill(true);
    /* [한국어] fill 도착 시 byte_mask를 갱신하여 어느 바이트가 쓰였는지 추적 */

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));
    /* [한국어] WRITE_ALLOCATE_SENT 이벤트 기록 */

    if (do_miss) {
      /* [한국어] 읽기 요청이 실제로 MSHR에 추가된 경우 */
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        /* [한국어] 교체된 블록이 dirty이고 write-through 정책이 아니면 writeback 필요 */
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());
        /* [한국어] 교체된 dirty 블록의 writeback mem_fetch 생성 */
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        /* [한국어] 고급 L2 해싱 시 chip ID 보정 */
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        /* [한국어] 올바른 메모리 파티션으로 설정 */
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
        /* [한국어] writeback 요청 전송 및 이벤트 기록 */
      }
      return MISS; /* [한국어] 부분 쓰기 write-allocate 처리 성공 */
    }
    return RESERVATION_FAIL; /* [한국어] 읽기 요청 처리 실패 — warp 스톨 */
  }
}

/*
 * [한국어]
 * data_cache::wr_miss_wa_lazy_fetch_on_read - lazy fetch-on-read write-allocate 처리
 *
 * @addr: 쓰기 요청 주소
 * @cache_index: 태그 배열에서 할당된 캐시 라인 인덱스
 * @mf: 원본 쓰기 요청 mem_fetch 패킷
 * @time: 현재 사이클
 * @events: 이 사이클에 발생한 캐시 이벤트 리스트
 * @status: tag_array::access()가 반환한 상태 (호출자가 이 함수 내에서는 m_status로 재명명)
 * @return: MISS (성공) 또는 RESERVATION_FAIL (자원 부족)
 *
 * "Lazy fetch-on-read" 방식: 쓰기 미스 시 즉시 하위 메모리에서 라인을 fetch하지 않고,
 * 캐시에 라인을 할당하고 MODIFIED로 마킹한 뒤, 나중에 읽기 미스가 발생할 때 fetch.
 * 처리 단계:
 *   1. miss_queue에 최소 1개 슬롯이 있는지 확인 (writeback 대비)
 *   2. write-through 정책이면 즉시 하위 메모리로 쓰기 전송
 *   3. tag_array::access()로 캐시 라인 할당 (MISS 또는 HIT_RESERVED)
 *   4. 블록을 MODIFIED로 마킹, byte_mask 기록
 *   5. HIT_RESERVED이면 fill 도착 시 우리 쓰기를 보존하는 플래그 설정
 *      (ignore_on_fill, modified_on_fill, byte_mask_on_fill)
 *   6. 전체 라인 쓰기: set_m_readable(true) → 나중에 읽어도 유효
 *      부분 라인 쓰기: set_m_readable(false) → 나중에 읽기 시 SECTOR_MISS 발생하여 fetch
 *   7. update_m_readable()로 dirty 바이트 마스크 기반 readable 상태 업데이트
 *   8. 교체된 블록이 MODIFIED + write-through 아니면 writeback 전송
 *
 * 호출 체인:
 *   data_cache::access() → m_wr_miss 함수 포인터 → [이 함수]
 *     → m_tag_array->access() (라인 할당)
 *     → update_m_readable() (readable 상태 갱신)
 *     → send_write_request() (write-through 및 writeback, 조건부)
 */
enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 캐시 라인 정렬 주소 */

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0)) {
    /* [한국어] miss_queue 가용성 확인 (writeback을 위한 최소 슬롯 1개 필요) */
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
    /* [한국어] miss_queue 가득 참 — 이 사이클 처리 불가, warp 스톨 */
  }

  if (m_config.m_write_policy == WRITE_THROUGH) {
    /* [한국어] write-through 정책인 경우: 캐시와 동시에 하위 메모리에도 즉시 쓰기
     * lazy_fetch_on_read는 주로 write-back 정책과 조합되지만 write-through도 지원 */
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
    /* [한국어] 원본 쓰기 요청을 하위 메모리로 전송 */
  }

  bool wb = false;            /* [한국어] 교체로 인한 writeback 필요 여부 */
  evicted_block_info evicted; /* [한국어] 교체된 블록 정보 */

  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
  /* [한국어] 태그 배열에 접근하여 캐시 라인 할당 — MISS 또는 HIT_RESERVED 반환
   * 파라미터 status와 구분하기 위해 m_status로 명명 */
  assert(m_status != HIT); /* [한국어] 쓰기 미스 경로이므로 HIT 불가 — 안전성 검증 */
  cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 할당된 캐시 블록 포인터 */
  if (!block->is_modified_line()) {
    /* [한국어] 어떤 섹터도 MODIFIED 아니었다면 dirty 카운터 증가 */
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  /* [한국어] 해당 섹터를 MODIFIED로 마킹 — 하위 메모리보다 새로운 데이터가 있음을 표시 */
  block->set_byte_mask(mf);
  /* [한국어] mf의 byte_mask를 블록에 기록 — 어떤 바이트가 실제로 쓰였는지 추적 */
  if (m_status == HIT_RESERVED) {
    /* [한국어] HIT_RESERVED: 이 라인에 대한 읽기 fetch가 이미 진행 중인 경우
     * fill 도착 시 우리 쓰기 데이터를 보존하기 위한 플래그 설정 */
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    /* [한국어] fill이 와도 이 섹터의 데이터를 덮어쓰지 않도록 (우리 쓰기 보호) */
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    /* [한국어] fill 도착 시에도 MODIFIED 상태 유지 (fill이 상태를 VALID로 바꾸지 않도록) */
    block->set_byte_mask_on_fill(true);
    /* [한국어] fill 도착 시 byte_mask 갱신하여 어느 바이트가 쓰였는지 계속 추적 */
  }

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    /* [한국어] 전체 섹터를 쓰는 경우: 모든 바이트가 유효하므로 즉시 readable=true */
    block->set_m_readable(true, mf->get_access_sector_mask());
    /* [한국어] 이 섹터는 fetch 없이도 읽기 가능 — 나중에 읽기 히트 처리 가능 */
  } else {
    /* [한국어] 부분 섹터 쓰기: 나머지 바이트가 아직 유효하지 않으므로 readable=false
     * 나중에 이 섹터를 읽으면 SECTOR_MISS → 하위 메모리에서 fetch 후 merge */
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
    /* [한국어] HIT_RESERVED인 경우: fill 도착 시 readable=true로 자동 갱신
     * fill이 오면 나머지 바이트가 채워지므로 그때부터 읽기 가능 */
  }
  update_m_readable(mf, cache_index);
  /* [한국어] dirty byte_mask를 기반으로 블록 전체의 readable 상태를 재계산
   * 섹터별 readable 비트를 종합하여 블록 수준의 readable 상태 결정 */

  if (m_status != RESERVATION_FAIL) {
    /* [한국어] 라인 할당 성공 시 교체 블록 writeback 처리 */
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      /* [한국어] 교체된 블록이 dirty이고 write-through 아닌 경우 writeback 전송 */
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      /* [한국어] 교체된 dirty 블록의 writeback mem_fetch 생성 */
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      /* [한국어] 고급 L2 해싱 시 chip ID 보정 */
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      /* [한국어] 올바른 메모리 파티션으로 설정 */
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
      /* [한국어] writeback 요청 전송 및 이벤트 기록 */
    }
    return MISS; /* [한국어] lazy fetch-on-read 쓰기 미스 처리 성공 */
  }
  return RESERVATION_FAIL; /* [한국어] 라인 할당 실패 (RESERVATION_FAIL) — warp 스톨 */
}

/// No write-allocate miss: Simply send write request to lower level memory
/*
 * [한국어]
 * data_cache::wr_miss_no_wa - 쓰기 미스 시 no-write-allocate 처리
 *
 * @addr: 쓰기 요청 주소 (이 함수에서는 직접 사용하지 않음)
 * @cache_index: 태그 배열에서의 인덱스 (이 함수에서는 사용하지 않음)
 * @mf: 원본 쓰기 요청 mem_fetch 패킷
 * @time: 현재 사이클
 * @events: 이 사이클에 발생한 캐시 이벤트 리스트
 * @status: tag_array::access()가 반환한 상태 (이 함수에서는 사용하지 않음)
 * @return: MISS (성공) 또는 RESERVATION_FAIL (miss_queue 가득 참)
 *
 * No-write-allocate 정책: 쓰기 미스 발생 시 캐시에 라인을 할당하지 않고
 * 단순히 하위 메모리(L2/DRAM)로 쓰기 요청만 전송한다.
 * GPU에서는 스레드 수가 매우 많아 write buffer를 유지하기 어려우므로 write-through 방식으로 처리.
 * 캐시 라인을 할당하지 않으므로 eviction이나 MSHR 관리가 불필요하여 가장 단순한 쓰기 미스 처리.
 * write-miss가 빈번한 워크로드에서 캐시 오염(pollution)을 방지하는 효과가 있다.
 *
 * 호출 체인:
 *   data_cache::access() → m_wr_miss 함수 포인터 → [이 함수]
 *     → send_write_request() (하위 메모리로 즉시 전송)
 */
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(0)) {
    /* [한국어] miss_queue 가용성 확인 — 쓰기 요청을 위한 슬롯 필요 */
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    /* [한국어] MISS_QUEUE_FULL 실패 통계 기록 */
    return RESERVATION_FAIL;  // cannot handle request this cycle
    /* [한국어] miss_queue 가득 참 — 이 사이클 처리 불가, warp 스톨 */
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  /* [한국어] 캐시 라인 할당 없이 원본 쓰기 요청을 하위 메모리로 바로 전송
   * WRITE_REQUEST_SENT 이벤트를 events에 추가하여 상위 파이프라인에 알림 */

  return MISS; /* [한국어] no-write-allocate 처리 완료 — MISS 반환 (캐시에는 라인 없음) */
}

/****** Read hit functions (Set by config file) ******/
/* [한국어] 읽기 히트 처리 함수들 — 설정 파일의 캐시 정책에 따라 m_rd_hit 함수 포인터로 선택됨 */

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
/*
 * [한국어]
 * data_cache::rd_hit_base - 읽기 히트의 기본 처리: LRU 갱신 + 원자 연산 특수 처리
 *
 * @addr: 읽기 주소
 * @cache_index: 히트한 캐시 라인 인덱스
 * @mf: 읽기 요청 패킷
 * @time: 현재 사이클 (LRU 갱신용)
 * @events: [출력] 이벤트 목록
 * @status: probe 결과 (HIT)
 * @return: HIT
 *
 * 일반 읽기 히트: LRU 타임스탬프만 갱신하고 반환.
 * 원자 연산(atomicAdd 등): 읽기이지만 내부적으로 수정을 포함하므로 MODIFIED 표시.
 * GPGPU-Sim에서 원자 연산은 GLOBAL_ACC_R 타입으로 처리된다.
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (m_rd_hit 함수 포인터)
 */
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  m_tag_array->access(block_addr, time, cache_index, mf); /* [한국어] LRU 타임스탬프 갱신 */
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  /* [한국어] 원자 연산은 읽기+수정이므로 읽기 히트이지만 MODIFIED로 표시해야 함 */
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R); /* [한국어] 원자 연산은 항상 GLOBAL_ACC_R 타입 */
    cache_block_t *block = m_tag_array->get_block(cache_index); /* [한국어] 캐시 블록 포인터 획득 */
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty(); /* [한국어] 새로 MODIFIED 상태가 되는 경우 더티 카운터 증가 */
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    /* [한국어] 원자 연산의 쓰기 결과를 반영하기 위해 MODIFIED로 표시 */
    block->set_byte_mask(mf); /* [한국어] 수정된 바이트 위치 기록 */
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/
/* [한국어] 읽기 미스 처리 함수들 — 설정 파일의 캐시 정책에 따라 m_rd_miss 함수 포인터로 선택됨 */

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
/*
 * [한국어]
 * data_cache::rd_miss_base - 읽기 미스 기본 처리: 하위 메모리 read 요청 + 필요시 writeback
 *
 * @addr: 읽기 주소
 * @cache_index: 교체 후보 라인 인덱스
 * @mf: 읽기 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @status: probe 결과 (MISS, SECTOR_MISS, HIT_RESERVED 등)
 * @return: MISS 또는 RESERVATION_FAIL
 *
 * 읽기 미스 처리의 기본 함수. 동작 단계:
 * 1. miss_queue 공간 확인 (writeback까지 최대 1+1=2개 요청 가능 — miss_queue_full(1) 체크)
 * 2. send_read_request()로 MSHR 등록 및 miss_queue에 read 요청 추가
 * 3. 교체 대상이 MODIFIED이고 write-back 정책이면 writeback mem_fetch 생성 후 miss_queue 추가
 * 4. chip/partition 주소는 원본 mf에서 복사 (advanced L2 hashing 시 올바른 주소 보장)
 *
 * 호출 체인:
 *   data_cache::process_tag_probe() → [이 함수] (m_rd_miss 함수 포인터)
 */
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    /* [한국어] miss_queue_full(1): read 요청 1개 + 잠재적 writeback 1개를 위한 공간 확인 */
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID()); /* [한국어] 실패 통계 기록 */
    return RESERVATION_FAIL;
  }

  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  bool do_miss = false;   /* [한국어] 요청이 성공적으로 처리되었는지 — send_read_request가 설정 */
  bool wb = false;        /* [한국어] 교체 대상 라인이 MODIFIED(더티)인지 — writeback 필요 여부 */
  evicted_block_info evicted; /* [한국어] evict된 라인 정보 (블록 주소, 수정 크기, 마스크) */
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false); /* [한국어] MSHR 등록 + miss_queue read 요청 추가 */

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    /* [한국어] 요청이 성공적으로 처리된 경우 writeback 필요성 확인 */
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      /* [한국어] 교체 대상이 MODIFIED이고 write-through가 아닌 경우 — writeback 필요
       * write-through는 이미 하위 메모리에 최신 데이터가 있으므로 writeback 불필요 */
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      /* [한국어] evicted 블록 정보를 기반으로 writeback 전용 mem_fetch 생성
       * m_wrbk_type은 WRITE_BACK_REQUEST 타입, modified_size만큼만 전송 */
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      /* [한국어] Advanced L2 hashing 사용 시 evicted 블록의 chip ID가 잘못될 수 있음
       * 원본 요청(mf)의 chip/partition 주소를 writeback mf에 복사하여 올바른 메모리 파티션으로 전송 */
      wb->set_chip(mf->get_tlx_addr().chip);           /* [한국어] chip ID 복사 */
      wb->set_partition(mf->get_tlx_addr().sub_partition); /* [한국어] sub-partition 복사 */
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
      /* [한국어] writeback 요청을 miss_queue에 추가 */
    }
    return MISS; /* [한국어] 읽기 미스 처리 완료 */
  }
  return RESERVATION_FAIL; /* [한국어] send_read_request 실패 — MSHR/miss_queue 공간 부족 */
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
/*
 * [한국어]
 * read_only_cache::access - 읽기 전용 캐시 접근 (상수/텍스처/명령어 캐시 기반)
 *
 * @addr: 접근 주소
 * @mf: 읽기 요청 패킷 (쓰기 불가)
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @return: HIT(→cache_status로 변환), MISS, RESERVATION_FAIL
 *
 * 읽기 전용 캐시(상수 캐시, 명령어 캐시)의 접근 함수.
 * 쓰기 요청은 assert로 차단된다(READ_ONLY 정책 강제).
 * HIT이면 LRU 갱신, MISS이면 send_read_request로 하위 메모리 요청,
 * RESERVATION_FAIL이면 LINE_ALLOC_FAIL 통계 기록 후 반환.
 * 통계는 select_stats_status로 probe/cache_status를 정규화하여 기록.
 *
 * 호출 체인:
 *   shader.cc (상수/명령어 캐시 접근) → [이 함수] → tag_array::probe/access + send_read_request
 */
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz()); /* [한국어] 요청 크기가 원자 크기 이하인지 검증 */
  assert(m_config.m_write_policy == READ_ONLY); /* [한국어] 이 캐시가 READ_ONLY 정책임을 검증 */
  assert(!mf->get_is_write()); /* [한국어] 읽기 전용 캐시에 쓰기 요청 금지 */
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  unsigned cache_index = (unsigned)-1; /* [한국어] probe가 설정할 캐시 라인 인덱스 — 초기값은 "미설정" 센티넬 */
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write()); /* [한국어] 태그 탐색 */
  enum cache_request_status cache_status = RESERVATION_FAIL; /* [한국어] 최종 처리 결과 — 기본값 FAIL */

  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
    /* [한국어] HIT: LRU 타임스탬프 갱신, cache_status는 access 반환값(통상 HIT) */
  } else if (status != RESERVATION_FAIL) {
    /* [한국어] MISS 또는 SECTOR_MISS: 하위 메모리로 읽기 요청 발송 */
    if (!miss_queue_full(0)) {
      /* [한국어] miss_queue에 공간이 있는 경우에만 처리 가능 */
      bool do_miss = false; /* [한국어] send_read_request가 성공하면 true로 설정 */
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false); /* [한국어] read_only=true로 호출 */
      if (do_miss)
        cache_status = MISS; /* [한국어] 성공적으로 miss 처리됨 */
      else
        cache_status = RESERVATION_FAIL; /* [한국어] MSHR 공간 부족 등으로 실패 */
    } else {
      cache_status = RESERVATION_FAIL; /* [한국어] miss_queue 가득 참 */
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID()); /* [한국어] 실패 통계 기록 */
    }
  } else {
    /* [한국어] RESERVATION_FAIL: 모든 라인이 RESERVED — LINE_ALLOC_FAIL 통계 기록 */
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                           mf->get_streamID());
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID()); /* [한국어] 전체 접근 통계 기록 */
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID()); /* [한국어] 윈도우별 접근 통계 기록 */
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
/*
 * [한국어]
 * data_cache::process_tag_probe - 태그 탐색 결과를 바탕으로 쓰기/읽기 정책 함수 디스패치
 *
 * @wr: 쓰기 요청이면 true, 읽기이면 false
 * @probe_status: tag_array::probe() 반환값 (HIT/MISS/HIT_RESERVED/RESERVATION_FAIL/SECTOR_MISS)
 * @addr: 접근 주소
 * @cache_index: probe가 결정한 캐시 라인 인덱스
 * @mf: 메모리 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @return: 최종 처리 결과 (HIT/MISS/RESERVATION_FAIL 등)
 *
 * 캐시 정책(쓰기/읽기 히트/미스)별 처리를 함수 포인터로 디스패치한다.
 * 생성자에서 gpgpusim.config 설정에 따라 4개의 함수 포인터가 설정된다:
 *   m_wr_hit: write-back/through/evict/global_we_local_wb 중 하나
 *   m_wr_miss: write-allocate naive/fetch_on_write/lazy/no-wa 중 하나
 *   m_rd_hit: rd_hit_base
 *   m_rd_miss: rd_miss_base
 *
 * 쓰기 처리:
 *   HIT → m_wr_hit 호출
 *   MISS/SECTOR_MISS/HIT_RESERVED (또는 RESERVATION_FAIL + NO_WRITE_ALLOCATE) → m_wr_miss 호출
 *   RESERVATION_FAIL (write-allocate) → LINE_ALLOC_FAIL 통계만 기록
 *
 * 읽기 처리:
 *   HIT → m_rd_hit 호출
 *   MISS/SECTOR_MISS/HIT_RESERVED → m_rd_miss 호출
 *   RESERVATION_FAIL → LINE_ALLOC_FAIL 통계만 기록
 *
 * 처리 완료 후 bandwidth_management.use_data_port()로 포트 사용 기록.
 *
 * 호출 체인:
 *   data_cache::access() → [이 함수] → m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss
 */
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  /* [한국어] 4개의 함수 포인터(m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss)는 생성자에서
   * gpgpusim.config에 따라 초기화됨 — 많은 조건 분기를 피하기 위한 함수 포인터 패턴 */
  cache_request_status access_status = probe_status; /* [한국어] 기본값: probe 결과 그대로 — 아래에서 업데이트 */
  if (wr) {  // Write
    /* [한국어] 쓰기 요청 처리 */
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
      /* [한국어] 쓰기 히트: m_wr_hit 함수 포인터가 가리키는 정책 함수 호출
       * (wb/wt/we/global_we_local_wb 중 설정된 하나) */
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      /* [한국어] MISS/SECTOR_MISS/HIT_RESERVED이거나 (RESERVATION_FAIL이지만 NO_WRITE_ALLOCATE인 경우):
       * NO_WRITE_ALLOCATE는 미스 시 라인 할당 없이 쓰기만 하므로 RESERVATION_FAIL 무관 */
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
      /* [한국어] 쓰기 미스: m_wr_miss 함수 포인터가 가리키는 정책 함수 호출 */
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      /* [한국어] RESERVATION_FAIL + WRITE_ALLOCATE: 모든 라인이 RESERVED이므로 할당 불가
       * LINE_ALLOC_FAIL 통계만 기록하고 RESERVATION_FAIL 반환 */
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  } else {  // Read
    /* [한국어] 읽기 요청 처리 */
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
      /* [한국어] 읽기 히트: m_rd_hit(= rd_hit_base) 호출 — LRU 갱신 + 원자 연산 특수 처리 */
    } else if (probe_status != RESERVATION_FAIL) {
      /* [한국어] MISS/SECTOR_MISS/HIT_RESERVED: 읽기 미스 처리 */
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
      /* [한국어] 읽기 미스: m_rd_miss(= rd_miss_base) 호출 — MSHR 등록 + miss_queue 추가 */
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      /* [한국어] RESERVATION_FAIL: 모든 라인 RESERVED — LINE_ALLOC_FAIL 통계 기록 */
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  /* [한국어] 처리 결과에 따른 데이터 포트 사용 기록 — 대역폭 모델 업데이트 */
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
/*
 * [한국어]
 * data_cache::access - 데이터 캐시의 최상위 접근 함수 (L1D, L2 공통)
 *
 * @addr: 접근 주소
 * @mf: 메모리 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이 접근 중 발생한 캐시 이벤트 목록
 * @return: 최종 접근 결과 (HIT/MISS/RESERVATION_FAIL 등)
 *
 * L1 데이터 캐시와 L2 캐시가 공통으로 사용하는 접근 함수.
 * L1/L2 간 차이는 gpgpusim.config의 캐시 정책 설정으로 처리됨.
 * 동작 단계:
 * 1. 요청 크기가 원자 크기 이하인지 검증
 * 2. probe_mode=true로 tag_array::probe() 호출 (실제 상태 변경 전 탐색)
 * 3. process_tag_probe()로 쓰기/읽기 히트/미스 함수 디스패치
 * 4. probe/access 결과를 select_stats_status로 정규화하여 통계 기록
 *
 * probe_mode=true 사용 이유: probe()에서 LRU 변경 없이 탐색만 하고,
 * process_tag_probe → m_wr/rd_hit/miss 함수에서 실제 상태 변경 수행.
 *
 * 호출 체인:
 *   shader.cc (ldst_unit::cycle) → l1_cache::access() → [이 함수]
 *   mem_sub_partition → l2_cache::access() → [이 함수]
 */
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz()); /* [한국어] 요청 크기 검증 — 원자 크기(라인/섹터) 이하여야 함 */
  bool wr = mf->get_is_write();           /* [한국어] 쓰기 요청 여부 */
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  unsigned cache_index = (unsigned)-1;    /* [한국어] probe가 설정할 라인 인덱스 — 초기값은 "미설정" 센티넬 */
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  /* [한국어] probe_mode=true: 상태 변경 없이 탐색만 수행 — cache_index와 결과 상태 결정 */
  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  /* [한국어] 탐색 결과에 따라 쓰기/읽기 히트/미스 처리 함수 디스패치 */
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status),
                    mf->get_streamID()); /* [한국어] 전체 누적 통계 기록 */
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(probe_status, access_status),
                       mf->get_streamID()); /* [한국어] 윈도우별 통계 기록 */
  return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
/*
 * [한국어]
 * l1_cache::access - L1 데이터 캐시 접근 (Fermi 이후 GPU 모델)
 *
 * @addr: 접근 주소
 * @mf: 메모리 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @return: 접근 결과
 *
 * Fermi GPU의 L1 데이터 캐시를 모델링한다. 글로벌 메모리 쓰기는 write-evict,
 * 로컬/공유 메모리 쓰기는 write-back 정책을 사용한다 (gpgpusim.config 설정).
 * 현재는 data_cache::access()를 직접 호출하며, L1 특화 동작이 필요시 여기에 추가한다.
 *
 * 호출 체인:
 *   shader.cc (ldst_unit) → [이 함수] → data_cache::access()
 */
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events); /* [한국어] L1 특화 동작 없이 기본 data_cache::access 호출 */
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
/*
 * [한국어]
 * l2_cache::access - L2 캐시 접근
 *
 * @addr: 접근 주소
 * @mf: 메모리 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @return: 접근 결과
 *
 * L2 캐시 접근 함수. 현재는 data_cache::access()를 직접 호출한다.
 * L1과의 차이는 gpgpusim.config의 캐시 설정(l2_cache_config)으로 처리된다.
 * L2 특화 동작(파티션별 통계 등)이 필요시 이 함수에 추가한다.
 *
 * 호출 체인:
 *   mem_sub_partition → [이 함수] → data_cache::access()
 */
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events); /* [한국어] L2 특화 동작 없이 기본 data_cache::access 호출 */
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
/*
 * [한국어]
 * tex_cache::access - 텍스처 캐시 접근 함수 (fragment FIFO + ROB 구조)
 *
 * @addr: 텍스처 접근 주소
 * @mf: 메모리 요청 패킷
 * @time: 현재 사이클
 * @events: [출력] 이벤트 목록
 * @return: RESERVATION_FAIL, MISS, 또는 HIT_RESERVED (절대 HIT 반환 안 함)
 *
 * 텍스처 캐시는 일반 CPU 캐시와 달리 HIT이어도 데이터가 즉시 SM에 반환되지 않는다.
 * fragment FIFO → ROB(Reorder Buffer)를 통해 in-order로 결과를 SM에 전달하기 때문이다.
 * 따라서 HIT이어도 HIT_RESERVED를 반환한다.
 *
 * 동작 단계:
 * 1. fragment_fifo, request_fifo, rob 중 하나라도 가득 차면 RESERVATION_FAIL
 * 2. tag_array::access()로 태그 접근 (즉시 라인 할당 — ON_MISS 방식)
 *    - HIT_RESERVED나 RESERVATION_FAIL은 assert로 차단 (항상 HIT 또는 MISS)
 * 3. fragment_fifo에 요청 등록 (HIT이면 miss=false, MISS이면 miss=true)
 * 4. MISS이면 ROB에 등록, extra_mf_fields 저장, tags.fill()로 VALID 표시,
 *    request_fifo에 추가(→ cycle()에서 하위 메모리로 발송), READ_REQUEST_SENT 이벤트
 * 5. HIT이면 fragment_fifo에서 결과를 기다림 (데이터는 이미 캐시에 있음)
 *
 * 텍스처 캐시의 특수성:
 *   - fragment FIFO: 요청을 순서대로 처리하기 위한 큐
 *   - ROB(Reorder Buffer): out-of-order 응답을 in-order로 정렬
 *   - result_fifo: 최종적으로 SM에 반환되는 순서가 보장된 결과 큐
 *
 * 호출 체인:
 *   shader.cc (텍스처 접근) → [이 함수] → tex_cache::cycle() → result_fifo → SM
 */
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    /* [한국어] 내부 큐 중 하나라도 가득 차면 이 사이클에 처리 불가 */
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz()); /* [한국어] 텍스처 요청 크기가 라인 크기 이하인지 검증 */

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  /* [한국어] 이 시점에서 요청 수용 결정 — 태그 접근 및 즉시 라인 할당 */
  new_addr_type block_addr = m_config.block_addr(addr); /* [한국어] 블록 정렬 주소 계산 */
  unsigned cache_index = (unsigned)-1; /* [한국어] tag_array::access가 설정할 라인 인덱스 */
  enum cache_request_status status =
      m_tags.access(block_addr, time, cache_index, mf); /* [한국어] 태그 접근 + 즉시 라인 할당 */
  enum cache_request_status cache_status = RESERVATION_FAIL; /* [한국어] 최종 결과 초기값 */
  assert(status != RESERVATION_FAIL); /* [한국어] 큐 여유 확인 후이므로 RESERVATION_FAIL 불가 */
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  /* [한국어] 텍스처 캐시는 즉시 할당이므로 HIT_RESERVED 상태도 없음 — 항상 HIT 또는 MISS */
  m_fragment_fifo.push(
      fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  /* [한국어] fragment_fifo에 등록 — (요청 패킷, 캐시 인덱스, 미스 여부, 데이터 크기) */
  if (status == MISS) {
    // we need to send a memory request...
    /* [한국어] 캐시 미스: 하위 메모리 요청 필요 */
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    /* [한국어] ROB에 등록 — 응답이 돌아왔을 때 인덱스로 fragment_fifo와 매핑 */
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config); /* [한국어] fill() 시 ROB 인덱스 복원을 위한 보조 정보 저장 */
    mf->set_data_size(m_config.get_line_sz()); /* [한국어] 전체 라인 크기로 변경 — 하위 메모리는 라인 단위로 반환 */
    m_tags.fill(cache_index, time, mf);  // mark block as valid
    /* [한국어] 태그를 VALID로 표시 — 이후 같은 블록에 대한 요청은 HIT로 처리됨 */
    m_request_fifo.push(mf); /* [한국어] request_fifo에 추가 — cycle()에서 하위 메모리 포트로 발송 */
    mf->set_status(m_request_queue_status, time); /* [한국어] 요청 상태를 request_queue 대기 중으로 설정 */
    events.push_back(cache_event(READ_REQUEST_SENT)); /* [한국어] READ_REQUEST_SENT 이벤트 기록 */
    cache_status = MISS;
  } else {
    // the value *will* *be* in the cache already
    /* [한국어] HIT: 데이터는 캐시에 있으나 fragment_fifo를 통해 in-order 전달 필요
     * HIT이어도 데이터가 즉시 SM에 반환되지 않으므로 HIT_RESERVED 반환 */
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID()); /* [한국어] 전체 누적 통계 기록 */
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID()); /* [한국어] 윈도우별 통계 기록 */
  return cache_status;
}

/*
 * [한국어]
 * tex_cache::cycle - 텍스처 캐시의 매 사이클 처리 (요청 발송 + 응답 in-order 정렬)
 *
 * 텍스처 캐시의 두 가지 주요 동작을 매 사이클 수행한다:
 * 1. request_fifo의 head를 하위 메모리 포트로 발송 시도
 * 2. fragment_fifo의 head를 result_fifo로 이동 (in-order 처리)
 *    - MISS였던 경우: ROB head가 ready 상태인지 확인 후 이동
 *    - HIT였던 경우: 캐시에 데이터가 있으므로 바로 result_fifo로 이동
 *
 * fragment_fifo + ROB 구조의 핵심:
 *   fragment_fifo는 요청 순서 보장. ROB는 out-of-order 응답을 in-order로 정렬.
 *   fragment_fifo의 head와 ROB의 head가 항상 같은 요청을 가리켜야 함(assert로 검증).
 *   ROB head가 ready일 때만 fragment_fifo를 pop하여 result_fifo에 전달.
 *
 * 호출 체인:
 *   shader.cc (텍스처 파이프라인 사이클) → [이 함수]
 */
void tex_cache::cycle() {
  // send next request to lower level of memory
  // TODO: Use different full() for sst_mem_interface?
  if (!m_request_fifo.empty()) {
    /* [한국어] request_fifo에 발송 대기 중인 요청이 있으면 하위 메모리 포트로 push 시도 */
    mem_fetch *mf = m_request_fifo.peek(); /* [한국어] 큐 head 확인 (pop 안 함) */
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      /* [한국어] 하위 메모리 포트에 공간이 있으면 발송 */
      m_request_fifo.pop();  /* [한국어] request_fifo에서 제거 */
      m_memport->push(mf);   /* [한국어] 하위 메모리 포트에 추가 — ICNT를 통해 L2/DRAM으로 전달 */
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    /* [한국어] fragment_fifo에 처리할 요청이 있고 result_fifo에 공간이 있으면 처리 진행 */
    const fragment_entry &e = m_fragment_fifo.peek(); /* [한국어] fragment_fifo head 확인 */
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      /* [한국어] 미스 요청: ROB head가 ready 상태(응답 도착)인지 확인 */
      unsigned rob_index = m_rob.next_pop_index(); /* [한국어] ROB에서 다음으로 pop할 인덱스 */
      const rob_entry &r = m_rob.peek(rob_index); /* [한국어] ROB head 항목 확인 */
      assert(r.m_request == e.m_request); /* [한국어] fragment_fifo와 ROB의 head가 동일한 요청이어야 함 — in-order 보장 */
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        /* [한국어] ROB head가 ready(하위 메모리 응답 도착)이면 result_fifo로 이동 */
        assert(r.m_index == e.m_cache_index); /* [한국어] ROB와 fragment_fifo의 캐시 인덱스 일치 검증 */
        m_cache[r.m_index].m_valid = true;         /* [한국어] 캐시 라인을 VALID로 표시 */
        m_cache[r.m_index].m_block_addr = r.m_block_addr; /* [한국어] 캐시 라인의 블록 주소 갱신 */
        m_result_fifo.push(e.m_request); /* [한국어] 완료된 요청을 result_fifo에 추가 — SM에 반환될 준비 완료 */
        m_rob.pop();          /* [한국어] ROB head 제거 */
        m_fragment_fifo.pop(); /* [한국어] fragment_fifo head 제거 */
      }
      /* [한국어] r.m_ready=false이면 아직 응답 미도착 — 이 사이클은 처리 안 함 */
    } else {
      // hit:
      /* [한국어] HIT 요청: 데이터가 이미 캐시에 있으므로 ROB 확인 없이 바로 result_fifo로 이동 */
      assert(m_cache[e.m_cache_index].m_valid); /* [한국어] HIT 라인이 여전히 valid인지 검증 */
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr())); /* [한국어] 캐시 라인 주소와 요청 주소 일치 검증 */
      m_result_fifo.push(e.m_request); /* [한국어] 완료된 요청을 result_fifo에 추가 */
      m_fragment_fifo.pop(); /* [한국어] fragment_fifo head 제거 */
    }
  }
}

/// Place returning cache block into reorder buffer
/*
 * [한국어]
 * tex_cache::fill - 하위 메모리에서 반환된 텍스처 캐시 데이터를 ROB에 배치
 *
 * @mf: 하위 메모리에서 fill 응답으로 반환된 mem_fetch 패킷
 * @time: 현재 사이클
 *
 * 텍스처 캐시 미스로 발송된 읽기 요청에 대한 응답이 하위 메모리에서 도착했을 때 호출된다.
 * ROB(Reorder Buffer)는 텍스처 접근의 순서를 보장하기 위한 구조로,
 * cycle()에서 ROB 선두 엔트리가 ready 상태가 되면 result_fifo로 이동한다.
 *
 * 처리 단계:
 *   1. SECTOR_TEX_FIFO 모드: 섹터 단위 응답을 처리 — pending_read를 감소시키고
 *      모든 섹터가 도착하면 원본 mf로 교체 (아직 대기 중이면 이 mf 삭제 후 즉시 반환)
 *   2. m_extra_mf_fields에서 이 mf의 ROB 인덱스(rob_index)를 조회
 *   3. mf 상태를 m_rob_status로 설정 (텍스처 fill 완료 상태)
 *   4. ROB에서 해당 엔트리를 찾아 m_ready=true, m_time=time으로 마킹
 *   5. cycle()이 다음 사이클에 ROB 선두를 체크하여 result_fifo로 이동
 *
 * 호출 체인:
 *   memory_sub_partition::push() (하위 메모리 응답 수신) → [이 함수]
 *   tex_cache::cycle() (ROB에서 ready 엔트리 처리)
 */
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    /* [한국어] SECTOR_TEX_FIFO 모드: 섹터 기반 텍스처 캐시에서 섹터별 응답 처리
     * 하나의 캐시 라인이 여러 섹터로 나뉘어 요청될 수 있으므로 모든 섹터 응답 대기 */
    assert(mf->get_original_mf()); /* [한국어] 섹터 응답은 반드시 원본 mf를 가져야 함 */
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    /* [한국어] 원본 mf로 extra_mf_fields 조회 — ROB 인덱스 및 pending_read 카운터 위치 */
    assert(e != m_extra_mf_fields.end()); /* [한국어] 등록되지 않은 mf는 버그 */
    e->second.pending_read--;
    /* [한국어] 이 섹터가 도착했으므로 대기 중인 섹터 수 감소 */

    if (e->second.pending_read > 0) {
      /* [한국어] 아직 다른 섹터 응답이 남아 있는 경우 — 모든 섹터 도착까지 대기 */
      // wait for the other requests to come back
      delete mf; /* [한국어] 이 섹터 패킷은 더 이상 필요 없으므로 해제 */
      return;     /* [한국어] 모든 섹터 도착 전까지 ROB 처리를 건너뜀 */
    } else {
      /* [한국어] 모든 섹터 응답이 도착한 경우 — 원본 mf로 교체하여 이후 처리 진행 */
      mem_fetch *temp = mf;          /* [한국어] 마지막 섹터 패킷을 임시 보관 */
      mf = mf->get_original_mf();   /* [한국어] 이후 처리는 원본 mf 기준으로 진행 */
      delete temp;                   /* [한국어] 마지막 섹터 패킷 해제 */
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  /* [한국어] 원본 mf로 extra_mf_fields 조회 — ROB 인덱스 및 유효성 정보 */
  assert(e != m_extra_mf_fields.end()); /* [한국어] 등록되지 않은 mf는 버그 */
  assert(e->second.m_valid);            /* [한국어] 유효하지 않은 엔트리는 버그 */
  assert(!m_rob.empty());               /* [한국어] ROB가 비어 있으면 대응하는 엔트리가 없는 버그 */
  mf->set_status(m_rob_status, time);
  /* [한국어] mf 상태를 m_rob_status(텍스처 fill 완료 상태)로 업데이트
   * m_rob_status는 tex_cache 초기화 시 설정된 상수 (예: IN_SHADER_FETCHED) */

  unsigned rob_index = e->second.m_rob_index; /* [한국어] 이 mf에 해당하는 ROB 엔트리 인덱스 */
  rob_entry &r = m_rob.peek(rob_index);       /* [한국어] ROB에서 해당 인덱스의 엔트리 참조 */
  assert(!r.m_ready); /* [한국어] 아직 ready가 아닌 상태여야 함 (중복 fill 방지) */
  r.m_ready = true;   /* [한국어] ROB 엔트리를 ready 상태로 마킹 — cycle()에서 처리 가능 */
  r.m_time = time;    /* [한국어] fill 완료 시각 기록 — 지연 시간 추적에 사용 */
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
  /* [한국어] ROB 엔트리의 블록 주소와 도착한 mf의 블록 주소가 일치하는지 검증 */
}

/*
 * [한국어]
 * tex_cache::display_state - 텍스처 캐시 내부 상태를 파일로 출력 (디버그용)
 *
 * @fp: 출력 대상 파일 포인터
 *
 * 텍스처 캐시의 현재 내부 상태를 사람이 읽을 수 있는 형식으로 출력한다.
 * 시뮬레이터 디버깅, 교착 상태 탐지, 상태 검증 등의 목적으로 사용된다.
 * 출력 내용:
 *   - fragment_fifo: 텍스처 프래그먼트(접근 요청) 대기 큐의 현재 사용량/최대 용량
 *   - ROB(Reorder Buffer): 미스 응답 대기 버퍼의 사용량/최대 용량 및 각 엔트리 상태
 *     (ready/pending, 완료 시각, 캐시 라인 인덱스, 요청 정보)
 *   - request_fifo: 하위 메모리로 전송 대기 중인 요청 큐의 사용량/최대 용량
 *   - fragment_fifo의 최선두(가장 오래된) 엔트리의 히트/미스 상태 및 요청 정보
 * const 함수이므로 내부 상태를 변경하지 않는다.
 *
 * 호출 체인:
 *   shader.cc (디버그 출력) 또는 시뮬레이터 진단 코드 → [이 함수]
 */
void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  /* [한국어] 텍스처 캐시 이름과 함께 상태 출력 헤더 */
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  /* [한국어] fragment_fifo 현재 사용량 / 최대 용량 출력
   * fragment_fifo: cycle()에서 접근 요청을 단계별로 처리하는 FIFO 큐 */
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  /* [한국어] ROB 현재 사용량 / 최대 용량 출력
   * ROB: 하위 메모리에서 데이터가 도착하기를 기다리는 엔트리들의 버퍼 */
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  /* [한국어] request_fifo 현재 사용량 / 최대 용량 출력
   * request_fifo: 하위 메모리로 전송 대기 중인 읽기 요청 FIFO */
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  /* [한국어] ROB가 비어 있지 않은 경우에만 엔트리 상세 출력 */
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    /* [한국어] ROB 엔트리를 오래된 순서(최선두)부터 최신 순으로 순회
     * n=size-1이 가장 최근에 추가된 엔트리, n=0이 가장 오래된 엔트리 */
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    /* [한국어] 순환 버퍼에서 실제 배열 인덱스를 계산
     * next_pop_index(): 다음에 pop될 위치(선두), +n으로 n번째 뒤 엔트리 접근 */
    const rob_entry &r = m_rob.peek(index); /* [한국어] 해당 인덱스의 ROB 엔트리 참조 */
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    /* [한국어] ROB 인덱스와 ready/pending 상태 출력 */
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    /* [한국어] ready 상태이면 완료된 사이클 시각 출력 */
    else
      fprintf(fp, "       "); /* [한국어] pending 상태이면 공백으로 정렬 유지 */
    fprintf(fp, "[idx=%4u]", r.m_index);
    /* [한국어] 이 ROB 엔트리가 가리키는 캐시 라인 인덱스 출력 */
    r.m_request->print(fp, false);
    /* [한국어] 이 엔트리에 연결된 mem_fetch 요청 정보 출력 (false: 축약 형식) */
  }
  if (!m_fragment_fifo.empty()) {
    /* [한국어] fragment_fifo가 비어 있지 않은 경우 선두 엔트리 출력 */
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    /* [한국어] fragment_fifo의 선두(가장 오래된) 엔트리 참조 */
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    /* [한국어] 선두 엔트리가 미스 처리 중인지 히트인지 출력 */
    f.m_request->print(fp, false);
    /* [한국어] 선두 엔트리에 연결된 요청 정보 출력 */
  }
}
/******************************************************************************************************************************************/
