// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Vijay Kandiah,
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
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
 * [한국어 설명] GPU 캐시 계층 선언 (gpu-cache.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 전체 캐시 계층 구조(L1 데이터 캐시, L2 캐시, 읽기 전용 캐시,
 * 텍스처 캐시)를 정의하는 핵심 헤더 파일이다. 캐시 블록 상태(INVALID/RESERVED/
 * VALID/MODIFIED), MSHR(Miss Status Holding Register), 태그 배열, 대역폭
 * 관리자, 통계 수집 구조체를 포함하며, 타이밍 시뮬레이션의 메모리 계층 동작
 * 전체를 선언한다. 모든 캐시 구현체(baseline_cache 하위 클래스들)는 이 헤더의
 * 인터페이스를 통해 shader.cc 등 상위 모듈과 통신한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPU 타이밍 모델(gpgpu-sim/) 내 메모리 서브시스템의 캐시 계층을 담당한다.
 * 실행 흐름:
 *   shader.cc (SM 파이프라인, ldst_unit) → l1_cache::access()
 *     → baseline_cache::cycle() → [MSHR/miss_queue → memport → ICNT]
 *     → memory_sub_partition → l2_cache::access()
 *     → dram.cc (DRAM 컨트롤러)
 * 사이클마다 shader.cc의 ldst_unit이 l1_cache::access()를 호출하고,
 * baseline_cache::cycle()이 miss_queue의 요청을 하위 메모리(ICNT)로 전달한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 사이클-레벨 타이밍 시뮬레이션.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - abstract_hardware_model.h: new_addr_type, mem_access_type, mem_fetch_interface,
 *     mem_access_byte_mask_t, mem_access_sector_mask_t, FuncCache 등 기반 타입
 *   - mem_fetch.h: 캐시를 통과하는 메모리 요청 패킷(mem_fetch) 정의
 *   - addrdec.h: SECTOR_SIZE, SECTOR_CHUNCK_SIZE 상수 및 주소 디코딩 유틸
 *   - gpu-misc.h: LOGB2 매크로 등 보조 유틸리티
 * 이 헤더에 의존하는 모듈:
 *   - shader.cc/h: l1_cache, read_only_cache(상수/텍스처) 인스턴스 생성 및 접근
 *   - gpu-sim.cc: 전체 통계 집계 시 get_stats()/get_sub_stats() 호출
 *   - memory_sub_partition.cc: l2_cache 인스턴스 생성 및 fill() 호출
 * 데이터 흐름: mem_fetch 객체가 SM(shader)에서 생성되어 l1_cache → ICNT →
 * l2_cache → DRAM 순으로 이동하며, 응답(fill)은 역방향으로 전달된다.
 *
 * === 주요 함수/구조체 요약 ===
 * cache_block_t      : 캐시 라인 하나를 표현하는 추상 기반 클래스 (tag + block_addr 보유)
 * line_cache_block   : 캐시라인 전체(128B)를 단일 상태로 관리하는 구체 구현
 * sector_cache_block : 캐시라인을 4개의 32B 섹터로 분할 관리하는 구체 구현
 * cache_config       : gpgpusim.config 문자열을 파싱하여 캐시 파라미터를 저장
 * tag_array          : 세트-연상 태그 배열 (probe/access/fill 핵심 로직 구현)
 * mshr_table         : 동일 캐시라인 중복 미스를 병합하는 MSHR 구조체
 * baseline_cache     : read_only_cache·data_cache의 공통 기반 클래스
 * data_cache         : 함수 포인터 기반 쓰기 정책 선택을 지원하는 L1/L2 데이터 캐시
 * tex_cache          : fragment_fifo + ROB 구조의 텍스처 캐시 (순서 보장 필요)
 */

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include "gpu-misc.h"
#include "mem_fetch.h"

#include <iostream>
#include "addrdec.h"

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

/* [한국어] 캐시 블록(캐시라인 또는 섹터)의 상태를 나타내는 열거형.
 * 타이밍 시뮬레이션에서 cache_block_t::m_status 필드로 사용된다.
 * 상태 전이: INVALID → (allocate) → RESERVED → (fill) → VALID → (write) → MODIFIED
 * RESERVED: miss 후 MSHR에 등록되어 메모리 요청이 진행 중이지만 데이터는 아직 도착 안 함.
 * MODIFIED: L1 write-back 정책에서 캐시에 쓰여졌지만 하위 메모리에 아직 반영 안 됨(더티). */
enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };

/* [한국어] 캐시 접근 요청의 결과 상태를 나타내는 열거형.
 * tag_array::probe()/access()와 baseline_cache::access()의 반환값으로 사용된다.
 * 각 값의 의미:
 *   HIT              : 태그 히트, 데이터가 VALID 상태 — 즉시 데이터 공급 가능
 *   HIT_RESERVED     : 태그 히트이지만 RESERVED 상태 — 아직 fill 대기 중 (pending hit)
 *   MISS             : 태그 미스 — MSHR 등록 후 하위 메모리로 요청 전송
 *   RESERVATION_FAIL : MSHR 또는 miss_queue가 가득 차 요청 수용 불가 — 재시도 필요
 *   SECTOR_MISS      : 섹터 캐시에서 라인은 히트이지만 해당 32B 섹터가 없음
 *   MSHR_HIT         : 동일 캐시라인에 이미 MSHR entry가 존재 — 병합 처리
 *   NUM_CACHE_REQUEST_STATUS : 배열 크기 계산용 센티넬 */
enum cache_request_status {
  HIT = 0,
  HIT_RESERVED,
  MISS,
  RESERVATION_FAIL,
  SECTOR_MISS,
  MSHR_HIT,
  NUM_CACHE_REQUEST_STATUS
};

/* [한국어] RESERVATION_FAIL 발생 시 세부 실패 원인을 추적하는 열거형.
 * cache_stats::inc_fail_stats()에서 통계 집계에 사용된다.
 * 각 값의 의미:
 *   LINE_ALLOC_FAIL        : 캐시 내 모든 라인이 이미 RESERVED 상태 — 빈 슬롯 없음
 *   MISS_QUEUE_FULL        : miss_queue(ICNT/DRAM 방향) 포화 — 더 이상 요청 보낼 수 없음
 *   MSHR_ENRTY_FAIL        : MSHR 엔트리 수 한계 초과 — 새 블록 주소 등록 불가
 *   MSHR_MERGE_ENRTY_FAIL  : MSHR 엔트리 존재하지만 merge 한계(m_max_merged) 초과
 *   MSHR_RW_PENDING        : 같은 주소에 read-after-write 위험 — 쓰기 완료 대기 중 */
enum cache_reservation_fail_reason {
  LINE_ALLOC_FAIL = 0,  // all line are reserved
  MISS_QUEUE_FULL,      // MISS queue (i.e. interconnect or DRAM) is full
  MSHR_ENRTY_FAIL,
  MSHR_MERGE_ENRTY_FAIL,
  MSHR_RW_PENDING,
  NUM_CACHE_RESERVATION_FAIL_STATUS
};

/* [한국어] 캐시 접근 처리 중 하위 메모리로 발행된 요청 유형을 나타내는 열거형.
 * cache_event 구조체의 m_cache_event_type 필드로 사용되며,
 * was_write_sent()/was_read_sent()/was_writeallocate_sent() 헬퍼 함수로 검사한다.
 * 각 값의 의미:
 *   WRITE_BACK_REQUEST_SENT  : dirty 라인 축출 시 write-back 요청이 발행됨
 *   READ_REQUEST_SENT        : 캐시 미스로 하위 메모리로부터 데이터 fetch 요청 발행됨
 *   WRITE_REQUEST_SENT       : write-through/write-evict 정책에서 쓰기 요청 직접 발행됨
 *   WRITE_ALLOCATE_SENT      : write-allocate 정책에서 write-alloc 요청 발행됨 */
enum cache_event_type {
  WRITE_BACK_REQUEST_SENT,
  READ_REQUEST_SENT,
  WRITE_REQUEST_SENT,
  WRITE_ALLOCATE_SENT
};

/* [한국어] GPU 메모리 계층에서 캐시의 레벨을 나타내는 열거형.
 * baseline_cache 생성자에서 m_level 필드로 저장되며, 통계 집계 시
 * inc_aggregated_stats() 함수가 어느 캐시 레벨의 통계인지 구분할 때 사용된다.
 * L1_GPU_CACHE : SM 내부의 L1 데이터 캐시 (per-SM, 사이클 레이턴시 낮음)
 * L2_GPU_CACHE : 메모리 서브파티션의 L2 통합 캐시 (공유, 더 큰 용량)
 * OTHER_GPU_CACHE : 텍스처·상수 캐시 등 기타 특수 캐시 */
enum cache_gpu_level {
  L1_GPU_CACHE = 0,
  L2_GPU_CACHE,
  OTHER_GPU_CACHE,
  NUM_CACHE_GPU_LEVELS
};

/* [한국어] 캐시에서 축출(evict)된 블록의 정보를 담는 구조체.
 * tag_array::access()에서 write-back이 필요한 dirty 블록을 축출할 때 생성되며,
 * baseline_cache::send_read_request()가 이 정보를 상위 호출자로 전달한다.
 * 상위 호출자는 이 구조체를 참조하여 write-back mem_fetch를 생성하고
 * 하위 메모리(ICNT/L2)로 write-back 요청을 발행한다. */
struct evicted_block_info {
  new_addr_type m_block_addr;
  /* [한국어] 축출된 블록의 베이스 주소(캐시라인 정렬 주소).
   * 설정자: tag_array::access()에서 evicted.set_info()를 통해 설정.
   * 읽는 자: data_cache의 write-back 로직이 이 주소로 write-back mem_fetch를 생성.
   * 값 범위: 캐시라인 크기(m_line_sz)로 정렬된 물리 주소. 0이면 유효하지 않음.
   * 동기화: tag_array 접근은 단일 SM 컨텍스트에서 이루어지므로 별도 락 불필요. */
  unsigned m_modified_size;
  /* [한국어] 축출된 블록에서 실제 더티(수정된) 바이트 수.
   * 설정자: tag_array::access() → evicted.set_info()에서 cache_block_t::get_modified_size()로 설정.
   * 읽는 자: write-back mem_fetch 생성 시 데이터 크기 지정에 사용.
   * 값 범위: 0 ~ m_line_sz(line_cache_block) 또는 0 ~ SECTOR_SIZE*N(sector_cache_block).
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  mem_access_byte_mask_t m_byte_mask;
  /* [한국어] 축출된 블록에서 더티 바이트의 비트마스크 (128B 라인 기준 128비트).
   * 설정자: sector_cache_block::get_dirty_byte_mask()에서 채워짐.
   * 읽는 자: write-back 요청 시 실제로 쓰여진 바이트만 전송하기 위해 참조.
   * 값 범위: bitset<128>로 각 비트가 해당 바이트의 더티 여부를 나타냄.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  mem_access_sector_mask_t m_sector_mask;
  /* [한국어] 축출된 블록에서 더티 섹터의 비트마스크 (4비트, 각 비트 = 32B 섹터 하나).
   * 설정자: sector_cache_block::get_dirty_sector_mask()에서 채워짐.
   * 읽는 자: 섹터 단위 write-back 처리 시 어느 섹터만 전송할지 결정.
   * 값 범위: bitset<4>, 비트 i=1이면 i번째 섹터가 더티.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  evicted_block_info() {
    m_block_addr = 0;          /* [한국어] 축출 블록 주소 초기화 — 아직 축출 정보 없음 */
    m_modified_size = 0;       /* [한국어] 더티 크기 0으로 초기화 */
    m_byte_mask.reset();       /* [한국어] 바이트 마스크 전체 클리어 — 더티 바이트 없음 */
    m_sector_mask.reset();     /* [한국어] 섹터 마스크 전체 클리어 — 더티 섹터 없음 */
  }
  /*
   * [한국어]
   * set_info (2인자 오버로드) - 블록 주소와 더티 크기만 설정 (line_cache_block용)
   *
   * @block_addr: 축출된 캐시라인의 베이스 주소
   * @modified_size: 더티 바이트 수 (line_cache_block의 경우 라인 전체 크기)
   *
   * line_cache_block처럼 섹터 분할 없이 라인 전체를 하나의 단위로 관리할 때 사용.
   * byte_mask/sector_mask는 설정하지 않으므로 sector_cache_block에는 사용 불가.
   *
   * 호출 체인: tag_array::access() → evicted.set_info(block_addr, modified_size)
   */
  void set_info(new_addr_type block_addr, unsigned modified_size) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
  }
  /*
   * [한국어]
   * set_info (4인자 오버로드) - 블록 주소, 더티 크기, 바이트/섹터 마스크 모두 설정
   *
   * @block_addr: 축출된 캐시라인의 베이스 주소
   * @modified_size: 더티 바이트 수
   * @byte_mask: 더티 바이트 비트마스크 (128비트)
   * @sector_mask: 더티 섹터 비트마스크 (4비트)
   *
   * sector_cache_block처럼 섹터 단위로 더티 상태를 추적할 때 사용.
   * write-back 요청 시 실제 더티 섹터만 선택적으로 전송하기 위해 마스크 정보 보존.
   *
   * 호출 체인: tag_array::access() → evicted.set_info(block_addr, size, byte_mask, sector_mask)
   */
  void set_info(new_addr_type block_addr, unsigned modified_size,
                mem_access_byte_mask_t byte_mask,
                mem_access_sector_mask_t sector_mask) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
    m_byte_mask = byte_mask;
    m_sector_mask = sector_mask;
  }
};

/* [한국어] 캐시 접근 처리 중 발생한 부수 이벤트(write-back 발행, 읽기/쓰기 요청 발행 등)를
 * 기록하는 구조체. cache_t::access()는 events 리스트(std::list<cache_event>)를 인자로 받아
 * 처리 중 발생한 모든 이벤트를 append한다. 호출자(ldst_unit 등)는 이 리스트를 검사하여
 * ICNT로 실제 요청 패킷을 전송한다. was_write_sent()/was_read_sent() 등 헬퍼 함수로 조회. */
struct cache_event {
  enum cache_event_type m_cache_event_type;
  /* [한국어] 발생한 이벤트의 유형 (WRITE_BACK_REQUEST_SENT, READ_REQUEST_SENT 등).
   * 설정자: cache_event 생성자에서 초기화.
   * 읽는 자: was_write_sent(), was_read_sent(), was_writeallocate_sent() 헬퍼.
   * 값 범위: cache_event_type 열거형의 4가지 값.
   * 동기화: 단일 SM 컨텍스트에서 생성·소비되므로 락 불필요. */
  evicted_block_info m_evicted_block;  // if it was write_back event, fill the
                                       // the evicted block info
  /* [한국어] WRITE_BACK_REQUEST_SENT 이벤트 시 축출된 블록의 상세 정보.
   * 설정자: dirty 라인 축출 시 evicted_block_info를 포함한 cache_event 생성자가 설정.
   * 읽는 자: data_cache의 write-back 처리 로직이 이 정보로 write-back mem_fetch를 생성.
   * 값 범위: 이벤트 유형이 WRITE_BACK_REQUEST_SENT인 경우에만 유효.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  /*
   * [한국어]
   * cache_event (1인자 생성자) - 이벤트 유형만 설정 (축출 블록 정보 없음)
   *
   * @m_cache_event: 발생한 캐시 이벤트 유형
   *
   * READ_REQUEST_SENT, WRITE_REQUEST_SENT, WRITE_ALLOCATE_SENT 이벤트처럼
   * 축출 블록 정보가 필요 없는 경우에 사용.
   */
  cache_event(enum cache_event_type m_cache_event) {
    m_cache_event_type = m_cache_event; /* [한국어] 이벤트 유형 저장, 축출 정보는 기본값 */
  }

  /*
   * [한국어]
   * cache_event (2인자 생성자) - 이벤트 유형과 축출 블록 정보 모두 설정
   *
   * @cache_event: 발생한 캐시 이벤트 유형 (보통 WRITE_BACK_REQUEST_SENT)
   * @evicted_block: write-back 대상 블록의 주소, 더티 크기, 바이트/섹터 마스크
   *
   * dirty 라인 축출 시 write-back 요청이 필요할 때 사용.
   * 호출자가 m_evicted_block을 참조하여 write-back mem_fetch를 별도 생성한다.
   */
  cache_event(enum cache_event_type cache_event,
              evicted_block_info evicted_block) {
    m_cache_event_type = cache_event;     /* [한국어] 이벤트 유형 설정 */
    m_evicted_block = evicted_block;      /* [한국어] 축출 블록 정보 복사 저장 */
  }
};

/* [한국어]
 * cache_request_status_str - 캐시 요청 상태 코드를 사람이 읽을 수 있는 문자열로 변환
 *
 * @status: 변환할 cache_request_status 열거값
 * @return: 해당 상태의 영문 이름 문자열 (예: "HIT", "MISS", "RESERVATION_FAIL")
 *
 * 주로 디버그/통계 출력 시 상태 코드를 문자열로 로깅하기 위해 사용.
 * 구현은 gpu-cache.cc에 있으며 switch-case로 각 열거값을 매핑.
 *
 * 호출 체인: print_stats() / display_state() → cache_request_status_str()
 */
const char *cache_request_status_str(enum cache_request_status status);

/* [한국어] 캐시 블록(캐시라인 또는 섹터) 하나를 표현하는 추상 기반 클래스(인터페이스).
 * line_cache_block(라인 전체 단위)과 sector_cache_block(32B 섹터 단위)이 이 클래스를 상속.
 * tag_array는 이 타입의 포인터 배열(cache_block_t** m_lines)로 블록을 관리하여
 * 런타임에 NORMAL/SECTOR 캐시 유형을 투명하게 처리한다.
 * allocate()/fill() 순서로 블록의 라이프사이클이 진행:
 *   cache miss → allocate() [INVALID→RESERVED] → 메모리 요청 → fill() [RESERVED→VALID/MODIFIED] */
struct cache_block_t {
  /*
   * [한국어]
   * cache_block_t 기본 생성자 - 태그와 블록 주소를 0으로 초기화
   *
   * tag_array 생성자에서 m_lines 배열을 초기화할 때 호출.
   * 파생 클래스의 생성자(line_cache_block(), sector_cache_block())에서
   * 각자의 추가 필드도 함께 초기화한다.
   */
  cache_block_t() {
    m_tag = 0;           /* [한국어] 태그 값 0으로 초기화 — 아직 어떤 주소도 매핑 안 됨 */
    m_block_addr = 0;    /* [한국어] 블록 베이스 주소 0으로 초기화 */
  }

  /* [한국어] allocate - 캐시 미스 후 이 블록에 새 주소를 할당 (INVALID→RESERVED 전이)
   * @tag: 이 블록에 매핑할 메모리 태그 (tag_array::tag() 계산 결과)
   * @block_addr: 캐시라인 정렬 주소
   * @time: 현재 시뮬레이터 사이클 (LRU 타임스탬프용)
   * @sector_mask: sector_cache_block에서 어느 섹터를 먼저 할당할지 지정 (line 무시)
   * 호출 체인: tag_array::access() → cache_block_t::allocate() */
  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned time,
                        mem_access_sector_mask_t sector_mask) = 0;

  /* [한국어] fill - 메모리에서 데이터가 도착했을 때 블록 상태를 갱신 (RESERVED→VALID/MODIFIED)
   * @time: fill이 완료된 사이클
   * @sector_mask: 도착한 데이터가 속한 섹터 (sector_cache_block에서 사용)
   * @byte_mask: fill과 함께 설정할 더티 바이트 마스크
   * 호출 체인: tag_array::fill() → cache_block_t::fill() */
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) = 0;

  /* [한국어] is_invalid_line - 블록(또는 모든 섹터)이 INVALID 상태인지 확인
   * tag_array의 교체 대상 탐색(LRU 교체 등)에서 우선 교체 후보로 선택할 때 사용. */
  virtual bool is_invalid_line() = 0;
  /* [한국어] is_valid_line - 블록에 유효한 데이터가 존재하는지 확인 */
  virtual bool is_valid_line() = 0;
  /* [한국어] is_reserved_line - 미스 처리 중(메모리 요청 진행 중)인지 확인
   * RESERVED 블록은 데이터 미도착이므로 교체 대상에서 제외한다. */
  virtual bool is_reserved_line() = 0;
  /* [한국어] is_modified_line - write-back 필요한 더티 상태인지 확인
   * 이 블록을 교체할 때 write-back 요청 발행이 필요한지 여부를 결정. */
  virtual bool is_modified_line() = 0;

  /* [한국어] get_status - 특정 섹터 마스크에 대응하는 블록 상태 반환
   * sector_cache_block에서는 sector_mask로 섹터를 선택; line_cache_block은 무시. */
  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] set_status - 특정 섹터의 상태를 강제 설정
   * tag_array에서 축출 처리 시 INVALID로 강제 전환하거나, write 시 MODIFIED로 변경. */
  virtual void set_status(enum cache_block_state m_status,
                          mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] set_byte_mask (mem_fetch 오버로드) - mem_fetch의 접근 바이트 마스크를 더티 마스크에 OR 합산 */
  virtual void set_byte_mask(mem_fetch *mf) = 0;
  /* [한국어] set_byte_mask (byte_mask 오버로드) - 직접 바이트 마스크를 더티 마스크에 OR 합산 */
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) = 0;
  /* [한국어] get_dirty_byte_mask - 현재 더티 바이트 비트마스크 반환 (write-back 크기 계산용) */
  virtual mem_access_byte_mask_t get_dirty_byte_mask() = 0;
  /* [한국어] get_dirty_sector_mask - 더티 섹터 비트마스크 반환 (섹터 단위 write-back용) */
  virtual mem_access_sector_mask_t get_dirty_sector_mask() = 0;
  /* [한국어] get_last_access_time - LRU 교체 정책에서 가장 오래된 라인 식별용 마지막 접근 시각 반환 */
  virtual unsigned long long get_last_access_time() = 0;
  /* [한국어] set_last_access_time - 접근 시각 갱신 (LRU 카운터 업데이트) */
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] get_alloc_time - 이 블록이 allocate된 사이클 반환 (FIFO 교체 정책용) */
  virtual unsigned long long get_alloc_time() = 0;
  /* [한국어] set_ignore_on_fill - fill 시 상태 전이를 무시할지 플래그 설정
   * streaming 캐시 등 특수 정책에서 fill 도착 시 RESERVED→VALID 전이를 건너뛰기 위해 사용. */
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] set_modified_on_fill - fill 완료 시 VALID 대신 MODIFIED로 전이할지 플래그 설정
   * write-allocate 정책에서 쓰기 데이터를 fill과 함께 더티로 표시할 때 사용. */
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] set_readable_on_fill - fill 완료 시 readable 플래그를 true로 설정할지 지정
   * LAZY_FETCH_ON_READ 정책에서 fill 전까지 readable=false로 두다가 fill 완료 후 읽기 허용. */
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] set_byte_mask_on_fill - fill 완료 시 byte_mask를 적용할지 플래그 설정 */
  virtual void set_byte_mask_on_fill(bool m_modified) = 0;
  /* [한국어] get_modified_size - write-back에 필요한 더티 데이터 크기(바이트) 반환
   * line_cache_block: 항상 라인 전체 크기(SECTOR_CHUNCK_SIZE * SECTOR_SIZE = 128B)
   * sector_cache_block: 더티 섹터 개수 * SECTOR_SIZE */
  virtual unsigned get_modified_size() = 0;
  /* [한국어] set_m_readable - 특정 섹터의 readable 플래그 직접 설정 */
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] is_readable - 특정 섹터가 읽기 가능한 상태인지 확인
   * LAZY_FETCH_ON_READ 정책에서 fill 완료 전 읽기 접근 차단에 사용. */
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) = 0;
  /* [한국어] print_status - 디버그 출력: 블록 주소와 현재 상태를 stdout에 출력 */
  virtual void print_status() = 0;
  virtual ~cache_block_t() {}

  new_addr_type m_tag;
  /* [한국어] 이 캐시 블록에 매핑된 메모리 태그.
   * 설정자: allocate() 호출 시 cache_config::tag(addr)로 계산된 값이 저장됨.
   * 읽는 자: tag_array::probe()에서 태그 비교(hit/miss 판별)에 사용.
   * 값 범위: 캐시라인 정렬 주소(block_addr와 동일, 풀 태그+인덱스 포함).
   * 동기화: 단일 SM 컨텍스트에서만 접근, 락 불필요. */
  new_addr_type m_block_addr;
  /* [한국어] 이 캐시 블록의 캐시라인 베이스 주소 (m_tag와 동일값).
   * 설정자: allocate() 호출 시 block_addr 인자로 설정.
   * 읽는 자: evicted_block_info 생성, print_status(), fill() 후 통계 기록 등.
   * 값 범위: m_line_sz로 정렬된 주소. INVALID 상태일 때는 이전 값이 잔류할 수 있음.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
};

/* [한국어] 캐시라인(128B) 전체를 단일 상태로 관리하는 캐시 블록 구현체.
 * NORMAL 캐시(cache_type == NORMAL)에서 사용된다. 섹터 분할 없이 캐시라인 전체가
 * 하나의 상태 머신(INVALID/RESERVED/VALID/MODIFIED)으로 관리된다.
 * L1/L2 데이터 캐시의 non-sector 모드 또는 read-only 캐시에서 인스턴스화.
 * 라이프사이클: 생성 시 INVALID → allocate()로 RESERVED → fill()로 VALID/MODIFIED.
 * sector_cache_block과 달리 is_invalid_line() 등이 단일 m_status 필드만 검사한다. */
struct line_cache_block : public cache_block_t {
  /*
   * [한국어]
   * line_cache_block 기본 생성자 - 모든 타임스탬프와 상태를 초기화
   *
   * tag_array 생성자에서 m_lines 배열을 초기화할 때 new로 생성되어 호출된다.
   * INVALID 상태, 모든 타임스탬프 0, readable=true(읽기 가능)로 초기화.
   *
   * 호출 체인: tag_array::tag_array() → new line_cache_block() → 이 생성자
   */
  line_cache_block() {
    m_alloc_time = 0;                /* [한국어] 할당 시각 0 — 아직 allocate된 적 없음 */
    m_fill_time = 0;                 /* [한국어] fill 완료 시각 0 — 아직 데이터 도착 안 함 */
    m_last_access_time = 0;          /* [한국어] 마지막 접근 시각 0 — LRU 정책에서 최우선 교체 후보 */
    m_status = INVALID;              /* [한국어] 상태 INVALID — 유효한 데이터 없음 */
    m_ignore_on_fill_status = false; /* [한국어] fill 시 상태 전이 무시 플래그 false(기본 동작) */
    m_set_modified_on_fill = false;  /* [한국어] fill 완료 시 MODIFIED로 전이 플래그 false(기본: VALID) */
    m_set_readable_on_fill = false;  /* [한국어] fill 완료 시 readable 설정 플래그 false */
    m_readable = true;               /* [한국어] 기본적으로 읽기 가능 상태로 초기화 */
  }
  /*
   * [한국어]
   * allocate - 캐시 미스 후 이 블록에 새 주소를 할당하고 RESERVED 상태로 전이
   *
   * @tag: cache_config::tag(addr)로 계산된 태그 값 (블록 주소와 동일)
   * @block_addr: 캐시라인 정렬 주소
   * @time: 현재 시뮬레이터 사이클 (LRU 타임스탬프로 저장)
   * @sector_mask: line_cache_block에서는 사용하지 않음 (sector_cache_block용 인터페이스)
   *
   * 캐시 미스 시 tag_array::access()가 교체 대상 라인을 선택한 후 이 함수를 호출.
   * 이 블록은 RESERVED 상태가 되어 중복 미스 요청 시 MSHR 병합이 가능해진다.
   * fill() 호출 전까지는 데이터가 없으므로 HIT_RESERVED 상태를 반환한다.
   *
   * 호출 체인: tag_array::access() → 교체 후보 블록.allocate()
   */
  void allocate(new_addr_type tag, new_addr_type block_addr, unsigned time,
                mem_access_sector_mask_t sector_mask) {
    m_tag = tag;                         /* [한국어] 새 태그 저장 — 이전 블록 태그 덮어씀 */
    m_block_addr = block_addr;           /* [한국어] 캐시라인 베이스 주소 저장 */
    m_alloc_time = time;                 /* [한국어] 할당 시각 기록 — FIFO 교체 정책의 순서 기준 */
    m_last_access_time = time;           /* [한국어] 마지막 접근 시각 = 할당 시각으로 초기화 */
    m_fill_time = 0;                     /* [한국어] fill 시각 초기화 — 아직 데이터 미도착 */
    m_status = RESERVED;                 /* [한국어] RESERVED 전이 — 메모리 요청 진행 중 */
    m_ignore_on_fill_status = false;     /* [한국어] fill 시 상태 전이 정상 수행 */
    m_set_modified_on_fill = false;      /* [한국어] fill 완료 시 VALID로 전이(기본값) */
    m_set_readable_on_fill = false;      /* [한국어] fill 완료 시 readable 자동 설정 안 함 */
    m_set_byte_mask_on_fill = false;     /* [한국어] fill 완료 시 byte_mask 적용 안 함 */
  }
  /*
   * [한국어]
   * fill - 하위 메모리에서 데이터가 도착했을 때 블록 상태를 VALID 또는 MODIFIED로 전이
   *
   * @time: fill이 완료된 사이클 (m_fill_time으로 기록)
   * @sector_mask: line_cache_block에서는 무시 (sector_cache_block 인터페이스 호환용)
   * @byte_mask: fill과 함께 적용할 더티 바이트 마스크
   *
   * RESERVED 상태의 블록에 데이터가 도착할 때 호출된다.
   * m_set_modified_on_fill 플래그에 따라 VALID(읽기 로드) 또는 MODIFIED(write-alloc)로 전이.
   * m_ignore_on_fill_status가 true면 현재 상태와 무관하게 전이를 허용한다.
   * 이 호출 후 MSHR에서 대기 중인 모든 pending hit 요청이 처리될 수 있다.
   *
   * 호출 체인: tag_array::fill() → 해당 블록.fill() → 이 함수
   */
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) {
    // if(!m_ignore_on_fill_status)
    //	assert( m_status == RESERVED );

    m_status = m_set_modified_on_fill ? MODIFIED : VALID; /* [한국어] write-alloc이면 MODIFIED, 일반 로드면 VALID로 전이 */

    if (m_set_readable_on_fill) m_readable = true;        /* [한국어] LAZY_FETCH_ON_READ 정책: fill 완료 시 readable 활성화 */
    if (m_set_byte_mask_on_fill) set_byte_mask(byte_mask);/* [한국어] fill 시 byte_mask 적용 플래그가 설정된 경우 더티 마스크 업데이트 */

    m_fill_time = time;  /* [한국어] fill 완료 시각 기록 — 디버그/통계용 */
  }
  /* [한국어] is_invalid_line - m_status가 INVALID인지 단순 비교 (교체 후보 탐색에서 우선순위 최상) */
  /* [한국어] is_invalid_line - m_status가 INVALID인지 단순 비교 (교체 후보 탐색에서 우선순위 최상) */
  virtual bool is_invalid_line() { return m_status == INVALID; }
  /* [한국어] is_valid_line - m_status가 VALID인지 확인 (RESERVED, MODIFIED는 false) */
  virtual bool is_valid_line() { return m_status == VALID; }
  /* [한국어] is_reserved_line - RESERVED 상태인지 확인 — 이 라인은 교체 불가, 미스 진행 중 */
  virtual bool is_reserved_line() { return m_status == RESERVED; }
  /* [한국어] is_modified_line - MODIFIED(dirty) 상태인지 확인 — 교체 시 write-back 필요 */
  virtual bool is_modified_line() { return m_status == MODIFIED; }

  /* [한국어] get_status - sector_mask 인자는 무시하고 단일 m_status 반환
   * line_cache_block은 섹터 개념이 없으므로 sector_mask는 사용하지 않는다. */
  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    return m_status; /* [한국어] 라인 전체 상태 반환 */
  }
  /* [한국어] set_status - sector_mask 무시, 라인 전체 상태를 직접 설정
   * 주로 flush() 처리 시 INVALID로 강제 전환하거나, write 히트 시 MODIFIED로 변경. */
  virtual void set_status(enum cache_block_state status,
                          mem_access_sector_mask_t sector_mask) {
    m_status = status; /* [한국어] 라인 전체 상태 변경 */
  }
  /* [한국어] set_byte_mask (mem_fetch 오버로드) - mem_fetch가 접근한 바이트들을 더티 마스크에 추가
   * write-back 정책에서 쓰기 히트 시 어느 바이트가 수정됐는지 기록. */
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask(); /* [한국어] OR로 더티 바이트 누적 */
  }
  /* [한국어] set_byte_mask (byte_mask 오버로드) - 직접 바이트 마스크를 더티 마스크에 OR 합산 */
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask; /* [한국어] OR로 더티 바이트 누적 */
  }
  /* [한국어] get_dirty_byte_mask - 현재까지 축적된 더티 바이트 비트마스크 반환
   * write-back 요청 발행 시 실제로 수정된 바이트 범위를 결정하는 데 사용. */
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask; /* [한국어] 더티 바이트 마스크 반환 */
  }
  /* [한국어] get_dirty_sector_mask - MODIFIED 상태이면 모든 섹터(4비트 all-set) 반환, VALID이면 0
   * line_cache_block은 라인 단위 관리이므로 MODIFIED이면 전체 섹터가 더티. */
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;           /* [한국어] 반환할 섹터 마스크, 기본값 0 */
    if (m_status == MODIFIED) sector_mask.set();    /* [한국어] MODIFIED이면 모든 4개 섹터 비트 set */
    return sector_mask;
  }
  /* [한국어] get_last_access_time - LRU 교체 정책에서 가장 오래전에 접근된 라인 선택용 */
  virtual unsigned long long get_last_access_time() {
    return m_last_access_time; /* [한국어] 마지막 접근 사이클 반환 */
  }
  /* [한국어] set_last_access_time - 접근 시 LRU 카운터 갱신
   * sector_mask는 line_cache_block에서 무시 (라인 단위 관리). */
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    m_last_access_time = time; /* [한국어] 현재 사이클로 마지막 접근 시각 업데이트 */
  }
  /* [한국어] get_alloc_time - FIFO 교체 정책에서 가장 오래된 라인 선택용 할당 시각 반환 */
  virtual unsigned long long get_alloc_time() { return m_alloc_time; }
  /* [한국어] set_ignore_on_fill - fill 시 RESERVED 상태 검사 우회 플래그 설정
   * streaming 캐시 등 특수 정책에서 이미 VALID인 블록에 re-fill할 때 assert를 방지. */
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    m_ignore_on_fill_status = m_ignore; /* [한국어] fill 시 상태 무시 여부 설정 */
  }
  /* [한국어] set_modified_on_fill - fill 완료 시 MODIFIED로 전이할지 플래그 설정
   * write-allocate 정책: 쓰기 미스 시 fetch 후 즉시 MODIFIED로 표시 필요. */
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_modified_on_fill = m_modified; /* [한국어] fill 완료 시 MODIFIED 전이 여부 설정 */
  }
  /* [한국어] set_readable_on_fill - fill 완료 시 m_readable을 true로 설정할지 플래그 지정
   * LAZY_FETCH_ON_READ 정책: fill 전에는 readable=false로 두어 읽기를 차단. */
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_readable_on_fill = readable; /* [한국어] fill 완료 시 readable 활성화 여부 설정 */
  }
  /* [한국어] set_byte_mask_on_fill - fill 완료 시 byte_mask를 dirty mask에 적용할지 플래그 설정 */
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified; /* [한국어] fill 시 byte_mask 적용 여부 플래그 설정 */
  }
  /* [한국어] get_modified_size - write-back 크기: line_cache_block은 항상 캐시라인 전체 크기
   * SECTOR_CHUNCK_SIZE(4) * SECTOR_SIZE(32B) = 128B = 전체 캐시라인 크기를 반환. */
  virtual unsigned get_modified_size() {
    return SECTOR_CHUNCK_SIZE * SECTOR_SIZE;  // i.e. cache line size
    /* [한국어] 라인 전체(128B)가 write-back 대상 — 섹터 단위 관리 없음 */
  }
  /* [한국어] set_m_readable - sector_mask 무시, 라인 전체의 readable 플래그 직접 설정 */
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    m_readable = readable; /* [한국어] readable 플래그 직접 변경 */
  }
  /* [한국어] is_readable - 현재 라인이 읽기 가능한지 반환
   * LAZY_FETCH_ON_READ 정책에서 fill 완료 전 읽기 접근을 차단하는 데 사용. */
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    return m_readable; /* [한국어] readable 플래그 반환 */
  }
  /* [한국어] print_status - 디버그용: 블록 주소와 상태 코드를 stdout에 출력 */
  virtual void print_status() {
    printf("m_block_addr is %llu, status = %u\n", m_block_addr, m_status);
    /* [한국어] 블록 주소(64비트 unsigned)와 cache_block_state 숫자값 출력 */
  }

 private:
  unsigned long long m_alloc_time;
  /* [한국어] 이 블록이 allocate()된 사이클.
   * 설정자: allocate() 호출 시 현재 사이클로 설정.
   * 읽는 자: FIFO 교체 정책에서 get_alloc_time()으로 가장 오래된 블록 선택.
   * 값 범위: 0(미할당) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned long long m_last_access_time;
  /* [한국어] 마지막으로 이 블록에 접근(읽기 또는 쓰기)된 사이클.
   * 설정자: set_last_access_time() — tag_array::access()에서 히트 시 갱신.
   * 읽는 자: LRU 교체 정책에서 get_last_access_time()으로 교체 후보 선택.
   * 값 범위: 0(접근 없음) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned long long m_fill_time;
  /* [한국어] 마지막으로 fill()이 완료된 사이클.
   * 설정자: fill() 호출 시 현재 사이클로 설정.
   * 읽는 자: 현재 직접 사용되지 않지만 디버그/통계 목적으로 보존.
   * 값 범위: 0(fill 없음) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  cache_block_state m_status;
  /* [한국어] 이 캐시 블록 전체의 현재 상태 (INVALID/RESERVED/VALID/MODIFIED).
   * 설정자: allocate()로 RESERVED, fill()로 VALID/MODIFIED, set_status()로 직접 설정.
   * 읽는 자: is_invalid_line(), is_valid_line(), is_reserved_line(), is_modified_line(), get_status().
   * 값 범위: cache_block_state enum의 4가지 값.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_ignore_on_fill_status;
  /* [한국어] fill() 시 RESERVED 상태 검사를 무시할지 여부 플래그.
   * 설정자: set_ignore_on_fill() — streaming 캐시 등 특수 정책에서 활성화.
   * 읽는 자: fill() 내부에서 assert 우회 여부 결정 (현재 코드에서 주석 처리).
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_modified_on_fill;
  /* [한국어] fill() 완료 시 VALID 대신 MODIFIED로 전이할지 플래그.
   * 설정자: set_modified_on_fill() — write-allocate 미스 처리 시 활성화.
   * 읽는 자: fill() 내부 상태 전이 분기.
   * 값 범위: true이면 fill 후 MODIFIED, false이면 VALID.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_readable_on_fill;
  /* [한국어] fill() 완료 시 m_readable을 true로 설정할지 플래그.
   * 설정자: set_readable_on_fill() — LAZY_FETCH_ON_READ 정책에서 활성화.
   * 읽는 자: fill() 내부 readable 갱신 분기.
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_byte_mask_on_fill;
  /* [한국어] fill() 완료 시 byte_mask를 dirty mask에 적용할지 플래그.
   * 설정자: set_byte_mask_on_fill() — 특정 쓰기 정책에서 fill 시 byte 단위 더티 기록 필요.
   * 읽는 자: fill() 내부 byte_mask 적용 분기.
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_readable;
  /* [한국어] 이 블록이 현재 읽기 가능한지 여부.
   * 설정자: 생성자에서 true, set_m_readable(), set_readable_on_fill() 연계로 변경.
   * 읽는 자: is_readable() — LAZY_FETCH_ON_READ 정책에서 fill 전 읽기 접근 차단.
   * 값 범위: true(읽기 가능) / false(fill 대기 중, 읽기 차단).
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  mem_access_byte_mask_t m_dirty_byte_mask;
  /* [한국어] 이 블록에서 수정된(더티) 바이트의 비트마스크 (128비트, 캐시라인 128B 대응).
   * 설정자: set_byte_mask()로 쓰기 접근마다 OR 누적.
   * 읽는 자: get_dirty_byte_mask() — write-back 요청 생성 시 실제 수정 바이트 범위 결정.
   * 값 범위: bitset<128>, 각 비트가 해당 바이트 오프셋의 더티 여부.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
};

/* [한국어] 캐시라인(128B)을 4개의 32B 섹터로 분할 관리하는 캐시 블록 구현체.
 * SECTOR 캐시(cache_type == SECTOR)에서 사용된다. 각 섹터는 독립적인 상태 머신
 * (INVALID/RESERVED/VALID/MODIFIED)을 가지므로 필요한 섹터만 선택적으로 fetch/write할 수 있다.
 * L2 캐시 및 섹터 모드 L1에서 인스턴스화. SECTOR_CHUNCK_SIZE=4, SECTOR_SIZE=32B로 하드코딩.
 * 라인 레벨과 섹터 레벨 두 가지 타임스탬프를 동시에 관리한다:
 *   - 라인 레벨: m_line_alloc_time, m_line_last_access_time (LRU/FIFO 교체 정책용)
 *   - 섹터 레벨: m_sector_alloc_time[i], m_last_sector_access_time[i] (섹터별 통계용) */
struct sector_cache_block : public cache_block_t {
  /*
   * [한국어]
   * sector_cache_block 기본 생성자 - init()을 호출하여 모든 필드 초기화
   *
   * tag_array 생성자에서 new로 생성 시 호출. 4개 섹터 모두 INVALID로 초기화.
   *
   * 호출 체인: tag_array::tag_array() → new sector_cache_block() → init()
   */
  sector_cache_block() { init(); } /* [한국어] init() 위임 — 모든 필드를 초기화 */

  /*
   * [한국어]
   * init - 모든 섹터와 라인 레벨 필드를 초기 상태(INVALID, 타임스탬프 0)로 리셋
   *
   * 생성자에서 초기화 목적으로 호출되거나, allocate_line()에서 기존 블록을 재사용할 때
   * 이전 상태를 완전히 초기화하기 위해 호출된다.
   * 4개 섹터 모두를 루프로 초기화하여 하드코딩된 SECTOR_CHUNCK_SIZE에 의존.
   *
   * 호출 체인: sector_cache_block() → init() / allocate_line() → init()
   */
  void init() {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 섹터를 순서대로 초기화 */
      m_sector_alloc_time[i] = 0;          /* [한국어] 섹터 할당 시각 0으로 초기화 */
      m_sector_fill_time[i] = 0;           /* [한국어] 섹터 fill 시각 0으로 초기화 */
      m_last_sector_access_time[i] = 0;    /* [한국어] 섹터 마지막 접근 시각 0으로 초기화 */
      m_status[i] = INVALID;               /* [한국어] 섹터 상태 INVALID — 유효 데이터 없음 */
      m_ignore_on_fill_status[i] = false;  /* [한국어] fill 시 상태 무시 플래그 false */
      m_set_modified_on_fill[i] = false;   /* [한국어] fill 완료 시 MODIFIED 전이 플래그 false */
      m_set_readable_on_fill[i] = false;   /* [한국어] fill 완료 시 readable 설정 플래그 false */
      m_readable[i] = true;                /* [한국어] 기본적으로 읽기 가능 상태 */
    }
    m_line_alloc_time = 0;          /* [한국어] 라인 레벨 할당 시각 초기화 */
    m_line_last_access_time = 0;    /* [한국어] 라인 레벨 마지막 접근 시각 초기화 */
    m_line_fill_time = 0;           /* [한국어] 라인 레벨 fill 완료 시각 초기화 */
    m_dirty_byte_mask.reset();      /* [한국어] 더티 바이트 마스크 전체 클리어 */
  }

  /*
   * [한국어]
   * allocate (virtual 오버라이드) - cache_block_t 인터페이스 → allocate_line()으로 위임
   *
   * @tag: 새로 매핑할 태그 (블록 주소와 동일)
   * @block_addr: 캐시라인 정렬 주소
   * @time: 현재 사이클
   * @sector_mask: 첫 번째로 할당할 섹터 지정
   *
   * tag_array가 교체 대상 블록을 선택하면 이 함수를 호출. 실제 로직은 allocate_line()에 있음.
   *
   * 호출 체인: tag_array::access() → cache_block_t::allocate() → allocate_line()
   */
  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned time, mem_access_sector_mask_t sector_mask) {
    allocate_line(tag, block_addr, time, sector_mask); /* [한국어] 라인+섹터 초기화를 allocate_line()에 위임 */
  }

  /*
   * [한국어]
   * allocate_line - 완전히 새로운 라인을 할당하고 첫 번째 섹터를 RESERVED로 설정
   *
   * @tag: 새 태그 값
   * @block_addr: 캐시라인 베이스 주소
   * @time: 현재 사이클
   * @sector_mask: 최초 할당할 섹터 (1비트만 set 되어야 함)
   *
   * 이전 블록 내용을 init()으로 완전히 초기화 후 새 주소를 할당한다.
   * 라인 레벨 타임스탬프(m_line_alloc_time)는 첫 섹터 할당 시에만 설정되어
   * LRU/FIFO 교체 정책의 기준이 된다. 추가 섹터는 allocate_sector()로 처리.
   *
   * 호출 체인: allocate() → allocate_line()
   */
  void allocate_line(new_addr_type tag, new_addr_type block_addr, unsigned time,
                     mem_access_sector_mask_t sector_mask) {
    // allocate a new line
    // assert(m_block_addr != 0 && m_block_addr != block_addr);
    init();                              /* [한국어] 이전 블록 상태 완전 초기화 — 재사용 준비 */
    m_tag = tag;                         /* [한국어] 새 태그 설정 */
    m_block_addr = block_addr;           /* [한국어] 새 블록 베이스 주소 설정 */

    unsigned sidx = get_sector_index(sector_mask); /* [한국어] sector_mask에서 섹터 인덱스(0~3) 추출 */

    // set sector stats
    m_sector_alloc_time[sidx] = time;    /* [한국어] 이 섹터의 할당 시각 기록 */
    m_last_sector_access_time[sidx] = time; /* [한국어] 초기 접근 시각 = 할당 시각 */
    m_sector_fill_time[sidx] = 0;        /* [한국어] fill 시각 초기화 — 아직 데이터 미도착 */
    m_status[sidx] = RESERVED;           /* [한국어] 이 섹터 RESERVED 전이 — 메모리 요청 진행 중 */
    m_ignore_on_fill_status[sidx] = false; /* [한국어] fill 시 상태 전이 정상 수행 */
    m_set_modified_on_fill[sidx] = false;  /* [한국어] fill 완료 시 VALID로 전이(기본값) */
    m_set_readable_on_fill[sidx] = false;  /* [한국어] fill 완료 시 readable 자동 설정 안 함 */
    m_set_byte_mask_on_fill = false;       /* [한국어] fill 시 byte_mask 적용 안 함 */

    // set line stats
    m_line_alloc_time = time;  // only set this for the first allocated sector
    /* [한국어] 라인의 첫 섹터 할당 시 라인 레벨 할당 시각 설정 — FIFO 교체 기준 */
    m_line_last_access_time = time; /* [한국어] 라인 마지막 접근 시각 초기화 */
    m_line_fill_time = 0;           /* [한국어] 라인 fill 시각 초기화 */
  }

  /*
   * [한국어]
   * allocate_sector - 이미 유효한 라인의 추가 섹터를 RESERVED로 할당 (SECTOR_MISS 처리)
   *
   * @time: 현재 사이클
   * @sector_mask: 새로 할당할 섹터 (1비트만 set 되어야 함)
   *
   * SECTOR_MISS 상태: 라인은 히트이지만 요청한 32B 섹터가 INVALID인 경우 호출된다.
   * 라인 전체를 재할당(allocate_line)하지 않고 해당 섹터만 RESERVED로 전이.
   * FETCH_ON_WRITE 정책에서 MODIFIED 섹터를 다시 fetch할 때는 m_set_modified_on_fill=true로
   * 설정하여 fill 완료 후에도 MODIFIED 상태를 유지한다.
   *
   * 호출 체인: tag_array::access() → (SECTOR_MISS 경로) → sector_cache_block::allocate_sector()
   */
  void allocate_sector(unsigned time, mem_access_sector_mask_t sector_mask) {
    // allocate invalid sector of this allocated valid line
    assert(is_valid_line()); /* [한국어] 이미 유효한 라인에서만 추가 섹터 할당 가능 */
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 할당할 섹터 인덱스 추출 */

    // set sector stats
    m_sector_alloc_time[sidx] = time;    /* [한국어] 이 섹터의 할당 시각 기록 */
    m_last_sector_access_time[sidx] = time; /* [한국어] 초기 접근 시각 = 할당 시각 */
    m_sector_fill_time[sidx] = 0;        /* [한국어] fill 시각 초기화 */
    if (m_status[sidx] == MODIFIED)  // this should be the case only for
                                     // fetch-on-write policy //TO DO
      /* [한국어] FETCH_ON_WRITE: MODIFIED 섹터를 다시 fetch 시 fill 후에도 MODIFIED 유지 */
      m_set_modified_on_fill[sidx] = true;
    else
      /* [한국어] 일반 SECTOR_MISS: fill 완료 시 VALID로 전이 */
      m_set_modified_on_fill[sidx] = false;

    m_set_readable_on_fill[sidx] = false; /* [한국어] fill 완료 시 readable 자동 설정 안 함 */

    m_status[sidx] = RESERVED;           /* [한국어] 해당 섹터 RESERVED 전이 */
    m_ignore_on_fill_status[sidx] = false; /* [한국어] fill 시 상태 전이 정상 수행 */
    // m_set_modified_on_fill[sidx] = false;
    m_readable[sidx] = true;             /* [한국어] 섹터 readable 초기화 */

    // set line stats
    m_line_last_access_time = time; /* [한국어] 라인 레벨 마지막 접근 시각 갱신 */
    m_line_fill_time = 0;           /* [한국어] 라인 fill 시각 초기화 */
  }

  /*
   * [한국어]
   * fill (sector_cache_block 오버라이드) - 특정 섹터에 메모리 응답 데이터 도착 처리
   *
   * @time: fill 완료 사이클
   * @sector_mask: 도착한 데이터의 섹터 (1비트만 set)
   * @byte_mask: fill과 함께 적용할 더티 바이트 마스크
   *
   * sector_mask로 섹터 인덱스를 계산하여 해당 섹터만 RESERVED → VALID/MODIFIED로 전이.
   * 다른 섹터들의 상태는 변경하지 않는다 (섹터 독립 관리의 핵심).
   * m_set_readable_on_fill[sidx] 플래그는 적용 후 false로 리셋되어 재사용 방지.
   *
   * 호출 체인: tag_array::fill() → sector_cache_block::fill()
   */
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] sector_mask에서 섹터 인덱스(0~3) 추출 */

    //	if(!m_ignore_on_fill_status[sidx])
    //	         assert( m_status[sidx] == RESERVED );
    m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;
    /* [한국어] write-alloc/FETCH_ON_WRITE이면 MODIFIED, 일반 로드이면 VALID로 섹터 상태 전이 */

    if (m_set_readable_on_fill[sidx]) {
      m_readable[sidx] = true;              /* [한국어] LAZY_FETCH_ON_READ: fill 완료 후 읽기 허용 */
      m_set_readable_on_fill[sidx] = false; /* [한국어] 플래그 리셋 — 다음 allocate까지 재활성화 안 함 */
    }
    if (m_set_byte_mask_on_fill) set_byte_mask(byte_mask); /* [한국어] 필요시 더티 바이트 마스크 업데이트 */

    m_sector_fill_time[sidx] = time; /* [한국어] 이 섹터의 fill 완료 시각 기록 */
    m_line_fill_time = time;         /* [한국어] 라인 레벨 fill 시각도 갱신 */
  }

  /*
   * [한국어]
   * is_invalid_line - 모든 4개 섹터가 INVALID 상태인지 확인 (라인 전체 미사용 여부)
   *
   * @return: 모든 섹터가 INVALID이면 true — 이 블록은 교체 1순위 후보
   *
   * 하나라도 INVALID가 아닌 섹터가 있으면 false 반환 (라인이 부분적으로 사용 중).
   * tag_array의 교체 후보 탐색에서 INVALID 라인을 우선 선택한다.
   */
  virtual bool is_invalid_line() {
    // all the sectors should be invalid
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 섹터 모두 검사 */
      if (m_status[i] != INVALID) return false; /* [한국어] INVALID 아닌 섹터 발견 시 즉시 false 반환 */
    }
    return true; /* [한국어] 모든 섹터 INVALID — 라인 전체가 비어있음 */
  }
  /* [한국어] is_valid_line - 하나라도 INVALID가 아닌 섹터가 있으면 true (라인이 어느 섹터든 사용 중)
   * allocate_sector() 호출 전 라인이 유효한지 확인하는 assert에서 사용. */
  virtual bool is_valid_line() { return !(is_invalid_line()); }
  /*
   * [한국어]
   * is_reserved_line - 하나라도 RESERVED 섹터가 있으면 true (라인이 메모리 요청 대기 중)
   *
   * @return: RESERVED 섹터가 하나라도 있으면 true
   *
   * RESERVED 섹터가 있는 라인은 교체 대상에서 제외된다.
   * 여러 섹터가 동시에 다른 메모리 요청을 처리 중일 수 있다.
   */
  virtual bool is_reserved_line() {
    // if any of the sector is reserved, then the line is reserved
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 섹터 순서대로 검사 */
      if (m_status[i] == RESERVED) return true; /* [한국어] RESERVED 섹터 발견 시 즉시 true 반환 */
    }
    return false; /* [한국어] RESERVED 섹터 없음 */
  }
  /*
   * [한국어]
   * is_modified_line - 하나라도 MODIFIED 섹터가 있으면 true (write-back 필요)
   *
   * @return: MODIFIED 섹터가 하나라도 있으면 true
   *
   * 이 라인을 교체할 때 write-back 요청 발행이 필요한지 여부를 결정.
   * write-back은 get_dirty_sector_mask()로 더티 섹터만 선택하여 처리.
   */
  virtual bool is_modified_line() {
    // if any of the sector is modified, then the line is modified
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 섹터 순서대로 검사 */
      if (m_status[i] == MODIFIED) return true; /* [한국어] MODIFIED 섹터 발견 시 즉시 true 반환 */
    }
    return false; /* [한국어] MODIFIED 섹터 없음 — write-back 불필요 */
  }

  /* [한국어] get_status - sector_mask가 가리키는 섹터의 상태 반환
   * 섹터 캐시에서 특정 섹터가 VALID/RESERVED/MODIFIED인지 개별 확인할 때 사용. */
  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 마스크 → 인덱스(0~3) 변환 */

    return m_status[sidx]; /* [한국어] 해당 섹터의 현재 상태 반환 */
  }

  /* [한국어] set_status - sector_mask가 가리키는 섹터의 상태를 직접 설정
   * flush() 또는 invalidate() 처리 시 INVALID로 강제 변경할 때 사용. */
  virtual void set_status(enum cache_block_state status,
                          mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 마스크 → 인덱스 변환 */
    m_status[sidx] = status; /* [한국어] 해당 섹터 상태 변경 */
  }

  /* [한국어] set_byte_mask (mem_fetch 오버로드) - mem_fetch 접근 바이트를 더티 마스크에 OR 누적
   * write 히트/write-alloc fill 시 실제 수정된 바이트 추적. */
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask(); /* [한국어] 더티 바이트 누적 (OR) */
  }
  /* [한국어] set_byte_mask (byte_mask 오버로드) - 직접 바이트 마스크를 더티 마스크에 OR 누적 */
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask; /* [한국어] 더티 바이트 누적 (OR) */
  }
  /* [한국어] get_dirty_byte_mask - 라인 전체의 더티 바이트 비트마스크 반환
   * write-back 요청 생성 시 실제 수정된 바이트 범위 지정에 사용. */
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask; /* [한국어] 128비트 더티 바이트 마스크 반환 */
  }
  /* [한국어] get_dirty_sector_mask - MODIFIED 상태인 섹터들의 비트마스크 반환 (4비트)
   * write-back 시 더티 섹터만 선택적으로 전송하기 위해 evicted_block_info에 저장됨. */
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask; /* [한국어] 결과 섹터 마스크, 초기값 0 */
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) { /* [한국어] 4개 섹터 순서대로 검사 */
      if (m_status[i] == MODIFIED) sector_mask.set(i); /* [한국어] MODIFIED 섹터 비트 set */
    }
    return sector_mask; /* [한국어] 더티 섹터 마스크 반환 */
  }
  /* [한국어] get_last_access_time - LRU 교체 정책에서 라인 레벨 마지막 접근 시각 반환
   * 섹터 캐시에서 교체 단위는 라인이므로 라인 레벨 타임스탬프를 사용. */
  virtual unsigned long long get_last_access_time() {
    return m_line_last_access_time; /* [한국어] 라인 레벨 마지막 접근 시각 반환 */
  }

  /* [한국어] set_last_access_time - 섹터 레벨과 라인 레벨 접근 시각 동시 갱신
   * 특정 섹터에 접근할 때 해당 섹터의 타임스탬프와 라인 전체의 타임스탬프를 모두 업데이트. */
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 인덱스 추출 */

    m_last_sector_access_time[sidx] = time; /* [한국어] 해당 섹터의 마지막 접근 시각 갱신 */
    m_line_last_access_time = time;         /* [한국어] 라인 레벨 마지막 접근 시각도 갱신 (LRU용) */
  }

  /* [한국어] get_alloc_time - FIFO 교체 정책에서 라인 레벨 할당 시각 반환
   * 라인 첫 번째 섹터 할당 시 설정된 m_line_alloc_time을 반환. */
  virtual unsigned long long get_alloc_time() { return m_line_alloc_time; }

  /* [한국어] set_ignore_on_fill - 특정 섹터의 fill 시 상태 검사 무시 플래그 설정 */
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 인덱스 추출 */
    m_ignore_on_fill_status[sidx] = m_ignore;      /* [한국어] 해당 섹터의 무시 플래그 설정 */
  }

  /* [한국어] set_modified_on_fill - 특정 섹터의 fill 완료 시 MODIFIED 전이 플래그 설정 */
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);   /* [한국어] 섹터 인덱스 추출 */
    m_set_modified_on_fill[sidx] = m_modified;       /* [한국어] 해당 섹터의 MODIFIED 전이 플래그 설정 */
  }
  /* [한국어] set_byte_mask_on_fill - 라인 전체에 적용되는 byte_mask-on-fill 플래그 설정
   * sector_cache_block에서도 m_set_byte_mask_on_fill은 섹터별이 아닌 라인 단위. */
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified; /* [한국어] 라인 레벨 byte_mask-on-fill 플래그 설정 */
  }

  /* [한국어] set_readable_on_fill - 특정 섹터의 fill 완료 시 readable 활성화 플래그 설정
   * LAZY_FETCH_ON_READ 정책: fill 전까지 해당 섹터 읽기 차단. */
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 인덱스 추출 */
    m_set_readable_on_fill[sidx] = readable;       /* [한국어] 해당 섹터의 readable-on-fill 플래그 설정 */
  }
  /* [한국어] set_m_readable - 특정 섹터의 readable 플래그 직접 설정 */
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 인덱스 추출 */
    m_readable[sidx] = readable;                   /* [한국어] 해당 섹터 readable 플래그 직접 변경 */
  }

  /* [한국어] is_readable - 특정 섹터가 읽기 가능한지 확인
   * LAZY_FETCH_ON_READ 정책: fill 완료 전 읽기 접근 차단에 사용. */
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask); /* [한국어] 섹터 인덱스 추출 */
    return m_readable[sidx];                       /* [한국어] 해당 섹터 readable 플래그 반환 */
  }

  /* [한국어] get_modified_size - 더티 섹터 수 × SECTOR_SIZE(32B) 반환
   * write-back에 실제로 전송해야 할 바이트 수를 계산. MODIFIED 섹터만 카운트. */
  virtual unsigned get_modified_size() {
    unsigned modified = 0;                         /* [한국어] 더티 섹터 카운터 초기화 */
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 섹터 순서대로 검사 */
      if (m_status[i] == MODIFIED) modified++;     /* [한국어] MODIFIED 섹터 카운트 증가 */
    }
    return modified * SECTOR_SIZE; /* [한국어] 더티 섹터 수 × 32B = write-back 데이터 크기 */
  }

  /* [한국어] print_status - 디버그용: 블록 주소와 4개 섹터 상태 코드를 stdout에 출력 */
  virtual void print_status() {
    printf("m_block_addr is %llu, status = %u %u %u %u\n", m_block_addr,
           m_status[0], m_status[1], m_status[2], m_status[3]);
    /* [한국어] 블록 주소와 섹터 0~3의 cache_block_state 숫자값 순서대로 출력 */
  }

 private:
  unsigned m_sector_alloc_time[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터가 allocate(RESERVED 전이)된 사이클 배열 [섹터0, 섹터1, 섹터2, 섹터3].
   * 설정자: allocate_line()/allocate_sector()에서 해당 섹터 할당 시 현재 사이클로 설정.
   * 읽는 자: 현재 직접 참조되지 않지만 섹터별 통계 확장 목적으로 보존.
   * 값 범위: 0(할당 안 됨) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_last_sector_access_time[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터에 마지막으로 접근된 사이클 배열.
   * 설정자: set_last_access_time()에서 갱신 (allocate 시에도 초기값으로 설정).
   * 읽는 자: 섹터별 LRU 교체 정책(현재 미구현, 라인 레벨 LRU 사용).
   * 값 범위: 0 ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_sector_fill_time[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터에 fill()이 완료된 사이클 배열.
   * 설정자: fill()에서 해당 섹터 완료 시 현재 사이클로 설정.
   * 읽는 자: 현재 직접 참조되지 않지만 섹터별 지연시간 분석 목적으로 보존.
   * 값 범위: 0(fill 없음) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_line_alloc_time;
  /* [한국어] 이 라인의 첫 번째 섹터가 할당된 사이클 (FIFO 교체 정책의 기준).
   * 설정자: allocate_line()에서 최초 섹터 할당 시 설정. allocate_sector()에서는 변경 안 함.
   * 읽는 자: get_alloc_time() — FIFO 교체 정책에서 가장 오래된 라인 선택.
   * 값 범위: 0(할당 안 됨) ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_line_last_access_time;
  /* [한국어] 이 라인의 어느 섹터든 마지막으로 접근된 사이클 (LRU 교체 정책의 기준).
   * 설정자: set_last_access_time()에서 섹터 접근 시 라인 레벨도 함께 갱신.
   * 읽는 자: get_last_access_time() — LRU 교체 정책에서 교체 후보 선택.
   * 값 범위: 0 ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_line_fill_time;
  /* [한국어] 가장 최근에 어느 섹터든 fill()된 사이클 (라인 레벨 fill 완료 시각).
   * 설정자: fill()에서 매번 현재 사이클로 업데이트.
   * 읽는 자: 현재 직접 참조되지 않음 — 디버그/향후 통계 목적.
   * 값 범위: 0 ~ 현재 시뮬레이터 사이클.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  cache_block_state m_status[SECTOR_CHUNCK_SIZE];
  /* [한국어] 4개 섹터 각각의 현재 상태 배열 (INVALID/RESERVED/VALID/MODIFIED).
   * 설정자: allocate_line()/allocate_sector()로 RESERVED, fill()로 VALID/MODIFIED, set_status()로 직접.
   * 읽는 자: is_invalid_line(), is_reserved_line(), is_modified_line(), get_status(), get_dirty_sector_mask().
   * 값 범위: cache_block_state enum의 4가지 값, 섹터별 독립적.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터의 fill 시 상태 검사 무시 플래그 배열.
   * 설정자: set_ignore_on_fill() — streaming 캐시 등 특수 정책에서 활성화.
   * 읽는 자: fill() 내부 assert 우회 여부 (현재 코드에서 주석 처리).
   * 값 범위: true/false, 섹터별 독립적.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_modified_on_fill[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터의 fill 완료 시 MODIFIED 전이 플래그 배열.
   * 설정자: allocate_line()/allocate_sector()에서 초기화; set_modified_on_fill()로 변경.
   * 읽는 자: fill() 내부 상태 전이 분기.
   * 값 범위: true(fill 후 MODIFIED) / false(fill 후 VALID), 섹터별 독립적.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_readable_on_fill[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터의 fill 완료 시 readable 활성화 플래그 배열.
   * 설정자: set_readable_on_fill() — LAZY_FETCH_ON_READ 정책에서 활성화.
   * 읽는 자: fill() 내부 readable 갱신 분기 (적용 후 false로 리셋).
   * 값 범위: true/false, 섹터별 독립적.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_set_byte_mask_on_fill;
  /* [한국어] 라인 전체에 적용되는 byte_mask-on-fill 플래그 (섹터별이 아닌 라인 단위).
   * 설정자: set_byte_mask_on_fill() — 특정 쓰기 정책에서 fill 시 byte 단위 더티 기록.
   * 읽는 자: fill() 내부 byte_mask 적용 분기.
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  bool m_readable[SECTOR_CHUNCK_SIZE];
  /* [한국어] 각 섹터의 읽기 가능 여부 플래그 배열.
   * 설정자: 생성자(init())에서 true, set_m_readable(), fill()의 set_readable_on_fill 연계.
   * 읽는 자: is_readable() — LAZY_FETCH_ON_READ 정책에서 fill 전 읽기 접근 차단.
   * 값 범위: true(읽기 가능) / false(fill 대기 중), 섹터별 독립적.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  mem_access_byte_mask_t m_dirty_byte_mask;
  /* [한국어] 라인 전체의 더티 바이트 비트마스크 (128비트, 섹터 분할 없이 라인 단위 관리).
   * 설정자: set_byte_mask()로 쓰기 접근마다 OR 누적.
   * 읽는 자: get_dirty_byte_mask() — write-back 요청 생성 시 실제 수정 바이트 범위 결정.
   * 값 범위: bitset<128>, 각 비트가 해당 바이트 오프셋의 더티 여부.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  /*
   * [한국어]
   * get_sector_index - sector_mask(4비트 bitset)에서 섹터 인덱스(0~3)를 추출하는 내부 헬퍼
   *
   * @sector_mask: 1비트만 set된 4비트 섹터 마스크 (불변 조건: count()==1)
   * @return: set된 비트 위치(0~3) — 오류 시 SECTOR_CHUNCK_SIZE(4) 반환
   *
   * sector_mask는 항상 정확히 1비트만 set되어야 한다 (assert로 검증).
   * 비트 스캔으로 set 비트의 위치를 찾아 배열 인덱스로 사용.
   *
   * 호출 체인: get_status(), set_status(), fill(), set_last_access_time() 등 내부에서 사용
   */
  unsigned get_sector_index(mem_access_sector_mask_t sector_mask) {
    assert(sector_mask.count() == 1); /* [한국어] sector_mask에 정확히 1비트만 set되어야 한다는 불변 조건 검증 */
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) { /* [한국어] 4개 비트 위치 순서대로 검사 */
      if (sector_mask.to_ulong() & (1 << i)) return i;  /* [한국어] i번째 비트가 set이면 인덱스 i 반환 */
    }
    return SECTOR_CHUNCK_SIZE;  // error
    /* [한국어] 여기 도달하면 오류 — assert로 이미 차단되어야 하지만 안전망으로 존재 */
  }
};

/* [한국어] 캐시 라인 교체 정책 열거형.
 * tag_array::access()에서 교체 후보 라인 선택 시 m_replacement_policy 필드로 적용.
 * LRU: 가장 오래전에 접근된 라인 교체 — 시간적 지역성 활용에 유리하나 연산 오버헤드 존재.
 * FIFO: 가장 먼저 할당된 라인 교체 — 할당 순서만 기록하면 되어 구현이 단순함. */
enum replacement_policy_t {
  LRU,  /* [한국어] Least Recently Used — 마지막 접근 시각 기준으로 가장 오래된 라인 교체 */
  FIFO  /* [한국어] First In First Out — 할당 시각 기준으로 가장 먼저 들어온 라인 교체 */
};

/* [한국어] 캐시 쓰기 정책 열거형.
 * gpgpusim.config 문자열의 'wp' 필드로 지정되며 cache_config::init()에서 m_write_policy에 저장.
 * data_cache::init()에서 m_wr_hit 함수 포인터를 이 정책에 따라 선택한다.
 * 설정 문자 매핑: R=READ_ONLY, B=WRITE_BACK, T=WRITE_THROUGH, E=WRITE_EVICT, L=LOCAL_WB_GLOBAL_WT */
enum write_policy_t {
  READ_ONLY,        /* [한국어] 쓰기 불가 읽기 전용 캐시 — 텍스처/상수 캐시에 사용. 현재는 별도 read_only_cache 클래스로 대체됨 */
  WRITE_BACK,       /* [한국어] 더티 라인 축출 시에만 하위 메모리 갱신 — L1/L2 데이터 캐시 주요 정책 */
  WRITE_THROUGH,    /* [한국어] 매 쓰기마다 캐시와 하위 메모리를 동시 갱신 — 쓰기 대역폭 증가 */
  WRITE_EVICT,      /* [한국어] 쓰기 히트 시 해당 라인 무효화 후 하위 메모리로 직접 쓰기 — L1 전역 메모리 정책 */
  LOCAL_WB_GLOBAL_WT /* [한국어] 로컬 메모리는 WRITE_BACK, 전역 메모리는 WRITE_THROUGH (Fermi 혼합 정책) */
};

/* [한국어] 캐시 라인 할당 시점 정책 열거형.
 * gpgpusim.config 'ap' 필드로 지정되며 cache_config::init()에서 m_alloc_policy에 저장.
 * ON_MISS와 STREAMING의 핵심 차이: STREAMING은 내부적으로 ON_FILL로 변환되어
 * 라인 할당 실패(LINE_ALLOC_FAIL) 스톨을 제거한다. 설정 문자: m=ON_MISS, f=ON_FILL, s=STREAMING */
enum allocation_policy_t {
  ON_MISS,    /* [한국어] 캐시 미스 탐지 즉시 RESERVED 라인 할당 — 전통적인 방식, MSHR와 라인 수 일치 */
  ON_FILL,    /* [한국어] 하위 메모리 응답(fill) 도착 시 라인 할당 — RESERVED 상태 없음, 교체 시 dirty 라인 evict 불가 */
  STREAMING   /* [한국어] 스트리밍 워크로드용 특수 정책 — 내부적으로 ON_FILL로 변환, MSHR=라인 수로 설정 (Pascal/Volta 스타일) */
};

/* [한국어] 쓰기 미스 시 캐시 라인 할당 방식 열거형.
 * gpgpusim.config 'wap' 필드로 지정되며 cache_config::init()에서 m_write_alloc_policy에 저장.
 * data_cache::init()에서 m_wr_miss 함수 포인터를 이 정책에 따라 선택한다.
 * 참고: Jouppi, "Cache write policies and performance", ISCA 1993.
 * 설정 문자 매핑: N=NO_WRITE_ALLOCATE, W=WRITE_ALLOCATE, F=FETCH_ON_WRITE, L=LAZY_FETCH_ON_READ */
enum write_allocate_policy_t {
  NO_WRITE_ALLOCATE,    /* [한국어] 쓰기 미스 시 라인 할당 없이 하위 메모리로 직접 전달 — wr_miss_no_wa() 사용 */
  WRITE_ALLOCATE,       /* [한국어] 쓰기 미스 시 WRITE+READ 동시 발행 후 라인 할당 — GPGPU-Sim 3.x 레거시 방식, wr_miss_wa_naive() 사용 */
  FETCH_ON_WRITE,       /* [한국어] 쓰기 미스 시 먼저 라인 fetch 후 쓰기 적용 — Volta L1 방식, wr_miss_wa_fetch_on_write() 사용 */
  LAZY_FETCH_ON_READ    /* [한국어] 쓰기 시 MODIFIED 표시만 하고 읽기 미스 시에만 fetch — 메모리 트래픽 최소화, wr_miss_wa_lazy_fetch_on_read() 사용 */
};

/* [한국어] MSHR(Miss Status Holding Register) 구현 방식 열거형.
 * gpgpusim.config 'mshr_type' 필드로 지정되며 cache_config::init()에서 m_mshr_type에 저장.
 * tex_cache는 TEX_FIFO/SECTOR_TEX_FIFO, baseline_cache 파생 클래스는 ASSOC/SECTOR_ASSOC 사용.
 * 설정 문자 매핑: F=TEX_FIFO, T=SECTOR_TEX_FIFO, A=ASSOC, S=SECTOR_ASSOC */
enum mshr_config_t {
  TEX_FIFO,         /* [한국어] 텍스처 캐시용 FIFO MSHR — 요청 순서 보장, mshr_table 아닌 tex_cache 내부 ROB로 구현 */
  ASSOC,            /* [한국어] 일반 데이터 캐시용 완전 연상 MSHR — mshr_table 클래스로 구현, 같은 블록 요청 병합 */
  SECTOR_TEX_FIFO,  /* [한국어] 섹터 캐시(L2)에 요청하는 텍스처 캐시용 FIFO MSHR — 섹터 단위 pending_read 추적 */
  SECTOR_ASSOC      /* [한국어] 섹터 캐시에 요청하는 일반 캐시용 연상 MSHR — pending_read로 섹터 응답 카운트 관리 */
};

/* [한국어] 캐시 세트 인덱스 계산 함수 선택 열거형.
 * gpgpusim.config 'sif' 필드로 지정되며 cache_config::init()에서 m_set_index_function에 저장.
 * cache_config::set_index()가 이 값에 따라 hash_function() 또는 직접 계산을 선택.
 * hashing.cc의 ipoly_hash_function()/bitwise_hash_function() 등이 실제 구현체.
 * 설정 문자 매핑: L=LINEAR, X=BITWISE_XORING, P=HASH_IPOLY, H=FERMI_HASH, C=CUSTOM */
enum set_index_function {
  LINEAR_SET_FUNCTION = 0,  /* [한국어] 주소 하위 nset_log2 비트를 그대로 세트 인덱스로 사용 — 단순하나 스트라이드 패턴에 충돌 취약 */
  BITWISE_XORING_FUNCTION,  /* [한국어] 주소 상위/하위 비트 XOR — 세트 분산 향상, bitwise_hash_function() 구현 */
  HASH_IPOLY_FUNCTION,      /* [한국어] GF(2) 위의 기약다항식 기반 해시 — 가장 균등한 분산, ipoly_hash_function() 구현 */
  FERMI_HASH_SET_FUNCTION,  /* [한국어] Fermi GPU 실제 하드웨어 해시 재현 (Nugteren et al., HPCA 2014) — 실제 GPU와 동일한 매핑 */
  CUSTOM_SET_FUNCTION       /* [한국어] 사용자 정의 해시 함수 — 현재 미구현, 확장성을 위한 플레이스홀더 */
};

/* [한국어] 캐시 블록 관리 단위를 선택하는 열거형.
 * gpgpusim.config 'ct' 필드로 지정되며 cache_config::init()에서 m_cache_type에 저장.
 * NORMAL: line_cache_block 사용 (128B 라인 전체를 단일 상태로 관리).
 * SECTOR: sector_cache_block 사용 (128B 라인을 4개×32B 섹터로 분할 관리 — Volta L2 스타일).
 * 설정 문자 매핑: N=NORMAL, S=SECTOR */
enum cache_type {
  NORMAL = 0, /* [한국어] 라인 전체(128B)를 단일 상태로 관리 — line_cache_block 인스턴스화, 단순하나 섹터 선택적 fetch 불가 */
  SECTOR      /* [한국어] 라인을 4개 32B 섹터로 분할 관리 — sector_cache_block 인스턴스화, 필요한 섹터만 fetch하여 대역폭 절감 */
};

#define MAX_WARP_PER_SHADER 64  /* [한국어] SM(Streaming Multiprocessor) 하나가 동시에 수용할 수 있는 최대 warp 수 (Fermi/Kepler 기준 64) */
#define INCT_TOTAL_BUFFER 64    /* [한국어] ICNT(인터커넥트 네트워크) 입출력 버퍼의 총 슬롯 수 — 인터커넥트 포화 여부 판단에 사용 */
#define L2_TOTAL 64             /* [한국어] L2 캐시 파티션의 총 버퍼 슬롯 수 — L2 포화 판단 기준 */
#define MAX_WARP_PER_SHADER 64  /* [한국어] MAX_WARP_PER_SHADER 재정의 (중복 — 포함 순서 보호용) */
#define MAX_WARP_PER_SHADER 64  /* [한국어] MAX_WARP_PER_SHADER 재정의 (중복 — 포함 순서 보호용) */

class cache_config {
 public:
  /* [한국어] cache_config 기본 생성자 — 모든 파라미터를 미초기화 상태로 설정.
   * option_parser에 의해 m_config_string 등이 채워진 후 init()가 호출되어야 실제로 사용 가능해진다.
   * m_valid=false이므로 init() 전에 get_line_sz() 등 accessor 호출 시 assert로 차단된다.
   * 호출 체인: 시뮬레이터 초기화 → cache_config 생성자 → option_parser 파싱 → init() */
  cache_config() {
    m_valid = false;              /* [한국어] 초기화 미완료 플래그 — init() 호출 전까지 accessor 사용 차단 */
    m_disabled = false;           /* [한국어] 캐시 비활성화 플래그 초기화 — "none" 설정 문자열 시 true로 변경 */
    m_config_string = NULL;       // set by option parser
    /* [한국어] 설정 문자열 포인터 — option_parser가 gpgpusim.config 문자열을 가리키도록 설정함 */
    m_config_stringPrefL1 = NULL; /* [한국어] L1 캐시 prefetch 설정 문자열 포인터 초기화 */
    m_config_stringPrefShared = NULL; /* [한국어] Shared memory prefetch 설정 문자열 포인터 초기화 */
    m_data_port_width = 0;        /* [한국어] 데이터 포트 폭 0으로 초기화 — init()에서 미지정 시 m_line_sz로 설정됨 */
    m_set_index_function = LINEAR_SET_FUNCTION; /* [한국어] 기본 세트 인덱싱 = LINEAR (하위 비트 직접 추출) */
    m_is_streaming = false;       /* [한국어] 스트리밍 캐시 모드 비활성화로 초기화 */
    m_wr_percent = 0;             /* [한국어] 더티 라인 비율 임계값 0으로 초기화 */
  }
  /*
   * [한국어]
   * cache_config::init - gpgpusim.config 설정 문자열을 파싱하여 캐시 파라미터 초기화
   *
   * @config: gpgpusim.config에서 읽은 캐시 설정 문자열
   *   형식: "ct:m_nset:m_line_sz:m_assoc,rp:wp:ap:wap:sif,mshr_type:m_mshr_entries:m_mshr_max_merge,
   *          miss_queue_size:result_fifo_entries,data_port_width"
   *   예시: "N:32:128:4,L:B:m:W:L,A:32:10,32:4,128"
   *   ct: N=NORMAL, S=SECTOR
   *   rp: L=LRU, F=FIFO
   *   wp: R=READ_ONLY, B=WRITE_BACK, T=WRITE_THROUGH, E=WRITE_EVICT, L=LOCAL_WB_GLOBAL_WT
   *   ap: m=ON_MISS, f=ON_FILL, s=STREAMING
   *   wap: N=NO_WRITE_ALLOCATE, W=WRITE_ALLOCATE, F=FETCH_ON_WRITE, L=LAZY_FETCH_ON_READ
   *   sif: H=FERMI_HASH, P=HASH_IPOLY, C=CUSTOM, L=LINEAR, X=BITWISE_XORING
   *   mshr_type: F=TEX_FIFO, T=SECTOR_TEX_FIFO, A=ASSOC, S=SECTOR_ASSOC
   * @status: FuncCache 상태 (gpgpu_func_cache_prefer 등) — cache_status 필드에 저장
   *
   * STREAMING 정책은 내부적으로 ON_FILL로 변환되고 m_is_streaming=true 플래그가 설정된다.
   * 파싱 후 여러 무효 설정 조합(ON_FILL+WRITE_BACK 등)을 assert로 차단한다.
   * 호출 체인: option_parser 파싱 완료 후 → cache_config::init()
   */
  void init(char *config, FuncCache status) {
    cache_status = status; /* [한국어] FuncCache 상태 저장 — get_cache_status()로 조회 가능 */
    assert(config);        /* [한국어] 설정 문자열이 NULL이면 즉시 종료 */
    char ct, rp, wp, ap, mshr_type, wap, sif;
    /* [한국어] 설정 문자열에서 파싱할 단일 문자 변수들:
     *   ct=캐시 유형, rp=교체 정책, wp=쓰기 정책, ap=할당 정책,
     *   wap=쓰기 미스 할당 정책, sif=세트 인덱스 함수, mshr_type=MSHR 유형 */

    int ntok =
        sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
               &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
               &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
               &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);
    /* [한국어] 설정 문자열에서 최대 15개 토큰을 파싱:
     *   ct, m_nset(세트 수), m_line_sz(라인 크기), m_assoc(연상도),
     *   rp, wp, ap, wap, sif, mshr_type, m_mshr_entries(MSHR 엔트리 수),
     *   m_mshr_max_merge(최대 병합 수), m_miss_queue_size, m_result_fifo_entries, m_data_port_width
     *   ntok < 12이면 파싱 실패 (최소 필수 토큰 수 = 12) */

    if (ntok < 12) {
      /* [한국어] 파싱 성공 토큰 수 12 미만 — 유효하지 않은 설정 */
      if (!strcmp(config, "none")) {
        /* [한국어] "none" 문자열이면 캐시 비활성화 처리 후 즉시 반환 */
        m_disabled = true; /* [한국어] 비활성화 플래그 설정 — disabled() 조회 시 true 반환 */
        return;
      }
      exit_parse_error(); /* [한국어] "none"도 아니고 파싱도 실패 — 에러 출력 후 abort() */
    }

    switch (ct) { /* [한국어] 캐시 유형(ct) 설정: N=NORMAL, S=SECTOR */
      case 'N':
        m_cache_type = NORMAL; /* [한국어] line_cache_block 사용 — 라인 전체 단위 관리 */
        break;
      case 'S':
        m_cache_type = SECTOR; /* [한국어] sector_cache_block 사용 — 32B 섹터 단위 분할 관리 */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 캐시 유형 — abort() */
    }
    switch (rp) { /* [한국어] 교체 정책(rp) 설정: L=LRU, F=FIFO */
      case 'L':
        m_replacement_policy = LRU;  /* [한국어] Least Recently Used 교체 정책 */
        break;
      case 'F':
        m_replacement_policy = FIFO; /* [한국어] First In First Out 교체 정책 */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 교체 정책 — abort() */
    }
    switch (wp) { /* [한국어] 쓰기 정책(wp) 설정: R/B/T/E/L */
      case 'R':
        m_write_policy = READ_ONLY;         /* [한국어] 읽기 전용 — 현재는 deprecated, read_only_cache 클래스 사용 권장 */
        break;
      case 'B':
        m_write_policy = WRITE_BACK;        /* [한국어] 쓰기 되돌림 — dirty 라인 축출 시에만 하위 메모리 갱신 */
        break;
      case 'T':
        m_write_policy = WRITE_THROUGH;     /* [한국어] 즉시 쓰기 — 매 쓰기마다 캐시+하위 메모리 동시 갱신 */
        break;
      case 'E':
        m_write_policy = WRITE_EVICT;       /* [한국어] 쓰기 축출 — 쓰기 히트 시 라인 무효화 후 하위로 전달 */
        break;
      case 'L':
        m_write_policy = LOCAL_WB_GLOBAL_WT; /* [한국어] 로컬=WRITE_BACK, 전역=WRITE_THROUGH 혼합 정책 */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 쓰기 정책 — abort() */
    }
    switch (ap) { /* [한국어] 라인 할당 정책(ap) 설정: m=ON_MISS, f=ON_FILL, s=STREAMING */
      case 'm':
        m_alloc_policy = ON_MISS;    /* [한국어] 미스 즉시 라인 RESERVED 할당 */
        break;
      case 'f':
        m_alloc_policy = ON_FILL;    /* [한국어] fill 응답 도착 시 라인 할당 */
        break;
      case 's':
        m_alloc_policy = STREAMING;  /* [한국어] 스트리밍 모드 — 아래에서 ON_FILL로 변환됨 */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 할당 정책 — abort() */
    }
    if (m_alloc_policy == STREAMING) {
      /*
      For streaming cache:
      (1) we set the alloc policy to be on-fill to remove all line_alloc_fail
      stalls. if the whole memory is allocated to the L1 cache, then make the
      allocation to be on_MISS otherwise, make it ON_FILL to eliminate line
      allocation fails. i.e. MSHR throughput is the same, independent on the L1
      cache size/associativity So, we set the allocation policy per kernel
      basis, see shader.cc, max_cta() function

      (2) We also set the MSHRs to be equal to max
      allocated cache lines. This is possible by moving TAG to be shared
      between cache line and MSHR enrty (i.e. for each cache line, there is
      an MSHR rntey associated with it). This is the easiest think we can
      think of to model (mimic) L1 streaming cache in Pascal and Volta

      For more information about streaming cache, see:
      http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
      https://ieeexplore.ieee.org/document/8344474/
      */
      /* [한국어] STREAMING → ON_FILL 변환:
       *   (1) LINE_ALLOC_FAIL 스톨 제거: fill 시 할당하므로 미스 시점에 빈 슬롯 필요 없음.
       *       단, L1 전체가 캐시에 할당된 경우 shader.cc max_cta()에서 다시 ON_MISS로 조정.
       *   (2) MSHR 수 = 최대 캐시 라인 수: 각 캐시 라인에 MSHR 엔트리가 1:1 대응.
       *       이를 통해 Pascal/Volta의 L1 스트리밍 캐시 동작을 모델링. */
      m_is_streaming = true;     /* [한국어] 스트리밍 모드 플래그 활성화 */
      m_alloc_policy = ON_FILL;  /* [한국어] 실제 동작은 ON_FILL — STREAMING은 설정상 별칭 */
    }
    switch (mshr_type) { /* [한국어] MSHR 유형(mshr_type) 설정: F/T/A/S */
      case 'F':
        m_mshr_type = TEX_FIFO;      /* [한국어] 텍스처 캐시용 FIFO MSHR */
        assert(ntok == 14);          /* [한국어] TEX_FIFO는 15번째 토큰(data_port_width) 없음 — 14개만 파싱 */
        break;
      case 'T':
        m_mshr_type = SECTOR_TEX_FIFO; /* [한국어] 섹터 캐시로 요청하는 텍스처 FIFO MSHR */
        assert(ntok == 14);            /* [한국어] SECTOR_TEX_FIFO도 14개 토큰 */
        break;
      case 'A':
        m_mshr_type = ASSOC;         /* [한국어] 일반 데이터 캐시용 완전 연상 MSHR */
        break;
      case 'S':
        m_mshr_type = SECTOR_ASSOC;  /* [한국어] 섹터 캐시용 완전 연상 MSHR */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 MSHR 유형 — abort() */
    }
    m_line_sz_log2 = LOGB2(m_line_sz); /* [한국어] 캐시라인 크기의 log2 계산 — 비트 추출에 사용 */
    m_nset_log2 = LOGB2(m_nset);       /* [한국어] 세트 수의 log2 계산 — 세트 인덱스 비트 수 */
    m_valid = true;                     /* [한국어] 파싱 완료 — accessor 사용 허용 */
    m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
    /* [한국어] 원자적 접근 단위 크기: SECTOR 캐시이면 32B(SECTOR_SIZE), NORMAL이면 라인 전체.
     *   mshr_addr() 계산 시 이 크기로 정렬하여 같은 원자 단위 요청을 MSHR에 병합. */
    m_sector_sz_log2 = LOGB2(SECTOR_SIZE); /* [한국어] 섹터 크기(32B)의 log2 = 5 — 섹터 오프셋 비트 수 */
    original_m_assoc = m_assoc;             /* [한국어] 원본 연상도 저장 — Volta 동적 크기 조정 후 복원에 사용 */

    // For more details about difference between FETCH_ON_WRITE and WRITE
    // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
    // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
    // GPGPU-sim 3.x, that send WRITE and READ for every write request
    switch (wap) { /* [한국어] 쓰기 미스 할당 정책(wap) 설정: N/W/F/L */
      case 'N':
        m_write_alloc_policy = NO_WRITE_ALLOCATE; /* [한국어] 쓰기 미스 시 라인 할당 없이 바로 하위 메모리로 */
        break;
      case 'W':
        m_write_alloc_policy = WRITE_ALLOCATE;    /* [한국어] 쓰기 미스 시 WRITE+READ 동시 발행 (구식 방식) */
        break;
      case 'F':
        m_write_alloc_policy = FETCH_ON_WRITE;    /* [한국어] 쓰기 미스 시 fetch 먼저, Volta L1 방식 */
        break;
      case 'L':
        m_write_alloc_policy = LAZY_FETCH_ON_READ; /* [한국어] 쓰기 시 MODIFIED만 표시, 읽기 시에만 fetch */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 쓰기 미스 할당 정책 — abort() */
    }

    // detect invalid configuration
    if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
        m_write_policy == WRITE_BACK) {
      // A writeback cache with allocate-on-fill policy will inevitably lead to
      // deadlock: The deadlock happens when an incoming cache-fill evicts a
      // dirty line, generating a writeback request.  If the memory subsystem is
      // congested, the interconnection network may not have sufficient buffer
      // for the writeback request.  This stalls the incoming cache-fill.  The
      // stall may propagate through the memory subsystem back to the output
      // port of the same core, creating a deadlock where the wrtieback request
      // and the incoming cache-fill are stalling each other.
      /* [한국어] ON_FILL + WRITE_BACK 조합은 교착 상태(deadlock) 발생 가능:
       *   fill 시 dirty 라인이 축출되며 write-back 요청이 생성되는데,
       *   ICNT 버퍼가 꽉 찬 상태에서 write-back 요청을 보낼 수 없으면
       *   fill 자체가 블록되고 이 stall이 전파되어 교착 상태가 된다. */
      assert(0 &&
             "Invalid cache configuration: Writeback cache cannot allocate new "
             "line on fill. ");
    }

    if ((m_write_alloc_policy == FETCH_ON_WRITE ||
         m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        m_alloc_policy == ON_FILL) {
      /* [한국어] FETCH_ON_WRITE/LAZY_FETCH_ON_READ는 ON_FILL 할당 정책과 함께 작동 불가:
       *   미스 탐지 시점에 라인 RESERVED가 필요한데 ON_FILL은 fill 시점에 할당하므로 충돌. */
      assert(
          0 &&
          "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
          "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    }

    if (m_cache_type == SECTOR) {
      /* [한국어] SECTOR 캐시 유효성 검증: 라인 크기 = SECTOR_SIZE × SECTOR_CHUNCK_SIZE 여야 함
       *   (기본값: 128B = 32B × 4) — 하드코딩된 상수와 맞지 않으면 시뮬레이션 불가 */
      bool cond = m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
                  m_line_sz % SECTOR_SIZE == 0;
      if (!cond) {
        std::cerr << "error: For sector cache, the simulator uses hard-coded "
                     "SECTOR_SIZE and SECTOR_CHUNCK_SIZE. The line size "
                     "must be product of both values.\n";
        assert(0); /* [한국어] 라인 크기 조건 불충족 — abort() */
      }
    }

    // default: port to data array width and granularity = line size
    if (m_data_port_width == 0) {
      /* [한국어] data_port_width가 설정 문자열에 없거나 0이면 기본값 = 라인 크기(128B) */
      m_data_port_width = m_line_sz; /* [한국어] 포트 폭 = 라인 크기로 설정 (한 사이클에 라인 전체 접근 가능) */
    }
    assert(m_line_sz % m_data_port_width == 0);
    /* [한국어] 라인 크기가 포트 폭의 배수여야 함 — 라인을 정수 사이클 수로 전송 가능해야 함 */

    switch (sif) { /* [한국어] 세트 인덱스 함수(sif) 설정: H/P/C/L/X */
      case 'H':
        m_set_index_function = FERMI_HASH_SET_FUNCTION; /* [한국어] Fermi GPU 실제 하드웨어 해시 */
        break;
      case 'P':
        m_set_index_function = HASH_IPOLY_FUNCTION;     /* [한국어] GF(2) 기약다항식 해시 */
        break;
      case 'C':
        m_set_index_function = CUSTOM_SET_FUNCTION;     /* [한국어] 사용자 정의 해시 (미구현) */
        break;
      case 'L':
        m_set_index_function = LINEAR_SET_FUNCTION;     /* [한국어] 하위 비트 직접 추출 (기본값) */
        break;
      case 'X':
        m_set_index_function = BITWISE_XORING_FUNCTION; /* [한국어] 비트 XOR 해시 */
        break;
      default:
        exit_parse_error(); /* [한국어] 알 수 없는 세트 인덱스 함수 — abort() */
    }
  }
  /* [한국어] disabled - 캐시가 "none"으로 비활성화됐는지 확인 (true이면 이 캐시는 존재하지 않음) */
  bool disabled() const { return m_disabled; }
  /* [한국어] get_line_sz - 캐시라인 크기(바이트) 반환 (일반적으로 128B).
   * init() 완료 후에만 유효 — assert(m_valid)로 보호. */
  unsigned get_line_sz() const {
    assert(m_valid); /* [한국어] 초기화 완료 검증 */
    return m_line_sz; /* [한국어] 캐시라인 크기 반환 */
  }
  /* [한국어] get_atom_sz - 원자적 접근 단위 크기(바이트) 반환.
   * SECTOR 캐시: SECTOR_SIZE(32B), NORMAL 캐시: m_line_sz(128B).
   * mshr_addr() 계산의 정렬 단위로 사용. */
  unsigned get_atom_sz() const {
    assert(m_valid); /* [한국어] 초기화 완료 검증 */
    return m_atom_sz; /* [한국어] 원자 단위 크기 반환 */
  }
  /* [한국어] get_num_lines - 현재 캐시의 총 라인 수 반환 (m_nset × m_assoc).
   * tag_array의 m_lines 배열 크기와 동일. Volta 동적 리사이징 후 현재 크기 반영. */
  unsigned get_num_lines() const {
    assert(m_valid);          /* [한국어] 초기화 완료 검증 */
    return m_nset * m_assoc;  /* [한국어] 세트 수 × 연상도 = 총 캐시 라인 수 */
  }
  /* [한국어] get_max_num_lines - 이 캐시가 가질 수 있는 최대 라인 수 반환.
   * Volta 동적 리사이징: get_max_cache_multiplier() × m_nset × original_m_assoc.
   * l1d_cache_config에서 오버라이드되어 통합 캐시 크기 기반으로 계산. */
  unsigned get_max_num_lines() const {
    assert(m_valid); /* [한국어] 초기화 완료 검증 */
    return get_max_cache_multiplier() * m_nset * original_m_assoc;
    /* [한국어] 최대 배율 × 세트 수 × 원본 연상도 = 최대 라인 수 */
  }
  /* [한국어] get_max_assoc - 동적 리사이징 시 최대 연상도 반환.
   * Volta 통합 캐시에서 L1이 최대로 확장됐을 때의 연상도. */
  unsigned get_max_assoc() const {
    assert(m_valid); /* [한국어] 초기화 완료 검증 */
    return get_max_cache_multiplier() * original_m_assoc;
    /* [한국어] 최대 배율 × 원본 연상도 = 최대 연상도 */
  }
  /* [한국어] print - 캐시 설정 요약을 파일에 출력 (Size, Set수, Way수, 라인 크기) */
  void print(FILE *fp) const {
    fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
            m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
    /* [한국어] 총 바이트 크기, 세트 수, 연상도, 라인 크기를 사람이 읽을 수 있는 형태로 출력 */
  }

  /* [한국어] set_index - 주소에서 캐시 세트 인덱스를 계산 (virtual — l2_cache_config에서 오버라이드).
   * m_set_index_function에 따라 LINEAR/BITWISE_XOR/IPOLY/FERMI_HASH 중 하나를 선택.
   * 구현은 gpu-cache.cc에 있음. */
  virtual unsigned set_index(new_addr_type addr) const;

  /* [한국어] get_max_cache_multiplier - 동적 리사이징 최대 배율 반환 (기본값=4).
   * l1d_cache_config에서 오버라이드: 통합 캐시 크기 / 원본 캐시 크기로 계산.
   * Volta 아키텍처에서 L1/Shared 메모리 비율을 동적으로 조정할 때 사용. */
  virtual unsigned get_max_cache_multiplier() const {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER; /* [한국어] 기본 최대 배율 = 4 */
  }

  /* [한국어] hash_function - 세트 인덱스 해시 계산 헬퍼.
   * m_index_function에 따라 FERMI_HASH/IPOLY/BITWISE_XOR 해시를 hashing.cc 함수로 계산.
   * set_index()가 LINEAR가 아닌 경우 이 함수를 호출. */
  unsigned hash_function(new_addr_type addr, unsigned m_nset,
                         unsigned m_line_sz_log2, unsigned m_nset_log2,
                         unsigned m_index_function) const;

  /* [한국어] tag - 주소에서 캐시 태그를 계산 (태그 = 블록 주소 = 라인 정렬 주소).
   * 일반적인 태그 추출(상위 비트만)과 달리 풀 블록 주소를 태그로 사용.
   * 이유: 복잡한 세트 인덱스 함수를 쓸 때 인덱스+태그 전체 비교가 필요함 — 히트 판별 정확도 보장.
   * 구현: addr의 하위 m_line_sz 비트를 0으로 마스킹 → 라인 정렬 주소 반환. */
  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.

    // return addr >> (m_line_sz_log2+m_nset_log2);
    return addr & ~(new_addr_type)(m_line_sz - 1);
    /* [한국어] 하위 m_line_sz 비트 마스킹 → 캐시라인 정렬 주소(= 태그) 반환 */
  }
  /* [한국어] block_addr - 주소를 캐시라인 경계로 정렬 (tag()와 동일 결과, 의미 명확화용 별칭) */
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
    /* [한국어] 하위 m_line_sz 비트 마스킹 → 라인 베이스 주소 반환 */
  }
  /* [한국어] mshr_addr - MSHR 엔트리 키로 사용할 원자 단위 정렬 주소 반환.
   * NORMAL 캐시: 라인 정렬(= block_addr), SECTOR 캐시: 32B 섹터 정렬.
   * 같은 원자 단위로 정렬된 요청들이 동일 MSHR 엔트리에 병합된다. */
  new_addr_type mshr_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_atom_sz - 1);
    /* [한국어] 하위 m_atom_sz 비트 마스킹 → 원자 단위 정렬 주소 반환 */
  }
  /* [한국어] get_mshr_type - MSHR 구현 유형 반환 (TEX_FIFO/ASSOC/SECTOR_TEX_FIFO/SECTOR_ASSOC) */
  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
  /* [한국어] set_assoc - 연상도(way 수)를 동적으로 변경 (Volta 통합 캐시 크기 조정에 사용).
   * shader.cc max_cta() 함수가 커널별로 L1/Shared 메모리 분할 비율에 따라 호출. */
  void set_assoc(unsigned n) {
    // set new assoc. L1 cache dynamically resized in Volta
    m_assoc = n; /* [한국어] 연상도 변경 — tag_array의 get_num_lines()에 즉시 반영 */
  }
  /* [한국어] get_nset - 캐시 세트 수 반환. tag_array 인덱싱의 기준. */
  unsigned get_nset() const {
    assert(m_valid);  /* [한국어] 초기화 완료 검증 */
    return m_nset;    /* [한국어] 세트 수 반환 */
  }
  /* [한국어] get_total_size_inKB - 현재 캐시 총 크기를 KB 단위로 반환 (m_assoc × m_nset × m_line_sz / 1024) */
  unsigned get_total_size_inKB() const {
    assert(m_valid); /* [한국어] 초기화 완료 검증 */
    return (m_assoc * m_nset * m_line_sz) / 1024; /* [한국어] 연상도 × 세트 수 × 라인 크기(B) → KB */
  }
  /* [한국어] is_streaming - 스트리밍 캐시 모드(STREAMING 설정)인지 반환.
   * shader.cc max_cta()에서 ON_MISS↔ON_FILL 재조정 여부 결정에 사용. */
  bool is_streaming() { return m_is_streaming; }
  /* [한국어] get_cache_status - FuncCache 상태(gpgpu_func_cache_prefer 설정값) 반환.
   * 어떤 캐시 구성(L1이 shared memory 우선인지 data cache 우선인지)이 활성화됐는지 나타냄. */
  FuncCache get_cache_status() { return cache_status; }
  /* [한국어] set_allocation_policy - 라인 할당 정책 동적 변경 (ON_MISS ↔ ON_FILL 전환).
   * shader.cc max_cta()에서 스트리밍 캐시의 전체 사용 여부에 따라 정책을 런타임에 전환. */
  void set_allocation_policy(enum allocation_policy_t alloc) {
    m_alloc_policy = alloc; /* [한국어] 할당 정책 변경 — 다음 사이클부터 새 정책 적용 */
  }
  char *m_config_string;
  /* [한국어] gpgpusim.config에서 읽은 캐시 설정 문자열 포인터.
   * 설정자: option_parser가 파싱 시 이 포인터를 설정 문자열로 지정.
   * 읽는 자: cache_config::init()에서 sscanf 파싱 입력으로 사용; exit_parse_error()에서 에러 메시지 출력.
   * 값 범위: "none" 또는 "ct:nset:line_sz:assoc,..." 형식의 문자열; NULL은 option_parser 설정 전.
   * 동기화: 단일 스레드(시뮬레이터 초기화 단계)에서만 설정, 이후 읽기 전용. */
  char *m_config_stringPrefL1;
  /* [한국어] L1 캐시 prefetch 전용 설정 문자열 포인터.
   * 설정자: option_parser가 prefetch L1 옵션 파싱 시 설정.
   * 읽는 자: l1d_cache_config::init() 또는 관련 prefetch 초기화 코드.
   * 값 범위: NULL(미설정) 또는 설정 문자열 포인터.
   * 동기화: 단일 스레드(초기화 단계), 이후 읽기 전용. */
  char *m_config_stringPrefShared;
  /* [한국어] Shared memory prefetch 전용 설정 문자열 포인터.
   * 설정자: option_parser가 prefetch shared 옵션 파싱 시 설정.
   * 읽는 자: 관련 prefetch 초기화 코드.
   * 값 범위: NULL(미설정) 또는 설정 문자열 포인터.
   * 동기화: 단일 스레드(초기화 단계), 이후 읽기 전용. */
  FuncCache cache_status;
  /* [한국어] FuncCache 상태 — gpgpu_func_cache_prefer 설정에서 온 캐시 구성 우선순위.
   * 설정자: cache_config::init()에서 status 인자로 저장.
   * 읽는 자: get_cache_status() — shader.cc에서 캐시 구성 결정에 사용.
   * 값 범위: FuncCache enum 값(abstract_hardware_model.h 정의).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_wr_percent;
  /* [한국어] 더티(수정) 라인이 전체 캐시에서 차지할 최대 비율 임계값 (현재 직접 사용되지 않음).
   * 설정자: cache_config 생성자에서 0으로 초기화, 필요 시 option_parser가 설정.
   * 읽는 자: 더티 라인 조기 축출 정책 구현 시 참조 예정.
   * 값 범위: 0~100 (퍼센트).
   * 동기화: 단일 스레드(초기화 단계). */
  /* [한국어] get_write_allocate_policy - 쓰기 미스 할당 정책 반환 (data_cache::process_tag_probe에서 사용) */
  write_allocate_policy_t get_write_allocate_policy() {
    return m_write_alloc_policy; /* [한국어] 쓰기 미스 할당 정책 반환 */
  }
  /* [한국어] get_write_policy - 쓰기 정책 반환 (data_cache::init에서 m_wr_hit 함수 포인터 선택에 사용) */
  write_policy_t get_write_policy() { return m_write_policy; }

 protected:
  void exit_parse_error() {
    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
           m_config_string);
    abort();
  }

  bool m_valid;
  /* [한국어] 설정 파싱 완료 여부 플래그.
   * 설정자: cache_config() 생성자에서 false, init() 완료 시 true.
   * 읽는 자: get_line_sz() 등 accessor 함수의 assert(m_valid) 보호.
   * 값 범위: true(init 완료) / false(미초기화).
   * 동기화: 단일 스레드(초기화 단계). */
  bool m_disabled;
  /* [한국어] 캐시 비활성화 플래그 ("none" 설정 시 true).
   * 설정자: init()에서 config=="none"이면 true로 설정.
   * 읽는 자: disabled() — 상위 코드가 이 캐시를 스킵할지 결정.
   * 값 범위: true(비활성화) / false(활성).
   * 동기화: 단일 스레드(초기화 단계). */
  unsigned m_line_sz;
  /* [한국어] 캐시라인 크기(바이트), 일반적으로 128B.
   * 설정자: init()에서 sscanf 파싱으로 설정.
   * 읽는 자: tag(), block_addr(), get_line_sz(), m_data_port_width 기본값 등.
   * 값 범위: 2의 거듭제곱 값 (32, 64, 128 등).
   * 동기화: 초기화 후 읽기 전용(set_assoc 제외). */
  unsigned m_line_sz_log2;
  /* [한국어] m_line_sz의 log2 값 (비트 추출 연산에 사용).
   * 설정자: init()에서 LOGB2(m_line_sz)로 계산.
   * 읽는 자: set_index(), hash_function() — 주소에서 인덱스 비트 추출 시.
   * 값 범위: 5(32B) ~ 7(128B) 등.
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_nset;
  /* [한국어] 캐시 세트 수.
   * 설정자: init()에서 sscanf 파싱으로 설정.
   * 읽는 자: set_index(), get_nset(), get_num_lines(), get_total_size_inKB().
   * 값 범위: 2의 거듭제곱 (예: 4, 8, 32, 64 등).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_nset_log2;
  /* [한국어] m_nset의 log2 값 (세트 인덱스 추출에 필요한 비트 수).
   * 설정자: init()에서 LOGB2(m_nset)로 계산.
   * 읽는 자: hash_function() — 세트 인덱스 비트 수 계산.
   * 값 범위: 2(nset=4) ~ 6(nset=64) 등.
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_assoc;
  /* [한국어] 현재 연상도(way 수). Volta 동적 리사이징으로 runtime에 변경될 수 있음.
   * 설정자: init()에서 sscanf 파싱으로 초기 설정; set_assoc()으로 동적 변경 가능.
   * 읽는 자: get_num_lines(), get_total_size_inKB() — 현재 유효 캐시 크기 계산.
   * 값 범위: 2 이상의 정수 (예: 4, 8, 16, 32).
   * 동기화: shader.cc max_cta()에서 변경하므로 커널 실행 전/후에만 변경. */
  unsigned m_atom_sz;
  /* [한국어] 원자적 접근 단위 크기(바이트).
   * 설정자: init()에서 SECTOR 캐시이면 SECTOR_SIZE(32B), NORMAL이면 m_line_sz로 설정.
   * 읽는 자: mshr_addr() — MSHR 엔트리 키 정렬에 사용.
   * 값 범위: 32(SECTOR) 또는 m_line_sz(128B, NORMAL).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_sector_sz_log2;
  /* [한국어] SECTOR_SIZE(32B)의 log2 = 5 (고정값).
   * 설정자: init()에서 LOGB2(SECTOR_SIZE)로 계산.
   * 읽는 자: 섹터 인덱스 계산 등 섹터 캐시 관련 비트 연산.
   * 값 범위: 5 (고정).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned original_m_assoc;
  /* [한국어] 설정 파일에서 파싱된 원본 연상도 (동적 리사이징 전 값 보존용).
   * 설정자: init()에서 m_assoc와 동일 값으로 초기화; set_assoc()은 m_assoc만 변경.
   * 읽는 자: get_max_num_lines(), get_max_assoc(), l1d_cache_config::get_max_cache_multiplier().
   * 값 범위: init() 시점의 m_assoc 값.
   * 동기화: 초기화 후 읽기 전용. */
  bool m_is_streaming;
  /* [한국어] 스트리밍 캐시 모드 플래그 (STREAMING 설정 시 true).
   * 설정자: init()에서 ap=='s'이면 true로 설정.
   * 읽는 자: is_streaming() — shader.cc max_cta()에서 ON_MISS↔ON_FILL 재조정 여부 결정.
   * 값 범위: true/false.
   * 동기화: 초기화 후 읽기 전용. */

  enum replacement_policy_t m_replacement_policy;  // 'L' = LRU, 'F' = FIFO
  /* [한국어] 교체 정책 (LRU 또는 FIFO).
   * 설정자: init()에서 rp 문자로 설정.
   * 읽는 자: tag_array::access()에서 교체 후보 선택 시 사용.
   * 값 범위: LRU / FIFO.
   * 동기화: 초기화 후 읽기 전용. */
  enum write_policy_t
      m_write_policy;  // 'T' = write through, 'B' = write back, 'R' = read only
  /* [한국어] 쓰기 정책 (READ_ONLY/WRITE_BACK/WRITE_THROUGH/WRITE_EVICT/LOCAL_WB_GLOBAL_WT).
   * 설정자: init()에서 wp 문자로 설정.
   * 읽는 자: data_cache::init()에서 m_wr_hit 함수 포인터 선택; get_write_policy().
   * 값 범위: write_policy_t enum 5가지 값.
   * 동기화: 초기화 후 읽기 전용. */
  enum allocation_policy_t
      m_alloc_policy;  // 'm' = allocate on miss, 'f' = allocate on fill
  /* [한국어] 라인 할당 정책 (ON_MISS / ON_FILL).
   * 설정자: init()에서 ap 문자로 설정; set_allocation_policy()로 동적 변경 가능.
   * 읽는 자: tag_array::access() 및 baseline_cache::fill() — 할당 시점 결정.
   * 값 범위: ON_MISS / ON_FILL (STREAMING은 init()에서 ON_FILL로 변환됨).
   * 동기화: shader.cc max_cta()에서 커널 실행 전 변경 가능. */
  enum mshr_config_t m_mshr_type;
  /* [한국어] MSHR 구현 유형 (TEX_FIFO / ASSOC / SECTOR_TEX_FIFO / SECTOR_ASSOC).
   * 설정자: init()에서 mshr_type 문자로 설정.
   * 읽는 자: get_mshr_type() — baseline_cache::init()에서 ASSOC/SECTOR_ASSOC 여부 assert.
   * 값 범위: mshr_config_t enum 4가지 값.
   * 동기화: 초기화 후 읽기 전용. */
  enum cache_type m_cache_type;
  /* [한국어] 캐시 블록 관리 단위 (NORMAL=라인 전체 / SECTOR=32B 섹터).
   * 설정자: init()에서 ct 문자로 설정.
   * 읽는 자: tag_array 생성자에서 line_cache_block/sector_cache_block 선택; m_atom_sz 계산.
   * 값 범위: NORMAL(0) / SECTOR.
   * 동기화: 초기화 후 읽기 전용. */

  write_allocate_policy_t
      m_write_alloc_policy;  // 'W' = Write allocate, 'N' = No write allocate
  /* [한국어] 쓰기 미스 시 라인 할당 정책.
   * 설정자: init()에서 wap 문자로 설정.
   * 읽는 자: data_cache::init()에서 m_wr_miss 함수 포인터 선택; get_write_allocate_policy().
   * 값 범위: NO_WRITE_ALLOCATE / WRITE_ALLOCATE / FETCH_ON_WRITE / LAZY_FETCH_ON_READ.
   * 동기화: 초기화 후 읽기 전용. */

  union {
    unsigned m_mshr_entries;
    /* [한국어] MSHR 엔트리 수 (ASSOC/SECTOR_ASSOC 캐시용).
     * 설정자: init()에서 sscanf 파싱으로 설정 (11번째 토큰).
     * 읽는 자: baseline_cache 생성자의 m_mshrs(config.m_mshr_entries, ...) 초기화.
     * 값 범위: 양의 정수 (예: 32, 64, 128).
     * 동기화: 초기화 후 읽기 전용. */
    unsigned m_fragment_fifo_entries;
    /* [한국어] 텍스처 캐시 fragment FIFO 크기 (TEX_FIFO/SECTOR_TEX_FIFO용 별칭).
     * tex_cache 생성자의 m_fragment_fifo(config.m_fragment_fifo_entries) 초기화에 사용. */
  };
  union {
    unsigned m_mshr_max_merge;
    /* [한국어] MSHR 엔트리 하나에 병합할 수 있는 최대 요청 수.
     * 설정자: init()에서 sscanf 파싱 (12번째 토큰).
     * 읽는 자: baseline_cache 생성자의 m_mshrs(..., config.m_mshr_max_merge) 초기화.
     * 값 범위: 양의 정수 (예: 8, 10, 16).
     * 동기화: 초기화 후 읽기 전용. */
    unsigned m_request_fifo_entries;
    /* [한국어] 텍스처 캐시 request FIFO 크기 (TEX_FIFO용 별칭).
     * tex_cache 생성자의 m_request_fifo(config.m_request_fifo_entries) 초기화에 사용. */
  };
  union {
    unsigned m_miss_queue_size;
    /* [한국어] miss_queue(ICNT 방향 미스 요청 대기열) 최대 크기.
     * 설정자: init()에서 sscanf 파싱 (13번째 토큰).
     * 읽는 자: baseline_cache::miss_queue_full() — 새 미스 요청 수용 여부 판단.
     * 값 범위: 양의 정수 (예: 32, 64).
     * 동기화: 초기화 후 읽기 전용. */
    unsigned m_rob_entries;
    /* [한국어] 텍스처 캐시 Reorder Buffer 크기 (tex_cache용 별칭).
     * tex_cache 생성자의 m_rob(config.m_rob_entries) 초기화에 사용. */
  };
  unsigned m_result_fifo_entries;
  /* [한국어] 결과 FIFO 크기 (텍스처 캐시 m_result_fifo 또는 읽기 완료 결과 버퍼).
   * 설정자: init()에서 sscanf 파싱 (14번째 토큰).
   * 읽는 자: tex_cache 생성자의 m_result_fifo(config.m_result_fifo_entries) 초기화.
   * 값 범위: 양의 정수 (예: 4, 8).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_data_port_width;  //< number of byte the cache can access per cycle
  /* [한국어] 한 사이클에 캐시 데이터 배열에 접근할 수 있는 바이트 수.
   * 설정자: init()에서 15번째 토큰; 미지정(==0)이면 m_line_sz로 기본 설정.
   * 읽는 자: bandwidth_management — 데이터 포트 사용 사이클 계산.
   * 값 범위: m_line_sz의 약수 (assert 보장).
   * 동기화: 초기화 후 읽기 전용. */
  enum set_index_function
      m_set_index_function;  // Hash, linear, or custom set index function
  /* [한국어] 세트 인덱스 계산 함수 선택.
   * 설정자: init()에서 sif 문자로 설정; 생성자에서 LINEAR_SET_FUNCTION으로 초기화.
   * 읽는 자: cache_config::set_index() — 주소에서 세트 인덱스 계산 시 분기.
   * 값 범위: set_index_function enum 5가지 값.
   * 동기화: 초기화 후 읽기 전용. */

  friend class tag_array;
  friend class baseline_cache;
  friend class read_only_cache;
  friend class tex_cache;
  friend class data_cache;
  friend class l1_cache;
  friend class l2_cache;
  friend class memory_sub_partition;
};

/* [한국어] L1 데이터 캐시 전용 설정 클래스 (cache_config 파생).
 * SM(Streaming Multiprocessor) 내부의 L1 데이터 캐시에 특화된 추가 파라미터를 보유한다.
 * 주요 추가 사항:
 *   - l1_banks: L1 캐시 뱅크 수 (뱅크 인터리빙으로 동시 접근 처리량 향상)
 *   - m_unified_cache_size: Volta 통합 캐시 크기(KB) — L1/Shared 메모리 동적 분할 지원
 * get_max_cache_multiplier()가 오버라이드되어 통합 캐시 크기에 따른 최대 배율을 계산.
 * 전체 흐름: gpgpusim.config 파싱 → l1d_cache_config::init() → cache_config::init() */
class l1d_cache_config : public cache_config {
 public:
  /* [한국어] l1d_cache_config 기본 생성자 — cache_config() 위임, L1 전용 필드는 별도 초기화 필요 */
  l1d_cache_config() : cache_config() {}
  /* [한국어] set_bank - 주소에서 L1 캐시 뱅크 인덱스를 계산 (뱅크 인터리빙 기반).
   * l1_banks_hashing_function에 따라 LINEAR/BITWISE_XOR 해시로 뱅크를 선택.
   * shader.cc의 ldst_unit이 같은 뱅크로의 충돌(bank conflict) 여부를 이 함수로 판단. */
  unsigned set_bank(new_addr_type addr) const;
  /* [한국어] l1d_cache_config::init - L1 전용 필드(뱅크 로그값) 선계산 후 부모 init() 호출.
   * @config: gpgpusim.config L1 캐시 설정 문자열
   * @status: FuncCache 상태
   * l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving) 선계산으로
   * set_bank() 호출 시 log2 연산 없이 시프트만 수행. */
  void init(char *config, FuncCache status) {
    l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
    /* [한국어] 바이트 인터리빙 크기의 log2 계산 — set_bank()에서 인터리빙 오프셋 추출에 사용 */
    l1_banks_log2 = LOGB2(l1_banks); /* [한국어] 뱅크 수의 log2 — set_bank()에서 뱅크 인덱스 비트 수 */
    cache_config::init(config, status); /* [한국어] 공통 캐시 파라미터 파싱 — 부모 클래스 init() 위임 */
  }
  unsigned l1_latency;
  /* [한국어] L1 캐시 히트 레이턴시 (사이클 수).
   * 설정자: option_parser가 -gpgpu_l1_latency 옵션으로 설정.
   * 읽는 자: shader.cc ldst_unit — 히트 시 이 사이클 후 워프에 결과 공급.
   * 값 범위: 1~수십 사이클 (Pascal/Volta: 28~32 사이클 등).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned l1_banks;
  /* [한국어] L1 캐시 뱅크 수 (동시 접근 처리량 제어).
   * 설정자: option_parser가 -gpgpu_cache:dl1_banks 등으로 설정.
   * 읽는 자: set_bank() — 주소에서 뱅크 인덱스 계산; shader.cc bank conflict 감지.
   * 값 범위: 2의 거듭제곱 (1, 2, 4, 8 등).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned l1_banks_log2;
  /* [한국어] l1_banks의 log2 — set_bank()에서 비트 시프트 연산에 사용.
   * 설정자: l1d_cache_config::init()에서 LOGB2(l1_banks)로 계산.
   * 읽는 자: set_bank() 내부.
   * 값 범위: 0(banks=1) ~ 3(banks=8) 등.
   * 동기화: 초기화 후 읽기 전용. */
  unsigned l1_banks_byte_interleaving;
  /* [한국어] L1 뱅크 인터리빙 단위(바이트) — 연속 주소가 어떤 단위로 다른 뱅크에 분산되는지.
   * 설정자: option_parser가 설정.
   * 읽는 자: set_bank() — 인터리빙 오프셋으로 뱅크 인덱스 계산.
   * 값 범위: 2의 거듭제곱 (4, 8, 16, 32 등).
   * 동기화: 초기화 후 읽기 전용. */
  unsigned l1_banks_byte_interleaving_log2;
  /* [한국어] l1_banks_byte_interleaving의 log2.
   * 설정자: l1d_cache_config::init()에서 LOGB2(l1_banks_byte_interleaving)로 계산.
   * 읽는 자: set_bank() 내부 인터리빙 비트 추출.
   * 값 범위: 2(4B) ~ 5(32B) 등.
   * 동기화: 초기화 후 읽기 전용. */
  unsigned l1_banks_hashing_function;
  /* [한국어] L1 뱅크 선택에 사용할 해시 함수 종류 (set_bank()에서 분기 기준).
   * 설정자: option_parser가 설정.
   * 읽는 자: set_bank() — 0이면 LINEAR, 그 외 BITWISE_XOR 등 선택.
   * 값 범위: 0(LINEAR) 또는 기타 정수.
   * 동기화: 초기화 후 읽기 전용. */
  unsigned m_unified_cache_size;
  /* [한국어] Volta 통합 캐시 총 크기(KB) — L1 데이터 캐시와 Shared 메모리가 공유하는 SRAM 용량.
   * 설정자: option_parser가 -gpgpu_unified_cache_size로 설정.
   * 읽는 자: get_max_cache_multiplier() — L1 최대 크기 배율 계산에 사용.
   * 값 범위: 0(미사용, 기본 배율 4 적용) 또는 KB 단위 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */
  /*
   * [한국어]
   * l1d_cache_config::get_max_cache_multiplier - Volta 통합 캐시 기반 최대 크기 배율 반환
   *
   * @return: m_unified_cache_size가 설정됐으면 통합 캐시 크기 / 원본 L1 크기, 아니면 4
   *
   * Volta 아키텍처에서 L1 데이터 캐시와 Shared 메모리는 동일한 SRAM을 공유한다.
   * shader.cc max_cta()가 커널별 Shared 메모리 사용량을 파악하여 L1에 남은 용량을 계산하고,
   * set_assoc()로 연상도를 조정한 후 이 함수로 최대 배율을 확인한다.
   * 예시: 통합 캐시=128KB, 원본 L1=32KB → 배율=4 (최대 4배로 확장 가능).
   *
   * 호출 체인: get_max_num_lines() / get_max_assoc() → get_max_cache_multiplier()
   */
  virtual unsigned get_max_cache_multiplier() const {
    // set * assoc * cacheline size. Then convert Byte to KB
    // gpgpu_unified_cache_size is in KB while original_sz is in B
    if (m_unified_cache_size > 0) {
      /* [한국어] 통합 캐시 크기가 설정된 경우: 원본 L1 크기 계산 후 배율 반환 */
      unsigned original_size = m_nset * original_m_assoc * m_line_sz / 1024;
      /* [한국어] 원본 L1 크기(KB) = 세트 수 × 원본 연상도 × 라인 크기(B) / 1024 */
      assert(m_unified_cache_size % original_size == 0);
      /* [한국어] 통합 캐시가 원본 크기의 정수 배여야 함 — 정수 연상도 조정 가능하도록 */
      return m_unified_cache_size / original_size; /* [한국어] 최대 배율 반환 */
    } else {
      return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER; /* [한국어] 통합 캐시 미설정 시 기본 배율 4 반환 */
    }
  }
};

/* [한국어] L2 캐시 전용 설정 클래스 (cache_config 파생).
 * 메모리 서브파티션(memory_sub_partition)의 L2 캐시에 특화된 설정을 보유한다.
 * 주요 추가 사항:
 *   - m_address_mapping: 선형 주소를 실제 DRAM 물리 주소로 변환하는 매핑 객체
 * set_index()가 오버라이드되어 L2 파티션 주소 공간을 고려한 세트 인덱스를 계산한다.
 * 전체 흐름: memory_sub_partition 초기화 → l2_cache_config::init(address_mapping) → set_index() */
class l2_cache_config : public cache_config {
 public:
  /* [한국어] l2_cache_config 기본 생성자 — cache_config() 위임 */
  l2_cache_config() : cache_config() {}
  /* [한국어] l2_cache_config::init - 주소 매핑 객체를 저장하고 L2용 설정 초기화.
   * @address_mapping: 선형→물리 주소 변환 객체 (addrdec.h의 linear_to_raw_address_translation).
   *   L2 세트 인덱스 계산 시 파티션 경계를 고려한 주소 변환에 사용됨. */
  void init(linear_to_raw_address_translation *address_mapping);
  /* [한국어] l2_cache_config::set_index - L2용 오버라이드 세트 인덱스 계산.
   * 주소 매핑(m_address_mapping)으로 변환된 물리 주소를 기반으로 세트 인덱스를 계산.
   * 여러 L2 파티션이 존재하므로 파티션 내 오프셋을 고려하여 계산. */
  virtual unsigned set_index(new_addr_type addr) const;

 private:
  linear_to_raw_address_translation *m_address_mapping;
  /* [한국어] 선형 주소 → DRAM 물리 주소 변환 객체 포인터.
   * 설정자: l2_cache_config::init()에서 인자로 받아 저장.
   * 읽는 자: set_index()에서 주소 변환 후 세트 인덱스 계산.
   * 값 범위: 유효한 linear_to_raw_address_translation 포인터 (NULL 불가).
   * 동기화: 초기화 후 읽기 전용. */
};

/* [한국어] 세트-연상(set-associative) 태그 배열 — 캐시의 핵심 자료구조.
 * probe()/access()/fill()을 통해 캐시 히트/미스 판별과 라인 상태 전이를 수행한다.
 * 내부적으로 cache_block_t* 배열(m_lines)을 관리하며, 설정에 따라
 * line_cache_block(NORMAL) 또는 sector_cache_block(SECTOR) 객체를 생성한다.
 * baseline_cache가 tag_array를 소유하고(new로 생성), fill()/cycle()에서 접근한다.
 * 사이클-레벨 타이밍 모델의 캐시 상태를 정확히 추적하는 핵심 모듈.
 * 호출 체인: baseline_cache::access() → tag_array::probe() → (miss) → tag_array::access()
 *            → (응답 도착) → tag_array::fill() */
class tag_array {
 public:
  /* [한국어] tag_array 주 생성자 — m_lines 배열을 동적으로 할당하고 init() 호출.
   * @config: 캐시 설정 (m_nset, m_assoc, m_cache_type 등)
   * @core_id: 이 캐시를 소유한 SM 인덱스 (통계 구분용)
   * @type_id: 캐시 유형 인덱스 (normal/texture/constant 구분) */
  // Use this constructor
  tag_array(cache_config &config, int core_id, int type_id);
  /* [한국어] tag_array 소멸자 — m_lines 배열과 각 cache_block_t 객체 메모리 해제 */
  ~tag_array();

  /* [한국어] probe (mem_fetch 오버로드) - 읽기/쓰기 접근 전 태그 배열 탐색 (부작용 없음).
   * @addr: 접근할 주소
   * @idx: [출력] 히트 또는 교체 후보 라인 인덱스
   * @mf: 접근 요청 mem_fetch (sector_mask, is_write 정보 포함)
   * @is_write: 쓰기 접근 여부
   * @probe_mode: true이면 통계 업데이트 없이 순수 탐색만 수행 (예비 확인용)
   * @return: HIT/HIT_RESERVED/MISS/RESERVATION_FAIL/SECTOR_MISS */
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_fetch *mf, bool is_write,
                                  bool probe_mode = false) const;
  /* [한국어] probe (sector_mask 오버로드) - 섹터 마스크를 직접 지정하는 탐색.
   * data_cache가 특정 섹터에 대한 탐색을 요청할 때 사용.
   * @mask: 탐색할 섹터 마스크 (4비트 bitset)
   * 나머지 인자는 위 오버로드와 동일. */
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_access_sector_mask_t mask, bool is_write,
                                  bool probe_mode = false,
                                  mem_fetch *mf = NULL) const;
  /* [한국어] access (2출력 오버로드) - 태그 탐색 + 히트/미스 통계 업데이트만 수행 (write-back 없음).
   * @addr: 접근 주소, @time: 현재 사이클 (LRU 갱신), @idx: [출력] 라인 인덱스, @mf: 요청
   * read_only_cache에서 사용 — dirty 라인 축출 불필요 */
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, mem_fetch *mf);
  /* [한국어] access (4출력 오버로드) - 태그 탐색 + dirty 라인 축출 정보까지 반환.
   * @wb: [출력] true이면 교체된 라인이 dirty — write-back 요청 발행 필요
   * @evicted: [출력] 축출된 블록의 주소/크기/마스크 정보
   * data_cache에서 write-back이 필요할 수 있을 때 사용. */
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, bool &wb,
                                   evicted_block_info &evicted, mem_fetch *mf);

  /* [한국어] fill (mem_fetch 오버로드) - 하위 메모리 응답 도착 시 라인 상태 VALID/MODIFIED로 전이.
   * @addr: fill할 블록 주소, @time: 현재 사이클, @mf: fill을 트리거한 원본 요청, @is_write: 쓰기 fill 여부 */
  void fill(new_addr_type addr, unsigned time, mem_fetch *mf, bool is_write);
  /* [한국어] fill (인덱스 오버로드) - 이미 알려진 캐시 라인 인덱스로 직접 fill.
   * extra_mf_fields_lookup에서 m_cache_index를 찾아 직접 전달할 때 사용. */
  void fill(unsigned idx, unsigned time, mem_fetch *mf);
  /* [한국어] fill (sector_mask 오버로드) - 특정 섹터만 fill (SECTOR 캐시용).
   * @mask: fill할 섹터 마스크, @byte_mask: 더티 바이트 마스크, @is_write: 쓰기 여부 */
  void fill(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask,
            mem_access_byte_mask_t byte_mask, bool is_write);

  /* [한국어] size - 총 캐시 라인 수 반환 (m_config.get_num_lines() = m_nset × m_assoc) */
  unsigned size() const { return m_config.get_num_lines(); }
  /* [한국어] get_block - 인덱스로 특정 캐시 블록 포인터 반환 (force_tag_access 등에서 직접 접근) */
  cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

  /* [한국어] flush - MODIFIED 상태인 모든 라인을 INVALID로 초기화 (커널 종료 시 dirty 제거) */
  void flush();       // flush all written entries
  /* [한국어] invalidate - 모든 라인을 INVALID로 강제 전환 (캐시 전체 무효화, flush보다 강력) */
  void invalidate();  // invalidate all entries
  /* [한국어] new_window - AerialVision 통계 윈도우 갱신: m_prev_snapshot_* 필드를 현재 값으로 스냅샷 */
  void new_window();

  /* [한국어] print - 캐시 접근/미스 통계를 stream에 출력하고 total_access/total_misses에 누계 */
  void print(FILE *stream, unsigned &total_access,
             unsigned &total_misses) const;
  /* [한국어] windowed_miss_rate - 마지막 new_window() 이후의 미스율 반환 (AerialVision용) */
  float windowed_miss_rate() const;
  /* [한국어] get_stats - 누적 접근/미스/pending_hit/RESERVATION_FAIL 통계를 출력 인자로 반환 */
  void get_stats(unsigned &total_access, unsigned &total_misses,
                 unsigned &total_hit_res, unsigned &total_res_fail) const;

  /* [한국어] update_cache_parameters - 동적 리사이징(Volta L1) 후 설정 갱신 (m_config 참조 업데이트) */
  void update_cache_parameters(cache_config &config);
  /* [한국어] add_pending_line - 이 블록 주소에 대한 pending 요청 카운터 증가 (sector 캐시용) */
  void add_pending_line(mem_fetch *mf);
  /* [한국어] remove_pending_line - pending 요청 카운터 감소 및 완료 시 제거 */
  void remove_pending_line(mem_fetch *mf);
  /* [한국어] inc_dirty - 더티 라인 카운터 증가 (write hit 시 호출) */
  void inc_dirty() { m_dirty++; }

 protected:
  // This constructor is intended for use only from derived classes that wish to
  // avoid unnecessary memory allocation that takes place in the
  // other tag_array constructor
  /* [한국어] tag_array 파생 클래스용 생성자 — 외부에서 할당된 new_lines 배열을 받아 사용.
   * tex_cache처럼 tag_array를 직접 소유하는 클래스에서 m_lines 메모리를 직접 관리할 때 사용.
   * @new_lines: 외부에서 생성된 cache_block_t* 배열 (이 생성자는 메모리 할당 안 함) */
  tag_array(cache_config &config, int core_id, int type_id,
            cache_block_t **new_lines);
  /* [한국어] init - 통계 카운터 초기화 및 m_core_id/m_type_id 설정 (두 생성자 공통 로직) */
  void init(int core_id, int type_id);

 protected:
  cache_config &m_config;
  /* [한국어] 이 태그 배열의 캐시 설정 참조.
   * 설정자: 생성자에서 config 인자로 바인딩 (참조이므로 동적 변경이 m_config에 반영됨).
   * 읽는 자: probe()/access()/fill() 전반 — m_nset, m_assoc, m_line_sz 등 참조.
   * 값 범위: 유효한 cache_config 인스턴스 참조.
   * 동기화: 단일 SM 컨텍스트에서만 접근. */

  cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */
  /* [한국어] 세트-연상 캐시 라인 배열 (크기: m_nset × m_assoc = get_num_lines()).
   * 설정자: 주 생성자에서 new line_cache_block[] 또는 new sector_cache_block[]로 동적 할당.
   * 읽는 자: probe()/access()/fill()/flush()/invalidate() — 모든 캐시 라인 접근.
   * 값 범위: 유효한 cache_block_t* 포인터 배열; 소멸자에서 해제.
   * 동기화: 단일 SM 컨텍스트에서만 접근, 별도 락 불필요. */

  unsigned m_access;
  /* [한국어] 이 태그 배열에 대한 총 접근 횟수 (probe/access 호출 수).
   * 설정자: access()가 호출될 때마다 증가.
   * 읽는 자: get_stats(), print(), windowed_miss_rate() 계산.
   * 값 범위: 0 ~ 전체 시뮬레이션 사이클 수.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_miss;
  /* [한국어] 캐시 미스 횟수 (MISS 또는 RESERVATION_FAIL 반환 횟수).
   * 설정자: access()에서 MISS 결과 시 증가.
   * 읽는 자: get_stats(), print(), windowed_miss_rate().
   * 값 범위: 0 ~ m_access.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_pending_hit;  // number of cache miss that hit a line that is
                           // allocated but not filled
  /* [한국어] HIT_RESERVED 횟수 — 라인이 RESERVED 상태일 때의 히트 수 (pending hit).
   * 설정자: access()에서 HIT_RESERVED 결과 시 증가.
   * 읽는 자: get_stats() — 전체 미스 처리 통계에서 MSHR 병합 효율 분석.
   * 값 범위: 0 ~ m_access.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_res_fail;
  /* [한국어] RESERVATION_FAIL 횟수 — MSHR/miss_queue 포화로 요청 거부된 횟수.
   * 설정자: access()에서 RESERVATION_FAIL 결과 시 증가.
   * 읽는 자: get_stats() — 캐시 병목 분석.
   * 값 범위: 0 ~ m_access.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_sector_miss;
  /* [한국어] SECTOR_MISS 횟수 — 라인은 히트이지만 섹터가 없는 경우.
   * 설정자: access()에서 SECTOR_MISS 결과 시 증가.
   * 읽는 자: get_stats() — 섹터 캐시의 부분 히트 효율 분석.
   * 값 범위: 0 ~ m_access (SECTOR 캐시에서만 의미 있음).
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_dirty;
  /* [한국어] 현재 MODIFIED 상태(더티)인 캐시 라인 수.
   * 설정자: inc_dirty()로 쓰기 히트 시 증가; flush()/fill() 등에서 감소.
   * 읽는 자: 통계 수집 — 더티 라인 비율 모니터링.
   * 값 범위: 0 ~ get_num_lines().
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  // performance counters for calculating the amount of misses within a time
  // window
  unsigned m_prev_snapshot_access;
  /* [한국어] 마지막 new_window() 호출 시점의 m_access 스냅샷.
   * 설정자: new_window()에서 현재 m_access 값으로 업데이트.
   * 읽는 자: windowed_miss_rate()에서 윈도우 내 증분 계산.
   * 값 범위: 0 ~ m_access.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_prev_snapshot_miss;
  /* [한국어] 마지막 new_window() 시점의 m_miss 스냅샷.
   * 설정자: new_window()에서 업데이트.
   * 읽는 자: windowed_miss_rate() — 윈도우 미스율 분자.
   * 값 범위: 0 ~ m_miss.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned m_prev_snapshot_pending_hit;
  /* [한국어] 마지막 new_window() 시점의 m_pending_hit 스냅샷.
   * 설정자: new_window()에서 업데이트.
   * 읽는 자: windowed_miss_rate() — 윈도우 내 pending hit 증분.
   * 값 범위: 0 ~ m_pending_hit.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  int m_core_id;  // which shader core is using this
  /* [한국어] 이 태그 배열을 소유한 SM(Shader Core) 인덱스.
   * 설정자: init()에서 생성자 인자 core_id로 설정.
   * 읽는 자: 통계 출력 시 어느 SM의 캐시 통계인지 구분.
   * 값 범위: 0 ~ (SM 수 - 1). -1이면 L2 등 SM 비귀속 캐시.
   * 동기화: 초기화 후 읽기 전용. */
  int m_type_id;  // what kind of cache is this (normal, texture, constant)
  /* [한국어] 캐시 유형 구분자 (0=일반 데이터, 1=텍스처, 2=상수 등).
   * 설정자: init()에서 생성자 인자 type_id로 설정.
   * 읽는 자: 통계 출력 시 캐시 유형 구분.
   * 값 범위: 정수 (유형별 convention은 shader.cc 참조).
   * 동기화: 초기화 후 읽기 전용. */

  bool is_used;  // a flag if the whole cache has ever been accessed before
  /* [한국어] 이 캐시가 시뮬레이션 시작 후 한 번이라도 접근됐는지 여부.
   * 설정자: access()에서 첫 접근 시 true로 설정.
   * 읽는 자: 통계 출력 시 사용 여부 확인 (미사용 캐시 통계 스킵).
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  typedef tr1_hash_map<new_addr_type, unsigned> line_table;
  /* [한국어] 블록 주소 → pending 요청 수 매핑 타입 (SECTOR 캐시용 섹터 응답 추적).
   * 같은 블록 주소의 여러 섹터 요청이 독립적으로 발행될 때 모든 섹터가 도착했는지 추적. */
  line_table pending_lines;
  /* [한국어] 블록 주소별 아직 도착하지 않은 섹터 요청 수를 추적하는 테이블.
   * 설정자: add_pending_line() — 섹터 요청 발행 시 카운터 증가.
   * 읽는 자: remove_pending_line() — 섹터 응답 도착 시 감소, 0이 되면 엔트리 제거.
   * 값 범위: 블록 주소 → (1 ~ SECTOR_CHUNCK_SIZE).
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
};

/* [한국어] MSHR (Miss Status Holding Register) 테이블.
 * 캐시 미스 발생 시 같은 캐시라인에 대한 중복 요청을 하나의 엔트리에 병합(merge)하여
 * 하위 메모리로의 중복 요청을 방지하고, fill 완료 시 병합된 모든 요청을 함께 처리한다.
 * 완전 연상(fully associative) 구조로, 최대 m_num_entries개의 서로 다른 블록 주소를 추적한다.
 * fill 응답이 도착하면 mark_ready()로 엔트리를 준비 완료 표시하고,
 * next_access()로 병합된 요청들을 순서대로 꺼내 처리한다.
 * 호출 체인: baseline_cache::send_read_request() → mshr_table::add()
 *            → (fill 도착) → mark_ready() → next_access() */
class mshr_table {
 public:
  /*
   * [한국어]
   * mshr_table 생성자 — MSHR 엔트리 수와 최대 병합 수 초기화
   *
   * @num_entries: MSHR 최대 엔트리 수 (서로 다른 블록 주소 추적 한계)
   * @max_merged: 엔트리 하나에 병합할 수 있는 최대 요청 수
   *
   * tr1_hash_map이 std::map이 아닌 경우 해시맵 버킷을 2*num_entries로 초기화하여
   * 해시 충돌을 최소화한다.
   * 호출 체인: baseline_cache 생성자 → m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge)
   */
  mshr_table(unsigned num_entries, unsigned max_merged)
      : m_num_entries(num_entries),
        m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
        ,
        m_data(2 * num_entries) /* [한국어] 해시맵 버킷 수를 엔트리 수의 2배로 설정 — 충돌 최소화 */
#endif
  {
  }

  /* [한국어] probe - 이 블록 주소에 대한 진행 중인 MSHR 엔트리가 있는지 확인.
   * @block_addr: 확인할 캐시라인 베이스 주소
   * @return: true이면 이미 동일 블록에 대한 요청이 MSHR에 등록되어 있음 (MSHR_HIT 가능)
   * 호출: baseline_cache::send_read_request() — MSHR_HIT 처리 전 확인 */
  /// Checks if there is a pending request to the lower memory level already
  bool probe(new_addr_type block_addr) const;
  /* [한국어] full - MSHR에 새 엔트리 추가 또는 기존 엔트리에 병합이 가능한지 확인.
   * @block_addr: 추가하려는 블록 주소
   * @return: true이면 MSHR 포화 (MSHR_ENRTY_FAIL 또는 MSHR_MERGE_ENRTY_FAIL 발생)
   * 이미 이 블록의 엔트리가 있으면 m_max_merged 미만인지, 없으면 m_num_entries 미만인지 검사 */
  /// Checks if there is space for tracking a new memory access
  bool full(new_addr_type block_addr) const;
  /* [한국어] add - 새 요청을 MSHR에 추가하거나 기존 엔트리에 병합.
   * @block_addr: 캐시라인 베이스 주소
   * @mf: 추가할 mem_fetch (원자 연산 여부 m_has_atomic 업데이트에도 사용)
   * full() 확인 후 호출해야 함 — full() 상태에서 호출 시 assert 실패 */
  /// Add or merge this access
  void add(new_addr_type block_addr, mem_fetch *mf);
  /* [한국어] busy - MSHR이 새 fill 응답을 받을 수 없는 상태인지 반환.
   * 현재 구현은 항상 false를 반환 — 실제 바쁨 판정은 미구현 */
  /// Returns true if cannot accept new fill responses
  bool busy() const { return false; }
  /* [한국어] mark_ready - fill 응답 도착 시 해당 블록 주소 엔트리를 처리 준비 완료로 표시.
   * @block_addr: fill이 완료된 블록 주소
   * @has_atomic: [출력] 이 엔트리에 원자 연산이 포함됐는지 여부
   * m_current_response 리스트에 block_addr를 추가하여 next_access()가 꺼낼 수 있게 함 */
  /// Accept a new cache fill response: mark entry ready for processing
  void mark_ready(new_addr_type block_addr, bool &has_atomic);
  /* [한국어] access_ready - 처리 준비된 완료 요청이 있는지 반환.
   * m_current_response 리스트가 비어있지 않으면 true — baseline_cache::cycle()에서 확인 */
  /// Returns true if ready accesses exist
  bool access_ready() const { return !m_current_response.empty(); }
  /* [한국어] next_access - 처리 준비된 다음 mem_fetch를 반환하고 MSHR에서 제거.
   * m_current_response에서 block_addr를 꺼내고, m_data에서 해당 엔트리의 첫 mem_fetch를 pop.
   * 엔트리의 모든 요청이 처리되면 m_data에서 엔트리 완전 제거 */
  /// Returns next ready access
  mem_fetch *next_access();
  /* [한국어] display - 현재 MSHR 상태를 fp에 출력 (디버그/dump용) */
  void display(FILE *fp) const;
  /* [한국어] is_read_after_write_pending - 이 블록에 쓰기 요청 후 읽기 요청이 대기 중인지 확인.
   * @block_addr: 확인할 블록 주소
   * @return: true이면 RAW(Read-After-Write) 위험이 있음 — 쓰기 완료 전 읽기 차단 필요
   * LAZY_FETCH_ON_READ 정책에서 쓰기 후 읽기 시 교착 방지에 사용 */
  // Returns true if there is a pending read after write
  bool is_read_after_write_pending(new_addr_type block_addr);

  /* [한국어] check_mshr_parameters - 커널 간 MSHR 파라미터 변경 여부 검증.
   * @num_entries: 확인할 MSHR 엔트리 수
   * @max_merged: 확인할 최대 병합 수
   * GPGPU-Sim은 커널 간 MSHR 설정 변경을 허용하지 않음 (assert로 강제) */
  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(m_num_entries == num_entries &&
           "Change of MSHR parameters between kernels is not allowed");
    /* [한국어] MSHR 엔트리 수 변경 시도 감지 — 허용되지 않음 */
    assert(m_max_merged == max_merged &&
           "Change of MSHR parameters between kernels is not allowed");
    /* [한국어] 최대 병합 수 변경 시도 감지 — 허용되지 않음 */
  }

 private:
  // finite sized, fully associative table, with a finite maximum number of
  // merged requests
  /* [한국어] MSHR 구현: 유한 크기 완전 연상 테이블, 병합 수 제한 있음 */
  const unsigned m_num_entries;
  /* [한국어] MSHR 최대 엔트리 수 (서로 다른 블록 주소 동시 추적 한계).
   * 설정자: 생성자에서 config.m_mshr_entries로 초기화; const이므로 이후 변경 불가.
   * 읽는 자: full() — 새 엔트리 추가 가능 여부 판단.
   * 값 범위: 양의 정수 (예: 32, 64, 128).
   * 동기화: 생성자 이후 불변. */
  const unsigned m_max_merged;
  /* [한국어] 하나의 MSHR 엔트리에 병합할 수 있는 최대 요청 수.
   * 설정자: 생성자에서 config.m_mshr_max_merge로 초기화; const이므로 이후 변경 불가.
   * 읽는 자: full() — 기존 엔트리 병합 한계 초과 여부 판단.
   * 값 범위: 양의 정수 (예: 8, 10, 16).
   * 동기화: 생성자 이후 불변. */

  struct mshr_entry {
    /* [한국어] MSHR 엔트리 하나를 표현하는 내부 구조체.
     * 같은 블록 주소로 향하는 요청들의 리스트와 원자 연산 포함 여부를 보유한다.
     * m_list: 동일 블록 주소에 병합된 mem_fetch 포인터 리스트.
     *   설정자: add()에서 mf를 push_back; next_access()에서 pop_front.
     *   읽는 자: next_access(), mark_ready() — 처리 순서 보장을 위한 FIFO.
     * m_has_atomic: 이 엔트리에 원자 연산(LD/ST atomic) 요청이 포함됐는지.
     *   설정자: add()에서 mf->is_atomic() 확인 후 true로 설정.
     *   읽는 자: mark_ready()에서 has_atomic 출력 인자로 반환 → shader.cc에서 원자 연산 완료 처리. */
    std::list<mem_fetch *> m_list;  /* [한국어] 병합된 mem_fetch 요청 리스트 (FIFO 순서) */
    bool m_has_atomic;              /* [한국어] 원자 연산 포함 여부 — fill 완료 시 원자 처리 경로 분기 */
    mshr_entry() : m_has_atomic(false) {} /* [한국어] 기본 생성자 — 원자 연산 없음으로 초기화 */
  };
  typedef tr1_hash_map<new_addr_type, mshr_entry> table;      /* [한국어] 블록 주소 → mshr_entry 해시맵 타입 */
  typedef tr1_hash_map<new_addr_type, mshr_entry> line_table; /* [한국어] 라인 단위 pending 추적용 해시맵 타입 (sector 캐시용) */
  table m_data;
  /* [한국어] MSHR 엔트리 해시맵: 블록 주소 → mshr_entry.
   * 설정자: add()에서 새 엔트리 삽입 또는 기존 엔트리 m_list에 append.
   * 읽는 자: probe()/full() — 엔트리 존재 및 크기 확인; next_access() — 요청 꺼내기.
   * 값 범위: 최대 m_num_entries개 엔트리; 각 엔트리는 최대 m_max_merged개 요청.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  line_table pending_lines;
  /* [한국어] sector 캐시에서 라인 단위 pending 요청 추적 (현재 활용 미확인, 확장 목적).
   * 설정자/읽는 자: 현재 mshr_table 내부에서 직접 사용되지 않을 수 있음.
   * 동기화: 단일 SM 컨텍스트. */

  // it may take several cycles to process the merged requests
  bool m_current_response_ready;
  /* [한국어] 현재 처리 준비된 응답이 있는지 플래그 (access_ready()의 보조).
   * 설정자: mark_ready()에서 true로 설정.
   * 읽는 자: 현재 access_ready()는 m_current_response.empty()를 사용하므로 이 필드는 보조.
   * 값 범위: true/false.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  std::list<new_addr_type> m_current_response;
  /* [한국어] fill 완료 후 처리 대기 중인 블록 주소 리스트.
   * 설정자: mark_ready()에서 block_addr를 push_back.
   * 읽는 자: access_ready() — 비어있지 않으면 처리 준비 완료; next_access() — front를 pop하여 처리.
   * 값 범위: 처리 대기 중인 블록 주소 0~N개.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
};

/***************************************************************** Caches
 * *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and
/// reservation fails.
///
/* [한국어] 캐시 접근 통계를 집계하는 간단한 구조체.
 * cache_stats::get_sub_stats()가 이 구조체에 통계를 채우고,
 * gpu-sim.cc의 통계 출력 함수들이 이 값을 참조하여 캐시 효율을 보고한다.
 * cache_sub_stats_pw와 달리 read/write 구분 없이 통합 집계한다.
 * 사용 패턴: cache_sub_stats css; cache.get_sub_stats(css); // css에 통계 채움 */
struct cache_sub_stats {
  unsigned long long accesses;
  /* [한국어] 총 캐시 접근 횟수 (HIT + HIT_RESERVED + MISS + RESERVATION_FAIL).
   * 설정자: cache_stats::get_sub_stats()에서 m_stats 배열을 합산하여 채움.
   * 읽는 자: gpu-sim.cc 통계 출력, 미스율 계산 (misses / accesses).
   * 값 범위: 0 ~ 전체 시뮬레이션 접근 수.
   * 동기화: 통계 출력 단계에서만 읽음. */
  unsigned long long misses;
  /* [한국어] 총 캐시 미스 횟수 (실제 메모리 요청을 발생시킨 접근 수).
   * 설정자: cache_stats::get_sub_stats()에서 MISS 상태 통계 합산.
   * 읽는 자: 미스율 계산 (misses / accesses), 성능 분석 보고.
   * 값 범위: 0 ~ accesses.
   * 동기화: 통계 출력 단계에서만 읽음. */
  unsigned long long pending_hits;
  /* [한국어] HIT_RESERVED(pending hit) 횟수 — MSHR에 진행 중인 요청과 같은 라인 hit.
   * 설정자: cache_stats::get_sub_stats()에서 HIT_RESERVED 상태 통계 합산.
   * 읽는 자: MSHR 효율(병합 히트율) 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 통계 출력 단계에서만 읽음. */
  unsigned long long res_fails;
  /* [한국어] RESERVATION_FAIL 횟수 — MSHR/miss_queue 포화로 요청이 거부된 횟수.
   * 설정자: cache_stats::get_sub_stats()에서 RESERVATION_FAIL 상태 통계 합산.
   * 읽는 자: 캐시 병목(backpressure) 분석 — 이 값이 크면 MSHR 확장 필요.
   * 값 범위: 0 ~ accesses.
   * 동기화: 통계 출력 단계에서만 읽음. */

  unsigned long long port_available_cycles;
  /* [한국어] 데이터 포트가 사용 가능했던 총 사이클 수 (포트 활용률 분모).
   * 설정자: bandwidth_management::replenish_port_bandwidth()에서 사이클마다 증가.
   * 읽는 자: print_port_stats() — 포트 활용률 = data_port_busy_cycles / port_available_cycles.
   * 값 범위: 0 ~ 전체 시뮬레이션 사이클 수.
   * 동기화: 통계 출력 단계에서만 읽음. */
  unsigned long long data_port_busy_cycles;
  /* [한국어] 데이터 포트(캐시 배열 읽기/쓰기 포트)가 사용 중이었던 사이클 수.
   * 설정자: bandwidth_management::use_data_port()에서 접근 시 업데이트.
   * 읽는 자: print_port_stats() — 데이터 포트 포화도 계산.
   * 값 범위: 0 ~ port_available_cycles.
   * 동기화: 통계 출력 단계에서만 읽음. */
  unsigned long long fill_port_busy_cycles;
  /* [한국어] fill 포트(하위 메모리 응답 수신 포트)가 사용 중이었던 사이클 수.
   * 설정자: bandwidth_management::use_fill_port()에서 fill 처리 시 업데이트.
   * 읽는 자: print_port_stats() — fill 포트 포화도 계산.
   * 값 범위: 0 ~ port_available_cycles.
   * 동기화: 통계 출력 단계에서만 읽음. */

  /* [한국어] cache_sub_stats 기본 생성자 — clear()로 모든 카운터 0 초기화 */
  cache_sub_stats() { clear(); }
  /*
   * [한국어]
   * clear - 모든 통계 카운터를 0으로 초기화
   *
   * AerialVision 윈도우 초기화 또는 통계 집계 전 클리어에 사용.
   * 호출 체인: cache_sub_stats() → clear() / clear_pw()에서 직접 호출
   */
  void clear() {
    accesses = 0;            /* [한국어] 총 접근 수 초기화 */
    misses = 0;              /* [한국어] 총 미스 수 초기화 */
    pending_hits = 0;        /* [한국어] HIT_RESERVED 수 초기화 */
    res_fails = 0;           /* [한국어] RESERVATION_FAIL 수 초기화 */
    port_available_cycles = 0;  /* [한국어] 포트 사용 가능 사이클 초기화 */
    data_port_busy_cycles = 0;  /* [한국어] 데이터 포트 사용 사이클 초기화 */
    fill_port_busy_cycles = 0;  /* [한국어] fill 포트 사용 사이클 초기화 */
  }
  /*
   * [한국어]
   * operator+= - 다른 cache_sub_stats를 현재 통계에 누적 합산
   *
   * @css: 합산할 통계 구조체
   * @return: 누적된 this 참조
   *
   * 여러 캐시 인스턴스(SM별 L1, L2 파티션 등)의 통계를 단일 구조체에 누계할 때 사용.
   * 호출 체인: gpu-sim.cc 통계 집계 루프에서 각 SM/파티션 통계 합산
   */
  cache_sub_stats &operator+=(const cache_sub_stats &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;                         /* [한국어] 접근 수 누적 */
    misses += css.misses;                             /* [한국어] 미스 수 누적 */
    pending_hits += css.pending_hits;                 /* [한국어] pending hit 수 누적 */
    res_fails += css.res_fails;                       /* [한국어] RESERVATION_FAIL 수 누적 */
    port_available_cycles += css.port_available_cycles; /* [한국어] 포트 사용 가능 사이클 누적 */
    data_port_busy_cycles += css.data_port_busy_cycles; /* [한국어] 데이터 포트 사용 사이클 누적 */
    fill_port_busy_cycles += css.fill_port_busy_cycles; /* [한국어] fill 포트 사용 사이클 누적 */
    return *this; /* [한국어] 누적된 자신 반환 */
  }

  /*
   * [한국어]
   * operator+ - 두 cache_sub_stats의 합을 새 객체로 반환
   *
   * @cs: 합산할 통계 구조체
   * @return: 두 통계의 합을 담은 새 cache_sub_stats 객체
   *
   * 두 캐시의 통계를 새 변수에 합산할 때 사용 (예: L1 + L2 통계 결합).
   */
  cache_sub_stats operator+(const cache_sub_stats &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats ret;                               /* [한국어] 결과를 담을 새 통계 구조체 생성 */
    ret.accesses = accesses + cs.accesses;             /* [한국어] 접근 수 합산 */
    ret.misses = misses + cs.misses;                   /* [한국어] 미스 수 합산 */
    ret.pending_hits = pending_hits + cs.pending_hits; /* [한국어] pending hit 합산 */
    ret.res_fails = res_fails + cs.res_fails;          /* [한국어] RESERVATION_FAIL 합산 */
    ret.port_available_cycles =
        port_available_cycles + cs.port_available_cycles; /* [한국어] 포트 사용 가능 사이클 합산 */
    ret.data_port_busy_cycles =
        data_port_busy_cycles + cs.data_port_busy_cycles; /* [한국어] 데이터 포트 사용 사이클 합산 */
    ret.fill_port_busy_cycles =
        fill_port_busy_cycles + cs.fill_port_busy_cycles; /* [한국어] fill 포트 사용 사이클 합산 */
    return ret; /* [한국어] 합산 결과 반환 */
  }

  /* [한국어] print_port_stats - 포트 활용률 통계를 fout에 출력.
   * data_port_busy_cycles / port_available_cycles 비율로 포트 포화도를 보고한다.
   * 구현은 gpu-cache.cc에 있음. */
  void print_port_stats(FILE *fout, const char *cache_name) const;
};

// Used for collecting AerialVision per-window statistics
/* [한국어] AerialVision 성능 가시화 도구용 시간 윈도우(per-window) 캐시 통계 구조체.
 * cache_sub_stats와 달리 읽기/쓰기를 구분하여 집계한다.
 * new_window() 호출마다 clear()로 초기화되어 윈도우 단위 성능 추이를 추적한다.
 * 사용 패턴: 주기적으로 get_sub_stats_pw()로 읽고 clear_pw()로 초기화 반복. */
struct cache_sub_stats_pw {
  unsigned accesses;
  /* [한국어] 윈도우 내 총 캐시 접근 횟수 (읽기+쓰기 통합).
   * 설정자: inc_stats_pw()에서 접근 시마다 증가.
   * 읽는 자: AerialVision 가시화 도구 — 시간대별 접근 패턴 표시.
   * 값 범위: 0 ~ 윈도우 기간 내 접근 수.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned write_misses;
  /* [한국어] 윈도우 내 쓰기 미스 횟수.
   * 설정자: inc_stats_pw()에서 쓰기+MISS 조합 시 증가.
   * 읽는 자: AerialVision — 쓰기 효율 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned write_hits;
  /* [한국어] 윈도우 내 쓰기 히트 횟수.
   * 설정자: inc_stats_pw()에서 쓰기+HIT 조합 시 증가.
   * 읽는 자: AerialVision — 쓰기 히트율 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned write_pending_hits;
  /* [한국어] 윈도우 내 쓰기 HIT_RESERVED 횟수 (쓰기 pending hit).
   * 설정자: inc_stats_pw()에서 쓰기+HIT_RESERVED 조합 시 증가.
   * 읽는 자: AerialVision — 쓰기 MSHR 병합 효율.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned write_res_fails;
  /* [한국어] 윈도우 내 쓰기 RESERVATION_FAIL 횟수.
   * 설정자: inc_stats_pw()에서 쓰기+RESERVATION_FAIL 시 증가.
   * 읽는 자: AerialVision — 쓰기 캐시 병목 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  unsigned read_misses;
  /* [한국어] 윈도우 내 읽기 미스 횟수.
   * 설정자: inc_stats_pw()에서 읽기+MISS 조합 시 증가.
   * 읽는 자: AerialVision — 읽기 효율 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned read_hits;
  /* [한국어] 윈도우 내 읽기 히트 횟수.
   * 설정자: inc_stats_pw()에서 읽기+HIT 조합 시 증가.
   * 읽는 자: AerialVision — 읽기 히트율 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned read_pending_hits;
  /* [한국어] 윈도우 내 읽기 HIT_RESERVED 횟수.
   * 설정자: inc_stats_pw()에서 읽기+HIT_RESERVED 조합 시 증가.
   * 읽는 자: AerialVision — 읽기 MSHR 병합 효율.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */
  unsigned read_res_fails;
  /* [한국어] 윈도우 내 읽기 RESERVATION_FAIL 횟수.
   * 설정자: inc_stats_pw()에서 읽기+RESERVATION_FAIL 시 증가.
   * 읽는 자: AerialVision — 읽기 캐시 병목 분석.
   * 값 범위: 0 ~ accesses.
   * 동기화: 단일 SM 컨텍스트, 락 불필요. */

  /* [한국어] cache_sub_stats_pw 기본 생성자 — clear()로 모든 카운터 0 초기화 */
  cache_sub_stats_pw() { clear(); }
  /*
   * [한국어]
   * clear - 윈도우 통계 카운터 전체 초기화
   *
   * AerialVision 윈도우 전환 시 호출되어 새 윈도우의 통계를 깨끗이 시작한다.
   * 호출 체인: cache_stats::clear_pw() → cache_sub_stats_pw::clear()
   */
  void clear() {
    accesses = 0;              /* [한국어] 총 접근 수 초기화 */
    write_misses = 0;          /* [한국어] 쓰기 미스 수 초기화 */
    write_hits = 0;            /* [한국어] 쓰기 히트 수 초기화 */
    write_pending_hits = 0;    /* [한국어] 쓰기 pending hit 수 초기화 */
    write_res_fails = 0;       /* [한국어] 쓰기 RESERVATION_FAIL 수 초기화 */
    read_misses = 0;           /* [한국어] 읽기 미스 수 초기화 */
    read_hits = 0;             /* [한국어] 읽기 히트 수 초기화 */
    read_pending_hits = 0;     /* [한국어] 읽기 pending hit 수 초기화 */
    read_res_fails = 0;        /* [한국어] 읽기 RESERVATION_FAIL 수 초기화 */
  }
  /*
   * [한국어]
   * operator+= - 다른 cache_sub_stats_pw를 현재 윈도우 통계에 누적
   *
   * @css: 합산할 윈도우 통계 구조체
   * @return: 누적된 this 참조
   */
  cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;                       /* [한국어] 총 접근 수 누적 */
    write_misses += css.write_misses;               /* [한국어] 쓰기 미스 수 누적 */
    read_misses += css.read_misses;                 /* [한국어] 읽기 미스 수 누적 */
    write_pending_hits += css.write_pending_hits;   /* [한국어] 쓰기 pending hit 수 누적 */
    read_pending_hits += css.read_pending_hits;     /* [한국어] 읽기 pending hit 수 누적 */
    write_res_fails += css.write_res_fails;         /* [한국어] 쓰기 RESERVATION_FAIL 누적 */
    read_res_fails += css.read_res_fails;           /* [한국어] 읽기 RESERVATION_FAIL 누적 */
    return *this; /* [한국어] 누적된 자신 반환 */
  }

  /*
   * [한국어]
   * operator+ - 두 윈도우 통계의 합을 새 객체로 반환
   *
   * @cs: 합산할 윈도우 통계 구조체
   * @return: 합산 결과를 담은 새 cache_sub_stats_pw 객체
   */
  cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats_pw ret;                                 /* [한국어] 결과 구조체 생성 */
    ret.accesses = accesses + cs.accesses;                  /* [한국어] 총 접근 수 합산 */
    ret.write_misses = write_misses + cs.write_misses;      /* [한국어] 쓰기 미스 합산 */
    ret.read_misses = read_misses + cs.read_misses;         /* [한국어] 읽기 미스 합산 */
    ret.write_pending_hits = write_pending_hits + cs.write_pending_hits; /* [한국어] 쓰기 pending hit 합산 */
    ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;    /* [한국어] 읽기 pending hit 합산 */
    ret.write_res_fails = write_res_fails + cs.write_res_fails;          /* [한국어] 쓰기 RESERVATION_FAIL 합산 */
    ret.read_res_fails = read_res_fails + cs.read_res_fails;             /* [한국어] 읽기 RESERVATION_FAIL 합산 */
    return ret; /* [한국어] 합산 결과 반환 */
  }
};

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
 public:
  cache_stats();
  void clear();
  // Clear AerialVision cache stats after each window
  void clear_pw();
  void inc_stats(int access_type, int access_outcome,
                 unsigned long long streamID);
  // Increment AerialVision cache stats
  void inc_stats_pw(int access_type, int access_outcome,
                    unsigned long long streamID);
  void inc_fail_stats(int access_type, int fail_outcome,
                      unsigned long long streamID);
  enum cache_request_status select_stats_status(
      enum cache_request_status probe, enum cache_request_status access) const;
  unsigned long long &operator()(int access_type, int access_outcome,
                                 bool fail_outcome,
                                 unsigned long long streamID);
  unsigned long long operator()(int access_type, int access_outcome,
                                bool fail_outcome,
                                unsigned long long streamID) const;
  cache_stats operator+(const cache_stats &cs);
  cache_stats &operator+=(const cache_stats &cs);
  void print_stats(FILE *fout, unsigned long long streamID,
                   const char *cache_name = "Cache_stats") const;
  void print_fail_stats(FILE *fout, unsigned long long streamID,
                        const char *cache_name = "Cache_fail_stats") const;

  unsigned long long get_stats(enum mem_access_type *access_type,
                               unsigned num_access_type,
                               enum cache_request_status *access_status,
                               unsigned num_access_status) const;
  void get_sub_stats(struct cache_sub_stats &css) const;

  // Get per-window cache stats for AerialVision
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

  void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);

 private:
  bool check_valid(int type, int status) const;
  bool check_fail_valid(int type, int fail) const;

  // CUDA streamID -> cache stats[NUM_MEM_ACCESS_TYPE]
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_stats;
  // AerialVision cache stats (per-window)
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_stats_pw;
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_fail_stats;

  unsigned long long m_cache_port_available_cycles;
  unsigned long long m_cache_data_port_busy_cycles;
  unsigned long long m_cache_fill_port_busy_cycles;
};

class cache_t {
 public:
  virtual ~cache_t() {}
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;

  // accessors for cache bandwidth availability
  virtual bool data_port_free() const = 0;
  virtual bool fill_port_free() const = 0;
};

bool was_write_sent(const std::list<cache_event> &events);
bool was_read_sent(const std::list<cache_event> &events);
bool was_writeallocate_sent(const std::list<cache_event> &events);

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
class baseline_cache : public cache_t {
 public:
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, enum cache_gpu_level level,
                 gpgpu_sim *gpu)
      : m_config(config),
        m_tag_array(new tag_array(config, core_id, type_id)),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config),
        m_level(level),
        m_gpu(gpu) {
    init(name, config, memport, status);
  }

  void init(const char *name, const cache_config &config,
            mem_fetch_interface *memport, enum mem_fetch_status status) {
    m_name = name;
    assert(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
    m_memport = memport;
    m_miss_queue_status = status;
  }

  virtual ~baseline_cache() { delete m_tag_array; }

  void update_cache_parameters(cache_config &config) {
    m_config = config;
    m_tag_array->update_cache_parameters(config);
    m_mshrs.check_mshr_parameters(config.m_mshr_entries,
                                  config.m_mshr_max_merge);
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;
  /// Sends next request to lower level of memory
  void cycle();
  /// Interface for response from lower memory level (model bandwidth
  /// restictions in caller)
  void fill(mem_fetch *mf, unsigned time);
  /// Checks if mf is waiting to be filled by lower memory level
  bool waiting_for_fill(mem_fetch *mf);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  bool access_ready() const { return m_mshrs.access_ready(); }
  /// Pop next ready access (does not include accesses that "HIT")
  mem_fetch *next_access() { return m_mshrs.next_access(); }
  // flash invalidate all entries in cache
  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }
  void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
  void display_state(FILE *fp) const;

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }
  void get_sub_stats(struct cache_sub_stats &css) const {
    m_stats.get_sub_stats(css);
  }
  // Clear per-window stats for AerialVision support
  void clear_pw() { m_stats.clear_pw(); }
  // Per-window sub stats for AerialVision support
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
    m_stats.get_sub_stats_pw(css);
  }

  // accessors for cache bandwidth availability
  bool data_port_free() const {
    return m_bandwidth_management.data_port_free();
  }
  bool fill_port_free() const {
    return m_bandwidth_management.fill_port_free();
  }
  void inc_aggregated_stats(cache_request_status status,
                            cache_request_status cache_status, mem_fetch *mf,
                            enum cache_gpu_level level);
  void inc_aggregated_fail_stats(cache_request_status status,
                                 cache_request_status cache_status,
                                 mem_fetch *mf, enum cache_gpu_level level);
  void inc_aggregated_stats_pw(cache_request_status status,
                               cache_request_status cache_status, mem_fetch *mf,
                               enum cache_gpu_level level);

  // This is a gapping hole we are poking in the system to quickly handle
  // filling the cache on cudamemcopies. We don't care about anything other than
  // L2 state after the memcopy - so just force the tag array to act as though
  // something is read or written without doing anything else.
  void force_tag_access(new_addr_type addr, unsigned time,
                        mem_access_sector_mask_t mask) {
    mem_access_byte_mask_t byte_mask;
    m_tag_array->fill(addr, time, mask, byte_mask, true);
  }

 protected:
  // Constructor that can be used by derived classes with custom tag arrays
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, tag_array *new_tag_array)
      : m_config(config),
        m_tag_array(new_tag_array),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config) {
    init(name, config, memport, status);
  }

 protected:
  std::string m_name;
  cache_config &m_config;
  tag_array *m_tag_array;
  mshr_table m_mshrs;
  std::list<mem_fetch *> m_miss_queue;
  enum mem_fetch_status m_miss_queue_status;
  mem_fetch_interface *m_memport;
  cache_gpu_level m_level;
  gpgpu_sim *m_gpu;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(new_addr_type a, new_addr_type ad, unsigned i, unsigned d,
                    const cache_config &m_config) {
      m_valid = true;
      m_block_addr = a;
      m_addr = ad;
      m_cache_index = i;
      m_data_size = d;
      pending_read = m_config.m_mshr_type == SECTOR_ASSOC
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    new_addr_type m_block_addr;
    new_addr_type m_addr;
    unsigned m_cache_index;
    unsigned m_data_size;
    // this variable is used when a load request generates multiple load
    // transactions For example, a read request from non-sector L1 request sends
    // a request to sector L2
    unsigned pending_read;
  };

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;

  cache_stats m_stats;

  /// Checks whether this request can be handled on this cycle. num_miss equals
  /// max # of misses to be handled on this cycle
  bool miss_queue_full(unsigned num_miss) {
    return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  }
  /// Read miss handler without writeback
  void send_read_request(new_addr_type addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, std::list<cache_event> &events,
                         bool read_only, bool wa);
  /// Read miss handler. Check MSHR hit or MSHR available
  void send_read_request(new_addr_type addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, bool &wb, evicted_block_info &evicted,
                         std::list<cache_event> &events, bool read_only,
                         bool wa);

  /// Sub-class containing all metadata for port bandwidth management
  class bandwidth_management {
   public:
    bandwidth_management(cache_config &config);

    /// use the data port based on the outcome and events generated by the
    /// mem_fetch request
    void use_data_port(mem_fetch *mf, enum cache_request_status outcome,
                       const std::list<cache_event> &events);

    /// use the fill port
    void use_fill_port(mem_fetch *mf);

    /// called every cache cycle to free up the ports
    void replenish_port_bandwidth();

    /// query for data port availability
    bool data_port_free() const;
    /// query for fill port availability
    bool fill_port_free() const;

   protected:
    const cache_config &m_config;

    int m_data_port_occupied_cycles;  //< Number of cycle that the data port
                                      // remains used
    int m_fill_port_occupied_cycles;  //< Number of cycle that the fill port
                                      // remains used
  };

  bandwidth_management m_bandwidth_management;
};

/// Read only cache
class read_only_cache : public baseline_cache {
 public:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, enum cache_gpu_level level,
                  gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {}

  /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
  /// could not be accepted (for any reason)
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

  virtual ~read_only_cache() {}

 protected:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, tag_array *new_tag_array)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {}
};

/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
 public:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, mem_access_type wr_alloc_type,
             mem_access_type wrbk_type, class gpgpu_sim *gpu,
             enum cache_gpu_level level)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  virtual ~data_cache() {}

  virtual void init(mem_fetch_allocator *mfcreator) {
    m_memfetch_creator = mfcreator;

    // Set read hit function
    m_rd_hit = &data_cache::rd_hit_base;

    // Set read miss function
    m_rd_miss = &data_cache::rd_miss_base;

    // Set write hit function
    switch (m_config.m_write_policy) {
      // READ_ONLY is now a separate cache class, config is deprecated
      case READ_ONLY:
        assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
        break;
      case WRITE_BACK:
        m_wr_hit = &data_cache::wr_hit_wb;
        break;
      case WRITE_THROUGH:
        m_wr_hit = &data_cache::wr_hit_wt;
        break;
      case WRITE_EVICT:
        m_wr_hit = &data_cache::wr_hit_we;
        break;
      case LOCAL_WB_GLOBAL_WT:
        m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
        break;
      default:
        assert(0 && "Error: Must set valid cache write policy\n");
        break;  // Need to set a write hit function
    }

    // Set write miss function
    switch (m_config.m_write_alloc_policy) {
      case NO_WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_no_wa;
        break;
      case WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_wa_naive;
        break;
      case FETCH_ON_WRITE:
        m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
        break;
      case LAZY_FETCH_ON_READ:
        m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
        break;
      default:
        assert(0 && "Error: Must set valid cache write miss policy\n");
        break;  // Need to set a write miss function
    }
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

 protected:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, tag_array *new_tag_array,
             mem_access_type wr_alloc_type, mem_access_type wrbk_type,
             class gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  mem_access_type m_wr_alloc_type;  // Specifies type of write allocate request
                                    // (e.g., L1 or L2)
  mem_access_type
      m_wrbk_type;  // Specifies type of writeback request (e.g., L1 or L2)
  class gpgpu_sim *m_gpu;

  //! A general function that takes the result of a tag_array probe
  //  and performs the correspding functions based on the cache configuration
  //  The access fucntion calls this function
  enum cache_request_status process_tag_probe(bool wr,
                                              enum cache_request_status status,
                                              new_addr_type addr,
                                              unsigned cache_index,
                                              mem_fetch *mf, unsigned time,
                                              std::list<cache_event> &events);

 protected:
  mem_fetch_allocator *m_memfetch_creator;

  // Functions for data cache access
  /// Sends write request to lower level memory (write or writeback)
  void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
                          std::list<cache_event> &events);
  void update_m_readable(mem_fetch *mf, unsigned cache_index);
  // Member Function pointers - Set by configuration options
  // to the functions below each grouping
  /******* Write-hit configs *******/
  enum cache_request_status (data_cache::*m_wr_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Marks block as MODIFIED and updates block LRU
  enum cache_request_status wr_hit_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-back
  enum cache_request_status wr_hit_wt(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-through

  /// Marks block as INVALID and sends write request to lower level memory
  enum cache_request_status wr_hit_we(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-evict
  enum cache_request_status wr_hit_global_we_local_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  // global write-evict, local write-back

  /******* Write-miss configs *******/
  enum cache_request_status (data_cache::*m_wr_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Sends read request, and possible write-back request,
  //  to lower level memory for a write miss with write-allocate
  enum cache_request_status wr_miss_wa_naive(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate-send-write-and-read-request
  enum cache_request_status wr_miss_wa_fetch_on_write(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate with fetch-on-every-write
  enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-allocate with read-fetch-only
  enum cache_request_status wr_miss_wa_write_validate(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate that writes with no read fetch
  enum cache_request_status wr_miss_no_wa(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // no write-allocate

  // Currently no separate functions for reads
  /******* Read-hit configs *******/
  enum cache_request_status (data_cache::*m_rd_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_hit_base(new_addr_type addr,
                                        unsigned cache_index, mem_fetch *mf,
                                        unsigned time,
                                        std::list<cache_event> &events,
                                        enum cache_request_status status);

  /******* Read-miss configs *******/
  enum cache_request_status (data_cache::*m_rd_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_miss_base(new_addr_type addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned time,
                                         std::list<cache_event> &events,
                                         enum cache_request_status status);
};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
class l1_cache : public data_cache {
 public:
  l1_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L1_WR_ALLOC_R, L1_WRBK_ACC, gpu, level) {}

  virtual ~l1_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

 protected:
  l1_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, tag_array *new_tag_array,
           class gpgpu_sim *gpu)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC, gpu) {}
};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
 public:
  l2_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {}

  virtual ~l2_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);
};

/*****************************************************************************/

// See the following paper to understand this cache model:
//
// Igehy, et al., Prefetching in a Texture Cache Architecture,
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
 public:
  tex_cache(const char *name, cache_config &config, int core_id, int type_id,
            mem_fetch_interface *memport, enum mem_fetch_status request_status,
            enum mem_fetch_status rob_status)
      : m_config(config),
        m_tags(config, core_id, type_id),
        m_fragment_fifo(config.m_fragment_fifo_entries),
        m_request_fifo(config.m_request_fifo_entries),
        m_rob(config.m_rob_entries),
        m_result_fifo(config.m_result_fifo_entries) {
    m_name = name;
    assert(config.m_mshr_type == TEX_FIFO ||
           config.m_mshr_type == SECTOR_TEX_FIFO);
    assert(config.m_write_policy == READ_ONLY);
    assert(config.m_alloc_policy == ON_MISS);
    m_memport = memport;
    m_cache = new data_block[config.get_num_lines()];
    m_request_queue_status = request_status;
    m_rob_status = rob_status;
  }

  /// Access function for tex_cache
  /// return values: RESERVATION_FAIL if request could not be accepted
  /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
  /// since unlike a normal CPU cache, a "HIT" in texture cache does not
  /// mean the data is ready (still need to get through fragment fifo)
  enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                   unsigned time,
                                   std::list<cache_event> &events);
  void cycle();
  /// Place returning cache block into reorder buffer
  void fill(mem_fetch *mf, unsigned time);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  bool access_ready() const { return !m_result_fifo.empty(); }
  /// Pop next ready access (includes both accesses that "HIT" and those that
  /// "MISS")
  mem_fetch *next_access() { return m_result_fifo.pop(); }
  void display_state(FILE *fp) const;

  // accessors for cache bandwidth availability - stubs for now
  bool data_port_free() const { return true; }
  bool fill_port_free() const { return true; }

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }

  void get_sub_stats(struct cache_sub_stats &css) const {
    m_stats.get_sub_stats(css);
  }

 private:
  std::string m_name;
  const cache_config &m_config;

  struct fragment_entry {
    fragment_entry() {}
    fragment_entry(mem_fetch *mf, unsigned idx, bool m, unsigned d) {
      m_request = mf;
      m_cache_index = idx;
      m_miss = m;
      m_data_size = d;
    }
    mem_fetch *m_request;    // request information
    unsigned m_cache_index;  // where to look for data
    bool m_miss;             // true if sent memory request
    unsigned m_data_size;
  };

  struct rob_entry {
    rob_entry() {
      m_ready = false;
      m_time = 0;
      m_request = NULL;
    }
    rob_entry(unsigned i, mem_fetch *mf, new_addr_type a) {
      m_ready = false;
      m_index = i;
      m_time = 0;
      m_request = mf;
      m_block_addr = a;
    }
    bool m_ready;
    unsigned m_time;   // which cycle did this entry become ready?
    unsigned m_index;  // where in cache should block be placed?
    mem_fetch *m_request;
    new_addr_type m_block_addr;
  };

  struct data_block {
    data_block() { m_valid = false; }
    bool m_valid;
    new_addr_type m_block_addr;
  };

  // TODO: replace fifo_pipeline with this?
  template <class T>
  class fifo {
   public:
    fifo(unsigned size) {
      m_size = size;
      m_num = 0;
      m_head = 0;
      m_tail = 0;
      m_data = new T[size];
    }
    bool full() const { return m_num == m_size; }
    bool empty() const { return m_num == 0; }
    unsigned size() const { return m_num; }
    unsigned capacity() const { return m_size; }
    unsigned push(const T &e) {
      assert(!full());
      m_data[m_head] = e;
      unsigned result = m_head;
      inc_head();
      return result;
    }
    T pop() {
      assert(!empty());
      T result = m_data[m_tail];
      inc_tail();
      return result;
    }
    const T &peek(unsigned index) const {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek(unsigned index) {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek() const { return m_data[m_tail]; }
    unsigned next_pop_index() const { return m_tail; }

   private:
    void inc_head() {
      m_head = (m_head + 1) % m_size;
      m_num++;
    }
    void inc_tail() {
      assert(m_num > 0);
      m_tail = (m_tail + 1) % m_size;
      m_num--;
    }

    unsigned m_head;  // next entry goes here
    unsigned m_tail;  // oldest entry found here
    unsigned m_num;   // how many in fifo?
    unsigned m_size;  // maximum number of entries in fifo
    T *m_data;
  };

  tag_array m_tags;
  fifo<fragment_entry> m_fragment_fifo;
  fifo<mem_fetch *> m_request_fifo;
  fifo<rob_entry> m_rob;
  data_block *m_cache;
  fifo<mem_fetch *> m_result_fifo;  // next completed texture fetch

  mem_fetch_interface *m_memport;
  enum mem_fetch_status m_request_queue_status;
  enum mem_fetch_status m_rob_status;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(unsigned i, const cache_config &m_config) {
      m_valid = true;
      m_rob_index = i;
      pending_read = m_config.m_mshr_type == SECTOR_TEX_FIFO
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    unsigned m_rob_index;
    unsigned pending_read;
  };

  cache_stats m_stats;

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
