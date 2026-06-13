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

enum replacement_policy_t { LRU, FIFO };

enum write_policy_t {
  READ_ONLY,
  WRITE_BACK,
  WRITE_THROUGH,
  WRITE_EVICT,
  LOCAL_WB_GLOBAL_WT
};

enum allocation_policy_t { ON_MISS, ON_FILL, STREAMING };

enum write_allocate_policy_t {
  NO_WRITE_ALLOCATE,
  WRITE_ALLOCATE,
  FETCH_ON_WRITE,
  LAZY_FETCH_ON_READ
};

enum mshr_config_t {
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
};

enum set_index_function {
  LINEAR_SET_FUNCTION = 0,
  BITWISE_XORING_FUNCTION,
  HASH_IPOLY_FUNCTION,
  FERMI_HASH_SET_FUNCTION,
  CUSTOM_SET_FUNCTION
};

enum cache_type { NORMAL = 0, SECTOR };

#define MAX_WARP_PER_SHADER 64
#define INCT_TOTAL_BUFFER 64
#define L2_TOTAL 64
#define MAX_WARP_PER_SHADER 64
#define MAX_WARP_PER_SHADER 64

class cache_config {
 public:
  cache_config() {
    m_valid = false;
    m_disabled = false;
    m_config_string = NULL;  // set by option parser
    m_config_stringPrefL1 = NULL;
    m_config_stringPrefShared = NULL;
    m_data_port_width = 0;
    m_set_index_function = LINEAR_SET_FUNCTION;
    m_is_streaming = false;
    m_wr_percent = 0;
  }
  void init(char *config, FuncCache status) {
    cache_status = status;
    assert(config);
    char ct, rp, wp, ap, mshr_type, wap, sif;

    int ntok =
        sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
               &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
               &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
               &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);

    if (ntok < 12) {
      if (!strcmp(config, "none")) {
        m_disabled = true;
        return;
      }
      exit_parse_error();
    }

    switch (ct) {
      case 'N':
        m_cache_type = NORMAL;
        break;
      case 'S':
        m_cache_type = SECTOR;
        break;
      default:
        exit_parse_error();
    }
    switch (rp) {
      case 'L':
        m_replacement_policy = LRU;
        break;
      case 'F':
        m_replacement_policy = FIFO;
        break;
      default:
        exit_parse_error();
    }
    switch (wp) {
      case 'R':
        m_write_policy = READ_ONLY;
        break;
      case 'B':
        m_write_policy = WRITE_BACK;
        break;
      case 'T':
        m_write_policy = WRITE_THROUGH;
        break;
      case 'E':
        m_write_policy = WRITE_EVICT;
        break;
      case 'L':
        m_write_policy = LOCAL_WB_GLOBAL_WT;
        break;
      default:
        exit_parse_error();
    }
    switch (ap) {
      case 'm':
        m_alloc_policy = ON_MISS;
        break;
      case 'f':
        m_alloc_policy = ON_FILL;
        break;
      case 's':
        m_alloc_policy = STREAMING;
        break;
      default:
        exit_parse_error();
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
      m_is_streaming = true;
      m_alloc_policy = ON_FILL;
    }
    switch (mshr_type) {
      case 'F':
        m_mshr_type = TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'T':
        m_mshr_type = SECTOR_TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'A':
        m_mshr_type = ASSOC;
        break;
      case 'S':
        m_mshr_type = SECTOR_ASSOC;
        break;
      default:
        exit_parse_error();
    }
    m_line_sz_log2 = LOGB2(m_line_sz);
    m_nset_log2 = LOGB2(m_nset);
    m_valid = true;
    m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
    m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
    original_m_assoc = m_assoc;

    // For more details about difference between FETCH_ON_WRITE and WRITE
    // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
    // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
    // GPGPU-sim 3.x, that send WRITE and READ for every write request
    switch (wap) {
      case 'N':
        m_write_alloc_policy = NO_WRITE_ALLOCATE;
        break;
      case 'W':
        m_write_alloc_policy = WRITE_ALLOCATE;
        break;
      case 'F':
        m_write_alloc_policy = FETCH_ON_WRITE;
        break;
      case 'L':
        m_write_alloc_policy = LAZY_FETCH_ON_READ;
        break;
      default:
        exit_parse_error();
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
      assert(0 &&
             "Invalid cache configuration: Writeback cache cannot allocate new "
             "line on fill. ");
    }

    if ((m_write_alloc_policy == FETCH_ON_WRITE ||
         m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        m_alloc_policy == ON_FILL) {
      assert(
          0 &&
          "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
          "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    }

    if (m_cache_type == SECTOR) {
      bool cond = m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
                  m_line_sz % SECTOR_SIZE == 0;
      if (!cond) {
        std::cerr << "error: For sector cache, the simulator uses hard-coded "
                     "SECTOR_SIZE and SECTOR_CHUNCK_SIZE. The line size "
                     "must be product of both values.\n";
        assert(0);
      }
    }

    // default: port to data array width and granularity = line size
    if (m_data_port_width == 0) {
      m_data_port_width = m_line_sz;
    }
    assert(m_line_sz % m_data_port_width == 0);

    switch (sif) {
      case 'H':
        m_set_index_function = FERMI_HASH_SET_FUNCTION;
        break;
      case 'P':
        m_set_index_function = HASH_IPOLY_FUNCTION;
        break;
      case 'C':
        m_set_index_function = CUSTOM_SET_FUNCTION;
        break;
      case 'L':
        m_set_index_function = LINEAR_SET_FUNCTION;
        break;
      case 'X':
        m_set_index_function = BITWISE_XORING_FUNCTION;
        break;
      default:
        exit_parse_error();
    }
  }
  bool disabled() const { return m_disabled; }
  unsigned get_line_sz() const {
    assert(m_valid);
    return m_line_sz;
  }
  unsigned get_atom_sz() const {
    assert(m_valid);
    return m_atom_sz;
  }
  unsigned get_num_lines() const {
    assert(m_valid);
    return m_nset * m_assoc;
  }
  unsigned get_max_num_lines() const {
    assert(m_valid);
    return get_max_cache_multiplier() * m_nset * original_m_assoc;
  }
  unsigned get_max_assoc() const {
    assert(m_valid);
    return get_max_cache_multiplier() * original_m_assoc;
  }
  void print(FILE *fp) const {
    fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
            m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
  }

  virtual unsigned set_index(new_addr_type addr) const;

  virtual unsigned get_max_cache_multiplier() const {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
  }

  unsigned hash_function(new_addr_type addr, unsigned m_nset,
                         unsigned m_line_sz_log2, unsigned m_nset_log2,
                         unsigned m_index_function) const;

  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.

    // return addr >> (m_line_sz_log2+m_nset_log2);
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type mshr_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_atom_sz - 1);
  }
  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
  void set_assoc(unsigned n) {
    // set new assoc. L1 cache dynamically resized in Volta
    m_assoc = n;
  }
  unsigned get_nset() const {
    assert(m_valid);
    return m_nset;
  }
  unsigned get_total_size_inKB() const {
    assert(m_valid);
    return (m_assoc * m_nset * m_line_sz) / 1024;
  }
  bool is_streaming() { return m_is_streaming; }
  FuncCache get_cache_status() { return cache_status; }
  void set_allocation_policy(enum allocation_policy_t alloc) {
    m_alloc_policy = alloc;
  }
  char *m_config_string;
  char *m_config_stringPrefL1;
  char *m_config_stringPrefShared;
  FuncCache cache_status;
  unsigned m_wr_percent;
  write_allocate_policy_t get_write_allocate_policy() {
    return m_write_alloc_policy;
  }
  write_policy_t get_write_policy() { return m_write_policy; }

 protected:
  void exit_parse_error() {
    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
           m_config_string);
    abort();
  }

  bool m_valid;
  bool m_disabled;
  unsigned m_line_sz;
  unsigned m_line_sz_log2;
  unsigned m_nset;
  unsigned m_nset_log2;
  unsigned m_assoc;
  unsigned m_atom_sz;
  unsigned m_sector_sz_log2;
  unsigned original_m_assoc;
  bool m_is_streaming;

  enum replacement_policy_t m_replacement_policy;  // 'L' = LRU, 'F' = FIFO
  enum write_policy_t
      m_write_policy;  // 'T' = write through, 'B' = write back, 'R' = read only
  enum allocation_policy_t
      m_alloc_policy;  // 'm' = allocate on miss, 'f' = allocate on fill
  enum mshr_config_t m_mshr_type;
  enum cache_type m_cache_type;

  write_allocate_policy_t
      m_write_alloc_policy;  // 'W' = Write allocate, 'N' = No write allocate

  union {
    unsigned m_mshr_entries;
    unsigned m_fragment_fifo_entries;
  };
  union {
    unsigned m_mshr_max_merge;
    unsigned m_request_fifo_entries;
  };
  union {
    unsigned m_miss_queue_size;
    unsigned m_rob_entries;
  };
  unsigned m_result_fifo_entries;
  unsigned m_data_port_width;  //< number of byte the cache can access per cycle
  enum set_index_function
      m_set_index_function;  // Hash, linear, or custom set index function

  friend class tag_array;
  friend class baseline_cache;
  friend class read_only_cache;
  friend class tex_cache;
  friend class data_cache;
  friend class l1_cache;
  friend class l2_cache;
  friend class memory_sub_partition;
};

class l1d_cache_config : public cache_config {
 public:
  l1d_cache_config() : cache_config() {}
  unsigned set_bank(new_addr_type addr) const;
  void init(char *config, FuncCache status) {
    l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
    l1_banks_log2 = LOGB2(l1_banks);
    cache_config::init(config, status);
  }
  unsigned l1_latency;
  unsigned l1_banks;
  unsigned l1_banks_log2;
  unsigned l1_banks_byte_interleaving;
  unsigned l1_banks_byte_interleaving_log2;
  unsigned l1_banks_hashing_function;
  unsigned m_unified_cache_size;
  virtual unsigned get_max_cache_multiplier() const {
    // set * assoc * cacheline size. Then convert Byte to KB
    // gpgpu_unified_cache_size is in KB while original_sz is in B
    if (m_unified_cache_size > 0) {
      unsigned original_size = m_nset * original_m_assoc * m_line_sz / 1024;
      assert(m_unified_cache_size % original_size == 0);
      return m_unified_cache_size / original_size;
    } else {
      return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
    }
  }
};

class l2_cache_config : public cache_config {
 public:
  l2_cache_config() : cache_config() {}
  void init(linear_to_raw_address_translation *address_mapping);
  virtual unsigned set_index(new_addr_type addr) const;

 private:
  linear_to_raw_address_translation *m_address_mapping;
};

class tag_array {
 public:
  // Use this constructor
  tag_array(cache_config &config, int core_id, int type_id);
  ~tag_array();

  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_fetch *mf, bool is_write,
                                  bool probe_mode = false) const;
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_access_sector_mask_t mask, bool is_write,
                                  bool probe_mode = false,
                                  mem_fetch *mf = NULL) const;
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, mem_fetch *mf);
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, bool &wb,
                                   evicted_block_info &evicted, mem_fetch *mf);

  void fill(new_addr_type addr, unsigned time, mem_fetch *mf, bool is_write);
  void fill(unsigned idx, unsigned time, mem_fetch *mf);
  void fill(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask,
            mem_access_byte_mask_t byte_mask, bool is_write);

  unsigned size() const { return m_config.get_num_lines(); }
  cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  void new_window();

  void print(FILE *stream, unsigned &total_access,
             unsigned &total_misses) const;
  float windowed_miss_rate() const;
  void get_stats(unsigned &total_access, unsigned &total_misses,
                 unsigned &total_hit_res, unsigned &total_res_fail) const;

  void update_cache_parameters(cache_config &config);
  void add_pending_line(mem_fetch *mf);
  void remove_pending_line(mem_fetch *mf);
  void inc_dirty() { m_dirty++; }

 protected:
  // This constructor is intended for use only from derived classes that wish to
  // avoid unnecessary memory allocation that takes place in the
  // other tag_array constructor
  tag_array(cache_config &config, int core_id, int type_id,
            cache_block_t **new_lines);
  void init(int core_id, int type_id);

 protected:
  cache_config &m_config;

  cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */

  unsigned m_access;
  unsigned m_miss;
  unsigned m_pending_hit;  // number of cache miss that hit a line that is
                           // allocated but not filled
  unsigned m_res_fail;
  unsigned m_sector_miss;
  unsigned m_dirty;

  // performance counters for calculating the amount of misses within a time
  // window
  unsigned m_prev_snapshot_access;
  unsigned m_prev_snapshot_miss;
  unsigned m_prev_snapshot_pending_hit;

  int m_core_id;  // which shader core is using this
  int m_type_id;  // what kind of cache is this (normal, texture, constant)

  bool is_used;  // a flag if the whole cache has ever been accessed before

  typedef tr1_hash_map<new_addr_type, unsigned> line_table;
  line_table pending_lines;
};

class mshr_table {
 public:
  mshr_table(unsigned num_entries, unsigned max_merged)
      : m_num_entries(num_entries),
        m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
        ,
        m_data(2 * num_entries)
#endif
  {
  }

  /// Checks if there is a pending request to the lower memory level already
  bool probe(new_addr_type block_addr) const;
  /// Checks if there is space for tracking a new memory access
  bool full(new_addr_type block_addr) const;
  /// Add or merge this access
  void add(new_addr_type block_addr, mem_fetch *mf);
  /// Returns true if cannot accept new fill responses
  bool busy() const { return false; }
  /// Accept a new cache fill response: mark entry ready for processing
  void mark_ready(new_addr_type block_addr, bool &has_atomic);
  /// Returns true if ready accesses exist
  bool access_ready() const { return !m_current_response.empty(); }
  /// Returns next ready access
  mem_fetch *next_access();
  void display(FILE *fp) const;
  // Returns true if there is a pending read after write
  bool is_read_after_write_pending(new_addr_type block_addr);

  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(m_num_entries == num_entries &&
           "Change of MSHR parameters between kernels is not allowed");
    assert(m_max_merged == max_merged &&
           "Change of MSHR parameters between kernels is not allowed");
  }

 private:
  // finite sized, fully associative table, with a finite maximum number of
  // merged requests
  const unsigned m_num_entries;
  const unsigned m_max_merged;

  struct mshr_entry {
    std::list<mem_fetch *> m_list;
    bool m_has_atomic;
    mshr_entry() : m_has_atomic(false) {}
  };
  typedef tr1_hash_map<new_addr_type, mshr_entry> table;
  typedef tr1_hash_map<new_addr_type, mshr_entry> line_table;
  table m_data;
  line_table pending_lines;

  // it may take several cycles to process the merged requests
  bool m_current_response_ready;
  std::list<new_addr_type> m_current_response;
};

/***************************************************************** Caches
 * *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and
/// reservation fails.
///
struct cache_sub_stats {
  unsigned long long accesses;
  unsigned long long misses;
  unsigned long long pending_hits;
  unsigned long long res_fails;

  unsigned long long port_available_cycles;
  unsigned long long data_port_busy_cycles;
  unsigned long long fill_port_busy_cycles;

  cache_sub_stats() { clear(); }
  void clear() {
    accesses = 0;
    misses = 0;
    pending_hits = 0;
    res_fails = 0;
    port_available_cycles = 0;
    data_port_busy_cycles = 0;
    fill_port_busy_cycles = 0;
  }
  cache_sub_stats &operator+=(const cache_sub_stats &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    misses += css.misses;
    pending_hits += css.pending_hits;
    res_fails += css.res_fails;
    port_available_cycles += css.port_available_cycles;
    data_port_busy_cycles += css.data_port_busy_cycles;
    fill_port_busy_cycles += css.fill_port_busy_cycles;
    return *this;
  }

  cache_sub_stats operator+(const cache_sub_stats &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats ret;
    ret.accesses = accesses + cs.accesses;
    ret.misses = misses + cs.misses;
    ret.pending_hits = pending_hits + cs.pending_hits;
    ret.res_fails = res_fails + cs.res_fails;
    ret.port_available_cycles =
        port_available_cycles + cs.port_available_cycles;
    ret.data_port_busy_cycles =
        data_port_busy_cycles + cs.data_port_busy_cycles;
    ret.fill_port_busy_cycles =
        fill_port_busy_cycles + cs.fill_port_busy_cycles;
    return ret;
  }

  void print_port_stats(FILE *fout, const char *cache_name) const;
};

// Used for collecting AerialVision per-window statistics
struct cache_sub_stats_pw {
  unsigned accesses;
  unsigned write_misses;
  unsigned write_hits;
  unsigned write_pending_hits;
  unsigned write_res_fails;

  unsigned read_misses;
  unsigned read_hits;
  unsigned read_pending_hits;
  unsigned read_res_fails;

  cache_sub_stats_pw() { clear(); }
  void clear() {
    accesses = 0;
    write_misses = 0;
    write_hits = 0;
    write_pending_hits = 0;
    write_res_fails = 0;
    read_misses = 0;
    read_hits = 0;
    read_pending_hits = 0;
    read_res_fails = 0;
  }
  cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    write_misses += css.write_misses;
    read_misses += css.read_misses;
    write_pending_hits += css.write_pending_hits;
    read_pending_hits += css.read_pending_hits;
    write_res_fails += css.write_res_fails;
    read_res_fails += css.read_res_fails;
    return *this;
  }

  cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats_pw ret;
    ret.accesses = accesses + cs.accesses;
    ret.write_misses = write_misses + cs.write_misses;
    ret.read_misses = read_misses + cs.read_misses;
    ret.write_pending_hits = write_pending_hits + cs.write_pending_hits;
    ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;
    ret.write_res_fails = write_res_fails + cs.write_res_fails;
    ret.read_res_fails = read_res_fails + cs.read_res_fails;
    return ret;
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
