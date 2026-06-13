/*
 * [한국어 설명] PTX 시뮬레이션용 메모리 주소 공간 추상화 (memory.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이션(cuda-sim) 계층에서 GPU 메모리 주소 공간을
 * 소프트웨어로 에뮬레이션하는 핵심 추상화를 정의한다. GPU 스레드가 PTX 명령어를
 * 실행할 때 접근하는 shared memory, local memory, global memory, param memory,
 * const memory 등 모든 CUDA 메모리 공간이 이 파일의 클래스를 통해 구현된다.
 * 실제 GPU 하드웨어 메모리 대신 호스트 CPU의 힙(heap) 메모리를 사용하여
 * 주소→값 매핑을 관리하며, 타이밍 모델(gpgpu-sim/)과는 독립된 기능 정확성만을 목표로 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: PTX 명령어 실행(instructions.cc) → memory_space::read/write() → mem_storage
 * 이 파일은 기능 시뮬레이션(cuda-sim/) 계층에 속하며, 타이밍 시뮬레이션(gpgpu-sim/)의
 * 캐시/DRAM 모델과는 별개로 동작한다. PTX 인스트럭션 시뮬레이터(ptx_sim.h,
 * instructions.cc)가 ld/st 명령어를 처리할 때 이 인터페이스를 호출한다.
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드 (기능 시뮬레이션 단계).
 *
 * === 타 모듈과의 연결 ===
 * 의존: abstract_hardware_model.h (address_type, addr_t), tr1_hash_map.h (해시맵 구현)
 * 의존받음: cuda-sim/instructions.cc (ld/st 명령어 실행), cuda-sim/ptx_sim.h
 *   (ptx_thread_info가 m_local_mem, m_shared_mem 등으로 memory_space* 보유)
 * 데이터 흐름: PTX 스레드가 주소를 계산 → memory_space::read/write 호출 →
 *   내부적으로 mem_storage 블록에 바이트 배열로 저장/조회
 * 공유 구조체: mem_storage<BSIZE>는 고정 크기 블록 단위로 raw 바이트를 관리,
 *   memory_space_impl<BSIZE>는 해시맵으로 블록 집합을 관리한다.
 *
 * === 주요 함수/구조체 요약 ===
 * mem_storage<BSIZE>: 고정 크기(BSIZE 바이트) 메모리 블록 — 실제 데이터 저장소
 * memory_space: 순수 가상 인터페이스 — read/write/print/set_watch 추상화
 * memory_space_impl<BSIZE>: memory_space의 구체 구현 — 해시맵으로 블록을 관리
 * write(): 주소 범위가 단일 블록 내에 있으면 fast path, 블록 경계를 넘으면 slow path
 * read(): write와 동일한 fast/slow 경로 분기 — 미초기화 주소는 0 반환
 * set_watch(): 감시점(watchpoint) 주소 등록 — 해당 주소 write 시 시뮬레이터에 알림
 */
// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#ifndef memory_h_INCLUDED /* [한국어] 헤더 중복 포함 방지 가드 — 다중 .cc 파일에서 include 시 중복 정의 오류 방지 */
#define memory_h_INCLUDED

#include "../abstract_hardware_model.h" /* [한국어] address_type, addr_t 등 GPU 하드웨어 추상 타입 정의 포함 */

#include "../tr1_hash_map.h" /* [한국어] 컴파일러/STL 버전에 따라 unordered_map 또는 std::map을 선택하는 래퍼 — 플랫폼 호환성 확보 */
#define mem_map tr1_hash_map /* [한국어] 메모리 블록 맵 타입을 tr1_hash_map으로 별칭 — 추후 구현 교체를 용이하게 함 */
#if tr1_hash_map_ismap == 1 /* [한국어] tr1_hash_map이 실제로 std::map(정렬된 트리)이면 rehash 불필요 */
#define MEM_MAP_RESIZE(hash_size) /* [한국어] std::map은 rehash 개념이 없으므로 no-op 매크로로 대체 */
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size)) /* [한국어] unordered_map의 경우 미리 버킷 수를 지정하여 삽입 시 리해시 비용 절감 */
#endif

#include <assert.h> /* [한국어] assert() — 블록 경계 검사, 초기화 확인 등 런타임 불변 조건 검증 */
#include <stdio.h>  /* [한국어] fprintf, fflush — print() 디버그 출력에 사용 */
#include <stdlib.h> /* [한국어] calloc, free — mem_storage 내부 바이트 배열 할당/해제 */
#include <string.h> /* [한국어] memcpy — 블록 내 바이트 복사 (write/read 핵심 연산) */
#include <map>      /* [한국어] std::map<unsigned, mem_addr_t> — watchpoint 주소 등록 테이블 */
#include <string>   /* [한국어] std::string — m_name (메모리 공간 이름, 예: "shared", "local") */

typedef address_type mem_addr_t; /* [한국어] GPU 메모리 주소 타입 별칭 — abstract_hardware_model.h에서 정의된 address_type을 mem_addr_t로 재명명하여 메모리 모듈 내 일관된 사용 */

#define MEM_BLOCK_SIZE (4 * 1024) /* [한국어] 기본 메모리 블록 크기 = 4KB. memory_space_impl의 기본 BSIZE로 사용 가능; 블록 단위로 해시맵에 저장하여 희소(sparse) 주소 공간 효율적 처리 */

/*
 * [한국어]
 * mem_storage<BSIZE> - 고정 크기 메모리 블록 저장소
 *
 * @BSIZE: 블록 크기(바이트) — 반드시 2의 거듭제곱이어야 함 (비트 마스킹에 사용)
 *
 * GPU 메모리 주소 공간을 BSIZE 바이트 단위 블록으로 분할하여 관리하는 최하위 저장소.
 * memory_space_impl의 해시맵 값(value)으로 사용되며, 각 블록은 calloc으로
 * 0초기화된 연속 바이트 배열을 유지한다. 복사 생성자를 제공하여 해시맵이
 * 내부적으로 블록을 복사할 때 올바르게 동작한다.
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 * 동기화: 별도 락 없음 — 기능 시뮬레이션은 단일 스레드로 실행됨.
 *
 * 호출 체인:
 *   instructions.cc (ld/st 실행) → memory_space_impl::read/write
 *     → mem_storage::read/write
 */
template <unsigned BSIZE>
class mem_storage {
 public:
  /*
   * [한국어]
   * 복사 생성자 - 기존 블록의 데이터를 새 블록으로 deep copy
   *
   * @another: 복사 원본 블록
   *
   * 해시맵(unordered_map)이 내부적으로 블록을 복사할 때 호출됨.
   * calloc으로 새 BSIZE 바이트 버퍼를 할당하고 원본 데이터를 memcpy.
   * shallow copy만 하면 m_data 포인터가 공유되어 double-free 발생 — deep copy 필수.
   */
  mem_storage(const mem_storage &another) {
    m_data = (unsigned char *)calloc(1, BSIZE); /* [한국어] BSIZE 바이트 0초기화 버퍼 새로 할당 */
    memcpy(m_data, another.m_data, BSIZE);      /* [한국어] 원본 블록의 전체 데이터를 새 버퍼로 복사 */
  }
  /*
   * [한국어]
   * 기본 생성자 - 0초기화된 빈 블록 생성
   *
   * calloc으로 BSIZE 바이트를 할당하고 0으로 초기화.
   * 미초기화 메모리 주소에 대한 read가 0을 반환하는 동작(memory_space_impl::read_single_block)의
   * 기반이 됨.
   */
  mem_storage() { m_data = (unsigned char *)calloc(1, BSIZE); } /* [한국어] BSIZE 바이트 0초기화 버퍼 할당 */
  /*
   * [한국어]
   * 소멸자 - 내부 바이트 버퍼 해제
   *
   * calloc으로 할당한 m_data를 free. 해시맵에서 블록이 제거되거나
   * memory_space_impl 소멸 시 자동 호출됨.
   */
  ~mem_storage() { free(m_data); } /* [한국어] 내부 데이터 버퍼 메모리 해제 */

  /*
   * [한국어]
   * write - 블록 내 지정 오프셋에 데이터 기록
   *
   * @offset: 블록 시작으로부터의 바이트 오프셋 (0 ~ BSIZE-1)
   * @length: 기록할 바이트 수
   * @data: 기록할 데이터 소스 포인터
   *
   * memcpy를 통해 m_data[offset..offset+length-1]에 데이터를 기록.
   * offset+length > BSIZE이면 블록 경계 초과 — assert로 감지.
   * 호출자(memory_space_impl::write)가 블록 경계 내에 들어오도록 보장한 후 호출.
   *
   * 호출 체인: memory_space_impl::write → [이 함수]
   */
  void write(unsigned offset, size_t length, const unsigned char *data) {
    assert(offset + length <= BSIZE); /* [한국어] 블록 경계 초과 여부 확인 — 초과 시 즉시 abort */
    memcpy(m_data + offset, data, length); /* [한국어] 소스 데이터를 블록 내 오프셋 위치에 복사 */
  }

  /*
   * [한국어]
   * read - 블록 내 지정 오프셋에서 데이터 읽기
   *
   * @offset: 블록 시작으로부터의 바이트 오프셋
   * @length: 읽을 바이트 수
   * @data: 읽은 데이터를 저장할 목적지 버퍼
   *
   * m_data[offset..offset+length-1]에서 data 버퍼로 memcpy.
   * 블록이 calloc으로 0초기화되었으므로, write 전에 read하면 0 반환.
   *
   * 호출 체인: memory_space_impl::read_single_block → [이 함수]
   */
  void read(unsigned offset, size_t length, unsigned char *data) const {
    assert(offset + length <= BSIZE); /* [한국어] 블록 경계 초과 여부 확인 */
    memcpy(data, m_data + offset, length); /* [한국어] 블록 내 데이터를 목적지 버퍼로 복사 */
  }

  /*
   * [한국어]
   * print - 블록 내용을 지정 형식으로 파일에 출력 (디버그용)
   *
   * @format: printf 형식 문자열 (예: "%08x")
   * @fout: 출력 대상 파일 포인터
   *
   * 블록 전체를 unsigned int 단위로 순회하며 fout에 출력.
   * memory_space_impl::print에서 각 블록을 덤프할 때 호출됨.
   *
   * 호출 체인: memory_space_impl::print → [이 함수]
   */
  void print(const char *format, FILE *fout) const {
    unsigned int *i_data = (unsigned int *)m_data; /* [한국어] 바이트 배열을 4바이트 단위로 재해석하여 순회 */
    for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) { /* [한국어] 블록 내 모든 4바이트 워드 순회 */
      if (d % 1 == 0) { /* [한국어] 매 워드마다 줄바꿈 (현재 조건은 항상 참 — 디버그 목적 포맷팅) */
        fprintf(fout, "\n");
      }
      fprintf(fout, format, i_data[d]); /* [한국어] 워드를 지정 형식으로 출력 */
      fprintf(fout, " ");               /* [한국어] 워드 사이 구분 공백 출력 */
    }
    fprintf(fout, "\n");  /* [한국어] 블록 끝 줄바꿈 */
    fflush(fout);         /* [한국어] 출력 버퍼 즉시 플러시 — 시뮬레이터 크래시 시에도 출력 보장 */
  }

 private:
  unsigned m_nbytes;      /* [한국어] 사용되지 않는 레거시 필드 (실제 크기는 템플릿 파라미터 BSIZE로 고정).
                           * 설정자: 없음 (초기화 코드 없음).
                           * 읽는 자: 없음.
                           * 값 범위: 미초기화 (calloc에 의해 0으로 설정됨).
                           * 동기화: 불필요 (미사용). */
  unsigned char *m_data;  /* [한국어] BSIZE 바이트 메모리 블록의 실제 데이터 버퍼.
                           * 설정자: 생성자(calloc), write(), 복사 생성자(memcpy).
                           * 읽는 자: read(), print().
                           * 값 범위: calloc 초기화 시 전부 0; write 이후 임의 데이터.
                           * 동기화: 단일 스레드 접근이므로 별도 락 불필요. */
};

class ptx_thread_info; /* [한국어] 순환 의존 방지를 위한 전방 선언 — PTX 스레드 상태 (레지스터, 메모리 포인터 등) */
class ptx_instruction;  /* [한국어] 순환 의존 방지를 위한 전방 선언 — PTX 명령어 표현 (소스 파일/라인 정보 포함) */

/*
 * [한국어]
 * memory_space - GPU 메모리 주소 공간 순수 가상 인터페이스
 *
 * GPU의 다양한 메모리 공간(shared/local/global/param/const)에 대한
 * 통일된 추상 인터페이스를 제공한다. 호출자(instructions.cc 등)는 어떤
 * 메모리 공간인지 구체 타입을 알 필요 없이 이 인터페이스를 통해 접근한다.
 * 구체 구현은 memory_space_impl<BSIZE>이며, 메모리 공간별로 다른 BSIZE를
 * 사용하여 인스턴스화된다 (예: shared=4KB, local=8KB 등).
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 */
class memory_space {
 public:
  virtual ~memory_space() {} /* [한국어] 다형적 소멸을 위한 가상 소멸자 — 파생 클래스 자원이 올바르게 해제되도록 보장 */
  /*
   * [한국어]
   * write - 주소 공간에 데이터 기록 (워치포인트 감시 포함)
   *
   * @addr: 기록할 메모리 주소 (절대 주소)
   * @length: 기록할 바이트 수
   * @data: 기록할 데이터 소스
   * @thd: 현재 실행 중인 PTX 스레드 — 워치포인트 히트 시 알림 대상
   * @pI: 현재 실행 중인 PTX 명령어 — 워치포인트 히트 시 컨텍스트 전달
   *
   * 기능 시뮬레이션의 store 명령어(st.*)가 이 함수를 호출한다.
   * 워치포인트가 등록된 경우, 기록 범위와 겹치면 gpu::hit_watchpoint() 호출.
   */
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI) = 0;
  /*
   * [한국어]
   * write_only - 워치포인트 검사 없이 직접 블록에 데이터 기록
   *
   * @index: 블록 인덱스 (블록 번호)
   * @offset: 블록 내 오프셋
   * @length: 기록할 바이트 수
   * @data: 기록할 데이터 소스
   *
   * 시뮬레이터 초기화 단계(커널 인자 로딩, const 메모리 초기화 등)에서
   * 워치포인트 오버헤드 없이 직접 메모리를 채울 때 사용.
   */
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data) = 0;
  /*
   * [한국어]
   * read - 주소 공간에서 데이터 읽기
   *
   * @addr: 읽을 메모리 주소
   * @length: 읽을 바이트 수
   * @data: 읽은 데이터를 저장할 목적지 버퍼
   *
   * 기능 시뮬레이션의 load 명령어(ld.*)가 이 함수를 호출한다.
   * 아직 write된 적 없는 주소는 0을 반환한다.
   */
  virtual void read(mem_addr_t addr, size_t length, void *data) const = 0;
  /*
   * [한국어]
   * print - 메모리 공간 전체 내용을 파일에 덤프 (디버그용)
   *
   * @format: printf 형식 문자열
   * @fout: 출력 파일 포인터
   */
  virtual void print(const char *format, FILE *fout) const = 0;
  /*
   * [한국어]
   * set_watch - 특정 주소에 감시점(watchpoint) 등록
   *
   * @addr: 감시할 메모리 주소
   * @watchpoint: 감시점 ID (디버거에서 식별 목적)
   *
   * 등록 후, 해당 주소에 write가 발생하면 gpu::hit_watchpoint()가 호출됨.
   */
  virtual void set_watch(addr_t addr, unsigned watchpoint) = 0;
};

/*
 * [한국어]
 * memory_space_impl<BSIZE> - memory_space 인터페이스의 구체 구현
 *
 * @BSIZE: 내부 메모리 블록 크기(바이트) — 2의 거듭제곱 필수
 *   instantiation 예: memory_space_impl<32> (shared), memory_space_impl<8192> (global param)
 *
 * 해시맵(mem_map)으로 블록 인덱스(addr >> log2(BSIZE)) → mem_storage<BSIZE> 매핑을 유지.
 * 희소(sparse) 주소 공간: 실제로 접근된 블록만 메모리를 할당하므로 수 GB 공간도 효율적.
 * 블록 경계 내 접근은 fast path (단순 오프셋 계산), 블록 경계를 넘는 접근은 slow path
 * (루프를 통한 블록 단위 분할 처리).
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 */
template <unsigned BSIZE>
class memory_space_impl : public memory_space {
 public:
  /*
   * [한국어]
   * 생성자 - 메모리 공간 초기화 (이름 설정 + 해시맵 예비 할당 + log2 블록 크기 계산)
   *
   * @name: 메모리 공간 이름 (예: "shared", "local", "global") — 디버그 출력용
   * @hash_size: 해시맵 초기 버킷 수 — 예상 블록 수에 맞게 설정하면 리해시 비용 감소
   *
   * BSIZE의 log2를 비트 스캔으로 계산하여 m_log2_block_size에 저장.
   * 이후 addr >> m_log2_block_size 로 블록 인덱스, addr & (BSIZE-1) 로 오프셋 계산.
   * 구현 파일: memory.cc
   */
  memory_space_impl(std::string name, unsigned hash_size);

  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  virtual void print(const char *format, FILE *fout) const;

  virtual void set_watch(addr_t addr, unsigned watchpoint);

 private:
  /*
   * [한국어]
   * read_single_block - 단일 블록 내에서 데이터 읽기 (fast path 헬퍼)
   *
   * @blk_idx: 블록 인덱스
   * @addr: 실제 읽기 주소
   * @length: 읽을 바이트 수
   * @data: 목적지 버퍼
   *
   * 블록이 해시맵에 없으면 0 반환 (미초기화 메모리 = 0).
   * 있으면 mem_storage::read로 바이트 복사.
   * 호출자가 반드시 단일 블록 내에 들어오도록 보장해야 함 — 초과 시 예외(throw 1).
   */
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  std::string m_name;           /* [한국어] 메모리 공간 식별 이름 (예: "shared_cta0").
                                  * 설정자: 생성자에서 초기화.
                                  * 읽는 자: read_single_block() 오류 메시지, print().
                                  * 값 범위: 임의 문자열.
                                  * 동기화: 생성 후 변경 없음, 락 불필요. */
  unsigned m_log2_block_size;   /* [한국어] BSIZE의 밑 2 로그값 — 비트 이동으로 블록 인덱스/오프셋 계산.
                                  * 설정자: 생성자에서 비트 스캔으로 계산.
                                  * 읽는 자: write(), read() — addr >> m_log2_block_size, addr & (BSIZE-1).
                                  * 값 범위: 예) BSIZE=32이면 5, BSIZE=4096이면 12.
                                  * 동기화: 생성 후 불변, 락 불필요. */
  typedef mem_map<mem_addr_t, mem_storage<BSIZE> > map_t; /* [한국어] 블록 인덱스 → mem_storage 매핑 타입 별칭 */
  map_t m_data;                 /* [한국어] 실제 접근된 블록들의 해시맵 (희소 주소 공간 구현 핵심).
                                  * 설정자: write() 시 해당 블록이 없으면 자동 삽입(operator[]).
                                  * 읽는 자: read_single_block() (find), print() (순회).
                                  * 값 범위: 접근된 블록만 존재; 나머지는 맵에 없음.
                                  * 동기화: 단일 스레드 접근이므로 별도 락 불필요. */
  std::map<unsigned, mem_addr_t> m_watchpoints; /* [한국어] 감시점 ID → 감시 주소 매핑 테이블.
                                  * 설정자: set_watch()로 등록.
                                  * 읽는 자: write() — 기록 범위와 감시 주소 겹침 여부 검사.
                                  * 값 범위: 등록된 감시점만 존재; 비어있으면 write에서 검사 스킵.
                                  * 동기화: 단일 스레드 사용, 락 불필요. */
};

#endif
