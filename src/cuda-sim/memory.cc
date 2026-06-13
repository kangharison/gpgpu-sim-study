/*
 * [한국어 설명] PTX 시뮬레이션용 메모리 주소 공간 구현 (memory.cc)
 *
 * === 파일의 역할 ===
 * memory.h에서 선언된 memory_space_impl<BSIZE> 클래스의 멤버 함수를 구현한다.
 * GPU 스레드가 PTX 명령어로 메모리에 접근할 때 실제로 호출되는 read/write 로직을 담고 있다.
 * 블록 단위로 희소(sparse) 주소 공간을 관리하며, 블록 경계 내 접근(fast path)과
 * 블록 경계를 넘는 접근(slow path)을 분리하여 처리한다.
 * 파일 하단에는 단위 테스트(UNIT_TEST)와 g_print_memory_space 유틸리티 함수가 포함되어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: instructions.cc (ld/st 명령어 실행) → memory_space_impl::write/read
 *   → mem_storage::write/read (바이트 레벨 복사)
 * 기능 시뮬레이션(cuda-sim/) 계층에 속하며 타이밍 모델과 독립적으로 동작.
 * 실행 컨텍스트: 호스트 CPU, 기능 시뮬레이션 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: memory.h, libcuda/gpgpu_context.h (워치포인트 히트 시 gpgpu_context 접근),
 *   debug.h (g_debug_execution 플래그)
 * 의존받음: cuda-sim/instructions.cc, cuda-sim/ptx_sim.h (ptx_thread_info 내 m_local_mem 등)
 * 데이터 흐름: PTX 스레드 → 주소 계산 → memory_space_impl → 해시맵 블록 → 바이트 복사
 * 명시적 인스턴스화: memory_space_impl<32>, <64>, <8192>, <16384> 네 가지 크기
 *
 * === 주요 함수/구조체 요약 ===
 * memory_space_impl() 생성자: BSIZE의 log2 계산, 해시맵 초기화
 * write(): fast/slow path 분기 후 블록 기록, 워치포인트 검사
 * read(): fast/slow path 분기 후 블록 읽기
 * read_single_block(): 단일 블록 내 읽기 — 미초기화 블록은 0 반환
 * write_only(): 워치포인트 없는 직접 쓰기 (초기화용)
 * set_watch(): 감시점 주소 등록
 */
// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include "memory.h"      /* [한국어] memory_space_impl, mem_storage 등 자체 헤더 */
#include <stdlib.h>      /* [한국어] (현재 미사용 — memory.h에서 이미 포함) */
#include "../../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context::the_gpgpusim → g_the_gpu::hit_watchpoint() 접근용 */
#include "../debug.h"    /* [한국어] g_debug_execution 플래그 — 디버그 레벨에 따른 출력 제어 */

/*
 * [한국어]
 * memory_space_impl<BSIZE>::memory_space_impl - 메모리 공간 초기화
 *
 * @name: 메모리 공간 식별 이름 (예: "shared_cta0", "local_tid5")
 * @hash_size: 해시맵 초기 버킷 수 — 예상 블록 수 근사값으로 설정
 * @return: 없음 (생성자)
 *
 * 생성자는 세 가지 작업을 수행:
 *   1. m_name 설정
 *   2. 해시맵 버킷 수 사전 할당 (MEM_MAP_RESIZE) — 리해시 방지
 *   3. BSIZE의 이진 로그(log2) 계산 및 m_log2_block_size 저장
 * log2 계산: 비트 마스크(1<<n)를 좌측 이동하며 BSIZE의 유일한 1-비트 위치를 탐색.
 * BSIZE는 반드시 2의 거듭제곱이어야 하므로 비트가 정확히 하나여야 함.
 * 실행 컨텍스트: 커널 실행 준비 단계 (단일 스레드).
 *
 * 호출 체인:
 *   ptx_thread_info 초기화 → new memory_space_impl<BSIZE>() → [이 함수]
 */
template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size) {
  m_name = name;              /* [한국어] 메모리 공간 이름 저장 — 오류 메시지와 print()에서 사용 */
  MEM_MAP_RESIZE(hash_size);  /* [한국어] 해시맵 버킷 수 사전 할당 — 이후 블록 삽입 시 리해시 비용 절감 */

  m_log2_block_size = -1;     /* [한국어] log2 계산 미완료를 나타내는 초기값 (-1 = 0xFFFFFFFF) */
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) { /* [한국어] 비트 마스크를 1씩 좌측 이동하며 BSIZE의 1-비트 위치 탐색 */
    if (BSIZE & mask) {       /* [한국어] BSIZE의 n번째 비트가 1인 경우 — 2의 거듭제곱이면 정확히 한 번 진입 */
      assert(m_log2_block_size == (unsigned)-1); /* [한국어] 두 번 이상 진입 = BSIZE가 2의 거듭제곱이 아님 → abort */
      m_log2_block_size = n;  /* [한국어] log2(BSIZE) = n 저장 (예: BSIZE=32이면 n=5) */
    }
  }
  assert(m_log2_block_size != (unsigned)-1); /* [한국어] BSIZE에 1-비트가 하나도 없음 = BSIZE=0 오류 → abort */
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::write_only - 워치포인트 없는 직접 블록 쓰기
 *
 * @offset: 블록 내 바이트 오프셋
 * @index: 블록 인덱스 (블록 번호)
 * @length: 기록할 바이트 수
 * @data: 기록할 데이터 소스
 *
 * 커널 실행 전 초기화(상수 메모리 채우기, 커널 파라미터 적재 등)를 위한
 * 단순 블록 쓰기. 워치포인트 검사와 ptx_thread_info 참조가 없어 초기화 단계에서
 * 안전하게 사용 가능. 단일 블록 경계 내에 있어야 함 (경계 초과 시 mem_storage::write에서 assert).
 * 실행 컨텍스트: 커널 실행 준비 단계 (단일 스레드).
 *
 * 호출 체인:
 *   cuda-sim/ptx_sim.cc (초기화) → [이 함수] → mem_storage::write
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  m_data[index].write(offset, length, (const unsigned char *)data); /* [한국어] 지정 블록의 오프셋 위치에 데이터 직접 기록 */
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::write - PTX st 명령어 실행 시 메모리 기록
 *
 * @addr: 기록할 절대 메모리 주소
 * @length: 기록할 바이트 수
 * @data: 기록할 데이터 소스 포인터
 * @thd: 현재 실행 중인 PTX 스레드 — 워치포인트 히트 시 알림
 * @pI: 현재 실행 중인 PTX 명령어 — 워치포인트 히트 컨텍스트
 *
 * 블록 인덱스를 addr >> m_log2_block_size 비트 이동으로 계산.
 * Fast path: (addr + length)가 동일 블록 내에 있으면 단일 mem_storage::write 호출.
 * Slow path: 블록 경계를 넘으면 블록 단위로 분할하여 반복 기록.
 * Watchpoint 검사: m_watchpoints가 비어있지 않으면 각 감시 주소와 쓰기 범위 겹침 확인.
 * 실행 컨텍스트: 기능 시뮬레이션 단일 스레드.
 *
 * 호출 체인:
 *   instructions.cc (st.* 실행) → ptx_thread_info::store → [이 함수] → mem_storage::write
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  mem_addr_t index = addr >> m_log2_block_size; /* [한국어] 주소를 블록 크기로 나누어 블록 인덱스 계산 (비트 이동이 나눗셈보다 빠름) */

  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    /* [한국어] Fast path: 쓰기 범위가 현재 블록 내에 완전히 포함됨 */
    unsigned offset = addr & (BSIZE - 1); /* [한국어] 블록 내 오프셋 = 주소 하위 log2(BSIZE) 비트 */
    unsigned nbytes = length;             /* [한국어] 기록할 바이트 수 */
    m_data[index].write(offset, nbytes, (const unsigned char *)data); /* [한국어] 해당 블록에 직접 기록 — 블록 없으면 operator[]로 자동 생성 */
  } else {
    // slow route for inter-block access
    /* [한국어] Slow path: 쓰기 범위가 블록 경계를 넘음 — 블록 단위로 분할 처리 */
    unsigned nbytes_remain = length;      /* [한국어] 아직 기록하지 않은 바이트 수 */
    unsigned src_offset = 0;             /* [한국어] 소스 버퍼(data)에서의 현재 오프셋 */
    mem_addr_t current_addr = addr;      /* [한국어] 현재 기록 중인 주소 */

    while (nbytes_remain > 0) {                            /* [한국어] 모든 바이트를 기록할 때까지 반복 */
      unsigned offset = current_addr & (BSIZE - 1);        /* [한국어] 현재 주소의 블록 내 오프셋 */
      mem_addr_t page = current_addr >> m_log2_block_size; /* [한국어] 현재 주소가 속하는 블록 인덱스 */
      mem_addr_t access_limit = offset + nbytes_remain;    /* [한국어] 이번 블록에서 기록 가능한 한계 (초과 시 클램프) */
      if (access_limit > BSIZE) {
        access_limit = BSIZE; /* [한국어] 블록 끝까지만 처리 — 나머지는 다음 반복에서 처리 */
      }

      size_t tx_bytes = access_limit - offset;             /* [한국어] 이번 블록에서 실제 기록할 바이트 수 */
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]); /* [한국어] 현재 블록에 부분 기록 */

      // advance pointers
      src_offset += tx_bytes;    /* [한국어] 소스 포인터를 기록한 바이트만큼 전진 */
      current_addr += tx_bytes;  /* [한국어] 현재 기록 주소를 전진 */
      nbytes_remain -= tx_bytes; /* [한국어] 남은 바이트 수 감소 */
    }
    assert(nbytes_remain == 0); /* [한국어] 모든 바이트가 기록됨을 확인 — 루프 논리 버그 감지 */
  }
  if (!m_watchpoints.empty()) { /* [한국어] 감시점이 등록된 경우에만 검사 — 성능 최적화를 위한 조기 반환 */
    std::map<unsigned, mem_addr_t>::iterator i;
    for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) { /* [한국어] 모든 감시점을 순회하며 충돌 여부 확인 */
      mem_addr_t wa = i->second; /* [한국어] 이 감시점이 감시하는 절대 주소 */
      if (((addr <= wa) && ((addr + length) > wa)) ||
          ((addr > wa) && (addr < (wa + 4))))
        /* [한국어] 쓰기 범위[addr, addr+length)가 감시 주소 wa를 포함하거나,
         *          감시 주소[wa, wa+4) 범위 내에 addr이 있는 경우 — watchpoint 히트 처리 */
        thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
            i->first, thd, pI); /* [한국어] 시뮬레이터에 감시점 히트 알림 — 감시점 ID, 스레드, 명령어 전달 */
    }
  }
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::read_single_block - 단일 블록 내에서 데이터 읽기 (내부 헬퍼)
 *
 * @blk_idx: 읽을 블록의 인덱스
 * @addr: 실제 읽기 시작 주소 (블록 경계 검증에 사용)
 * @length: 읽을 바이트 수
 * @data: 읽은 데이터를 저장할 목적지 버퍼
 *
 * 블록 경계 초과 여부를 먼저 검증하고, 해시맵에서 블록을 찾는다.
 * 블록이 없으면(미초기화 주소) data를 0으로 채움 — 실제 GPU의 미초기화 메모리 동작과 유사.
 * 블록이 있으면 mem_storage::read로 바이트 복사.
 * 반드시 단일 블록 내에 addr~addr+length가 포함되어야 함 — 상위 함수(read)가 보장.
 * 실행 컨텍스트: 기능 시뮬레이션 단일 스레드.
 *
 * 호출 체인:
 *   memory_space_impl::read → [이 함수] → mem_storage::read
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  if ((addr + length) > (blk_idx + 1) * BSIZE) { /* [한국어] 읽기 범위가 블록 끝을 넘으면 오류 — 호출자의 범위 계산 버그 */
    printf(
        "GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
        "addr=0x%llx, length=%zu\n",
        m_name.c_str(), addr, length); /* [한국어] 정렬 위반 오류 메시지 — 메모리 공간 이름, 주소, 길이 출력 */
    printf(
        "GPGPU-Sim PTX: (addr+length)=0x%llx > 0x%llx=(index+1)*BSIZE, "
        "index=0x%llx, BSIZE=0x%x\n",
        (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE); /* [한국어] 상세 경계 값 출력 — 디버깅 보조 */
    throw 1; /* [한국어] 예외 throw(정수 1) — 호출 스택을 따라 상위에서 catch하여 시뮬레이션 중단 */
  }
  typename map_t::const_iterator i = m_data.find(blk_idx); /* [한국어] 해시맵에서 해당 블록 인덱스 검색 */
  if (i == m_data.end()) { /* [한국어] 블록이 없음 = 아직 write된 적 없는 미초기화 주소 */
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0; /* [한국어] 미초기화 메모리는 0으로 반환 — GPU 실제 동작과 유사 */
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
    /* [한국어] 미초기화 경고 출력 코드 — 현재 주석 처리됨 (너무 많은 출력 방지) */
  } else { /* [한국어] 블록이 존재하는 경우 — 저장된 데이터 읽기 */
    unsigned offset = addr & (BSIZE - 1); /* [한국어] 블록 내 오프셋 = 주소 하위 log2(BSIZE) 비트 */
    unsigned nbytes = length;             /* [한국어] 읽을 바이트 수 */
    i->second.read(offset, nbytes, (unsigned char *)data); /* [한국어] mem_storage에서 바이트 복사 */
  }
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::read - PTX ld 명령어 실행 시 메모리 읽기
 *
 * @addr: 읽을 절대 메모리 주소
 * @length: 읽을 바이트 수
 * @data: 읽은 데이터를 저장할 목적지 버퍼
 *
 * write()와 동일한 fast/slow path 구조:
 * Fast path: (addr + length)가 동일 블록 내이면 read_single_block 단 1회 호출.
 * Slow path: 블록 경계를 넘으면 블록 단위로 분할하여 반복 읽기.
 * 각 분할 읽기는 read_single_block을 통해 처리되며, 미초기화 블록은 0으로 채워짐.
 * 워치포인트 검사 없음 — read는 감시하지 않음 (write만 감시).
 * 실행 컨텍스트: 기능 시뮬레이션 단일 스레드.
 *
 * 호출 체인:
 *   instructions.cc (ld.* 실행) → ptx_thread_info::load → [이 함수]
 *     → read_single_block → mem_storage::read
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  mem_addr_t index = addr >> m_log2_block_size; /* [한국어] 읽기 시작 주소의 블록 인덱스 계산 */
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    /* [한국어] Fast path: 읽기 범위가 단일 블록 내에 완전히 포함 */
    read_single_block(index, addr, length, data); /* [한국어] 단일 블록에서 직접 읽기 */
  } else {
    // slow route for inter-block access
    /* [한국어] Slow path: 읽기 범위가 블록 경계를 넘음 — 블록 단위로 분할 */
    unsigned nbytes_remain = length;      /* [한국어] 아직 읽지 않은 바이트 수 */
    unsigned dst_offset = 0;             /* [한국어] 목적지 버퍼(data)에서의 현재 오프셋 */
    mem_addr_t current_addr = addr;      /* [한국어] 현재 읽기 중인 주소 */

    while (nbytes_remain > 0) {                            /* [한국어] 모든 바이트를 읽을 때까지 반복 */
      unsigned offset = current_addr & (BSIZE - 1);        /* [한국어] 현재 주소의 블록 내 오프셋 */
      mem_addr_t page = current_addr >> m_log2_block_size; /* [한국어] 현재 주소의 블록 인덱스 */
      mem_addr_t access_limit = offset + nbytes_remain;    /* [한국어] 이번 블록에서 읽기 가능한 한계 */
      if (access_limit > BSIZE) {
        access_limit = BSIZE; /* [한국어] 블록 끝까지만 읽기 — 나머지는 다음 반복에서 처리 */
      }

      size_t tx_bytes = access_limit - offset;             /* [한국어] 이번 블록에서 실제 읽을 바이트 수 */
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]); /* [한국어] 현재 블록에서 부분 읽기 → 목적지 버퍼에 저장 */

      // advance pointers
      dst_offset += tx_bytes;    /* [한국어] 목적지 포인터를 읽은 바이트만큼 전진 */
      current_addr += tx_bytes;  /* [한국어] 현재 읽기 주소를 전진 */
      nbytes_remain -= tx_bytes; /* [한국어] 남은 바이트 수 감소 */
    }
    assert(nbytes_remain == 0); /* [한국어] 모든 바이트가 읽혔음을 확인 */
  }
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::print - 메모리 공간 전체 내용 덤프 (디버그용)
 *
 * @format: printf 형식 문자열 (예: "%08x" — 16진수 8자리)
 * @fout: 출력 파일 포인터
 *
 * 해시맵의 모든 블록을 순회하며 블록 인덱스와 데이터를 fout에 출력.
 * 각 블록은 "<m_name> <블록인덱스>:\n<블록데이터>" 형식으로 출력.
 * g_print_memory_space 유틸리티 함수나 디버거에서 메모리 상태 확인 시 사용.
 * 실행 컨텍스트: 기능 시뮬레이션 종료 후 또는 디버그 중단점.
 *
 * 호출 체인:
 *   g_print_memory_space() / 디버그 코드 → [이 함수] → mem_storage::print
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page; /* [한국어] 해시맵 순회 이터레이터 */

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) { /* [한국어] 모든 할당된 블록 순회 */
    fprintf(fout, "%s %08llx:", m_name.c_str(), i_page->first); /* [한국어] "<메모리공간이름> <블록인덱스(16진수8자리)>:" 헤더 출력 */
    i_page->second.print(format, fout); /* [한국어] 해당 블록의 데이터를 지정 형식으로 출력 */
  }
}

/*
 * [한국어]
 * memory_space_impl<BSIZE>::set_watch - 감시점(watchpoint) 주소 등록
 *
 * @addr: 감시할 메모리 주소 — 이 주소에 write 발생 시 hit_watchpoint() 호출
 * @watchpoint: 감시점 ID (디버거에서 감시점 식별용)
 *
 * m_watchpoints 맵에 watchpoint ID → 감시 주소 쌍을 등록.
 * 이후 write() 호출 시마다 등록된 모든 감시점과 겹침 여부를 확인.
 * 실행 컨텍스트: 시뮬레이터 디버그 모드에서 호출.
 *
 * 호출 체인:
 *   사용자 디버그 명령 → memory_space::set_watch → [이 함수]
 */
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr; /* [한국어] 감시점 ID → 감시 주소 등록 — 기존 같은 ID가 있으면 덮어씀 */
}

/* [한국어] 명시적 템플릿 인스턴스화 — 링커가 이 네 가지 특수화만 사용 가능하도록 제한
 * 각 크기는 용도별로 다름:
 *   32바이트: 소규모 레지스터/파라미터 블록
 *   64바이트: 캐시라인 크기 기반 블록
 *   8192바이트(8KB): shared memory 등 중간 크기 블록
 *   16384바이트(16KB): 대형 파라미터/글로벌 메모리 블록 */
template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16 * 1024>;

/*
 * [한국어]
 * g_print_memory_space - 메모리 공간 덤프 글로벌 유틸리티 함수
 *
 * @mem: 덤프할 memory_space 포인터
 * @format: printf 형식 문자열 (기본값: "%08x")
 * @fout: 출력 파일 포인터 (기본값: stdout)
 *
 * memory_space::print에 대한 편의 래퍼 — 외부 디버그 코드에서 직접 호출 가능.
 * 실행 컨텍스트: 디버그 출력 시 임의 위치에서 호출 가능.
 *
 * 호출 체인:
 *   [디버그 코드] → [이 함수] → memory_space_impl::print
 */
void g_print_memory_space(memory_space *mem, const char *format = "%08x",
                          FILE *fout = stdout) {
  mem->print(format, fout); /* [한국어] 가상 함수를 통해 구체 구현의 print 호출 */
}

#ifdef UNIT_TEST
/* [한국어] 단위 테스트 진입점 — UNIT_TEST 매크로 정의 시에만 컴파일됨.
 * memory_space_impl<32>의 read/write 정확성을 두 가지 시나리오로 검증:
 *   1. 4바이트 주소 정렬된 쓰기/읽기 (aligned 접근)
 *   2. 1바이트 비정렬 쓰기/읽기 — 블록 경계 넘는 slow path 포함
 * 빌드: g++ -DUNIT_TEST memory.cc -o memory_test */

int main(int argc, char *argv[]) {
  int errors_found = 0;  /* [한국어] 오류 발견 카운터 — 0이면 테스트 통과 */
  memory_space *mem = new memory_space_impl<32>("test", 4); /* [한국어] 32바이트 블록, 버킷 4개로 테스트 메모리 공간 생성 */
  // write address to [address]
  /* [한국어] 테스트 1: 4바이트 단위로 각 주소에 해당 주소값을 기록 */
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4)
    mem->write(addr, 4, &addr, NULL, NULL); /* [한국어] mem[addr] = addr (4바이트 정렬 쓰기) */

  /* [한국어] 테스트 1 검증: 쓴 값과 읽은 값이 일치하는지 확인 */
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4) {
    unsigned tmp = 0;    /* [한국어] 읽기 결과를 저장할 버퍼 */
    mem->read(addr, 4, &tmp); /* [한국어] 4바이트 읽기 */
    if (tmp != addr) {        /* [한국어] 쓴 값(addr)과 읽은 값(tmp) 불일치 = 버그 */
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr);
    }
  }

  /* [한국어] 테스트 2: 1바이트 단위 쓰기 — addr % 256 + 128 값으로 덮어씀 */
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned char val = (addr + 128) % 256; /* [한국어] 0~255 범위 순환 패턴 값 생성 */
    mem->write(addr, 1, &val, NULL, NULL);   /* [한국어] 1바이트 쓰기 */
  }

  /* [한국어] 테스트 2 검증: 쓴 1바이트 패턴이 올바르게 읽히는지 확인 */
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned tmp = 0;    /* [한국어] 읽기 결과를 저장할 버퍼 (1바이트만 유효) */
    mem->read(addr, 1, &tmp); /* [한국어] 1바이트 읽기 */
    unsigned char val = (addr + 128) % 256; /* [한국어] 기대값 재계산 */
    if (tmp != val) {    /* [한국어] 불일치 = 1바이트 경계 처리 버그 */
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp,
             (unsigned)val);
    }
  }

  if (errors_found) {
    printf("SUMMARY:  ERRORS FOUND\n"); /* [한국어] 하나 이상의 오류 발견 — 테스트 실패 */
  } else {
    printf("SUMMARY: UNIT TEST PASSED\n"); /* [한국어] 모든 검증 통과 — 테스트 성공 */
  }
}

#endif
