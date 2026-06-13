// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt, Ali Bakhoda,
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

/*
 * [한국어 설명] GPU 메모리 주소 디코딩 구현 (addrdec.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍 시뮬레이터에서 64비트 선형(가상) GPU 메모리 주소를
 * DRAM 물리 위치 — 채널(chip), 뱅크(bk), 행(row), 열(col), 버스트(burst) — 로
 * 분해하는 주소 디코딩 로직을 구현한다. 실제 NVIDIA GPU 메모리 컨트롤러가 내부적으로
 * 수행하는 주소 인터리빙/뱅크 선택 로직을 소프트웨어로 재현하여, dram_t 타이밍 모델이
 * row-buffer hit/miss 및 bank conflict를 정확히 판단할 수 있도록 기반 데이터를 제공한다.
 * 또한 CONSECUTIVE(연속 비트), BITWISE_PERMUTATION(XOR 해시), IPOLY(다항식 해시),
 * RANDOM(소프트웨어 해시테이블) 등 여러 채널 매핑 정책을 구현하여 메모리 인터리빙
 * 연구를 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일은 GPGPU-Sim 타이밍 시뮬레이션 계층(gpgpu-sim/)에 속하며,
 * 기능 시뮬레이션 계층(cuda-sim/)과는 독립적이다. 실행 컨텍스트는 호스트 CPU
 * 유저스페이스이며, GPU 디바이스 코드(SASS/PTX)와 직접 관련이 없다.
 * 메모리 계층 파이프라인에서의 위치:
 *   SM(셰이더 코어) → L1 캐시 → interconnect(intersim2/NoC)
 *   → L2 캐시 서브파티션(memory_sub_partition)
 *   → memory_partition_unit → [addrdec: 선형 주소→DRAM 필드 변환]
 *   → dram_t (채널별 사이클-레벨 DRAM 타이밍 시뮬레이션)
 * addrdec_tlx()는 매 사이클 DRAM으로 내려가는 mem_fetch(메모리 요청 패킷)마다
 * memory_partition_unit::dram_cycle()에서 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 이 파일이 의존하는 모듈:
 *   - addrdec.h: linear_to_raw_address_translation 클래스 선언, addrdec_t 구조체, enum 정의
 *   - abstract_hardware_model.h: new_addr_type (64비트 주소 typedef), memory_config
 *   - option_parser.h: addrdec_setoption()의 옵션 등록 API
 *   - gpu-sim.h: gpgpu_sim 전역 시뮬레이터 컨텍스트 (전방 참조 목적)
 *   - hashing.h: bitwise_hash_function(), ipoly_hash_function() (BITWISE_PERMUTATION/IPOLY 해시)
 *   - tr1_hash_map.h: RANDOM 정책의 address_random_interleaving 해시테이블과 sweep_test용 history_map
 *
 * 이 파일에 의존하는 모듈:
 *   - l2cache.cc / memory_partition_unit: addrdec_tlx()로 채널 ID 결정, partition_address()로 L2 태그 정규화
 *   - dram.cc / dram_t: chip/bk/row/col 필드로 타이밍 시뮬레이션 수행
 *   - gpu-sim.cc: init()을 호출해 채널 수/서브파티션 수로 디코더 초기화
 *
 * 데이터 흐름:
 *   mem_fetch::m_addr (선형 주소)
 *   → addrdec_tlx() → addrdec_t { chip, bk, row, col, burst, sub_partition }
 *   → dram_t::issue() (chip 확인 후 bk/row/col로 타이밍 결정)
 *
 * === 주요 함수/구조체 요약 ===
 * linear_to_raw_address_translation()   : 생성자 — 기본 비트마스크와 ADDR_CHIP_S 초기화
 * addrdec_setoption()                   : gpgpusim.config 옵션 4개를 option_parser에 등록
 * partition_address()                   : CHIP 비트와 서브파티션 비트를 제거한 정규화 주소 반환
 * addrdec_tlx()                         : 핵심 디코딩 — 선형 주소를 DRAM 5개 필드로 분해 + 해시 적용
 * addrdec_parseoption()                 : "dramid@N;RRRBBCCC..." 형식 문자열 파싱 → 비트마스크 배열 구성
 * init()                                : 채널 수 기반 비트마스크 최종 조정, sub_partition_id_mask 생성
 * sweep_test()                          : 16MB 범위 전수 검사로 주소 중복 매핑 탐지
 * addrdec_packbits()                    : 64비트 값에서 마스크 비트 위치의 비트만 추출·압축
 * addrdec_getmasklimit()                : 마스크에서 최하위/최상위 set 비트 위치(low/high) 탐색
 * LOGB2_32()                            : 이진 탐색 비트 조작으로 floor(log2(v)) 계산
 * next_powerOf2()                       : n 이상의 최소 2의 거듭제곱 계산
 */

#include "addrdec.h"    /* [한국어] linear_to_raw_address_translation 클래스, addrdec_t 구조체,
                         * partition_index_function 열거형 선언. 이 파일의 모든 핵심 타입이 여기서 온다. */
#include <math.h>       /* [한국어] 수학 함수 헤더 — 직접 사용하지는 않지만 하위 호환성을 위해 포함.
                         * LOGB2_32()와 powli()가 자체 구현이라 실제 math.h 함수는 사용 안 함. */
#include <string.h>     /* [한국어] memset(), memcmp(), strchr() 제공.
                         * 생성자의 memset(addrdec_mklow/mkhigh),
                         * operator==의 memcmp(), addrdec_parseoption의 strchr()에 필요. */
#include "../option_parser.h"  /* [한국어] option_parser_t 타입과 option_parser_register() API.
                                * addrdec_setoption()이 gpgpusim.config의 주소 매핑 옵션을
                                * 파서에 등록할 때 사용. 상위 디렉토리의 공통 유틸리티. */
#include "gpu-sim.h"    /* [한국어] gpgpu_sim 클래스 전방 참조. 직접 멤버를 쓰지는 않으나
                         * gpu-sim.h가 포함하는 타입들(memory_config 등)을 간접적으로 의존. */
#include "hashing.h"    /* [한국어] bitwise_hash_function()과 ipoly_hash_function() 선언.
                         * addrdec_tlx()의 BITWISE_PERMUTATION, IPOLY 케이스에서 채널 해시에 사용. */

/* [한국어] 이 파일 내부에서만 사용하는 정적(static) 헬퍼 함수들의 전방 선언.
 * 파일 하단에 정의되어 있으며, C++ 링커가 외부에서 이 심볼을 볼 수 없도록 static으로 제한한다.
 * 전방 선언 순서는 먼저 사용되는 함수가 위에 오도록 배치되었다. */
static long int powli(long int x, long int y);
/* [한국어] powli — 정수 거듭제곱 계산 (x^y). init()에서 2^nchipbits를 계산해 gap을 구할 때 사용.
 * <math.h>의 pow()는 부동소수점이라 정확도 문제가 있으므로 별도 구현. */

static unsigned int LOGB2_32(unsigned int v);
/* [한국어] LOGB2_32 — 32비트 정수의 floor(log2(v))를 비트 조작으로 계산.
 * init()에서 채널 수의 로그(nchipbits), 서브파티션 수의 로그를 구할 때 사용.
 * 이진 탐색 방식이라 분기 없이 O(1)로 동작. */

static unsigned next_powerOf2(unsigned n);
/* [한국어] next_powerOf2 — n 이상의 최소 2의 거듭제곱을 계산.
 * init()에서 nextPowerOf2_m_n_channel을 구하는 데 사용.
 * 채널 수가 2의 거듭제곱이 아닐 때(gap != 0) IPOLY 해시의 모듈러 범위를 결정. */

static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low);
/* [한국어] addrdec_packbits — 64비트 값(val)에서 마스크(mask)가 1인 비트 위치의 값만
 * 추출하여 하위 비트부터 연속적으로 압축(pack)한 결과를 반환.
 * low~high 범위만 순회하도록 최적화되어 있다.
 * addrdec_tlx()와 partition_address()에서 각 DRAM 필드를 추출하는 핵심 비트 연산. */

static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low);
/* [한국어] addrdec_getmasklimit — 비트마스크에서 set 비트의 최하위 위치(low)와
 * 최상위 위치+1(high)을 찾아 반환.
 * init()에서 각 필드 마스크의 유효 범위를 addrdec_mklow/mkhigh에 저장할 때 호출.
 * addrdec_packbits()가 전체 64비트 대신 low~high 범위만 순회하도록 최적화를 제공. */

/*
 * [한국어]
 * linear_to_raw_address_translation - 기본 생성자
 *
 * @return: 없음 (생성자)
 *
 * 주소 디코더 객체를 안전한 초기 상태로 설정한다. init()이 호출되기 전까지
 * addrdec_tlx()를 호출해서는 안 된다. 이 생성자가 설정하는 값들은 init()에서
 * gpgpusim.config의 채널 수와 옵션에 따라 덮어씌워진다.
 *
 * 초기화 내용:
 *   - addrdec_option: NULL (명시적 옵션 문자열 없음, 기본 마스크 사용)
 *   - ADDR_CHIP_S: 10 (비트 10이 채널 필드 시작 위치 — 캐시라인 크기 64바이트=2^6,
 *                      워드 단위=2^4 고려한 경험적 기본값)
 *   - addrdec_mklow: 전체 0 (모든 필드의 유효 비트 하한 = bit 0)
 *   - addrdec_mkhigh: 전체 64 (모든 필드의 유효 비트 상한 = bit 63)
 *   - addrdec_mask[CHIP=0]: 0x0000000000001C00 = bits[12:10] — 채널 선택 3비트
 *   - addrdec_mask[BK=1]:   0x0000000000000300 = bits[9:8] — 뱅크 2비트
 *   - addrdec_mask[ROW=2]:  0x000000000FFF0000 = bits[27:16] — 행 12비트
 *   - addrdec_mask[COL=3]:  0x000000000000E0FF = bits[13:13]+bits[7:0] — 열 11비트
 *   - addrdec_mask[BURST=4]:0x000000000000000F = bits[3:0] — 버스트 4비트
 * 이 기본 마스크들은 초기 GPGPU-Sim 개발 시 사용된 구성으로,
 * 실제 시뮬레이션에서는 gpgpu_mem_address_mask와 addrdec_option에 의해 재설정된다.
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이터 초기화 시 단 1회 실행.
 *
 * 호출 체인:
 *   gpu-sim.cc::memory_config 생성 → [이 생성자] → (이후) addrdec_setoption() → init()
 */
linear_to_raw_address_translation::linear_to_raw_address_translation() {
  addrdec_option = NULL;    /* [한국어] 명시적 주소 매핑 옵션 문자열 없음 — init()에서
                             * addrdec_option != NULL 인지 검사하여 parseoption을 조건부 호출 */
  ADDR_CHIP_S = 10;         /* [한국어] 채널(chip) 비트 시작 위치 = bit 10 (기본값).
                             * 2^10 = 1024바이트 = 16개 캐시라인 단위로 채널을 나눔.
                             * init()의 gpgpu_mem_address_mask에 따라 5~16으로 재설정될 수 있음. */
  memset(addrdec_mklow, 0, N_ADDRDEC);   /* [한국어] 각 필드의 low 경계를 0으로 초기화.
                                          * N_ADDRDEC=5개 필드 × 1바이트(unsigned char) = 5바이트.
                                          * init() 전까지의 안전한 기본값. */
  memset(addrdec_mkhigh, 64, N_ADDRDEC); /* [한국어] 각 필드의 high 경계를 64로 초기화.
                                          * 64 = 64비트 주소의 전체 비트 폭.
                                          * 실제 마스크 분석 후 addrdec_getmasklimit()이 덮어씀. */
  addrdec_mask[0] = 0x0000000000001C00;  /* [한국어] CHIP(채널) 마스크 초기값.
                                          * 0x1C00 = bits[12:10] = 0b0001_1100_0000_0000.
                                          * bit 10~12의 3비트로 최대 8채널을 구분. */
  addrdec_mask[1] = 0x0000000000000300;  /* [한국어] BK(뱅크) 마스크 초기값.
                                          * 0x0300 = bits[9:8] = 0b0000_0011_0000_0000.
                                          * bit 8~9의 2비트로 4뱅크를 구분. */
  addrdec_mask[2] = 0x000000000FFF0000;  /* [한국어] ROW(행) 마스크 초기값.
                                          * 0x0FFF0000 = bits[27:16] = 12비트로 4096개 행 구분.
                                          * DRAM row address는 일반적으로 13~15비트이나 여기선 12비트 사용. */
  addrdec_mask[3] = 0x000000000000E0FF;  /* [한국어] COL(열) 마스크 초기값.
                                          * 0xE0FF = bits[13:13]+bits[7:0] — 비연속 비트 필드.
                                          * 버스트 비트(bit 0~3)와 캐시라인 내 오프셋 비트를 포함. */
  addrdec_mask[4] = 0x000000000000000F;  /* [한국어] BURST(버스트) 마스크 초기값.
                                          * 0x000F = bits[3:0] = 4비트로 16-word 버스트 구분.
                                          * DDR 버스트 길이(BL=4~16)에 대응. */
}

/*
 * [한국어]
 * addrdec_setoption - gpgpusim.config 옵션을 option_parser에 등록
 *
 * @opp: option_parser 핸들 — gpu-sim.cc의 gpgpu_sim_config::reg_options()가 전달하는 전역 파서
 * @return: 없음 (void)
 *
 * gpgpusim.config 파일에서 주소 매핑 관련 옵션 4개를 option_parser에 등록한다.
 * 이 함수 자체는 값을 파싱하지 않으며, 파서가 나중에 config 파일을 읽을 때
 * 각 멤버 변수(addrdec_option, run_test, gpgpu_mem_address_mask, memory_partition_indexing)에
 * 값을 직접 저장하도록 포인터를 등록한다.
 * 등록된 옵션:
 *   -gpgpu_mem_addr_mapping          : 비트 단위 채널/뱅크/행/열 매핑 문자열 (NULL=기본값)
 *   -gpgpu_mem_addr_test             : sweep_test() 실행 여부 (기본 0=false)
 *   -gpgpu_mem_address_mask          : 사전 정의 비트마스크 세트 선택 (기본 0=레거시 마스크)
 *   -gpgpu_memory_partition_indexing : 채널 해시 정책 선택 (기본 0=CONSECUTIVE)
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이터 옵션 등록 단계에서 1회 실행.
 * 에러: option_parser_register 내부에서 중복 옵션 등록 시 assert로 중단.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim_config::reg_options() → [이 함수] → option_parser_register() × 4
 */
void linear_to_raw_address_translation::addrdec_setoption(option_parser_t opp) {
  option_parser_register(opp, "-gpgpu_mem_addr_mapping", OPT_CSTR,
                         &addrdec_option,
                         "mapping memory address to dram model {dramid@<start "
                         "bit>;<memory address map>}",
                         NULL);
  /* [한국어] -gpgpu_mem_addr_mapping: 주소 매핑 문자열 옵션 등록.
   * OPT_CSTR = C 문자열 타입. 파서가 읽으면 addrdec_option 포인터에 문자열 주소가 저장됨.
   * 형식: "dramid@<N>;<비트 문자열>" (예: "dramid@10;RRRRBBDDDCCCC").
   * 기본값 NULL이면 gpgpu_mem_address_mask로 결정되는 사전 정의 마스크를 사용. */

  option_parser_register(
      opp, "-gpgpu_mem_addr_test", OPT_BOOL, &run_test,
      "run sweep test to check address mapping for aliased address", "0");
  /* [한국어] -gpgpu_mem_addr_test: sweep_test() 활성화 플래그 등록.
   * OPT_BOOL = boolean. 기본값 "0" = false.
   * true로 설정하면 init() 말미에서 sweep_test()가 호출되어
   * 16MB 주소 공간의 모든 4바이트 경계 주소에 대한 매핑 중복 여부를 검사. */

  option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32,
                         &gpgpu_mem_address_mask,
                         "0 = old addressing mask, 1 = new addressing mask, 2 "
                         "= new add. mask + flipped bank sel and chip sel bits",
                         "0");
  /* [한국어] -gpgpu_mem_address_mask: 사전 정의된 비트마스크 세트 선택 옵션.
   * OPT_INT32 = 32비트 정수. 기본값 "0" = 레거시 마스크(ADDR_CHIP_S=10, BK=bits[9:8]).
   * init()의 switch(gpgpu_mem_address_mask) 분기에서 각 케이스별 사전 정의 마스크를 로드.
   * addrdec_option이 NULL이 아니면 이 옵션 대신 addrdec_option 문자열이 우선 적용됨. */

  option_parser_register(
      opp, "-gpgpu_memory_partition_indexing", OPT_UINT32,
      &memory_partition_indexing,
      "0 = no indexing, 1 = bitwise xoring, 2 = IPoly, 3 = custom indexing",
      "0");
  /* [한국어] -gpgpu_memory_partition_indexing: 채널 매핑 해시 정책 선택 옵션.
   * OPT_UINT32 = 부호 없는 32비트 정수. 기본값 "0" = CONSECUTIVE (해시 없음).
   * 파서가 읽은 정수값이 partition_index_function enum으로 캐스트되어 저장됨.
   * addrdec_tlx()의 switch(memory_partition_indexing)에서 해시 알고리즘을 분기. */
}

/*
 * [한국어]
 * partition_address - 서브파티션 비트를 제거한 채널-레벨 정규화 주소 반환
 *
 * @addr:   정규화할 64비트 선형 GPU 메모리 주소 (mem_fetch::m_addr)
 * @return: CHIP 비트와 sub_partition_id_mask 비트를 제거한 나머지 주소.
 *          두 선형 주소가 같은 파티션의 같은 내부 위치를 가리키는지 비교할 때 사용.
 *
 * L2 캐시 태그 비교 시, 서로 다른 파티션(채널) 또는 서로 다른 서브파티션에 매핑된
 * 주소라도 파티션 내부 오프셋이 같으면 같은 캐시 라인에 해당한다.
 * 이 함수는 채널 구분 비트(addrdec_mask[CHIP])와 서브파티션 구분 비트(sub_partition_id_mask)를
 * 모두 제거한 '파티션 내부 주소'를 반환함으로써 L2 태그를 정규화한다.
 * sweep_test()에서 두 선형 주소가 같은 파티션 내부 위치로 중복 매핑되는지 검증할 때도 사용.
 *
 * gap==0 경로 (채널 수가 2의 거듭제곱): 단순히 addrdec_packbits로 CHIP+sub_partition 비트 제거.
 * gap!=0 경로 (채널 수가 2의 거듭제곱 아님): addrdec_tlx()와 동일한 모듈러 분해 후 sub_partition 비트 제거.
 *   1단계: addr >> ADDR_CHIP_S의 몫(quotient) = 채널 비트 제거 후 상위 비트
 *   2단계: addr & ((1<<ADDR_CHIP_S)-1) = ADDR_CHIP_S 하위 비트 (채널 선택과 무관)
 *   3단계: 두 부분을 이어붙인 뒤 sub_partition_id_mask 비트 추가 제거
 *
 * 실행 컨텍스트: const 함수, memory_sub_partition::cache_cycle()에서 매 사이클 호출될 수 있음.
 * 에러 처리: 별도 에러 반환 없음 — 잘못된 마스크 설정 시 무결성은 sweep_test()가 보장.
 *
 * 호출 체인:
 *   memory_sub_partition::cache_cycle() → [이 함수] → addrdec_packbits()
 *   sweep_test() → [이 함수] (검증 목적)
 */
new_addr_type linear_to_raw_address_translation::partition_address(
    new_addr_type addr) const {
  if (!gap) {
    /* [한국어] 채널 수가 2의 거듭제곱인 경우 (gap==0):
     * addrdec_mask[CHIP]과 sub_partition_id_mask의 합집합 비트들을 반전한 마스크로
     * addrdec_packbits를 호출하면, 해당 비트들을 제외한 나머지 비트들만 압축 추출된다.
     * 결과적으로 채널 ID와 서브파티션 ID 비트가 모두 제거된 '내부 주소'가 반환된다. */
    return addrdec_packbits(~(addrdec_mask[CHIP] | sub_partition_id_mask), addr,
                            64, 0);
    /* [한국어] ~(addrdec_mask[CHIP] | sub_partition_id_mask) : CHIP과 서브파티션 마스크의
     * 비트를 모두 0으로 반전 → packbits가 이 마스크의 1인 비트(= 채널/서브파티션이 아닌 비트)만 추출.
     * high=64, low=0으로 전체 64비트 범위를 순회. */
  } else {
    // see addrdec_tlx for explanation
    /* [한국어] 채널 수가 2의 거듭제곱이 아닌 경우 (gap!=0):
     * addrdec_tlx()의 gap 처리와 동일한 방식으로 채널 비트를 제거한 뒤
     * sub_partition_id_mask 비트를 추가로 제거한다.
     * 아래 주석은 addrdec_tlx()의 gap 분기 설명과 대응됨. */
    unsigned long long int partition_addr;  /* [한국어] 재조합할 파티션 내부 주소 임시 변수 */
    partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    /* [한국어] (addr >> ADDR_CHIP_S) / m_n_channel = 채널 비트를 모듈러로 제거한 상위 비트 부분.
     * << ADDR_CHIP_S로 다시 상위 비트 위치로 올려서 하위 비트와 결합할 준비. */
    partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
    /* [한국어] addr & ((1<<ADDR_CHIP_S)-1) = ADDR_CHIP_S 아래의 하위 비트들 (채널 선택과 무관).
     * OR 연산으로 상위 부분(채널 제거됨)과 하위 부분을 이어붙임. */
    // remove the part of address that constributes to the sub partition ID
    partition_addr =
        addrdec_packbits(~sub_partition_id_mask, partition_addr, 64, 0);
    /* [한국어] 재조합된 partition_addr에서 서브파티션 ID 비트(sub_partition_id_mask)를 추가 제거.
     * ~sub_partition_id_mask = 서브파티션 비트를 0으로 반전 → packbits가 서브파티션 비트 제외한 값 반환. */
    return partition_addr;
  }
}

/*
 * [한국어]
 * addrdec_tlx - 핵심 주소 디코딩 함수: 선형 주소 → DRAM 물리 위치(addrdec_t) 분해
 *
 * @addr: 디코딩할 64비트 선형 GPU 메모리 주소 (mem_fetch가 가진 m_addr)
 * @tlx:  결과를 채울 addrdec_t 포인터 (호출자 스택의 임시 변수, 입력값은 무시됨)
 * @return: 없음 (void). tlx의 chip/bk/row/col/burst/sub_partition 필드에 결과 저장.
 *
 * 이 함수는 두 단계로 동작한다:
 *   1단계 — 비트마스크 추출 (gap 여부에 따라 분기):
 *     - gap==0 (채널 수가 2의 거듭제곱): addrdec_packbits()로 마스크 기반 비트 추출.
 *       64비트 주소의 각 DRAM 필드 비트를 직접 추출하여 tlx->chip/bk/row/col/burst에 저장.
 *     - gap!=0 (채널 수가 2의 거듭제곱 아님): 모듈러 분해 방식 사용.
 *       MSB = addr >> ADDR_CHIP_S, 채널 = MSB % m_n_channel,
 *       나머지 주소 = (MSB / m_n_channel) << ADDR_CHIP_S | LSB.
 *       이렇게 재조합된 rest_of_addr에서 BK/ROW/COL/BURST를 packbits로 추출.
 *   2단계 — 채널 해시 적용 (memory_partition_indexing에 따라 분기):
 *     - CONSECUTIVE: 1단계 결과 그대로 사용 (no-op).
 *     - BITWISE_PERMUTATION: bitwise_hash_function()으로 chip 재매핑.
 *     - IPOLY: ipoly_hash_function()으로 sub_partition 단위 재매핑, 함수 즉시 return.
 *     - RANDOM: 소프트웨어 해시테이블(address_random_interleaving)로 sub_partition 재매핑, 즉시 return.
 *     - CUSTOM: 사용자 구현 대기 (현재 no-op).
 *   단계2 이후 (IPOLY/RANDOM이 아닌 경우):
 *     sub_partition = chip * m_n_sub_partition_in_channel + (bk & sub_partition_addr_mask).
 *
 * 실행 컨텍스트: const 함수. memory_partition_unit::dram_cycle()에서 매 사이클 호출.
 * 재진입: RANDOM 케이스에서 address_random_interleaving에 쓰기가 발생하므로
 *         동일 chip 주소에 대한 동시 호출 시 데이터 레이스 가능. 실제로는 단일 스레드 사용.
 * 에러: assert로 범위 검사(chip < m_n_channel, sub_partition < 전체 서브파티션 수).
 *
 * 호출 체인:
 *   memory_partition_unit::dram_cycle() → [이 함수]
 *     → addrdec_packbits() (필드 추출)
 *     → bitwise_hash_function() / ipoly_hash_function() (해시 적용)
 *   이후: tlx 필드 → dram_t::issue() (타이밍 시뮬레이션)
 */
void linear_to_raw_address_translation::addrdec_tlx(new_addr_type addr,
                                                    addrdec_t *tlx) const {
  unsigned long long int addr_for_chip, rest_of_addr, rest_of_addr_high_bits;
  /* [한국어] 세 임시 변수:
   * addr_for_chip         : gap!=0 경로에서 모듈러 연산으로 추출한 채널 인덱스
   * rest_of_addr          : gap!=0 경로에서 채널 비트를 제거하고 재조합한 나머지 주소
   * rest_of_addr_high_bits: ADDR_CHIP_S 위의 상위 비트 부분 (해시 함수의 입력 고비트) */

  if (!gap) {
    /* [한국어] === gap==0 경로: 채널 수가 2의 거듭제곱 ===
     * 비트마스크 방식으로 각 DRAM 필드를 직접 추출한다.
     * addrdec_mask[CHIP]의 비트 위치에서 addr의 비트를 추출 → 연속 정수로 압축.
     * addrdec_mkhigh/mklow는 마스크의 유효 범위를 미리 계산해 두어 순회 범위를 최소화. */
    tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP],
                                 addrdec_mklow[CHIP]);
    /* [한국어] CHIP 필드 추출: addrdec_mask[CHIP]에서 1인 비트 위치의 addr 비트들을
     * 압축하여 채널 인덱스(0 ~ m_n_channel-1)를 얻는다. */
    tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
                               addrdec_mklow[BK]);
    /* [한국어] BK(뱅크) 필드 추출: addrdec_mask[BK]의 비트 위치에서 addr의 비트를 추출. */
    tlx->row = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW],
                                addrdec_mklow[ROW]);
    /* [한국어] ROW(행) 필드 추출: addrdec_mask[ROW]의 비트 위치에서 행 인덱스를 추출. */
    tlx->col = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL],
                                addrdec_mklow[COL]);
    /* [한국어] COL(열) 필드 추출: addrdec_mask[COL]의 비트 위치에서 열 인덱스를 추출. */
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
    /* [한국어] BURST(버스트 오프셋) 필드 추출: 버스트 전송 내에서의 위치를 나타내는 하위 비트들. */
    rest_of_addr_high_bits =
        (addr >> (ADDR_CHIP_S + (log2channel + log2sub_partition)));
    /* [한국어] 해시 함수용 상위 비트 계산:
     * addr에서 CHIP 비트와 서브파티션 비트 위 상위 비트들만 남긴다.
     * ADDR_CHIP_S + log2channel + log2sub_partition = 채널+서브파티션 비트 폭 합산 지점.
     * 이 상위 비트들이 BITWISE_PERMUTATION/IPOLY 해시의 "상위 주소" 입력으로 사용된다. */

  } else {
    // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
    // - extract chip address using modulus of MSBs
    // - recreate the rest of the address by stitching the quotient of MSBs and
    // the LSBs
    /* [한국어] === gap!=0 경로: 채널 수가 2의 거듭제곱이 아님 ===
     * 비트마스크 방식으로는 2의 거듭제곱이 아닌 채널 수를 올바르게 처리할 수 없다.
     * 대신 ADDR_CHIP_S를 기준으로 주소를 두 부분(MSB, LSB)으로 나눈 뒤,
     * MSB를 m_n_channel로 나눈 나머지(remainder)를 채널 인덱스로,
     * 몫(quotient)을 재조합하여 나머지 주소(rest_of_addr)를 만든다.
     *
     * 예: addr=0x1234, ADDR_CHIP_S=4, m_n_channel=3
     *   MSB = 0x1234 >> 4 = 0x123
     *   채널 = 0x123 % 3 = 0
     *   몫  = 0x123 / 3 = 0x61
     *   rest_of_addr = (0x61 << 4) | (0x1234 & 0xF) = 0x614
     *   이 rest_of_addr에서 BK/ROW/COL/BURST를 마스크 방식으로 추출 */
    addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel;
    /* [한국어] 채널 인덱스 = (상위 비트) mod (채널 수).
     * >> ADDR_CHIP_S로 하위 ADDR_CHIP_S개 비트를 제거한 뒤
     * % m_n_channel로 채널 인덱스를 0~(m_n_channel-1) 범위로 구한다. */
    rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    /* [한국어] (상위 비트 / 채널 수) << ADDR_CHIP_S = 채널 비트를 제거한 상위 부분을
     * 다시 ADDR_CHIP_S만큼 올려 하위 비트와 이어붙일 자리를 만든다. */
    rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel);
    /* [한국어] 해시 함수용 상위 비트 = shift 이전 몫(quotient).
     * gap==0 경로의 rest_of_addr_high_bits와 같은 역할: 해시 입력용 고비트. */
    rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
    /* [한국어] OR로 ADDR_CHIP_S 아래의 하위 비트(= 캐시라인/버스트 오프셋 등)를 이어붙임.
     * (1 << ADDR_CHIP_S) - 1 = ADDR_CHIP_S 아래 비트들의 마스크. */

    tlx->chip = addr_for_chip;    /* [한국어] 모듈러로 구한 채널 인덱스를 저장 */
    tlx->bk = addrdec_packbits(addrdec_mask[BK], rest_of_addr,
                               addrdec_mkhigh[BK], addrdec_mklow[BK]);
    /* [한국어] 재조합된 rest_of_addr에서 뱅크 비트 추출.
     * gap!=0 경로에서는 원래 addr 대신 rest_of_addr를 입력으로 사용하는 점이 gap==0과의 차이. */
    tlx->row = addrdec_packbits(addrdec_mask[ROW], rest_of_addr,
                                addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
    /* [한국어] rest_of_addr에서 행 비트 추출. */
    tlx->col = addrdec_packbits(addrdec_mask[COL], rest_of_addr,
                                addrdec_mkhigh[COL], addrdec_mklow[COL]);
    /* [한국어] rest_of_addr에서 열 비트 추출. */
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], rest_of_addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
    /* [한국어] rest_of_addr에서 버스트 오프셋 비트 추출. */
  }

  switch (memory_partition_indexing) {
    /* [한국어] 2단계: 채널 해시 정책에 따라 tlx->chip (및 sub_partition)을 재매핑.
     * 1단계에서 추출한 chip 값을 입력으로, 주소 비트를 추가로 혼합하여
     * 채널 간 부하 분산 품질을 높인다. */

    case CONSECUTIVE:
      // Do nothing
      /* [한국어] CONSECUTIVE: 해시 없음. 1단계에서 추출한 chip/bk/row/col/burst를 그대로 사용.
       * 연속된 선형 주소가 연속된 채널에 순서대로 매핑된다. 가장 단순하고 예측 가능한 방식. */
      break;

    case BITWISE_PERMUTATION: {
      /* [한국어] BITWISE_PERMUTATION: XOR 기반 비트 치환 해시.
       * rest_of_addr_high_bits(채널 비트 위의 상위 주소 비트)와 현재 chip 값을
       * bitwise_hash_function()에 입력하여 채널 인덱스를 재매핑한다.
       * 연속 접근이 다양한 채널로 분산되어 채널 편중을 줄이는 효과가 있다.
       * gap==0만 지원 (assert로 강제): 모듈러 분해와 XOR 해시를 조합하는 로직이 없음. */
      assert(!gap);
      /* [한국어] gap!=0이면 BITWISE_PERMUTATION을 사용할 수 없음: assert 위반으로 즉시 중단.
       * gap==0, 즉 채널 수가 2의 거듭제곱인 경우에만 이 해시가 정의됨. */
      tlx->chip =
          bitwise_hash_function(rest_of_addr_high_bits, tlx->chip, m_n_channel);
      /* [한국어] bitwise_hash_function(상위비트, 현재채널, 채널수):
       * 상위 비트와 현재 채널 인덱스를 XOR 혼합하여 새 채널 인덱스를 반환.
       * hashing.h에 정의된 함수. 결과는 [0, m_n_channel) 범위. */
      assert(tlx->chip < m_n_channel);
      /* [한국어] 해시 결과가 유효한 채널 범위 안에 있는지 확인. 이를 위반하면 잘못된 hashing.h 구현. */
      break;
    }

    case IPOLY: {
      /* [한국어] IPOLY: 비가역 다항식(Irreducible Polynomial) 기반 해시.
       * 채널 + 서브파티션을 합친 "총 서브파티션 인덱스" 단위로 해시를 적용한다.
       * GF(2^k) 위 다항식 나머지 연산으로, BITWISE_PERMUTATION보다 분산 품질이 높다.
       * 이 케이스는 sub_partition을 직접 설정하고 return하므로 함수 하단의 공통 sub_partition 계산을 건너뜀. */
      // assert(!gap);
      unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
      /* [한국어] sub_partition_addr_mask: m_n_sub_partition_in_channel이 2의 거듭제곱이므로
       * -1로 하위 비트 마스크를 만든다. bk & mask = 서브파티션 선택용 뱅크 하위 비트. */
      unsigned sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                               (tlx->bk & sub_partition_addr_mask);
      /* [한국어] 현재 chip과 bk의 하위 비트를 조합하여 전체 서브파티션 인덱스(0 ~ 전체 서브파티션 수)를 계산.
       * chip × 채널당서브파티션수 = 채널 기여분, bk & mask = 채널 내 서브파티션 위치. */
      sub_partition = ipoly_hash_function(
          rest_of_addr_high_bits, sub_partition,
          nextPowerOf2_m_n_channel * m_n_sub_partition_in_channel);
      /* [한국어] ipoly_hash_function(상위비트, 현재서브파티션, 2의거듭제곱범위):
       * 다항식 해시로 sub_partition을 재매핑. nextPowerOf2_m_n_channel × 서브파티션수 = 해시 범위.
       * 이 범위는 실제 서브파티션 수보다 크거나 같으므로 gap!=0이면 아래에서 모듈러 보정이 필요. */

      if (gap)  // if it is not 2^n partitions, then take modular
        sub_partition =
            sub_partition % (m_n_channel * m_n_sub_partition_in_channel);
      /* [한국어] gap!=0이면 해시 결과가 실제 서브파티션 수를 초과할 수 있으므로 모듈러 보정.
       * m_n_channel × m_n_sub_partition_in_channel = 실제 전체 서브파티션 수. */

      tlx->chip = sub_partition / m_n_sub_partition_in_channel;
      /* [한국어] 재매핑된 sub_partition에서 채널 인덱스를 역산:
       * 서브파티션 인덱스 / 채널당 서브파티션 수 = 채널 인덱스. */
      tlx->sub_partition = sub_partition;
      /* [한국어] sub_partition 필드를 직접 설정하고 return. 함수 하단의 공통 계산을 건너뜀. */
      assert(tlx->chip < m_n_channel);
      /* [한국어] 재매핑된 채널 인덱스가 유효 범위 안에 있는지 확인. */
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      /* [한국어] 서브파티션 인덱스가 전체 서브파티션 수 이내인지 확인. */
      return;
      /* [한국어] 함수 하단의 공통 sub_partition 계산 코드를 건너뛰고 즉시 반환. */
      break;
    }

    case RANDOM: {
      // This is an unrealistic hashing using software hashtable
      // we generate a random set for each memory address and save the value in
      /* [한국어] RANDOM: 소프트웨어 해시테이블(address_random_interleaving)을 이용한 완전 랜덤 매핑.
       * 각 "chip_address" 단위로 처음 방문 시 rand()로 서브파티션 ID를 무작위 할당하고,
       * 이후 동일 chip_address가 들어오면 테이블에서 저장된 값을 재사용한다.
       * 이 방식은 실제 하드웨어에 구현 불가능하며, 이상적인 완전 분산의 상한선을 연구용으로 모델링.
       * address_random_interleaving는 const 함수 내에서 쓰기가 일어나므로 mutable로 선언되어 있음. */
      new_addr_type chip_address = (addr >> (ADDR_CHIP_S - log2sub_partition));
      /* [한국어] chip_address: 서브파티션 단위로 정렬된 주소 상위 비트.
       * ADDR_CHIP_S - log2sub_partition 비트를 오른쪽 시프트하여 서브파티션 경계 단위로
       * 주소를 구분하는 키를 만든다. 같은 서브파티션에 매핑되어야 하는 주소들이
       * 동일한 chip_address를 갖도록 설계된 키. */
      tr1_hash_map<new_addr_type, unsigned>::const_iterator got =
          address_random_interleaving.find(chip_address);
      /* [한국어] 해시테이블에서 이 chip_address에 해당하는 서브파티션 ID를 조회.
       * tr1_hash_map: C++ TR1/unordered_map 호환 타입 (tr1_hash_map.h 참고).
       * const_iterator를 사용하지만 조회 실패 시 아래에서 삽입이 일어남(const 함수의 예외). */
      if (got == address_random_interleaving.end()) {
        /* [한국어] 처음 방문하는 chip_address: 새 서브파티션 ID를 무작위로 할당. */
        unsigned new_chip_id =
            rand() % (m_n_channel * m_n_sub_partition_in_channel);
        /* [한국어] rand()로 0 ~ (전체 서브파티션 수 - 1) 범위의 랜덤 서브파티션 ID 생성.
         * rand()는 srand(1)로 시드가 고정되어 있어 (init() 말미 참조) 재현 가능. */
        address_random_interleaving[chip_address] = new_chip_id;
        /* [한국어] 해시테이블에 저장하여 이후 같은 chip_address에 대해 동일 매핑 보장. */
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
        /* [한국어] sub_partition ID를 채널당 서브파티션 수로 나누어 채널 인덱스 역산. */
        tlx->sub_partition = new_chip_id;
        /* [한국어] sub_partition 필드를 전체 서브파티션 인덱스로 직접 설정. */
      } else {
        /* [한국어] 이미 방문한 chip_address: 테이블에서 저장된 서브파티션 ID 재사용. */
        unsigned new_chip_id = got->second;  /* [한국어] 테이블에서 기존에 할당된 서브파티션 ID 조회 */
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;  /* [한국어] 채널 인덱스 역산 */
        tlx->sub_partition = new_chip_id;                        /* [한국어] 서브파티션 ID 설정 */
      }

      assert(tlx->chip < m_n_channel);
      /* [한국어] 채널 인덱스가 유효 범위 [0, m_n_channel) 안에 있는지 확인. */
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      /* [한국어] 서브파티션 인덱스가 전체 서브파티션 수 이내인지 확인. */
      return;
      /* [한국어] sub_partition과 chip이 이미 설정되었으므로 함수 하단의 공통 계산을 건너뛰고 즉시 반환. */
      break;
    }

    case CUSTOM:
      /* No custom set function implemented */
      // Do you custom index here
      /* [한국어] CUSTOM: 사용자 정의 해시 알고리즘 확장 포인트.
       * 현재 구현이 없으며, 연구자가 이 위치에 새로운 채널 매핑 로직을 삽입한다.
       * 구현 없이 이 케이스를 선택하면 CONSECUTIVE와 동일하게 동작(1단계 결과 그대로 사용). */
      break;

    default:
      assert("\nUndefined set index function.\n" && 0);
      /* [한국어] 정의되지 않은 memory_partition_indexing 값이 들어오면 assert 위반으로 즉시 중단.
       * 문자열 && 0: C++에서 문자열 포인터는 항상 truthy이므로 && 0은 항상 false → assert 실패.
       * 에러 메시지를 assert 문에 포함시키는 관용 패턴. */
      break;
  }

  // combine the chip address and the lower bits of DRAM bank address to form
  // the subpartition ID
  /* [한국어] 공통 sub_partition 계산 경로 (IPOLY, RANDOM은 이미 return했으므로 여기 도달 안 함):
   * CONSECUTIVE와 BITWISE_PERMUTATION의 경우 sub_partition을 bk의 하위 비트로 결정. */
  unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
  /* [한국어] m_n_sub_partition_in_channel이 2의 거듭제곱이므로 -1이 하위 비트 마스크가 됨.
   * 예: 2개 서브파티션 → mask=0x1, 4개 → mask=0x3. */
  tlx->sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                       (tlx->bk & sub_partition_addr_mask);
  /* [한국어] 전체 서브파티션 인덱스 = 채널 인덱스 × 채널당서브파티션수 + 채널 내 서브파티션 위치.
   * 채널 내 서브파티션 위치는 bk의 하위 log2(m_n_sub_partition_in_channel)비트로 결정.
   * 이로써 tlx의 모든 필드(chip, bk, row, col, burst, sub_partition)가 채워진다. */
}
}

/*
 * [한국어]
 * addrdec_parseoption - 옵션 문자열을 파싱하여 addrdec_mask 배열 구성
 *
 * @option: gpgpusim.config의 -gpgpu_mem_addr_mapping 값 문자열.
 *          형식: "dramid@<N>;<매핑 문자열>" 또는 "<매핑 문자열>"
 *          예: "dramid@10;RRRRRRRRRRRRRBBDDDCCCCCCCCC"
 * @return: 없음 (void). addrdec_mask[CHIP..BURST], ADDR_CHIP_S를 직접 수정.
 *
 * 이 함수는 두 단계로 문자열을 처리한다:
 *   1단계 — dramid@N 파싱: "dramid@" 토큰이 있으면 N을 ADDR_CHIP_S에 저장.
 *            없으면 ADDR_CHIP_S = -1 (매핑 문자열 자체에 채널 비트 위치가 인코딩됨).
 *   2단계 — 매핑 문자열 파싱: 문자열의 각 문자를 bit 63부터 순서대로 처리.
 *            문자마다 해당 비트 위치를 해당 필드 마스크에 OR로 추가.
 *
 * 문자 의미:
 *   D/d: CHIP(채널) 마스크 비트 설정 — dramid@N이 없을 때만 허용
 *   B/b: BK(뱅크) 마스크 비트 설정
 *   R/r: ROW(행) 마스크 비트 설정
 *   C/c: COL(열) 마스크 비트 설정
 *   S/s: BURST 및 COL 마스크 비트 동시 설정 (버스트 오프셋 = 열 하위 비트와 중첩)
 *   0  : 이 비트 무시 (사용하지 않는 주소 비트)
 *   |/./ (공백): 구분자, ofs 변경 없이 스킵
 *   기타: stderr에 에러 출력
 *
 * 실행 컨텍스트: init() 내부 또는 option_parser 콜백에서 1회 실행.
 * 에러: 매핑 문자열 길이가 정확히 64비트를 커버하지 않으면 assert 위반.
 *
 * 호출 체인:
 *   init() → (addrdec_option != NULL이면) → [이 함수]
 */
void linear_to_raw_address_translation::addrdec_parseoption(
    const char *option) {
  unsigned int dramid_start = 0;  /* [한국어] dramid@N에서 파싱될 N 값 저장용 임시 변수. 기본값 0. */
  int dramid_parsed = sscanf(option, "dramid@%d", &dramid_start);
  /* [한국어] sscanf로 옵션 문자열에서 "dramid@" 접두사 뒤의 정수를 파싱.
   * 파싱 성공(1개 항목 읽음) → dramid_parsed == 1, dramid_start에 채널 시작 비트 저장.
   * 파싱 실패("dramid@" 없음) → dramid_parsed == 0 (또는 EOF 반환). */
  if (dramid_parsed == 1) {
    ADDR_CHIP_S = dramid_start;   /* [한국어] dramid@N이 있으면 N을 채널 비트 시작 위치로 사용.
                                   * 예: "dramid@8" → ADDR_CHIP_S = 8. */
  } else {
    ADDR_CHIP_S = -1;             /* [한국어] dramid@N이 없으면 ADDR_CHIP_S = -1.
                                   * 이 경우 매핑 문자열의 'D' 문자가 직접 채널 비트 위치를 지정.
                                   * init()에서 ADDR_CHIP_S == -1이면 n_channel이 2의 거듭제곱인지 assert. */
  }

  const char *cmapping = strchr(option, ';');
  /* [한국어] strchr로 ';' 구분자를 찾아 "dramid@N;" 이후의 매핑 문자열 시작 포인터를 찾음.
   * 예: "dramid@10;RRRBB..." → cmapping은 ';' 위치를 가리킴. */
  if (cmapping == NULL) {
    cmapping = option;            /* [한국어] ';' 없으면 전체 문자열이 매핑 문자열.
                                   * dramid@N 없이 바로 매핑 문자열만 전달된 경우. */
  } else {
    cmapping += 1;                /* [한국어] ';' 다음 문자부터 매핑 문자열 시작. */
  }

  addrdec_mask[CHIP] = 0x0;   /* [한국어] 파싱 시작 전 모든 마스크를 0으로 초기화.
                                * 이전에 생성자나 init()에서 설정된 기본값을 덮어씀. */
  addrdec_mask[BK] = 0x0;     /* [한국어] BK 마스크 초기화 */
  addrdec_mask[ROW] = 0x0;    /* [한국어] ROW 마스크 초기화 */
  addrdec_mask[COL] = 0x0;    /* [한국어] COL 마스크 초기화 */
  addrdec_mask[BURST] = 0x0;  /* [한국어] BURST 마스크 초기화 */

  int ofs = 63;
  /* [한국어] 비트 오프셋 카운터. 매핑 문자열의 첫 문자가 bit 63에 대응하고,
   * 유효 문자(D/B/R/C/S/0)가 나올 때마다 1씩 감소하여 bit 62, 61, ... 0 순으로 처리.
   * 구분자(|/. /공백)는 ofs를 감소시키지 않는다. */
  while ((*cmapping) != '\0') {
    /* [한국어] 매핑 문자열을 끝까지 1문자씩 처리하는 주 루프. */
    switch (*cmapping) {
      case 'D':
      case 'd':
        /* [한국어] D/d: 현재 비트 위치(ofs)를 CHIP(채널) 마스크에 추가.
         * dramid@N이 이미 설정된 경우(dramid_parsed==1)에는 D를 사용할 수 없음:
         * dramid@N 방식과 D 문자 방식은 채널 비트 위치 지정 방법이 상호 배타적. */
        assert(dramid_parsed != 1);
        /* [한국어] dramid@N이 파싱되었으면(=1) 'D' 사용 금지 — assert 위반으로 중단. */
        addrdec_mask[CHIP] |= (1ULL << ofs);
        /* [한국어] 현재 비트 위치 ofs에 해당하는 비트를 CHIP 마스크에 OR로 세트. */
        ofs--;
        /* [한국어] 다음 유효 문자는 한 비트 아래(ofs-1)에 대응. */
        break;
      case 'B':
      case 'b':
        /* [한국어] B/b: 현재 비트 위치를 BK(뱅크) 마스크에 추가. */
        addrdec_mask[BK] |= (1ULL << ofs);
        /* [한국어] ofs 비트를 뱅크 마스크에 세트. */
        ofs--;
        break;
      case 'R':
      case 'r':
        /* [한국어] R/r: 현재 비트 위치를 ROW(행) 마스크에 추가. */
        addrdec_mask[ROW] |= (1ULL << ofs);
        /* [한국어] ofs 비트를 행 마스크에 세트. */
        ofs--;
        break;
      case 'C':
      case 'c':
        /* [한국어] C/c: 현재 비트 위치를 COL(열) 마스크에 추가. */
        addrdec_mask[COL] |= (1ULL << ofs);
        /* [한국어] ofs 비트를 열 마스크에 세트. */
        ofs--;
        break;
      case 'S':
      case 's':
        /* [한국어] S/s: 현재 비트를 BURST 마스크와 COL 마스크에 동시 추가.
         * DRAM 버스트 전송에서 버스트 내 오프셋 비트는 열(COL) 주소의 하위 비트와 중첩된다.
         * 즉, 동일한 주소 비트가 버스트 오프셋이기도 하고 열 주소의 일부이기도 하다. */
        addrdec_mask[BURST] |= (1ULL << ofs);
        /* [한국어] ofs 비트를 버스트 마스크에 세트. */
        addrdec_mask[COL] |= (1ULL << ofs);
        /* [한국어] 동일한 ofs 비트를 COL 마스크에도 세트 (버스트 비트 = 열 하위 비트). */
        ofs--;
        break;
      // ignore bit
      case '0':
        /* [한국어] 0: 이 비트 위치는 어떤 필드에도 속하지 않는 무시 비트.
         * ofs만 감소시켜 다음 비트로 넘어간다. 보통 상위 미사용 비트나 예약 비트에 사용. */
        ofs--;
        break;
      // ignore character
      case '|':
      case ' ':
      case '.':
        /* [한국어] |/공백/.: 가독성을 위한 구분자 문자. ofs를 변경하지 않고 스킵.
         * 예: "RRRR|BBBB|CCCC" 형식으로 필드 그룹을 시각적으로 구분할 수 있음. */
        break;
      default:
        /* [한국어] 정의되지 않은 문자: stderr에 오류 메시지를 출력하고 계속 진행.
         * assert로 즉시 중단하지 않는 이유는 일부 오류를 경고로만 처리하는 레거시 동작 유지. */
        fprintf(
            stderr,
            "ERROR: Invalid address mapping character '%c' in option '%s'\n",
            *cmapping, option);
    }
    cmapping += 1;
    /* [한국어] 다음 문자로 포인터 전진. '\0' 종료 조건은 while 조건에서 확인. */
  }

  if (ofs != -1) {
    /* [한국어] 매핑 문자열 처리 후 ofs가 -1이 아니면 64비트 전체를 커버하지 못한 것.
     * ofs == -1은 bit 63부터 bit 0까지 정확히 64개의 유효 문자(D/B/R/C/S/0)가 처리되었음을 의미.
     * 63 - ofs = 처리된 유효 비트 수. ofs != -1이면 부족한 것. */
    fprintf(stderr,
            "ERROR: Invalid address mapping length (%d) in option '%s'\n",
            63 - ofs, option);
    /* [한국어] 처리된 비트 수(63-ofs)와 원본 옵션 문자열을 stderr에 출력. */
    assert(ofs == -1);
    /* [한국어] 길이 오류는 복구 불가능한 설정 오류 — assert 위반으로 시뮬레이터 즉시 중단. */
  }
}

/*
 * [한국어]
 * init - 채널/서브파티션 수를 기반으로 주소 디코더 내부 상태 최종 초기화
 *
 * @n_channel:                   DRAM 채널(메모리 파티션) 총 수. gpgpusim.config -gpgpu_n_mem.
 * @n_sub_partition_in_channel:  채널당 L2 서브파티션 수. 2의 거듭제곱이어야 함.
 * @return: 없음 (void). 클래스 멤버 변수들을 모두 최종값으로 설정.
 *
 * 이 함수는 addrdec_setoption() 이후, addrdec_tlx() 호출 이전에 반드시 1회 호출되어야 한다.
 * 처리 순서:
 *   1) log2channel, log2sub_partition, m_n_channel, m_n_sub_partition_in_channel 등 기본 멤버 설정.
 *   2) gap 계산: n_channel이 2의 거듭제곱이 아니면 gap!=0 → 모듈러 분해 경로 활성화.
 *   3) gpgpu_mem_address_mask에 따라 사전 정의 비트마스크(ADDR_CHIP_S + addrdec_mask 세트) 선택.
 *   4) addrdec_option != NULL이면 addrdec_parseoption()으로 옵션 문자열 파싱 (3단계 덮어씀).
 *   5) ADDR_CHIP_S != -1이고 gap==0이면: BK/ROW/COL 마스크를 nchipbits만큼 shift하여
 *      CHIP 마스크 공간을 확보하고, ADDR_CHIP_S 위치에 nchipbits개 비트로 CHIP 마스크 설정.
 *      ADDR_CHIP_S == -1이면: n_channel이 2의 거듭제곱인지 assert.
 *   6) m_n_sub_partition_in_channel이 2의 거듭제곱인지 assert.
 *   7) addrdec_getmasklimit()로 각 필드의 addrdec_mklow/mkhigh 계산.
 *   8) 마스크 정보를 stdout에 출력 (디버깅/검증용).
 *   9) sub_partition_id_mask 구성: addrdec_mask[BK]의 하위 n_sub_partition_log2 비트.
 *   10) run_test이면 sweep_test() 호출.
 *   11) RANDOM 정책이면 srand(1)로 랜덤 시드 초기화.
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이터 초기화 시 1회 실행.
 * 에러: 각 단계에서 assert로 잘못된 설정 탐지.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::init() → [이 함수]
 *     → addrdec_parseoption() (조건부)
 *     → addrdec_getmasklimit() × 5
 *     → sweep_test() (조건부)
 */
void linear_to_raw_address_translation::init(
    unsigned int n_channel, unsigned int n_sub_partition_in_channel) {
  unsigned i;                        /* [한국어] CHIP 마스크 비트 설정 루프의 인덱스 변수 */
  unsigned long long int mask;       /* [한국어] 비트 조작용 임시 마스크 변수 */
  unsigned int nchipbits = ::LOGB2_32(n_channel);
  /* [한국어] nchipbits = floor(log2(n_channel)) = 채널 수를 표현하는 데 필요한 비트 수.
   * 예: 4채널 → nchipbits=2, 8채널 → nchipbits=3.
   * n_channel이 2의 거듭제곱이 아닐 때는 아래에서 nchipbits++로 보정. */
  log2channel = nchipbits;           /* [한국어] 채널 비트 폭 저장. BITWISE_PERMUTATION/IPOLY에서 사용. */
  log2sub_partition = ::LOGB2_32(n_sub_partition_in_channel);
  /* [한국어] log2sub_partition = floor(log2(n_sub_partition_in_channel)) = 서브파티션 비트 폭.
   * n_sub_partition_in_channel은 2의 거듭제곱이어야 하므로 이 값은 정확한 log2임. */
  m_n_channel = n_channel;           /* [한국어] 채널 수 저장. addrdec_tlx()의 모듈러 연산에 사용. */
  m_n_sub_partition_in_channel = n_sub_partition_in_channel;
  /* [한국어] 채널당 서브파티션 수 저장. sub_partition_id_mask 계산과 sub_partition 결정에 사용. */
  nextPowerOf2_m_n_channel = ::next_powerOf2(n_channel);
  /* [한국어] n_channel 이상의 최소 2의 거듭제곱 계산.
   * 예: n_channel=6 → nextPowerOf2=8. IPOLY 해시의 범위로 사용. */
  m_n_sub_partition_total = n_channel * n_sub_partition_in_channel;
  /* [한국어] 전체 서브파티션 총 수 = 채널 수 × 채널당 서브파티션 수.
   * 통계/스케줄링에서 전체 서브파티션 범위를 알 때 참조. */

  gap = (n_channel - ::powli(2, nchipbits));
  /* [한국어] gap = n_channel - 2^nchipbits.
   * n_channel이 2의 거듭제곱이면 gap==0 (예: 4채널 → 4-4=0).
   * n_channel이 2의 거듭제곱이 아니면 gap>0 (예: 6채널 → 6-4=2).
   * gap==0: 비트마스크 추출 방식 사용 가능.
   * gap!=0: 모듈러 분해 방식 필요. */
  if (gap) {
    nchipbits++;
    /* [한국어] 채널 수가 2의 거듭제곱이 아니면 비트 1개를 추가로 확보해야 함.
     * 예: 6채널 → log2(6)=2이지만 3비트(0~7)는 있어야 6채널을 인덱싱 가능.
     * 이후 BK/ROW/COL 마스크 shift에 이 값이 사용된다. */
  }
  switch (gpgpu_mem_address_mask) {
    /* [한국어] gpgpu_mem_address_mask 값에 따라 사전 정의된 ADDR_CHIP_S와 비트마스크 세트를 선택.
     * 이 switch 결과는 addrdec_option != NULL이면 addrdec_parseoption()이 덮어씀. */

    case 0:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      /* [한국어] case 0: 레거시 마스크. ADDR_CHIP_S=10, 구형 비트 레이아웃.
       * BK=bits[9:8](0x0300), ROW=bits[26:13](0x07FFE000), COL=bits[12:13]+bits[7:0](0x1CFF).
       * "added 2row bits"는 초기 구현 대비 ROW 비트가 2개 추가된 이력을 나타냄. */
      ADDR_CHIP_S = 10;
      addrdec_mask[CHIP] = 0x0000000000000000;  /* [한국어] 채널 마스크 0: init()의 5단계에서 ADDR_CHIP_S 위치에 삽입 */
      addrdec_mask[BK] = 0x0000000000000300;    /* [한국어] bit[9:8] = 2비트 뱅크, 4뱅크 구분 */
      addrdec_mask[ROW] = 0x0000000007FFE000;   /* [한국어] bit[26:13] = 14비트 행, 16384개 행 */
      addrdec_mask[COL] = 0x0000000000001CFF;   /* [한국어] bit[12:11]+bit[7:0] = 비연속 10비트 열 */
      break;
    case 1:
      /* [한국어] case 1: 신형 마스크. ADDR_CHIP_S=13.
       * BK=bits[12:11](0x1800), ROW=bits[26:13](0x07FFE000), COL=bits[10:0](0x07FF). */
      ADDR_CHIP_S = 13;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;    /* [한국어] bit[12:11] = 2비트 뱅크 */
      addrdec_mask[ROW] = 0x0000000007FFE000;   /* [한국어] bit[26:13] = 14비트 행 */
      addrdec_mask[COL] = 0x00000000000007FF;   /* [한국어] bit[10:0] = 11비트 열 */
      break;
    case 2:
      /* [한국어] case 2: ADDR_CHIP_S=11. case 1과 동일한 BK/ROW/COL 마스크이나
       * 채널 시작 위치가 bit 11로 낮아짐. 더 많은 캐시라인 그레인으로 채널 분리. */
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 3:
      /* [한국어] case 3: ADDR_CHIP_S=11, ROW 마스크가 case 1보다 1비트 더 넓음(0x0FFFE000).
       * bit[27:13] = 15비트 행 → 32768개 행. 대용량 DRAM 구성에 적합. */
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x000000000FFFE000;   /* [한국어] bit[27:13] = 15비트 행 (case 1보다 1비트 더) */
      addrdec_mask[COL] = 0x00000000000007FF;
      break;

    case 14:
      /* [한국어] case 14: ADDR_CHIP_S=14. bit 14부터 채널 비트 시작. */
      ADDR_CHIP_S = 14;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 15:
      /* [한국어] case 15: ADDR_CHIP_S=15. bit 15부터 채널 비트 시작. */
      ADDR_CHIP_S = 15;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 16:
      /* [한국어] case 16: ADDR_CHIP_S=16. bit 16부터 채널 비트 시작 (64KB 페이지 단위). */
      ADDR_CHIP_S = 16;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 6:
      /* [한국어] case 6: ADDR_CHIP_S=6. bit 6(64바이트=캐시라인 크기)부터 채널 비트 시작.
       * 캐시라인 단위로 채널을 나눠 인접한 캐시라인이 서로 다른 채널에 분산됨. */
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 5:
      /* [한국어] case 5: ADDR_CHIP_S=5. bit 5(32바이트)부터 채널 비트 시작. */
      ADDR_CHIP_S = 5;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 100:
      /* [한국어] case 100: ADDR_CHIP_S=1, BK=bits[1:0](0x0003), COL=bits[12:2](0x1FFC).
       * 매우 낮은 ADDR_CHIP_S(bit 1)와 뱅크가 최하위 2비트에 위치하는 특수 구성.
       * 인접한 워드들이 서로 다른 뱅크에 분산되는 최대 뱅크 인터리빙 효과를 연구용으로 사용. */
      ADDR_CHIP_S = 1;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;    /* [한국어] bit[1:0] = 2비트 뱅크 (최하위 비트) */
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;   /* [한국어] bit[12:2] = 11비트 열 (bit 1,0은 BK에 할당됨) */
      break;
    case 103:
      /* [한국어] case 103: ADDR_CHIP_S=3, case 100과 동일한 BK/COL 마스크.
       * 채널 시작을 bit 3으로 올려 byte-level보다 word-level 채널 분리. */
      ADDR_CHIP_S = 3;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;    /* [한국어] bit[1:0] = 2비트 뱅크 */
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;   /* [한국어] bit[12:2] = 11비트 열 */
      break;
    case 106:
      /* [한국어] case 106: ADDR_CHIP_S=6, BK/ROW/COL은 case 6(=case 1)과 동일.
       * case 6과 마스크가 같지만 별도 번호로 관리하는 레거시 호환 케이스. */
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 160:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      /* [한국어] case 160: 레거시 case 0와 동일한 BK/ROW/COL 마스크이나 ADDR_CHIP_S=6.
       * case 0(ADDR_CHIP_S=10)보다 채널 시작을 낮춰 더 세밀한 채널 인터리빙 적용.
       * break가 없어 default로 fall-through됨 — 의도적인 레거시 동작. */
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;    /* [한국어] bit[9:8] = 2비트 뱅크 (case 0 스타일) */
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;   /* [한국어] case 0 스타일 비연속 열 마스크 */

    default:
      break;
      /* [한국어] default: 알 수 없는 gpgpu_mem_address_mask 값이거나 case 160 fall-through.
       * 별도 처리 없이 이전에 설정된 값(또는 생성자 기본값)을 유지. */
  }

  if (addrdec_option != NULL) addrdec_parseoption(addrdec_option);
  /* [한국어] addrdec_option이 설정된 경우(-gpgpu_mem_addr_mapping이 비-NULL),
   * 위 switch에서 설정한 기본 마스크를 addrdec_parseoption()이 덮어씀.
   * 사용자 정의 비트 레이아웃이 사전 정의 마스크보다 우선한다. */

  if (ADDR_CHIP_S != -1) {
    /* [한국어] ADDR_CHIP_S가 명시적으로 설정된 경우 (dramid@N이 있거나 switch에서 결정됨):
     * BK/ROW/COL 마스크를 ADDR_CHIP_S 위치에서 nchipbits만큼 shift하여
     * CHIP 마스크가 들어갈 공간을 확보한 뒤, CHIP 마스크를 삽입한다.
     * 개념적으로: [기존 BK/ROW/COL 비트] → [CHIP 비트 삽입] → [기존 BK/ROW/COL 비트] */
    if (!gap) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      /* [한국어] gap==0 (채널 수가 2의 거듭제곱) 시에만 마스크 shift를 수행.
       * gap!=0이면 모듈러 분해를 사용하므로 마스크 위치 조정이 불필요. */
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      /* [한국어] mask = ADDR_CHIP_S 아래 비트들의 마스크.
       * 예: ADDR_CHIP_S=10 → mask=0x3FF (bit 0~9).
       * 이 마스크를 기준으로 각 필드 마스크를 "아래 부분"과 "위 부분"으로 분리하여 처리. */
      addrdec_mask[BK] =
          ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
      /* [한국어] BK 마스크 shift:
       * (addrdec_mask[BK] & ~mask) = ADDR_CHIP_S 위의 BK 비트들
       * << nchipbits = CHIP 비트 폭만큼 올려서 CHIP 공간 확보
       * | (addrdec_mask[BK] & mask) = ADDR_CHIP_S 아래의 BK 비트들(변경 없음)
       * 결과: CHIP 비트 자리를 건너뛰어 BK 마스크가 올바른 위치에 배치됨. */
      addrdec_mask[ROW] = ((addrdec_mask[ROW] & ~mask) << nchipbits) |
                          (addrdec_mask[ROW] & mask);
      /* [한국어] ROW 마스크도 동일한 방식으로 shift. */
      addrdec_mask[COL] = ((addrdec_mask[COL] & ~mask) << nchipbits) |
                          (addrdec_mask[COL] & mask);
      /* [한국어] COL 마스크도 동일한 방식으로 shift. */

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits); i++) {
        /* [한국어] CHIP 마스크를 ADDR_CHIP_S부터 ADDR_CHIP_S+nchipbits-1까지의 비트들로 구성.
         * 예: ADDR_CHIP_S=10, nchipbits=2 → bit 10과 bit 11을 CHIP 마스크에 세트. */
        mask = (unsigned long long int)1 << i;
        /* [한국어] i번째 비트만 1인 단일 비트 마스크 생성. */
        addrdec_mask[CHIP] |= mask;
        /* [한국어] CHIP 마스크에 i번째 비트를 OR로 추가. */
      }
    }  // otherwise, no need to change the masks
    /* [한국어] gap!=0이면 마스크 변경 불필요 — 모듈러 분해 경로가 채널 비트를 별도 처리. */
  } else {
    // make sure n_channel is power of two when explicit dram id mask is used
    assert((n_channel & (n_channel - 1)) == 0);
    /* [한국어] ADDR_CHIP_S == -1 (dramid@N 없이 D 문자로 채널 비트를 직접 지정한 경우):
     * 이 경우 n_channel이 반드시 2의 거듭제곱이어야 함.
     * (n_channel & (n_channel-1)) == 0: 2의 거듭제곱 판별 비트 트릭.
     * 2의 거듭제곱이 아닌 채널 수에 D 문자 방식을 사용하면 매핑이 불완전해지므로 assert로 방지. */
  }
  // make sure m_n_sub_partition_in_channel is power of two
  assert((m_n_sub_partition_in_channel & (m_n_sub_partition_in_channel - 1)) ==
         0);
  /* [한국어] m_n_sub_partition_in_channel이 반드시 2의 거듭제곱이어야 함을 검증.
   * sub_partition_id_mask 계산에서 -1로 마스크를 만들고 log2를 구하는 로직이
   * 2의 거듭제곱을 전제로 하기 때문에 이 조건이 반드시 성립해야 한다. */

  addrdec_getmasklimit(addrdec_mask[CHIP], &addrdec_mkhigh[CHIP],
                       &addrdec_mklow[CHIP]);
  /* [한국어] CHIP 마스크에서 유효 비트 범위(low/high) 계산 → addrdec_mklow[CHIP], addrdec_mkhigh[CHIP] 저장.
   * addrdec_packbits() 호출 시 전체 64비트 대신 이 범위만 순회하도록 최적화. */
  addrdec_getmasklimit(addrdec_mask[BK], &addrdec_mkhigh[BK],
                       &addrdec_mklow[BK]);
  /* [한국어] BK 마스크 유효 범위 계산 → addrdec_mklow[BK], addrdec_mkhigh[BK] 저장. */
  addrdec_getmasklimit(addrdec_mask[ROW], &addrdec_mkhigh[ROW],
                       &addrdec_mklow[ROW]);
  /* [한국어] ROW 마스크 유효 범위 계산 → addrdec_mklow[ROW], addrdec_mkhigh[ROW] 저장. */
  addrdec_getmasklimit(addrdec_mask[COL], &addrdec_mkhigh[COL],
                       &addrdec_mklow[COL]);
  /* [한국어] COL 마스크 유효 범위 계산 → addrdec_mklow[COL], addrdec_mkhigh[COL] 저장. */
  addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST],
                       &addrdec_mklow[BURST]);
  /* [한국어] BURST 마스크 유효 범위 계산 → addrdec_mklow[BURST], addrdec_mkhigh[BURST] 저장. */

  printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
         addrdec_mask[CHIP], addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
  /* [한국어] CHIP 마스크 값과 유효 비트 범위(high/low)를 stdout에 출력.
   * 시뮬레이터 실행 시 주소 매핑 설정을 사용자가 확인할 수 있도록 하는 진단 출력. */
  printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask[BK],
         addrdec_mkhigh[BK], addrdec_mklow[BK]);
  /* [한국어] BK 마스크 진단 출력. */
  printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask[ROW],
         addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
  /* [한국어] ROW 마스크 진단 출력. */
  printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask[COL],
         addrdec_mkhigh[COL], addrdec_mklow[COL]);
  /* [한국어] COL 마스크 진단 출력. */
  printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
         addrdec_mask[BURST], addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
  /* [한국어] BURST 마스크 진단 출력. */

  // create the sub partition ID mask (for removing the sub partition ID from
  // the partition address)
  /* [한국어] sub_partition_id_mask 구성: BK 마스크의 하위 n_sub_partition_log2 비트.
   * 이 마스크는 partition_address()에서 서브파티션 구분 비트를 제거하여 L2 태그를 정규화할 때 사용. */
  sub_partition_id_mask = 0;           /* [한국어] 초기값 0 (서브파티션 마스킹 없음). */
  if (m_n_sub_partition_in_channel > 1) {
    /* [한국어] 서브파티션이 2개 이상인 경우에만 마스크 구성 필요.
     * 1개면 서브파티션 구분 비트가 없으므로 마스크는 0 그대로 유지. */
    unsigned n_sub_partition_log2 = LOGB2_32(m_n_sub_partition_in_channel);
    /* [한국어] 서브파티션 수의 log2 = 서브파티션 ID를 표현하는 비트 수.
     * 예: 2개 서브파티션 → log2=1, 4개 → log2=2. */
    unsigned pos = 0;                  /* [한국어] 수집한 서브파티션 비트 수 카운터. */
    for (unsigned i = addrdec_mklow[BK]; i < addrdec_mkhigh[BK]; i++) {
      /* [한국어] BK 마스크의 유효 범위(mklow~mkhigh)에서 낮은 비트부터 순회.
       * BK 마스크에서 n_sub_partition_log2개의 최하위 set 비트를 sub_partition_id_mask에 추가. */
      if ((addrdec_mask[BK] & ((unsigned long long int)1 << i)) != 0) {
        /* [한국어] i번째 비트가 BK 마스크에 속하는지 확인. */
        sub_partition_id_mask |= ((unsigned long long int)1 << i);
        /* [한국어] BK 마스크의 최하위 set 비트부터 n_sub_partition_log2개를 sub_partition_id_mask에 추가. */
        pos++;
        /* [한국어] 수집한 비트 수 증가. */
        if (pos >= n_sub_partition_log2) break;
        /* [한국어] 필요한 비트 수(n_sub_partition_log2개)를 모두 수집했으면 루프 종료.
         * 나머지 BK 비트들은 순수 뱅크 선택용이므로 서브파티션 마스크에 포함하지 않음. */
      }
    }
  }
  printf("sub_partition_id_mask = %016llx\n", sub_partition_id_mask);
  /* [한국어] 서브파티션 ID 마스크 값을 stdout에 출력. 주소 매핑 검증에 사용. */

  if (run_test) {
    sweep_test();
    /* [한국어] run_test가 true이면 sweep_test()를 호출해 16MB 범위의 주소 중복 매핑 전수 검사.
     * 초기화 직후에 한 번만 실행되며, 잘못된 매핑 설정을 조기에 탐지한다. */
  }

  if (memory_partition_indexing == RANDOM) srand(1);
  /* [한국어] RANDOM 정책이 선택된 경우 랜덤 시드를 1로 고정.
   * srand(1)로 시드를 고정하면 동일한 시뮬레이션 실행 시 항상 동일한 랜덤 채널 매핑이 생성되어
   * 재현 가능한 결과를 보장한다. RANDOM 정책 외에는 srand를 호출하지 않아 시스템 기본 시드 사용. */
}

#include "../tr1_hash_map.h"
/* [한국어] tr1_hash_map 템플릿 정의 포함.
 * C++ TR1 unordered_map의 시뮬레이터 내부 래퍼로, RANDOM 정책의 address_random_interleaving과
 * sweep_test()의 history_map에 사용된다.
 * 파일 중간에 include하는 이유: operator==, operator<, hash_addrdec_t가 이 헤더보다 앞에 나오므로
 * hash_addrdec_t가 정의된 뒤 tr1_hash_map에서 해시 펑터로 사용될 수 있도록 순서를 배치. */

/*
 * [한국어]
 * operator== - addrdec_t 동등 비교 연산자
 *
 * @x: 첫 번째 addrdec_t 레코드 (chip/bk/row/col/burst/sub_partition 포함)
 * @y: 두 번째 addrdec_t 레코드
 * @return: x와 y의 모든 필드가 동일하면 true, 하나라도 다르면 false.
 *
 * sweep_test()에서 history_map<addrdec_t, ...>의 키 비교에 사용된다.
 * memcmp로 구조체 전체를 바이트 단위로 비교하므로 패딩 바이트에 의한
 * 오탐지 가능성이 있으나, addrdec_t는 unsigned 필드만 있어 패딩이 없다고 가정.
 *
 * 호출 체인:
 *   sweep_test() → history_map.find() → [이 operator==]
 */
bool operator==(const addrdec_t &x, const addrdec_t &y) {
  return (memcmp(&x, &y, sizeof(addrdec_t)) == 0);
  /* [한국어] memcmp(&x, &y, sizeof(addrdec_t)):
   * x와 y의 메모리 표현을 sizeof(addrdec_t) 바이트만큼 바이트 단위 비교.
   * == 0이면 모든 바이트가 동일(모든 필드가 같음) → true 반환.
   * 패딩 없는 POD 구조체이므로 memcmp가 올바르게 동작. */
}

/*
 * [한국어]
 * operator< - addrdec_t 사전순 비교 연산자 (정렬/맵 키 비교용)
 *
 * @x: 비교 왼쪽 operand
 * @y: 비교 오른쪽 operand
 * @return: x가 y보다 "작으면" true (chip → bk → row → col → burst 순으로 비교).
 *
 * tr1_hash_map_ismap == 1인 경우(std::map 사용 시) 키 정렬에 필요.
 * 비교 순서: chip → bk → row → col → burst (sub_partition은 비교 안 함).
 * x.chip >= y.chip이면 즉시 false를 반환하는 조기 종료 방식.
 * 주의: "x가 y보다 크거나 같으면 false"를 연속으로 검사하는 방식이라
 * 정확한 사전순이 아닌 일종의 lexicographic 비교임 (chip이 같으면 bk 비교 등).
 *
 * 호출 체인:
 *   sweep_test() (tr1_hash_map_ismap==1 시) → [이 operator<]
 */
bool operator<(const addrdec_t &x, const addrdec_t &y) {
  if (x.chip >= y.chip)
    return false;   /* [한국어] x.chip이 y.chip 이상이면 x는 y보다 작지 않음 → false. */
  else if (x.bk >= y.bk)
    return false;   /* [한국어] chip은 같고 bk가 y.bk 이상이면 → false. */
  else if (x.row >= y.row)
    return false;   /* [한국어] chip, bk 같고 row가 y.row 이상이면 → false. */
  else if (x.col >= y.col)
    return false;   /* [한국어] chip, bk, row 같고 col이 y.col 이상이면 → false. */
  else if (x.burst >= y.burst)
    return false;   /* [한국어] chip, bk, row, col 같고 burst가 y.burst 이상이면 → false. */
  else
    return true;    /* [한국어] 모든 비교를 통과한 경우: x가 y보다 작음 → true. */
}

/*
 * [한국어]
 * hash_addrdec_t - addrdec_t를 키로 사용하는 해시맵을 위한 해시 펑터(functor) 클래스
 *
 * tr1_hash_map<addrdec_t, ..., hash_addrdec_t>에서 키의 해시 값을 계산할 때 사용된다.
 * sweep_test()에서 history_map의 해시 펑터로 전달된다.
 * XOR 해시: 모든 필드를 XOR하면 빠르고 간단한 해시 값을 얻을 수 있으나,
 * 충돌 가능성이 있어 성능 저하가 발생할 수 있다. 테스트 목적으로는 충분.
 */
class hash_addrdec_t {
 public:
  /*
   * [한국어]
   * operator() - addrdec_t의 해시 값 계산
   *
   * @x: 해시할 addrdec_t 레코드
   * @return: chip ^ bk ^ row ^ col ^ burst를 XOR한 size_t 해시 값.
   *
   * 각 필드를 XOR하는 단순 해시 함수. 동일한 addrdec_t에 대해 항상 같은 값 반환.
   * sub_partition은 chip과 bk로 결정되므로 해시에서 제외해도 충돌 증가 미미.
   */
  size_t operator()(const addrdec_t &x) const {
    return (x.chip ^ x.bk ^ x.row ^ x.col ^ x.burst);
    /* [한국어] chip, bk, row, col, burst 5개 필드를 XOR하여 해시 값 생성.
     * XOR은 교환법칙·결합법칙을 만족하므로 필드 순서에 무관하지만,
     * 서로 다른 필드 배열이 같은 XOR 값을 가질 수 있어 이상적인 해시는 아님.
     * sweep_test()의 진단 도구용으로는 충분한 분산을 제공. */
  }
};

// a simple sweep test to ensure that two linear addresses are not mapped to the
// same raw address
/*
 * [한국어]
 * sweep_test - 주소 매핑 비트마스크의 유효성을 전수 검사하는 진단 함수
 *
 * @return: 없음 (void). 오류 발견 시 printf + abort()로 즉시 종료.
 *
 * 0부터 16MB(16×1024×1024) 범위의 모든 4바이트 경계 주소(raw_addr += 4)에 대해
 * addrdec_tlx()를 호출하여 addrdec_t 결과를 얻고,
 * 동일한 addrdec_t가 이미 다른 선형 주소에서 생성된 적이 있는지 history_map으로 검사한다.
 * 만약 두 개의 서로 다른 선형 주소가 동일한 addrdec_t(= 동일한 DRAM 물리 위치)로 매핑되면,
 * 메모리 주소 알리아싱(aliasing)이 발생한 것이므로 오류를 출력하고 abort()로 시뮬레이터를 중단한다.
 *
 * 추가 검증: partition_address(raw_addr) != raw_addr인지 확인하여
 *            CHIP 비트가 partition_address에서 실제로 제거되는지 검증.
 *
 * 실행 컨텍스트: init()에서 run_test==true일 때 1회 실행. 시간 복잡도 O(N log N).
 * 에러: 오류 발견 시 printf + abort(). assert가 아닌 abort()를 사용하는 이유는
 *       구조체 비교 방식 오류(패딩 등)가 아닌 실제 매핑 오류임을 명확히 표현하기 위함.
 *
 * 호출 체인:
 *   init() → [이 함수] → addrdec_tlx() × N → history_map.find() × N
 */
void linear_to_raw_address_translation::sweep_test() const {
  new_addr_type sweep_range = 16 * 1024 * 1024;
  /* [한국어] 전수 검사 범위: 16MB = 16 × 2^20 = 2^24 바이트.
   * 4바이트 단위로 순회하므로 총 4M(4,194,304)회 addrdec_tlx() 호출.
   * 실제 GPU 메모리보다 작은 범위이지만 매핑의 기본 유효성 검증에 충분. */

#if tr1_hash_map_ismap == 1
  typedef tr1_hash_map<addrdec_t, new_addr_type> history_map_t;
  /* [한국어] tr1_hash_map_ismap==1 이면 std::map 기반 — operator< 를 키 비교에 사용.
   * 정렬된 맵이므로 삽입/조회가 O(log N). */
#else
  typedef tr1_hash_map<addrdec_t, new_addr_type, hash_addrdec_t> history_map_t;
  /* [한국어] 기본값: std::unordered_map 기반 — hash_addrdec_t를 해시 펑터로, operator==을 키 비교에 사용.
   * 해시맵이므로 삽입/조회가 평균 O(1). sweep_test() 성능에 적합. */
#endif
  history_map_t history_map;
  /* [한국어] 지금까지 방문한 addrdec_t → 선형 주소 매핑 테이블.
   * 키: addrdec_t (분해된 DRAM 위치), 값: 최초 매핑된 선형 주소. */

  for (new_addr_type raw_addr = 4; raw_addr < sweep_range; raw_addr += 4) {
    /* [한국어] 4부터 시작하는 이유: raw_addr=0은 NULL 포인터와 혼동될 수 있어 제외.
     * += 4: 4바이트(32비트) 정렬된 주소만 검사 (캐시라인보다 세밀한 검사). */
    addrdec_t tlx;               /* [한국어] 각 주소의 DRAM 분해 결과를 담을 임시 구조체. */
    addrdec_tlx(raw_addr, &tlx); /* [한국어] raw_addr을 DRAM 필드(chip/bk/row/col/burst/sub_partition)로 분해. */

    history_map_t::iterator h = history_map.find(tlx);
    /* [한국어] 이 addrdec_t 값이 이전에 다른 주소에서 나온 적이 있는지 조회. */

    if (h != history_map.end()) {
      /* [한국어] 동일한 addrdec_t가 이미 존재 → 알리아싱(aliasing) 오류 발견!
       * h->second: 이전에 이 addrdec_t에 매핑된 선형 주소.
       * 두 서로 다른 선형 주소(h->second, raw_addr)가 동일한 DRAM 위치로 매핑됨. */
      printf(
          "[AddrDec] ** Error: address decoding mapping aliases two addresses "
          "to same partition with same intra-partition address: %llx %llx\n",
          h->second, raw_addr);
      /* [한국어] 충돌한 두 선형 주소를 16진수로 출력.
       * h->second: 먼저 방문한 주소, raw_addr: 현재 주소. */
      abort();
      /* [한국어] 복구 불가능한 매핑 오류 — abort()로 시뮬레이터 즉시 비정상 종료.
       * assert 대신 abort()를 사용하는 이유: NDEBUG로 assert가 비활성화되어도 항상 중단 보장. */
    } else {
      assert(tlx.chip < m_n_channel);
      /* [한국어] 채널 인덱스가 유효 범위 안에 있는지 확인 — 잘못된 마스크 설정 탐지. */
      // ensure that partition_address() returns the concatenated address
      if ((ADDR_CHIP_S != -1 and raw_addr >= (1ULL << ADDR_CHIP_S)) or
          (ADDR_CHIP_S == -1 and raw_addr >= (1ULL << addrdec_mklow[CHIP]))) {
        /* [한국어] raw_addr이 CHIP 비트 영역에 도달했을 때(= 채널 비트가 실제로 세트되는 주소):
         * partition_address(raw_addr) != raw_addr 임을 확인.
         * 만약 같다면 partition_address()가 CHIP 비트를 제대로 제거하지 못한 것.
         * ADDR_CHIP_S != -1: dramid@N 방식 — 1ULL << ADDR_CHIP_S 이상이면 채널 비트 활성.
         * ADDR_CHIP_S == -1: D 문자 방식 — CHIP 마스크의 첫 비트(mklow[CHIP]) 이상. */
        assert(raw_addr != partition_address(raw_addr));
        /* [한국어] raw_addr != partition_address(raw_addr): CHIP 비트가 실제로 제거됨을 검증.
         * 같으면 CHIP 마스크 설정 또는 partition_address() 구현에 오류가 있는 것. */
      }
      history_map[tlx] = raw_addr;
      /* [한국어] 처음 방문하는 addrdec_t를 history_map에 삽입.
       * 값: raw_addr (이후 동일 addrdec_t가 나오면 이 주소와 충돌했다고 보고할 것). */
    }

    if ((raw_addr & 0xffff) == 0) printf("%llu scaned\n", raw_addr);
    /* [한국어] 65536(64KB) 단위마다 진행 상황을 stdout에 출력.
     * raw_addr & 0xffff == 0: 하위 16비트가 0인 시점 = 64KB 경계.
     * 전수 검사가 오래 걸리므로 진행률을 사용자에게 표시. */
  }
}

/*
 * [한국어]
 * addrdec_t::print - addrdec_t 구조체의 모든 필드를 파일 스트림에 출력
 *
 * @fp: 출력할 FILE 포인터 (stdout, stderr, 또는 로그 파일)
 * @return: 없음 (void, const 멤버 함수)
 *
 * 디버깅 목적으로 addrdec_t의 6개 필드(chip, row, col, bk, burst, sub_partition)를
 * 탭 구분으로 16진수 형식으로 출력한다. 출력 줄 끝에 줄바꿈 없이 끝나므로
 * 호출자가 필요에 따라 fprintf(fp, "\n")을 추가해야 한다.
 *
 * 실행 컨텍스트: const 멤버 함수. 단순 I/O이므로 어떤 스레드에서도 호출 가능.
 *
 * 호출 체인:
 *   디버깅 코드 또는 sweep_test() 오류 출력 → [이 함수] → fprintf() × 6
 */
void addrdec_t::print(FILE *fp) const {
  fprintf(fp, "\tchip:%x ", chip);          /* [한국어] 채널(chip) 인덱스를 16진수로 출력 */
  fprintf(fp, "\trow:%x ", row);            /* [한국어] 행(row) 인덱스를 16진수로 출력 */
  fprintf(fp, "\tcol:%x ", col);            /* [한국어] 열(col) 인덱스를 16진수로 출력 */
  fprintf(fp, "\tbk:%x ", bk);             /* [한국어] 뱅크(bk) 인덱스를 16진수로 출력 */
  fprintf(fp, "\tburst:%x ", burst);        /* [한국어] 버스트 오프셋을 16진수로 출력 */
  fprintf(fp, "\tsub_partition:%x ", sub_partition);
  /* [한국어] 서브파티션 ID를 16진수로 출력. 마지막 필드로 출력 후 줄바꿈 없이 종료. */
}

/*
 * [한국어]
 * powli - 정수 거듭제곱 계산: x^y를 long int로 반환
 *
 * @x: 밑(base) 값 (long int)
 * @y: 지수(exponent) 값 (long int)
 * @return: x를 y번 곱한 결과 (x^y). y==0이면 1 반환.
 *
 * <math.h>의 pow()는 double 부동소수점 연산이라 큰 정수에서 반올림 오류가 생길 수 있다.
 * init()에서 2^nchipbits를 정확히 계산하여 gap = n_channel - 2^nchipbits를 구할 때
 * 정수 정밀도가 필요하므로 별도로 구현.
 * 오버플로우 방지: long int는 최소 32비트이므로 x=2, y=32 이상에서 오버플로우 발생.
 * 실제 사용: powli(2, nchipbits)이고 nchipbits <= 6(64채널 이하)이므로 안전.
 *
 * 실행 컨텍스트: static 함수, init()에서 1회 호출.
 *
 * 호출 체인:
 *   init() → [이 함수]
 */
static long int powli(long int x, long int y)  // compute x to the y
{
  long int r = 1;    /* [한국어] 결과 누산기, 초기값 1 (x^0 = 1). */
  int i;             /* [한국어] 루프 인덱스. */
  for (i = 0; i < y; ++i) {
    /* [한국어] y번 반복하여 r에 x를 곱해 x^y를 계산. */
    r *= x;          /* [한국어] r = r × x → i번째 곱셈. y번 반복하면 r = x^y. */
  }
  return r;          /* [한국어] 최종 계산 결과 x^y 반환. */
}

/*
 * [한국어]
 * LOGB2_32 - 32비트 정수의 floor(log2(v))를 비트 조작으로 빠르게 계산
 *
 * @v: log2를 구할 32비트 부호 없는 정수. v==0이면 동작 미정의.
 * @return: floor(log2(v)) — v의 최상위 set 비트 위치(0-based).
 *          예: v=1→0, v=2→1, v=3→1, v=4→2, v=8→3.
 *
 * 이진 탐색(binary search) 비트 조작으로 루프 없이 5단계 비교+시프트로 계산.
 * 각 단계에서 비트 폭을 절반씩 좁혀가며 최상위 비트의 위치를 찾는다.
 * 단계별 처리:
 *   1단계: 상위 16비트(bits[31:16]) 확인 → shift = 16 or 0
 *   2단계: 상위 8비트(bits[15:8] 또는 bits[7:0]) 확인 → shift += 8 or 0
 *   3단계: 상위 4비트 확인 → shift += 4 or 0
 *   4단계: 상위 2비트 확인 → shift += 2 or 0
 *   5단계: 상위 1비트 확인 → shift += 1 or 0
 * 결과 r에 각 단계의 shift가 OR로 누적되어 최종 log2 값이 됨.
 *
 * 실행 컨텍스트: static 함수, init()에서 nchipbits와 log2sub_partition 계산에 사용.
 *
 * 호출 체인:
 *   init() → [이 함수]
 */
static unsigned int LOGB2_32(unsigned int v) {
  unsigned int shift;   /* [한국어] 각 이진 탐색 단계에서 결정된 비트 시프트 양. */
  unsigned int r;       /* [한국어] 최종 log2 결과 누산기. */

  r = 0;               /* [한국어] 결과 초기화. 각 단계에서 OR로 비트 추가. */

  shift = ((v & 0xFFFF0000) != 0) << 4;
  /* [한국어] 1단계: bits[31:16]이 0인지 확인.
   * v & 0xFFFF0000 != 0이면 최상위 비트는 bit 16 이상에 있음 → shift = 1<<4 = 16.
   * v & 0xFFFF0000 == 0이면 최상위 비트는 bit 15 이하에 있음 → shift = 0.
   * 불리언 비교 결과(0 or 1)를 << 4로 4비트 시프트하여 shift = 16 or 0을 만드는 트릭. */
  v >>= shift;           /* [한국어] shift가 16이면 v를 16비트 오른쪽 시프트하여 하위 16비트로 좁힘. */
  r |= shift;            /* [한국어] 결과에 이 단계의 shift 기여분 누적. */
  shift = ((v & 0xFF00) != 0) << 3;
  /* [한국어] 2단계: 현재 v의 bits[15:8]이 0인지 확인 (1단계 후 좁혀진 범위 내에서).
   * v & 0xFF00 != 0이면 shift = 1<<3 = 8. 그렇지 않으면 shift = 0. */
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  /* [한국어] 3단계: bits[7:4] 확인 → shift = 4 or 0. */
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  /* [한국어] 4단계: bits[3:2] 확인 → shift = 2 or 0.
   * 0xC = 0b1100 = bits[3:2]. */
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  /* [한국어] 5단계: bit[1] 확인 → shift = 1 or 0.
   * 0x2 = bit 1. 이 단계에서 마지막 1비트 위치를 결정.
   * << 0은 의미 없으나 코드 일관성을 위해 유지 (shift = 1 or 0). */
  v >>= shift;
  r |= shift;

  return r;
  /* [한국어] 5단계에서 누적된 shift 합 = floor(log2(original v)).
   * 각 단계에서 비트 폭이 32→16→8→4→2→1로 절반씩 줄어들며 최상위 비트 위치를 이진 탐색. */
}

// compute power of two greater than or equal to n
// https://www.techiedelight.com/round-next-highest-power-2/
/*
 * [한국어]
 * next_powerOf2 - n 이상의 최소 2의 거듭제곱 계산
 *
 * @n: 2의 거듭제곱으로 올림할 기준 값 (unsigned). n==0이면 0 반환 (미정의 동작 주의).
 * @return: n 이상의 최소 2의 거듭제곱.
 *          예: n=1→1, n=3→4, n=4→4, n=5→8, n=6→8.
 *
 * 알고리즘 (출처: https://www.techiedelight.com/round-next-highest-power-2/):
 *   1단계: n = n - 1 (n이 이미 2의 거듭제곱인 경우를 올바르게 처리하기 위해 1 감소).
 *          예: n=4 → n=3=0b011 (1 감소 후 최하위 비트 제거 과정을 거침).
 *   2단계: while (n & (n-1)): 최하위 set 비트를 반복 제거하여 단 1개의 비트만 남김.
 *          n & (n-1)은 최하위 set 비트를 0으로 만드는 연산.
 *          종료 시: n은 2의 거듭제곱(단 1개 비트만 남음).
 *   3단계: return n << 1: 남은 1개 비트를 1 왼쪽 시프트하여 2배(다음 2의 거듭제곱)로 반환.
 *
 * init()에서 nextPowerOf2_m_n_channel 계산에 사용.
 *
 * 호출 체인:
 *   init() → [이 함수]
 */
unsigned next_powerOf2(unsigned n) {
  // decrement n (to handle the case when n itself
  // is a power of 2)
  n = n - 1;
  /* [한국어] n 자신이 2의 거듭제곱인 경우를 처리하기 위해 1 감소.
   * 예: n=8=0b1000 → n=7=0b0111. 이후 비트 제거 후 << 1하면 8을 올바르게 얻음.
   * n=6=0b110 → n=5=0b101. 비트 제거 후 4=0b100, <<1 → 8. 맞음. */

  // do till only one bit is left
  while (n & (n - 1)) n = n & (n - 1);  // unset rightmost bit
  /* [한국어] n & (n-1): n의 최하위 set 비트를 0으로 만드는 비트 트릭.
   * 예: n=7=0b0111 → 0b0110 → 0b0100 (단 1개 비트 남을 때까지 반복).
   * n & (n-1) == 0이 되면(= 2의 거듭제곱 이하) 루프 종료.
   * 루프 종료 후 n은 원래 입력 이하의 가장 큰 2의 거듭제곱(= 2^k). */

  // n is now a power of two (less than n)

  // return next power of 2
  return n << 1;
  /* [한국어] n(현재 2^k) << 1 = 2^(k+1) = n 이상의 최소 2의 거듭제곱.
   * 예: n=4=0b0100 → 0b1000=8. 올바른 결과. */
}

/*
 * [한국어]
 * addrdec_packbits - 64비트 값에서 마스크 비트 위치의 값만 추출하여 하위 비트부터 압축(pack)
 *
 * @mask: 추출할 비트 위치를 지정하는 64비트 마스크. 1인 비트 위치의 val 비트를 추출.
 * @val:  비트를 추출할 64비트 원본 값 (선형 주소).
 * @high: 순회 상한 비트 위치 (exclusive). addrdec_mkhigh[N]으로 최적화된 범위 지정.
 * @low:  순회 하한 비트 위치 (inclusive). addrdec_mklow[N]으로 최적화된 범위 지정.
 * @return: mask의 1인 비트 위치에서 val의 비트를 추출하여 bit 0부터 연속으로 압축한 값.
 *
 * 알고리즘:
 *   low부터 high-1까지 i를 순회하면서:
 *   1) mask의 bit i가 1인지 확인.
 *   2) 1이면 val의 bit i 값을 추출: (val >> i) & 1.
 *   3) 추출한 비트를 결과의 bit pos 위치에 배치.
 *   4) pos 증가.
 *   결과는 추출된 비트들이 bit 0부터 연속으로 붙은 정수.
 *
 * 예시: mask=0b1010, val=0b1110, low=0, high=4
 *   i=0: mask bit 0 = 0 → 스킵
 *   i=1: mask bit 1 = 1 → val bit 1 = 1 → result bit 0 = 1, pos=1
 *   i=2: mask bit 2 = 0 → 스킵
 *   i=3: mask bit 3 = 1 → val bit 3 = 1 → result bit 1 = 1, pos=2
 *   결과: 0b11 = 3
 *
 * addrdec_tlx()에서 각 DRAM 필드(chip/bk/row/col/burst)를 추출할 때 사용.
 * partition_address()에서 CHIP/sub_partition 비트를 제거할 때도 사용(마스크 반전).
 *
 * 실행 컨텍스트: static 함수, addrdec_tlx() / partition_address()에서 매 호출 시 실행.
 * 에러 처리: 없음. mask가 0이면 result=0 반환.
 *
 * 호출 체인:
 *   addrdec_tlx() → [이 함수] (5회, 각 DRAM 필드마다)
 *   partition_address() → [이 함수] (1~2회)
 */
static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low) {
  unsigned pos = 0;          /* [한국어] 결과 값에 비트를 배치할 현재 위치. bit 0부터 시작. */
  new_addr_type result = 0;  /* [한국어] 압축 추출된 비트를 누적하는 결과 변수. 초기값 0. */
  for (unsigned i = low; i < high; i++) {
    /* [한국어] low부터 high-1까지 비트 위치를 순회.
     * addrdec_getmasklimit()이 계산한 마스크 유효 범위만 순회하여 불필요한 반복을 최소화. */
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      /* [한국어] mask의 bit i가 1인지 확인.
       * (1ULL << i): i번째 비트만 1인 마스크. mask와 AND하여 0이 아니면 set된 것. */
      result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
      /* [한국어] val의 bit i 값을 추출하여 result의 bit pos에 배치:
       *   (val & (1ULL << i))       : val에서 bit i만 추출 (다른 비트는 0)
       *   >> i                      : 추출한 비트를 bit 0 위치로 내림 (= 0 or 1)
       *   << pos                    : bit pos 위치로 올림
       *   |= result                 : result의 bit pos에 OR로 합침
       * pos가 증가할수록 추출된 비트들이 result의 하위 비트부터 차례로 채워짐. */
      pos++;                 /* [한국어] 다음 비트는 result의 한 자리 위에 배치. */
    }
  }
  return result;             /* [한국어] mask 비트 위치에서 추출한 비트들이 압축된 정수 반환. */
}

/*
 * [한국어]
 * addrdec_getmasklimit - 비트마스크에서 유효 비트의 최하위(low)와 최상위+1(high) 위치 탐색
 *
 * @mask: 분석할 64비트 비트마스크 (addrdec_mask[N] 중 하나).
 * @high: 결과를 저장할 포인터 — 마지막 set 비트 위치 + 1을 저장.
 *        mask가 0이면 초기값 64 그대로 유지.
 * @low:  결과를 저장할 포인터 — 첫 번째 set 비트 위치를 저장.
 *        mask가 0이면 초기값 0 그대로 유지.
 *
 * bit 0부터 63까지 순회하며 set 비트를 찾는다.
 * 첫 번째 set 비트: *low에 저장, *high = 해당 위치 + 1로 초기화.
 * 이후 set 비트가 나올 때마다: *high = 현재 위치 + 1로 갱신.
 * 순회 완료 후 *high는 마지막 set 비트 위치 + 1이 됨.
 *
 * 결과 사용: addrdec_packbits(mask, val, *high, *low)에서 *low~*high-1 범위만 순회.
 * mask가 연속 비트가 아닌 경우(비연속 마스크)에도 올바르게 동작하나,
 * low~high 사이에 0인 비트가 있어도 packbits 루프에서 마스크 확인으로 걸러짐.
 *
 * 실행 컨텍스트: static 함수, init()에서 5회 호출 (각 DRAM 필드마다).
 *
 * 호출 체인:
 *   init() → [이 함수] × 5 (CHIP, BK, ROW, COL, BURST 필드 각각)
 */
static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low) {
  *high = 64;       /* [한국어] high 초기값: 64 (mask가 0이면 이 값 유지).
                     * 64는 "마스크에 set 비트 없음"을 나타내는 sentinel 값.
                     * addrdec_mklow/mkhigh 초기화와 일치. */
  *low = 0;         /* [한국어] low 초기값: 0 (mask가 0이면 이 값 유지). */
  int i;            /* [한국어] 비트 위치 순회 인덱스. */
  int low_found = 0; /* [한국어] 첫 번째 set 비트를 찾았는지 여부 플래그. 0=미발견, 1=발견. */

  for (i = 0; i < 64; i++) {
    /* [한국어] bit 0부터 63까지 순회하며 mask의 set 비트를 탐색. */
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      /* [한국어] bit i가 mask에서 set되어 있는지 확인. */
      if (low_found) {
        *high = i + 1;
        /* [한국어] 이미 low를 찾은 상태 — 현재 set 비트가 새 상한이 됨.
         * *high = i + 1: exclusive 상한 (i까지 포함이므로 i+1이 상한).
         * 더 높은 set 비트가 나올수록 *high가 계속 갱신됨. */
      } else {
        *high = i + 1;
        /* [한국어] 처음 발견한 set 비트: low와 high 모두 초기화.
         * *high = i + 1: 현재 비트가 마지막이라고 가정한 초기 상한. */
        *low = i;
        /* [한국어] *low = i: 첫 번째 set 비트 위치를 low로 저장. */
        low_found = 1;
        /* [한국어] 이제부터는 low_found 분기(고정된 *low, 갱신만 되는 *high)를 탐. */
      }
    }
  }
  /* [한국어] 순회 완료 후:
   * *low = 마스크의 최하위 set 비트 위치 (mask==0이면 0 그대로).
   * *high = 마스크의 최상위 set 비트 위치 + 1 (mask==0이면 64 그대로).
   * addrdec_packbits()는 [*low, *high) 범위만 순회하므로 불필요한 64회 전체 순회를 피함. */
}
