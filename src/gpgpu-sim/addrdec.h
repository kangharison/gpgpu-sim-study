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

/*
 * [한국어 설명] GPU 메모리 주소 디코딩 모듈 (addrdec.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU 시뮬레이터 내에서 64비트 선형(가상) 메모리 주소를 DRAM의
 * 물리적 위치 — 채널(chip), 뱅크(bk), 행(row), 열(col), 버스트(burst) —
 * 로 분해하는 주소 디코딩 계층을 정의한다. 실제 DRAM 컨트롤러가 수행하는
 * 주소 인터리빙/뱅크 선택 로직을 소프트웨어로 재현하여, 사이클-레벨
 * DRAM 타이밍 시뮬레이션이 정확한 물리적 위치를 기반으로 row-buffer hit/miss
 * 등을 판단할 수 있게 한다. 또한 partition_index_function을 통해
 * 다양한 주소 해시 정책(연속/XOR/다항식/랜덤 등)을 선택적으로 적용하여
 * 메모리 채널 간 부하 분산 효과를 연구할 수 있도록 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 메모리 계층은 다음 순서로 구성된다:
 *   SM(셰이더 코어) → L1 캐시 → interconnect(NoC/intersim2)
 *   → L2 캐시 서브파티션(memory_sub_partition)
 *   → memory_partition_unit → [addrdec 모듈] → dram_t(채널별 DRAM 타이밍)
 *
 * addrdec 모듈은 L2 캐시 미스 이후 mem_fetch 패킷이 DRAM으로 내려갈 때
 * 호출된다. memory_partition_unit::dram_cycle()에서 addrdec_tlx()를 호출해
 * 선형 주소를 DRAM 필드로 변환하고, 그 결과를 dram_t의 타이밍 모델에 전달한다.
 * 이 모듈은 타이밍 시뮬레이션 계층(gpgpu-sim/)에 속하며, 기능 시뮬레이션
 * 계층(cuda-sim/)과는 독립적이다. 호스트 유저스페이스에서 실행된다.
 *
 * 호출 체인 (L2 미스 기준):
 *   memory_sub_partition::L2_dram_queue
 *   → memory_partition_unit::dram_cycle()
 *   → linear_to_raw_address_translation::addrdec_tlx()
 *   → dram_t::issue() (뱅크/행 기반 타이밍 결정)
 *
 * === 타 모듈과의 연결 ===
 * 의존(이 파일이 사용하는 모듈):
 *   - abstract_hardware_model.h: new_addr_type(64비트 주소 typedef),
 *     memory_config 구조체 (linear_to_raw_address_translation 인스턴스 포함)
 *   - option_parser.h: addrdec_setoption()에서 gpgpusim.config 옵션을 등록할 때 사용
 *   - <assert.h>, <stdio.h>, <stdlib.h>: 표준 진단/입출력/메모리 유틸
 *
 * 역의존(이 파일을 사용하는 모듈):
 *   - l2cache.h / memory_partition_unit: addrdec_tlx()로 DRAM 채널 결정
 *   - dram.h / dram_t: 채널 ID(chip)와 뱅크(bk)/행(row)/열(col)을 받아 타이밍 모델 실행
 *   - gpu-sim.cc: init()을 통해 채널 수와 서브파티션 수 초기화
 *
 * 데이터 흐름:
 *   mem_fetch::m_addr (선형 주소)
 *   → addrdec_tlx() → addrdec_t { chip, bk, row, col, burst, sub_partition }
 *   → dram_t가 chip 인덱스로 자신의 채널인지 확인 후 요청 수락
 *
 * gpgpusim.config 관련 옵션:
 *   -gpgpu_mem_addr_mapping     : 주소 매핑 문자열 (각 필드의 비트 범위 지정)
 *   -gpgpu_mem_address_mask     : 비트마스크 선택 모드
 *   -gpgpu_memory_partition_indexing : partition_index_function 선택 (0=CONSECUTIVE 등)
 *
 * === 주요 함수/구조체 요약 ===
 * addrdec_t                              : 선형 주소를 분해한 DRAM 물리 위치 레코드
 * partition_index_function               : 채널 매핑 해시 정책 열거형
 * linear_to_raw_address_translation      : 주소 디코더 클래스 (핵심)
 *   - addrdec_setoption()                : gpgpusim.config 옵션을 option_parser에 등록
 *   - init()                             : 채널/서브파티션 수를 기반으로 비트마스크 초기화
 *   - addrdec_tlx()                      : 선형 주소 → addrdec_t 분해 (핵심 디코딩 함수)
 *   - partition_address()                : 서브파티션 비트를 제거한 채널-레벨 주소 반환
 *   - addrdec_parseoption()              : 옵션 문자열 파싱 → 비트마스크 배열 구성
 *   - sweep_test()                       : 주소 매핑에 중복/누락이 없는지 전수 검사
 */

#include <assert.h>  // [한국어] assert() 매크로 — sweep_test() 등에서 비트 일관성 검증에 사용
#include <stdio.h>   // [한국어] FILE*, fprintf() — addrdec_t::print()의 출력 스트림에 필요
#include <stdlib.h>  // [한국어] malloc/free 등 범용 유틸 — 옵션 파싱 보조 버퍼에 사용

#include "../option_parser.h"
// [한국어] option_parser_t 타입 및 option_parser_register() API 제공.
// addrdec_setoption()이 gpgpusim.config의 주소 매핑 옵션을 파서에 등록할 때 필요하다.
// 이 include가 #ifndef 가드 밖에 있는 이유는 option_parser.h 자체에 include guard가
// 있어 중복 포함이 무해하며, 일부 컴파일 유닛이 가드 없이 이 헤더를 먼저 볼 수 있기
// 때문이다 (레거시 include 순서 유지).

#ifndef ADDRDEC_H  // [한국어] 헤더 중복 포함 방지 가드 시작
#define ADDRDEC_H

#include "../abstract_hardware_model.h"
// [한국어] new_addr_type (= unsigned long long, 64비트 선형 주소 typedef) 및
// memory_config 구조체(linear_to_raw_address_translation 멤버 포함) 정의.
// addrdec_tlx()의 인자와 반환값 타입이 여기서 온다.

/*
 * [한국어] partition_index_function — 메모리 파티션(채널+서브파티션) 인덱스 결정 방식
 *
 * DRAM 채널(파티션)로의 주소 매핑 해시 정책을 선택하는 열거형이다.
 * gpgpusim.config의 -gpgpu_memory_partition_indexing 옵션으로 선택하며,
 * linear_to_raw_address_translation::memory_partition_indexing 멤버에 저장된다.
 * addrdec_tlx()에서 이 값을 분기 조건으로 사용해 서로 다른 채널 결정 알고리즘을 실행한다.
 * 채널 매핑 방식은 bank conflict 분포와 DRAM 부하 분산에 직접적인 영향을 미친다.
 */
enum partition_index_function {
  CONSECUTIVE = 0,
  /* [한국어] 연속 비트 슬라이스 방식 (기본값).
   * 선형 주소에서 ADDR_CHIP_S 위치를 기준으로 log2(채널 수)개의 연속 비트를
   * 그대로 채널 인덱스로 사용한다. 구현이 단순하고 예측 가능하지만,
   * 연속 접근 패턴에서 특정 채널에 부하가 집중될 수 있다.
   * addrdec_tlx()의 기본 경로에서 비트 마스크 추출 후 별도 해시 없이 사용. */

  BITWISE_PERMUTATION,
  /* [한국어] XOR 기반 비트 치환 해시 방식.
   * 주소의 여러 비트 그룹을 XOR 연산으로 혼합하여 채널 인덱스를 생성한다.
   * 연속 접근의 채널 편중을 줄여 부하를 분산하는 효과가 있다.
   * 하드웨어 구현 비용이 낮아 실제 GPU 메모리 컨트롤러에서도 사용된다.
   * addrdec_tlx()에서 CONSECUTIVE 추출 후 XOR 단계가 추가로 수행된다. */

  IPOLY,
  /* [한국어] 비가역 다항식(Irreducible Polynomial) 해시 방식.
   * GF(2^k) 위의 다항식 나머지 연산으로 채널 인덱스를 결정한다.
   * XOR 기반보다 분산 품질이 높아 bank conflict를 더 효과적으로 줄인다.
   * 연구 논문에서 "DRAM bank conflict 최소화" 목적으로 제안된 방식이다.
   * 연산량이 BITWISE_PERMUTATION보다 많지만 시뮬레이션 정확도 향상에 기여. */

  PAE,
  /* [한국어] PAE(Page Address Extension) 기반 매핑 방식.
   * 물리 페이지 주소의 상위 비트를 활용해 채널을 결정한다.
   * 페이지 단위로 채널이 달라지므로, 같은 페이지 내 접근은 항상 동일 채널에 집중된다.
   * 특정 워크로드(페이지 단위 지역성이 강한 경우)의 분석 실험에 적합하다. */

  RANDOM,
  /* [한국어] 소프트웨어 해시테이블 기반 완전 랜덤 매핑 방식.
   * 시뮬레이터가 내부적으로 주소-채널 매핑 테이블을 구성해 완전히 무작위로 분산한다.
   * 실제 하드웨어에서 구현 불가능한 이상적인 분산을 연구용으로 모델링할 때 사용한다.
   * 초기화 비용과 런타임 조회 비용이 높으므로 소규모 실험에만 권장. */

  CUSTOM
  /* [한국어] 사용자 정의 확장 포인트.
   * 연구자가 addrdec.cc에 자체 알고리즘을 구현할 때 사용하는 플레이스홀더이다.
   * GPGPU-Sim을 기반으로 새로운 메모리 인터리빙 정책을 실험할 때 이 값을 선택한다.
   * 미구현 상태에서 선택하면 addrdec_tlx()가 assert 또는 기본 경로로 폴백한다. */
};

/*
 * [한국어] addrdec_t — 선형 주소를 DRAM 물리 위치로 분해한 결과 레코드
 *
 * addrdec_tlx()가 64비트 선형 주소를 입력받아 채우는 출력 구조체이다.
 * 각 필드는 DRAM 계층의 서로 다른 선택 수준(채널 → 뱅크 → 행 → 열 → 버스트)에
 * 대응하며, dram_t의 타이밍 모델이 row-buffer hit/miss, bank conflict 등을
 * 판단하는 기준 데이터가 된다. 이 구조체는 메모리 요청(mem_fetch)당 하나씩
 * 스택에 생성되며(임시 변수), DRAM 채널로 전달된 후에는 소멸한다.
 * 멀티스레드 접근 없이 단일 memory_partition_unit 스레드가 단독으로 읽고 쓴다.
 */
struct addrdec_t {
  void print(FILE *fp) const;
  // [한국어] 디버깅용 출력 함수. 분해된 모든 필드(chip, bk, row, col, burst, sub_partition)를
  // fp 스트림에 16진수/10진수로 출력한다. 주소 매핑 검증 및 sweep_test() 결과 확인에 사용.

  unsigned chip;
  /* [한국어] DRAM 채널(메모리 파티션) 인덱스.
   * 설정자: addrdec_tlx()가 addrdec_mask[CHIP] 비트마스크와
   *         partition_index_function에 따른 해시 연산으로 결정.
   * 읽는 자: memory_partition_unit이 이 값으로 자신이 담당하는 채널인지 확인;
   *          dram_t::issue()가 요청을 수락할 채널을 식별.
   * 값 범위: [0, m_n_channel) — 채널 수는 gpgpusim.config의 -gpgpu_n_mem으로 결정.
   * 동기화: 단일 memory_partition_unit 스레드에서만 읽고 쓰므로 별도 락 불필요.
   *         각 memory_partition_unit은 자신의 chip 인덱스와 비교해 라우팅을 결정. */

  unsigned bk;
  /* [한국어] DRAM 뱅크 인덱스.
   * 설정자: addrdec_tlx()가 addrdec_mask[BK] 마스크로 추출.
   * 읽는 자: dram_t의 타이밍 모델이 해당 뱅크의 open row 상태를 확인해
   *          row-buffer hit(빠름) / miss(느림, PRECHARGE+ACTIVATE 필요)를 판단.
   * 값 범위: [0, 뱅크 수) — 뱅크 수는 -gpgpu_n_mem_per_ctrlr 등의 옵션으로 결정.
   * 동기화: dram_t 내부의 뱅크 상태(dram_req_t 큐)와 함께 관리되며,
   *          dram_t는 채널당 단일 객체이므로 채널 간 동기화는 불필요. */

  unsigned row;
  /* [한국어] DRAM 행(row) 인덱스 — row buffer에 활성화될 워드라인 위치.
   * 설정자: addrdec_tlx()가 addrdec_mask[ROW] 마스크로 추출.
   * 읽는 자: dram_t가 현재 활성화된 row(open_row[bk])와 비교하여
   *          row-buffer hit 여부를 결정. hit이면 CAS만, miss이면 PRECHARGE+ACTIVATE+CAS.
   * 값 범위: [0, 행 수) — 일반적으로 16K~64K 행.
   * 동기화: 뱅크 단위로 open_row 상태가 관리되며, dram_t 내부에서 직렬 처리. */

  unsigned col;
  /* [한국어] DRAM 열(column) 인덱스 — 활성화된 행 내에서의 열 위치.
   * 설정자: addrdec_tlx()가 addrdec_mask[COL] 마스크로 추출.
   * 읽는 자: dram_t가 CAS(Column Address Strobe) 명령을 발행할 때 열 주소로 사용.
   * 값 범위: [0, 열 수) — 캐시라인 크기와 DRAM 버스 폭에 따라 결정.
   * 동기화: row와 동일하게 dram_t 내부에서 직렬 처리. */

  unsigned burst;
  /* [한국어] 버스트 오프셋 — 연속 전송에서 첫 번째 전송 단위의 위치.
   * 설정자: addrdec_tlx()가 addrdec_mask[BURST] 마스크로 추출.
   * 읽는 자: dram_t가 버스트 길이(BL) 계산 및 데이터 전송 타이밍 결정에 사용.
   * 값 범위: [0, 버스트 길이) — DDR4 기준 BL=8이 일반적.
   * 동기화: 요청당 독립적이므로 별도 동기화 불필요. */

  unsigned sub_partition;
  /* [한국어] L2 캐시 서브파티션 ID — 동일 채널(chip) 내 여러 L2 슬라이스 중 담당 슬라이스.
   * 설정자: addrdec_tlx()가 bk 값의 하위 비트(sub_partition_id_mask)로 결정.
   *          한 채널 안에서도 복수의 L2 서브파티션이 존재할 때 이 값으로 구분.
   * 읽는 자: memory_partition_unit이 mem_fetch를 올바른 memory_sub_partition으로
   *          라우팅할 때 사용; L2 태그/데이터 슬라이스가 서브파티션 단위로 분리되어 있음.
   * 값 범위: [0, n_sub_partition_in_channel) — 서브파티션 수는 init()의 인자로 지정.
   * 동기화: chip과 동일하게 단일 스레드 내 임시 변수로만 사용. */
};

/*
 * [한국어] linear_to_raw_address_translation — GPU 선형 주소 → DRAM 물리 주소 디코더 클래스
 *
 * GPU 메모리 컨트롤러가 수행하는 주소 인터리빙/분해 로직을 소프트웨어로 구현한다.
 * memory_config 구조체의 멤버로 단 하나의 인스턴스가 존재하며,
 * 시뮬레이터 전역 설정(gpgpusim.config)에 따라 비트마스크 배열을 구성한다.
 * addrdec_tlx()가 이 클래스의 핵심 public API이며, memory_partition_unit에서
 * 매 사이클 DRAM으로 내려가는 mem_fetch마다 호출된다.
 * 호스트 CPU 유저스페이스에서 실행되며 GPU 디바이스 코드와는 무관하다.
 *
 * 초기화 순서:
 *   1. 생성자(linear_to_raw_address_translation()) — 기본값 설정
 *   2. addrdec_setoption(opp) — gpgpusim.config 옵션 등록 (gpu-sim.cc에서 호출)
 *   3. init(n_channel, n_sub_partition_in_channel) — 비트마스크/갭 계산 (gpu-sim.cc에서 호출)
 *   이후 addrdec_tlx() / partition_address() 호출 가능
 */
class linear_to_raw_address_translation {
 public:
  /*
   * [한국어] 생성자 — 멤버 변수를 안전한 초기값(0 또는 nullptr)으로 초기화.
   * init()이 호출되기 전까지는 addrdec_tlx()를 호출하면 안 된다.
   * 호출 체인: gpu-sim.cc의 gpgpu_sim 생성자 → memory_config 생성 → 이 생성자.
   */
  linear_to_raw_address_translation();

  /*
   * [한국어] addrdec_setoption — gpgpusim.config 옵션을 option_parser에 등록.
   * @opp: option_parser 핸들 — gpu-sim.cc가 전역 파서를 전달.
   *
   * -gpgpu_mem_addr_mapping, -gpgpu_mem_address_mask,
   * -gpgpu_memory_partition_indexing 등의 옵션을 파서에 등록한다.
   * 파서가 실제로 config 파일을 읽은 뒤 addrdec_parseoption()이 간접 호출되어
   * 비트마스크 배열이 구성된다.
   * 호출 체인: gpu-sim.cc::gpgpu_sim_config::reg_options() → 이 함수 → option_parser_register()
   */
  void addrdec_setoption(option_parser_t opp);

  /*
   * [한국어] init — 채널/서브파티션 수를 기반으로 주소 디코더 내부 상태 최종 초기화.
   * @n_channel: DRAM 채널(메모리 파티션) 총 수 — gpgpusim.config의 -gpgpu_n_mem
   * @n_sub_partition_in_channel: 채널당 L2 서브파티션 수
   *
   * gap(2의 거듭제곱 여부), log2channel, log2sub_partition,
   * nextPowerOf2_m_n_channel 등을 계산하여 이후 addrdec_tlx()에서
   * 채널 수가 2의 거듭제곱이 아닐 때도 올바르게 모듈러 연산할 수 있도록 준비한다.
   * run_test가 true이면 sweep_test()를 호출해 비트 매핑 일관성을 검증한다.
   * 호출 체인: gpu-sim.cc::gpgpu_sim::init() → 이 함수
   */
  void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel);

  // accessors

  /*
   * [한국어] addrdec_tlx — 핵심 주소 디코딩 함수. 선형 주소 → addrdec_t 분해.
   * @addr: 디코딩할 64비트 선형 GPU 메모리 주소 (mem_fetch::m_addr)
   * @tlx:  결과를 채울 addrdec_t 포인터 (호출자가 스택에 선언한 임시 변수)
   *
   * addrdec_mask[N_ADDRDEC] 비트마스크 배열을 이용해 addr에서 각 DRAM 필드를
   * 추출하고, memory_partition_indexing에 따라 CONSECUTIVE/BITWISE_PERMUTATION/
   * IPOLY 등의 해시를 적용하여 최종 채널(chip) 인덱스를 결정한다.
   * gap != 0인 경우 nextPowerOf2_m_n_channel 기반 모듈러 연산으로 채널 인덱스를 보정한다.
   * sub_partition은 bk 값의 하위 log2sub_partition 비트에서 추출한다.
   * 이 함수는 const이므로 클래스 상태를 변경하지 않으며, 매 사이클 반복 호출에 안전하다.
   *
   * 호출 체인:
   *   memory_partition_unit::dram_cycle() → [이 함수] → tlx 필드 → dram_t::issue()
   */
  void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const;

  /*
   * [한국어] partition_address — 서브파티션 비트를 제거한 채널-레벨 정규화 주소 반환.
   * @addr: 64비트 선형 주소
   * @return: sub_partition_id_mask에 해당하는 비트를 0으로 마스킹한 주소.
   *
   * L2 캐시 태그 비교 시 서브파티션 구분 비트를 제거하여 동일 채널의 서로 다른
   * 서브파티션에 매핑된 주소가 같은 캐시 태그를 갖도록 정규화한다.
   * memory_sub_partition의 L2 tag_array 조회에서 이 함수의 결과를 태그 주소로 사용한다.
   *
   * 호출 체인:
   *   memory_sub_partition::cache_cycle() → [이 함수] → L2 tag_array::probe()
   */
  new_addr_type partition_address(new_addr_type addr) const;

 private:
  /*
   * [한국어] addrdec_parseoption — 옵션 문자열을 파싱해 addrdec_mask 배열을 구성.
   * @option: gpgpusim.config의 -gpgpu_mem_addr_mapping 값 문자열
   *          (예: "dramid@8;...;RRRRRRRRRRRRR:...;BBBBB:...")
   *
   * 문자열에서 CHIP/BK/ROW/COL/BURST 각 필드의 비트 위치를 파싱하여
   * addrdec_mask[CHIP..BURST], addrdec_mklow[], addrdec_mkhigh[]를 설정한다.
   * addrdec_setoption()이 option_parser에 콜백으로 등록하거나, init() 시점에 호출된다.
   * 호출 체인: addrdec_setoption() 등록 콜백 or init() → [이 함수]
   */
  void addrdec_parseoption(const char *option);

  void sweep_test() const;  // sanity check to ensure no overlapping
  /* [한국어] sweep_test — 비트마스크 배열의 완전성과 비중복성을 전수 검사.
   * 모든 비트가 CHIP/BK/ROW/COL/BURST 중 정확히 하나에만 매핑되어 있는지 확인한다.
   * init()에서 run_test==true일 때 호출되며, 잘못된 gpgpusim.config 옵션이
   * 주소 매핑 오류를 일으키는 상황을 조기에 탐지한다.
   * 오류 발견 시 assert로 시뮬레이터를 즉시 중단한다. */

  enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, N_ADDRDEC };
  // [한국어] addrdec_mask/mklow/mkhigh 배열의 인덱스 상수.
  // CHIP=0: 채널(메모리 파티션) 필드 인덱스
  // BK=1  : 뱅크 필드 인덱스
  // ROW=2 : 행 필드 인덱스
  // COL=3 : 열 필드 인덱스
  // BURST=4: 버스트 오프셋 필드 인덱스
  // N_ADDRDEC=5: 배열 크기 (필드 수) — 루프 상한값으로 사용

  const char *addrdec_option;
  /* [한국어] gpgpusim.config에서 읽은 -gpgpu_mem_addr_mapping 옵션 문자열 포인터.
   * 설정자: option_parser가 config 파일을 파싱할 때 이 멤버에 문자열 주소를 저장.
   * 읽는 자: addrdec_parseoption()이 이 문자열을 파싱해 비트마스크를 구성.
   * 값 범위: NULL이면 기본 매핑 적용; 비-NULL이면 "dramid@..." 형식의 매핑 문자열.
   * 동기화: 초기화 단계에서만 쓰이고 이후 읽기 전용이므로 락 불필요. */

  int gpgpu_mem_address_mask;
  /* [한국어] 비트마스크 선택 모드 정수 — -gpgpu_mem_address_mask 옵션.
   * 설정자: option_parser가 config 파일에서 읽어 저장.
   * 읽는 자: addrdec_parseoption()이 매핑 방식을 선택할 때 참조.
   * 값 범위: 0~N (특정 사전 정의된 마스크 세트 선택; 0=기본).
   * 동기화: 초기화 후 읽기 전용. */

  partition_index_function memory_partition_indexing;
  /* [한국어] 채널 매핑 해시 정책 — -gpgpu_memory_partition_indexing 옵션.
   * 설정자: option_parser가 정수값으로 읽은 뒤 enum으로 캐스트하여 저장.
   * 읽는 자: addrdec_tlx()가 CONSECUTIVE/BITWISE_PERMUTATION/IPOLY 등으로 분기.
   * 값 범위: partition_index_function 열거형 값 (0=CONSECUTIVE ~ 5=CUSTOM).
   * 동기화: 초기화 후 읽기 전용. */

  bool run_test;
  /* [한국어] init() 완료 시 sweep_test() 실행 여부 플래그.
   * 설정자: option_parser 또는 생성자 기본값(false).
   * 읽는 자: init() 말미에서 이 값이 true이면 sweep_test()를 호출.
   * 값 범위: true(전수 검증 수행) / false(검증 생략, 시뮬레이션 속도 우선).
   * 동기화: 초기화 후 읽기 전용. */

  int ADDR_CHIP_S;
  /* [한국어] 선형 주소에서 채널(chip) 비트가 시작하는 비트 오프셋.
   * "DRAM ID shift" 값으로도 불리며, 옵션 문자열의 "dramid@<N>" 부분에서 파싱된다.
   * 설정자: addrdec_parseoption()이 "dramid@" 토큰 뒤의 정수를 파싱하여 저장.
   * 읽는 자: addrdec_tlx()에서 CONSECUTIVE 방식으로 채널 비트를 추출할 때 shift 값으로 사용.
   * 값 범위: 0~63 (비트 위치). 일반적으로 7~10 (캐시라인 크기 이상의 오프셋).
   * 동기화: 초기화 후 읽기 전용. */

  unsigned char addrdec_mklow[N_ADDRDEC];
  /* [한국어] 각 DRAM 필드 마스크에서 유효 비트의 최하위(low) 위치 배열.
   * 인덱스: CHIP=0, BK=1, ROW=2, COL=3, BURST=4.
   * 설정자: addrdec_parseoption()이 addrdec_mask[i]에서 첫 번째 set 비트 위치를 계산.
   * 읽는 자: addrdec_tlx()에서 마스크 추출 후 >> addrdec_mklow[i] 오른쪽 시프트로
   *          필드 값을 정수로 변환 (packed bits 추출 최적화).
   * 값 범위: [0, 63] (비트 위치); addrdec_mask[i]==0이면 의미 없음.
   * 동기화: 초기화 후 읽기 전용. */

  unsigned char addrdec_mkhigh[N_ADDRDEC];
  /* [한국어] 각 DRAM 필드 마스크에서 유효 비트의 최상위(high) 위치 배열.
   * 인덱스: addrdec_mklow와 동일.
   * 설정자: addrdec_parseoption()이 addrdec_mask[i]에서 마지막 set 비트 위치를 계산.
   * 읽는 자: sweep_test()에서 비트 범위가 겹치지 않는지 검증할 때 사용.
   *          addrdec_tlx()에서는 mklow와 함께 마스크 폭 계산에 참조될 수 있음.
   * 값 범위: [0, 63]; addrdec_mklow[i] <= addrdec_mkhigh[i] 항상 성립해야 함.
   * 동기화: 초기화 후 읽기 전용. */

  new_addr_type addrdec_mask[N_ADDRDEC];
  /* [한국어] CHIP/BK/ROW/COL/BURST 각 필드를 64비트 주소에서 추출하는 비트마스크 배열.
   * 인덱스: CHIP=0, BK=1, ROW=2, COL=3, BURST=4.
   * 설정자: addrdec_parseoption()이 옵션 문자열의 'C','B','R','c','b' 문자 위치를
   *          파싱하여 해당 비트를 1로 세트.
   * 읽는 자: addrdec_tlx()가 (addr & addrdec_mask[i]) >> addrdec_mklow[i] 로
   *          각 필드 값을 추출. 이 마스크가 잘못되면 전체 주소 매핑이 틀어진다.
   * 값 범위: 64비트 비트마스크 — 각 비트는 정확히 한 필드에만 속해야 함(비중복 조건).
   * 동기화: 초기화 후 읽기 전용. */

  new_addr_type sub_partition_id_mask;
  /* [한국어] 서브파티션 ID 비트를 나타내는 마스크 — partition_address() 구현의 핵심.
   * bk 마스크(addrdec_mask[BK])의 하위 log2sub_partition 비트에 해당한다.
   * 설정자: init()이 log2sub_partition을 계산한 뒤 addrdec_mask[BK]의 하위 비트로 구성.
   * 읽는 자: partition_address()에서 addr & ~sub_partition_id_mask 연산으로
   *          서브파티션 구분 비트를 제거하여 L2 태그 정규화 주소를 반환.
   * 값 범위: 64비트 마스크; n_sub_partition_in_channel==1이면 0(마스킹 없음).
   * 동기화: 초기화 후 읽기 전용. */

  unsigned int gap;
  /* [한국어] 채널 수가 2의 거듭제곱이 아닐 때의 보정 값.
   * gap = nextPowerOf2_m_n_channel - m_n_channel.
   * 설정자: init()이 m_n_channel을 설정한 직후 계산.
   * 읽는 자: addrdec_tlx()가 gap != 0일 때 모듈러 연산으로 채널 인덱스를 보정.
   *          gap == 0이면 단순 비트 마스크 추출로 충분(2의 거듭제곱 채널 수).
   * 값 범위: [0, nextPowerOf2_m_n_channel - 1]; 0이면 2의 거듭제곱 채널 구성.
   * 동기화: 초기화 후 읽기 전용. */

  unsigned m_n_channel;
  /* [한국어] DRAM 채널(메모리 파티션)의 총 수.
   * 설정자: init()의 n_channel 인자로 결정; gpgpusim.config -gpgpu_n_mem에서 기원.
   * 읽는 자: addrdec_tlx()에서 gap != 0일 때 % m_n_channel 모듈러 연산에 사용.
   *          sweep_test()에서 전수 검사 범위 상한으로 사용.
   * 값 범위: 1 이상의 정수; 일반적으로 1, 2, 4, 8, 16 등.
   * 동기화: 초기화 후 읽기 전용. */

  int m_n_sub_partition_in_channel;
  /* [한국어] 채널당 L2 서브파티션 수.
   * 설정자: init()의 n_sub_partition_in_channel 인자.
   * 읽는 자: init()에서 log2sub_partition과 sub_partition_id_mask를 계산할 때 사용.
   * 값 범위: 1 이상의 정수; 1이면 서브파티션 없이 채널 단위로만 구분.
   * 동기화: 초기화 후 읽기 전용. */

  int m_n_sub_partition_total;
  /* [한국어] 전체 서브파티션 총 수 = m_n_channel × m_n_sub_partition_in_channel.
   * 설정자: init()이 위 두 값을 곱해 저장.
   * 읽는 자: 통계 출력 또는 전체 파티션 수 기반 스케줄링 결정에 사용될 수 있음.
   * 값 범위: m_n_channel 이상; 실제 L2 서브파티션 객체 수와 일치해야 함.
   * 동기화: 초기화 후 읽기 전용. */

  unsigned log2channel;
  /* [한국어] log2(nextPowerOf2_m_n_channel) — 채널 비트 필드 폭.
   * 설정자: init()이 nextPowerOf2_m_n_channel을 계산한 후 log2를 취해 저장.
   * 읽는 자: BITWISE_PERMUTATION / IPOLY 해시에서 비트 폭을 결정할 때 사용.
   * 값 범위: 0~6 (1~64채널 범위에서); m_n_channel==1이면 0.
   * 동기화: 초기화 후 읽기 전용. */

  unsigned log2sub_partition;
  /* [한국어] log2(m_n_sub_partition_in_channel) — 서브파티션 비트 필드 폭.
   * 설정자: init()이 m_n_sub_partition_in_channel에서 계산.
   * 읽는 자: init()이 sub_partition_id_mask 구성 시 하위 비트 수를 결정할 때 사용;
   *          addrdec_tlx()에서 bk 값에서 서브파티션 ID를 분리할 때 shift 값으로 사용.
   * 값 범위: 0~N; m_n_sub_partition_in_channel==1이면 0.
   * 동기화: 초기화 후 읽기 전용. */

  unsigned nextPowerOf2_m_n_channel;
  /* [한국어] m_n_channel보다 크거나 같은 최소 2의 거듭제곱 값.
   * 설정자: init()이 m_n_channel을 읽어 계산 (예: m_n_channel=6 → 8).
   * 읽는 자: gap 계산(nextPowerOf2 - m_n_channel) 및 addrdec_tlx()의 모듈러 보정에 사용.
   *          log2channel = log2(이 값)으로도 사용.
   * 값 범위: m_n_channel 이상의 2의 거듭제곱; m_n_channel이 이미 2의 거듭제곱이면 같은 값.
   * 동기화: 초기화 후 읽기 전용. */
};

#endif  // [한국어] ADDRDEC_H include guard 종료
