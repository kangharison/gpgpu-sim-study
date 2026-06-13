// Copyright (c) 2009-2011, Tor M. Aamodt
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
 * [한국어 설명] GPU 내부 메모리 요청 패킷 정의 (mem_fetch.h)
 *
 * === 파일의 역할 ===
 * mem_fetch(Memory Fetch)는 GPGPU-Sim에서 GPU 내부를 흐르는 메모리 요청 패킷의
 * 핵심 자료구조이다. 실제 NVIDIA GPU 하드웨어에서 메모리 서브시스템 내부를 오가는
 * 요청 패킷에 해당하며, 시뮬레이터에서는 이 구조체 하나가 하나의 캐시라인 단위 읽기/쓰기
 * 요청을 완전히 캡슐화한다. L1 캐시 미스 발생 시 생성되어 ICNT(Interconnect Network,
 * SM과 L2 사이의 네트워크온칩)을 거쳐 L2 캐시, DRAM 컨트롤러까지 전달되고, 응답 시
 * 반대 방향으로 돌아온다. 이 헤더는 해당 패킷의 타입·필드·인터페이스를 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 시뮬레이션 호출 체인에서의 위치:
 *   shader core (SM 파이프라인, shader.cc)
 *     → ldst_unit::process_memory_access_queue()
 *     → [mem_fetch 생성] new mem_fetch(access, inst, ...)
 *     → ICNT (intersim2/) 전송 (IN_ICNT_TO_MEM 상태)
 *     → memory_partition_unit (gpu-sim.cc)
 *     → L2 cache (gpu-cache.cc: baseline_cache::access())
 *     → DRAM controller (dram.cc: dram_t::push())
 *     → 응답: L2 fill → ICNT → shader L1D fill → warp 재개
 *
 * 실행 컨텍스트: 호스트 유저스페이스(시뮬레이터 메인 루프). 단일 스레드 시뮬레이션
 * 사이클 루프 내에서 생성·이동·소멸한다. GPU 커널 코드가 아니라 시뮬레이터 코드임에
 * 유의한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - abstract_hardware_model.h: mem_access_t(메모리 접근 정보), warp_inst_t(워프
 *     명령어), active_mask_t(활성 스레드 마스크), mem_access_type(접근 타입 enum) 등
 *     핵심 타입을 가져온다.
 *   - addrdec.h: addrdec_t(DRAM 주소 디코딩 결과: chip/bank/row/col)를 가져온다.
 *   - mem_fetch_status.tup: X-매크로 패턴으로 mem_fetch_status enum을 생성한다.
 *     이 .tup 파일에 상태 값을 추가하면 enum과 문자열 배열이 동시에 갱신된다.
 *
 * 이 모듈에 의존하는 모듈:
 *   - gpu-cache.cc: baseline_cache::access()가 mem_fetch*를 받아 캐시 룩업 수행.
 *   - dram.cc: dram_t::push()가 mem_fetch*를 DRAM 큐에 삽입.
 *   - shader.cc: ldst_unit이 mem_fetch를 생성하고, L1D fill 응답으로 수신.
 *   - mem_latency_stat.cc: 타임스탬프 필드를 읽어 레이턴시 통계 집계.
 *   - intersim2/: ICNT 플릿(flit) 수 계산에 get_num_flits()를 호출.
 *
 * 데이터 흐름:
 *   mem_access_t (워프 접근 정보, 가상 주소·마스크 포함)
 *     → mem_fetch 생성 (물리 주소 디코딩, 메타데이터 초기화)
 *     → 각 큐를 거치며 m_status가 순차적으로 갱신됨
 *     → 응답(READ_REPLY / WRITE_ACK)으로 변환되어 shader로 복귀
 *     → 소멸 (~mem_fetch, m_status = MEM_FETCH_DELETED)
 *
 * === 주요 함수/구조체 요약 ===
 * - mem_fetch (class): 메모리 요청 패킷 전체. 주소·크기·타입·상태·타임스탬프·
 *   명령어 정보를 캡슐화한다.
 * - mf_type (enum): READ_REQUEST / WRITE_REQUEST / READ_REPLY / WRITE_ACK.
 *   패킷이 요청 방향인지 응답 방향인지를 구분.
 * - mem_fetch_status (enum): MEM_FETCH_INITIALIZED ~ MEM_FETCH_DELETED 까지
 *   패킷이 현재 어떤 큐/구조에 있는지를 나타내는 생애 주기 상태.
 * - set_reply(): 요청 방향 패킷을 응답 방향으로 변환 (READ_REQUEST→READ_REPLY,
 *   WRITE_REQUEST→WRITE_ACK).
 * - get_num_flits(): ICNT 전송 시 필요한 플릿 수 계산. 방향(simt_to_mem)에 따라
 *   데이터 크기 포함 여부가 달라진다.
 */

#ifndef MEM_FETCH_H
/* [한국어] 인클루드 가드(include guard) 시작.
 * 이 헤더가 여러 translation unit에서 중복 포함되어도 한 번만 처리되도록 보호한다.
 * MEM_FETCH_H 심볼이 정의되어 있지 않을 때만 아래 내용을 컴파일한다. */
#define MEM_FETCH_H
/* [한국어] MEM_FETCH_H 심볼을 정의하여, 이후 재포함 시 건너뛰도록 표시한다. */

#include <bitset>
/* [한국어] std::bitset<N>을 사용하기 위한 표준 라이브러리 헤더.
 * mem_access_sector_mask_t 등 섹터 마스크가 bitset으로 구현되어 있어 필요하다. */
#include "../abstract_hardware_model.h"
/* [한국어] GPGPU-Sim의 핵심 하드웨어 추상 모델 헤더.
 * mem_access_t, warp_inst_t, active_mask_t, mem_access_type, new_addr_type,
 * mem_access_byte_mask_t, mem_access_sector_mask_t 등 이 파일 전반에 걸쳐
 * 사용되는 타입들이 모두 여기서 정의된다. */
#include "addrdec.h"
/* [한국어] DRAM 물리 주소 디코딩 구조체(addrdec_t)와 선형-물리 주소 변환 클래스를
 * 가져온다. mem_fetch는 생성 시 가상 주소를 addrdec_t(chip/bank/row/col)로
 * 분해하여 m_raw_addr에 저장한다. */

/*
 * [한국어]
 * mf_type — mem_fetch 패킷의 방향/종류를 나타내는 enum.
 *
 * GPU 메모리 서브시스템에서 패킷은 크게 두 방향으로 흐른다:
 *   1) SM → 메모리 방향 (요청): READ_REQUEST, WRITE_REQUEST
 *   2) 메모리 → SM 방향 (응답): READ_REPLY, WRITE_ACK
 *
 * 이 enum은 mem_fetch::m_type 필드에 저장되며, set_reply()가 호출될 때
 * 요청 타입에서 응답 타입으로 전환된다.
 * intersim2/ NoC 및 캐시 코드가 이 값을 검사하여 패킷 처리 경로를 결정한다.
 */
enum mf_type {
  READ_REQUEST = 0,
  /* [한국어] SM(shader core)이 메모리로 보내는 읽기 요청.
   * L1D 캐시 미스 또는 L1T/L1C 미스 시 생성된다.
   * ICNT를 통해 L2 파티션으로 전달된다. */
  WRITE_REQUEST,
  /* [한국어] SM이 메모리로 보내는 쓰기 요청.
   * write-back 정책에서 dirty 캐시라인 축출(eviction) 시, 또는
   * write-through 정책에서 직접 쓰기 시 생성된다. */
  READ_REPLY,  // send to shader
  /* [한국어] 메모리(L2 또는 DRAM)에서 SM으로 보내는 읽기 응답.
   * L2 히트 또는 DRAM 데이터 반환 후 set_reply()에 의해 READ_REQUEST가
   * 이 타입으로 변환된다. ICNT를 역방향으로 통과하여 shader의
   * IN_CLUSTER_TO_SHADER_QUEUE → IN_SHADER_LDST_RESPONSE_FIFO로 들어온다. */
  WRITE_ACK
  /* [한국어] 메모리에서 SM으로 보내는 쓰기 확인 응답.
   * WRITE_REQUEST가 처리된 후 set_reply()에 의해 이 타입으로 변환된다.
   * fetch-on-write 정책 하에서는 original_wr_mf 포인터로 원본 쓰기 요청을
   * 추적한다. */
};

/*
 * [한국어]
 * X-매크로 패턴을 사용한 mem_fetch_status enum 생성.
 *
 * MF_TUP_BEGIN / MF_TUP / MF_TUP_END 매크로를 정의한 뒤
 * mem_fetch_status.tup 파일을 include하면, .tup 파일의 내용이
 * "enum mem_fetch_status { ... };"로 확장된다.
 *
 * 이 패턴의 장점: mem_fetch.cc에서 같은 .tup 파일을 다른 매크로 정의로
 * include하면 "static const char* Status_str[] = {...};"를 만들 수 있어,
 * enum 값과 문자열 배열을 항상 동기화 상태로 유지할 수 있다.
 *
 * mem_fetch_status 값의 의미 (mem_fetch_status.tup 참고):
 *   MEM_FETCH_INITIALIZED       - 생성 직후, 아직 큐에 진입 전
 *   IN_L1I_MISS_QUEUE           - L1 인스트럭션 캐시 미스 큐 대기 중
 *   IN_L1D_MISS_QUEUE           - L1 데이터 캐시 미스 큐 대기 중
 *   IN_L1T_MISS_QUEUE           - L1 텍스처 캐시 미스 큐 대기 중
 *   IN_L1C_MISS_QUEUE           - L1 상수 캐시 미스 큐 대기 중
 *   IN_L1TLB_MISS_QUEUE         - L1 TLB 미스 큐 대기 중
 *   IN_VM_MANAGER_QUEUE         - 가상 메모리 관리자 큐 대기 중
 *   IN_ICNT_TO_MEM              - ICNT를 통해 메모리 파티션으로 전송 중
 *   IN_PARTITION_ROP_DELAY      - 파티션 ROP(Raster Operations Pipeline) 지연 중
 *   IN_PARTITION_ICNT_TO_L2_QUEUE   - 파티션 내 ICNT→L2 큐 대기 중
 *   IN_PARTITION_L2_TO_DRAM_QUEUE   - L2→DRAM 큐 대기 중 (L2 미스)
 *   IN_PARTITION_DRAM_LATENCY_QUEUE - DRAM 레이턴시 시뮬레이션 큐
 *   IN_PARTITION_L2_MISS_QUEUE      - L2 미스 처리 큐
 *   IN_PARTITION_MC_INTERFACE_QUEUE - 메모리 컨트롤러 인터페이스 큐
 *   IN_PARTITION_MC_INPUT_QUEUE     - 메모리 컨트롤러 입력 큐
 *   IN_PARTITION_MC_BANK_ARB_QUEUE  - DRAM 뱅크 중재(arbitration) 큐
 *   IN_PARTITION_DRAM               - DRAM 내부 처리 중
 *   IN_PARTITION_MC_RETURNQ         - 메모리 컨트롤러 반환 큐
 *   IN_PARTITION_DRAM_TO_L2_QUEUE   - DRAM→L2 fill 큐
 *   IN_PARTITION_L2_FILL_QUEUE      - L2 캐시 fill 큐
 *   IN_PARTITION_L2_TO_ICNT_QUEUE   - L2→ICNT(SM 방향) 큐
 *   IN_ICNT_TO_SHADER               - ICNT를 통해 SM으로 전송 중
 *   IN_CLUSTER_TO_SHADER_QUEUE      - 클러스터→셰이더 큐
 *   IN_SHADER_LDST_RESPONSE_FIFO    - 셰이더 내 L/S 응답 FIFO
 *   IN_SHADER_FETCHED               - 셰이더가 응답 수신 완료
 *   IN_SHADER_L1T_ROB               - L1 텍스처 캐시 ROB(reorder buffer) 내
 *   MEM_FETCH_DELETED               - 소멸자 호출 후 (디버그용 센티넬 값)
 *   NUM_MEM_REQ_STAT                - 상태 수 계산용 센티넬 (배열 크기 등에 사용)
 */
#define MF_TUP_BEGIN(X) enum X {
/* [한국어] X-매크로 패턴: enum 선언 시작. X는 enum 이름(mem_fetch_status)이 된다. */
#define MF_TUP(X) X
/* [한국어] X-매크로 패턴: 각 enum 값을 그대로 이름으로 확장한다. */
#define MF_TUP_END(X) \
  }                   \
  ;
/* [한국어] X-매크로 패턴: enum 선언 종료. 중괄호와 세미콜론으로 마무리한다. */
#include "mem_fetch_status.tup"
/* [한국어] mem_fetch_status enum의 실제 값 목록이 이 파일에 정의되어 있다.
 * 위에서 정의한 MF_TUP_BEGIN/MF_TUP/MF_TUP_END 매크로가 적용되어
 * "enum mem_fetch_status { MEM_FETCH_INITIALIZED, IN_L1I_MISS_QUEUE, ... };"
 * 형태로 확장된다. */
#undef MF_TUP_BEGIN
/* [한국어] X-매크로 오염 방지: include 후 즉시 매크로를 해제한다.
 * mem_fetch.cc에서 같은 .tup 파일을 다른 정의로 재사용하기 위함. */
#undef MF_TUP
/* [한국어] MF_TUP 매크로 해제. */
#undef MF_TUP_END
/* [한국어] MF_TUP_END 매크로 해제. */

class memory_config;
/* [한국어] memory_config의 전방 선언(forward declaration).
 * 전방 선언만으로도 포인터/참조 타입을 선언할 수 있다.
 * 실제 정의는 gpu-sim.h / memory_config.h에 있다.
 * gpgpusim.config 파일에서 파싱된 메모리 관련 설정(DRAM 파티션 수,
 * 캐시 크기, ICNT 플릿 크기 등)을 담는다. */

/*
 * [한국어]
 * mem_fetch — GPGPU-Sim GPU 내부 메모리 요청/응답 패킷 클래스.
 *
 * 이 클래스 인스턴스 하나가 GPU 메모리 서브시스템을 흐르는
 * 단일 캐시라인 단위 메모리 트랜잭션을 완전히 표현한다.
 *
 * 생성: shader core의 ldst_unit이 L1D/L1T/L1C 캐시 미스 시 new mem_fetch()로 생성.
 * 소멸: 응답이 shader에 전달된 후, 또는 write-back 완료 후 delete로 소멸.
 *
 * 생애 주기 (m_type + m_status 조합으로 추적):
 *   생성(READ_REQUEST/WRITE_REQUEST, MEM_FETCH_INITIALIZED)
 *   → L1 미스 큐(IN_L1D_MISS_QUEUE 등)
 *   → ICNT 전송(IN_ICNT_TO_MEM)
 *   → L2 파티션 내부 큐들 (IN_PARTITION_* 시리즈)
 *   → DRAM 처리(IN_PARTITION_DRAM)
 *   → L2 fill / L2→ICNT
 *   → set_reply() 호출 → READ_REPLY / WRITE_ACK로 타입 변환
 *   → ICNT 역방향(IN_ICNT_TO_SHADER)
 *   → shader 응답 큐(IN_SHADER_LDST_RESPONSE_FIFO)
 *   → 소멸(MEM_FETCH_DELETED)
 *
 * 사이클-레벨 모델링: 각 사이클에 시뮬레이터 메인 루프가 큐를 순회하며
 * mem_fetch를 다음 단계로 이동시킨다. mem_fetch 자체는 사이클 개념을 갖지 않고,
 * 외부 루프(gpgpu_sim::cycle())가 타이밍을 관리한다.
 */
class mem_fetch {
 public:
  /*
   * [한국어]
   * mem_fetch 생성자 — 메모리 요청 패킷을 초기화한다.
   *
   * @access:        메모리 접근 정보 (가상 주소, 접근 타입, 바이트/섹터 마스크,
   *                 워프 활성 마스크 포함). abstract_hardware_model.h의 mem_access_t.
   *                 L1D 캐시 미스 처리 코드(gpu-cache.cc)가 생성하여 전달한다.
   * @inst:          이 메모리 요청을 발생시킨 워프 명령어 포인터 (NULL 가능).
   *                 NULL이면 m_inst가 empty 상태로 유지된다 (writeback 요청 등).
   * @streamID:      이 요청이 속한 CUDA 스트림(stream)의 ID.
   *                 스트림별 통계 및 동기화에 사용된다.
   * @ctrl_size:     하드웨어 관점의 메타데이터(컨트롤) 패킷 크기 (바이트).
   *                 실제 C++ 객체 크기와 다를 수 있다; NoC 플릿 수 계산에 사용.
   * @wid:           요청 워프 ID. m_inst가 있을 때 m_inst.warp_id()와 일치하는지
   *                 assert로 검증한다.
   * @sid:           요청 SM(shader core)의 ID (shader index).
   * @tpc:           TPC(Texture Processing Cluster) ID. SM들의 클러스터 단위.
   * @config:        시뮬레이터 메모리 설정 포인터. DRAM 파티션 수, 주소 매핑,
   *                 ICNT 플릿 크기 등을 제공한다. gpgpusim.config에서 파싱됨.
   * @cycle:         생성 시각 (gpu_sim_cycle + gpu_tot_sim_cycle). 레이턴시 측정 기준.
   * @original_mf:   L2 캐시에서 섹터 단위로 분할될 때 원본 요청 포인터 (NULL 가능).
   *                 요청이 L2 섹터 크기보다 크면 여러 개로 나뉘며, 각 조각이
   *                 원본 패킷을 이 포인터로 참조한다.
   * @original_wr_mf: fetch-on-write 정책에서 원본 쓰기 요청 포인터 (NULL 가능).
   *                 쓰기 전 읽기(fetch)를 위해 READ_REQUEST가 생성될 때 설정된다.
   * @return: (없음, 생성자)
   *
   * 동작 과정:
   *   1) 전역 카운터 sm_next_mf_request_uid에서 고유 ID를 할당한다.
   *   2) mem_access_t를 복사하고 warp_inst_t를 복사(inst != NULL이면)한다.
   *   3) SST 모드가 아니면 가상 주소를 addrdec_tlx()로 물리 DRAM 주소로 변환한다.
   *   4) 쓰기 여부에 따라 m_type을 WRITE_REQUEST / READ_REQUEST로 설정한다.
   *   5) 타임스탬프, 상태(MEM_FETCH_INITIALIZED), 설정 포인터를 초기화한다.
   *   6) original_mf가 있으면 chip/sub_partition 정보를 복사한다.
   *
   * 실행 컨텍스트: 시뮬레이터 메인 루프의 단일 스레드. 재진입 불필요.
   * 호출 체인: ldst_unit::process_memory_access_queue() / baseline_cache::send_read_request()
   *            → [mem_fetch 생성자] → (패킷 큐 삽입)
   */
  mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
            unsigned long long streamID, unsigned ctrl_size, unsigned wid,
            unsigned sid, unsigned tpc, const memory_config *config,
            unsigned long long cycle, mem_fetch *original_mf = NULL,
            mem_fetch *original_wr_mf = NULL);

  /*
   * [한국어]
   * ~mem_fetch 소멸자 — 패킷 소멸 시 상태를 MEM_FETCH_DELETED로 표시한다.
   *
   * @return: (없음, 소멸자)
   *
   * 실제로 동적으로 할당된 멤버가 없으므로 메모리 해제 작업은 없다.
   * m_status를 MEM_FETCH_DELETED로 설정하는 이유: 이미 삭제된 패킷을
   * 잘못 참조하는 댕글링 포인터 버그를 디버깅할 때 상태값으로 탐지하기 위함.
   *
   * 호출 체인: 응답 처리 완료 후 delete mf → [~mem_fetch()]
   */
  ~mem_fetch();

  /*
   * [한국어]
   * set_status — 패킷의 현재 위치/상태를 갱신한다.
   *
   * @status: 새로운 mem_fetch_status 값. 패킷이 진입한 큐나 처리 단계를 나타낸다.
   * @cycle:  상태 변경이 일어난 시뮬레이션 사이클 번호. m_status_change에 저장되어
   *          레이턴시 분석 및 디버깅에 사용된다.
   * @return: (없음)
   *
   * m_status와 m_status_change를 동시에 갱신하여 "언제 어디 있었는지"를 추적한다.
   * 각 큐/모듈이 패킷을 받을 때 이 함수를 호출하여 상태를 갱신한다.
   *
   * 호출 체인: baseline_cache::access() / dram_t::push() / icnt_push() 등
   *            → [set_status()] → (m_status, m_status_change 갱신)
   */
  void set_status(enum mem_fetch_status status, unsigned long long cycle);

  /*
   * [한국어]
   * set_reply — 요청 패킷을 응답 패킷으로 변환한다 (인라인 함수).
   *
   * @return: (없음)
   *
   * L2 캐시 히트 또는 DRAM 데이터 반환 후, 패킷을 SM 방향 응답으로 변환할 때 호출.
   * 변환 규칙:
   *   READ_REQUEST  → READ_REPLY  (읽기 요청 → 데이터 포함 응답)
   *   WRITE_REQUEST → WRITE_ACK   (쓰기 요청 → 쓰기 확인)
   *
   * 사전조건 (assert로 검증):
   *   - L1/L2 writeback 접근 타입(L1_WRBK_ACC, L2_WRBK_ACC)은 reply 변환 금지.
   *     writeback은 단방향 요청이므로 응답 패킷이 존재하지 않는다.
   *   - READ_REQUEST의 경우 is_write()가 false이어야 한다.
   *   - WRITE_REQUEST의 경우 is_write()가 true이어야 한다.
   *
   * 호출 체인: memory_sub_partition::service_mem_req() / L2 히트 처리 코드
   *            → [set_reply()] → m_type 변환 → ICNT 역방향 전송
   */
  void set_reply() {
    assert(m_access.get_type() != L1_WRBK_ACC &&
           m_access.get_type() != L2_WRBK_ACC);
    /* [한국어] L1/L2 write-back 접근은 reply 변환 불가 확인.
     * write-back 패킷은 SM으로 돌아갈 응답이 없으므로 이 경로를 타면 안 된다. */
    if (m_type == READ_REQUEST) {
      /* [한국어] 읽기 요청인 경우: READ_REQUEST → READ_REPLY로 변환. */
      assert(!get_is_write());
      /* [한국어] 읽기 요청임에도 is_write가 true이면 일관성 오류. */
      m_type = READ_REPLY;
      /* [한국어] 타입을 읽기 응답으로 변환. 이후 ICNT 역방향으로 전송된다. */
    } else if (m_type == WRITE_REQUEST) {
      /* [한국어] 쓰기 요청인 경우: WRITE_REQUEST → WRITE_ACK로 변환. */
      assert(get_is_write());
      /* [한국어] 쓰기 요청임을 확인. */
      m_type = WRITE_ACK;
      /* [한국어] 타입을 쓰기 확인 응답으로 변환. */
    }
  }

  /*
   * [한국어]
   * do_atomic — 이 패킷이 요청하는 원자적(atomic) 연산을 실제로 실행한다.
   *
   * @return: (없음)
   *
   * GPU의 atomicAdd, atomicCAS 등 원자 연산은 메모리 계층에서 실행된다.
   * DRAM 또는 L2 캐시에서 데이터가 반환될 때, 이 함수를 호출하여
   * 해당 warp 명령어의 원자 연산 시맨틱을 적용한다.
   * m_access.get_warp_mask()를 사용하여 활성 스레드에 대해서만 원자 연산을 수행.
   *
   * 실행 컨텍스트: 메모리 파티션 처리 코드(memory_sub_partition)에서 호출.
   * 호출 체인: memory_sub_partition::service_mem_req()
   *            → [do_atomic()] → m_inst.do_atomic(warp_mask)
   */
  void do_atomic();

  /*
   * [한국어]
   * print — 패킷의 모든 정보를 파일 스트림에 사람이 읽을 수 있는 형태로 출력한다.
   *
   * @fp:         출력 대상 파일 포인터 (stdout, stderr, 또는 로그 파일).
   * @print_inst: true이면 연관된 warp 명령어 정보도 함께 출력한다 (기본값: true).
   * @return: (없음)
   *
   * 디버깅 및 성능 분석 시 사용. 출력 형식:
   *   "  mf: uid=<uid>, sid<SM_ID>:w<warp_ID>, part=<chip>, <access정보>,
   *    status = <상태문자열> (<상태변경사이클>), [명령어정보]"
   *
   * 호출 체인: 디버그 출력 코드, visualizer.cc, 성능 분석 도구
   *            → [print()] → m_access.print() + m_inst.print()
   */
  void print(FILE *fp, bool print_inst = true) const;

  /*
   * [한국어]
   * get_tlx_addr — DRAM 물리 주소 디코딩 결과(addrdec_t)를 반환한다.
   *
   * @return: m_raw_addr 상수 참조. chip/bk/row/col/burst/sub_partition 필드 포함.
   *
   * DRAM 컨트롤러(dram.cc)가 요청을 특정 뱅크/행으로 라우팅할 때 사용.
   * 또한 L2 파티션 선택(sub_partition)에도 사용된다.
   */
  const addrdec_t &get_tlx_addr() const { return m_raw_addr; }

  /*
   * [한국어]
   * set_chip — m_raw_addr의 chip(메모리 파티션/채널) 번호를 설정한다.
   *
   * @chip_id: 메모리 파티션 번호 (0 ~ N_DRAM_PARTITION-1).
   * @return: (없음)
   *
   * 섹터 분할 요청에서 original_mf의 chip 정보를 파생 요청에 복사할 때 사용.
   * memory_sub_partition이 요청을 특정 DRAM 파티션으로 라우팅한 후 호출.
   */
  void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }

  /*
   * [한국어]
   * set_partition — m_raw_addr의 sub_partition 번호를 설정한다.
   *
   * @sub_partition_id: 채널 내 서브 파티션 번호 (0 ~ N_SUB_PARTITION-1).
   * @return: (없음)
   *
   * 생성자에서 addrdec_tlx()로 자동 설정되지만, 섹터 분할 시
   * original_mf의 값을 파생 요청에 복사하기 위해 명시적으로 호출된다.
   */
  void set_partition(unsigned sub_partition_id) {
    m_raw_addr.sub_partition = sub_partition_id;
  }

  /*
   * [한국어]
   * get_data_size — 이 요청이 전송하는 실제 데이터 크기(바이트)를 반환한다.
   *
   * @return: m_data_size (바이트). L1D 캐시라인 크기(보통 128B) 또는 섹터 크기.
   *
   * 읽기 응답 시 반환되는 데이터 크기이며, NoC 플릿 수 계산에도 사용된다.
   */
  unsigned get_data_size() const { return m_data_size; }

  /*
   * [한국어]
   * set_data_size — 이 요청의 데이터 크기를 설정한다.
   *
   * @size: 새로운 데이터 크기(바이트).
   * @return: (없음)
   *
   * L2에서 섹터 단위로 요청을 분할할 때 각 파생 요청의 크기를 조정하기 위해 사용.
   */
  void set_data_size(unsigned size) { m_data_size = size; }

  /*
   * [한국어]
   * get_ctrl_size — 하드웨어 관점의 컨트롤(메타데이터) 패킷 크기를 반환한다.
   *
   * @return: m_ctrl_size (바이트). 주소, 타입 등 메타데이터의 하드웨어 크기.
   *
   * 쓰기 요청(SM→메모리)에서는 데이터 없이 컨트롤 크기만 ICNT로 전송되는 경우도 있다.
   * get_num_flits()가 이 값을 데이터 크기와 합산하여 플릿 수를 계산한다.
   */
  unsigned get_ctrl_size() const { return m_ctrl_size; }

  /*
   * [한국어]
   * size — 이 패킷의 전체 크기(데이터 + 컨트롤)를 반환한다.
   *
   * @return: m_data_size + m_ctrl_size (바이트).
   *
   * 원자 연산, 데이터 포함 전송 시의 전체 ICNT 페이로드 크기를 나타낸다.
   * get_num_flits()에서 원자/읽기응답/쓰기요청 시 이 값을 기준으로 플릿 수 계산.
   */
  unsigned size() const { return m_data_size + m_ctrl_size; }

  /*
   * [한국어]
   * is_write — 이 요청이 쓰기(write) 요청인지 반환한다 (비상수 버전).
   *
   * @return: true면 쓰기 요청, false면 읽기 요청.
   *
   * m_access.is_write()에 위임한다.
   * write-back 처리 경로, DRAM 스케줄러가 읽기/쓰기를 구분할 때 사용.
   */
  bool is_write() { return m_access.is_write(); }

  /*
   * [한국어]
   * set_addr — m_access의 주소를 재설정한다.
   *
   * @addr: 새로운 메모리 주소 (new_addr_type = 64비트 정수).
   * @return: (없음)
   *
   * 캐시 miss penalty 계산이나 주소 정렬 조정이 필요한 경우 사용.
   */
  void set_addr(new_addr_type addr) { m_access.set_addr(addr); }

  /*
   * [한국어]
   * get_addr — 이 요청의 메모리 주소(가상 또는 물리, 캐시라인 정렬)를 반환한다.
   *
   * @return: m_access.get_addr(). 캐시라인 크기로 정렬된 주소값.
   *
   * 캐시 태그 비교, MSHR(Miss Status Holding Register) 병합 검색 등에 사용.
   */
  new_addr_type get_addr() const { return m_access.get_addr(); }

  /*
   * [한국어]
   * get_access_size — mem_access_t에 기록된 접근 크기를 반환한다.
   *
   * @return: m_access.get_size() (바이트). 워프가 실제 접근하는 크기.
   *
   * m_data_size(캐시라인 단위)와 다를 수 있다. mem_access_t는 워프의
   * 실제 접근 범위를 나타내고, m_data_size는 캐시 라인 전체 크기다.
   */
  unsigned get_access_size() const { return m_access.get_size(); }

  /*
   * [한국어]
   * get_partition_addr — DRAM 파티션 내부의 선형 주소를 반환한다.
   *
   * @return: m_partition_addr. 뱅크 선택 비트를 제거한 파티션 내 선형 주소.
   *
   * 파티션 내에서 행(row) 버퍼 히트 판단, DRAM 스케줄링에 사용.
   * addrdec.h의 partition_address()가 계산한 값으로 생성자에서 초기화된다.
   */
  new_addr_type get_partition_addr() const { return m_partition_addr; }

  /*
   * [한국어]
   * get_sub_partition_id — 채널 내 서브 파티션 번호를 반환한다.
   *
   * @return: m_raw_addr.sub_partition (0 ~ N_SUB_PARTITION-1).
   *
   * memory_sub_partition이 자신에게 들어온 요청을 확인하거나,
   * L2 파티션 큐 라우팅에 사용된다.
   */
  unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }

  /*
   * [한국어]
   * get_is_write — 이 요청이 쓰기인지 반환한다 (상수 버전).
   *
   * @return: true면 쓰기, false면 읽기.
   *
   * is_write()의 const 버전. set_reply() 내부의 assert 검증 및
   * get_num_flits()의 방향 판단에 사용된다.
   */
  bool get_is_write() const { return m_access.is_write(); }

  /*
   * [한국어]
   * get_request_uid — 이 패킷의 전역 고유 ID를 반환한다.
   *
   * @return: m_request_uid. 시뮬레이터 전체에서 유일한 패킷 식별자.
   *
   * 디버깅 출력, 성능 분석 도구에서 특정 패킷을 추적할 때 사용.
   * sm_next_mf_request_uid(전역 카운터)에서 생성 시 할당된다.
   */
  unsigned get_request_uid() const { return m_request_uid; }

  /*
   * [한국어]
   * get_sid — 이 요청을 발생시킨 SM(shader core)의 인덱스를 반환한다.
   *
   * @return: m_sid (0 ~ N_SM-1).
   *
   * 통계 집계, 응답 패킷을 올바른 SM으로 라우팅할 때 사용.
   */
  unsigned get_sid() const { return m_sid; }

  /*
   * [한국어]
   * get_tpc — 이 요청이 속한 TPC(Texture Processing Cluster) ID를 반환한다.
   *
   * @return: m_tpc. SM들의 클러스터 단위 ID.
   *
   * NoC 라우팅에서 소스 노드 식별, 통계 집계에 사용.
   */
  unsigned get_tpc() const { return m_tpc; }

  /*
   * [한국어]
   * get_wid — 이 요청을 발생시킨 워프(warp)의 ID를 반환한다.
   *
   * @return: m_wid (0 ~ N_WARP_PER_SM-1).
   *
   * 응답 수신 후 해당 워프를 재개(ready 상태로 전환)할 때,
   * scoreboard 해제 시 워프를 특정하는 데 사용.
   */
  unsigned get_wid() const { return m_wid; }

  /*
   * [한국어]
   * istexture — 이 요청이 텍스처 메모리 접근인지 반환한다.
   *
   * @return: true면 tex_space 접근.
   *
   * 텍스처 캐시(L1T) 경로로 라우팅할지, 일반 L1D 경로로 라우팅할지 결정.
   * m_inst가 비어 있으면 false를 반환한다 (writeback 등).
   *
   * 호출 체인: shader core 내 메모리 접근 분류 코드 → [istexture()]
   */
  bool istexture() const;

  /*
   * [한국어]
   * isconst — 이 요청이 상수(const) 또는 커널 파라미터 메모리 접근인지 반환한다.
   *
   * @return: true면 const_space 또는 param_space_kernel 접근.
   *
   * 상수 캐시(L1C) 경로로 라우팅할지 결정. m_inst가 비어 있으면 false.
   *
   * 호출 체인: shader core 메모리 접근 분류 코드 → [isconst()]
   */
  bool isconst() const;

  /*
   * [한국어]
   * get_type — 이 패킷의 현재 방향/종류(mf_type)를 반환한다.
   *
   * @return: READ_REQUEST / WRITE_REQUEST / READ_REPLY / WRITE_ACK 중 하나.
   *
   * ICNT, 캐시, DRAM 코드가 패킷을 처리하는 방법을 결정할 때 사용.
   */
  enum mf_type get_type() const { return m_type; }

  /*
   * [한국어]
   * isatomic — 이 요청이 원자적(atomic) 연산인지 반환한다.
   *
   * @return: true면 atomicAdd, atomicCAS 등의 원자 연산.
   *
   * 원자 연산은 데이터를 읽고 수정하여 다시 쓰는 과정이 원자적으로 일어나야 하므로
   * 특별한 처리가 필요하다. m_inst.isatomic()에 위임하며, m_inst가 비어 있으면 false.
   *
   * 호출 체인: memory_sub_partition / get_num_flits() → [isatomic()]
   */
  bool isatomic() const;

  /*
   * [한국어]
   * set_return_timestamp — 패킷이 ICNT를 통해 SM으로 전송될 때의 시각을 기록한다.
   *
   * @t: 기록할 시뮬레이션 사이클 번호.
   * @return: (없음)
   *
   * m_timestamp2에 저장되며, 읽기 응답의 ICNT→SM 구간 레이턴시 측정에 사용.
   * shader 방향 ICNT 큐에 삽입될 때 호출된다.
   */
  void set_return_timestamp(unsigned t) { m_timestamp2 = t; }

  /*
   * [한국어]
   * set_icnt_receive_time — 고정 ICNT 레이턴시 모드에서 수신 예상 시각을 기록한다.
   *
   * @t: gpu_sim_cycle + interconnect_latency 로 계산된 예상 도착 사이클.
   * @return: (없음)
   *
   * fixed latency 모드(gpgpusim.config에서 설정)에서 ICNT 전송 대신
   * 단순 지연 큐로 시뮬레이션할 때 이 값을 기준으로 도착 여부를 판단한다.
   */
  void set_icnt_receive_time(unsigned t) { m_icnt_receive_time = t; }

  /*
   * [한국어]
   * get_timestamp — 패킷 생성 시각을 반환한다.
   *
   * @return: m_timestamp (gpu_sim_cycle + gpu_tot_sim_cycle at 생성 시점).
   *
   * 전체 메모리 레이턴시(생성~응답 수신) 계산의 시작점으로 사용.
   * mem_latency_stat.cc가 이 값을 읽어 레이턴시 히스토그램을 생성한다.
   */
  unsigned get_timestamp() const { return m_timestamp; }

  /*
   * [한국어]
   * get_return_timestamp — 패킷이 SM 방향 ICNT로 전송된 시각을 반환한다.
   *
   * @return: m_timestamp2. 읽기 전용(READ_REPLY)에서만 유효하다.
   *
   * 메모리→SM 구간 레이턴시 측정에 사용. 쓰기 ACK에서는 의미 없는 값이다.
   */
  unsigned get_return_timestamp() const { return m_timestamp2; }

  /*
   * [한국어]
   * get_icnt_receive_time — 고정 ICNT 레이턴시 모드의 예상 수신 시각을 반환한다.
   *
   * @return: m_icnt_receive_time. set_icnt_receive_time()으로 설정된 값.
   *
   * fixed latency ICNT 모드에서 패킷이 아직 도착하지 않았는지 확인할 때 사용.
   */
  unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

  /*
   * [한국어]
   * get_streamID — 이 요청이 속한 CUDA 스트림 ID를 반환한다.
   *
   * @return: m_streamID (64비트 정수). CUDA 스트림 핸들.
   *
   * 스트림별 동기화, 통계 집계, 멀티-스트림 시나리오에서 요청을 구분할 때 사용.
   */
  unsigned long long get_streamID() const { return m_streamID; }

  /*
   * [한국어]
   * get_access_type — mem_access_t의 접근 타입(mem_access_type enum)을 반환한다.
   *
   * @return: GLOBAL_ACC_R, GLOBAL_ACC_W, L1_WRBK_ACC, L2_WRBK_ACC, TEXTURE_ACC_R 등.
   *
   * 캐시 정책(write-back/write-through/write-evict) 결정, 통계 분류에 사용.
   * set_reply()의 assert에서 L1_WRBK_ACC / L2_WRBK_ACC를 걸러내는 데도 사용.
   */
  enum mem_access_type get_access_type() const { return m_access.get_type(); }

  /*
   * [한국어]
   * get_access_warp_mask — 이 요청을 발생시킨 워프의 활성 스레드 마스크를 반환한다.
   *
   * @return: active_mask_t 상수 참조 (32비트 bitset). 비트 i가 1이면 스레드 i 활성.
   *
   * 원자 연산 시 어떤 스레드의 연산을 수행할지 결정하는 데 사용.
   * do_atomic()이 m_access.get_warp_mask()로 이 값을 가져온다.
   */
  const active_mask_t &get_access_warp_mask() const {
    return m_access.get_warp_mask();
  }

  /*
   * [한국어]
   * get_access_byte_mask — 캐시라인 내 접근된 바이트들의 마스크를 반환한다.
   *
   * @return: mem_access_byte_mask_t. 캐시라인 크기만큼의 비트 배열.
   *          비트 i가 1이면 오프셋 i 바이트에 접근함.
   *
   * write-byte-enable, partial write 처리, 통계 집계에 사용.
   */
  mem_access_byte_mask_t get_access_byte_mask() const {
    return m_access.get_byte_mask();
  }

  /*
   * [한국어]
   * get_access_sector_mask — 접근된 캐시 섹터들의 마스크를 반환한다.
   *
   * @return: mem_access_sector_mask_t. 캐시라인을 섹터 단위로 나눈 마스크.
   *
   * L2 캐시가 섹터 단위로 요청을 처리할 때, 어느 섹터가 필요한지 판단.
   * 섹터 캐시(sector cache) 모드에서 중요하다.
   */
  mem_access_sector_mask_t get_access_sector_mask() const {
    return m_access.get_sector_mask();
  }

  /*
   * [한국어]
   * get_pc — 이 요청을 발생시킨 명령어의 PC(Program Counter)를 반환한다.
   *
   * @return: m_inst.pc (명령어 주소). m_inst가 비어 있으면 -1을 반환.
   *
   * 디버깅, 명령어-레벨 메모리 접근 추적, visualizer에 사용.
   */
  address_type get_pc() const { return m_inst.empty() ? -1 : m_inst.pc; }

  /*
   * [한국어]
   * get_inst — 이 요청을 발생시킨 warp_inst_t 객체의 참조를 반환한다.
   *
   * @return: m_inst 참조. isatomic(), istexture(), isconst() 등의 판단에 사용.
   *
   * 비상수 참조를 반환하므로 호출자가 명령어 상태를 수정할 수 있다.
   * do_atomic()이 m_inst.do_atomic()으로 원자 연산을 실제 수행할 때 사용.
   */
  const warp_inst_t &get_inst() { return m_inst; }

  /*
   * [한국어]
   * get_status — 패킷의 현재 위치/상태(mem_fetch_status)를 반환한다.
   *
   * @return: m_status. MEM_FETCH_INITIALIZED ~ MEM_FETCH_DELETED 범위.
   *
   * 디버깅, 시각화 도구(visualizer.cc), 레이턴시 통계 수집에 사용.
   */
  enum mem_fetch_status get_status() const { return m_status; }

  /*
   * [한국어]
   * get_mem_config — 이 패킷이 사용하는 메모리 설정 포인터를 반환한다.
   *
   * @return: m_mem_config 포인터 (memory_config*).
   *
   * 주로 내부 검증이나 설정 참조가 필요한 외부 코드에서 사용.
   * DRAM 파티션 수, 캐시 설정, ICNT 플릿 크기 등에 접근 가능.
   */
  const memory_config *get_mem_config() { return m_mem_config; }

  /*
   * [한국어]
   * get_num_flits — ICNT 전송 시 필요한 플릿(flit) 수를 계산한다.
   *
   * @simt_to_mem: true면 SM→메모리 방향, false면 메모리→SM 방향.
   * @return: 이 패킷이 ICNT를 통과할 때 사용하는 플릿 수 (최소 1).
   *
   * 전송 크기 결정 규칙:
   *   - 원자 연산: 항상 전체 크기(ctrl + data) 사용.
   *   - SM→메모리 방향 쓰기 요청: 데이터 포함 → 전체 크기.
   *   - 메모리→SM 방향 읽기 응답: 데이터 포함 → 전체 크기.
   *   - 그 외(SM→메모리 읽기 요청, 메모리→SM 쓰기 ACK): ctrl 크기만.
   *
   * 플릿 수 = ceil(전송크기 / icnt_flit_size). 나머지가 있으면 1 플릿 추가.
   *
   * 호출 체인: intersim2 ICNT 삽입 코드 → [get_num_flits(simt_to_mem)]
   *            → (플릿 수만큼 ICNT 자원 예약)
   */
  unsigned get_num_flits(bool simt_to_mem);

  /*
   * [한국어]
   * get_original_mf — L2 섹터 분할 시 원본 요청 포인터를 반환한다.
   *
   * @return: original_mf 포인터. 분할되지 않은 요청이면 NULL.
   *
   * 섹터 단위로 분할된 파생 요청들이 완료될 때, 원본 패킷을 다시 찾아
   * 상위 레벨로 응답을 전달하는 데 사용된다.
   */
  mem_fetch *get_original_mf() { return original_mf; }

  /*
   * [한국어]
   * get_original_wr_mf — fetch-on-write 시 원본 쓰기 요청 포인터를 반환한다.
   *
   * @return: original_wr_mf 포인터. fetch-on-write가 아닌 경우 NULL.
   *
   * fetch-on-write 정책: 쓰기 전에 먼저 해당 캐시라인을 읽어 오는 방식.
   * 이때 읽기 요청(READ_REQUEST)이 생성되며, 원본 쓰기 요청을 이 포인터로 참조한다.
   * 읽기 완료 후 원본 쓰기 요청을 다시 처리한다.
   */
  mem_fetch *get_original_wr_mf() { return original_wr_mf; }

 private:
  // request source information
  unsigned m_request_uid;
  /* [한국어] 이 패킷의 전역 고유 식별자(Unique ID).
   * 설정자: 생성자에서 sm_next_mf_request_uid++ 로 원자적으로 할당.
   *         (단일 스레드 시뮬레이터이므로 실제 원자 연산은 불필요하지만
   *          스타일상 전역 카운터를 후위 증가로 사용.)
   * 읽는 자: print() 출력, 디버거, visualizer.cc 추적 코드.
   * 값 범위: 1부터 시작하여 단조 증가 (0은 유효하지 않은 ID로 예약).
   * 동기화: 단일 스레드 시뮬레이터이므로 별도 락 불필요. */

  unsigned m_sid;
  /* [한국어] 이 요청을 발생시킨 SM(shader core)의 인덱스.
   * 설정자: 생성자의 @sid 파라미터로 초기화.
   * 읽는 자: get_sid(), print(), ICNT 라우팅 소스 노드 계산 코드.
   * 값 범위: 0 ~ (총 SM 수 - 1). gpgpusim.config의 -gpgpu_n_clusters 등에 의존.
   * 동기화: 생성 후 불변(immutable). 락 불필요. */

  unsigned m_tpc;
  /* [한국어] 이 요청이 속한 TPC(Texture Processing Cluster)의 인덱스.
   * TPC는 여러 SM을 묶는 클러스터 단위로, NVIDIA GPU 하드웨어 구조를 반영한다.
   * 설정자: 생성자의 @tpc 파라미터로 초기화.
   * 읽는 자: get_tpc(), NoC 라우팅, 통계 집계.
   * 값 범위: 0 ~ (총 TPC 수 - 1).
   * 동기화: 생성 후 불변. */

  unsigned m_wid;
  /* [한국어] 이 요청을 발생시킨 워프(warp)의 SM 내 인덱스.
   * 설정자: 생성자의 @wid 파라미터로 초기화.
   *         inst != NULL이면 assert(wid == m_inst.warp_id())로 일관성 검증.
   * 읽는 자: get_wid(), 응답 수신 후 해당 워프 재개 코드(shader.cc의 warp 재활성화),
   *           scoreboard 해제 코드.
   * 값 범위: 0 ~ (SM당 최대 워프 수 - 1).
   * 동기화: 생성 후 불변. */

  // where is this request now?
  enum mem_fetch_status m_status;
  /* [한국어] 패킷의 현재 위치/상태.
   * MEM_FETCH_INITIALIZED에서 시작하여 각 큐 진입 시 갱신되며
   * MEM_FETCH_DELETED로 끝난다.
   * 설정자: 생성자(MEM_FETCH_INITIALIZED), set_status()로 각 큐 진입 시,
   *         소멸자(MEM_FETCH_DELETED).
   * 읽는 자: get_status(), print(), visualizer.cc, mem_latency_stat.cc.
   * 값 범위: mem_fetch_status enum의 모든 값 (MEM_FETCH_INITIALIZED ~
   *           MEM_FETCH_DELETED).
   * 동기화: 단일 스레드 시뮬레이터. 큐를 이동할 때마다 set_status()로 갱신. */

  unsigned long long m_status_change;
  /* [한국어] 마지막으로 m_status가 변경된 시뮬레이션 사이클 번호.
   * 설정자: 생성자(생성 사이클), set_status()에서 status 변경 시점의 cycle.
   * 읽는 자: print() — 상태와 함께 "(사이클번호)"로 출력.
   *           디버깅 시 "이 패킷이 얼마나 오랫동안 특정 상태에 있었나" 파악에 사용.
   * 값 범위: 0 이상의 사이클 번호 (gpu_sim_cycle + gpu_tot_sim_cycle).
   * 동기화: m_status와 항상 쌍으로 갱신. 단일 스레드. */

  // request type, address, size, mask
  mem_access_t m_access;
  /* [한국어] 이 요청의 핵심 접근 정보를 담는 mem_access_t 구조체.
   * 메모리 주소, 접근 타입(GLOBAL_ACC_R/W, TEXTURE 등), 접근 크기,
   * 활성 워프 스레드 마스크, 바이트 마스크, 섹터 마스크를 포함한다.
   * 설정자: 생성자의 @access 파라미터로 복사 초기화.
   *         set_addr()로 주소만 재설정 가능.
   * 읽는 자: get_addr(), get_access_type(), get_access_warp_mask() 등
   *           모든 m_access.xxx() 래퍼 메서드.
   * 값 범위: abstract_hardware_model.h의 mem_access_t 정의에 따름.
   * 동기화: 생성 후 주소 외 필드는 불변. 단일 스레드. */

  unsigned m_data_size;  // how much data is being written
  /* [한국어] 이 요청이 전송하는 데이터 페이로드의 크기(바이트).
   * 일반적으로 캐시라인 크기(예: 128B)와 일치하지만,
   * 섹터 분할 시 섹터 크기(예: 32B)가 될 수 있다.
   * 설정자: 생성자에서 access.get_size()로 초기화.
   *         set_data_size()로 섹터 분할 후 조정 가능.
   * 읽는 자: get_data_size(), size(), get_num_flits().
   * 값 범위: 1 ~ 캐시라인 크기(보통 128) 바이트.
   * 동기화: 단일 스레드. */

  unsigned
      m_ctrl_size;  // how big would all this meta data be in hardware (does not
                    // necessarily match actual size of mem_fetch)
  /* [한국어] 하드웨어에서 이 패킷의 메타데이터(제어 정보)가 차지하는 크기(바이트).
   * 실제 C++ mem_fetch 객체의 sizeof(mem_fetch)와 다르다.
   * 이 값은 하드웨어 구현에서 주소, 타입, ID 등의 컨트롤 필드가 차지하는
   * 비트 수를 바이트로 환산한 값이다.
   * 설정자: 생성자의 @ctrl_size 파라미터로 초기화.
   * 읽는 자: get_ctrl_size(), size(), get_num_flits().
   *           — 특히 get_num_flits()에서 읽기 요청(SM→메모리 방향, 데이터 없음)
   *             의 ICNT 플릿 수를 계산할 때 이 값만 사용한다.
   * 값 범위: 하드웨어 설계에 따라 다름 (보통 수 바이트 ~ 수십 바이트).
   * 동기화: 생성 후 불변. */

  new_addr_type
      m_partition_addr;  // linear physical address *within* dram partition
                         // (partition bank select bits squeezed out)
  /* [한국어] DRAM 파티션 내부의 선형 물리 주소.
   * 전체 물리 주소에서 파티션 선택 비트(bank interleaving에 쓰이는 비트)를
   * 제거(squeeze out)한 나머지 주소이다.
   * 설정자: 생성자에서 config->m_address_mapping.partition_address(addr)로 계산.
   *         SST 모드에서는 설정되지 않는다.
   * 읽는 자: get_partition_addr(). DRAM 스케줄러가 row-buffer hit 판단에 사용.
   * 값 범위: 0 ~ (DRAM 파티션 용량 - 1).
   * 동기화: 생성 후 불변. */

  addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
                         // chip-row-bank-column address)
  /* [한국어] DRAM 물리 주소를 chip/bank/row/col/burst/sub_partition으로 분해한 구조체.
   * addrdec.h의 addrdec_t 타입이다. GPU DRAM은 여러 채널(chip)과 각 채널 내
   * 여러 뱅크(bank)로 구성되며, 이 구조체가 그 분해 결과를 저장한다.
   * 설정자: 생성자에서 config->m_address_mapping.addrdec_tlx(addr, &m_raw_addr)로 초기화.
   *         set_chip()으로 chip만 재설정 가능.
   *         set_partition()으로 sub_partition만 재설정 가능.
   *         섹터 분할 시 original_mf의 chip/sub_partition을 복사.
   * 읽는 자: get_tlx_addr() — DRAM 컨트롤러, L2 파티션 라우팅 코드.
   *           get_sub_partition_id() — 서브 파티션 ID 접근.
   * 값 범위: chip(0~N_CHANNEL-1), bk(0~N_BANK-1), row/col은 DRAM 용량에 따라 다름.
   * 동기화: 초기화 후 원칙적으로 불변이나, 섹터 분할 시 chip/sub_partition이 조정됨. */

  enum mf_type m_type;
  /* [한국어] 패킷의 현재 방향과 종류.
   * READ_REQUEST / WRITE_REQUEST (SM→메모리) 또는
   * READ_REPLY / WRITE_ACK (메모리→SM).
   * 설정자: 생성자에서 m_access.is_write() ? WRITE_REQUEST : READ_REQUEST.
   *         set_reply()에서 응답 방향으로 변환.
   * 읽는 자: get_type(), set_reply()의 분기 조건, ICNT/캐시/DRAM 처리 코드.
   * 값 범위: mf_type enum (READ_REQUEST=0, WRITE_REQUEST=1, READ_REPLY=2, WRITE_ACK=3).
   * 동기화: 단일 스레드. set_reply() 호출은 한 번만 발생. */

  // statistics
  unsigned
      m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
  /* [한국어] 패킷 생성 시각 (시뮬레이션 사이클 단위).
   * gpu_sim_cycle + gpu_tot_sim_cycle 값으로, 여러 커널에 걸친 누적 사이클 포함.
   * 설정자: 생성자의 @cycle 파라미터로 초기화.
   * 읽는 자: get_timestamp(). mem_latency_stat.cc가 응답 수신 시점에서
   *           이 값을 빼어 전체 메모리 레이턴시를 계산.
   * 값 범위: 0 이상의 누적 사이클 번호.
   * 동기화: 생성 후 불변. */

  unsigned m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
                          // onto icnt to shader; only used for reads
  /* [한국어] 패킷이 메모리→SM 방향 ICNT에 삽입된 시각 (읽기 응답 전용).
   * 메모리 파티션에서 응답을 SM으로 전송할 때 기록한다.
   * 쓰기 ACK에서는 의미 없는 값(0으로 초기화)이다.
   * 설정자: ICNT 삽입 코드에서 set_return_timestamp(cycle)로 기록.
   * 읽는 자: get_return_timestamp(). ICNT→SM 구간 레이턴시 측정에 사용.
   * 값 범위: 0 (초기/미설정) 또는 응답 ICNT 삽입 사이클 번호.
   * 동기화: 생성 후 한 번 설정. 단일 스레드. */

  unsigned m_icnt_receive_time;  // set to gpu_sim_cycle + interconnect_latency
                                 // when fixed icnt latency mode is enabled
  /* [한국어] 고정 레이턴시 ICNT 모드에서의 예상 수신 사이클.
   * gpgpusim.config에서 -gpgpu_fixed_latency_enabled 1 등으로 고정 레이턴시
   * 모드를 활성화하면, ICNT를 intersim2로 시뮬레이션하는 대신 단순 지연 큐로 대체.
   * 이 경우 삽입 사이클 + interconnect_latency 값을 이 필드에 저장하고,
   * 현재 사이클이 이 값 이상이 되면 도착한 것으로 처리.
   * 설정자: ICNT 삽입 코드에서 set_icnt_receive_time()으로 기록.
   * 읽는 자: get_icnt_receive_time(). 도착 여부 확인 코드.
   * 값 범위: 0 (기본값, 미설정) 또는 예상 수신 사이클.
   * 동기화: 한 번 설정. 단일 스레드. */

  // requesting instruction (put last so mem_fetch prints nicer in gdb)
  warp_inst_t m_inst;
  /* [한국어] 이 메모리 요청을 발생시킨 워프 명령어(warp_inst_t)의 복사본.
   * GDB에서 mem_fetch를 보기 좋게 출력하기 위해 마지막 필드에 배치한다.
   * (warp_inst_t는 큰 구조체이므로 다른 필드들을 먼저 표시하는 것이 디버깅에 유리.)
   *
   * 설정자: 생성자에서 inst != NULL이면 *inst를 복사하여 초기화.
   *         NULL이면 empty 상태(m_inst.empty() == true).
   * 읽는 자: get_inst(), get_pc(), isatomic(), istexture(), isconst(), do_atomic().
   *           print()에서 명령어 정보 출력.
   * 값 범위: empty 상태(writeback/fill 등 명령어 없는 요청) 또는
   *           유효한 warp_inst_t (shader core가 발행한 load/store/atomic 명령어).
   * 동기화: 생성 후 불변. 명령어 실행 결과는 do_atomic()이 in-place로 갱신. */

  unsigned long long m_streamID;
  /* [한국어] 이 요청이 속한 CUDA 스트림(stream)의 ID.
   * CUDA 런타임의 cudaStream_t 핸들 값이다.
   * 설정자: 생성자의 @streamID 파라미터로 초기화.
   * 읽는 자: get_streamID(). 스트림별 통계, 멀티-스트림 동기화에 사용.
   * 값 범위: 0(기본 스트림) 이상의 64비트 정수.
   * 동기화: 생성 후 불변. */

  static unsigned sm_next_mf_request_uid;
  /* [한국어] 다음에 생성될 mem_fetch에 할당할 UID의 전역 카운터.
   * static 멤버이므로 모든 mem_fetch 인스턴스가 공유하는 클래스-레벨 변수이다.
   * 설정자: mem_fetch.cc에서 1로 초기화. 생성자에서 m_request_uid = sm_next_mf_request_uid++.
   * 읽는 자: 생성자 내부에서만 직접 접근.
   * 값 범위: 1부터 시작하여 무한 증가 (unsigned 오버플로우 시 0으로 순환,
   *           실용적으로는 문제없는 수준).
   * 동기화: 단일 스레드 시뮬레이터이므로 별도 락 불필요. */

  const memory_config *m_mem_config;
  /* [한국어] 이 패킷이 참조하는 메모리 시스템 설정 포인터.
   * gpgpusim.config에서 파싱된 메모리 관련 파라미터 전체를 담는다:
   * DRAM 파티션 수, 캐시 크기, ICNT 플릿 크기, SST 모드 여부, 주소 매핑 방식 등.
   * 설정자: 생성자의 @config 파라미터로 초기화 (포인터 복사).
   * 읽는 자: get_mem_config(). get_num_flits()에서 icnt_flit_size를 참조.
   *           생성자 내부에서 is_SST_mode(), addrdec_tlx() 등에 사용.
   * 값 범위: 유효한 memory_config* (NULL 불가, 생성자 assert로 보장).
   * 동기화: 생성 후 불변 (포인터 값과 가리키는 설정 모두 불변). */

  unsigned icnt_flit_size;
  /* [한국어] ICNT(Interconnect Network) 전송 단위인 플릿(flit)의 크기(바이트).
   * 생성자에서 config->icnt_flit_size 값을 복사하여 저장한다.
   * 설정자: 생성자에서 config->icnt_flit_size로 초기화.
   * 읽는 자: get_num_flits()에서 패킷 크기를 나누어 플릿 수를 계산.
   * 값 범위: gpgpusim.config의 -gpgpu_icnt_flit_size 파라미터에 의존 (보통 32~128 바이트).
   * 동기화: 생성 후 불변. */

  mem_fetch
      *original_mf;  // this pointer is set up when a request is divided into
                     // sector requests at L2 cache (if the req size > L2 sector
                     // size), so the pointer refers to the original request
  /* [한국어] L2 캐시에서 섹터 단위로 요청을 분할할 때, 원본 요청 패킷을 가리키는 포인터.
   * L2 캐시 요청 크기가 L2 섹터 크기보다 크면 여러 개의 파생 mem_fetch가 생성되며,
   * 각 파생 요청이 original_mf로 원본 패킷을 참조한다.
   * 모든 파생 요청의 응답이 완료되면 원본 패킷이 완료 처리된다.
   *
   * 설정자: 생성자의 @original_mf 파라미터로 초기화. 분할이 없으면 NULL.
   *         original_mf != NULL이면 생성자가 chip/sub_partition을 original_mf에서 복사.
   * 읽는 자: get_original_mf(). L2 파티션 처리 코드가 파생 요청 완료 후
   *           원본 요청 완료 처리를 위해 참조.
   * 값 범위: NULL (분할 없음) 또는 유효한 mem_fetch* (원본 요청).
   * 동기화: 생성 후 불변. */

  mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
                              // when fetch-on-write policy is used
  /* [한국어] fetch-on-write 정책에서 원본 쓰기 요청을 가리키는 포인터.
   * fetch-on-write: 쓰기 전에 해당 캐시라인을 먼저 메모리에서 가져오는 정책.
   * 이때 쓰기 요청을 처리하려고 READ_REQUEST가 생성되며,
   * 그 READ_REQUEST의 original_wr_mf 필드가 원본 WRITE_REQUEST를 가리킨다.
   * 읽기 완료(L2/DRAM에서 데이터 반환) 후 original_wr_mf를 이용해
   * 원본 쓰기를 재시작한다.
   *
   * 설정자: 생성자의 @original_wr_mf 파라미터로 초기화. fetch-on-write가 아니면 NULL.
   * 읽는 자: get_original_wr_mf(). L2 파티션 / 캐시 miss 처리 코드.
   * 값 범위: NULL (fetch-on-write 미적용) 또는 유효한 쓰기 mem_fetch*.
   * 동기화: 생성 후 불변. */
};

#endif
/* [한국어] MEM_FETCH_H 인클루드 가드 종료.
 * #ifndef MEM_FETCH_H 블록의 끝을 표시한다. */
