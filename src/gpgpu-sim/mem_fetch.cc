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
 * [한국어 설명] GPU 내부 메모리 요청 패킷 구현 (mem_fetch.cc)
 *
 * === 파일의 역할 ===
 * mem_fetch.h에서 선언된 mem_fetch 클래스의 멤버 함수를 구현한다.
 * mem_fetch는 GPGPU-Sim에서 GPU 내부 메모리 서브시스템을 흐르는 요청/응답
 * 패킷이며, 이 파일은 그 생성(생성자), 소멸(소멸자), 상태 관리(set_status),
 * 원자 연산 실행(do_atomic), 텍스처/상수 접근 판별(istexture/isconst),
 * ICNT 플릿 수 계산(get_num_flits), 디버그 출력(print) 등을 구현한다.
 * 파일 크기는 작지만, 시뮬레이터 전체에서 가장 빈번하게 호출되는 경로에 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일의 코드는 시뮬레이터 메인 루프(gpu-sim.cc의 gpgpu_sim::cycle())가
 * 매 사이클마다 수행하는 메모리 파이프라인 처리 중에 호출된다.
 *
 * 생성 경로 (SM → 메모리):
 *   shader.cc (ldst_unit, L1D 미스 처리)
 *     → new mem_fetch() [이 파일의 생성자]
 *     → ICNT push (intersim2, IN_ICNT_TO_MEM)
 *     → gpu-sim.cc의 memory_partition_unit
 *     → gpu-cache.cc (L2 access)
 *     → dram.cc (DRAM push)
 *
 * 응답 경로 (메모리 → SM):
 *   dram.cc (DRAM return) / gpu-cache.cc (L2 hit)
 *     → set_reply() [mem_fetch.h 인라인, 타입 변환]
 *     → ICNT push (SM 방향)
 *     → shader.cc (L1D fill, warp 재개)
 *     → delete mem_fetch() [이 파일의 소멸자]
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드 시뮬레이터 루프.
 * GPU 디바이스 코드가 아니라 시뮬레이터 호스트 코드임에 유의한다.
 *
 * === 타 모듈과의 연결 ===
 * include하는 헤더:
 *   - mem_fetch.h: 이 파일이 구현하는 클래스 선언.
 *   - gpu-sim.h: gpgpu_sim 클래스, 메모리 설정(memory_config) 타입.
 *   - mem_latency_stat.h: 메모리 레이턴시 통계 수집 인터페이스.
 *   - shader.h: warp_inst_t의 do_atomic(), isatomic() 등 명령어 메서드.
 *   - visualizer.h: 시각화 도구(AerialVision)에 데이터를 전달하는 인터페이스.
 *
 * 이 파일이 사용하는 외부 상태:
 *   - sm_next_mf_request_uid (전역 카운터): 패킷 UID 할당.
 *   - Status_str[] (파일 내 정적 배열): print()에서 상태 이름 출력.
 *   - memory_config: 주소 매핑(addrdec_tlx, partition_address), SST 모드 여부,
 *     ICNT 플릿 크기 등 모든 메모리 설정에 접근.
 *
 * === 주요 함수/구조체 요약 ===
 * - mem_fetch() 생성자: 메모리 접근 정보(mem_access_t)를 받아 패킷을 완전히
 *   초기화한다. 가상 주소를 DRAM 물리 주소(chip/bank/row/col)로 변환하는
 *   addrdec_tlx()를 이 시점에 호출한다.
 * - ~mem_fetch() 소멸자: m_status를 MEM_FETCH_DELETED로 표시. 댕글링 포인터 감지용.
 * - set_status(): 패킷이 새로운 큐에 진입할 때 m_status와 m_status_change를 갱신.
 * - print(): UID, SM, 워프, 파티션, 접근 정보, 현재 상태, 명령어를 파일에 출력.
 * - get_num_flits(): 방향과 타입에 따라 ICNT 플릿 수를 계산한다.
 *   원자/쓰기/읽기응답은 ctrl+data, 읽기요청/쓰기ACK는 ctrl만 계산.
 * - isatomic(), istexture(), isconst(): m_inst 필드를 검사하여 접근 종류 판별.
 * - do_atomic(): m_inst.do_atomic(warp_mask)를 호출하여 원자 연산 실제 실행.
 */

#include "mem_fetch.h"
/* [한국어] mem_fetch 클래스 선언 및 mf_type, mem_fetch_status enum을 포함하는 헤더.
 * 이 .cc 파일이 구현하는 모든 멤버 함수의 선언이 여기에 있다. */
#include "gpu-sim.h"
/* [한국어] gpgpu_sim 클래스 및 memory_config 클래스 전체 정의를 포함.
 * 생성자에서 config->is_SST_mode(), config->m_address_mapping.addrdec_tlx(),
 * config->icnt_flit_size 등에 접근하기 위해 필요하다. */
#include "mem_latency_stat.h"
/* [한국어] 메모리 레이턴시 통계 수집 인터페이스.
 * 이 파일에서 직접 통계를 기록하지는 않지만, 다른 모듈과의 통합을 위해 포함. */
#include "shader.h"
/* [한국어] warp_inst_t의 isatomic(), do_atomic(), isspace() 등
 * 명령어 수준 메서드가 완전히 정의된 헤더.
 * isatomic(), istexture(), isconst(), do_atomic()이 m_inst의 이 메서드들을 호출. */
#include "visualizer.h"
/* [한국어] AerialVision(GPGPU-Sim 성능 시각화 도구) 인터페이스 헤더.
 * 시각화 데이터 수집 코드가 mem_fetch의 타임스탬프/상태를 읽는 데 사용. */

unsigned mem_fetch::sm_next_mf_request_uid = 1;
/* [한국어] 전역 mem_fetch 요청 UID 카운터의 초기화.
 * 클래스 static 멤버 변수는 이 위치(번역 단위 전역 초기화)에서 한 번 초기화된다.
 * 1로 시작하여 각 mem_fetch 생성 시 후위 증가(sm_next_mf_request_uid++)된다.
 * 0은 "유효하지 않은 UID"로 예약되어 있어 1부터 시작한다. */

/*
 * [한국어]
 * mem_fetch::mem_fetch — 메모리 요청 패킷 생성자.
 *
 * @access:          메모리 접근 정보 (가상 주소, 타입, 크기, warp/byte/sector 마스크).
 *                   gpu-cache.cc의 L1D 미스 처리 코드가 생성하여 전달한다.
 * @inst:            요청을 발생시킨 warp 명령어 포인터. NULL이면 명령어 없는 요청
 *                   (예: L2 write-back, fill 패킷 등).
 * @streamID:        CUDA 스트림 ID. 스트림별 통계 및 동기화에 사용.
 * @ctrl_size:       하드웨어에서 이 패킷의 컨트롤 메타데이터 크기 (바이트).
 * @wid:             요청 워프의 SM 내 인덱스.
 * @sid:             요청 SM(shader core)의 인덱스.
 * @tpc:             요청 TPC(Texture Processing Cluster) 인덱스.
 * @config:          메모리 시스템 설정 포인터 (DRAM 구조, 캐시 크기, ICNT 플릿 크기 등).
 * @cycle:           생성 시점의 시뮬레이션 사이클 번호 (레이턴시 측정 기준).
 * @m_original_mf:   L2 섹터 분할 시 원본 패킷 포인터 (NULL 가능).
 * @m_original_wr_mf: fetch-on-write 시 원본 쓰기 패킷 포인터 (NULL 가능).
 * @return:          (없음, 생성자)
 *
 * 동작 단계:
 *   1) 전역 카운터에서 고유 UID를 할당한다.
 *   2) mem_access_t를 복사하고, inst가 있으면 warp_inst_t도 복사한다.
 *      wid와 m_inst.warp_id()의 일치를 assert로 검증한다.
 *   3) SST 모드가 아닐 때: addrdec_tlx()로 가상 주소를 DRAM 물리 주소
 *      (chip/bank/row/col)로 분해하고, partition_address()로 파티션 내 선형
 *      주소를 계산한다. SST 모드에서는 SST 메모리 계층이 주소 매핑을 담당.
 *   4) 쓰기 여부로 m_type(WRITE_REQUEST / READ_REQUEST)을 결정한다.
 *   5) 타임스탬프, 초기 상태(MEM_FETCH_INITIALIZED), 설정 포인터 등을 초기화.
 *   6) original_mf != NULL이면 chip/sub_partition을 원본에서 복사한다.
 *      섹터 분할 파생 요청이 원본과 같은 메모리 파티션에 매핑되도록 보장.
 *
 * 실행 컨텍스트: 시뮬레이터 메인 루프 (단일 스레드). 매 사이클 L1D 미스가
 *               발생할 때마다 호출될 수 있다.
 * 호출 체인: shader.cc (ldst_unit::process_memory_access_queue) /
 *            gpu-cache.cc (baseline_cache::send_read_request)
 *            → [mem_fetch 생성자] → (패킷이 L1 미스 큐에 삽입됨)
 */
mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned long long streamID, unsigned ctrl_size,
                     unsigned wid, unsigned sid, unsigned tpc,
                     const memory_config *config, unsigned long long cycle,
                     mem_fetch *m_original_mf, mem_fetch *m_original_wr_mf)
    : m_access(access)
/* [한국어] 멤버 초기화 리스트: m_access를 @access로 복사 초기화.
 * mem_access_t는 주소, 타입, 마스크 등을 포함하는 값 타입이므로 복사된다. */

{
  m_request_uid = sm_next_mf_request_uid++;
  /* [한국어] 전역 카운터에서 고유 ID를 할당하고 카운터를 증가시킨다.
   * 후위 증가(++)이므로 현재 값을 m_request_uid에 대입한 뒤 카운터를 1 증가.
   * 단일 스레드 시뮬레이터이므로 별도 원자 연산 없이 안전하다. */

  m_access = access;
  /* [한국어] m_access를 다시 한번 복사한다.
   * 멤버 초기화 리스트에서 이미 복사했지만, 이 대입문이 명시적으로 재복사한다.
   * 중복이지만 기존 코드를 유지한다. */

  if (inst) {
    /* [한국어] 요청을 발생시킨 warp 명령어 포인터가 NULL이 아닌 경우.
     * writeback, fill 등 명령어가 없는 요청은 inst == NULL이므로 이 블록을 건너뜀. */
    m_inst = *inst;
    /* [한국어] warp_inst_t를 값으로 복사한다.
     * m_inst는 warp_inst_t 타입의 멤버이므로 복사 생성자가 호출된다.
     * 이 복사로 패킷은 원본 명령어 객체와 독립적으로 warp 정보를 보유한다. */
    assert(wid == m_inst.warp_id());
    /* [한국어] 전달받은 wid와 명령어에 기록된 warp ID가 일치하는지 검증.
     * 불일치하면 호출자가 잘못된 wid를 전달한 버그. 시뮬레이션을 즉시 중단시킨다. */
  }

  m_streamID = streamID;
  /* [한국어] CUDA 스트림 ID를 저장한다. 스트림별 통계 및 동기화에 사용. */

  m_data_size = access.get_size();
  /* [한국어] 메모리 접근의 데이터 크기를 mem_access_t에서 가져와 저장.
   * 일반적으로 캐시라인 크기(128B)이며, 섹터 분할 후 set_data_size()로 조정 가능. */

  m_ctrl_size = ctrl_size;
  /* [한국어] 하드웨어 컨트롤 패킷 크기를 저장.
   * 호출자(shader.cc, gpu-cache.cc)가 하드웨어 프로토콜에 따라 계산하여 전달한다. */

  m_sid = sid;
  /* [한국어] 요청 SM(shader core) 인덱스를 저장한다.
   * 응답 패킷을 올바른 SM으로 라우팅하는 데 사용된다. */

  m_tpc = tpc;
  /* [한국어] 요청 TPC(Texture Processing Cluster) 인덱스를 저장한다.
   * NoC 라우팅에서 소스 노드를 식별하는 데 사용된다. */

  m_wid = wid;
  /* [한국어] 요청 warp ID를 저장한다.
   * 응답 수신 후 해당 warp를 ready 상태로 전환할 때 필요하다. */

  if (!config->is_SST_mode()) {
    /* [한국어] SST(Structural Simulation Toolkit) 메모리 모델 모드가 아닌 경우.
     * SST 모드에서는 외부 SST 메모리 계층이 주소 매핑을 담당하므로 이 블록을 건너뜀.
     * 일반 GPGPU-Sim 모드에서는 시뮬레이터가 직접 주소를 물리 주소로 변환한다. */

    // In SST memory model, the SST memory hierarchy is
    // responsible to generate the correct address mapping
    config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);
    /* [한국어] 가상(또는 시뮬레이션 선형) 주소를 DRAM 물리 주소 구조체로 변환.
     * addrdec_tlx()는 addrdec.h의 linear_to_raw_address_translation 클래스가 제공.
     * 결과로 m_raw_addr의 chip(채널), bk(뱅크), row(행), col(열), burst, sub_partition이
     * 채워진다. gpgpusim.config의 -gpgpu_mem_address_mapping 옵션으로 매핑 방식 제어.
     * 이 변환은 DRAM 뱅크 충돌 분석, row-buffer hit rate 계산의 기반이 된다. */

    m_partition_addr =
        config->m_address_mapping.partition_address(access.get_addr());
    /* [한국어] DRAM 파티션 내부의 선형 주소를 계산하여 저장한다.
     * 전체 주소에서 뱅크 선택 비트를 제거한 나머지 주소이다.
     * DRAM 스케줄러가 row-buffer hit 여부(같은 파티션 내 연속 접근)를 판단할 때 사용. */
  }

  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  /* [한국어] mem_access_t의 is_write() 결과로 패킷 타입을 결정한다.
   * 쓰기 접근이면 WRITE_REQUEST, 읽기 접근이면 READ_REQUEST.
   * 이후 set_reply()에 의해 각각 WRITE_ACK / READ_REPLY로 변환될 예정. */

  m_timestamp = cycle;
  /* [한국어] 생성 시각(사이클)을 기록한다.
   * gpu_sim_cycle + gpu_tot_sim_cycle 값이며, 전체 메모리 레이턴시 측정의 시작점. */

  m_timestamp2 = 0;
  /* [한국어] SM 방향 ICNT 삽입 시각을 0으로 초기화한다.
   * 읽기 응답 패킷이 ICNT에 삽입될 때 set_return_timestamp()로 갱신된다. */

  /* [한국어] m_icnt_receive_time은 여기서 명시적으로 초기화되지 않는다.
   * 고정 레이턴시 ICNT 모드(-gpgpu_fixed_latency_enabled 1)가 활성화된 경우에만
   * ICNT 삽입 시 set_icnt_receive_time()이 호출되어 값이 설정된다.
   * 일반 intersim2 시뮬레이션 모드에서는 이 필드를 읽는 코드 경로가 없으므로
   * 미초기화 상태가 실제 동작에 영향을 주지 않는다. */

  m_status = MEM_FETCH_INITIALIZED;
  /* [한국어] 초기 상태를 MEM_FETCH_INITIALIZED로 설정한다.
   * 이 상태는 패킷이 생성되었으나 아직 어떤 큐에도 진입하지 않은 상태를 나타낸다. */

  m_status_change = cycle;
  /* [한국어] 상태 변경 시각을 생성 사이클로 초기화한다.
   * 이후 set_status() 호출마다 갱신된다. */

  m_mem_config = config;
  /* [한국어] 메모리 설정 포인터를 저장한다.
   * get_num_flits()에서 icnt_flit_size 접근 등 생애 전반에 걸쳐 참조된다. */

  icnt_flit_size = config->icnt_flit_size;
  /* [한국어] ICNT 플릿 크기를 설정에서 복사하여 저장한다.
   * get_num_flits()가 매 호출마다 config를 역참조하는 대신 이 값을 사용한다.
   * gpgpusim.config의 -gpgpu_icnt_flit_size 파라미터로 조정된다. */

  original_mf = m_original_mf;
  /* [한국어] L2 섹터 분할 시 원본 패킷 포인터를 저장한다.
   * 분할이 없으면 NULL. 이 포인터는 파생 요청의 완료 처리에 사용된다. */

  original_wr_mf = m_original_wr_mf;
  /* [한국어] fetch-on-write 정책의 원본 쓰기 요청 포인터를 저장한다.
   * fetch-on-write가 아니면 NULL. 읽기 완료 후 원본 쓰기를 재시작하는 데 사용. */

  if (m_original_mf) {
    /* [한국어] 이 패킷이 섹터 분할로 생성된 파생 요청인 경우.
     * original_mf != NULL이면 원본 패킷의 DRAM 파티션 정보를 복사한다. */
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    /* [한국어] 원본 패킷의 chip(채널/파티션) 번호를 복사한다.
     * 파생 요청들이 원본과 동일한 메모리 채널로 전송되도록 보장한다.
     * 섹터 분할 시 주소 기반으로 계산한 chip이 일부 경우 달라질 수 있어
     * 원본의 값을 강제로 사용하여 일관성을 유지한다. */
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
    /* [한국어] 원본 패킷의 sub_partition 번호를 복사한다.
     * L2 파티션 라우팅에서 모든 파생 요청이 동일한 L2 서브 파티션으로
     * 전달되도록 보장한다. */
  }
}

/*
 * [한국어]
 * mem_fetch::~mem_fetch — 메모리 요청 패킷 소멸자.
 *
 * @return: (없음, 소멸자)
 *
 * m_status를 MEM_FETCH_DELETED로 설정하여 이미 해제된 패킷을 나타낸다.
 * 실제 동적 할당 멤버가 없으므로 메모리 해제 작업은 없다.
 * MEM_FETCH_DELETED 상태는 댕글링 포인터 버그 디버깅 시 이미 소멸된 패킷을
 * 잘못 참조하는 코드를 발견하는 센티넬 역할을 한다.
 *
 * 호출 체인: shader.cc (응답 처리 완료) / gpu-cache.cc (writeback 완료)
 *            → delete mf → [~mem_fetch()]
 */
mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }
/* [한국어] 소멸자 본문: m_status를 MEM_FETCH_DELETED로 표시.
 * 이후 이 포인터를 역참조하는 코드가 있다면 비정상적인 상태값을 보게 된다. */

/*
 * [한국어]
 * X-매크로 패턴으로 Status_str[] 문자열 배열 생성.
 *
 * mem_fetch_status.tup 파일을 다른 매크로 정의(MF_TUP_BEGIN → 배열 시작,
 * MF_TUP → 문자열화, MF_TUP_END → 배열 종료)로 include하여
 * "static const char *Status_str[] = { "MEM_FETCH_INITIALIZED", ..., };"
 * 를 생성한다.
 *
 * 이 패턴의 핵심 이점:
 *   - mem_fetch_status.tup에 새 상태를 추가하면 enum(mem_fetch.h)과
 *     문자열 배열(이 위치) 양쪽이 자동으로 동기화된다.
 *   - 두 곳을 별도로 관리할 필요가 없어 불일치 버그를 방지한다.
 *
 * Status_str[]은 print() 함수에서 m_status 값을 사람이 읽을 수 있는
 * 문자열("IN_L1D_MISS_QUEUE" 등)로 변환할 때 사용된다.
 */
#define MF_TUP_BEGIN(X) static const char *Status_str[] = {
/* [한국어] 문자열 배열 선언 시작. X(=mem_fetch_status)는 무시되고
 * 배열 이름은 Status_str로 고정된다. static이므로 이 번역 단위에만 유효. */
#define MF_TUP(X) #X
/* [한국어] 각 enum 값 이름을 문자열 리터럴로 변환한다.
 * #X는 C 전처리기의 문자열화(stringification) 연산자:
 * MF_TUP(IN_L1D_MISS_QUEUE) → "IN_L1D_MISS_QUEUE" */
#define MF_TUP_END(X) \
  }                   \
  ;
/* [한국어] 배열 선언 종료. 중괄호와 세미콜론으로 마무리. */
#include "mem_fetch_status.tup"
/* [한국어] 실제 상태 목록이 담긴 .tup 파일을 포함한다.
 * 위에서 정의한 MF_TUP_BEGIN/MF_TUP/MF_TUP_END 매크로가 적용되어
 * Status_str[]에 "MEM_FETCH_INITIALIZED", "IN_L1I_MISS_QUEUE", ...
 * 형태의 문자열 배열이 생성된다. */
#undef MF_TUP_BEGIN
/* [한국어] MF_TUP_BEGIN 매크로 해제. 다른 코드에서의 충돌 방지. */
#undef MF_TUP
/* [한국어] MF_TUP 매크로 해제. */
#undef MF_TUP_END
/* [한국어] MF_TUP_END 매크로 해제. */

/*
 * [한국어]
 * mem_fetch::print — 패킷의 전체 정보를 사람이 읽을 수 있는 형식으로 출력한다.
 *
 * @fp:         출력 대상 파일 포인터 (stdout, stderr, 로그 파일 등).
 * @print_inst: true이면 연관 warp 명령어 정보도 함께 출력 (기본값: true).
 * @return:     (없음)
 *
 * 출력 형식 (한 줄 기본):
 *   "  mf: uid=<UID>, sid<SM_ID>:w<warp_ID>, part=<chip번호>, <mem_access정보>,
 *    status = <상태이름> (<상태변경사이클>), [명령어정보 또는 개행]"
 *
 * 동작 과정:
 *   1) uid, SM ID, warp ID, DRAM chip(파티션) 번호를 출력한다.
 *   2) m_access.print(fp)로 주소, 타입, 크기, 마스크를 출력한다.
 *   3) m_status가 유효 범위(< NUM_MEM_REQ_STAT)이면 Status_str[]로 이름 출력,
 *      그렇지 않으면 정수값에 "???"를 붙여 비정상 상태임을 표시한다.
 *   4) m_inst가 비어 있지 않고 print_inst가 true이면 m_inst.print(fp) 호출.
 *      그렇지 않으면 개행만 출력.
 *
 * 실행 컨텍스트: 디버깅, 시각화, 통계 출력 코드에서 호출.
 * 호출 체인: visualizer.cc / 디버그 출력 코드 → [print(fp, print_inst)]
 *            → m_access.print() + m_inst.print()
 */
void mem_fetch::print(FILE *fp, bool print_inst) const {
  // if (this == NULL) { // doenst make sense!
  //   fprintf(fp, " <NULL mem_fetch pointer>\n");
  //   return;
  // }
  /* [한국어] NULL 포인터 검사 코드가 주석 처리되어 있다.
   * C++에서 NULL 포인터의 멤버 함수 호출은 undefined behavior이므로
   * 의미 없는 검사라는 이유로 비활성화되었다. */

  fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid,
          m_wid, m_raw_addr.chip);
  /* [한국어] 기본 식별 정보를 출력한다:
   *   uid: 6자리 패딩으로 패킷 고유 ID.
   *   sid: 2자리 패딩으로 SM(shader core) 인덱스.
   *   w:   2자리 패딩으로 warp 인덱스.
   *   part: DRAM chip(채널/파티션) 번호. */

  m_access.print(fp);
  /* [한국어] mem_access_t의 print()를 호출하여 메모리 주소, 접근 타입,
   * 크기, 활성 마스크 등의 접근 세부 정보를 출력한다. */

  if ((unsigned)m_status < NUM_MEM_REQ_STAT)
    /* [한국어] m_status 값이 유효한 enum 범위 내인지 확인한다.
     * NUM_MEM_REQ_STAT는 mem_fetch_status enum의 마지막 센티넬 값으로,
     * 이 값보다 작으면 Status_str[]로 이름을 조회할 수 있다. */
    fprintf(fp, " status = %s (%llu), ", Status_str[m_status], m_status_change);
    /* [한국어] 유효한 상태: Status_str[] 배열로 상태 이름 문자열을 출력하고,
     * m_status_change(상태 변경 사이클)를 함께 출력한다. */
  else
    fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
    /* [한국어] 비정상 상태(enum 범위 초과): 정수값 뒤에 "???"를 붙여
     * 알 수 없는 상태임을 표시한다. 메모리 오염(memory corruption) 의심 신호. */

  if (!m_inst.empty() && print_inst)
    /* [한국어] m_inst가 유효한 명령어를 담고 있고, 출력이 요청된 경우. */
    m_inst.print(fp);
    /* [한국어] warp_inst_t::print()를 호출하여 PTX 명령어, 피연산자,
     * 접근 메모리 공간 등 명령어 세부 정보를 출력한다. */
  else
    fprintf(fp, "\n");
    /* [한국어] 명령어가 없거나 출력이 비활성화된 경우: 개행으로 줄을 마무리한다. */
}

/*
 * [한국어]
 * mem_fetch::set_status — 패킷의 현재 위치/상태를 갱신한다.
 *
 * @status: 새로운 mem_fetch_status 값. 패킷이 진입한 큐나 처리 단계.
 * @cycle:  상태 변경 시점의 시뮬레이션 사이클 번호.
 * @return: (없음)
 *
 * m_status와 m_status_change를 원자적으로(단일 스레드이므로 실제 원자 연산
 * 불필요) 갱신한다. 각 큐/모듈이 패킷을 수신할 때 이 함수를 호출하여
 * "이 패킷이 지금 어디에 있는가"를 추적한다.
 *
 * 상태 전이 예시:
 *   MEM_FETCH_INITIALIZED → IN_L1D_MISS_QUEUE (L1D 미스 큐 진입)
 *   IN_L1D_MISS_QUEUE → IN_ICNT_TO_MEM (ICNT 삽입)
 *   IN_ICNT_TO_MEM → IN_PARTITION_ICNT_TO_L2_QUEUE (L2 파티션 도착)
 *   ... → MEM_FETCH_DELETED (소멸)
 *
 * 실행 컨텍스트: 시뮬레이터 메인 루프 단일 스레드.
 * 호출 체인: baseline_cache::access() / dram_t::push() / icnt_push() 등
 *            → [set_status(status, cycle)]
 */
void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  /* [한국어] 새로운 상태를 기록한다. 이 값은 get_status()로 외부에 노출되고,
   * print()에서 Status_str[] 인덱스로 사용된다. */
  m_status_change = cycle;
  /* [한국어] 상태가 변경된 시뮬레이션 사이클을 기록한다.
   * print() 출력과 레이턴시 분석 도구에서 "이 상태에 얼마나 머물렀는지"
   * 계산하는 기준점으로 사용된다. */
}

/*
 * [한국어]
 * mem_fetch::isatomic — 이 요청이 원자적(atomic) 연산인지 반환한다.
 *
 * @return: true면 atomicAdd/atomicCAS 등의 원자 연산 요청. false면 일반 load/store.
 *
 * m_inst가 비어 있으면(writeback, fill 등 명령어 없는 요청) false를 반환한다.
 * 비어 있지 않으면 m_inst.isatomic()에 위임한다 (warp_inst_t가 판단).
 *
 * 원자 연산은 do_atomic()으로 실제 실행되며, get_num_flits()에서 플릿 수
 * 계산 시 "항상 데이터 포함" 경로를 선택하도록 하는 데 사용된다.
 *
 * 호출 체인: memory_sub_partition::service_mem_req() / get_num_flits()
 *            → [isatomic()] → m_inst.isatomic()
 */
bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  /* [한국어] m_inst가 empty이면 명령어가 없는 요청(writeback 등)이므로
   * 원자 연산일 수 없다. 즉시 false를 반환한다. */
  return m_inst.isatomic();
  /* [한국어] warp_inst_t::isatomic()에 위임한다.
   * PTX 명령어가 atom.* 계열이면 true를 반환한다. */
}

/*
 * [한국어]
 * mem_fetch::do_atomic — 이 패킷이 요청하는 원자적 연산을 실제로 실행한다.
 *
 * @return: (없음)
 *
 * GPU의 atomicAdd, atomicCAS, atomicExch 등은 read-modify-write 연산이다.
 * 메모리에서 데이터가 반환된 후(DRAM 또는 L2 캐시에서), 이 함수를 호출하여
 * 실제 연산 시맨틱을 적용한다. 결과는 m_inst 내부의 레지스터 상태에 반영된다.
 *
 * m_access.get_warp_mask()로 이 요청을 발생시킨 활성 스레드들의 마스크를 가져와
 * m_inst.do_atomic()에 전달한다. 비활성 스레드에 대해서는 연산을 수행하지 않는다.
 *
 * 실행 컨텍스트: memory_sub_partition::service_mem_req()에서 응답 반환 시 호출.
 * 호출 체인: memory_sub_partition::service_mem_req()
 *            → [do_atomic()] → m_inst.do_atomic(warp_mask)
 *                            → 각 활성 스레드의 atomic 연산 수행
 */
void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }
/* [한국어] m_inst.do_atomic()을 warp 마스크와 함께 호출한다.
 * warp 마스크(active_mask_t)는 이 요청에 참여한 스레드들을 나타낸다.
 * warp_inst_t::do_atomic()이 각 활성 스레드에 대해 원자 연산 시맨틱을 수행한다. */

/*
 * [한국어]
 * mem_fetch::istexture — 이 요청이 텍스처 메모리 접근인지 반환한다.
 *
 * @return: true면 tex_space(텍스처 메모리) 접근. false면 다른 메모리 공간.
 *
 * m_inst가 비어 있으면 false를 반환한다.
 * 비어 있지 않으면 m_inst.space.get_type()이 tex_space인지 확인한다.
 *
 * 텍스처 접근은 L1 텍스처 캐시(L1T) 경로로 라우팅되며,
 * IN_L1T_MISS_QUEUE 상태를 거친다. istexture()는 shader.cc에서
 * 올바른 캐시 경로를 선택하는 분기 조건으로 사용된다.
 *
 * 호출 체인: shader.cc (메모리 경로 선택) → [istexture()]
 *            → m_inst.space.get_type() == tex_space
 */
bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  /* [한국어] 명령어 없는 요청은 텍스처 접근이 아니다. false 반환. */
  return m_inst.space.get_type() == tex_space;
  /* [한국어] 명령어의 메모리 공간 타입(_memory_space_t)이 tex_space이면 true.
   * tex_space는 abstract_hardware_model.h의 _memory_space_t enum 값으로,
   * CUDA의 텍스처 메모리(__texture__)에 해당한다.
   * 텍스처 캐시는 2D 공간 지역성에 최적화된 별도의 L1 캐시이다. */
}

/*
 * [한국어]
 * mem_fetch::isconst — 이 요청이 상수 또는 커널 파라미터 메모리 접근인지 반환한다.
 *
 * @return: true면 const_space 또는 param_space_kernel 접근. false면 다른 공간.
 *
 * m_inst가 비어 있으면 false를 반환한다.
 * 비어 있지 않으면 m_inst.space.get_type()이 const_space 또는
 * param_space_kernel인지 확인한다.
 *
 * 상수 메모리(__constant__)와 커널 파라미터(커널 런치 시 전달되는 인자)는
 * 모든 스레드가 동일한 값을 읽으므로 브로드캐스트 최적화가 가능하다.
 * 이들은 L1 상수 캐시(L1C)를 통해 처리되며, IN_L1C_MISS_QUEUE 상태를 거친다.
 *
 * 호출 체인: shader.cc (메모리 경로 선택) → [isconst()]
 *            → m_inst.space.get_type() 비교
 */
bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  /* [한국어] 명령어 없는 요청은 상수 접근이 아니다. false 반환. */
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
  /* [한국어] 두 가지 경우 중 하나이면 상수 접근으로 분류:
   *   const_space: CUDA __constant__ 변수. 디바이스 전역에서 읽기 전용.
   *   param_space_kernel: 커널 파라미터 공간. 모든 스레드가 동일한 값을 읽는
   *                       읽기 전용 공간. CUDA에서 커널 인자 전달에 사용. */
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
/*
 * [한국어]
 * mem_fetch::get_num_flits — ICNT 전송 시 필요한 플릿(flit) 수를 계산한다.
 *
 * @simt_to_mem: true면 SM→메모리 방향, false면 메모리→SM 방향.
 * @return: 이 패킷의 ICNT 전송에 필요한 플릿 수 (최소 1).
 *
 * 플릿(flit)은 ICNT(NoC)의 최소 전송 단위이다. intersim2/가 사용하는 개념이며,
 * 패킷 크기를 플릿 크기로 나누어 몇 개의 플릿이 필요한지 계산한다.
 *
 * 전송 크기 결정 규칙 (하드웨어 프로토콜 반영):
 *   케이스 1: 원자 연산 (isatomic() == true)
 *     → 항상 ctrl + data 전체 크기. 읽기와 쓰기가 모두 포함된 연산이므로.
 *   케이스 2: SM→메모리 방향 쓰기 (simt_to_mem && is_write)
 *     → ctrl + data 전체 크기. 실제 데이터를 메모리로 전송하므로.
 *   케이스 3: 메모리→SM 방향 읽기 응답 (!simt_to_mem && !is_write)
 *     → ctrl + data 전체 크기. 읽어온 데이터를 SM으로 반환하므로.
 *   케이스 4: 그 외 (SM→메모리 읽기 요청, 메모리→SM 쓰기 ACK)
 *     → ctrl 크기만. 데이터 없이 제어 정보만 전송.
 *
 * 플릿 수 계산: ceil(전송크기 / icnt_flit_size) = (sz / flit_size) + (나머지 > 0 ? 1 : 0)
 *
 * 호출 체인: intersim2 / icnt_push() 코드 → [get_num_flits(simt_to_mem)]
 *            → (플릿 수 반환, ICNT 자원 예약에 사용)
 */
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  /* [한국어] 전송 크기(바이트)를 저장할 변수를 0으로 초기화한다. */

  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    /* [한국어] 데이터 포함 전송이 필요한 세 가지 경우를 OR 조건으로 검사:
     *   1) isatomic(): 원자 연산 — read-modify-write이므로 항상 데이터 포함.
     *   2) simt_to_mem && get_is_write(): SM→메모리 방향 쓰기 — 데이터를 메모리로 전송.
     *   3) !(simt_to_mem || get_is_write()): 드모르간 법칙 변환 결과:
     *      !simt_to_mem && !get_is_write() 즉 메모리→SM 방향 읽기 응답 — 데이터 반환.
     * 세 경우 모두 실제 데이터 페이로드가 ICNT를 통해 전달되어야 한다. */
    sz = size();
    /* [한국어] 전체 크기 = m_data_size + m_ctrl_size 를 사용.
     * 데이터 페이로드와 컨트롤 메타데이터 합산. */
  else
    sz = get_ctrl_size();
    /* [한국어] 데이터 없이 컨트롤 정보만 전송하는 경우 ctrl 크기만 사용.
     * SM→메모리 방향 읽기 요청 또는 메모리→SM 방향 쓰기 ACK에 해당.
     * 읽기 요청에는 주소·타입 등 메타데이터만 있고 데이터가 없다. */

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
  /* [한국어] 플릿 수 = ceil(sz / icnt_flit_size) 를 정수 산술로 계산한다.
   *   sz / icnt_flit_size: 정수 나눗셈 (소수점 이하 버림).
   *   (sz % icnt_flit_size) ? 1 : 0: 나머지가 있으면 플릿 1개 추가.
   * 예: sz=100B, flit_size=32B → 100/32=3, 100%32=4 → 3+1=4 플릿.
   * 이 값은 intersim2 NoC에서 해당 패킷이 몇 개의 플릿을 차지하는지 나타내며,
   * 채널 대역폭 소비 계산에 직접 사용된다. */
}
