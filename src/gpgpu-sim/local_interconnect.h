// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
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
 * [한국어 설명] 로컬 인터커넥트(간이 크로스바 ICNT) 헤더 (local_interconnect.h)
 *
 * === 파일의 역할 ===
 * BookSim(intersim2) 없이 단순 크로스바 구조로 SM 클러스터와 L2 메모리 파티션을
 * 연결하는 LOCAL_XBAR ICNT 구현을 정의한다. 두 핵심 클래스를 선언한다:
 *   - xbar_router: 단일 크로스바 스위치로 N개 입력에서 M개 출력으로 패킷을 전달.
 *     NAIVE_RR(라운드로빈) 또는 iSLIP 중재 알고리즘으로 경합을 해소한다.
 *   - LocalInterconnect: 2개 서브넷(요청망 REQ_NET + 응답망 REPLY_NET)으로 구성된
 *     최상위 인터커넥트. icnt_wrapper.cc의 함수 포인터 인터페이스와 연결된다.
 * 시뮬레이션 속도가 중요하거나 NoC 정확도가 불필요한 경우 BookSim 대신 선택한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * icnt_wrapper.cc에서 g_network_mode == LOCAL_XBAR일 때 이 구현이 선택된다.
 * SM(shader core) → 요청 패킷 → REQ_NET xbar_router → 메모리 파티션
 * 메모리 파티션 → 응답 패킷 → REPLY_NET xbar_router → SM(shader core)
 * 호출 체인: gpu-sim.cc 사이클 루프 → icnt_transfer() → LocalInterconnect::Advance()
 *            → xbar_router::Advance() → RR_Advance() 또는 iSLIP_Advance()
 * 실행 컨텍스트: 시뮬레이터 메인 스레드, 매 시뮬레이션 사이클마다 동기 실행.
 *
 * === 타 모듈과의 연결 ===
 * - icnt_wrapper.h/cc: LocalInterconnect를 감싸는 함수 포인터 인터페이스
 * - mem_fetch.h: Push() 데이터 포인터가 실제로는 mem_fetch*인 경우가 많음
 * - local_interconnect.cc: 이 헤더의 모든 클래스/함수 구현
 * 데이터 흐름: shader.cc의 icnt_push() → LocalInterconnect::Push() →
 *              xbar_router::in_buffers → Advance()로 out_buffers로 이동 →
 *              gpu-sim.cc의 icnt_pop() → LocalInterconnect::Pop() 반환
 *
 * === 주요 함수/구조체 요약 ===
 * inct_config          - local_xbar 설정 구조체 (버퍼 한도, 서브넷, 중재 알고리즘 등)
 * xbar_router::Push()  - 입력 버퍼에 패킷 추가
 * xbar_router::Advance()  - 한 사이클 중재하여 입력→출력 버퍼로 패킷 이동
 * LocalInterconnect::Push/Pop() - 서브넷 선택 후 xbar_router 호출
 * LocalInterconnect::HasBuffer() - back-pressure 확인
 */

#ifndef _LOCAL_INTERCONNECT_HPP_
#define _LOCAL_INTERCONNECT_HPP_

#include <iostream>  /* [한국어] 표준 입출력 스트림 — DisplayStats 등 출력에 간접 사용 */
#include <map>       /* [한국어] std::map — 현재 직접 사용하지 않지만 헤더 의존으로 포함 */
#include <queue>     /* [한국어] std::queue — xbar_router의 in_buffers, out_buffers (패킷 FIFO 큐) */
#include <vector>    /* [한국어] std::vector — in_buffers, out_buffers, next_node, net 등 크기 가변 배열 */
using namespace std; /* [한국어] std:: 접두사 생략을 위한 using 지시자 — queue, vector 등 직접 사용 */

/* [한국어] 인터커넥트 서브넷 타입 열거형.
 * REQ_NET(0): SM → 메모리 파티션 방향 요청(request) 패킷을 전달하는 서브넷.
 * REPLY_NET(1): 메모리 파티션 → SM 방향 응답(reply) 패킷을 전달하는 서브넷.
 * xbar_router 생성 시 m_type으로 전달되어 active_in_buffers/active_out_buffers 방향 결정. */
enum Interconnect_type { REQ_NET = 0, REPLY_NET = 1 };

/* [한국어] 크로스바 중재 알고리즘 타입 열거형.
 * NAIVE_RR(0): 단순 라운드로빈(Round-Robin) — 구현 단순, 모든 입력에 공평한 기회 부여.
 * iSLIP(1): iterative SLIP(Scheduling for Large Input-queued Packet switches) 알고리즘.
 *           McKeown(1999) 논문 기반. accept/grant 단계로 출력 포트 경합을 더 효율적으로 해소.
 * inct_config::arbiter_algo 필드와 -icnt_arbiter_algo 설정으로 선택. */
enum Arbiteration_type { NAIVE_RR = 0, iSLIP = 1 };

/*
 * [한국어]
 * inct_config - local_xbar ICNT 설정 구조체
 *
 * icnt_wrapper.cc의 g_inct_config 전역 변수로 관리되며,
 * option_parser를 통해 gpgpusim.config의 -icnt_* 옵션들로 채워진다.
 * LocalInterconnect 및 xbar_router 생성자에 전달되어 내부 자료구조 크기를 결정한다.
 */
struct inct_config {
  // config for local interconnect
  unsigned in_buffer_limit;
  /* [한국어] xbar_router의 각 입력 노드 버퍼 최대 패킷 수.
   * 설정자: -icnt_in_buffer_limit 옵션 (기본값 64).
   * 읽는 자: xbar_router 생성자에서 in_buffer_limit 필드에 복사.
   * 값 범위: 1 이상의 정수. 클수록 버퍼 포화 감소, 메모리 사용 증가.
   * 동기화: 초기화 이후 읽기 전용. */

  unsigned out_buffer_limit;
  /* [한국어] xbar_router의 각 출력 노드 버퍼 최대 패킷 수.
   * 설정자: -icnt_out_buffer_limit 옵션 (기본값 64).
   * 읽는 자: xbar_router 생성자에서 out_buffer_limit 필드에 복사.
   * 값 범위: 1 이상의 정수.
   * 동기화: 초기화 이후 읽기 전용. */

  unsigned subnets;
  /* [한국어] 생성할 서브네트워크(xbar_router 객체) 수.
   * 설정자: -icnt_subnets 옵션 (기본값 2 = REQ_NET + REPLY_NET).
   * 읽는 자: LocalInterconnect 생성자에서 n_subnets 설정.
   * 값 범위: 1 또는 2. 1이면 단방향 단일 서브넷, 2이면 요청/응답 분리.
   * 동기화: 초기화 이후 읽기 전용. */

  Arbiteration_type arbiter_algo;
  /* [한국어] xbar_router의 중재 알고리즘 선택.
   * 설정자: -icnt_arbiter_algo 옵션 (기본값 1=iSLIP).
   * 읽는 자: xbar_router 생성자에서 arbit_type 필드에 복사.
   * 값 범위: 0(NAIVE_RR) 또는 1(iSLIP).
   * 동기화: 초기화 이후 읽기 전용. */

  unsigned verbose;
  /* [한국어] xbar_router 디버그 출력 활성화 수준.
   * 설정자: -icnt_verbose 옵션 (기본값 0 = 비활성).
   * 읽는 자: xbar_router 생성자에서 verbose 필드에 복사.
   *          Advance() 내부에서 verbose != 0이면 패킷 이동 로그 출력.
   * 값 범위: 0(비활성) 또는 1(활성).
   * 동기화: 초기화 이후 읽기 전용. */

  unsigned grant_cycles;
  /* [한국어] iSLIP 알고리즘에서 하나의 grant를 몇 사이클 유지할지.
   * 설정자: -icnt_grant_cycles 옵션 (기본값 1).
   * 읽는 자: xbar_router 생성자에서 grant_cycles 및 grant_cycles_count 설정.
   * 값 범위: 1 이상. 1이면 매 사이클 새 grant, 크면 같은 출력→입력 연결을 여러 사이클 유지.
   * 동기화: 초기화 이후 읽기 전용. */
};

/*
 * [한국어]
 * xbar_router - 단일 크로스바 스위치 라우터
 *
 * N개 입력 노드에서 M개 출력 노드로 패킷을 전달하는 단순화된 크로스바 모델.
 * 각 입력 노드마다 in_buffers[node], 각 출력 노드마다 out_buffers[node]를 가진다.
 * 매 사이클 Advance()가 호출되면 RR_Advance() 또는 iSLIP_Advance()로 중재하여
 * in_buffers에서 out_buffers로 패킷을 이동한다. 실제 NoC 라우팅 레이턴시 없이
 * 버퍼-to-버퍼 전달로 단순화된 모델이다.
 */
class xbar_router {
 public:
  /*
   * [한국어]
   * xbar_router::xbar_router - 크로스바 라우터 생성자
   *
   * @router_id: 이 라우터의 고유 ID (LocalInterconnect의 net[] 인덱스와 일치)
   * @m_type: REQ_NET 또는 REPLY_NET — 트래픽 방향(입력/출력 활성 노드 방향) 결정
   * @n_shader: SM 클러스터 수 (소스 방향 노드 수)
   * @n_mem: 메모리 파티션 수 (목적지 방향 노드 수)
   * @m_localinct_config: 버퍼 크기, 중재 알고리즘 등 설정 구조체 참조
   * @return: 없음 (생성자)
   *
   * in_buffers와 out_buffers를 (n_shader + n_mem) 크기로 resize하고,
   * m_type에 따라 active_in_buffers와 active_out_buffers 방향을 설정한다.
   * REQ_NET이면 SM들이 입력, 메모리 파티션들이 출력.
   * REPLY_NET이면 메모리 파티션들이 입력, SM들이 출력.
   * 모든 통계 카운터를 0으로 초기화한다.
   *
   * 호출 체인: LocalInterconnect::CreateInterconnect() → [이 함수]
   */
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config);

  /* [한국어] 소멸자 — 현재 빈 구현. in_buffers/out_buffers는 vector<queue> 자동 해제. */
  ~xbar_router();

  /*
   * [한국어]
   * xbar_router::Push - 입력 노드의 버퍼에 패킷 추가
   *
   * @input_deviceID: 삽입 소스 노드 ID (0 ~ total_nodes-1)
   * @output_deviceID: 패킷의 목적지 노드 ID
   * @data: 전달할 데이터 포인터 (통상 mem_fetch*)
   * @size: 패킷 크기 (현재 local_xbar에서는 1 플릿으로 처리, size 미사용)
   * @return: 없음 (void)
   *
   * Packet 구조체로 data와 output_deviceID를 묶어 in_buffers[input_deviceID]에 push.
   * Has_Buffer_In()으로 사전 확인 후 호출해야 한다 (assert로 인덱스 범위만 검사).
   *
   * 호출 체인: LocalInterconnect::Push() → [이 함수]
   */
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);

  /*
   * [한국어]
   * xbar_router::Pop - 출력 노드의 버퍼에서 도착 패킷 꺼내기
   *
   * @ouput_deviceID: 수신 노드 ID (0 ~ total_nodes-1)
   * @return: 도착한 데이터 포인터 (없으면 NULL)
   *
   * out_buffers[ouput_deviceID]에서 front 패킷의 data를 꺼내 반환.
   * 버퍼가 비어 있으면 NULL 반환.
   *
   * 호출 체인: LocalInterconnect::Pop() → [이 함수]
   */
  void* Pop(unsigned ouput_deviceID);

  /*
   * [한국어]
   * xbar_router::Advance - 한 사이클 중재하여 입력 버퍼에서 출력 버퍼로 패킷 이동
   *
   * @return: 없음 (void)
   *
   * arbit_type에 따라 RR_Advance() 또는 iSLIP_Advance()를 호출한다.
   * 각 메서드는 입력 버퍼의 패킷을 출력 포트 경합 없이 out_buffers로 이동시킨다.
   * 사이클 카운터(cycles)를 증가시키고 버퍼 utilization 통계를 갱신한다.
   *
   * 호출 체인: LocalInterconnect::Advance() → [이 함수] → RR_Advance() 또는 iSLIP_Advance()
   */
  void Advance();

  /*
   * [한국어]
   * xbar_router::Busy - 버퍼에 패킷이 남아 있는지 확인
   *
   * @return: in_buffers 또는 out_buffers 중 하나라도 비어 있지 않으면 true
   *
   * 시뮬레이션 종료 시 ICNT가 드레인되었는지 확인하는 데 사용된다.
   *
   * 호출 체인: LocalInterconnect::Busy() → [이 함수]
   */
  bool Busy() const;

  /*
   * [한국어]
   * xbar_router::Has_Buffer_In - 입력 노드 버퍼에 여유 공간이 있는지 확인
   *
   * @input_deviceID: 확인할 소스 노드 ID
   * @size: 삽입하려는 패킷 수 (통상 1)
   * @update_counter: true이면 버퍼 가득 참 카운터(in_buffer_full) 증가 (통계용)
   * @return: (현재 크기 + size <= in_buffer_limit)이면 true, 아니면 false
   *
   * LocalInterconnect::HasBuffer()에서 back-pressure 판단에 사용된다.
   *
   * 호출 체인: LocalInterconnect::HasBuffer() → [이 함수]
   */
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);

  /*
   * [한국어]
   * xbar_router::Has_Buffer_Out - 출력 노드 버퍼에 여유 공간이 있는지 확인
   *
   * @output_deviceID: 확인할 목적지 노드 ID
   * @size: 이동하려는 패킷 수 (통상 1)
   * @return: (현재 크기 + size <= out_buffer_limit)이면 true, 아니면 false
   *
   * RR_Advance() 및 iSLIP_Advance() 내부에서 출력 버퍼 포화 여부를 확인하는 데 사용.
   *
   * 호출 체인: xbar_router::RR_Advance() 또는 iSLIP_Advance() → [이 함수]
   */
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // some stats
  /* [한국어] xbar_router 성능 통계 필드들 — DisplayStats()에서 출력되며 시뮬레이션 분석에 사용 */
  unsigned long long cycles;
  /* [한국어] 이 라우터가 처리한 총 사이클 수.
   * 설정자: Advance() 계열 함수의 마지막에서 증가.
   * 읽는 자: DisplayStats()에서 normalizer로 사용.
   * 동기화: 단일 스레드. */

  unsigned long long conflicts;
  /* [한국어] 총 출력 포트 경합 횟수 (여러 입력이 동일 출력을 동시에 요청한 횟수).
   * 설정자: RR_Advance(), iSLIP_Advance()에서 갱신.
   * 읽는 자: DisplayStats()에서 conflicts/cycles로 나누어 per-cycle 경합률 계산.
   * 동기화: 단일 스레드. */

  unsigned long long conflicts_util;
  /* [한국어] 활성 사이클(트래픽이 있는 사이클)에서의 경합 횟수.
   * active가 true인 사이클에서만 누적 — 유휴 사이클을 제외한 실제 부하 상태의 경합률 측정용.
   * 동기화: 단일 스레드. */

  unsigned long long cycles_util;
  /* [한국어] 활성 사이클 수 (in_buffers에 패킷이 있어 Advance()에서 실제 처리가 일어난 사이클).
   * 설정자: Advance() 계열 함수에서 active == true인 경우 증가.
   * 읽는 자: DisplayStats()에서 utilization 관련 비율 계산의 분모.
   * 동기화: 단일 스레드. */

  unsigned long long reqs_util;
  /* [한국어] 활성 사이클 동안 성공적으로 전달된 총 패킷 수.
   * 설정자: Advance() 계열 함수에서 패킷이 out_buffers로 이동할 때마다 증가.
   * 읽는 자: DisplayStats()에서 bank-level parallelism(BLP) = reqs_util/cycles_util 계산.
   * 동기화: 단일 스레드. */

  unsigned long long out_buffer_full;
  /* [한국어] 출력 버퍼가 가득 차서 패킷 이동이 실패한 횟수.
   * 설정자: RR_Advance()에서 Has_Buffer_Out() false 시 증가.
   * 읽는 자: DisplayStats()에서 out_buffer_full/cycles로 per-cycle 포화율 계산.
   * 동기화: 단일 스레드. */

  unsigned long long out_buffer_util;
  /* [한국어] 모든 사이클에 걸친 출력 버퍼의 누적 점유량 (버퍼 활용도 추적용).
   * 설정자: 매 Advance() 호출마다 모든 out_buffers[i].size()를 합산하여 누적.
   * 읽는 자: DisplayStats()에서 out_buffer_util/cycles/active_out_buffers로 평균 활용도 계산.
   * 동기화: 단일 스레드. */

  unsigned long long in_buffer_full;
  /* [한국어] 입력 버퍼가 가득 차서 삽입이 거부된 횟수.
   * 설정자: Has_Buffer_In(update_counter=true)에서 버퍼 가득 참 시 증가.
   * 읽는 자: DisplayStats()에서 in_buffer_full/cycles로 per-cycle 포화율 계산.
   * 동기화: 단일 스레드. */

  unsigned long long in_buffer_util;
  /* [한국어] 모든 사이클에 걸친 입력 버퍼의 누적 점유량.
   * 설정자: 매 Advance() 호출마다 모든 in_buffers[i].size()를 합산하여 누적.
   * 읽는 자: DisplayStats()에서 평균 입력 버퍼 활용도 계산.
   * 동기화: 단일 스레드. */

  unsigned long long packets_num;
  /* [한국어] 이 라우터에 Push()로 삽입된 총 패킷 수.
   * 설정자: Push() 호출마다 증가.
   * 읽는 자: DisplayStats()에서 throughput 계산의 분자.
   * 동기화: 단일 스레드. */

 private:
  /*
   * [한국어]
   * xbar_router::iSLIP_Advance - iSLIP 알고리즘 기반 크로스바 중재 및 패킷 이동
   *
   * @return: 없음 (void)
   *
   * iSLIP(iterative Scheduling for Large Input-queued Packet switches) 알고리즘을 구현.
   * 참고: McKeown (1999) "The iSLIP scheduling algorithm for input-queued switches."
   * 각 출력 포트 i에 대해 next_node[i]에서 시작하는 라운드로빈 순으로 입력 버퍼를 탐색하여
   * 자신을 목적지로 하는 패킷을 찾아 out_buffers[i]로 이동한다.
   * grant가 발생하면 next_node[i]를 전진시켜 다음 사이클에 다른 입력을 먼저 서비스한다.
   * grant_cycles_count로 같은 grant를 여러 사이클 유지할 수 있다.
   *
   * 호출 체인: xbar_router::Advance() → [이 함수]
   */
  void iSLIP_Advance();

  /*
   * [한국어]
   * xbar_router::RR_Advance - 단순 라운드로빈 기반 크로스바 중재 및 패킷 이동
   *
   * @return: 없음 (void)
   *
   * next_node_id를 시작점으로 모든 입력 노드를 순서대로 순회한다.
   * 각 입력 버퍼에 패킷이 있고 목적지 출력 버퍼에 공간이 있으며
   * 해당 사이클에 아직 그 출력 포트로 전달이 없었으면(issued[] == false)
   * 패킷을 in_buffers에서 out_buffers로 이동한다.
   * 처리 후 next_node_id를 1 전진하여 다음 사이클 공평성을 유지한다.
   *
   * 호출 체인: xbar_router::Advance() → [이 함수]
   */
  void RR_Advance();

  /*
   * [한국어]
   * xbar_router::Packet - 라우터 내부에서 데이터와 목적지를 묶는 내부 패킷 구조체
   *
   * Push()에서 생성되어 in_buffers에 저장되고, Advance()에서 out_buffers로 이동된다.
   */
  struct Packet {
    /* [한국어] Packet 생성자 — data와 output_deviceID를 받아 구조체 필드에 저장 */
    Packet(void* m_data, unsigned m_output_deviceID) {
      data = m_data;
      output_deviceID = m_output_deviceID;
    }
    void* data;
    /* [한국어] 전달할 실제 데이터 포인터 (통상 mem_fetch* 또는 기타 시뮬레이터 객체).
     * 설정자: Push() 에서 인자로 전달된 data 저장.
     * 읽는 자: Pop()에서 반환, 또는 Advance()에서 out_buffers로 이동 시 보존.
     * 값 범위: 유효한 포인터 (NULL일 경우 상위 레벨에서 패킷 없음으로 해석).
     * 동기화: 단일 스레드. */

    unsigned output_deviceID;
    /* [한국어] 이 패킷의 목적지 노드 ID.
     * 설정자: Push() 에서 output_deviceID 인자로 설정.
     * 읽는 자: RR_Advance() 및 iSLIP_Advance()에서 어느 out_buffers에 넣을지 결정.
     * 값 범위: 0 ~ total_nodes-1.
     * 동기화: 설정 후 읽기 전용. */
  };

  vector<queue<Packet> > in_buffers;
  /* [한국어] 각 입력 노드의 수신 큐 벡터 (크기: total_nodes = n_shader + n_mem).
   * Push()로 삽입된 패킷이 저장되며, Advance()에서 중재 후 out_buffers로 이동된다.
   * 인덱스 0 ~ n_shader-1: SM 노드 입력 큐 (REQ_NET에서 활성 소스).
   * 인덱스 n_shader ~ total_nodes-1: 메모리 노드 입력 큐 (REPLY_NET에서 활성 소스).
   * 동기화: 단일 스레드. */

  vector<queue<Packet> > out_buffers;
  /* [한국어] 각 출력 노드의 전달 완료 큐 벡터 (크기: total_nodes).
   * Advance()에서 중재 후 패킷이 이동되며, Pop()으로 상위 레벨이 꺼내간다.
   * 인덱스 의미는 in_buffers와 동일하나 방향이 반대 (REQ_NET에서는 메모리 노드 출력).
   * 동기화: 단일 스레드. */

  unsigned _n_shader, _n_mem, total_nodes;
  /* [한국어] SM 클러스터 수, 메모리 파티션 수, 전체 노드 수(n_shader + n_mem).
   * 생성자에서 설정되며 이후 읽기 전용. 루프 상한 및 노드 ID 범위 검사에 사용. */

  unsigned in_buffer_limit, out_buffer_limit;
  /* [한국어] 입력/출력 버퍼의 최대 패킷 수 (inct_config에서 복사).
   * Has_Buffer_In(), Has_Buffer_Out()의 가득 참 판단 기준.
   * 생성자에서 inct_config로부터 초기화, 이후 읽기 전용. */

  vector<unsigned> next_node;  // used for iSLIP arbit
  /* [한국어] iSLIP 중재에서 각 출력 포트 i의 다음 그랜트 시작 입력 노드 인덱스 벡터 (크기: total_nodes).
   * iSLIP_Advance()에서 grant가 발생할 때 next_node[i]를 전진하여 다음 사이클 공평성 유지.
   * 생성자에서 모두 0으로 초기화. */

  unsigned next_node_id;       // used for RR arbit
  /* [한국어] RR 중재에서 다음 사이클 순회 시작 입력 노드 ID.
   * RR_Advance() 마지막에 1씩 전진 (모듈러 연산).
   * 생성자에서 0으로 초기화. */

  unsigned m_id;
  /* [한국어] 이 라우터의 고유 ID (net[] 배열 인덱스, REQ_NET=0, REPLY_NET=1).
   * 디버그 출력에서 라우터를 식별하는 데 사용. 생성자에서 router_id로 설정. */

  enum Interconnect_type router_type;
  /* [한국어] 이 라우터의 타입 (REQ_NET 또는 REPLY_NET).
   * 생성자에서 active_in_buffers, active_out_buffers 방향을 결정하는 데 사용.
   * 생성자 이후 읽기 전용. */

  unsigned active_in_buffers, active_out_buffers;
  /* [한국어] 실제로 활성화된 입력/출력 버퍼 수.
   * REQ_NET: active_in_buffers = n_shader, active_out_buffers = n_mem.
   * REPLY_NET: active_in_buffers = n_mem, active_out_buffers = n_shader.
   * DisplayStats()에서 평균 버퍼 활용도 계산의 분모로 사용. */

  Arbiteration_type arbit_type;
  /* [한국어] 이 라우터가 사용하는 중재 알고리즘 타입 (NAIVE_RR 또는 iSLIP).
   * 생성자에서 inct_config::arbiter_algo로 설정, Advance()에서 분기 조건으로 사용.
   * 생성자 이후 읽기 전용. */

  unsigned verbose;
  /* [한국어] 디버그 출력 활성화 여부 (0=비활성, 1=활성).
   * 생성자에서 inct_config::verbose로 설정.
   * RR_Advance(), iSLIP_Advance()에서 verbose != 0이면 사이클별 패킷 이동 로그 printf.
   * 생성자 이후 읽기 전용. */

  unsigned grant_cycles;
  /* [한국어] iSLIP에서 하나의 grant를 유지할 사이클 수 (inct_config에서 복사).
   * grant_cycles_count와 함께 사용: count가 1이 되면 grant를 갱신하고 count를 grant_cycles로 재설정.
   * 생성자에서 초기화, 이후 읽기 전용. */

  unsigned grant_cycles_count;
  /* [한국어] 현재 grant가 남은 사이클 카운터.
   * iSLIP_Advance()에서 active && count == 1이면 next_node 전진 후 grant_cycles로 재설정.
   * active && count > 1이면 count 감소. 비활성 사이클에는 변경 없음.
   * 생성자에서 grant_cycles로 초기화. */

  friend class LocalInterconnect;
  /* [한국어] LocalInterconnect가 xbar_router의 private 멤버에 직접 접근하도록 허용.
   * DisplayStats()에서 packets_num, cycles 등 통계 필드를 직접 읽기 위해 필요. */
};

/*
 * [한국어]
 * LocalInterconnect - 2-서브넷 로컬 크로스바 인터커넥트 최상위 클래스
 *
 * icnt_wrapper의 함수 포인터 인터페이스와 연결되는 LOCAL_XBAR ICNT 구현체.
 * 내부에 REQ_NET과 REPLY_NET 두 개의 xbar_router를 net[] 벡터로 관리한다.
 * Push/Pop 호출 시 소스/목적지 노드 ID를 기반으로 어느 서브넷을 사용할지 자동 결정한다:
 *   SM 노드(ID < n_shader)가 push하면 REQ_NET, 메모리 노드가 push하면 REPLY_NET.
 *   SM 노드로의 pop은 REPLY_NET, 메모리 노드로의 pop은 REQ_NET에서 꺼낸다.
 */
class LocalInterconnect {
 public:
  /*
   * [한국어]
   * LocalInterconnect::LocalInterconnect - 생성자 (직접 호출 비권장, New() 사용 권장)
   *
   * @m_localinct_config: 설정 구조체 const 참조 (멤버 m_inct_config에 바인딩)
   * @return: 없음 (생성자)
   *
   * n_shader, n_mem을 0으로, n_subnets를 m_localinct_config.subnets로 초기화.
   * net[] 벡터는 CreateInterconnect() 호출 시 할당된다.
   *
   * 호출 체인: LocalInterconnect::New() → [이 함수]
   */
  LocalInterconnect(const struct inct_config& m_localinct_config);

  /* [한국어] 소멸자 — net[] 벡터의 모든 xbar_router 포인터를 delete. */
  ~LocalInterconnect();

  /*
   * [한국어]
   * LocalInterconnect::New - LocalInterconnect 팩토리 메서드 (정적)
   *
   * @m_inct_config: 설정 구조체 const 참조
   * @return: 새로 생성된 LocalInterconnect 객체 포인터
   *
   * new LocalInterconnect(m_inct_config)를 호출하여 객체를 힙에 생성하고 반환한다.
   * icnt_wrapper_init()에서 g_localicnt_interface에 저장된다.
   *
   * 호출 체인: icnt_wrapper_init() → [이 함수] → LocalInterconnect 생성자
   */
  static LocalInterconnect* New(const struct inct_config& m_inct_config);

  /*
   * [한국어]
   * LocalInterconnect::CreateInterconnect - xbar_router 객체 생성 및 net[] 초기화
   *
   * @n_shader: SM 클러스터 수
   * @n_mem: 메모리 파티션 수
   * @return: 없음 (void)
   *
   * n_subnets 개의 xbar_router를 생성하여 net[]에 저장한다.
   * 각 라우터는 인덱스(0=REQ_NET, 1=REPLY_NET)로 타입을 결정한다.
   *
   * 호출 체인: icnt_create(n_shader, n_mem) → LocalInterconnect_create() → [이 함수]
   */
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  /*
   * [한국어]
   * LocalInterconnect::Init - 인터커넥트 초기화 (현재 no-op)
   *
   * @return: 없음 (void)
   *
   * 현재 빈 함수. 인터페이스 일관성을 위해 존재한다.
   *
   * 호출 체인: icnt_init() → LocalInterconnect_init() → [이 함수]
   */
  void Init();

  /*
   * [한국어]
   * LocalInterconnect::Push - 서브넷 선택 후 패킷 삽입
   *
   * @input_deviceID: 소스 노드 ID
   * @output_deviceID: 목적지 노드 ID
   * @data: 전달할 데이터 포인터
   * @size: 패킷 크기
   * @return: 없음 (void)
   *
   * n_subnets == 1이면 항상 subnet=0(REQ_NET).
   * n_subnets == 2이면 input_deviceID < n_shader이면 subnet=0(REQ_NET),
   * 그 외(메모리 노드)이면 subnet=1(REPLY_NET).
   * Has_Buffer_In() assert 후 xbar_router::Push() 호출.
   *
   * 호출 체인: icnt_push() → LocalInterconnect_push() → [이 함수] → xbar_router::Push()
   */
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);

  /*
   * [한국어]
   * LocalInterconnect::Pop - 서브넷 선택 후 도착 패킷 수신
   *
   * @ouput_deviceID: 수신 노드 ID
   * @return: 도착한 데이터 포인터 (없으면 NULL)
   *
   * ouput_deviceID < n_shader이면 subnet=1(REPLY_NET),
   * 그 외이면 subnet=0(REQ_NET)에서 꺼낸다.
   * (Push와 반대 방향: SM은 REPLY_NET에서 응답을 받고, 메모리는 REQ_NET에서 요청을 받음)
   *
   * 호출 체인: icnt_pop() → LocalInterconnect_pop() → [이 함수] → xbar_router::Pop()
   */
  void* Pop(unsigned ouput_deviceID);

  /*
   * [한국어]
   * LocalInterconnect::Advance - 모든 서브넷의 한 사이클 진행
   *
   * @return: 없음 (void)
   *
   * net[] 벡터의 모든 xbar_router->Advance()를 순서대로 호출한다.
   * REQ_NET, REPLY_NET 모두 동시에 한 사이클 진행된다.
   *
   * 호출 체인: icnt_transfer() → LocalInterconnect_transfer() → [이 함수] → xbar_router::Advance()
   */
  void Advance();

  /*
   * [한국어]
   * LocalInterconnect::Busy - 모든 서브넷에 전달 중 패킷이 있는지 확인
   *
   * @return: 어느 서브넷이든 Busy()이면 true, 모두 비어 있으면 false
   *
   * 호출 체인: icnt_busy() → LocalInterconnect_busy() → [이 함수] → xbar_router::Busy()
   */
  bool Busy() const;

  /*
   * [한국어]
   * LocalInterconnect::HasBuffer - back-pressure 판단을 위한 버퍼 여유 확인
   *
   * @deviceID: 삽입하려는 소스 노드 ID
   * @size: 삽입하려는 패킷 크기 (현재 로직에서는 1로 고정)
   * @return: 해당 서브넷의 입력 버퍼에 여유가 있으면 true, 없으면 false
   *
   * n_subnets > 1이고 deviceID >= n_shader이면 REPLY_NET의 Has_Buffer_In(),
   * 그 외이면 REQ_NET의 Has_Buffer_In()을 호출한다. update_counter=true로 전달.
   *
   * 호출 체인: icnt_has_buffer() → LocalInterconnect_has_buffer() → [이 함수] → xbar_router::Has_Buffer_In()
   */
  bool HasBuffer(unsigned deviceID, unsigned int size) const;

  /*
   * [한국어]
   * LocalInterconnect::DisplayStats - 각 서브넷의 성능 통계 출력
   *
   * @return: 없음 (void)
   *
   * REQ_NET과 REPLY_NET 각각의 packets_num, cycles, throughput,
   * conflict rate, BLP(Bank-Level Parallelism), buffer utilization 등을 printf로 출력한다.
   *
   * 호출 체인: icnt_display_stats() → LocalInterconnect_display_stats() → [이 함수]
   */
  void DisplayStats() const;

  /* [한국어] 전체 누적 통계 출력 — 현재 빈 구현. 인터페이스 일관성을 위해 존재. */
  void DisplayOverallStats() const;

  /* [한국어] 플릿 크기 반환 — LOCAL_INCT_FLIT_SIZE(40바이트) 고정 반환.
   * local_xbar는 모든 패킷을 1플릿으로 처리하는 단순화 모델이므로 플릿 크기 고정. */
  unsigned GetFlitSize() const;

  /* [한국어] 내부 상태 fp에 덤프 — 현재 "Under implementation" 메시지만 출력. 디버그 예약 슬롯. */
  void DisplayState(FILE* fp) const;

 protected:
  const inct_config& m_inct_config;
  /* [한국어] 생성자에서 전달된 설정 구조체의 const 참조.
   * 소멸자에서 subnets 값에 접근하기 위해 참조를 유지.
   * 설정자: 생성자 초기화 리스트에서 바인딩.
   * 읽는 자: 생성자(n_subnets 설정), 소멸자(루프 상한).
   * 동기화: const 참조로 변경 불가. */

  unsigned n_shader, n_mem;
  /* [한국어] SM 클러스터 수와 메모리 파티션 수.
   * 설정자: CreateInterconnect() 호출 시 설정.
   * 읽는 자: Push(), Pop(), HasBuffer()에서 노드 ID 범위 판단.
   * 동기화: CreateInterconnect() 이후 읽기 전용. */

  unsigned n_subnets;
  /* [한국어] 서브넷(xbar_router) 수 — 생성자에서 m_inct_config.subnets로 초기화.
   * 설정자: 생성자에서 설정.
   * 읽는 자: CreateInterconnect()에서 net.resize() 크기 결정, Push()/Advance() 루프 상한.
   * 동기화: 초기화 이후 읽기 전용. */

  vector<xbar_router*> net;
  /* [한국어] 서브넷별 xbar_router 포인터 벡터 (크기: n_subnets, 기본 [REQ_NET, REPLY_NET]).
   * 설정자: CreateInterconnect()에서 resize 및 new xbar_router로 채움.
   * 읽는 자: Push(), Pop(), Advance(), Busy(), HasBuffer(), DisplayStats() 등 대부분의 함수.
   * 동기화: 초기화 이후 구조 변경 없음, 내부 xbar_router 상태만 변경됨. */
};

#endif
