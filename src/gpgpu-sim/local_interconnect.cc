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
 * [한국어 설명] 로컬 인터커넥트(간이 크로스바 ICNT) 구현 (local_interconnect.cc)
 *
 * === 파일의 역할 ===
 * local_interconnect.h에서 선언된 xbar_router 및 LocalInterconnect 클래스의 모든
 * 메서드를 구현한다. xbar_router는 NAIVE_RR과 iSLIP 두 가지 중재 알고리즘으로
 * 크로스바 스위치의 한 사이클 동작을 구현하며, LocalInterconnect는 REQ_NET과
 * REPLY_NET 두 서브넷을 관리하며 icnt_wrapper의 함수 포인터 인터페이스에 연결된다.
 * 모든 패킷을 1 플릿(40바이트 고정)으로 처리하는 단순화 모델로, 실제 NoC 레이턴시
 * 없이 단순 버퍼-to-버퍼 전달로 SM ↔ 메모리 파티션 간 통신을 시뮬레이션한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * icnt_wrapper.cc에서 g_network_mode == LOCAL_XBAR로 설정된 경우 선택되는 ICNT 구현체.
 * gpu-sim.cc의 사이클 루프 → icnt_transfer() → LocalInterconnect::Advance()
 * → 각 서브넷의 xbar_router::Advance()로 한 사이클씩 처리된다.
 * 실행 컨텍스트: 시뮬레이터 메인 스레드, 매 시뮬레이션 사이클마다 동기 실행.
 *
 * === 타 모듈과의 연결 ===
 * - local_interconnect.h: 이 파일이 구현하는 클래스 선언 포함
 * - mem_fetch.h: 패킷의 실제 페이로드 타입 (void* data는 통상 mem_fetch*로 캐스팅)
 * - icnt_wrapper.cc: LocalInterconnect의 멤버 함수를 래핑하여 함수 포인터에 바인딩
 * 데이터 흐름: icnt_push() → LocalInterconnect::Push() → xbar_router::in_buffers
 *              → Advance()로 out_buffers로 이동 → icnt_pop() → LocalInterconnect::Pop()
 *              → 목적지 모듈(shader.cc 또는 gpu-sim.cc)
 *
 * === 주요 함수/구조체 요약 ===
 * xbar_router::xbar_router()  - 크로스바 라우터 초기화 (버퍼 resize, 통계 초기화)
 * xbar_router::RR_Advance()   - 단순 라운드로빈 중재: 입력 버퍼→출력 버퍼 이동
 * xbar_router::iSLIP_Advance()- iSLIP 알고리즘 중재: 출력 포트별 공평한 스케줄링
 * LocalInterconnect::Push()   - 서브넷 선택 후 xbar_router에 패킷 삽입
 * LocalInterconnect::Pop()    - 수신 노드 방향의 서브넷에서 패킷 꺼내기
 * LocalInterconnect::DisplayStats() - 각 서브넷의 throughput, 충돌률, 활용도 출력
 */

#include <algorithm>   /* [한국어] std::find — iSLIP_Advance()에서 출력 포트 중복 탐지용 */
#include <cmath>       /* [한국어] cmath 수학 함수 — 현재 직접 사용 없으나 헤더 의존으로 포함 */
#include <fstream>     /* [한국어] std::fstream — DisplayState() 등에서 파일 I/O 가능하도록 포함 */
#include <iomanip>     /* [한국어] std::setw, setprecision 등 — DisplayStats() 출력 포맷에 간접 사용 */
#include <iostream>    /* [한국어] std::cout — 디버그 출력용 (현재 printf 위주이나 의존으로 포함) */
#include <sstream>     /* [한국어] std::stringstream — 현재 직접 사용 없으나 의존으로 포함 */
#include <utility>     /* [한국어] std::pair 등 — 현재 직접 사용 없으나 의존으로 포함 */

#include "local_interconnect.h"  /* [한국어] xbar_router, LocalInterconnect 클래스 선언 */
#include "mem_fetch.h"           /* [한국어] mem_fetch 타입 — Push()의 void* data는 통상 mem_fetch* */

/*
 * [한국어]
 * xbar_router::xbar_router - 크로스바 라우터 생성자
 *
 * @router_id: 이 라우터의 고유 ID (net[] 배열 인덱스, 0=REQ_NET, 1=REPLY_NET)
 * @m_type: REQ_NET 또는 REPLY_NET — 트래픽 방향 결정
 * @n_shader: SM 클러스터 수
 * @n_mem: 메모리 파티션 수
 * @m_localinct_config: 버퍼 크기, 중재 알고리즘 등 설정 구조체
 * @return: 없음 (생성자)
 *
 * total_nodes = n_shader + n_mem 크기로 in_buffers, out_buffers, next_node 벡터를 할당.
 * m_type에 따라 REQ_NET이면 SM이 입력(active_in_buffers = n_shader), 메모리가 출력,
 * REPLY_NET이면 그 반대로 설정한다.
 * 모든 통계 카운터를 0으로 초기화한다.
 *
 * 호출 체인: LocalInterconnect::CreateInterconnect() → [이 함수]
 */
xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config& m_localinct_config) {
  m_id = router_id;                              /* [한국어] 라우터 고유 ID 저장 (디버그 출력용) */
  router_type = m_type;                          /* [한국어] 라우터 타입(REQ_NET/REPLY_NET) 저장 */
  _n_mem = n_mem;                                /* [한국어] 메모리 파티션 수 저장 */
  _n_shader = n_shader;                          /* [한국어] SM 클러스터 수 저장 */
  total_nodes = n_shader + n_mem;                /* [한국어] 전체 노드 수 = SM 수 + 메모리 파티션 수 */
  verbose = m_localinct_config.verbose;          /* [한국어] 디버그 출력 레벨 설정 */
  grant_cycles = m_localinct_config.grant_cycles; /* [한국어] iSLIP grant 유지 사이클 수 설정 */
  grant_cycles_count = m_localinct_config.grant_cycles; /* [한국어] grant 사이클 카운터를 초기값으로 초기화 */
  in_buffers.resize(total_nodes);                /* [한국어] 각 노드의 입력 큐 벡터를 total_nodes 크기로 초기화 */
  out_buffers.resize(total_nodes);               /* [한국어] 각 노드의 출력 큐 벡터를 total_nodes 크기로 초기화 */
  next_node.resize(total_nodes, 0);              /* [한국어] iSLIP 중재용 next_node 벡터를 total_nodes 크기, 모두 0으로 초기화 */
  in_buffer_limit = m_localinct_config.in_buffer_limit;   /* [한국어] 입력 버퍼 최대 패킷 수 설정 */
  out_buffer_limit = m_localinct_config.out_buffer_limit; /* [한국어] 출력 버퍼 최대 패킷 수 설정 */
  arbit_type = m_localinct_config.arbiter_algo;  /* [한국어] 중재 알고리즘 타입 설정 (NAIVE_RR 또는 iSLIP) */
  next_node_id = 0;                              /* [한국어] RR 중재 시작 노드 ID를 0으로 초기화 */
  if (m_type == REQ_NET) {
    /* [한국어] REQ_NET: SM 클러스터들이 요청을 보내는 입력 방향, 메모리 파티션들이 출력 방향 */
    active_in_buffers = n_shader;   /* [한국어] REQ_NET 활성 입력 버퍼 수 = SM 클러스터 수 */
    active_out_buffers = n_mem;     /* [한국어] REQ_NET 활성 출력 버퍼 수 = 메모리 파티션 수 */
  } else if (m_type == REPLY_NET) {
    /* [한국어] REPLY_NET: 메모리 파티션들이 응답을 보내는 입력 방향, SM 클러스터들이 출력 방향 */
    active_in_buffers = n_mem;      /* [한국어] REPLY_NET 활성 입력 버퍼 수 = 메모리 파티션 수 */
    active_out_buffers = n_shader;  /* [한국어] REPLY_NET 활성 출력 버퍼 수 = SM 클러스터 수 */
  }

  /* [한국어] 모든 통계 카운터를 0으로 초기화 */
  cycles = 0;          /* [한국어] 총 사이클 카운터 초기화 */
  conflicts = 0;       /* [한국어] 총 출력 포트 경합 횟수 초기화 */
  out_buffer_full = 0; /* [한국어] 출력 버퍼 포화로 인한 전달 실패 횟수 초기화 */
  in_buffer_full = 0;  /* [한국어] 입력 버퍼 포화로 인한 삽입 거부 횟수 초기화 */
  out_buffer_util = 0; /* [한국어] 출력 버퍼 누적 점유량 초기화 */
  in_buffer_util = 0;  /* [한국어] 입력 버퍼 누적 점유량 초기화 */
  packets_num = 0;     /* [한국어] 삽입된 총 패킷 수 초기화 */
  conflicts_util = 0;  /* [한국어] 활성 사이클에서의 경합 횟수 초기화 */
  cycles_util = 0;     /* [한국어] 활성 사이클 수 초기화 */
  reqs_util = 0;       /* [한국어] 활성 사이클에서 전달 성공한 패킷 수 초기화 */
}

/* [한국어] xbar_router 소멸자 — in_buffers, out_buffers, next_node는 vector 소멸 시 자동 해제됨 */
xbar_router::~xbar_router() {}

/*
 * [한국어]
 * xbar_router::Push - 입력 노드의 버퍼에 패킷 추가
 *
 * @input_deviceID: 삽입 소스 노드 ID (0 ~ total_nodes-1)
 * @output_deviceID: 패킷의 목적지 노드 ID
 * @data: 전달할 데이터 포인터 (통상 mem_fetch*)
 * @size: 패킷 크기 (현재 local_xbar에서는 1플릿 고정이므로 실질적으로 미사용)
 * @return: 없음 (void)
 *
 * input_deviceID 범위 검사 후 Packet 구조체로 묶어 in_buffers[input_deviceID]에 push.
 * packets_num 통계를 증가시킨다.
 * 사전에 Has_Buffer_In()으로 버퍼 여유를 확인해야 한다 (assert는 인덱스 범위만 검사).
 *
 * 호출 체인: LocalInterconnect::Push() → [이 함수]
 */
void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void* data, unsigned int size) {
  assert(input_deviceID < total_nodes);  /* [한국어] 노드 ID 범위 검사 — total_nodes 초과 시 즉시 종료 */
  in_buffers[input_deviceID].push(Packet(data, output_deviceID));
  /* [한국어] data와 output_deviceID를 Packet으로 묶어 해당 입력 노드의 버퍼 뒤에 추가 */
  packets_num++;  /* [한국어] 이 라우터에 삽입된 총 패킷 수 증가 — DisplayStats()의 throughput 계산용 */
}

/*
 * [한국어]
 * xbar_router::Pop - 출력 노드의 버퍼에서 도착 패킷 꺼내기
 *
 * @ouput_deviceID: 수신 노드 ID (0 ~ total_nodes-1)
 * @return: 도착한 데이터 포인터 (out_buffers가 비어 있으면 NULL)
 *
 * out_buffers[ouput_deviceID]의 front에서 Packet.data를 꺼내 반환.
 * 버퍼가 비어 있으면 NULL 반환.
 *
 * 호출 체인: LocalInterconnect::Pop() → [이 함수]
 */
void* xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);  /* [한국어] 노드 ID 범위 검사 */
  void* data = NULL;  /* [한국어] 기본값은 NULL — 버퍼가 비어 있을 경우 그대로 반환 */

  if (!out_buffers[ouput_deviceID].empty()) {
    /* [한국어] 출력 버퍼에 도착한 패킷이 있는 경우 */
    data = out_buffers[ouput_deviceID].front().data;
    /* [한국어] 큐의 앞쪽(가장 먼저 도착한) 패킷의 data 포인터 추출 */
    out_buffers[ouput_deviceID].pop();
    /* [한국어] 꺼낸 패킷을 출력 버퍼에서 제거 (FIFO 순서) */
  }

  return data;  /* [한국어] 도착한 데이터 포인터 반환 (없으면 NULL) */
}

/*
 * [한국어]
 * xbar_router::Has_Buffer_In - 입력 노드 버퍼의 여유 공간 확인
 *
 * @input_deviceID: 확인할 소스 노드 ID
 * @size: 삽입하려는 패킷 수 (통상 1)
 * @update_counter: true이면 버퍼 포화 시 in_buffer_full 통계 증가
 * @return: (현재 큐 크기 + size <= in_buffer_limit)이면 true, 아니면 false
 *
 * LocalInterconnect::HasBuffer()에서 back-pressure 판단에 사용된다.
 * update_counter=true로 호출 시 버퍼 포화 통계도 함께 갱신한다.
 *
 * 호출 체인: LocalInterconnect::HasBuffer() → [이 함수]
 *            또는 xbar_router::RR_Advance()/iSLIP_Advance() 내부에서 직접 호출되지 않음
 *            (LocalInterconnect::Push()의 assert에서 사전 확인)
 */
bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size,
                                bool update_counter) {
  assert(input_deviceID < total_nodes);  /* [한국어] 노드 ID 범위 검사 */

  bool has_buffer =
      (in_buffers[input_deviceID].size() + size <= in_buffer_limit);
  /* [한국어] 현재 버퍼 크기 + 삽입 요청 크기가 한도 이하이면 여유 있음 */
  if (update_counter && !has_buffer) in_buffer_full++;
  /* [한국어] 통계 갱신 요청이고 버퍼가 가득 찼다면 포화 카운터 증가 */

  return has_buffer;  /* [한국어] 버퍼 여유 여부 반환 — false이면 상위에서 back-pressure 발동 */
}

/*
 * [한국어]
 * xbar_router::Has_Buffer_Out - 출력 노드 버퍼의 여유 공간 확인
 *
 * @output_deviceID: 확인할 목적지 노드 ID
 * @size: 이동하려는 패킷 수 (통상 1)
 * @return: (현재 큐 크기 + size <= out_buffer_limit)이면 true, 아니면 false
 *
 * RR_Advance()와 iSLIP_Advance() 내부에서 패킷을 out_buffers로 이동하기 전에 호출한다.
 * false이면 해당 사이클에 패킷을 전달할 수 없어 경합으로 처리된다.
 *
 * 호출 체인: xbar_router::RR_Advance() 또는 iSLIP_Advance() → [이 함수]
 */
bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size) {
  return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
  /* [한국어] 출력 버퍼 현재 크기 + 이동 요청 크기가 한도 이하이면 이동 가능 */
}

/*
 * [한국어]
 * xbar_router::Advance - 한 사이클 중재 실행 (arbit_type에 따라 분기)
 *
 * @return: 없음 (void)
 *
 * arbit_type이 NAIVE_RR이면 RR_Advance(), iSLIP이면 iSLIP_Advance()를 호출한다.
 * 알 수 없는 타입은 assert(0)으로 종료한다.
 * 이 함수는 LocalInterconnect::Advance()에서 각 서브넷마다 호출된다.
 *
 * 호출 체인: LocalInterconnect::Advance() → [이 함수] → RR_Advance() 또는 iSLIP_Advance()
 */
void xbar_router::Advance() {
  if (arbit_type == NAIVE_RR)
    RR_Advance();          /* [한국어] NAIVE_RR 모드: 단순 라운드로빈 중재 실행 */
  else if (arbit_type == iSLIP)
    iSLIP_Advance();       /* [한국어] iSLIP 모드: iterative SLIP 중재 알고리즘 실행 */
  else
    assert(0);             /* [한국어] 알 수 없는 중재 알고리즘 타입 — 즉시 오류 종료 */
}

/*
 * [한국어]
 * xbar_router::RR_Advance - 단순 라운드로빈(NAIVE_RR) 크로스바 중재 실행
 *
 * @return: 없음 (void)
 *
 * next_node_id를 기준으로 라운드로빈 순서로 모든 입력 노드를 순회한다.
 * 각 입력 노드에 패킷이 있고 목적지 출력 버퍼에 공간이 있으며 해당 사이클에
 * 그 출력 포트로의 전달이 아직 없었으면(issued[] == false) 패킷을 이동한다.
 * issued[] 배열로 각 출력 포트는 한 사이클에 최대 1개 패킷만 수신한다.
 * 처리 후 next_node_id를 1 전진하여 다음 사이클에 다른 노드를 먼저 서비스한다.
 * 마지막에 버퍼 utilization 통계를 갱신하고 cycles를 증가시킨다.
 *
 * 호출 체인: xbar_router::Advance() → [이 함수]
 */
void xbar_router::RR_Advance() {
  bool active = false;                           /* [한국어] 이 사이클에 처리할 패킷이 있었는지 여부 — 활성 사이클 통계용 */
  vector<bool> issued(total_nodes, false);       /* [한국어] 각 출력 포트의 이번 사이클 전달 여부 배열 — 포트당 1패킷 제한 */
  unsigned conflict_sub = 0;                     /* [한국어] 이번 사이클의 출력 포트 경합(충돌) 횟수 */
  unsigned reqs = 0;                             /* [한국어] 이번 사이클에 성공적으로 전달된 패킷 수 */

  for (unsigned i = 0; i < total_nodes; ++i) {
    /* [한국어] next_node_id를 시작점으로 라운드로빈 순서로 모든 입력 노드 순회 */
    unsigned node_id = (i + next_node_id) % total_nodes;
    /* [한국어] 라운드로빈 오프셋 적용 — next_node_id에서 시작하여 전체를 순환 */

    if (!in_buffers[node_id].empty()) {
      /* [한국어] 해당 입력 노드에 대기 중인 패킷이 있는 경우 */
      active = true;  /* [한국어] 활성 사이클임을 표시 */
      Packet _packet = in_buffers[node_id].front();
      /* [한국어] 입력 버퍼의 앞쪽(가장 오래 대기한) 패킷 참조 */
      // ensure that the outbuffer has space and not issued before in this cycle
      if (Has_Buffer_Out(_packet.output_deviceID, 1)) {
        /* [한국어] 목적지 출력 버퍼에 1개 패킷을 추가할 공간이 있는 경우 */
        if (!issued[_packet.output_deviceID]) {
          /* [한국어] 이번 사이클에 아직 해당 출력 포트로 전달된 패킷이 없는 경우 */
          out_buffers[_packet.output_deviceID].push(_packet);
          /* [한국어] 패킷을 목적지 출력 버퍼로 이동 */
          in_buffers[node_id].pop();
          /* [한국어] 입력 버퍼에서 패킷 제거 */
          issued[_packet.output_deviceID] = true;
          /* [한국어] 이번 사이클에 이 출력 포트로 이미 전달했음을 표시 */
          reqs++;  /* [한국어] 성공적으로 전달된 패킷 수 증가 */
        } else
          conflict_sub++;
          /* [한국어] 동일 출력 포트로 이미 이번 사이클에 전달이 있었음 — 경합 발생 */
      } else {
        out_buffer_full++;
        /* [한국어] 출력 버퍼가 가득 차서 패킷을 이동할 수 없음 — out_buffer_full 통계 증가 */

        if (issued[_packet.output_deviceID]) conflict_sub++;
        /* [한국어] 출력 버퍼 포화이면서 이번 사이클에 이미 전달이 있었다면 경합으로도 집계 */
      }
    }
  }
  next_node_id = next_node_id + 1;               /* [한국어] 다음 사이클의 라운드로빈 시작 노드 전진 */
  next_node_id = (next_node_id % total_nodes);   /* [한국어] 모듈러 연산으로 total_nodes 범위 내 순환 */

  conflicts += conflict_sub;  /* [한국어] 이번 사이클 경합 횟수를 총 경합 카운터에 누적 */
  if (active) {
    /* [한국어] 이번 사이클에 처리할 패킷이 있었던 경우 활성 사이클 통계 갱신 */
    conflicts_util += conflict_sub;  /* [한국어] 활성 사이클에서의 경합 횟수 누적 */
    cycles_util++;                   /* [한국어] 활성 사이클 수 증가 */
    reqs_util += reqs;               /* [한국어] 활성 사이클에서 전달 성공한 패킷 수 누적 */
  }

  if (verbose) {
    /* [한국어] 디버그 출력 활성화 시 이번 사이클의 경합 및 전달 현황 출력 */
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    /* [한국어] 라우터 ID, 현재 사이클, 이번 사이클 경합 횟수 출력 */
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
    /* [한국어] 라우터 ID, 현재 사이클, 이번 사이클 전달 성공 패킷 수 출력 */
  }

  // collect some stats about buffer util
  /* [한국어] 버퍼 utilization 통계 수집 — 모든 노드의 현재 버퍼 점유량 누적 */
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();    /* [한국어] 각 입력 버퍼의 현재 크기를 누적 점유량에 추가 */
    out_buffer_util += out_buffers[i].size();  /* [한국어] 각 출력 버퍼의 현재 크기를 누적 점유량에 추가 */
  }

  cycles++;  /* [한국어] 총 사이클 카운터 증가 */
}

// iSLIP algorithm
// McKeown, Nick. "The iSLIP scheduling algorithm for input-queued switches."
// IEEE/ACM transactions on networking 2 (1999): 188-201.
// https://www.cs.rutgers.edu/~sn624/552-F18/papers/islip.pdf
/*
 * [한국어]
 * xbar_router::iSLIP_Advance - iSLIP 알고리즘 기반 크로스바 중재 실행
 *
 * @return: 없음 (void)
 *
 * iSLIP(iterative Scheduling for Large Input-queued Packet switches) 알고리즘.
 * 참고: McKeown (1999), IEEE/ACM Transactions on Networking.
 *
 * 동작 방식:
 * 1) 경합 통계 계산 단계: 모든 입력 노드의 front 패킷 목적지를 수집하여
 *    동일 목적지 요청이 있으면 conflict_sub 카운트.
 * 2) iSLIP 중재 단계: 출력 포트 i에 대해 next_node[i]를 기준으로 라운드로빈 순서로
 *    입력 노드를 탐색하여 자신을 목적지로 하는 패킷을 찾으면 out_buffers[i]로 이동.
 *    grant가 발생하고 grant_cycles_count == 1이면 next_node[i]를 전진하여
 *    다음 grant 시 다른 입력을 먼저 선택하게 한다 (공평성 보장).
 * 3) grant_cycles_count 관리: 활성 사이클에서 count가 1이 되면 grant_cycles로 재설정,
 *    아직 남아 있으면 감소시켜 같은 grant를 여러 사이클 유지한다.
 *
 * RR_Advance와의 차이: iSLIP은 출력 포트별로 독립적으로 입력을 순회하므로
 * 여러 출력 포트가 동시에 서로 다른 입력으로부터 패킷을 받을 수 있어 더 높은 throughput.
 *
 * 호출 체인: xbar_router::Advance() → [이 함수]
 */
void xbar_router::iSLIP_Advance() {
  vector<unsigned> node_tmp;  /* [한국어] 이번 사이클의 요청 목적지 집합 — 경합 검출에 사용 */
  bool active = false;        /* [한국어] 이번 사이클에 처리할 패킷이 있는지 여부 */

  unsigned conflict_sub = 0;  /* [한국어] 이번 사이클의 경합 횟수 */
  unsigned reqs = 0;          /* [한국어] 이번 사이클에 성공적으로 전달된 패킷 수 */

  // calcaulte how many conflicts are there for stats
  /* [한국어] 1단계: 경합 통계 계산 — 각 입력의 front 패킷 목적지를 수집하여 중복 탐지 */
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) {
      /* [한국어] 입력 노드 i에 대기 중인 패킷이 있는 경우 */
      Packet _packet_tmp = in_buffers[i].front();
      /* [한국어] 입력 버퍼의 front 패킷(가장 오래 대기한 패킷) 참조 */
      if (!node_tmp.empty()) {
        /* [한국어] 이미 수집된 목적지가 있는 경우 */
        if (std::find(node_tmp.begin(), node_tmp.end(),
                      _packet_tmp.output_deviceID) != node_tmp.end()) {
          conflict_sub++;
          /* [한국어] 동일 목적지 노드를 원하는 요청이 이미 있음 → 경합 발생 */
        } else
          node_tmp.push_back(_packet_tmp.output_deviceID);
          /* [한국어] 새로운 목적지 추가 */
      } else {
        node_tmp.push_back(_packet_tmp.output_deviceID);
        /* [한국어] 첫 번째 목적지 등록 */
      }
      active = true;  /* [한국어] 처리할 패킷이 존재함을 표시 */
    }
  }

  conflicts += conflict_sub;  /* [한국어] 이번 사이클 경합 횟수를 총 경합 카운터에 누적 */
  if (active) {
    conflicts_util += conflict_sub;  /* [한국어] 활성 사이클에서의 경합 횟수 누적 */
    cycles_util++;                   /* [한국어] 활성 사이클 수 증가 */
  }
  // do iSLIP
  /* [한국어] 2단계: iSLIP 중재 실행 — 각 출력 포트 i에 대해 next_node[i]부터 탐색 */
  for (unsigned i = 0; i < total_nodes; ++i) {
    /* [한국어] 출력 포트 i에 대해 버퍼 공간이 있는지 확인 */
    if (Has_Buffer_Out(i, 1)) {
      /* [한국어] 출력 버퍼에 공간이 있는 경우: 입력 노드를 라운드로빈 순으로 탐색 */
      for (unsigned j = 0; j < total_nodes; ++j) {
        unsigned node_id = (j + next_node[i]) % total_nodes;
        /* [한국어] next_node[i]를 시작점으로 라운드로빈 순서의 입력 노드 ID 계산 */

        if (!in_buffers[node_id].empty()) {
          /* [한국어] 해당 입력 노드에 패킷이 있는 경우 */
          Packet _packet = in_buffers[node_id].front();
          /* [한국어] 입력 버퍼 front 패킷 참조 */
          if (_packet.output_deviceID == i) {
            /* [한국어] 이 패킷의 목적지가 현재 출력 포트 i와 일치하면 전달 */
            out_buffers[_packet.output_deviceID].push(_packet);
            /* [한국어] 패킷을 출력 버퍼로 이동 */
            in_buffers[node_id].pop();
            /* [한국어] 입력 버퍼에서 패킷 제거 */
            if (verbose)
              printf("%d : cycle %llu : send req from %d to %d\n", m_id, cycles,
                     node_id, i - _n_shader);
            /* [한국어] 디버그 출력: 패킷을 node_id에서 메모리 파티션(i - _n_shader)으로 전달 */
            if (grant_cycles_count == 1)
              next_node[i] = (++node_id % total_nodes);
            /* [한국어] grant 유지 기간이 끝났으면 next_node[i]를 전진 — 다음 grant 시 다른 입력 우선 */
            if (verbose) {
              /* [한국어] 디버그 모드: 같은 출력을 원하지만 서비스받지 못한 다른 입력 노드들 출력 */
              for (unsigned k = j + 1; k < total_nodes; ++k) {
                unsigned node_id2 = (k + next_node[i]) % total_nodes;
                if (!in_buffers[node_id2].empty()) {
                  Packet _packet2 = in_buffers[node_id2].front();

                  if (_packet2.output_deviceID == i)
                    printf("%d : cycle %llu : cannot send req from %d to %d\n",
                           m_id, cycles, node_id2, i - _n_shader);
                  /* [한국어] 이번 사이클에 i로의 전달 기회를 얻지 못한 대기 중인 패킷 로그 */
                }
              }
            }

            reqs++;  /* [한국어] 이번 사이클 전달 성공 패킷 수 증가 */
            break;   /* [한국어] 출력 포트 i에 대한 grant 완료 — 내부 루프 종료 (포트당 1패킷) */
          }
        }
      }
    } else
      out_buffer_full++;
    /* [한국어] 출력 버퍼 i가 가득 찬 경우 — 이 출력 포트로는 이번 사이클에 전달 불가 */
  }

  if (active) {
    reqs_util += reqs;  /* [한국어] 활성 사이클에서 전달 성공한 패킷 수 누적 */
  }

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);
  /* [한국어] 디버그 출력: 현재 grant_cycles 설정값 확인 */

  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  /* [한국어] 활성 사이클이고 카운터가 1이 되었으면 grant_cycles로 재설정 (다음 grant 준비) */
  else if (active)
    grant_cycles_count--;
  /* [한국어] 활성 사이클이고 아직 남은 count가 있으면 감소 — 현재 grant를 계속 유지 */

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
    /* [한국어] 디버그 출력: 이번 사이클의 경합 횟수와 전달 성공 패킷 수 */
  }

  // collect some stats about buffer util
  /* [한국어] 버퍼 utilization 통계 수집 — 모든 노드의 현재 버퍼 점유량 누적 */
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();    /* [한국어] 각 입력 버퍼 현재 크기 누적 */
    out_buffer_util += out_buffers[i].size();  /* [한국어] 각 출력 버퍼 현재 크기 누적 */
  }

  cycles++;  /* [한국어] 총 사이클 카운터 증가 */
}

/*
 * [한국어]
 * xbar_router::Busy - 버퍼에 전달 중 패킷이 있는지 확인
 *
 * @return: in_buffers 또는 out_buffers 중 하나라도 비어 있지 않으면 true
 *
 * 모든 노드의 입력/출력 버퍼를 순회하여 패킷이 남아 있으면 true 반환.
 * 시뮬레이션 종료 시 ICNT가 완전히 드레인되었는지 확인하는 데 사용된다.
 *
 * 호출 체인: LocalInterconnect::Busy() → [이 함수]
 */
bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) return true;   /* [한국어] 입력 버퍼에 아직 대기 중인 패킷이 있으면 busy */

    if (!out_buffers[i].empty()) return true;  /* [한국어] 출력 버퍼에 아직 꺼내지 않은 패킷이 있으면 busy */
  }
  return false;  /* [한국어] 모든 버퍼가 비어 있음 — 이 라우터는 idle 상태 */
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// assume all the packets are one flit
/* [한국어] local_xbar는 모든 패킷을 1개의 플릿으로 처리하는 단순화 가정을 사용.
 * 실제 NoC에서는 큰 패킷을 여러 플릿으로 분할하지만, local_xbar는 이를 생략. */
#define LOCAL_INCT_FLIT_SIZE 40
/* [한국어] local_xbar의 플릿 크기: 40바이트 고정.
 * GetFlitSize()가 이 값을 반환하며, 상위 레벨에서 패킷 크기를 플릿 수로 환산할 때 사용.
 * 실제 mem_fetch 크기(통상 cache line 64바이트)와 다를 수 있으나 단순화 모델에서는 무관. */

/*
 * [한국어]
 * LocalInterconnect::New - LocalInterconnect 팩토리 메서드 (정적)
 *
 * @m_localinct_config: 설정 구조체 const 참조
 * @return: 힙에 생성된 LocalInterconnect 객체 포인터
 *
 * new LocalInterconnect(m_localinct_config)를 호출하여 객체를 생성하고 반환한다.
 * icnt_wrapper_init()에서 g_localicnt_interface에 저장된다.
 *
 * 호출 체인: icnt_wrapper_init() → [이 함수] → LocalInterconnect 생성자
 */
LocalInterconnect* LocalInterconnect::New(
    const struct inct_config& m_localinct_config) {
  LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);
  /* [한국어] LocalInterconnect 객체를 힙에 생성 — g_localicnt_interface에 저장될 객체 */

  return icnt_interface;  /* [한국어] 생성된 객체 포인터 반환 */
}

/*
 * [한국어]
 * LocalInterconnect::LocalInterconnect - 생성자
 *
 * @m_localinct_config: 설정 구조체 const 참조 (m_inct_config 멤버에 바인딩)
 * @return: 없음 (생성자)
 *
 * m_inct_config 참조를 바인딩하고 n_shader, n_mem을 0으로, n_subnets를
 * m_localinct_config.subnets로 초기화한다.
 * net[] 벡터는 CreateInterconnect() 호출 시 채워진다.
 *
 * 호출 체인: LocalInterconnect::New() → [이 함수]
 */
LocalInterconnect::LocalInterconnect(
    const struct inct_config& m_localinct_config)
    : m_inct_config(m_localinct_config) {  /* [한국어] 설정 구조체 const 참조를 멤버에 바인딩 */
  n_shader = 0;  /* [한국어] SM 클러스터 수 초기화 — CreateInterconnect()에서 설정됨 */
  n_mem = 0;     /* [한국어] 메모리 파티션 수 초기화 */
  n_subnets = m_localinct_config.subnets;  /* [한국어] 서브넷 수 설정 (기본값 2: REQ_NET + REPLY_NET) */
}

/*
 * [한국어]
 * LocalInterconnect::~LocalInterconnect - 소멸자
 *
 * net[] 벡터에 저장된 모든 xbar_router 포인터를 delete하여 메모리를 해제한다.
 * m_inct_config.subnets 만큼 루프를 돌며 각 xbar_router 객체를 해제한다.
 *
 * 호출 체인: 시뮬레이션 종료 시 자동 호출
 */
LocalInterconnect::~LocalInterconnect() {
  for (unsigned i = 0; i < m_inct_config.subnets; ++i) {
    delete net[i];  /* [한국어] 각 서브넷의 xbar_router 객체 메모리 해제 */
  }
}

/*
 * [한국어]
 * LocalInterconnect::CreateInterconnect - xbar_router 서브넷 생성 및 초기화
 *
 * @m_n_shader: SM 클러스터 수
 * @m_n_mem: 메모리 파티션 수
 * @return: 없음 (void)
 *
 * n_shader, n_mem을 설정하고 net[]을 n_subnets 크기로 resize한다.
 * 각 서브넷(인덱스 i)에 대해 xbar_router 객체를 새로 생성하여 net[i]에 저장한다.
 * 인덱스 i를 Interconnect_type으로 캐스팅하므로 0=REQ_NET, 1=REPLY_NET이 된다.
 *
 * 호출 체인: icnt_create(n_shader, n_mem) → LocalInterconnect_create() → [이 함수]
 */
void LocalInterconnect::CreateInterconnect(unsigned m_n_shader,
                                           unsigned m_n_mem) {
  n_shader = m_n_shader;  /* [한국어] SM 클러스터 수 저장 — Push/Pop에서 서브넷 결정에 사용 */
  n_mem = m_n_mem;        /* [한국어] 메모리 파티션 수 저장 */

  net.resize(n_subnets);  /* [한국어] net 벡터를 서브넷 수만큼 크기 조정 */
  for (unsigned i = 0; i < n_subnets; ++i) {
    /* [한국어] 각 서브넷에 대해 xbar_router 생성: i가 타입(0=REQ_NET, 1=REPLY_NET)을 결정 */
    net[i] = new xbar_router(i, static_cast<Interconnect_type>(i), m_n_shader,
                             m_n_mem, m_inct_config);
    /* [한국어] router_id=i, type=Interconnect_type(i), n_shader, n_mem, 설정으로 xbar_router 생성 */
  }
}

/*
 * [한국어]
 * LocalInterconnect::Init - 인터커넥트 초기화 (현재 no-op)
 *
 * @return: 없음 (void)
 *
 * 현재 아무 동작도 하지 않는 빈 함수.
 * 인터페이스 일관성을 위해 존재 (intersim2의 Init()과 동일한 인터페이스 유지).
 *
 * 호출 체인: icnt_init() → LocalInterconnect_init() → [이 함수]
 */
void LocalInterconnect::Init() {
  // empty
  // there is nothing to do
  /* [한국어] local_xbar는 생성자/CreateInterconnect에서 모든 초기화가 완료되므로 추가 Init 불필요 */
}

/*
 * [한국어]
 * LocalInterconnect::Push - 서브넷을 선택하여 xbar_router에 패킷 삽입
 *
 * @input_deviceID: 소스 노드 ID
 * @output_deviceID: 목적지 노드 ID
 * @data: 전달할 데이터 포인터 (통상 mem_fetch*)
 * @size: 패킷 크기 (local_xbar에서는 1플릿 고정으로 처리)
 * @return: 없음 (void)
 *
 * 서브넷 선택 규칙:
 *   n_subnets == 1: 항상 subnet=0 (단일 서브넷 모드)
 *   n_subnets == 2: input_deviceID < n_shader이면 subnet=0(REQ_NET: SM→메모리),
 *                   그 외(메모리 노드)이면 subnet=1(REPLY_NET: 메모리→SM)
 * 선택된 서브넷의 Has_Buffer_In() assert로 버퍼 여유를 검사한 후 Push()를 호출한다.
 *
 * 호출 체인: icnt_push() → LocalInterconnect_push() → [이 함수] → xbar_router::Push()
 */
void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                             void* data, unsigned int size) {
  unsigned subnet;
  if (n_subnets == 1) {
    subnet = 0;  /* [한국어] 단일 서브넷 모드: 항상 subnet=0 */
  } else {
    if (input_deviceID < n_shader) {
      subnet = 0;  /* [한국어] SM 클러스터(ID < n_shader)가 보내는 요청 → REQ_NET(subnet=0) */
    } else {
      subnet = 1;  /* [한국어] 메모리 파티션(ID >= n_shader)이 보내는 응답 → REPLY_NET(subnet=1) */
    }
  }

  // it should have free buffer
  // assume all the packets have size of one
  // no flits are implemented
  assert(net[subnet]->Has_Buffer_In(input_deviceID, 1));
  /* [한국어] 버퍼 여유 assert — Push 호출 전 HasBuffer()로 확인했어야 함.
   * local_xbar는 패킷을 1단위로 처리하므로 크기는 항상 1 */

  net[subnet]->Push(input_deviceID, output_deviceID, data, size);
  /* [한국어] 선택된 서브넷의 xbar_router에 패킷 삽입 */
}

/*
 * [한국어]
 * LocalInterconnect::Pop - 수신 노드 방향의 서브넷에서 패킷 꺼내기
 *
 * @ouput_deviceID: 수신 노드 ID
 * @return: 도착한 데이터 포인터 (없으면 NULL)
 *
 * 서브넷 선택 규칙 (Push와 반대 방향):
 *   ouput_deviceID < n_shader(SM 노드): REPLY_NET(subnet=1)에서 꺼냄
 *     → 메모리 파티션이 REPLY_NET으로 보낸 응답을 SM이 수신
 *   그 외(메모리 노드): REQ_NET(subnet=0)에서 꺼냄
 *     → SM이 REQ_NET으로 보낸 요청을 메모리 파티션이 수신
 *
 * 호출 체인: icnt_pop() → LocalInterconnect_pop() → [이 함수] → xbar_router::Pop()
 */
void* LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  /* [한국어] 수신 노드 ID에 따라 서브넷 결정: SM(0~n_shader-1) → REPLY_NET, 메모리 → REQ_NET */
  int subnet = 0;                              /* [한국어] 기본값: REQ_NET(메모리 파티션 수신 방향) */
  if (ouput_deviceID < n_shader) subnet = 1;  /* [한국어] SM 노드 수신: REPLY_NET에서 응답을 꺼냄 */

  return net[subnet]->Pop(ouput_deviceID);
  /* [한국어] 선택된 서브넷의 출력 버퍼에서 패킷을 꺼내 반환 */
}

/*
 * [한국어]
 * LocalInterconnect::Advance - 모든 서브넷의 한 사이클 중재 실행
 *
 * @return: 없음 (void)
 *
 * n_subnets 개의 모든 서브넷(REQ_NET, REPLY_NET)에 대해 xbar_router::Advance()를
 * 순서대로 호출하여 각 크로스바의 중재를 실행하고 패킷을 이동시킨다.
 * 매 시뮬레이션 사이클마다 gpu-sim.cc 사이클 루프에서 호출된다.
 *
 * 호출 체인: icnt_transfer() → LocalInterconnect_transfer() → [이 함수] → xbar_router::Advance()
 */
void LocalInterconnect::Advance() {
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i]->Advance();  /* [한국어] 서브넷 i(REQ_NET 또는 REPLY_NET)의 한 사이클 중재 실행 */
  }
}

/*
 * [한국어]
 * LocalInterconnect::Busy - 모든 서브넷에 전달 중 패킷이 있는지 확인
 *
 * @return: 어느 서브넷이든 Busy()이면 true, 모두 비어 있으면 false
 *
 * 모든 서브넷의 xbar_router::Busy()를 호출하여 하나라도 true이면 true 반환.
 * 시뮬레이션 종료 전 ICNT 드레인 완료 여부 확인에 사용된다.
 *
 * 호출 체인: icnt_busy() → LocalInterconnect_busy() → [이 함수] → xbar_router::Busy()
 */
bool LocalInterconnect::Busy() const {
  for (unsigned i = 0; i < n_subnets; ++i) {
    if (net[i]->Busy()) return true;  /* [한국어] 서브넷 i에 전달 중 패킷이 있으면 즉시 true 반환 */
  }
  return false;  /* [한국어] 모든 서브넷의 모든 버퍼가 비어 있음 — ICNT 완전 드레인 상태 */
}

/*
 * [한국어]
 * LocalInterconnect::HasBuffer - back-pressure 판단을 위한 버퍼 여유 확인
 *
 * @deviceID: 삽입 소스 노드 ID
 * @size: 삽입하려는 패킷 크기 (현재 1 고정)
 * @return: 해당 서브넷 입력 버퍼에 여유가 있으면 true, 없으면 false
 *
 * n_subnets > 1이고 deviceID가 메모리 노드(>= n_shader)이면 REPLY_NET의 Has_Buffer_In(),
 * 그 외이면 REQ_NET의 Has_Buffer_In()을 호출한다.
 * update_counter=true로 호출하여 버퍼 포화 시 통계도 함께 갱신한다.
 * false 반환 시 상위 레벨에서 삽입을 미루고 back-pressure를 발동해야 한다.
 *
 * 호출 체인: icnt_has_buffer() → LocalInterconnect_has_buffer() → [이 함수] → xbar_router::Has_Buffer_In()
 */
bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  bool has_buffer = false;

  if ((n_subnets > 1) && deviceID >= n_shader)  // deviceID is memory node
    has_buffer = net[REPLY_NET]->Has_Buffer_In(deviceID, 1, true);
  /* [한국어] 메모리 노드(deviceID >= n_shader)가 보내는 응답: REPLY_NET 입력 버퍼 확인.
   * update_counter=true: 버퍼 가득 참 시 in_buffer_full 통계 자동 증가. */
  else
    has_buffer = net[REQ_NET]->Has_Buffer_In(deviceID, 1, true);
  /* [한국어] SM 노드(deviceID < n_shader)가 보내는 요청: REQ_NET 입력 버퍼 확인.
   * n_subnets == 1인 경우도 이 경로로 처리됨. */

  return has_buffer;  /* [한국어] 버퍼 여유 여부 반환 — false이면 상위에서 icnt_push 지연 */
}

/*
 * [한국어]
 * LocalInterconnect::DisplayStats - REQ_NET과 REPLY_NET의 성능 통계 출력
 *
 * @return: 없음 (void)
 *
 * 요청 네트워크(REQ_NET)와 응답 네트워크(REPLY_NET) 각각에 대해 다음 통계를 출력:
 * - injected_packets_num: 총 삽입 패킷 수
 * - cycles: 총 사이클 수
 * - injected_packets_per_cycle: 사이클당 처리량 (throughput)
 * - conflicts_per_cycle: 사이클당 출력 포트 경합률
 * - conflicts_per_cycle_util: 활성 사이클 기준 경합률
 * - Bank_Level_Parallelism: 활성 사이클당 전달 성공 패킷 수 (BLP)
 * - in/out_buffer_full_per_cycle: 사이클당 버퍼 포화 빈도
 * - in/out_buffer_avg_util: 버퍼 평균 활용도
 *
 * 시뮬레이션 완료 후 결과 리포트에 포함된다.
 *
 * 호출 체인: icnt_display_stats() → LocalInterconnect_display_stats() → [이 함수]
 */
void LocalInterconnect::DisplayStats() const {
  /* [한국어] REQ_NET(SM→메모리 방향) 통계 출력 블록 */
  printf("Req_Network_injected_packets_num = %lld\n",
         net[REQ_NET]->packets_num);
  /* [한국어] REQ_NET에 삽입된 총 패킷 수 — SM들이 보낸 메모리 요청의 총수 */
  printf("Req_Network_cycles = %lld\n", net[REQ_NET]->cycles);
  /* [한국어] REQ_NET의 총 시뮬레이션 사이클 수 */
  printf("Req_Network_injected_packets_per_cycle = %12.4f \n",
         (float)(net[REQ_NET]->packets_num) / (net[REQ_NET]->cycles));
  /* [한국어] REQ_NET 처리량 = 총 패킷 수 / 총 사이클 수 */
  printf("Req_Network_conflicts_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->conflicts) / (net[REQ_NET]->cycles));
  /* [한국어] REQ_NET의 총 사이클 기준 경합률 = 총 경합 / 총 사이클 */
  printf("Req_Network_conflicts_per_cycle_util = %12.4f\n",
         (float)(net[REQ_NET]->conflicts_util) / (net[REQ_NET]->cycles_util));
  /* [한국어] REQ_NET의 활성 사이클 기준 경합률 = 활성 사이클 경합 / 활성 사이클 수 */
  printf("Req_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REQ_NET]->reqs_util) / (net[REQ_NET]->cycles_util));
  /* [한국어] REQ_NET BLP(Bank-Level Parallelism) = 활성 사이클당 전달 성공 패킷 수 */
  printf("Req_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->in_buffer_full) / (net[REQ_NET]->cycles));
  /* [한국어] REQ_NET 입력 버퍼 포화 빈도 = 총 in_buffer_full / 총 사이클 */
  printf("Req_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->in_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_in_buffers));
  /* [한국어] REQ_NET 평균 입력 버퍼 활용도 = 누적 점유량 / (총 사이클 × 활성 입력 수) */
  printf("Req_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->out_buffer_full) / (net[REQ_NET]->cycles));
  /* [한국어] REQ_NET 출력 버퍼 포화 빈도 = 총 out_buffer_full / 총 사이클 */
  printf("Req_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->out_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_out_buffers));
  /* [한국어] REQ_NET 평균 출력 버퍼 활용도 = 누적 점유량 / (총 사이클 × 활성 출력 수) */

  printf("\n");  /* [한국어] REQ_NET과 REPLY_NET 통계 사이 빈 줄 구분 */

  /* [한국어] REPLY_NET(메모리→SM 방향) 통계 출력 블록 */
  printf("Reply_Network_injected_packets_num = %lld\n",
         net[REPLY_NET]->packets_num);
  /* [한국어] REPLY_NET에 삽입된 총 패킷 수 — 메모리 파티션들이 보낸 응답의 총수 */
  printf("Reply_Network_cycles = %lld\n", net[REPLY_NET]->cycles);
  /* [한국어] REPLY_NET의 총 시뮬레이션 사이클 수 */
  printf("Reply_Network_injected_packets_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->packets_num) / (net[REPLY_NET]->cycles));
  /* [한국어] REPLY_NET 처리량 */
  printf("Reply_Network_conflicts_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->conflicts) / (net[REPLY_NET]->cycles));
  /* [한국어] REPLY_NET의 총 사이클 기준 경합률 */
  printf(
      "Reply_Network_conflicts_per_cycle_util = %12.4f\n",
      (float)(net[REPLY_NET]->conflicts_util) / (net[REPLY_NET]->cycles_util));
  /* [한국어] REPLY_NET의 활성 사이클 기준 경합률 */
  printf("Reply_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REPLY_NET]->reqs_util) / (net[REPLY_NET]->cycles_util));
  /* [한국어] REPLY_NET BLP — 응답 네트워크의 활성 사이클당 전달 성공 패킷 수 */
  printf("Reply_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->in_buffer_full) / (net[REPLY_NET]->cycles));
  /* [한국어] REPLY_NET 입력 버퍼 포화 빈도 */
  printf("Reply_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->in_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_in_buffers));
  /* [한국어] REPLY_NET 평균 입력 버퍼 활용도 */
  printf("Reply_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->out_buffer_full) / (net[REPLY_NET]->cycles));
  /* [한국어] REPLY_NET 출력 버퍼 포화 빈도 */
  printf("Reply_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->out_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_out_buffers));
  /* [한국어] REPLY_NET 평균 출력 버퍼 활용도 */
}

/* [한국어] LocalInterconnect::DisplayOverallStats - 전체 누적 통계 출력 (현재 빈 구현).
 * 인터페이스 일관성을 위해 존재. 향후 여러 커널에 걸친 누적 통계 출력에 활용 예정. */
void LocalInterconnect::DisplayOverallStats() const {}

/* [한국어] LocalInterconnect::GetFlitSize - local_xbar의 플릿 크기(40바이트) 반환.
 * 상위 레벨에서 패킷을 플릿 단위로 환산할 때 분모로 사용.
 * local_xbar는 모든 패킷을 1플릿으로 처리하므로 이 값은 패킷당 1플릿을 의미. */
unsigned LocalInterconnect::GetFlitSize() const { return LOCAL_INCT_FLIT_SIZE; }

/* [한국어] LocalInterconnect::DisplayState - ICNT 내부 상태를 fp에 출력.
 * 현재 "Under implementation" 메시지만 출력하는 미완성 상태.
 * 디버그 목적의 상태 덤프 인터페이스로 예약된 슬롯. */
void LocalInterconnect::DisplayState(FILE* fp) const {
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
  /* [한국어] 현재 구현이 완료되지 않아 "구현 중" 메시지만 출력 */
}
