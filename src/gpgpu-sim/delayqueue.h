// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt
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
 * [한국어 설명] 파이프라인 레이턴시 시뮬레이션용 지연 큐 템플릿 (delayqueue.h)
 *
 * === 파일의 역할 ===
 * GPU 파이프라인의 고정 레이턴시를 시뮬레이션하기 위한 제네릭 FIFO 큐 템플릿
 * fifo_pipeline<T>를 정의한다. 핵심 아이디어는 최소 길이(min_len)를 지정하면
 * 큐가 항상 min_len개 슬롯을 유지하도록 빈 슬롯(NULL)을 자동 삽입한다는 점이다.
 * 이로써 push한 항목이 min_len 사이클 후에야 pop()으로 꺼내진다 — 즉, N 사이클의
 * 파이프라인 스테이지 지연을 자연스럽게 모델링한다. 최대 길이(max_len)를 지정하면
 * 큐 오버플로를 방지하여 파이프라인 역압(backpressure)도 표현할 수 있다.
 * 이 파일은 헤더 전용(header-only)으로, 모든 구현이 인라인 메서드로 포함된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * shader.cc의 SM 파이프라인 스테이지 간 연결(예: 발행 큐 → 실행 유닛, 메모리 응답
 * 큐 → 레지스터 라이트백 큐), mem_fetch.cc의 메모리 요청/응답 전달, dram.cc의 DRAM
 * 응답 버스 모델링 등 고정 레이턴시가 필요한 거의 모든 파이프라인 경계에서 사용된다.
 * 사이클 루프(gpu-sim.cc::cycle())가 각 파이프라인 스테이지를 순서대로 진행시킬 때
 * fifo_pipeline의 push/pop으로 스테이지 간 데이터가 전달된다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 사이클 루프 내에서 매 사이클 접근됨.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: statwrapper.h (통계 지원), gpu-misc.h (LOGB2 등 유틸),
 *   assert.h/stdio.h/stdlib.h (표준 라이브러리)
 * - 이 파일에 의존하는 모듈: shader.cc, mem_fetch.cc, dram.cc, gpu-cache.cc 등
 *   고정 레이턴시 파이프라인 스테이지가 필요한 모든 타이밍 모델 파일
 * - 데이터 흐름: T* 포인터(보통 mem_fetch*, warp_inst_t* 등)를 push → min_len 사이클
 *   후 pop()으로 꺼냄 → 다음 파이프라인 스테이지 처리 → 최종 완료
 * - 공유 자료구조: fifo_data<T> (단방향 연결 리스트 노드), fifo_pipeline<T> (큐 본체)
 *
 * === 주요 함수/구조체 요약 ===
 * - fifo_data<T>: 연결 리스트 노드 — m_data(페이로드), m_next(다음 노드 포인터)
 * - fifo_pipeline<T>::fifo_pipeline: 생성자 — min_len개의 NULL 슬롯을 사전 삽입하여 초기 지연 설정
 * - fifo_pipeline<T>::push: 데이터 삽입 — 꼬리에 새 노드 추가 또는 빈 꼬리 슬롯 재사용
 * - fifo_pipeline<T>::pop: 데이터 꺼내기 — 머리 노드 제거, min_len 유지를 위해 NULL 재삽입
 * - fifo_pipeline<T>::top: 꺼내지 않고 머리 데이터 참조 (peek)
 * - fifo_pipeline<T>::set_min_length: 동적으로 최소 길이 변경 — 파이프라인 레이턴시 재설정
 */

/* [한국어] assert.h, stdio.h, stdlib.h — assert(), printf(), malloc()/free() 사용을 위한 포함 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DELAYQUEUE_H
#define DELAYQUEUE_H

/* [한국어] statwrapper.h — 통계 수집 지원 헤더 (fifo_pipeline이 통계 출력 기능에 의존) */
#include "../statwrapper.h"
/* [한국어] gpu-misc.h — LOGB2, gs_min2 등 공통 유틸리티 매크로/함수 포함 */
#include "gpu-misc.h"

/*
 * [한국어] fifo_data<T> - fifo_pipeline의 단방향 연결 리스트 노드
 *
 * fifo_pipeline이 내부적으로 사용하는 연결 리스트의 개별 노드 구조체이다.
 * 큐의 각 슬롯이 이 구조체 하나에 대응하며, m_data가 NULL이면 "빈 슬롯"으로
 * 파이프라인 지연 패딩을 나타낸다.
 */
template <class T>
struct fifo_data {
  T* m_data;
  /* [한국어] 이 슬롯이 보관하는 실제 데이터 포인터.
   * 설정자: fifo_pipeline::push()에서 m_tail->m_data = data로 설정.
   * 읽는 자: fifo_pipeline::top()이 참조하고, pop()이 반환한 후 호출자가 사용.
   * 값 범위: 유효한 T* 포인터 또는 NULL(빈 슬롯 — 파이프라인 지연 패딩).
   *          NULL이 반환되면 해당 사이클에 처리할 데이터가 없음을 의미한다.
   * 동기화: 단일 스레드(사이클 루프)에서만 접근되므로 별도 락 불필요. */

  fifo_data* m_next;
  /* [한국어] 이 노드의 다음 노드를 가리키는 포인터 (FIFO 순서 유지).
   * 설정자: push()에서 새 노드를 꼬리에 연결할 때 이전 꼬리의 m_next에 설정.
   * 읽는 자: pop()이 m_head->m_next로 다음 머리를 결정할 때 참조.
   * 값 범위: 유효한 fifo_data<T>* 또는 NULL(현재 노드가 마지막 노드임을 의미).
   * 동기화: 단일 스레드 접근. */
};

/*
 * [한국어] fifo_pipeline<T> - N 사이클 지연을 자동 모델링하는 파이프라인 FIFO 큐 템플릿
 *
 * min_len 파라미터로 지정한 사이클 수만큼 데이터 전달이 지연되는 파이프라인 스테이지를
 * 모델링한다. 내부적으로 단방향 연결 리스트(fifo_data<T>)를 사용하며, 머리(m_head)에서
 * 꺼내고(pop) 꼬리(m_tail)에 삽입(push)하는 표준 FIFO 방식으로 동작한다.
 * min_len > 0이면 큐가 항상 min_len개 이상의 슬롯을 유지하여 지연을 보장한다.
 */
template <class T>
class fifo_pipeline {
 public:
  /*
   * [한국어]
   * fifo_pipeline - 파이프라인 큐 생성자
   *
   * @nm: 큐 이름 문자열 — print() 디버그 출력에 사용. 호출자가 정적/리터럴 문자열을 제공한다.
   * @minlen: 큐가 유지해야 하는 최소 슬롯 수 — 이 값이 파이프라인 지연(레이턴시) 사이클 수.
   *          예) minlen=4이면 push한 데이터는 최소 4 사이클 후에 pop 가능.
   * @maxlen: 큐의 최대 슬롯 수. 0이면 무제한(assert로 막힘). push 시 이 길이 초과 불가.
   *
   * 생성 시 min_len개의 NULL 슬롯을 미리 push하여 초기 지연을 설정한다.
   * 이 NULL 슬롯들이 파이프라인 지연 패딩 역할을 한다 — 데이터가 밀려 들어올 때
   * 이 슬롯들을 채우면서 자동으로 지연이 발생한다.
   *
   * 호출 체인: shader.cc::shader_core_ctx 초기화 → [fifo_pipeline 생성자]
   */
  fifo_pipeline(const char* nm, unsigned int minlen, unsigned int maxlen) {
    assert(maxlen); /* [한국어] maxlen이 0이면 무한 큐가 되어 메모리를 무제한 소비 — 사전 방지 */
    m_name = nm;       /* [한국어] 디버그 출력용 큐 이름 저장 */
    m_min_len = minlen; /* [한국어] 최소 슬롯 수(= 파이프라인 레이턴시 사이클 수) 설정 */
    m_max_len = maxlen; /* [한국어] 최대 슬롯 수 설정 — push 시 초과 방지를 위한 상한선 */
    m_length = 0;       /* [한국어] 현재 슬롯 수 0으로 초기화 */
    m_n_element = 0;    /* [한국어] 유효 데이터 수 0으로 초기화 (NULL 슬롯은 제외) */
    m_head = NULL;      /* [한국어] 연결 리스트 머리 포인터 초기화 */
    m_tail = NULL;      /* [한국어] 연결 리스트 꼬리 포인터 초기화 */
    /* [한국어] 초기 지연 패딩: min_len개의 NULL 슬롯을 미리 push하여 큐를 채운다.
     * 이 NULL 슬롯들이 파이프라인 초기 지연을 만든다 — 실제 데이터가 push되면
     * 이 NULL들이 먼저 pop되어야 하므로 자연스럽게 min_len 사이클 지연이 생긴다. */
    for (unsigned i = 0; i < m_min_len; i++) push(NULL);
  }

  /*
   * [한국어]
   * ~fifo_pipeline - 소멸자: 남아있는 모든 연결 리스트 노드를 순서대로 해제
   *
   * m_head부터 순회하며 delete로 각 fifo_data<T> 노드를 해제한다.
   * m_data가 가리키는 실제 데이터(T 객체)는 호출자가 관리하므로 여기서 해제하지 않는다.
   *
   * 호출 체인: GPU 시뮬레이션 종료 → shader_core_ctx 소멸 → [~fifo_pipeline]
   */
  ~fifo_pipeline() {
    /* [한국어] m_head가 NULL이 될 때까지 순회하며 모든 노드 해제 */
    while (m_head) {
      m_tail = m_head;         /* [한국어] 현재 머리를 임시 저장 (삭제할 노드) */
      m_head = m_head->m_next; /* [한국어] 머리를 다음 노드로 전진 */
      delete m_tail;           /* [한국어] 이전 머리 노드 해제 (m_data 자체는 해제 안 함) */
    }
  }

  /*
   * [한국어]
   * push - 큐의 꼬리에 데이터 포인터를 삽입한다.
   *
   * @data: 삽입할 T* 포인터. NULL이면 빈 슬롯(지연 패딩)으로 삽입된다.
   *        유효한 포인터이면 실제 파이프라인 데이터로 취급된다.
   *
   * 삽입 방식은 꼬리 슬롯 상태에 따라 두 가지로 나뉜다:
   * (1) 꼬리 슬롯이 이미 유효한 데이터를 가지거나 길이가 min_len 미만이면 새 노드를 생성하여 꼬리에 연결.
   * (2) 꼬리 슬롯이 비어 있고(NULL) 길이가 min_len 이상이면 기존 빈 꼬리 슬롯을 재사용 — 새 노드 생성 없음.
   * 이 방식으로 큐가 min_len 슬롯을 유지하면서 불필요한 메모리 할당을 줄인다.
   *
   * 호출 체인: shader.cc::파이프라인 스테이지 진행 → [push]
   *            fifo_pipeline 생성자 → [push(NULL)] (초기화)
   *            pop() → [push(NULL)] (min_len 유지)
   */
  void push(T* data) {
    assert(m_length < m_max_len); /* [한국어] 큐가 가득 찬 상태에서 push 시도 방지 — 역압(backpressure) 확인 */
    if (m_head) { /* [한국어] 큐에 이미 노드가 있는 경우 */
      if (m_tail->m_data || m_length < m_min_len) {
        /* [한국어] 꼬리가 유효 데이터를 갖고 있거나 아직 min_len에 못 미치는 경우:
         * 새 fifo_data 노드를 동적 할당하여 꼬리에 연결한다. */
        m_tail->m_next = new fifo_data<T>(); /* [한국어] 새 노드 할당 및 꼬리의 m_next에 연결 */
        m_tail = m_tail->m_next;             /* [한국어] 꼬리 포인터를 새 노드로 전진 */
        m_length++;                          /* [한국어] 전체 슬롯 수 증가 */
        m_n_element++;                       /* [한국어] 유효 요소 카운터 증가 (NULL 여부는 아래에서 조정) */
      }
      /* [한국어] 꼬리가 비어 있고(NULL) 길이가 min_len 이상인 경우: 기존 꼬리 슬롯 재사용 — 아무것도 안 함 */
    } else {
      /* [한국어] 큐가 완전히 비어 있는 경우: 첫 번째 노드를 생성하여 머리와 꼬리 모두 설정 */
      m_head = m_tail = new fifo_data<T>(); /* [한국어] 첫 노드 할당, 머리=꼬리=동일 노드 */
      m_length++;   /* [한국어] 슬롯 수 1로 증가 */
      m_n_element++; /* [한국어] 유효 요소 카운터 증가 */
    }
    m_tail->m_next = NULL; /* [한국어] 꼬리 노드의 next를 NULL로 — 연결 리스트 끝 표시 */
    m_tail->m_data = data; /* [한국어] 꼬리 슬롯에 데이터 저장 (NULL이면 빈 패딩 슬롯) */
  }

  /*
   * [한국어]
   * pop - 큐의 머리에서 데이터 포인터를 꺼내 반환한다.
   *
   * @return: 큐 머리의 T* 포인터. 큐가 비어 있으면 NULL 반환.
   *          유효 데이터(non-NULL)가 반환되면 해당 사이클에 처리할 항목이 있음을 의미.
   *          NULL이 반환되면 파이프라인이 현재 사이클에 처리할 데이터가 없음을 의미.
   *
   * 머리 노드를 제거한 후, min_len이 설정되어 있고 큐 길이가 min_len 미만이 되면
   * push(NULL)을 호출하여 빈 슬롯을 자동 보충한다 — 이로써 지연 보장이 유지된다.
   * m_n_element는 NULL 슬롯을 카운트하지 않으므로, 보충된 NULL 슬롯 수를 다시 차감한다.
   *
   * 호출 체인: shader.cc::파이프라인 스테이지 진행 → [pop] → 반환값으로 다음 스테이지 처리
   */
  T* pop() {
    fifo_data<T>* next; /* [한국어] 머리 노드 제거 후 새 머리가 될 다음 노드 임시 저장 */
    T* data;            /* [한국어] 반환할 데이터 포인터 */
    if (m_head) { /* [한국어] 큐에 노드가 있는 경우 */
      next = m_head->m_next; /* [한국어] 현재 머리의 다음 노드 저장 — 머리 제거 후 새 머리 */
      data = m_head->m_data; /* [한국어] 현재 머리의 데이터 포인터 추출 — 반환할 값 */
      if (m_head == m_tail) {
        /* [한국어] 노드가 하나뿐인 경우: 머리 제거 후 꼬리도 NULL로 설정해야 함 */
        assert(next == NULL); /* [한국어] 단일 노드이면 next는 반드시 NULL */
        m_tail = NULL;        /* [한국어] 꼬리 포인터 NULL로 — 빈 큐 상태 */
      }
      delete m_head; /* [한국어] 현재 머리 노드 해제 (데이터 포인터 자체는 해제 안 함) */
      m_head = next; /* [한국어] 머리 포인터를 다음 노드로 전진 */
      m_length--;    /* [한국어] 전체 슬롯 수 감소 */
      if (m_length == 0) {
        /* [한국어] 슬롯이 0개가 된 경우: 머리와 꼬리 모두 NULL이어야 함을 검증 */
        assert(m_head == NULL);
        m_tail = m_head; /* [한국어] 꼬리도 NULL로 설정 (이미 m_head=NULL이므로 동일 효과) */
      }
      m_n_element--; /* [한국어] 유효 요소 카운터 감소 */
      if (m_min_len && m_length < m_min_len) {
        /* [한국어] min_len이 설정된 상태에서 슬롯 수가 min_len 미만이 되었을 때:
         * NULL 슬롯을 push하여 최소 길이를 복원한다 — 다음 사이클에도 지연 보장을 유지 */
        push(NULL);
        m_n_element--;  // uncount NULL elements inserted to create delays
        /* [한국어] push(NULL)이 m_n_element를 증가시켰으므로, NULL은 유효 요소가 아니라
         * 지연 패딩이기 때문에 다시 감소시켜 카운터를 올바르게 유지한다 */
      }
    } else {
      /* [한국어] 큐가 비어 있는 경우: NULL 반환 — 이 사이클에 처리할 데이터 없음 */
      data = NULL;
    }
    return data; /* [한국어] 꺼낸 데이터 포인터 반환 (NULL이면 빈 슬롯이었음) */
  }

  /*
   * [한국어]
   * top - 큐의 머리 데이터를 꺼내지 않고 참조(peek)한다.
   *
   * @return: 머리 노드의 m_data 포인터. 큐가 비어 있으면 NULL.
   *          pop()과 달리 노드를 제거하지 않으므로 큐 상태가 변하지 않는다.
   *
   * 현재 사이클에 처리 가능한 항목이 있는지 확인할 때 pop() 전에 호출한다.
   * 반환값이 NULL이면 이 사이클에 처리할 데이터가 없음 (지연 패딩 중).
   *
   * 호출 체인: shader.cc::파이프라인 스테이지 확인 → [top] → non-NULL이면 pop()
   */
  T* top() const {
    if (m_head) {
      return m_head->m_data; /* [한국어] 머리 노드의 데이터 포인터 반환 (제거 없이) */
    } else {
      return NULL; /* [한국어] 큐가 비어 있으면 NULL 반환 */
    }
  }

  /*
   * [한국어]
   * set_min_length - 파이프라인 최소 길이(= 지연 사이클 수)를 동적으로 변경한다.
   *
   * @new_min_len: 새로운 최소 슬롯 수. 현재 m_min_len과 같으면 즉시 반환한다.
   *
   * 레이턴시가 동적으로 변하는 파이프라인 스테이지(예: 메모리 레이턴시 재설정)에
   * 사용된다. 새 값이 더 크면 부족한 슬롯만큼 NULL을 push하여 채운다.
   * 새 값이 더 작으면 꼬리에서 빈(NULL) 슬롯을 역방향으로 찾아 제거한다.
   * 꼬리 노드 제거 시 역방향 탐색이 필요하므로 O(n) 복잡도이다.
   *
   * 호출 체인: shader.cc 또는 gpu-sim.cc에서 레이턴시 재설정 → [set_min_length]
   */
  void set_min_length(unsigned int new_min_len) {
    if (new_min_len == m_min_len) return; /* [한국어] 변경 없으면 즉시 반환 — 불필요한 작업 방지 */

    if (new_min_len > m_min_len) {
      /* [한국어] 새 min_len이 더 큰 경우: 부족한 슬롯만큼 NULL을 push하여 지연 증가 */
      m_min_len = new_min_len; /* [한국어] 최소 길이 업데이트 */
      while (m_length < m_min_len) {
        /* [한국어] 현재 길이가 새 min_len에 도달할 때까지 NULL 슬롯 삽입 */
        push(NULL);
        m_n_element--;  // uncount NULL elements inserted to create delays
        /* [한국어] push(NULL)이 m_n_element를 증가시켰으므로, 지연 패딩은 유효 요소가 아니므로 차감 */
      }
    } else {
      // in this branch imply that the original min_len is larger then 0
      // ie. head != 0
      /* [한국어] 새 min_len이 더 작은 경우: 꼬리의 불필요한 NULL 슬롯을 제거하여 지연 감소 */
      assert(m_head); /* [한국어] 원래 min_len > 0이었으므로 반드시 노드가 있어야 함 */
      m_min_len = new_min_len; /* [한국어] 최소 길이 업데이트 */
      /* [한국어] 길이가 새 min_len을 초과하고 꼬리가 빈 슬롯인 동안 꼬리를 제거한다 */
      while ((m_length > m_min_len) && (m_tail->m_data == 0)) {
        fifo_data<T>* iter;
        /* [한국어] 꼬리 직전 노드를 찾기 위해 머리부터 순방향 탐색 — O(n) */
        iter = m_head;
        while (iter && (iter->m_next != m_tail)) iter = iter->m_next; /* [한국어] 꼬리 직전 노드까지 전진 */
        if (!iter) {
          // there is only one node, and that node is empty
          /* [한국어] 노드가 하나뿐이고 그 노드가 비어 있는 경우: pop()으로 제거 */
          assert(m_head->m_data == 0); /* [한국어] 단일 노드이고 비어 있어야 함을 검증 */
          pop(); /* [한국어] 단일 빈 노드 제거 */
        } else {
          // there are more than one node, and tail node is empty
          /* [한국어] 노드가 둘 이상이고 꼬리가 빈 경우: 꼬리를 직접 제거하고 새 꼬리 설정 */
          assert(iter->m_next == m_tail); /* [한국어] iter가 꼬리 직전 노드임을 검증 */
          delete m_tail;        /* [한국어] 기존 꼬리 노드 해제 */
          m_tail = iter;        /* [한국어] 꼬리를 이전 노드로 후퇴 */
          m_tail->m_next = 0;   /* [한국어] 새 꼬리의 next를 NULL로 — 연결 리스트 끝 표시 */
          m_length--;           /* [한국어] 슬롯 수 감소 */
        }
      }
    }
  }

  /*
   * [한국어] full - 큐가 가득 찼는지 확인한다.
   * @return: true이면 push 불가 상태 (역압/backpressure). false이면 push 가능.
   * m_max_len이 0이 아니고 현재 슬롯 수가 최대에 도달했을 때 true.
   */
  bool full() const { return (m_max_len && m_length >= m_max_len); }

  /*
   * [한국어] is_avilable_size - 지정한 size만큼 추가로 push 가능한지 확인한다.
   * @size: 확인할 추가 슬롯 수.
   * @return: true이면 size개 추가 삽입 시 최대 길이 초과 (공간 부족). false이면 가능.
   * 주의: 함수명 "avilable"은 "available"의 오탈자이나 원본 코드 그대로 유지.
   */
  bool is_avilable_size(unsigned size) const {
    return (m_max_len && m_length + size - 1 >= m_max_len); /* [한국어] 현재 길이 + size - 1이 max_len 이상이면 공간 부족 */
  }

  /*
   * [한국어] empty - 큐에 유효 데이터가 전혀 없는지 확인한다.
   * @return: m_head가 NULL이면 true (완전히 빈 상태). 빈 슬롯만 있어도 m_head != NULL이므로 false.
   * NULL 패딩 슬롯만 있는 경우는 empty()가 false를 반환하므로 주의.
   * 실제 유효 데이터 유무는 top()이 NULL인지로 판단한다.
   */
  bool empty() const { return m_head == NULL; }

  /* [한국어] get_n_element - 큐에 삽입된 유효 데이터 수(NULL 슬롯 제외) 반환 */
  unsigned get_n_element() const { return m_n_element; }

  /* [한국어] get_length - 현재 전체 슬롯 수(NULL 슬롯 포함) 반환 */
  unsigned get_length() const { return m_length; }

  /* [한국어] get_max_len - 설정된 최대 슬롯 수 반환 */
  unsigned get_max_len() const { return m_max_len; }

  /*
   * [한국어]
   * print - 큐의 현재 상태를 디버그 출력한다.
   *
   * 큐 이름, 현재 길이, 각 슬롯의 m_data 포인터 주소를 순서대로 출력한다.
   * NULL 슬롯은 "(nil)" 또는 "0x0"으로 출력되므로 지연 패딩 슬롯을 시각적으로 확인 가능.
   * 단일 셰이더 구성의 파이프라인 디버깅 시 사용.
   *
   * 호출 체인: 디버그/테스트 코드 → [print]
   */
  void print() const {
    fifo_data<T>* ddp = m_head; /* [한국어] 머리부터 순회 시작 */
    printf("%s(%d): ", m_name, m_length); /* [한국어] 큐 이름과 현재 슬롯 수 출력 */
    while (ddp) {
      printf("%p ", ddp->m_data); /* [한국어] 각 슬롯의 데이터 포인터 주소 출력 (NULL이면 "(nil)") */
      ddp = ddp->m_next;          /* [한국어] 다음 노드로 전진 */
    }
    printf("\n"); /* [한국어] 줄 바꿈으로 출력 종료 */
  }

 private:
  const char* m_name;
  /* [한국어] 큐 이름 문자열 포인터 — 디버그 출력(print())에 사용.
   * 설정자: 생성자에서 nm 인자로 설정. 수명 동안 변경되지 않는다.
   * 읽는 자: print() 메서드.
   * 값 범위: NULL 불가. 호출자가 제공한 정적/리터럴 문자열 포인터.
   * 동기화: 읽기 전용이므로 동기화 불필요. */

  unsigned int m_min_len;
  /* [한국어] 최소 슬롯 수 = 파이프라인 지연 사이클 수.
   * 설정자: 생성자에서 minlen으로 초기화; set_min_length()로 동적 변경 가능.
   * 읽는 자: push()에서 새 노드 생성 여부 결정; pop()에서 NULL 패딩 삽입 조건 확인.
   * 값 범위: 0 이상 m_max_len 이하. 0이면 지연 없음(즉시 pop 가능).
   * 동기화: 단일 스레드 접근. */

  unsigned int m_max_len;
  /* [한국어] 최대 슬롯 수 = 큐 용량 상한선.
   * 설정자: 생성자에서 maxlen으로 설정. 이후 변경 불가(const 아니지만 관례상 고정).
   * 읽는 자: push()에서 assert로 초과 방지; full()과 is_avilable_size()에서 참조.
   * 값 범위: 1 이상 (생성자에서 assert(maxlen)으로 0 금지).
   * 동기화: 단일 스레드 접근. */

  unsigned int m_length;
  /* [한국어] 현재 전체 슬롯 수 (NULL 패딩 슬롯 포함).
   * 설정자: push()에서 새 노드 생성 시 증가; pop()과 set_min_length()에서 노드 제거 시 감소.
   * 읽는 자: push()의 assert; full(); is_avilable_size(); get_length(); print().
   * 값 범위: m_min_len 이상 m_max_len 이하.
   * 동기화: 단일 스레드 접근. */

  unsigned int m_n_element;
  /* [한국어] 유효 데이터(non-NULL) 슬롯 수 — NULL 패딩 슬롯은 카운트하지 않음.
   * 설정자: push(non-NULL)에서 증가; push(NULL)의 m_n_element-- 보정으로 감소;
   *         pop()에서 감소; set_min_length()의 NULL 삽입 후 보정으로 감소.
   * 읽는 자: get_n_element()에서 반환.
   * 값 범위: 0 이상 m_length 이하.
   * 동기화: 단일 스레드 접근. */

  fifo_data<T>* m_head;
  /* [한국어] 연결 리스트의 머리 노드 포인터 — pop()이 이 노드에서 꺼낸다.
   * 설정자: 생성자에서 NULL로 초기화; push()에서 첫 노드 생성 시 설정;
   *         pop()에서 머리 제거 후 다음 노드로 전진.
   * 읽는 자: pop()에서 데이터 추출; top()에서 peek; empty()에서 NULL 여부 확인.
   * 값 범위: 유효한 fifo_data<T>* 또는 NULL(큐가 빈 경우).
   * 동기화: 단일 스레드 접근. */

  fifo_data<T>* m_tail;
  /* [한국어] 연결 리스트의 꼬리 노드 포인터 — push()가 이 노드 뒤에 삽입한다.
   * 설정자: 생성자에서 NULL로 초기화; push()에서 새 노드 생성 시 전진;
   *         pop()에서 단일 노드 제거 시 NULL로 설정;
   *         set_min_length()에서 꼬리 노드 제거 시 이전 노드로 후퇴.
   * 읽는 자: push()에서 꼬리 슬롯 상태 확인; set_min_length()에서 꼬리 빈 여부 확인.
   * 값 범위: 유효한 fifo_data<T>* 또는 NULL(큐가 빈 경우).
   * 동기화: 단일 스레드 접근. */
};

#endif
