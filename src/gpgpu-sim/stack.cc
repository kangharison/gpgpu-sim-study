// Copyright (c) 2009-2011, Tor M. Aamodt,  Ali Bakhoda, Ivan Sham,
// Wilson W.L. Fung
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
 * [한국어 설명] SIMT 스택 구현 (stack.cc)
 *
 * === 파일의 역할 ===
 * stack.h에 선언된 Stack 자료구조와 관련 함수들의 구현을 제공한다.
 * Stack은 GPU warp의 SIMT 분기/재합류 처리에 사용되는 address_type 배열 기반
 * 고정 크기 스택이다. 분기 명령어 처리 시 재합류 PC(reconvergence program counter)와
 * 분기 대상 PC를 push하고, 분기가 끝나면 pop하여 warp의 PC를 복원한다.
 * C 스타일의 단순한 구현으로 배열과 top 인덱스만으로 스택을 관리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * shader.cc의 SM 파이프라인에서 warp 분기 처리 단계에 사용된다. GPGPU-Sim의
 * SIMT 실행 모델에서 warp 내 스레드들이 서로 다른 분기 경로를 택할 때(warp
 * divergence), 재합류 지점(reconvergence point)의 PC를 이 스택에 저장하고
 * 분기가 완료되면 꺼내어 모든 스레드가 재합류하도록 한다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 사이클 루프 내 warp 스케줄링 단계에서 접근됨.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: stack.h (Stack 구조체 및 함수 선언),
 *   assert.h (push_stack의 오버플로 방지), stdlib.h (malloc/calloc/free)
 * - 이 파일에 의존하는 모듈: shader.cc (warp 분기 처리 로직에서 호출)
 * - 데이터 흐름: shader.cc가 분기 명령어 처리 → push_stack()으로 재합류 PC 저장 →
 *   분기 블록 완료 → pop_stack()으로 PC 복원 → warp 재합류
 *
 * === 주요 함수/구조체 요약 ===
 * - new_stack(size): malloc+calloc으로 Stack 할당 및 초기화
 * - push_stack(S, val): S->v[S->top++] = val (오버플로 assert 포함)
 * - pop_stack(S): return S->v[--S->top] (언더플로 보호 없음 — 호출자 책임)
 * - top_stack(S): return S->v[S->top-1] (빈 스택 assert 포함)
 * - element_exist_stack(S, value): O(n) 선형 탐색으로 값 존재 여부 반환
 * - reset_stack(S): S->top = 0으로 스택 내용 무효화 (메모리 해제 없음)
 */

/* [한국어] stack.h — Stack 구조체 및 모든 스택 함수 선언 포함 */
#include "stack.h"

/* [한국어] assert.h — push_stack의 스택 오버플로 방지 및 top_stack의 빈 스택 검사에 사용 */
#include <assert.h>
/* [한국어] stdlib.h — new_stack의 malloc/calloc, free_stack의 free 사용 */
#include <stdlib.h>

/*
 * [한국어]
 * push_stack - 스택에 address_type 값을 하나 삽입한다.
 *
 * @S: 삽입할 대상 스택 포인터. new_stack()으로 할당된 유효한 포인터여야 한다.
 * @val: 삽입할 주소 값. 보통 재합류 PC(reconvergence PC) 또는 분기 대상 PC이다.
 *
 * S->top이 S->max_size에 도달하면 assert로 즉시 시뮬레이터를 종료한다 — 스택
 * 오버플로는 SIMT 스택 깊이 설정 오류를 나타내므로 조용히 처리하지 않는다.
 * 삽입 후 top을 증가시켜 다음 push를 위한 슬롯을 준비한다.
 *
 * 호출 체인: shader.cc::warp 분기 처리 → [push_stack(S, reconvergence_pc)]
 */
void push_stack(Stack *S, address_type val) {
  assert(S->top < S->max_size); /* [한국어] 스택 오버플로 사전 방지 — top이 max_size에 도달하면 assert로 종료 */
  S->v[S->top] = val;           /* [한국어] 현재 top 위치에 값 저장 — v[top]이 새 최상위 슬롯 */
  (S->top)++;                   /* [한국어] top 증가 — 다음 push를 위한 위치를 한 칸 올림 */
}

/*
 * [한국어]
 * pop_stack - 스택에서 최상위 값을 꺼내 반환한다.
 *
 * @S: 꺼낼 대상 스택 포인터.
 * @return: 꺼낸 address_type 값 (재합류 PC 또는 분기 대상 PC).
 *
 * top을 먼저 감소시킨 후 v[top]을 반환한다 (post-decrement 아닌 pre-decrement 방식).
 * 주의: 스택이 비어 있을 때(top==0) 호출하면 top이 음수(-1)로 언더플로가 발생하고
 * v[-1] 접근으로 미정의 동작이 된다 — 호출자가 empty_stack()으로 사전 확인해야 한다.
 *
 * 호출 체인: shader.cc::warp 재합류 처리 → [pop_stack(S)]
 */
address_type pop_stack(Stack *S) {
  (S->top)--;          /* [한국어] top 감소 — 최상위 슬롯의 인덱스를 top-1로 만듦 */
  return (S->v[S->top]); /* [한국어] 감소된 top 위치의 값 반환 — 이제 이 슬롯은 유효하지 않음 */
}

/*
 * [한국어]
 * top_stack - 스택에서 최상위 값을 꺼내지 않고 참조(peek)한다.
 *
 * @S: 참조할 대상 스택 포인터.
 * @return: v[top-1]의 address_type 값. pop 없이 현재 최상위를 확인할 때 사용.
 *
 * top >= 1이어야 한다는 것을 assert로 검사한다 — 빈 스택에서 호출하면 종료.
 * pop()과 달리 S->top을 변경하지 않으므로 큐 상태가 변하지 않는다.
 *
 * 호출 체인: shader.cc에서 분기 처리 전 재합류 PC 확인 → [top_stack(S)]
 */
address_type top_stack(Stack *S) {
  assert(S->top >= 1);        /* [한국어] 빈 스택 접근 방지 — top이 0이면 최상위 없음 */
  return (S->v[S->top - 1]);  /* [한국어] top-1 위치의 값 반환 — top은 변경하지 않음(peek) */
}

/*
 * [한국어]
 * new_stack - 지정한 크기의 Stack을 동적 할당하여 초기화한 후 반환한다.
 *
 * @size: 스택 최대 크기 (슬롯 수). 이 값이 max_size로 설정된다.
 * @return: 초기화된 Stack* 포인터. 사용 후 반드시 free_stack()으로 해제해야 한다.
 *
 * Stack 구조체 자체를 malloc으로, 내부 배열 v를 calloc으로 할당한다. calloc을 사용하므로
 * v의 모든 슬롯이 0으로 초기화된다. top은 0으로 설정되어 비어 있는 상태로 시작한다.
 *
 * 호출 체인: shader.cc::shader_core_ctx 초기화 → [new_stack(max_warp_depth)]
 */
Stack *new_stack(int size) {
  Stack *S;                                                  /* [한국어] 할당할 Stack 포인터 선언 */
  S = (Stack *)malloc(sizeof(Stack));                        /* [한국어] Stack 구조체 메모리 할당 (sizeof(Stack) = v 포인터 + max_size + top) */
  S->max_size = size;                                        /* [한국어] 최대 크기 설정 — push_stack의 오버플로 검사에 사용 */
  S->top = 0;                                                /* [한국어] top을 0으로 초기화 — 비어 있는 스택 상태 */
  S->v = (address_type *)calloc(size, sizeof(address_type)); /* [한국어] address_type 배열 할당 + 0으로 초기화 (calloc) */
  return S; /* [한국어] 초기화된 Stack 포인터 반환 — 호출자가 free_stack()으로 해제 책임 */
}

/*
 * [한국어]
 * free_stack - new_stack()으로 할당된 Stack과 내부 배열을 해제한다.
 *
 * @S: 해제할 Stack 포인터. new_stack()으로 할당된 유효한 포인터여야 한다.
 *     이후 S에 접근하면 미정의 동작(use-after-free).
 *
 * 먼저 내부 배열 v를 free하고, 그 다음 Stack 구조체 자체를 free한다.
 * 순서가 중요하다 — S 먼저 해제하면 S->v 접근이 미정의 동작이 된다.
 *
 * 호출 체인: shader.cc::shader_core_ctx 소멸 → [free_stack(S)]
 */
void free_stack(Stack *S) {
  free(S->v); /* [한국어] 내부 주소 배열 v 먼저 해제 — Stack 구조체보다 먼저 해제해야 안전 */
  free(S);    /* [한국어] Stack 구조체 자체 해제 */
}

/*
 * [한국어]
 * size_stack - 현재 스택에 저장된 요소 수를 반환한다.
 * @S: 크기를 조회할 스택 포인터.
 * @return: S->top — 유효 요소 수 (0 이상 max_size 이하).
 */
int size_stack(Stack *S) { return S->top; /* [한국어] top이 곧 현재 요소 수 — 0이면 빈 스택 */ }

/*
 * [한국어]
 * full_stack - 스택이 가득 찼는지 확인한다.
 * @S: 확인할 스택 포인터.
 * @return: top >= max_size이면 1(true, 가득 참), 그렇지 않으면 0(false).
 *          push_stack() 호출 전에 확인하여 오버플로를 방지하는 데 사용한다.
 */
int full_stack(Stack *S) { return S->top >= S->max_size; /* [한국어] top이 max_size에 도달하면 가득 찬 상태 */ }

/*
 * [한국어]
 * empty_stack - 스택이 비어 있는지 확인한다.
 * @S: 확인할 스택 포인터.
 * @return: top == 0이면 1(true, 비어 있음), 그렇지 않으면 0(false).
 *          pop_stack() 또는 top_stack() 호출 전에 확인하여 언더플로를 방지한다.
 */
int empty_stack(Stack *S) { return S->top == 0; /* [한국어] top이 0이면 요소가 없는 빈 스택 */ }

/*
 * [한국어]
 * element_exist_stack - 스택에 지정한 값이 존재하는지 선형 탐색한다.
 *
 * @S: 탐색할 스택 포인터.
 * @value: 찾을 address_type 값 (보통 PC 주소).
 * @return: 값이 존재하면 1(true), 없으면 0(false).
 *
 * 스택의 모든 유효 슬롯(v[0] ~ v[top-1])을 순서대로 비교한다. O(n) 선형 복잡도이다.
 * SIMT 스택은 깊이가 매우 얕으므로(최대 중첩 분기 수) 성능 영향은 무시할 수 있다.
 *
 * 호출 체인: shader.cc에서 재합류 PC 중복 여부 확인 → [element_exist_stack(S, pc)]
 */
int element_exist_stack(Stack *S, address_type value) {
  int i;                          /* [한국어] 탐색 인덱스 */
  for (i = 0; i < S->top; ++i) { /* [한국어] 유효 슬롯(v[0]~v[top-1]) 순서대로 탐색 */
    if (value == S->v[i]) {       /* [한국어] 현재 슬롯의 값이 찾는 값과 일치하는지 확인 */
      return 1;                   /* [한국어] 발견 — 즉시 1 반환 (early exit) */
    }
  }
  return 0; /* [한국어] 모든 슬롯 탐색 후 미발견 — 0 반환 */
}

/*
 * [한국어]
 * reset_stack - 스택의 top을 0으로 초기화하여 비어 있는 상태로 만든다.
 * @S: 초기화할 스택 포인터.
 *
 * top만 0으로 설정하고 내부 배열 v는 그대로 둔다 — 기존 데이터는 유효하지 않게 되지만
 * 메모리는 해제되지 않는다. free_stack()이 필요 없으므로 같은 Stack을 재사용할 때 호출한다.
 * 사이클 루프에서 warp를 재시작하거나 커널 경계에서 스택을 초기화할 때 사용한다.
 *
 * 호출 체인: shader.cc에서 warp 재초기화 → [reset_stack(S)]
 */
void reset_stack(Stack *S) { S->top = 0; /* [한국어] top을 0으로 설정 — 모든 슬롯을 무효화 (배열 해제 없음) */ }
