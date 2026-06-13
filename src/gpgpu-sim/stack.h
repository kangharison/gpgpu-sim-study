// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Ivan Sham
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
 * [한국어 설명] SIMT 스택 (warp 분기 처리용 주소 스택) 선언 (stack.h)
 *
 * === 파일의 역할 ===
 * GPU warp의 SIMT(Single Instruction Multiple Thread) 분기/재합류(reconvergence)
 * 처리를 위한 C 스타일 정수 스택 자료구조(Stack)와 관련 함수들을 선언한다.
 * Stack은 address_type(GPU 프로그램 카운터 주소 타입) 값을 저장하는 배열 기반
 * 고정 크기 스택으로 구현된다. GPGPU-Sim에서 warp가 조건 분기를 만날 때 재합류
 * 지점(reconvergence PC) 및 분기 대상 PC를 이 스택에 push하고, 분기가 끝나면
 * pop하여 SIMT 실행 마스크를 복원한다. 이 파일은 헤더 선언만 포함하며 구현은
 * stack.cc에 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * abstract_hardware_model.h의 simt_stack 클래스(C++ 클래스 기반의 더 완전한 SIMT
 * 스택 구현)와 별개로, 이 파일은 C 스타일의 단순한 주소 스택을 제공한다.
 * shader.cc의 warp 스케줄러 및 분기 처리 로직에서 재합류 PC 관리에 사용된다.
 * 사이클 루프(gpu-sim.cc::cycle())의 SM 실행 단계에서 분기 명령어를 처리할 때
 * 이 스택으로 분기 대상 주소를 추적한다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 사이클 루프 내에서 warp별로 접근됨.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: abstract_hardware_model.h (address_type 타입 정의)
 * - 이 파일에 의존하는 모듈: shader.cc (warp 분기 처리), stack.cc (구현 제공)
 * - 데이터 흐름: 분기 명령어 처리 시 재합류 PC → push_stack()으로 저장 →
 *   분기 블록 완료 후 pop_stack()으로 꺼내어 warp PC 복원
 * - 공유 자료구조: Stack — warp별로 독립적으로 존재하며 warp 스레드 간 공유 안 함
 *
 * === 주요 함수/구조체 요약 ===
 * - Stack: 배열 기반 고정 크기 스택 구조체 (v: 주소 배열, max_size: 최대 크기, top: 현재 크기)
 * - new_stack/free_stack: 동적 할당 및 해제
 * - push_stack/pop_stack/top_stack: 표준 스택 연산
 * - size_stack/full_stack/empty_stack: 상태 조회
 * - element_exist_stack: 값 존재 여부 선형 탐색
 * - reset_stack: top을 0으로 초기화 (메모리 해제 없이 재사용)
 */

#ifndef _MY_STACK_
#define _MY_STACK_

/* [한국어] abstract_hardware_model.h — address_type (GPU 프로그램 카운터 주소 타입,
 * 보통 unsigned int 또는 unsigned long) 정의 포함 */
#include "../abstract_hardware_model.h"

/*
 * [한국어] Stack - SIMT 분기 처리를 위한 배열 기반 고정 크기 정수 스택 구조체
 *
 * warp가 조건 분기 명령어(BRA, CALL 등)를 만날 때 재합류 PC(reconvergence PC)와
 * 분기 대상 PC를 저장하는 스택이다. 배열 v에 address_type 값을 순서대로 저장하고
 * top 인덱스로 현재 크기를 추적하는 표준 배열 스택 구현이다.
 */
typedef struct {
  address_type *v;
  /* [한국어] 스택 내용을 저장하는 동적 배열. new_stack()에서 calloc으로 할당된다.
   * 설정자: new_stack()에서 calloc으로 할당; push_stack()에서 v[top]에 값 저장.
   * 읽는 자: pop_stack()과 top_stack()에서 v[top-1]로 최상위 값 접근;
   *          element_exist_stack()에서 선형 탐색.
   * 값 범위: address_type 값 배열 (0으로 초기화된 max_size개 슬롯).
   *          유효 범위는 v[0] ~ v[top-1].
   * 동기화: warp별로 독립 스택 — 단일 스레드(사이클 루프)에서만 접근, 락 불필요. */

  int max_size;
  /* [한국어] 스택의 최대 크기 (슬롯 수). new_stack(size)의 인자로 결정된다.
   * 설정자: new_stack()에서 size 인자로 초기화. 이후 변경 불가.
   * 읽는 자: push_stack()의 assert(top < max_size); full_stack()에서 비교.
   * 값 범위: 1 이상의 양의 정수 (0이면 의미 없음).
   * 동기화: 읽기 전용 (초기화 이후 불변). */

  int top;
  /* [한국어] 현재 스택에 저장된 요소 수이자, 다음 push 시 사용할 배열 인덱스.
   * top == 0이면 빈 스택, top == max_size이면 가득 찬 상태이다.
   * 설정자: push_stack()에서 v[top] = val 후 top++; pop_stack()에서 top--;
   *         reset_stack()에서 0으로 초기화.
   * 읽는 자: pop_stack(), top_stack()에서 v[top-1]로 최상위 접근;
   *          size_stack(), full_stack(), empty_stack()에서 직접 반환/비교.
   * 값 범위: 0 이상 max_size 이하.
   * 동기화: warp별 독립 접근, 락 불필요. */
} Stack;

/*
 * [한국어]
 * push_stack - 스택에 address_type 값을 하나 삽입한다.
 * @S: 삽입할 대상 스택 포인터.
 * @val: 삽입할 주소 값 (재합류 PC 또는 분기 대상 PC).
 * 스택이 가득 차면 assert로 시뮬레이터 종료.
 */
void push_stack(Stack *S, address_type val);

/*
 * [한국어]
 * pop_stack - 스택에서 최상위 값을 꺼내 반환한다.
 * @S: 꺼낼 대상 스택 포인터.
 * @return: 꺼낸 address_type 값. 스택이 비어 있을 때 호출하면 top이 음수가 되어 미정의 동작.
 */
address_type pop_stack(Stack *S);

/*
 * [한국어]
 * top_stack - 스택에서 최상위 값을 꺼내지 않고 참조(peek)한다.
 * @S: 참조할 대상 스택 포인터.
 * @return: 최상위 address_type 값. 스택에 적어도 1개 이상의 요소가 있어야 한다 (assert).
 */
address_type top_stack(Stack *S);

/*
 * [한국어]
 * new_stack - 지정한 크기의 Stack을 동적 할당하여 초기화한 후 반환한다.
 * @size: 스택 최대 크기 (슬롯 수).
 * @return: 초기화된 Stack* 포인터. 사용 후 free_stack()으로 해제 필요.
 */
Stack *new_stack(int size);

/*
 * [한국어]
 * free_stack - new_stack()으로 할당된 Stack과 내부 배열을 해제한다.
 * @S: 해제할 Stack 포인터. 이후 S에 접근하면 미정의 동작.
 */
void free_stack(Stack *S);

/*
 * [한국어]
 * size_stack - 현재 스택에 저장된 요소 수를 반환한다.
 * @S: 크기를 조회할 스택 포인터.
 * @return: S->top (0 이상 max_size 이하).
 */
int size_stack(Stack *S);

/*
 * [한국어]
 * full_stack - 스택이 가득 찼는지 확인한다.
 * @S: 확인할 스택 포인터.
 * @return: top >= max_size이면 1(true), 그렇지 않으면 0(false).
 */
int full_stack(Stack *S);

/*
 * [한국어]
 * empty_stack - 스택이 비어 있는지 확인한다.
 * @S: 확인할 스택 포인터.
 * @return: top == 0이면 1(true), 그렇지 않으면 0(false).
 */
int empty_stack(Stack *S);

/*
 * [한국어]
 * element_exist_stack - 스택에 지정한 값이 존재하는지 선형 탐색한다.
 * @S: 탐색할 스택 포인터.
 * @value: 찾을 address_type 값.
 * @return: 존재하면 1(true), 없으면 0(false). O(n) 복잡도.
 */
int element_exist_stack(Stack *S, address_type value);

/*
 * [한국어]
 * reset_stack - 스택의 top을 0으로 초기화하여 비어 있는 상태로 만든다.
 * @S: 초기화할 스택 포인터.
 * 배열 메모리는 해제하지 않으므로 재사용 가능. 기존 데이터는 덮어씌워질 때까지 남아 있다.
 */
void reset_stack(Stack *S);
#endif  // _MY_STACK_
