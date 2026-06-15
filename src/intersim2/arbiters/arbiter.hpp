// $Id: arbiter.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * [한국어 설명] Arbiter 기반 클래스 선언 (arbiter.hpp)
 *
 * === 파일의 역할 ===
 * BookSim2 NoC 시뮬레이터에서 사용하는 모든 중재자(arbiter)의 기반 클래스를 선언한다.
 * Arbiter는 여러 입력 포트 중 하나를 선택(중재)하는 역할을 한다. 이 추상 클래스를
 * RoundRobinArbiter(라운드로빈), MatrixArbiter(행렬 우선순위), TreeArbiter(트리 계층)가 상속한다.
 * SeparableAllocator에서 입력/출력 중재자로 사용되며, SelAlloc/iSLIP의 하위 컴포넌트이다.
 * GPU SM↔L2 트래픽의 VC/스위치 할당 시 각 포트에서 경합하는 플릿을 중재하는 데 쓰인다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SeparableAllocator → Arbiter (입력/출력 중재자)
 * TreeArbiter → Arbiter (그룹/전역 중재자)
 * Arbiter::NewArbiter() 팩토리로 생성.
 *
 * === 타 모듈과의 연결 ===
 * - module.hpp: Module 기반 클래스 상속 — 계층적 이름/디버그 지원
 * - roundrobin_arb, matrix_arb, tree_arb: 구체 구현 클래스들
 * - SeparableAllocator: _input_arb/_output_arb로 이 클래스의 인스턴스 보유
 * - BookSimConfig / gpgpusim.config: 다음 옵션이 이 중재자의 생성과 동작에 영향을 준다.
 *     * arb_type        — 중재 알고리즘 기본("round_robin"/"matrix"/"tree(...)")
 *     * vc_allocator    — VC 할당기("islip"/"prio" 등)가 남부 Arbiter를 생성
 *     * sw_allocator    — 스위치 할당기가 남부 Arbiter를 생성
 *     * spec_sw_allocator — 투기적 스위치 할당기("prio" 등)
 *     * alloc_iters     — 반복 할당 횟수(iSLIP 등에서 Arbiter Clear/Add/Arbitrate/Update 반복 수)
 *     * class_priority  — 클래스별 기본 우선순위(pri 값) 제공
 *
 * === 주요 함수/구조체 요약 ===
 * - entry_t: 요청 엔트리 (valid, id, pri)
 * - _request[]: 크기 _size의 요청 배열, 각 입력의 요청 상태
 * - AddRequest(): 특정 입력의 요청 등록
 * - Arbitrate(): _selected 입력을 반환 (id, pri 포인터 업데이트)
 * - UpdateState(): 중재 결과에 따라 우선순위 상태 갱신 (순수 가상)
 * - Clear(): 모든 요청 초기화
 * - NewArbiter(): round_robin/matrix/tree(...) 타입에 따른 팩토리 함수
 */

// ----------------------------------------------------------------------
//
//  Arbiter: Base class for Matrix and Round Robin Arbiter
//
// ----------------------------------------------------------------------

#ifndef _ARBITER_HPP_
#define _ARBITER_HPP_

#include <vector>

#include "module.hpp"

class Arbiter : public Module {

protected:

  typedef struct {
    bool valid ; /* [한국어] 이 입력에 요청이 등록되어 있는지 여부 */
    int id ;     /* [한국어] 요청의 식별자 (레이블) — 상위 알고리즘에서 사용 */
    int pri ;    /* [한국어] 요청의 우선순위 값 — 높을수록 먼저 선택 */
  } entry_t ;
  /* [한국어] 각 입력 포트의 요청 정보를 담는 구조체.
   * 설정자: Arbiter::AddRequest()에서 valid=true, id/pri 설정.
   * 읽는 자: Arbitrate()에서 우선순위 비교, Clear()에서 valid=false 초기화.
   * 값 범위: valid=true이면 id≥0, pri는 임의 정수. valid=false이면 id/pri 의미 없음. */

  vector<entry_t> _request ;
  /* [한국어] 크기 _size의 요청 배열. _request[i] = 입력 i의 요청 엔트리.
   * 설정자: 생성자에서 valid=false로 초기화; AddRequest()에서 유효 요청 설정.
   * 읽는 자: Arbitrate()에서 최적 입력 선택; Clear()에서 전체 초기화.
   * 동기화: 단일 사이클 내에서만 유효. 사이클마다 Clear() 호출 후 AddRequest() 재등록. */

  int  _size ;
  /* [한국어] 이 중재자가 처리하는 입력 포트 수. 생성자에서 설정되고 이후 불변.
   * 값 범위: 1 이상 정수. _request 배열의 크기와 동일. */

  int  _selected ;
  /* [한국어] 이번 사이클 Arbitrate()가 선택한 입력 포트 번호.
   * 설정자: 각 서브클래스 Arbitrate()에서 설정. Clear()에서 -1로 리셋.
   * 읽는 자: Arbiter::Arbitrate()에서 반환값으로 사용; LastWinner()로 외부에서 접근.
   * 값 범위: -1(선택 없음) 또는 0.._size-1. */

  int _highest_pri;
  /* [한국어] AddRequest() 과정에서 현재까지 발견된 최고 우선순위 값.
   * RoundRobinArbiter가 AddRequest()마다 갱신하여 최적 입력 추적에 사용.
   * 설정자: RoundRobinArbiter::AddRequest()에서 갱신; Clear()에서 int_min으로 리셋.
   * 값 범위: numeric_limits<int>::min() (초기/리셋) 또는 실제 우선순위 값. */

  int _best_input;
  /* [한국어] AddRequest() 과정에서 현재까지 발견된 최선 입력 포트 번호.
   * RoundRobinArbiter가 Supersedes() 기준으로 갱신. Arbitrate() 시 _selected로 복사.
   * 설정자: RoundRobinArbiter::AddRequest()에서 Supersedes() 조건 만족 시 갱신.
   * 값 범위: -1(없음) 또는 0.._size-1. */

public:
  int  _num_reqs ;
  /* [한국어] 현재 사이클에 등록된 요청 수. AddRequest() 호출 시마다 증가.
   * Clear()에서 0으로 리셋됨. 0이면 Arbitrate()는 -1 반환.
   * 설정자: Arbiter::AddRequest()에서 증가; Arbiter::Clear()에서 0으로 리셋.
   * 읽는 자: Clear()에서 진입 조건 판단; MatrixArbiter에서 요청 수 최적화 분기. */

  /*
   * [한국어] Arbiter 생성자 — 요청 배열 초기화
   * @parent: 부모 모듈 (계층 이름에 사용)
   * @name: 이 중재자의 이름
   * @size: 입력 포트 수 (_request 배열 크기)
   * 호출 체인: 서브클래스 생성자 → Arbiter()
   */
  Arbiter( Module *parent, const string &name, int size ) ;

  // Print priority matrix to standard output
  /*
   * [한국어] PrintState — 현재 우선순위 상태를 stdout에 출력 (디버그용, 순수 가상)
   */
  virtual void PrintState() const = 0 ;

  // Register request with arbiter
  /*
   * [한국어] AddRequest — 입력 input의 요청을 id/pri와 함께 등록
   * @input: 요청하는 입력 포트 번호 (0.._size-1)
   * @id: 요청 식별자 (레이블)
   * @pri: 요청 우선순위 (높을수록 우선)
   * _request[input].valid=true, id/pri 설정, _num_reqs 증가.
   */
  virtual void AddRequest( int input, int id, int pri ) ;

  // Update priority matrix based on last aribtration result
  /*
   * [한국어] UpdateState — 마지막 중재 결과(_selected)를 반영하여 우선순위 상태 갱신 (순수 가상)
   * RoundRobin: _pointer = (_selected+1)%_size
   * Matrix: _matrix[_selected][*] = 0, _matrix[*][_selected] = 1
   */
  virtual void UpdateState() = 0 ;

  // Arbitrate amongst requests. Returns winning input and
  // updates pointers to metadata when valid pointers are passed
  /*
   * [한국어] Arbitrate — _selected 입력을 반환; id/pri 포인터가 유효하면 해당 값도 채움
   * @id: (옵션) 선택된 입력의 식별자를 저장할 포인터
   * @pri: (옵션) 선택된 입력의 우선순위를 저장할 포인터
   * @return: _selected (선택된 입력, -1이면 요청 없음)
   * 서브클래스에서 _selected를 결정한 후 이 메서드를 super 호출.
   */
  virtual int Arbitrate( int* id = 0, int* pri = 0 ) ;

  /*
   * [한국어] Clear — 모든 요청 초기화 (_request[*].valid=false, _num_reqs=0, _selected=-1)
   * 사이클마다 Clear() → AddRequest() → Arbitrate() → UpdateState() 순서로 사용.
   */
  virtual void Clear();

  /*
   * [한국어] LastWinner — 이번 사이클 Arbitrate()가 선택한 입력 반환 (인라인)
   * TreeArbiter에서 그룹 중재자의 결과를 전역 중재자에 전달할 때 사용.
   */
  inline int LastWinner() const {
    return _selected;
  }

  /*
   * [한국어] NewArbiter — 중재자 타입 문자열에 따라 적절한 Arbiter 서브클래스를 생성하는 팩토리
   * @arb_type: "round_robin" / "matrix" / "tree(groups,sub_arb_type)"
   * @size: 입력 포트 수
   * @return: 동적 할당된 Arbiter 포인터 (호출자가 해제)
   * 호출 체인: SeparableAllocator 생성자 → NewArbiter()
   */
  static Arbiter *NewArbiter( Module *parent, const string &name,
			      const string &arb_type, int size );
} ;

#endif
