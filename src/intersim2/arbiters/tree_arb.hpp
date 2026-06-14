// $Id: tree_arb.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] TreeArbiter 선언 (tree_arb.hpp)
 *
 * === 파일의 역할 ===
 * 2단계 계층 트리 중재자를 선언한다. 입력을 groups개 그룹으로 나누고,
 * 각 그룹에서 그룹 중재자(_group_arbiters[i])가 로컬 승자를 뽑은 뒤,
 * 전역 중재자(_global_arbiter)가 groups개 로컬 승자 중 하나를 최종 선택한다.
 * 이 계층 구조 덕분에 대규모 입력 수에서도 중재 지연을 줄일 수 있다.
 * 그룹/전역 중재자 모두 Arbiter::NewArbiter()로 생성되며 타입은 설정으로 결정된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("tree(groups,arb_type)") → TreeArbiter()
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): _request, _size, _selected, _num_reqs
 * - Arbiter::NewArbiter(): 그룹/전역 중재자 동적 생성
 *
 * === 주요 함수/구조체 요약 ===
 * - _group_size: size/groups — 각 그룹의 입력 수
 * - _group_arbiters[i]: 그룹 i의 로컬 중재자 (size=_group_size)
 * - _global_arbiter: 그룹 승자들 간 전역 중재자 (size=groups)
 * - _group_reqs[i]: 그룹 i의 요청 수 (AddRequest()에서 증가, Clear()에서 리셋)
 * - Arbitrate(): 그룹 중재 → 전역 중재 → _selected = group*_group_size + group_sel
 * - UpdateState(): 승리한 그룹의 그룹 중재자 + 전역 중재자 상태 갱신
 */

// ----------------------------------------------------------------------
//
//  TreeArbiter
//
// ----------------------------------------------------------------------

#ifndef _TREE_ARB_HPP_
#define _TREE_ARB_HPP_

#include "arbiter.hpp"

class TreeArbiter : public Arbiter {

  int  _group_size ;
  /* [한국어] 각 그룹의 입력 포트 수. = _size / groups.
   * 입력 번호 input은 그룹 (input/_group_size)의 (input%_group_size)번째 포트. */

  vector<Arbiter *> _group_arbiters;
  /* [한국어] groups개 그룹 중재자 배열. _group_arbiters[i]는 그룹 i의 로컬 중재자.
   * 설정자: 생성자에서 Arbiter::NewArbiter()로 생성.
   * 읽는 자: AddRequest()에서 그룹 중재자에 요청 등록; Arbitrate()에서 그룹 중재 수행.
   * 동기화: 소멸자에서 delete로 해제. */

  Arbiter * _global_arbiter;
  /* [한국어] 전역 중재자. 각 그룹의 로컬 승자(그룹 인덱스)를 입력으로 받아 최종 승자 그룹 결정.
   * 크기 = groups (그룹 수). */

  vector<int> _group_reqs;
  /* [한국어] 그룹별 요청 수. _group_reqs[i] = 그룹 i에 AddRequest()된 횟수.
   * 설정자: AddRequest()에서 ++; Clear()에서 0으로 리셋.
   * 읽는 자: Arbitrate()에서 요청이 있는 그룹만 그룹 중재 수행. */

public:

  // Constructors
  /*
   * [한국어] TreeArbiter 생성자 — 그룹/전역 중재자 생성
   * @size: 전체 입력 포트 수 (size%groups==0이어야 함)
   * @groups: 그룹 수
   * @arb_type: 그룹/전역 중재자 타입 ("round_robin" / "matrix")
   * 호출 체인: Arbiter::NewArbiter("tree(groups,arb_type)") → TreeArbiter()
   */
  TreeArbiter( Module *parent, const string &name, int size, int groups, const string & arb_type ) ;

  /*
   * [한국어] ~TreeArbiter — _group_arbiters 및 _global_arbiter 메모리 해제
   */
  ~TreeArbiter();

  // Print priority matrix to standard output
  /*
   * [한국어] PrintState — 각 그룹 중재자와 전역 중재자의 상태를 출력 (디버그용)
   */
  virtual void PrintState() const ;

  // Update priority matrix based on last aribtration result
  /*
   * [한국어] UpdateState — 마지막 승리 그룹의 그룹 중재자와 전역 중재자 상태 갱신
   */
  virtual void UpdateState() ;

  // Arbitrate amongst requests. Returns winning input and
  // updates pointers to metadata when valid pointers are passed
  /*
   * [한국어] Arbitrate — 2단계 트리 중재: 그룹 → 전역 → _selected 결정
   * 각 그룹에서 로컬 승자를 뽑고, 전역 중재자가 그룹 중 최종 승자를 결정한다.
   * _selected = group * _group_size + group_sel
   */
  virtual int Arbitrate( int* id = 0, int* pri = 0) ;

  /*
   * [한국어] AddRequest — 입력을 그룹에 배정하여 그룹 중재자에 등록 + _group_reqs 증가
   */
  virtual void AddRequest( int input, int id, int pri ) ;

  /*
   * [한국어] Clear — 요청이 있는 경우: 모든 그룹 중재자 + 전역 중재자 초기화, _group_reqs 리셋
   */
  virtual void Clear();

} ;

#endif
