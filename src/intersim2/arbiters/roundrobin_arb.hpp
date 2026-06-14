// $Id: roundrobin_arb.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] RoundRobinArbiter 선언 (roundrobin_arb.hpp)
 *
 * === 파일의 역할 ===
 * 라운드로빈 중재 알고리즘을 구현하는 RoundRobinArbiter를 선언한다.
 * _pointer 오프셋을 기준으로 입력들을 순환 탐색하여 가장 높은 우선순위(동점이면 _pointer에 가까운)
 * 요청을 선택한다. AddRequest()에서 Supersedes() 비교를 통해 O(1)로 최선 후보를 갱신하므로
 * Arbitrate()는 단순히 _best_input을 반환하면 된다. 이 덕분에 O(1) 중재가 가능하다.
 * SeparableAllocator의 입력/출력 중재자로 가장 널리 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("round_robin") → RoundRobinArbiter()
 * SeparableAllocator → RoundRobinArbiter (입력/출력 중재자)
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): _request, _size, _selected, _num_reqs, _highest_pri, _best_input
 *
 * === 주요 함수/구조체 요약 ===
 * - _pointer: 라운드로빈 시작 오프셋 (UpdateState()에서 _selected+1로 갱신)
 * - AddRequest(): Supersedes() 비교로 _best_input 즉시 갱신 (O(1))
 * - Arbitrate(): _best_input을 _selected로 설정 후 Arbiter::Arbitrate() 호출
 * - UpdateState(): _pointer = (_selected+1)%_size 갱신
 * - Supersedes(): 두 요청의 우선순위/위치를 비교하는 static inline 함수
 */

// ----------------------------------------------------------------------
//
//  RoundRobin: Round Robin Arbiter
//
// ----------------------------------------------------------------------

#ifndef _ROUNDROBIN_HPP_
#define _ROUNDROBIN_HPP_

#include "arbiter.hpp"

class RoundRobinArbiter : public Arbiter {

  // Priority pointer
  int  _pointer ;
  /* [한국어] 라운드로빈 탐색 시작 오프셋. _pointer 위치에 가까운 입력이 동점 시 우선순위를 가짐.
   * 설정자: UpdateState()에서 (_selected+1)%_size로 갱신; 생성자에서 0으로 초기화.
   * 읽는 자: AddRequest()의 Supersedes() 호출 시 공정성 기준으로 사용.
   * 사이클 간 유지되어 라운드로빈 공정성을 구현함. */

public:

  // Constructors
  /*
   * [한국어] RoundRobinArbiter 생성자 — _pointer=0, Arbiter 초기화
   * 호출 체인: Arbiter::NewArbiter("round_robin") → RoundRobinArbiter()
   */
  RoundRobinArbiter( Module *parent, const string &name, int size ) ;

  // Print priority matrix to standard output
  /*
   * [한국어] PrintState — 현재 _pointer 값 출력 (디버그용)
   */
  virtual void PrintState() const ;

  // Update priority matrix based on last aribtration result
  /*
   * [한국어] UpdateState — 마지막 중재 결과 이후 _pointer 전진
   * _selected != -1이면 _pointer = (_selected+1)%_size
   */
  virtual void UpdateState() ;

  // Arbitrate amongst requests. Returns winning input and
  // updates pointers to metadata when valid pointers are passed
  /*
   * [한국어] Arbitrate — _best_input을 _selected로 설정 후 Arbiter::Arbitrate() 호출
   * AddRequest()에서 이미 _best_input이 결정되어 있으므로 단순 위임 가능.
   */
  virtual int Arbitrate( int* id = 0, int* pri = 0) ;

  /*
   * [한국어] AddRequest — 요청 등록 + _best_input 즉시 갱신 (O(1))
   * Supersedes(input, pri, _best_input, _highest_pri, _pointer, _size)로 새 요청이 현재 최선보다
   * 우수한지 비교하여 _best_input/_highest_pri를 업데이트한다.
   */
  virtual void AddRequest( int input, int id, int pri ) ;

  /*
   * [한국어] Clear — _highest_pri=int_min, _best_input=-1 리셋 후 Arbiter::Clear() 호출
   */
  virtual void Clear();

  /*
   * [한국어] Supersedes — 두 요청 중 어느 것이 우선하는지 판정하는 static inline 헬퍼
   * @input1, pri1: 새로 들어온 요청
   * @input2, pri2: 현재 최선 후보
   * @offset: 라운드로빈 시작 위치 (_pointer)
   * @size: 중재자 크기
   * @return: pri1 > pri2이거나, pri1==pri2이고 input1이 offset 기준으로 input2보다 가까우면 true
   * 수식: (input - offset + size) % size — offset 기준 순환 거리. 거리가 작을수록 우선.
   */
  static inline bool Supersedes(int input1, int pri1, int input2, int pri2, int offset, int size)
  {
    // in a round-robin scheme with the given number of positions and current
    // offset, should a request at input1 with priority pri1 supersede a
    // request at input2 with priority pri2?
    return ((pri1 > pri2) ||
	    ((pri1 == pri2) &&
	     (((input1 - offset + size) % size) < ((input2 - offset + size) % size))));
    // [한국어] pri가 높으면 무조건 우선; 동점이면 offset 기준 거리가 짧은 쪽(더 가까운 쪽)을 선택
  }

} ;

#endif
