// $Id: roundrobin_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] RoundRobinArbiter 구현 (roundrobin_arb.cpp)
 *
 * === 파일의 역할 ===
 * 라운드로빈 중재 알고리즘 구현. AddRequest() 호출마다 Supersedes() 비교로 최선 후보를 즉시 갱신하여
 * Arbitrate()를 O(1)로 수행할 수 있게 한다. _pointer가 사이클 간 유지되어 공정성을 보장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("round_robin") → RoundRobinArbiter
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): AddRequest(), Arbitrate(), Clear() super 호출
 * - BookSimConfig / gpgpusim.config:
 *     * arb_type = "round_robin"일 때 NewArbiter()가 이 클래스를 생성(기본값).
 *     * alloc_iters 횟수만큼 매 사이클 Arbitrate() 이후 UpdateState()가 반복되어
 *       _pointer가 전진하며 공정성이 유지된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - RoundRobinArbiter(): Arbiter 초기화 + _pointer=0
 * - AddRequest(): _best_input 갱신 후 Arbiter::AddRequest() 호출
 * - Arbitrate(): _selected=_best_input 후 Arbiter::Arbitrate() 위임
 * - UpdateState(): _pointer=(_selected+1)%_size
 * - Clear(): _highest_pri/best_input 리셋 후 Arbiter::Clear() 위임
 */

// ----------------------------------------------------------------------
//
//  RoundRobin: RoundRobin Arbiter
//
// ----------------------------------------------------------------------

#include "roundrobin_arb.hpp" // [한국어] RoundRobinArbiter 선언
#include <iostream>           // [한국어] PrintState()의 cout 사용
#include <limits>             // [한국어] numeric_limits<int>::min() — _highest_pri 리셋값

using namespace std ;

/*
 * [한국어] RoundRobinArbiter 생성자 — Arbiter 초기화 + _pointer=0
 * 호출 체인: Arbiter::NewArbiter("round_robin") → RoundRobinArbiter()
 */
RoundRobinArbiter::RoundRobinArbiter( Module *parent, const string &name,
				      int size )
  : Arbiter( parent, name, size ), // [한국어] _request 배열 할당, 기본 상태 초기화
    _pointer( 0 ) {                // [한국어] 라운드로빈 포인터를 0번 입력부터 시작
}

/*
 * [한국어] PrintState — 디버그용 _pointer 값 출력
 */
void RoundRobinArbiter::PrintState() const  {
  cout << "Round Robin Priority Pointer: " << endl ;
  cout << "  _pointer = " << _pointer << endl ; // [한국어] 현재 라운드로빈 오프셋 출력
}

/*
 * [한국어] UpdateState — 중재 결과 이후 _pointer 전진
 * Arbitrate() 후 UpdateState()를 호출하면 _selected+1 위치부터 다음 사이클 중재 시작.
 */
void RoundRobinArbiter::UpdateState() {
  // update priority matrix using last grant
  if ( _selected > -1 )               // [한국어] 선택된 입력이 있을 때만 포인터 갱신
    _pointer = ( _selected + 1 ) % _size ; // [한국어] 공정성: 마지막 승자 다음부터 탐색
}

/*
 * [한국어] RoundRobinArbiter::AddRequest — 요청 등록 + _best_input 즉시 갱신
 * 중복 요청 시 더 높은 우선순위로 요청이 갱신되는 경우에만 _best_input을 비교한다.
 * Supersedes()로 새 요청이 현재 최선보다 우선하면 _best_input을 갱신한다.
 * 이 방식으로 O(1) Arbitrate()가 가능하다.
 */
void RoundRobinArbiter::AddRequest( int input, int id, int pri )
{
  if(!_request[input].valid || (_request[input].pri < pri)) {
    // [한국어] 이 입력에 요청이 없거나, 새 요청이 기존보다 더 높은 우선순위인 경우에만 후보 비교
    if((_num_reqs == 0) ||                                               // [한국어] 첫 번째 요청이면 무조건 최선
       Supersedes(input, pri, _best_input, _highest_pri, _pointer,_size )) { // [한국어] 현재 최선보다 우월하면
      _highest_pri = pri;   // [한국어] 최고 우선순위 갱신
      _best_input = input;  // [한국어] 최선 입력 갱신
    }
  }
  Arbiter::AddRequest(input, id, pri); // [한국어] _request[input] 설정 + _num_reqs 증가 (super 호출)
}

/*
 * [한국어] RoundRobinArbiter::Arbitrate — _best_input을 _selected로 설정 후 위임
 * AddRequest()에서 이미 _best_input이 결정되어 있으므로 단순 복사 후 super 호출.
 */
int RoundRobinArbiter::Arbitrate( int* id, int* pri ) {

  _selected = _best_input; // [한국어] AddRequest()에서 선정된 최선 입력을 최종 선택으로 확정

  return Arbiter::Arbitrate(id, pri); // [한국어] id/pri 포인터 채우기 + _selected 반환 (super 위임)
}

/*
 * [한국어] RoundRobinArbiter::Clear — _highest_pri/_best_input 리셋 후 Arbiter::Clear() 호출
 * 사이클 간 상태 정리. _pointer는 리셋하지 않음 — 사이클 간 유지되어야 공정성 보장.
 */
void RoundRobinArbiter::Clear()
{
  _highest_pri = numeric_limits<int>::min(); // [한국어] 최고 우선순위 리셋 (최솟값으로)
  _best_input = -1;                          // [한국어] 최선 입력 리셋
  Arbiter::Clear();                          // [한국어] _request[].valid=false, _num_reqs=0, _selected=-1
}
