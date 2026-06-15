// $Id: prio_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] PriorityArbiter 구현 (prio_arb.cpp)
 *
 * === 파일의 역할 ===
 * PriorityArbiter의 모든 메서드를 구현한다. _requests 리스트를 in 오름차순으로 유지하며,
 * Arbitrate()에서 _rr_ptr 기준 순환 탐색으로 최고 우선순위 입력을 선택한다.
 * SelAlloc 등에서 각 포트의 우선순위 기반 중재에 활용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SelAlloc → PriorityArbiter::Arbitrate()
 *
 * === 타 모듈과의 연결 ===
 * - Module (상속): 계층적 이름 지원
 * - BookSimConfig / gpgpusim.config:
 *     * spec_sw_allocator = "prio"일 때 투기적 스위치 할당기가 이 중재자를 사용(기본값).
 *     * vc_allocator/sw_allocator가 "prio"를 지정하면 해당 할당기도 PriorityArbiter를 사용.
 *     * class_priority 값이 요청의 pri로 전달되어 클래스별 우선순위 차별에 활용.
 *     * alloc_iters 횟수만큼 매 사이클 Arbitrate()가 반복되어 _rr_ptr이 갱신.
 *
 * === 주요 함수/구조체 요약 ===
 * - PriorityArbiter(): _rr_ptr=0, _inputs 설정
 * - AddRequest(): in 오름차순 삽입, 중복 시 더 높은 우선순위만 유지
 * - RemoveRequest(): in으로 검색하여 제거
 * - Arbitrate(): _rr_ptr 기준 순환 탐색으로 최고 우선순위 선택, _rr_ptr 갱신
 * - Update(): _rr_ptr 단순 전진
 */

#include "booksim.hpp" // [한국어] BookSim 공통 헤더 (using namespace std 등)
#include <cassert>     // [한국어] assert() 사용

#include "prio_arb.hpp" // [한국어] PriorityArbiter 선언

/*
 * [한국어] PriorityArbiter 생성자 — _rr_ptr=0, _inputs 설정
 * 호출 체인: SelAlloc() → PriorityArbiter()
 */
PriorityArbiter::PriorityArbiter( const Configuration &config,
				  Module *parent, const string& name,
				  int inputs )
: Module( parent, name ),  // [한국어] Module 계층 등록
  _rr_ptr(0),              // [한국어] 라운드로빈 포인터 0으로 초기화
  _inputs( inputs )        // [한국어] 입력 포트 수 저장
{

}

/*
 * [한국어] PriorityArbiter::Clear — 요청 리스트 전체 삭제
 */
void PriorityArbiter::Clear( )
{
  _requests.clear( ); // [한국어] 리스트의 모든 sRequest 엔트리 제거
}

/*
 * [한국어] PriorityArbiter::AddRequest — in 오름차순 위치에 요청 삽입
 * @in: 입력 포트 번호
 * @label: 요청 식별자
 * @pri: 우선순위
 * 동일 in에 이미 요청이 있으면 더 높은 우선순위의 요청으로 교체한다.
 * (새 요청이 낮은 우선순위이면 무시)
 */
void PriorityArbiter::AddRequest( int in, int label, int pri )
{
  sRequest r;                          // [한국어] 새 요청 엔트리
  list<sRequest>::iterator insert_point; // [한국어] 삽입 위치 이터레이터

  r.in = in; r.label = label; r.pri = pri; // [한국어] 요청 정보 설정

  insert_point = _requests.begin( ); // [한국어] 리스트 처음부터 탐색
  while( ( insert_point != _requests.end( ) ) &&
	 ( insert_point->in < in ) ) { // [한국어] in 오름차순 삽입 위치 탐색
    insert_point++;
  }

  bool del = false; // [한국어] 기존 요청 삭제 여부
  bool add = true;  // [한국어] 새 요청 삽입 여부

  // For consistant behavior, delete the existing request
  // if it is for the same input and has a higher priority
  // [한국어] 동일 in의 기존 요청이 있는지 확인
  if ( ( insert_point != _requests.end( ) ) &&
       ( insert_point->in == in ) ) {
    if ( insert_point->pri < pri ) { // [한국어] 기존 우선순위가 더 낮으면 교체
      del = true;                    // [한국어] 기존 요청 삭제 표시
    } else {
      add = false;                   // [한국어] 기존 요청이 더 높은 우선순위 — 새 요청 무시
    }
  }

  if ( add ) {
    _requests.insert( insert_point, r ); // [한국어] in 오름차순 위치에 새 요청 삽입
  }

  if ( del ) {
    _requests.erase( insert_point ); // [한국어] 낮은 우선순위의 기존 요청 제거
  }
}

/*
 * [한국어] PriorityArbiter::RemoveRequest — in 번호로 검색하여 제거
 * @in: 제거할 요청의 입력 포트 번호
 */
void PriorityArbiter::RemoveRequest( int in, int label )
{
  list<sRequest>::iterator erase_point; // [한국어] 제거 위치 이터레이터

  erase_point = _requests.begin( );     // [한국어] 리스트 처음부터 검색
  while( ( erase_point != _requests.end( ) ) &&
	 ( erase_point->in < in ) ) {   // [한국어] in 번호에 해당하는 위치 탐색
    erase_point++;
  }

  assert( erase_point != _requests.end( ) ); // [한국어] 반드시 존재해야 함 (없으면 오류)
  _requests.erase( erase_point );            // [한국어] 해당 요청 제거
}

/*
 * [한국어] PriorityArbiter::Match — 마지막 Arbitrate() 결과 반환
 */
int PriorityArbiter::Match( ) const
{
  return _match; // [한국어] -1이면 매칭 없음, 아니면 선택된 입력 번호
}

/*
 * [한국어] PriorityArbiter::Arbitrate — _rr_ptr 기준 순환 탐색으로 최고 우선순위 선택
 * 요청 리스트가 비어 있으면 _match=-1. 비어 있지 않으면:
 * 1. _rr_ptr 이상인 첫 번째 요청부터 시작
 * 2. _rr_ptr까지 순환하며 pri 최댓값 추적
 * 3. 최고 우선순위 입력을 _match로 설정, _rr_ptr = (_match+1)%_inputs 갱신
 */
void PriorityArbiter::Arbitrate( )
{
  list<sRequest>::iterator p; // [한국어] 요청 리스트 순회 이터레이터

  int max_index, max_pri; // [한국어] 현재까지 최고 우선순위 입력 번호와 값
  bool wrapped;           // [한국어] 리스트 순환 여부

  //MERGENOTE
  //booksim does not have this if statement
  //as far as I can tell they are identical in function
  // [한국어] 요청 리스트가 비어 있지 않은 경우에만 중재 수행
  if ( _requests.begin( ) != _requests.end( ) ) {
    // A round-robin arbiter between input requests
    // [한국어] _rr_ptr 이상인 첫 요청 위치부터 탐색 시작
    p = _requests.begin( );
    while( ( p != _requests.end( ) ) &&
	   ( p->in < _rr_ptr ) ) { // [한국어] _rr_ptr 이전 요청 건너뜀
      p++;
    }

    max_index = -1; // [한국어] 최고 우선순위 입력 초기화
    max_pri   = 0;  // [한국어] 최고 우선순위 값 초기화

    // [한국어] 순환 탐색: wrapped=false이면 끝까지, wrapped=true이면 _rr_ptr 전까지
    wrapped = false;
    while( (!wrapped) || ( p->in < _rr_ptr ) ) {
      if ( p == _requests.end( ) ) {
	if ( wrapped ) { break; } // [한국어] 이미 순환했고 끝에 도달 — 탐색 완료
	// p is valid here because empty lists
	// are skipped (above)
	p = _requests.begin( ); // [한국어] 리스트 처음으로 순환
	wrapped = true;          // [한국어] 순환 발생 표시
      }

      // check if request is the highest priority so far
      // [한국어] 현재 요청이 지금까지의 최고 우선순위보다 높으면 갱신
      if ( ( p->pri > max_pri ) || ( max_index == -1 ) ) {
	max_pri   = p->pri;  // [한국어] 최고 우선순위 갱신
	max_index = p->in;   // [한국어] 최고 우선순위 입력 갱신
      }

      p++; // [한국어] 다음 요청으로
    }

    _match = max_index; // -1 for no match
    // [한국어] 최고 우선순위 입력을 _match에 저장
    if ( _match != -1 ) {
      _rr_ptr = ( _match + 1 ) % _inputs; // [한국어] 다음 사이클 탐색 시작 위치 갱신
    }

  } else {
    _match = -1; // [한국어] 요청 없음 — 매칭 없음
  }
}

//MERGENOTE
//added update function to priorityarbiter

/*
 * [한국어] PriorityArbiter::Update — _rr_ptr을 1 전진
 * Arbitrate() 없이 포인터만 갱신할 때 사용 (GPGPU-Sim 통합 시 추가된 함수).
 */
void PriorityArbiter::Update( )
{
  _rr_ptr = ( _rr_ptr + 1 ) % _inputs; // [한국어] 라운드로빈 포인터 1 전진
}
