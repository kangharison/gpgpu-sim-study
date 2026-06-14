// $Id: arbiter.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] Arbiter 기반 클래스 구현 (arbiter.cpp)
 *
 * === 파일의 역할 ===
 * Arbiter 기반 클래스의 공통 메서드(생성자, AddRequest, Arbitrate, Clear, NewArbiter)를 구현한다.
 * 서브클래스(RoundRobinArbiter, MatrixArbiter, TreeArbiter)가 오버라이드하지 않는 공통 로직을 담당한다.
 * NewArbiter() 팩토리 함수도 여기에 구현되어 문자열 타입으로 적절한 중재자를 생성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SeparableAllocator/iSLIP → Arbiter 서브클래스 → Arbiter 기반 메서드
 *
 * === 타 모듈과의 연결 ===
 * - roundrobin_arb.hpp, matrix_arb.hpp, tree_arb.hpp: NewArbiter() 팩토리에서 인스턴스화
 *
 * === 주요 함수/구조체 요약 ===
 * - Arbiter(): _size/selected/_highest_pri/_best_input/_num_reqs 초기화; _request 배열 할당
 * - AddRequest(): 유효성 검사 후 _request[input] 설정, _num_reqs 증가
 * - Arbitrate(): _selected 반환, id/pri 포인터 업데이트
 * - Clear(): 요청 있을 때만 _request[].valid=false, _num_reqs=0, _selected=-1
 * - NewArbiter(): "round_robin"/"matrix"/"tree(N,type)" 문자열 파싱 후 인스턴스 생성
 */

// ----------------------------------------------------------------------
//
//  Arbiter: Base class for Matrix and Round Robin Arbiter
//
// ----------------------------------------------------------------------

#include "arbiter.hpp"
#include "roundrobin_arb.hpp" // [한국어] RoundRobinArbiter 정의
#include "matrix_arb.hpp"     // [한국어] MatrixArbiter 정의
#include "tree_arb.hpp"       // [한국어] TreeArbiter 정의

#include <limits>  // [한국어] numeric_limits<int>::min() — _highest_pri 초기값
#include <cassert> // [한국어] assert() — 인자 유효성 검사

using namespace std ;

/*
 * [한국어] Arbiter 생성자 — 기본 상태 초기화
 * @parent: 부모 모듈 (Module 계층 등록)
 * @name: 이 중재자 이름
 * @size: 입력 포트 수 (_request 배열 크기)
 * _request 배열을 할당하고 모든 valid를 false로 초기화한다.
 * _highest_pri를 int 최솟값으로 설정하여 첫 요청이 반드시 최고 우선순위가 되도록 한다.
 * 호출 체인: 서브클래스 생성자 → Arbiter()
 */
Arbiter::Arbiter( Module *parent, const string &name, int size )
  : Module( parent, name ),                        // [한국어] Module 계층 등록
    _size(size), _selected(-1),                     // [한국어] 포트 수 / 초기 선택 없음
    _highest_pri(numeric_limits<int>::min()),        // [한국어] 최솟값으로 초기화 — 첫 요청이 최고 우선순위
    _best_input(-1), _num_reqs(0)                   // [한국어] 최선 입력 없음, 요청 수 0
{
  _request.resize(size);                            // [한국어] 입력 수만큼 요청 배열 할당
  for ( int i = 0 ; i < size ; i++ )
    _request[i].valid = false ;                     // [한국어] 모든 요청을 무효 상태로 초기화
}

/*
 * [한국어] Arbiter::AddRequest — 입력 포트의 요청을 id/pri와 함께 등록
 * @input: 요청하는 입력 포트 번호 (0.._size-1)
 * @id: 요청 식별자
 * @pri: 요청 우선순위 (높을수록 선호)
 * 동일 입력의 중복 요청은 assert로 방지한다.
 * 호출 체인: 서브클래스 AddRequest() → Arbiter::AddRequest() (super 호출)
 */
void Arbiter::AddRequest( int input, int id, int pri )
{
  assert( 0 <= input && input < _size ) ; // [한국어] 유효 범위 검사
  assert( !_request[input].valid );       // [한국어] 중복 요청 방지 — 이미 등록된 요청 없어야 함

  _num_reqs++ ;                           // [한국어] 요청 수 증가
  _request[input].valid = true ;          // [한국어] 요청 유효화
  _request[input].id = id ;              // [한국어] 요청 식별자 저장
  _request[input].pri = pri ;            // [한국어] 요청 우선순위 저장
}

/*
 * [한국어] Arbiter::Arbitrate — _selected 반환; id/pri 포인터가 유효하면 채움
 * @id: (옵션) 선택된 입력의 식별자를 저장할 포인터
 * @pri: (옵션) 선택된 입력의 우선순위를 저장할 포인터
 * @return: _selected (-1이면 요청 없음)
 * 서브클래스에서 _selected를 결정한 후 이 메서드를 super 호출하는 패턴.
 */
int Arbiter::Arbitrate( int* id, int* pri )
{
  if ( _selected != -1 ) {                          // [한국어] 선택된 입력이 있을 때만 id/pri 채움
    if ( id )
      *id  = _request[_selected].id ;               // [한국어] 선택된 입력의 식별자 반환
    if ( pri )
      *pri = _request[_selected].pri ;              // [한국어] 선택된 입력의 우선순위 반환
  }

  assert((_selected >= 0) || (_num_reqs == 0));    // [한국어] 요청이 있으면 반드시 선택이 있어야 함

  return _selected ; // [한국어] 선택된 입력 번호 반환
}

/*
 * [한국어] Arbiter::Clear — 모든 요청 초기화; 요청이 없으면 즉시 리턴
 * 사이클 간 상태 정리에 사용. 요청이 없을 때는 불필요한 루프를 건너뜀.
 */
void Arbiter::Clear()
{
  if(_num_reqs > 0) { // [한국어] 요청이 있을 때만 정리 (불필요한 루프 방지)

    // clear the request vector
    for ( int i = 0; i < _size ; i++ )
      _request[i].valid = false ; // [한국어] 모든 요청을 무효화
    _num_reqs = 0 ;               // [한국어] 요청 수 리셋
    _selected = -1;               // [한국어] 선택 결과 리셋
  }
}

/*
 * [한국어] Arbiter::NewArbiter — 중재자 타입 문자열에 따른 팩토리 함수
 * @arb_type: "round_robin" / "matrix" / "tree(groups,sub_arb_type)"
 * @size: 입력 포트 수
 * @return: 동적 할당된 Arbiter 포인터 (호출자가 해제 책임)
 *
 * "tree(N,type)" 형식은 파싱을 통해 N개 그룹과 서브 중재자 타입을 추출한다.
 * 예: "tree(4,round_robin)" → 4그룹, 각 그룹에 RoundRobinArbiter 사용하는 TreeArbiter.
 * 호출 체인: SeparableAllocator() → NewArbiter(); TreeArbiter() → NewArbiter()
 */
Arbiter *Arbiter::NewArbiter( Module *parent, const string& name,
			      const string &arb_type, int size)
{
  Arbiter *a = NULL;                        // [한국어] 생성될 중재자 포인터 초기화
  if(arb_type == "round_robin") {           // [한국어] 라운드로빈 중재자 생성
    a = new RoundRobinArbiter( parent, name, size );
  } else if(arb_type == "matrix") {         // [한국어] 행렬 우선순위 중재자 생성
    a = new MatrixArbiter( parent, name, size );
  } else if(arb_type.substr(0, 5) == "tree(") { // [한국어] 트리 중재자 — 파라미터 파싱 필요
    size_t left = 4;                        // [한국어] "tree(" 이후 첫 문자 오프셋
    size_t middle = arb_type.find_first_of(','); // [한국어] 첫 ',' 위치 (groups와 sub_arb_type 경계)
    assert(middle != string::npos);         // [한국어] ',' 없으면 형식 오류
    size_t right = arb_type.find_last_of(')');   // [한국어] 마지막 ')' 위치
    assert(right != string::npos);          // [한국어] ')' 없으면 형식 오류
    string groups_str = arb_type.substr(left+1, middle-left-1); // [한국어] 그룹 수 문자열 추출
    int groups = atoi(groups_str.c_str()); // [한국어] 문자열을 정수로 변환
    string sub_arb_type = arb_type.substr(middle+1, right-middle-1); // [한국어] 서브 중재자 타입 추출
    a = new TreeArbiter( parent, name, size, groups, sub_arb_type ); // [한국어] TreeArbiter 생성
  } else assert(false);                     // [한국어] 알 수 없는 중재자 타입 — 종료
  return a; // [한국어] 생성된 중재자 포인터 반환
}
