// $Id: matrix_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] MatrixArbiter 구현 (matrix_arb.cpp)
 *
 * === 파일의 역할 ===
 * 행렬 우선순위 중재 알고리즘 구현. 생성자에서 _matrix 하삼각을 1로 초기화하고,
 * UpdateState()에서 승자를 최저 우선순위로 강등한다. Arbitrate()는 요청이 1개 이하이면
 * O(1), 2개 이상이면 O(N^2) 탐색으로 승자를 결정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("matrix") → MatrixArbiter
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): AddRequest(), Arbitrate(), Clear() super 호출
 * - BookSimConfig / gpgpusim.config:
 *     * arb_type = "matrix"일 때 NewArbiter()가 이 클래스를 생성한다.
 *     * alloc_iters 횟수만큼 매 사이클 Arbitrate() 이후 UpdateState()가 반복 호출되어
 *       우선순위 행렬이 갱신된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - MatrixArbiter(): _matrix[i][j]=1 for i>j (초기 우선순위 설정)
 * - UpdateState(): 승자 행=0, 승자 열=1 설정 (최저 우선순위 강등)
 * - Arbitrate(): _num_reqs<2이면 즉시 리턴; 아니면 O(N^2) 행렬 탐색
 */

// ----------------------------------------------------------------------
//
//  Matrix: Matrix Arbiter
//
// ----------------------------------------------------------------------

#include "matrix_arb.hpp" // [한국어] MatrixArbiter 선언
#include <iostream>       // [한국어] PrintState()의 cout 사용
using namespace std ;

/*
 * [한국어] MatrixArbiter 생성자 — 우선순위 행렬 초기화
 * _matrix[i][j]=1 for i>j: 초기 상태에서 낮은 번호의 입력이 높은 번호보다 우선.
 * 대각선(_matrix[i][i])은 0으로 유지 (자기 자신과의 비교 불필요).
 * 호출 체인: Arbiter::NewArbiter("matrix") → MatrixArbiter() → Arbiter()
 */
MatrixArbiter::MatrixArbiter( Module *parent, const string &name, int size )
  : Arbiter( parent, name, size ), // [한국어] _request 배열 할당, 기본 상태 초기화
    _last_req(-1) {                // [한국어] 마지막 요청 없음으로 초기화
  _matrix.resize(size);           // [한국어] size개 행 할당
  for ( int i = 0 ; i < size ; i++ ) {
    _matrix[i].resize(size);      // [한국어] 각 행에 size개 열 할당 (기본값 0)
    for ( int j = 0; j < i; j++ ) {
      _matrix[i][j] = 1;          // [한국어] 하삼각 초기화: i가 j보다 낮은 우선순위
    }
  }
}

/*
 * [한국어] PrintState — _matrix 전체를 stdout에 출력 (디버그용)
 */
void MatrixArbiter::PrintState() const  {
  cout << "Priority Matrix: " << endl ;
  for ( int r = 0; r < _size ; r++ ) {        // [한국어] 행 순회
    for ( int c = 0 ; c < _size ; c++ ) {     // [한국어] 열 순회
      cout << _matrix[r][c] << " " ;          // [한국어] 각 셀 출력
    }
    cout << endl ;
  }
  cout << endl ;
}

/*
 * [한국어] MatrixArbiter::UpdateState — 마지막 승자를 최저 우선순위로 강등
 * _matrix[_selected][*]=0: _selected는 모든 다른 입력에 대해 우선권을 잃음.
 * _matrix[*][_selected]=1: 모든 다른 입력이 _selected에 대해 우선권을 가짐.
 * 이후 _selected가 다시 선택되려면 다른 모든 요청이 먼저 처리되어야 함.
 */
void MatrixArbiter::UpdateState() {
  // update priority matrix using last grant
  if ( _selected > -1 ) {                         // [한국어] 선택된 입력이 있을 때만 갱신
    for ( int i = 0; i < _size ; i++ ) {           // [한국어] 모든 다른 입력에 대해
      if( _selected != i ) {                       // [한국어] 자기 자신은 제외
	_matrix[_selected][i] = 0 ;               // [한국어] _selected는 i에 대해 우선권 없음
	_matrix[i][_selected] = 1 ;              // [한국어] i는 _selected에 대해 우선권 가짐
      }
    }
  }
}

/*
 * [한국어] MatrixArbiter::AddRequest — 요청 등록 + _last_req 갱신
 */
void MatrixArbiter::AddRequest( int input, int id, int pri )
{
  _last_req = input;                    // [한국어] 가장 최근 요청 입력 갱신 (1개 이하 최적화용)
  Arbiter::AddRequest(input, id, pri);  // [한국어] _request[input] 설정 + _num_reqs 증가 (super 호출)
}

/*
 * [한국어] MatrixArbiter::Arbitrate — 행렬 기반 우선순위 중재
 * 요청이 1개 이하이면 행렬 탐색을 건너뛰고 _last_req를 즉시 선택한다.
 * 요청이 2개 이상이면 입력을 0부터 순서대로 탐색하며:
 *   - 같은 우선순위(pri)에서 _matrix[k][input]=1인 요청 k가 없고
 *   - 더 높은 우선순위(pri>_request[input].pri)의 요청도 없는 입력을 승자로 선택.
 */
int MatrixArbiter::Arbitrate( int* id, int* pri ) {

  // avoid running arbiter if it has not recevied at least two requests
  // (in this case, requests and grants are identical)
  if ( _num_reqs < 2 ) {    // [한국어] 요청이 0개 또는 1개 — 즉시 처리
    _selected = _last_req ; // [한국어] 0개이면 -1, 1개이면 그 입력

  } else {
    // [한국어] 요청이 2개 이상 — O(N^2) 행렬 탐색
    _selected = -1 ;

    for ( int input = 0 ; input < _size ; input++ ) { // [한국어] 모든 입력 순회
      if(_request[input].valid) {                      // [한국어] 유효한 요청이 있는 입력만 검사

	bool grant = true; // [한국어] 이 입력이 승자가 될 수 있는지 여부
	for ( int i = 0 ; i < _size ; i++ ) { // [한국어] 다른 모든 요청과 비교
	  if ( _request[i].valid &&
	       ( ( ( _request[i].pri == _request[input].pri ) &&
		   _matrix[i][input]) || // [한국어] 같은 우선순위에서 i가 input보다 행렬 우선
		 ( _request[i].pri > _request[input].pri ) // [한국어] i의 우선순위가 더 높음
		 ) ) {
	    grant = false ; // [한국어] 이 입력은 승자가 될 수 없음 — 다음 입력으로
	    break ;
	  }
	}

	if ( grant ) {
	  _selected = input ; // [한국어] 승자 결정
	  break ;             // [한국어] 첫 번째 승자로 즉시 탈출 (안정 중재)
	}
      }

    }
  }

  return Arbiter::Arbitrate(id, pri); // [한국어] id/pri 포인터 채우기 + _selected 반환
}

/*
 * [한국어] MatrixArbiter::Clear — _last_req=-1 리셋 후 Arbiter::Clear() 호출
 * _matrix는 리셋하지 않음 — 사이클 간 유지되어야 우선순위 기록이 보존됨.
 */
void MatrixArbiter::Clear()
{
  _last_req = -1;   // [한국어] 마지막 요청 리셋
  Arbiter::Clear(); // [한국어] _request[].valid=false, _num_reqs=0, _selected=-1
}
