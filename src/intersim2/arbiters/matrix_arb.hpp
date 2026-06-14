// $Id: matrix_arb.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] MatrixArbiter 선언 (matrix_arb.hpp)
 *
 * === 파일의 역할 ===
 * 행렬(matrix) 우선순위 중재 알고리즘을 구현하는 MatrixArbiter를 선언한다.
 * _matrix[i][j] = 1이면 입력 i가 입력 j보다 우선순위가 낮음을 의미한다.
 * 마지막 승자(_selected)가 UpdateState()에서 _matrix 갱신으로 최저 우선순위를 가지게 된다.
 * Arbitrate()에서 요청이 2개 이상일 때 O(N^2) 탐색으로 무조건 낮지 않은 입력을 찾는다.
 * 1개 이하 요청이면 O(1)로 처리. 완전한 공정성을 보장하는 중재 알고리즘이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("matrix") → MatrixArbiter()
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): _request, _size, _selected, _num_reqs
 *
 * === 주요 함수/구조체 요약 ===
 * - _matrix[i][j]: 우선순위 행렬. 1이면 i보다 j가 우선. 생성자에서 하삼각을 1로 초기화.
 * - _last_req: 가장 최근 AddRequest()된 입력 (요청 1개 이하 최적화용)
 * - Arbitrate(): _num_reqs<2면 _last_req 즉시 선택; 아니면 행렬 탐색
 * - UpdateState(): _selected행을 0으로, _selected열을 1로 설정 (최저 우선순위 강등)
 */

// ----------------------------------------------------------------------
//
//  Matrix: Matrix Arbiter
//
// ----------------------------------------------------------------------

#ifndef _MATRIX_ARB_HPP_
#define _MATRIX_ARB_HPP_

#include <vector>

#include "arbiter.hpp"

using namespace std;

class MatrixArbiter : public Arbiter {

  // Priority matrix
  vector<vector<int> > _matrix ;
  /* [한국어] _size × _size 우선순위 행렬. _matrix[i][j]=1이면 i가 j에 비해 낮은 우선순위를 가짐.
   * 설정자: 생성자에서 하삼각(i>j인 [i][j])을 1로 초기화; UpdateState()에서 승자 기준 갱신.
   * 읽는 자: Arbitrate()에서 입력 i가 승자인지 판별할 때 (_matrix[i][input] 존재 시 grant=false).
   * 해석: 입력 i에 대해 _matrix[k][i]==1인 요청 k가 있으면 i는 k에게 양보해야 함. */

  int  _last_req ;
  /* [한국어] 가장 최근에 AddRequest()된 입력 번호. 요청이 0개 또는 1개인 경우 즉시 선택에 사용.
   * 설정자: AddRequest()마다 갱신; Clear()에서 -1로 리셋.
   * 읽는 자: Arbitrate()에서 _num_reqs<2이면 _selected=_last_req로 즉시 결정. */

public:

  // Constructors
  /*
   * [한국어] MatrixArbiter 생성자 — 우선순위 행렬 초기화
   * size×size 행렬을 생성하고 하삼각(i>j인 [i][j])을 1로 초기화한다.
   * 이 초기 상태는 높은 번호의 입력이 낮은 번호의 입력보다 낮은 우선순위를 가짐을 의미.
   * 호출 체인: Arbiter::NewArbiter("matrix") → MatrixArbiter()
   */
  MatrixArbiter( Module *parent, const string &name, int size ) ;

  /*
   * [한국어] PrintState — _matrix 전체를 stdout에 출력 (디버그용)
   */
  virtual void PrintState() const ;

  /*
   * [한국어] UpdateState — 마지막 승자(_selected)를 최저 우선순위로 강등
   * _matrix[_selected][*]=0, _matrix[*][_selected]=1 설정.
   * 이후 다음 사이클에서 _selected가 다시 선택되려면 모든 다른 입력들이 먼저 선택되어야 함.
   */
  virtual void UpdateState() ;

  /*
   * [한국어] Arbitrate — 행렬 기반 우선순위 중재
   * 요청이 1개 이하이면 _last_req 즉시 반환. 2개 이상이면 _request[input].valid인
   * 입력 중 _matrix[k][input]=1인 다른 요청 k가 없는 첫 번째 입력을 선택.
   */
  virtual int Arbitrate( int* id = 0, int* pri = 0) ;

  /*
   * [한국어] AddRequest — 요청 등록 + _last_req 갱신
   */
  virtual void AddRequest( int input, int id, int pri ) ;

  /*
   * [한국어] Clear — _last_req=-1 리셋 후 Arbiter::Clear() 호출
   */
  virtual void Clear();

} ;

#endif
