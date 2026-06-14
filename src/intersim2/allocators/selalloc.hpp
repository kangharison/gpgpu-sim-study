// $Id: selalloc.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] SelAlloc 우선순위 기반 선택 할당기 선언 (selalloc.hpp)
 *
 * === 파일의 역할 ===
 * SelAlloc(Selective Allocator) 클래스를 선언한다. iSLIP과 유사한 2단계 반복 매칭이지만,
 * Grant/Accept 단계에서 단순 라운드로빈 대신 우선순위(out_pri/in_pri)가 가장 높은 요청을
 * 선택한다는 점이 다르다. 또한 MaskOutput()으로 특정 출력을 할당에서 제외할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("select") 또는 "select(N)"으로 생성. IQRouter VC/스위치 할당기.
 *
 * === 타 모듈과의 연결 ===
 * - SparseAllocator: 상속 — _in_req, _out_req 맵 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - SelAlloc(): 생성자 — _gptrs, _aptrs, _outmask 초기화
 * - Allocate(): 우선순위 기반 반복 Grant-Accept 매칭
 * - MaskOutput(): 특정 출력을 이번 사이클 할당에서 제외
 * - _outmask: 마스크된 출력 표시 벡터 (0=정상, 1=마스크)
 */

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

#ifndef _SELALLOC_HPP_
#define _SELALLOC_HPP_

#include <vector>

#include "allocator.hpp"

class SelAlloc : public SparseAllocator {
  int _iter;
  /* 반복 횟수 — iSLIP과 같이 Grant-Accept를 최대 _iter회 반복.
   * 설정자: 생성자에서 iters 파라미터로 설정.
   * 읽는 자: Allocate() 반복 루프 종료 조건.
   * 값 범위: 1 이상. 설정 파일 "alloc_iters" 또는 기본값 1.
   * 동기화: 불변. */

  vector<int> _aptrs;
  /* Accept 단계 라운드로빈 포인터. _aptrs[input] = Accept 시작 오프셋.
   * iSLIP과 달리 Accept 단계에서 우선순위가 가장 높은 출력을 선택하며,
   * 동점 시 라운드로빈 포인터(_aptrs)로 공정성을 보완한다.
   * 설정자: Allocate()에서 첫 번째 반복 Accept 성공 시 갱신. 사이클 간 유지.
   * 값 범위: 0.._outputs-1. */

  vector<int> _gptrs;
  /* Grant 단계 라운드로빈 포인터. _gptrs[output] = Grant 시작 오프셋.
   * 설정자: Allocate()에서 첫 번째 반복 Accept 성공 시 갱신. 사이클 간 유지.
   * 값 범위: 0.._inputs-1. */

  vector<int> _outmask;
  /* 출력 마스크 벡터. _outmask[out] != 0 이면 이 출력은 할당에서 제외.
   * 설정자: MaskOutput()에서 사용자가 직접 설정. 초기값 0(마스크 없음).
   * 읽는 자: Allocate() Grant 단계에서 마스크된 출력을 건너뛸 때 사용.
   * 값 범위: 0(정상) 또는 1(마스크). */

public:
  /*
   * [한국어] SelAlloc 생성자 — 포인터와 마스크 벡터 초기화
   * @parent: 부모 모듈
   * @name: 이름
   * @inputs, @outputs: 입출력 포트 수
   * @iters: Grant-Accept 반복 횟수
   */
  SelAlloc( Module *parent, const string& name,
	    int inputs, int outputs, int iters );

  void Allocate( );  // [한국어] 우선순위 기반 반복 Grant-Accept 매칭 실행

  void MaskOutput( int out, int mask = 1 ); // [한국어] 출력 out을 마스크 (할당에서 제외)

  virtual void PrintRequests( ostream * os = NULL ) const; // [한국어] 마스크 상태 포함 요청 출력

};

#endif 
