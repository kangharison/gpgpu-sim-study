// $Id: separable.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] 분리 할당기(Separable Allocator) 기반 클래스 선언 (separable.hpp)
 *
 * === 파일의 역할 ===
 * 분리 할당기의 공통 구조를 정의한다. 분리 할당기는 입력 중재기(_input_arb)와 출력
 * 중재기(_output_arb)를 별도로 운영하여 크로스바 매칭을 수행한다. "분리(Separable)"는
 * 입력 측 결정과 출력 측 결정이 독립적으로 이루어짐을 뜻한다. 이 구조는 O(N) 복잡도로
 * iSLIP보다 단순하지만, 공정성과 처리량이 다소 낮을 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SparseAllocator를 상속하고, SeparableInputFirstAllocator / SeparableOutputFirstAllocator가
 * 이 클래스를 다시 상속하여 구체 알고리즘을 구현한다.
 * 호출 체인: IQRouter → SeparableInputFirstAllocator::Allocate() (상속)
 *                     → SeparableAllocator::Clear() 등
 *
 * === 타 모듈과의 연결 ===
 * - SparseAllocator: 상속 — 요청 맵과 occupied 집합 제공
 * - Arbiter (arbiters/arbiter.hpp): _input_arb, _output_arb의 원소 타입
 * - SeparableInputFirstAllocator/SeparableOutputFirstAllocator: 이 클래스를 상속
 *
 * === 주요 함수/구조체 요약 ===
 * - SeparableAllocator(): 생성자 — inputs개의 입력 중재기와 outputs개의 출력 중재기 생성
 * - ~SeparableAllocator(): 소멸자 — 모든 Arbiter 인스턴스 delete
 * - Clear(): 모든 Arbiter의 요청을 초기화 후 SparseAllocator::Clear() 호출
 * - _input_arb: 입력 i가 어떤 출력을 선택할지 중재하는 Arbiter 배열
 * - _output_arb: 출력 j를 담당할 입력을 중재하는 Arbiter 배열
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

// ----------------------------------------------------------------------
//
//  SeparableAllocator: Separable Allocator Base Class
//
// ----------------------------------------------------------------------

#ifndef _SEPARABLE_HPP_
#define _SEPARABLE_HPP_

#include <vector>

#include "allocator.hpp"

class Arbiter; // [한국어] 전방 선언 — 구현에서 arbiter.hpp 포함

/*
 * [한국어] SeparableAllocator - 입출력 중재기를 각각 두는 분리형 할당기 기반 클래스
 *
 * 입력별로 Arbiter 하나(_input_arb[i])와 출력별로 Arbiter 하나(_output_arb[j])를 보유한다.
 * 구체 알고리즘(InputFirst/OutputFirst)이 이 Arbiter들을 호출하여 매칭을 결정한다.
 */
class SeparableAllocator : public SparseAllocator {

protected:

  vector<Arbiter*> _input_arb ;
  /* 입력 중재기 배열. _input_arb[i]는 입력 i가 요청하는 출력들 중 하나를 중재.
   * 설정자: 생성자에서 arb_type에 맞는 Arbiter::NewArbiter()로 생성.
   * 읽는 자: Allocate() 구현에서 AddRequest()/Arbitrate()/UpdateState() 호출.
   * 값 범위: _inputs개의 Arbiter 포인터. NULL 불가.
   * 동기화: 단일 스레드. */

  vector<Arbiter*> _output_arb ;
  /* 출력 중재기 배열. _output_arb[j]는 출력 j를 요청하는 입력들 중 하나를 중재.
   * 설정자: 생성자에서 Arbiter::NewArbiter()로 생성.
   * 읽는 자: Allocate() 구현에서 AddRequest()/Arbitrate()/UpdateState() 호출.
   * 값 범위: _outputs개의 Arbiter 포인터. NULL 불가.
   * 동기화: 단일 스레드. */

public:

  /*
   * [한국어] SeparableAllocator 생성자 — 입출력 Arbiter 배열 생성
   * @parent: 부모 모듈
   * @name: 이름
   * @inputs: 입력 포트 수
   * @outputs: 출력 포트 수
   * @arb_type: 중재기 타입 문자열 ("round_robin", "matrix", "tree(N,round_robin)" 등)
   * 각 입력/출력에 대해 Arbiter::NewArbiter()를 호출하여 중재기를 생성한다.
   * 호출 체인: SeparableInputFirstAllocator() → SeparableAllocator()
   */
  SeparableAllocator( Module* parent, const string& name, int inputs,
		      int outputs, const string& arb_type ) ;

  /*
   * [한국어] SeparableAllocator 소멸자 — _input_arb와 _output_arb의 모든 Arbiter 해제
   */
  virtual ~SeparableAllocator() ;

  /*
   * [한국어] Clear - 요청이 있는 Arbiter만 초기화 후 SparseAllocator::Clear() 호출
   * _num_reqs > 0인 중재기만 Clear()하여 불필요한 초기화를 피한다.
   */
  virtual void Clear() ;

} ;

#endif
