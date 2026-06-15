// $Id: separable_output_first.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] 출력 먼저 분리 할당기 선언 (separable_output_first.hpp)
 *
 * === 파일의 역할 ===
 * SeparableOutputFirstAllocator를 선언한다. "출력 먼저(Output-First)"는 1단계에서 각 출력
 * 중재기가 먼저 담당할 입력을 하나 선택하고, 2단계에서 선택된 입력 중재기가 최종 출력을
 * 결정하는 순서를 의미한다. 이 방식은 출력 측 공정성을 우선한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("separable_output_first")로 생성. IQRouter VC/스위치 할당기.
 *
 * === 타 모듈과의 연결 ===
 * - SeparableAllocator: 상속
 * - Arbiter: 1단계 출력 중재, 2단계 입력 중재에 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - Allocate(): 1단계(출력 중재) → 2단계(입력 중재) 순서로 매칭 결정
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
//  SeparableOutputFirstAllocator: Separable Output-First Allocator
//
// ----------------------------------------------------------------------

#ifndef _SEPARABLE_OUTPUT_FIRST_HPP_
#define _SEPARABLE_OUTPUT_FIRST_HPP_

#include "separable.hpp"

class SeparableOutputFirstAllocator : public SeparableAllocator {

public:
  
  /*
   * [한국어] 출력 먼저 분리 할당기 생성자
   * @parent: 부모 모듈
   * @name: 이름
   * @inputs, @outputs: 입출력 포트 수
   * @arb_type: 사용할 Arbiter 타입 문자열 ("round_robin", "matrix", "tree(N,round_robin)" 등)
   * 호출 체인: Allocator::NewAllocator("separable_output_first") → SeparableOutputFirstAllocator() → SeparableAllocator()
   */
  SeparableOutputFirstAllocator( Module* parent, const string& name, int inputs,
				 int outputs, const string& arb_type ) ;

  /*
   * [한국어] 출력 먼저 분리 매칭 실행
   * 1단계 출력 중재 → 2단계 입력 중재 순서로 _inmatch/_outmatch를 채운다.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → SeparableOutputFirstAllocator::Allocate()
   */
  virtual void Allocate() ;

} ;

#endif
