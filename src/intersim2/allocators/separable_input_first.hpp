// $Id: separable_input_first.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] 입력 먼저 분리 할당기 선언 (separable_input_first.hpp)
 *
 * === 파일의 역할 ===
 * SeparableInputFirstAllocator를 선언한다. "입력 먼저(Input-First)"는 1단계에서 각 입력
 * 중재기가 먼저 원하는 출력을 하나 선택하고, 2단계에서 그 결과를 출력 중재기에 전달하여
 * 최종 출력 할당을 결정하는 순서를 의미한다. 이 방식은 입력 측 공정성을 우선한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("separable_input_first") 또는 "separable_input_first(arb_type)"으로
 * 생성되며, IQRouter의 VC 할당기 또는 스위치 할당기로 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - SeparableAllocator: 상속 — _input_arb, _output_arb 배열과 Clear() 제공
 * - Arbiter: _input_arb[i]->Arbitrate(), _output_arb[j]->Arbitrate()로 중재 실행
 *
 * === 주요 함수/구조체 요약 ===
 * - Allocate(): 1단계(입력 중재) → 2단계(출력 중재) 순서로 매칭 결정
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
//  SeparableInputFirstAllocator: Separable Input-First Allocator
//
// ----------------------------------------------------------------------

#ifndef _SEPARABLE_INPUT_FIRST_HPP_
#define _SEPARABLE_INPUT_FIRST_HPP_

#include <set>

#include "separable.hpp"

class SeparableInputFirstAllocator : public SeparableAllocator {

public:
  
  SeparableInputFirstAllocator( Module* parent, const string& name, int inputs,
				int outputs, const string& arb_type ) ;

  virtual void Allocate() ;

} ;

#endif
