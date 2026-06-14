// $Id: separable.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] 분리 할당기 기반 클래스 구현 (separable.cpp)
 *
 * === 파일의 역할 ===
 * SeparableAllocator의 생성자, 소멸자, Clear()를 구현한다.
 * 생성자에서 inputs개의 입력 Arbiter와 outputs개의 출력 Arbiter를 생성하며,
 * 소멸자에서 모든 Arbiter를 delete한다. Clear()는 활성 Arbiter만 초기화한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SeparableInputFirstAllocator/SeparableOutputFirstAllocator가 이 클래스를 상속하여
 * 구체 Allocate() 알고리즘을 구현한다.
 *
 * === 타 모듈과의 연결 ===
 * - arbiter.hpp: Arbiter::NewArbiter()로 구체 Arbiter 인스턴스 생성
 * - SparseAllocator: 상속 — Clear() 체인
 *
 * === 주요 함수/구조체 요약 ===
 * - SeparableAllocator(): inputs+outputs개의 Arbiter 생성
 * - ~SeparableAllocator(): 모든 Arbiter delete
 * - Clear(): _num_reqs > 0인 중재기만 초기화
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

#include "separable.hpp"

#include <sstream>

#include "arbiter.hpp"

/*
 * [한국어] SeparableAllocator 생성자 — 입출력 Arbiter 배열 초기화
 *
 * @parent: 부모 모듈 (IQRouter)
 * @name: 이름 문자열
 * @inputs: 입력 포트 수
 * @outputs: 출력 포트 수
 * @arb_type: 중재기 타입 ("round_robin", "matrix", "tree(N,round_robin)" 등)
 *
 * 입력 Arbiter는 각 입력이 원하는 출력들(size=outputs) 중 하나를 중재하고,
 * 출력 Arbiter는 각 출력을 원하는 입력들(size=inputs) 중 하나를 중재한다.
 *
 * 호출 체인: SeparableInputFirstAllocator() → SeparableAllocator() → SparseAllocator()
 */
SeparableAllocator::SeparableAllocator( Module* parent, const string& name,
					int inputs, int outputs,
					const string& arb_type )
  : SparseAllocator( parent, name, inputs, outputs ) // [한국어] SparseAllocator 초기화
{

  _input_arb.resize(inputs); // [한국어] inputs개의 입력 중재기 슬롯 할당

  for (int i = 0; i < inputs; ++i) { // [한국어] 각 입력에 대한 Arbiter 생성
    ostringstream arb_name("arb_i"); // [한국어] "arb_i0", "arb_i1", ... 형식의 이름 생성
    arb_name << i;
    _input_arb[i] = Arbiter::NewArbiter(this, arb_name.str(), arb_type, outputs);
    // [한국어] 입력 i의 중재기 — outputs개의 후보 출력 중 하나를 선택 (size=outputs)
  }

  _output_arb.resize(outputs); // [한국어] outputs개의 출력 중재기 슬롯 할당

  for (int i = 0; i < outputs; ++i) { // [한국어] 각 출력에 대한 Arbiter 생성
    ostringstream arb_name("arb_o"); // [한국어] "arb_o0", "arb_o1", ... 형식의 이름 생성
    arb_name << i;
    _output_arb[i] = Arbiter::NewArbiter(this, arb_name.str( ), arb_type, inputs);
    // [한국어] 출력 j의 중재기 — inputs개의 후보 입력 중 하나를 선택 (size=inputs)
  }

}

/*
 * [한국어] SeparableAllocator 소멸자 — 생성된 모든 Arbiter 인스턴스 해제
 * IQRouter 소멸 시 할당기가 delete되면 이 소멸자가 모든 Arbiter를 정리한다.
 */
SeparableAllocator::~SeparableAllocator() {

  for (int i = 0; i < _inputs; ++i) {  // [한국어] 입력 중재기들 delete
    delete _input_arb[i];
  }

  for (int i = 0; i < _outputs; ++i) { // [한국어] 출력 중재기들 delete
    delete _output_arb[i];
  }

}

/*
 * [한국어] SeparableAllocator::Clear — 활성 Arbiter만 초기화 후 SparseAllocator 초기화
 *
 * _num_reqs > 0인 Arbiter(이번 사이클에 요청을 받은 것)만 Clear()하여 불필요한 반복 방지.
 * 마지막에 SparseAllocator::Clear()를 호출하여 요청 맵과 매칭 결과도 초기화한다.
 * 매 사이클 시작 시 IQRouter가 호출한다.
 */
void SeparableAllocator::Clear() {
  for ( int i = 0 ; i < _inputs ; i++ ) {    // [한국어] 모든 입력 중재기 순회
    if(_input_arb[i]-> _num_reqs)            // [한국어] 요청이 있었던 중재기만 초기화 (최적화)
      _input_arb[i]->Clear();               // [한국어] 요청 배열 초기화 및 _num_reqs = 0
  }
  for ( int o = 0; o < _outputs; o++ ) {    // [한국어] 모든 출력 중재기 순회
    if(_output_arb[o]->_num_reqs)           // [한국어] 요청이 있었던 중재기만 초기화
      _output_arb[o]->Clear();             // [한국어] 요청 배열 초기화
  }
  SparseAllocator::Clear(); // [한국어] _in_req 맵, _in_occ 집합, _inmatch/_outmatch 초기화
}
