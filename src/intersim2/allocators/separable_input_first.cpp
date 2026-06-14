// $Id: separable_input_first.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] 입력 먼저 분리 할당기 구현 (separable_input_first.cpp)
 *
 * === 파일의 역할 ===
 * SeparableInputFirstAllocator::Allocate()를 구현한다.
 * 1단계: 각 활성 입력(_in_occ)의 요청을 _input_arb[input]에 등록하고 Arbitrate()로 출력 선택.
 * 2단계: 선택된 (input, output) 쌍을 _output_arb[output]에 등록하고 Arbitrate()로 최종 선택.
 * 이 방식은 입력이 원하는 출력을 먼저 결정한 후 출력이 최종 수락자를 결정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → SeparableInputFirstAllocator::Allocate()
 *
 * === 타 모듈과의 연결 ===
 * - SeparableAllocator: 상속 (_input_arb, _output_arb, _in_occ, _out_occ, _in_req, _out_req)
 * - Arbiter: AddRequest(), Arbitrate(), UpdateState() 3단계로 중재 수행
 *
 * === 주요 함수/구조체 요약 ===
 * - Allocate(): 입력 먼저 → 출력 최종 결정 2단계 분리 매칭 알고리즘
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

#include "separable_input_first.hpp"

#include "booksim.hpp"
#include "arbiter.hpp"

#include <vector>
#include <iostream>
#include <cstring>

SeparableInputFirstAllocator::
SeparableInputFirstAllocator( Module* parent, const string& name, int inputs,
			      int outputs, const string& arb_type )
  : SeparableAllocator( parent, name, inputs, outputs, arb_type )
{}

/*
 * [한국어] SeparableInputFirstAllocator::Allocate - 입력 먼저 분리 매칭 알고리즘 실행
 *
 * 1단계 (입력 중재): 활성 입력(_in_occ)을 순회하며 입력 Arbiter로 출력 하나를 선택하고,
 *                   선택 결과를 출력 Arbiter(_output_arb[output])에 전달.
 * 2단계 (출력 중재): 활성 출력(_out_occ)을 순회하며 출력 Arbiter로 최종 입력을 선택하고,
 *                   _inmatch/_outmatch에 기록 후 양쪽 Arbiter 상태 갱신(UpdateState).
 *
 * 호출 체인:
 *   IQRouter::_SWAllocEvaluate() → SeparableInputFirstAllocator::Allocate()
 */
void SeparableInputFirstAllocator::Allocate() {

  // [한국어] 1단계: 입력 중재 — 각 활성 입력이 원하는 출력 하나를 선택
  set<int>::const_iterator port_iter = _in_occ.begin(); // [한국어] 활성 입력 집합 순회 시작
  while(port_iter != _in_occ.end()) {

    const int & input = *port_iter; // [한국어] 현재 처리 중인 입력 포트

    // add requests to the input arbiter
    // [한국어] 이 입력이 보낸 요청들을 _input_arb[input]에 등록
    map<int, sRequest>::const_iterator req_iter = _in_req[input].begin();
    while(req_iter != _in_req[input].end()) { // [한국어] 이 입력의 모든 요청 열거

      const sRequest & req = req_iter->second; // [한국어] 요청 메타데이터

      _input_arb[input]->AddRequest(req.port, req.label, req.in_pri);
      // [한국어] 입력 중재기에 (출력 포트, 레이블, 입력측 우선순위)로 요청 등록

      ++req_iter;
    }

    // Execute the input arbiters and propagate the grants to the
    // output arbiters.
    // [한국어] 입력 중재기 실행 — 이 입력이 선택한 출력을 결정
    int label = -1; // [한국어] 선택된 요청의 레이블 (out 파라미터)
    const int output = _input_arb[input]->Arbitrate(&label, NULL);
    // [한국어] 입력 중재기가 최우선 출력 포트를 선택. output = 선택된 출력 번호.
    assert(output > -1); // [한국어] 활성 입력(_in_occ에 있음)이므로 반드시 요청이 존재해야 함

    const sRequest & req = _out_req[output][input]; // [한국어] 역방향 맵에서 이 요청의 out_pri 조회
    assert((req.port == input) && (req.label == label)); // [한국어] 일관성 검사

    _output_arb[output]->AddRequest(req.port, req.label, req.out_pri);
    // [한국어] 출력 중재기에 (입력 포트, 레이블, 출력측 우선순위)로 요청 전달

    ++port_iter; // [한국어] 다음 활성 입력으로
  }

  // [한국어] 2단계: 출력 중재 — 각 활성 출력이 최종 수락할 입력을 선택
  port_iter = _out_occ.begin(); // [한국어] 활성 출력 집합 순회 시작
  while(port_iter != _out_occ.end()) {

    const int & output = *port_iter; // [한국어] 현재 처리 중인 출력 포트

    // Execute the output arbiters.
    // [한국어] 출력 중재기 실행 — 1단계에서 이 출력을 선택한 입력 중 최우선 입력 선택
    const int input = _output_arb[output]->Arbitrate(NULL, NULL);

    if(input > -1) { // [한국어] 이 출력에 대한 요청이 존재하면 매칭 기록
      assert((_inmatch[input] == -1) && (_outmatch[output] == -1)); // [한국어] 중복 매칭 방지

      _inmatch[input] = output ;         // [한국어] 입력 측 매칭 결과 기록
      _outmatch[output] = input ;        // [한국어] 출력 측 매칭 결과 기록
      _input_arb[input]->UpdateState() ; // [한국어] 입력 중재기의 라운드로빈 포인터 갱신 (공정성)
      _output_arb[output]->UpdateState() ; // [한국어] 출력 중재기의 라운드로빈 포인터 갱신 (공정성)
    }

    ++port_iter; // [한국어] 다음 활성 출력으로
  }
}
