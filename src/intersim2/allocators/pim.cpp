// $Id: pim.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] PIM(Parallel Iterative Matching) 할당기 구현 (pim.cpp)
 *
 * === 파일의 역할 ===
 * PIM 알고리즘을 구현한다. Grant 단계에서 출력이 미매칭 요청 중 무작위로 하나를 선택하고,
 * Accept 단계에서 입력이 Grant받은 출력 중 무작위로 하나를 수락한다.
 * iSLIP과 구조는 동일하지만 포인터(_gptrs, _aptrs) 대신 RandomInt()로 오프셋을 결정한다.
 * 포인터 상태 유지가 없어 구현이 단순하지만, 공정성은 확률적으로만 보장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → PIM::Allocate()
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 (_request, _inmatch, _outmatch)
 * - random_utils.hpp: RandomInt() — 무작위 탐색 시작 위치 결정
 *
 * === 주요 함수/구조체 요약 ===
 * - PIM(): DenseAllocator 초기화 + _PIM_iter 저장
 * - ~PIM(): 빈 소멸자
 * - Allocate(): _PIM_iter 반복 Grant(무작위 입력 선택) + Accept(무작위 출력 선택)
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

#include "booksim.hpp"
#include <iostream>

#include "pim.hpp"
#include "random_utils.hpp"

//#define DEBUG_PIM

/*
 * [한국어] PIM 생성자 — DenseAllocator 초기화 및 반복 횟수 저장
 * 호출 체인: Allocator::NewAllocator("pim") → PIM() → DenseAllocator()
 */
PIM::PIM( Module *parent, const string& name,
	  int inputs, int outputs, int iters ) :
  DenseAllocator( parent, name, inputs, outputs ), // [한국어] _request 2D 배열 초기화
  _PIM_iter(iters)                                 // [한국어] Grant-Accept 반복 횟수 저장
{
}

/*
 * [한국어] PIM 소멸자 — DenseAllocator가 _request 해제하므로 별도 작업 없음
 */
PIM::~PIM( )
{
}

/*
 * [한국어] PIM::Allocate — 무작위 Grant-Accept 반복으로 이분 매칭 수행
 *
 * iSLIP과 동일한 2단계 반복 구조이지만 포인터(_gptrs/_aptrs) 없이 RandomInt()를 사용한다.
 *
 * Grant 단계: 각 미매칭 출력이 자신의 요청 중 미매칭 입력을 무작위 오프셋 순서로 탐색하여
 *   처음 발견한 미매칭 입력에 Grant한다. grants[output] = input으로 기록.
 *
 * Accept 단계: 각 입력이 자신에게 Grant된 출력을 무작위 오프셋 순서로 탐색하여
 *   처음 발견한 Grant 출력을 수락하고 _inmatch/_outmatch에 매칭을 기록한다.
 *
 * 포인터 갱신 없이 매 사이클 완전 무작위 — 공정성은 확률적으로만 보장.
 * 호출 체인: IQRouter::_SWAllocEvaluate() → PIM::Allocate()
 */
void PIM::Allocate( )
{
  int input;   // [한국어] 현재 처리 중인 입력 포트
  int output;  // [한국어] 현재 처리 중인 출력 포트

  int input_offset;  // [한국어] Grant 단계 무작위 탐색 시작 위치
  int output_offset; // [한국어] Accept 단계 무작위 탐색 시작 위치

  for ( int iter = 0; iter < _PIM_iter; ++iter ) { // [한국어] 최대 _PIM_iter 반복
    // Grant phase --- outputs randomly choose
    // between one of their requests
    // [한국어] Grant 단계: 각 출력이 요청 중 미매칭 입력을 무작위로 선택

    vector<int> grants(_outputs, -1); // [한국어] 이번 반복 Grant 결과. -1=미승인

    for ( output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력 포트 순회

      // A random arbiter between input requests
      input_offset  = RandomInt( _inputs - 1 ); // [한국어] 무작위 탐색 시작 위치 결정

      for ( int i = 0; i < _inputs; ++i ) {          // [한국어] _inputs개 입력을 오프셋부터 순환
	input = ( i + input_offset ) % _inputs;        // [한국어] 오프셋 적용 후 실제 입력 번호

	if ( ( _request[input][output].label != -1 ) && // [한국어] 이 입력이 이 출력을 요청하는가?
	     ( _inmatch[input] == -1 ) &&               // [한국어] 이 입력이 아직 미매칭인가?
	     ( _outmatch[output] == -1 ) ) {            // [한국어] 이 출력이 아직 미매칭인가?

	  // Grant
	  grants[output] = input; // [한국어] 이 출력이 이 입력에 Grant 결정
	  break;                  // [한국어] 출력당 하나의 입력에만 Grant — 탐색 종료
	}
      }
    }

    // Accept phase -- inputs randomly choose
    // between input_speedup of their grants
    // [한국어] Accept 단계: 각 입력이 Grant받은 출력 중 하나를 무작위로 수락

    for ( input = 0; input < _inputs; ++input ) { // [한국어] 모든 입력 포트 순회

      // A random arbiter between output grants
      output_offset  = RandomInt( _outputs - 1 ); // [한국어] 무작위 탐색 시작 위치 결정

      for ( int o = 0; o < _outputs; ++o ) {          // [한국어] _outputs개 출력을 오프셋부터 순환
	output = ( o + output_offset ) % _outputs;     // [한국어] 오프셋 적용 후 실제 출력 번호

	if ( grants[output] == input ) { // [한국어] 이 출력이 이 입력에 Grant했는가?

	  // Accept
	  // [한국어] 이 (input, output) 쌍 매칭 확정
	  _inmatch[input]   = output; // [한국어] 입력 측 매칭 결과 기록
	  _outmatch[output] = input;  // [한국어] 출력 측 매칭 결과 기록

	  break; // [한국어] 입력당 하나의 출력만 수락 — 탐색 종료
	}
      }
    }
  } // [한국어] 반복 종료

#ifdef DEBUG_PIM
  if ( _outputs == 8 ) {
    cout << "input match: " << endl;
    for ( int i = 0; i < _inputs; ++i ) {
      cout << "  from " << i << " to " << _inmatch[i] << endl;
    }
    cout << endl;
  }

  cout << "output match: ";
  for ( int j = 0; j < _outputs; ++j ) {
    cout << _outmatch[j] << " ";
  }
  cout << endl;
#endif
}


