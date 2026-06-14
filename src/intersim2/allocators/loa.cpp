// $Id: loa.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] LOA(Lonely Output Arbiter) 할당기 구현 (loa.cpp)
 *
 * === 파일의 역할 ===
 * LOA::Allocate()를 구현한다. 3단계로 동작:
 * 1. Count: 각 출력에 대한 요청 수를 _counts[j]에 집계
 * 2. Request: 각 입력이 _rptr[input] 오프셋 기준으로 _counts가 가장 낮은 출력을 선택
 * 3. Grant: 각 출력이 자신을 선택한 입력 중 _gptr[output] 기준 라운드로빈으로 하나를 Grant
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → LOA::Allocate()
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 (_request 2D 배열)
 *
 * === 주요 함수/구조체 요약 ===
 * - LOA(): _counts, _req, _rptr, _gptr 벡터 inputs/outputs 크기로 초기화
 * - Allocate(): Count → Request → Grant 3단계 LOA 알고리즘
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

#include "loa.hpp"
#include "random_utils.hpp"

/*
 * [한국어] LOA 생성자 — 내부 상태 벡터 초기화
 * @parent, @name: 부모/이름
 * @inputs, @outputs: 입출력 포트 수
 * _req, _counts, _rptr, _gptr를 inputs 또는 outputs 크기로 초기화(값은 미초기화).
 * 호출 체인: Allocator::NewAllocator("loa") → LOA() → DenseAllocator()
 */
LOA::LOA( Module *parent, const string& name,
	  int inputs, int outputs ) :
  DenseAllocator( parent, name, inputs, outputs )
{
  _req.resize(inputs);     // [한국어] 각 입력이 선택한 외로운 출력 저장 공간
  _counts.resize(outputs); // [한국어] 각 출력에 대한 요청 수 카운터

  _rptr.resize(inputs);    // [한국어] 입력별 Request 단계 탐색 오프셋 (초기값 미설정 — Allocate에서 사용)
  _gptr.resize(outputs);   // [한국어] 출력별 Grant 단계 탐색 오프셋
}

/*
 * [한국어] LOA::Allocate — Count → Request → Grant 3단계 LOA 알고리즘
 *
 * 1단계 Count: 출력 j에 요청하는 입력 수를 _counts[j]에 계산 (O(outputs × inputs))
 * 2단계 Request: 각 입력이 자신의 요청 중 _counts가 최소인 출력(외로운 출력)을 선택
 *   - _rptr[input] 오프셋부터 _outputs개 출력을 순환 탐색
 *   - _request[input][output].label != -1이고 _counts[output] < lonely_cnt인 최소값 선택
 * 3단계 Grant: 각 출력이 _req[input]==output인 입력 중 _gptr[output] 기준 라운드로빈으로 1개 선택
 *   - Grant 성공 시 _rptr[input]과 _gptr[output] 모두 1 전진 (공정성 보장)
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → LOA::Allocate()
 */
void LOA::Allocate( )
{
  int input;        // [한국어] 현재 처리 중인 입력 포트
  int output;       // [한국어] 현재 처리 중인 출력 포트

  int input_offset;  // [한국어] Grant 단계 탐색 시작 오프셋
  int output_offset; // [한국어] Request 단계 탐색 시작 오프셋

  int lonely;     // [한국어] 현재 입력이 선택한 외로운 출력 번호
  int lonely_cnt; // [한국어] 현재까지 발견한 최소 요청 수

  // Count phase --- the number of requests
  // per output is counted
  // [한국어] 1단계 Count: 각 출력을 요청하는 입력의 수를 집계

  for ( int j = 0; j < _outputs; ++j ) { // [한국어] 모든 출력 포트 순회
    _counts[j] = 0;                       // [한국어] 카운터 초기화
    for ( int i = 0; i < _inputs; ++i ) { // [한국어] 모든 입력 포트 순회
      _counts[j] += ( _request[i][j].label != -1 ) ? 1 : 0;
      // [한국어] 출력 j를 요청하는 입력이면 카운트 증가
    }
  }

  // Request phase
  // [한국어] 2단계 Request: 각 입력이 가장 외로운(요청이 적은) 출력을 선택
  for ( input = 0; input < _inputs; ++input ) {

    // Find the lonely output
    output_offset = _rptr[input]; // [한국어] 탐색 시작 위치 (동점 시 공정성을 위한 오프셋)
    lonely        = -1;           // [한국어] 아직 외로운 출력을 찾지 못함
    lonely_cnt    = _inputs + 1;  // [한국어] 최솟값 초기화 (_inputs+1 = 불가능한 큰 값)

    for ( int o = 0; o < _outputs; ++o ) {       // [한국어] _outputs개 출력을 오프셋부터 순환 탐색
      output = ( o + output_offset ) % _outputs; // [한국어] 오프셋 적용 후 실제 출력 번호

      if ( ( _request[input][output].label != -1 ) && // [한국어] 이 입력이 이 출력을 요청하는가?
	   ( _counts[output] < lonely_cnt ) ) {       // [한국어] 현재까지 발견한 최소 요청 수보다 적은가?
	lonely = output;                              // [한국어] 더 외로운 출력 발견 — 갱신
	lonely_cnt = _counts[output];                 // [한국어] 최솟값 갱신
      }
    }

    // Request the lonely output (-1 for no request)
    _req[input] = lonely; // [한국어] 외로운 출력 번호 저장 (-1이면 요청할 외로운 출력 없음)
  }

  // Grant phase
  // [한국어] 3단계 Grant: 각 출력이 자신을 선택한 입력 중 라운드로빈으로 1개를 Grant
  for ( output = 0; output < _outputs; ++output ) {
    input_offset = _gptr[output]; // [한국어] Grant 탐색 시작 위치 (라운드로빈 공정성)

    for ( int i = 0; i < _inputs; ++i ) {            // [한국어] _inputs개 입력을 오프셋부터 순환 탐색
      input = ( i + input_offset ) % _inputs;         // [한국어] 오프셋 적용 후 실제 입력 번호

      if ( _req[input] == output ) { // [한국어] 이 입력이 이 출력을 외로운 출력으로 선택했는가?
	// Grant!
	// [한국어] 이 (input, output) 쌍 매칭 결정
	_inmatch[input]   = output; // [한국어] 입력 측 매칭 결과 기록
	_outmatch[output] = input;  // [한국어] 출력 측 매칭 결과 기록

	_rptr[input] = ( _rptr[input] + 1 ) % _outputs; // [한국어] Request 포인터 전진 (다음 사이클 공정성)
	_gptr[output] = ( _gptr[output] + 1 ) % _inputs; // [한국어] Grant 포인터 전진

	break; // [한국어] 이 출력은 하나의 입력에만 Grant — 다음 출력으로
      }
    }
  }

}


