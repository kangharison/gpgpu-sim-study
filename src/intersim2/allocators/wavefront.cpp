// $Id: wavefront.cpp 5262 2012-09-20 23:39:40Z dub $

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

/*wavefront.cpp
 *
 * [한국어 설명] Wavefront 크로스바 할당기 구현 (wavefront.cpp)
 *
 * === 파일의 역할 ===
 * Wavefront 알고리즘을 구현한다. 요청 행렬의 대각선(diagonal)을 _pri 포인터에서 시작하여
 * 순환 탐색하고, 각 대각선에서 충돌 없이 최대한 많은 (input, output) 쌍을 매칭한다.
 * 요청 수가 0이면 즉시 리턴, 1이면 즉시 매칭, 2 이상이면 대각선 순환 탐색 수행.
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → Wavefront::Allocate()
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 — _request[in][out].label로 요청 확인
 *
 * === 주요 함수/구조체 요약 ===
 * - Wavefront(): _pri=0, _num_requests=0으로 초기화
 * - AddRequest(): 요청 등록 + _priorities에 (out_pri, in_pri) 삽입
 * - Allocate(): 0/1/다수 요청을 분기하여 대각선 탐색 후 _pri 갱신
 *
 *The wave front allocator
 *
 */
#include "booksim.hpp"

#include "wavefront.hpp"

/*
 * [한국어] Wavefront 생성자 — 상태 변수 초기화
 * _pri=0(대각선 0부터 시작), _num_requests=0(요청 없음), _last_in/out=-1(미설정)
 * skip_diags=false: 고정 wavefront, true: rr_wavefront(라운드로빈 대각선)
 * 호출 체인: Allocator::NewAllocator("wavefront"/"rr_wavefront") → Wavefront()
 */
Wavefront::Wavefront( Module *parent, const string& name,
		      int inputs, int outputs, bool skip_diags ) :
  DenseAllocator( parent, name, inputs, outputs ),   // [한국어] DenseAllocator 초기화 (요청 행렬)
  _last_in(-1), _last_out(-1), _skip_diags(skip_diags), // [한국어] 마지막 요청 포트, 모드 설정
  _square(max(inputs, outputs)), _pri(0), _num_requests(0) // [한국어] 정방 크기, 대각선 포인터, 요청 수
{
}

/*
 * [한국어] Wavefront::AddRequest — DenseAllocator 등록 후 상태 변수 갱신
 * DenseAllocator::AddRequest()로 실제 요청을 _request 배열에 저장하고,
 * _num_requests를 증가시키고, _last_in/_last_out을 갱신하고,
 * (out_pri, in_pri) 쌍을 _priorities 집합에 삽입한다.
 * 호출 체인: IQRouter → Wavefront::AddRequest()
 */
void Wavefront::AddRequest( int in, int out, int label,
			    int in_pri, int out_pri )
{
  DenseAllocator::AddRequest(in, out, label, in_pri, out_pri); // [한국어] _request[in][out] 등록
  _num_requests++;                                              // [한국어] 요청 수 증가
  _last_in = in;                                               // [한국어] 1개 요청 최적화를 위한 마지막 입력 저장
  _last_out = out;                                             // [한국어] 마지막 출력 저장
  _priorities.insert(make_pair(out_pri, in_pri));              // [한국어] 우선순위 쌍 등록 (set이라 자동 정렬)
}

/*
 * [한국어] Wavefront::Allocate — 대각선 탐색으로 충돌 없는 최대 매칭 결정
 *
 * 요청 수에 따라 세 가지 경로로 분기:
 * 1. _num_requests == 0: 요청 없음 — 즉시 리턴 (아무것도 하지 않음)
 * 2. _num_requests == 1: 요청이 1개뿐 — 무조건 Grant (대각선 탐색 불필요)
 * 3. _num_requests >= 2: _priorities를 역순으로 순회하며 우선순위별로 대각선 탐색
 *
 * 대각선 탐색: 대각선 d는 input + output = d (mod _square)인 쌍들의 집합.
 *   p=0부터 _square-1까지: 대각선 번호 = (_pri + p) % _square
 *   각 대각선에서 output=0부터 _square-1까지 순환하며:
 *     input = (_pri + p + _square - output) % _square
 *   이 (input, output)이 유효하고 미매칭이고 요청이 있으면 매칭.
 *
 * Allocate() 후 _pri 갱신:
 *   _skip_diags=false: _pri = (_pri + 1) % _square (단순 1 증가)
 *   _skip_diags=true:  _pri = (first_diag + 1) % _square (첫 매칭 대각선 다음으로 점프)
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → Wavefront::Allocate()
 */
void Wavefront::Allocate( )
{

  int first_diag = -1; // [한국어] 첫 번째 매칭이 발생한 대각선 번호 (-1이면 아직 매칭 없음)

  if(_num_requests == 0)
    // bypass allocator completely if there were no requests
    return; // [한국어] 요청이 없으면 아무것도 하지 않고 즉시 리턴

  if(_num_requests == 1) {

    // if we only had a single request, we can immediately grant it
    // [한국어] 요청이 1개뿐 — 대각선 탐색 없이 즉시 _last_in → _last_out 매칭
    _inmatch[_last_in] = _last_out;   // [한국어] 입력 측 매칭 기록
    _outmatch[_last_out] = _last_in;  // [한국어] 출력 측 매칭 기록
    first_diag = _last_in + _last_out; // [한국어] 이 요청의 대각선 번호 계산

  } else {

    // otherwise we have to loop through the diagonals of request matrix
    // [한국어] 요청이 2개 이상 — 우선순위 내림차순으로 대각선 탐색

    for(set<pair<int, int> >::const_reverse_iterator iter =
	  _priorities.rbegin(); // [한국어] 높은 (out_pri, in_pri) 쌍부터 역순 순회
	iter != _priorities.rend(); ++iter) { // [한국어] 모든 우선순위 레벨 처리

      for ( int p = 0; p < _square; ++p ) { // [한국어] _pri에서 시작하여 _square개 대각선 순환
	for ( int output = 0; output < _square; ++output ) { // [한국어] 각 대각선 내 출력 순회
	  int input = ( ( _pri + p ) + ( _square - output ) ) % _square;
	  // [한국어] 대각선 (_pri+p)에서 output에 대응하는 input 계산
	  // input + output ≡ _pri + p (mod _square)를 만족
	  if ( ( input < _inputs ) && ( output < _outputs ) && // [한국어] 유효 범위 내 확인
	       ( _inmatch[input] == -1 ) && ( _outmatch[output] == -1 ) && // [한국어] 미매칭 확인
	       ( _request[input][output].label != -1 ) &&  // [한국어] 요청 존재 확인
	       ( _request[input][output].in_pri == iter->second ) && // [한국어] in_pri 일치
	       ( _request[input][output].out_pri == iter->first ) ) { // [한국어] out_pri 일치
	    // Grant!
	    _inmatch[input] = output;  // [한국어] 매칭 기록 — 입력 측
	    _outmatch[output] = input; // [한국어] 매칭 기록 — 출력 측
	    if(first_diag < 0) { // [한국어] 첫 번째 매칭 대각선 번호 기록
	      first_diag = input + output; // [한국어] 이 (input, output)의 대각선 번호
	    }
	  }
	}
      }
    }
  }

  _num_requests = 0;     // [한국어] 요청 수 리셋 (다음 사이클 대비)
  _last_in = -1;         // [한국어] 마지막 요청 포트 리셋
  _last_out = -1;        // [한국어] 마지막 요청 포트 리셋
  _priorities.clear();   // [한국어] 우선순위 집합 초기화

  assert(first_diag >= 0); // [한국어] 요청이 있었으면 반드시 매칭이 하나 이상 있어야 함

  // Round-robin the priority diagonal
  // [한국어] _skip_diags 모드에 따라 다음 사이클의 시작 대각선 번호 갱신
  _pri = ( ( _skip_diags ? first_diag : _pri ) + 1 ) % _square;
  // [한국어] false(wavefront): (_pri+1) % _square — 단순 1 증가
  // [한국어] true(rr_wavefront): (first_diag+1) % _square — 첫 매칭 대각선 다음으로 점프
}


