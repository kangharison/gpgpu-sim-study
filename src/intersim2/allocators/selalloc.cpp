// $Id: selalloc.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] SelAlloc 우선순위 선택 할당기 구현 (selalloc.cpp)
 *
 * === 파일의 역할 ===
 * SelAlloc의 생성자, Allocate(), MaskOutput(), PrintRequests()를 구현한다.
 * iSLIP과 구조는 같지만 Grant와 Accept 단계에서 라운드로빈 탐색 중
 * 최고 우선순위(max_pri)를 가진 요청을 선택하는 점이 다르다.
 * MaskOutput()으로 특정 출력(예: 오류 링크)을 할당에서 제외할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → SelAlloc::Allocate()
 *
 * === 타 모듈과의 연결 ===
 * - SparseAllocator: 상속 (_in_req, _out_req, _in_occ, _out_occ)
 *
 * === 주요 함수/구조체 요약 ===
 * - SelAlloc(): _gptrs, _aptrs 0으로 초기화, _outmask 0으로 초기화
 * - Allocate(): 반복 Grant(우선순위 최대 입력 선택) + Accept(우선순위 최대 출력 선택)
 * - MaskOutput(): _outmask[out] = mask 설정
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

#include "selalloc.hpp"
#include "random_utils.hpp"

//#define DEBUG_SELALLOC

/*
 * [한국어] SelAlloc 생성자 — 반복 횟수, 포인터, 마스크 벡터 초기화
 * @iters: Grant-Accept 최대 반복 횟수 (설정 파일 "alloc_iters")
 * 호출 체인: Allocator::NewAllocator("select") → SelAlloc() → SparseAllocator()
 */
SelAlloc::SelAlloc( Module *parent, const string& name,
		    int inputs, int outputs, int iters ) :
  SparseAllocator( parent, name, inputs, outputs )
{
  _iter = iters;                  // [한국어] Grant-Accept 반복 횟수 저장

  _gptrs.resize(outputs, 0);     // [한국어] Grant 포인터: 출력 수만큼, 초기값 0
  _aptrs.resize(inputs, 0);      // [한국어] Accept 포인터: 입력 수만큼, 초기값 0
  _outmask.resize(outputs, 0);   // [한국어] 출력 마스크: 출력 수만큼, 초기값 0(마스크 없음)
}

/*
 * [한국어] SelAlloc::Allocate - 우선순위 기반 반복 Grant-Accept 크로스바 매칭
 *
 * iSLIP과 동일한 2단계 반복 구조이지만, Grant/Accept 단계에서 라운드로빈 탐색 중
 * 가장 높은 우선순위(out_pri/in_pri)를 가진 요청을 선택한다.
 *
 * Grant 단계: _gptrs 오프셋부터 순환 탐색하며 미매칭 입력 중 out_pri 최대인 것을 선택.
 * Accept 단계: _aptrs 오프셋부터 순환 탐색하며 Grant된 출력 중 in_pri 최대인 것을 수락.
 * 첫 번째 반복(iter==0)에서 수락된 경우에만 _gptrs/_aptrs 포인터를 갱신한다.
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → SelAlloc::Allocate()
 */
void SelAlloc::Allocate( )
{
  int input;         // [한국어] 현재 처리 중인 입력 포트
  int output;        // [한국어] 현재 처리 중인 출력 포트

  int input_offset;  // [한국어] Grant 단계 탐색 시작 위치
  int output_offset; // [한국어] Accept 단계 탐색 시작 위치

  map<int, sRequest>::iterator p; // [한국어] 요청 맵 순회 이터레이터
  set<int>::iterator outer_iter;  // [한국어] occupied 집합 순회 이터레이터
  bool wrapped;                   // [한국어] 순환 발생 여부 플래그

  int max_index; // [한국어] 현재까지 발견한 최고 우선순위 요청의 포트 번호
  int max_pri;   // [한국어] 현재까지의 최고 우선순위 값

  vector<int> grants(_outputs, -1); // [한국어] 이번 반복 Grant 결과 배열. -1=미승인

  for ( int iter = 0; iter < _iter; ++iter ) { // [한국어] 최대 _iter회 반복
    // Grant phase
    // [한국어] Grant 단계: 각 활성 출력이 최고 우선순위 미매칭 입력을 선택

    for( outer_iter = _out_occ.begin( );  // [한국어] 활성 출력 집합 순회
	 outer_iter != _out_occ.end( ); ++outer_iter ) {
      output = *outer_iter; // [한국어] 현재 출력 포트

      // Skip loop if there are no requests
      // or the output is already matched or
      // the output is masked
      if ( ( _out_req[output].empty( ) ) || // [한국어] 이 출력에 요청 없음 — 건너뜀
	   ( _outmatch[output] != -1 ) ||   // [한국어] 이미 매칭됨 — 건너뜀
	   ( _outmask[output] != 0 ) ) {    // [한국어] 마스크됨 — 건너뜀 (링크 오류 등)
	continue;
      }

      // A round-robin arbiter between input requests
      input_offset = _gptrs[output]; // [한국어] Grant 탐색 시작 위치 (라운드로빈 포인터)

      p = _out_req[output].begin( ); // [한국어] 맵 처음부터 시작
      while( ( p != _out_req[output].end( ) ) &&
	     ( p->second.port < input_offset ) ) { // [한국어] 오프셋 이전 항목 건너뜀
	p++;
      }

      max_index = -1; // [한국어] 아직 후보 없음
      max_pri   = 0;  // [한국어] 우선순위 초기값

      // [한국어] 오프셋부터 순환 탐색하며 최고 out_pri 찾기
      wrapped = false;
      while( (!wrapped) ||
	     ( ( p != _out_req[output].end() ) &&
	       ( p->second.port < input_offset ) ) ) {
	if ( p == _out_req[output].end( ) ) {
	  if ( wrapped ) { break; }           // [한국어] 순환 완료 — 종료
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _out_req[output].begin( );      // [한국어] 처음으로 감기
	  wrapped = true;
	}

	input = p->second.port; // [한국어] 이 요청의 입력 포트

	// we know the output is free (above) and
	// if the input is free, check if request is the
	// highest priority so far
	if ( ( _inmatch[input] == -1 ) &&    // [한국어] 미매칭 입력만 후보
	     ( ( p->second.out_pri > max_pri ) || ( max_index == -1 ) ) ) {
	  // [한국어] out_pri가 현재 최대이거나 첫 번째 후보이면 업데이트
	  max_pri   = p->second.out_pri; // [한국어] 최고 out_pri 갱신
	  max_index = input;             // [한국어] 최고 우선순위 입력 갱신
	}

	p++; // [한국어] 다음 요청 탐색
      }

      if ( max_index != -1 ) { // grant
	grants[output] = max_index; // [한국어] 최고 우선순위 입력에게 Grant
      }
    }

#ifdef DEBUG_SELALLOC
    cout << "grants: ";
    for ( int i = 0; i < _outputs; ++i ) {
      cout << grants[i] << " ";
    }
    cout << endl;

    cout << "aptrs: ";
    for ( int i = 0; i < _inputs; ++i ) {
      cout << _aptrs[i] << " ";
    }
    cout << endl;
#endif

    // Accept phase
    // [한국어] Accept 단계: 각 활성 입력이 Grant받은 출력 중 최고 in_pri를 수락

    for ( outer_iter = _in_occ.begin( );   // [한국어] 활성 입력 집합 순회
	  outer_iter != _in_occ.end( ); ++outer_iter ) {
      input = *outer_iter; // [한국어] 현재 입력 포트

      if ( _in_req[input].empty( ) ) { // [한국어] 요청 없으면 건너뜀
	continue;
      }

      // A round-robin arbiter between output grants
      output_offset = _aptrs[input]; // [한국어] Accept 탐색 시작 위치

      p = _in_req[input].begin( ); // [한국어] 맵 처음부터
      while( ( p != _in_req[input].end( ) ) &&
	     ( p->second.port < output_offset ) ) { // [한국어] 오프셋 이전 건너뜀
	p++;
      }

      max_index = -1; // [한국어] 아직 최선 후보 없음
      max_pri   = 0;  // [한국어] 우선순위 초기값

      // [한국어] 오프셋부터 순환 탐색하며 최고 in_pri Grant를 찾기
      wrapped = false;
      while( (!wrapped) ||
	     ( ( p != _in_req[input].end() ) &&
	       ( p->second.port < output_offset ) ) ) {
	if ( p == _in_req[input].end( ) ) {
	  if ( wrapped ) { break; } // [한국어] 순환 완료
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _in_req[input].begin( ); // [한국어] 처음으로 감기
	  wrapped = true;
	}

	output = p->second.port; // [한국어] 이 요청의 출력 포트

	// we know the output is free (above) and
	// if the input is free, check if the highest
	// priroity
	if ( ( grants[output] == input ) &&          // [한국어] 이 출력이 이 입력에 Grant했는지 확인
	     ( !_out_req[output].empty( ) ) &&       // [한국어] 이 출력에 요청이 있어야 함
	     ( ( p->second.in_pri > max_pri ) || ( max_index == -1 ) ) ) {
	  // [한국어] in_pri가 현재 최대이거나 첫 번째 후보이면 갱신
	  max_pri   = p->second.in_pri; // [한국어] 최고 in_pri 갱신
	  max_index = output;           // [한국어] 최고 우선순위 출력 갱신
	}

	p++; // [한국어] 다음 요청 탐색
      }

      if ( max_index != -1 ) {
	// Accept
	output = max_index; // [한국어] 수락할 출력 결정

	_inmatch[input]   = output; // [한국어] 입력 측 매칭 결과 기록
	_outmatch[output] = input;  // [한국어] 출력 측 매칭 결과 기록

	// Only update pointers if accepted during the 1st iteration
	if ( iter == 0 ) { // [한국어] 첫 번째 반복에서만 포인터 갱신 (이후 반복은 공정성 유지)
	  _gptrs[output] = ( input + 1 ) % _inputs;   // [한국어] Grant 포인터 전진
	  _aptrs[input]  = ( output + 1 ) % _outputs; // [한국어] Accept 포인터 전진
	}
      }
    }
  } // [한국어] 반복 종료

#ifdef DEBUG_SELALLOC
  cout << "input match: ";
  for ( int i = 0; i < _inputs; ++i ) {
    cout << _inmatch[i] << " ";
  }
  cout << endl;

  cout << "output match: ";
  for ( int j = 0; j < _outputs; ++j ) {
    cout << _outmatch[j] << " ";
  }
  cout << endl;
#endif 
}

/*
 * [한국어] SelAlloc::MaskOutput - 특정 출력 포트를 이번 사이클 할당에서 제외
 * @out: 마스크할 출력 포트 번호
 * @mask: 마스크 값 (기본값 1 = 마스크 활성, 0 = 정상)
 * 링크 오류나 특정 출력을 강제로 유휴 상태로 만들 때 사용한다.
 */
void SelAlloc::MaskOutput( int out, int mask )
{
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 범위 검사
  _outmask[out] = mask; // [한국어] 0이 아닌 값이면 Allocate() Grant 단계에서 이 출력을 건너뜀
}

void SelAlloc::PrintRequests( ostream * os ) const
{
  map<int, sRequest>::const_iterator iter;
  
  if(!os) os = &cout;
  
  *os << "Input requests = [ ";
  for ( int input = 0; input < _inputs; ++input ) {
    *os << input << " -> [ ";
    for ( iter = _in_req[input].begin( ); 
	  iter != _in_req[input].end( ); iter++ ) {
      *os << iter->second.port << " ";
    }
    *os << "]  ";
  }
  *os << "], output requests = [ ";
  for ( int output = 0; output < _outputs; ++output ) {
    *os << output << " -> ";
    if ( _outmask[output] == 0 ) {
      *os << "[ ";
      for ( iter = _out_req[output].begin( ); 
	    iter != _out_req[output].end( ); iter++ ) {
	*os << iter->second.port << " ";
      }
      *os << "]  ";
    } else {
      *os << "masked  ";
    }
  }
  *os << "]." << endl;
}

