// $Id: islip.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] iSLIP 크로스바 할당기 구현 (islip.cpp)
 *
 * === 파일의 역할 ===
 * iSLIP(Iterative Schedule with LInear Priority) 알고리즘을 구현한다.
 * Grant 단계: 각 출력이 _gptrs 기준 라운드로빈으로 요청 중인 미매칭 입력을 하나 선택.
 * Accept 단계: 각 입력이 _aptrs 기준 라운드로빈으로 자신을 Grant한 출력 중 하나를 수락.
 * 이 두 단계를 _iSLIP_iter회 반복하여 최대 가중 이분 매칭에 수렴한다.
 * GPGPU-Sim GPU NoC에서 스위치 할당의 기본 알고리즘이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 IQRouter 내부 스위치 할당 단계. 매 사이클 IQRouter::_SWAllocEvaluate()가
 * Allocate()를 호출하며, 결과는 _inmatch/_outmatch에 기록된다.
 * 실행 컨텍스트: GPGPU-Sim 타이밍 사이클 루프(단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * - SparseAllocator: 상속 — _in_req, _out_req, _in_occ, _out_occ, _inmatch, _outmatch 사용
 * - IQRouter: _sw_allocator, _vc_allocator로 보유
 * - random_utils.hpp: 이 파일에서는 #include하지만 iSLIP 알고리즘 자체는 랜덤 미사용
 *
 * === 주요 함수/구조체 요약 ===
 * - iSLIP_Sparse(): _gptrs를 outputs 크기, _aptrs를 inputs 크기로 0 초기화
 * - Allocate(): Grant + Accept 2단계를 _iSLIP_iter회 반복, 포인터는 첫 반복 성공 시만 갱신
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

#include "booksim.hpp"       // [한국어] using namespace std; 및 공통 헤더
#include <iostream>          // [한국어] DEBUG_ISLIP 시 cout 사용

#include "islip.hpp"         // [한국어] iSLIP_Sparse 클래스 선언
#include "random_utils.hpp"  // [한국어] #include되어 있으나 iSLIP 자체는 랜덤 미사용 (PIM과 달리)

//#define DEBUG_ISLIP  // [한국어] 이 매크로를 활성화하면 Grant/Accept 결과를 cout으로 출력

/*
 * [한국어] iSLIP_Sparse 생성자 — Grant/Accept 라운드로빈 포인터 초기화
 *
 * @parent: 부모 모듈 (IQRouter)
 * @name: 디버그용 이름 문자열
 * @inputs: 입력 포트 수
 * @outputs: 출력 포트 수
 * @iters: iSLIP 반복 횟수
 *
 * _gptrs[output]=0, _aptrs[input]=0으로 초기화 — 모든 포인터가 포트 0부터 시작.
 * 포인터는 매칭 성공 시 갱신되며 사이클 간에 유지되어 공정성을 보장한다.
 *
 * 호출 체인: Allocator::NewAllocator("islip") → iSLIP_Sparse() → SparseAllocator()
 */
iSLIP_Sparse::iSLIP_Sparse( Module *parent, const string& name,
			    int inputs, int outputs, int iters ) :
  SparseAllocator( parent, name, inputs, outputs ), // [한국어] SparseAllocator 초기화 (_in_req 등)
  _iSLIP_iter(iters)                                // [한국어] 반복 횟수 설정
{
  _gptrs.resize(_outputs, 0); // [한국어] Grant 포인터: 출력 수만큼 할당, 초기값 0(입력 0부터 탐색)
  _aptrs.resize(_inputs, 0);  // [한국어] Accept 포인터: 입력 수만큼 할당, 초기값 0(출력 0부터 탐색)
}

/*
 * [한국어] iSLIP_Sparse::Allocate - iSLIP 크로스바 할당 알고리즘 실행
 *
 * @return: 없음 (결과는 _inmatch[input] = output, _outmatch[output] = input에 기록)
 *
 * _iSLIP_iter회 반복하며 Grant와 Accept 두 단계를 수행한다.
 *
 * [Grant 단계]: 각 출력 포트가 _gptrs[output] 기준 라운드로빈으로 미매칭 입력 하나를 선택.
 *   - 이미 매칭된 출력(_outmatch != -1)은 건너뜀.
 *   - _out_req[output] 맵을 _gptrs[output] 오프셋부터 순환 탐색.
 *   - 미매칭 입력(_inmatch == -1)을 발견하면 grants[output] = input으로 임시 승인.
 *
 * [Accept 단계]: 각 입력 포트가 _aptrs[input] 기준 라운드로빈으로 Grant된 출력 중 하나를 수락.
 *   - _in_req[input] 맵을 _aptrs[input] 오프셋부터 순환 탐색.
 *   - grants[output] == input인 경우에만 수락 가능.
 *   - 수락 시 _inmatch[input] = output, _outmatch[output] = input 기록.
 *   - 포인터 갱신은 첫 번째 반복(iter==0) 성공 시에만 수행 — 이후 반복에서 공정성 유지.
 *
 * 실행 컨텍스트: GPGPU-Sim 타이밍 사이클 루프 내 단일 스레드. 재진입 불가.
 *
 * 호출 체인:
 *   IQRouter::_SWAllocEvaluate() → iSLIP_Sparse::Allocate()
 *                                 → (결과) IQRouter::_SwitchEvaluate() 에서 OutputAssigned() 조회
 */
void iSLIP_Sparse::Allocate( )
{
  int input;        // [한국어] 현재 처리 중인 입력 포트 번호
  int output;       // [한국어] 현재 처리 중인 출력 포트 번호

  int input_offset; // [한국어] Grant 단계 라운드로빈 시작 오프셋 (_gptrs[output])
  int output_offset;// [한국어] Accept 단계 라운드로빈 시작 오프셋 (_aptrs[input])

  map<int, sRequest>::iterator p; // [한국어] _in_req/_out_req 맵 순회 이터레이터
  bool wrapped;     // [한국어] 라운드로빈 순환(wraparound) 발생 여부

  for ( int iter = 0; iter < _iSLIP_iter; ++iter ) { // [한국어] 최대 _iSLIP_iter회 반복
    // Grant phase
    // [한국어] Grant 단계: 각 출력이 자신을 요청하는 미매칭 입력 중 하나를 라운드로빈으로 선택

    vector<int> grants(_outputs, -1); // [한국어] 이번 반복에서 각 출력이 Grant한 입력 (-1 = 미승인)

    for ( output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력 포트에 대해 Grant 시도

      // Skip loop if there are no requests
      // or the output is already matched
      if ( ( _out_req[output].empty( ) ) ||    // [한국어] 이 출력을 요청하는 입력이 없으면 건너뜀
	   ( _outmatch[output] != -1 ) ) {     // [한국어] 이미 이전 반복에서 매칭됐으면 건너뜀
	continue;
      }

      // A round-robin arbiter between input requests
      input_offset = _gptrs[output]; // [한국어] 이 출력의 Grant 시작 위치 (_gptrs로 라운드로빈 공정성 구현)

      p = _out_req[output].begin( ); // [한국어] _out_req[output] 맵의 첫 항목부터 시작
      while( ( p != _out_req[output].end( ) ) &&
	     ( p->second.port < input_offset ) ) { // [한국어] input_offset 이전 항목들 건너뜀
	p++;
      }

      // [한국어] input_offset부터 순환 탐색 — wrapped 플래그로 무한 루프 방지
      wrapped = false;
      while( (!wrapped) ||
	     ( ( p != _out_req[output].end( ) ) &&
	       ( p->second.port < input_offset ) ) ) { // [한국어] 순환 후 input_offset 이전 항목 허용
	if ( p == _out_req[output].end( ) ) {
	  if ( wrapped ) { break; } // [한국어] 이미 한 바퀴 순환했으면 종료
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _out_req[output].begin( ); // [한국어] 맵 끝 도달 — 처음으로 감기 (wraparound)
	  wrapped = true;                // [한국어] 순환 발생 표시
	}

	input = p->second.port; // [한국어] 이 요청의 입력 포트 번호

	// we know the output is free (above) and
	// if the input is free, grant request
	if ( _inmatch[input] == -1 ) { // [한국어] 이 입력이 아직 매칭되지 않았으면 Grant
	  grants[output] = input;      // [한국어] grants[output] = input으로 임시 승인 기록
	  break;                       // [한국어] Grant 완료 — 다음 출력으로 이동
	}

	p++; // [한국어] 이 입력은 이미 매칭됨 — 다음 요청 탐색
      }
    }

#ifdef DEBUG_ISLIP
    cout << "grants: ";
    for ( int i = 0; i < _outputs; ++i ) { // [한국어] 이번 반복의 Grant 결과 출력
      cout << grants[i] << " ";
    }
    cout << endl;

    cout << "aptrs: ";
    for ( int i = 0; i < _inputs; ++i ) { // [한국어] 현재 Accept 포인터 상태 출력
      cout << _aptrs[i] << " ";
    }
    cout << endl;
#endif

    // Accept phase
    // [한국어] Accept 단계: 각 입력이 Grant된 출력 중 라운드로빈으로 하나를 수락

    for ( input = 0; input < _inputs; ++input ) { // [한국어] 모든 입력 포트에 대해 Accept 시도

      if ( _in_req[input].empty( ) ) { // [한국어] 이 입력이 요청을 보내지 않았으면 건너뜀
	continue;
      }

      // A round-robin arbiter between output grants
      output_offset = _aptrs[input]; // [한국어] Accept 시작 위치 (_aptrs로 공정성 보장)

      p = _in_req[input].begin( ); // [한국어] _in_req[input] 맵의 첫 항목부터
      while( ( p != _in_req[input].end( ) ) &&
	     ( p->second.port < output_offset ) ) { // [한국어] output_offset 이전 건너뜀
	p++;
      }

      // [한국어] output_offset부터 순환 탐색
      wrapped = false;
      while( (!wrapped) ||
	     ( ( p != _in_req[input].end( ) ) &&
	       ( p->second.port < output_offset ) ) ) {
	if ( p == _in_req[input].end( ) ) {
	  if ( wrapped ) { break; } // [한국어] 한 바퀴 순환 완료 — 종료
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _in_req[input].begin( ); // [한국어] 처음으로 감기
	  wrapped = true;
	}

	output = p->second.port; // [한국어] 이 요청의 출력 포트 번호

	// we know the output is free (above) and
	// if the input is free, grant request
	if ( grants[output] == input ) { // [한국어] 이 출력이 이 입력을 Grant했는지 확인
	  // Accept
	  _inmatch[input]   = output; // [한국어] 입력 측 매칭 결과 기록
	  _outmatch[output] = input;  // [한국어] 출력 측 매칭 결과 기록

	  // Only update pointers if accepted during the 1st iteration
	  if ( iter == 0 ) { // [한국어] 첫 번째 반복에서 수락된 경우에만 포인터 갱신
	    _gptrs[output] = ( input + 1 ) % _inputs;   // [한국어] 다음 Grant는 방금 선택한 입력 다음부터
	    _aptrs[input]  = ( output + 1 ) % _outputs; // [한국어] 다음 Accept는 방금 선택한 출력 다음부터
	  }

	  break; // [한국어] Accept 완료 — 이 입력은 처리 끝
	}

	p++; // [한국어] 이 출력은 Grant하지 않았음 — 다음 요청 탐색
      }
    }
  } // [한국어] _iSLIP_iter 반복 종료

#ifdef DEBUG_ISLIP
  cout << "input match: ";
  for ( int i = 0; i < _inputs; ++i ) { // [한국어] 최종 입력 매칭 결과 출력
    cout << _inmatch[i] << " ";
  }
  cout << endl;

  cout << "output match: ";
  for ( int j = 0; j < _outputs; ++j ) { // [한국어] 최종 출력 매칭 결과 출력
    cout << _outmatch[j] << " ";
  }
  cout << endl;
#endif
}
