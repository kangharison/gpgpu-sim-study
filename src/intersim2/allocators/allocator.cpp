// $Id: allocator.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] NoC 크로스바 할당기 기반 클래스 구현 (allocator.cpp)
 *
 * === 파일의 역할 ===
 * Allocator(순수 가상 기반), DenseAllocator(2D 배열 저장), SparseAllocator(맵 저장)의
 * 공통 메서드를 구현한다. 또한 NewAllocator() 팩토리 함수를 통해 설정 문자열에 따라
 * MaxSizeMatch, PIM, iSLIP, LOA, Wavefront, SelAlloc, Separable 등 구체 할당기를 생성한다.
 * GPGPU-Sim에서 GPU NoC 라우터(IQRouter)의 VC/스위치 할당에 매 사이클 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터의 라우터 내부에서 동작한다. IQRouter는 생성 시 NewAllocator()를
 * 호출하여 설정에 맞는 할당기를 한 번 생성하고, 이후 매 사이클 Clear/AddRequest/Allocate
 * 3단계를 반복한다. 실행 컨텍스트: GPGPU-Sim 타이밍 사이클 루프(단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * - IQRouter: _vc_allocator, _sw_allocator, _spec_sw_allocator로 할당기 인스턴스 보유
 * - maxsize.hpp/pim.hpp/islip.hpp 등: #include로 구체 알고리즘 클래스 포함
 * - module.hpp: Allocator가 Module을 상속
 *
 * === 주요 함수/구조체 요약 ===
 * - Allocator(): _inmatch/_outmatch를 -1로 초기화
 * - Clear(): _dirty 확인 후 _inmatch/_outmatch 리셋
 * - AddRequest(): 유효성 검사 및 _dirty 설정
 * - OutputAssigned()/InputAssigned(): 매칭 결과 조회
 * - PrintGrants(): 매칭 결과 디버그 출력
 * - NewAllocator(): 설정 문자열로 구체 Allocator 인스턴스 생성 팩토리
 * - DenseAllocator/SparseAllocator: 각자 요청 저장 방식에 맞는 CRUD 구현
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

#include "booksim.hpp"  // [한국어] using namespace std; 및 공통 헤더 포함
#include <iostream>    // [한국어] cout, cerr 출력 스트림
#include <sstream>     // [한국어] ostringstream — 이름 문자열 생성에 사용
#include <cassert>     // [한국어] 경계값 유효성 검사용 assert 매크로
#include "allocator.hpp" // [한국어] Allocator, DenseAllocator, SparseAllocator 선언

/////////////////////////////////////////////////////////////////////////
// [한국어] 구체 할당기 알고리즘 헤더 — NewAllocator() 팩토리에서 인스턴스 생성에 필요
#include "maxsize.hpp"              // [한국어] MaxSizeMatch — 최대 크기 이분 매칭 (O(N^3))
#include "pim.hpp"                  // [한국어] PIM — 병렬 반복 매칭 (Parallel Iterative Matching)
#include "islip.hpp"                // [한국어] iSLIP — 반복 라운드로빈 매칭 (기본 GPU NoC 알고리즘)
#include "loa.hpp"                  // [한국어] LOA — 외로운 출력 우선 매칭 (Lonely Output Arbiter)
#include "wavefront.hpp"            // [한국어] Wavefront — 대각선 우선순위 매칭
#include "selalloc.hpp"             // [한국어] SelAlloc — 선택적 할당 (iSLIP 변형, 우선순위 지원)
#include "separable_input_first.hpp"  // [한국어] SeparableInputFirst — 입력 먼저 분리 할당기
#include "separable_output_first.hpp" // [한국어] SeparableOutputFirst — 출력 먼저 분리 할당기
//
/////////////////////////////////////////////////////////////////////////

//==================================================
// Allocator base class
//==================================================

/*
 * [한국어] Allocator 생성자 — 입출력 매칭 배열을 초기화
 *
 * @parent: BookSim 모듈 계층의 부모 모듈 (IQRouter)
 * @name: 디버그용 이름 문자열
 * @inputs: 입력 포트 수 (VC 포함)
 * @outputs: 출력 포트 수 (VC 포함)
 *
 * _inmatch와 _outmatch를 -1(미할당)로 초기화하고 _dirty를 false로 설정한다.
 * 이후 매 사이클마다 Clear() → AddRequest() × N → Allocate() 순서로 재사용된다.
 *
 * 호출 체인: Allocator::NewAllocator() → [구체 할당기 생성자] → Allocator()
 */
Allocator::Allocator( Module *parent, const string& name,
		      int inputs, int outputs ) :
Module( parent, name ), _inputs( inputs ), _outputs( outputs ), _dirty( false )
{
  _inmatch.resize(_inputs, -1);   // [한국어] 입력 i가 매칭된 출력 포트 초기화 (-1 = 미매칭)
  _outmatch.resize(_outputs, -1); // [한국어] 출력 j를 담당하는 입력 포트 초기화 (-1 = 유휴)
}

/*
 * [한국어] Clear - 매칭 결과 및 dirty 플래그 초기화 (매 사이클 시작 시 호출)
 *
 * _dirty가 true인 경우(이번 사이클에 요청이 있었던 경우)에만 _inmatch와 _outmatch를
 * 모두 -1로 초기화하고 _dirty를 false로 리셋한다. 요청이 전혀 없었으면 건너뛰어 성능 최적화.
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → Allocator::Clear()
 */
void Allocator::Clear( )
{
  if(_dirty) { // [한국어] 이번 사이클에 요청이 하나라도 등록되었으면 초기화 실행
    _inmatch.assign(_inputs, -1);   // [한국어] 모든 입력의 매칭 결과를 -1(미매칭)로 초기화
    _outmatch.assign(_outputs, -1); // [한국어] 모든 출력의 매칭 결과를 -1(유휴)로 초기화
    _dirty = false;                 // [한국어] 다음 Clear() 호출 시 불필요한 재초기화 방지
  }
}

/*
 * [한국어] AddRequest - 요청 유효성 검사 및 dirty 플래그 설정
 *
 * @in: 요청하는 입력 포트 번호 (0.._inputs-1 범위 확인)
 * @out: 요청하는 출력 포트 번호 (0.._outputs-1 범위 확인)
 * @label: 요청 레이블 / VC ID (0 이상이어야 함)
 * @in_pri: 수락 단계 우선순위
 * @out_pri: 승인 단계 우선순위
 *
 * Allocator 기반 클래스의 AddRequest()는 유효성 검사와 _dirty 설정만 담당한다.
 * 실제 요청 저장은 DenseAllocator 또는 SparseAllocator의 AddRequest()에서 처리한다.
 * 서브클래스 구현은 반드시 Allocator::AddRequest()를 먼저 호출해야 한다.
 *
 * 호출 체인: IQRouter → DenseAllocator::AddRequest() → Allocator::AddRequest()
 */
void Allocator::AddRequest( int in, int out, int label, int in_pri,
			    int out_pri ) {

  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 입력 포트 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 출력 포트 범위 검사
  assert( label >= 0 );                         // [한국어] 레이블은 0 이상이어야 함 (-1은 "요청 없음"을 의미)
  _dirty = true;                                // [한국어] 이번 사이클에 요청이 있음을 표시
}

/*
 * [한국어] OutputAssigned - 입력 in이 이번 사이클 Allocate()에서 할당받은 출력 포트 번호 반환
 *
 * @in: 입력 포트 번호 (0.._inputs-1)
 * @return: 할당된 출력 포트 번호. 미할당이면 -1.
 * Allocate() 호출 후 IQRouter가 스위치 트래버설 여부를 결정할 때 사용한다.
 *
 * 호출 체인: IQRouter::_SwitchEvaluate() → Allocator::OutputAssigned()
 */
int Allocator::OutputAssigned( int in ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) ); // [한국어] 범위 검사

  return _inmatch[in]; // [한국어] _inmatch[in] = -1(미할당) 또는 출력 포트 번호
}

/*
 * [한국어] InputAssigned - 출력 out을 이번 사이클에 담당하는 입력 포트 번호 반환
 *
 * @out: 출력 포트 번호 (0.._outputs-1)
 * @return: 해당 출력을 담당하는 입력 포트 번호. 미사용이면 -1.
 */
int Allocator::InputAssigned( int out ) const
{
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 범위 검사

  return _outmatch[out]; // [한국어] _outmatch[out] = -1(유휴) 또는 입력 포트 번호
}

/*
 * [한국어] PrintGrants - 현재 _inmatch/_outmatch 매칭 결과를 스트림에 출력 (디버그용)
 *
 * @os: 출력 스트림. NULL이면 cout 사용.
 * 할당된 매칭 쌍만 출력한다 (미매칭 -1은 생략).
 * 예: "Input grants = [ 0 -> 2  1 -> 0 ], output grants = [ 0 -> 1  2 -> 0 ]."
 */
void Allocator::PrintGrants( ostream * os ) const
{
  if(!os) os = &cout; // [한국어] NULL이면 표준 출력으로 대체

  *os << "Input grants = [ ";
  for ( int input = 0; input < _inputs; ++input ) { // [한국어] 모든 입력을 순회하며 매칭된 것만 출력
    if(_inmatch[input] >= 0) { // [한국어] -1이면 미할당이므로 출력 생략
      *os << input << " -> " << _inmatch[input] << "  ";
    }
  }
  *os << "], output grants = [ ";
  for ( int output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력을 순회하며 사용된 것만 출력
    if(_outmatch[output] >= 0) {
      *os << output << " -> " << _outmatch[output] << "  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// DenseAllocator
//==================================================

/*
 * [한국어] DenseAllocator 생성자 — inputs × outputs 크기의 요청 행렬 초기화
 *
 * @parent: 부모 모듈 (IQRouter)
 * @name: 디버그용 이름
 * @inputs: 입력 포트 수
 * @outputs: 출력 포트 수
 *
 * _request를 2D 벡터로 할당하고 모든 슬롯의 label을 -1로 초기화한다.
 * MaxSizeMatch, PIM, Wavefront, LOA 등이 이 생성자를 통해 초기화된다.
 *
 * 호출 체인: Allocator::NewAllocator() → [MaxSizeMatch 등 생성자] → DenseAllocator()
 */
DenseAllocator::DenseAllocator( Module *parent, const string& name,
				int inputs, int outputs ) :
  Allocator( parent, name, inputs, outputs )
{
  _request.resize(_inputs); // [한국어] 행(입력 포트) 수만큼 외부 벡터 크기 설정

  for ( int i = 0; i < _inputs; ++i ) {
    _request[i].resize(_outputs);  // [한국어] 열(출력 포트) 수만큼 내부 벡터 크기 설정
    for ( int j = 0; j < _outputs; ++j ) {
      _request[i][j].label = -1; // [한국어] 초기 상태는 요청 없음(-1)으로 설정
    }
  }
}

/*
 * [한국어] DenseAllocator::Clear - 모든 요청 슬롯을 -1로 초기화
 *
 * 2D 행렬 전체를 순회하며 label을 -1로 리셋한다 (O(inputs × outputs)).
 * 이후 상위 Allocator::Clear()를 호출하여 _inmatch/_outmatch도 리셋한다.
 * 매 사이클 Allocate() 전에 호출된다.
 */
void DenseAllocator::Clear( )
{
  for ( int i = 0; i < _inputs; ++i ) {   // [한국어] 모든 입력 포트 행 순회
    for ( int j = 0; j < _outputs; ++j ) { // [한국어] 모든 출력 포트 열 순회
      _request[i][j].label = -1;          // [한국어] 요청 없음 상태로 초기화
    }
  }
  Allocator::Clear(); // [한국어] 상위 클래스의 _inmatch/_outmatch도 초기화
}

/*
 * [한국어] DenseAllocator::ReadRequest (레이블 버전) - 특정 입출력 쌍의 요청 레이블 반환
 * @in: 입력 포트 번호
 * @out: 출력 포트 번호
 * @return: _request[in][out].label — -1이면 요청 없음, >=0이면 유효 요청
 */
int DenseAllocator::ReadRequest( int in, int out ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 입력 인덱스 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 출력 인덱스 범위 검사

  return _request[in][out].label; // [한국어] -1(요청 없음) 또는 레이블 값 반환
}

/*
 * [한국어] DenseAllocator::ReadRequest (구조체 버전) - 요청 전체 메타데이터를 req에 복사
 * @req: 결과를 저장할 sRequest 구조체 (out 파라미터)
 * @in: 입력 포트 번호
 * @out: 출력 포트 번호
 * @return: req.label >= 0이면 true (요청 존재), 그렇지 않으면 false
 */
bool DenseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 범위 검사

  req = _request[in][out]; // [한국어] 요청 메타데이터 복사 (label, in_pri, out_pri)

  return ( req.label >= 0 ); // [한국어] label이 0 이상이면 유효한 요청 존재
}

/*
 * [한국어] DenseAllocator::AddRequest - 특정 입출력 쌍의 요청을 등록
 *
 * @in: 요청하는 입력 포트
 * @out: 요청하는 출력 포트
 * @label: 요청 레이블 (VC ID 등)
 * @in_pri: 입력 측 우선순위
 * @out_pri: 출력 측 우선순위
 *
 * 상위 Allocator::AddRequest()로 유효성 검사 및 _dirty 설정 후 실제 저장 수행.
 * 이미 같은 슬롯에 요청이 있으면 assert로 에러 — 중복 요청은 허용되지 않는다.
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → DenseAllocator::AddRequest()
 */
void DenseAllocator::AddRequest( int in, int out, int label,
				 int in_pri, int out_pri )
{
  Allocator::AddRequest(in, out, label, in_pri, out_pri); // [한국어] 유효성 검사 + _dirty = true
  assert( _request[in][out].label == -1 ); // [한국어] 중복 요청 방지 — 이미 요청이 있으면 오류

  _request[in][out].label   = label;   // [한국어] 요청 레이블(VC ID) 저장
  _request[in][out].in_pri  = in_pri;  // [한국어] 수락 단계 우선순위 저장
  _request[in][out].out_pri = out_pri; // [한국어] 승인 단계 우선순위 저장
}

/*
 * [한국어] DenseAllocator::RemoveRequest - 특정 입출력 쌍의 요청을 취소
 * @in: 취소할 입력 포트
 * @out: 취소할 출력 포트
 * @label: 취소할 요청의 레이블 (검사용)
 * _request[in][out].label를 -1로 초기화하여 요청 없음 상태로 변경한다.
 */
void DenseAllocator::RemoveRequest( int in, int out, int label )
{
  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 범위 검사

  _request[in][out].label = -1; // [한국어] 요청 취소 — 슬롯을 빈 상태로 표시
}

/*
 * [한국어] DenseAllocator::InputHasRequests - 입력 in이 어떤 출력에든 요청을 보냈는지 확인
 * @in: 확인할 입력 포트 번호
 * @return: 요청이 하나라도 있으면 true
 * _request[in][*]을 순회하며 label >= 0인 슬롯이 있으면 즉시 true 반환.
 */
bool DenseAllocator::InputHasRequests( int in ) const
{
  for(int out = 0; out < _outputs; ++out) {    // [한국어] 입력 in의 모든 출력 요청 확인
    if(_request[in][out].label >= 0) {         // [한국어] 유효한 요청이 있으면 즉시 반환
      return true;
    }
  }
  return false; // [한국어] 어떤 출력에도 요청 없음
}

/*
 * [한국어] DenseAllocator::OutputHasRequests - 출력 out에 대한 요청이 존재하는지 확인
 * @out: 확인할 출력 포트 번호
 * @return: 요청이 하나라도 있으면 true
 */
bool DenseAllocator::OutputHasRequests( int out ) const
{
  for(int in = 0; in < _inputs; ++in) {     // [한국어] 출력 out을 요청하는 모든 입력 확인
    if(_request[in][out].label >= 0) {      // [한국어] 유효한 요청이 있으면 즉시 반환
      return true;
    }
  }
  return false; // [한국어] 어떤 입력도 이 출력을 요청하지 않음
}

/*
 * [한국어] DenseAllocator::NumInputRequests - 입력 in이 보낸 요청의 총 수 반환
 * @in: 입력 포트 번호
 * @return: label >= 0인 출력 슬롯 수
 */
int DenseAllocator::NumInputRequests( int in ) const
{
  int result = 0;                              // [한국어] 유효 요청 수 카운터
  for(int out = 0; out < _outputs; ++out) {   // [한국어] 모든 출력 포트 순회
    if(_request[in][out].label >= 0) {        // [한국어] 요청이 있으면 카운트 증가
      ++result;
    }
  }
  return result; // [한국어] 입력 in이 이번 사이클에 요청한 출력 포트 수
}

/*
 * [한국어] DenseAllocator::NumOutputRequests - 출력 out에 대한 요청의 총 수 반환
 * @out: 출력 포트 번호
 * @return: 이 출력을 요청한 입력 수
 */
int DenseAllocator::NumOutputRequests( int out ) const
{
  int result = 0;                            // [한국어] 유효 요청 수 카운터
  for(int in = 0; in < _inputs; ++in) {     // [한국어] 모든 입력 포트 순회
    if(_request[in][out].label >= 0) {      // [한국어] 요청이 있으면 카운트 증가
      ++result;
    }
  }
  return result; // [한국어] 이번 사이클에 출력 out을 요청한 입력 수
}

/*
 * [한국어] DenseAllocator::PrintRequests - 등록된 요청 행렬을 텍스트로 출력 (디버그용)
 * @os: 출력 스트림 (NULL이면 cout)
 * 입력 측과 출력 측 요청을 각각 "input -> [출력@우선순위 ...]" 형식으로 출력한다.
 */
void DenseAllocator::PrintRequests( ostream * os ) const
{
  if(!os) os = &cout; // [한국어] NULL이면 표준 출력 사용

  *os << "Input requests = [ ";
  for ( int input = 0; input < _inputs; ++input ) { // [한국어] 각 입력 포트 순회
    bool print = false;
    ostringstream ss;
    for ( int output = 0; output < _outputs; ++output ) { // [한국어] 이 입력이 요청한 출력들 탐색
      const sRequest & req = _request[input][output]; // [한국어] 요청 참조
      if ( req.label >= 0 ) {  // [한국어] 유효한 요청만 출력
	print = true;
	ss << output << "@" << req.in_pri << " "; // [한국어] "출력번호@입력우선순위" 형식
      }
    }
    if(print) { // [한국어] 이 입력이 요청을 하나라도 보냈으면 출력
      *os << input << " -> [ " << ss.str() << "]  ";
    }
  }
  *os << "], output requests = [ ";
  for ( int output = 0; output < _outputs; ++output ) { // [한국어] 각 출력 포트 순회
    bool print = false;
    ostringstream ss;
    for ( int input = 0; input < _inputs; ++input ) { // [한국어] 이 출력을 요청한 입력들 탐색
      const sRequest & req = _request[input][output];
      if ( req.label >= 0 ) {
	print = true;
	ss << input << "@" << req.out_pri << " "; // [한국어] "입력번호@출력우선순위" 형식
      }
    }
    if(print) {
      *os << output << " -> [ " << ss.str() << "]  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// SparseAllocator
//==================================================

/*
 * [한국어] SparseAllocator 생성자 — 희소 요청 맵 구조 초기화
 *
 * @parent: 부모 모듈 (IQRouter)
 * @name: 디버그용 이름
 * @inputs: 입력 포트 수
 * @outputs: 출력 포트 수
 *
 * _in_req를 inputs 크기, _out_req를 outputs 크기의 빈 맵 벡터로 초기화한다.
 * iSLIP, SelAlloc, SeparableAllocator가 상속하며 이 생성자를 통해 초기화한다.
 *
 * 호출 체인: Allocator::NewAllocator() → iSLIP_Sparse 생성자 → SparseAllocator()
 */
SparseAllocator::SparseAllocator( Module *parent, const string& name,
				  int inputs, int outputs ) :
  Allocator( parent, name, inputs, outputs )
{
  _in_req.resize(_inputs);    // [한국어] 입력 포트별 요청 맵 벡터 초기화 (각 맵은 빈 상태)
  _out_req.resize(_outputs);  // [한국어] 출력 포트별 요청 맵 벡터 초기화
}


/*
 * [한국어] SparseAllocator::Clear - 요청 맵과 occupied 집합을 초기화
 *
 * 비어있지 않은 맵만 clear()하고 _in_occ, _out_occ 집합을 비운다.
 * 마지막으로 상위 Allocator::Clear()를 호출하여 _inmatch/_outmatch도 리셋한다.
 * DenseAllocator::Clear()와 달리 실제로 요청이 있었던 맵만 초기화하여 효율적이다.
 */
void SparseAllocator::Clear( )
{
  for ( int i = 0; i < _inputs; ++i ) {         // [한국어] 모든 입력 포트 순회
    if(!_in_req[i].empty())                      // [한국어] 비어있지 않은 맵만 초기화 (최적화)
      _in_req[i].clear( );                       // [한국어] 이 입력이 보낸 모든 요청 제거
  }

  for ( int j = 0; j < _outputs; ++j ) {        // [한국어] 모든 출력 포트 순회
    if(!_out_req[j].empty())                     // [한국어] 비어있지 않은 맵만 초기화 (최적화)
      _out_req[j].clear( );                      // [한국어] 이 출력을 요청하는 모든 항목 제거
  }

  _in_occ.clear( );   // [한국어] 활성 입력 포트 집합 초기화
  _out_occ.clear( );  // [한국어] 활성 출력 포트 집합 초기화

  Allocator::Clear(); // [한국어] 상위 클래스의 _inmatch/_outmatch 초기화 + _dirty = false
}

/*
 * [한국어] SparseAllocator::ReadRequest (레이블 버전)
 * @in: 입력 포트 번호
 * @out: 출력 포트 번호
 * @return: 요청이 있으면 레이블, 없으면 -1
 * 구조체 버전 ReadRequest()를 통해 내부적으로 맵 탐색을 수행한다.
 */
int SparseAllocator::ReadRequest( int in, int out ) const
{
  sRequest r; // [한국어] 요청 메타데이터를 담을 임시 구조체

  if ( ! ReadRequest( r, in, out ) ) { // [한국어] 맵에서 (in, out) 요청 탐색
    r.label = -1; // [한국어] 요청 없음 — label을 -1로 설정
  }

  return r.label; // [한국어] -1(요청 없음) 또는 레이블 값 반환
}

/*
 * [한국어] SparseAllocator::ReadRequest (구조체 버전) — 맵 탐색으로 요청 메타데이터 조회
 * @req: 결과를 담을 sRequest 구조체 (out 파라미터)
 * @in: 입력 포트 번호
 * @out: 출력 포트 번호
 * @return: 요청이 있으면 true, 없으면 false
 */
bool SparseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
  bool found; // [한국어] 요청 발견 여부

  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 입력 포트 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 출력 포트 범위 검사

  map<int, sRequest>::const_iterator match = _in_req[in].find(out); // [한국어] _in_req[in] 맵에서 out 키 탐색
  if ( match != _in_req[in].end( ) ) { // [한국어] 탐색 성공 — 요청 존재
    req = match->second; // [한국어] 요청 메타데이터를 req에 복사
    found = true;
  } else { // [한국어] 탐색 실패 — 요청 없음
    found = false;
  }

  return found; // [한국어] 요청 존재 여부 반환
}

/*
 * [한국어] SparseAllocator::AddRequest - 입출력 요청을 _in_req와 _out_req에 양방향 등록
 *
 * @in: 요청하는 입력 포트
 * @out: 요청하는 출력 포트
 * @label: 요청 레이블
 * @in_pri: 입력 측 우선순위
 * @out_pri: 출력 측 우선순위
 *
 * 동일한 (in, out) 쌍에 중복 요청을 방지하고, 처음 요청 등록 시 occupied 집합에 추가한다.
 * _in_req[in][out]에는 port = out, _out_req[out][in]에는 port = in으로 저장하여
 * 양방향 순회를 O(log N)에 지원한다.
 *
 * 호출 체인: IQRouter → iSLIP_Sparse(상속) → SparseAllocator::AddRequest()
 */
void SparseAllocator::AddRequest( int in, int out, int label,
				  int in_pri, int out_pri )
{
  Allocator::AddRequest(in, out, label, in_pri, out_pri); // [한국어] 범위 검사 + _dirty = true
  assert( _in_req[in].count(out) == 0 );    // [한국어] 중복 요청 방지 — 이미 같은 요청이 있으면 오류
  assert( _out_req[out].count(in) == 0 );   // [한국어] 역방향도 중복 방지

  // insert into occupied inputs set if
  // input is currently empty
  if ( _in_req[in].empty( ) ) {             // [한국어] 이 입력 포트의 첫 번째 요청인 경우
    _in_occ.insert(in);                     // [한국어] occupied 입력 집합에 추가 — Grant/Accept 루프에서 건너뜀 방지
  }

  // similarly for the output
  if ( _out_req[out].empty( ) ) {           // [한국어] 이 출력 포트를 처음 요청하는 경우
    _out_occ.insert(out);                   // [한국어] occupied 출력 집합에 추가
  }

  sRequest req;          // [한국어] _in_req에 저장할 요청 구조체
  req.port    = out;     // [한국어] 입력 측 맵에서 port는 출력 포트 번호
  req.label   = label;   // [한국어] 요청 레이블
  req.in_pri  = in_pri;  // [한국어] 수락 단계 우선순위
  req.out_pri = out_pri; // [한국어] 승인 단계 우선순위

  _in_req[in][out] = req; // [한국어] 입력 방향 맵에 등록 (key=출력번호, value=요청)

  req.port  = in;         // [한국어] 출력 측 맵에서 port는 입력 포트 번호로 변경

  _out_req[out][in] = req; // [한국어] 출력 방향 맵에 등록 (key=입력번호, value=요청)
}

/*
 * [한국어] SparseAllocator::RemoveRequest - 특정 입출력 요청을 양방향으로 제거
 *
 * @in: 제거할 입력 포트
 * @out: 제거할 출력 포트
 * @label: 일치 검사용 레이블
 *
 * _in_req[in]에서 out 키를 erase하고, _out_req[out]에서 in 키를 erase한다.
 * 맵이 비어지면 occupied 집합에서도 해당 포트를 제거한다.
 */
void SparseAllocator::RemoveRequest( int in, int out, int label )
{
  assert( ( in >= 0 ) && ( in < _inputs ) );   // [한국어] 범위 검사
  assert( ( out >= 0 ) && ( out < _outputs ) ); // [한국어] 범위 검사

  assert( _in_req[in].count( out ) > 0 );       // [한국어] 존재하는 요청만 제거 가능
  assert( _in_req[in][out].label == label );     // [한국어] 레이블 일치 확인
  _in_req[in].erase( out );                      // [한국어] 입력 방향 맵에서 요청 제거

  // remove from occupied inputs list if
  // input is now empty
  if ( _in_req[in].empty( ) ) {                  // [한국어] 이 입력의 마지막 요청이 제거된 경우
    _in_occ.erase(in);                           // [한국어] occupied 집합에서도 제거
  }

  // similarly for the output
  assert( _out_req[out].count( in ) > 0 );       // [한국어] 역방향 요청도 존재해야 함
  assert( _out_req[out][in].label == label );     // [한국어] 역방향 레이블 일치 확인
  _out_req[out].erase( in );                      // [한국어] 출력 방향 맵에서 요청 제거

  if ( _out_req[out].empty( ) ) {                 // [한국어] 이 출력의 마지막 요청이 제거된 경우
    _out_occ.erase(out);                          // [한국어] occupied 집합에서도 제거
  }
}

/*
 * [한국어] InputHasRequests/OutputHasRequests - occupied 집합으로 요청 존재 여부 O(log N) 조회
 */
bool SparseAllocator::InputHasRequests( int in ) const
{
  return _in_occ.count(in) > 0; // [한국어] in이 occupied 집합에 있으면 요청 존재
}

bool SparseAllocator::OutputHasRequests( int out ) const
{
  return _out_occ.count(out) > 0; // [한국어] out이 occupied 집합에 있으면 요청 존재
}

/*
 * [한국어] NumInputRequests/NumOutputRequests - occupied 집합의 count()로 요청 수 반환
 * SparseAllocator에서는 동일 입출력 쌍에 요청이 하나만 허용되므로 count()는 0 또는 1.
 */
int SparseAllocator::NumInputRequests( int in ) const
{
  return _in_occ.count(in); // [한국어] 0 또는 1 — occupied 집합에 있는지 여부
}

int SparseAllocator::NumOutputRequests( int out ) const
{
  return _out_occ.count(out); // [한국어] 0 또는 1
}

/*
 * [한국어] SparseAllocator::PrintRequests - 활성 요청 맵을 텍스트로 출력 (디버그용)
 * @os: 출력 스트림 (NULL이면 cout)
 * 비어있는 맵은 건너뛰고 활성 입출력의 요청만 출력한다.
 */
void SparseAllocator::PrintRequests( ostream * os ) const
{
  map<int, sRequest>::const_iterator iter; // [한국어] 맵 순회 이터레이터

  if(!os) os = &cout; // [한국어] NULL이면 표준 출력 사용

  *os << "Input requests = [ ";
  for ( int input = 0; input < _inputs; ++input ) { // [한국어] 모든 입력 포트 순회
    if(!_in_req[input].empty()) {                   // [한국어] 요청이 없는 포트는 출력 생략
      *os << input << " -> [ ";
      for ( iter = _in_req[input].begin( );
	    iter != _in_req[input].end( ); iter++ ) { // [한국어] 이 입력이 요청한 모든 출력 열거
	*os << iter->second.port << "@" << iter->second.in_pri << " "; // [한국어] "출력번호@입력우선순위"
      }
      *os << "]  ";
    }
  }
  *os << "], output requests = [ ";
  for ( int output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력 포트 순회
    if(!_out_req[output].empty()) {                     // [한국어] 요청 없는 포트는 생략
      *os << output << " -> ";
      *os << "[ ";
      for ( iter = _out_req[output].begin( );
	    iter != _out_req[output].end( ); iter++ ) { // [한국어] 이 출력을 요청하는 모든 입력 열거
	*os << iter->second.port << "@" << iter->second.out_pri << " "; // [한국어] "입력번호@출력우선순위"
      }
      *os << "]  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// Global allocator allocation function
//==================================================

/*
 * [한국어] Allocator::NewAllocator - 설정 문자열에 따라 구체 Allocator 인스턴스를 생성하는 팩토리
 *
 * @parent: 부모 모듈 (IQRouter)
 * @name: 할당기 이름 문자열
 * @alloc_type: 할당기 타입 문자열. 괄호 안에 파라미터 포함 가능
 *              예: "islip", "islip(4)", "pim(2)", "wavefront", "separable_input_first(round_robin)"
 * @inputs: 입력 포트 수
 * @outputs: 출력 포트 수
 * @config: gpgpusim.config 설정 — param_str가 비어있을 때 alloc_iters, arb_type 등을 읽는다
 * @return: 생성된 Allocator 인스턴스. 알 수 없는 타입이면 NULL(0).
 *
 * IQRouter 생성자에서 VC 할당기와 스위치 할당기를 생성할 때 한 번 호출된다.
 * 설정 파일의 "vc_allocator", "sw_allocator" 항목이 이 함수에 전달된다.
 *
 * GPGPU-Sim GPU NoC 기본값: "islip" (단순 라운드로빈으로 충분히 효율적).
 *
 * 호출 체인: IQRouter 생성자 → Allocator::NewAllocator()
 */
Allocator *Allocator::NewAllocator( Module *parent, const string& name,
				    const string &alloc_type,
				    int inputs, int outputs,
				    Configuration const * const config )
{
  Allocator *a = 0; // [한국어] 생성된 할당기 포인터 초기화

  string alloc_name; // [한국어] 파라미터 제거 후 순수 타입 이름
  string param_str;  // [한국어] 괄호 안의 파라미터 문자열
  size_t left = alloc_type.find_first_of('('); // [한국어] 괄호 위치 탐색
  if(left == string::npos) {                   // [한국어] 괄호 없음 — 파라미터 없는 타입
    alloc_name = alloc_type;                   // [한국어] 전체 문자열이 타입 이름
  } else {                                     // [한국어] 괄호 있음 — 파라미터 분리
    alloc_name = alloc_type.substr(0, left);   // [한국어] 괄호 앞 부분이 타입 이름
    size_t right = alloc_type.find_last_of(')'); // [한국어] 닫는 괄호 위치
    if(right == string::npos) {                // [한국어] 닫는 괄호 없음 — 끝까지를 파라미터로
      param_str = alloc_type.substr(left+1);
    } else {                                   // [한국어] 정상적인 "name(param)" 형식
      param_str = alloc_type.substr(left+1, right-left-1); // [한국어] 괄호 사이 문자열 추출
    }
  }

  // [한국어] 타입 이름에 따라 구체 할당기 인스턴스 생성
  if ( alloc_name == "max_size" ) {
    // [한국어] MaxSizeMatch: 최대 크기 이분 매칭 — O(N^3) 최적 매칭, 연구용
    a = new MaxSizeMatch( parent, name, inputs, outputs );
  } else if ( alloc_name == "pim" ) {
    // [한국어] PIM(Parallel Iterative Matching): 병렬 랜덤 매칭 반복
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    // [한국어] iters: 반복 횟수 — param_str 우선, 없으면 config의 alloc_iters, 없으면 기본값 1
    a = new PIM( parent, name, inputs, outputs, iters );
  } else if ( alloc_name == "islip" ) {
    // [한국어] iSLIP: 반복 라운드로빈 매칭 — GPGPU-Sim GPU NoC 기본 할당기
    // 각 반복은 Grant(출력 → 입력 선택) + Accept(입력 → 출력 선택) 2단계로 구성됨
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    a = new iSLIP_Sparse( parent, name, inputs, outputs, iters );
  } else if ( alloc_name == "loa" ) {
    // [한국어] LOA(Lonely Output Arbiter): 요청이 가장 적은 출력에 우선 접근
    a = new LOA( parent, name, inputs, outputs );
  } else if ( alloc_name == "wavefront" ) {
    // [한국어] Wavefront: 대각선 우선순위 매칭 — skip_diags=false(고정 우선순위)
    a = new Wavefront( parent, name, inputs, outputs );
  } else if ( alloc_name == "rr_wavefront" ) {
    // [한국어] RR-Wavefront: 라운드로빈 대각선 매칭 — skip_diags=true(동적 우선순위)
    a = new Wavefront( parent, name, inputs, outputs, true );
  } else if ( alloc_name == "select" ) {
    // [한국어] SelAlloc: 우선순위 기반 선택적 할당 (iSLIP 변형)
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    a = new SelAlloc( parent, name, inputs, outputs, iters );
  } else if (alloc_name == "separable_input_first") {
    // [한국어] SeparableInputFirst: 입력 먼저 중재 후 출력 중재하는 분리 할당기
    // arb_type: 내부 Arbiter 타입 — "round_robin"(기본), "matrix", "tree(...)" 가능
    string arb_type = param_str.empty() ? (config ? config->GetStr("arb_type") : "round_robin") : param_str;
    a = new SeparableInputFirstAllocator( parent, name, inputs, outputs,
					  arb_type );
  } else if (alloc_name == "separable_output_first") {
    // [한국어] SeparableOutputFirst: 출력 먼저 중재 후 입력 중재하는 분리 할당기
    string arb_type = param_str.empty() ? (config ? config->GetStr("arb_type") : "round_robin") : param_str;
    a = new SeparableOutputFirstAllocator( parent, name, inputs, outputs,
					   arb_type );
  }

//==================================================
// Insert new allocators here, add another else if
//==================================================


  return a; // [한국어] 생성된 할당기 포인터 반환 (알 수 없는 타입이면 NULL=0)
}

