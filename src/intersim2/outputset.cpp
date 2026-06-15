// $Id: outputset.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*outputset.cpp
 *
 *output set assigns a flit which output to go to in a router
 *used by the VC class
 *the output assignment is done by the routing algorithms..
 *
 */

/*
 * [한국어 설명] BookSim NoC 라우터 출력 포트/VC 집합 구현 (outputset.cpp)
 *
 * === 파일의 역할 ===
 * outputset.hpp에 선언된 OutputSet 클래스의 메서드를 구현한다.
 * OutputSet은 NoC 라우터에서 라우팅 알고리즘이 플릿에 대해 선택 가능한
 * 출력 포트(output port)와 가상 채널(VC: Virtual Channel) 후보 집합을 관리한다.
 * 라우팅 알고리즘이 Add()/AddRange()로 후보를 등록하면, VC 할당기가
 * GetSet()/GetVC()로 후보를 읽어 실제 포트/VC를 결정한다.
 * 매 라우팅 단계 시작 시 Clear()로 초기화되고 새 후보로 채워진다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 라우터 파이프라인의 라우팅 단계(Route Computation)와 VC 할당 단계 사이에서
 * 중간 데이터 저장소 역할을 한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — 매 시뮬레이션 사이클마다 라우터 파이프라인에서 호출.
 * 호출 체인: 라우팅 알고리즘(routefunc.cc) → [OutputSet 메서드]
 *                                          → VC 할당기(allocator.cc).
 *
 * === 타 모듈과의 연결 ===
 * 의존: <cassert> (assert), booksim.hpp (공통 타입), outputset.hpp (자기 선언).
 * 피의존: VC 클래스 (OutputSet 인스턴스를 멤버로 보유),
 *         routefunc.cc (Add/AddRange 호출), allocator.cc (GetSet/GetVC 호출).
 * 데이터 흐름: 라우팅 함수 → [_outputs에 sSetElement 삽입] → VC 할당기 → 결정.
 * 공유 자료구조: _outputs (std::set<sSetElement>, 우선순위 내림차순 정렬).
 *
 * === 주요 함수/구조체 요약 ===
 * Clear()              - _outputs.clear()로 후보 집합 초기화
 * Add(port, vc, pri)   - AddRange(port, vc, vc, pri) 호출 래퍼
 * AddRange(port,s,e,p) - sSetElement 생성 후 _outputs에 삽입
 * NumVCs(port)         - 레거시: 선형 순회로 특정 포트의 VC 수 합산
 * OutputEmpty(port)    - 선형 순회로 특정 포트 후보 유무 확인
 * GetSet()             - _outputs의 const 참조 반환 (고성능 직접 접근)
 * GetVC(port, index)   - 레거시: 인덱스로 VC 번호 조회
 * GetPortVC(p, vc)     - 레거시: 단일 포트/VC 쌍 여부 확인
 */

#include <cassert> // [한국어] assert(): 내부 불변조건 검증에 사용 (현재 코드에서 직접 사용 안 하지만 booksim 관례)

#include "booksim.hpp"    // [한국어] BookSim 공통 타입, 매크로 기반 헤더
#include "outputset.hpp"  // [한국어] 이 파일에서 구현하는 OutputSet 클래스 선언 헤더

/*
 * [한국어]
 * OutputSet::Clear - 출력 후보 집합을 모두 비운다.
 *
 * @return: 없음 (void).
 *
 * 매 라우팅 단계 시작 시 호출되어 이전 사이클의 후보 항목들을 제거한다.
 * 이후 라우팅 알고리즘이 Add()/AddRange()로 새 후보를 등록한다.
 * _outputs.clear()는 std::set의 모든 원소를 제거하지만 메모리를 해제하지는 않는다.
 * 실행 컨텍스트: 호스트 CPU, 매 시뮬레이션 사이클의 라우팅 단계.
 *
 * 호출 체인:
 *   Router::_Route() → vc->outputs.Clear() → [Clear()] → _outputs.clear()
 */
void OutputSet::Clear( )
{
  _outputs.clear( ); // [한국어] _outputs set의 모든 sSetElement 항목을 제거
                     // [한국어] std::set::clear()는 O(n) 시간 복잡도; 각 원소 소멸자 호출
}

/*
 * [한국어]
 * OutputSet::Add - 단일 VC를 출력 후보로 등록한다.
 *
 * @output_port: 등록할 출력 포트 번호.
 * @vc:          등록할 단일 VC 번호.
 * @pri:         우선순위 (기본값 0; 높을수록 std::set에서 앞에 위치).
 * @return:      없음 (void).
 *
 * AddRange(output_port, vc, vc, pri)의 래퍼 함수이다.
 * 결정론적 라우팅처럼 정확히 하나의 VC를 지정하는 경우에 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 매 사이클 라우팅 단계.
 *
 * 호출 체인:
 *   라우팅 함수(routefunc.cc) → [Add(port, vc, pri)] → AddRange(port, vc, vc, pri)
 */
void OutputSet::Add( int output_port, int vc, int pri  )
{
  AddRange( output_port, vc, vc, pri ); // [한국어] vc_start와 vc_end를 동일(vc)하게 설정하여 단일 VC를 범위 형태로 등록
}

/*
 * [한국어]
 * OutputSet::AddRange - VC 범위를 출력 후보로 등록한다.
 *
 * @output_port: 등록할 출력 포트 번호.
 * @vc_start:    VC 범위 시작값 (포함).
 * @vc_end:      VC 범위 끝값 (포함). vc_start == vc_end이면 단일 VC.
 * @pri:         우선순위 (기본값 0).
 * @return:      없음 (void).
 *
 * sSetElement 구조체를 생성하고 _outputs set에 삽입한다.
 * operator<(pri 내림차순)로 정렬되므로, 높은 우선순위 항목이 set 앞에 위치한다.
 * 적응형 라우팅에서 여러 출력 포트 또는 넓은 VC 범위를 한 번에 등록할 때 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 매 사이클 라우팅 단계.
 *
 * 호출 체인:
 *   라우팅 함수 / Add() → [AddRange(port, s, e, pri)] → sSetElement 생성 → _outputs.insert()
 */
void OutputSet::AddRange( int output_port, int vc_start, int vc_end, int pri )
{

  sSetElement s; // [한국어] 새 출력 후보 항목 구조체 로컬 변수 선언

  s.vc_start = vc_start;     // [한국어] VC 범위 시작값 설정
  s.vc_end   = vc_end;       // [한국어] VC 범위 끝값 설정 (단일 VC이면 vc_start와 동일)
  s.pri      = pri;          // [한국어] 우선순위 설정 (높을수록 set 내에서 앞에 배치됨)
  s.output_port = output_port; // [한국어] 출력 포트 번호 설정
  _outputs.insert( s );      // [한국어] sSetElement를 _outputs set에 삽입
                             // [한국어] operator<(pri 내림차순)에 따라 정렬 위치가 결정됨
}

//legacy support, for performance, just use GetSet()
/*
 * [한국어]
 * OutputSet::NumVCs - 특정 출력 포트에 등록된 VC 후보의 총 수를 반환한다 (레거시).
 *
 * @output_port: 확인할 출력 포트 번호.
 * @return:      해당 포트에 등록된 VC 후보의 총 수 (모든 범위의 합산).
 *
 * _outputs set을 선형 순회하여 output_port 일치 항목들의
 * (vc_end - vc_start + 1)을 누적 합산한다. O(n) 시간 복잡도.
 * 성능을 위해서는 GetSet()으로 직접 set을 순회하는 것이 권장된다.
 * 실행 컨텍스트: 호스트 CPU, VC 할당 또는 검증 코드에서 호출.
 *
 * 호출 체인:
 *   할당기 / 검증 코드 → [NumVCs(port)] → _outputs 순회 → 합산 반환
 */
int OutputSet::NumVCs( int output_port ) const
{
  int total = 0; // [한국어] 이 포트에 대한 VC 후보 수 누산기 초기화
  set<sSetElement>::const_iterator i = _outputs.begin( ); // [한국어] _outputs set 순회 시작 반복자
  while(i!=_outputs.end( )){ // [한국어] 모든 후보 항목을 끝까지 순회
    if(i->output_port == output_port){ // [한국어] 현재 항목이 요청한 출력 포트인지 확인
      total += (i->vc_end - i->vc_start + 1); // [한국어] VC 범위의 크기(vc_end - vc_start + 1)를 누산
                                               // [한국어] +1은 vc_end 포함(inclusive) 범위이므로 필요
    }
    i++; // [한국어] 다음 항목으로 이동
  }
  return total; // [한국어] 해당 포트의 전체 VC 후보 수 반환
}

/*
 * [한국어]
 * OutputSet::OutputEmpty - 특정 출력 포트에 대한 후보가 없는지 확인한다.
 *
 * @output_port: 확인할 출력 포트 번호.
 * @return:      해당 포트의 후보가 하나도 없으면 true, 하나 이상 있으면 false.
 *
 * _outputs set을 선형 순회하여 output_port 일치 항목을 찾는다.
 * 하나라도 발견하면 즉시 false를 반환하여 불필요한 순회를 중단한다.
 * VC 할당기가 특정 포트가 라우팅 후보에 있는지 빠르게 판별할 때 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 매 사이클 VC 할당 단계.
 *
 * 호출 체인:
 *   VC 할당기 → [OutputEmpty(port)] → true/false 반환
 */
bool OutputSet::OutputEmpty( int output_port ) const
{
  set<sSetElement>::const_iterator i = _outputs.begin( ); // [한국어] _outputs set 순회 시작 반복자
  while(i!=_outputs.end( )){ // [한국어] 모든 후보 항목을 끝까지 순회
    if(i->output_port == output_port){ // [한국어] 요청한 포트와 일치하는 항목 발견 시
      return false; // [한국어] 해당 포트에 후보가 존재함 → false 반환하여 순회 즉시 종료
    }
    i++; // [한국어] 다음 항목으로 이동
  }
  return true; // [한국어] 순회 완료 후 일치 항목 없음 → 해당 포트는 후보 없음
}


/*
 * [한국어]
 * OutputSet::GetSet - 내부 _outputs set의 const 참조를 반환한다.
 *
 * @return: set<sSetElement>의 const 참조. 복사 없이 직접 순회 가능.
 *
 * VC 할당기와 라우팅 코드가 모든 후보를 효율적으로 순회할 수 있도록
 * 내부 컨테이너를 직접 노출한다. 레거시 함수(GetVC, NumVCs, GetPortVC)보다
 * 성능이 우수하므로 새 코드에서는 이 함수 사용이 권장된다.
 * 실행 컨텍스트: 호스트 CPU, 매 사이클 VC 할당 단계.
 *
 * 호출 체인:
 *   VC 할당기 / Router → [GetSet()] → set<sSetElement> 순회
 */
const set<OutputSet::sSetElement> & OutputSet::GetSet() const{
  return _outputs; // [한국어] 내부 _outputs set의 const 참조 반환 (복사 비용 없음)
}

//legacy support, for performance, just use GetSet()
/*
 * [한국어]
 * OutputSet::GetVC - 특정 출력 포트에서 인덱스로 VC 번호를 조회한다 (레거시).
 *
 * @output_port: 조회할 출력 포트 번호.
 * @vc_index:    0-based 인덱스 (해당 포트의 모든 VC 후보 중 vc_index번째를 반환).
 * @pri:         NULL이 아니면 해당 항목의 우선순위를 저장하는 출력 파라미터.
 * @return:      해당 인덱스의 VC 번호. 인덱스가 범위를 초과하면 -1.
 *
 * _outputs를 선형 순회하며 output_port 일치 항목을 찾고, 범위 크기를 누적하여
 * vc_index번째 VC가 속한 범위를 찾아 실제 VC 번호(vc_start + remaining)를 반환한다.
 * 레거시 지원 함수이며, 성능을 위해서는 GetSet() 직접 사용이 권장된다.
 * 실행 컨텍스트: 호스트 CPU.
 *
 * 호출 체인:
 *   레거시 할당기 코드 → [GetVC(port, index, &pri)] → VC 번호 반환
 */
int OutputSet::GetVC( int output_port, int vc_index, int *pri ) const
{

  int range;           // [한국어] 현재 항목의 VC 범위 크기 (vc_end - vc_start + 1)
  int remaining = vc_index; // [한국어] 아직 건너뛰어야 할 VC 수; 처음엔 요청된 vc_index 전체
  int vc = -1;         // [한국어] 반환할 VC 번호; 찾지 못하면 -1 (에러 표시)

  if ( pri ) { *pri = -1; } // [한국어] pri 출력 파라미터가 NULL이 아니면 -1로 초기화 (못 찾은 경우의 기본값)

  set<sSetElement>::const_iterator i = _outputs.begin( ); // [한국어] _outputs set 순회 시작
  while(i!=_outputs.end( )){ // [한국어] 모든 후보 항목 순회
    if(i->output_port == output_port){ // [한국어] 요청한 출력 포트와 일치하는 항목만 처리
      range = i->vc_end - i->vc_start + 1; // [한국어] 현재 범위에 속한 VC 수 계산 (+1: inclusive)
      if ( remaining >= range ) { // [한국어] 현재 범위 전체를 건너뛰어야 하는 경우
	remaining -= range; // [한국어] 건너뛴 만큼 remaining을 차감하고 다음 항목으로 이동
      } else { // [한국어] 현재 범위 내에 vc_index번째 VC가 존재하는 경우
	vc = i->vc_start + remaining; // [한국어] 범위 시작에서 remaining만큼 더한 값이 목표 VC 번호
	if ( pri ) { // [한국어] 우선순위 출력 파라미터가 제공된 경우
	  *pri = i->pri; // [한국어] 해당 항목의 우선순위를 출력 파라미터에 저장
	}
	break; // [한국어] 목표 VC를 찾았으므로 순회 중단
      }
    }
    i++; // [한국어] 다음 항목으로 이동
  }
  return vc; // [한국어] 찾은 VC 번호 반환; 못 찾으면 -1
}

//legacy support, for performance, just use GetSet()
/*
 * [한국어]
 * OutputSet::GetPortVC - 집합이 단일 포트/VC 쌍으로만 이루어졌는지 확인하고 반환한다 (레거시).
 *
 * @out_port: (출력) 단일 포트 번호가 저장될 포인터.
 * @out_vc:   (출력) 단일 VC 번호가 저장될 포인터.
 * @return:   집합의 모든 항목이 동일 포트의 단일 VC(vc_start==vc_end)이면 true.
 *            여러 포트가 있거나 VC 범위(vc_start != vc_end)가 있으면 false.
 *
 * 결정론적 라우팅에서 정확히 하나의 포트/VC로 결정된 경우를 판별할 때 사용한다.
 * 레거시 지원 함수이며, 성능을 위해서는 GetSet() 직접 사용이 권장된다.
 * 실행 컨텍스트: 호스트 CPU.
 *
 * 알고리즘 설명:
 *   - used_outputs: 첫 번째 항목의 output_port를 기준값으로 설정
 *   - 각 항목이 단일 VC(vc_start==vc_end)인지 확인; 아니면 false 반환
 *   - 항목이 처음 기준 포트와 다른 포트이면 false 반환
 *   - 모든 항목이 통과하면 true 반환
 *
 * 호출 체인:
 *   레거시 할당기 코드 → [GetPortVC(&port, &vc)] → true/false 반환
 */
bool OutputSet::GetPortVC( int *out_port, int *out_vc ) const
{

  bool single_output = false;  // [한국어] 단일 포트/VC 여부 플래그; 초기값은 false
  int  used_outputs  = 0;      // [한국어] 기준 포트 번호; 첫 번째 항목의 포트로 초기화

  set<sSetElement>::const_iterator i = _outputs.begin( ); // [한국어] _outputs set 순회 시작
  if(i!=_outputs.end( )){ // [한국어] 집합이 비어 있지 않으면 첫 번째 항목으로 기준 포트 설정
    used_outputs = i->output_port; // [한국어] 첫 번째 항목의 포트를 기준 포트로 설정
  }
  while(i!=_outputs.end( )){ // [한국어] 모든 항목 순회

    if ( i->vc_start == i->vc_end ) { // [한국어] 현재 항목이 단일 VC(범위가 아닌 정확히 하나)인 경우
      *out_vc   = i->vc_start;        // [한국어] 단일 VC 번호를 출력 파라미터에 저장
      *out_port = i->output_port;     // [한국어] 해당 출력 포트를 출력 파라미터에 저장
      single_output = true;           // [한국어] 지금까지 단일 VC로 구성됨 표시
    } else { // [한국어] 현재 항목이 VC 범위(vc_start != vc_end)를 가지는 경우
      // multiple vc's selected
      break; // [한국어] 여러 VC가 포함된 항목이 있으면 단일 출력이 아님 → 순회 중단
    }
    if (used_outputs != i->output_port) { // [한국어] 현재 항목의 포트가 기준 포트와 다른 경우
      // multiple outputs selected
      single_output = false; // [한국어] 여러 출력 포트가 있음 → false로 재설정
      break;                 // [한국어] 단일 포트/VC 조건 불만족 → 순회 중단
    }
       i++; // [한국어] 다음 항목으로 이동
  }
  return single_output; // [한국어] 단일 포트/VC 쌍이면 true, 아니면 false 반환
}
