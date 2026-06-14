// $Id: loa.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] LOA(Lonely Output Arbiter) 할당기 선언 (loa.hpp)
 *
 * === 파일의 역할 ===
 * LOA 알고리즘을 구현하는 LOA 클래스를 선언한다. LOA는 "외로운 출력(Lonely Output)"을
 * 우선하는 전략으로, 각 입력은 자신이 요청하는 출력 중 다른 입력들이 가장 적게 요청한
 * (가장 외로운) 출력을 하나 선택하여 요청한다(Request 단계). 이후 Grant 단계에서
 * 각 출력은 자신을 요청한 입력 중 라운드로빈으로 하나를 선택한다.
 * 부하 분산(load balancing) 효과가 있어 트래픽이 고르게 분산될 때 효율적이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("loa")로 생성. DenseAllocator 상속.
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 — _request 2D 배열 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - LOA(): 생성자 — _counts, _req, _rptr, _gptr 벡터 초기화
 * - Allocate(): Count → Request → Grant 3단계 LOA 알고리즘 실행
 * - _counts[j]: 출력 j를 요청하는 입력의 수 (Count 단계에서 계산)
 * - _req[i]: 입력 i가 이번에 선택한 "외로운 출력" (-1이면 요청 없음)
 * - _rptr[i]: 입력 i의 Request 단계 라운드로빈 포인터
 * - _gptr[j]: 출력 j의 Grant 단계 라운드로빈 포인터
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

#ifndef _LOA_HPP_
#define _LOA_HPP_

#include <vector>

#include "allocator.hpp"

class LOA : public DenseAllocator {
  vector<int> _counts;
  /* _counts[j]: 이번 사이클에 출력 j를 요청하는 입력의 수 (Count 단계에서 계산).
   * 설정자: Allocate()의 Count 단계에서 _request 행렬 순회 후 계산.
   * 읽는 자: Request 단계에서 각 입력이 가장 _counts가 낮은 출력을 선택하는 데 사용.
   * 값 범위: 0.._inputs.
   * 동기화: 단일 사이클 내에서만 유효. */

  vector<int> _req;
  /* _req[i]: 입력 i가 Request 단계에서 선택한 "외로운 출력" 번호.
   * -1이면 이 입력에서 요청할 수 있는 외로운 출력이 없음.
   * 설정자: Allocate()의 Request 단계에서 _counts가 최소인 출력으로 설정.
   * 읽는 자: Grant 단계에서 각 출력이 자신을 선택한 입력 중 하나를 결정할 때 사용.
   * 값 범위: -1 또는 0.._outputs-1. */

  vector<int> _rptr;
  /* Request 단계 라운드로빈 포인터. _rptr[i] = 입력 i의 탐색 시작 출력 오프셋.
   * Count 단계에서 동점인 출력이 여럿일 때 _rptr[i] 오프셋 기준으로 탐색.
   * 설정자: Allocate() Grant 성공 후 (_rptr[i]+1)%_outputs로 갱신. 사이클 간 유지.
   * 값 범위: 0.._outputs-1. */

  vector<int> _gptr;
  /* Grant 단계 라운드로빈 포인터. _gptr[j] = 출력 j가 입력 탐색 시작 오프셋.
   * 설정자: Allocate() Grant 성공 후 (_gptr[j]+1)%_inputs로 갱신. 사이클 간 유지.
   * 값 범위: 0.._inputs-1. */

public:
  /*
   * [한국어] LOA 생성자 — 내부 상태 벡터 초기화
   * @parent, @name: 부모/이름
   * @inputs, @outputs: 입출력 포트 수
   * 호출 체인: Allocator::NewAllocator("loa") → LOA() → DenseAllocator()
   */
  LOA( Module *parent, const string& name,
       int inputs, int outputs );

  /*
   * [한국어] Allocate - Count → Request → Grant 3단계 LOA 알고리즘 실행
   * 각 입력이 가장 외로운(요청이 적은) 출력에 집중하여 부하를 분산한다.
   */
  void Allocate( );
};

#endif
