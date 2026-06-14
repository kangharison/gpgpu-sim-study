// $Id: pim.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] PIM(Parallel Iterative Matching) 할당기 선언 (pim.hpp)
 *
 * === 파일의 역할 ===
 * 크로스바 할당을 위한 PIM(Parallel Iterative Matching) 알고리즘을 선언한다.
 * PIM은 iSLIP과 달리 Grant/Accept 단계에서 라운드로빈 대신 RandomInt()를 사용하여
 * 완전히 무작위로 중재한다. 반복 횟수(_PIM_iter)가 많을수록 매칭 효율이 높아진다.
 * 간단하고 하드웨어 구현이 쉬우나, 공정성은 확률적으로만 보장된다.
 * GPU NoC 시뮬레이션에서 무작위 트래픽 패턴을 빠르게 처리하는 데 적합하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("pim")로 생성.
 * IQRouter → PIM::Allocate() — VC/스위치 할당 단계에서 사용.
 * DenseAllocator 상속 — _request[inputs][outputs] 2D 배열로 요청 저장.
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator (상속): _request, _inmatch, _outmatch, _inputs, _outputs 사용
 * - random_utils.hpp: RandomInt() — 무작위 오프셋 생성
 *
 * === 주요 함수/구조체 요약 ===
 * - PIM(): DenseAllocator 초기화 + _PIM_iter 저장
 * - ~PIM(): 빈 소멸자 (DenseAllocator가 _request 해제)
 * - Allocate(): _PIM_iter 횟수 반복하며 Grant(무작위 입력 선택) + Accept(무작위 출력 선택)
 * - _PIM_iter: Grant-Accept 반복 횟수 (설정 파일 "alloc_iters"로 지정)
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

#ifndef _PIM_HPP_
#define _PIM_HPP_

#include <vector>

#include "allocator.hpp"

class PIM : public DenseAllocator {
  int _PIM_iter;
  /* [한국어] Grant-Accept 반복 횟수. 횟수가 많을수록 매칭 효율이 높아진다.
   * iSLIP이 라운드로빈 포인터를 사용하는 것과 달리, PIM은 RandomInt()로 완전 무작위 중재.
   * 설정자: PIM() 생성자에서 iters 인자를 받아 저장.
   * 읽는 자: Allocate()의 for(iter) 루프 상한으로 사용.
   * 값 범위: 1 이상 정수. gpgpusim.config의 "alloc_iters"로 제어. */

public:
  /*
   * [한국어] PIM 생성자 — DenseAllocator 초기화 및 반복 횟수 저장
   * @parent, @name: 모듈 계층
   * @inputs, @outputs: 입출력 포트 수
   * @iters: Grant-Accept 최대 반복 횟수
   * 호출 체인: Allocator::NewAllocator("pim") → PIM() → DenseAllocator()
   */
  PIM( Module *parent, const string& name,
       int inputs, int outputs, int iters );

  /*
   * [한국어] PIM 소멸자 — 별도 해제 불필요 (DenseAllocator가 _request 배열 해제)
   */
  ~PIM( );

  /*
   * [한국어] Allocate — _PIM_iter 반복 무작위 Grant-Accept 매칭
   * 각 반복마다:
   *   Grant: 출력이 요청 중 미매칭 입력을 RandomInt 오프셋으로 무작위 선택
   *   Accept: 입력이 자신에게 Grant된 출력을 RandomInt 오프셋으로 무작위 선택
   * 포인터 갱신 없이 완전 무작위 — 공정성은 확률적으로만 보장.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → PIM::Allocate()
   */
  void Allocate( );
};

#endif
