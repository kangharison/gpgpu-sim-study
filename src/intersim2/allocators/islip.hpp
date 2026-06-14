// $Id: islip.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] iSLIP 할당기 선언 (islip.hpp)
 *
 * === 파일의 역할 ===
 * iSLIP(Iterative Schedule with LInear Priority) 알고리즘을 구현하는 iSLIP_Sparse 클래스를
 * 선언한다. iSLIP은 Nick McKeown이 제안한 반복 라운드로빈 매칭 알고리즘으로, 입력 큐드
 * 크로스바 스위치에서 최대 처리량에 수렴하면서 공정성을 보장한다.
 * GPGPU-Sim의 GPU NoC에서 기본 스위치 할당기로 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터 내 IQRouter의 스위치 할당(_sw_allocator)으로 동작한다.
 * 매 사이클: Clear() → AddRequest() × N → Allocate() → OutputAssigned()로 결과 조회.
 * Allocate() 내부는 Grant 단계(출력→입력 선택)와 Accept 단계(입력→출력 선택)로 나뉘며,
 * 최대 _iSLIP_iter 회 반복한다. 1회 반복만으로도 높은 처리량 달성 가능.
 *
 * === 타 모듈과의 연결 ===
 * - SparseAllocator: 상속 — _in_req/_out_req 맵과 _in_occ/_out_occ 집합 사용
 * - Allocator::NewAllocator(): "islip" 또는 "islip(N)" 타입으로 인스턴스 생성
 * - IQRouter: _sw_allocator 및 _vc_allocator로 보유
 *
 * === 주요 함수/구조체 요약 ===
 * - iSLIP_Sparse(): 생성자 — _gptrs(Grant 포인터)와 _aptrs(Accept 포인터)를 0으로 초기화
 * - Allocate(): iSLIP 알고리즘 실행 — _iSLIP_iter회 반복하여 최대 매칭 수행
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

#ifndef _ISLIP_HPP_
#define _ISLIP_HPP_

#include <vector>       // [한국어] _gptrs, _aptrs 벡터 사용
#include "allocator.hpp" // [한국어] SparseAllocator 기반 클래스

/*
 * [한국어] iSLIP_Sparse - 희소 저장 기반 iSLIP 크로스바 할당기
 *
 * SparseAllocator를 상속하여 요청 맵을 사용하고, 그 위에 iSLIP 알고리즘을 구현한다.
 * iSLIP은 Grant와 Accept 두 단계를 최대 _iSLIP_iter회 반복하며 매칭을 수행한다.
 * Grant: 각 출력이 요청 중인 입력을 라운드로빈(_gptrs)으로 하나 선택.
 * Accept: 각 입력이 자신을 승인한 출력 중 라운드로빈(_aptrs)으로 하나 수락.
 */
class iSLIP_Sparse : public SparseAllocator {
  int _iSLIP_iter;
  /* iSLIP 반복 횟수.
   * 반복을 늘릴수록 더 많은 매칭을 찾을 수 있으나 사이클당 연산 비용 증가.
   * 실제 GPU NoC에서는 1~4회가 일반적이며, 1회에도 높은 처리량 달성 가능.
   * 설정자: 생성자에서 iters 파라미터로 설정. const로 고정.
   * 읽는 자: Allocate() 반복 루프의 종료 조건.
   * 값 범위: 1 이상 양의 정수. 설정 파일의 "alloc_iters" 값.
   * 동기화: 변경 없음. 락 불필요. */

  vector<int> _gptrs;
  /* Grant 단계 라운드로빈 포인터 배열. _gptrs[output] = 다음 승인 시작할 입력 인덱스.
   * iSLIP의 공정성 핵심 — 마지막으로 승인한 입력의 다음부터 탐색 시작.
   * 설정자: Allocate()에서 첫 번째 반복(iter==0)의 Accept 성공 시 (input+1)%_inputs로 갱신.
   *         이후 반복에서는 갱신하지 않아 공정성 유지.
   * 읽는 자: Allocate() Grant 단계에서 _out_req[output] 순회 시작 위치로 사용.
   * 값 범위: 0.._inputs-1. 초기값 0.
   * 동기화: 사이클 간에 상태가 유지됨. 단일 스레드. */

  vector<int> _aptrs;
  /* Accept 단계 라운드로빈 포인터 배열. _aptrs[input] = 다음 수락 시작할 출력 인덱스.
   * _gptrs와 대칭 구조 — 마지막으로 수락한 출력의 다음부터 탐색 시작.
   * 설정자: Allocate()에서 첫 번째 반복(iter==0)의 Accept 성공 시 (output+1)%_outputs로 갱신.
   * 읽는 자: Allocate() Accept 단계에서 _in_req[input] 순회 시작 위치로 사용.
   * 값 범위: 0.._outputs-1. 초기값 0.
   * 동기화: 사이클 간에 상태 유지. 단일 스레드. */

public:
  /*
   * [한국어] iSLIP_Sparse 생성자 — Grant/Accept 포인터를 0으로 초기화
   * @parent: 부모 모듈 (IQRouter)
   * @name: 디버그용 이름
   * @inputs: 입력 포트 수
   * @outputs: 출력 포트 수
   * @iters: iSLIP 반복 횟수 (설정 파일의 alloc_iters 또는 기본값 1)
   * 호출 체인: Allocator::NewAllocator() → iSLIP_Sparse() → SparseAllocator()
   */
  iSLIP_Sparse( Module *parent, const string& name,
		int inputs, int outputs, int iters );

  /*
   * [한국어] Allocate - iSLIP 알고리즘 실행으로 크로스바 매칭 결정
   * _iSLIP_iter회 반복하며 Grant → Accept 2단계를 수행한다.
   * 결과는 _inmatch[input] = output, _outmatch[output] = input에 기록된다.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → iSLIP_Sparse::Allocate()
   */
  void Allocate( );
};

#endif 
