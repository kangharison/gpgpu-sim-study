// $Id: maxsize.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] MaxSizeMatch(최대 크기 이분 매칭) 할당기 선언 (maxsize.hpp)
 *
 * === 파일의 역할 ===
 * 최단 증가 경로(Shortest Augmenting Path) BFS 알고리즘을 사용한 최대 이분 매칭 할당기를 선언한다.
 * 이 알고리즘은 이분 그래프(입력 포트 집합 ↔ 출력 포트 집합)에서 가능한 최대의 매칭 수를 찾는다.
 * 복잡도는 O(N^3)으로 교통량이 많을 때 최적 처리량을 보장한다. 단, 공정성은 _prio 포인터로만
 * 부분적으로 보장하므로 스타베이션이 발생할 수 있다. GPU NoC에서는 처리량 최대화가 중요한
 * 시뮬레이션 실험 기준 알고리즘으로 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("maxsize")로 생성.
 * IQRouter → MaxSizeMatch::Allocate() — 스위치/VC 할당에 사용 가능.
 * DenseAllocator 상속 — _request[inputs][outputs] 2D 배열로 요청 저장.
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator (상속): _request, _inmatch, _outmatch, _inputs, _outputs 사용
 * - _ShortestAugmenting(): 내부 BFS 증가 경로 탐색 함수 — Allocate()가 반복 호출
 *
 * === 주요 함수/구조체 요약 ===
 * - MaxSizeMatch(): _from 벡터 + _s/_ns 동적 배열 할당, _prio=0 초기화
 * - ~MaxSizeMatch(): _s, _ns 해제
 * - Allocate(): _ShortestAugmenting()이 false를 반환할 때까지 반복 호출 → 최대 매칭
 * - _ShortestAugmenting(): BFS로 최단 증가 경로를 탐색, 발견 시 경로를 따라 매칭 보강
 * - _from[j]: 출력 j에 도달한 직전 입력 노드 (BFS 트리의 부모 포인터, -1=미방문)
 * - _s/_ns: BFS 현재/다음 레벨의 미매칭 입력 스택
 * - _prio: 다음 사이클 BFS 시작 입력 번호 (공정성을 위한 라운드로빈 오프셋)
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

#ifndef _MAXSIZE_HPP_
#define _MAXSIZE_HPP_

#include <vector>

#include "allocator.hpp"

class MaxSizeMatch : public DenseAllocator {
  vector<int> _from;   // array to hold breadth-first tree
  /* [한국어] BFS 트리의 부모 포인터 배열. _from[j] = 출력 j에 최단 경로로 도달한 입력 i.
   * BFS가 출력 j를 방문할 때 직전에 방문한 입력 번호를 기록하여 경로 역추적에 사용한다.
   * 설정자: _ShortestAugmenting()에서 BFS 진행 중 _from[j] = i로 설정.
   * 읽는 자: 증가 경로 발견 후 found_augmenting 레이블에서 경로를 역추적할 때 사용.
   * 값 범위: -1(미방문) 또는 0.._inputs-1. 각 _ShortestAugmenting() 호출마다 -1로 리셋. */

  int *_s;      // stack of leaf nodes in tree
  /* [한국어] BFS 현재 레벨의 미매칭 입력 노드 스택.
   * 처음에는 모든 미매칭 입력이 이 스택에 들어가고, 각 BFS 레벨 처리 후 _ns와 교환된다.
   * 설정자: MaxSizeMatch() 생성자에서 new int[inputs]로 할당.
   * 동기화: 단일 사이클 내 _ShortestAugmenting() 호출 중에만 사용. */

  int *_ns;     // next stack
  /* [한국어] BFS 다음 레벨의 입력 노드 스택.
   * 출력 j가 매칭된 입력 _outmatch[j]가 다음 레벨 탐색 대상으로 여기에 추가된다.
   * 처리 후 _s와 포인터만 교환(swap)하여 BFS 레벨 전환을 효율적으로 수행한다. */

  int _prio;    // priority pointer to ensure fairness
  /* [한국어] 공정성을 위한 라운드로빈 시작 오프셋. 매 Allocate() 호출마다 1 증가.
   * _ShortestAugmenting()에서 미매칭 입력을 스택에 넣을 때 _prio 오프셋부터 순환하여
   * 매 사이클 다른 입력부터 BFS를 시작함으로써 장기적 공정성을 부분적으로 보장한다.
   * 설정자: Allocate() 마지막에 (_prio+1) % _inputs로 갱신.
   * 값 범위: 0.._inputs-1. */

  /*
   * [한국어] _ShortestAugmenting — BFS로 최단 증가 경로 탐색 후 매칭 보강
   * @return: true이면 증가 경로 발견 및 매칭 보강 완료; false이면 경로 없음(최대 매칭 완료)
   *
   * 이분 매칭 이론에서 증가 경로(augmenting path)란:
   *   미매칭 입력 → 매칭 안된 출력으로의 경로, 또는
   *   미매칭 입력 → 매칭된 출력 → (이미 매칭된 입력으로 역방향) → 다른 출력 ... → 미매칭 출력
   * 이 경로를 찾으면 경로 상의 매칭을 반전(augment)하여 매칭 수를 1 늘릴 수 있다.
   *
   * 호출 체인: MaxSizeMatch::Allocate() → while(_ShortestAugmenting())
   */
  bool _ShortestAugmenting( );

public:
  /*
   * [한국어] MaxSizeMatch 생성자 — BFS 관련 자료구조 초기화
   * @parent, @name: 모듈 계층 위치
   * @inputs, @outputs: 입출력 포트 수
   * 호출 체인: Allocator::NewAllocator("maxsize") → MaxSizeMatch()
   */
  MaxSizeMatch( Module *parent, const string& name,
		int inputs, int ouputs );

  /*
   * [한국어] MaxSizeMatch 소멸자 — _s, _ns 동적 배열 해제
   */
  ~MaxSizeMatch( );

  /*
   * [한국어] Allocate — 최단 증가 경로 반복으로 최대 이분 매칭 수행 (O(N^3))
   * _ShortestAugmenting()을 더 이상 증가 경로가 없을 때까지 반복 호출하여
   * 최대 매칭을 구성하고, 마지막에 _prio를 1 전진한다.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → MaxSizeMatch::Allocate()
   */
  void Allocate( );
};

#endif 
