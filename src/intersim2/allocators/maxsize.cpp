// $Id: maxsize.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] MaxSizeMatch(최단 증가 경로 BFS) 할당기 구현 (maxsize.cpp)
 *
 * === 파일의 역할 ===
 * 이분 그래프 최대 매칭을 최단 증가 경로(BFS) 알고리즘으로 구현한다.
 * Allocate()는 _ShortestAugmenting()을 증가 경로가 없을 때까지 반복 호출하여
 * 최대 크기의 매칭을 구성한다. 알고리즘 복잡도는 O(N^3)이다.
 * 공정성은 _prio 포인터의 라운드로빈으로 부분적으로 보장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * IQRouter → MaxSizeMatch::Allocate()
 * 최대 매칭이 필요한 스위치/VC 할당 단계에서 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 (_request, _inmatch, _outmatch)
 *
 * === 주요 함수/구조체 요약 ===
 * - MaxSizeMatch(): _from 벡터 + _s/_ns 배열 할당, _prio=0
 * - ~MaxSizeMatch(): _s, _ns 해제
 * - Allocate(): while(_ShortestAugmenting()) 반복 후 _prio 전진
 * - _ShortestAugmenting(): BFS로 증가 경로를 찾아 매칭 보강; 없으면 false 반환
 */

/*
  Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this 
  list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.
  Neither the name of the Stanford University nor the names of its contributors 
  may be used to endorse or promote products derived from this software without 
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
  POSSIBILITY OF SUCH DAMAGE.
*/

#include "booksim.hpp"
#include <iostream>

#include "maxsize.hpp"

// shortest augmenting path:
//
// for all unmatched left nodes,
//    push node onto work stack
// end
//
// for all j,
//   from[j] = undefined
// end
//
// do,
//
//   while( !stack.empty ),
//     
//     nl = stack.pop
//     for each edge (nl,j),
//       if ( ( lmatch[nl] != j ) && ( from[j] == undefined ) ),
//         if ( rmatch[j] == undefined ),
//           stop // augmenting path found
//         else
//           from[j] = nl
//           newstack.push( rmatch[j] ) 
//         end
//       end
//     end
//   end
//
//   stack = newstack
// end
//

//#define DEBUG_MAXSIZE
//#define PRINT_MATCHING

/*
 * [한국어] MaxSizeMatch 생성자 — BFS 보조 자료구조 할당 및 초기화
 * @parent, @name: 모듈 계층
 * @inputs, @outputs: 포트 수
 * 호출 체인: Allocator::NewAllocator("maxsize") → MaxSizeMatch() → DenseAllocator()
 */
MaxSizeMatch::MaxSizeMatch( Module *parent, const string& name,
			    int inputs, int outputs ) :
  DenseAllocator( parent, name, inputs, outputs ) // [한국어] DenseAllocator 초기화 (_request 2D 배열)
{
  _from.resize(outputs);          // [한국어] BFS 부모 포인터 배열 — 출력 수만큼
  _s    = new int [inputs];       // [한국어] BFS 현재 레벨 스택 — 입력 수만큼
  _ns   = new int [inputs];       // [한국어] BFS 다음 레벨 스택 — 입력 수만큼
  _prio = 0;                      // [한국어] 공정성 오프셋 초기화
}

/*
 * [한국어] MaxSizeMatch 소멸자 — 동적 할당 배열 해제
 */
MaxSizeMatch::~MaxSizeMatch( )
{
  delete [] _s;  // [한국어] BFS 현재 레벨 스택 해제
  delete [] _ns; // [한국어] BFS 다음 레벨 스택 해제
}

/*
 * [한국어] MaxSizeMatch::Allocate — 최단 증가 경로 반복으로 최대 이분 매칭 구성
 *
 * _ShortestAugmenting()을 더 이상 증가 경로가 없을 때까지 반복 호출한다.
 * 각 호출마다 매칭 수가 1씩 증가하므로 전체 복잡도는 O(N^3)이다.
 * 마지막에 _prio를 1 전진하여 다음 사이클 BFS 시작 입력을 변경함으로써 공정성을 보장한다.
 *
 * 호출 체인: IQRouter::_SWAllocEvaluate() → MaxSizeMatch::Allocate()
 */
void MaxSizeMatch::Allocate( )
{

  // augment as many times as possible
  // (this is an O(N^3) maximum-size matching algorithm)
  // [한국어] 증가 경로가 없을 때까지 반복하여 최대 매칭 구성 (O(N^3))
  while( _ShortestAugmenting( ) );

  // next time, start at next input to ensure fairness
  // [한국어] 다음 사이클에서는 다음 입력부터 BFS를 시작하여 공정성 보장
  _prio = (_prio + 1) % _inputs;
}

/*
 * [한국어] MaxSizeMatch::_ShortestAugmenting — BFS로 최단 증가 경로를 탐색하고 매칭을 보강
 * @return: true = 증가 경로 발견 후 매칭 수 1 증가; false = 경로 없음(최대 매칭 완성)
 *
 * 알고리즘 개요 (최단 증가 경로 BFS):
 * 1. 모든 미매칭 입력을 초기 스택(_s)에 넣는다 (_prio 오프셋 순서로)
 * 2. _from을 모두 -1로 초기화 (미방문 상태)
 * 3. 각 BFS 레벨에서:
 *    - 스택의 입력 i마다 그 입력이 요청하는 모든 출력 j를 탐색
 *    - 에지 (i,j)가 유효하고(_request[i][j].label!=-1), 현재 매칭 에지가 아니고, _from[j]==-1이면:
 *      · _from[j] = i 기록 (경로 추적용)
 *      · j가 미매칭(_outmatch[j]==-1): 증가 경로 발견! found_augmenting으로 점프
 *      · j가 매칭됨: _outmatch[j](=j와 매칭된 입력)를 다음 스택(_ns)에 추가
 * 4. 스택 교환 후 반복 (최대 _inputs 레벨)
 * 5. 증가 경로 발견 시:
 *    - _from을 따라 경로를 역추적하며 매칭을 반전(augment)
 *    - _outmatch[j]=i, _inmatch[i]=j 순서로 갱신
 *
 * 호출 체인: MaxSizeMatch::Allocate() → while(_ShortestAugmenting())
 */
bool MaxSizeMatch::_ShortestAugmenting( )
{
  int i, j, jn;   // [한국어] i=입력 포트, j=출력 포트, jn=이전 매칭 출력(경로 역추적용)
  int slen, nslen; // [한국어] 현재/다음 스택 길이

  // start with empty stack
  slen = 0; // [한국어] 초기 스택 비움

  // push all unassigned inputs to the stack
  // [한국어] _prio 오프셋 순서로 모든 미매칭 입력을 스택에 추가
  for ( i = 0; i < _inputs; ++i ) {
    j = (i + _prio) % _inputs;            // [한국어] _prio 기준 순환 오프셋 적용
    if ( _inmatch[j] == -1 ) {            // start with unmatched left nodes
      // [한국어] 미매칭 입력만 초기 스택에 추가 — 이 입력들이 BFS 탐색의 출발점
      _s[slen++] = j;
    }
  }

  _from.assign(_inputs, -1); // [한국어] BFS 부모 배열 초기화 (-1 = 미방문)

  // [한국어] BFS 레벨 반복 (최대 _inputs 레벨 — 경로 길이 상한)
  for ( int iter = 0; iter < _inputs; iter++ ) {
    nslen = 0; // [한국어] 다음 레벨 스택 길이 초기화

    for ( int e = 0; e < slen; ++e ) { // [한국어] 현재 레벨 스택의 모든 입력 처리
      i = _s[e]; // [한국어] 현재 입력 포트

      for ( j = 0; j < _outputs; ++j ) { // [한국어] 이 입력이 요청하는 모든 출력 탐색
	if ( ( _request[i][j].label != -1 ) && // edge (i,j) exists
	     // [한국어] 에지 (i,j)가 요청 행렬에 존재하고
	     ( _inmatch[i] != j ) &&     // (i,j) is not contained in the current matching
	     // [한국어] 현재 매칭에 이 에지가 없고 (중복 방문 방지)
	     ( _from[j] == -1 ) ) {      // no shorter path to j exists
	  // [한국어] 아직 더 짧은 경로로 j에 도달한 적 없으면 탐색
	  _from[j] = i;                  // how did we get to j?
	  // [한국어] j에 도달한 경로의 직전 입력을 기록 (경로 역추적용 부모 포인터)

#ifdef DEBUG_MAXSIZE
	  cout << "  got to " << j << " from " << i << endl;
#endif
	  if ( _outmatch[j] == -1 ) {   // j is unmatched -- augmenting path found
	    // [한국어] j가 미매칭 출력 — 증가 경로를 발견! 경로 보강으로 점프
	    goto found_augmenting;
	  } else {                      // j is matched
	    // [한국어] j가 이미 매칭됨 — j의 매칭 파트너를 다음 BFS 레벨에 추가
	    _ns[nslen] = _outmatch[j];  // add the destination of this edge to the leaf nodes
	    // [한국어] j와 매칭된 입력(_outmatch[j])을 다음 탐색 대상으로 추가
	    nslen++;

#ifdef DEBUG_MAXSIZE
	    cout << "  adding " << _outmatch[j] << endl;
#endif
	  }
	}
      }
    }

    // no augmenting path found yet, swap stacks
    // [한국어] 이 레벨에서 증가 경로 없음 — 스택을 교환하고 다음 레벨로 진행
    int * t = _s; // [한국어] 임시 포인터로 _s, _ns 교환 (O(1))
    _s = _ns;
    _ns = t;
    slen = nslen; // [한국어] 다음 레벨 스택 길이 적용
  }

  return false; // no augmenting paths
  // [한국어] 모든 레벨 탐색 후에도 증가 경로 없음 — 현재 매칭이 최대

 found_augmenting:

  // the augmenting path ends at node j on the right
  // [한국어] 증가 경로의 오른쪽(출력 측) 끝 노드가 j — 경로를 역추적하며 매칭 보강

#ifdef DEBUG_MAXSIZE
  cout << "Found path: " << j << "c <- ";
#endif

  i = _from[j];    // [한국어] 출력 j에 도달한 직전 입력
  _outmatch[j] = i; // [한국어] 출력 j ↔ 입력 i 매칭 설정

#ifdef DEBUG_MAXSIZE
  cout << i;
#endif

  // [한국어] i가 이전 매칭을 갖고 있는 동안 (증가 경로 역방향 추적)
  while ( _inmatch[i] != -1 ) {  // loop until the end of the path
    jn = _inmatch[i];            // remove previous edge (i,jn) and add (i,j)
    // [한국어] 현재 i가 매칭된 이전 출력 jn을 저장 (이 에지를 끊고 새 에지(i,j)로 교체)
    _inmatch[i] = j; // [한국어] 입력 i의 매칭을 jn에서 j로 갱신

#ifdef DEBUG_MAXSIZE
    cout << " <- " << j << "c <- ";
#endif

    j = jn;                    // add edge from (jn,in)
    // [한국어] j를 이전 출력 jn으로 이동하여 다음 역추적 단계 진행
    i = _from[j]; // [한국어] jn에 도달한 직전 입력
    _outmatch[j] = i; // [한국어] 출력 jn ↔ 새 입력 i로 매칭 갱신

#ifdef DEBUG_MAXSIZE
    cout << i;
#endif
  }

#ifdef DEBUG_MAXSIZE
  cout << endl;
#endif

  _inmatch[i] = j; // [한국어] 경로의 끝(미매칭이었던 입력 i)을 출력 j에 최종 매칭

#ifdef PRINT_MATCHING
  cout << "left  matching: ";

  for ( i = 0; i < _inputs; i++ ) {
    cout << _inmatch[i] << " ";
  }
  cout << endl;

  cout << "right matching: ";
  for ( i = 0; i < _outputs; i++ ) {
    cout << _outmatch[i] << " ";
  }
  cout << endl;
#endif

  return true; // [한국어] 증가 경로 발견 및 매칭 보강 완료 — 매칭 수 1 증가
}
