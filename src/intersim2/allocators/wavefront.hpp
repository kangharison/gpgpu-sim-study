// $Id: wavefront.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] Wavefront(파면) 크로스바 할당기 선언 (wavefront.hpp)
 *
 * === 파일의 역할 ===
 * Wavefront 알고리즘을 구현하는 Wavefront 클래스를 선언한다. 이 알고리즘은 요청 행렬의
 * 대각선(diagonal)을 우선순위 순서로 탐색하여 충돌 없이 최대 매칭을 찾는다.
 * "파면(wavefront)"은 대각선이 행렬을 가로질러 이동하는 모습에서 유래했다.
 * 행렬의 모든 대각선을 순서대로 처리하므로 공정성이 보장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Allocator::NewAllocator("wavefront") 또는 "rr_wavefront"로 생성.
 * DenseAllocator를 상속하여 전체 요청 행렬(_request)을 사용한다.
 *
 * === 타 모듈과의 연결 ===
 * - DenseAllocator: 상속 — _request 2D 배열, AddRequest 기반 클래스
 *
 * === 주요 함수/구조체 요약 ===
 * - Wavefront(): 생성자 — _last_in/out, _skip_diags, _square, _pri, _num_requests 초기화
 * - AddRequest(): DenseAllocator 등록 후 _num_requests++, _priorities에 우선순위 쌍 추가
 * - Allocate(): _priorities를 역순(높은 우선순위)으로 대각선 탐색하여 매칭 결정
 * - _skip_diags: true이면 rr_wavefront — 매칭된 대각선 이후로 _pri 갱신 (라운드로빈)
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

#ifndef _WAVEFRONT_HPP_
#define _WAVEFRONT_HPP_

#include <set>

#include "allocator.hpp"

class Wavefront : public DenseAllocator {

private:
  int _last_in;
  /* 마지막으로 AddRequest()된 입력 포트. 요청이 1개인 경우 즉시 그 입력을 매칭하는 최적화에 사용.
   * 설정자: AddRequest() 호출마다 갱신. Allocate() 후 -1로 리셋.
   * 읽는 자: Allocate()에서 _num_requests==1일 때 즉시 매칭. */
  int _last_out;
  /* 마지막으로 AddRequest()된 출력 포트. _last_in과 짝으로 사용됨. */
  set<pair<int, int> > _priorities;
  /* AddRequest() 시 등록된 (out_pri, in_pri) 쌍들의 집합. 중복 없이 내림차순 정렬됨.
   * Allocate()에서 rbegin()부터 역순(높은 우선순위)으로 순회하며 우선순위별 대각선 탐색.
   * 설정자: AddRequest()에서 make_pair(out_pri, in_pri) 삽입.
   * 읽는 자: Allocate()의 외부 루프에서 우선순위 레벨 결정. */
  bool _skip_diags;
  /* true이면 rr_wavefront 모드 — Allocate() 후 첫 매칭이 발생한 대각선+1로 _pri 이동.
   * false이면 단순 wavefront — _pri는 1씩 증가하여 모든 대각선을 순환. */

protected:
  int _square;
  /* max(inputs, outputs) — 정사각 행렬 탐색에 필요한 변. 비정방 행렬에서도 안전하게 탐색. */
  int _pri;
  /* 현재 우선순위 대각선 번호 (0.._square-1). 이 대각선부터 탐색 시작.
   * 설정자: Allocate() 마지막에 갱신 (_skip_diags 여부에 따라 first_diag+1 또는 _pri+1).
   * 사이클 간 유지되어 라운드로빈 공정성을 구현함. */
  int _num_requests;
  /* 이번 사이클에 AddRequest()가 호출된 횟수. 0이면 Allocate() 즉시 리턴. 1이면 즉시 매칭.
   * 설정자: AddRequest()에서 증가. Allocate() 마지막에 0으로 리셋.
   * 읽는 자: Allocate() 앞부분에서 최적화 분기에 사용. */

public:
  /*
   * [한국어] Wavefront 생성자
   * @parent, @name: 부모/이름
   * @inputs, @outputs: 입출력 포트 수
   * @skip_diags: true이면 rr_wavefront(동적 우선순위), false이면 wavefront(정적 대각선 순환)
   */
  Wavefront( Module *parent, const string& name,
	     int inputs, int outputs, bool skip_diags = false );

  /*
   * [한국어] AddRequest - DenseAllocator에 등록 후 카운터/우선순위 집합 갱신
   * _num_requests 증가, _last_in/_last_out 갱신, _priorities에 (out_pri, in_pri) 삽입.
   */
  virtual void AddRequest( int in, int out, int label = 1,
			   int in_pri = 0, int out_pri = 0 );

  /*
   * [한국어] Allocate - 우선순위 순서로 대각선을 탐색하여 충돌 없는 최대 매칭 결정
   * _priorities를 높은 우선순위 순으로 순회하며 각 우선순위 레벨에서 _pri 대각선부터 탐색.
   * 매칭 후 _num_requests, _last_in, _last_out, _priorities를 초기화하고 _pri를 갱신.
   */
  virtual void Allocate( );
};

#endif
