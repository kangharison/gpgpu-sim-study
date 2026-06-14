// $Id: prio_arb.hpp 5188 2012-08-30 00:31:31Z dub $

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

/*
 * [한국어 설명] PriorityArbiter 선언 (prio_arb.hpp)
 *
 * === 파일의 역할 ===
 * SelAlloc 내부에서 사용하는 우선순위 기반 라운드로빈 중재자를 선언한다.
 * Arbiter 클래스와 달리 SparseAllocator 스타일의 list<sRequest>로 요청을 관리하여
 * RemoveRequest()가 가능하다. Arbitrate()는 _rr_ptr 오프셋 기준으로 최고 우선순위 입력을 선택한다.
 * GPGPU-Sim에 통합 시 intersim2/icnt_wrapper.cc에서 SelAlloc이 이 중재자를 활용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SelAlloc → PriorityArbiter (내부 중재자)
 *
 * === 타 모듈과의 연결 ===
 * - Module (상속): 계층적 이름 지원
 * - config_utils.hpp: Configuration — 설정 파일 파라미터 읽기 (현재는 미사용)
 *
 * === 주요 함수/구조체 요약 ===
 * - sRequest: 요청 엔트리 (in=입력번호, label=식별자, pri=우선순위)
 * - _requests: in 번호 오름차순으로 정렬된 요청 리스트
 * - _rr_ptr: 라운드로빈 탐색 시작 입력 번호 (Arbitrate()에서 갱신)
 * - _match: 마지막 Arbitrate() 결과 (-1이면 매칭 없음)
 * - Arbitrate(): _rr_ptr 오프셋 기준 순환 탐색으로 최고 우선순위 입력 선택
 * - Update(): _rr_ptr을 1 전진
 */

#ifndef _PRIO_ARB_HPP_
#define _PRIO_ARB_HPP_

#include <list>

#include "module.hpp"
#include "config_utils.hpp"

class PriorityArbiter : public Module {
  int _rr_ptr;
  /* [한국어] 라운드로빈 탐색 시작 입력 번호. Arbitrate()에서 _rr_ptr 기준으로 순환 탐색하여
   * 동점 시 _rr_ptr에 가까운 입력이 우선 선택된다.
   * 설정자: Arbitrate()에서 매칭 성공 시 (_match+1)%_inputs로 갱신; Update()에서도 1 전진.
   * 읽는 자: Arbitrate()의 라운드로빈 탐색 오프셋.
   * 값 범위: 0.._inputs-1. 사이클 간 유지됨. */

protected:
  const int _inputs;
  /* [한국어] 이 중재자가 처리하는 입력 포트 수. 생성자에서 설정되고 이후 불변.
   * _rr_ptr의 모듈러 연산 기준으로 사용됨. */

  struct sRequest {
    int in;    /* [한국어] 요청하는 입력 포트 번호 */
    int label; /* [한국어] 요청 식별자 (레이블) */
    int pri;   /* [한국어] 요청 우선순위 — 높을수록 먼저 선택 */
  };
  /* [한국어] 각 요청의 정보를 담는 구조체.
   * AddRequest()에서 in 번호 오름차순으로 _requests 리스트에 삽입된다.
   * RemoveRequest()에서 in 번호로 검색하여 제거된다. */

  list<sRequest> _requests;
  /* [한국어] 현재 등록된 모든 요청을 in 번호 오름차순으로 관리하는 리스트.
   * 중복 요청 시 더 높은 우선순위만 유지된다 (AddRequest 로직).
   * 설정자: AddRequest()에서 삽입; RemoveRequest()에서 제거; Clear()에서 전체 삭제.
   * 읽는 자: Arbitrate()에서 _rr_ptr 기준 순환 탐색. */

  int _match;
  /* [한국어] 마지막 Arbitrate() 결과. 선택된 입력 번호, 없으면 -1.
   * 설정자: Arbitrate()에서 설정.
   * 읽는 자: Match() 접근자를 통해 외부에서 읽음. */

public:
  /*
   * [한국어] PriorityArbiter 생성자 — _rr_ptr=0, _inputs 설정
   * @config: Configuration (현재 미사용, 인터페이스 호환을 위해 유지)
   * @inputs: 입력 포트 수
   * 호출 체인: SelAlloc() → PriorityArbiter()
   */
  PriorityArbiter( const Configuration &config,
		   Module *parent, const string& name,
		   int inputs );

  /*
   * [한국어] Clear — 모든 요청 리스트 초기화
   */
  void Clear( );

  /*
   * [한국어] AddRequest — in 번호 오름차순 위치에 삽입; 중복 시 더 높은 우선순위만 유지
   * @in: 입력 포트 번호
   * @label: 요청 식별자
   * @pri: 우선순위 (기본값 0)
   */
  void AddRequest( int in, int label = 0, int pri = 0 );

  /*
   * [한국어] RemoveRequest — in 번호로 검색하여 _requests에서 제거
   */
  void RemoveRequest( int in, int label = 0 );

  /*
   * [한국어] Match — 마지막 Arbitrate() 결과 반환 (const)
   */
  int Match( ) const;

  /*
   * [한국어] Arbitrate — _rr_ptr 기준 순환 탐색으로 최고 우선순위 입력 선택
   * 최고 우선순위(동점 시 _rr_ptr에 가까운)를 _match에 저장하고 _rr_ptr을 갱신한다.
   */
  void Arbitrate( );

  /*
   * [한국어] Update — _rr_ptr을 1 전진 (Arbitrate() 없이 포인터만 갱신)
   */
  void Update( );
};

#endif 
