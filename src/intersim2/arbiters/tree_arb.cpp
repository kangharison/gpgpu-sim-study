// $Id: tree_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] TreeArbiter 구현 (tree_arb.cpp)
 *
 * === 파일의 역할 ===
 * 2단계 계층 트리 중재자 구현. 생성자에서 groups개 그룹 중재자와 1개의 전역 중재자를 생성하고,
 * Arbitrate()에서 2단계로 중재한다: 각 그룹의 로컬 승자 → 전역 중재자로 최종 승자 결정.
 * UpdateState()는 승리한 그룹의 그룹 중재자와 전역 중재자만 갱신한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Arbiter::NewArbiter("tree(N,type)") → TreeArbiter
 *
 * === 타 모듈과의 연결 ===
 * - Arbiter (상속): AddRequest(), Arbitrate(), Clear() super 호출
 * - Arbiter::NewArbiter(): 그룹/전역 중재자 생성
 * - BookSimConfig / gpgpusim.config:
 *     * arb_type = "tree(groups,sub_arb_type)"일 때 NewArbiter()가 이 클래스를 생성.
 *     * sub_arb_type("round_robin"/"matrix")이 그룹/전역 중재자 알고리즘으로 사용됨.
 *     * alloc_iters가 1 이상이면 매 사이클 그룹/전역 중재자의 UpdateState()가 반복 호출됨.
 *
 * === 주요 함수/구조체 요약 ===
 * - TreeArbiter(): groups개 그룹 중재자(크기=_group_size) + 전역 중재자(크기=groups) 생성
 * - ~TreeArbiter(): 모든 동적 할당 중재자 해제
 * - Arbitrate(): 2단계 중재 → _selected = group*_group_size + group_sel
 * - UpdateState(): 승리 그룹의 그룹 중재자 + 전역 중재자 UpdateState()
 */

// ----------------------------------------------------------------------
//
//  TreeArbiter
//
// ----------------------------------------------------------------------

#include "tree_arb.hpp" // [한국어] TreeArbiter 선언
#include <iostream>     // [한국어] PrintState()의 cout 사용
#include <sstream>      // [한국어] ostringstream — 그룹 중재자 이름 생성

using namespace std ;

/*
 * [한국어] TreeArbiter 생성자 — 그룹/전역 중재자 생성 및 초기화
 * @size: 전체 입력 포트 수 (size % groups == 0이어야 함)
 * @groups: 그룹 수 (_global_arbiter 크기)
 * @arb_type: 그룹/전역 중재자 타입
 * 호출 체인: Arbiter::NewArbiter("tree(...)") → TreeArbiter() → Arbiter()
 */
TreeArbiter::TreeArbiter( Module *parent, const string &name,
			  int size, int groups, const string & arb_type )
  : Arbiter( parent, name, size ) {  // [한국어] Arbiter 기반 초기화 (_request 배열 등)
  assert(size % groups == 0);        // [한국어] size가 groups의 배수여야 함
  _group_arbiters.resize(groups);    // [한국어] groups개 그룹 중재자 포인터 배열 할당
  _group_reqs.resize(groups, 0);     // [한국어] 그룹별 요청 수 배열 초기화 (모두 0)
  _group_size = size / groups;       // [한국어] 그룹당 입력 수 계산
  for(int i = 0; i < groups; ++i) { // [한국어] 각 그룹 중재자 생성
    ostringstream group_arb_name;
    group_arb_name << "group_arb" << i; // [한국어] 이름: "group_arb0", "group_arb1", ...
    _group_arbiters[i] = Arbiter::NewArbiter(this, group_arb_name.str(), arb_type, _group_size);
    // [한국어] 그룹 i의 중재자 생성 (크기=_group_size, 부모=this)
  }
  _global_arbiter = Arbiter::NewArbiter(this, "global_arb", arb_type, groups);
  // [한국어] 전역 중재자 생성 (크기=groups — 그룹 수가 입력)
}

/*
 * [한국어] ~TreeArbiter — 동적 할당된 중재자들 해제
 */
TreeArbiter::~TreeArbiter() {
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) {
    delete _group_arbiters[i]; // [한국어] 각 그룹 중재자 해제
  }
  delete _global_arbiter; // [한국어] 전역 중재자 해제
}

/*
 * [한국어] PrintState — 모든 그룹/전역 중재자의 상태를 출력 (디버그용)
 */
void TreeArbiter::PrintState() const  {
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) { // [한국어] 각 그룹 중재자 출력
    cout << "Group arbiter " << i << ":" << endl;
    _group_arbiters[i]->PrintState();
  }
  cout << "Global arbiter:" << endl;
  _global_arbiter->PrintState(); // [한국어] 전역 중재자 출력
}

/*
 * [한국어] TreeArbiter::UpdateState — 마지막 승리 그룹의 상태 갱신
 * _global_arbiter->LastWinner()로 승리 그룹을 얻어 그 그룹 중재자와 전역 중재자만 갱신.
 * 승리하지 않은 그룹은 상태 유지 (공정성 보존).
 */
void TreeArbiter::UpdateState() {
  if(_selected > -1) {                             // [한국어] 선택된 입력이 있을 때만 갱신
    int last_winner = _global_arbiter->LastWinner(); // [한국어] 마지막으로 선택된 그룹 번호
    assert(last_winner >= 0 && last_winner < (int)_group_arbiters.size());
    _group_arbiters[last_winner]->UpdateState();   // [한국어] 승리 그룹 중재자 상태 갱신
    _global_arbiter->UpdateState();                // [한국어] 전역 중재자 상태 갱신
  }
}

/*
 * [한국어] TreeArbiter::AddRequest — 그룹 중재자에 요청 등록 + _group_reqs 증가
 * @input: 전체 입력 포트 번호
 * 그룹 번호 = input / _group_size, 그룹 내 포트 번호 = input % _group_size.
 */
void TreeArbiter::AddRequest( int input, int id, int pri )
{
  Arbiter::AddRequest(input, id, pri);             // [한국어] 기반 _request[input] 등록 (super)
  int group_index = input / _group_size;           // [한국어] 이 입력이 속한 그룹 번호 계산
  _group_arbiters[group_index]->AddRequest( input % _group_size, id, pri );
  // [한국어] 그룹 내 로컬 포트 번호로 그룹 중재자에 등록
  ++_group_reqs[group_index];                      // [한국어] 이 그룹의 요청 수 증가
}

/*
 * [한국어] TreeArbiter::Arbitrate — 2단계 트리 중재
 * 1단계: 요청이 있는 각 그룹에서 그룹 중재자를 실행하여 로컬 승자(group_id, group_pri) 결정
 * 2단계: 각 그룹의 로컬 승자를 전역 중재자에 AddRequest()하여 최종 승리 그룹 결정
 * 결과: _selected = group * _group_size + group_sel
 */
int TreeArbiter::Arbitrate( int* id, int* pri ) {
  if(!_num_reqs) {      // [한국어] 요청이 전혀 없으면 즉시 -1 반환
    return -1;
  }
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) { // [한국어] 각 그룹 처리
    if(_group_reqs[i]) {                                   // [한국어] 이 그룹에 요청이 있으면
      int group_id, group_pri;
      _group_arbiters[i]->Arbitrate(&group_id, &group_pri); // [한국어] 그룹 내 로컬 중재 수행
      _global_arbiter->AddRequest(i, group_id, group_pri);  // [한국어] 로컬 승자를 전역 중재자에 등록
    }
  }
  int group = _global_arbiter->Arbitrate(NULL, NULL); // [한국어] 전역 중재 수행 → 승리 그룹 결정
  assert(group >= 0 && group < (int)_group_arbiters.size());
  int group_sel = _group_arbiters[group]->LastWinner(); // [한국어] 승리 그룹의 로컬 승자 번호
  assert(group_sel >= 0 && group_sel < _group_size);
  _selected = group * _group_size + group_sel; // [한국어] 전역 입력 번호로 변환
  assert(_selected >= 0 && _selected < _size);
  return Arbiter::Arbitrate(id, pri); // [한국어] id/pri 포인터 채우기 + _selected 반환
}

/*
 * [한국어] TreeArbiter::Clear — 요청이 있을 때 모든 그룹/전역 중재자 초기화
 * 요청이 없으면 즉시 리턴하여 불필요한 루프 방지.
 */
void TreeArbiter::Clear()
{
  if(!_num_reqs) { // [한국어] 요청이 없으면 즉시 리턴 (최적화)
    return;
  }
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) { // [한국어] 각 그룹 중재자 초기화
    _group_arbiters[i]->Clear();
    _group_reqs[i] = 0; // [한국어] 그룹별 요청 수 리셋
  }
  _global_arbiter->Clear(); // [한국어] 전역 중재자 초기화
  Arbiter::Clear();         // [한국어] _request[].valid=false, _num_reqs=0, _selected=-1
}
