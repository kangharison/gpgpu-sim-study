// $Id: allocator.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 * [한국어 설명] NoC 크로스바 할당기 추상 기반 클래스 및 Dense/Sparse 구현 (allocator.hpp)
 *
 * === 파일의 역할 ===
 * intersim2(BookSim2) NoC 시뮬레이터에서 IQ 라우터(Input-Queued Router)의 크로스바(crossbar)
 * 스위치 할당을 담당하는 Allocator 클래스 계층구조를 정의한다. 할당기는 매 사이클 어떤
 * 입력 포트가 어떤 출력 포트를 사용할 수 있는지를 결정하는 알고리즘의 공통 인터페이스다.
 * GPGPU-Sim에서는 SM(Streaming Multiprocessor) 간 혹은 SM과 L2 캐시 사이를 연결하는
 * NoC 스위치가 이 할당기를 통해 매 사이클 트래픽을 처리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델 계층에서 NoC(intersim2) 서브시스템의 라우터 내부에 위치한다.
 * 호출 체인: IQRouter::_SWAllocEvaluate() / _VCAllocEvaluate()
 *            → Allocator::AddRequest() (요청 등록)
 *            → Allocator::Allocate()   (매칭 계산)
 *            → Allocator::OutputAssigned() / InputAssigned() (결과 조회)
 * 실행 컨텍스트: GPGPU-Sim 타이밍 사이클 루프 내 단일 스레드로 호출됨.
 * 각 사이클마다 Clear() → AddRequest() × N → Allocate() → 결과 판독 순서로 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - IQRouter (routers/iq_router.{hpp,cpp}): VC 할당기(_vc_allocator)와 스위치 할당기
 *   (_sw_allocator, _spec_sw_allocator)로 이 클래스의 인스턴스를 보유한다.
 * - Arbiter (arbiters/arbiter.hpp): SeparableAllocator의 내부 중재기로 사용된다.
 * - Module (module.hpp): 부모 클래스 — BookSim 모듈 계층구조에 통합된다.
 * - Configuration (config_utils.hpp): 설정 파일에서 할당기 타입("islip", "pim" 등)을 읽는다.
 * 데이터 흐름: AddRequest()로 요청이 등록 → Allocate()로 매칭 결과가 _inmatch/_outmatch에
 *             기록 → OutputAssigned()/InputAssigned()로 IQRouter가 결과를 읽는다.
 *
 * === 주요 함수/구조체 요약 ===
 * - Allocator: 순수 가상 기반 클래스. 입력N × 출력M 매칭 알고리즘의 공통 인터페이스.
 * - DenseAllocator: 전체 요청 행렬을 2D 벡터로 저장하는 구현. 소규모 NoC에 적합.
 * - SparseAllocator: 실제 요청만 map으로 저장하는 구현. 희소 트래픽에서 효율적.
 * - sRequest: 단일 입출력 요청의 메타데이터 (포트, 레이블, 입력/출력 우선순위).
 * - NewAllocator(): 설정 문자열("islip", "pim", "wavefront" 등)로 구체 할당기를 생성하는 팩토리.
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

#ifndef _ALLOCATOR_HPP_
#define _ALLOCATOR_HPP_

#include <string>
#include <map>
#include <set>
#include <vector>

#include "module.hpp"     // [한국어] BookSim 모듈 계층의 기반 클래스 — 이름, 부모-자식 관계 관리
#include "config_utils.hpp" // [한국어] gpgpusim.config/booksim.cfg 파싱 Configuration 클래스

/*
 * [한국어] Allocator - 크로스바 입출력 매칭 알고리즘의 순수 가상 기반 클래스
 *
 * GPU NoC 라우터(IQRouter)의 VC 할당(VC Allocation)과 스위치 할당(Switch Allocation)에
 * 사용되는 알고리즘의 공통 인터페이스를 정의한다. 구체 알고리즘(iSLIP, PIM, Wavefront 등)은
 * 모두 이 클래스를 상속한다. 매 사이클 Clear() → AddRequest() × N → Allocate() 순서로 사용된다.
 */
class Allocator : public Module {
protected:
  const int _inputs;
  /* _inputs: 이 할당기가 처리하는 입력 포트 수 (= 라우터 입력 × VC 수).
   * 설정자: 생성자에서 1회 설정 후 const로 고정.
   * 읽는 자: Allocate() 구현에서 루프 경계로 사용.
   * 값 범위: 1 이상 양의 정수. GPU NoC에서는 입력포트수 × VC수.
   * 동기화: const이므로 락 불필요. */

  const int _outputs;
  /* _outputs: 이 할당기가 처리하는 출력 포트 수.
   * 설정자: 생성자에서 1회 설정 후 const로 고정.
   * 읽는 자: Allocate() 구현, OutputAssigned(), InputAssigned()에서 사용.
   * 값 범위: 1 이상 양의 정수. GPU NoC에서는 출력포트수 × VC수.
   * 동기화: const이므로 락 불필요. */

  bool _dirty;
  /* _dirty: 이번 사이클에 AddRequest()가 한 번이라도 호출되었는지 여부.
   * 설정자: AddRequest() 호출 시 true로 설정. Clear()에서 false로 리셋.
   * 읽는 자: Clear()에서 실제 초기화가 필요한지 확인하는 데 사용(최적화).
   * 값 범위: true(요청 있음) / false(요청 없음, 초기화 불필요).
   * 동기화: 단일 스레드 사이클 루프 내에서만 사용. 별도 락 불필요. */

  vector<int> _inmatch;
  /* _inmatch: Allocate() 결과 — 입력 i가 할당된 출력 포트 번호.
   * _inmatch[i] = j 이면 입력 i → 출력 j로 매칭됨.
   * _inmatch[i] = -1 이면 입력 i는 이번 사이클에 미할당.
   * 설정자: Allocate() 구현에서 매칭 결과 기록. Clear()에서 -1로 초기화.
   * 읽는 자: OutputAssigned(int in) 및 IQRouter에서 스위치 트래버설 결정에 사용.
   * 값 범위: -1(미할당) 또는 0.._outputs-1.
   * 동기화: 단일 스레드. Clear() → AddRequest × N → Allocate() 순서 보장. */

  vector<int> _outmatch;
  /* _outmatch: Allocate() 결과 — 출력 j를 담당하게 된 입력 포트 번호.
   * _outmatch[j] = i 이면 출력 j ← 입력 i로 매칭됨.
   * _outmatch[j] = -1 이면 출력 j는 이번 사이클에 유휴 상태.
   * 설정자: Allocate() 구현에서 매칭 결과 기록. Clear()에서 -1로 초기화.
   * 읽는 자: InputAssigned(int out) 및 Allocate() 내 이미 매칭된 출력 스킵 로직에 사용.
   * 값 범위: -1(미할당) 또는 0.._inputs-1.
   * 동기화: 단일 스레드. _inmatch와 쌍으로 항상 동시에 설정됨. */

public:

  struct sRequest {
    /* [한국어] 단일 입출력 요청의 메타데이터 구조체.
     * AddRequest() 호출 시 이 구조체로 요청이 기록된다. */
    int port;
    /* 상대 포트 번호 — in_req[i]에서는 출력 포트, out_req[j]에서는 입력 포트.
     * 설정자: AddRequest()에서 기록.
     * 읽는 자: Allocate() 내에서 매칭 탐색 시 상대 포트 식별에 사용. */
    int label;
    /* 요청 레이블 (VC ID 또는 식별자). -1이면 요청 없음.
     * DenseAllocator에서 요청 유무 판단 기준값으로 사용된다(label == -1이면 빈 슬롯).
     * 설정자: AddRequest()에서 기록. RemoveRequest()에서 -1로 초기화.
     * 읽는 자: ReadRequest()에서 요청 유무 판단. */
    int in_pri;
    /* 입력 측 우선순위 — 수락(Accept) 단계에서 동점 시 높은 값이 우선.
     * 설정자: AddRequest()에서 호출자가 지정.
     * 읽는 자: SelAlloc, iSLIP 등의 Accept 단계 우선순위 비교에서 사용. */
    int out_pri;
    /* 출력 측 우선순위 — 승인(Grant) 단계에서 동점 시 높은 값이 우선.
     * 설정자: AddRequest()에서 호출자가 지정.
     * 읽는 자: SelAlloc, Wavefront 등의 Grant 단계 우선순위 비교에서 사용. */
  };

  /*
   * [한국어] Allocator 생성자
   * @parent: BookSim 모듈 트리에서의 부모 모듈 (라우터)
   * @name: 디버그/출력용 이름 문자열
   * @inputs: 이 할당기가 처리할 입력 포트 수
   * @outputs: 이 할당기가 처리할 출력 포트 수
   * _inmatch와 _outmatch를 -1로 초기화하고 _dirty를 false로 설정한다.
   * 호출 체인: IQRouter 생성자 → Allocator::NewAllocator() → [구체 클래스 생성자] → Allocator()
   */
  Allocator( Module *parent, const string& name,
	     int inputs, int outputs );

  /*
   * [한국어] Clear - 할당 결과 및 요청을 초기화하여 다음 사이클에 재사용 준비
   * @return: 없음
   * _dirty 플래그가 true일 때만 _inmatch와 _outmatch를 -1로 초기화하여 성능 최적화.
   * 매 사이클 시작 시 IQRouter가 호출한다.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → Allocator::Clear()
   */
  virtual void Clear( );

  /*
   * [한국어] ReadRequest - 입력 in이 출력 out에 대해 등록한 요청의 레이블을 반환
   * @in: 입력 포트 번호
   * @out: 출력 포트 번호
   * @return: 요청이 있으면 레이블(>=0), 없으면 -1
   * 순수 가상 함수 — DenseAllocator와 SparseAllocator에서 각각 구현됨.
   */
  virtual int  ReadRequest( int in, int out ) const = 0;

  /*
   * [한국어] ReadRequest (구조체 버전) - 요청의 전체 메타데이터를 sRequest 구조체로 반환
   * @req: 반환할 sRequest 구조체 (out 파라미터)
   * @in: 입력 포트 번호
   * @out: 출력 포트 번호
   * @return: 요청이 존재하면 true, 없으면 false
   */
  virtual bool ReadRequest( sRequest &req, int in, int out ) const = 0;

  /*
   * [한국어] AddRequest - 입력 in → 출력 out 방향의 요청을 등록
   * @in: 요청하는 입력 포트 번호 (0.._inputs-1)
   * @out: 요청하는 출력 포트 번호 (0.._outputs-1)
   * @label: 요청 레이블 / VC ID (기본값 1)
   * @in_pri: 수락 단계 우선순위 (기본값 0)
   * @out_pri: 승인 단계 우선순위 (기본값 0)
   * _dirty를 true로 설정하고 유효성 검사를 수행한다. 서브클래스에서 실제 저장을 담당.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → Allocator::AddRequest()
   */
  virtual void AddRequest( int in, int out, int label = 1,
			   int in_pri = 0, int out_pri = 0 );

  /*
   * [한국어] RemoveRequest - 등록된 요청을 취소
   * @in: 취소할 입력 포트
   * @out: 취소할 출력 포트
   * @label: 취소할 요청의 레이블
   * 순수 가상 함수 — 서브클래스에서 저장 구조에 따라 구현.
   */
  virtual void RemoveRequest( int in, int out, int label = 1 ) = 0;

  /*
   * [한국어] Allocate - 등록된 모든 요청에 대해 입출력 매칭 알고리즘을 실행
   * 순수 가상 함수 — iSLIP, PIM, Wavefront 등 각 알고리즘이 구현한다.
   * 결과는 _inmatch[in] = out, _outmatch[out] = in 형태로 기록된다.
   * 호출 체인: IQRouter::_SWAllocEvaluate() → Allocator::Allocate()
   */
  virtual void Allocate( ) = 0;

  /*
   * [한국어] OutputAssigned - 입력 in이 할당받은 출력 포트 번호를 반환
   * @in: 입력 포트 번호
   * @return: 할당된 출력 포트 번호. 미할당이면 -1.
   * Allocate() 후 IQRouter가 스위치 트래버설 여부를 판단할 때 호출한다.
   */
  int OutputAssigned( int in ) const;

  /*
   * [한국어] InputAssigned - 출력 out을 차지한 입력 포트 번호를 반환
   * @out: 출력 포트 번호
   * @return: 해당 출력을 담당하는 입력 포트 번호. 미사용이면 -1.
   */
  int InputAssigned( int out ) const;

  virtual bool OutputHasRequests( int out ) const = 0; // [한국어] 출력 out에 대한 요청이 존재하면 true
  virtual bool InputHasRequests( int in ) const = 0;   // [한국어] 입력 in에서 보낸 요청이 존재하면 true

  virtual int NumOutputRequests( int out ) const = 0;  // [한국어] 출력 out에 대한 요청 수 반환
  virtual int NumInputRequests( int in ) const = 0;    // [한국어] 입력 in이 보낸 요청 수 반환

  virtual void PrintRequests( ostream * os = NULL ) const = 0; // [한국어] 등록된 요청 상태를 os에 출력 (디버그용)

  /*
   * [한국어] PrintGrants - 현재 매칭 결과(_inmatch/_outmatch)를 os에 출력
   * @os: 출력 스트림 (NULL이면 cout 사용)
   * 디버그 목적으로 입력→출력, 출력→입력 할당 결과를 한 줄에 표시한다.
   */
  void PrintGrants( ostream * os = NULL ) const;

  /*
   * [한국어] NewAllocator - 설정 문자열에 따라 구체 Allocator 인스턴스를 생성하는 팩토리
   * @parent: 부모 모듈
   * @name: 모듈 이름
   * @alloc_type: 할당기 타입 문자열 (예: "islip(4)", "pim", "wavefront", "separable_input_first")
   * @inputs: 입력 포트 수
   * @outputs: 출력 포트 수
   * @config: gpgpusim.config에서 읽은 설정 (alloc_iters, arb_type 등을 가져올 때 사용)
   * @return: 생성된 Allocator 인스턴스 포인터. 알 수 없는 타입이면 NULL.
   * IQRouter 생성 시 한 번 호출되며, 이후 매 사이클마다 재사용된다.
   * 호출 체인: IQRouter 생성자 → Allocator::NewAllocator()
   */
  static Allocator *NewAllocator( Module *parent, const string& name,
				  const string &alloc_type,
				  int inputs, int outputs,
				  Configuration const * const config = NULL );
};

/*
 * [한국어] DenseAllocator — 전체 요청 행렬을 2D 벡터로 저장하는 할당기 기반 구현
 *
 * _inputs × _outputs 크기의 2D sRequest 배열(_request)을 항상 메모리에 유지한다.
 * 요청이 없는 슬롯은 label == -1로 표시된다.
 * MaxSizeMatch, PIM, Wavefront, LOA 등 밀집 요청 행렬을 순회해야 하는 알고리즘이 상속한다.
 * 메모리 사용량 O(inputs × outputs). GPU NoC에서 작은 크기에서는 캐시 친화적이다.
 */
class DenseAllocator : public Allocator {
protected:
  vector<vector<sRequest> > _request;
  /* _request[i][j]: 입력 i → 출력 j에 대한 요청 메타데이터.
   * _request[i][j].label == -1이면 요청 없음. >= 0이면 유효한 요청.
   * 설정자: AddRequest(i, j, ...)에서 기록. Clear()/RemoveRequest()에서 -1로 초기화.
   * 읽는 자: Allocate() 구현(LOA, PIM, Wavefront, MaxSizeMatch)에서 요청 행렬 순회에 사용.
   * 값 범위: sRequest{port, label, in_pri, out_pri} — label은 -1 또는 >=0.
   * 동기화: 단일 스레드 사이클 루프 내에서 사용. 락 불필요. */

public:
  /*
   * [한국어] DenseAllocator 생성자 — _request 2D 벡터를 inputs × outputs로 초기화
   * 모든 슬롯의 label을 -1로 초기화하여 요청 없음 상태로 시작.
   * 호출 체인: MaxSizeMatch/PIM/Wavefront/LOA 생성자 → DenseAllocator() → Allocator()
   */
  DenseAllocator( Module *parent, const string& name,
		  int inputs, int outputs );

  void Clear( );  // [한국어] 모든 _request[i][j].label을 -1로 초기화 후 상위 Clear() 호출

  int  ReadRequest( int in, int out ) const;           // [한국어] _request[in][out].label 반환 (-1이면 요청 없음)
  bool ReadRequest( sRequest &req, int in, int out ) const; // [한국어] _request[in][out]을 req에 복사, 유효 여부 반환

  void AddRequest( int in, int out, int label = 1,
		   int in_pri = 0, int out_pri = 0 );   // [한국어] _request[in][out]에 요청 메타데이터 기록
  void RemoveRequest( int in, int out, int label = 1 ); // [한국어] _request[in][out].label을 -1로 초기화

  bool OutputHasRequests( int out ) const; // [한국어] 출력 out을 요청하는 입력이 하나라도 있으면 true
  bool InputHasRequests( int in ) const;   // [한국어] 입력 in이 요청하는 출력이 하나라도 있으면 true

  int NumOutputRequests( int out ) const;  // [한국어] 출력 out에 대한 요청 수 (label >= 0인 입력 수)
  int NumInputRequests( int in ) const;    // [한국어] 입력 in이 보낸 요청 수

  void PrintRequests( ostream * os = NULL ) const; // [한국어] 요청 행렬을 텍스트로 출력 (디버그용)

};

/*
 * [한국어] SparseAllocator — 실제 요청만 map으로 저장하는 희소 할당기 기반 구현
 *
 * 전체 입출력 조합 행렬 대신 실제로 요청이 있는 (입력, 출력) 쌍만 map으로 저장한다.
 * _in_req[i][j]와 _out_req[j][i]에 양방향으로 요청을 기록하여 입력/출력 방향 순회 모두 지원.
 * iSLIP, SelAlloc, SeparableAllocator 등이 이 클래스를 상속한다.
 * 트래픽이 희소할수록 DenseAllocator보다 효율적이다.
 */
class SparseAllocator : public Allocator {
protected:
  set<int> _in_occ;
  /* 이번 사이클에 요청을 보낸 입력 포트의 집합.
   * 설정자: AddRequest()에서 입력 요청이 처음 등록될 때 삽입. RemoveRequest()/Clear()에서 제거.
   * 읽는 자: iSLIP, SelAlloc의 Accept 단계 루프에서 활성 입력만 순회할 때 사용.
   * 값 범위: 0.._inputs-1 범위의 정수들의 부분집합.
   * 동기화: 단일 스레드 사이클 루프. 락 불필요. */

  set<int> _out_occ;
  /* 이번 사이클에 요청을 받은 출력 포트의 집합.
   * 설정자: AddRequest()에서 출력 요청이 처음 등록될 때 삽입. RemoveRequest()/Clear()에서 제거.
   * 읽는 자: iSLIP, SelAlloc의 Grant 단계 루프에서 활성 출력만 순회할 때 사용.
   * 값 범위: 0.._outputs-1 범위의 정수들의 부분집합.
   * 동기화: 단일 스레드. 락 불필요. */

  vector<map<int, sRequest> > _in_req;
  /* _in_req[i]: 입력 i가 보낸 요청의 map. key = 출력 포트 번호, value = sRequest.
   * 설정자: AddRequest(i, j, ...)에서 _in_req[i][j]에 요청 추가.
   *         RemoveRequest()에서 해당 항목 제거. Clear()에서 map 전체 초기화.
   * 읽는 자: iSLIP Accept 단계에서 _in_req[input] 맵을 순회하며 승인된 출력 탐색.
   * 값 범위: 맵 내 key는 0.._outputs-1, value의 label >= 0.
   * 동기화: 단일 스레드. */

  vector<map<int, sRequest> > _out_req;
  /* _out_req[j]: 출력 j를 요청하는 입력들의 map. key = 입력 포트 번호, value = sRequest.
   * 설정자: AddRequest(i, j, ...)에서 _out_req[j][i]에 요청 추가.
   * 읽는 자: iSLIP Grant 단계에서 _out_req[output] 맵을 순회하며 할당할 입력 탐색.
   * 값 범위: 맵 내 key는 0.._inputs-1.
   * 동기화: 단일 스레드. */

public:
  /*
   * [한국어] SparseAllocator 생성자 — _in_req와 _out_req를 inputs/outputs 크기로 초기화
   * 각 맵은 빈 상태(요청 없음)로 시작한다. _in_occ와 _out_occ도 빈 집합.
   */
  SparseAllocator( Module *parent, const string& name,
		   int inputs, int outputs );

  void Clear( );  // [한국어] 모든 맵과 occupied 집합을 초기화 후 상위 Clear() 호출

  int  ReadRequest( int in, int out ) const;           // [한국어] _in_req[in][out].label 반환. 요청 없으면 -1.
  bool ReadRequest( sRequest &req, int in, int out ) const; // [한국어] _in_req[in].find(out) 탐색, 결과 req에 기록

  void AddRequest( int in, int out, int label = 1,
		   int in_pri = 0, int out_pri = 0 );   // [한국어] _in_req[in][out]과 _out_req[out][in] 양방향 등록
  void RemoveRequest( int in, int out, int label = 1 ); // [한국어] _in_req[in][out]과 _out_req[out][in] 양방향 제거

  bool OutputHasRequests( int out ) const; // [한국어] _out_occ.count(out) > 0 이면 true
  bool InputHasRequests( int in ) const;   // [한국어] _in_occ.count(in) > 0 이면 true

  int NumOutputRequests( int out ) const;  // [한국어] _out_occ.count(out) 반환 (0 또는 1 — 존재 여부)
  int NumInputRequests( int in ) const;    // [한국어] _in_occ.count(in) 반환

  void PrintRequests( ostream * os = NULL ) const; // [한국어] 요청 맵을 텍스트로 출력 (디버그용)

};

#endif
