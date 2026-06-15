// $Id: tree4.hpp 5188 2012-08-30 00:31:31Z dub $

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

////////////////////////////////////////////////////////////////////////
//
// Tree4: Network with 64 Terminal Nodes arranged in a tree topology
//        with 4 routers at the root of the tree
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: tree4.hpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

/*
 * [한국어 설명] Tree4 4진 트리 네트워크 헤더 (tree4.hpp)
 *
 * === 파일의 역할 ===
 * Tree4는 k=4, n=3 고정 파라미터의 4진 직접 트리 NoC 토폴로지를 구현하는
 * Network 파생 클래스의 헤더 선언 파일이다. QTree(간접 트리)와 달리 Tree4는
 * 루트 레벨에서도 4개의 라우터를 두는 방식으로, 레벨별 라우터 수가 QTree와 다르다:
 *   Level 0(루트): 4개 8×8 라우터 (= (4>>0)*k^0 = 4개)
 *   Level 1(중간): 8개 8×8 라우터 (= (4>>1)*k^1 = 2*4 = 8개)
 *   Level 2(리프): 16개 6×6 라우터 (= (4>>2)*k^2 = 1*16 = 16개)
 * 총 28개 라우터와 64개 터미널 노드로 구성된다.
 * 이 헤더는 클래스 선언, 멤버 변수 타입, 핵심 함수 프로토타입을 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 시뮬레이션 계층(intersim2) 내 Tree 계열 토폴로지 중 하나이다.
 * icnt_wrapper.cc가 Network::New("tree4", ...)를 호출하면 Tree4 인스턴스가 생성된다.
 * 초기화 순서:
 *   1) _ComputeSize(): k=4/n=3 assert, _nodes=64, _size=28, _channels 계산
 *   2) _Alloc(): 라우터/채널 배열 동적 할당
 *   3) _BuildNet(): 3레벨 라우터 생성, inject/eject 채널(리프), 레벨 간 채널 연결
 * 포트 규칙: 출력 포트 0~3=DOWN(하위 방향), 4~7=UP(상위 방향).
 * 실행 컨텍스트: 초기화는 메인 스레드; 이후 시뮬레이션 루프에서 매 사이클 step() 호출.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - network.hpp: Network 기반 클래스 (라우터/채널 배열, _Alloc, step 등)
 *   - booksim.hpp: 전역 파라미터 (gK, gN)
 *   - misc_utils.hpp: powi() — 정수 거듭제곱
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: Network::New()를 통해 Tree4 생성
 *   - 라우팅 함수들: HeightFromID(), PosFromID()로 라우터 트리 위치 파악
 * 공유 자료구조:
 *   - gK=4, gN=3 전역 변수 (_ComputeSize()에서 설정)
 *
 * === 주요 함수/구조체 요약 ===
 * Tree4()              : 생성자 — _ComputeSize → _Alloc → _BuildNet 표준 3단계 초기화
 * _ComputeSize()       : k=4/n=3 assert, _nodes=64, _size=28, _channels 계산
 * _BuildNet()          : 레벨별 라우터 생성, inject/eject(리프), h=1↔h=2 및 h=0↔h=1 채널 연결
 * _Router(h, pos)      : (height, pos) → _routers[] 참조 반환 (Tree4 특유의 접두 합 계산)
 * _WireLatency(h1,p1,h2,p2): 물리 위치 기반 레이턴시 계산 (레거시, 현재 SetLatency(1) 사용)
 * HeightFromID(id)     : id / k^(n-1) 으로 height 추출
 * PosFromID(id)        : id % k^(n-1) 으로 pos 추출
 * SpeedUp(height)      : 상위 레벨 링크 속도 배율 반환 (현재 미구현)
 */

#ifndef _TREE4_HPP_
#define _TREE4_HPP_
#include <cassert>     // [한국어] assert() — k=4, n=3 강제 및 라우터 위치 범위 검증
#include "network.hpp" // [한국어] Network 기반 클래스 — 라우터/채널 배열, _Alloc(), step() 등

/*
 * [한국어] Tree4 - 4진 트리 NoC 토폴로지 클래스 (64 터미널 노드, 3레벨)
 *
 * k=4, n=3 고정. QTree와의 차이: 루트 레벨에 4개 라우터(QTree는 1개).
 * Tree4는 (4>>h)*k^h 공식으로 각 레벨의 라우터 수를 계산한다:
 *   h=0: (4>>0)*4^0 = 4  (루트 4개 라우터, 각 8포트)
 *   h=1: (4>>1)*4^1 = 8  (중간 8개 라우터, 각 8포트)
 *   h=2: (4>>2)*4^2 = 16 (리프 16개 라우터, 각 6포트 = inject/eject 4 + 상위 2)
 * 총 _size = 4+8+16 = 28 라우터.
 *
 * 포트 방향 규칙:
 *   출력 포트 0~3: DOWN 방향 (상위→하위 또는 inject/eject)
 *   출력 포트 4~7: UP 방향 (하위→상위)
 *
 * 라우터 ID 인코딩: id = h * k^(n-1) + pos = h*16 + pos
 * (HeightFromID: id/16, PosFromID: id%16)
 */
class Tree4 : public Network {

  int _k;
  /* [한국어] 트리 방사(radix) = 4로 고정.
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 설정, assert(_k==4) 검증.
   * 읽는 자: _ComputeSize()에서 _nodes/_size/_channels 계산 및 레벨별 라우터 수 공식 사용.
   *           _BuildNet()에서 inject/eject 채널 수, 레벨 간 채널 연결 포트 수 결정.
   *           _Router(), _WireLatency(), HeightFromID(), PosFromID()에서 인덱스 계산.
   * 값 범위: 항상 4 (다른 값이면 assert 실패).
   * 동기화: 초기화 단계에서만 쓰이며 이후 읽기 전용. */

  int _n;
  /* [한국어] 트리 레벨(depth) = 3으로 고정.
   * 설정자: _ComputeSize()에서 config.GetInt("n")로 설정, assert(_n==3) 검증.
   * 읽는 자: _ComputeSize()에서 _nodes=k^n, _size=sum(...), _channels 계산.
   *           _BuildNet()에서 레벨 루프 상한, 리프 레벨(h=_n-1) 결정.
   *           _Router()에서 범위 검증 (assert height < _n).
   *           HeightFromID(), PosFromID()에서 k^(n-1) 계산.
   * 값 범위: 항상 3 (다른 값이면 assert 실패).
   * 동기화: 초기화 단계에서만 쓰이며 이후 읽기 전용. */

  int _channelWidth;
  /* [한국어] 채널 폭 (플릿 당 비트 수) — 현재 사용되지 않는 레거시 필드.
   * 설정자: 명시적 초기화 없음 (쓰레기 값 가능성).
   * 읽는 자: _WireLatency()나 _BuildNet()에서 참조하지 않음.
   * 값 범위: 정의 없음 (사용 안 함).
   * 동기화: 불필요 (미사용). */

  /*
   * [한국어] _ComputeSize - Tree4 크기 파라미터 계산
   *
   * @config: BooksimConfig — "k"=4, "n"=3 읽음
   * @return: 없음 (_k, _n, _nodes, _size, _channels 설정)
   *
   * k=4, n=3을 assert로 강제하고 이하를 계산한다:
   *   _nodes    = k^n = 64 터미널 노드
   *   _size     = sum((4>>h)*k^h, h=0..n-1) = 4+8+16 = 28 라우터
   *   _channels = 2 * (2*k^1) * (2*k) = 2 * 8 * 8 = 128 채널
   *               (공식이 Tree4 레벨 간 연결 구조에서 유도됨)
   * gK=_k, gN=_n으로 전역 파라미터 업데이트.
   *
   * 호출 체인:
   *   Tree4() → [_ComputeSize()]
   */
  void _ComputeSize( const Configuration& config );

  /*
   * [한국어] _BuildNet - Tree4 라우터 생성 및 채널 연결
   *
   * @config: BooksimConfig — Router::NewRouter()에 전달
   * @return: 없음
   *
   * 세 단계로 구성된다:
   * 1단계 — 라우터 생성:
   *   각 레벨 h=0..n-1, 각 위치 pos=0..nPos-1 (nPos=(4>>h)*k^h)에 라우터 생성.
   *   h<n-1이면 degree=8(8포트), h=n-1이면 degree=6(6포트).
   *   라우터 ID = h*k^(n-1) + pos = h*16+pos.
   *   _Router(h,pos)로 _routers[] 위치에 저장.
   * 2단계 — inject/eject 채널 (리프):
   *   리프(h=n-1=2)의 각 16개 라우터에 k=4개의 inject/eject 채널 연결.
   *   모두 SetLatency(1)로 1사이클 레이턴시 설정.
   * 3단계 — 레벨 간 내부 채널:
   *   h=1↔h=2: 채널 c=0부터 시작, pp=pos(중간), pc=k*(pos/2)+port(리프),
   *             각 연결에 DOWN(AddOutput→AddInput)과 UP(AddInput←AddOutput) 각 1채널.
   *   h=0↔h=1: pp=pos(루트), pc=port(중간), 각 연결에 DOWN+UP 2채널.
   *   모두 SetLatency(1) 설정.
   *
   * 호출 체인:
   *   Tree4() → _ComputeSize() → _Alloc() → [_BuildNet()]
   */
  void _BuildNet( const Configuration& config );


  /*
   * [한국어] _Router - (height, pos) 좌표로 _routers[] 참조 반환
   *
   * @height: 트리 레벨 (0=루트, 1=중간, 2=리프); assert(height < _n) 검증
   * @pos: 레벨 내 위치; assert(pos < (4>>height)*k^height) 검증
   * @return: Router*& — _routers[] 배열 원소 참조 (대입 가능한 참조형)
   *
   * Tree4 특유의 라우터 수 공식 (4>>h)*k^h로 레벨별 라우터 수를 계산하고
   * prefix sum으로 _routers[] 선형 인덱스를 결정한다:
   *   h=0: 시작 인덱스 0, 라우터 4개 → [0..3]
   *   h=1: 시작 인덱스 4, 라우터 8개 → [4..11]
   *   h=2: 시작 인덱스 12, 라우터 16개 → [12..27]
   * 반환이 참조형(Router*&)이므로 _Router(h,pos) = r; 형태로 대입 가능하다.
   *
   * 호출 체인:
   *   _BuildNet() → [_Router(h, pos)] (라우터 생성 시 대입 및 채널 연결 시 조회)
   */
  Router*& _Router( int height, int pos );

  /*
   * [한국어] _WireLatency - 두 라우터 간 물리적 거리 기반 레이턴시 계산 (레거시)
   *
   * @height1, pos1: 첫 번째 라우터의 트리 좌표
   * @height2, pos2: 두 번째 라우터의 트리 좌표
   * @return: 두 라우터를 잇는 채널의 레이턴시 (사이클)
   *
   * 두 라우터 중 높이가 낮은 쪽을 Parent, 높은 쪽을 Child로 분류한다.
   * heightChild - heightParent == 1인 경우만 지원 (assert 검증).
   * 레이턴시 값(_length_d2_d1=2, _length_d1_d0_0~3=2/2/6/6)이 하드코딩되어 있으며
   * 물리적 칩 레이아웃 기반 추정치이다.
   * 현재 _BuildNet()에서는 모든 채널을 SetLatency(1)로 고정하므로
   * 이 함수는 레거시 코드로 실제 시뮬레이션에 사용되지 않는다.
   * (관련 SetLatency(_WireLatency(...)) 호출은 _BuildNet()에서 주석 처리됨)
   *
   * 호출 체인:
   *   (현재 호출되지 않음 — 레거시)
   */
  int _WireLatency( int height1, int pos1, int height2, int pos2 );

public:

  /*
   * [한국어] Tree4 생성자 — 4진 트리 네트워크 초기화
   *
   * @config: BooksimConfig — "k"=4, "n"=3, 라우터 내부 설정 포함
   * @name: 네트워크 객체 이름 문자열 (디버그/로그 식별자)
   * @return: 없음 (생성자)
   *
   * Network 기반 클래스 생성자 호출 후 _ComputeSize → _Alloc → _BuildNet 순서로
   * Tree4 네트워크를 완전히 구성한다.
   *
   * 호출 체인:
   *   Network::New() → [Tree4()] → _ComputeSize() → _Alloc() → _BuildNet()
   */
  Tree4( const Configuration& config, const string & name );

  /*
   * [한국어] RegisterRoutingFunctions - Tree4 전용 라우팅 함수 등록 (현재 미구현)
   *
   * @return: 없음
   *
   * tree4.cpp에서 빈 함수로 구현됨 — Tree4 전용 라우팅 함수가 아직 없음.
   * 실제 시뮬레이션에서는 전역 기본 라우팅 함수나 설정 파일로 지정된 함수 사용.
   *
   * 호출 체인:
   *   Network::RegisterRoutingFunctions() → [Tree4::RegisterRoutingFunctions()]
   */
  static void RegisterRoutingFunctions() ;

  /*
   * [한국어] HeightFromID - 라우터 ID에서 트리 레벨(height) 추출
   *
   * @id: 라우터 ID (_BuildNet()에서 h*k^(n-1)+pos = h*16+pos로 인코딩)
   * @return: 트리 레벨 h (0=루트, 1=중간, 2=리프)
   *
   * 반환값 = id / k^(n-1) = id / 16
   * 정적 함수이므로 인스턴스 없이 라우터 ID만으로 트리 위치 파악 가능.
   *
   * 호출 체인:
   *   라우팅 함수 또는 디버그 코드 → [Tree4::HeightFromID(id)]
   */
  static int HeightFromID( int id );

  /*
   * [한국어] PosFromID - 라우터 ID에서 레벨 내 위치(pos) 추출
   *
   * @id: 라우터 ID (h*k^(n-1)+pos 형식으로 인코딩)
   * @return: 레벨 내 위치 pos (0부터 (4>>h)*k^h - 1까지)
   *
   * 반환값 = id % k^(n-1) = id % 16
   * HeightFromID()와 함께 사용하여 (height, pos) 좌표를 완전히 복원한다.
   *
   * 호출 체인:
   *   라우팅 함수 또는 디버그 코드 → [Tree4::PosFromID(id)]
   */
  static int PosFromID( int id );

  /*
   * [한국어] SpeedUp - 트리 레벨별 링크 속도 배율 반환 (현재 미구현)
   *
   * @height: 링크가 위치한 트리 레벨
   * @return: 속도 배율 (현재 미구현, 항상 1 반환 예정)
   *
   * 상위 레벨(루트에 가까운) 링크에 더 높은 속도 배율을 부여하는 개념적 인터페이스.
   * 현재 tree4.cpp에는 이 함수의 구현 코드가 없으며 (선언만 존재하고 정의가 누락됨),
   * _BuildNet()에서도 호출되지 않는다. 향후 이기종 속도 링크 지원을 위한 플레이스홀더.
   *
   * 호출 체인:
   *   (현재 호출되지 않음 — 미구현)
   */
  static int SpeedUp( int height );
};

#endif
