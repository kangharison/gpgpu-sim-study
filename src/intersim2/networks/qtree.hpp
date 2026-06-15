// $Id: qtree.hpp 5188 2012-08-30 00:31:31Z dub $

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
// QTree: A Quad-Tree Indirect Network.
//
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/05/17 17:14:07 $
//  $Id: qtree.hpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

/*
 * [한국어 설명] QTree 4진 간접 트리 네트워크 헤더 (qtree.hpp)
 *
 * === 파일의 역할 ===
 * QTree(Quad-Tree)는 k=4, n=3 고정 파라미터의 4진 간접 트리 토폴로지 NoC를 구현하는
 * Network 파생 클래스의 헤더 선언 파일이다. 간접 트리(Indirect Tree)란 터미널 노드가
 * 오직 리프(leaf) 레이어에만 연결되고, 내부 노드는 모두 순수 라우터로만 구성되는
 * 멀티레벨 스위치 패브릭이다. k=4, n=3이므로 루트부터 리프까지 3레벨이 존재하며
 * 64(=4^3)개의 터미널 노드와 21(=1+4+16)개의 라우터로 구성된다.
 * 이 헤더는 클래스 선언, 멤버 변수 타입, 핵심 인덱싱 함수 프로토타입을 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 시뮬레이션 계층(intersim2)에서 Tree 계열 토폴로지 중 하나이다.
 * icnt_wrapper.cc가 Network::New()를 통해 QTree를 생성하면 초기화 순서가 실행된다:
 *   1) _ComputeSize(): _k=4, _n=3 검증, _nodes=64, _size=21, _channels 계산
 *   2) _Alloc(): Network 기반 클래스가 _routers[], _chan[], _inject[], _eject[] 할당
 *   3) _BuildNet(): 3레벨 트리 라우터 생성 및 채널 연결
 * 시뮬레이션 루프에서 step() 호출 시 등록된 라우팅 함수(qtree.cpp에서 등록)가 사용된다.
 * 실행 컨텍스트: 초기화는 단일 메인 스레드; 이후 시뮬레이션 스레드에서 매 사이클 호출.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - network.hpp: Network 기반 클래스 — 라우터/채널 배열, _Alloc(), step() 등 제공
 *   - booksim.hpp: 전역 시뮬레이션 파라미터 (gK, gN, powi 등)
 *   - misc_utils.hpp: powi() — 정수 거듭제곱 계산 유틸리티
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: Network::New()를 통해 QTree 인스턴스 생성
 *   - 라우팅 함수들: HeightFromID(), PosFromID()로 라우터의 트리 위치 파악
 * 공유 자료구조:
 *   - gK, gN (전역): _ComputeSize()에서 _k=4, _n=3으로 설정하여 라우팅 함수와 공유
 *
 * === 주요 함수/구조체 요약 ===
 * QTree()            : 생성자 — _ComputeSize → _Alloc → _BuildNet 순서로 트리 초기화
 * _ComputeSize()     : k=4, n=3 assert 검증, _nodes=64, _size=21, _channels 계산
 * _BuildNet()        : 3레벨 라우터 생성, inject/eject 채널(리프), 부모-자식 내부 채널 연결
 * _RouterIndex(h,p)  : (height, pos) → _routers[] 배열 인덱스 변환 (prefix sum)
 * _InputIndex(h,p,p) : 자식→부모 방향(UP) 채널 인덱스 계산
 * _OutputIndex(h,p,p): 부모→자식 방향(DOWN) 채널 인덱스 계산 (_channels/2 오프셋)
 * HeightFromID(id)   : 라우터 ID(h*256+pos)에서 height 추출 (id / 256)
 * PosFromID(id)      : 라우터 ID(h*256+pos)에서 pos 추출 (id % 256)
 */

#ifndef _QTREE_HPP_
#define _QTREE_HPP_
#include <cassert>    // [한국어] assert() — k=4, n=3 강제 및 채널 인덱스 범위 검증
#include "network.hpp" // [한국어] Network 기반 클래스 — 라우터/채널 배열, _Alloc(), step() 등

/*
 * [한국어] QTree - 4진 간접 트리 NoC 토폴로지 클래스
 *
 * k=4, n=3 고정의 4진 트리를 구현한다. assert로 파라미터를 엄격히 강제한다.
 *
 * 트리 구조:
 *   h=0 (루트, 1개 라우터):  k=4 포트, 하위(자식) 연결만 (inject/eject 없음)
 *   h=1 (중간, 4개 라우터):  k+1=5 포트 (하위 4개 + 상위 1개)
 *   h=2 (리프, 16개 라우터): k+1=5 포트 (inject/eject 4개 + 상위 1개)
 *
 * 채널 번호 분류:
 *   _channels / 2 = 자식→부모(UP) 방향 채널 수 = 부모→자식(DOWN) 방향 채널 수
 *   [0, _channels/2)     : _InputIndex()  — UP 방향 (자식이 부모로 보내는 채널)
 *   [_channels/2, _channels): _OutputIndex() — DOWN 방향 (부모가 자식으로 보내는 채널)
 *
 * 라우터 ID 인코딩: id = h * 256 + pos (HeightFromID/PosFromID으로 복원)
 */
class QTree : public Network {

  int _k;
  /* [한국어] 트리의 팬아웃(fan-out) 수 = 방사(radix) = 4로 고정.
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 설정, 즉시 assert(_k==4) 검증.
   * 읽는 자: _ComputeSize()에서 _nodes/_size/_channels 계산,
   *           _BuildNet()에서 포트 수, 채널 인덱스, 루프 상한값 결정.
   *           _RouterIndex(), _InputIndex(), _OutputIndex()에서 지수/계수로 사용.
   * 값 범위: 항상 4 (다른 값이면 assert 실패).
   * 동기화: 초기화 단계에서만 쓰이며 이후 읽기 전용. */

  int _n;
  /* [한국어] 트리의 레벨(깊이) 수 = 3으로 고정.
   * 설정자: _ComputeSize()에서 config.GetInt("n")로 설정, 즉시 assert(_n==3) 검증.
   * 읽는 자: _ComputeSize()에서 _nodes=k^n, _size=sum(k^i,i=0..n-1), _channels 계산.
   *           _BuildNet()에서 레벨 루프 상한, 리프 레벨(h=_n-1) 판별에 사용.
   *           _InputIndex(), _OutputIndex()에서 범위 검증에 사용.
   * 값 범위: 항상 3 (다른 값이면 assert 실패).
   * 동기화: 초기화 단계에서만 쓰이며 이후 읽기 전용. */

  /*
   * [한국어] _ComputeSize - QTree 크기 파라미터 계산
   *
   * @config: BooksimConfig — "k"=4, "n"=3 설정값 읽음
   * @return: 없음 (_k, _n, _nodes, _size, _channels 멤버 설정)
   *
   * k=4, n=3을 assert로 강제하고 이하를 계산한다:
   *   _nodes    = k^n = 64 터미널 노드
   *   _size     = sum(k^i, i=0..n-1) = 1+4+16 = 21 라우터
   *   _channels = sum(2*k^j, j=1..n-1) = 2*4 + 2*16 = 8+32 = 40 채널
   *              (각 레벨 연결마다 UP+DOWN 방향 채널 2개)
   * gK=_k, gN=_n으로 전역 시뮬레이션 파라미터도 업데이트한다.
   *
   * 호출 체인:
   *   QTree() → [_ComputeSize()] → _Alloc() → _BuildNet()
   */
  void _ComputeSize( const Configuration& config );

  /*
   * [한국어] _BuildNet - QTree 라우터 생성 및 채널 연결
   *
   * @config: BooksimConfig — Router::NewRouter()에 전달
   * @return: 없음
   *
   * 두 단계로 구성된다:
   * 1단계 — 라우터 생성:
   *   모든 레벨(h=0..n-1)과 모든 위치(pos=0..k^h-1)에 대해 라우터를 생성한다.
   *   라우터 ID = h*256+pos, 포트 수 d = k(h=0일 때) 또는 k+1(h>0일 때).
   *   _RouterIndex(h,pos)로 _routers[] 배열 인덱스를 계산하여 라우터를 저장한다.
   * 2단계 — 채널 연결:
   *   리프(h=n-1) 라우터에 inject/eject 채널을 4개씩 연결한다.
   *   각 레벨 h의 각 라우터에 대해:
   *     - h < n-1이면 하위 채널(_InputIndex, _OutputIndex)을 자식 방향으로 연결
   *     - h > 0이면 상위 채널(_OutputIndex(h-1,...), _InputIndex(h-1,...))을 부모 방향으로 연결
   *   부모와 자식이 서로 반대 방향으로 같은 채널을 공유한다 (AddInputChannel/AddOutputChannel 반전).
   *
   * 호출 체인:
   *   QTree() → _ComputeSize() → _Alloc() → [_BuildNet()]
   */
  void _BuildNet( const Configuration& config );

  /*
   * [한국어] _RouterIndex - (height, pos) 좌표를 _routers[] 배열 인덱스로 변환
   *
   * @height: 라우터의 트리 레벨 (0=루트, 1=중간, 2=리프)
   * @pos: 해당 레벨 내 위치 (0부터 k^height-1까지)
   * @return: _routers[] 배열의 선형 인덱스
   *
   * 반환값 = sum(k^h, h=0..height-1) + pos
   *   h=0: 0+pos = pos (루트 라우터 1개만 있으므로 0)
   *   h=1: 1+pos (루트 1개 이후의 중간 라우터)
   *   h=2: 1+4+pos = 5+pos (루트+중간 5개 이후의 리프 라우터)
   *
   * 호출 체인:
   *   _BuildNet() → [_RouterIndex(h, pos)] (라우터 생성 및 채널 연결 시)
   */
  int _RouterIndex( int height, int pos );

  /*
   * [한국어] _InputIndex - UP 방향(자식→부모) 채널의 _chan[] 배열 인덱스 계산
   *
   * @height: 부모 라우터의 트리 레벨 (0부터 n-2까지)
   * @pos: 부모 라우터의 레벨 내 위치
   * @port: 부모 라우터의 포트 번호 (0부터 k-1까지)
   * @return: _chan[] 배열 인덱스 ([0, _channels/2) 범위)
   *
   * 채널 번호 공간의 앞쪽 절반(_channels/2개)을 UP 채널에 배정한다.
   * 반환값 = sum(k^(h+1), h=0..height-1) + k*pos + port
   * 이 채널은 자식 라우터에서는 AddOutputChannel, 부모 라우터에서는 AddInputChannel로 사용.
   *
   * 호출 체인:
   *   _BuildNet() → [_InputIndex(h, pos, port)] (부모 라우터에 입력 채널 등록 시)
   */
  int _InputIndex( int height, int pos, int port );

  /*
   * [한국어] _OutputIndex - DOWN 방향(부모→자식) 채널의 _chan[] 배열 인덱스 계산
   *
   * @height: 부모 라우터의 트리 레벨 (0부터 n-2까지)
   * @pos: 부모 라우터의 레벨 내 위치
   * @port: 부모 라우터의 포트 번호 (0부터 k-1까지)
   * @return: _chan[] 배열 인덱스 ([_channels/2, _channels) 범위)
   *
   * 채널 번호 공간의 뒤쪽 절반을 DOWN 채널에 배정한다.
   * 반환값 = _channels/2 + sum(k^(h+1), h=0..height-1) + k*pos + port
   * 이 채널은 부모 라우터에서는 AddOutputChannel, 자식 라우터에서는 AddInputChannel로 사용.
   *
   * 호출 체인:
   *   _BuildNet() → [_OutputIndex(h, pos, port)] (부모 라우터에 출력 채널 등록 시)
   */
  int _OutputIndex( int height, int pos, int port );

public:

  /*
   * [한국어] QTree 생성자 — 4진 트리 네트워크 초기화
   *
   * @config: BooksimConfig — "k"=4, "n"=3, 라우터 내부 설정 포함
   * @name: 네트워크 객체 이름 문자열 (디버그/로그 식별자)
   * @return: 없음 (생성자)
   *
   * Network 기반 클래스 생성자 호출 후 _ComputeSize → _Alloc → _BuildNet 순서로
   * 4진 트리 네트워크를 완전히 구성한다.
   *
   * 호출 체인:
   *   Network::New() → [QTree()] → _ComputeSize() → _Alloc() → _BuildNet()
   */
  QTree( const Configuration& config, const string & name );

  /*
   * [한국어] RegisterRoutingFunctions - QTree 전용 라우팅 함수 등록 (현재 미구현)
   *
   * @return: 없음
   *
   * qtree.cpp에서 빈 함수로 구현됨 — QTree 전용 라우팅 함수가 아직 등록되지 않음.
   * 실제 시뮬레이션에서는 전역 기본 라우팅 함수를 사용하거나 설정 파일로 지정해야 한다.
   *
   * 호출 체인:
   *   Network::RegisterRoutingFunctions() → [QTree::RegisterRoutingFunctions()]
   */
  static void RegisterRoutingFunctions() ;

  /*
   * [한국어] HeightFromID - 라우터 ID에서 트리 레벨(height) 추출
   *
   * @id: 라우터 ID (h*256 + pos 형식으로 인코딩됨)
   * @return: 트리 레벨 h (0=루트, 1=중간, 2=리프)
   *
   * _BuildNet()에서 라우터 ID를 h*256+pos로 인코딩하며, 이를 역으로 복원한다.
   * 반환값 = id / 256
   * 라우팅 함수나 디버그 유틸이 라우터 ID만으로 트리 위치를 파악할 때 사용.
   *
   * 호출 체인:
   *   라우팅 함수 또는 디버그 코드 → [HeightFromID(id)]
   */
  static int HeightFromID( int id );

  /*
   * [한국어] PosFromID - 라우터 ID에서 레벨 내 위치(pos) 추출
   *
   * @id: 라우터 ID (h*256 + pos 형식으로 인코딩됨)
   * @return: 레벨 내 위치 pos (0부터 k^h-1까지)
   *
   * 반환값 = id % 256
   * HeightFromID와 함께 사용하여 라우터의 트리상 좌표 (h, pos)를 완전히 복원한다.
   *
   * 호출 체인:
   *   라우팅 함수 또는 디버그 코드 → [PosFromID(id)]
   */
  static int PosFromID( int id );

};

#endif
