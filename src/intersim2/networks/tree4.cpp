// $Id: tree4.cpp 5188 2012-08-30 00:31:31Z dub $

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
//  Level 0 :  4  8 x 8 Routers   (8 Descending Links per Router)
//  Level 1 :  8  8 x 8 Routers   (4 Descending Links per Router)
//  Level 2 : 16  6 x 6 Routers   (4 Descending Links per Router)
//  Level 3 : 64  Terminal Nodes
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: tree4.cpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

/*
 * [한국어 설명] Tree4 4진 트리 네트워크 구현 (tree4.cpp)
 *
 * === 파일의 역할 ===
 * Tree4 클래스의 전체 구현을 담는다. k=4, n=3 고정의 4진 직접 트리 NoC를 구성하는
 * 모든 로직이 이 파일에 있다. QTree(간접 트리)와 달리 Tree4는 루트 레벨에 1개가 아닌
 * 4개의 라우터를 두며, 각 레벨의 라우터 수를 (4>>h)*k^h 공식으로 계산한다.
 * 라우터 객체 생성(_BuildNet), 레이턴시 설정(SetLatency(1)), inject/eject 채널 연결,
 * 레벨 간 내부 채널 연결의 3단계로 초기화가 완료된다.
 * _WireLatency()는 물리적 레이아웃 기반 레이턴시를 계산하는 레거시 함수이나
 * 현재 _BuildNet()에서는 모두 SetLatency(1)로 대체되어 사용되지 않는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이션 계층의 Tree 계열 토폴로지 구현이다.
 * 네트워크 레벨 구조:
 *   Level 0 (루트): 4개 8×8 라우터 (8 DOWN 포트만, 8 UP 포트 없음)
 *   Level 1 (중간): 8개 8×8 라우터 (4 DOWN + 4 UP 포트)
 *   Level 2 (리프): 16개 6×6 라우터 (4 inject/eject + 2 UP 포트)
 *   Level 3 (단말): 64개 터미널 노드
 * 포트 방향: 출력 0~3=DOWN, 출력 4~7=UP (이 규칙은 라우팅 함수가 사용)
 * 설정 파일에서 topology=tree4, k=4, n=3으로 지정하여 활성화한다.
 * icnt_wrapper.cc → Network::New() → Tree4() 순서로 초기화된다.
 * 이후 매 사이클 gpgpu-sim의 icnt_push/icnt_pop → Network::step()으로 동작한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - booksim.hpp: 전역 시뮬레이션 파라미터 (gK, gN 등)
 *   - tree4.hpp: Tree4 클래스 선언 (멤버 변수, 함수 프로토타입)
 *   - network.hpp: Network 기반 클래스 (_routers, _chan, _inject, _eject, step, _Alloc)
 *   - misc_utils.hpp: powi() — 정수 거듭제곱 (레벨별 라우터 수 계산에 필수)
 *   - router.hpp: Router::NewRouter() — 각 라우터 객체 생성
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: Network::New("tree4", ...)를 통해 Tree4 생성 및 관리
 * 데이터 흐름:
 *   터미널 노드 → _inject[id] → 리프 라우터(h=2) → UP 채널 → 중간 라우터(h=1)
 *   → UP 채널 → 루트 라우터(h=0) → DOWN 채널 → 중간 라우터 → DOWN 채널
 *   → 리프 라우터 → _eject[id] → 목적지 노드
 *
 * === 주요 함수/구조체 요약 ===
 * Tree4()            : 생성자 — _ComputeSize → _Alloc → _BuildNet 표준 3단계 초기화
 * _ComputeSize()     : k=4/n=3 assert, _nodes=64, _size=28, _channels=128 계산
 * RegisterRoutingFunctions(): 빈 함수 — 전용 라우팅 함수 미구현
 * _BuildNet()        : 3레벨 라우터 생성, inject/eject(리프), h=1↔h=2 및 h=0↔h=1 채널 연결
 * _Router(h, pos)    : (h,pos) → _routers[] 참조 반환 (접두 합, 대입 가능한 참조형)
 * _WireLatency(...)  : 물리 위치 기반 레이턴시 계산 (레거시 — _BuildNet에서 미사용)
 * HeightFromID(id)   : id / k^(n-1) = id/16 으로 height 추출
 * PosFromID(id)      : id % k^(n-1) = id%16 으로 pos 추출
 */

#include "booksim.hpp"    // [한국어] 전역 시뮬레이션 파라미터(gK, gN 등) 및 공통 매크로
#include <vector>         // [한국어] std::vector — (현재 직접 사용 없으나 헤더 포함)
#include <sstream>        // [한국어] ostringstream — 라우터 이름 문자열 조합
#include <cmath>          // [한국어] (현재 직접 사용 없으나 레이턴시 계산용으로 포함)

#include "tree4.hpp"      // [한국어] Tree4 클래스 선언 (멤버 변수, 함수 프로토타입)
#include "misc_utils.hpp" // [한국어] powi(base, exp) — 정수 거듭제곱 (_size/_channels 계산에 필수)

/*
 * [한국어]
 * Tree4::Tree4 - 4진 트리 네트워크 생성자
 *
 * @config: BooksimConfig — "k"=4, "n"=3, 라우터 내부 설정(VC 수, 버퍼 크기 등) 포함
 * @name: 네트워크 객체 이름 문자열 (디버그/로그 식별자)
 * @return: 없음 (생성자)
 *
 * Network 기반 클래스 생성자를 먼저 호출한 뒤 표준 3단계 초기화를 수행한다:
 *   1) _ComputeSize(): k=4/n=3 강제 검증, _nodes=64, _size=28, _channels 계산
 *   2) _Alloc():       Network 기반이 _routers[], _chan[], _inject[], _eject[] 배열 할당
 *   3) _BuildNet():    28개 라우터 생성, inject/eject 채널, 레벨 간 UP/DOWN 채널 연결
 * 생성자 반환 시 시뮬레이션 준비 완료 상태.
 *
 * 호출 체인:
 *   Network::New() → [Tree4()] → _ComputeSize() → _Alloc() → _BuildNet()
 */
Tree4::Tree4( const Configuration& config, const string & name )
: Network ( config, name )  // [한국어] Network 기반 클래스 생성자 — _routers/_chan/_inject/_eject 포인터 초기화
{
  _ComputeSize( config );  // [한국어] k=4/n=3 assert 검증, _nodes=64/_size=28/_channels 계산
  _Alloc( );               // [한국어] Network::_Alloc() — 계산된 크기로 라우터/채널 배열 동적 할당
  _BuildNet( config );     // [한국어] 라우터 인스턴스 생성, inject/eject 및 내부 채널 연결
}

/*
 * [한국어]
 * Tree4::_ComputeSize - Tree4 크기 파라미터 계산
 *
 * @config: BooksimConfig — "k"=4, "n"=3 설정값 읽음
 * @return: 없음 (_k, _n, _nodes, _size, _channels 멤버 및 gK, gN 전역 변수 설정)
 *
 * k=4, n=3을 assert로 강제하고 이하를 계산한다:
 *   _nodes    = k^n = 4^3 = 64 터미널 노드
 *   _size     = sum((4>>h)*k^h, h=0..n-1)
 *             = (4>>0)*1 + (4>>1)*4 + (4>>2)*16 = 4 + 8 + 16 = 28 라우터
 *   _channels = 2 * (2*k^1) * (2*k)
 *             = 2 * 8 * 8 = 128 채널
 *             (공식의 의미: 중간 라우터(h=1)가 2*k^1=8개이고 각 중간 라우터가 2*k=8개의
 *              UP+DOWN 채널을 가지며, 이를 2배(UP/DOWN 방향)로 계산)
 *             실제 _BuildNet()에서 사용하는 채널 수는 이보다 적을 수 있음 (일부 미사용).
 * gK=_k, gN=_n으로 전역 라우팅 파라미터 업데이트.
 *
 * 호출 체인:
 *   Tree4() → [_ComputeSize()]
 */
void Tree4::_ComputeSize( const Configuration& config )
{
  int h;  // [한국어] 레벨 루프 변수

  _k = config.GetInt( "k" );  // [한국어] 트리 방사(radix) 읽기 — Tree4는 k=4 고정
  assert(_k == 4);             // [한국어] k=4 강제 — 다른 값이면 즉시 abort
  _n = config.GetInt( "n" );  // [한국어] 트리 레벨(depth) 읽기 — Tree4는 n=3 고정
  assert(_n == 3);             // [한국어] n=3 강제 — 다른 값이면 즉시 abort

  gK = _k; gN = _n;  // [한국어] 전역 시뮬레이션 파라미터 업데이트 — 라우팅 함수에서 gK/gN 참조

  _nodes = powi( _k, _n );  // [한국어] 터미널 노드 수 = k^n = 4^3 = 64

  _size = 0;
  for ( h = 0; h < _n; ++h )
    _size += (4 >> h) * powi( _k, h );
  // [한국어] 레벨별 라우터 수 누적:
  //         h=0: (4>>0)*4^0 = 4*1 = 4  (루트 레벨 4개 라우터)
  //         h=1: (4>>1)*4^1 = 2*4 = 8  (중간 레벨 8개 라우터)
  //         h=2: (4>>2)*4^2 = 1*16= 16 (리프 레벨 16개 라우터)
  //         합계 _size = 28 라우터

  _channels = 2                  // Two Channels per Connection
    * ( 2 * powi( _k, 1) )       // Number of Middle Routers
    * ( 2 * _k );                // Connectivity of Middle Routers
  // [한국어] _channels = 2*(2*k^1)*(2*k) = 2*8*8 = 128 채널
  //         인수 설명:
  //           2: 각 연결마다 UP+DOWN 두 방향 채널
  //           2*powi(_k,1)=8: 중간 라우터(h=1)의 수 (2*4=8)
  //           2*_k=8: 각 중간 라우터의 연결 수 (h=0↔h=1 간 k=4 + h=1↔h=2 간 k=4 = 8)
}

/*
 * [한국어]
 * Tree4::RegisterRoutingFunctions - 라우팅 함수 등록 (현재 빈 함수)
 *
 * @return: 없음
 *
 * Tree4 전용 라우팅 함수가 아직 구현/등록되지 않은 상태이다.
 * 실제 라우팅은 전역 기본 라우팅 함수나 설정 파일의 routing_function 키로 지정된 함수가 담당.
 *
 * 호출 체인:
 *   Network::RegisterRoutingFunctions() → [Tree4::RegisterRoutingFunctions()] (no-op)
 */
void Tree4::RegisterRoutingFunctions(){

}

/*
 * [한국어]
 * Tree4::_BuildNet - Tree4 라우터 생성 및 채널 연결
 *
 * @config: BooksimConfig — Router::NewRouter()에 전달하여 라우터별 VC/버퍼 설정 적용
 * @return: 없음
 *
 * 세 단계로 구성된다:
 *
 * 1단계 — 라우터 생성:
 *   h=0..n-1, pos=0..nPos-1 이중 루프 (nPos=(4>>h)*k^h):
 *     h<n-1이면 degree=8 (8×8 라우터), h=n-1이면 degree=6 (6×6 라우터)
 *     라우터 ID = h*k^(n-1)+pos = h*16+pos
 *     _Router(h,pos)에 생성된 라우터 저장 (_timed_modules에도 추가)
 *
 * 2단계 — inject/eject 채널 (리프 h=n-1=2):
 *   nPos=k^(n-1)=16개 리프 라우터에 각 k=4개 inject/eject 채널 연결.
 *   _Router(n-1,pos)->AddInputChannel(_inject[k*pos+port]): 노드→라우터 inject
 *   _Router(n-1,pos)->AddOutputChannel(_eject[k*pos+port]): 라우터→노드 eject
 *   모두 SetLatency(1) — inject/eject 1사이클 레이턴시
 *
 * 3단계 — 레벨 간 내부 채널:
 *   h=1 ↔ h=2 연결 (중간 라우터 ↔ 리프 라우터):
 *     c=0부터, pos=0..2*k^1-1(=7), port=0..k-1(=3)
 *     pp=pos(중간 라우터 인덱스), pc=k*(pos/2)+port(리프 라우터 인덱스)
 *     _Router(1,pp) 출력 → _chan[c] → _Router(2,pc) 입력 (DOWN)
 *     _Router(1,pp) 입력 ← _chan[c+1] ← _Router(2,pc) 출력 (UP)
 *     각 연결 SetLatency(1)
 *   h=0 ↔ h=1 연결 (루트 라우터 ↔ 중간 라우터):
 *     pos=0..4*k^0-1(=3), port=0..2*k-1(=7)
 *     pp=pos(루트 인덱스), pc=port(중간 라우터 인덱스 — 0~7 전체에 직접 연결)
 *     _Router(0,pp) 출력 → _chan[c] → _Router(1,pc) 입력 (DOWN)
 *     _Router(0,pp) 입력 ← _chan[c+1] ← _Router(1,pc) 출력 (UP)
 *
 * 호출 체인:
 *   Tree4() → _ComputeSize() → _Alloc() → [_BuildNet()]
 */
void Tree4::_BuildNet( const Configuration& config )
{

  //
  // Allocate Routers
  //
  ostringstream name;                  // [한국어] 라우터 이름 문자열 조합용 스트림
  int h, pos, nPos, degree, id;       // [한국어] 루프 변수: h=레벨, pos=위치, nPos=레벨 내 라우터 수, degree=포트 수, id=라우터 ID

  for ( h = 0; h < _n; h++ ) {        // [한국어] 트리의 각 레벨(h=0:루트, h=1:중간, h=2:리프) 순회
    nPos = (4 >> h) * powi( _k, h );  // [한국어] 해당 레벨의 라우터 수 계산: (4>>h)*k^h
    for ( pos = 0; pos < nPos; ++pos) { // [한국어] 해당 레벨의 각 라우터 위치 순회
      if ( h < _n-1 )
	degree = 8;                    // [한국어] 내부 라우터(h=0,1): 8포트 (8×8 라우터)
      else
	degree = 6;                    // [한국어] 리프 라우터(h=2): 6포트 (6×6 라우터 — inject/eject 4 + UP 2)

      name.str("");                    // [한국어] 이전 라우터 이름 지우고 새 이름 조합 준비
      name << "router_" << h << "_" << pos;  // [한국어] "router_<height>_<pos>" 형태 이름 생성
      id = h * powi( _k, _n-1 ) + pos; // [한국어] 라우터 ID 인코딩 = h*k^(n-1)+pos = h*16+pos
      Router * r = Router::NewRouter( config, this, name.str( ),
				      id, degree, degree );  // [한국어] 입력=degree, 출력=degree 포트의 라우터 생성
      _Router( h, pos ) = r;           // [한국어] _Router()의 참조 반환을 활용해 _routers[] 적절한 위치에 저장
      _timed_modules.push_back(r);     // [한국어] 타이밍 모듈 목록에 추가 — 매 사이클 step() 호출 대상
    }
  }

  //
  // Connect Channels to Routers
  //
  int pp, pc;  // [한국어] pp=부모 라우터 위치, pc=자식 라우터 위치
  //
  // Connection Rule: Output Ports 0:3 Move DOWN Network
  //                  Output Ports 4:7 Move UP Network
  //

  // Injection & Ejection Channels
  nPos = powi( _k, _n - 1 );  // [한국어] 리프 라우터 수 = k^(n-1) = 4^2 = 16
  for ( pos = 0 ; pos < nPos ; ++pos ) {     // [한국어] 16개 리프 라우터 순회
    for ( int port = 0 ; port < _k ; ++port ) {  // [한국어] 각 리프 라우터에 k=4개 inject/eject 포트 연결

      _Router( _n-1, pos)->AddInputChannel( _inject[_k*pos+port],
					    _inject_cred[_k*pos+port]);
      // [한국어] inject 채널 등록: 노드 (_k*pos+port)의 패킷이 이 리프 라우터(h=2, pos)로 진입

      _inject[_k*pos+port]->SetLatency( 1 );       // [한국어] inject 채널 레이턴시 1사이클 설정
      _inject_cred[_k*pos+port]->SetLatency( 1 );  // [한국어] inject 크레딧(역방향 흐름 제어) 레이턴시 1사이클

      _Router( _n-1, pos)->AddOutputChannel( _eject[_k*pos+port],
					     _eject_cred[_k*pos+port]);
      // [한국어] eject 채널 등록: 리프 라우터(h=2, pos)에서 노드 (_k*pos+port)로 패킷 배달

      _eject[_k*pos+port]->SetLatency( 1 );        // [한국어] eject 채널 레이턴시 1사이클 설정
      _eject_cred[_k*pos+port]->SetLatency( 1 );   // [한국어] eject 크레딧 레이턴시 1사이클

    }
  }

  // Connections between h = 1 and h = 2 Levels
  int c = 0;                            // [한국어] 채널 번호 카운터 — _chan[] 배열의 현재 위치
  nPos = 2 * powi( _k, 1 );            // [한국어] h=1 레벨 중간 라우터 수 = 2*k^1 = 8
  for ( pos = 0; pos < nPos; ++pos ) { // [한국어] 8개 중간 라우터 순회
    for ( int port = 0; port < _k; ++port ) {  // [한국어] 각 중간 라우터의 k=4개 하위 포트 순회

      pp = pos;                         // [한국어] 부모(중간) 라우터 위치 = pos (h=1 내 위치)
      pc = _k * ( pos / 2 ) + port;    // [한국어] 자식(리프) 라우터 위치 계산
      // [한국어] pos/2: pos번째 중간 라우터가 속한 루트 그룹 (0..3 범위, 2개씩 같은 그룹)
      //         _k*(pos/2): 해당 그룹의 리프 시작 위치
      //         +port: 그룹 내 port번째 리프 라우터

      // cout << "connecting (1,"<<pp<<") <-> (2,"<<pc<<")"<<endl;

      _Router( 1, pp)->AddOutputChannel( _chan[c], _chan_cred[c] );  // [한국어] 중간 라우터 출력 → DOWN 채널 등록
      _Router( 2, pc)->AddInputChannel(  _chan[c], _chan_cred[c] );  // [한국어] 리프 라우터 입력 ← DOWN 채널 등록

      //_chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );

      _chan[c]->SetLatency( 1 );       // [한국어] DOWN 채널(중간→리프) 레이턴시 1사이클 설정
      _chan_cred[c]->SetLatency( 1 );  // [한국어] DOWN 크레딧 채널 레이턴시 1사이클 설정

      c++;  // [한국어] 다음 채널 번호로 증가

      _Router(1, pp)->AddInputChannel( _chan[c], _chan_cred[c] );     // [한국어] 중간 라우터 입력 ← UP 채널 등록
      _Router(2, pc)->AddOutputChannel( _chan[c], _chan_cred[c] );    // [한국어] 리프 라우터 출력 → UP 채널 등록

      //_chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );       // [한국어] UP 채널(리프→중간) 레이턴시 1사이클 설정
      _chan_cred[c]->SetLatency( 1 );  // [한국어] UP 크레딧 채널 레이턴시 1사이클 설정

      c++;  // [한국어] 다음 채널 번호로 증가 (한 연결에 DOWN+UP 2채널 소비)
    }
  }

  // Connections between h = 0 and h = 1 Levels
  nPos = 4 * powi( _k, 0 );            // [한국어] h=0 레벨 루트 라우터 수 = 4*k^0 = 4*1 = 4
  for ( pos  = 0; pos < nPos; ++pos ) {  // [한국어] 4개 루트 라우터 순회
    for ( int port = 0; port < 2 * _k; ++port ) {  // [한국어] 각 루트 라우터의 2*k=8개 하위 포트 순회
      pp = pos;   // [한국어] 부모(루트) 라우터 위치 = pos (h=0 내 위치, 0..3)
      pc = port;  // [한국어] 자식(중간) 라우터 위치 = port (h=1 내 위치, 0..7)
      // [한국어] 루트 라우터 1개가 모든 8개 중간 라우터에 연결됨 (port=0..7이 pc=0..7로 직접 매핑)
      // [한국어] 루트 4개가 각각 중간 8개 전부에 연결 → 루트-중간 연결이 4*8=32쌍 (64채널 UP+DOWN)

      // cout << "connecting (0,"<<pp<<") <-> (1,"<<pc<<")"<<endl;

      _Router(0, pp)->AddOutputChannel( _chan[c], _chan_cred[c] );  // [한국어] 루트 라우터 출력 → DOWN 채널 등록
      _Router(1, pc)->AddInputChannel( _chan[c], _chan_cred[c] );   // [한국어] 중간 라우터 입력 ← DOWN 채널 등록

      //      _chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );       // [한국어] DOWN 채널(루트→중간) 레이턴시 1사이클 설정
      _chan_cred[c]->SetLatency( 1 );  // [한국어] DOWN 크레딧 채널 레이턴시 1사이클 설정

      c++;  // [한국어] 다음 채널 번호로 증가

      _Router(0, pp)->AddInputChannel( _chan[c], _chan_cred[c] );    // [한국어] 루트 라우터 입력 ← UP 채널 등록
      _Router(1, pc)->AddOutputChannel( _chan[c], _chan_cred[c] );   // [한국어] 중간 라우터 출력 → UP 채널 등록

      //  _chan[c]->SetLatency( L );
      // _chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );       // [한국어] UP 채널(중간→루트) 레이턴시 1사이클 설정
      _chan_cred[c]->SetLatency( 1 );  // [한국어] UP 크레딧 채널 레이턴시 1사이클 설정
      c++;  // [한국어] 다음 채널 번호로 증가
    }
  }

  // cout << "Used " << c << " of " << _channels << " channels" << endl;

}

/*
 * [한국어]
 * Tree4::_Router - (height, pos) 좌표로 _routers[] 배열 참조 반환
 *
 * @height: 트리 레벨 (0=루트, 1=중간, 2=리프); assert(height < _n) 검증
 * @pos: 레벨 내 위치 (0부터 (4>>height)*k^height-1); assert로 범위 검증
 * @return: Router*& — _routers[] 배열 원소 참조 (읽기 및 대입 모두 가능)
 *
 * Tree4 특유의 (4>>h)*k^h 공식으로 각 레벨의 라우터 수를 계산하고
 * prefix sum으로 _routers[] 선형 인덱스를 결정한다:
 *   h=0: 시작 인덱스 0,  라우터 4개  → [0..3]
 *   h=1: 시작 인덱스 4,  라우터 8개  → [4..11]
 *   h=2: 시작 인덱스 12, 라우터 16개 → [12..27]
 * 반환이 Router*& 참조형이므로 _Router(h,pos) = r; 형태로 대입 가능하다 (_BuildNet() 사용).
 * QTree::_RouterIndex()와 달리 이 함수는 직접 참조를 반환하는 접근자 패턴이다.
 *
 * 호출 체인:
 *   _BuildNet() → [_Router(h, pos)] (라우터 생성 시 대입, 채널 연결 시 포인터 조회)
 */
Router*& Tree4::_Router( int height, int pos )
{
  assert( height < _n );                                    // [한국어] 유효한 레벨 범위 검증 (0..n-1)
  assert( pos < (4 >> height) * powi( _k, height) );       // [한국어] 해당 레벨의 유효한 위치 범위 검증

  int i = 0;
  for ( int h = 0; h < height; ++h )
    i += (4 >> h) * powi( _k, h );                         // [한국어] h=0..height-1 레벨의 라우터 수를 누적합 (prefix sum)
  return _routers[i+pos];                                   // [한국어] 레벨 시작 인덱스(i)에 레벨 내 위치(pos)를 더한 참조 반환
}

/*
 * [한국어]
 * Tree4::_WireLatency - 두 라우터 간 물리적 거리 기반 채널 레이턴시 계산 (레거시)
 *
 * @height1: 첫 번째 라우터의 트리 레벨
 * @pos1: 첫 번째 라우터의 레벨 내 위치
 * @height2: 두 번째 라우터의 트리 레벨
 * @pos2: 두 번째 라우터의 레벨 내 위치
 * @return: 두 라우터를 잇는 채널의 레이턴시 값 (사이클)
 *
 * 두 라우터를 height 기준으로 Parent(낮은 레벨)와 Child(높은 레벨)로 분류한다.
 * assert로 인접 레벨(heightChild == heightParent+1)임을 강제한다.
 * 레이턴시 값은 물리적 칩 레이아웃 기반 추정치로 하드코딩되어 있다:
 *   _length_d2_d1 = 2: h=2↔h=1 사이의 모든 연결은 2사이클
 *   h=1↔h=0 연결은 posChild 및 posParent에 따라:
 *     posChild 0/6, posParent 0: 2, 1: 2, 2: 6, 3: 6
 *     posChild 1/7, posParent 0: 6, 1: 6, 2: 2, 3: 2
 *     posChild 2/4, posParent 0: 2, 1: 2, 2: 6, 3: 6
 *     posChild 3/5, posParent 0: 6, 1: 6, 2: 2, 3: 2
 *     (물리적으로 가까운 루트-중간 쌍은 2사이클, 먼 쌍은 6사이클)
 * 주의: 현재 _BuildNet()에서는 모두 SetLatency(1)로 교체되어 이 함수는 호출되지 않음.
 *       향후 물리 레이아웃 기반 시뮬레이션 재활성화 시 사용 가능.
 *
 * 호출 체인:
 *   (현재 호출되지 않음 — 레거시)
 */
int Tree4::_WireLatency( int height1, int pos1, int height2, int pos2 )
{
  int heightChild, heightParent, posChild, posParent;  // [한국어] 부모/자식 라우터 좌표 분리용 변수

  int L;  // [한국어] 계산된 레이턴시 결과값

  if (height1 < height2) {     // [한국어] height1이 더 낮으면(루트에 가까우면) height1이 부모
    heightChild  = height2;    // [한국어] 더 높은 레벨(리프에 가까운)이 자식
    posChild     = pos2;       // [한국어] 자식 위치
    heightParent = height1;    // [한국어] 더 낮은 레벨(루트에 가까운)이 부모
    posParent    = pos1;       // [한국어] 부모 위치
  } else {                     // [한국어] height2가 더 낮으면 height2가 부모
    heightChild  = height1;    // [한국어] 자식 = height1 (더 높은 레벨)
    posChild     = pos1;
    heightParent = height2;    // [한국어] 부모 = height2 (더 낮은 레벨)
    posParent    = pos2;
  }

  int _length_d2_d1   = 2 ;   // [한국어] h=2(리프)↔h=1(중간) 연결 레이턴시 = 2사이클 (물리적 인접)
  int _length_d1_d0_0 = 2 ;   // [한국어] h=1↔h=0, 루트 위치 0에 대한 최단 거리 레이턴시 = 2사이클
  int _length_d1_d0_1 = 2 ;   // [한국어] h=1↔h=0, 루트 위치 1에 대한 레이턴시 = 2사이클
  int _length_d1_d0_2 = 6 ;   // [한국어] h=1↔h=0, 루트 위치 2에 대한 원거리 레이턴시 = 6사이클
  int _length_d1_d0_3 = 6 ;   // [한국어] h=1↔h=0, 루트 위치 3에 대한 원거리 레이턴시 = 6사이클

  assert( heightChild == heightParent+1 );  // [한국어] 인접 레벨 간 연결만 지원 — 비인접 레벨 연결은 오류

  // We must decrement the delays by one to account for how the
  //  simulator interprets the specified delay (with 0 indicating one
  //  cycle of delay).
  // [한국어] 시뮬레이터는 지정된 레이턴시를 오프셋으로 해석하므로 (0=1사이클)
  //         실제 값에서 1을 빼야 하지만, 이 로직은 현재 코드에 반영되지 않았다 (레거시 주석).

  if ( heightChild == 2 )
    L = _length_d2_d1;  // [한국어] 리프↔중간 연결 — 레이턴시 2사이클 (물리적 근거리)
  else {
       if ( posChild == 0 || posChild == 6 )  // [한국어] 중간 라우터 위치 0 또는 6
      switch ( posParent ) {
      case 0: L =_length_d1_d0_0; break;  // [한국어] 루트 0↔중간 0/6: 2사이클 (물리적 근거리)
      case 1: L =_length_d1_d0_1; break;  // [한국어] 루트 1↔중간 0/6: 2사이클
      case 2: L =_length_d1_d0_2; break;  // [한국어] 루트 2↔중간 0/6: 6사이클 (물리적 원거리)
      case 3: L =_length_d1_d0_3; break;  // [한국어] 루트 3↔중간 0/6: 6사이클
      }
    if ( posChild == 1 || posChild == 7 )  // [한국어] 중간 라우터 위치 1 또는 7
      switch ( posParent ) {
      case 0: L =_length_d1_d0_3; break;  // [한국어] 루트 0↔중간 1/7: 6사이클 (원거리)
      case 1: L =_length_d1_d0_2; break;  // [한국어] 루트 1↔중간 1/7: 6사이클
      case 2: L =_length_d1_d0_1; break;  // [한국어] 루트 2↔중간 1/7: 2사이클
      case 3: L =_length_d1_d0_0; break;  // [한국어] 루트 3↔중간 1/7: 2사이클
      }
    if ( posChild == 2 || posChild == 4 )  // [한국어] 중간 라우터 위치 2 또는 4
      switch ( posParent ) {
      case 0: L = _length_d1_d0_0; break;  // [한국어] 루트 0↔중간 2/4: 2사이클
      case 1: L = _length_d1_d0_1; break;  // [한국어] 루트 1↔중간 2/4: 2사이클
      case 2: L = _length_d1_d0_2; break;  // [한국어] 루트 2↔중간 2/4: 6사이클
      case 3: L = _length_d1_d0_3; break;  // [한국어] 루트 3↔중간 2/4: 6사이클
      }
    if ( posChild == 3|| posChild == 5 )   // [한국어] 중간 라우터 위치 3 또는 5
      switch ( posParent ) {
      case 0: L =_length_d1_d0_3; break;   // [한국어] 루트 0↔중간 3/5: 6사이클
      case 1: L =_length_d1_d0_2; break;   // [한국어] 루트 1↔중간 3/5: 6사이클
      case 2: L =_length_d1_d0_1; break;   // [한국어] 루트 2↔중간 3/5: 2사이클
      case 3: L =_length_d1_d0_0; break;   // [한국어] 루트 3↔중간 3/5: 2사이클
      }
  }
  return L;  // [한국어] 계산된 레이턴시 반환 (사이클 단위, 현재 _BuildNet에서 미사용)
}
