// $Id: qtree.cpp 5188 2012-08-30 00:31:31Z dub $

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
//  $Id: qtree.cpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

/*
 * [한국어 설명] QTree 4진 간접 트리 네트워크 구현 (qtree.cpp)
 *
 * === 파일의 역할 ===
 * QTree 클래스의 전체 구현을 담는다. k=4, n=3 고정 파라미터로 4진 간접 트리 NoC를
 * 구성한다. 3레벨(루트/중간/리프)의 라우터 객체를 생성하고, 각 레벨 간 UP/DOWN
 * 채널을 체계적인 인덱싱 함수(_RouterIndex, _InputIndex, _OutputIndex)로 연결한다.
 * 리프(h=2) 라우터에만 inject/eject 채널이 연결되어 터미널 노드의 패킷 진입/진출이
 * 이루어지며, 내부(h=0,1) 라우터는 순수 스위칭 역할만 한다.
 * 채널은 모두 SetLatency(기본값)로 레이턴시가 설정되며 현재는 명시 설정 없음(기본 1사이클).
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 시뮬레이션 계층(intersim2) 내 Tree 계열 토폴로지 구현이다.
 * icnt_wrapper.cc가 Network::New("qtree", ...)를 호출하면 QTree 인스턴스가 생성된다.
 * 초기화 완료 후 gpgpu-sim의 icnt_push/icnt_pop 인터페이스를 통해 매 사이클 step()
 * 호출로 플릿이 트리 토폴로지를 따라 라우팅된다.
 * 설정 파일(gpgpusim.config)에서 topology=qtree, k=4, n=3으로 지정하여 활성화한다.
 * 실행 컨텍스트: 초기화는 메인 스레드; 이후 시뮬레이션 루프(gpgpusim 시뮬 스레드)에서
 * 매 사이클 step()이 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - booksim.hpp: 전역 파라미터(gK, gN) 및 전처리 매크로
 *   - qtree.hpp: QTree 클래스 선언 및 멤버 타입
 *   - network.hpp: Network 기반 클래스 (_routers, _chan, _inject, _eject, _Alloc, step)
 *   - misc_utils.hpp: powi(base, exp) — 정수 거듭제곱 계산
 *   - router.hpp: Router::NewRouter() — 각 라우터 객체 생성
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: Network::New()를 통해 QTree 생성 및 관리
 * 데이터 흐름:
 *   터미널 노드 → _inject[node_id] → 리프 라우터(h=2) → UP 채널 → 중간 라우터(h=1)
 *   → UP 채널 → 루트 라우터(h=0) → DOWN 채널 → 중간 라우터 → DOWN 채널
 *   → 리프 라우터 → _eject[node_id] → 목적지 노드
 *
 * === 주요 함수/구조체 요약 ===
 * QTree()            : 생성자 — _ComputeSize → _Alloc → _BuildNet 표준 3단계 초기화
 * _ComputeSize()     : k=4/n=3 assert, _nodes=64/_size=21/_channels=40 계산, gK/gN 전역 설정
 * RegisterRoutingFunctions(): 빈 함수 — 전용 라우팅 함수 미구현
 * _BuildNet()        : 3레벨 라우터 생성, 리프에 inject/eject 연결, 레벨 간 UP/DOWN 채널 연결
 * _RouterIndex(h,p)  : prefix sum으로 (h,p) → _routers[] 선형 인덱스 변환
 * _InputIndex(h,p,p) : UP 방향 채널 번호 계산 ([0, _channels/2) 공간)
 * _OutputIndex(h,p,p): DOWN 방향 채널 번호 계산 ([_channels/2, _channels) 공간)
 * HeightFromID(id)   : id/256 으로 height 추출
 * PosFromID(id)      : id%256 으로 pos 추출
 */

#include "booksim.hpp"      // [한국어] 전역 시뮬레이션 파라미터(gK, gN 등) 및 공통 매크로
#include <vector>           // [한국어] std::vector — _BuildNet()에서 채널 순서 관리용
#include <sstream>          // [한국어] ostringstream — 라우터 이름 문자열 조합
#include "qtree.hpp"        // [한국어] QTree 클래스 선언 (멤버 변수, 함수 프로토타입)
#include "misc_utils.hpp"   // [한국어] powi(base, exp) — 정수 거듭제곱 (_size/_channels/_nodes 계산에 필수)

/*
 * [한국어]
 * QTree::QTree - 4진 간접 트리 네트워크 생성자
 *
 * @config: BooksimConfig — "k"=4, "n"=3, 라우터 내부 파라미터(VC 수, 버퍼 크기 등) 포함
 * @name: 네트워크 객체 이름 문자열 (디버그/로그 식별자)
 * @return: 없음 (생성자)
 *
 * Network 기반 클래스 생성자 호출 후 표준 3단계로 QTree를 완전히 초기화한다:
 *   1) _ComputeSize(): k/n 검증, _nodes=64, _size=21, _channels=40 계산
 *   2) _Alloc():       Network 기반 클래스가 배열(_routers, _chan, _inject, _eject) 동적 할당
 *   3) _BuildNet():    21개 라우터 생성, inject/eject 채널, UP/DOWN 내부 채널 연결
 *
 * 호출 체인:
 *   Network::New() → [QTree()] → _ComputeSize() → _Alloc() → _BuildNet()
 */
QTree::QTree( const Configuration& config, const string & name )
: Network ( config, name )  // [한국어] Network 기반 클래스 초기화 — _routers/_chan/_inject/_eject 포인터를 nullptr로 초기화
{
  _ComputeSize( config );  // [한국어] k=4/n=3 검증 및 _nodes=64, _size=21, _channels=40 계산
  _Alloc( );               // [한국어] Network::_Alloc() — 계산된 크기로 라우터/채널 배열 동적 할당
  _BuildNet( config );     // [한국어] 라우터 인스턴스 생성, inject/eject 및 내부 채널 연결
}


/*
 * [한국어]
 * QTree::_ComputeSize - 4진 트리 네트워크 크기 파라미터 계산
 *
 * @config: BooksimConfig — "k", "n" 설정 값 읽음
 * @return: 없음 (_k, _n, _nodes, _size, _channels 멤버 및 gK, gN 전역 변수 설정)
 *
 * k=4, n=3을 assert로 강제한 뒤 이하를 계산한다:
 *   _nodes    = powi(k, n) = 4^3 = 64 터미널 노드
 *   _size     = sum(powi(k,i), i=0..n-1) = 1 + 4 + 16 = 21 라우터
 *   _channels = sum(2*powi(k,j), j=1..n-1) = 2*4 + 2*16 = 40 채널
 *               (j=1: h=0↔h=1 레벨 간 연결 2*k^1=8, j=2: h=1↔h=2 레벨 간 2*k^2=32)
 * gK=_k, gN=_n으로 전역 시뮬레이션 파라미터를 업데이트하여 라우팅 함수에서 참조 가능하게 한다.
 *
 * 호출 체인:
 *   QTree() → [_ComputeSize()] → (gK=4, gN=3 설정)
 */
void QTree::_ComputeSize( const Configuration& config )
{

  _k = config.GetInt( "k" );  // [한국어] 트리 방사(radix) k 읽기 — QTree는 k=4 고정
  _n = config.GetInt( "n" );  // [한국어] 트리 레벨(depth) n 읽기 — QTree는 n=3 고정

  assert( _k == 4 && _n == 3 );  // [한국어] QTree는 k=4, n=3만 지원 — 다른 값이면 즉시 abort

  gK = _k; gN = _n;  // [한국어] 전역 시뮬레이션 파라미터 업데이트 — 라우팅 함수에서 gK/gN 참조

  _nodes = powi( _k, _n );  // [한국어] 터미널 노드 수 = k^n = 4^3 = 64

  _size = 0;
  for (int i = 0; i < _n; i++)
    _size += powi( _k, i );  // [한국어] 라우터 수 = sum(k^i, i=0..n-1) = 1+4+16 = 21
                              //         h=0: k^0=1(루트), h=1: k^1=4(중간), h=2: k^2=16(리프)

  _channels = 0;
  for (int j = 1; j < _n; j++)
    _channels += 2 * powi( _k, j );  // [한국어] 채널 수 = sum(2*k^j, j=1..n-1)
                                      //         j=1: 2*4=8 (h=0↔h=1 레벨), j=2: 2*16=32 (h=1↔h=2 레벨)
                                      //         각 연결에 UP+DOWN 2개 채널 할당 → 합계 40

}

/*
 * [한국어]
 * QTree::RegisterRoutingFunctions - 라우팅 함수 등록 (현재 빈 함수)
 *
 * @return: 없음
 *
 * QTree 전용 라우팅 함수가 아직 구현/등록되지 않은 상태이다.
 * 실제 라우팅은 전역 기본 라우팅 함수나 설정 파일로 지정된 다른 함수가 담당한다.
 *
 * 호출 체인:
 *   Network::RegisterRoutingFunctions() → [QTree::RegisterRoutingFunctions()] (no-op)
 */
void QTree::RegisterRoutingFunctions(){

}

/*
 * [한국어]
 * QTree::_BuildNet - 4진 트리 라우터 생성 및 채널 연결
 *
 * @config: BooksimConfig — Router::NewRouter()에 전달하여 라우터별 설정 적용
 * @return: 없음
 *
 * 세 단계로 네트워크를 구성한다:
 *
 * 1단계 — 라우터 생성:
 *   높이 h=0..n-1, 위치 pos=0..k^h-1 이중 루프로 모든 라우터를 생성한다.
 *   라우터 ID = h*256+pos (HeightFromID/PosFromID로 역 복원 가능).
 *   포트 수 d: h=0이면 k(4), h>0이면 k+1(5) — 루트는 하위 포트만, 나머지는 상위 포트 추가.
 *   _RouterIndex(h,pos)로 _routers[] 배열 위치를 결정한다.
 *
 * 2단계 — inject/eject 채널:
 *   리프(h=n-1=2)의 각 라우터(pos=0..15)에 k=4개의 inject/eject 채널을 연결한다.
 *   inject[k*pos+port]: 노드 k*pos+port의 inject 채널 → 리프 라우터 입력
 *   eject[k*pos+port]:  리프 라우터 출력 → 노드 k*pos+port의 eject 채널
 *
 * 3단계 — 내부 채널 (레벨 간 UP/DOWN 연결):
 *   h=0..n-1, pos=0..k^h-1, port=0..k-1 삼중 루프:
 *   h < n-1이면 DOWN 채널(_OutputIndex)을 부모 출력으로, UP 채널(_InputIndex)을 부모 입력으로 등록
 *   h > 0이면 반대 방향: 부모의 DOWN채널을 자신의 입력으로, 부모의 UP채널을 자신의 출력으로 등록
 *   부모 라우터의 (h-1, pos/k, pos%k)로 채널 인덱스를 계산한다.
 *
 * 채널 레이턴시: 명시적 SetLatency 호출 없음 → Network 기반 클래스 기본값(1사이클) 사용.
 *
 * 호출 체인:
 *   QTree() → _ComputeSize() → _Alloc() → [_BuildNet()]
 */
void QTree::_BuildNet( const Configuration& config )
{

  ostringstream routerName;          // [한국어] 라우터 이름 문자열 조합용 스트림
  int h, r = 0 , pos, port;         // [한국어] 루프 변수: h=레벨, r=배열 인덱스, pos=레벨 내 위치, port=포트 번호

  for (h = 0; h < _n; h++) {        // [한국어] 트리의 각 레벨(h=0:루트, h=1:중간, h=2:리프) 순회
    for (pos = 0 ; pos < powi( _k, h ) ; ++pos ) {  // [한국어] 해당 레벨의 라우터 수(k^h)만큼 순회

      int id = h * 256 + pos;        // [한국어] 라우터 ID 인코딩 = h*256+pos (HeightFromID/PosFromID로 역 복원)
      r = _RouterIndex( h, pos );    // [한국어] (h,pos) → _routers[] 배열 선형 인덱스 계산

      routerName << "router_" << h << "_" << pos;  // [한국어] "router_<height>_<pos>" 형태의 이름 생성

      int d = ( h == 0 ) ? _k : _k + 1;  // [한국어] 포트 수: 루트(h=0)는 k=4(하위만), 나머지는 k+1=5(상위 포트 추가)
      _routers[r] = Router::NewRouter( config, this,
				       routerName.str( ),
				       id, d, d);        // [한국어] 입력=d, 출력=d 포트를 가진 라우터 객체 생성
      _timed_modules.push_back(_routers[r]);       // [한국어] 타이밍 시뮬레이션 모듈 목록에 추가 — 매 사이클 step() 호출 대상
      routerName.str("");                           // [한국어] 다음 라우터 이름을 위해 스트림 버퍼 초기화
    }
  }

  // Injection & Ejection Channels
  for ( pos = 0 ; pos < powi( _k, _n-1 ) ; ++pos ) {  // [한국어] 리프 라우터 수(k^(n-1)=16)만큼 순회
    r = _RouterIndex( _n-1, pos );                      // [한국어] 리프 레벨(h=n-1=2)의 pos번째 라우터 인덱스
    for ( port = 0 ; port < _k ; port++ ) {             // [한국어] 각 리프 라우터에 k=4개의 inject/eject 포트 연결

      _routers[r]->AddInputChannel( _inject[_k*pos+port],
				    _inject_cred[_k*pos+port]);
      // [한국어] inject 채널 등록: 노드 (_k*pos+port)의 패킷이 이 리프 라우터로 진입
      //         _inject_cred는 역방향 크레딧 채널 (흐름 제어)

      _routers[r]->AddOutputChannel( _eject[_k*pos+port],
				     _eject_cred[_k*pos+port]);
      // [한국어] eject 채널 등록: 이 리프 라우터에서 노드 (_k*pos+port)로 패킷 배달
    }
  }

  int c;
  for ( h = 0 ; h < _n ; ++h ) {              // [한국어] 모든 레벨 순회
    for ( pos = 0 ; pos < powi( _k, h ) ; ++pos ) {  // [한국어] 해당 레벨의 모든 라우터 순회
      for ( port = 0 ; port < _k ; port++ ) { // [한국어] 각 라우터의 k=4개 하위 포트 순회

	r = _RouterIndex( h, pos );           // [한국어] 현재 처리 중인 라우터(h,pos)의 배열 인덱스

	if ( h < _n-1 ) {
	  // Channels to Children Nodes
	  c = _InputIndex( h , pos, port );   // [한국어] (h,pos)의 port번째 자식이 보내는 UP 채널 인덱스
	  _routers[r]->AddInputChannel( _chan[c],
					_chan_cred[c] );
          // [한국어] 부모 라우터(h,pos)의 입력 포트에 UP 채널 등록 — 자식에서 올라오는 패킷 수신

	  c = _OutputIndex( h, pos, port );   // [한국어] (h,pos)에서 port번째 자식으로 내려가는 DOWN 채널 인덱스
	  _routers[r]->AddOutputChannel( _chan[c],
					 _chan_cred[c] );
          // [한국어] 부모 라우터(h,pos)의 출력 포트에 DOWN 채널 등록 — 자식으로 내려가는 패킷 송신

	}
      }
      if ( h > 0 ) {
	// Channels to Parent Nodes
	c = _OutputIndex( h - 1, pos / _k, pos % _k );  // [한국어] 부모(h-1, pos/k)에서 현재 자식으로의 DOWN 채널
        // [한국어] pos/_k: 현재 라우터의 부모 위치; pos%_k: 부모의 몇 번째 자식인지
	_routers[r]->AddInputChannel( _chan[c],
				      _chan_cred[c] );
        // [한국어] 자식 라우터(h,pos)의 입력 포트에 DOWN 채널 등록 — 부모가 내려보내는 패킷 수신

	c = _InputIndex( h - 1, pos / _k, pos % _k );   // [한국어] 현재 자식에서 부모(h-1, pos/k)로의 UP 채널
	_routers[r]->AddOutputChannel( _chan[c],
				       _chan_cred[c]);
        // [한국어] 자식 라우터(h,pos)의 출력 포트에 UP 채널 등록 — 부모로 올라가는 패킷 송신
      }
    }
  }
}

/*
 * [한국어]
 * QTree::_RouterIndex - (height, pos) 좌표를 _routers[] 배열 인덱스로 변환
 *
 * @height: 트리 레벨 (0=루트, 1=중간, 2=리프)
 * @pos: 해당 레벨 내 위치 (0부터 k^height-1까지)
 * @return: _routers[] 배열의 선형 인덱스
 *
 * prefix sum 방식: 레벨 0..height-1까지의 라우터 수를 누적합으로 계산한 뒤 pos를 더한다.
 *   h=0: 0 + pos = pos         (루트: [0])
 *   h=1: 1 + pos               (중간: [1..4])
 *   h=2: 1+4 + pos = 5+pos     (리프: [5..20])
 * 이 인덱스는 _routers[], _BuildNet()의 라우터 생성 루프 순서와 일치해야 한다.
 *
 * 호출 체인:
 *   _BuildNet() → [_RouterIndex(h, pos)] (라우터 생성 및 채널 연결 시 반복 호출)
 */
int QTree::_RouterIndex( int height, int pos )
{
  int r = 0;
  for ( int h = 0; h < height; h++ )
    r += powi( _k, h );  // [한국어] 레벨 h의 라우터 수(k^h)를 누적 합산 (prefix sum)
  return (r + pos);       // [한국어] 해당 레벨 시작 인덱스에 레벨 내 위치 추가
}

/*
 * [한국어]
 * QTree::_InputIndex - UP 방향(자식→부모) 채널의 _chan[] 배열 인덱스 계산
 *
 * @height: 부모 라우터의 트리 레벨 (0 이상 k^(n-1) 미만 — 즉 n-2까지)
 * @pos: 부모 라우터의 레벨 내 위치
 * @port: 부모 라우터의 하위 포트 번호 (0부터 k-1까지 — 몇 번째 자식인지)
 * @return: _chan[] 배열 인덱스 ([0, _channels/2) 범위의 UP 채널)
 *
 * _channels의 앞쪽 절반([0, _channels/2))을 UP 채널(자식→부모)에 배정한다.
 * 반환값 = sum(k^(h+1), h=0..height-1) + k*pos + port
 *   height=0, pos=0, port=0: c=0  (루트의 첫 번째 UP 채널)
 *   height=0, pos=0, port=3: c=3  (루트의 마지막 UP 채널)
 *   height=1, pos=0, port=0: c=4  (첫 번째 중간 라우터의 첫 UP 채널)
 * 같은 (height,pos,port) 쌍에 대해 _InputIndex와 _OutputIndex는 다른 인덱스를 반환하므로
 * 같은 물리 연결의 UP/DOWN 두 채널을 구별한다.
 *
 * 호출 체인:
 *   _BuildNet() → [_InputIndex(h, pos, port)] (부모 라우터 입력 채널 등록 및 자식 출력 채널 등록 시)
 */
int QTree::_InputIndex( int height, int pos, int port )
{
  assert( height >= 0 && height < powi( _k,_n-1 ) );  // [한국어] height 범위 검증 (리프는 하위 채널 없으므로 n-2까지)
  int c = 0;
  for ( int h = 0; h < height; h++)
    c += powi( _k, h+1 );                // [한국어] 레벨 h의 UP 채널 수(k^(h+1))를 누적 합산
  return ( c + _k * pos + port );         // [한국어] 레벨 시작 오프셋 + 라우터 내 포트 위치
}

/*
 * [한국어]
 * QTree::_OutputIndex - DOWN 방향(부모→자식) 채널의 _chan[] 배열 인덱스 계산
 *
 * @height: 부모 라우터의 트리 레벨 (0 이상 k^(n-1) 미만)
 * @pos: 부모 라우터의 레벨 내 위치
 * @port: 부모 라우터의 하위 포트 번호 (0부터 k-1까지)
 * @return: _chan[] 배열 인덱스 ([_channels/2, _channels) 범위의 DOWN 채널)
 *
 * _channels의 뒤쪽 절반([_channels/2, _channels))을 DOWN 채널(부모→자식)에 배정한다.
 * 반환값 = _channels/2 + sum(k^(h+1), h=0..height-1) + k*pos + port
 * _channels/2가 UP 채널 공간의 끝이자 DOWN 채널 공간의 시작이며,
 * 이후 오프셋 계산은 _InputIndex와 동일한 방식을 따른다.
 *
 * 호출 체인:
 *   _BuildNet() → [_OutputIndex(h, pos, port)] (부모 라우터 출력 채널 등록 및 자식 입력 채널 등록 시)
 */
int QTree::_OutputIndex( int height, int pos, int port )
{
  assert( height >= 0 && height < powi( _k,_n-1 ) );  // [한국어] height 범위 검증
  int c = _channels / 2;                               // [한국어] DOWN 채널 공간 시작 오프셋 (_channels/2)
  for ( int h = 0; h < height; h++)
    c += powi( _k, h+1 );                              // [한국어] 레벨 h의 DOWN 채널 수(k^(h+1)) 누적 합산
  return ( c + _k * pos + port );                      // [한국어] 레벨 시작 오프셋 + 라우터 내 포트 위치
}


/*
 * [한국어]
 * QTree::HeightFromID - 라우터 ID에서 트리 레벨(height) 추출
 *
 * @id: 라우터 ID (_BuildNet()에서 h*256+pos로 인코딩됨)
 * @return: 트리 레벨 h (0=루트, 1=중간, 2=리프)
 *
 * _BuildNet()에서 라우터 ID를 h*256+pos로 인코딩하므로 h = id/256으로 복원 가능하다.
 * 정적 함수이므로 QTree 인스턴스 없이 라우터 ID만으로 트리 위치를 파악할 수 있다.
 * 라우팅 함수나 디버그 코드에서 호출된다.
 *
 * 호출 체인:
 *   라우팅 함수 또는 디버그 코드 → [QTree::HeightFromID(id)]
 */
int QTree::HeightFromID( int id )
{
  return id / 256;  // [한국어] ID 인코딩 h*256+pos에서 정수 나눗셈으로 height 추출
}

/*
 * [한국어]
 * QTree::PosFromID - 라우터 ID에서 레벨 내 위치(pos) 추출
 *
 * @id: 라우터 ID (_BuildNet()에서 h*256+pos로 인코딩됨)
 * @return: 레벨 내 위치 pos (0부터 k^h-1까지)
 *
 * 반환값 = id % 256 으로 pos를 복원한다.
 * HeightFromID()와 함께 사용하여 라우터의 (height, pos) 좌표를 완전히 복원한다.
 * 256을 모듈러로 사용하므로 pos가 0..255 범위임을 전제한다 (k^2=16이므로 충분).
 *
 * 호출 체인:
 *   라우팅 함수 또는 디버그 코드 → [QTree::PosFromID(id)]
 */
int QTree::PosFromID( int id )
{
  return id % 256;  // [한국어] ID 인코딩 h*256+pos에서 나머지 연산으로 pos 추출
}
