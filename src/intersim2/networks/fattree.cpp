// $Id: fattree.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] Fat Tree 네트워크 토폴로지 구현 (fattree.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 NoC 시뮬레이터에서 k진 n레벨 Fat Tree 토폴로지를 완전히
 * 구성하는 FatTree 클래스의 구현을 담는다. Fat Tree(비만 트리)는 상위 계층으로
 * 갈수록 링크 대역폭이 넓어지는 계층적 트리로, 임의의 두 노드 사이에 여러 경로가
 * 존재해 bisection bandwidth가 높다. 이 파일은 (1) 네트워크 크기 계산,
 * (2) 라우터 생성, (3) inject/eject/down/up 채널 배선의 세 단계를 순서대로
 * 수행하여 시뮬레이션 준비를 완료한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim에서 SM(Shader Core)과 L2 캐시/메모리 컨트롤러 사이 인터커넥트 역할:
 *   SM → icnt_wrapper.cc → FatTree(intersim2) → L2/DRAM
 * Booksim2 계층에서는 Network 기반 클래스를 상속하며, 생성자에서 라우터와 채널을
 * 완전히 배선한 뒤 사이클마다 Network::Step() → Router::Step()으로 패킷을
 * 전진시킨다. 실행 컨텍스트: 호스트 유저스페이스 시뮬레이터 스레드(단일 스레드).
 * 호출 체인: icnt_wrapper_init() → FatTree 생성자 → _ComputeSize/_Alloc/_BuildNet
 *            ; 사이클마다: gpu-sim.cc cycle() → icnt_push/pop → Network::Step()
 *
 * === 타 모듈과의 연결 ===
 * - 의존: network.hpp (Network 기반 클래스 — _routers[], _chan[], _inject[],
 *         _eject[], _Alloc() 제공), router.hpp (Router::NewRouter()),
 *         misc_utils.hpp (powi() — 정수 거듭제곱), booksim.hpp (전역 gK/gN),
 *         configuration.hpp (Configuration::GetInt())
 * - 피의존: icnt_wrapper.cc (GPGPU-Sim ↔ Booksim2 브리지, FatTree를 Network*로
 *           업캐스팅하여 사용), booksim_config.cpp (topology="fattree" 시 선택)
 * - 공유 자료구조: Network::_routers[] (크기 _n*k^(n-1), 레벨×위치 행렬로 접근),
 *                 Network::_chan[] (내부 채널 배열, 크기 _channels),
 *                 _inject[]/_eject[] (단말-라우터 외부 채널, 크기 _nodes)
 * - 전역 변수 gK/gN: booksim.hpp 정의, 이 파일에서 _k/_n으로 설정함.
 *
 * === 주요 함수/구조체 요약 ===
 * - FatTree()         : 생성자 — _ComputeSize→_Alloc→_BuildNet 순서로 초기화
 * - _ComputeSize()    : _k,_n 읽어 _nodes=k^n, _size=n*k^(n-1), _channels=2*k*k^(n-1)*(n-1) 계산
 * - _BuildNet()       : 라우터 생성 + inject/eject/down-output/up-output/down-input/up-input 연결
 * - _Router(d,p)      : depth*k^(n-1)+pos 공식으로 _routers[] 원소 참조 반환
 * - RegisterRoutingFunctions(): 빈 함수 (라우팅 함수 미구현)
 *
 * 채널 번호 체계 요약:
 *   chan_per_direction = k * k^(n-1)   (단방향 채널 수)
 *   chan_per_level     = 2 * chan_per_direction  (레벨당 채널 수)
 *   down-output 채널: level*chan_per_level + pos*k + port
 *   up-output 채널:   level*chan_per_level - chan_per_direction + pos*k + port
 *   down-input / up-input 채널: 이웃(neighborhood) 단위의 인터리브 번호 체계 사용
 */

////////////////////////////////////////////////////////////////////////
//
// FatTree
//
//       Each level of the hierarchical indirect Network has
//       k^(n-1) Routers. The Routers are organized such that
//       each node has k descendents, and each parent is
//       replicated k  times.
//      most routers has 2K ports, excep the top level has only K
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:50:48 $
//  $Id: fattree.cpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////


#include "booksim.hpp"
/* [한국어] Booksim2 공통 헤더 포함 — 전역 변수 gK, gN, gC 등 선언과
 * 시뮬레이터 전역 설정(assert 매크로, 타입 정의)을 제공한다.
 * gK/_n은 이 파일에서 _ComputeSize()를 통해 설정된다. */

#include <vector>
/* [한국어] std::vector 포함 — 이 파일에서는 직접 사용하지 않으나
 * 포함된 헤더(network.hpp 등)의 의존성을 위해 포함한다. */

#include <sstream>
/* [한국어] std::ostringstream 포함 — _BuildNet()에서 라우터 이름 문자열
 * "router_level<N>_<P>"를 동적으로 생성할 때 사용한다. */

#include <cmath>
/* [한국어] C 수학 함수 포함 — pow() 등. 이 파일에서는 powi()
 * (misc_utils.hpp 제공 정수 거듭제곱)를 사용하므로 직접 호출은 없으나
 * 의존 헤더가 요구하므로 포함한다. */

#include "fattree.hpp"
/* [한국어] FatTree 클래스 선언 헤더 포함 — 이 .cpp 파일은 fattree.hpp에서
 * 선언된 FatTree 클래스의 멤버 함수들을 정의한다. */

#include "misc_utils.hpp"
/* [한국어] Booksim2 유틸리티 함수 포함 — powi(base, exp) (정수 거듭제곱),
 * ilog2() 등을 제공한다. powi()는 이 파일 전반에서 k^(n-1), k^n 등의
 * 계산에 광범위하게 사용된다. */


 //#define FATTREE_DEBUG
/* [한국어] FATTREE_DEBUG 매크로 — 주석 처리된 디버그 출력 토글.
 * 정의 시 _BuildNet()의 각 채널 배선 단계에서 상세 로그를 콘솔에 출력한다.
 * 채널 번호 추적이 필요할 때 주석을 해제한다. */

/*
 * [한국어]
 * FatTree::FatTree - FatTree 네트워크 생성자
 *
 * @param config: gpgpusim.config 파싱 결과 — k(방사), n(레벨 수) 포함
 * @param name:   네트워크 인스턴스 이름 (로그/디버그용 문자열)
 *
 * FatTree 네트워크를 완전히 초기화하는 유일한 진입점이다.
 * 초기화는 세 단계로 이루어진다:
 *   1. _ComputeSize(config): k, n에서 _nodes, _size, _channels 계산
 *   2. _Alloc():             계산된 크기로 _routers[], _chan[], _inject[],
 *                            _eject[], 크레딧 채널 배열 동적 할당
 *   3. _BuildNet(config):    라우터 생성 및 채널 배선 완료
 * 이 생성자가 반환되면 Fat Tree 토폴로지가 시뮬레이션 준비 완료 상태이다.
 * 실행 컨텍스트: 시뮬레이터 초기화 시점의 CPU 단일 스레드.
 *
 * 호출 체인:
 *   icnt_wrapper_init() / Booksim2 main → [FatTree 생성자]
 *     → _ComputeSize() → _Alloc() → _BuildNet()
 */
FatTree::FatTree( const Configuration& config,const string & name )
  : Network( config ,name)
  /* [한국어] Network 기반 클래스 생성자 호출 — config와 name을 전달하여
   * Network 공통 멤버(모듈 이름, 부모 포인터 등)를 초기화한다.
   * 이 시점에서는 아직 _routers, _chan 배열이 할당되지 않았다. */
{


  _ComputeSize( config );
  /* [한국어] 1단계: k, n 파라미터를 읽어 _nodes, _size, _channels를 계산한다.
   * _Alloc()이 이 값들을 사용하므로 반드시 먼저 호출되어야 한다. */

  _Alloc( );
  /* [한국어] 2단계: 계산된 크기로 _routers[], _chan[], _inject[], _eject[],
   * 크레딧 채널(_chan_cred[], _inject_cred[], _eject_cred[]) 배열을 동적 할당한다.
   * Network 기반 클래스에 정의된 메서드이다. */

  _BuildNet( config );
  /* [한국어] 3단계: 라우터를 생성하고 채널을 모두 배선한다.
   * 이 호출이 완료되면 FatTree 토폴로지의 물리적 연결이 확정된다. */

}

/*
 * [한국어]
 * FatTree::_ComputeSize - 네트워크 크기 파라미터 계산
 *
 * @param config: Configuration 객체 — "k", "n" 값을 읽어온다
 * @return: 없음 (부모 클래스의 _nodes, _size, _channels를 직접 설정)
 *
 * gpgpusim.config에서 k(방사)와 n(레벨 수)을 읽어 Fat Tree의 전체
 * 크기를 계산한다. 계산 공식:
 *   _nodes    = k^n           : 단말 노드(inject/eject 포트) 수
 *   _size     = n * k^(n-1)  : 전체 라우터 수 (레벨 수 × 레벨당 라우터 수)
 *   _channels = 2*k*k^(n-1)*(n-1) : 전체 내부 채널 수
 *               (단방향 chan_per_direction=k*k^(n-1),
 *                up+down=2배, 레벨 간 연결=(n-1)개 레벨)
 * 전역 변수 gK, gN도 여기서 설정된다 (다른 Booksim2 모듈에서 참조).
 * 실행 컨텍스트: 생성자 내 단일 스레드 — 동시성 없음.
 *
 * 호출 체인:
 *   FatTree 생성자 → [_ComputeSize] → (종료 후 _Alloc 호출됨)
 */
void FatTree::_ComputeSize( const Configuration& config )
{

  _k = config.GetInt( "k" );
  /* [한국어] gpgpusim.config의 "k" 파라미터를 읽어 _k에 저장.
   * k는 각 라우터의 방사(radix), 즉 출력 포트 수의 절반이다.
   * 예: k=4이면 비최상위 라우터는 4개 down + 4개 up = 8 포트. */

  _n = config.GetInt( "n" );
  /* [한국어] gpgpusim.config의 "n" 파라미터를 읽어 _n에 저장.
   * n은 Fat Tree의 레벨(깊이) 수로, 레벨 0이 최상위(루트에 가장 가까운),
   * 레벨 n-1이 최하위(단말에 가장 가까운)이다. */

  gK = _k; gN = _n;
  /* [한국어] 전역 변수 gK, gN에 _k, _n을 복사.
   * gK/gN은 booksim.hpp에 선언된 전역 변수로, Booksim2의 라우팅 함수,
   * 통계 수집 등 다른 모듈에서 토폴로지 파라미터를 참조할 때 사용한다.
   * FatTree 전용이 아니라 시뮬레이터 전역에서 공유된다. */

  _nodes = powi( _k, _n );
  /* [한국어] 단말 노드 수 = k^n.
   * Fat Tree에서 단말 노드는 레벨 n-1의 각 라우터에 k개씩 붙으므로
   * 전체 단말 수 = k^(n-1) 라우터 × k 포트 = k^n 개.
   * 이 값이 _inject[] / _eject[] 배열의 크기가 된다. */

  //levels * routers_per_level
  _size = _n * powi( _k , _n - 1 );
  /* [한국어] 전체 라우터 수 = 레벨 수(n) × 레벨당 라우터 수(k^(n-1)).
   * 모든 레벨에 동일하게 k^(n-1)개의 라우터가 존재한다.
   * 이 값이 _routers[] 배열의 크기가 된다. */

  //(channels per level = k*routers_per_level* up/down) * (levels-1)
  _channels = (2*_k * powi( _k , _n-1 ))*(_n-1);
  /* [한국어] 전체 내부 채널 수 계산.
   * 레벨 간 연결 수 (n-1)개 레벨 경계 × 레벨당 채널 수.
   * 레벨당 채널 = up 방향 k*k^(n-1) + down 방향 k*k^(n-1) = 2*k*k^(n-1).
   * 레벨 0(최상위)는 up 채널 없고, 레벨 n-1(최하위)는 down 채널이
   * inject/eject 채널로 대체되므로 내부 채널은 (n-1)개 레벨 경계만 해당.
   * 이 값이 _chan[] / _chan_cred[] 배열의 크기가 된다. */


}


/*
 * [한국어]
 * FatTree::RegisterRoutingFunctions - 라우팅 함수 등록 훅 (빈 함수)
 *
 * @return: 없음
 *
 * Booksim2의 토폴로지 초기화 규약에 따라 각 토폴로지는 이 정적 함수에서
 * 전역 라우팅 테이블에 자신의 라우팅 함수를 등록해야 한다. 그러나 FatTree는
 * 현재 전용 라우팅 함수가 구현되어 있지 않으므로 빈 함수(no-op)로 남아 있다.
 * 참고: Mesh, Torus 등의 다른 토폴로지는 여기서 routing_function_t 함수 포인터를
 * gRoutingFunctionMap에 삽입한다.
 * 실행 컨텍스트: 시뮬레이터 시작 시 한 번 호출됨 (단일 스레드).
 *
 * 호출 체인:
 *   Booksim2 초기화 루틴 → [RegisterRoutingFunctions()] → (아무 동작 없음)
 */
void FatTree::RegisterRoutingFunctions() {

}

/*
 * [한국어]
 * FatTree::_BuildNet - Fat Tree 라우터 생성 및 채널 배선
 *
 * @param config: Configuration 객체 — 라우터 타입 등 추가 설정에 사용
 * @return: 없음 (_routers[], _chan[], _inject[], _eject[] 직접 변경)
 *
 * Fat Tree의 물리적 연결을 완전히 구성하는 핵심 함수이다. 다음 순서로 동작:
 *   1. 라우터 생성: 모든 레벨의 모든 위치에 Router 객체 생성.
 *      - 레벨 0 (최상위): degree=k (down 포트만 k개, up 없음)
 *      - 레벨 1~(n-1): degree=2*k (down k개 + up k개)
 *   2. inject/eject 채널 연결: 레벨 n-1 라우터에 단말 채널 연결.
 *   3. down 출력 채널 연결: 레벨 0~(n-2)의 출력 포트(port 0~k-1)에 _chan[link].
 *   4. up 출력 채널 연결: 레벨 1~(n-1)의 출력 포트(port k~2k-1)에 _chan[link].
 *   5. down 입력 채널 연결: 레벨 0~(n-2) 라우터의 입력 포트에 인터리브 번호 체계로 배선.
 *   6. up 입력 채널 연결: 레벨 1~(n-1) 라우터의 입력 포트에 인터리브 번호 체계로 배선.
 * 모든 채널 레이턴시는 1 사이클로 설정된다.
 *
 * 포트 규칙:
 *   출력 포트 0~k-1: DOWN 방향 (하위 레벨로 패킷 전달)
 *   출력 포트 k~2k-1: UP 방향 (상위 레벨로 패킷 전달)
 *   입력 포트 0~k-1: 하위 레벨(DOWN)에서 올라온 패킷 수신
 *   입력 포트 k~2k-1: 상위 레벨(UP)에서 내려온 패킷 수신
 *
 * 실행 컨텍스트: 생성자 내 단일 스레드.
 * 에러 처리: _Router() 내 assert가 깊이/위치 범위를 검사한다.
 *
 * 호출 체인:
 *   FatTree 생성자 → _Alloc() 완료 후 → [_BuildNet()]
 */
void FatTree::_BuildNet( const Configuration& config )
{
 cout << "Fat Tree" << endl;
 /* [한국어] 시뮬레이터 초기화 시 Fat Tree 토폴로지가 선택되었음을 콘솔에 출력.
  * 디버그/로그 목적의 출력이다. */

  cout << " k = " << _k << " levels = " << _n << endl;
  /* [한국어] k(방사)와 n(레벨 수) 파라미터를 콘솔에 출력.
   * 예: k=4, n=3 이면 "k = 4 levels = 3" 출력. */

  cout << " each switch - total radix =  "<< 2*_k << endl;
  /* [한국어] 비최상위 라우터의 총 포트 수(radix) = 2*k 출력.
   * 최상위(레벨 0) 라우터는 k 포트이지만, 일반적인 경우로 2*k를 표시한다. */

  cout << " # of switches = "<<  _size << endl;
  /* [한국어] 전체 라우터(스위치) 수 = n * k^(n-1) 출력. */

  cout << " # of channels = "<<  _channels << endl;
  /* [한국어] 전체 내부 채널 수 = 2*k*k^(n-1)*(n-1) 출력. */

  cout << " # of nodes ( size of network ) = " << _nodes << endl;
  /* [한국어] 단말 노드 수 = k^n 출력. GPU의 경우 SM과 메모리 컨트롤러의 합이다. */


  // Number of router positions at each depth of the network
  const int nPos = powi( _k, _n-1);
  /* [한국어] 레벨당 라우터 수 = k^(n-1).
   * 모든 레벨에 동일하게 nPos개의 라우터가 존재한다.
   * 레벨 루프(level 0~n-1)의 내부 루프 상한값으로 사용된다. */

  //
  // Allocate Routers
  //
  ostringstream name;
  /* [한국어] 라우터 이름 생성을 위한 문자열 스트림.
   * 루프마다 name.str("")으로 초기화하고 "router_level<N>_<P>" 형식으로 재사용한다. */

  int level, pos, id, degree, port;
  /* [한국어] 루프 변수 선언:
   * - level: 현재 레벨 (0=최상위 ~ n-1=최하위)
   * - pos:   현재 레벨 내 라우터 위치 (0 ~ nPos-1)
   * - id:    _routers[] 배열 내 선형 인덱스
   * - degree: 라우터의 입/출력 포트 수
   * - port:  내부 포트 루프 변수 */

  for ( level = 0 ; level < _n ; ++level ) {
  /* [한국어] 레벨 0(최상위)부터 레벨 n-1(최하위)까지 순회하며 라우터를 생성한다. */

    for ( pos = 0 ; pos < nPos ; ++pos ) {
    /* [한국어] 각 레벨의 모든 위치(0 ~ nPos-1)를 순회한다. */

      if ( level == 0 ) //top routers is zero
	degree = _k;
      /* [한국어] 최상위 레벨(level 0) 라우터는 down 포트만 k개를 가진다.
       * 레벨 0 위의 상위 레벨이 없으므로 up 포트가 없어 degree=k. */
      else
	degree = 2 * _k;
      /* [한국어] 나머지 레벨(level 1~n-1) 라우터는 down k개 + up k개 = 2k 포트.
       * AddInputChannel/AddOutputChannel을 각각 degree번 호출하게 된다. */

      id = level * nPos + pos;
      /* [한국어] 라우터의 선형 ID = 레벨 * 레벨당_라우터수 + 위치.
       * _Router(level, pos) 계산식과 동일하다. 라우터 이름과 통계 식별에 사용. */

      name.str("");
      /* [한국어] 이전 루프에서 쌓인 문자열 스트림을 초기화한다.
       * str("")은 내부 버퍼를 빈 문자열로 교체한다. */

      name << "router_level" << level << "_" << pos;
      /* [한국어] 라우터 이름 생성: "router_level<level>_<pos>" 형식.
       * 예: level=1, pos=3 → "router_level1_3".
       * Router 내부에서 디버그 출력과 통계 레이블에 사용된다. */

      Router * r = Router::NewRouter( config, this, name.str( ), id,
				      degree, degree );
      /* [한국어] 팩토리 함수 Router::NewRouter()로 라우터 객체를 생성한다.
       * - config: 버퍼 크기, VC 수 등 라우터 내부 파라미터
       * - this:   소유 네트워크 포인터 (부모 참조)
       * - name.str(): 이름 문자열
       * - id:     선형 ID
       * - degree, degree: 입력 포트 수, 출력 포트 수 (둘 다 동일) */

      _Router( level, pos ) = r;
      /* [한국어] 생성된 라우터 포인터를 _Router() 접근자로 _routers[] 배열에 저장.
       * _Router(level, pos)는 참조를 반환하므로 대입으로 배열 원소를 설정한다. */

      _timed_modules.push_back(r);
      /* [한국어] 라우터를 시뮬레이터의 timed module 목록에 등록한다.
       * Network::Step() 호출 시 _timed_modules의 모든 모듈에 대해 Step()이
       * 순서대로 호출되어 1 사이클 진행이 이루어진다. */
    }
  }

  //
  // Connect Channels to Routers
  //

  //
  // Router Connection Rule: Output Ports <gK Move DOWN Network
  //                         Output Ports >=gK Move UP Network
  //                         Input Ports <gK from DOWN Network
  //                         Input Ports >=gK  from up Network

  // Connecting  Injection & Ejection Channels
  for ( pos = 0 ; pos < nPos ; ++pos ) {
  /* [한국어] 레벨 n-1(최하위) 라우터의 모든 위치를 순회하며 inject/eject 채널 연결.
   * inject: 단말 노드 → 라우터 (패킷 주입), eject: 라우터 → 단말 노드 (패킷 배출).
   * _inject[]/_eject[] 배열은 _nodes=k^n 크기이며, 위치 pos의 라우터에는
   * 인덱스 pos*k ~ pos*k+k-1 까지 k개의 채널이 붙는다. */

    for(int index = 0; index<_k; index++){
    /* [한국어] 각 최하위 라우터에는 k개의 inject/eject 포트가 있으므로 k번 반복. */

      int link = pos*_k + index;
      /* [한국어] inject/eject 채널 인덱스 = pos*k + index (0 ~ k^n-1).
       * pos번째 라우터의 index번째 포트가 담당하는 단말 채널 번호. */

      _Router( _n-1, pos)->AddInputChannel( _inject[link],
					    _inject_cred[link]);
      /* [한국어] 최하위 라우터(level n-1, pos)의 입력 포트에 inject 채널 연결.
       * 패킷이 단말(SM 또는 메모리 컨트롤러)에서 주입될 때 이 채널을 통해 입력된다.
       * _inject_cred[link]는 흐름 제어를 위한 크레딧 채널이다. */

      _Router( _n-1, pos)->AddOutputChannel( _eject[link],
					     _eject_cred[link]);
      /* [한국어] 최하위 라우터(level n-1, pos)의 출력 포트에 eject 채널 연결.
       * 패킷이 목적지 단말로 배출될 때 이 채널을 통해 출력된다. */

      _inject[link]->SetLatency( 1 );
      /* [한국어] inject 채널의 전파 레이턴시를 1 사이클로 설정.
       * 단말→라우터 구간이 1 사이클 지연됨을 의미한다. */

      _inject_cred[link]->SetLatency( 1 );
      /* [한국어] inject 크레딧 채널의 레이턴시를 1 사이클로 설정.
       * 흐름 제어 크레딧이 라우터→단말로 1 사이클 후 전달된다. */

      _eject[link]->SetLatency( 1 );
      /* [한국어] eject 채널의 전파 레이턴시를 1 사이클로 설정.
       * 라우터→단말 구간이 1 사이클 지연됨을 의미한다. */

      _eject_cred[link]->SetLatency( 1 );
      /* [한국어] eject 크레딧 채널의 레이턴시를 1 사이클로 설정. */
    }
  }

#ifdef FATTREE_DEBUG
  cout<<"\nAssigning output\n";
  /* [한국어] FATTREE_DEBUG 정의 시 출력 채널 배선 시작을 알리는 디버그 메시지. */
#endif

  //channels are numbered sequentially from an output channel perspective
  int chan_per_direction = (_k * powi( _k , _n-1 )); //up or down
  /* [한국어] 단방향(up 또는 down) 채널 수 = k * k^(n-1) = k^n.
   * 레벨당 라우터 수(k^(n-1)) × 라우터당 해당 방향 포트 수(k).
   * 이 값은 채널 인덱스 계산의 기준 단위로 아래 모든 배선 루프에서 사용된다. */

  int chan_per_level = 2*(_k * powi( _k , _n-1 )); //up+down
  /* [한국어] 레벨당 전체 채널 수(up+down) = 2 * chan_per_direction.
   * 각 레벨 경계에는 up 방향 chan_per_direction개와 down 방향
   * chan_per_direction개 채널이 있으므로 합계 2*chan_per_direction.
   * level*chan_per_level 을 채널 번호의 레벨 오프셋으로 사용한다. */

  //connect all down output channels
  //level n-1's down channel are injection channels
  for (level = 0; level<_n-1; level++){
  /* [한국어] 레벨 0~(n-2)의 down 출력 채널을 연결한다.
   * 레벨 n-1(최하위)의 down 채널은 inject 채널로 이미 연결했으므로 제외.
   * down 출력: 상위 라우터에서 하위 라우터로 패킷이 내려가는 방향. */

    for ( pos = 0; pos < nPos; ++pos ) {
    /* [한국어] 현재 레벨의 모든 위치를 순회한다. */

      for ( port = 0; port < _k; ++port ) {
      /* [한국어] 각 라우터의 down 출력 포트 0~k-1을 순회한다. */

	int link = (level*chan_per_level) + pos*_k + port;
	/* [한국어] down 출력 채널 번호 계산:
	 *   level*chan_per_level: 현재 레벨까지의 채널 블록 오프셋
	 *   pos*_k + port:        현재 레벨 내 위치와 포트로 결정되는 채널 번호
	 * down 채널은 각 레벨 블록의 앞쪽 chan_per_direction개를 차지한다. */

	_Router(level, pos)->AddOutputChannel( _chan[link],
						_chan_cred[link] );
	/* [한국어] 라우터 (level, pos)의 출력 포트에 _chan[link] 채널 연결.
	 * AddOutputChannel() 호출 순서가 포트 번호를 결정한다 (0번부터 순차). */

	_chan[link]->SetLatency( 1 );
	/* [한국어] down 채널 레이턴시 1 사이클 설정. */

	_chan_cred[link]->SetLatency( 1 );
	/* [한국어] down 채널 크레딧 레이턴시 1 사이클 설정. */

#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"down output "<<port<<" "
	    <<"channel_id "<<link<<endl;
	/* [한국어] 디버그: 라우터 이름, "down output", 포트 번호, 채널 ID 출력.
	 * 채널 배선이 의도대로 이루어졌는지 검증할 때 FATTREE_DEBUG를 정의한다. */
#endif

      }
    }
  }

  //connect all up output channels
  //level 0 has no up chnanels
  for (level = 1; level<_n; level++){
  /* [한국어] 레벨 1~(n-1)의 up 출력 채널을 연결한다.
   * 레벨 0(최상위)는 상위 레벨이 없으므로 up 채널 없음 → level=1부터 시작.
   * up 출력: 하위 라우터에서 상위 라우터로 패킷이 올라가는 방향. */

    for ( pos = 0; pos < nPos; ++pos ) {
    /* [한국어] 현재 레벨의 모든 위치를 순회한다. */

      for ( port = 0; port < _k; ++port ) {
      /* [한국어] 각 라우터의 up 출력 포트 0~k-1을 순회한다.
       * 라우터 내부에서는 실제로 포트 k~2k-1번에 매핑되지만,
       * AddOutputChannel() 호출 순서로 포트 번호가 결정되므로
       * down 출력이 이미 0~k-1에 등록된 후 이 루프가 k~2k-1을 채운다. */

	int link = (level*chan_per_level - chan_per_direction) + pos*_k + port ;
	/* [한국어] up 출력 채널 번호 계산:
	 *   level*chan_per_level - chan_per_direction:
	 *     레벨 경계 (level-1) ↔ level 사이의 채널 블록 중 up 방향 오프셋.
	 *     up 채널은 해당 레벨 경계 블록의 뒤쪽 chan_per_direction개를 차지.
	 *     수식 유도: level번째 경계의 down 채널 시작 = (level-1)*chan_per_level,
	 *               up 채널 시작 = down 시작 + chan_per_direction
	 *                            = (level-1)*chan_per_level + chan_per_direction
	 *                            = level*chan_per_level - chan_per_direction
	 *   pos*_k + port: 위치·포트 오프셋 */

	_Router(level, pos)->AddOutputChannel( _chan[link],
						_chan_cred[link] );
	/* [한국어] 라우터 (level, pos)의 up 출력 포트에 _chan[link] 채널 연결. */

	_chan[link]->SetLatency( 1 );
	/* [한국어] up 채널 레이턴시 1 사이클 설정. */

	_chan_cred[link]->SetLatency( 1 );
	/* [한국어] up 채널 크레딧 레이턴시 1 사이클 설정. */

#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"up output "<<port<<" "
	    <<"channel_id "<<link<<endl;
	/* [한국어] 디버그: 라우터 이름, "up output", 포트 번호, 채널 ID 출력. */
#endif
      }
    }
  }

#ifdef FATTREE_DEBUG
  cout<<"\nAssigning Input\n";
  /* [한국어] 디버그: 입력 채널 배선 시작 알림. */
#endif

  //connect all down input channels
  for (level = 0; level<_n-1; level++){
  /* [한국어] 레벨 0~(n-2) 라우터의 down 입력 채널을 연결한다.
   * "down 입력"은 하위 레벨(level+1)에서 상위 레벨(level)로 올라오는
   * up 출력 채널의 반대편 수신단이다.
   * 즉, 이 루프는 level 라우터의 입력에 level+1 라우터의 up 출력을 연결한다.
   * 채널 번호는 인터리브 방식으로 계산된다(아래 설명 참조). */

    //input channel are numbered interleavely, the interleaev depends on level
    int routers_per_neighborhood = powi(_k,_n-1-(level));
    /* [한국어] 이웃(neighborhood) 크기 = k^(n-1-level).
     * level이 높을수록(상위 레벨) 한 이웃이 커진다.
     * 이웃이란 같은 상위 라우터(level-1)에 연결된 라우터들의 집합이다. */

    int routers_per_branch = powi(_k,_n-1-(level+1));
    /* [한국어] 브랜치 크기 = k^(n-2-level).
     * level+1(하위 레벨)의 이웃 내 한 브랜치에 속한 라우터 수.
     * 인터리브 채널 번호 계산에서 서브-이웃 내 포트 매핑에 사용된다. */

    int level_offset = routers_per_neighborhood*_k;
    /* [한국어] 레벨 내 이웃 단위 오프셋 = routers_per_neighborhood * k.
     * 하나의 이웃 영역이 차지하는 채널 수로, 이웃 간 채널 블록을 구분한다. */

    for ( pos = 0; pos < nPos; ++pos ) {
    /* [한국어] level 레벨의 모든 위치(0~nPos-1)를 순회한다. */

      int neighborhood = pos/routers_per_neighborhood;
      /* [한국어] 현재 라우터 pos가 속한 이웃 번호.
       * 이웃은 routers_per_neighborhood 단위로 구분된다.
       * 예: nPos=8, routers_per_neighborhood=4 이면 pos 0~3 → neighborhood 0, 4~7 → 1. */

      int neighborhood_pos = pos%routers_per_neighborhood;
      /* [한국어] 이웃 내에서 현재 라우터의 상대 위치.
       * 0 ~ routers_per_neighborhood-1 범위의 값을 가진다. */

      for ( port = 0; port < _k; ++port ) {
      /* [한국어] 각 라우터의 down 입력 포트 0~k-1을 순회한다.
       * down 입력은 하위 레벨에서 올라오는 up 출력 채널을 수신한다. */

	int link =
	  ((level+1)*chan_per_level - chan_per_direction)  //which levellevel
	  /* [한국어] (level+1)*chan_per_level - chan_per_direction:
	   * 레벨 경계 level↔(level+1)의 up 출력 채널 블록 시작 오프셋.
	   * up 출력 채널 번호 계산과 동일한 기준으로, 같은 채널의 반대쪽 끝을 연결한다. */

	  +neighborhood*level_offset   //region in level
	  /* [한국어] 이웃 번호 × level_offset: 이웃 단위 채널 블록 선택. */

	  +port*routers_per_branch*gK  //sub region in region
	  /* [한국어] port × routers_per_branch × k: 포트에 대응하는 서브 영역 시작.
	   * 각 포트는 routers_per_branch*k개의 채널 서브 블록을 차지한다. */

	  +(neighborhood_pos)%routers_per_branch*gK  //router in subregion
	  /* [한국어] (neighborhood_pos % routers_per_branch) × k:
	   * 서브 영역 내 라우터 위치에 대응하는 오프셋. */

	  +(neighborhood_pos)/routers_per_branch; //port on router
	  /* [한국어] neighborhood_pos / routers_per_branch: 서브 영역 내 포트 번호.
	   * 최종 link 값이 up 출력 채널 배선에서 같은 채널을 가리키도록 설계된다. */

	_Router(level, pos)->AddInputChannel( _chan[link],
					      _chan_cred[link] );
	/* [한국어] 라우터 (level, pos)의 down 입력 포트에 _chan[link]를 연결.
	 * 이 채널은 앞서 level+1 라우터의 up 출력으로 이미 연결된 채널과 동일하다.
	 * 즉 down 입력 배선이 up 출력 배선의 수신단을 완성한다. */

#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"down input "<<port<<" "
	    <<"channel_id "<<link<<endl;
	/* [한국어] 디버그: 라우터 이름, "down input", 포트 번호, 채널 ID 출력. */
#endif
      }
    }
  }


 //connect all up input channels
  for (level = 1; level<_n; level++){
  /* [한국어] 레벨 1~(n-1) 라우터의 up 입력 채널을 연결한다.
   * "up 입력"은 상위 레벨(level-1)에서 내려오는 down 출력 채널의 수신단이다.
   * 즉, level 라우터의 up 입력 포트(k~2k-1)에 level-1 라우터의 down 출력을 연결한다.
   * level=0(최상위)는 상위가 없으므로 level=1부터 시작한다. */

    //input channel are numbered interleavely, the interleaev depends on level
    int routers_per_neighborhood = powi(_k,_n-1-(level-1));
    /* [한국어] 이웃 크기 = k^(n-level).
     * 상위 레벨(level-1) 기준의 이웃 크기 — down 입력 계산의 routers_per_neighborhood와
     * level을 하나 올린 값이다. */

    int routers_per_branch = powi(_k,_n-1-(level));
    /* [한국어] 브랜치 크기 = k^(n-1-level).
     * level 기준의 이웃 내 브랜치 크기. */

    int level_offset = routers_per_neighborhood*_k;
    /* [한국어] 이웃 단위 채널 블록 크기 = routers_per_neighborhood * k. */

    for ( pos = 0; pos < nPos; ++pos ) {
    /* [한국어] level 레벨의 모든 위치를 순회한다. */

      int neighborhood = pos/routers_per_neighborhood;
      /* [한국어] 현재 라우터가 속한 이웃 번호. */

      int neighborhood_pos = pos%routers_per_neighborhood;
      /* [한국어] 이웃 내 상대 위치. */

      for ( port = 0; port < _k; ++port ) {
      /* [한국어] 각 라우터의 up 입력 포트 0~k-1을 순회한다.
       * (라우터 내부에서는 포트 번호 k~2k-1로 등록되지만 AddInputChannel
       * 호출 순서로 결정된다 — down 입력이 먼저 0~k-1을 채웠으므로
       * up 입력은 k~2k-1에 해당한다.) */

	int link =
	  ((level-1)*chan_per_level) //which levellevel
	  /* [한국어] (level-1)*chan_per_level: 레벨 경계 (level-1)↔level의
	   * down 출력 채널 블록 시작 오프셋. down 출력 배선과 동일한 기준. */

	  +neighborhood*level_offset   //region in level
	  /* [한국어] 이웃 번호 × level_offset: 이웃 단위 블록 선택. */

	  +port*routers_per_branch*gK  //sub region in region
	  /* [한국어] port × routers_per_branch × k: 포트에 대응하는 서브 영역 시작. */

	  +(neighborhood_pos)%routers_per_branch*gK //router in subregion
	  /* [한국어] 서브 영역 내 라우터 위치 오프셋. */

	  +(neighborhood_pos)/routers_per_branch; //port on router
	  /* [한국어] 서브 영역 내 포트 번호.
	   * 최종 link 값이 level-1 라우터의 down 출력 채널과 동일한 채널을 가리킨다. */

	_Router(level, pos)->AddInputChannel( _chan[link],
					      _chan_cred[link] );
	/* [한국어] 라우터 (level, pos)의 up 입력 포트에 _chan[link]를 연결.
	 * 이 채널은 레벨 level-1 라우터의 down 출력으로 이미 등록된 채널이다.
	 * up 입력 배선이 down 출력 배선의 수신단을 완성하여 양방향 경로를 구성한다. */

#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"up input "<<port<<" "
	    <<"channel_id "<<link<<endl;
	/* [한국어] 디버그: 라우터 이름, "up input", 포트 번호, 채널 ID 출력. */
#endif
      }
    }
  }
#ifdef FATTREE_DEBUG
  cout<<"\nChannel assigned\n";
  /* [한국어] 디버그: 모든 채널 배선 완료 알림 메시지. */
#endif
}

/*
 * [한국어]
 * FatTree::_Router - (레벨, 위치) 쌍으로 _routers[] 배열에 접근하는 인덱서
 *
 * @param depth: 레벨 번호 (0=최상위 ~ _n-1=최하위); [0, _n) 범위이어야 함
 * @param pos:   레벨 내 라우터 위치 (0 ~ k^(n-1)-1); [0, k^(n-1)) 범위이어야 함
 * @return: _routers[depth * k^(n-1) + pos]에 대한 Router*& 참조.
 *          참조이므로 읽기(_Router(l,p)->AddChannel())와 쓰기(_Router(l,p) = r) 모두 가능.
 *
 * Fat Tree의 라우터는 2차원 (레벨, 위치)으로 논리적으로 배치되지만 _routers[]는
 * 1차원 배열이므로, 이 함수가 인덱스 변환을 캡슐화한다.
 * 공식: index = depth * k^(n-1) + pos
 * assert로 범위를 검사하여 잘못된 인덱스 접근을 즉시 감지한다.
 * 실행 컨텍스트: _BuildNet() 내 단일 스레드 — 동시성 없음.
 *
 * 호출 체인:
 *   _BuildNet() → [_Router(level, pos)] → _routers[index] 참조 반환
 */
Router*& FatTree::_Router( int depth, int pos )
{
  assert( depth < _n && pos < powi( _k, _n-1) );
  /* [한국어] 범위 검사 assert:
   * - depth < _n: 레벨 번호가 유효한 범위(0~n-1) 내에 있는지 확인.
   * - pos < k^(n-1): 위치 번호가 레벨당 라우터 수 이내인지 확인.
   * 위반 시 즉시 프로그램을 중단하여 버그를 조기에 발견한다. */

  return _routers[depth * powi( _k, _n-1) + pos];
  /* [한국어] 2차원 좌표 (depth, pos)를 1차원 배열 인덱스로 변환하여 참조를 반환한다.
   * 공식: index = depth * k^(n-1) + pos
   * - depth * k^(n-1): 레벨 오프셋 (각 레벨에 k^(n-1)개 라우터가 있으므로)
   * - pos: 해당 레벨 내 위치
   * 반환값이 참조(Router*&)이므로 호출자가 이 원소에 읽기/쓰기 모두 가능하다. */
}
