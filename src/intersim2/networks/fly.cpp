// $Id: fly.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] KNFly — k-ary n-fly 버터플라이 네트워크 구현 (fly.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 fly.hpp에 선언된 KNFly 클래스의 모든 멤버 함수를 구현한다.
 * KNFly는 k-ary n-fly(버터플라이) 토폴로지를 표현하며, n개 스테이지에 k^(n-1)개 라우터를
 * 배치하고 인접 스테이지를 "자릿수 교환(digit-swap)" 배선 규칙으로 연결한다.
 * 핵심 책임은 두 가지이다:
 *   (1) 네트워크 규모 계산(_ComputeSize): 터미널·라우터·채널 수 도출
 *   (2) 토폴로지 구축(_BuildNet): 라우터 생성 및 주입/이젝트/내부 채널 연결
 * 구성이 완료된 후에는 Network 기반 클래스의 Evaluate() 루프에서 사이클마다 동작한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이션 계층에서 이 파일은 intersim2 NoC 시뮬레이터 계층에 속한다.
 * 실행 컨텍스트: 호스트 유저스페이스 (CPU 스레드).
 * 초기화 경로:
 *   gpgpusim_entrypoint → icnt_wrapper_init() → intersim 선택 시
 *   → IcntWrapper → Network::NewNetwork("fly") → KNFly 생성자
 *       → _ComputeSize()  : 토폴로지 파라미터 계산
 *       → _Alloc()        : (부모) 라우터·채널 배열 동적 할당
 *       → _BuildNet()     : 라우터 인스턴스 생성 + 채널 배선
 * 시뮬레이션 루프:
 *   gpu-sim::cycle() → icnt_push/pop → Network::Evaluate() → 각 라우터 Evaluate()
 *
 * === 타 모듈과의 연결 ===
 * - network.hpp (Network 기반 클래스): _size, _nodes, _channels, _routers, _inject,
 *   _eject, _chan, _chan_cred, _timed_modules 등 핵심 자료구조를 상속.
 *   _Alloc()로 배열을 할당하고, ReadInputs/Evaluate/WriteOutputs로 사이클 단계 실행.
 * - router.hpp (Router 추상 클래스): Router::NewRouter()로 각 스테이지의 k×k 라우터를 생성.
 *   생성된 라우터 포인터는 _routers[node]와 _timed_modules에 저장.
 * - misc_utils.hpp: powi(base, exp) — k^n, k^(n-1) 등의 정수 거듭제곱 계산.
 * - booksim.hpp: Booksim 공통 타입·매크로 포함.
 * - globals.hpp (간접): gK/gN 전역 변수를 통해 라우팅 함수들이 k/n 값을 읽는다.
 * - Configuration: config.GetInt("k")/config.GetInt("n")으로 토폴로지 파라미터 입력.
 * - FlitChannel / CreditChannel: _chan[c]/_chan_cred[c]를 라우터 포트에 연결.
 *   채널 레이턴시는 _InChannel 경로에서 SetLatency(1)로 1 사이클로 설정.
 *
 * === 주요 함수/구조체 요약 ===
 * - KNFly(): 생성자. _ComputeSize→_Alloc→_BuildNet 3단계로 네트워크 완성.
 * - _ComputeSize(): k/n 설정 읽기; _nodes=k^n, _size=n*k^(n-1), _channels=(n-1)*k^n 계산.
 * - _BuildNet(): n*k^(n-1)개 라우터를 스테이지 순서로 생성; 스테이지별 채널 종류(inject/내부/eject) 결정.
 * - _OutChannel(stage, addr, port): 출력 채널 인덱스 = stage*_nodes + addr*_k + port.
 * - _InChannel(stage, addr, port): 버터플라이 자릿수 교환 규칙으로 입력 채널 인덱스 계산.
 * - GetN() / GetK(): 외부에서 _n/_k 조회용 접근자.
 * - Capacity(): 이분 대역폭 비율 1.0 반환.
 */

#include "booksim.hpp"
// [한국어] Booksim 공통 타입·매크로·전역 설정 포함 — assert, 디버그 매크로 등이 정의되어 있음
#include <vector>
// [한국어] std::vector 포함 — 동적 크기 배열 자료구조; 라우터 이름 등 임시 컨테이너에 사용됨
#include <sstream>
// [한국어] std::ostringstream 포함 — router_name 문자열을 동적으로 조합하기 위해 사용

#include "fly.hpp"
// [한국어] KNFly 클래스 선언 포함 — _k, _n, _ComputeSize, _BuildNet 등 이 파일이 구현할 인터페이스
#include "misc_utils.hpp"
// [한국어] powi(x, y) 정수 거듭제곱 함수 포함 — k^n, k^(n-1) 등 버터플라이 크기 계산에 필수

//#define DEBUG_FLY
// [한국어] 디버그 출력 활성화 매크로 (현재 비활성) — 정의 시 _BuildNet()에서 채널 연결 정보를 콘솔에 출력.
//         개발 중 배선이 올바른지 검증할 때 주석을 해제하여 사용한다.

/*
 * [한국어]
 * KNFly::KNFly — k-ary n-fly 버터플라이 네트워크 생성자
 *
 * @config: Booksim 설정 객체 — "k"/"n" 파라미터 및 라우터 내부 설정 포함.
 * @name:   네트워크 객체 이름 문자열 — TimedModule 식별자로 사용.
 * @return: (생성자, 반환값 없음)
 *
 * Network 기반 클래스 생성자를 통해 _size/_nodes/_channels를 -1로 초기화한 뒤,
 * 아래 3단계를 순서대로 호출하여 완전한 네트워크를 구성한다:
 *   1. _ComputeSize(config): 설정에서 k/n을 읽어 토폴로지 규모(_nodes, _size, _channels)를 확정.
 *   2. _Alloc():             확정된 규모에 따라 _routers, _chan, _inject, _eject 등 배열을 동적 할당.
 *   3. _BuildNet(config):    각 스테이지의 라우터를 생성하고 올바른 채널에 연결.
 * 생성자 완료 후 네트워크는 사이클-레벨 Evaluate() 루프를 수신할 준비 상태가 된다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 시 단일 스레드에서 1회 호출된다.
 *
 * 호출 체인:
 *   Network::NewNetwork("fly") → [KNFly 생성자] → _ComputeSize() → _Alloc() → _BuildNet()
 */
KNFly::KNFly( const Configuration &config, const string & name ) :
Network( config, name )
// [한국어] Network 기반 클래스 생성자 호출 — config와 name을 전달하여 TimedModule 등록 및
//         _size/_nodes/_channels를 초기값(-1)으로 초기화한다.
{
  _ComputeSize( config ); // [한국어] 토폴로지 파라미터(k, n) 읽기 및 _nodes/_size/_channels 계산
  _Alloc( );              // [한국어] 계산된 크기에 따라 _routers/_chan/_inject/_eject 배열 동적 할당
  _BuildNet( config );    // [한국어] 라우터 인스턴스 생성 및 채널 연결 — 실제 토폴로지 배선 완성
}

/*
 * [한국어]
 * KNFly::_ComputeSize — 버터플라이 토폴로지의 규모 계산
 *
 * @config: Booksim 설정 객체 — "k"(기수)와 "n"(스테이지 수)를 읽는다.
 * @return: 없음 (부모의 _nodes/_size/_channels/_k/_n을 직접 설정)
 *
 * 버터플라이 네트워크의 수학적 구조에 따라 규모를 도출한다:
 *   - _nodes    = k^n:          총 터미널(단말) 노드 수. 주입/이젝트 채널 배열의 크기와 동일.
 *   - _size     = n * k^(n-1): 총 라우터 수. n개 스테이지 각각에 k^(n-1)개 라우터 배치.
 *   - _channels = (n-1) * k^n: 스테이지 간 내부 채널 수. (n-1)개 스테이지 경계 × 경계당 k^n개.
 * gK, gN 전역 변수도 동시에 설정하여 라우팅 함수들이 파라미터를 별도 인자 없이 참조할 수 있게 한다.
 *
 * 실행 컨텍스트: KNFly 생성자 초기화 시 1회 호출. 단일 스레드.
 *
 * 호출 체인:
 *   KNFly 생성자 → [_ComputeSize] → _Alloc()
 */
void KNFly::_ComputeSize( const Configuration &config )
{
  _k = config.GetInt( "k" ); // [한국어] 기수(radix) 읽기 — 라우터당 입출력 포트 수이자 주소 자릿수의 밑.
                              //         예: k=2이면 이진 버터플라이(나비형), k=4이면 쿼터너리 버터플라이.
  _n = config.GetInt( "n" ); // [한국어] 스테이지 수(차원 수) 읽기 — 네트워크 지름 결정.
                              //         패킷이 목적지에 도달하기 위한 최단 홉 수 = n-1.

  gK = _k; gN = _n;
  // [한국어] 전역 변수 gK/gN에 k/n을 저장 — 라우팅 함수들이 토폴로지 파라미터를
  //         인스턴스 참조 없이 전역으로 접근할 수 있도록 설정하는 Booksim 관례.

  _nodes = powi( _k, _n );
  // [한국어] 터미널 노드 수 = k^n 계산 — 버터플라이의 단말 수는 k를 n번 곱한 값.
  //         예: k=4, n=3이면 _nodes=64 (64개 단말 노드).
  //         _inject 배열과 _eject 배열의 크기가 _nodes로 결정된다.

  // n stages of k^(n-1) k x k switches
  _size = _n*powi( _k, _n-1 );
  // [한국어] 총 라우터 수 = n * k^(n-1) 계산 — 각 스테이지에 k^(n-1)개의 k×k 스위치가 배치됨.
  //         예: k=4, n=3이면 _size = 3*4^2 = 48 (48개 라우터).
  //         _routers 배열의 크기와 동일하다.

  // n-1 sets of wiring between the stages
  _channels = (_n-1)*_nodes;
  // [한국어] 내부 채널 수 = (n-1) * k^n 계산 — 인접 스테이지 쌍의 수는 n-1개이며,
  //         각 경계마다 k^n (=_nodes)개의 채널이 필요하다.
  //         예: k=4, n=3이면 _channels = 2*64 = 128 (128개 내부 채널).
  //         _chan, _chan_cred 배열의 크기와 동일하다.
}

/*
 * [한국어]
 * KNFly::_BuildNet — 버터플라이 네트워크의 라우터를 생성하고 채널을 연결
 *
 * @config: Booksim 설정 객체 — Router::NewRouter()에 전달되어 IQRouter의 VC 수·버퍼 크기 등을 설정.
 * @return: 없음
 *
 * 스테이지(0 ~ n-1)와 스테이지 내 주소(0 ~ k^(n-1)-1)를 이중 루프로 순회하며
 * 총 _size개의 라우터를 생성하고 각 포트에 채널을 연결한다. 연결 규칙:
 *   - stage == 0 (첫 스테이지):   입력 포트에 _inject[addr*k+port] 주입 채널 연결.
 *   - stage == n-1 (마지막 스테이지): 출력 포트에 _eject[addr*k+port] 이젝트 채널 연결.
 *   - 그 외 스테이지:
 *       입력 포트: _InChannel()으로 계산한 _chan[c] 연결 + 레이턴시 1 사이클 설정.
 *       출력 포트: _OutChannel()으로 계산한 _chan[c] 연결.
 * 생성된 각 라우터는 _timed_modules에도 등록되어 매 사이클 Evaluate() 호출 대상이 된다.
 *
 * 실행 컨텍스트: KNFly 생성자 초기화 시 1회 호출. 단일 스레드.
 *
 * 호출 체인:
 *   KNFly 생성자 → _ComputeSize() → _Alloc() → [_BuildNet]
 *   [_BuildNet] → Router::NewRouter() (각 라우터 생성)
 *              → _routers[node]->AddInputChannel() (입력 포트 채널 연결)
 *              → _routers[node]->AddOutputChannel() (출력 포트 채널 연결)
 *              → _InChannel() / _OutChannel() (채널 인덱스 계산)
 */
void KNFly::_BuildNet( const Configuration &config )
{
  ostringstream router_name;
  // [한국어] 라우터 이름을 동적으로 생성하기 위한 문자열 스트림 — "router_<stage>_<addr>" 형식으로 조합됨.
  //         각 라우터는 고유 이름으로 디버그 출력 및 통계 식별에 사용된다.

  int per_stage = powi( _k, _n-1 );
  // [한국어] 스테이지당 라우터 수 = k^(n-1) 계산 — 각 스테이지에 배치되는 k×k 라우터의 수.
  //         예: k=4, n=3이면 per_stage=16 (각 스테이지에 16개 라우터).
  //         addr 루프의 상한값으로 사용된다.

  int node = 0;
  // [한국어] 전체 라우터 배열(_routers)에서의 선형 인덱스 — 스테이지 순서로 0부터 _size-1까지 증가.
  //         node = stage * per_stage + addr (이중 루프를 통해 순차 증가).

  int c;
  // [한국어] 채널 인덱스 임시 변수 — _inject/_eject/_chan 배열 중 어느 채널에 연결할지 저장.
  //         각 포트 처리마다 _InChannel()/_OutChannel() 또는 직접 계산으로 값이 갱신된다.

  for ( int stage = 0; stage < _n; ++stage ) {
    // [한국어] 스테이지 루프 (0 ~ n-1) — 버터플라이의 각 열(column)에 해당하는 스테이지를 순회.
    //         stage=0은 주입 채널 연결, stage=n-1은 이젝트 채널 연결, 나머지는 내부 채널 연결.
    for ( int addr = 0; addr < per_stage; ++addr ) {
      // [한국어] 스테이지 내 라우터 주소 루프 (0 ~ per_stage-1) — 각 스테이지의 라우터를 순서대로 처리.
      //         addr은 k진법 n-1자리 주소로 해석되며, 이것이 버터플라이 배선 계산의 기반이 된다.

      router_name << "router_" << stage << "_" << addr;
      // [한국어] 라우터 이름 조합 — "router_<stage>_<addr>" 형식.
      //         예: stage=1, addr=3이면 "router_1_3". 디버그·통계에서 라우터를 식별하는 레이블.

      _routers[node] = Router::NewRouter( config, this, router_name.str( ),
					  node, _k, _k );
      // [한국어] Router::NewRouter()로 k×k IQRouter 인스턴스 생성 후 _routers[node]에 저장.
      //         인자 순서: (설정, 부모 네트워크, 이름, 라우터 번호, 입력 포트 수=k, 출력 포트 수=k).
      //         내부적으로 IQRouter 또는 설정에 따른 다른 라우터 서브클래스가 생성됨.

      _timed_modules.push_back(_routers[node]);
      // [한국어] 생성된 라우터를 _timed_modules 큐에 추가 — 매 사이클 Network::Evaluate()에서
      //         이 큐를 순회하며 각 모듈의 ReadInputs/Evaluate/WriteOutputs를 호출한다.
      //         채널 모듈들은 _Alloc()에서 이미 등록되어 있으므로, 라우터를 여기서 추가.

      router_name.str("");
      // [한국어] 이름 스트림 초기화 — 다음 라우터 이름 생성을 위해 스트림 내용을 비움.
      //         초기화하지 않으면 이전 이름이 누적되어 잘못된 이름이 생성됨.

#ifdef DEBUG_FLY
      cout << "connecting node " << node << " to:" << endl;
#endif
      // [한국어] DEBUG_FLY 활성 시 현재 노드 연결 시작을 콘솔에 출력.

      for ( int port = 0; port < _k; ++port ) {
        // [한국어] 포트 루프 (0 ~ k-1) — 현재 라우터의 k개 입력 포트와 k개 출력 포트를 순서대로 처리.
        //         버터플라이에서 k는 기수(radix)이므로 각 라우터는 k-way 스위치.

	// Input connections
	if ( stage == 0 ) {
          // [한국어] 스테이지 0 (첫 번째 스테이지): 입력 포트에 주입 채널(_inject)을 연결.
          //         주입 채널은 외부(SM/L2 캐시)에서 NoC로 플릿을 보내는 채널.
	  c = addr*_k + port;
          // [한국어] 주입 채널 인덱스 계산 = addr*k + port.
          //         스테이지 0의 라우터 addr, 포트 port에 연결되는 단말 노드 번호와 동일하다.
          //         _inject와 _inject_cred 배열은 _nodes(=k^n)개이며, 각 단말이 하나씩 대응.

	  _routers[node]->AddInputChannel( _inject[c], _inject_cred[c] );
          // [한국어] 주입 플릿 채널(_inject[c])과 크레딧 역방향 채널(_inject_cred[c])을
          //         현재 라우터의 port번째 입력 포트에 연결.
          //         이후 외부에서 WriteFlit(c) 호출 시 이 채널을 통해 플릿이 라우터로 진입.

#ifdef DEBUG_FLY
	  cout << "  injection channel " << c << endl;
#endif
          // [한국어] DEBUG_FLY 활성 시 주입 채널 번호를 출력하여 배선 정확성 검증.

	} else {
          // [한국어] 스테이지 1 이상 (중간 또는 마지막 스테이지): 입력 포트에 내부 채널(_chan)을 연결.
          //         이전 스테이지의 라우터 출력 포트에서 오는 채널을 버터플라이 배선 규칙으로 계산.

	  c = _InChannel( stage, addr, port );
          // [한국어] 현재(stage, addr, port)에 연결되는 이전 스테이지 채널 인덱스 계산.
          //         _InChannel()은 버터플라이 자릿수 교환 규칙(digit-swap)을 적용하여
          //         (stage-1)*_nodes + in_addr*_k + in_port 형태의 인덱스를 반환한다.

	  _routers[node]->AddInputChannel( _chan[c], _chan_cred[c] );
          // [한국어] 내부 플릿 채널(_chan[c])과 크레딧 역방향 채널(_chan_cred[c])을
          //         현재 라우터의 port번째 입력 포트에 연결.
          //         동일한 _chan[c] 객체가 이전 스테이지에서 AddOutputChannel로도 연결되어 있음.

	  _chan[c]->SetLatency( 1 );
          // [한국어] 이 내부 채널의 플릿 전송 레이턴시를 1 사이클로 설정.
          //         버터플라이에서 스테이지 간 링크 레이턴시는 1 사이클로 가정하며,
          //         이것이 NoC 홉당 지연의 기본값이다.

#ifdef DEBUG_FLY
	  cout << "  input channel " << c << endl;
#endif
          // [한국어] DEBUG_FLY 활성 시 입력 내부 채널 번호를 출력.
	}

	// Output connections
	if ( stage == _n - 1 ) {
          // [한국어] 스테이지 n-1 (마지막 스테이지): 출력 포트에 이젝트 채널(_eject)을 연결.
          //         이젝트 채널은 NoC에서 외부(SM/L2 캐시)로 플릿을 전달하는 채널.

	  c = addr*_k + port;
          // [한국어] 이젝트 채널 인덱스 계산 = addr*k + port.
          //         주입 채널 인덱스 계산과 동일한 공식으로, 단말 노드 번호에 대응.
          //         _eject, _eject_cred 배열도 _nodes개 원소를 가진다.

	  _routers[node]->AddOutputChannel( _eject[c], _eject_cred[c] );
          // [한국어] 이젝트 플릿 채널(_eject[c])과 크레딧 역방향 채널(_eject_cred[c])을
          //         현재 라우터의 port번째 출력 포트에 연결.
          //         이후 외부에서 ReadFlit(c) 호출 시 이 채널에서 플릿을 수신.

#ifdef DEBUG_FLY
	  cout << "  ejection channel " << c << endl;
#endif
          // [한국어] DEBUG_FLY 활성 시 이젝트 채널 번호를 출력.

	} else {
          // [한국어] 마지막 스테이지 이외: 출력 포트에 다음 스테이지로 향하는 내부 채널 연결.

	  c = _OutChannel( stage, addr, port );
          // [한국어] 현재(stage, addr, port)의 출력 채널 인덱스 계산.
          //         _OutChannel() 공식: stage*_nodes + addr*_k + port.
          //         다음 스테이지(stage+1)의 입력 포트에서 _InChannel()로 동일한 c를 계산함으로써
          //         양쪽이 같은 _chan[c] 객체를 공유하게 된다.

	  _routers[node]->AddOutputChannel( _chan[c], _chan_cred[c] );
          // [한국어] 내부 플릿 채널(_chan[c])과 크레딧 역방향 채널(_chan_cred[c])을
          //         현재 라우터의 port번째 출력 포트에 연결.
          //         이 채널은 다음 스테이지의 라우터 입력 포트에도 연결(AddInputChannel)되어 있음.

#ifdef DEBUG_FLY
	  cout << "  output channel " << c << endl;
#endif
          // [한국어] DEBUG_FLY 활성 시 출력 내부 채널 번호를 출력.
	}
      } // [한국어] port 루프 종료 — k개 포트 처리 완료

      ++node;
      // [한국어] 다음 라우터로 선형 인덱스 전진 — per_stage*n 번 실행되어 최종적으로 node==_size가 됨.
    }
  } // [한국어] stage 루프 종료 — 모든 n개 스테이지 처리 완료; 전체 버터플라이 토폴로지 배선 확정
}

/*
 * [한국어]
 * KNFly::_OutChannel — 현재 스테이지에서 다음 스테이지로의 내부 채널 인덱스 계산
 *
 * @stage: 출력 측 라우터의 스테이지 번호 (0-based, 0 ~ n-2)
 * @addr:  해당 스테이지에서 라우터의 주소 (0 ~ k^(n-1)-1)
 * @port:  해당 라우터의 출력 포트 번호 (0 ~ k-1)
 * @return: _chan[] 배열에서의 채널 인덱스 (0 ~ (n-1)*k^n - 1)
 *
 * 내부 채널은 스테이지 경계별로 _chan 배열에 선형 배치된다:
 *   - 스테이지 s의 채널들: _chan[s*_nodes] ~ _chan[(s+1)*_nodes - 1]
 *   - 스테이지 s, 라우터 addr, 포트 port의 채널: s*_nodes + addr*_k + port
 *
 * 동일한 채널 인덱스를 다음 스테이지(s+1)의 라우터가 _InChannel()로 참조함으로써
 * 출력 포트와 입력 포트가 같은 FlitChannel 객체를 공유한다.
 *
 * 실행 컨텍스트: _BuildNet() 내에서 각 출력 포트 처리 시 호출. const 함수.
 *
 * 호출 체인:
 *   _BuildNet() → [_OutChannel] → (반환된 c로) _routers[node]->AddOutputChannel(_chan[c], ...)
 */
int KNFly::_OutChannel( int stage, int addr, int port ) const
{
  return stage*_nodes + addr*_k + port;
  // [한국어] 출력 채널 인덱스 = stage*_nodes + addr*_k + port.
  //         분해:
  //           stage*_nodes : 이 스테이지 경계의 채널 블록 시작 오프셋 (경계당 _nodes=k^n개 채널)
  //           addr*_k      : 현재 라우터의 첫 번째 채널 인덱스 (라우터당 k개 채널)
  //           port         : 해당 라우터의 포트 번호 오프셋
  //         예: k=4, n=3, stage=1, addr=2, port=3이면 = 1*64 + 2*4 + 3 = 75.
}

/*
 * [한국어]
 * KNFly::_InChannel — 이전 스테이지에서 현재 스테이지로의 입력 채널 인덱스 계산
 *                     (버터플라이 자릿수 교환 배선 규칙 적용)
 *
 * @stage: 입력 측 라우터의 스테이지 번호 (1-based, 1 ~ n-1)
 * @addr:  현재 스테이지에서 라우터의 주소 (0 ~ k^(n-1)-1)
 * @port:  현재 라우터의 입력 포트 번호 (0 ~ k-1)
 * @return: _chan[] 배열에서의 채널 인덱스 (0 ~ (n-1)*k^n - 1)
 *
 * 버터플라이 네트워크의 배선 원칙:
 *   스테이지 s의 라우터들은 k진법 n-1자리 주소(d_{n-2}...d_0)를 가진다.
 *   스테이지 s에서 스테이지 s+1로 가는 채널은 주소의 (n-s-1)번째 자릿수(digit)를
 *   포트 번호와 교환(swap)함으로써 연결된다.
 *
 * 계산 과정:
 *   shift      = k^(n-stage-1):  현재 처리 차원의 stride — 해당 자릿수의 가중치.
 *   zero_digit = (addr/shift)%k: 현재 라우터 addr에서 해당 차원 자릿수의 값.
 *   last_digit = port:           현재 입력 포트 번호.
 *   in_addr    = addr - zero_digit*shift + last_digit*shift:
 *                두 자릿수를 교환한 이전 스테이지 라우터 주소.
 *                (현재 차원 자릿수 자리에 last_digit을, zero_digit은 포트로 이동)
 *   in_port    = zero_digit:     이전 라우터에서의 출력 포트 번호.
 *
 * 반환값: (stage-1)*_nodes + in_addr*_k + in_port
 *        = 이전 스테이지 경계(_OutChannel(stage-1, in_addr, in_port))와 동일한 인덱스.
 *
 * 실행 컨텍스트: _BuildNet() 내에서 각 입력 포트 처리 시 호출. const 함수.
 *
 * 호출 체인:
 *   _BuildNet() → [_InChannel] → (반환된 c로) _routers[node]->AddInputChannel(_chan[c], ...)
 */
int KNFly::_InChannel( int stage, int addr, int port ) const
{
  int in_addr; // [한국어] 이전 스테이지에서 연결되는 라우터 주소 (자릿수 교환 결과)
  int in_port; // [한국어] 이전 라우터에서의 출력 포트 번호 (= zero_digit)

  // Channels are between {node,port}
  //   { d_{n-1} ... d_{n-stage} ... d_0 } and
  //   { d_{n-1} ... d_0         ... d_{n-stage} }
  // [한국어] 버터플라이 배선 원칙 주석: 두 연결 라우터의 주소는 특정 자릿수 위치를 교환한 형태.
  //         스테이지 stage에서 처리하는 차원의 자릿수(d_{n-stage-1})와 마지막 자릿수(d_0)를 교환.

  int shift = powi( _k, _n-stage-1 );
  // [한국어] 현재 처리 차원의 stride 계산 = k^(n-stage-1).
  //         addr을 k진법으로 볼 때 (n-stage-1)번째 자릿수의 가중치(자릿값).
  //         stage=1이면 shift=k^(n-2), stage=n-1이면 shift=k^0=1.
  //         이 값으로 addr에서 해당 자릿수를 추출하고 교환하는 데 사용.

  int last_digit = port;
  // [한국어] 현재 입력 포트 번호 = 이전 스테이지 라우터의 해당 자릿수가 가져야 할 값.
  //         버터플라이에서 포트 번호는 목적지 주소의 해당 차원 좌표에 대응한다.

  int zero_digit = ( addr / shift ) % _k;
  // [한국어] 현재 라우터 addr에서 (n-stage-1)번째 자릿수 추출.
  //         addr / shift: 해당 자릿수를 최하위 자리로 이동.
  //         % _k: k진법에서의 나머지로 해당 자릿수 값만 추출.
  //         이 값이 이전 스테이지 라우터의 출력 포트(in_port)가 된다.

  // swap zero and last digit to get first node's address
  in_addr = addr - zero_digit*shift + last_digit*shift;
  // [한국어] 이전 스테이지 라우터 주소 계산 — 자릿수 교환(digit-swap) 수행.
  //         addr에서 zero_digit 자릿수를 제거하고(-zero_digit*shift)
  //         그 자리에 last_digit(=port)을 넣어(+last_digit*shift) in_addr을 구성.
  //         결과: in_addr의 해당 자릿수 = port, 나머지 자릿수는 addr과 동일.

  in_port = zero_digit;
  // [한국어] 이전 라우터에서의 출력 포트 번호 = 현재 라우터의 해당 차원 자릿수 값.
  //         즉, 이전 라우터 in_addr의 포트 zero_digit에서 나오는 채널이
  //         현재 라우터 addr의 포트 port(=last_digit)에 연결됨을 의미한다.

  return (stage-1)*_nodes + in_addr*_k + in_port;
  // [한국어] 이전 스테이지(stage-1) 경계 채널의 인덱스 반환.
  //         이 값은 _OutChannel(stage-1, in_addr, in_port)와 동일하므로,
  //         _BuildNet()에서 이전 스테이지 라우터가 AddOutputChannel()로 등록한 채널과 동일한 객체를 가리킴.
  //         분해:
  //           (stage-1)*_nodes: 이전 스테이지 경계 채널 블록 시작 오프셋
  //           in_addr*_k:       이전 라우터의 첫 채널 오프셋
  //           in_port:          이전 라우터의 포트 오프셋
}

/*
 * [한국어]
 * KNFly::GetN — 버터플라이 네트워크의 스테이지 수(_n) 반환
 *
 * @return: int — _n (설정 파일의 "n" 파라미터, 스테이지 수)
 *
 * 외부 코드(라우팅 함수, 통계 모듈 등)가 네트워크 차원 수를 조회할 때 사용한다.
 * const 함수이므로 네트워크 상태를 변경하지 않으며 어떤 컨텍스트에서도 안전하게 호출 가능.
 *
 * 호출 체인:
 *   외부 라우팅/통계 코드 → [GetN] → _n 반환
 */
int KNFly::GetN( ) const
{
  return _n; // [한국어] 버터플라이 스테이지 수 반환 — 설정된 "n" 파라미터 값.
}

/*
 * [한국어]
 * KNFly::GetK — 버터플라이 네트워크의 기수(_k) 반환
 *
 * @return: int — _k (설정 파일의 "k" 파라미터, 라우터당 포트 수)
 *
 * 외부 코드(라우팅 함수, 통계 모듈 등)가 네트워크 기수를 조회할 때 사용한다.
 * const 함수이므로 네트워크 상태를 변경하지 않으며 어떤 컨텍스트에서도 안전하게 호출 가능.
 *
 * 호출 체인:
 *   외부 라우팅/통계 코드 → [GetK] → _k 반환
 */
int KNFly::GetK( ) const
{
  return _k; // [한국어] 버터플라이 기수(라우터당 포트 수) 반환 — 설정된 "k" 파라미터 값.
}

/*
 * [한국어]
 * KNFly::Capacity — 버터플라이 네트워크의 이분 대역폭 비율 반환
 *
 * @return: double 1.0 — 버터플라이 네트워크의 이론적 이분 대역폭 비율.
 *
 * 이분 대역폭(bisection bandwidth): 네트워크를 두 절반으로 나눌 때 절단되는 링크들의 총 대역폭.
 * 이분 대역폭 비율 = 이분 대역폭 / (단말 수 × 링크 대역폭).
 *
 * k-ary n-fly에서 이분 대역폭 비율이 1.0인 이유:
 *   - 네트워크의 중간 스테이지 경계에는 k^n개의 채널이 존재하고,
 *   - 단말 수도 k^n이므로, 각 단말당 1개의 채널이 이분 경계를 통과함.
 *   - 따라서 비율 = k^n / k^n = 1.0 (이론적 최적 이분 대역폭).
 * 실제 처리량은 라우터 혼잡과 라우팅 알고리즘에 따라 이보다 낮을 수 있다.
 *
 * 호출 체인:
 *   통계 출력 또는 설정 검증 코드 → [Capacity] → 1.0 반환
 */
double KNFly::Capacity( ) const
{
  return 1.0;
  // [한국어] 버터플라이의 이분 대역폭 비율 1.0 반환 — 이론적 최적값.
  //         이 값은 시뮬레이터가 처리량 분석 시 정규화 기준으로 사용한다.
}
