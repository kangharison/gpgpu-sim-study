// $Id: fattree.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] Fat Tree 네트워크 토폴로지 선언 (fattree.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 NoC(Network-on-Chip) 시뮬레이터에서 Fat Tree 토폴로지를
 * 구현하는 FatTree 클래스를 선언한다. Fat Tree는 HPC 클러스터와 데이터센터
 * 네트워크에서 널리 사용되는 계층적 트리 구조로, 상위 계층으로 갈수록 링크 폭이
 * 넓어지는 특성("fat") 때문에 bisection bandwidth가 높다. 이 헤더는 FatTree
 * 클래스의 인터페이스, 핵심 멤버 변수, private 내부 빌드 함수들을 선언한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델에서 SM(Shader Core)과 L2/메모리 컨트롤러 사이의
 * 인터커넥트(ICNT) 레이어에 위치한다. 구체적으로:
 *   SM(Shader Core) → ICNT(intersim2/FatTree) → L2/Memory Controller
 * Booksim2 내에서는 Network 기반 클래스(network.hpp)를 상속하여, 사이클-레벨
 * 라우팅 시뮬레이션의 진입점이 된다. 실행 컨텍스트: 호스트 유저스페이스 (CPU
 * 시뮬레이터 스레드) — GPU 디바이스 코드가 아님.
 * 호출 체인: icnt_wrapper.cc (GPGPU-Sim) → Network::Step() → Router::Step()
 *            → FatTree 채널 및 라우터 연결 설정은 생성자에서 완료됨.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: network.hpp (Network 기반 클래스, _routers/_chan/_inject/_eject 배열
 *         및 _Alloc()/_ComputeSize() 훅 제공), router.hpp (Router 클래스),
 *         configuration.hpp (Configuration 파라미터 파싱)
 * - 피의존: fattree.cpp (구현), booksim_config.cpp (토폴로지 선택 시 인스턴스화),
 *           icnt_wrapper.cc (GPGPU-Sim ↔ Booksim2 브리지)
 * - 공유 자료구조: Network::_routers[] (라우터 포인터 배열, size=_n*k^(n-1)),
 *                 Network::_chan[] (내부 채널 배열), _inject[]/_eject[] (외부 채널)
 * - gpgpusim.config의 "topology = fattree", "k", "n" 파라미터가 이 클래스의
 *   동작을 결정한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - FatTree(config, name)    : 생성자 — _ComputeSize → _Alloc → _BuildNet 순으로 초기화
 * - _ComputeSize(config)     : k, n 읽어 _nodes/_size/_channels 계산
 * - _BuildNet(config)        : 라우터 생성 + 채널 연결 (inject/eject/down/up)
 * - _Router(depth, pos)      : (레벨, 위치) → _routers[] 인덱스 접근자
 * - RegisterRoutingFunctions(): 라우팅 함수 등록 훅 (현재 미구현, 빈 함수)
 * - PreferedPort(r, index)   : 라우팅 보조 함수 (현재 미구현)
 */

////////////////////////////////////////////////////////////////////////
//
//  FatTree
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: fattree.hpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

#ifndef _FatTree_HPP_
/* [한국어] 헤더 가드 매크로 — fattree.hpp의 중복 포함을 방지.
 * 다수의 translation unit(소스 파일)이 이 헤더를 include 해도 FatTree
 * 클래스 선언이 한 번만 처리되도록 보장한다. */
#define _FatTree_HPP_

#include "network.hpp"
/* [한국어] Booksim2의 Network 기반 클래스 포함.
 * Network는 _routers, _chan, _inject, _eject, _inject_cred, _eject_cred,
 * _chan_cred 배열과 _Alloc(), AddChannel() 등 공통 빌드 인프라를 제공한다.
 * FatTree는 이를 상속받아 Fat Tree 특화 연결 로직만 추가한다. */

class FatTree : public Network {
/* [한국어] FatTree — Booksim2 Network를 상속하는 Fat Tree 토폴로지 구현 클래스.
 * 계층적 k진 트리를 n 레벨로 구성한다:
 *   - 레벨 0(최상위): k 포트 라우터 k^(n-1)개
 *   - 레벨 1~(n-1): 2k 포트 라우터 k^(n-1)개 (k개 down + k개 up)
 *   - 단말 노드: k^n개 (레벨 n-1 라우터에 주입/배출 채널로 연결)
 * 라우터 total: n * k^(n-1), 채널 total: 2*k*k^(n-1)*(n-1).
 * 호출 컨텍스트: 유저스페이스 CPU 시뮬레이터 단일 스레드에서 생성 및 실행. */

  int _k;
  /* [한국어] Fat Tree의 방사(radix) 파라미터 k.
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 읽어 설정.
   * 읽는 자: _ComputeSize()(_nodes/_size/_channels 계산),
   *          _BuildNet()(라우터 포트 수, 채널 수 계산에 반복 사용),
   *          _Router()(인덱스 계산).
   * 값 범위: 양의 정수, 일반적으로 2~8. 클수록 라우터 fanout 증가.
   * 동기화: 생성자에서 1회 설정 후 읽기 전용 — 별도 락 불필요. */

  int _n;
  /* [한국어] Fat Tree의 레벨 수(깊이) n.
   * 설정자: _ComputeSize()에서 config.GetInt("n")으로 읽어 설정.
   * 읽는 자: _ComputeSize()(_size/_channels 계산), _BuildNet()(레벨 루프),
   *          _Router()(assert 경계 검사).
   * 값 범위: 양의 정수, 보통 2~4. n=2이면 2계층 fat tree(리프+루트).
   * 동기화: 생성자에서 1회 설정 후 읽기 전용 — 별도 락 불필요. */


  void _ComputeSize( const Configuration& config );
  /* [한국어] _ComputeSize — k, n 파라미터로 네트워크 크기를 계산하는 내부 함수.
   * 부모 클래스 Network의 _nodes, _size, _channels를 채운다.
   * 생성자에서 _Alloc() 호출 전에 반드시 먼저 호출되어야 한다. */

  void _BuildNet(    const Configuration& config );
  /* [한국어] _BuildNet — 라우터를 생성하고 채널을 연결하는 내부 함수.
   * _Alloc() 이후에 호출되며, inject/eject/down/up 채널을 순서대로 배선한다.
   * 이 함수가 완료되면 FatTree 토폴로지의 물리적 연결이 확정된다. */

  Router*& _Router( int depth, int pos );
  /* [한국어] _Router — (레벨, 위치) 쌍으로 _routers[] 배열에 접근하는 인덱서.
   * 반환값이 참조(Router*&)이므로 읽기와 쓰기 모두 가능하다.
   * _BuildNet() 내에서 라우터 생성 직후 대입(_Router(l,p) = r)과
   * 채널 연결(_Router(l,p)->AddInputChannel())에 모두 사용된다. */

  int  _mapSize;
  /* [한국어] _mapSize — _inputChannelMap/_outputChannelMap/_latencyMap 배열의 크기.
   * 설정자: 현재 구현에서는 명시적으로 설정되지 않음 (레거시 필드).
   * 읽는 자: 현재 _BuildNet()에서 사용되지 않음 — 구버전 라우팅 맵 코드의 잔재.
   * 값 범위: 미정의 (초기화 코드 없음).
   * 동기화: 미사용이므로 동기화 불필요. */

  int* _inputChannelMap;
  /* [한국어] _inputChannelMap — 입력 채널 번호 재매핑 테이블 (레거시, 미사용).
   * 설정자: 현재 코드에서 할당/초기화 없음.
   * 읽는 자: 현재 코드에서 참조 없음 — 구버전 채널 매핑 로직의 잔재.
   * 값 범위: 미정의 (dangling pointer 위험 있음, 실제로는 호출되지 않음).
   * 동기화: 미사용. */

  int* _outputChannelMap;
  /* [한국어] _outputChannelMap — 출력 채널 번호 재매핑 테이블 (레거시, 미사용).
   * 설정자: 현재 코드에서 할당/초기화 없음.
   * 읽는 자: 현재 코드에서 참조 없음 — _inputChannelMap과 대칭되는 잔재 필드.
   * 값 범위: 미정의.
   * 동기화: 미사용. */

  int* _latencyMap;
  /* [한국어] _latencyMap — 채널별 레이턴시 매핑 테이블 (레거시, 미사용).
   * 설정자: 현재 코드에서 할당/초기화 없음.
   * 읽는 자: 현재 코드에서 참조 없음. 실제 채널 레이턴시는 _BuildNet()에서
   *          직접 _chan[link]->SetLatency(1)로 설정한다.
   * 값 범위: 미정의.
   * 동기화: 미사용. */



public:

  FatTree( const Configuration& config ,const string & name );
  /* [한국어] FatTree 생성자 — Fat Tree 네트워크를 완전히 초기화한다.
   * @param config: gpgpusim.config 파싱 결과 (k, n 등 네트워크 파라미터 포함).
   * @param name:   네트워크 이름 문자열 (디버그/로그 출력용).
   * 내부적으로 _ComputeSize → _Alloc → _BuildNet 순서로 호출하여
   * 라우터 생성, 채널 배열 할당, 연결 배선을 완료한다.
   * 호출 체인: Booksim2 main / icnt_wrapper_init() → FatTree 생성자. */

  static void RegisterRoutingFunctions() ;
  /* [한국어] RegisterRoutingFunctions — Fat Tree용 라우팅 함수를 전역 라우팅
   * 테이블에 등록하는 정적 훅 함수. 현재 구현은 빈 함수(no-op)이다.
   * 다른 토폴로지(예: Mesh, Torus)는 이 함수에서 routing_function_t 함수 포인터를
   * 등록하지만, FatTree는 별도 라우팅 함수 없이 기본 경로를 사용한다.
   * 호출 체인: Booksim2 초기화 루틴 → RegisterRoutingFunctions(). */

  //
  // Methods to Assit Routing Functions
  //
  static int PreferedPort( const Router* r, int index );
  /* [한국어] PreferedPort — 라우팅 함수가 선호 출력 포트를 결정할 때 사용하는
   * 보조 정적 함수. 현재 구현이 없는 미완성 함수이다(선언만 존재).
   * @param r:     현재 패킷이 위치한 라우터 포인터.
   * @param index: 목적지 또는 라우팅 힌트 인덱스 (사용처 미정의).
   * @return: 선호 출력 포트 번호 (현재 구현 없음 — 반환값 미정의).
   * Fat Tree의 UP/DOWN 라우팅에서 상위 레벨로 올라갈 때 어느 포트를 선택할지
   * 보조하는 용도로 설계되었으나, 실제 코드에서는 호출되지 않는다. */

};

#endif
/* [한국어] _FatTree_HPP_ 헤더 가드 종료. */
