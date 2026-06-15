// $Id: fly.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] KNFly — k-ary n-fly 버터플라이 네트워크 클래스 선언 (fly.hpp)
 *
 * === 파일의 역할 ===
 * 이 헤더 파일은 k-ary n-fly(버터플라이) 토폴로지를 구현하는 KNFly 클래스를
 * 선언한다. 버터플라이 네트워크는 n개 스테이지에 k×k 스위치를 배열하여
 * k^n개의 터미널 노드를 O(n) 홉(hop)으로 연결하는 클래식 인터커넥트 토폴로지이다.
 * 이 파일은 KNFly의 공개 인터페이스(생성자, GetN, GetK, Capacity)와
 * 내부 헬퍼(채널 인덱스 계산, 크기 계산, 네트워크 구축)를 선언한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC(네트워크온칩) 계층은 intersim2/로 구현되며,
 * 이 파일은 intersim2/networks/ 디렉토리의 토폴로지 구현 중 하나이다.
 * 실행 컨텍스트: 호스트 유저스페이스 (시뮬레이터 초기화 시 인스턴스 생성).
 * 호출 체인:
 *   icnt_wrapper_init() → LocalInterconnect 또는 intersim2 선택
 *   → Network::NewNetwork("fly") → KNFly 생성자 → _ComputeSize() → _Alloc() → _BuildNet()
 * 이후 매 사이클마다 Network::Evaluate()가 호출되어 내부 라우터들이 순서대로 동작한다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모 클래스: Network (networks/network.hpp) — _size/_nodes/_channels/_routers/_chan 등
 *   기반 자료구조 및 _Alloc(), ReadInputs/Evaluate/WriteOutputs 인터페이스 제공.
 * - Router: _BuildNet()에서 Router::NewRouter()로 k×k IQRouter 인스턴스를 생성한다.
 * - misc_utils.hpp: powi(k, n) 정수 거듭제곱 계산에 사용.
 * - globals.hpp: gK, gN 전역 변수 — 라우팅 함수들이 토폴로지 파라미터를 공유하기 위해 읽는다.
 * - Configuration(config): "k"/"n" 파라미터를 읽어 _k/_n을 초기화한다.
 * - FlitChannel / CreditChannel (channel.hpp): _chan, _chan_cred 배열에 저장되는 채널 객체.
 *
 * === 주요 함수/구조체 요약 ===
 * - KNFly(config, name): 생성자. _ComputeSize→_Alloc→_BuildNet 순서로 네트워크 구성.
 * - _ComputeSize(config): _k, _n 읽기, _nodes=k^n, _size=n*k^(n-1), _channels=(n-1)*k^n 계산.
 * - _BuildNet(config): 모든 라우터를 생성하고 주입/이젝트/내부 채널을 연결.
 * - _OutChannel(stage, addr, port): stage→stage+1 방향 채널 인덱스 반환.
 * - _InChannel(stage, addr, port): stage-1→stage 방향 채널 인덱스를 버터플라이 배선 규칙으로 계산.
 * - Capacity(): 버터플라이의 이분 대역폭 비율(bisection bandwidth ratio) 1.0 반환.
 * - GetN()/GetK(): 설정 파라미터 _n/_k를 외부에 제공하는 접근자.
 * - RegisterRoutingFunctions(): 빈 함수 — 버터플라이는 별도 라우팅 함수 등록 없이 기본 라우팅 사용.
 */

#ifndef _FLY_HPP_
// [한국어] 중복 include 방지 헤더 가드 시작 — 이 파일이 두 번 이상 포함되더라도 내용이 한 번만 컴파일됨
#define _FLY_HPP_

#include "network.hpp"
// [한국어] Network 기반 클래스 포함 — _size/_nodes/_channels/_routers/_chan 등 기반 자료구조와
//         _Alloc(), ReadInputs/Evaluate/WriteOutputs 등 인터페이스를 상속받는다.

/*
 * [한국어] KNFly — k-ary n-fly 버터플라이 토폴로지 네트워크 클래스.
 *
 * 버터플라이(butterfly) 네트워크는 아래 구조를 가진다:
 *   - 스테이지 수: n (_n)
 *   - 각 스테이지의 라우터 수: k^(n-1) (per_stage)
 *   - 각 라우터의 입출력 포트 수: k×k (_k)
 *   - 터미널(단말) 노드 수: k^n (_nodes)
 *   - 총 라우터 수: n * k^(n-1) = _size
 *   - 스테이지 간 내부 채널 수: (n-1) * k^n = _channels
 *
 * 라우터 번호는 스테이지 순서로 선형 나열된다:
 *   노드 번호 = stage * per_stage + addr (0-based)
 *
 * 채널 연결 규칙:
 *   - 스테이지 0의 입력 포트: _inject[addr*k + port] (외부 주입 채널)
 *   - 스테이지 n-1의 출력 포트: _eject[addr*k + port] (외부 이젝트 채널)
 *   - 중간 스테이지: _OutChannel / _InChannel 계산으로 _chan[c] 연결
 *
 * GPU 시뮬레이션에서는 CMesh 등이 더 많이 쓰이며, 이 클래스는
 * 학술 연구용(이론적 이분 대역폭 분석 등)으로 제공된다.
 */
class KNFly : public Network {

  int _k;
  /* [한국어] 버터플라이 네트워크의 기수(radix) — 각 라우터의 입력/출력 포트 수이자
   * 주소 공간의 각 자릿수의 밑(base).
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 읽어 저장.
   * 읽는 자: _ComputeSize()에서 _nodes/_size/_channels 계산;
   *          _BuildNet()에서 라우터 생성(k×k)과 채널 주소 계산;
   *          _OutChannel()/_InChannel()에서 채널 인덱스 계산;
   *          GetK()에서 외부에 반환.
   * 값 범위: 2 이상 정수. 흔히 k=2(이진 버터플라이), k=4, k=8 등.
   * 동기화: 초기화 후 읽기 전용이므로 별도 동기화 불필요. */

  int _n;
  /* [한국어] 버터플라이 네트워크의 차원 수(number of stages) — 네트워크 지름과 직결.
   * 터미널 노드 수 = k^n, 홉 수 = n-1 (최단 경로).
   * 설정자: _ComputeSize()에서 config.GetInt("n")로 읽어 저장.
   * 읽는 자: _ComputeSize()에서 _size/_channels 계산;
   *          _BuildNet()에서 스테이지 루프 상한;
   *          _InChannel()에서 shift 계산(powi(_k, _n-stage-1));
   *          GetN()에서 외부에 반환.
   * 값 범위: 2 이상 정수. n=2이면 2스테이지 버터플라이.
   * 동기화: 초기화 후 읽기 전용이므로 별도 동기화 불필요. */

  /*
   * [한국어]
   * _ComputeSize — config에서 k/n을 읽고 네트워크 규모(_nodes/_size/_channels) 계산
   *
   * @config: Booksim 설정 객체 — "k"/"n" 파라미터를 읽는다.
   * @return: 없음 (부모 Network의 _nodes/_size/_channels를 직접 설정)
   *
   * Network 기반 클래스의 순수 가상 함수 오버라이드. 버터플라이 토폴로지 공식 적용:
   *   _nodes    = k^n          (터미널 노드 수 = 전체 단말 수)
   *   _size     = n * k^(n-1) (총 라우터 수 = n 스테이지 × 스테이지당 k^(n-1)개)
   *   _channels = (n-1) * k^n (스테이지 간 내부 채널 수 = (n-1)개 구간 × 각 k^n개)
   * gK/gN 전역 변수도 설정하여 라우팅 함수들이 파라미터를 공유할 수 있게 한다.
   *
   * 호출 체인:
   *   KNFly 생성자 → [_ComputeSize] → _Alloc() → _BuildNet()
   */
  void _ComputeSize( const Configuration &config );

  /*
   * [한국어]
   * _BuildNet — 버터플라이 네트워크의 라우터를 생성하고 채널을 연결
   *
   * @config: Booksim 설정 객체 — Router::NewRouter()로 전달되어 라우터 내부 파라미터를 설정.
   * @return: 없음
   *
   * Network 기반 클래스의 순수 가상 함수 오버라이드. 총 _size개의 라우터(k×k IQRouter)를
   * 스테이지 순서로 생성하고, 각 라우터의 k개 입력 포트와 k개 출력 포트에 채널을 연결한다:
   *   - 스테이지 0 입력 포트: _inject[addr*k + port] 주입 채널 연결
   *   - 스테이지 n-1 출력 포트: _eject[addr*k + port] 이젝트 채널 연결
   *   - 중간 스테이지 입력 포트: _InChannel(stage, addr, port)로 _chan 연결
   *   - 중간 스테이지 출력 포트: _OutChannel(stage, addr, port)로 _chan 연결
   * DEBUG_FLY 매크로가 정의된 경우 채널 연결 정보를 콘솔에 출력한다.
   *
   * 호출 체인:
   *   KNFly 생성자 → _ComputeSize() → _Alloc() → [_BuildNet]
   */
  void _BuildNet( const Configuration &config );

  /*
   * [한국어]
   * _OutChannel — stage에서 stage+1 방향으로의 내부 채널 인덱스 계산
   *
   * @stage: 현재 스테이지 번호 (0-based, 0 ~ n-2)
   * @addr:  현재 스테이지에서의 라우터 주소 (0 ~ k^(n-1)-1)
   * @port:  해당 라우터의 출력 포트 번호 (0 ~ k-1)
   * @return: _chan[] 배열에서의 채널 인덱스
   *          공식: stage*_nodes + addr*_k + port
   *          = (stage 번호 기반 오프셋) + (라우터 내 포트 위치)
   *
   * 내부 채널(_chan)은 (n-1)개 스테이지 구간에 걸쳐 선형 배열된다.
   * stage s의 채널들은 _chan[s*_nodes] ~ _chan[(s+1)*_nodes - 1]에 위치한다.
   * 동일한 채널 인덱스를 _BuildNet()에서 출력 쪽과 다음 스테이지 입력 쪽 모두 사용하므로
   * _OutChannel()과 _InChannel()은 같은 채널 객체를 서로 다른 끝에서 참조한다.
   *
   * 호출 체인:
   *   _BuildNet() → [_OutChannel] → _chan[c]에 라우터 출력 포트 연결
   */
  int _OutChannel( int stage, int addr, int port ) const;

  /*
   * [한국어]
   * _InChannel — stage-1에서 stage 방향으로의 내부 채널 인덱스 계산 (버터플라이 배선 규칙)
   *
   * @stage: 현재 스테이지 번호 (1-based, 1 ~ n-1)
   * @addr:  현재 스테이지에서의 라우터 주소 (0 ~ k^(n-1)-1)
   * @port:  해당 라우터의 입력 포트 번호 (0 ~ k-1)
   * @return: _chan[] 배열에서의 채널 인덱스
   *          공식: (stage-1)*_nodes + in_addr*_k + in_port
   *
   * 버터플라이 배선의 핵심: 스테이지 간 채널은 단순 직결이 아니라 bit-reversal에 기반한
   * 자릿수 교환(digit-swap) 패턴으로 연결된다.
   *   shift      = k^(n-stage-1)  — 현재 처리 차원의 stride
   *   last_digit = port            — 현재 입력 포트 번호 (목적지의 해당 차원 좌표)
   *   zero_digit = (addr/shift)%k  — 현재 라우터 addr에서 해당 차원의 좌표
   *   in_addr    = addr - zero_digit*shift + last_digit*shift  (두 자릿수 교환)
   *   in_port    = zero_digit
   * 즉, 현재 라우터의 port 입력 포트에 연결되는 채널은 이전 스테이지의 in_addr 라우터의
   * in_port 출력 포트로부터 온다.
   *
   * 호출 체인:
   *   _BuildNet() → [_InChannel] → _chan[c]에 라우터 입력 포트 연결
   */
  int _InChannel( int stage, int addr, int port ) const;

public:
  /*
   * [한국어]
   * KNFly — k-ary n-fly 버터플라이 네트워크 생성자
   *
   * @config: Booksim 설정 객체 — "k"/"n" 등 토폴로지 파라미터와 라우터 설정 포함.
   * @name:   이 네트워크 객체의 이름 문자열 (TimedModule 식별자로 사용).
   * @return: (생성자, 반환값 없음)
   *
   * Network 기반 클래스 생성자를 호출한 뒤 아래 3단계로 네트워크를 완성한다:
   *   1. _ComputeSize(config): _k/_n 읽기, _nodes/_size/_channels 계산
   *   2. _Alloc(): _routers/_chan 등 배열 동적 할당 및 FlitChannel 객체 생성
   *   3. _BuildNet(config): 라우터 인스턴스 생성 및 채널 연결
   * 생성 완료 후 네트워크는 매 사이클 Evaluate() 호출을 받을 준비가 된다.
   *
   * 호출 체인:
   *   Network::NewNetwork("fly") → [KNFly 생성자] → _ComputeSize → _Alloc → _BuildNet
   */
  KNFly( const Configuration &config, const string & name );

  /*
   * [한국어]
   * GetN — 버터플라이 네트워크의 스테이지 수(_n) 반환
   *
   * @return: int — _n 값 (설정 파일의 "n" 파라미터)
   *
   * 외부 모듈(라우팅 함수, 통계 수집 등)이 네트워크 차원 수를 조회할 때 사용한다.
   * 읽기 전용 const 함수이므로 어느 컨텍스트에서도 안전하게 호출 가능하다.
   *
   * 호출 체인:
   *   외부 라우팅 함수 또는 통계 모듈 → [GetN]
   */
  int GetN( ) const;

  /*
   * [한국어]
   * GetK — 버터플라이 네트워크의 기수(_k) 반환
   *
   * @return: int — _k 값 (설정 파일의 "k" 파라미터, 라우터당 포트 수)
   *
   * 외부 모듈(라우팅 함수, 통계 수집 등)이 네트워크 기수를 조회할 때 사용한다.
   * 읽기 전용 const 함수이므로 어느 컨텍스트에서도 안전하게 호출 가능하다.
   *
   * 호출 체인:
   *   외부 라우팅 함수 또는 통계 모듈 → [GetK]
   */
  int GetK( ) const;

  /*
   * [한국어]
   * RegisterRoutingFunctions — 버터플라이 전용 라우팅 함수 등록 (빈 구현)
   *
   * @return: 없음 (static void)
   *
   * Network 서브클래스들은 이 함수에서 라우팅 함수를 전역 라우팅 함수 맵에 등록한다.
   * KNFly는 별도의 전용 라우팅 함수를 사용하지 않고 기본 라우팅(oblivious 등)을 그대로 쓰므로
   * 이 함수는 빈 body({})로 선언되어 있다.
   * static 함수이므로 인스턴스 없이 호출 가능하다.
   *
   * 호출 체인:
   *   Booksim 초기화 시 → [RegisterRoutingFunctions] (아무 동작 없음)
   */
  static void RegisterRoutingFunctions(){};

  /*
   * [한국어]
   * Capacity — 버터플라이 네트워크의 이분 대역폭 비율 반환
   *
   * @return: double 1.0 — 버터플라이는 이론적으로 이분 대역폭 비율이 1.0임.
   *
   * 이분 대역폭(bisection bandwidth)은 네트워크를 반으로 나눌 때 절단되는 링크의 총 대역폭이다.
   * k-ary n-fly는 각 스테이지 경계에서 k^n개 채널이 존재하고, 단말 수도 k^n이므로
   * 비율이 정확히 1.0이 된다 (이론적 최적값).
   * 실제 GPU 시뮬레이션에서는 라우터 혼잡, 라우팅 알고리즘 등으로 인해 실효 처리량이 낮아진다.
   *
   * 호출 체인:
   *   통계 출력 또는 설정 검증 코드 → [Capacity]
   */
  double Capacity( ) const;
};

#endif
// [한국어] 헤더 가드 종료 — #ifndef _FLY_HPP_ 블록 닫기
