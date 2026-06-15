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
 * [한국어 설명] DragonFly 네트워크 토폴로지 헤더 (dragonfly.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Dragonfly 고-기수(high-radix)/저-직경(low-diameter) 네트워크
 * 토폴로지를 구현하는 DragonFlyNew 클래스와 관련 라우팅 함수를 선언한다.
 * Dragonfly 토폴로지는 대규모 HPC(고성능 컴퓨팅) 시스템에서 사용되며,
 * 여러 라우터가 하나의 그룹을 이루고 그룹 내는 짧은 구리 케이블(intra-group),
 * 그룹 간은 고비용 광학 링크(inter-group)로 연결하는 2-level 계층 구조이다.
 * GPGPU-Sim의 intersim2 서브시뮬레이터 내에서 NoC(Network-on-Chip) 토폴로지
 * 옵션 중 하나로 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 전체 흐름 상 이 파일은 intersim2(Booksim 기반 NoC 시뮬레이터) 내에서
 * 구체적인 네트워크 토폴로지를 정의하는 계층에 속한다.
 * 호출 체인:
 *   gpgpu_sim::cycle()
 *     → icnt_wrapper (intersim2 인터페이스)
 *         → Booksim 시뮬레이션 루프
 *             → DragonFlyNew::_BuildNet()  (네트워크 구조 생성)
 *             → min_dragonflynew() / ugal_dragonflynew()  (패킷 라우팅)
 * 실행 컨텍스트: 호스트 유저스페이스 (시뮬레이터 CPU 스레드)에서 실행된다.
 * GPU 디바이스 코드가 아니며, 사이클마다 호출되는 타이밍 모델의 일부이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - network.hpp: 부모 클래스 Network 정의 (라우터/채널/주입-추출 포트 관리)
 *   - routefunc.hpp: 라우팅 함수 타입 및 gRoutingFunctionMap 전역 맵 선언
 *   - router.hpp (간접): Router 클래스 — 각 스위치 노드 표현
 *   - flit.hpp (간접): Flit 클래스 — 패킷 플릿 표현 (ph, intm, src, dest 필드)
 * 의존받는 모듈:
 *   - dragonfly.cpp: 이 헤더에서 선언한 모든 클래스/함수를 구현
 *   - booksim_main.cpp: NetworkFactory를 통해 DragonFlyNew를 생성
 *   - icnt_wrapper.cc (GPGPU-Sim): intersim2 초기화 시 이 네트워크를 선택
 * 데이터 흐름:
 *   GPU SM의 메모리 요청 → icnt_push() → Flit 생성 → 라우팅 함수 호출
 *   → dragonfly_port()로 출력 포트 결정 → 다음 라우터 채널로 전달
 * 공유 핵심 자료구조:
 *   - gP, gA, gG (globals.hpp): 전역 Dragonfly 파라미터 (gP=노드수/라우터,
 *     gA=라우터수/그룹, gG=총그룹수) — 라우팅 함수와 공유
 *   - Flit::ph: 현재 라우팅 페이즈 (0=비최소, 1=최소inter-group, 2=intra-group)
 *   - Flit::intm: UGAL 라우팅의 중간 노드 ID
 *
 * === 주요 함수/구조체 요약 ===
 * DragonFlyNew 클래스:
 *   - DragonFlyNew(): 생성자 — 크기 계산 → 메모리 할당 → 네트워크 빌드 순서로 초기화
 *   - _ComputeSize(): gpgpusim.config의 k, n 값으로 _p, _a, _g, _nodes 등 계산
 *   - _BuildNet(): 라우터 생성, intra/inter-group 채널 연결 (latency 차등 적용)
 *   - RegisterRoutingFunctions(): min/ugal 라우팅 함수를 전역 맵에 등록
 *   - GetN(), GetK(), Capacity(): 네트워크 파라미터 접근자
 * 전역 함수:
 *   - dragonfly_port(): 현재 라우터 위치, 소스, 목적지로 출력 포트 번호 계산
 *   - min_dragonflynew(): 최소 경로 라우팅 (2 VC, dateline 기반 데드락 방지)
 *   - ugal_dragonflynew(): UGAL 적응형 라우팅 (3 VC, 큐 길이 기반 경로 선택)
 */




#ifndef _DragonFly_HPP_
#define _DragonFly_HPP_

#include "network.hpp"    // [한국어] Network 기본 클래스 — 라우터 배열, 채널 배열, inject/eject 포트 관리
#include "routefunc.hpp"  // [한국어] 라우팅 함수 타입 정의 및 gRoutingFunctionMap 전역 등록 맵

/*
 * [한국어]
 * DragonFlyNew - Dragonfly 고-기수 저-직경 네트워크 토폴로지 클래스
 *
 * Dragonfly 구조 개요:
 *   - 전체 네트워크는 여러 개의 "그룹"으로 분할된다.
 *   - 각 그룹은 _a개의 라우터로 구성되며, 라우터들은 intra-group 링크로 완전 연결에
 *     가깝게 연결된다 (all-to-all 또는 near all-to-all).
 *   - 그룹 간은 inter-group 광학 링크로 연결되어 직경을 낮게 유지한다.
 *   - n=1 제약: 현재 구현은 그룹 내 1차원 구조만 지원한다.
 *
 * 파라미터 관계 (n=1 경우):
 *   _p = k (설정값): 라우터당 터미널(처리 노드) 포트 수
 *   _a = 2*_p:       그룹당 라우터 수
 *   _g = _a*_p + 1:  총 그룹 수
 *   _nodes = _a*_p*_g: 총 터미널 노드 수
 *   _k = _p + (2*_p-1) + _p: 라우터 전체 radix
 *        = 터미널 포트 + intra-group 포트 + inter-group 포트
 *
 * 상속: Network (intersim2/network.hpp)
 * 실행 컨텍스트: 호스트 시뮬레이터 스레드 (단일 스레드로 사이클 구동)
 */
class DragonFlyNew : public Network {

  int _m;
  /* [한국어] 차원 관련 임시 변수 (현재 구현에서 실질적으로 미사용).
   * 설정자: 초기화되지 않음 — _n=1 제약으로 다차원 구조가 비활성화되어 있음.
   * 읽는 자: 없음 (미래 확장을 위한 플레이스홀더로 보임).
   * 값 범위: 미초기화 상태.
   * 동기화: 단일 스레드 환경이므로 동기화 불필요. */

  int _n;
  /* [한국어] 그룹 내 차원 수 (number of dimensions within a group).
   * 설정자: _ComputeSize()에서 config.GetInt("n")으로 설정됨.
   * 읽는 자: _ComputeSize()에서 _a, _k 계산 시; _BuildNet()에서 n>1 오류 검사 시.
   * 값 범위: 반드시 1이어야 함 (assert(_n==1)). n>1은 지원하지 않음.
   * 동기화: 생성자에서 한 번 설정 후 읽기 전용 — 락 불필요. */

  int _r;
  /* [한국어] 라우터 관련 임시 변수 (현재 구현에서 실질적으로 미사용).
   * 설정자: 초기화되지 않음.
   * 읽는 자: 없음.
   * 값 범위: 미초기화.
   * 동기화: 해당 없음. */

  int _k;
  /* [한국어] 라우터의 전체 radix (총 포트 수).
   * n=1일 때: _k = _p + _p + (2*_p - 1) = 터미널(_p) + inter-group(_p) + intra-group(2*_p-1).
   * 설정자: _ComputeSize()에서 계산됨.
   * 읽는 자: _BuildNet()에서 Router::NewRouter() 호출 시 입력/출력 포트 수로 사용;
   *           GetK()로 외부에서 조회.
   * 값 범위: n=1일 때 _k = 4*_p - 1.
   * 동기화: 생성 후 읽기 전용. */

  int _p, _a, _g;
  /* [한국어] Dragonfly 핵심 3 파라미터.
   * _p: 라우터당 프로세서(터미널) 포트 수 — config의 "k" 값으로 설정됨.
   *     전역 변수 gP와 동기화되어 라우팅 함수에서 사용됨.
   * _a: 그룹당 라우터 수 — n=1이면 2*_p, n>1이면 _p^n.
   *     전역 변수 gA와 동기화됨.
   * _g: 총 그룹 수 — _a*_p + 1.
   *     전역 변수 gG와 동기화됨.
   * 설정자: _ComputeSize()에서 계산됨.
   * 읽는 자: _BuildNet()에서 채널 인덱스 계산 시; GetK(), GetN(), Capacity()에서 조회.
   * 값 범위: _p >= 1 (실용적으로 _p >= 2 이상 사용).
   * 동기화: 생성 후 읽기 전용. */

  int _radix;
  /* [한국어] 라우터 radix의 별칭 (현재 구현에서 _k와 별도로 사용되지 않는 것으로 보임).
   * 설정자: 초기화 여부 불명확.
   * 읽는 자: 없음 (미사용 필드로 보임).
   * 값 범위: 미정.
   * 동기화: 해당 없음. */

  int _net_size;
  /* [한국어] 네트워크 크기 (라우터 수 또는 노드 수 — 맥락에 따라 다름).
   * 설정자: 직접 초기화되지 않는 것으로 보임 (_nodes 또는 _size로 대체됨).
   * 읽는 자: 없음 (사실상 _nodes 또는 _num_of_switch로 대체됨).
   * 값 범위: 미정.
   * 동기화: 해당 없음. */

  int _stageout;
  /* [한국어] 멀티스테이지 네트워크 출력 스테이지 수 (현재 Dragonfly 구현에서 미사용).
   * 설정자: 초기화되지 않음.
   * 읽는 자: 없음.
   * 값 범위: 미정.
   * 동기화: 해당 없음. */

  int _numinput;
  /* [한국어] 총 입력 채널 수 (현재 구현에서 직접 사용되지 않는 것으로 보임).
   * 설정자: 초기화 여부 불명확.
   * 읽는 자: 없음 (Network 기본 클래스 필드로 관리될 가능성 있음).
   * 값 범위: 미정.
   * 동기화: 해당 없음. */

  int _stages;
  /* [한국어] 멀티스테이지 파이프라인 스테이지 수 (현재 Dragonfly에서 미사용).
   * 설정자: 초기화되지 않음.
   * 읽는 자: 없음.
   * 값 범위: 미정.
   * 동기화: 해당 없음. */

  int _num_of_switch;
  /* [한국어] 네트워크 내 총 스위치(라우터) 수.
   * = _nodes / _p = (_a * _g) 개.
   * 설정자: _ComputeSize()에서 _nodes/_p로 계산됨.
   * 읽는 자: _BuildNet()에서 라우터 생성 루프의 상한으로 사용;
   *           채널 수 계산(_channels = _num_of_switch * (_k - _p))에 사용.
   * 값 범위: _a * _g (양수 정수).
   * 동기화: 생성 후 읽기 전용. */

  int _grp_num_routers;
  /* [한국어] 그룹당 라우터 수 (_a와 동일값, gA 전역변수와 동기화됨).
   * 설정자: _ComputeSize()에서 gA로 설정됨.
   * 읽는 자: _BuildNet()에서 그룹 ID 계산 시 (grp_ID = node / _grp_num_routers).
   * 값 범위: _a = 2*_p (n=1 경우).
   * 동기화: 생성 후 읽기 전용; 전역 변수 gA도 동일 값 유지. */

  int _grp_num_nodes;
  /* [한국어] 그룹당 터미널 노드 수 (_grp_num_routers * gP).
   * 설정자: _ComputeSize()에서 _grp_num_routers*gP로 계산됨.
   * 읽는 자: _BuildNet()에서 inter-group 채널 인덱스 계산 시.
   * 값 범위: _a * _p (n=1 경우, = 2*_p^2).
   * 동기화: 생성 후 읽기 전용. */


  void _ComputeSize( const Configuration &config );
  /* [한국어] _ComputeSize — config에서 k, n을 읽어 네트워크 크기 파라미터 계산.
   * @config: gpgpusim.config에서 파싱된 설정 객체 — "k"(터미널 포트수), "n"(차원수) 읽기.
   * 반환값: 없음 (클래스 멤버 필드들을 직접 설정).
   * private 메서드로 생성자에서만 호출된다.
   * 계산 결과로 _p, _a, _g, _nodes, _k, _num_of_switch, _channels, _size,
   * gP, gA, gG(전역) 등이 결정된다. n!=1이면 assert로 즉시 종료. */

  void _BuildNet( const Configuration &config );
  /* [한국어] _BuildNet — 라우터 객체 생성 및 채널 연결로 실제 네트워크 구조 구축.
   * @config: 라우터 생성 시 Router::NewRouter()에 전달되는 설정 객체.
   * 반환값: 없음 (_routers[] 배열 채우기 및 채널 연결 완료).
   * private 메서드로 생성자에서 _Alloc() 이후에 호출된다.
   * 각 라우터에 대해:
   *   1) inject/eject 채널 연결 (터미널 포트)
   *   2) intra-group output 채널 연결 (latency=10)
   *   3) inter-group output 채널 연결 (latency=100, 광학 링크 모델링)
   *   4) intra-group input 채널 연결
   *   5) inter-group input 채널 연결
   * n>1이면 에러 메시지 출력 후 exit(-1)로 종료. */


public:
  DragonFlyNew( const Configuration &config, const string & name );
  /* [한국어] DragonFlyNew 생성자 — Dragonfly 네트워크를 완전히 초기화.
   * @config: 시뮬레이터 설정 객체 ("k", "n" 파라미터 포함).
   * @name: 이 네트워크 인스턴스의 이름 문자열 (디버그/로그용).
   * Network 부모 생성자 호출 후 _ComputeSize() → _Alloc() → _BuildNet() 순서로
   * 네트워크를 완전히 구성한다. 완료 시 _routers[], _chan[], _inject[], _eject[]
   * 배열이 모두 채워진 상태가 된다. */

  int GetN( ) const;
  /* [한국어] GetN — 그룹 내 차원 수 반환.
   * @return: _n 값 (현재 항상 1).
   * 외부에서 네트워크 파라미터를 조회할 때 사용된다. */

  int GetK( ) const;
  /* [한국어] GetK — 라우터 전체 radix(총 포트 수) 반환.
   * @return: _k 값 (n=1일 때 4*_p-1).
   * 외부에서 네트워크 파라미터를 조회할 때 사용된다. */

  double Capacity( ) const;
  /* [한국어] Capacity — 네트워크 용량 지표 반환.
   * @return: (double)_k / 8.0 — 라우터 radix를 8로 나눈 정규화 값.
   * Booksim 내부 통계 계산에서 사용된다. */

  static void RegisterRoutingFunctions();
  /* [한국어] RegisterRoutingFunctions — Dragonfly용 라우팅 함수들을 전역 맵에 등록.
   * 정적(static) 메서드로, 인스턴스 없이 호출 가능하다.
   * gRoutingFunctionMap["min_dragonflynew"] 및 "ugal_dragonflynew" 항목을 추가한다.
   * booksim_main의 초기화 단계에서 모든 토폴로지의 RegisterRoutingFunctions()가
   * 호출되어 라우팅 함수 선택이 가능해진다. */

  void InsertRandomFaults( const Configuration &config );
  /* [한국어] InsertRandomFaults — 랜덤 링크 결함 주입 (현재 미구현).
   * @config: 결함 주입 관련 설정 (사용되지 않음).
   * 함수 본체가 비어 있어 실제로 아무 동작도 하지 않는다.
   * 결함 허용(fault-tolerant) 시뮬레이션 확장을 위한 인터페이스만 선언된 상태. */

};

int dragonfly_port(int rID, int source, int dest);
/* [한국어] dragonfly_port — 현재 라우터에서 패킷의 최소 경로 출력 포트 번호 계산.
 * @rID:    현재 라우터 ID (전역 라우터 인덱스, 0 ~ _num_of_switch-1).
 * @source: 패킷 출발 터미널 노드 ID.
 * @dest:   패킷 목적지 터미널 노드 ID.
 * @return: 출력 포트 번호 (0 ~ _k-1).
 *          0 ~ gP-1: 터미널(eject) 포트 — 최종 홉에서만 선택됨.
 *          gP ~ gP+(gA-2): intra-group 포트 — 같은 그룹 내 다른 라우터로 이동.
 *          gP+(gA-1) ~ gP+(gA-1)+gP-1: inter-group 포트 — 광학 링크로 다른 그룹 이동.
 * dragonfly.cpp에서 구현되며, min_dragonflynew()와 ugal_dragonflynew() 라우팅 함수
 * 모두에서 호출된다. */

void ugal_dragonflynew( const Router *r, const Flit *f, int in_channel,
		       OutputSet *outputs, bool inject );
/* [한국어] ugal_dragonflynew — UGAL(Universal Globally-Adaptive Load-balanced) 적응형 라우팅.
 * @r:          현재 라우터 객체 (GetID(), GetUsedCredit() 사용).
 * @f:          현재 처리 중인 Flit 객체 (ph, intm, src, dest 필드 읽기/쓰기).
 * @in_channel: 패킷이 도착한 입력 채널 번호 (< gP이면 소스 라우터에서 주입된 것).
 * @outputs:    선택한 출력 포트/VC 결과를 저장하는 집합.
 * @inject:     true이면 소스 주입 단계 — 랜덤 VC 배정 후 즉시 반환.
 * 3개의 VC(Virtual Channel)가 필요하며 gNumVCs==3을 assert로 검증한다.
 * f->ph 값에 따른 3단계 FSM:
 *   ph=0: 비최소 경로 — 중간 그룹(intm)으로 향하는 inter-group 링크 사용.
 *   ph=1: 최소 경로 — 목적지 그룹으로 향하는 inter-group 링크 사용.
 *   ph=2: 최소 경로 (intra-group) — 목적지 라우터로 intra-group 링크 사용.
 * adaptive_threshold=30: 최소 경로 편향값 (양수이므로 비최소 경로보다 최소 경로 선호). */

void min_dragonflynew( const Router *r, const Flit *f, int in_channel,
		       OutputSet *outputs, bool inject );
/* [한국어] min_dragonflynew — 최소 경로 라우팅 (Minimal Routing).
 * @r:          현재 라우터 객체.
 * @f:          현재 처리 중인 Flit (ph, src, dest 필드 사용).
 * @in_channel: 입력 채널 번호 (< gP이면 주입 포트에서 온 것).
 * @outputs:    선택 결과를 저장할 OutputSet.
 * @inject:     true이면 소스 주입 — 랜덤 VC 배정 후 즉시 반환.
 * 항상 최단 경로를 사용하며, dateline 기법으로 VC를 0→1로 업그레이드하여
 * 데드락을 방지한다 (inter-group 광학 링크를 통과할 때 ph가 0에서 1로 전환됨).
 * gNumVCs >= 2 환경에서 동작한다. */

#endif
