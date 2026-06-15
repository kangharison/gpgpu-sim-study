// $Id: flatfly_onchip.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] Flattened Butterfly On-Chip NoC 헤더 (flatfly_onchip.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 GPU 내부 NoC(Network-on-Chip) 시뮬레이터(intersim2)에서
 * "Flattened Butterfly(평탄화된 버터플라이)" 토폴로지 네트워크 클래스와
 * 해당 토폴로지에 특화된 모든 라우팅 함수를 선언한다.
 * Flattened Butterfly는 전통적인 버터플라이의 각 차원 내 라우터들을 모두 직접
 * 연결(all-to-all within dimension)하여 홉 수를 줄이고 대역폭을 높인 고-기수(high-radix)
 * 토폴로지로, GPU 내부 SM(Streaming Multiprocessor) 간 통신 네트워크로 사용된다.
 * 라우팅 알고리즘으로는 최소(min), XY/YX, 적응형 XY/YX, Valiant, UGAL 등 7종을 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델의 NoC 계층에 해당한다.
 * 실행 흐름:
 *   gpgpu-sim/shader.cc (SM 메모리 요청 발생)
 *     → gpgpu-sim/icnt_wrapper.cc (intersim2 진입점)
 *       → intersim2/networks/flatfly_onchip.cpp (FlatFlyOnChip 토폴로지)
 *         → intersim2/routers/ (각 라우터 사이클 처리)
 *           → 목적지 SM/L2 캐시 도착
 * 실행 컨텍스트: 호스트 CPU 싱글스레드 시뮬레이션 루프 (GPU 사이클마다 호출).
 * Network::_Step()이 매 사이클 라우터들의 Advance()를 호출하며, 라우팅 결정 함수는
 * 플릿(flit)이 라우터에 도착할 때 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * - 상위 의존: intersim2/network.hpp (Network 기반 클래스), intersim2/routefunc.hpp (라우팅 함수 타입 및 gRoutingFunctionMap)
 * - 하위 호출: intersim2/routers/router.hpp (Router 생성 및 채널 연결), intersim2/globals.hpp (gK, gN, gC, gNumVCs 등 전역 설정)
 * - 데이터 흐름: Flit 객체가 inject 채널 → FlatFlyOnChip 내부 라우터 → eject 채널 경로로 흐름.
 *   라우팅 함수는 OutputSet에 출력 포트와 VC 범위를 기록하고, 라우터가 이를 읽어 플릿을 전달.
 * - 공유 전역 변수: _xcount, _ycount, _xrouter, _yrouter (flatfly_onchip.cpp에서 static 정의),
 *   gK, gN, gC (globals.hpp), gRoutingFunctionMap (routefunc.hpp)
 *
 * === 주요 함수/구조체 요약 ===
 * FlatFlyOnChip        - Flattened Butterfly 토폴로지 네트워크 클래스. 라우터 생성 및 채널 배선 담당.
 * _ComputeSize()       - k, n, c 파라미터로 라우터 수, 채널 수, 노드 수를 계산.
 * _BuildNet()          - inject/eject 채널 및 라우터 간 채널을 실제로 연결.
 * RegisterRoutingFunctions() - 라우팅 함수 이름→함수포인터 맵에 7종 라우팅 알고리즘 등록.
 * flatfly_transformation()   - 물리 노드 번호를 라우팅 좌표계 노드 번호로 변환 (GPU SM 그리드 배치 보정).
 * flatfly_outport()          - XY 순서 최소 출력 포트 계산.
 * flatfly_outport_yx()       - YX 순서 최소 출력 포트 계산.
 * find_distance()            - 두 노드 간 홉 수(차원 불일치 수) 계산.
 * find_ran_intm()            - UGAL 비최소 라우팅용 무작위 중간 노드 선택.
 * min_flatfly()              - 순수 최소 라우팅 (항상 dim 0부터 탐색).
 * xyyx_flatfly()             - 최초 홉에서 XY/YX 무작위 선택 + VC 분리.
 * adaptive_xyyx_flatfly()    - 크레딧(큐 점유율) 기반 적응형 XY/YX 선택.
 * valiant_flatfly()          - Valiant 무작위 중간 노드 경유 비최소 라우팅.
 * ugal_flatfly_onchip()      - UGAL: hop*queue 비교로 최소/비최소 동적 선택.
 * ugal_xyyx_flatfly_onchip() - UGAL + XY/YX 결합 (4분의 VC 구획).
 * ugal_pni_flatfly_onchip()  - PNI(Partially Non-Interfering) UGAL: 목적지 해시로 VC 분리.
 */

#ifndef _FlatFlyOnChip_HPP_
#define _FlatFlyOnChip_HPP_

#include "network.hpp"       // [한국어] intersim2 Network 기반 클래스 — 라우터 배열, 채널 배열, _Step() 루프 제공

#include "routefunc.hpp"     // [한국어] gRoutingFunctionMap (이름→함수포인터 전역 맵) 및 라우팅 함수 시그니처 정의
#include <cassert>           // [한국어] assert() 매크로 — 설정값 유효성 검사 및 포트 범위 검사에 사용


/*
 * [한국어]
 * FlatFlyOnChip - Flattened Butterfly On-Chip NoC 토폴로지 클래스
 *
 * Network를 상속받아 k^n개의 라우터와 노드(SM)들을 Flattened Butterfly 방식으로 연결한다.
 * 각 라우터는 같은 차원(행 또는 열)의 모든 다른 라우터와 직접 채널로 연결된다.
 * 이를 통해 임의의 두 SM 사이를 최대 n홉(차원 수)만에 도달할 수 있다.
 *
 * 호출 관계:
 *   intersim2/booksim_main.cpp 또는 icnt_wrapper.cc의 초기화 코드
 *     → new FlatFlyOnChip(config, name)  [생성]
 *     → Network::_Step() 매 사이클 호출  [동작]
 */
class FlatFlyOnChip : public Network {

  int _m;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * 설정자: 없음 (초기화만 되고 사용되지 않는 잔재 필드).
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _n;
  /* [한국어] 토폴로지 차원 수(n). gpgpusim.config의 "n" 파라미터.
   * 일반적으로 2(2D Flattened Butterfly). 최대 4차원까지 코드에서 지원.
   * 설정자: _ComputeSize()에서 config.GetInt("n")으로 설정.
   * 읽는 자: GetN(), _BuildNet() 내 차원 루프, 라우팅 함수(gN 전역으로도 공유).
   * 값 범위: 2~4 (코드 내 dim==0,1,2,3 분기 존재).
   * 동기화: 초기화 후 읽기 전용이므로 동기화 불필요. */

  int _r;
  /* [한국어] 각 라우터의 전체 radix(포트 수) = _c + (_k-1)*_n.
   * 즉, 집중 포트(_c개) + 각 차원별 직접 링크 수((_k-1)*_n개)의 합.
   * 예: k=4, n=2, c=4이면 _r = 4 + 3*2 = 10 (입력 10개, 출력 10개).
   * 설정자: _ComputeSize()에서 계산.
   * 읽는 자: _BuildNet()에서 Router::NewRouter(_r, _r)에 전달.
   * 값 범위: _c + (_k-1)*_n (항상 양수).
   * 동기화: 초기화 후 읽기 전용. */

  int _k;
  /* [한국어] 각 차원의 라우터 수 (radix per dimension). gpgpusim.config의 "k" 파라미터.
   * 총 라우터 수 = k^n. 예: k=4, n=2이면 라우터 16개.
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 설정, gK 전역에도 복사됨.
   * 읽는 자: _BuildNet() 내 채널 인덱스 계산, GetK(), 라우팅 함수들(gK 전역으로).
   * 값 범위: 2 이상의 양의 정수 (2의 거듭제곱 권장).
   * 동기화: 초기화 후 읽기 전용. */

  int _c;
  /* [한국어] concentration(집중도) — 라우터 하나에 연결되는 노드(SM) 수.
   * gpgpusim.config의 "c" 파라미터. _xrouter * _yrouter와 일치해야 함.
   * 설정자: _ComputeSize()에서 config.GetInt("c")로 설정, gC 전역에도 복사됨.
   * 읽는 자: _BuildNet() 내 inject/eject 채널 수 계산, 라우팅 함수들(gC 전역으로).
   * 값 범위: _xrouter * _yrouter (assert로 강제).
   * 동기화: 초기화 후 읽기 전용. */

  int _radix;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * _r과 의미상 동일한 잔재 필드로 보임.
   * 설정자: 없음.
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _net_size;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * 설정자: 없음.
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _stageout;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * 설정자: 없음.
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _numinput;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * 설정자: 없음.
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _stages;
  /* [한국어] 미사용 멤버 (코드 내 실제로 참조되지 않음).
   * 버터플라이 원형에서 사용하던 스테이지 수 개념의 잔재.
   * 설정자: 없음.
   * 읽는 자: 없음.
   * 값 범위: 초기화되지 않음.
   * 동기화: 해당 없음. */

  int _num_of_switch;
  /* [한국어] 네트워크 내 라우터(스위치) 총 개수 = k^n = _nodes / _c.
   * 설정자: _ComputeSize()에서 _nodes / _c로 계산. _size(Network 기반 클래스 멤버)에도 동일값 복사.
   * 읽는 자: _BuildNet() 내 라우터 생성 루프, 채널 수 계산(_channels = _num_of_switch * (_r - _c)).
   * 값 범위: k^n (항상 양수).
   * 동기화: 초기화 후 읽기 전용. */

  /*
   * [한국어]
   * _ComputeSize - 설정 파라미터로부터 네트워크 크기를 계산하는 내부 함수
   *
   * @config: gpgpusim.config에서 읽어온 설정 객체 (k, n, c, x, y, xr, yr 포함)
   * @return: void
   *
   * k, n, c를 읽어 _r(라우터 radix), _nodes(총 노드 수), _num_of_switch(라우터 수),
   * _channels(채널 수), _size(Network 기반 클래스의 라우터 배열 크기)를 계산한다.
   * 또한 gK, gN, gC 전역 변수에 값을 복사하여 라우팅 함수들이 참조할 수 있게 한다.
   * _xcount == _ycount, _xrouter == _yrouter, _c == _xrouter*_yrouter를 assert로 검증한다.
   *
   * 호출 체인:
   *   FlatFlyOnChip::FlatFlyOnChip() → [_ComputeSize] → (config 파라미터 읽기만)
   */
  void _ComputeSize( const Configuration &config );

  /*
   * [한국어]
   * _BuildNet - 라우터 객체 생성 및 채널 배선을 수행하는 내부 함수
   *
   * @config: gpgpusim.config 설정 객체 (use_noc_latency, gTrace 등 포함)
   * @return: void
   *
   * 두 단계로 네트워크를 구성한다:
   * 1) inject/eject 채널: 각 라우터에 _c개의 inject/eject 채널을 연결.
   *    use_noc_latency가 true면 SM까지 물리적 거리(tile 단위)를 레이턴시로 설정.
   * 2) 라우터 간 채널: 각 차원 d에서 서로 다른 라우터 쌍을 직접 연결.
   *    채널 인덱스 = (_k-1)*_n*node + (_k-1)*dim + cnt + offset (node<other이면 offset=-1).
   *    use_noc_latency가 true면 물리적 거리(tile 단위)를 레이턴시로 설정.
   *
   * 호출 체인:
   *   FlatFlyOnChip::FlatFlyOnChip() → [_BuildNet] → Router::NewRouter(), Router::AddInputChannel(), Router::AddOutputChannel()
   */
  void _BuildNet( const Configuration &config );

  /*
   * [한국어]
   * _OutChannel - 스테이지/주소/포트로 출력 채널 인덱스를 계산하는 내부 함수 (현재 미사용)
   *
   * @stage: 버터플라이 스테이지 번호 (Flattened Butterfly에서는 실질적으로 미사용)
   * @addr: 라우터 주소
   * @port: 포트 번호
   * @outputs: 출력 포트 수
   * @return: 출력 채널 인덱스
   *
   * 원래 다단계 버터플라이에서 사용하던 채널 인덱스 계산 함수의 잔재.
   * 현재 FlatFlyOnChip 코드에서는 호출되지 않는다.
   *
   * 호출 체인: (미사용)
   */
  int _OutChannel( int stage, int addr, int port, int outputs ) const;

  /*
   * [한국어]
   * _InChannel - 스테이지/주소/포트로 입력 채널 인덱스를 계산하는 내부 함수 (현재 미사용)
   *
   * @stage: 버터플라이 스테이지 번호
   * @addr: 라우터 주소
   * @port: 포트 번호
   * @return: 입력 채널 인덱스
   *
   * _OutChannel과 마찬가지로 원래 다단계 버터플라이용 잔재 함수.
   * 현재 FlatFlyOnChip 코드에서는 호출되지 않는다.
   *
   * 호출 체인: (미사용)
   */
  int _InChannel( int stage, int addr, int port ) const;

public:
  /*
   * [한국어]
   * FlatFlyOnChip - 생성자: 설정에서 파라미터를 읽어 네트워크를 완전히 초기화
   *
   * @config: gpgpusim.config 설정 객체 (k, n, c, x, y, xr, yr, use_noc_latency 등)
   * @name: 네트워크 이름 문자열 (디버그 출력용)
   * @return: 없음 (생성자)
   *
   * Network 기반 클래스 생성자를 먼저 호출한 뒤,
   * _ComputeSize() → _Alloc() → _BuildNet() 순서로 초기화한다.
   * _Alloc()은 Network 기반 클래스에서 _size, _channels 기반으로 라우터/채널 배열을 할당.
   *
   * 호출 체인:
   *   icnt_wrapper_init() 또는 booksim_main() → [FlatFlyOnChip()] → _ComputeSize → _Alloc → _BuildNet
   */
  FlatFlyOnChip( const Configuration &config, const string & name );

  /*
   * [한국어]
   * GetN - 차원 수 _n을 반환하는 접근자
   *
   * @return: 토폴로지 차원 수 (일반적으로 2)
   *
   * 외부 모듈이 네트워크 차원 수를 조회할 때 사용.
   * 라우팅 함수들은 전역 gN을 직접 사용하므로, 주로 테스트/디버그 목적.
   *
   * 호출 체인: 외부 코드 → [GetN]
   */
  int GetN( ) const;

  /*
   * [한국어]
   * GetK - 차원당 라우터 수 _k를 반환하는 접근자
   *
   * @return: 차원당 라우터 수 (각 차원의 크기)
   *
   * 외부 모듈이 네트워크 크기(k)를 조회할 때 사용.
   * 라우팅 함수들은 전역 gK를 직접 사용.
   *
   * 호출 체인: 외부 코드 → [GetK]
   */
  int GetK( ) const;

  /*
   * [한국어]
   * RegisterRoutingFunctions - Flattened Butterfly용 라우팅 함수들을 전역 맵에 등록
   *
   * @return: void
   *
   * gRoutingFunctionMap에 7종의 라우팅 함수를 이름 문자열로 등록한다.
   * gpgpusim.config의 "routing_function" 설정값이 이 이름과 일치하면 해당 함수가 사용된다.
   * 등록 항목: ran_min_flatfly, adaptive_xyyx_flatfly, xyyx_flatfly, valiant_flatfly,
   *            ugal_flatfly, ugal_pni_flatfly, ugal_xyyx_flatfly
   *
   * 호출 체인:
   *   icnt_wrapper_init() 또는 네트워크 초기화 시 → [RegisterRoutingFunctions]
   */
  static void RegisterRoutingFunctions() ;

  /*
   * [한국어]
   * Capacity - 네트워크 용량(bisection bandwidth 등 정규화된 척도)을 반환
   *
   * @return: (double)_k / 8.0 — k에 비례하는 단순 용량 추정값
   *
   * intersim2에서 네트워크 용량 비교/리포트 목적으로 사용.
   * 정확한 이론적 bisection bandwidth는 아니며, 상대적 비교용 단순 수식.
   *
   * 호출 체인: 외부 코드 → [Capacity]
   */
  double Capacity( ) const;

  /*
   * [한국어]
   * InsertRandomFaults - 네트워크에 무작위 결함을 삽입하는 함수 (현재 미구현)
   *
   * @config: gpgpusim.config 설정 객체
   * @return: void
   *
   * 결함 허용(fault tolerance) 실험을 위한 인터페이스이나, FlatFlyOnChip에서는
   * 함수 본체가 비어 있어 아무 동작도 하지 않는다.
   *
   * 호출 체인: Network 기반 클래스 또는 외부 코드 → [InsertRandomFaults]
   */
  void InsertRandomFaults( const Configuration &config );
};

/*
 * [한국어]
 * adaptive_xyyx_flatfly - 크레딧(큐 점유율) 기반 적응형 XY/YX 최소 라우팅 함수
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿(flit) 포인터 — dest, vc, type, watch 필드 참조
 * @in_channel: 플릿이 들어온 입력 채널 번호 (< gC이면 인젝션 포트)
 * @outputs:    라우팅 결정 결과를 기록할 OutputSet 포인터
 * @inject:     true이면 인젝션 단계(out_port=-1 반환)
 * @return:     void (outputs에 결과 기록)
 *
 * 첫 홉(in_channel < gC, 즉 SM에서 처음 진입)에서만 XY vs YX 방향을
 * 두 포트의 크레딧(사용 중인 VC 수)를 비교하여 적응적으로 선택한다.
 * 이후 홉에서는 현재 VC 번호 위치로 방향을 유지한다.
 * VC를 절반씩 나누어 XY방향(하위 절반)과 YX방향(상위 절반)에 할당함으로써
 * 데드락을 방지한다.
 *
 * 호출 체인:
 *   Router::Advance() [각 사이클] → gRoutingFunctionMap["adaptive_xyyx_flatfly"]
 *     → [adaptive_xyyx_flatfly] → flatfly_transformation, flatfly_outport, flatfly_outport_yx
 */
void adaptive_xyyx_flatfly( const Router *r, const Flit *f, int in_channel,
		  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * xyyx_flatfly - 무작위 XY/YX 최소 라우팅 함수
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터
 * @in_channel: 입력 채널 번호 (< gC이면 인젝션)
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * 첫 홉에서 XY 또는 YX 방향을 무작위(50:50)로 선택한다.
 * 이후 홉에서는 현재 VC 범위로 방향을 결정(하위 절반=XY, 상위 절반=YX).
 * adaptive_xyyx_flatfly와 구조는 동일하나, 크레딧 비교 대신 RandomInt(1) 사용.
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["xyyx_flatfly"]
 *     → [xyyx_flatfly] → flatfly_transformation, flatfly_outport, flatfly_outport_yx
 */
void xyyx_flatfly( const Router *r, const Flit *f, int in_channel,
		  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * min_flatfly - 순수 최소 라우팅 함수 (dim 0 우선 XY 순서)
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터
 * @in_channel: 입력 채널 번호
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * flatfly_outport()를 사용하여 항상 dim 0(X방향)부터 확인하는 XY 순서 최소 라우팅.
 * VC 분리 없이 전체 VC 범위를 사용. 단순하지만 편향 트래픽에서 데드락 위험.
 * "ran_min_flatfly" 이름으로 등록되며, 최소 경로 중 랜덤성은 없다
 * (이름의 "ran"은 다른 랜덤 중간 노드 방식과 구별을 위한 잔재 명칭).
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["ran_min_flatfly"]
 *     → [min_flatfly] → flatfly_transformation, flatfly_outport
 */
void min_flatfly( const Router *r, const Flit *f, int in_channel,
		  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * ugal_xyyx_flatfly_onchip - UGAL(Universal Globally Adaptive Load-balanced) + XY/YX 결합 라우팅
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터 — ph(phase), intm(중간 노드), vc 필드 수정
 * @in_channel: 입력 채널 번호
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * ugal_flatfly_onchip에 XY/YX 방향 선택을 추가한 확장 버전.
 * VC를 4등분하여: XY/YX 방향 × 최소(ph=2)/비최소(ph=1) 조합으로 데드락 방지.
 * f->ph: 0=초기 결정, 1=중간 노드로 비최소 라우팅 중, 2=목적지로 최소 라우팅 중.
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["ugal_xyyx_flatfly"]
 *     → [ugal_xyyx_flatfly_onchip] → flatfly_transformation, flatfly_outport,
 *        flatfly_outport_yx, find_distance, find_ran_intm
 */
void ugal_xyyx_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * ugal_flatfly_onchip - UGAL 기본 구현 (XY 방향 고정 + 최소/비최소 동적 선택)
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터
 * @in_channel: 입력 채널 번호
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * 최소 홉 경로의 (홉수 × 큐점유율)과 비최소 경로의 동일 척도를 비교하여
 * threshold=2 범위 내에서 최소 라우팅(f->ph=2)을, 초과 시 비최소(f->ph=1)를 선택.
 * VC를 절반씩 나누어 최소/비최소 분리. 방향은 XY 고정(flatfly_outport 사용).
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["ugal_flatfly"]
 *     → [ugal_flatfly_onchip] → flatfly_transformation, flatfly_outport,
 *        find_distance, find_ran_intm
 */
void ugal_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * ugal_pni_flatfly_onchip - PNI(Partially Non-Interfering) UGAL 라우팅
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터
 * @in_channel: 입력 채널 번호
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * ugal_flatfly_onchip에 더해, 인젝션 또는 라우터 간 전달 시 목적지 좌표의
 * 다음 차원 ID를 해시로 사용해 VC를 세분화(vcs_per_dest = VC 수 / gK).
 * 동일 목적지를 향하는 패킷들이 동일 VC에 집중되도록 하여 패킷 간 간섭을 줄임.
 * gK == gC인 "proper" Flattened Butterfly 설정에서만 동작(assert 포함).
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["ugal_pni_flatfly"]
 *     → [ugal_pni_flatfly_onchip] → flatfly_transformation, flatfly_outport,
 *        find_distance, find_ran_intm
 */
void ugal_pni_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			      OutputSet *outputs, bool inject );

/*
 * [한국어]
 * valiant_flatfly - Valiant 무작위 중간 노드 경유 비최소 라우팅
 *
 * @r:          현재 플릿이 위치한 라우터 포인터
 * @f:          라우팅 중인 플릿 포인터 — ph, intm 필드 수정
 * @in_channel: 입력 채널 번호
 * @outputs:    출력 결과를 기록할 OutputSet
 * @inject:     true이면 인젝션 단계
 * @return:     void
 *
 * 인젝션 시(in_channel < gC) 전체 네트워크 노드 중 무작위로 중간 노드(f->intm)를 선택.
 * f->ph=0: 중간 노드로 최소 라우팅. 중간 노드 또는 목적지 도달 시 f->ph=1로 전환.
 * f->ph=1: 최종 목적지로 최소 라우팅. VC 분리로 데드락 방지(하위=비최소, 상위=최소).
 * UGAL과 달리 경로 부하를 비교하지 않고 항상 비최소 경로를 사용(부하 균등화 목적).
 *
 * 호출 체인:
 *   Router::Advance() → gRoutingFunctionMap["valiant_flatfly"]
 *     → [valiant_flatfly] → flatfly_transformation, flatfly_outport
 */
void valiant_flatfly( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );

/*
 * [한국어]
 * find_distance - 두 노드 사이의 최소 홉 수(차원 불일치 수)를 계산
 *
 * @src:  출발 노드 번호 (flatfly_transformation() 적용 후 좌표계)
 * @dest: 목적지 노드 번호 (flatfly_transformation() 적용 후 좌표계)
 * @return: 홉 수 (0 이상 정수, 최대 gN)
 *
 * 각 차원 d에서 src_router_id % gK != dest_router_id % gK이면 dist++.
 * Flattened Butterfly에서 각 차원은 최대 1홉이므로 결과는 0~gN 범위.
 * UGAL에서 최소 경로 비용과 비최소 경로 비용 계산에 사용됨.
 *
 * 호출 체인:
 *   ugal_flatfly_onchip, ugal_xyyx_flatfly_onchip, ugal_pni_flatfly_onchip
 *     → [find_distance]
 */
int find_distance (int src, int dest);

/*
 * [한국어]
 * find_ran_intm - UGAL 비최소 라우팅용 무작위 중간 노드를 선택
 *
 * @src:  출발 노드 번호 (flatfly_transformation() 적용 후)
 * @dest: 목적지 노드 번호 (flatfly_transformation() 적용 후)
 * @return: 선택된 중간 노드 번호 (변환 좌표계 기준)
 *
 * 각 차원 d에서 src와 dest가 같은 위치이면 중간 노드도 같은 좌표를 사용하고,
 * 다른 차원이면 0~gK-1 중 무작위로 좌표를 선택한다.
 * 이를 통해 같은 차원의 불필요한 우회를 방지하면서 부하 분산을 달성한다.
 *
 * 호출 체인:
 *   ugal_flatfly_onchip, ugal_xyyx_flatfly_onchip, ugal_pni_flatfly_onchip
 *     → [find_ran_intm]
 */
int find_ran_intm (int src, int dest);

/*
 * [한국어]
 * flatfly_outport - XY 순서(dim 0 우선) 최소 출력 포트 계산
 *
 * @dest: 목적지 노드 번호 (flatfly_transformation() 적용 후)
 * @rID:  현재 라우터 ID
 * @return: 출력 포트 번호. dest가 현재 라우터면 dest % gC (로컬 포트). 에러 시 exit(-1).
 *
 * dim 0(X방향)부터 순차적으로 확인하여 처음으로 차이가 나는 차원의 포트를 반환.
 * 포트 번호 계산: gC + ((gK-1)*d) - 1 + (dID > sID ? dID : dID+1).
 * XY 순서 라우팅(min_flatfly, ugal_flatfly_onchip 등)에서 사용.
 *
 * 호출 체인:
 *   min_flatfly, ugal_flatfly_onchip, ugal_xyyx_flatfly_onchip 등
 *     → [flatfly_outport]
 */
int flatfly_outport(int dest, int rID);

/*
 * [한국어]
 * flatfly_transformation - 물리 노드 번호를 라우팅 좌표계 노드 번호로 변환
 *
 * @dest: GPGPU-Sim이 할당한 물리 노드 번호 (SM의 그리드 배치 기준)
 * @return: intersim2 라우팅 좌표계에서의 노드 번호
 *
 * GPU에서 SM들은 (_xcount*_xrouter) × (_ycount*_yrouter) 물리 그리드로 배치되어 있다.
 * intersim2의 라우팅 함수는 라우터 단위 좌표(vertical*_xcount + horizontal 등)를 기대하므로
 * 물리 배치 → 라우팅 좌표 변환이 필요하다.
 * 변환 공식: dest = (vertical*_xcount + horizontal)*gC + _xrouter*vertical_rem + horizontal_rem
 * 모든 라우팅 함수에서 f->dest, f->src, f->intm에 적용 전에 반드시 호출해야 한다.
 *
 * 호출 체인:
 *   min_flatfly, xyyx_flatfly, valiant_flatfly, ugal_* 등 모든 라우팅 함수
 *     → [flatfly_transformation]
 */
int flatfly_transformation(int dest);

/*
 * [한국어]
 * flatfly_outport_yx - YX 순서(dim n-1 우선) 최소 출력 포트 계산
 *
 * @dest: 목적지 노드 번호 (flatfly_transformation() 적용 후)
 * @rID:  현재 라우터 ID
 * @return: 출력 포트 번호. 에러 시 exit(-1).
 *
 * flatfly_outport()와 동일한 공식이나 dim 순서를 역방향(gN-1 → 0)으로 탐색.
 * YX 방향 라우팅(xyyx_flatfly, adaptive_xyyx_flatfly, ugal_xyyx_flatfly_onchip)에서
 * XY와 함께 쌍으로 사용되어 데드락을 방지한다.
 *
 * 호출 체인:
 *   xyyx_flatfly, adaptive_xyyx_flatfly, ugal_xyyx_flatfly_onchip
 *     → [flatfly_outport_yx]
 */
int flatfly_outport_yx(int dest, int rID);

#endif
