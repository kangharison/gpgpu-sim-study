// $Id: cmesh.hpp 5188 2012-08-30 00:31:31Z dub $

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
// CMesh: Mesh topology with concentration and express links along the
//         edge of the network
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: cmesh.hpp 5188 2012-08-30 00:31:31Z dub $
//
////////////////////////////////////////////////////////////////////////

/*
 * [한국어 설명] CMesh (Concentrated Mesh) 네트워크 토폴로지 헤더 (cmesh.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 GPU 내부 NoC(Network-on-Chip) 시뮬레이터(intersim2)에서
 * CMesh(Concentrated Mesh) 토폴로지를 구현하는 클래스 CMesh와 4개의 라우팅 함수를
 * 선언한다. CMesh는 여러 GPU SM(Streaming Multiprocessor)을 하나의 라우터에 묶어
 * (concentration) 라우터 수를 줄임으로써 홉 수와 레이턴시를 낮추는 메시 변형 토폴로지이다.
 * 또한 메시 엣지(가장자리) 라우터끼리 express channel로 직접 연결해 장거리 통신을 추가로
 * 단축한다. GPGPU-Sim에서 기본적으로 사용되는 GPU 내부 인터커넥트 네트워크 토폴로지이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름상 이 파일은 intersim2 NoC 시뮬레이터 계층에 위치한다:
 *   gpgpu-sim/gpu-sim.cc (사이클 루프)
 *     → gpgpu-sim/icnt_wrapper.cc (NoC 진입점 래퍼)
 *         → intersim2/networks/cmesh.cpp (CMesh 토폴로지 구축·라우팅)
 *             → intersim2/routers/ (각 라우터 사이클 처리)
 *                 → intersim2/allocators/, arbiters/ (VC·포트 할당)
 * CMesh 클래스는 Network 추상 클래스를 상속하며, 각 GPU 사이클마다
 * Router::Evaluate() / FlitChannel::Evaluate() 를 통해 플릿(flit)을 라우팅한다.
 * 실행 컨텍스트: CPU 호스트 시뮬레이션 스레드 (싱글 스레드로 사이클-by-사이클 동작).
 *
 * === 타 모듈과의 연결 ===
 * 의존(사용)하는 모듈:
 *   - network.hpp   : Network 추상 기반 클래스 (_nodes, _size, _channels, _routers 등 공유)
 *   - routefunc.hpp : gRoutingFunctionMap, 라우팅 함수 포인터 타입 RoutingFunction 정의
 *   - router.hpp    : Router 클래스 (AddInputChannel/AddOutputChannel, GetID 등)
 *   - flit.hpp      : Flit 구조체 (dest, vc, type 필드 참조)
 *   - globals/gK, gN, gC : 글로벌 토폴로지 파라미터 (booksim.hpp에서 extern 선언)
 * 이 파일에 의존하는 모듈:
 *   - gpgpu-sim/icnt_wrapper.cc : CMesh 객체를 생성하고 NoC 시뮬레이션 진입
 *   - intersim2/trafficmanager.cc : 네트워크를 통해 플릿 주입/배출 관리
 * 데이터 흐름: SM → inject 채널 → CMesh 라우터 (라우팅 함수 결정) → 라우터 간 채널
 *              → eject 채널 → 메모리 컨트롤러 / 다른 SM
 *
 * === 주요 함수/구조체 요약 ===
 * CMesh::CMesh()           - 생성자: 크기 계산 → 메모리 할당 → 네트워크 배선 순서로 초기화
 * CMesh::_ComputeSize()    - k, n, c 파라미터로 노드 수/_size/_channels 계산 및 글로벌 변수 설정
 * CMesh::_BuildNet()       - 라우터 생성, injection/ejection 채널, 라우터 간 채널, express 채널 연결
 * CMesh::NodeToRouter()    - 노드(SM) 번호 → 연결된 라우터 번호 변환 (라우팅 함수에서 사용)
 * CMesh::NodeToPort()      - 노드(SM) 번호 → 해당 라우터 내 포트 번호 변환 (ejection 포트 결정)
 * xy_yx_cmesh()            - 랜덤 XY/YX 차원 순서 + express 채널 활용 라우팅 (권장 기본 라우팅)
 * dor_cmesh()              - Dimension-Order Routing (항상 XY 순서) + express 채널 활용 라우팅
 * dor_no_express_cmesh()   - DOR 라우팅, express 채널 미사용 (순수 메시 라우팅)
 * xy_yx_no_express_cmesh() - 랜덤 XY/YX 차원 순서, express 채널 미사용
 */

#ifndef _CMESH_HPP_
// [한국어] 중복 포함 방지 가드: 이 헤더가 여러 번 include되어도 한 번만 처리되도록 보장
#define _CMESH_HPP_

#include "network.hpp"
// [한국어] Network 추상 기반 클래스 포함: CMesh가 상속하는 클래스, _nodes/_routers/_chan 등
//          공용 필드와 _Alloc() 같은 공통 초기화 메서드를 제공
#include "routefunc.hpp"
// [한국어] 라우팅 함수 포인터 타입(RoutingFunction)과 gRoutingFunctionMap(이름→함수 매핑) 제공.
//          RegisterRoutingFunctions()에서 이 맵에 CMesh 라우팅 함수를 등록함

/*
 * [한국어]
 * CMesh - Concentrated Mesh(집중 메시) NoC 토폴로지 클래스
 *
 * 상속: Network (순수 가상 인터페이스 포함)
 *
 * CMesh의 핵심 구조:
 *   - k×k 격자로 배치된 _size(=k^n) 개의 라우터
 *   - 각 라우터에 _c(=4)개의 SM/L2 노드가 집중 연결 (concentration)
 *   - 라우터 간 채널: +x/-x/+y/-y 각 1개씩 (총 4×_size 개)
 *   - 엣지 라우터끼리 express channel로 k/2 거리를 1홉으로 단축
 *   - 포트 번호 배치: [0..c-1]=injection/ejection, c+0=+x, c+1=-x, c+2=+y, c+3=-y
 *
 * 제약 사항 (현재 코드 구현):
 *   - n <= 2 (2차원 메시까지만 지원)
 *   - c == 4 (라우터당 노드 수는 항상 4)
 *   - _xcount == _ycount (정방형 메시만 지원)
 *   - _xrouter == _yrouter (x/y 방향 concentration 동일)
 *
 * 실행 컨텍스트: 호스트 CPU 시뮬레이션 스레드 (싱글 스레드)
 * 생명주기: gpgpu_sim 생성 시 icnt_wrapper_init()에 의해 한 번 생성, 시뮬레이션 종료 시 소멸
 */
class CMesh : public Network {
public:
  /*
   * [한국어]
   * CMesh - 생성자
   *
   * @config: Booksim 설정 객체. k/n/c/x/y/xr/yr/use_noc_latency 등 토폴로지
   *          파라미터를 담고 있음. icnt_wrapper_init()에서 gpgpusim.config를 파싱해 전달.
   * @name:   시뮬레이터 내부 식별자 문자열 (예: "cmesh")
   * @return: 없음 (생성자)
   *
   * 초기화 순서:
   *   1. Network(config, name) 기반 클래스 생성자 호출 → _timed_modules 등 공통 초기화
   *   2. _ComputeSize(config) → 노드 수, 라우터 수, 채널 수 계산 및 글로벌 변수 설정
   *   3. _Alloc() → _routers, _chan, _inject, _eject 배열 메모리 할당 (Network 제공)
   *   4. _BuildNet(config) → 라우터 생성 및 채널 연결 (토폴로지 배선)
   *
   * 호출 체인:
   *   icnt_wrapper_init() → new CMesh(config, name) → [이 생성자]
   *     → _ComputeSize() → _Alloc() → _BuildNet()
   */
  CMesh( const Configuration &config, const string & name );

  /*
   * [한국어]
   * GetN - 네트워크 차원 수 반환
   *
   * @return: _n 값 (현재 항상 2, 즉 2차원 메시)
   *
   * Network 기반 클래스의 순수 가상 함수 구현체.
   * 라우팅 분석, 통계 출력 등에서 차원 수를 조회할 때 사용.
   *
   * 호출 체인: TrafficManager 또는 외부 → [GetN]
   */
  int GetN() const;

  /*
   * [한국어]
   * GetK - 각 차원의 라우터 수 반환
   *
   * @return: _k 값 (각 x/y 방향의 라우터 개수, 예: 4×4 메시면 4 반환)
   *
   * Network 기반 클래스의 순수 가상 함수 구현체.
   * 통계 출력이나 라우팅 거리 계산 시 사용.
   *
   * 호출 체인: TrafficManager 또는 외부 → [GetK]
   */
  int GetK() const;

  /*
   * [한국어]
   * NodeToRouter - SM/노드 번호를 해당 라우터 번호로 변환하는 static 헬퍼 함수
   *
   * @address: 글로벌 노드 번호 (0 ~ _nodes-1). SM 번호 또는 L2 뱅크 번호.
   *           Flit::dest 필드로 들어오는 값.
   * @return:  이 노드가 연결된 라우터의 번호 (0 ~ _size-1).
   *           y*gK + x 형태로 인코딩됨.
   *
   * 라우팅 함수(xy_yx_cmesh, dor_cmesh 등)에서 목적지 SM의 라우터를 파악해
   * 어느 방향으로 플릿을 전달할지 결정하는 데 사용.
   * static 함수이므로 CMesh 인스턴스 없이 전역 라우팅 함수에서 직접 호출 가능.
   *
   * 호출 체인: xy_yx_cmesh / dor_cmesh → [NodeToRouter(f->dest)]
   */
  static int NodeToRouter( int address ) ;

  /*
   * [한국어]
   * NodeToPort - SM/노드 번호를 해당 라우터 내 ejection 포트 번호로 변환하는 static 헬퍼
   *
   * @address: 글로벌 노드 번호 (0 ~ _nodes-1). Flit::dest 필드로 들어오는 값.
   * @return:  해당 노드가 연결된 라우터의 포트 번호 (0 ~ _c-1).
   *           이 포트를 통해 플릿이 목적지 SM으로 ejection됨.
   *
   * 플릿이 목적지 라우터에 도착했을 때 어느 로컬 포트(노드)로 내보낼지 결정.
   * 반환된 포트 번호가 outputs->AddRange()의 out_port로 사용됨.
   *
   * 호출 체인: xy_yx_cmesh / dor_cmesh (dest_router == cur_router 시) → [NodeToPort(f->dest)]
   */
  static int NodeToPort( int address ) ;

  /*
   * [한국어]
   * RegisterRoutingFunctions - CMesh 라우팅 함수를 전역 맵에 등록하는 static 초기화 함수
   *
   * @return: 없음
   *
   * gRoutingFunctionMap에 "dor_cmesh", "dor_no_express_cmesh", "xy_yx_cmesh",
   * "xy_yx_no_express_cmesh" 4가지 라우팅 함수를 이름→함수 포인터로 등록.
   * Booksim 초기화 시 RegisterRoutingFunctions()가 각 네트워크 타입별로 호출됨.
   * 이후 TrafficManager가 설정 파일의 "routing_function" 옵션을 읽어
   * 해당 이름으로 gRoutingFunctionMap을 조회해 실제 라우팅 함수를 결정.
   *
   * 호출 체인: main() / icnt_wrapper_init() → InitializeRoutingMap()
   *            → CMesh::RegisterRoutingFunctions() → [gRoutingFunctionMap에 등록]
   */
  static void RegisterRoutingFunctions() ;

private:

  static int _cX ;
  /* [한국어] x 방향 concentration: 라우터 하나에 x 방향으로 연결되는 노드(SM) 수.
   * 설정자: _ComputeSize()에서 _c / _n 으로 계산되어 설정 (c=4, n=2이면 _cX=2).
   * 읽는 자: NodeToRouter(), NodeToPort(), _BuildNet()의 내부 링크 번호 계산.
   * 값 범위: 현재 구현에서는 항상 2 (c=4, n=2 제약).
   * 동기화: static 멤버이므로 프로그램 실행 중 단 한 번 초기화, 이후 읽기 전용. */

  static int _cY ;
  /* [한국어] y 방향 concentration: 라우터 하나에 y 방향으로 연결되는 노드(SM) 수.
   * 설정자: _ComputeSize()에서 _c / _cX 로 계산되어 설정 (c=4, _cX=2이면 _cY=2).
   * 읽는 자: NodeToRouter(), NodeToPort(), _BuildNet()의 링크 번호 계산.
   * 값 범위: 현재 구현에서는 항상 2.
   * 동기화: static 멤버, 초기화 후 읽기 전용. */

  static int _memo_NodeShiftX ;
  /* [한국어] NodeToPort() 내부 x 방향 비트 마스크 계산에 사용되는 메모이제이션 값.
   * 설정자: _ComputeSize()에서 _cX >> 1 로 계산 (즉 _cX/2 = 1).
   * 읽는 자: NodeToPort() 에서 주소에서 로컬 x 오프셋 추출 시 참조.
   * 값 범위: 현재 항상 1 (_cX=2인 경우).
   * 동기화: static 멤버, 초기화 후 읽기 전용. */

  static int _memo_NodeShiftY ;
  /* [한국어] NodeToPort() 내부 y 방향 비트 마스크 계산에 사용되는 메모이제이션 값.
   * 설정자: _ComputeSize()에서 log_two(gK * _cX) + (_cY >> 1) 로 계산.
   *         예: gK=4, _cX=2, _cY=2이면 log_two(8) + 1 = 3 + 1 = 4.
   * 읽는 자: NodeToPort() 에서 주소에서 로컬 y 오프셋 추출 시 참조.
   * 동기화: static 멤버, 초기화 후 읽기 전용. */

  static int _memo_PortShiftY ;
  /* [한국어] NodeToPort()에서 y 방향 포트 번호 계산의 시프트 크기.
   * 설정자: _ComputeSize()에서 log_two(gK * _cX) 로 계산 (예: gK=4, _cX=2이면 3).
   * 읽는 자: NodeToPort() 에서 y 성분을 포트 번호로 변환 시 사용.
   * 동기화: static 멤버, 초기화 후 읽기 전용. */

  /*
   * [한국어]
   * _ComputeSize - 설정 파라미터로부터 네트워크 크기를 계산하는 private 초기화 함수
   *
   * @config: Booksim 설정 객체. k/n/c/x/y/xr/yr 키로 파라미터를 읽음.
   * @return: 없음 (멤버 변수와 글로벌 변수 gK/gN/gC/_nodes/_size/_channels를 설정)
   *
   * 계산 내용:
   *   - _nodes    = c * k^n    (전체 처리 노드 수 = SM 수)
   *   - _size     = k^n        (전체 라우터 수)
   *   - _channels = 2*n*_size  (라우터 간 채널 수, 양방향이므로 2배)
   *   - _cX, _cY  = x/y 방향 concentration
   *   - _memo_* 값들 = NodeToPort/NodeToRouter 연산용 메모이제이션
   * 또한 assert로 n<=2, c==4, _xcount==_ycount, _xrouter==_yrouter 제약을 검사.
   *
   * 호출 체인: CMesh() → [_ComputeSize()] → (멤버/글로벌 변수 설정 완료)
   */
  void _ComputeSize( const Configuration &config );

  /*
   * [한국어]
   * _BuildNet - 실제 네트워크 배선을 수행하는 private 초기화 함수
   *
   * @config: 라우터 타입 및 use_noc_latency 옵션 참조용.
   * @return: 없음 (내부적으로 _routers, _chan, _inject, _eject에 배선)
   *
   * 수행 내용 (노드 반복 루프):
   *   1. 각 라우터 위치(y_index, x_index) 계산 (node = y*k + x)
   *   2. Router::NewRouter()로 라우터 객체 생성 (입출력 포트 수: 2*n + c)
   *   3. injection/ejection 채널 _c개 연결 (포트 0 ~ c-1)
   *   4. 라우터 간 채널 +x/-x/+y/-y 연결 (포트 c, c+1, c+2, c+3)
   *   5. 엣지 라우터에서 express channel 재배선 (입력 채널 변경)
   *   6. use_noc_latency에 따라 채널 레이턴시 설정 (express: k/2 * concentration, 일반: concentration)
   *
   * express channel 규칙:
   *   - x==0 (좌측 엣지): -x 입력이 같은 x=0 열의 k/2 거리 라우터 -x 출력에 연결
   *   - x==k-1 (우측 엣지): +x 입력이 같은 x=k-1 열의 k/2 거리 라우터 +x 출력에 연결
   *   - y==0 (하단 엣지): -y 입력이 같은 y=0 행의 k/2 거리 라우터 -y 출력에 연결
   *   - y==k-1 (상단 엣지): +y 입력이 같은 y=k-1 행의 k/2 거리 라우터 +y 출력에 연결
   *
   * 호출 체인: CMesh() → [_BuildNet()] → Router::NewRouter(), AddInputChannel(), AddOutputChannel()
   */
  void _BuildNet( const Configuration& config );

  int _k ;
  /* [한국어] 각 차원(x/y)의 라우터 개수. 예: 4×4 CMesh이면 _k=4.
   * 설정자: _ComputeSize()에서 config.GetInt("k")로 읽어 설정. gK 글로벌 변수에도 복사됨.
   * 읽는 자: _BuildNet()에서 라우터 인덱스 계산, 라우팅 함수에서 gK로 참조.
   * 값 범위: 보통 2의 거듭제곱 (2, 4, 8...). 현재 4가 일반적.
   * 동기화: 초기화 후 읽기 전용. */

  int _n ;
  /* [한국어] 메시의 차원 수. 현재 구현은 n<=2만 지원 (2차원 메시).
   * 설정자: _ComputeSize()에서 config.GetInt("n")으로 읽어 설정. gN 글로벌 변수에도 복사됨.
   * 읽는 자: _channels 계산 (2*n*_size), 라우터 degree 계산 (2*n + _c).
   * 값 범위: 현재 항상 2.
   * 동기화: 초기화 후 읽기 전용. */

  int _c ;
  /* [한국어] concentration: 라우터 하나에 연결되는 노드(SM) 수. 현재 구현은 c==4만 지원.
   * 설정자: _ComputeSize()에서 config.GetInt("c")로 읽어 설정. gC 글로벌 변수에도 복사됨.
   * 읽는 자: _BuildNet()의 injection/ejection 채널 루프 (_cX, _cY 반복), 라우팅 함수에서 gC 참조.
   * 값 범위: 현재 항상 4 (xr=2, yr=2, 즉 2×2 grid per router).
   * 동기화: 초기화 후 읽기 전용. */

  int _xcount;
  /* [한국어] x 방향 라우터 개수. _k와 동일한 값.
   * 설정자: _ComputeSize()에서 config.GetInt("x")로 읽어 설정.
   * 읽는 자: assert(_xcount == _ycount)로 정방형 제약 검사, 이후는 _k로 대체 참조.
   * 동기화: 초기화 후 읽기 전용. */

  int _ycount;
  /* [한국어] y 방향 라우터 개수. _k와 동일한 값.
   * 설정자: _ComputeSize()에서 config.GetInt("y")로 읽어 설정.
   * 읽는 자: assert(_xcount == _ycount)로 정방형 제약 검사.
   * 동기화: 초기화 후 읽기 전용. */

  int _xrouter;
  /* [한국어] 라우터당 x 방향 노드 수 (= _cX). xr 파라미터로 설정.
   * 설정자: _ComputeSize()에서 config.GetInt("xr")로 읽어 설정.
   * 읽는 자: assert(_xrouter == _yrouter)로 대칭 제약 검사, assert(c == _xrouter*_yrouter)로 c 검증.
   * 동기화: 초기화 후 읽기 전용. */

  int _yrouter;
  /* [한국어] 라우터당 y 방향 노드 수 (= _cY). yr 파라미터로 설정.
   * 설정자: _ComputeSize()에서 config.GetInt("yr")로 읽어 설정.
   * 읽는 자: assert(_xrouter == _yrouter)로 대칭 제약 검사.
   * 동기화: 초기화 후 읽기 전용. */

  bool _express_channels;
  /* [한국어] express 채널 사용 여부를 나타내는 플래그 (현재 미사용 — 실제로는 항상 express 배선).
   * 설정자: 명시적으로 설정되지 않음 (현재 코드에서는 항상 express channel이 배선됨).
   * 읽는 자: 직접 참조하는 코드 없음. 라우팅 함수 선택(dor_cmesh vs dor_no_express_cmesh)으로 제어.
   * 동기화: 해당 없음. */
};

//
// Routing Functions
//

/*
 * [한국어]
 * xy_yx_cmesh - 랜덤 XY/YX 차원 순서 + express 채널 활용 라우팅 함수 (권장 기본 설정)
 *
 * @r:          현재 플릿이 있는 라우터 객체 포인터. GetID()로 현재 라우터 번호 조회.
 * @f:          라우팅할 플릿 포인터. dest(목적지 노드), vc(현재 가상 채널), type(트래픽 클래스) 참조.
 * @in_channel: 플릿이 들어온 입력 채널 번호. < gC이면 injection(첫 홉)을 의미.
 * @outputs:    출력 포트 및 VC 범위를 저장하는 집합. AddRange()로 출력 경로 추가.
 * @inject:     true이면 플릿이 injection 단계 (out_port = -1 설정).
 * @return:     없음 (outputs에 결과 기록)
 *
 * 동작:
 *   1. 트래픽 클래스(READ_REQUEST/WRITE_REQUEST 등)에 따라 vcBegin/vcEnd 범위 결정
 *   2. inject이면 out_port = -1 (트래픽 매니저가 injection 처리)
 *   3. dest_router == cur_router이면 NodeToPort(f->dest)로 ejection 포트 결정
 *   4. 다른 라우터이면: 첫 홉(in_channel < gC)이면 RandomInt(1)로 XY/YX 무작위 선택;
 *      이후 홉에서는 f->vc로 이미 선택된 차원 순서 유지 (하위 절반 VC = XY, 상위 절반 VC = YX)
 *   5. XY 선택 시 vcEnd -= available_vcs (하위 VC 절반 사용), YX 선택 시 vcBegin += available_vcs
 *   6. cmesh_xy() 또는 cmesh_yx()를 호출해 express 채널 포함 출력 포트 결정
 *
 * 데드락 방지: XY/YX 차원 순서를 VC로 분리해 순환 의존성 제거.
 * 각 클래스에 최소 2개 VC 필요 (assert로 검사).
 *
 * 호출 체인: TrafficManager::Step() → Router::Evaluate() → [xy_yx_cmesh]
 *             → cmesh_xy() 또는 cmesh_yx() → 출력 포트 결정
 */
void xy_yx_cmesh( const Router *r, const Flit *f, int in_channel,
		  OutputSet *outputs, bool inject ) ;

/*
 * [한국어]
 * xy_yx_no_express_cmesh - 랜덤 XY/YX 차원 순서, express 채널 미사용 라우팅 함수
 *
 * @r:          현재 라우터 포인터.
 * @f:          라우팅할 플릿 포인터.
 * @in_channel: 플릿 입력 채널 번호.
 * @outputs:    출력 경로 저장 집합.
 * @inject:     injection 단계 여부.
 * @return:     없음
 *
 * xy_yx_cmesh와 동일한 구조이나 express 채널 우선 선택 로직 없이
 * 순수하게 X→Y 또는 Y→X 차원 순서로만 라우팅.
 * cmesh_xy_no_express() / cmesh_yx_no_express()를 사용.
 * express 채널이 배선되어 있어도 라우팅 함수가 express 경로를 선택하지 않음.
 *
 * 호출 체인: TrafficManager::Step() → Router::Evaluate() → [xy_yx_no_express_cmesh]
 *             → cmesh_xy_no_express() 또는 cmesh_yx_no_express()
 */
void xy_yx_no_express_cmesh( const Router *r, const Flit *f, int in_channel,
			     OutputSet *outputs, bool inject ) ;

/*
 * [한국어]
 * dor_cmesh - Dimension-Order Routing (항상 XY 순서) + express 채널 활용 라우팅 함수
 *
 * @r:          현재 라우터 포인터.
 * @f:          라우팅할 플릿 포인터.
 * @in_channel: 플릿 입력 채널 번호.
 * @outputs:    출력 경로 저장 집합.
 * @inject:     injection 단계 여부.
 * @return:     없음
 *
 * 항상 X 방향 먼저, 그 다음 Y 방향(DOR: Dimension-Order Routing)으로 라우팅.
 * xy_yx_cmesh와 달리 차원 순서를 결정론적으로 고정해 VC를 절반씩 나누지 않음.
 * cmesh_next()를 통해 express 채널을 우선 활용한다.
 * DOR 특성상 순환 없음 → 이론적 데드락 프리. 단, 특정 트래픽 패턴에서 핫스팟 가능.
 *
 * 호출 체인: TrafficManager::Step() → Router::Evaluate() → [dor_cmesh]
 *             → cmesh_next() → 출력 포트 결정
 */
void dor_cmesh( const Router *r, const Flit *f, int in_channel,
		OutputSet *outputs, bool inject ) ;

/*
 * [한국어]
 * dor_no_express_cmesh - DOR 라우팅, express 채널 미사용
 *
 * @r:          현재 라우터 포인터.
 * @f:          라우팅할 플릿 포인터.
 * @in_channel: 플릿 입력 채널 번호.
 * @outputs:    출력 경로 저장 집합.
 * @inject:     injection 단계 여부.
 * @return:     없음
 *
 * dor_cmesh와 동일하지만 cmesh_next_no_express()를 사용해 express 채널 불사용.
 * 순수 메시 토폴로지 성능과 비교하거나, express 채널 효과를 검증할 때 활용.
 *
 * 호출 체인: TrafficManager::Step() → Router::Evaluate() → [dor_no_express_cmesh]
 *             → cmesh_next_no_express() → 출력 포트 결정
 */
void dor_no_express_cmesh( const Router *r, const Flit *f, int in_channel,
			   OutputSet *outputs, bool inject ) ;

#endif
