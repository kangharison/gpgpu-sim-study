// $Id: network.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] Network 기반 클래스 선언 (network.hpp)
 *
 * === 파일의 역할 ===
 * BookSim2 NoC 시뮬레이터에서 모든 네트워크 토폴로지의 기반 클래스를 선언한다.
 * 라우터 배열(_routers), 주입/이젝트/내부 플릿 채널, 크레딧 채널을 관리하며
 * 사이클 단위 시뮬레이션 루프(ReadInputs → Evaluate → WriteOutputs)를 구동한다.
 * Network::New() 팩토리로 설정 파일의 "topology" 옵션에 따라 적절한 서브클래스를 생성한다.
 * GPGPU-Sim의 icnt_wrapper.cc가 이 클래스의 인터페이스로 SM↔L2 트래픽을 주입/이젝트한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * icnt_wrapper.cc → Network::WriteFlit() / ReadFlit() / ReadInputs() / Evaluate() / WriteOutputs()
 * Network 서브클래스: KNCube(mesh/torus), CMesh, KNFly, QTree, Tree4, FatTree, FlatFlyOnChip, AnyNet, DragonFlyNew
 *
 * === 타 모듈과의 연결 ===
 * - TimedModule (상속): ReadInputs/Evaluate/WriteOutputs 사이클 구조 제공
 * - Router: 각 _routers[r]는 IQRouter 등의 실제 라우터 인스턴스
 * - FlitChannel / CreditChannel: 플릿(데이터) 및 크레딧(백프레셔) 채널
 * - _timed_modules: 사이클마다 순서대로 업데이트할 모듈 목록 (채널+라우터)
 *
 * === 주요 함수/구조체 요약 ===
 * - Network(): _size/_nodes/_channels 초기화, _classes 설정
 * - ~Network(): 모든 라우터/채널 해제
 * - New(): "topology" 문자열로 적절한 서브클래스 인스턴스 생성
 * - _Alloc(): _routers/_inject/_eject/_chan 배열 동적 할당 및 FlitChannel 생성
 * - WriteFlit()/ReadFlit(): 외부(SM/L2)에서 플릿 주입/수신
 * - WriteCredit()/ReadCredit(): 크레딧 역방향 전송
 * - ReadInputs()/Evaluate()/WriteOutputs(): 전체 _timed_modules 일괄 업데이트
 */

#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <vector>
#include <deque>

#include "module.hpp"      // [한국어] Module 기반 클래스
#include "flit.hpp"        // [한국어] Flit 데이터 패킷 정의
#include "credit.hpp"      // [한국어] Credit 백프레셔 패킷 정의
#include "router.hpp"      // [한국어] Router 기반 클래스
#include "module.hpp"      // [한국어] (중복 포함 — 무해)
#include "timed_module.hpp" // [한국어] TimedModule: ReadInputs/Evaluate/WriteOutputs 인터페이스
#include "flitchannel.hpp"  // [한국어] FlitChannel: 레이턴시를 가진 FIFO 플릿 채널
#include "channel.hpp"      // [한국어] Channel<T> 템플릿: 크레딧 채널에 사용
#include "config_utils.hpp" // [한국어] Configuration: gpgpusim.config 파싱 결과
#include "globals.hpp"      // [한국어] gNodes 등 전역 변수

typedef Channel<Credit> CreditChannel;
/* [한국어] CreditChannel = Channel<Credit>. 출력측 → 입력측으로 전달되는 백프레셔 채널.
 * FlitChannel의 반대 방향으로 크레딧(버퍼 가용 공간 신호)을 전달한다. */


class Network : public TimedModule {
protected:

  int _size;
  /* [한국어] 네트워크 내 라우터 수. 서브클래스의 _ComputeSize()에서 결정됨.
   * 설정자: _ComputeSize() 오버라이드에서 설정; 생성자에서 -1로 초기화.
   * 읽는 자: _Alloc()에서 _routers 배열 크기, Display() 등에서 사용. */

  int _nodes;
  /* [한국어] 네트워크에 연결된 터미널 노드(SM + L2) 수.
   * _inject/_eject 채널 배열의 크기와 동일. _ComputeSize()에서 설정. */

  int _channels;
  /* [한국어] 라우터 간 내부 링크(채널) 수. _chan/_chan_cred 배열의 크기. */

  int _classes;
  /* [한국어] 플릿 클래스(우선순위/VC 그룹) 수. 설정 파일 "classes"에서 읽음.
   * FlitChannel 생성 시 클래스 수를 파라미터로 전달. */

  vector<Router *> _routers;
  /* [한국어] _size개 라우터 포인터 배열. 각 라우터는 IQRouter 등의 서브클래스.
   * 설정자: _Alloc()에서 배열 할당; 서브클래스의 _BuildNet()에서 각 라우터 인스턴스 생성.
   * 읽는 자: ReadInputs/Evaluate/WriteOutputs의 _timed_modules 루프; Display(); DumpNodeMap().
   * 동기화: ~Network()에서 delete로 해제. */

  vector<FlitChannel *> _inject;
  /* [한국어] _nodes개 주입 FlitChannel 배열. 외부(SM/L2) → 라우터로 플릿 주입 경로.
   * WriteFlit(source)에서 _inject[source]->Send(f)로 플릿을 채널에 넣는다. */

  vector<CreditChannel *> _inject_cred;
  /* [한국어] _nodes개 주입 CreditChannel 배열. 라우터 → 외부(SM/L2)로 크레딧 역방향 전송 경로.
   * ReadCredit(source)에서 _inject_cred[source]->Receive()로 크레딧을 읽는다. */

  vector<FlitChannel *> _eject;
  /* [한국어] _nodes개 이젝트 FlitChannel 배열. 라우터 → 외부(SM/L2)로 플릿 이젝트 경로.
   * ReadFlit(dest)에서 _eject[dest]->Receive()로 플릿을 받는다. */

  vector<CreditChannel *> _eject_cred;
  /* [한국어] _nodes개 이젝트 CreditChannel 배열. 외부(SM/L2) → 라우터로 크레딧 역방향 전송.
   * WriteCredit(dest)에서 _eject_cred[dest]->Send(c)로 크레딧을 전송한다. */

  vector<FlitChannel *> _chan;
  /* [한국어] _channels개 라우터 간 내부 FlitChannel 배열. 라우터 A → 라우터 B 플릿 경로. */

  vector<CreditChannel *> _chan_cred;
  /* [한국어] _channels개 라우터 간 CreditChannel 배열. 라우터 B → 라우터 A 크레딧 역방향 경로. */

  deque<TimedModule *> _timed_modules;
  /* [한국어] 사이클마다 순서대로 ReadInputs/Evaluate/WriteOutputs를 호출할 모듈 목록.
   * _Alloc()에서 _inject, _inject_cred, _eject, _eject_cred, _chan, _chan_cred 채널들이 등록됨.
   * _BuildNet()에서 라우터들도 추가됨. deque를 사용하여 앞/뒤 삽입이 가능. */

  /*
   * [한국어] _ComputeSize — 토폴로지에 따른 _size/_nodes/_channels 계산 (순수 가상)
   * @config: 설정 파일 파라미터 (k, n, concentration 등)
   * 서브클래스(KNCube, CMesh 등)에서 구현.
   */
  virtual void _ComputeSize( const Configuration &config ) = 0;

  /*
   * [한국어] _BuildNet — 라우터 생성 및 채널 연결 (순수 가상)
   * @config: 설정 파일 파라미터
   * _ComputeSize() 이후 _Alloc()으로 배열 할당 후 이 함수에서 라우터와 채널을 실제로 연결.
   */
  virtual void _BuildNet( const Configuration &config ) = 0;

  /*
   * [한국어] _Alloc — 라우터/채널 배열 동적 할당 및 FlitChannel/CreditChannel 인스턴스 생성
   * _ComputeSize()로 결정된 _size/_nodes/_channels에 따라 배열을 할당하고
   * 각 채널 객체를 생성하여 _timed_modules에 등록한다.
   * 호출 체인: 서브클래스 생성자 → _ComputeSize() → _Alloc() → _BuildNet()
   */
  void _Alloc( );

public:
  /*
   * [한국어] Network 생성자 — 기본 상태 초기화 + _classes 설정
   * _size/_nodes/_channels를 -1로 초기화 (서브클래스 _ComputeSize()에서 설정).
   * @config: 설정 파일 ("classes" 읽기)
   * @name: 네트워크 이름
   */
  Network( const Configuration &config, const string & name );

  /*
   * [한국어] ~Network — 모든 라우터/채널 동적 할당 해제
   */
  virtual ~Network( );

  /*
   * [한국어] New — "topology" 설정에 따른 네트워크 서브클래스 인스턴스 생성 팩토리
   * @config: 설정 파일 ("topology" 등 파라미터)
   * @return: 동적 할당된 Network 포인터 (호출자가 해제 책임)
   * 지원 토폴로지: torus/mesh(KNCube), cmesh(CMesh), fly(KNFly), qtree, tree4, fattree, flatfly, anynet, dragonflynew
   */
  static Network *New( const Configuration &config, const string & name );

  /*
   * [한국어] WriteFlit — source 노드에서 플릿을 _inject[source] 채널로 주입
   * @f: 전송할 플릿 포인터
   * @source: 출발 노드 번호 (SM 또는 L2 인덱스)
   * GPGPU-Sim의 icnt_wrapper.cc가 SM→L2 또는 SM→SM 플릿을 시뮬레이터에 주입할 때 사용.
   */
  virtual void WriteFlit( Flit *f, int source );

  /*
   * [한국어] ReadFlit — dest 노드의 _eject[dest] 채널에서 도착한 플릿을 수신
   * @dest: 목적지 노드 번호
   * @return: 도착한 플릿 포인터 (NULL이면 이번 사이클 도착 없음)
   */
  virtual Flit *ReadFlit( int dest );

  /*
   * [한국어] WriteCredit — dest 노드의 _eject_cred[dest] 채널로 크레딧 전송
   */
  virtual void    WriteCredit( Credit *c, int dest );

  /*
   * [한국어] ReadCredit — source 노드의 _inject_cred[source] 채널에서 크레딧 수신
   */
  virtual Credit *ReadCredit( int source );

  /*
   * [한국어] NumNodes — 네트워크에 연결된 터미널 노드 수 반환 (inline)
   */
  inline int NumNodes( ) const {return _nodes;}

  /*
   * [한국어] InsertRandomFaults — 랜덤 링크 장애 삽입 (기본 구현: 오류 출력)
   * 특정 토폴로지 서브클래스에서 오버라이드 가능. 기본은 Error() 호출.
   */
  virtual void InsertRandomFaults( const Configuration &config );

  /*
   * [한국어] OutChannelFault — 라우터 r의 채널 c에 장애/복구 설정
   */
  void OutChannelFault( int r, int c, bool fault = true );

  /*
   * [한국어] Capacity — 네트워크 처리 용량 반환 (기본 1.0)
   */
  virtual double Capacity( ) const;

  /*
   * [한국어] ReadInputs/Evaluate/WriteOutputs — 사이클 진행의 3단계
   * _timed_modules 큐의 모든 모듈(채널+라우터)에 대해 순서대로 호출.
   * GPGPU-Sim의 icnt_wrapper.cc가 매 사이클 이 순서로 호출한다.
   */
  virtual void ReadInputs( );
  virtual void Evaluate( );
  virtual void WriteOutputs( );

  /*
   * [한국어] Display/DumpChannelMap/DumpNodeMap — 네트워크 구조 출력 (디버그/분석용)
   */
  void Display( ostream & os = cout ) const;
  void DumpChannelMap( ostream & os = cout, string const & prefix = "" ) const;
  void DumpNodeMap( ostream & os = cout, string const & prefix = "" ) const;

  /* [한국어] 다양한 접근자 메서드 — 채널/라우터 배열에 대한 읽기 전용 인터페이스 */
  int NumChannels() const {return _channels;}
  const vector<FlitChannel *> & GetInject() {return _inject;}
  FlitChannel * GetInject(int index) {return _inject[index];}
  const vector<CreditChannel *> & GetInjectCred() {return _inject_cred;}
  CreditChannel * GetInjectCred(int index) {return _inject_cred[index];}
  const vector<FlitChannel *> & GetEject(){return _eject;}
  FlitChannel * GetEject(int index) {return _eject[index];}
  const vector<CreditChannel *> & GetEjectCred(){return _eject_cred;}
  CreditChannel * GetEjectCred(int index) {return _eject_cred[index];}
  const vector<FlitChannel *> & GetChannels(){return _chan;}
  const vector<CreditChannel *> & GetChannelsCred(){return _chan_cred;}
  const vector<Router *> & GetRouters(){return _routers;}
  Router * GetRouter(int index) {return _routers[index];}
  int NumRouters() const {return _size;}
};

#endif 

