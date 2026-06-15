// $Id: buffer.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 라우터 입력 포트 버퍼 클래스 선언 (buffer.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 NoC(Network-on-Chip) 시뮬레이터에서 라우터 입력 포트의
 * 물리 버퍼를 모델링하는 Buffer 클래스를 선언한다.
 * Buffer는 하나의 입력 포트에 존재하는 모든 VC(Virtual Channel)의 플릿(flit)
 * 큐를 통합 관리하며, 버퍼 전체 점유율(occupancy)과 크기(size)를 추적한다.
 * 각 VC는 내부에 별도의 VC 객체(vc.hpp)로 표현되어 독립적인 플릿 큐와
 * 상태 머신(idle/routing/vc_alloc/active 등)을 가진다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 계층: CUDA 앱 → libcuda → gpgpusim_entrypoint → gpu-sim
 *                 → shader(SM) → icnt_wrapper → intersim2 [여기]
 * intersim2 내부 위치: Router(라우터) 내부의 입력 포트 모델링.
 *   라우터의 ReadInputs() 단계에서 FlitChannel로부터 수신된 플릿이
 *   이 Buffer의 특정 VC에 AddFlit()으로 적재된다.
 *   이후 VC 할당(VA), 스위치 할당(SA) 단계를 거쳐 RemoveFlit()으로 꺼내진다.
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, 매 시뮬레이션 사이클마다 호출됨.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - vc.hpp: 개별 VC를 나타내는 VC 클래스 (플릿 큐, VC 상태 머신 포함)
 *   - flit.hpp: 플릿 객체 (Flit 클래스, 헤드/바디/테일 구분)
 *   - outputset.hpp: 라우팅 함수가 생성하는 출력 포트 후보 집합
 *   - routefunc.hpp: tRoutingFunction 함수 포인터 타입 정의
 *   - config_utils.hpp: Configuration 클래스 (num_vcs, buf_size 등 파라미터 읽기)
 *   - module.hpp: Module 기반 클래스 (계층형 이름 및 에러 출력 지원)
 * 이 모듈을 사용하는 모듈:
 *   - router.hpp/cc 계열 (IQRouter, KNCRouter 등): 입력 포트별로 Buffer를 생성·관리
 *
 * === 주요 함수/구조체 요약 ===
 * Buffer(config, outputs, parent, name): 생성자 — num_vcs개의 VC 객체 초기화
 * AddFlit(vc, f): 지정 VC에 플릿 적재, 전체 occupancy 증가
 * RemoveFlit(vc): 지정 VC 최상단 플릿 제거·반환, 전체 occupancy 감소
 * FrontFlit(vc): 제거하지 않고 최상단 플릿 조회 (peek)
 * Full(): 전체 버퍼 용량 초과 여부 확인 (흐름 제어 판단에 사용)
 * Route(vc, rf, router, f, in_channel): 해당 VC에서 라우팅 함수 실행
 */

#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

#include <vector> // [한국어] VC 객체 포인터 배열(_vc)에 사용

#include "vc.hpp"           // [한국어] 가상 채널(VC) 객체 — 플릿 FIFO 큐 + 상태 머신 포함
#include "flit.hpp"         // [한국어] 플릿(flit) 객체 — NoC 전송 기본 단위 (헤드/바디/테일)
#include "outputset.hpp"    // [한국어] 라우팅 결과인 출력 포트 후보 집합 (OutputSet)
#include "routefunc.hpp"    // [한국어] 라우팅 함수 포인터 타입 tRoutingFunction 정의
#include "config_utils.hpp" // [한국어] Configuration 클래스 — num_vcs, buf_size 등 읽기

/*
 * [한국어]
 * Buffer - 라우터 입력 포트의 물리 버퍼 (모든 VC 통합 관리)
 *
 * 하나의 라우터 입력 포트에 대응하는 버퍼 객체이다.
 * 내부적으로 num_vcs개의 VC 객체를 벡터로 보유하며,
 * 각 VC는 독립적인 플릿 FIFO 큐와 상태 머신을 갖는다.
 * 버퍼 전체의 플릿 총 점유율(_occupancy)을 별도로 추적해
 * Full() 판단을 O(1)로 수행한다.
 *
 * 상속: Module — 계층형 이름 트리에 등록, Error()/Display() 지원
 */
class Buffer : public Module {

  int _occupancy;
  /* 버퍼 전체에 현재 적재된 플릿의 총 수.
   * 설정자: AddFlit() 호출 시 ++, RemoveFlit() 호출 시 --.
   * 읽는 자: Full() 인라인 메서드가 _size와 비교해 흐름 제어 여부 결정.
   * 값 범위: 0 이상 _size 이하. 초기값 0 (생성자 초기화 리스트).
   * 동기화: 단일 스레드 시뮬레이터 — 별도 락 불필요. */

  int _size;
  /* 버퍼 전체가 수용 가능한 최대 플릿 수 (슬롯 수).
   * 설정자: 생성자에서 config의 "buf_size" 또는 num_vcs * "vc_buf_size"로 결정.
   * 읽는 자: Full() 인라인 메서드, AddFlit()의 오버플로 검사.
   * 값 범위: 양의 정수. "buf_size" < 0이면 num_vcs * vc_buf_size로 계산.
   * 동기화: 생성 이후 변경 없음 — 읽기 전용. */

  vector<VC*> _vc;
  /* 입력 포트에 속한 VC(Virtual Channel) 객체 포인터 배열.
   * 인덱스 i는 VC 번호 i에 대응한다.
   * 설정자: 생성자에서 num_vcs개의 VC를 new로 생성하여 채움.
   * 읽는 자: AddFlit/RemoveFlit/FrontFlit/Empty/GetState/SetState 등 모든 VC
   *          접근 메서드. vc 번호를 인덱스로 직접 접근.
   * 값 범위: num_vcs개의 유효한 VC 포인터 (NULL 없음). 소멸자에서 해제.
   * 동기화: 단일 스레드 — 불필요. */

#ifdef TRACK_BUFFERS
  vector<int> _class_occupancy;
  /* [TRACK_BUFFERS 전처리 조건] 트래픽 클래스별 현재 점유 플릿 수.
   * 설정자: AddFlit()에서 f->cl 인덱스로 증가, RemoveFlit()에서 감소.
   * 읽는 자: GetOccupancyForClass(c) 메서드 — 디버그/통계 수집용.
   * 값 범위: 각 원소 0 이상. 크기는 config의 "classes" 값.
   * 동기화: 단일 스레드 — 불필요. */
#endif

public:

  /*
   * [한국어]
   * Buffer 생성자 - VC 객체 배열 초기화 및 버퍼 크기 설정
   *
   * @config: NoC 설정 객체 — "num_vcs", "buf_size", "vc_buf_size", "classes" 읽기
   * @outputs: 이 라우터의 출력 포트 수 — VC 객체 생성 시 전달 (라우팅 테이블 크기)
   * @parent: 모듈 계층 트리 부모 (라우터 객체)
   * @name: 이 버퍼 모듈의 이름 문자열 (예: "buf_0")
   * @return: 없음 (생성자)
   *
   * num_vcs개의 VC 객체를 new로 생성하고 _vc 벡터에 저장한다.
   * _size는 "buf_size" 값을 우선 사용하되, 음수이면
   * num_vcs * "vc_buf_size"로 계산한다 (VC당 고정 슬롯 모델).
   * 실행 컨텍스트: 시뮬레이터 초기화 단계 (시뮬레이션 루프 시작 전).
   *
   * 호출 체인:
   *   IQRouter::IQRouter() → [Buffer::Buffer()] → VC::VC()
   */
  Buffer( const Configuration& config, int outputs,
	  Module *parent, const string& name );

  /*
   * [한국어]
   * Buffer 소멸자 - VC 객체 배열 메모리 해제
   *
   * @return: 없음 (소멸자)
   *
   * _vc 벡터의 모든 VC 포인터에 대해 delete를 호출한다.
   * 생성자에서 new로 할당된 VC 객체들을 정리하는 책임이 있다.
   *
   * 호출 체인:
   *   라우터 소멸 → [Buffer::~Buffer()] → VC::~VC() (각 VC별로)
   */
  ~Buffer();

  /*
   * [한국어]
   * AddFlit - 지정된 VC에 플릿을 적재하고 전체 occupancy를 증가시킴
   *
   * @vc: 플릿을 적재할 VC 번호 (0 이상 num_vcs 미만)
   * @f:  적재할 플릿 포인터 (헤드/바디/테일 중 하나)
   * @return: 없음
   *
   * 버퍼 전체 점유율 검사 후 플릿을 해당 VC의 큐 뒤에 삽입한다.
   * _occupancy >= _size이면 오버플로 에러를 발생시킨다.
   * TRACK_BUFFERS 활성 시 트래픽 클래스별 occupancy도 갱신한다.
   * 실행 컨텍스트: 라우터의 ReadInputs() 단계 — 매 사이클마다 호출 가능.
   *
   * 호출 체인:
   *   IQRouter::ReadInputs() → [Buffer::AddFlit()] → VC::AddFlit()
   */
  void AddFlit( int vc, Flit *f );

  /*
   * [한국어]
   * RemoveFlit - 지정된 VC의 최상단 플릿을 제거하여 반환 (인라인)
   *
   * @vc: 플릿을 꺼낼 VC 번호
   * @return: 제거된 플릿 포인터 (호출자가 소유권을 가져감)
   *
   * 전체 _occupancy를 먼저 감소시킨 뒤 VC::RemoveFlit()으로
   * 해당 VC의 FIFO 큐 헤드 플릿을 반환한다.
   * TRACK_BUFFERS 활성 시 클래스별 occupancy도 감소시킨다.
   * 실행 컨텍스트: SA(스위치 할당) 이후 WriteOutputs() 단계.
   *
   * 호출 체인:
   *   IQRouter::_OutputQueuing() 또는 WriteOutputs() → [Buffer::RemoveFlit()] → VC::RemoveFlit()
   */
  inline Flit *RemoveFlit( int vc )
  {
    --_occupancy; // [한국어] 버퍼 전체 점유율 1 감소 — 슬롯 하나가 비었음을 전체 수준에서 반영
#ifdef TRACK_BUFFERS
    int cl = _vc[vc]->FrontFlit()->cl; // [한국어] 꺼낼 플릿의 트래픽 클래스 번호 조회 (FIFO 헤드)
    assert(_class_occupancy[cl] > 0);  // [한국어] 클래스별 occupancy가 0 미만으로 내려가지 않아야 함
    --_class_occupancy[cl];            // [한국어] 해당 클래스의 점유 카운터 감소
#endif
    return _vc[vc]->RemoveFlit( ); // [한국어] VC FIFO 큐에서 헤드 플릿 제거 후 반환
  }

  /*
   * [한국어]
   * FrontFlit - 제거하지 않고 지정 VC의 최상단 플릿을 조회 (peek, 인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: 해당 VC의 FIFO 큐 헤드 플릿 포인터 (NULL 가능 — VC가 비어 있을 때)
   *
   * 플릿을 꺼내지 않고 헤드만 확인한다.
   * 라우팅 단계에서 헤드 플릿의 목적지/클래스를 읽을 때 사용.
   *
   * 호출 체인:
   *   IQRouter::_Routing() → [Buffer::FrontFlit()] → VC::FrontFlit()
   */
  inline Flit *FrontFlit( int vc ) const
  {
    return _vc[vc]->FrontFlit( ); // [한국어] VC 큐 헤드 플릿 반환 (제거 없음)
  }

  /*
   * [한국어]
   * Empty - 지정 VC가 비어 있는지 확인 (인라인)
   *
   * @vc: 확인할 VC 번호
   * @return: VC 큐가 비어 있으면 true, 그렇지 않으면 false
   *
   * VC 스케줄러가 라운드로빈/우선순위 선택 시 빈 VC를 건너뛰는 데 사용.
   *
   * 호출 체인:
   *   IQRouter::_PickableVC() 또는 라우팅 로직 → [Buffer::Empty()] → VC::Empty()
   */
  inline bool Empty( int vc ) const
  {
    return _vc[vc]->Empty( ); // [한국어] 해당 VC의 플릿 큐가 비어 있는지 여부
  }

  /*
   * [한국어]
   * Full - 버퍼 전체가 가득 찼는지 확인 (인라인)
   *
   * @return: 전체 occupancy >= 전체 size이면 true
   *
   * 업스트림 라우터(또는 SM의 패킷 주입기)가 새 플릿을 보낼 수 있는지
   * 흐름 제어 판단에 사용된다. true이면 플릿 주입 금지.
   *
   * 호출 체인:
   *   IQRouter::ReadInputs() 또는 injectionq → [Buffer::Full()]
   */
  inline bool Full( ) const
  {
    return _occupancy >= _size; // [한국어] 전체 점유 >= 전체 용량이면 버퍼 포화 상태
  }

  /*
   * [한국어]
   * GetState - 지정 VC의 현재 상태 머신 상태 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: VC::eVCState 열거값 (idle, routing, vc_alloc, active 등)
   *
   * 라우터의 VC 할당기(VA), 스위치 할당기(SA)가 VC 상태를 확인해
   * 처리 가능한 VC만 선택할 때 사용한다.
   *
   * 호출 체인:
   *   IQRouter::_VCAlloc() → [Buffer::GetState()] → VC::GetState()
   */
  inline VC::eVCState GetState( int vc ) const
  {
    return _vc[vc]->GetState( ); // [한국어] VC 상태 머신의 현재 상태 반환
  }

  /*
   * [한국어]
   * SetState - 지정 VC의 상태 머신 상태를 변경 (인라인)
   *
   * @vc: 상태를 변경할 VC 번호
   * @s:  설정할 새 상태 (VC::eVCState)
   * @return: 없음
   *
   * 라우팅 완료, VC 할당 완료, tail 플릿 전송 완료 등 각 단계에서
   * 라우터가 직접 VC 상태를 전이시킬 때 호출한다.
   *
   * 호출 체인:
   *   IQRouter::_VCAlloc() 또는 _SwitchAlloc() → [Buffer::SetState()] → VC::SetState()
   */
  inline void SetState( int vc, VC::eVCState s )
  {
    _vc[vc]->SetState(s); // [한국어] 해당 VC의 상태를 s로 강제 전이
  }

  /*
   * [한국어]
   * GetRouteSet - 지정 VC의 라우팅 결과(출력 포트 후보 집합) 포인터 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: OutputSet const 포인터 (라우팅 함수가 계산한 후보 집합)
   *
   * VC 할당기가 라우팅 결과를 바탕으로 출력 VC를 선택할 때 사용.
   *
   * 호출 체인:
   *   IQRouter::_VCAlloc() → [Buffer::GetRouteSet()] → VC::GetRouteSet()
   */
  inline const OutputSet *GetRouteSet( int vc ) const
  {
    return _vc[vc]->GetRouteSet( ); // [한국어] VC에 저장된 라우팅 결과 집합 반환
  }

  /*
   * [한국어]
   * SetRouteSet - 지정 VC에 라우팅 결과 집합 포인터 저장 (인라인)
   *
   * @vc: 저장할 VC 번호
   * @output_set: 라우팅 함수가 계산한 OutputSet 포인터 (소유권은 VC로 이전)
   * @return: 없음
   *
   * 라우팅 단계에서 tRoutingFunction 호출 후 결과를 VC에 저장할 때 사용.
   *
   * 호출 체인:
   *   IQRouter::_Routing() → [Buffer::SetRouteSet()] → VC::SetRouteSet()
   */
  inline void SetRouteSet( int vc, OutputSet * output_set )
  {
    _vc[vc]->SetRouteSet(output_set); // [한국어] 라우팅 결과를 해당 VC에 저장
  }

  /*
   * [한국어]
   * SetOutput - 지정 VC의 할당된 출력 포트/VC 번호 저장 (인라인)
   *
   * @vc: 설정할 입력 VC 번호
   * @out_port: 할당된 출력 포트 번호
   * @out_vc: 할당된 출력 VC 번호
   * @return: 없음
   *
   * VC 할당기가 특정 출력 VC를 이 입력 VC에 바인딩할 때 호출.
   * 이후 스위치 할당기가 GetOutputPort/GetOutputVC로 조회해 전달 경로 확정.
   *
   * 호출 체인:
   *   IQRouter::_VCAlloc() → [Buffer::SetOutput()] → VC::SetOutput()
   */
  inline void SetOutput( int vc, int out_port, int out_vc )
  {
    _vc[vc]->SetOutput(out_port, out_vc); // [한국어] 이 VC가 사용할 출력 포트/VC 번호 저장
  }

  /*
   * [한국어]
   * GetOutputPort - 지정 VC에 이미 할당된 출력 포트 번호 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: 할당된 출력 포트 번호 (-1이면 아직 미할당)
   *
   * 스위치 할당기(SA)가 경합할 출력 포트를 결정할 때 참조.
   *
   * 호출 체인:
   *   IQRouter::_SwitchAlloc() → [Buffer::GetOutputPort()] → VC::GetOutputPort()
   */
  inline int GetOutputPort( int vc ) const
  {
    return _vc[vc]->GetOutputPort( ); // [한국어] VC에 저장된 출력 포트 번호 반환
  }

  /*
   * [한국어]
   * GetOutputVC - 지정 VC에 이미 할당된 출력 VC 번호 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: 할당된 출력 VC 번호 (-1이면 미할당)
   *
   * SA 단계에서 출력 큐에 플릿을 보낼 때 목적 VC 번호를 결정.
   *
   * 호출 체인:
   *   IQRouter::_SwitchTraversal() → [Buffer::GetOutputVC()] → VC::GetOutputVC()
   */
  inline int GetOutputVC( int vc ) const
  {
    return _vc[vc]->GetOutputVC( ); // [한국어] VC에 저장된 출력 VC 번호 반환
  }

  /*
   * [한국어]
   * GetPriority - 지정 VC의 우선순위 값 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: 우선순위 정수 값 (높을수록 우선)
   *
   * SA 단계에서 복수의 VC가 같은 출력에 경합할 때 우선순위 중재에 사용.
   *
   * 호출 체인:
   *   IQRouter::_SwitchAlloc() → [Buffer::GetPriority()] → VC::GetPriority()
   */
  inline int GetPriority( int vc ) const
  {
    return _vc[vc]->GetPriority( ); // [한국어] VC 우선순위 값 반환
  }

  /*
   * [한국어]
   * Route - 지정 VC에서 라우팅 함수를 실행하여 출력 포트 후보를 계산 (인라인)
   *
   * @vc: 라우팅을 수행할 VC 번호
   * @rf: 사용할 라우팅 함수 포인터 (tRoutingFunction 타입)
   * @router: 현재 라우터 포인터 (라우팅 함수에게 위치 정보 제공)
   * @f: 헤드 플릿 포인터 (목적지 정보 포함)
   * @in_channel: 플릿이 진입한 입력 채널(포트) 번호
   * @return: 없음 (결과는 VC 내부 OutputSet에 저장)
   *
   * 라우팅 단계에서 헤드 플릿이 도착하면 라우팅 함수 rf를 호출해
   * 가능한 출력 포트 후보 집합을 계산하고 이를 VC에 저장한다.
   *
   * 호출 체인:
   *   IQRouter::_Routing() → [Buffer::Route()] → VC::Route() → tRoutingFunction(rf)
   */
  inline void Route( int vc, tRoutingFunction rf, const Router* router, const Flit* f, int in_channel )
  {
    _vc[vc]->Route(rf, router, f, in_channel); // [한국어] 해당 VC에서 라우팅 함수 실행, 결과를 VC 내부 OutputSet에 저장
  }

  // ==== Debug functions ====

  /*
   * [한국어]
   * SetWatch - 지정 VC에 watch 플래그 설정 (디버그용, 인라인)
   *
   * @vc: 설정할 VC 번호
   * @watch: true이면 이 VC의 이벤트를 상세 출력 (기본 true)
   * @return: 없음
   *
   * 시뮬레이션 중 특정 VC를 추적하여 상태 변화를 로그로 출력하는
   * 디버그 기능. 성능에는 영향 없음 (조건부 출력).
   *
   * 호출 체인:
   *   디버그 설정 코드 → [Buffer::SetWatch()] → VC::SetWatch()
   */
  inline void SetWatch( int vc, bool watch = true )
  {
    _vc[vc]->SetWatch(watch); // [한국어] 해당 VC의 watch 플래그 설정 (디버그 추적 활성화)
  }

  /*
   * [한국어]
   * IsWatched - 지정 VC의 watch 플래그 조회 (디버그용, 인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: watch 중이면 true
   *
   * 디버그 로그 출력 여부를 조건 분기할 때 사용.
   *
   * 호출 체인:
   *   IQRouter 디버그 경로 → [Buffer::IsWatched()] → VC::IsWatched()
   */
  inline bool IsWatched( int vc ) const
  {
    return _vc[vc]->IsWatched( ); // [한국어] 해당 VC의 watch 플래그 반환
  }

  /*
   * [한국어]
   * GetOccupancy() - 버퍼 전체의 현재 플릿 점유 수 반환 (인라인)
   *
   * @return: 전체 occupancy (0 이상 _size 이하)
   *
   * 통계 수집 또는 상태 출력 시 전체 버퍼 사용률 확인에 사용.
   *
   * 호출 체인:
   *   통계 수집 루틴 → [Buffer::GetOccupancy()]
   */
  inline int GetOccupancy( ) const
  {
    return _occupancy; // [한국어] 전체 버퍼의 현재 플릿 총 점유 수 반환
  }

  /*
   * [한국어]
   * GetOccupancy(vc) - 지정 VC의 현재 플릿 점유 수 반환 (인라인)
   *
   * @vc: 조회할 VC 번호
   * @return: 해당 VC의 점유 수
   *
   * VC별 버퍼 사용률 확인, 디버그 출력 시 사용.
   *
   * 호출 체인:
   *   통계 수집 또는 Display() → [Buffer::GetOccupancy(vc)] → VC::GetOccupancy()
   */
  inline int GetOccupancy( int vc ) const
  {
    return _vc[vc]->GetOccupancy( ); // [한국어] 해당 VC의 현재 점유 플릿 수 반환
  }

#ifdef TRACK_BUFFERS
  /*
   * [한국어]
   * GetOccupancyForClass - 트래픽 클래스별 점유 수 반환 (인라인, TRACK_BUFFERS 한정)
   *
   * @c: 트래픽 클래스 번호 (0 이상 classes 미만)
   * @return: 해당 클래스의 현재 점유 플릿 수
   *
   * 멀티 클래스 트래픽에서 클래스별 버퍼 공정성 분석에 사용.
   * TRACK_BUFFERS 매크로가 정의된 빌드에서만 컴파일됨.
   *
   * 호출 체인:
   *   통계 수집 → [Buffer::GetOccupancyForClass(c)]
   */
  inline int GetOccupancyForClass(int c) const
  {
    return _class_occupancy[c]; // [한국어] 클래스 c의 현재 점유 플릿 수 반환
  }
#endif

  /*
   * [한국어]
   * Display - 버퍼 내 모든 VC의 상태를 출력 스트림에 덤프 (디버그용)
   *
   * @os: 출력 스트림 (기본값 cout)
   * @return: 없음
   *
   * 모든 VC의 Display()를 순서대로 호출하여 각 VC의 상태를 출력한다.
   * 시뮬레이션 중 데드락 또는 이상 동작 디버깅 시 사용.
   *
   * 호출 체인:
   *   라우터 디버그 → [Buffer::Display()] → VC::Display() (각 VC별)
   */
  void Display( ostream & os = cout ) const;
};

#endif
