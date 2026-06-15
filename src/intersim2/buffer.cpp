// $Id: buffer.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 라우터 입력 포트 버퍼 구현 (buffer.cpp)
 *
 * === 파일의 역할 ===
 * Buffer 클래스의 비인라인 멤버 함수(생성자, 소멸자, AddFlit, Display)를 구현한다.
 * Buffer는 라우터의 입력 포트에 존재하는 물리 버퍼를 모델링하며, 내부적으로
 * num_vcs개의 VC 객체를 관리한다. 이 파일은 생성/해제 및 플릿 삽입의
 * 검증 로직(오버플로 검사)과 디버그 출력을 담당한다.
 * 인라인 메서드(RemoveFlit, FrontFlit, Full 등)는 buffer.hpp에 구현되어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터 내부의 라우터 입력 버퍼 계층.
 * 라우터(IQRouter, KNCRouter) 생성 시 입력 포트 수만큼 Buffer 인스턴스를 생성한다.
 * 매 시뮬레이션 사이클의 ReadInputs() 단계에서 FlitChannel로부터 플릿이 도착하면
 * AddFlit()이 호출되고, WriteOutputs() 단계에서 RemoveFlit()이 호출된다.
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, 매 사이클마다 호출됨.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - globals.hpp: GetSimTime() 등 시뮬레이터 전역 함수/변수
 *   - booksim.hpp: Error() 매크로, 공통 typedef 등
 *   - buffer.hpp: Buffer 클래스 선언 (VC*, Flit* 포함)
 *   - vc.hpp: VC 클래스 — 개별 가상 채널 FIFO 큐 + 상태 머신
 *   - flit.hpp: Flit 클래스 — 플릿 객체 (cl 필드: 트래픽 클래스)
 * 이 파일을 사용하는 모듈:
 *   - iqrouter.cc, kncrouter.cc 등: Buffer 객체를 입력 포트별로 생성·사용
 *
 * === 주요 함수/구조체 요약 ===
 * Buffer::Buffer(): num_vcs개 VC 객체를 new로 생성, _size 계산
 * Buffer::~Buffer(): _vc 벡터의 모든 VC 포인터 delete
 * Buffer::AddFlit(): 오버플로 검사 후 플릿을 지정 VC에 삽입
 * Buffer::Display(): 모든 VC의 Display()를 순서대로 호출하여 상태 덤프
 */

#include <sstream> // [한국어] VC 이름 생성 시 ostringstream 사용 ("vc_0", "vc_1" 등)

#include "globals.hpp" // [한국어] GetSimTime() 등 시뮬레이터 전역 유틸리티
#include "booksim.hpp" // [한국어] Error() 매크로, 공통 헤더 (assert, cout 등)
#include "buffer.hpp"  // [한국어] Buffer 클래스 선언 — VC, Flit, OutputSet, routefunc 포함

/*
 * [한국어]
 * Buffer::Buffer - Buffer 객체 생성자: VC 배열 초기화 및 버퍼 크기 결정
 *
 * @config:  NoC 설정 객체 — "num_vcs", "buf_size", "vc_buf_size", "classes" 참조
 * @outputs: 이 라우터의 출력 포트 수 — VC 객체 생성 시 전달 (라우팅 테이블 초기화에 사용)
 * @parent:  모듈 계층 부모 (라우터 객체)
 * @name:    이 버퍼 모듈의 식별 이름 (예: "buf_0")
 * @return:  없음 (생성자)
 *
 * 초기화 목록에서 _occupancy를 0으로 설정한 뒤:
 *   1. "num_vcs" 읽어 VC 수 결정
 *   2. "buf_size" < 0이면 num_vcs * "vc_buf_size"로 전체 버퍼 크기 계산
 *   3. _vc 벡터를 num_vcs 크기로 resize 후 각 VC 객체를 new로 생성
 *   4. TRACK_BUFFERS 활성 시 클래스별 occupancy 카운터 초기화
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (GPU-Sim 사이클 루프 시작 전).
 *
 * 호출 체인:
 *   IQRouter::IQRouter() → [Buffer::Buffer()] → VC::VC() (num_vcs회)
 */
Buffer::Buffer( const Configuration& config, int outputs,
		Module *parent, const string& name ) :
Module( parent, name ), _occupancy(0) // [한국어] 모듈 계층 등록, 전체 점유 수 0으로 초기화
{
  int num_vcs = config.GetInt( "num_vcs" ); // [한국어] 설정에서 VC(가상 채널) 수 읽기 — 버퍼 내 VC 수 결정

  _size = config.GetInt("buf_size"); // [한국어] 버퍼 전체 슬롯 수 읽기 (음수이면 아래에서 재계산)
  if(_size < 0) { // [한국어] buf_size가 음수(-1)이면 VC당 크기 방식으로 전체 크기 계산
    _size = num_vcs * config.GetInt( "vc_buf_size" ); // [한국어] 각 VC에 vc_buf_size 슬롯씩 할당하여 합산
  };

  _vc.resize(num_vcs); // [한국어] VC 포인터 벡터를 num_vcs 크기로 확장 (초기값 nullptr)

  for(int i = 0; i < num_vcs; ++i) { // [한국어] VC 번호 0부터 num_vcs-1까지 순서대로 초기화
    ostringstream vc_name;            // [한국어] 각 VC의 이름 문자열을 동적으로 생성하는 스트림
    vc_name << "vc_" << i;           // [한국어] "vc_0", "vc_1", ... 형태의 이름 조합
    _vc[i] = new VC(config, outputs, this, vc_name.str( ) ); // [한국어] VC 객체 생성: config, 출력 포트 수, 부모(이 Buffer), 이름 전달
  }

#ifdef TRACK_BUFFERS
  int classes = config.GetInt("classes"); // [한국어] 트래픽 클래스 수 읽기 (TRACK_BUFFERS 활성 시만 사용)
  _class_occupancy.resize(classes, 0);   // [한국어] 클래스별 점유 카운터를 classes 크기로 초기화 (모두 0)
#endif
}

/*
 * [한국어]
 * Buffer::~Buffer - Buffer 소멸자: VC 객체 배열 메모리 해제
 *
 * @return: 없음 (소멸자)
 *
 * 생성자에서 new로 할당된 모든 VC 객체를 순회하며 delete한다.
 * STL 반복자를 사용해 벡터 전체를 한 번에 해제한다.
 * 실행 컨텍스트: 시뮬레이터 종료 단계.
 *
 * 호출 체인:
 *   IQRouter::~IQRouter() 또는 시뮬레이터 종료 → [Buffer::~Buffer()] → VC::~VC() (각 VC별)
 */
Buffer::~Buffer()
{
  for(vector<VC*>::iterator i = _vc.begin(); i != _vc.end(); ++i) { // [한국어] _vc 벡터의 모든 VC 포인터 순회
    delete *i; // [한국어] 각 VC 객체 해제 — 내부 플릿 큐·OutputSet 등 VC 소멸자가 정리
  }
}

/*
 * [한국어]
 * Buffer::AddFlit - 지정된 VC에 플릿을 삽입하고 전체 점유 수 증가
 *
 * @vc: 플릿을 삽입할 VC 번호 (0 이상 num_vcs 미만)
 * @f:  삽입할 플릿 포인터 (헤드/바디/테일 중 하나)
 * @return: 없음
 *
 * 버퍼 전체 점유율을 먼저 검사하여 오버플로를 방지하고,
 * 이상이 없으면 _occupancy를 증가시킨 뒤 VC::AddFlit()에 위임한다.
 * TRACK_BUFFERS 활성 시 f->cl 인덱스로 클래스별 카운터도 증가시킨다.
 * 실행 컨텍스트: 라우터의 ReadInputs() 단계 — 매 사이클마다 호출 가능.
 * 에러: _occupancy >= _size이면 "Flit buffer overflow." 에러 출력 후 종료.
 *
 * 호출 체인:
 *   IQRouter::ReadInputs() → [Buffer::AddFlit()] → VC::AddFlit()
 */
void Buffer::AddFlit( int vc, Flit *f )
{
  if(_occupancy >= _size) { // [한국어] 버퍼 전체가 이미 포화 상태이면 — 흐름 제어 누락 또는 로직 버그
    Error("Flit buffer overflow."); // [한국어] 오버플로 에러 출력 후 시뮬레이션 강제 종료
  }
  ++_occupancy; // [한국어] 전체 점유 카운터 1 증가 — 슬롯 하나 사용됨을 전체 수준에서 기록
  _vc[vc]->AddFlit(f); // [한국어] 지정 VC의 FIFO 큐 뒤에 플릿 삽입 (VC 내부 occupancy도 갱신)
#ifdef TRACK_BUFFERS
  ++_class_occupancy[f->cl]; // [한국어] 이 플릿의 트래픽 클래스(f->cl)별 점유 카운터 1 증가
#endif
}

/*
 * [한국어]
 * Buffer::Display - 모든 VC의 상태를 출력 스트림에 덤프 (디버그용)
 *
 * @os: 출력 스트림 (기본값 cout)
 * @return: 없음
 *
 * _vc 벡터의 모든 VC 객체에 대해 VC::Display(os)를 호출한다.
 * 각 VC의 플릿 큐 내용, 상태 머신 상태 등이 순서대로 출력된다.
 * 시뮬레이션 중 데드락이나 이상 동작 디버깅 시 특정 입력 포트 버퍼의
 * 전체 상태를 빠르게 확인하는 데 사용한다.
 * 실행 컨텍스트: 필요 시 수동 호출 (시뮬레이션 루프 외부에서도 가능).
 *
 * 호출 체인:
 *   디버그 루틴 또는 라우터 Display() → [Buffer::Display()] → VC::Display() (각 VC별)
 */
void Buffer::Display( ostream & os ) const
{
  for(vector<VC*>::const_iterator i = _vc.begin(); i != _vc.end(); ++i) { // [한국어] 모든 VC를 const 반복자로 순회
    (*i)->Display(os); // [한국어] 각 VC의 내부 상태(큐 내용, 상태 머신 등)를 os에 출력
  }
}
