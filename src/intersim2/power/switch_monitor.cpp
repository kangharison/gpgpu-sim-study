// $Id: switch_monitor.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 크로스바 스위칭 활동 모니터 구현 (switch_monitor.cpp)
 *
 * === 파일의 역할 ===
 * SwitchMonitor 클래스의 생성자와 카운터 갱신/출력 함수를 구현한다.
 * IQRouter가 크로스바를 통해 플릿을 전달할 때마다 (입력, 출력, 클래스) 단위로
 * 이벤트를 집계하며, Power_Module::calcSwitch()가 이를 읽어 스위치 전력/면적을 계산한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/switch_monitor.cpp
 *     → routers/iq_router.cpp: 크로스바 traversal 시 traversal() 호출
 *     → power/power_module.cpp: Power_Module::calcSwitch()가 GetActivity() 조회
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - switch_monitor.hpp: 클래스 선언
 *   - flit.hpp: Flit::cl (트래픽 클래스) 접근
 * 피의존:
 *   - iq_router.cpp: 크로스바 연결 설정 후 플릿 통과 시 이벤트 기록
 *   - power_module.cpp: 최종 크로스바 전력/면적 산출
 * 설정 연동:
 *   - "classes"가 _classes 크기를 결정하고, 벡터 인덱스 계산에 사용됨
 *
 * === 주요 함수/구조체 요약 ===
 * SwitchMonitor()     - 이벤트 카운터 벡터를 0으로 초기화
 * index()             - (입력, 출력, 클래스) → 1차원 벡터 인덱스 변환 및 범위 검사
 * cycle()             - _cycles 증가
 * traversal()         - (입력 → 출력) 스위칭 이벤트에 해당하는 클래스 카운터 증가
 * display()           - 입출력 포트/클줄별 이벤트 횟수 출력
 */

#include "switch_monitor.hpp" // [한국어] SwitchMonitor 클래스 선언 포함

#include "flit.hpp"           // [한국어] Flit::cl 멤버 접근을 위한 헤더

/*
 * [한국어]
 * 생성자 — 입력/출력/클줄 수를 받아 이벤트 카운터 벡터를 할당하고 0으로 초기화
 * @inputs  : 크로스바 입력 포트 수
 * @outputs : 크로스바 출력 포트 수
 * @classes : 트래픽 클래스 수
 */
SwitchMonitor::SwitchMonitor( int inputs, int outputs, int classes )
: _cycles(0), _inputs(inputs), _outputs(outputs), _classes(classes) { // [한국어] 멤버 초기화 리스트
  _event.resize(inputs * outputs * classes, 0) ; // [한국어] 입력x출력x클줄 크기로 벡터 설정, 초기값 0
}

/*
 * [한국어]
 * index() — (input, output, cl) 삼중 쌍을 _event의 1차원 인덱스로 변환
 * 인덱스 공식: cl + _classes * (output + _outputs * input)
 * 범위를 벗어나면 assert로 시뮬레이션 중단
 */
int SwitchMonitor::index( int input, int output, int cl ) const {
  assert((input >= 0) && (input < _inputs));    // [한국어] 입력 포트 범위 검사
  assert((output >= 0) && (output < _outputs)); // [한국어] 출력 포트 범위 검사
  assert((cl >= 0) && (cl < _classes));         // [한국어] 클래스 범위 검사
  return cl + _classes * ( output + _outputs * input ) ; // [한국어] 3차원 좌표를 1차원으로 평탄화
}

/*
 * [한국어]
 * cycle() — 시뮬레이션 사이클이 한 단계 진행될 때 호출
 * 현재는 _cycles만 1 증가시키며, 카운터 정규화 등 추가 로직은 없다.
 */
void SwitchMonitor::cycle() {
  _cycles++ ; // [한국어] 경과 사이클 수 증가
}

/*
 * [한국어]
 * traversal() — 플릿 f가 입력 포트 input에서 출력 포트 output으로 스위칭되었을 때 호출
 * f->cl을 통해 트래픽 클래스를 식별하고 해당 _event 카운터를 증가시킨다.
 */
void SwitchMonitor::traversal( int input, int output, Flit const * f ) {
  _event[ index( input, output, f->cl) ]++ ; // [한국어] (입력, 출력, 플릿 클래스) 위치의 카운터 증가
}

/*
 * [한국어]
 * display() — 입출력 포트/클줄별 스위칭 이벤트 횟수를 스트림에 출력
 * Power_Module은 GetActivity()를 통해 직접 수치를 읽지만,
 * 이 함수는 디버깅이나 로깅 용도로 사용된다.
 */
void SwitchMonitor::display(ostream & os) const {
  for ( int i = 0 ; i < _inputs ; i++ ) {      // [한국어] 모든 입력 포트 순회
    for ( int o = 0 ; o < _outputs ; o++) {    // [한국어] 모든 출력 포트 순회
      os << "[" << i << " -> " << o << "] " ; // [한국어] 현재 입출력 쌍 출력
      for ( int c = 0 ; c < _classes ; c++ ) { // [한국어] 모든 트래픽 클래스 순회
	os << c << ":" << _event[index(i,o,c)] << " " ; // [한국어] 클래스별 이벤트 횟수 출력
      }
      os << endl ; // [한국어] 입출력 쌍 단위로 줄바꿈
    }
  }
}

/*
 * [한국어]
 * operator<<() — SwitchMonitor 객체를 ostream에 출력하는 비멤버 연산자
 * 낶은 display()를 호출하여 포맷된 결과를 출력한다.
 */
ostream & operator<<( ostream & os, SwitchMonitor const & obj ) {
  obj.display(os); // [한국어] display()에 출력 스트림 전달
  return os ;       // [한국어] ostream 참조 반환으로 << 연쇄 사용 가능
}
