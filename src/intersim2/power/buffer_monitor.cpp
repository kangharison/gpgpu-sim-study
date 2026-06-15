// $Id: buffer_monitor.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 입력 버퍼 접근 모니터 구현 (buffer_monitor.cpp)
 *
 * === 파일의 역할 ===
 * BufferMonitor 클래스의 생성자와 카운터 갱신/출력 함수를 구현한다.
 * IQRouter가 버퍼에 플릿을 쓰거나 읽을 때마다 트래픽 클래스별로 이벤트를 집계하며,
 * 시뮬레이션 종료 후 Power_Module이 이 카운터를 읽어 버퍼 전력/면적을 계산한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/buffer_monitor.cpp
 *     → routers/iq_router.cpp: 라우터의 버퍼 접근 시 write()/read() 호출
 *     → power/power_module.cpp: Power_Module::calcBuffer()가 GetReads()/GetWrites() 조회
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - buffer_monitor.hpp: 클래스 선언
 *   - flit.hpp: Flit::cl (트래픽 클래스) 접근
 * 피의존:
 *   - iq_router.cpp: 각 입력 포트의 버퍼 읽기/쓰기 이벤트 기록
 *   - power_module.cpp: 최종 전력/면적 산출
 * 설정 연동:
 *   - "classes"가 _classes 크기를 결정하고, 벡터 인덱스 계산에 사용됨
 *
 * === 주요 함수/구조체 요약 ===
 * BufferMonitor() - 카운터 벡터를 0으로 초기화
 * index()         - (입력 포트, 클래스) → 1차원 벡터 인덱스 변환 및 범위 검사
 * cycle()         - _cycles 증가 (현재는 단순 카운터)
 * write()/read()  - 이벤트에 해당하는 클래스 카운터 증가
 * display()       - 포트/클줄별 R/W 횟수 출력
 */

#include "buffer_monitor.hpp" // [한국어] BufferMonitor 클래스 선언 포함

#include "flit.hpp"           // [한국어] Flit::cl 멤버 접근을 위한 헤더

/*
 * [한국어]
 * 생성자 — 입력 포트 수와 클래스 수에 따라 read/write 카운터 벡터를 할당하고 0으로 초기화
 * @inputs  : 입력 포트 수
 * @classes : 트래픽 클래스 수
 */
BufferMonitor::BufferMonitor( int inputs, int classes ) 
: _cycles(0), _inputs(inputs), _classes(classes) { // [한국어] 멤버 초기화 리스트로 _cycles, _inputs, _classes 설정
  _reads.resize(inputs * classes, 0) ;  // [한국어] reads 벡터 크기를 inputs*classes로 설정, 초기값 0
  _writes.resize(inputs * classes, 0) ; // [한국어] writes 벡터 크기를 동일하게 설정, 초기값 0
}

/*
 * [한국어]
 * index() — (input, cl) 쌍을 _reads/_writes의 1차원 인덱스로 변환
 * 인덱스 공식: cl + _classes * input
 * 범위를 벗어나면 assert로 시뮬레이션 중단
 */
int BufferMonitor::index( int input, int cl ) const {
  assert((input >= 0) && (input < _inputs));  // [한국어] 입력 포트 범위 검사
  assert((cl >= 0) && (cl < _classes));       // [한국어] 클래스 범위 검사
  return cl + _classes * input ;              // [한국어] 2차원 좌표를 1차원으로 평탄화
}

/*
 * [한국어]
 * cycle() — 시뮬레이션 사이클이 한 단계 진행될 때 호출
 * 현재는 _cycles만 1 증가시키며, 카운터 정규화 등 추가 로직은 없다.
 */
void BufferMonitor::cycle() {
  _cycles++ ; // [한국어] 경과 사이클 수 증가
}

/*
 * [한국어]
 * write() — 입력 포트 input에 플릿 f가 버퍼로 쓰였을 때 호출
 * f->cl을 통해 트래픽 클래스를 식별하고 해당 writes 카운터를 증가시킨다.
 */
void BufferMonitor::write( int input, Flit const * f ) {
  _writes[ index(input, f->cl) ]++ ; // [한국어] (입력 포트, 플릿 클래스) 위치의 쓰기 카운터 증가
}

/*
 * [한국어]
 * read() — 입력 포트 input의 버퍼에서 플릿 f가 읽혔을 때 호출
 * f->cl을 통해 트래픽 클래스를 식별하고 해당 reads 카운터를 증가시킨다.
 */
void BufferMonitor::read( int input, Flit const * f ) {
  _reads[ index(input, f->cl) ]++ ; // [한국어] (입력 포트, 플릿 클래스) 위치의 읽기 카운터 증가
}

/*
 * [한국어]
 * display() — 입력 포트별/클줄별 read/write 카운터를 스트림에 출력
 * Power_Module은 GetReads()/GetWrites()를 통해 직접 수치를 읽지만,
 * 이 함수는 디버깅이나 로깅 용도로 사용된다.
 */
void BufferMonitor::display(ostream & os) const {
  for ( int i = 0 ; i < _inputs ; i++ ) { // [한국어] 모든 입력 포트 순회
    os << "[ " << i << " ] " ;             // [한국어] 현재 포트 번호 출력
    for ( int c = 0 ; c < _classes ; c++ ) { // [한국어] 모든 트래픽 클래스 순회
      os << "Type=" << c
	 << ":(R#" << _reads[ index( i, c) ]  << "," // [한국어] 읽기 횟수
	 << "W#" << _writes[ index( i, c) ] << ")" << " " ; // [한국어] 쓰기 횟수
    }
    os << endl ; // [한국어] 포트 단위로 줄바꿈
  }
}

/*
 * [한국어]
 * operator<<() — BufferMonitor 객체를 ostream에 출력하는 비멤버 연산자
 * 낶은 display()를 호출하여 포맷된 결과를 출력한다.
 */
ostream & operator<<( ostream & os, BufferMonitor const & obj ) {
  obj.display(os); // [한국어] display()에 출력 스트림 전달
  return os ;       // [한국어] ostream 참조 반환으로 << 연쇄 사용 가능
}
