// $Id: buffer_monitor.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 입력 버퍼 접근 모니터 헤더 (buffer_monitor.hpp)
 *
 * === 파일의 역할 ===
 * IQ(Input-Queued) 라우터의 각 입력 포트별 버퍼(Buffer)에 대한
 * 읽기(read)/쓰기(write) 이벤트를 트래픽 클래스(traffic class) 단위로 집계한다.
 * BookSim2의 전력 모델이 실제 시뮬레이션 중 발생한 버퍼 접근 횟수를 바탕으로
 * 동적/정적 전력을 추정할 때 필요한 활동량(activity) 데이터를 수집하는 역할을 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/ (NoC 전력 모델 서브 모듈)
 *     → routers/iq_router.hpp: 각 IQRouter가 입력 포트 수만큼 BufferMonitor 소유
 *     → power/power_module.cpp: Power_Module::calcBuffer()가 수집된 카운터를 읽어
 *       입력 버퍼의 동적 읽기/쓰기 전력 및 누설 전력, 면적을 계산
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - flit.hpp (전방 선언 class Flit): 버퍼에 쓰이거나 읽힌 플릿의 클래스 정보(f->cl) 사용
 * 피의존:
 *   - routers/iq_router.hpp: 라우터 생성 시 BufferMonitor를 입력 포트별로 인스턴스화하고
 *     버퍼 쓰기/읽기 시점에 write()/read() 호출
 *   - power_module.cpp: 시뮬레이션 종료 후 calcBuffer()에서 GetReads()/GetWrites() 조회
 * 설정 연동:
 *   - BookSimConfig / gpgpusim.config 의 "classes" 가 트래픽 클래스 수(_classes)를 결정
 *   - "num_vcs", "vc_buf_size" 는 Power_Module에서 버퍼 깊이(depth)를 계산할 때 사용
 *
 * === 주요 함수/구조체 요약 ===
 * BufferMonitor      - 입력 포트 x 트래픽 클래스 크기의 read/write 카운터를 유지
 * index(input, cl) - 2차원 인덱스를 1차원 벡터 오프셋으로 변환
 * write()/read()   - 버퍼 쓰기/읽기 이벤트 발생 시 해당 클래스 카운터 증가
 * cycle()          - 남겨진 사이클 카운터(_cycles) 증가 (현재는 단순히 증가만 수행)
 * GetReads()/GetWrites() - 집계된 카운터 벡터 반환
 * display()        - 포트/클줄별 read/write 횟수를 스트림에 출력
 */

#ifndef _BUFFER_MONITOR_HPP_
#define _BUFFER_MONITOR_HPP_

#include <vector>    // [한국어] read/write 카운터를 저장할 vector<int>용 표준 헤더
#include <iostream>  // [한국어] display() 및 operator<<()에서 ostream 사용

using namespace std; // [한국어] BookSim2 전역 스타일로 std 네임스페이스 노출

class Flit; // [한국어] 버퍼 모니터는 Flit 클래스의 정의 없이도 포인터/참조만 사용하므로 전방 선언

/*
 * [한국어]
 * BufferMonitor — IQ 라우터 입력 버퍼의 클래스별 읽기/쓰기 활동량을 집계하는 카운터
 *
 * 각 IQRouter는 입력 포트마다 하나의 BufferMonitor를 가지며,
 * 버퍼에 플릿이 쓰일 때(write)와 읽힐 때(read)를 기록한다.
 * _reads와 _writes는 (input_port * classes + class) 형태의 1차원 벡터로 저장되며,
 * Power_Module이 이 값을 totalTime으로 나누어 activity factor를 구한다.
 */
class BufferMonitor {
  int  _cycles ;  // [한국어] 현재까지 진행된 시뮬레이션 사이클 수 (cycle() 호출 시 증가)
  int  _inputs ;  // [한국어] 모니터링할 입력 포트 수 (IQRouter의 입력 포트 수)
  int  _classes ; // [한국어] 트래픽 클래스 수; BookSimConfig "classes" 값이 저장됨
  vector<int> _reads ;  // [한국어] 입력 포트 x 클래스별 버퍼 읽기 횟수
  vector<int> _writes ; // [한국어] 입력 포트 x 클래스별 버퍼 쓰기 횟수
  int index( int input, int cl ) const ; // [한국어] (input, cl)을 _reads/_writes 벡터 인덱스로 변환
public:
  /*
   * [한국어]
   * 생성자 — 입력 포트 수와 트래픽 클래스 수를 받아 카운터 벡터를 0으로 초기화
   *
   * @inputs  : 모니터링 대상 입력 포트 수
   * @classes : 트래픽 클래스 수 (보통 BookSimConfig "classes" 값)
   */
  BufferMonitor( int inputs, int classes ) ;

  /*
   * [한국어]
   * cycle() — 시뮬레이션 사이클이 진행될 때마다 호출
   * 현재 구현에서는 _cycles를 1 증가시키는 단순 카운터 역할만 수행한다.
   */
  void cycle() ;

  /*
   * [한국어]
   * write() — 입력 포트 input의 버퍼에 플릿 f가 쓰였음을 기록
   * f->cl 클래스에 해당하는 _writes 카운터를 1 증가시킨다.
   */
  void write( int input, Flit const * f ) ;

  /*
   * [한국어]
   * read() — 입력 포트 input의 버퍼에서 플릿 f가 읽혔음을 기록
   * f->cl 클래스에 해당하는 _reads 카운터를 1 증가시킨다.
   */
  void read( int input, Flit const * f ) ;

  /* [한국어] 클래스별 버퍼 읽기 횟수 벡터를 상수 참조로 반환 */
  inline const vector<int> & GetReads() const {
    return _reads;
  }
  /* [한국어] 클래스별 버퍼 쓰기 횟수 벡터를 상수 참조로 반환 */
  inline const vector<int> & GetWrites() const {
    return _writes;
  }
  /* [한국어] 모니터링 중인 입력 포트 수 반환 */
  inline int NumInputs() const {
    return _inputs;
  }
  /* [한국어] 트래픽 클래스 수 반환 */
  inline int NumClasses() const {
    return _classes;
  }

  /*
   * [한국어]
   * display() — 포트/클줄별 read/write 횟수를 사람이 읽기 쉬운 형태로 출력
   * Power_Module은 직접 이 함수를 사용하지 않고, operator<<()나 디버깅 용도로 활용
   */
  void display(ostream & os) const;

} ;

/* [한국어] BufferMonitor를 ostream에 출력하기 위한 비멤버 연산자 */
ostream & operator<<( ostream & os, BufferMonitor const & obj ) ;

#endif
