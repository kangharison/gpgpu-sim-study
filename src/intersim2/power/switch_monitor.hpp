// $Id: switch_monitor.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 크로스바(crossbar) 스위칭 활동 모니터 헤더 (switch_monitor.hpp)
 *
 * === 파일의 역할 ===
 * IQRouter 낶의 크로스바 스위치에서 발생하는 (입력 포트 → 출력 포트) 트래픽 클래스별
 * 스위칭 이벤트를 집계한다. Power_Module은 이 활동량을 바탕으로 크로스바 전달 전력,
 * 제어 전력, 누설 전력, 출력 포트 DFF/클록 전력을 추정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/ (NoC 전력 모델 서브 모듈)
 *     → routers/iq_router.hpp: 각 IQRouter가 입력x출력 크기의 SwitchMonitor 소유
 *     → power/power_module.cpp: Power_Module::calcSwitch()가 수집된 카운터를 읽어
 *       크로스바 전력/면적 계산
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - flit.hpp (전방 선언 class Flit): 스위칭 이벤트의 플릿 클래스(f->cl) 사용
 * 피의존:
 *   - routers/iq_router.cpp: 크로스바 통과(traversal) 시 traversal() 호출
 *   - power_module.cpp: 시뮬레이션 종료 후 GetActivity() 조회
 * 설정 연동:
 *   - BookSimConfig / gpgpusim.config 의 "classes" 가 트래픽 클래스 수(_classes)를 결정
 *   - 라우터 입력/출출 포트 수는 Network 토폴로지 및 "k", "n" 등에 의해 결정됨
 *
 * === 주요 함수/구조체 요약 ===
 * SwitchMonitor           - 입력x출력x클줄 크기의 스위칭 이벤트 카운터 유지
 * index(in,out,cl)        - 3차원 인덱스를 1차원 벡터 오프셋으로 변환
 * traversal(in,out,f)     - 플릿 f가 입력 in에서 출력 out으로 스위칭되었음을 기록
 * GetActivity()           - 집계된 카운터 벡터 반환
 * display()               - 입출력 포트/클줄별 이벤트 횟수 출력
 */

#ifndef _SWITCH_MONITOR_HPP_
#define _SWITCH_MONITOR_HPP_

#include <vector>    // [한국어] 스위칭 이벤트 카운터를 저장할 vector<int>용 표준 헤더
#include <iostream>  // [한국어] display() 및 operator<<()에서 ostream 사용

using namespace std; // [한국어] BookSim2 전역 스타일로 std 네임스페이스 노출

class Flit; // [한국어] SwitchMonitor는 Flit 클래스의 정의 없이도 포인터만 사용하므로 전방 선언

/*
 * [한국어]
 * SwitchMonitor — IQ 라우터 크로스바의 클래스별 스위칭 활동량을 집계하는 카운터
 *
 * 크로스바는 입력 포트 중 하나를 출력 포트 중 하나로 연결하는 스위치 행렬이다.
 * _event 벡터는 (input, output, class) 별로 해당 연결을 통과한 플릿 수를 기록하며,
 * Power_Module::calcSwitch()가 이를 읽어 activity factor를 계산한다.
 */
class SwitchMonitor {
  int  _cycles ;  // [한국어] 현재까지 진행된 시뮬레이션 사이클 수 (cycle() 호출 시 증가)
  int  _inputs ;  // [한국어] 크로스바 입력 포트 수
  int  _outputs ; // [한국어] 크로스바 출력 포트 수
  int  _classes ; // [한국어] 트래픽 클래스 수; BookSimConfig "classes" 값이 저장됨
  vector<int> _event ; // [한국어] 입력 x 출력 x 클래스 크기의 스위칭 이벤트 카운터
  int index( int input, int output, int cl ) const ; // [한국어] 3차원 인덱스 → 1차원 벡터 인덱스 변환
public:
  /*
   * [한국어]
   * 생성자 — 입력/출력/클줄 수를 받아 이벤트 카운터 벡터를 0으로 초기화
   * @inputs  : 크로스바 입력 포트 수
   * @outputs : 크로스바 출력 포트 수
   * @classes : 트래픽 클래스 수
   */
  SwitchMonitor( int inputs, int outputs, int classes ) ;

  /*
   * [한국어]
   * cycle() — 시뮬레이션 사이클이 진행될 때마다 호출
   * 현재 구현에서는 _cycles를 1 증가시키는 단순 카운터 역할만 수행한다.
   */
  void cycle() ;

  /* [한국어] 집계된 스위칭 이벤트 카운터 벡터를 상수 참조로 반환 */
  vector<int> const & GetActivity() const {
    return _event;
  }
  /* [한국어] 크로스바 입력 포트 수 반환 */
  inline int const & NumInputs() const {
    return _inputs;
  }
  /* [한국어] 크로스바 출력 포트 수 반환 */
  inline int const & NumOutputs() const {
    return _outputs;
  }
  /* [한국어] 트래픽 클래스 수 반환 */
  inline int const & NumClasses() const {
    return _classes;
  }

  /*
   * [한국어]
   * traversal() — 플릿 f가 입력 포트 input에서 출력 포트 output으로 스위칭되었음을 기록
   * f->cl 클래스에 해당하는 _event 카운터를 1 증가시킨다.
   */
  void traversal( int input, int output, Flit const * f ) ;

  /*
   * [한국어]
   * display() — 입출력 포트/클줄별 이벤트 횟수를 스트림에 출력
   */
  void display(ostream & os) const;
} ;

/* [한국어] SwitchMonitor를 ostream에 출력하기 위한 비멤버 연산자 */
ostream & operator<<( ostream & os, SwitchMonitor const & obj ) ;

#endif
