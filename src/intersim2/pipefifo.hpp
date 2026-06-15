// $Id: pipefifo.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 파이프라인 FIFO 템플릿 (pipefifo.hpp)
 *
 * === 파일의 역할 ===
 * PipelineFIFO<T>는 NoC(네트워크온칩) 시뮬레이터 intersim2 내부에서
 * 파이프라인 레지스터 스테이지(pipeline register stage)를 모델링하기 위한
 * 제네릭 템플릿 클래스이다. 실제 하드웨어 파이프라인과 마찬가지로 데이터를
 * 쓴 뒤 한 사이클이 지나야 읽을 수 있도록, 링 버퍼 방식의 지연(delay) 큐를
 * 구현한다. 라우터·스위치·중재기(arbiter) 등 사이클 정확(cycle-accurate)
 * 컴포넌트들이 이 클래스를 이용해 스테이지 간 플릿(flit)/크레딧(credit)
 * 전달 지연을 정확하게 표현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 시뮬레이터의 라우터·링크·트래픽 매니저는 각자
 * 내부에 PipelineFIFO 인스턴스를 보유하며, 매 사이클마다:
 *   1) Write()로 현재 사이클의 데이터를 기록,
 *   2) Read()로 'depth' 사이클 전에 쓰인 데이터를 꺼냄,
 *   3) Advance()로 링 버퍼 포인터를 한 칸 전진시킨다.
 * 이를 통해 파이프라인 스테이지 수(depth)만큼 정확히 지연된 값이 나오므로
 * 하드웨어의 플립플롭 체인을 소프트웨어로 정밀 모방한다.
 *
 * 호출 체인:
 *   TrafficManager::Step() → Router::ReadInputs/Evaluate/WriteOutputs()
 *     → PipelineFIFO::Write() / PipelineFIFO::Read() / PipelineFIFO::Advance()
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프 내부).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - module.hpp : 모든 intersim2 컴포넌트의 공통 베이스 클래스 Module을 제공.
 *   - <vector>   : 2차원 링 버퍼 저장에 사용.
 * 이 파일을 사용하는 모듈:
 *   - router/*.cc : 각 라우터 구현이 파이프라인 스테이지마다 플릿/크레딧
 *                   지연 큐로 사용.
 *   - trafficmanager.cc : 주입(injection) 파이프라인 모델에 사용 가능.
 * 데이터 흐름:
 *   상위 컴포넌트가 Write(val) → _data[lane][_pipe_ptr] 에 저장
 *   → Advance() 호출로 _pipe_ptr 이동
 *   → 다음 호출자가 Read()로 depth 사이클 전 값 획득.
 *
 * === 주요 함수/구조체 요약 ===
 * - PipelineFIFO(parent, name, lanes, depth) : 생성자. 2D 링 버퍼 초기화.
 * - Write(val, lane) : 현재 사이클에 특정 레인(lane)에 데이터 기록.
 * - WriteAll(val)    : 모든 레인에 동일한 값 기록 (브로드캐스트).
 * - Read(lane)       : 현재 포인터 위치(= depth 사이클 전 Write 지점)에서 읽기.
 * - Advance()        : 링 버퍼 포인터를 1 사이클 전진 (매 사이클 호출 필수).
 */

#ifndef _PIPEFIFO_HPP_
#define _PIPEFIFO_HPP_

#include <vector>  // [한국어] 2차원 링 버퍼(_data) 구현에 필요한 STL 벡터

#include "module.hpp"  // [한국어] intersim2 Module 베이스 클래스 — 계층적 이름/에러 출력 지원

/*
 * [한국어]
 * PipelineFIFO<T> — 멀티 레인 파이프라인 레지스터 지연 큐 템플릿
 *
 * 하드웨어 파이프라인의 플립플롭 체인을 소프트웨어로 모사하는 링 버퍼 클래스.
 * T는 주로 Flit*(플릿 포인터) 또는 Credit*(크레딧 포인터)로 인스턴스화된다.
 *
 * 동작 원리:
 *   depth=2 레인=1 예시: _pipe_len = 3 (depth+1)
 *   사이클 0: Write(A) → slot[0]=A,  Advance() → ptr=1
 *   사이클 1: Write(B) → slot[1]=B,  Advance() → ptr=2
 *   사이클 2: Read()   → slot[2]=null, Write(C) → slot[2]=C, Advance() → ptr=0
 *   사이클 3: Read()   → slot[0]=A  ← 2사이클 전에 쓴 A가 출력됨
 *
 * 동기화: intersim2는 단일 호스트 스레드에서 실행되므로 별도 락 불필요.
 */
template<class T> class PipelineFIFO : public Module {
  int _lanes;
  /* 병렬 레인 수 — 예를 들어 라우터의 포트 수(number of ports)만큼
   * 레인을 두어 각 포트의 파이프라인 데이터를 독립적으로 관리한다.
   * 설정자: 생성자 인자 lanes로 초기화되며 이후 변경 없음.
   * 읽는 자: Write/WriteAll/Read 메서드가 인덱스 범위로 사용.
   * 값 범위: 1 이상 (일반적으로 라우터 포트 수 = 입력 포트 수).
   * 동기화: 생성 후 불변 → 락 불필요. */

  int _depth;
  /* 파이프라인 깊이(스테이지 수) — Write()한 데이터가 Read()로 나오기까지
   * 걸리는 사이클 수를 결정한다. depth=1이면 1사이클 지연, depth=2이면
   * 2사이클 지연을 의미한다. 링 버퍼 크기는 _pipe_len = _depth + 1이다.
   * 설정자: 생성자 인자 depth로 초기화되며 이후 변경 없음.
   * 읽는 자: 생성자 내에서 _pipe_len 계산에 사용됨.
   * 값 범위: 1 이상 (0이면 즉시 통과, 의미 없음).
   * 동기화: 생성 후 불변 → 락 불필요. */

  int _pipe_len;
  /* 링 버퍼의 실제 슬롯 수 = _depth + 1.
   * depth+1개의 슬롯이 필요한 이유: Write와 Read가 동시에 같은 슬롯에
   * 접근하지 않도록 "쓰기 슬롯"과 "읽기 슬롯" 사이에 depth개의 중간
   * 슬롯이 항상 존재해야 하기 때문이다.
   * 설정자: 생성자에서 depth+1로 고정 설정.
   * 읽는 자: Write/Read/Advance에서 배열 크기로 사용.
   * 값 범위: depth+1 (양의 정수).
   * 동기화: 생성 후 불변 → 락 불필요. */

  int _pipe_ptr;
  /* 링 버퍼의 현재 "쓰기 위치" 포인터 (0 ~ _pipe_len-1 순환).
   * Advance()가 호출될 때마다 (ptr+1) % _pipe_len 으로 전진한다.
   * Write()는 _pipe_ptr 위치에 저장하고,
   * Read()는 _pipe_ptr 위치에서 읽는다 (= depth 사이클 전에 Write된 슬롯).
   * 설정자: 생성자에서 0으로 초기화, Advance()에서 매 사이클 갱신.
   * 읽는 자: Write(), WriteAll(), Read() 모두 이 값으로 슬롯 결정.
   * 값 범위: [0, _pipe_len-1].
   * 동기화: 단일 스레드 실행 → 락 불필요. */

  vector<vector<T*> > _data;
  /* 2차원 링 버퍼 본체 — _data[lane][slot] 형태.
   * 외부 벡터 크기 = _lanes (각 레인별 독립 파이프라인),
   * 내부 벡터 크기 = _pipe_len (링 버퍼 슬롯 배열).
   * 각 원소는 T* 포인터 (일반적으로 Flit* 또는 Credit*).
   * nullptr(0)은 "빈 슬롯"을 의미한다.
   * 설정자: 생성자에서 0으로 초기화; Write()/WriteAll()이 값 저장.
   * 읽는 자: Read()가 현재 포인터 위치의 값을 반환.
   * 값 범위: T* (유효 포인터 또는 nullptr).
   * 동기화: 단일 스레드 실행 → 락 불필요. */

public:
  PipelineFIFO( Module *parent, const string& name, int lanes, int depth );
  ~PipelineFIFO( );

  void Write( T* val, int lane = 0 );
  void WriteAll( T* val );

  T*   Read( int lane = 0 );

  void Advance( );
};

/*
 * [한국어]
 * PipelineFIFO<T>::PipelineFIFO — 생성자
 *
 * @parent: Module 계층 트리의 부모 노드 (라우터나 네트워크 객체). 이름 출력
 *          및 오류 메시지 계층화에 사용됨.
 * @name  : 이 FIFO 인스턴스의 식별 이름 (예: "credit_pipe"). 디버그/통계 출력에 사용.
 * @lanes : 병렬 레인 수. 라우터의 포트 수만큼 생성하는 것이 일반적.
 * @depth : 파이프라인 지연 깊이 (사이클 수). Write 후 Read까지의 지연.
 * @return: (생성자, 반환값 없음)
 *
 * 동작 과정:
 *   1) Module 베이스 클래스 생성자 호출(이름/부모 등록).
 *   2) _pipe_len = depth + 1 계산 (링 버퍼 슬롯 수).
 *   3) _pipe_ptr = 0 으로 초기화.
 *   4) _data를 [lanes][pipe_len] 크기로 할당하고 모두 nullptr(0)으로 초기화.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 단일 스레드.
 * 호출 체인: Router::Router() → PipelineFIFO<Flit*>::PipelineFIFO(...)
 */
template<class T> PipelineFIFO<T>::PipelineFIFO( Module *parent,
						 const string& name,
						 int lanes, int depth ) :
  Module( parent, name ),           // [한국어] Module 베이스 클래스 초기화: 계층 이름과 부모 포인터 등록
  _lanes( lanes ), _depth( depth )  // [한국어] 레인 수와 파이프라인 깊이 멤버 초기화
{
  _pipe_len = depth + 1; // [한국어] 링 버퍼 슬롯 수 = 깊이+1 (읽기·쓰기 슬롯 분리에 필요한 최소 크기)
  _pipe_ptr = 0;         // [한국어] 링 버퍼 현재 포인터를 슬롯 0에서 시작

  _data.resize(_lanes);                       // [한국어] 외부 벡터를 레인 수만큼 확장 (각 레인이 독립적인 링 버퍼를 가짐)
  for ( int l = 0; l < _lanes; ++l ) {        // [한국어] 각 레인에 대해 순회
    _data[l].resize(_pipe_len, 0);            // [한국어] 해당 레인의 슬롯 배열을 pipe_len 크기로 초기화, 모든 슬롯을 nullptr(0)으로 설정
  }
}

/*
 * [한국어]
 * PipelineFIFO<T>::~PipelineFIFO — 소멸자
 *
 * @return: (소멸자, 반환값 없음)
 *
 * 현재 구현은 빈 본체이다. _data 벡터는 STL 벡터이므로 자동으로 해제되며,
 * 저장된 T* 포인터가 가리키는 객체의 메모리 해제는 소유권을 가진 상위
 * 컴포넌트(라우터 등)가 담당한다. 따라서 여기서는 명시적 delete 불필요.
 *
 * 실행 컨텍스트: 시뮬레이터 종료 단계, 단일 스레드.
 * 호출 체인: Router::~Router() → ~PipelineFIFO()
 */
template<class T> PipelineFIFO<T>::~PipelineFIFO( )
{
  // [한국어] 본체 없음 — _data 벡터는 RAII에 의해 자동 소멸됨.
  // T* 포인터가 가리키는 Flit/Credit 객체는 상위 소유자가 별도 해제함.
}

/*
 * [한국어]
 * PipelineFIFO<T>::Write — 특정 레인의 현재 사이클 슬롯에 값 기록
 *
 * @val : 저장할 T 타입 포인터 (예: Flit*, Credit*, nullptr).
 *         nullptr를 전달하면 해당 슬롯을 "빈 상태"로 표시할 수 있다.
 * @lane: 쓸 레인 인덱스 (기본값 0). 0 ≤ lane < _lanes 이어야 한다.
 *         범위 초과 시 STL 벡터의 []연산이 미정의 동작(UB)을 유발한다.
 * @return: void
 *
 * 현재 사이클의 _pipe_ptr 슬롯에 값을 기록한다. Read()는 같은
 * _pipe_ptr 슬롯을 읽으므로, Advance()가 호출된 이후에는 이 슬롯이
 * depth 사이클 전 Write 결과가 된다. 즉, Write → Advance가 depth회
 * 반복된 뒤에야 Read()에서 이 값이 나온다.
 *
 * 실행 컨텍스트: 라우터의 WriteOutputs() 단계, 매 사이클 호출.
 * 호출 체인: Router::WriteOutputs() → PipelineFIFO::Write()
 */
template<class T> void PipelineFIFO<T>::Write( T* val, int lane )
{
  _data[lane][_pipe_ptr] = val; // [한국어] 지정된 레인의 현재 포인터 슬롯에 val 저장.
                                 // 같은 슬롯을 Read()도 사용하므로, Advance() 전에
                                 // Write와 Read 순서는 호출자가 명시적으로 관리해야 함.
}

/*
 * [한국어]
 * PipelineFIFO<T>::WriteAll — 모든 레인에 동일한 값 브로드캐스트 기록
 *
 * @val : 모든 레인의 현재 슬롯에 저장할 T 타입 포인터.
 * @return: void
 *
 * 단일 제어 신호(예: 유효 비트, 공통 크레딧)를 모든 레인에 동시에 전달할 때
 * 사용한다. 내부적으로 Write(val, lane)을 순차적으로 호출하는 것과 동일하지만,
 * 호출자 코드를 단순화하기 위한 편의 메서드이다.
 *
 * 실행 컨텍스트: 라우터의 WriteOutputs() 단계, 매 사이클 호출 가능.
 * 호출 체인: Router::WriteOutputs() → PipelineFIFO::WriteAll()
 */
template<class T> void PipelineFIFO<T>::WriteAll( T* val )
{
  for ( int l = 0; l < _lanes; ++l ) { // [한국어] 전체 레인 순회 (0 ~ _lanes-1)
    _data[l][_pipe_ptr] = val;          // [한국어] 각 레인의 현재 포인터 슬롯에 동일한 val 저장
  }
}

/*
 * [한국어]
 * PipelineFIFO<T>::Read — 특정 레인의 현재 포인터 슬롯에서 값 읽기
 *
 * @lane  : 읽을 레인 인덱스 (기본값 0). 0 ≤ lane < _lanes 이어야 한다.
 * @return: 현재 _pipe_ptr 슬롯에 저장된 T* 값.
 *           이 값은 depth 사이클 전에 Write()된 데이터이다.
 *           슬롯이 비어 있으면 nullptr(0)을 반환한다.
 *
 * 동작 원리:
 *   _pipe_ptr는 Advance()에 의해 매 사이클 전진한다. 링 버퍼이므로
 *   현재 _pipe_ptr 위치는 depth 사이클 전 Write() 슬롯과 일치한다.
 *   즉, Read()는 "depth 사이클 전에 기록된 값"을 반환한다.
 *
 * 실행 컨텍스트: 라우터의 ReadInputs() 단계, 매 사이클 호출.
 * 호출 체인: Router::ReadInputs() → PipelineFIFO::Read()
 */
template<class T> T* PipelineFIFO<T>::Read( int lane )
{
  return _data[lane][_pipe_ptr]; // [한국어] 지정 레인의 현재 포인터 슬롯 값 반환.
                                  // depth 사이클 전 Write() 결과가 여기에 있음.
                                  // nullptr이면 해당 슬롯에 데이터가 없었음을 의미.
}

/*
 * [한국어]
 * PipelineFIFO<T>::Advance — 링 버퍼 포인터를 1 사이클 전진
 *
 * @return: void
 *
 * 매 시뮬레이션 사이클의 끝에 반드시 호출해야 한다. _pipe_ptr을
 * (ptr+1) % _pipe_len 으로 갱신하여 링 버퍼를 순환시킨다.
 * 이 호출을 빠뜨리면 파이프라인 지연이 올바르게 동작하지 않는다.
 *
 * 설계 의도:
 *   Advance()를 명시적으로 분리한 이유는 ReadInputs → Evaluate → WriteOutputs
 *   3단계 실행 모델에서, 같은 사이클 내 Read와 Write가 올바른 순서로 동작한 뒤
 *   포인터를 이동하도록 제어권을 상위 루프에 위임하기 위해서다.
 *
 * 실행 컨텍스트: 라우터/트래픽 매니저의 Step() 루프 끝, 매 사이클 1회 호출.
 * 호출 체인: TrafficManager::Step() → Router::Advance() → PipelineFIFO::Advance()
 */
template<class T> void PipelineFIFO<T>::Advance( )
{
  _pipe_ptr = ( _pipe_ptr + 1 ) % _pipe_len; // [한국어] 포인터를 1 증가 후 _pipe_len으로 모듈러 연산하여 링 버퍼 순환.
                                               // _pipe_len = depth+1 이므로 depth 사이클 후 같은 슬롯으로 돌아옴.
}

#endif  // [한국어] _PIPEFIFO_HPP_ 헤더 가드 종료 — 중복 include 방지
