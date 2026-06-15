// $Id: stats.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*stats.cpp
 *
 *class stores statistics gnerated by the trafficmanager such as the latency
 *hope count of the the flits
 *
 *reset option resets the min and max alues of this statistiscs
 */

/*
 * [한국어 설명] 시뮬레이션 통계 수집 클래스 구현 (stats.cpp)
 *
 * === 파일의 역할 ===
 * stats.hpp에서 선언된 Stats 클래스의 모든 메서드를 구현한다. 생성자·
 * Clear·AddSample·Average·Variance·Min·Max·Sum·SquaredSum·NumSamples·
 * Display·operator<< 등 전체 통계 기능을 제공한다. TrafficManager가
 * 플릿(flit) 완료 시마다 AddSample()을 호출하여 레이턴시·홉 수 등을
 * 누적하고, 시뮬레이션 종료 후 Display()로 히스토그램을 출력한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 성능 측정 계층의 구현부. TrafficManager가 유일한 주요 호출자이며,
 * 개별 라우터(router/)도 내부 큐 통계를 위해 사용한다.
 *
 * 호출 체인:
 *   TrafficManager::_RetireFlit()
 *     → Stats::AddSample(latency) / Stats::AddSample(hops)
 *   TrafficManager::DisplayStats()
 *     → Stats::Display(cout)
 *         → operator<<(os, stats)
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - booksim.hpp   : BookSim 공통 매크로·타입 정의.
 *   - stats.hpp     : Stats 클래스 선언.
 *   - <iostream>    : ostream, cout.
 *   - <sstream>     : (현재 미사용, 과거 포맷용으로 남음).
 *   - <limits>      : numeric_limits<double>::quiet_NaN() — min/max 초기화.
 *   - <cmath>       : fmax(), floor() — 히스토그램 빈 계산.
 *   - <cstdio>      : (현재 미사용, 과거 printf용으로 남음).
 * 이 파일에 의존하는 모듈:
 *   - trafficmanager.cc, router/*.cc 등 Stats 인스턴스를 보유하는 모든 모듈.
 *
 * === 주요 함수/구조체 요약 ===
 * - Stats()         : Module 초기화 + _num_bins/_bin_size 설정 + Clear().
 * - Clear()         : 카운터·히스토그램·min/max 초기화.
 * - AddSample(val)  : 샘플 추가 — 합계·제곱합·min/max·히스토그램 갱신.
 * - Average()       : _sample_sum / _num_samples.
 * - Variance()      : (n·Σval² − (Σval)²) / n².
 * - operator<<      : "[ b0 b1 ... bN ]" 형식 히스토그램 출력.
 */

#include "booksim.hpp"  // [한국어] BookSim 공통 헤더 — 전처리 매크로, assert, 공통 타입 포함
#include <iostream>     // [한국어] ostream, cout — Display/operator<< 출력에 필요
#include <sstream>      // [한국어] stringstream — 현재 미사용이나 과거 포맷 코드 잔재로 유지
#include <limits>       // [한국어] numeric_limits<double>::quiet_NaN() — min/max 초기값 설정에 사용
#include <cmath>        // [한국어] fmax(), floor() — 히스토그램 빈 인덱스 계산에 사용
#include <cstdio>       // [한국어] printf() — 현재 미사용이나 과거 출력 코드 잔재로 유지

#include "stats.hpp"    // [한국어] Stats 클래스 선언 포함 — 구현 메서드의 원형 확인

/*
 * [한국어]
 * Stats::Stats — 통계 수집기 생성자
 *
 * @parent  : Module 계층 트리의 부모 노드 (TrafficManager 등).
 *             Module 베이스 초기화에 전달되어 이름 계층 등록에 사용.
 * @name    : 이 Stats 인스턴스의 식별 이름 (예: "packet_latency").
 *             Display() 출력 및 디버그 메시지에 사용.
 * @bin_size: 히스토그램 빈 너비 (기본값 1.0). 단위: 샘플 값의 단위와 동일.
 *             예: 레이턴시(사이클) 측정 시 bin_size=1이면 1사이클 단위로 집계.
 * @num_bins: 히스토그램 빈 수 (기본값 10). 분해능과 메모리 사용량 결정.
 * @return  : (생성자)
 *
 * 동작 과정:
 *   1) Module(parent, name) 베이스 생성자 호출 → 이름/부모 등록.
 *   2) _num_bins, _bin_size 멤버 초기화.
 *   3) Clear() 호출 → 모든 카운터 초기 상태로 설정.
 *
 * 실행 컨텍스트: TrafficManager 생성 시 (시뮬레이터 초기화 단계), 단일 스레드.
 * 호출 체인: TrafficManager::TrafficManager() → new Stats(this, "plat", ...)
 */
Stats::Stats( Module *parent, const string &name,
	      double bin_size, int num_bins ) :
  Module( parent, name ), _num_bins( num_bins ), _bin_size( bin_size )
  // [한국어] Module 베이스 생성자 호출: 계층 이름 등록.
  // _num_bins와 _bin_size는 초기화 목록에서 직접 설정 (Clear()에서 사용하므로 먼저 초기화 필요).
{
  Clear(); // [한국어] 모든 통계 카운터를 초기 상태로 리셋.
            // _hist 벡터 크기 조정 포함 (Clear() 내에서 _num_bins 기반 assign 호출).
}

/*
 * [한국어]
 * Stats::Clear — 모든 통계 카운터와 히스토그램 초기화
 *
 * @return: void
 *
 * 동작 과정:
 *   1) _num_samples = 0 : 샘플 카운터 리셋.
 *   2) _sample_sum = 0.0, _sample_squared_sum = 0.0 : 합계·제곱합 리셋.
 *   3) _hist.assign(_num_bins, 0) : 히스토그램 모든 빈 0으로 초기화.
 *   4) _min = quiet_NaN() : 최솟값을 "샘플 없음" NaN으로 설정.
 *   5) _max = -quiet_NaN() : 최댓값도 NaN으로 설정.
 *      (NaN의 부정도 NaN이므로 실질적으로 동일한 NaN 초기화)
 *
 * 웜업(warm-up) 사이클 제외 후 새 측정 구간 시작 시 TrafficManager가 호출.
 *
 * 실행 컨텍스트: 초기화 및 웜업 종료 시점, 단일 스레드.
 * 호출 체인: Stats::Stats() → Clear()
 *             TrafficManager::_ClearStats() → Stats::Clear()
 */
void Stats::Clear( )
{
  _num_samples = 0;          // [한국어] 누적 샘플 수를 0으로 리셋 — 새 측정 구간 시작
  _sample_sum  = 0.0;        // [한국어] 샘플 합계를 0으로 리셋
  _sample_squared_sum = 0.0; // [한국어] 샘플 제곱합을 0으로 리셋 (Variance 계산에 사용)

  _hist.assign(_num_bins, 0); // [한국어] 히스토그램 벡터를 _num_bins 크기로 재할당하고 모두 0으로 초기화.
                               // assign()은 현재 벡터 크기와 관계없이 항상 정확히 _num_bins개 원소를 생성.

  _min = numeric_limits<double>::quiet_NaN();  // [한국어] 최솟값 초기화: quiet_NaN()은 샘플 없음을 의미.
                                                // quiet NaN은 연산 시 예외를 발생시키지 않으며(IEEE 754),
                                                // AddSample()의 !(val >= _min) 조건이 NaN과 비교 시
                                                // 항상 true가 되어 첫 샘플이 자동으로 min이 됨.
  _max = -numeric_limits<double>::quiet_NaN(); // [한국어] 최댓값 초기화: NaN의 부정도 NaN (IEEE 754).
                                                // 동일하게 !(val <= _max) 조건이 첫 샘플에서 true가 되어
                                                // 첫 샘플이 자동으로 max로 설정됨.

  //  _reset = true;  // [한국어] 리셋 플래그 — 현재 주석 처리됨. NaN 초기화 방식으로 대체.
}

/*
 * [한국어]
 * Stats::Average — 샘플 산술 평균 반환
 *
 * @return: double = _sample_sum / _num_samples.
 *           _num_samples == 0 이면 0.0/0 = NaN 또는 부동소수점 예외 가능.
 *           호출 전 NumSamples() > 0 확인 권장.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 출력 단계, 단일 스레드.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Average()
 */
double Stats::Average( ) const
{
  return _sample_sum / (double)_num_samples; // [한국어] 합계를 샘플 수로 나눠 평균 계산.
                                              // _num_samples를 double로 캐스트하여 정수 나눗셈 방지.
}

/*
 * [한국어]
 * Stats::Variance — 샘플 분산 반환
 *
 * @return: double = (n·Σval² − (Σval)²) / n²
 *           여기서 n = _num_samples, Σval² = _sample_squared_sum, Σval = _sample_sum.
 *
 * 공식 유도:
 *   Var = E[X²] - (E[X])² = (Σval²/n) - (Σval/n)²
 *       = (n·Σval² - (Σval)²) / n²
 *
 * 주의: 이는 모분산(population variance, 분모 n) 추정량이며,
 *       표본 분산(분모 n-1)의 불편 추정량이 아님.
 *       _num_samples <= 0 이면 0/0 = NaN.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 출력 단계.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Variance()
 */
double Stats::Variance( ) const
{
  return (_sample_squared_sum * (double)_num_samples - _sample_sum * _sample_sum)
         / ((double)_num_samples * (double)_num_samples);
  // [한국어] 분자: n·Σval² − (Σval)²  = _sample_squared_sum * n - _sample_sum²
  // 분모: n² = _num_samples² (double로 캐스트하여 오버플로·정수 나눗셈 방지)
  // 이 방식은 수치적 안정성보다 구현 단순성을 우선한 것 — 매우 큰 값이나
  // 분산이 작은 경우 cancellation error가 발생할 수 있음.
}

/*
 * [한국어]
 * Stats::Min — 관측된 최솟값 반환
 *
 * @return: double = _min. 샘플 없으면 NaN.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 출력 단계.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Min()
 */
double Stats::Min( ) const
{
  return _min; // [한국어] 현재까지 AddSample()로 추가된 값 중 최솟값 반환. 샘플이 없으면 NaN.
}

/*
 * [한국어]
 * Stats::Max — 관측된 최댓값 반환
 *
 * @return: double = _max. 샘플 없으면 NaN.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 출력 단계.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Max()
 */
double Stats::Max( ) const
{
  return _max; // [한국어] 현재까지 AddSample()로 추가된 값 중 최댓값 반환. 샘플이 없으면 NaN.
}

/*
 * [한국어]
 * Stats::Sum — 샘플 합계 반환
 *
 * @return: double = _sample_sum.
 *
 * 평균 대신 합계 자체가 필요한 계산(예: 총 레이턴시)에 사용.
 * 실행 컨텍스트: 결과 출력 또는 외부 집계 단계.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Sum()
 */
double Stats::Sum( ) const
{
  return _sample_sum; // [한국어] 누적 합계 직접 반환 — Average() = Sum() / NumSamples()
}

/*
 * [한국어]
 * Stats::SquaredSum — 샘플 제곱합 반환
 *
 * @return: double = _sample_squared_sum = Σval².
 *
 * Variance()를 외부에서 재현하거나 합산된 통계를 병합할 때 사용.
 * 실행 컨텍스트: 결과 출력 또는 외부 집계 단계.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::SquaredSum()
 */
double Stats::SquaredSum( ) const
{
  return _sample_squared_sum; // [한국어] 누적 제곱합(Σval²) 직접 반환
}

/*
 * [한국어]
 * Stats::NumSamples — 현재까지 추가된 샘플 수 반환
 *
 * @return: int = _num_samples. 0이면 Average()/Variance() 호출 시 NaN 주의.
 *
 * 실행 컨텍스트: 결과 출력 및 호출 전 유효성 검사.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::NumSamples()
 */
int Stats::NumSamples( ) const
{
  return _num_samples; // [한국어] 누적 샘플 수 직접 반환
}

/*
 * [한국어]
 * Stats::AddSample (double) — 실수 샘플 추가 및 내부 모든 통계 갱신
 *
 * @val   : 추가할 측정값 (레이턴시 사이클 수, 홉 수 등 양의 실수).
 *           음수나 NaN을 전달하면 min/max 및 히스토그램 동작이 비정상.
 * @return: void
 *
 * 동작 과정:
 *   1) _num_samples 증가.
 *   2) _sample_sum += val.
 *   3) NaN-safe 비교로 _max, _min 갱신:
 *      - !(val <= _max) : val > _max 이거나 _max가 NaN이면 true → _max = val.
 *      - !(val >= _min) : val < _min 이거나 _min이 NaN이면 true → _min = val.
 *      이 방식은 IEEE 754 NaN 비교 규칙("NaN과의 모든 비교는 false")을 역으로 활용.
 *   4) 히스토그램 빈 계산:
 *      b = clamp(floor(val / _bin_size), 0, _num_bins-1)
 *      → _hist[b]++.
 *
 * 실행 컨텍스트: TrafficManager::_RetireFlit() 내부, 패킷 완료마다 호출.
 * 호출 체인:
 *   TrafficManager::_RetireFlit() → Stats::AddSample(latency)
 *   TrafficManager::_RetireFlit() → Stats::AddSample(hops)
 */
void Stats::AddSample( double val )
{
  ++_num_samples;       // [한국어] 샘플 카운터 1 증가 (프리픽스 ++로 효율적 갱신)
  _sample_sum += val;   // [한국어] 합계에 val 누적 (Average() 분자로 사용)

  // NOTE: the negation ensures that NaN values are handled correctly!
  // [한국어] NaN-safe 비교: IEEE 754에서 NaN과의 비교는 항상 false를 반환한다.
  // 따라서 !(val <= _max)는 _max가 NaN일 때 !(false) = true가 되어 val이 max로 설정됨.
  // 이는 Clear() 후 첫 번째 샘플이 자동으로 min·max가 되도록 보장하는 핵심 트릭.
  _max = !(val <= _max) ? val : _max; // [한국어] val이 현재 _max보다 크거나 _max가 NaN이면 _max를 val로 갱신
  _min = !(val >= _min) ? val : _min; // [한국어] val이 현재 _min보다 작거나 _min이 NaN이면 _min을 val로 갱신

  //double clamp between 0 and num_bins-1
  // [한국어] 히스토그램 빈 인덱스 계산: floor(val / bin_size)로 구간 결정 후 [0, num_bins-1]로 클램프
  int b = (int)fmax(floor( val / _bin_size ), 0.0); // [한국어] val을 _bin_size로 나눈 후 floor하여 빈 인덱스 계산.
                                                      // fmax(..., 0.0)으로 음수 인덱스를 0으로 클램프 (하한 보정).
                                                      // 예: val=25, bin_size=10 → b=2 (세 번째 빈에 해당).
  b = (b >= _num_bins) ? (_num_bins - 1) : b;        // [한국어] 인덱스가 상한(num_bins)을 초과하면 마지막 빈으로 클램프.
                                                       // 상한 초과 샘플은 모두 마지막 빈에 누적됨 (오버플로 방지).

  _hist[b]++;  // [한국어] 해당 빈 카운터 1 증가 — operator<</Display()에서 이 배열을 출력함
}

/*
 * [한국어]
 * Stats::Display — 히스토그램을 지정 ostream에 출력 후 줄 바꿈
 *
 * @os    : 출력 스트림 (기본값 cout). 파일이나 stringstream으로 변경 가능.
 * @return: void
 *
 * operator<<(*this)를 통해 히스토그램을 출력하고 endl로 버퍼를 플러시한다.
 * TrafficManager::DisplayStats()가 각 Stats 인스턴스에 대해 호출한다.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 출력 단계, 단일 스레드.
 * 호출 체인: TrafficManager::DisplayStats() → Stats::Display(cout)
 *                                            → operator<<(cout, *this)
 */
void Stats::Display( ostream & os ) const
{
  os << *this << endl; // [한국어] operator<<를 호출하여 "[ b0 b1 ... bN ]" 형식 출력 후 endl로 줄 바꿈
                        // endl은 '\n'과 달리 버퍼를 즉시 플러시하여 출력이 즉시 반영됨
}

/*
 * [한국어]
 * operator<< — Stats 히스토그램 ostream 출력 연산자 (전역 friend 함수)
 *
 * @os: 출력 스트림
 * @s : 출력할 Stats 인스턴스
 * @return: ostream& os — 연쇄 출력(chaining) 지원
 *
 * "[ bin0 bin1 ... binN ]" 형식으로 _hist 벡터를 공백 구분하여 출력한다.
 * friend 함수이므로 Stats의 private 멤버 _hist에 직접 접근 가능.
 * 각 빈 값은 해당 구간에 속하는 샘플 수를 나타낸다.
 *
 * 실행 컨텍스트: Display() 내부, 또는 std::cout << stats_instance 형태로 직접 호출.
 * 호출 체인: Stats::Display() → operator<<(os, *this)
 */
ostream & operator<<(ostream & os, const Stats & s) {
  vector<int> const & v = s._hist;  // [한국어] _hist 벡터에 const 참조로 접근 — 복사 비용 없이 순회
  os << "[ ";                        // [한국어] 히스토그램 시작 구분자 출력
  for(size_t i = 0; i < v.size(); ++i) {  // [한국어] _hist의 모든 빈을 순서대로 순회 (i=0이 최솟값 구간)
    os << v[i] << " ";                     // [한국어] i번 빈 카운트와 공백 출력
  }
  os << "]";  // [한국어] 히스토그램 종료 구분자 출력
  return os;  // [한국어] 스트림 참조 반환 — "cout << s1 << s2" 형태의 연쇄 출력 지원
}
