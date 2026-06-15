// $Id: stats.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 시뮬레이션 통계 수집 클래스 헤더 (stats.hpp)
 *
 * === 파일의 역할 ===
 * intersim2 NoC 시뮬레이터에서 발생하는 각종 측정값(레이턴시, 홉 수,
 * 처리량 등)을 축적·집계하는 Stats 클래스를 선언한다. 샘플을 AddSample()로
 * 추가하면 내부적으로 합계·제곱합·최솟값·최댓값·히스토그램을 갱신하며,
 * 시뮬레이션 종료 후 Average(), Variance(), Display() 등으로 결과를 출력한다.
 * TrafficManager가 플릿별 레이턴시와 홉 수를 이 클래스를 통해 집계한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * stats.hpp/cpp는 intersim2의 성능 측정 계층 최하단에 위치한다.
 * TrafficManager가 패킷 완료 시 Stats 인스턴스에 샘플을 추가하고,
 * 시뮬레이션 완료 후 gpgpusim_entrypoint 또는 TrafficManager::DisplayStats()가
 * Stats::Display()를 호출하여 결과를 출력한다.
 *
 * 호출 체인:
 *   TrafficManager::_UpdateStats() → Stats::AddSample(latency)
 *   TrafficManager::DisplayStats() → Stats::Display() / operator<<
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - module.hpp : Module 베이스 클래스 (계층 이름, 에러 출력 지원).
 * 이 파일을 사용하는 모듈:
 *   - trafficmanager.cc : _plat(패킷 레이턴시), _nlat(네트워크 레이턴시),
 *                         _hop_stats(홉 수) 등 Stats 인스턴스 보유.
 *   - router/*.cc       : 개별 라우터가 내부 큐 점유 통계를 수집할 때 사용.
 * 데이터 흐름:
 *   플릿 완료 → TrafficManager가 레이턴시 계산
 *   → Stats::AddSample() → _sample_sum, _hist[] 갱신
 *   → 시뮬레이션 종료 시 Average()/Variance()로 집계 출력.
 *
 * === 주요 함수/구조체 요약 ===
 * - Stats(parent, name, bin_size, num_bins) : 생성자. 히스토그램 설정 후 Clear().
 * - Clear()       : 모든 통계를 초기화 (측정 구간 재시작 시 호출).
 * - AddSample(val): 샘플 추가 — 합계·제곱합·min/max·히스토그램 갱신.
 * - Average()     : 산술 평균 = _sample_sum / _num_samples.
 * - Variance()    : 표본 분산 (편향 추정량 아님 — 공식 주의).
 * - Display(os)   : 히스토그램 벡터를 스트림에 출력.
 * - operator<<    : ostream 출력 연산자 오버로드 — Display()와 동일 출력.
 */

#ifndef _STATS_HPP_
#define _STATS_HPP_

#include "module.hpp"  // [한국어] Module 베이스 클래스 — 이름/계층 관리. Stats는 Module을 상속.

/*
 * [한국어]
 * Stats — NoC 시뮬레이터 성능 측정값 수집·집계 클래스
 *
 * 샘플(double/int/unsigned long long)을 AddSample()로 추가하면
 * 내부적으로 다음 6가지를 누적 갱신한다:
 *   1) _num_samples : 샘플 수
 *   2) _sample_sum  : 합계 (Average 계산에 사용)
 *   3) _sample_squared_sum : 제곱합 (Variance 계산에 사용)
 *   4) _min, _max   : 최솟값·최댓값
 *   5) _hist[]      : 히스토그램 (bin_size 간격으로 분류)
 *
 * 히스토그램은 val을 bin_size로 나눈 뒤 floor하여 빈(bin) 인덱스를 결정한다.
 * 인덱스가 범위를 벗어나면 마지막 빈(_num_bins-1)에 클램프한다.
 *
 * 동기화: TrafficManager가 단일 스레드에서 실행하므로 별도 락 불필요.
 */
class Stats : public Module {
  int    _num_samples;
  /* 지금까지 AddSample()로 추가된 샘플의 총 개수.
   * 설정자: Clear()에서 0으로 초기화; AddSample()에서 ++.
   * 읽는 자: Average(), Variance() — 분모로 사용; NumSamples() — 직접 반환.
   * 값 범위: [0, INT_MAX]. 오버플로 검사는 없으므로 매우 긴 시뮬레이션에서
   *           주의 필요 (실용적으로는 수백만 샘플 이내로 충분).
   * 동기화: 단일 스레드 실행 → 락 불필요. */

  double _sample_sum;
  /* 추가된 모든 샘플 값의 합계.
   * 설정자: Clear()에서 0.0으로 초기화; AddSample()에서 += val.
   * 읽는 자: Average() = _sample_sum / _num_samples; Sum() — 직접 반환.
   * 값 범위: 음수 불가 (레이턴시·홉 수는 양수). 부동소수점 누적 오차 가능.
   * 동기화: 단일 스레드 → 락 불필요. */

  double _sample_squared_sum;
  /* 추가된 모든 샘플 값의 제곱합 (Σ val²).
   * Variance() = (n·Σval² − (Σval)²) / n² 공식에서 분자 계산에 사용.
   * 설정자: Clear()에서 0.0으로 초기화. (주의: AddSample()에서 갱신되지
   *          않는 것처럼 보이나 stats.cpp 구현에서 실제로는 갱신됨 — 헤더에서
   *          선언만 됨.)
   * 읽는 자: Variance(), SquaredSum().
   * 값 범위: 양수. 값이 크면 부동소수점 정밀도 손실 주의.
   * 동기화: 단일 스레드 → 락 불필요. */

  //bool _reset;
  /* [한국어] 리셋 플래그 — 현재 주석 처리되어 미사용.
   * 과거에 min/max를 리셋하는 용도로 사용되었으나 제거된 것으로 추정.
   * NaN 초기화 방식으로 대체됨. */

  double _min;
  /* 관측된 최솟값.
   * 설정자: Clear()에서 quiet_NaN()으로 초기화 (유효 샘플 없음을 표시);
   *          AddSample()에서 val < _min 시 갱신.
   *          NaN 처리: !(val >= _min) 조건으로 NaN 초기 상태를 올바르게 처리.
   * 읽는 자: Min() — 직접 반환.
   * 값 범위: 유효 샘플 없으면 NaN, 이후 [0, _max].
   * 동기화: 단일 스레드 → 락 불필요. */

  double _max;
  /* 관측된 최댓값.
   * 설정자: Clear()에서 -quiet_NaN()으로 초기화 (NaN의 부정도 NaN이므로
   *          동일하게 미초기화 상태를 표시); AddSample()에서 val > _max 시 갱신.
   *          NaN 처리: !(val <= _max) 조건으로 NaN 초기 상태를 올바르게 처리.
   * 읽는 자: Max() — 직접 반환.
   * 값 범위: 유효 샘플 없으면 NaN, 이후 [_min, +∞).
   * 동기화: 단일 스레드 → 락 불필요. */

  int    _num_bins;
  /* 히스토그램 빈(bin)의 수 — 히스토그램 분해능을 결정한다.
   * 기본값: 10 (생성자 인자 num_bins).
   * 설정자: 생성자에서 초기화, 이후 불변.
   * 읽는 자: Clear()에서 _hist.assign(_num_bins, 0); AddSample()에서 클램프 상한.
   *           GetBin()에서 인덱스 상한으로 사용.
   * 값 범위: 1 이상 (0이면 _hist 접근 시 UB).
   * 동기화: 생성 후 불변 → 락 불필요. */

  double _bin_size;
  /* 히스토그램 각 빈의 너비 — 샘플 값을 빈 인덱스로 변환할 때 사용.
   * 빈 인덱스 = floor(val / _bin_size). 기본값: 1.0.
   * 예: bin_size=10이면 [0,10), [10,20), ... 구간으로 히스토그램 분류.
   * 설정자: 생성자에서 초기화, 이후 불변.
   * 읽는 자: AddSample()에서 빈 인덱스 계산 시 사용.
   * 값 범위: 양수 (0이면 제로 나누기 발생).
   * 동기화: 생성 후 불변 → 락 불필요. */

  vector<int> _hist;
  /* 히스토그램 빈 카운트 배열. _hist[i]는 i번 빈에 속하는 샘플 수.
   * 크기: _num_bins. 마지막 빈(_hist[_num_bins-1])은 상한 초과 샘플도 포함(클램프).
   * 설정자: Clear()에서 모두 0으로 초기화; AddSample()에서 해당 빈 ++.
   * 읽는 자: operator<<와 Display()에서 순서대로 출력; GetBin()으로 개별 접근.
   * 값 범위: 각 원소 [0, _num_samples].
   * 동기화: 단일 스레드 → 락 불필요. */

public:
  /*
   * [한국어]
   * Stats::Stats — 통계 수집기 생성자
   *
   * @parent  : Module 계층 트리의 부모 노드 (TrafficManager 등).
   * @name    : 이 Stats 인스턴스 식별 이름 (예: "packet_latency").
   * @bin_size: 히스토그램 빈 너비 (기본값 1.0). 레이턴시 측정 시 단위 사이클.
   * @num_bins: 히스토그램 빈 수 (기본값 10). 분해능 결정.
   * @return  : (생성자)
   *
   * Module 베이스 초기화 후 _num_bins, _bin_size 설정, Clear() 호출로
   * 모든 카운터를 초기 상태로 리셋한다.
   */
  Stats( Module *parent, const string &name,
	 double bin_size = 1.0, int num_bins = 10 );

  /*
   * [한국어]
   * Stats::Clear — 모든 통계 카운터 초기화
   *
   * @return: void
   *
   * 새 측정 구간 시작 시 호출. _num_samples=0, _sample_sum=0.0,
   * _sample_squared_sum=0.0으로 초기화하고 _hist를 모두 0으로 채운다.
   * _min은 quiet_NaN(), _max는 -quiet_NaN()으로 설정하여 "샘플 없음" 상태 표시.
   */
  void Clear( );

  /*
   * [한국어]
   * Stats::Average — 산술 평균 반환
   *
   * @return: double = _sample_sum / _num_samples.
   *           _num_samples == 0 이면 NaN 또는 0/0 = UB (호출 전 NumSamples() 확인 권장).
   */
  double Average( ) const;

  /*
   * [한국어]
   * Stats::Variance — 표본 분산 반환
   *
   * @return: double = (n·Σval² − (Σval)²) / n²  (편향 추정량).
   *           _num_samples <= 1 이면 의미 없는 값이 반환될 수 있음.
   *
   * 주의: 이 공식은 n(n-1) 분모의 불편 추정량이 아니라 n² 분모이므로
   * 통계적 엄밀성이 필요한 경우 수정이 필요하다.
   */
  double Variance( ) const;

  /*
   * [한국어]
   * Stats::Max — 관측된 최댓값 반환
   *
   * @return: double. 샘플이 없으면 NaN.
   */
  double Max( ) const;

  /*
   * [한국어]
   * Stats::Min — 관측된 최솟값 반환
   *
   * @return: double. 샘플이 없으면 NaN.
   */
  double Min( ) const;

  /*
   * [한국어]
   * Stats::Sum — 샘플 합계 반환
   *
   * @return: double = _sample_sum. 합계를 외부에서 직접 사용해야 할 때.
   */
  double Sum( ) const;

  /*
   * [한국어]
   * Stats::SquaredSum — 샘플 제곱합 반환
   *
   * @return: double = _sample_squared_sum. Variance() 계산 재현이 필요할 때.
   */
  double SquaredSum( ) const;

  /*
   * [한국어]
   * Stats::NumSamples — 지금까지 추가된 샘플 수 반환
   *
   * @return: int = _num_samples. 0이면 Average()/Variance() 호출 위험.
   */
  int    NumSamples( ) const;

  /*
   * [한국어]
   * Stats::AddSample (double) — 실수 샘플 추가 및 내부 통계 갱신
   *
   * @val   : 추가할 샘플 값 (레이턴시, 홉 수 등 양의 실수).
   * @return: void
   *
   * 1) _num_samples++, _sample_sum += val, _sample_squared_sum += val*val.
   * 2) NaN-safe 비교로 _min, _max 갱신.
   * 3) 히스토그램 빈 인덱스 계산 후 _hist[b]++.
   *
   * 실행 컨텍스트: TrafficManager::_UpdateStats(), 매 패킷 완료 시.
   * 호출 체인: TrafficManager → Stats::AddSample(latency)
   */
  void AddSample( double val );

  /*
   * [한국어]
   * Stats::AddSample (int) — 정수 샘플을 double로 변환 후 AddSample(double) 위임
   *
   * @val   : 정수 샘플 (홉 수 등).
   * @return: void
   *
   * 인라인 캐스트 래퍼로, 호출자가 오버로드 해소에 신경 쓰지 않아도 되게 함.
   */
  inline void AddSample( int val ) {
    AddSample( (double)val ); // [한국어] int를 double로 명시적 캐스트 후 double 버전 위임
  }

  /*
   * [한국어]
   * Stats::AddSample (unsigned long long) — 64비트 정수 샘플을 double로 변환 후 위임
   *
   * @val   : 64비트 정수 샘플 (바이트 수, 대역폭 카운터 등).
   * @return: void
   *
   * unsigned long long → double 변환 시 53비트 이상 값은 정밀도 손실 가능.
   * 레이턴시/홉 수 용도에서는 일반적으로 문제없음.
   */
  inline void AddSample( unsigned long long val ) {
    AddSample( (double)val ); // [한국어] unsigned long long을 double로 명시적 캐스트 후 double 버전 위임
  }

  /*
   * [한국어]
   * Stats::GetBin — 지정 인덱스의 히스토그램 빈 카운트 반환
   *
   * @b     : 빈 인덱스. 0 ≤ b < _num_bins 이어야 한다 (범위 검사 없음).
   * @return: int = _hist[b]. 해당 빈에 속하는 샘플 수.
   *
   * 외부에서 히스토그램 데이터를 직접 접근할 때 사용.
   * 범위를 벗어난 b 값은 STL 벡터의 []연산이 UB를 유발.
   */
  int GetBin(int b){ return _hist[b];}  // [한국어] 범위 검사 없이 _hist[b] 반환 — 호출자가 유효 인덱스 보장 필요

  /*
   * [한국어]
   * Stats::Display — 히스토그램을 지정 ostream에 출력
   *
   * @os    : 출력 스트림 (기본값 cout). 파일이나 stringstream으로 변경 가능.
   * @return: void
   *
   * operator<< 를 호출하고 endl을 추가한다. TrafficManager::DisplayStats()가
   * 시뮬레이션 종료 후 각 Stats 인스턴스에 대해 호출한다.
   */
  void Display( ostream & os = cout ) const;

  /*
   * [한국어]
   * operator<< — Stats 히스토그램 ostream 출력 연산자
   *
   * @os: 출력 스트림
   * @s : 출력할 Stats 인스턴스
   * @return: os 참조 (체인 출력 지원)
   *
   * "[ bin0 bin1 ... binN ]" 형식으로 히스토그램 카운트를 공백 구분하여 출력.
   * friend 선언이므로 Stats의 private 멤버(_hist)에 접근 가능.
   */
  friend ostream & operator<<(ostream & os, const Stats & s);

};

// [한국어] 전역 operator<< 선언 — Stats 객체를 ostream에 직접 <<로 출력 가능하게 함
ostream & operator<<(ostream & os, const Stats & s);

#endif  // [한국어] _STATS_HPP_ 헤더 가드 종료 — 중복 include 방지
