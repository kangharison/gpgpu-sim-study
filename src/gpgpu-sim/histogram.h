// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*
 * [한국어 설명] GPGPU-Sim 히스토그램 자료구조 헤더 (histogram.h)
 *
 * === 파일의 역할 ===
 * 시뮬레이션 통계 수집에 사용되는 세 가지 히스토그램 클래스를 정의한다.
 * binned_histogram은 기반 클래스로 bin 경계, 카운터, 최대값, 합계를 관리한다.
 * pow2_histogram은 2의 거듭제곱 단위로 빠른 비트 연산 log2 계산을 사용하는 변형이고,
 * linear_histogram은 고정 stride(간격)으로 선형 분류하는 변형이다.
 * 메모리 접근 레이턴시 분포, 워프 점유율 분포 등 다양한 성능 지표 수집에 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: 타이밍 시뮬레이션(shader.cc, gpu-cache.cc 등) → add2bin() 호출
 *   → 시뮬레이션 종료 후 fprint()로 결과 파일에 출력
 * stat-tool.h/cc가 이 클래스들을 포함하여 상위 수준 통계 로거를 구현.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내 임의 시점.
 *
 * === 타 모듈과의 연결 ===
 * 의존: stdio.h (FILE*), string (std::string)
 * 사용처: stat-tool.h의 linear_histogram_logger, insn_warp_occ_logger,
 *          shader_mem_lat_log() 등에서 내부 자료구조로 사용
 * 데이터 흐름: add2bin() → m_bin_cnts[] 누적 → fprint()로 출력
 *
 * === 주요 함수/구조체 요약 ===
 * binned_histogram::add2bin()   - 순수 가상(assert 실패) — 서브클래스 구현 필요
 * binned_histogram::fprint()    - bin 카운터, 최대값, 평균을 파일에 출력
 * pow2_histogram::add2bin()     - 비트 연산 log2로 O(1) bin 분류 (2의 거듭제곱 경계)
 * linear_histogram::add2bin()   - sample/stride로 bin 인덱스 계산 (선형 균등 분류)
 */

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#ifdef __cplusplus
/* [한국어] C++에서만 사용 가능한 클래스 정의. C 파일에서 포함 시 무시. */

#include <stdio.h>  /* [한국어] FILE* 타입 (fprint 메서드의 출력 대상) */
#include <string>   /* [한국어] std::string (m_name 통계 이름 필드) */

/*
 * [한국어]
 * binned_histogram - 임의 bin 경계를 지원하는 기반 히스토그램 클래스
 *
 * 사용자 지정 bin 경계 배열 또는 서브클래스가 정의한 방식으로 샘플을 분류한다.
 * add2bin()은 기반 클래스에서 assert(0)으로 구현되어 있어 반드시 서브클래스에서
 * 오버라이드해야 한다. fprint()는 모든 서브클래스가 공유하는 공통 출력 메서드.
 * m_sum을 통해 평균 계산 가능, m_maximum으로 최대 샘플값 추적.
 */
class binned_histogram {
 public:
  // creators
  /*
   * [한국어]
   * binned_histogram() - 기본 생성자
   *
   * @name: 통계 이름 (fprint 출력 시 접두어로 사용)
   * @nbins: bin 개수 (기본 32)
   * @bins: 사용자 지정 bin 경계 배열 포인터 (NULL이면 경계 없음 — 서브클래스가 결정)
   *
   * m_bin_cnts를 new int[nbins]로 동적 할당하고 reset_bins()로 0 초기화.
   * bins != NULL이면 m_bins도 동적 할당하여 복사.
   */
  binned_histogram(std::string name = "", int nbins = 32, int* bins = NULL);
  /*
   * [한국어]
   * binned_histogram(const binned_histogram&) - 복사 생성자
   *
   * @other: 복사 원본 히스토그램 객체
   *
   * m_bin_cnts[]를 other에서 복사. m_bins는 복사하지 않음(NULL로 남김).
   * stat-tool.cc에서 벡터 할당 시 복사 생성자 호출됨.
   */
  binned_histogram(const binned_histogram& other);
  virtual ~binned_histogram(); /* [한국어] 가상 소멸자: m_bins, m_bin_cnts 동적 메모리 해제 */

  // modifiers:
  /*
   * [한국어]
   * reset_bins() - 모든 bin 카운터를 0으로 초기화
   *
   * m_bin_cnts[0..m_nbins-1]를 모두 0으로 설정.
   * 생성자에서 초기화 및 인터벌 기반 리셋 시 호출.
   */
  void reset_bins();
  /*
   * [한국어]
   * add2bin() - 샘플값을 적절한 bin에 분류하여 카운터 증가 (순수 가상)
   *
   * @sample: 분류할 정수 샘플 값
   *
   * 기반 클래스에서는 assert(0)으로 구현 — 서브클래스(pow2_histogram,
   * linear_histogram)가 반드시 오버라이드해야 한다.
   * m_maximum 갱신과 m_sum 누적은 각 서브클래스에서 수행.
   */
  void add2bin(int sample);

  // accessors:
  /*
   * [한국어]
   * fprint() - 히스토그램 내용을 파일 스트림에 출력
   *
   * @fout: 출력 대상 FILE* 포인터
   *
   * 형식: "[name] = <bin0> <bin1> ... max=<max> avg=<avg>"
   * total_sample = sum of m_bin_cnts[]로 평균 계산.
   * 샘플이 없으면 avg=0.0 출력.
   */
  void fprint(FILE* fout) const;

 protected:
  std::string m_name;
  /* [한국어] 히스토그램 이름 문자열.
   * 설정자: 생성자. 읽는 자: fprint()가 출력 접두어로 사용.
   * 값 범위: 임의 문자열, 빈 문자열이면 이름 출력 생략. */

  int m_nbins;
  /* [한국어] 히스토그램 bin의 총 개수.
   * 설정자: 생성자 파라미터. 읽는 자: add2bin, fprint, reset_bins.
   * 값 범위: 양의 정수 (기본 32). */

  int* m_bins;                 // bin boundaries
  /* [한국어] 사용자 지정 bin 경계값 배열 포인터 (m_nbins개).
   * 설정자: 생성자에서 bins 파라미터가 NULL이 아닐 때 동적 할당 및 복사.
   * 읽는 자: 현재 기반 클래스 및 서브클래스에서 직접 사용하지 않음(참고용).
   * 값 범위: NULL이면 경계 배열 없음 (서브클래스가 자체 계산). */

  int* m_bin_cnts;             // counters
  /* [한국어] 각 bin의 샘플 카운터 배열 (m_nbins개).
   * 설정자: 생성자에서 new int[m_nbins] 동적 할당; add2bin()에서 해당 bin 증가.
   * 읽는 자: fprint()에서 출력; reset_bins()에서 0으로 초기화.
   * 동기화: 단일 스레드에서만 접근하므로 락 불필요. */

  int m_maximum;               // the maximum sample
  /* [한국어] 지금까지 추가된 샘플 중 최대값.
   * 설정자: add2bin()에서 (sample > m_maximum) ? sample : m_maximum 갱신.
   * 읽는 자: fprint()에서 "max=<maximum>" 형식으로 출력.
   * 값 범위: 0 이상의 정수 (초기값 0). */

  signed long long int m_sum;  // for calculating the average
  /* [한국어] 추가된 모든 샘플값의 합계 (평균 계산용).
   * 설정자: add2bin()에서 m_sum += sample.
   * 읽는 자: fprint()에서 avg = (float)m_sum / total_sample 계산.
   * 값 범위: signed 64비트 정수 (대규모 누적에도 오버플로 방지). */
};

/*
 * [한국어]
 * pow2_histogram - 2의 거듭제곱 경계 히스토그램 (비트 연산 log2 사용)
 *
 * bin i는 [2^(i-1), 2^i) 범위의 샘플을 수집한다 (i=0: sample=0).
 * add2bin()에서 비트 연산으로 O(1) log2를 계산하므로 메모리 레이턴시 분포 등
 * 넓은 범위의 값을 효율적으로 분류하는 데 적합하다.
 * stat-tool.cc의 shader_mem_lat_log()에서 레이턴시 분포 수집에 사용.
 */
class pow2_histogram : public binned_histogram {
 public:
  /*
   * [한국어]
   * pow2_histogram() - 2의 거듭제곱 히스토그램 생성자
   *
   * @name: 통계 이름, @nbins: bin 개수(기본 32), @bins: 사용자 bin 경계(기본 NULL)
   *
   * 기반 클래스 binned_histogram 생성자에 위임.
   * nbins=32이면 최대 2^31 = 2G 범위의 값까지 분류 가능.
   */
  pow2_histogram(std::string name = "", int nbins = 32, int* bins = NULL);
  ~pow2_histogram() {}

  /*
   * [한국어]
   * pow2_histogram::add2bin() - 비트 연산으로 log2 계산하여 bin 분류
   *
   * @sample: 분류할 비음수 정수 샘플 (assert(sample >= 0) 검사)
   *
   * 비트 시프트 연산으로 floor(log2(sample))+1을 O(1)로 계산.
   * 계산된 bin 인덱스의 m_bin_cnts[bin]을 증가시키고
   * m_maximum과 m_sum을 갱신한다.
   * sample=0이면 bin=0 (log2(0)은 정의 불가이지만 특별 처리됨).
   */
  void add2bin(int sample);
};

/*
 * [한국어]
 * linear_histogram - 고정 stride(간격) 선형 히스토그램
 *
 * bin i = sample / m_stride. 따라서 bin i는 [i*stride, (i+1)*stride) 범위 수집.
 * sample이 마지막 bin 범위를 초과하면 마지막 bin(m_nbins-1)에 분류(클리핑).
 * 워프 점유율(warp occupancy) 같이 균등하게 분포된 값 수집에 적합.
 */
class linear_histogram : public binned_histogram {
 public:
  /*
   * [한국어]
   * linear_histogram() - 선형 히스토그램 생성자
   *
   * @stride: bin 당 값 범위 간격 (기본 1). 예: stride=4이면 [0,4), [4,8), ...
   * @name: 통계 이름 (기본 NULL)
   * @nbins: bin 개수 (기본 32)
   * @bins: 사용자 bin 경계 (기본 NULL)
   */
  linear_histogram(int stride = 1, const char* name = NULL, int nbins = 32,
                   int* bins = NULL);
  ~linear_histogram() {}

  /*
   * [한국어]
   * linear_histogram::add2bin() - stride로 나눠 bin 인덱스 계산
   *
   * @sample: 분류할 비음수 정수 샘플 (assert(sample >= 0) 검사)
   *
   * bin = sample / m_stride. bin >= m_nbins이면 m_nbins-1로 클리핑(오버플로 방지).
   * m_bin_cnts[bin] 증가, m_maximum과 m_sum 갱신.
   */
  void add2bin(int sample);

 private:
  int m_stride;
  /* [한국어] bin 하나당 값 범위 간격.
   * 설정자: 생성자 파라미터. 읽는 자: add2bin()에서 bin = sample / m_stride 계산.
   * 값 범위: 양의 정수 (0이면 나눗셈 오류). */
};

#endif

#endif /* HISTOGRAM_H */
