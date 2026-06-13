// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung
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
 * [한국어 설명] GPGPU-Sim 히스토그램 구현 (histogram.cc)
 *
 * === 파일의 역할 ===
 * histogram.h에 선언된 세 히스토그램 클래스(binned_histogram, pow2_histogram,
 * linear_histogram)의 멤버 함수를 구현한다. 기반 클래스는 동적 메모리 관리, 출력,
 * 초기화를 담당하며, 두 서브클래스는 서로 다른 bin 분류 알고리즘을 구현한다.
 * pow2_histogram의 add2bin()은 비트 연산 기반 O(1) log2 계산을 사용하는
 * 최적화된 구현으로, 레이턴시 분포 수집에서 성능 병목을 회피한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: stat-tool.cc의 shader_mem_lat_log() → pow2_histogram::add2bin()
 *            stat-tool.cc의 linear_histogram_logger::log() → linear_histogram::add2bin()
 * 시뮬레이션 루프에서 빈번히 호출되므로 add2bin()의 성능이 중요.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: histogram.h, assert.h
 * 사용처: stat-tool.cc (linear_histogram, pow2_histogram 인스턴스 생성 및 사용)
 * 데이터 흐름: add2bin() → m_bin_cnts[] 누적 → fprint()로 결과 출력
 *
 * === 주요 함수/구조체 요약 ===
 * binned_histogram::binned_histogram() - m_bin_cnts 동적 할당 및 reset_bins() 초기화
 * binned_histogram::fprint()           - bin 카운터 + max + avg 파일 출력
 * pow2_histogram::add2bin()            - 비트 연산 log2로 bin 분류
 * linear_histogram::add2bin()          - sample/stride로 bin 분류
 */

#include "histogram.h"

#include <assert.h> /* [한국어] assert() — 비정상 입력 및 설계 오류 조기 감지 */

/*
 * [한국어]
 * binned_histogram::binned_histogram() - 기반 히스토그램 생성자
 *
 * @name: 통계 이름 문자열
 * @nbins: bin 개수 (기본 32)
 * @bins: 사용자 정의 bin 경계 배열 포인터 (NULL이면 서브클래스가 경계 결정)
 *
 * m_bin_cnts를 new int[m_nbins]로 동적 할당하고 reset_bins()로 0 초기화.
 * bins != NULL이면 m_bins도 동적 할당하여 전체 복사.
 * 실행 컨텍스트: 시뮬레이터 초기화 시 생성.
 *
 * 호출 체인:
 *   stat-tool.cc → [binned_histogram()] → new int[m_nbins] + reset_bins()
 */
binned_histogram::binned_histogram(std::string name, int nbins, int* bins)
    : m_name(name),           /* [한국어] 통계 이름 초기화 */
      m_nbins(nbins),         /* [한국어] bin 개수 초기화 */
      m_bins(NULL),           /* [한국어] bin 경계 배열 포인터 — bins 파라미터 처리 전 NULL */
      m_bin_cnts(new int[m_nbins]), /* [한국어] bin 카운터 배열 동적 할당 */
      m_maximum(0),           /* [한국어] 최대 샘플값 0으로 초기화 */
      m_sum(0) {              /* [한국어] 샘플 합계 0으로 초기화 */
  if (bins) {
    /* [한국어] 사용자 지정 bin 경계가 제공된 경우 복사 */
    m_bins = new int[m_nbins]; /* [한국어] bin 경계 배열 동적 할당 */
    for (int i = 0; i < nbins; i++) {
      m_bins[i] = bins[i]; /* [한국어] 각 bin 경계값 복사 */
    }
  }

  reset_bins(); /* [한국어] m_bin_cnts[] 모두 0으로 초기화 */
}

/*
 * [한국어]
 * binned_histogram::binned_histogram(const binned_histogram&) - 복사 생성자
 *
 * @other: 복사 원본 히스토그램 객체
 *
 * m_bin_cnts[]를 other에서 항목별 복사. m_bins는 NULL로 남김(경계 배열 복사 생략).
 * stat-tool.cc에서 std::vector에 히스토그램을 assign()할 때 호출됨.
 * m_maximum, m_sum은 0으로 초기화 (누적 데이터 미복사).
 *
 * 호출 체인:
 *   stat-tool.cc::shader_warp_occ_create() → vector::assign() → [복사 생성자]
 */
binned_histogram::binned_histogram(const binned_histogram& other)
    : m_name(other.m_name),   /* [한국어] 이름 복사 */
      m_nbins(other.m_nbins), /* [한국어] bin 개수 복사 */
      m_bins(NULL),           /* [한국어] bin 경계 배열은 복사하지 않음 */
      m_bin_cnts(new int[m_nbins]), /* [한국어] 새 카운터 배열 할당 */
      m_maximum(0),           /* [한국어] 최대값 0으로 재초기화 */
      m_sum(0) {              /* [한국어] 합계 0으로 재초기화 */
  for (int i = 0; i < m_nbins; i++) {
    m_bin_cnts[i] = other.m_bin_cnts[i]; /* [한국어] 기존 카운터 값 복사 */
  }
}

/*
 * [한국어]
 * binned_histogram::reset_bins() - 모든 bin 카운터를 0으로 초기화
 *
 * m_bin_cnts[0..m_nbins-1]을 모두 0으로 설정.
 * 생성자에서 초기화 목적, 또는 인터벌 통계 리셋 시 사용.
 */
void binned_histogram::reset_bins() {
  for (int i = 0; i < m_nbins; i++) {
    m_bin_cnts[i] = 0; /* [한국어] 각 bin 카운터를 0으로 초기화 */
  }
}

/*
 * [한국어]
 * binned_histogram::add2bin() - 기반 클래스 구현 (assert(0) — 서브클래스 강제)
 *
 * @sample: 분류할 샘플 (무시됨)
 *
 * 기반 클래스에서는 구현하지 않음. assert(0)으로 서브클래스 오버라이드를 강제.
 * 서브클래스(pow2_histogram, linear_histogram)가 반드시 오버라이드해야 함.
 * m_maximum 갱신은 남겨져 있지만 assert(0)으로 실제로 도달 불가.
 */
void binned_histogram::add2bin(int sample) {
  assert(0); /* [한국어] 기반 클래스 add2bin() 직접 호출 금지 — 서브클래스 오버라이드 필수 */
  m_maximum = (sample > m_maximum) ? sample : m_maximum;
  /* [한국어] 위 assert(0) 때문에 실제로 실행되지 않음. */
}

/*
 * [한국어]
 * binned_histogram::fprint() - 히스토그램 내용을 파일 스트림에 출력
 *
 * @fout: 출력 대상 FILE* 포인터
 *
 * 출력 형식: "[name] = <bin0> <bin1> ... max=<max> avg=<avg>"
 * total_sample = 모든 bin 카운터의 합으로 평균 계산.
 * 샘플이 전혀 없으면 avg = 0.0으로 출력 (0 나눗셈 방지).
 * 실행 컨텍스트: 시뮬레이션 종료 후 결과 파일 출력 단계.
 *
 * 호출 체인:
 *   stat-tool.cc → linear_histogram_logger → [fprint()]
 */
void binned_histogram::fprint(FILE* fout) const {
  if (m_name.c_str() != NULL) fprintf(fout, "%s = ", m_name.c_str());
  /* [한국어] 이름이 있으면 "이름 = " 접두어 출력 */
  int total_sample = 0;      /* [한국어] 전체 샘플 수 (평균 계산 분모용) */
  for (int i = 0; i < m_nbins; i++) {
    fprintf(fout, "%d ", m_bin_cnts[i]); /* [한국어] 각 bin의 카운터 출력 */
    total_sample += m_bin_cnts[i];       /* [한국어] 전체 샘플 수 누적 */
  }
  fprintf(fout, "max=%d ", m_maximum); /* [한국어] 최대 샘플값 출력 */
  float avg = 0.0f;                    /* [한국어] 평균값 초기화 (샘플 없는 경우 0 출력) */
  if (total_sample > 0) {
    avg = (float)m_sum / total_sample; /* [한국어] 합계 / 총 샘플 수로 평균 계산 */
  }
  fprintf(fout, "avg=%0.2f ", avg); /* [한국어] 소수점 2자리로 평균 출력 */
}

/*
 * [한국어]
 * binned_histogram::~binned_histogram() - 소멸자 (동적 메모리 해제)
 *
 * m_bins(bin 경계 배열, NULL일 수 있음)와 m_bin_cnts(카운터 배열)를 해제.
 * m_bins는 생성자에서 bins 파라미터가 NULL이 아닐 때만 할당됨.
 */
binned_histogram::~binned_histogram() {
  if (m_bins) delete[] m_bins; /* [한국어] bin 경계 배열이 있는 경우에만 해제 */
  delete[] m_bin_cnts;         /* [한국어] bin 카운터 배열은 항상 할당됨 — 무조건 해제 */
}

/*
 * [한국어]
 * pow2_histogram::pow2_histogram() - 2의 거듭제곱 히스토그램 생성자
 *
 * @name: 통계 이름, @nbins: bin 개수(기본 32), @bins: bin 경계(기본 NULL)
 *
 * 기반 클래스 binned_histogram 생성자에 전달 위임. 별도 초기화 없음.
 */
pow2_histogram::pow2_histogram(std::string name, int nbins, int* bins)
    : binned_histogram(name, nbins, bins) {}
/* [한국어] 기반 클래스 생성자에 모든 파라미터 위임 */

/*
 * [한국어]
 * pow2_histogram::add2bin() - 비트 연산 log2로 bin 분류
 *
 * @sample: 분류할 비음수 정수 샘플
 *
 * de Bruijn sequence 유사 비트 연산으로 floor(log2(v))를 O(1)로 계산.
 * bin = floor(log2(sample)) + 1 (sample > 0인 경우), sample=0이면 bin=0.
 * 알고리즘: 16비트/8비트/4비트/2비트/1비트 순으로 비트 시프트하며 log2 누적.
 * 레이턴시 값이 1~수백 사이클로 넓게 분포할 때 균등한 bin 분류를 제공.
 *
 * 호출 체인:
 *   stat-tool.cc::shader_mem_lat_log() → [pow2_histogram::add2bin()]
 */
void pow2_histogram::add2bin(int sample) {
  assert(sample >= 0); /* [한국어] 음수 샘플 금지 — 레이턴시는 항상 0 이상 */

  int bin;              /* [한국어] 최종 계산된 bin 인덱스 */
  int v = sample;       /* [한국어] 비트 연산을 위한 샘플 복사본 */
  register unsigned int shift; /* [한국어] 현재 단계의 비트 시프트 양 */

  bin = (v > 0xFFFF) << 4;  /* [한국어] 16비트 초과이면 bin 하위 4비트에 16 설정, v를 16 우측 시프트 준비 */
  v >>= bin;                 /* [한국어] bin 값만큼 v 우측 시프트 (16이면 상위 16비트 제거) */
  shift = (v > 0xFF) << 3;  /* [한국어] 8비트 초과이면 shift=8, 아니면 0 */
  v >>= shift;               /* [한국어] shift만큼 v 우측 시프트 */
  bin |= shift;              /* [한국어] 8비트 기여분을 bin에 누적 */
  shift = (v > 0xF) << 2;   /* [한국어] 4비트 초과이면 shift=4, 아니면 0 */
  v >>= shift;               /* [한국어] shift만큼 v 우측 시프트 */
  bin |= shift;              /* [한국어] 4비트 기여분을 bin에 누적 */
  shift = (v > 0x3) << 1;   /* [한국어] 2비트 초과이면 shift=2, 아니면 0 */
  v >>= shift;               /* [한국어] shift만큼 v 우측 시프트 */
  bin |= shift;              /* [한국어] 2비트 기여분을 bin에 누적 */
  bin |= (v >> 1);           /* [한국어] 나머지 최상위 비트 기여분 누적 → bin = floor(log2(sample)) */
  bin += (sample > 0) ? 1 : 0; /* [한국어] sample>0이면 bin+1. sample=0→bin=0, 1→bin=1, 2→bin=2, ... */

  m_bin_cnts[bin] += 1; /* [한국어] 계산된 bin의 카운터 증가 */

  m_maximum = (sample > m_maximum) ? sample : m_maximum; /* [한국어] 최대값 갱신 */
  m_sum += sample; /* [한국어] 합계에 현재 샘플 누적 (평균 계산용) */
}

/*
 * [한국어]
 * linear_histogram::linear_histogram() - 선형 히스토그램 생성자
 *
 * @stride: bin 당 값 간격. bin i = [i*stride, (i+1)*stride) 범위.
 * @name: 통계 이름, @nbins: bin 개수, @bins: 사용자 bin 경계
 *
 * 기반 클래스 생성자에 위임하고 m_stride 저장.
 */
linear_histogram::linear_histogram(int stride, const char* name, int nbins,
                                   int* bins)
    : binned_histogram(name, nbins, bins), m_stride(stride) {}
/* [한국어] 기반 클래스 생성자에 위임하고 stride 멤버 초기화 */

/*
 * [한국어]
 * linear_histogram::add2bin() - stride로 나눠 bin 인덱스 계산
 *
 * @sample: 분류할 비음수 정수 샘플
 *
 * bin = sample / m_stride (정수 나눗셈). bin >= m_nbins이면 마지막 bin으로 클리핑.
 * 워프 점유율(0~32) 같이 균등하게 분포된 값 수집에 적합.
 * stride=1이면 샘플값이 그대로 bin 인덱스가 됨.
 *
 * 호출 체인:
 *   stat-tool.cc::shader_warp_occ_log() → linear_histogram_logger::log() → [add2bin()]
 */
void linear_histogram::add2bin(int sample) {
  assert(sample >= 0); /* [한국어] 음수 샘플 금지 */

  int bin = sample / m_stride; /* [한국어] 선형 bin 인덱스: sample을 stride로 나눈 몫 */
  if (bin >= m_nbins) bin = m_nbins - 1; /* [한국어] 범위 초과 시 마지막 bin으로 클리핑 */

  m_bin_cnts[bin] += 1; /* [한국어] 해당 bin 카운터 증가 */

  m_maximum = (sample > m_maximum) ? sample : m_maximum; /* [한국어] 최대값 갱신 */
  m_sum += sample; /* [한국어] 합계에 현재 샘플 누적 (평균 계산용) */
}
