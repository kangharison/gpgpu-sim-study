// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Ahmed ElTantawy,
// Vijay Kandiah, Nikos Hardavellas The University of British Columbia,
// Northwestern University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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
 * [한국어 설명] AccelWattch GPU 전력 모델 래퍼 헤더 (gpgpu_sim_wrapper.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍 시뮬레이터와 AccelWattch 전력 모델(McPAT 기반) 사이의
 * 핵심 인터페이스를 정의한다. gpgpu_sim_wrapper 클래스는 GPGPU-Sim의 활동 카운터
 * (IPC, 캐시 히트/미스, 레지스터 접근, 실행 유닛 사용량 등)를 수집하여 McPAT XML
 * 설정 구조체(ParseXML)에 주입하고, McPAT의 proc->compute()를 호출해 각 하드웨어
 * 컴포넌트의 전력 소비(Watts)를 산출한다. AccelWattch는 기존 McPAT을 GPU 특화
 * 구조(SP/SFU/DP/Tensor/Tex 유닛, 공유 메모리, NoC, DVFS 등)로 확장하였으며,
 * MICRO 2021 논문("AccelWattch: A Power Modeling Framework for Modern GPUs")에서
 * 발표되었다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 시뮬레이션 루프(gpu-sim.cc의 gpgpu_sim::cycle())가 매
 * stat_sample_freq 사이클마다 gpgpu_sim_wrapper의 set_*_power() 메서드를 호출하여
 * 활동 카운터를 전달한다. 이후 compute()를 호출하면 McPAT 내부에서 에너지 계산이
 * 수행되고, update_components_power()가 결과를 컴포넌트별로 분리한다. 커널 종료
 * 시점에 print_power_kernel_stats()가 powerfile에 통계를 기록한다.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::cycle()
 *     → set_inst_power() / set_regfile_power() / set_*cache_power() / ...
 *     → compute()                  (McPAT 에너지 계산 트리거)
 *     → update_components_power()  (컴포넌트별 전력 추출)
 *     → power_metrics_calculations() (avg/max/min 갱신)
 *     → print_power_kernel_stats()  (커널 종료 시 리포트 출력)
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 시뮬레이터 스레드에서 호출된다.
 * 멀티스레드 접근은 없으며 별도 동기화 불필요.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - accelwattch/processor.h (McPAT Processor 클래스 — proc->compute() 호출)
 *   - accelwattch/XML_Parse.h (ParseXML — McPAT XML 설정 구조체, perf_count_label,
 *     NUM_PERFORMANCE_COUNTERS, perf_count_t enum 정의)
 *   - zlib (gzopen/gzprintf/gzclose — 압축 트레이스 파일 출력)
 *
 * 이 모듈을 사용하는 모듈:
 *   - src/gpgpu-sim/gpu-sim.cc (gpgpu_sim 클래스가 m_power_stats 멤버로 보유)
 *
 * 데이터 흐름:
 *   GPGPU-Sim 타이밍 시뮬레이터(shader.cc, gpu-cache.cc, dram.cc 등)에서 집계된
 *   활동 카운터 → set_*() 메서드 → ParseXML sys.core[0]/sys.l2/sys.mc/sys.NoC[0]
 *   → proc->compute() → rt_power.readOp.dynamic → sample_cmp_pwr[] → powerfile
 *
 * 공유 자료구조:
 *   - ParseXML* p: McPAT 설정 XML을 파싱한 구조체. 모든 활동 카운터는 이 구조체의
 *     sys.core[0], sys.l2, sys.mc, sys.NoC[0] 필드에 주입된다.
 *   - Processor* proc: McPAT Processor 객체. proc->compute() 호출 후 rt_power에
 *     컴포넌트별 에너지가 채워진다.
 *
 * === 주요 함수/구조체 요약 ===
 * - gpgpu_sim_wrapper(): 벡터 초기화 및 McPAT ParseXML/Processor 객체 생성
 * - init_mcpat(): 커널 실행 전 파일 핸들 오픈, XML 설정 초기화, 헤더 출력
 * - set_inst_power() / set_regfile_power() / set_*cache_power() / ...:
 *     GPGPU-Sim 활동 카운터를 ParseXML 구조체에 주입하는 setter 군
 * - compute(): McPAT proc->compute() 래퍼 — 에너지 계산 트리거
 * - update_components_power(): McPAT rt_power에서 컴포넌트별 전력(Watts) 추출
 * - calculate_static_power(): 평균 활성 스레드 수 기반 선형 누설 전력 계산
 * - power_metrics_calculations(): 샘플별 avg/max/min 전력 통계 누적
 * - print_power_kernel_stats(): 커널 종료 시 powerfile에 통계 출력
 * - detect_print_steady_state(): 슬라이딩 윈도우로 전력 정상 상태 구간 감지
 *
 * 주요 구조체:
 * - avg_max_min_counters<T>: avg/max/min 세 값을 묶는 범용 통계 컨테이너
 * - PowerscalingCoefficients: 실행 유닛별 스케일링 계수 모음 (XML에서 로드)
 */

#ifndef GPGPU_SIM_WRAPPER_H_
#define GPGPU_SIM_WRAPPER_H_

#include <assert.h>   // [한국어] assert() — 런타임 불변 조건 검사 (전력 합산 일치 등)
#include <stdio.h>    // [한국어] printf() — 디버그/에러 메시지 출력
#include <stdlib.h>   // [한국어] exit() — 파일 오픈 실패 등 치명적 오류 시 즉시 종료
#include <string.h>   // [한국어] 문자열 처리 (파일명 조작 등)
#include <zlib.h>     // [한국어] gzopen/gzprintf/gzclose — 전력 트레이스 파일을 압축(gzip) 형식으로 기록
#include <fstream>    // [한국어] std::ofstream — powerfile 텍스트 출력 스트림
#include <iostream>   // [한국어] 표준 입출력 (디버그용)
#include <string>     // [한국어] std::string — kernel_info_string 등 문자열 처리
#include "processor.h" // [한국어] McPAT Processor 클래스 선언 — proc->compute()로 에너지 계산

using namespace std;

/*
 * [한국어]
 * avg_max_min_counters<T> - 평균/최대/최솟값을 묶는 범용 통계 컨테이너 템플릿
 *
 * @T: 집계할 값의 타입 (실제 사용처: double — 전력(Watts), 성능 카운터)
 *
 * 전력 시뮬레이션에서 각 stat_sample_freq 사이클 구간의 샘플을 수집해
 * 커널 및 전체 GPU 수준에서 avg/max/min 전력 통계를 유지하기 위해 사용된다.
 * kernel_cmp_pwr[], kernel_cmp_perf_counters[], kernel_power, gpu_tot_power
 * 필드 모두 이 타입으로 선언된다.
 *
 * 호출 체인:
 *   power_metrics_calculations() → kernel_cmp_pwr[i].avg/max/min 갱신
 *   print_power_kernel_stats()   → kernel_power.avg/max/min 출력
 */
template <typename T>
struct avg_max_min_counters {
  T avg;
  /* [한국어] 평균값 누적 필드.
   * 설정자: power_metrics_calculations()에서 kernel_tot_power / kernel_sample_count
   *         또는 컴포넌트별로 += sample_cmp_pwr[ind]로 누적.
   * 읽는 자: print_power_kernel_stats()에서 avg / kernel_sample_count로 나눠 출력.
   * 값 범위: 0.0 이상의 double (단위: Watts 또는 카운터 횟수).
   * 동기화: 단일 시뮬레이터 스레드에서만 접근하므로 락 불필요. */

  T max;
  /* [한국어] 관찰된 최댓값 필드.
   * 설정자: power_metrics_calculations()에서 sample_power > kernel_power.max 조건 시 갱신.
   * 읽는 자: print_power_kernel_stats()에서 "kernel_max_power" 줄로 출력.
   * 값 범위: 0.0 이상의 double. 초기화 시 0으로 설정(avg_max_min_counters 생성자).
   * 동기화: 단일 스레드 접근. */

  T min;
  /* [한국어] 관찰된 최솟값 필드.
   * 설정자: power_metrics_calculations()에서 sample_power < kernel_power.min
   *         또는 kernel_power.min == 0 조건 시 갱신.
   * 읽는 자: print_power_kernel_stats()에서 "kernel_min_power" 줄로 출력.
   * 값 범위: 0.0 이상의 double. 초기화 시 0 (첫 샘플에서 무조건 갱신됨).
   * 동기화: 단일 스레드 접근. */

  /*
   * [한국어]
   * avg_max_min_counters() - avg/max/min 을 0으로 초기화하는 기본 생성자
   *
   * @return: 없음 (생성자)
   *
   * reset_counters() 내부에서 이 기본 생성자로 만든 init 객체를
   * kernel_cmp_pwr[], kernel_cmp_perf_counters[] 등에 대입하여 커널 간
   * 통계를 초기화한다.
   *
   * 호출 체인:
   *   reset_counters() → avg_max_min_counters<double> init; → kernel_cmp_pwr[i] = init;
   */
  avg_max_min_counters() {
    avg = 0; // [한국어] 평균 누적값 0으로 초기화 — 커널/샘플 시작 시 깨끗한 상태 보장
    max = 0; // [한국어] 최댓값 0으로 초기화 — 첫 샘플이 반드시 max를 갱신하도록
    min = 0; // [한국어] 최솟값 0으로 초기화 — min==0 조건을 power_metrics_calculations()에서 "아직 미설정"으로 활용
  }
};

#ifndef COEFF_STRUCT
#define COEFF_STRUCT

/*
 * [한국어]
 * PowerscalingCoefficients - AccelWattch 실행 유닛별 전력 스케일링 계수 구조체
 *
 * AccelWattch는 McPAT의 단일 FPU/SFU 모델을 GPU의 세분화된 실행 유닛
 * (SP FP32, SFU int-mul/div, DP FP64, Tensor, Tex 등)으로 분리하기 위해
 * 각 유닛에 독립적인 스케일링 계수를 부여한다.
 * 이 구조체의 값들은 ParseXML의 sys.scaling_coefficients[] 배열에서
 * get_scaling_coeffs()가 추출하여 채운다.
 *
 * 사용처: gpgpu_sim_wrapper::get_scaling_coeffs()가 이 구조체를 힙에 할당해
 *         반환하면, gpu-sim.cc의 shader_core_ctx가 이를 읽어 각 유닛의
 *         접근 횟수에 곱해 McPAT에 전달할 "유효 접근 횟수"를 산출한다.
 *
 * 동기화: 시뮬레이션 초기화 시 1회 생성 후 읽기 전용으로 사용.
 *         멀티스레드 접근 없음.
 */
struct PowerscalingCoefficients {
  double int_coeff;
  /* [한국어] 정수 ALU(IALU) 접근 횟수에 곱할 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[INT_ACC]에서 복사.
   * 읽는 자: shader_core_ctx::set_int_accesses() 호출 경로에서 ialu_accesses에 곱함.
   * 값 범위: 0.0 초과 실수 (XML gpgpusim.config의 scaling_coefficients 섹션에 의존).
   * 동기화: 초기화 후 읽기 전용. */

  double int_mul_coeff;
  /* [한국어] 정수 일반 곱셈(IMUL) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[INT_MUL_ACC]에서 복사.
   * 읽는 자: SFU 접근 횟수 분배 시 imul_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double int_mul24_coeff;
  /* [한국어] 24비트 정수 곱셈(IMUL24) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[INT_MUL24_ACC]에서 복사.
   * 읽는 자: imul24_accesses에 곱해 McPAT SFU 전력 계산에 반영.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double int_mul32_coeff;
  /* [한국어] 32비트 정수 곱셈(IMUL32) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[INT_MUL32_ACC]에서 복사.
   * 읽는 자: imul32_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double int_div_coeff;
  /* [한국어] 정수 나눗셈(IDIV) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[INT_DIV_ACC]에서 복사.
   * 읽는 자: idiv_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double fp_coeff;
  /* [한국어] 단정밀도 FP(FP32) ALU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_ACC]에서 복사.
   * 읽는 자: fpu_accesses(SP 유닛)에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double dp_coeff;
  /* [한국어] 배정밀도 FP(FP64, DP) ALU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[DP_ACC]에서 복사.
   * 읽는 자: dpu_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double fp_mul_coeff;
  /* [한국어] FP32 곱셈(FMUL) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_MUL_ACC]에서 복사.
   * 읽는 자: fpmul_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double fp_div_coeff;
  /* [한국어] FP32 나눗셈(FDIV) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_DIV_ACC]에서 복사.
   * 읽는 자: fpdiv_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double dp_mul_coeff;
  /* [한국어] FP64 곱셈(DMUL) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[DP_MUL_ACC]에서 복사.
   * 읽는 자: dpmul_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double dp_div_coeff;
  /* [한국어] FP64 나눗셈(DDIV) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[DP_DIV_ACC]에서 복사.
   * 읽는 자: dpdiv_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double sqrt_coeff;
  /* [한국어] FP 제곱근(SQRT) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_SQRT_ACC]에서 복사.
   * 읽는 자: sqrt_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double log_coeff;
  /* [한국어] FP 로그(LG2) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_LG_ACC]에서 복사.
   * 읽는 자: log_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double sin_coeff;
  /* [한국어] FP 사인(SIN) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_SIN_ACC]에서 복사.
   * 읽는 자: sin_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double exp_coeff;
  /* [한국어] FP 지수(EX2) SFU 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[FP_EXP_ACC]에서 복사.
   * 읽는 자: exp_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */

  double tensor_coeff;
  /* [한국어] Tensor Core 접근 횟수 스케일링 계수 (Volta/Turing 이후 GPU).
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[TENSOR_ACC]에서 복사.
   * 읽는 자: tensor_accesses에 곱함.
   * 값 범위: 0.0 이상 실수. Tensor Core 미지원 GPU에서는 0.
   * 동기화: 읽기 전용. */

  double tex_coeff;
  /* [한국어] 텍스처 유닛(TEX) 접근 횟수 스케일링 계수.
   * 설정자: get_scaling_coeffs()가 p->sys.scaling_coefficients[TEX_ACC]에서 복사.
   * 읽는 자: tex_accesses에 곱함.
   * 값 범위: 0.0 이상 실수.
   * 동기화: 읽기 전용. */
};

#endif

/*
 * [한국어]
 * gpgpu_sim_wrapper - GPGPU-Sim 타이밍 시뮬레이터와 AccelWattch(McPAT) 전력 모델의 중간 계층 클래스
 *
 * 이 클래스는 다음 세 역할을 수행한다:
 *   1. 활동 카운터 주입기: gpu-sim.cc로부터 전달받은 각종 접근 횟수를
 *      McPAT의 ParseXML 구조체(p->sys.core[0], p->sys.l2, p->sys.mc, p->sys.NoC[0])에
 *      스케일링 계수를 적용해 주입한다.
 *   2. McPAT 전력 계산 트리거: compute()로 proc->compute()를 호출해 에너지(J)를
 *      Watts 단위 전력으로 환산한다.
 *   3. 통계 관리자: 샘플별/커널별/전체 GPU 수준의 avg/max/min 전력 통계를 유지하고
 *      powerfile 및 gzip 압축 트레이스 파일에 기록한다.
 *
 * 인스턴스 수명: 시뮬레이터 실행 전체에 걸쳐 1개 인스턴스(gpgpu_sim의 멤버)가 존재한다.
 * 커널 경계마다 reset_counters()로 per-kernel 통계를 초기화하고 재사용한다.
 *
 * DVFS(Dynamic Voltage Frequency Scaling) 지원:
 *   g_dvfs_enabled가 true일 때 update_components_power()에서 voltage_ratio를 사용해
 *   동적 전력은 voltage_ratio^2, 누설 전력은 voltage_ratio^1로 스케일링한다.
 */
class gpgpu_sim_wrapper {
 public:
  /*
   * [한국어]
   * gpgpu_sim_wrapper() - 전력 모델 래퍼 생성자
   *
   * @power_simulation_enabled: 전력 시뮬레이션 활성화 여부 (false면 McPAT 파싱 건너뜀)
   * @xmlfile: McPAT XML 설정 파일 경로 (gpgpusim의 -accelwattch_xml_file 옵션)
   * @power_simulation_mode: 전력 시뮬레이션 모드 번호 (0=HW측정, 1=simulation 등)
   * @dvfs_enabled: DVFS 전압 스케일링 적용 여부
   * @return: 없음 (생성자)
   *
   * 벡터/카운터를 0으로 초기화하고, ParseXML을 파싱 후 Processor 객체를 생성한다.
   * gpu-sim.cc::gpgpu_sim 생성자에서 m_power_stats = new gpgpu_sim_wrapper(...) 형태로 호출.
   *
   * 호출 체인:
   *   gpgpu_sim::gpgpu_sim() → gpgpu_sim_wrapper()
   */
  gpgpu_sim_wrapper(bool power_simulation_enabled, char* xmlfile,
                    int power_simulation_mode, bool dvfs_enabled);

  /*
   * [한국어]
   * ~gpgpu_sim_wrapper() - 소멸자 (빈 구현)
   *
   * @return: 없음
   *
   * Processor*, ParseXML* 등은 소멸자에서 명시적으로 해제하지 않는다.
   * 시뮬레이터 종료 시 OS가 회수한다.
   *
   * 호출 체인:
   *   gpgpu_sim 소멸 → ~gpgpu_sim_wrapper()
   */
  ~gpgpu_sim_wrapper();

  /*
   * [한국어]
   * init_mcpat() - 커널 실행 전 McPAT 및 출력 파일 초기화
   *
   * @xmlfile: McPAT XML 설정 파일 경로
   * @powerfile: 전력 통계 텍스트 출력 파일 경로
   * @power_trace_file: 사이클별 전력 트레이스 파일 경로 (gzip 압축)
   * @metric_trace_file: 사이클별 성능 카운터 트레이스 파일 경로 (gzip 압축)
   * @steady_state_file: 정상 상태 전력 추적 파일 경로 (gzip 압축)
   * @power_sim_enabled: 전력 시뮬레이션 활성화 플래그
   * @trace_enabled: 트레이스 파일 기록 활성화 플래그
   * @steady_state_enabled: 정상 상태 감지 활성화 플래그
   * @power_per_cycle_dump: 사이클별 전력 덤프 활성화 플래그
   * @steady_power_deviation: 정상 상태 판단 전력 편차 임계값 (Watts)
   * @steady_min_period: 정상 상태로 간주할 최소 샘플 수
   * @zlevel: gzip 압축 수준 (0-9)
   * @init_val: 현재 커널 시작 시점의 전체 명령어 수 (정상 상태 IPC 계산용)
   * @stat_sample_freq: 전력 샘플링 주기 (GPU 사이클 수)
   * @power_sim_mode: 전력 시뮬레이션 모드
   * @dvfs_enabled: DVFS 스케일링 활성화 플래그
   * @clock_freq: GPU 코어 클럭 주파수 (MHz)
   * @num_shaders: 시뮬레이션할 SM 수
   * @return: 없음
   *
   * static bool mcpat_init 플래그로 최초 1회만 파일 헤더를 기록하고 파일을 오픈한다.
   * reset_counters()로 per-kernel 통계를 초기화한 뒤 XML 설정값을 반영한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() 또는 커널 런치 → init_mcpat()
   */
  void init_mcpat(char* xmlfile, char* powerfile, char* power_trace_file,
                  char* metric_trace_file, char* steady_state_file,
                  bool power_sim_enabled, bool trace_enabled,
                  bool steady_state_enabled, bool power_per_cycle_dump,
                  double steady_power_deviation, double steady_min_period,
                  int zlevel, double init_val, int stat_sample_freq,
                  int power_sim_mode, bool dvfs_enabled, unsigned clock_freq,
                  unsigned num_shaders);

  /*
   * [한국어]
   * init_mcpat_hw_mode() - HW 측정 모드에서 총 시뮬레이션 사이클 수를 McPAT에 설정
   *
   * @gpu_sim_cycle: 현재 커널의 시뮬레이션 사이클 수
   * @return: 없음
   *
   * p->sys.total_cycles에 gpu_sim_cycle을 대입해 McPAT가 실행 시간을 올바르게
   * 계산하도록 한다. HW 전력 측정값을 기반으로 하는 모드(power_sim_mode=0)에서
   * compute() 전에 호출된다.
   *
   * 호출 체인:
   *   gpu-sim.cc::gpgpu_sim::cycle() → init_mcpat_hw_mode()
   */
  void init_mcpat_hw_mode(unsigned gpu_sim_cycle);

  /*
   * [한국어]
   * detect_print_steady_state() - 슬라이딩 윈도우 방식으로 전력 정상 상태 구간을 감지·기록
   *
   * @position: 0이면 사이클 중간 샘플, 1이면 커널 종료 시점 강제 출력
   * @init_val: 이 시점까지의 총 명령어 수 (IPC 계산 기준값)
   * @return: 없음
   *
   * g_steady_power_levels_enabled가 true일 때만 동작한다.
   * 현재 샘플 전력이 슬라이딩 평균 ± gpu_steady_power_deviation 이내이면
   * samples[] 벡터에 추가하고, 편차 초과 시 print_steady_state()를 호출한다.
   * steady_state_tacking_file을 매 호출마다 open/close하여 실시간 기록을 보장한다.
   *
   * 호출 체인:
   *   print_power_kernel_stats() → detect_print_steady_state(1, ...)
   *   gpgpu_sim::cycle() → detect_print_steady_state(0, ...)
   */
  void detect_print_steady_state(int position, double init_val);

  /*
   * [한국어]
   * close_files() - gzip 트레이스 파일 핸들 닫기
   *
   * @return: 없음
   *
   * print_trace_files()가 open_files()와 짝으로 호출한다.
   * g_power_simulation_enabled && g_power_trace_enabled 조건에서만 동작한다.
   *
   * 호출 체인:
   *   print_trace_files() → open_files() / [트레이스 기록] / close_files()
   */
  void close_files();

  /*
   * [한국어]
   * open_files() - gzip 트레이스 파일을 append 모드로 열기
   *
   * @return: 없음
   *
   * print_trace_files() 내부에서 호출된다. 파일을 "a"(append) 모드로 열어
   * 이전 사이클 기록에 이어 새 샘플을 추가한다.
   *
   * 호출 체인:
   *   print_trace_files() → open_files()
   */
  void open_files();

  /*
   * [한국어]
   * compute() - McPAT proc->compute() 래퍼 — 에너지 계산 실행
   *
   * @return: 없음
   *
   * ParseXML에 모든 활동 카운터가 설정된 후 이 함수를 호출하면
   * McPAT 내부에서 각 컴포넌트의 에너지(J)와 전력(W)이 계산된다.
   * 결과는 proc->rt_power.readOp.dynamic 및 각 서브컴포넌트의 rt_power에 저장된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → compute() → proc->compute() [McPAT 내부]
   */
  void compute();

  /*
   * [한국어]
   * dump() - 사이클별 에너지 덤프 (g_power_per_cycle_dump 옵션)
   *
   * @return: 없음
   *
   * g_power_per_cycle_dump가 true일 때 proc->displayEnergy(2, 5)를 호출해
   * 표준 출력에 컴포넌트별 에너지를 출력한다. 디버깅 용도.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → dump()
   */
  void dump();

  /*
   * [한국어]
   * print_trace_files() - 현재 샘플의 성능 카운터와 전력 값을 gzip 파일에 기록
   *
   * @return: 없음
   *
   * metric_trace_file에 sample_perf_counters[], power_trace_file에 proc_power 및
   * sample_cmp_pwr[]를 쉼표 구분 형식으로 기록한다. 한 줄 = 한 샘플.
   *
   * 호출 체인:
   *   print_power_kernel_stats() → print_trace_files()
   *   → open_files() / gzprintf() / close_files()
   */
  void print_trace_files();

  /*
   * [한국어]
   * update_components_power() - McPAT rt_power에서 컴포넌트별 전력(Watts)을 추출
   *
   * @return: 없음
   *
   * compute() 호출 후 proc의 각 서브컴포넌트(ifu, lsu, exu, rfu 등)의
   * rt_power.readOp.dynamic을 executionTime으로 나눠 Watts 단위 전력을 얻고,
   * sample_cmp_pwr[] 배열에 저장한다. FPU/SFU 전력은 총 접근 횟수 대비 비율로
   * 세분화한다. DVFS 활성화 시 voltage_ratio^2 (동적), voltage_ratio^1 (누설) 적용.
   * CONSTP(상수 동적 전력)와 STATICP(calculate_static_power() 결과)도 여기서 설정.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → update_components_power()
   *   → update_coefficients() / calculate_static_power()
   */
  void update_components_power();

  /*
   * [한국어]
   * calculate_static_power() - 평균 활성 스레드 수 기반 선형 누설 전력 계산
   *
   * @return: 현재 샘플의 추정 누설 전력 (Watts)
   *
   * AccelWattch의 핵심 기여 중 하나로, GPU SM의 워프 수준 스레드 발산(divergence)에
   * 따라 활성 레인 수가 달라지므로 누설 전력도 달라진다.
   * 선형 모델: total_static_power = base_static_power + (avg_threads_per_warp - 1) * lane_static_power
   * 7가지 기능 유닛 조합(INT, INT_FP, INT_FP_DP, INT_FP_SFU, INT_FP_TEX,
   * INT_FP_TENSOR, LIGHT_SM)에 따라 base/lane 파라미터를 선택한다.
   * per_active_core = (num_cores - num_idle_cores) / num_cores 로 유휴 SM을 제외.
   *
   * 호출 체인:
   *   update_components_power() → calculate_static_power()
   */
  double calculate_static_power();

  /*
   * [한국어]
   * update_coefficients() - McPAT 내부 계수를 읽어 initpower_coeff/effpower_coeff 갱신
   *
   * @return: 없음
   *
   * compute() 직후 각 컴포넌트의 에너지/접근 횟수 비율을 McPAT proc->cores[0]의
   * get_coefficient_*() 메서드로 읽어 initpower_coeff[]와 effpower_coeff[]에 저장한다.
   * 이 계수들은 calculate_static_power()의 카테고리 판별 및 per-component 전력 분리에 쓰인다.
   * FPU/SFU 계수는 전체 접근 대비 각 유닛 접근 비율로 정규화한다.
   * 마지막으로 executionTime으로 나눠 시간 정규화(Watts = J/s)를 수행한다.
   *
   * 호출 체인:
   *   update_components_power() → update_coefficients()
   */
  void update_coefficients();

  /*
   * [한국어]
   * reset_counters() - 커널 경계에서 per-kernel 통계 카운터 초기화
   *
   * @return: 없음
   *
   * sample_perf_counters[], kernel_cmp_perf_counters[], sample_cmp_pwr[],
   * kernel_cmp_pwr[], kernel_sample_count, kernel_tot_power, kernel_power,
   * avg_threads_per_warp_tot를 모두 0으로 초기화한다.
   * init_mcpat()의 첫 단계에서 호출되어 이전 커널의 데이터가 새 커널에
   * 섞이지 않도록 보장한다.
   *
   * 호출 체인:
   *   init_mcpat() → reset_counters()
   */
  void reset_counters();

  /*
   * [한국어]
   * print_power_kernel_stats() - 커널 종료 시 전력 통계를 powerfile에 출력
   *
   * @gpu_sim_cycle: 현재 커널의 시뮬레이션 사이클 수
   * @gpu_tot_sim_cycle: 전체 누적 시뮬레이션 사이클 수
   * @init_value: 커널 시작 시점의 총 명령어 수
   * @kernel_info_string: 커널 이름/ID 등 식별 문자열
   * @print_trace: true이면 print_trace_files()도 호출
   * @return: 없음
   *
   * detect_print_steady_state(1, ...)를 먼저 호출해 정상 상태 기록을 마무리한다.
   * g_power_simulation_enabled가 true일 때만 powerfile에 avg/max/min 통계를 출력하고,
   * 전체 GPU 누적 통계(gpu_tot_power)도 함께 기록한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() (커널 종료 감지) → print_power_kernel_stats()
   */
  void print_power_kernel_stats(double gpu_sim_cycle, double gpu_tot_sim_cycle,
                                double init_value,
                                const std::string& kernel_info_string,
                                bool print_trace);

  /*
   * [한국어]
   * power_metrics_calculations() - 현재 샘플 전력으로 avg/max/min 통계 갱신
   *
   * @return: 없음
   *
   * compute() + update_components_power() 이후 매 샘플마다 호출된다.
   * sample_power = proc->rt_power.readOp.dynamic + CONSTP + STATICP 를 현재 샘플 전력으로 사용.
   * kernel_sample_count와 total_sample_count를 각각 증가시키고,
   * kernel_power 및 kernel_cmp_pwr[]의 avg/max/min을 갱신한다.
   * gpu_tot_power는 전체 커널에 걸친 누적 avg/max/min이다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → power_metrics_calculations()
   */
  void power_metrics_calculations();

  /*
   * [한국어]
   * set_model_voltage() - DVFS 시 모델링 전압 설정
   *
   * @model_voltage: 현재 동작 전압 (V)
   * @return: 없음
   *
   * modeled_chip_voltage에 저장되며, update_components_power()에서
   * voltage_ratio = model_voltage / p->sys.modeled_chip_voltage_ref 로
   * 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_model_voltage()
   */
  void set_model_voltage(double model_voltage);

  /*
   * [한국어]
   * set_inst_power() - 인스트럭션 실행 관련 활동 카운터를 McPAT XML 구조체에 주입
   *
   * @clk_gated_lanes: 클럭 게이팅된 SIMT 레인 유무 (클럭 게이팅 전력 감소 반영)
   * @tot_cycles: SM이 실행한 총 사이클 수
   * @busy_cycles: SM이 실제로 명령어를 실행한 사이클 수 (idle 제외)
   * @tot_inst: 총 실행 명령어 수 (모든 유닛 합산)
   * @int_inst: 정수 명령어 수
   * @fp_inst: FP 명령어 수
   * @load_inst: 로드 명령어 수
   * @store_inst: 스토어 명령어 수
   * @committed_inst: 커밋(완료) 명령어 수
   * @return: 없음
   *
   * p->sys.core[0]의 total_instructions, int_instructions, fp_instructions 등에
   * 스케일링 계수(scaling_coefficients[TOT_INST], [FP_INT])를 곱해 저장한다.
   * sample_perf_counters[FP_INT/TOT_INST]도 갱신하여 트레이스 파일에 기록.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_inst_power()
   */
  void set_inst_power(bool clk_gated_lanes, double tot_cycles,
                      double busy_cycles, double tot_inst, double int_inst,
                      double fp_inst, double load_inst, double store_inst,
                      double committed_inst);

  /*
   * [한국어]
   * set_regfile_power() - 레지스터 파일 접근 카운터를 McPAT XML 구조체에 주입
   *
   * @reads: 레지스터 읽기 횟수
   * @writes: 레지스터 쓰기 횟수
   * @ops: 비레지스터 오퍼랜드(immediate, 상수 등) 접근 횟수
   * @return: 없음
   *
   * p->sys.core[0].int_regfile_reads/writes, non_rf_operands에
   * scaling_coefficients[REG_RD/WR/NON_REG_OPs]를 곱해 저장한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_regfile_power()
   */
  void set_regfile_power(double reads, double writes, double ops);

  /*
   * [한국어]
   * set_icache_power() - 명령어 캐시(ICache) 접근 카운터를 McPAT XML에 주입
   *
   * @accesses: I-Cache 히트 횟수
   * @misses: I-Cache 미스 횟수
   * @return: 없음
   *
   * read_accesses = hits*IC_H_coeff + misses*IC_M_coeff, read_misses = misses*IC_M_coeff.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_icache_power()
   */
  void set_icache_power(double accesses, double misses);

  /*
   * [한국어]
   * set_ccache_power() - 상수 캐시(Constant Cache) 접근 카운터를 McPAT XML에 주입
   *
   * @accesses: 히트 횟수
   * @misses: 미스 횟수
   * @return: 없음
   *
   * p->sys.core[0].ccache.read_accesses/read_misses에 CC_H/CC_M 계수를 적용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_ccache_power()
   */
  void set_ccache_power(double accesses, double misses);

  /*
   * [한국어]
   * set_tcache_power() - 텍스처 캐시(Texture Cache) 접근 카운터를 McPAT XML에 주입
   *
   * @accesses: 히트 횟수
   * @misses: 미스 횟수
   * @return: 없음
   *
   * p->sys.core[0].tcache.read_accesses/read_misses에 TC_H/TC_M 계수를 적용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_tcache_power()
   */
  void set_tcache_power(double accesses, double misses);

  /*
   * [한국어]
   * set_shrd_mem_power() - 공유 메모리(Shared Memory) 접근 카운터를 McPAT XML에 주입
   *
   * @accesses: 공유 메모리 접근 횟수 (읽기+쓰기 합산)
   * @return: 없음
   *
   * p->sys.core[0].sharedmemory.read_accesses에 SHRD_ACC 계수를 적용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_shrd_mem_power()
   */
  void set_shrd_mem_power(double accesses);

  /*
   * [한국어]
   * set_l1cache_power() - L1 데이터 캐시(DCache) 접근 카운터를 McPAT XML에 주입
   *
   * @read_accesses: 읽기 히트 횟수
   * @read_misses: 읽기 미스 횟수
   * @write_accesses: 쓰기 히트 횟수
   * @write_misses: 쓰기 미스 횟수
   * @return: 없음
   *
   * p->sys.core[0].dcache의 read/write accesses/misses에 DC_RH/RM/WH/WM 계수 적용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_l1cache_power()
   */
  void set_l1cache_power(double read_accesses, double read_misses,
                         double write_accesses, double write_misses);

  /*
   * [한국어]
   * set_l2cache_power() - L2 캐시 접근 카운터를 McPAT XML에 주입
   *
   * @read_accesses: L2 읽기 히트 횟수
   * @read_misses: L2 읽기 미스 횟수
   * @write_accesses: L2 쓰기 히트 횟수
   * @write_misses: L2 쓰기 미스 횟수
   * @return: 없음
   *
   * p->sys.l2의 total_accesses, read_accesses, write_accesses, 히트/미스 필드 모두 설정.
   * L2는 SM 전체 공유 캐시이므로 sys.core[0]가 아닌 sys.l2를 사용한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_l2cache_power()
   */
  void set_l2cache_power(double read_accesses, double read_misses,
                         double write_accesses, double write_misses);

  /*
   * [한국어]
   * set_num_cores() - 전체 SM 수를 num_cores에 저장
   *
   * @num_core: 시뮬레이션 중인 총 SM 수
   * @return: 없음
   *
   * calculate_static_power()에서 per_active_core 비율 계산에 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_num_cores()
   */
  void set_num_cores(double num_core);

  /*
   * [한국어]
   * set_idle_core_power() - 유휴 SM 수를 McPAT XML 및 내부 필드에 설정
   *
   * @num_idle_core: 현재 샘플에서 유휴 상태인 SM 수
   * @return: 없음
   *
   * p->sys.num_idle_cores에 저장하고 num_idle_cores 내부 필드도 동기화.
   * calculate_static_power()의 per_active_core 계산에 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_idle_core_power()
   */
  void set_idle_core_power(double num_idle_core);

  /*
   * [한국어]
   * set_duty_cycle_power() - 파이프라인 duty cycle을 McPAT XML에 설정
   *
   * @duty_cycle: SM 파이프라인 사용률 (0.0~1.0)
   * @return: 없음
   *
   * p->sys.core[0].pipeline_duty_cycle에 PIPE_A 계수를 적용.
   * 파이프라인 레지스터 동적 전력(PIPEP) 계산에 영향을 준다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_duty_cycle_power()
   */
  void set_duty_cycle_power(double duty_cycle);

  /*
   * [한국어]
   * set_mem_ctrl_power() - DRAM 메모리 컨트롤러 접근 카운터를 McPAT XML에 주입
   *
   * @reads: DRAM 읽기 명령 횟수
   * @writes: DRAM 쓰기 명령 횟수
   * @dram_precharge: DRAM 프리차지 명령 횟수
   * @return: 없음
   *
   * p->sys.mc.memory_accesses/reads/writes와 dram_pre에
   * MEM_RD/MEM_WR/MEM_PRE 계수를 적용해 저장한다.
   * update_components_power()에서 MCP(메모리 컨트롤러)와 DRAMP(DRAM) 전력을 분리 추출.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_mem_ctrl_power()
   */
  void set_mem_ctrl_power(double reads, double writes, double dram_precharge);

  /*
   * [한국어]
   * set_exec_unit_power() - FPU/IALU/SFU 총 접근 횟수를 McPAT XML에 직접 설정
   *
   * @fpu_accesses: FP 유닛(SP FP32 + DP FP64) 총 접근 횟수
   * @ialu_accesses: 정수 ALU 접근 횟수
   * @sfu_accesses: SFU 총 접근 횟수 (int-mul/div, fp-sqrt/log/sin/exp, tensor, tex 합산)
   * @return: 없음
   *
   * p->sys.core[0].fpu_accesses, ialu_accesses, mul_accesses에 저장하고,
   * tot_fpu_accesses, tot_sfu_accesses에도 내부 사본을 유지한다.
   * update_components_power()에서 개별 유닛 전력을 비율로 분리할 때 분모로 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_exec_unit_power()
   */
  void set_exec_unit_power(double fpu_accesses, double ialu_accesses,
                           double sfu_accesses);

  /*
   * [한국어]
   * set_int_accesses() - 정수 실행 유닛별 세분화 접근 횟수를 sample_perf_counters에 기록
   *
   * @ialu_accesses: 정수 ALU(IADD 등) 접근 횟수
   * @imul24_accesses: 24비트 정수 곱셈 SFU 접근 횟수
   * @imul32_accesses: 32비트 정수 곱셈 SFU 접근 횟수
   * @imul_accesses: 일반 정수 곱셈 SFU 접근 횟수
   * @idiv_accesses: 정수 나눗셈 SFU 접근 횟수
   * @return: 없음
   *
   * sample_perf_counters[]에만 저장하고 McPAT XML에는 직접 주입하지 않는다.
   * update_components_power()에서 SFU 전력을 비율로 분배할 때 이 값들을 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_int_accesses()
   */
  void set_int_accesses(double ialu_accesses, double imul24_accesses,
                        double imul32_accesses, double imul_accesses,
                        double idiv_accesses);

  /*
   * [한국어]
   * set_dp_accesses() - FP64(배정밀도) 유닛별 접근 횟수를 sample_perf_counters에 기록
   *
   * @dpu_accesses: DP ALU 접근 횟수
   * @dpmul_accesses: DP 곱셈 접근 횟수
   * @dpdiv_accesses: DP 나눗셈 접근 횟수
   * @return: 없음
   *
   * update_components_power()에서 tot_fpu_accesses 대비 비율로 DPUP 전력 분배에 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_dp_accesses()
   */
  void set_dp_accesses(double dpu_accesses, double dpmul_accesses,
                       double dpdiv_accesses);

  /*
   * [한국어]
   * set_fp_accesses() - FP32(단정밀도) 유닛별 접근 횟수를 sample_perf_counters에 기록
   *
   * @fpu_accesses: SP FP32 ALU 접근 횟수
   * @fpmul_accesses: SP FP32 곱셈 접근 횟수
   * @fpdiv_accesses: SP FP32 나눗셈 접근 횟수
   * @return: 없음
   *
   * update_components_power()에서 FPUP 전력 분배에 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_fp_accesses()
   */
  void set_fp_accesses(double fpu_accesses, double fpmul_accesses,
                       double fpdiv_accesses);

  /*
   * [한국어]
   * set_trans_accesses() - 초월함수(SQRT/LOG/SIN/EXP) SFU 접근 횟수를 sample_perf_counters에 기록
   *
   * @sqrt_accesses: SQRT SFU 접근 횟수
   * @log_accesses: LG2 SFU 접근 횟수
   * @sin_accesses: SIN SFU 접근 횟수
   * @exp_accesses: EX2 SFU 접근 횟수
   * @return: 없음
   *
   * tot_sfu_accesses 대비 비율로 FP_SQRTP/FP_LGP/FP_SINP/FP_EXP 전력 분배에 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_trans_accesses()
   */
  void set_trans_accesses(double sqrt_accesses, double log_accesses,
                          double sin_accesses, double exp_accesses);

  /*
   * [한국어]
   * set_tensor_accesses() - Tensor Core 접근 횟수를 sample_perf_counters에 기록
   *
   * @tensor_accesses: Tensor Core 접근 횟수 (WMMA 명령어 등)
   * @return: 없음
   *
   * tot_sfu_accesses 대비 비율로 TENSORP 전력 분배에 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_tensor_accesses()
   */
  void set_tensor_accesses(double tensor_accesses);

  /*
   * [한국어]
   * set_tex_accesses() - 텍스처 유닛 접근 횟수를 sample_perf_counters에 기록
   *
   * @tex_accesses: 텍스처 페치 접근 횟수
   * @return: 없음
   *
   * tot_sfu_accesses 대비 비율로 TEXP 전력 분배에 사용.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_tex_accesses()
   */
  void set_tex_accesses(double tex_accesses);

  /*
   * [한국어]
   * set_avg_active_threads() - 평균 활성 스레드 수(warp당)를 내부 필드에 저장
   *
   * @active_threads: 현재 샘플의 warp당 평균 활성 스레드 수 (0~32)
   * @return: 없음
   *
   * ceil()로 올림하여 avg_threads_per_warp(unsigned)에 저장하고,
   * avg_threads_per_warp_tot에 누적한다. calculate_static_power()의 선형 모델에 사용.
   * 높은 스레드 수 = 낮은 warp divergence = 더 많은 활성 레인 = 더 높은 누설 전력.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_avg_active_threads()
   */
  void set_avg_active_threads(float active_threads);

  /*
   * [한국어]
   * set_active_lanes_power() - SP/SFU 유닛의 평균 활성 레인 수를 McPAT XML에 설정
   *
   * @sp_avg_active_lane: SP(FP32) 유닛의 평균 활성 SIMT 레인 수 (0~32)
   * @sfu_avg_active_lane: SFU 유닛의 평균 활성 SIMT 레인 수 (0~32)
   * @return: 없음
   *
   * p->sys.core[0].sp_average_active_lanes, sfu_average_active_lanes에 저장.
   * McPAT 내부에서 레인별 동적 전력 계산에 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_active_lanes_power()
   */
  void set_active_lanes_power(double sp_avg_active_lane,
                              double sfu_avg_active_lane);

  /*
   * [한국어]
   * set_NoC_power() - GPU 내부 NoC(Network-on-Chip, intersim2) 접근 횟수를 McPAT XML에 주입
   *
   * @noc_tot_acc: NoC 총 접근(패킷 전송) 횟수
   * @return: 없음
   *
   * p->sys.NoC[0].total_accesses에 NOC_A 계수를 적용해 저장.
   * update_components_power()에서 NOCP(NoC 전력) 계산에 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → set_NoC_power()
   */
  void set_NoC_power(double noc_tot_acc);

  /*
   * [한국어]
   * sanity_check() - 두 double 값이 거의 같은지 상대/절대 오차로 검사
   *
   * @a: 검사할 첫 번째 값 (보통 sum_pwr_cmp — 컴포넌트 합산 전력)
   * @b: 검사할 두 번째 값 (보통 proc_power — McPAT 총 전력)
   * @return: true이면 거의 동일 (오차 < 0.001%), false이면 불일치
   *
   * b==0이면 절대 오차 < 0.00001, 그 외에는 상대 오차 < 0.00001 기준.
   * update_components_power() 마지막에 컴포넌트 합산이 총 전력과 일치하는지
   * assert()와 함께 검증한다. DVFS 활성화 시 이 검사는 건너뛴다(주석 참조).
   *
   * 호출 체인:
   *   update_components_power() → sanity_check()
   *   print_power_kernel_stats() → sanity_check()
   */
  bool sanity_check(double a, double b);

  /*
   * [한국어]
   * get_scaling_coeffs() - XML에서 로드된 스케일링 계수를 PowerscalingCoefficients 구조체로 반환
   *
   * @return: 힙 할당된 PowerscalingCoefficients 포인터 (호출자가 delete 책임)
   *
   * p->sys.scaling_coefficients[] 배열에서 각 유닛의 계수를 읽어
   * PowerscalingCoefficients 멤버에 복사한 후 반환한다.
   * gpu-sim.cc에서 호출해 shader_core_ctx에 전달하며, 이 계수로 활동 카운터를
   * 스케일링한 값을 McPAT에 다시 입력하는 피드백 루프를 구성한다.
   *
   * 호출 체인:
   *   gpgpu_sim::init() → get_scaling_coeffs() → shader_core_ctx::set_scaling_coeffs()
   */
  PowerscalingCoefficients* get_scaling_coeffs();

 private:
  /*
   * [한국어]
   * print_steady_state() - 감지된 정상 상태 구간의 통계를 steady_state_tacking_file에 기록
   *
   * @position: 1이면 커널 종료 시점 (has_written_avg 미달 시 에러 메시지 출력)
   * @init_val: 이 시점까지의 총 명령어 수
   * @return: 없음
   *
   * samples[] 슬라이딩 윈도우의 평균 전력과 IPC를 계산해 gzip 파일에 기록한다.
   * 최소 기간(gpu_steady_min_period)보다 짧으면 기록하지 않고 에러 메시지만 출력.
   * 기록 후 samples, samples_counter, pwr_counter를 clear()하여 다음 구간 준비.
   *
   * 호출 체인:
   *   detect_print_steady_state() → print_steady_state()
   */
  void print_steady_state(int position, double init_val);

  Processor* proc;
  /* [한국어] McPAT Processor 객체 포인터 — 전력 계산의 핵심 엔진.
   * 설정자: gpgpu_sim_wrapper 생성자에서 new Processor(p)로 생성.
   * 읽는 자: compute()가 proc->compute()를 호출; update_components_power()가
   *          proc->rt_power, proc->cores[0], proc->l2array[0], proc->mc,
   *          proc->nocs[0]에서 전력 값을 읽음.
   * 값 범위: 비-NULL 포인터 (생성자에서 항상 할당).
   * 동기화: 단일 시뮬레이터 스레드에서만 접근. */

  ParseXML* p;
  /* [한국어] McPAT XML 설정 파싱 객체 포인터.
   * 설정자: 생성자에서 new ParseXML() 후 p->parse(xml_filename)으로 초기화.
   *         set_*() 메서드들이 p->sys.core[0]/l2/mc/NoC[0] 필드에 활동 카운터를 주입.
   * 읽는 자: proc->compute()가 이 구조체를 참조해 전력 계산.
   *          update_coefficients()가 p->sys.scaling_coefficients[]를 읽음.
   * 값 범위: 비-NULL 포인터.
   * 동기화: 단일 스레드 접근. */

  // power parameters
  double const_dynamic_power;
  /* [한국어] 상수 동적 전력 항 (클럭 분배 전력 등 회귀 모델로 추정된 고정 동적 전력, Watts).
   * 설정자: 생성자에서 0으로 초기화. (실제로는 현재 코드에서 proc->get_const_dynamic_power()
   *         경로가 주석 처리되어 0으로 유지되며, sample_cmp_pwr[CONSTP]가 대신 사용됨.)
   * 읽는 자: 현재 사용 경로 없음 (레거시 필드).
   * 값 범위: 0.0 이상 Watts.
   * 동기화: 단일 스레드. */

  double avg_threads_per_warp_tot;
  /* [한국어] 커널 내 전체 샘플에 걸친 warp당 평균 활성 스레드 수 누적합.
   * 설정자: set_avg_active_threads()에서 += active_threads로 누적.
   *         reset_counters()에서 0으로 초기화.
   * 읽는 자: print_power_kernel_stats()에서 / kernel_sample_count로 나눠 출력.
   * 값 범위: 0.0 이상 (샘플 수 * 평균 스레드 수).
   * 동기화: 단일 스레드. */

  double proc_power;
  /* [한국어] 현재 샘플의 McPAT 총 동적 전력 (Watts).
   * 설정자: update_components_power()에서 proc->rt_power.readOp.dynamic으로 설정.
   *         CONSTP + STATICP 추가 후 최종값 결정.
   * 읽는 자: print_trace_files()에서 power_trace_file의 첫 열로 기록.
   * 값 범위: 0.0 이상 Watts.
   * 동기화: 단일 스레드. */

  double num_cores;
  /* [한국어] 시뮬레이션 중인 총 SM 수.
   * 설정자: set_num_cores()에서 설정.
   * 읽는 자: calculate_static_power()에서 per_active_core = (num_cores - num_idle_cores) / num_cores.
   * 값 범위: 1 이상 양의 정수 (double로 저장).
   * 동기화: 단일 스레드. */

  double num_idle_cores;
  /* [한국어] 현재 샘플에서 유휴 상태인 SM 수.
   * 설정자: set_idle_core_power()에서 설정.
   * 읽는 자: calculate_static_power()에서 활성 SM 비율 계산.
   * 값 범위: 0 이상, num_cores 이하.
   * 동기화: 단일 스레드. */

  unsigned num_perf_counters;  // # of performance counters
  /* [한국어] ParseXML의 perf_count_t enum에 정의된 성능 카운터 총 수 (NUM_PERFORMANCE_COUNTERS).
   * 설정자: 생성자에서 NUM_PERFORMANCE_COUNTERS로 초기화.
   * 읽는 자: reset_counters(), power_metrics_calculations(), print_power_kernel_stats() 등의 루프.
   * 값 범위: 고정 상수 (빌드 시 결정).
   * 동기화: 읽기 전용. */

  unsigned num_pwr_cmps;       // # of components modelled
  /* [한국어] pwr_cmp_t enum에 정의된 전력 컴포넌트 총 수 (NUM_COMPONENTS_MODELLED = 33).
   * 설정자: 생성자에서 NUM_COMPONENTS_MODELLED로 초기화.
   * 읽는 자: update_components_power(), power_metrics_calculations() 등의 루프.
   * 값 범위: 33 (IBP~STATICP).
   * 동기화: 읽기 전용. */

  int kernel_sample_count;     // # of samples per kernel
  /* [한국어] 현재 커널에서 수집된 전력 샘플 횟수.
   * 설정자: power_metrics_calculations()에서 ++; reset_counters()에서 0으로 초기화.
   * 읽는 자: print_power_kernel_stats()에서 kernel_cmp_pwr[i].avg / kernel_sample_count.
   * 값 범위: 0 이상 정수.
   * 동기화: 단일 스레드. */

  int total_sample_count;      // # of samples per benchmark
  /* [한국어] 벤치마크 전체에 걸친 누적 샘플 횟수 (커널 경계에서 초기화 안 됨).
   * 설정자: power_metrics_calculations()에서 ++.
   * 읽는 자: print_power_kernel_stats()에서 gpu_tot_power.avg / total_sample_count.
   *          detect_print_steady_state()에서 gzprintf의 end 위치로 사용.
   * 값 범위: 0 이상 정수.
   * 동기화: 단일 스레드. */

  std::vector<avg_max_min_counters<double> >
      kernel_cmp_pwr;  // Per-kernel component power avg/max/min values
  /* [한국어] 컴포넌트별(NUM_COMPONENTS_MODELLED개) per-kernel 전력 avg/max/min 벡터.
   * 설정자: power_metrics_calculations()에서 각 샘플마다 avg += sample_cmp_pwr[ind],
   *         max/min 조건 갱신; reset_counters()에서 0으로 초기화.
   * 읽는 자: print_power_kernel_stats()에서 avg/kernel_sample_count, max, min 출력.
   * 값 범위: 각 원소는 Watts 단위 비음수 double.
   * 동기화: 단일 스레드. */

  std::vector<avg_max_min_counters<double> >
      kernel_cmp_perf_counters;  // Per-kernel component avg/max/min performance counters
  /* [한국어] 성능 카운터별(NUM_PERFORMANCE_COUNTERS개) per-kernel avg/max/min 벡터.
   * 설정자: power_metrics_calculations()에서 avg += sample_perf_counters[ind],
   *         max/min 조건 갱신; reset_counters()에서 초기화.
   * 읽는 자: print_power_kernel_stats()에서 각 카운터의 평균/최대/최소 출력.
   * 값 범위: 각 원소는 접근 횟수 등 비음수 카운터 값.
   * 동기화: 단일 스레드. */

  double kernel_tot_power;  // Total per-kernel power
  /* [한국어] 현재 커널의 전체 샘플 전력 누적합 (Watts * sample_count).
   * 설정자: power_metrics_calculations()에서 += sample_power.
   *         reset_counters()에서 0으로 초기화.
   * 읽는 자: kernel_power.avg = kernel_tot_power / kernel_sample_count.
   * 값 범위: 0.0 이상.
   * 동기화: 단일 스레드. */

  avg_max_min_counters<double>
      kernel_power;  // Per-kernel power avg/max/min values
  /* [한국어] 현재 커널의 총 전력 avg/max/min.
   * 설정자: power_metrics_calculations()에서 avg = kernel_tot_power / kernel_sample_count,
   *         max/min 조건 갱신; reset_counters()에서 초기화.
   * 읽는 자: print_power_kernel_stats()에서 kernel_avg/max/min_power 출력.
   * 값 범위: Watts 단위 비음수.
   * 동기화: 단일 스레드. */

  avg_max_min_counters<double>
      gpu_tot_power;  // Global GPU power avg/max/min values (across kernels)
  /* [한국어] 벤치마크 전체에 걸친 누적 GPU 전력 avg/max/min (커널 경계에서 초기화 안 됨).
   * 설정자: power_metrics_calculations()에서 매 샘플마다 avg += sample_power, max/min 갱신.
   * 읽는 자: print_power_kernel_stats()에서 gpu_tot_avg/max/min_power 출력.
   * 값 범위: Watts 단위 비음수.
   * 동기화: 단일 스레드. */

  bool has_written_avg;
  /* [한국어] 정상 상태(steady-state) 평균이 steady_state_tacking_file에 기록되었는지 여부.
   * 설정자: init_mcpat()에서 false로 초기화; print_steady_state()에서 true로 설정.
   * 읽는 자: print_steady_state()에서 has_written_avg==false && position!=0이면 에러 출력.
   * 값 범위: true/false.
   * 동기화: 단일 스레드. */

  std::vector<double> sample_cmp_pwr;  // Current sample component powers
  /* [한국어] 현재 샘플의 컴포넌트별 전력(Watts) 배열 (NUM_COMPONENTS_MODELLED개).
   * 설정자: update_components_power()에서 McPAT rt_power에서 추출해 저장.
   *         reset_counters()에서 0으로 초기화.
   * 읽는 자: power_metrics_calculations()에서 avg/max/min 갱신 소스.
   *          print_trace_files()에서 power_trace_file에 기록.
   *          detect_print_steady_state()에서 pwr_counter에 누적.
   * 값 범위: 비음수 Watts. DVFS 적용 시 스케일링됨.
   * 동기화: 단일 스레드. */

  std::vector<double>
      sample_perf_counters;  // Current sample component perf. counts
  /* [한국어] 현재 샘플의 성능 카운터 값 배열 (NUM_PERFORMANCE_COUNTERS개).
   * 설정자: set_inst_power(), set_regfile_power(), set_*cache_power() 등 setter 군.
   *         reset_counters()에서 0으로 초기화.
   * 읽는 자: update_components_power()에서 FPU/SFU 전력 비율 분배 계산.
   *          power_metrics_calculations()에서 kernel_cmp_perf_counters 갱신.
   *          print_trace_files()에서 metric_trace_file에 기록.
   * 값 범위: 비음수 실수 (접근 횟수, 사이클 수 등).
   * 동기화: 단일 스레드. */

  std::vector<double> initpower_coeff;
  /* [한국어] McPAT 초기(raw) 전력 계수 배열 (NUM_PERFORMANCE_COUNTERS개).
   * 설정자: update_coefficients()에서 McPAT proc->cores[0]->get_coefficient_*() 메서드로 읽어 저장.
   *         마지막으로 executionTime으로 나눠 시간 정규화.
   * 읽는 자: calculate_static_power()에서 int_accesses/fp_accesses 등 유닛별 활성도 판별.
   * 값 범위: 비음수 실수 (에너지/접근 비율, Joule/count 단위).
   * 동기화: 단일 스레드. */

  std::vector<double> effpower_coeff;
  /* [한국어] 스케일링 계수가 적용된 유효(effective) 전력 계수 배열.
   * 설정자: update_coefficients()에서 initpower_coeff[i] * scaling_coefficients[i]로 계산.
   * 읽는 자: 현재 코드에서 직접 사용 경로 없음 (AccelWattch 확장 연구용 보조 필드).
   * 값 범위: 비음수 실수.
   * 동기화: 단일 스레드. */

  // For calculating steady-state average
  unsigned sample_start;
  /* [한국어] 현재 정상 상태 구간이 시작된 샘플 인덱스 (total_sample_count 기준).
   * 설정자: detect_print_steady_state()에서 첫 샘플 진입 시 total_sample_count로 설정.
   *         print_steady_state() 이후 0으로 리셋.
   * 읽는 자: print_steady_state()에서 gzprintf의 start 값으로 출력.
   * 값 범위: 0 이상.
   * 동기화: 단일 스레드. */

  double sample_val;
  /* [한국어] 현재 정상 상태 구간의 McPAT 동적 전력 누적합.
   * 설정자: detect_print_steady_state()에서 += proc->rt_power.readOp.dynamic.
   *         print_steady_state() 이후 0으로 리셋.
   * 읽는 자: print_steady_state()에서 / samples.size()로 평균 계산.
   * 값 범위: 0.0 이상 Watts.
   * 동기화: 단일 스레드. */

  double init_inst_val;
  /* [한국어] 현재 정상 상태 구간 시작 시점의 총 명령어 수 (IPC 계산 기준).
   * 설정자: init_mcpat()에서 init_val로 초기화; print_steady_state()에서 init_val로 리셋.
   * 읽는 자: print_steady_state()에서 temp_ipc = (init_val - init_inst_val) / (samples.size() * freq).
   * 값 범위: 0 이상 명령어 수.
   * 동기화: 단일 스레드. */

  double tot_sfu_accesses;
  /* [한국어] 현재 샘플의 SFU 총 접근 횟수 (set_exec_unit_power()에서 설정).
   * 설정자: set_exec_unit_power()에서 sfu_accesses 인자로 설정.
   * 읽는 자: update_components_power()에서 SFU 유닛별 전력 분배의 분모.
   *          update_coefficients()에서 SFU 계수 분배의 분모.
   * 값 범위: 0 이상. 0이면 모든 SFU 세부 전력을 0으로 처리.
   * 동기화: 단일 스레드. */

  double tot_fpu_accesses;
  /* [한국어] 현재 샘플의 FPU(SP FP32 + DP FP64) 총 접근 횟수.
   * 설정자: set_exec_unit_power()에서 fpu_accesses 인자로 설정.
   * 읽는 자: update_components_power()에서 FPUP/DPUP 전력 분배의 분모.
   *          update_coefficients()에서 FP_ACC/DP_ACC 계수 분배의 분모.
   * 값 범위: 0 이상. 0이면 FPUP, DPUP를 0으로 처리.
   * 동기화: 단일 스레드. */

  double modeled_chip_voltage;
  /* [한국어] DVFS 모드에서 현재 동작 전압 (V).
   * 설정자: set_model_voltage()에서 설정.
   * 읽는 자: update_components_power()에서 voltage_ratio = modeled_chip_voltage / p->sys.modeled_chip_voltage_ref.
   * 값 범위: 양의 실수 (GPU 동작 전압, 보통 0.5~1.2 V 범위).
   * 동기화: 단일 스레드. */

  unsigned avg_threads_per_warp;
  /* [한국어] 현재 샘플에서 warp당 평균 활성 스레드 수 (올림 적용).
   * 설정자: set_avg_active_threads()에서 (unsigned)ceil(active_threads)로 설정.
   * 읽는 자: calculate_static_power()의 선형 모델 수식에서 사용:
   *          total_static = base + (avg_threads_per_warp - 1) * lane.
   *          avg_threads_per_warp == 0이면 LIGHT_SM / memory-only 경로로 분기.
   * 값 범위: 0~32 (warp 크기 = 32 스레드).
   * 동기화: 단일 스레드. */

  std::vector<double> samples;
  /* [한국어] 현재 정상 상태 구간의 McPAT 동적 전력 샘플 목록.
   * 설정자: detect_print_steady_state()에서 편차 이내 샘플을 push_back.
   *         print_steady_state()에서 clear().
   * 읽는 자: print_steady_state()에서 samples.size()로 평균 계산.
   *          detect_print_steady_state()에서 temp_avg = sample_val / samples.size().
   * 값 범위: 비음수 Watts의 double 목록.
   * 동기화: 단일 스레드. */

  std::vector<double> samples_counter;
  /* [한국어] 현재 정상 상태 구간의 성능 카운터 누적값 벡터 (num_perf_counters개).
   * 설정자: detect_print_steady_state()에서 첫 샘플에 push_back, 이후 += sample_perf_counters.
   *         print_steady_state()에서 clear().
   * 읽는 자: print_steady_state()에서 samples_counter.at(i) / samples.size()로 평균 출력.
   * 값 범위: 비음수 실수.
   * 동기화: 단일 스레드. */

  std::vector<double> pwr_counter;
  /* [한국어] 현재 정상 상태 구간의 컴포넌트별 전력 누적 벡터 (num_pwr_cmps개).
   * 설정자: detect_print_steady_state()에서 push_back + 누적.
   *         print_steady_state()에서 clear().
   * 읽는 자: print_steady_state()에서 (현재는 출력에 직접 사용하지 않으나 추후 확장용 보존).
   * 값 범위: 비음수 Watts.
   * 동기화: 단일 스레드. */

  char* xml_filename;
  /* [한국어] McPAT XML 설정 파일 경로 문자열.
   * 설정자: 생성자 및 init_mcpat()에서 인자로 전달받아 저장.
   * 읽는 자: p->parse(xml_filename) 호출 시 사용.
   * 값 범위: 유효한 파일 경로 C 문자열.
   * 동기화: 단일 스레드. */

  char* g_power_filename;
  /* [한국어] 전력 통계 텍스트 출력 파일 경로 (gpgpusim.power).
   * 설정자: init_mcpat()에서 최초 1회 설정.
   * 읽는 자: powerfile.open(g_power_filename)으로 파일 오픈.
   * 값 범위: 유효한 파일 경로 또는 NULL.
   * 동기화: 단일 스레드. */

  char* g_power_trace_filename;
  /* [한국어] 사이클별 전력 트레이스 gzip 파일 경로.
   * 설정자: init_mcpat()에서 설정.
   * 읽는 자: open_files()에서 gzopen().
   * 값 범위: 유효한 경로 또는 NULL.
   * 동기화: 단일 스레드. */

  char* g_metric_trace_filename;
  /* [한국어] 사이클별 성능 카운터 트레이스 gzip 파일 경로.
   * 설정자: init_mcpat()에서 설정.
   * 읽는 자: open_files()에서 gzopen().
   * 값 범위: 유효한 경로 또는 NULL.
   * 동기화: 단일 스레드. */

  char* g_steady_state_tracking_filename;
  /* [한국어] 정상 상태 전력 추적 gzip 파일 경로.
   * 설정자: init_mcpat()에서 설정.
   * 읽는 자: detect_print_steady_state()에서 gzopen().
   * 값 범위: 유효한 경로 또는 NULL.
   * 동기화: 단일 스레드. */

  bool g_power_simulation_enabled;
  /* [한국어] 전력 시뮬레이션 활성화 플래그.
   * 설정자: 생성자 및 init_mcpat()에서 인자로 전달받아 설정.
   * 읽는 자: print_power_kernel_stats(), open_files() 등의 조건 분기.
   * 값 범위: true/false.
   * 동기화: 읽기 전용 (초기화 후 변경 없음). */

  int g_power_simulation_mode;
  /* [한국어] 전력 시뮬레이션 모드 번호 (0=HW측정 기반, 1=시뮬레이션 기반 등).
   * 설정자: 생성자 및 init_mcpat()에서 설정.
   * 읽는 자: gpgpu_sim_wrapper 외부 (gpu-sim.cc)에서 모드 분기에 사용.
   * 값 범위: 정의된 모드 번호 (AccelWattch 논문 참조).
   * 동기화: 읽기 전용. */

  bool g_dvfs_enabled;
  /* [한국어] DVFS(Dynamic Voltage Frequency Scaling) 활성화 플래그.
   * 설정자: 생성자 및 init_mcpat()에서 설정.
   * 읽는 자: update_components_power()에서 voltage_ratio 스케일링 분기.
   * 값 범위: true/false.
   * 동기화: 읽기 전용. */

  bool g_steady_power_levels_enabled;
  /* [한국어] 정상 상태 전력 레벨 감지 기능 활성화 플래그.
   * 설정자: init_mcpat()에서 steady_state_enabled 인자로 설정.
   * 읽는 자: detect_print_steady_state()에서 기능 ON/OFF 조건.
   * 값 범위: true/false.
   * 동기화: 읽기 전용. */

  bool g_power_trace_enabled;
  /* [한국어] 사이클별 전력/성능 트레이스 파일 기록 활성화 플래그.
   * 설정자: init_mcpat()에서 trace_enabled 인자로 설정.
   * 읽는 자: open_files(), close_files(), print_trace_files()의 분기 조건.
   * 값 범위: true/false.
   * 동기화: 읽기 전용. */

  bool g_power_per_cycle_dump;
  /* [한국어] 사이클별 에너지 덤프 활성화 플래그 (dump() 함수 제어).
   * 설정자: init_mcpat()에서 power_per_cycle_dump 인자로 설정.
   * 읽는 자: dump()에서 proc->displayEnergy() 호출 조건.
   * 값 범위: true/false.
   * 동기화: 읽기 전용. */

  double gpu_steady_power_deviation;
  /* [한국어] 정상 상태 판별 전력 편차 임계값 (Watts).
   * 설정자: init_mcpat()에서 steady_power_deviation 인자로 설정.
   * 읽는 자: detect_print_steady_state()에서 abs(현재전력 - 평균) < 이 값이면 정상 상태로 판단.
   * 값 범위: 양의 실수 (예: 0.1W).
   * 동기화: 읽기 전용. */

  double gpu_steady_min_period;
  /* [한국어] 정상 상태로 기록되기 위한 최소 샘플 수 (슬라이딩 윈도우 최소 길이).
   * 설정자: init_mcpat()에서 steady_min_period 인자로 설정.
   * 읽는 자: print_steady_state()에서 samples.size() > gpu_steady_min_period 조건.
   * 값 범위: 양의 실수 (샘플 수로 비교).
   * 동기화: 읽기 전용. */

  int g_power_trace_zlevel;
  /* [한국어] gzip 압축 수준 (0=무압축, 9=최고압축).
   * 설정자: init_mcpat()에서 zlevel 인자로 설정.
   * 읽는 자: init_mcpat()에서 gzsetparams()의 level 인자로 사용.
   * 값 범위: 0~9.
   * 동기화: 읽기 전용. */

  double gpu_stat_sample_frequency;
  /* [한국어] 전력 샘플링 주파수 (레거시 필드, 현재 사용 안 됨).
   * 설정자: 없음 (초기화 없이 선언만).
   * 읽는 자: 없음.
   * 값 범위: 미사용.
   * 동기화: 해당 없음. */

  int gpu_stat_sample_freq;
  /* [한국어] 전력 샘플링 주기 (GPU 사이클 수). McPAT의 total_cycles 설정에 사용.
   * 설정자: init_mcpat()에서 stat_sample_freq 인자로 설정. 생성자에서 0으로 초기화.
   * 읽는 자: init_mcpat()에서 p->sys.total_cycles = gpu_stat_sample_freq.
   *          print_steady_state()에서 IPC 계산 분모.
   * 값 범위: 양의 정수 (gpgpusim.config의 -power_simulation_period 옵션).
   * 동기화: 읽기 전용 (초기화 후 변경 없음). */

  std::ofstream powerfile;
  /* [한국어] 전력 통계 텍스트 출력 스트림.
   * 설정자: init_mcpat()에서 powerfile.open(g_power_filename)으로 한 번 오픈.
   * 읽는 자: print_power_kernel_stats()에서 << 연산자로 통계 출력.
   * 값 범위: 유효한 ofstream (오픈 실패 시 abort).
   * 동기화: 단일 스레드. */

  gzFile power_trace_file;
  /* [한국어] 사이클별 전력 트레이스 gzip 파일 핸들.
   * 설정자: init_mcpat()의 헤더 기록 시 gzopen/gzclose; open_files()에서 append 모드 재오픈.
   * 읽는 자: print_trace_files()에서 gzprintf.
   * 값 범위: 유효한 gzFile 포인터 또는 NULL.
   * 동기화: 단일 스레드. */

  gzFile metric_trace_file;
  /* [한국어] 사이클별 성능 카운터 트레이스 gzip 파일 핸들.
   * 설정자: init_mcpat()의 헤더 기록 시 gzopen/gzclose; open_files()에서 재오픈.
   * 읽는 자: print_trace_files()에서 gzprintf.
   * 값 범위: 유효한 gzFile 포인터 또는 NULL.
   * 동기화: 단일 스레드. */

  gzFile steady_state_tacking_file;
  /* [한국어] 정상 상태 전력 구간 추적 gzip 파일 핸들.
   * 설정자: init_mcpat()의 헤더 기록 시; detect_print_steady_state()에서 매번 open/close.
   * 읽는 자: print_steady_state()에서 gzprintf.
   * 값 범위: 유효한 gzFile 포인터 또는 NULL.
   * 동기화: 단일 스레드. */
};

#endif /* GPGPU_SIM_WRAPPER_H_ */
