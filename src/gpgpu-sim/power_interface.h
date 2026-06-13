// Copyright (c) 2009-2021, Tor M. Aamodt, Ahmed El-Shafiey, Tayler
// Hetherington, Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers The University of British Columbia, Northwestern
// University, Purdue University All rights reserved.
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
 * [한국어 설명] AccelWattch 전력 모델 인터페이스 헤더 (power_interface.h)
 *
 * === 파일의 역할 ===
 * AccelWattch(GPGPU-Sim의 전력 모델)와 GPGPU-Sim 타이밍 시뮬레이터 간의 인터페이스를
 * 정의하는 헤더 파일이다. GPGPU-Sim이 수집한 마이크로아키텍처 카운터(SM ALU 사용량,
 * 캐시 접근 횟수 등)를 McPAT 기반 전력 모델에 전달하여 사이클별 전력 소비를 추정한다.
 * 세 가지 시뮬레이션 모드(Accel-Sim, HW 측정, 하이브리드)를 지원하며, 각 모드에서
 * 적절한 카운터 소스를 선택해 McPAT에 입력한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 계층:
 *   gpu-sim.cc (cycle 루프) → [power_interface.cc] → gpgpu_sim_wrapper (McPAT 래퍼)
 *   power_stat_t (카운터 수집) → [이 헤더의 함수들] → McPAT 전력 계산
 * 실행 컨텍스트: 호스트 유저스페이스, gpgpu_sim::cycle() 호출 시 stat_sample_freq
 * 사이클마다 주기적으로 호출된다.
 * 위치: 타이밍 모델(gpgpu-sim/)과 AccelWattch(accelwattch/) 사이의 접착 계층.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - gpu-sim.h: gpgpu_sim_config (설정), gpgpu_sim (시뮬레이터 상태)
 *   - power_stat.h: power_stat_t (마이크로아키텍처 카운터 집합)
 *   - shader.h: shader_core_config (SM 구성 정보)
 *   - gpgpu_sim_wrapper.h: gpgpu_sim_wrapper (McPAT 래퍼 — 실제 전력 계산 수행)
 * 이 파일에 의존하는 모듈:
 *   - gpu-sim.cc: 매 stat_sample_freq 사이클마다 mcpat_cycle() 또는
 *                 calculate_hw_mcpat() 호출
 * 데이터 흐름: shader_core_stats → power_stat_t → power_interface 함수 →
 *             gpgpu_sim_wrapper → McPAT → 전력(와트) 출력
 *
 * === 주요 함수/구조체 요약 ===
 * init_mcpat()            - 시뮬레이터 시작 시 McPAT 초기화 (설정 파일 경로 전달)
 * mcpat_cycle()           - 매 샘플링 주기에 Accel-Sim 모드로 전력 계산 수행
 * calculate_hw_mcpat()    - HW 측정값 기반 또는 하이브리드 모드 전력 계산
 * parse_hw_file()         - HW 전력 CSV 파일에서 벤치마크/커널 항목 파싱
 * mcpat_reset_perf_count()- 다음 샘플링 주기를 위해 McPAT 내부 카운터 초기화
 */

#ifndef POWER_INTERFACE_H_
#define POWER_INTERFACE_H_

#include "gpu-sim.h"      /* [한국어] gpgpu_sim_config, gpgpu_sim — 시뮬레이터 전역 설정 및 상태 */
#include "power_stat.h"   /* [한국어] power_stat_t — 마이크로아키텍처 전력 카운터 집합 */
#include "shader.h"       /* [한국어] shader_core_config — SM(Streaming Multiprocessor) 구성 파라미터 */

#include "gpgpu_sim_wrapper.h" /* [한국어] gpgpu_sim_wrapper — McPAT 전력 계산 엔진 래퍼 */

/*
 * [한국어]
 * init_mcpat - McPAT 전력 모델 초기화
 *
 * @config: gpgpu_sim_config — 전력 모델 설정 (config 파일에서 읽힌 g_power_config_name 등)
 * @wrapper: gpgpu_sim_wrapper — McPAT 계산 엔진을 감싸는 래퍼 객체
 * @stat_sample_freq: 전력 샘플링 주기 (사이클 단위) — 이 주기마다 전력 계산 수행
 * @tot_inst: 이전 커널까지의 누적 명령어 수
 * @inst: 현재 샘플링 창의 명령어 수
 * @return: 없음 (void)
 *
 * 시뮬레이터 시작 직후(첫 커널 실행 전) 한 번 호출되어 McPAT 내부 상태를 초기화한다.
 * gpgpusim.config의 -power_simulation_enabled 옵션이 활성화된 경우에만 의미가 있다.
 * wrapper->init_mcpat()을 통해 전력 설정 파일, 추적 파일 경로, 코어 주파수, SM 개수 등을
 * McPAT 래퍼에 전달한다.
 *
 * 호출 체인:
 *   gpgpu_sim::init() 또는 최초 kernel_launch → [init_mcpat] → wrapper->init_mcpat()
 */
void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst);

/*
 * [한국어]
 * mcpat_cycle - Accel-Sim 모드 주기적 전력 계산 (stat_sample_freq 사이클마다 호출)
 *
 * @config: gpgpu_sim_config — 시뮬레이터 전역 설정
 * @shdr_config: shader_core_config — SM 개수, 클럭 게이팅 레인 수 등 코어 구성
 * @wrapper: gpgpu_sim_wrapper — McPAT 래퍼 (set_*_power() 및 compute() 보유)
 * @power_stats: power_stat_t — 지난 샘플링 창의 마이크로아키텍처 카운터
 * @stat_sample_freq: 샘플링 주기 (사이클 수) — 평균화에 사용
 * @tot_cycle: 이전 커널까지 누적 사이클 수
 * @cycle: 현재 샘플링 창의 사이클 수
 * @tot_inst: 이전 커널까지 누적 명령어 수
 * @inst: 현재 샘플링 창의 명령어 수
 * @dvfs_enabled: DVFS(Dynamic Voltage/Frequency Scaling) 활성화 여부
 * @return: 없음 (void)
 *
 * Accel-Sim 시뮬레이션 모드(mode=0)에서 사용. 매 stat_sample_freq 사이클마다
 * power_stat_t에서 수집된 카운터(L1/L2 캐시 접근, ALU 사용량, DRAM 접근 등)를
 * wrapper의 set_*_power() 함수에 전달하고, wrapper->compute()를 호출해 McPAT이
 * 동적/정적 전력을 계산하게 한다.
 * 첫 번째 사이클에는 아직 카운터가 없으므로 mcpat_init 플래그로 스킵한다.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → [mcpat_cycle] → wrapper->set_*_power() → wrapper->compute()
 *                                                               → wrapper->dump()
 */
void mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, bool dvfs_enabled);

/*
 * [한국어]
 * calculate_hw_mcpat - HW 측정값 또는 하이브리드 모드 전력 계산
 *
 * @config: gpgpu_sim_config — 시뮬레이터 전역 설정
 * @shdr_config: shader_core_config — SM 구성 정보
 * @wrapper: gpgpu_sim_wrapper — McPAT 래퍼
 * @power_stats: power_stat_t — 시뮬레이션 카운터 (하이브리드 모드에서 일부 사용)
 * @stat_sample_freq: 샘플링 주기
 * @tot_cycle / @cycle: 누적/현재 사이클 수
 * @tot_inst / @inst: 누적/현재 명령어 수
 * @power_simulation_mode: 1=HW 전용, 2=하이브리드(HW+Accel-Sim)
 * @dvfs_enabled: DVFS 활성화 여부
 * @hwpowerfile: HW 전력/성능 측정값이 담긴 CSV 파일 경로
 * @benchname: CSV 파일에서 검색할 벤치마크 이름
 * @executed_kernelname: CSV 파일에서 검색할 커널 이름 (없으면 벤치마크 첫 항목 사용)
 * @accelwattch_hybrid_configuration: 하이브리드 모드에서 각 카운터를 HW/Accel-Sim 중
 *   어느 소스에서 가져올지 결정하는 불리언 배열 (인덱스는 HW_* 열거형)
 * @aggregate_power_stats: 여러 커널에 걸쳐 카운터를 누적할지 여부
 * @return: 없음 (void)
 *
 * 커널 실행 완료 후 한 번 호출되어 HW 측정값 CSV에서 성능 카운터를 읽고 McPAT에
 * 전달한다. power_simulation_mode==2이고 accelwattch_hybrid_configuration[카운터]==true인
 * 카운터는 CSV 대신 Accel-Sim 시뮬레이션 결과를 사용한다.
 * 계산 완료 후 power_stats의 커널 기준점(l1r_hits_kernel 등)을 현재 값으로 갱신하여
 * 다음 커널의 델타 계산을 준비하고, power_stats->clear()로 카운터를 리셋한다.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() (커널 완료 시) → [calculate_hw_mcpat] → parse_hw_file()
 *                                     → wrapper->set_*_power() → wrapper->compute()
 */
void calculate_hw_mcpat(
    const gpgpu_sim_config &config, const shader_core_config *shdr_config,
    class gpgpu_sim_wrapper *wrapper, class power_stat_t *power_stats,
    unsigned stat_sample_freq, unsigned tot_cycle, unsigned cycle,
    unsigned tot_inst, unsigned inst, int power_simulation_mode,
    bool dvfs_enabled, char *hwpowerfile, char *benchname,
    std::string executed_kernelname,
    const bool *accelwattch_hybrid_configuration, bool aggregate_power_stats);

/*
 * [한국어]
 * parse_hw_file - HW 전력/성능 측정 CSV 파일에서 벤치마크/커널 항목 검색
 *
 * @hwpowerfile: CSV 파일 경로 (쉼표 구분, 각 행이 하나의 실행 결과)
 * @find_target_kernel: true면 benchname+kernelname 정확 매칭, false면 benchname만 매칭
 * @hw_data: [출력] 매칭된 행의 CSV 필드를 string 벡터로 반환 (HW_* 열거형으로 인덱싱)
 * @benchname: 검색할 벤치마크 이름 (CSV의 HW_BENCH_NAME 열과 비교)
 * @executed_kernelname: 검색할 커널 이름 (CSV의 HW_KERNEL_NAME 열과 비교)
 * @return: 매칭 항목을 찾으면 true, 파일 끝까지 못 찾으면 false
 *
 * calculate_hw_mcpat()에서 두 단계로 호출된다:
 *   1단계: find_target_kernel=true — 벤치마크+커널 이름 정확 매칭
 *   2단계(실패 시): find_target_kernel=false — 벤치마크 이름만으로 첫 항목 반환
 * 파일 오픈/클로즈를 함수 내에서 처리하며, 매칭 즉시 파일을 닫고 반환한다.
 *
 * 호출 체인:
 *   calculate_hw_mcpat → [parse_hw_file] (최대 2회)
 */
bool parse_hw_file(char *hwpowerfile, bool find_target_kernel,
                   vector<string> &hw_data, char *benchname,
                   std::string executed_kernelname);

/*
 * [한국어]
 * mcpat_reset_perf_count - McPAT 래퍼 내부의 성능 카운터 초기화
 *
 * @wrapper: gpgpu_sim_wrapper — 카운터를 초기화할 McPAT 래퍼 객체
 * @return: 없음 (void)
 *
 * wrapper->reset_counters()를 호출하여 McPAT 내부 카운터를 0으로 리셋한다.
 * 커널 전환 시점 또는 샘플링 창 경계에서 호출되어 이전 커널의 카운터가
 * 다음 커널 계산에 섞이지 않도록 보장한다.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() (커널 완료 후) → [mcpat_reset_perf_count] → wrapper->reset_counters()
 */
void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper);

#endif /* POWER_INTERFACE_H_ */
