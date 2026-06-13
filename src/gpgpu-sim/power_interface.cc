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
 * [한국어 설명] AccelWattch 전력 모델 인터페이스 구현 (power_interface.cc)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim 타이밍 시뮬레이터가 수집한 마이크로아키텍처 카운터를 AccelWattch
 * (McPAT 기반 GPU 전력 모델)에 전달하는 접착 계층(glue layer)을 구현한다.
 * 세 가지 전력 시뮬레이션 모드를 지원한다:
 *   모드 0 (Accel-Sim): GPGPU-Sim 카운터만 사용 — mcpat_cycle()로 주기적 계산
 *   모드 1 (HW): 실측 HW 성능 카운터 CSV 사용 — calculate_hw_mcpat()
 *   모드 2 (하이브리드): 카운터별로 HW 또는 Accel-Sim 소스를 선택적 혼합
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *     → (매 stat_sample_freq 사이클) mcpat_cycle() 또는 calculate_hw_mcpat()
 *       → wrapper->set_*_power() — 카운터 값 McPAT 입력 레지스터에 적재
 *       → wrapper->compute()     — McPAT 동적/정적 전력 계산 실행
 *       → wrapper->dump()        — 결과를 전력 추적 파일에 출력
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드. 사이클 루프 안에서 동기적으로 호출.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - power_interface.h: 이 파일의 함수 선언
 *   - power_stat_t (power_stat.h): SM별, GPU 전체 마이크로아키텍처 카운터 집합
 *   - gpgpu_sim_wrapper (accelwattch/): McPAT 전력 계산 엔진 래퍼
 *   - gpgpu_sim_config (gpu-sim.h): 전력 모델 활성화 여부, 설정 파일 경로 등
 *   - shader_core_config (shader.h): SM 개수, SP/SFU 유닛 수 등 코어 구성
 * 데이터 흐름: shader_core_stats → power_stat_t → 이 파일 → gpgpu_sim_wrapper
 *             → McPAT → 전력(W) 출력 파일
 *
 * === 주요 함수/구조체 요약 ===
 * init_mcpat()            - 시뮬레이터 시작 시 McPAT 초기화
 * mcpat_cycle()           - Accel-Sim 모드 주기적 전력 계산 (stat_sample_freq마다)
 * calculate_hw_mcpat()    - HW/하이브리드 모드 커널별 전력 계산
 * parse_hw_file()         - HW 전력 측정 CSV 파일 파싱
 * mcpat_reset_perf_count()- McPAT 내부 카운터 초기화
 */

#include "power_interface.h" /* [한국어] 이 파일에서 구현하는 함수들의 선언 */

/*
 * [한국어]
 * init_mcpat - McPAT 전력 모델 초기화
 *
 * @config: gpgpu_sim_config — 전력 관련 설정 파라미터 (config 파일에서 파싱됨)
 * @wrapper: gpgpu_sim_wrapper — McPAT 래퍼 객체 (Accel-Sim이 생성해 전달)
 * @stat_sample_freq: 전력 샘플링 주기 (사이클) — 이 주기마다 mcpat_cycle 호출됨
 * @tot_inst: 이전 커널까지의 누적 명령어 수
 * @inst: 현재 샘플링 창의 명령어 수
 * @return: 없음 (void)
 *
 * McPAT 래퍼가 전력 계산을 수행하기 전에 설정 파일 경로, 출력 파일 경로,
 * 코어 주파수, SM 개수 등을 래퍼에 전달하여 내부 McPAT 모델을 초기화한다.
 * g_power_simulation_enabled가 false면 이 함수 호출 자체가 의미 없지만,
 * gpu-sim.cc에서 조건을 확인하고 호출하므로 이 함수 내에서는 별도 체크 없음.
 *
 * 호출 체인:
 *   gpgpu_sim 생성자 또는 첫 kernel_launch → [init_mcpat] → wrapper->init_mcpat()
 */
void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst) {
  wrapper->init_mcpat(
      config.g_power_config_name,             /* [한국어] McPAT XML 설정 파일 경로 (-power_config_name) */
      config.g_power_filename,                /* [한국어] 전력 결과 출력 파일 경로 (-power_output_file) */
      config.g_power_trace_filename,          /* [한국어] 사이클별 전력 추적 파일 경로 */
      config.g_metric_trace_filename,         /* [한국어] 성능 메트릭 추적 파일 경로 */
      config.g_steady_state_tracking_filename,/* [한국어] 정상 상태(steady-state) 추적 파일 경로 */
      config.g_power_simulation_enabled,      /* [한국어] 전력 시뮬레이션 활성화 플래그 */
      config.g_power_trace_enabled,           /* [한국어] 전력 추적 파일 생성 여부 */
      config.g_steady_power_levels_enabled,   /* [한국어] 정상 상태 전력 레벨 감지 활성화 여부 */
      config.g_power_per_cycle_dump,          /* [한국어] 사이클마다 전력 덤프 여부 */
      config.gpu_steady_power_deviation,      /* [한국어] 정상 상태 판별 전력 편차 임계값 */
      config.gpu_steady_min_period,           /* [한국어] 정상 상태 최소 지속 기간 */
      config.g_power_trace_zlevel,            /* [한국어] 전력 추적 파일 zlib 압축 레벨 */
      tot_inst + inst,                        /* [한국어] 전체 누적 명령어 수 = 이전 커널 + 현재 창 */
      stat_sample_freq,                       /* [한국어] 샘플링 주기 (사이클) */
      config.g_power_simulation_mode,         /* [한국어] 전력 시뮬레이션 모드 (0=AccelSim, 1=HW, 2=Hybrid) */
      config.g_dvfs_enabled,                  /* [한국어] DVFS 활성화 여부 */
      config.get_core_freq() / 1000000,       /* [한국어] 코어 클럭 주파수 (Hz→MHz 변환) */
      config.num_shader());                   /* [한국어] 전체 SM(Shader Core) 개수 */
}

/*
 * [한국어]
 * mcpat_cycle - Accel-Sim 모드 주기적 전력 계산
 *
 * @config: gpgpu_sim_config — 시뮬레이터 전역 설정 (현재 이 함수에서 직접 사용 안 함)
 * @shdr_config: shader_core_config — SM 개수, 클럭 게이팅 레인, SP/SFU 유닛 수
 * @wrapper: gpgpu_sim_wrapper — McPAT 래퍼 (set_*_power(), compute(), dump() 포함)
 * @power_stats: power_stat_t — 마이크로아키텍처 카운터 집합 (델타/누적 두 가지 모드)
 * @stat_sample_freq: 샘플링 주기 (사이클) — 평균화 분모로 사용
 * @tot_cycle + @cycle: 전체 누적 사이클 수 — stat_sample_freq 나눔으로 샘플링 시점 결정
 * @tot_inst + @inst: 전체 누적 명령어 수 — 정상 상태 감지에 사용
 * @dvfs_enabled: DVFS 활성화 여부 — true이면 모델 전압을 1로 설정 (임시 처리)
 * @return: 없음 (void)
 *
 * 동작 과정:
 *   1. mcpat_init 플래그로 첫 번째 사이클 스킵 (첫 사이클에는 카운터가 없음)
 *   2. (tot_cycle+cycle) % stat_sample_freq == 0 조건을 만족하는 사이클에만 실행
 *   3. power_stat_t에서 각 카운터의 델타값(aggregate_stat=false)을 읽어
 *      wrapper의 set_*_power() 함수에 전달
 *   4. wrapper->compute()로 McPAT 전력 계산 실행
 *   5. wrapper->update_components_power(), print_trace_files(), dump()로
 *      컴포넌트별 전력 업데이트 및 파일 출력
 *   6. power_stats->save_stats()로 현재 카운터를 PREV로 저장해 다음 창의 델타 기준 갱신
 * 실행 컨텍스트: 호스트 유저스페이스, 사이클 루프 내 단일 스레드. 재진입 없음.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → [mcpat_cycle] → wrapper->compute() → McPAT 전력 계산
 */
void mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, bool dvfs_enabled) {
  static bool mcpat_init = true; /* [한국어] 함수 최초 호출 여부를 추적하는 static 플래그 */

  if (mcpat_init) {  // If first cycle, don't have any power numbers yet
    /* [한국어] 첫 번째 호출: 아직 카운터가 수집되지 않았으므로 전력 계산 불가 — 플래그 해제 후 리턴 */
    mcpat_init = false; /* [한국어] 이후 호출부터는 정상 경로로 진입하도록 플래그 클리어 */
    return;
  }

  if ((tot_cycle + cycle) % stat_sample_freq == 0) {
    /* [한국어] 샘플링 주기 도달 — 전체 누적 사이클이 stat_sample_freq의 배수일 때만 계산 수행 */
    if (dvfs_enabled) {
      wrapper->set_model_voltage(1);  // performance model needs to support
                                      // this.
      /* [한국어] DVFS 모드: 현재는 전압 1로 고정 (McPAT 내 전압 스케일링 지원 시 개선 예정) */
    }

    /* [한국어] === 명령어 통계 설정 ===
     * aggregate_stat=false → CURRENT-PREV 델타값 사용 (이번 샘플링 창의 증분)
     * gpgpu_clock_gated_lanes: 클럭 게이팅된 레인 수 (비활성 레인 전력 절감 모델링)
     * stat_sample_freq: 샘플 창 크기 (사이클)
     * 총 명령어, 정수 명령어, FP 명령어, L1D 읽기/쓰기 접근, 커밋된 명령어 전달 */
    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
        stat_sample_freq, power_stats->get_total_inst(0),
        power_stats->get_total_int_inst(0), power_stats->get_total_fp_inst(0),
        power_stats->get_l1d_read_accesses(0),
        power_stats->get_l1d_write_accesses(0),
        power_stats->get_committed_inst(0));

    // Single RF for both int and fp ops
    /* [한국어] === 레지스터 파일 접근 통계 ===
     * GPU는 INT/FP 공용 단일 레지스터 파일 사용.
     * 읽기 접근 수, 쓰기 접근 수, 레지스터 파일 외 오퍼랜드(즉시값 등) 수 전달 */
    wrapper->set_regfile_power(power_stats->get_regfile_reads(0),
                               power_stats->get_regfile_writes(0),
                               power_stats->get_non_regfile_operands(0));

    // Instruction cache stats
    /* [한국어] === 명령어 캐시(L1I) 통계 ===
     * 히트/미스 수를 McPAT의 I-캐시 전력 모델에 전달 */
    wrapper->set_icache_power(power_stats->get_inst_c_hits(0),
                              power_stats->get_inst_c_misses(0));

    // Constant Cache, shared memory, texture cache
    /* [한국어] === 상수 캐시 통계 ===
     * 현재 모든 접근을 HIT으로 가정하여 미스 수=0 전달 (보수적 전력 추정) */
    wrapper->set_ccache_power(
        power_stats->get_const_accessess(0),
        0);  // assuming all HITS in constant cache for now
    /* [한국어] === 텍스처 캐시 통계 ===
     * 텍스처 캐시 히트/미스 수 전달 */
    wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
                              power_stats->get_texture_c_misses());
    /* [한국어] === 공유 메모리(Shared Memory) 통계 ===
     * 워프 내 스레드들이 공유하는 온-칩 SRAM 접근 수 전달 */
    wrapper->set_shrd_mem_power(power_stats->get_shmem_access(0));

    /* [한국어] === L1 데이터 캐시 통계 ===
     * 읽기 히트/미스, 쓰기 히트/미스 별도 전달 — McPAT은 R/W 분리 모델 사용 */
    wrapper->set_l1cache_power(power_stats->get_l1d_read_hits(0),
                               power_stats->get_l1d_read_misses(0),
                               power_stats->get_l1d_write_hits(0),
                               power_stats->get_l1d_write_misses(0));

    /* [한국어] === L2 캐시 통계 ===
     * L2 읽기/쓰기 히트/미스 수 전달 */
    wrapper->set_l2cache_power(
        power_stats->get_l2_read_hits(0), power_stats->get_l2_read_misses(0),
        power_stats->get_l2_write_hits(0), power_stats->get_l2_write_misses(0));

    /* [한국어] === 유휴 SM(코어) 수 계산 ===
     * m_active_sms: 이번 창에서 활성화된 SM 수의 합산 → 평균 구하기 위해 샘플 수로 나눔 */
    float active_sms = (*power_stats->m_active_sms) / stat_sample_freq; /* [한국어] 샘플링 창 평균 활성 SM 수 */
    float num_cores = shdr_config->num_shader();                         /* [한국어] 전체 SM 개수 */
    float num_idle_core = num_cores - active_sms;                        /* [한국어] 유휴(비활성) SM 수 = 전체 - 활성 */
    wrapper->set_num_cores(num_cores);                                   /* [한국어] McPAT에 전체 SM 수 전달 */
    wrapper->set_idle_core_power(num_idle_core);                         /* [한국어] 유휴 SM 전력 모델에 유휴 코어 수 전달 */

    // pipeline power - pipeline_duty_cycle *= percent_active_sms;
    /* [한국어] === 파이프라인 듀티 사이클 계산 ===
     * m_average_pipeline_duty_cycle: 이번 창에서 파이프라인이 활성이었던 비율의 합산
     * 샘플 수로 나눠 평균 구한 뒤 0.8로 상한 클리핑 (GPU 파이프라인의 현실적 최대 활용률 반영) */
    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq) /* [한국어] 0.8 미만이면 그대로 사용 */
            : 0.8;                                                                /* [한국어] 0.8 이상이면 0.8로 클리핑 */
    wrapper->set_duty_cycle_power(pipeline_duty_cycle); /* [한국어] McPAT 파이프라인 활용률 파라미터 전달 */

    // Memory Controller
    /* [한국어] === DRAM 메모리 컨트롤러 통계 ===
     * DRAM 읽기/쓰기/프리차지 명령 수를 메모리 컨트롤러 전력 모델에 전달 */
    wrapper->set_mem_ctrl_power(power_stats->get_dram_rd(0),
                                power_stats->get_dram_wr(0),
                                power_stats->get_dram_pre(0));

    // Execution pipeline accesses
    // FPU (SP) accesses, Integer ALU (not present in Tesla), Sfu accesses

    /* [한국어] === 정수 ALU 파이프라인 통계 ===
     * IALU 접근, 24비트/32비트 정수 곱셈, 범용 정수 곱셈, 정수 나눗셈 수 전달 */
    wrapper->set_int_accesses(power_stats->get_ialu_accessess(0),
                              power_stats->get_intmul24_accessess(0),
                              power_stats->get_intmul32_accessess(0),
                              power_stats->get_intmul_accessess(0),
                              power_stats->get_intdiv_accessess(0));

    /* [한국어] === 배정밀도(DP) FPU 파이프라인 통계 ===
     * DP 연산, DP 곱셈, DP 나눗셈 수 전달 */
    wrapper->set_dp_accesses(power_stats->get_dp_accessess(0),
                             power_stats->get_dpmul_accessess(0),
                             power_stats->get_dpdiv_accessess(0));

    /* [한국어] === 단정밀도(FP/SP) FPU 파이프라인 통계 ===
     * SP 연산, SP 곱셈, SP 나눗셈 수 전달 */
    wrapper->set_fp_accesses(power_stats->get_fp_accessess(0),
                             power_stats->get_fpmul_accessess(0),
                             power_stats->get_fpdiv_accessess(0));

    /* [한국어] === 초월 함수(SFU 트랜센덴탈) 파이프라인 통계 ===
     * sqrt(제곱근), log(로그), sin(사인), exp(지수) 연산 수 전달
     * 이 연산들은 특수 함수 유닛(SFU: Special Function Unit)에서 처리됨 */
    wrapper->set_trans_accesses(
        power_stats->get_sqrt_accessess(0), power_stats->get_log_accessess(0),
        power_stats->get_sin_accessess(0), power_stats->get_exp_accessess(0));

    /* [한국어] === 텐서 코어(Tensor Core) 접근 통계 ===
     * Volta/Turing 이상 아키텍처의 행렬 곱셈 전용 유닛 접근 수 전달 */
    wrapper->set_tensor_accesses(power_stats->get_tensor_accessess(0));

    /* [한국어] === 텍스처 유닛 접근 통계 ===
     * 텍스처 패치 연산 수 전달 */
    wrapper->set_tex_accesses(power_stats->get_tex_accessess(0));

    /* [한국어] === 실행 유닛 전체 통계 ===
     * 전체 FPU(SP+DP) 접근, IALU 접근, 전체 SFU 접근 수 전달
     * McPAT에서 실행 유닛 동적 전력 계산의 주 입력 */
    wrapper->set_exec_unit_power(power_stats->get_tot_fpu_accessess(0),
                                 power_stats->get_ialu_accessess(0),
                                 power_stats->get_tot_sfu_accessess(0));

    /* [한국어] === 평균 활성 스레드 수 ===
     * warp당 활성 스레드 수의 평균 — SIMT 실행 효율성 지표로 전력 스케일링에 사용 */
    wrapper->set_avg_active_threads(power_stats->get_active_threads(0));

    // Average active lanes for sp and sfu pipelines
    /* [한국어] === 평균 활성 레인 수 계산 ===
     * SP 파이프라인과 SFU 파이프라인의 평균 활성 레인 수 계산
     * get_sp/sfu_active_lanes()는 SM 전체 합산 값을 반환하므로
     * stat_sample_freq로 나눠 평균화, 최대 32(warp 크기)로 클리핑 */
    float avg_sp_active_lanes =
        (power_stats->get_sp_active_lanes()) / stat_sample_freq;  /* [한국어] SP 파이프라인 평균 활성 레인 수 */
    float avg_sfu_active_lanes =
        (power_stats->get_sfu_active_lanes()) / stat_sample_freq; /* [한국어] SFU 파이프라인 평균 활성 레인 수 */
    if (avg_sp_active_lanes > 32.0) avg_sp_active_lanes = 32.0;   /* [한국어] warp 크기(32레인) 상한 클리핑 */
    if (avg_sfu_active_lanes > 32.0) avg_sfu_active_lanes = 32.0; /* [한국어] warp 크기(32레인) 상한 클리핑 */
    assert(avg_sp_active_lanes <= 32);   /* [한국어] 클리핑 후 범위 검증 */
    assert(avg_sfu_active_lanes <= 32);  /* [한국어] 클리핑 후 범위 검증 */
    wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes); /* [한국어] McPAT에 레인 활성화율 전달 */

    /* [한국어] === NoC(Network-on-Chip, ICNT) 트래픽 통계 ===
     * SIMT 클러스터→메모리 파티션 방향과 그 역방향의 플릿(flit) 수 합산
     * intersim2(Booksim 기반 NoC 시뮬레이터)가 라우팅하는 패킷 단위 */
    double n_icnt_simt_to_mem = (double)power_stats->get_icnt_simt_to_mem(
        0);  // # flits from SIMT clusters
             // to memory partitions
    double n_icnt_mem_to_simt = (double)power_stats->get_icnt_mem_to_simt(
        0);  // # flits from memory
             // partitions to SIMT clusters
    wrapper->set_NoC_power(
        n_icnt_mem_to_simt +
        n_icnt_simt_to_mem);  // Number of flits traversing the interconnect
    /* [한국어] 양방향 플릿 수 합산 → McPAT NoC 전력 모델 입력 */

    wrapper->compute(); /* [한국어] McPAT 전력 계산 실행 — 입력된 모든 카운터로 동적+정적 전력 산출 */

    wrapper->update_components_power(); /* [한국어] 각 서브컴포넌트(L1, L2, SP, SFU 등)별 전력 값 업데이트 */
    wrapper->print_trace_files();       /* [한국어] 전력 추적 파일(-power_trace_enabled)에 이번 창 결과 기록 */
    power_stats->save_stats();          /* [한국어] 현재 카운터를 PREV에 복사 — 다음 창 델타 계산 기준점 갱신 */

    wrapper->detect_print_steady_state(0, tot_inst + inst); /* [한국어] 정상 상태(전력 편차 안정) 감지 및 출력 */

    wrapper->power_metrics_calculations(); /* [한국어] EDP(에너지-지연 곱) 등 파생 전력 메트릭 계산 */

    wrapper->dump(); /* [한국어] 이번 창의 전력 결과를 출력 파일에 덤프 */
  }
  // wrapper->close_files();
}

/*
 * [한국어]
 * mcpat_reset_perf_count - McPAT 래퍼 내부 성능 카운터 초기화
 *
 * @wrapper: gpgpu_sim_wrapper — 카운터를 초기화할 McPAT 래퍼 객체
 * @return: 없음 (void)
 *
 * wrapper->reset_counters()를 호출하여 McPAT 내부의 누적 카운터를 0으로 리셋한다.
 * 커널 전환 경계에서 calculate_hw_mcpat() 호출 전 또는 후에 호출되어
 * 이전 커널의 카운터가 다음 커널 계산 결과에 누적되지 않도록 방지한다.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() (커널 완료 후) → [mcpat_reset_perf_count] → wrapper->reset_counters()
 */
void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper) {
  wrapper->reset_counters(); /* [한국어] McPAT 래퍼 내부의 모든 성능 카운터 0으로 초기화 */
}

/*
 * [한국어]
 * parse_hw_file - HW 전력/성능 측정 CSV 파일에서 벤치마크/커널 항목 검색 및 파싱
 *
 * @hwpowerfile: 검색할 CSV 파일 경로 (행마다 하나의 [벤치마크, 커널, 카운터...] 항목)
 * @find_target_kernel: true=benchname+kernelname 정확 매칭, false=benchname만 매칭
 * @hw_data: [출력] 매칭 행의 CSV 필드 벡터 — HW_BENCH_NAME, HW_KERNEL_NAME, HW_CYCLES,
 *           HW_L1_RH 등 HW_* 열거형 인덱스로 접근
 * @benchname: 검색할 벤치마크 이름 (CSV HW_BENCH_NAME 열과 비교)
 * @executed_kernelname: 검색할 커널 이름 (CSV HW_KERNEL_NAME 열과 비교)
 * @return: 항목을 찾으면 true, 파일 끝까지 찾지 못하면 false
 *
 * calculate_hw_mcpat()에서 최대 2회 호출된다:
 *   1차: find_target_kernel=true — 정확한 커널 이름 매칭
 *   2차(1차 실패 시): find_target_kernel=false — 벤치마크 이름만으로 첫 항목 반환
 * CSV 파일을 행 단위로 읽으며 쉼표로 분리하여 hw_data에 채운다.
 * 매칭 즉시 파일을 닫고 반환하므로 전체 파일을 읽지 않는다.
 *
 * 에러 처리: 파일 열기 실패 시 하드 크래시(fstream은 예외 미발생, 이후 접근에서 실패)
 *
 * 호출 체인:
 *   calculate_hw_mcpat → [parse_hw_file] (최대 2회 호출)
 */
bool parse_hw_file(char *hwpowerfile, bool find_target_kernel,
                   vector<string> &hw_data, char *benchname,
                   std::string executed_kernelname) {
  fstream hw_file;                      /* [한국어] CSV 파일 스트림 */
  hw_file.open(hwpowerfile, ios::in);   /* [한국어] 읽기 모드로 HW 전력 측정 파일 열기 */
  string line, word, temp;              /* [한국어] 행 전체, 필드 단위, 임시 버퍼 */
  while (!hw_file.eof()) {              /* [한국어] 파일 끝까지 모든 행 순회 */
    hw_data.clear();                    /* [한국어] 이전 행의 파싱 결과 초기화 */
    getline(hw_file, line);             /* [한국어] 한 행 전체를 line에 읽기 */
    stringstream s(line);               /* [한국어] 행을 stringstream으로 변환해 쉼표 파싱 준비 */
    while (getline(s, word, ',')) {     /* [한국어] 쉼표 구분자로 필드별 분리 */
      hw_data.push_back(word);          /* [한국어] 파싱된 필드를 벡터에 순서대로 추가 */
    }
    if (hw_data[HW_BENCH_NAME] == std::string(benchname)) {
      /* [한국어] 벤치마크 이름 매칭 확인 — HW_BENCH_NAME 인덱스는 CSV 첫 번째 열 */
      if (find_target_kernel) {         /* [한국어] 커널 이름까지 정확히 매칭해야 하는 모드 */
        if (hw_data[HW_KERNEL_NAME] == "") {
          /* [한국어] CSV 행에 커널 이름이 없는 경우 — 벤치마크 레벨 통합 항목으로 간주하고 반환 */
          hw_file.close();
          return true;
        } else {
          if (hw_data[HW_KERNEL_NAME] == executed_kernelname) {
            /* [한국어] 커널 이름까지 정확히 일치 — 매칭 성공 */
            hw_file.close();
            return true;
          }
        }
      } else {
        /* [한국어] 벤치마크 이름만으로 매칭하는 폴백 모드 — 첫 번째 매칭 행 반환 */
        hw_file.close();
        return true;
      }
    }
  }
  hw_file.close();  /* [한국어] 파일 끝까지 매칭 실패 — 파일 닫고 false 반환 */
  return false;     /* [한국어] 대상 벤치마크/커널을 CSV에서 찾지 못함 */
}

/*
 * [한국어]
 * calculate_hw_mcpat - HW 측정값 또는 하이브리드 모드 커널별 전력 계산
 *
 * @config: gpgpu_sim_config — 전력 시뮬레이션 모드 등 전역 설정 (현재 직접 미사용)
 * @shdr_config: shader_core_config — SM 개수, SP/SFU 유닛 수 등 코어 구성
 * @wrapper: gpgpu_sim_wrapper — McPAT 래퍼 (set_*_power(), compute(), dump() 포함)
 * @power_stats: power_stat_t — Accel-Sim 마이크로아키텍처 카운터 (하이브리드 모드에서 부분 사용)
 * @stat_sample_freq: 샘플링 주기 (평균화에 사용)
 * @tot_cycle / @cycle: 누적/현재 사이클 수 (하이브리드 모드의 사이클 수 오버라이드에 사용)
 * @tot_inst / @inst: 누적/현재 명령어 수 (현재 이 함수에서 직접 사용 안 함)
 * @power_simulation_mode: 1=HW 전용, 2=하이브리드(HW+Accel-Sim 선택적 혼합)
 * @dvfs_enabled: DVFS 활성화 여부
 * @hwpowerfile: HW 성능/전력 측정값 CSV 파일 경로
 * @benchname: CSV에서 검색할 벤치마크 이름
 * @executed_kernelname: CSV에서 검색할 커널 이름
 * @accelwattch_hybrid_configuration: 길이 HW_NUM_STATS의 불리언 배열.
 *   [i]=true이면 i번 카운터를 Accel-Sim 값으로 대체, false이면 CSV HW 값 사용.
 * @aggregate_power_stats: true이면 여러 커널 실행을 누적하여 합산 계산
 * @return: 없음 (void)
 *
 * 동작 흐름:
 *   1. parse_hw_file()로 CSV에서 HW 카운터 로드 (실패 시 assert로 중단)
 *   2. perf_cycles 결정: HW 사이클 수 또는 하이브리드 모드에서 Accel-Sim 사이클 수
 *   3. 각 카운터(L1/L2 히트/미스, DRAM RD/WR, ALU 접근 등)를 CSV 또는 power_stats에서 선택
 *   4. wrapper->set_*_power()로 McPAT 입력 적재
 *   5. wrapper->compute() → wrapper->dump()로 전력 계산 및 출력
 *   6. power_stats의 커널 기준점 갱신 (다음 커널의 델타 계산을 위해)
 *   7. power_stats->clear()로 카운터 초기화
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 완료 후 단일 스레드에서 호출.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() (커널 완료 시) → [calculate_hw_mcpat] → parse_hw_file()
 *                                     → wrapper->compute() → wrapper->dump()
 */
void calculate_hw_mcpat(
    const gpgpu_sim_config &config, const shader_core_config *shdr_config,
    class gpgpu_sim_wrapper *wrapper, class power_stat_t *power_stats,
    unsigned stat_sample_freq, unsigned tot_cycle, unsigned cycle,
    unsigned tot_inst, unsigned inst, int power_simulation_mode,
    bool dvfs_enabled, char *hwpowerfile, char *benchname,
    std::string executed_kernelname,
    const bool *accelwattch_hybrid_configuration, bool aggregate_power_stats) {
  /* Reading HW data from CSV file */

  vector<string> hw_data;    /* [한국어] CSV 파싱 결과를 담을 문자열 벡터 — HW_* 인덱스로 접근 */
  bool kernel_found = false; /* [한국어] CSV에서 대상 커널/벤치마크를 찾았는지 여부 */
  kernel_found = parse_hw_file(
      hwpowerfile, true, hw_data, benchname,
      executed_kernelname);  // Searching for matching executed_kernelname.
  /* [한국어] 1차 검색: 벤치마크+커널 이름 정확 매칭 */
  if (!kernel_found)
    kernel_found = parse_hw_file(
        hwpowerfile, false, hw_data, benchname,
        executed_kernelname);  // Searching for any kernel with same benchname.
  /* [한국어] 2차 검색(폴백): 벤치마크 이름만으로 첫 항목 반환 */
  assert(
      "Could not find perf stats for the target benchmark in hwpowerfile.\n" &&
      (kernel_found));
  /* [한국어] 두 번의 검색 모두 실패 시 assert로 시뮬레이터 중단 — CSV 파일 내용 확인 필요 */
  unsigned perf_cycles =
      static_cast<unsigned int>(std::stod(hw_data[HW_CYCLES]) + 0.5);
  /* [한국어] CSV에서 HW 사이클 수를 읽어 부동소수점→정수 변환 (반올림) */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_CYCLES]))
    perf_cycles = cycle;
  /* [한국어] 하이브리드 모드에서 HW_CYCLES 카운터가 Accel-Sim 소스로 설정된 경우
   * CSV 값 대신 현재 커널의 Accel-Sim 사이클 수(cycle)로 대체 */
  wrapper->init_mcpat_hw_mode(
      perf_cycles);  // total PERF MODEL cycles for current kernel
  /* [한국어] McPAT에 이 커널의 총 사이클 수를 알려 에너지 계산(Power×Cycles)에 사용 */

  if (dvfs_enabled) {
    /* [한국어] DVFS 모드: 전압 설정 — HW_VOLTAGE 소스 선택 */
    if ((power_simulation_mode == 2) &&
        (accelwattch_hybrid_configuration[HW_VOLTAGE]))
      wrapper->set_model_voltage(1);  // performance model needs to support this
      /* [한국어] 하이브리드 모드에서 전압이 Accel-Sim 소스: 현재는 1로 고정 (추후 개선 필요) */
    else
      wrapper->set_model_voltage(std::stod(
          hw_data[HW_VOLTAGE]));  // performance model needs to support this
      /* [한국어] CSV에서 읽은 실측 전압 값(V)을 McPAT 전압 스케일링 파라미터로 설정 */
  }

  /* [한국어] === L1 데이터 캐시 카운터 초기값 (CSV에서 로드) ===
   * 하이브리드 모드에서 해당 카운터가 Accel-Sim 소스로 설정된 경우 아래에서 오버라이드됨 */
  double l1_read_hits = std::stod(hw_data[HW_L1_RH]);    /* [한국어] L1D 읽기 히트 수 (CSV) */
  double l1_read_misses = std::stod(hw_data[HW_L1_RM]);  /* [한국어] L1D 읽기 미스 수 (CSV) */
  double l1_write_hits = std::stod(hw_data[HW_L1_WH]);   /* [한국어] L1D 쓰기 히트 수 (CSV) */
  double l1_write_misses = std::stod(hw_data[HW_L1_WM]); /* [한국어] L1D 쓰기 미스 수 (CSV) */

  /* [한국어] 하이브리드 모드: L1 읽기 히트를 Accel-Sim 델타로 대체
   * aggregate_stat=1(누적) 현재값 - 커널 시작 기준점 = 이 커널의 L1 읽기 히트 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L1_RH]))
    l1_read_hits =
        power_stats->get_l1d_read_hits(1) - power_stats->l1r_hits_kernel;
  /* [한국어] 하이브리드 모드: L1 읽기 미스를 Accel-Sim 델타로 대체 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L1_RM]))
    l1_read_misses =
        power_stats->get_l1d_read_misses(1) - power_stats->l1r_misses_kernel;
  /* [한국어] 하이브리드 모드: L1 쓰기 히트를 Accel-Sim 델타로 대체 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L1_WH]))
    l1_write_hits =
        power_stats->get_l1d_write_hits(1) - power_stats->l1w_hits_kernel;
  /* [한국어] 하이브리드 모드: L1 쓰기 미스를 Accel-Sim 델타로 대체 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L1_WM]))
    l1_write_misses =
        power_stats->get_l1d_write_misses(1) - power_stats->l1w_misses_kernel;

  if (aggregate_power_stats) {
    /* [한국어] 누적 모드: 여러 커널에 걸친 명령어 수 합산
     * aggregate_stat=1(CURRENT값 전체)로 현재 커널까지의 누적 명령어 수 더하기 */
    power_stats->tot_inst_execution += power_stats->get_total_inst(1);          /* [한국어] 전체 명령어 수 누적 */
    power_stats->tot_int_inst_execution += power_stats->get_total_int_inst(1);  /* [한국어] 정수 명령어 수 누적 */
    power_stats->tot_fp_inst_execution += power_stats->get_total_fp_inst(1);    /* [한국어] FP 명령어 수 누적 */
    power_stats->commited_inst_execution += power_stats->get_committed_inst(1); /* [한국어] 커밋된 명령어 수 누적 */
    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes,
        cycle,  // TODO: core.[0] cycles counts don't matter, remove this
        /* [한국어] 현재 커널 사이클 수 (내부 코어 카운터는 무의미해 제거 예정) */
        cycle, power_stats->tot_inst_execution,              /* [한국어] 누적 전체 명령어 */
        power_stats->tot_int_inst_execution, power_stats->tot_fp_inst_execution,
        l1_read_hits + l1_read_misses, l1_write_hits + l1_write_misses,        /* [한국어] L1D 총 접근(히트+미스) */
        power_stats->commited_inst_execution);               /* [한국어] 누적 커밋된 명령어 수 */
  } else {
    /* [한국어] 비누적 모드: 이 커널의 명령어 수만 사용
     * aggregate_stat=1이지만 커널 기준점 차감 없이 현재 누적값 그대로 전달
     * (커널 시작 시 power_stats가 clear()되므로 사실상 이 커널만의 카운터) */
    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes,
        cycle,  // TODO: core.[0] cycles counts don't matter, remove this
        cycle, power_stats->get_total_inst(1),      /* [한국어] 이 커널의 전체 명령어 수 */
        power_stats->get_total_int_inst(1), power_stats->get_total_fp_inst(1),
        l1_read_hits + l1_read_misses, l1_write_hits + l1_write_misses,
        power_stats->get_committed_inst(1));         /* [한국어] 이 커널의 커밋된 명령어 수 */
  }

  // Single RF for both int and fp ops -- activity factor set to 0 for
  // Accelwattch HW and Accelwattch Hybrid because no HW Perf Stats for register
  // files
  /* [한국어] 레지스터 파일: HW/하이브리드 모드에서는 HW 성능 카운터로 RF 통계 없으므로
   * activity factor=0 (Accel-Sim 카운터는 있으나 HW CSV에는 없음 → 항상 Accel-Sim 값 사용)
   * 커널 시작 이후의 델타: aggregate_stat=1(누적값) 전달 (커널 시작 시 clear()됨) */
  wrapper->set_regfile_power(power_stats->get_regfile_reads(1),          /* [한국어] 레지스터 읽기 접근 수 */
                             power_stats->get_regfile_writes(1),         /* [한국어] 레지스터 쓰기 접근 수 */
                             power_stats->get_non_regfile_operands(1));  /* [한국어] 비 레지스터 오퍼랜드(즉시값 등) 수 */

  // Instruction cache stats -- activity factor set to 0 for Accelwattch HW and
  // Accelwattch Hybrid because no HW Perf Stats for instruction cache
  /* [한국어] L1 명령어 캐시: HW CSV에 I-캐시 통계 없으므로 항상 Accel-Sim 델타 사용
   * 커널 시작 기준점(l1i_hits_kernel, l1i_misses_kernel)을 차감하여 이 커널만의 증분 계산 */
  wrapper->set_icache_power(
      power_stats->get_inst_c_hits(1) - power_stats->l1i_hits_kernel,    /* [한국어] 이 커널의 L1I 히트 증분 */
      power_stats->get_inst_c_misses(1) - power_stats->l1i_misses_kernel);/* [한국어] 이 커널의 L1I 미스 증분 */

  // Constant Cache, shared memory, texture cache
  /* [한국어] 상수 캐시: 하이브리드 모드에서 HW_CC_ACC가 Accel-Sim 소스이면 시뮬 카운터 사용 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_CC_ACC]))
    wrapper->set_ccache_power(
        power_stats->get_const_accessess(1) - power_stats->cc_accesses_kernel,
        /* [한국어] 이 커널의 상수 캐시 접근 증분 */
        0);  // assuming all HITS in constant cache for now
  else
    wrapper->set_ccache_power(
        std::stod(hw_data[HW_CC_ACC]),  /* [한국어] CSV의 HW 상수 캐시 접근 수 */
        0);  // assuming all HITS in constant cache for now

  // wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
  //                           power_stats->get_texture_c_misses());
  /* [한국어] 텍스처 캐시: 현재 주석 처리됨 (HW 측정 모드에서 텍스처 통계 미지원) */

  /* [한국어] 공유 메모리: 하이브리드 모드에서 HW_SHRD_ACC가 Accel-Sim 소스이면 시뮬 카운터 사용 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_SHRD_ACC]))
    wrapper->set_shrd_mem_power(power_stats->get_shmem_access(1) -
                                power_stats->shared_accesses_kernel);
                                /* [한국어] 이 커널의 공유 메모리 접근 증분 */
  else
    wrapper->set_shrd_mem_power(std::stod(hw_data[HW_SHRD_ACC])); /* [한국어] CSV의 HW 공유 메모리 접근 수 */

  /* [한국어] L1D 캐시: 앞서 결정된 l1_read/write_hits/misses 전달
   * (CSV 값 또는 하이브리드 모드의 Accel-Sim 델타 중 선택됨) */
  wrapper->set_l1cache_power(l1_read_hits, l1_read_misses, l1_write_hits,
                             l1_write_misses);

  /* [한국어] === L2 캐시 카운터 초기값 (CSV에서 로드) ===
   * 하이브리드 모드에서 해당 카운터가 Accel-Sim 소스로 설정된 경우 아래에서 오버라이드됨 */
  double l2_read_hits = std::stod(hw_data[HW_L2_RH]);    /* [한국어] L2 읽기 히트 수 (CSV) */
  double l2_read_misses = std::stod(hw_data[HW_L2_RM]);  /* [한국어] L2 읽기 미스 수 (CSV) */
  double l2_write_hits = std::stod(hw_data[HW_L2_WH]);   /* [한국어] L2 쓰기 히트 수 (CSV) */
  double l2_write_misses = std::stod(hw_data[HW_L2_WM]); /* [한국어] L2 쓰기 미스 수 (CSV) */

  /* [한국어] 하이브리드 모드: 각 L2 카운터를 Accel-Sim 델타로 오버라이드 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L2_RH]))
    l2_read_hits =
        power_stats->get_l2_read_hits(1) - power_stats->l2r_hits_kernel;   /* [한국어] L2 읽기 히트 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L2_RM]))
    l2_read_misses =
        power_stats->get_l2_read_misses(1) - power_stats->l2r_misses_kernel; /* [한국어] L2 읽기 미스 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L2_WH]))
    l2_write_hits =
        power_stats->get_l2_write_hits(1) - power_stats->l2w_hits_kernel;   /* [한국어] L2 쓰기 히트 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_L2_WM]))
    l2_write_misses =
        power_stats->get_l2_write_misses(1) - power_stats->l2w_misses_kernel; /* [한국어] L2 쓰기 미스 증분 */

  /* [한국어] 결정된 L2 카운터를 McPAT L2 전력 모델에 전달 */
  wrapper->set_l2cache_power(l2_read_hits, l2_read_misses, l2_write_hits,
                             l2_write_misses);

  /* [한국어] === 유휴 SM 수 계산 ===
   * m_active_sms를 stat_sample_freq로 나눠 샘플링 창 평균 활성 SM 수 계산 */
  float active_sms = (*power_stats->m_active_sms) / stat_sample_freq; /* [한국어] 평균 활성 SM 수 */
  float num_cores = shdr_config->num_shader();                         /* [한국어] 전체 SM 개수 */
  float num_idle_core = num_cores - active_sms;                        /* [한국어] 평균 유휴 SM 수 */
  wrapper->set_num_cores(num_cores);                                   /* [한국어] McPAT에 전체 SM 수 전달 */
  /* [한국어] 하이브리드 모드: 유휴 SM 수를 Accel-Sim 값 또는 CSV HW 값 중 선택 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_NUM_SM_IDLE]))
    wrapper->set_idle_core_power(num_idle_core);                        /* [한국어] Accel-Sim 유휴 SM 수 */
  else
    wrapper->set_idle_core_power(std::stod(hw_data[HW_NUM_SM_IDLE]));  /* [한국어] CSV HW 유휴 SM 수 */

  /* [한국어] === 파이프라인 듀티 사이클 계산 ===
   * 이 커널 동안 파이프라인이 활성이었던 비율의 평균, 최대 0.8로 클리핑 */
  float pipeline_duty_cycle =
      ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) < 0.8)
          ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)  /* [한국어] 0.8 미만이면 그대로 */
          : 0.8;                                                                  /* [한국어] 0.8 이상이면 클리핑 */

  /* [한국어] 하이브리드 모드: 파이프라인 듀티 사이클 소스 선택 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_PIPE_DUTY]))
    wrapper->set_duty_cycle_power(pipeline_duty_cycle);                  /* [한국어] Accel-Sim 듀티 사이클 */
  else
    wrapper->set_duty_cycle_power(std::stod(hw_data[HW_PIPE_DUTY]));    /* [한국어] CSV HW 듀티 사이클 */

  // Memory Controller

  /* [한국어] === DRAM 메모리 컨트롤러 카운터 (CSV 초기값) ===
   * dram_pre: 프리차지 명령 수 — HW CSV에 별도 열 없어 0으로 초기화 후 하이브리드에서 오버라이드 */
  double dram_reads = std::stod(hw_data[HW_DRAM_RD]);  /* [한국어] DRAM 읽기 명령 수 (CSV) */
  double dram_writes = std::stod(hw_data[HW_DRAM_WR]); /* [한국어] DRAM 쓰기 명령 수 (CSV) */
  double dram_pre = 0;                                  /* [한국어] DRAM 프리차지 명령 수 (초기=0) */
  /* [한국어] 하이브리드 모드: DRAM 읽기/쓰기/프리차지를 Accel-Sim 델타로 오버라이드 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_DRAM_RD]))
    dram_reads = power_stats->get_dram_rd(1) - power_stats->dram_rd_kernel;   /* [한국어] 이 커널의 DRAM 읽기 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_DRAM_WR]))
    dram_writes = power_stats->get_dram_wr(1) - power_stats->dram_wr_kernel;  /* [한국어] 이 커널의 DRAM 쓰기 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_DRAM_RD]))
    dram_pre = power_stats->get_dram_pre(1) - power_stats->dram_pre_kernel;   /* [한국어] 이 커널의 DRAM 프리차지 증분 */

  wrapper->set_mem_ctrl_power(dram_reads, dram_writes, dram_pre); /* [한국어] McPAT 메모리 컨트롤러 전력 모델에 전달 */

  if (aggregate_power_stats) {
    /* [한국어] 누적 모드: 실행 유닛 카운터들을 여러 커널에 걸쳐 합산
     * 각 get_*_accessess(1)은 aggregate_stat=1 — 커널 시작 이후 누적값 반환
     * 이를 *_acc_execution 변수에 더해 전체 워크로드의 합산 카운터 유지 */
    power_stats->ialu_acc_execution += power_stats->get_ialu_accessess(1);       /* [한국어] 정수 ALU 접근 수 누적 */
    power_stats->imul24_acc_execution += power_stats->get_intmul24_accessess(1); /* [한국어] 24비트 정수 곱셈 누적 */
    power_stats->imul32_acc_execution += power_stats->get_intmul32_accessess(1); /* [한국어] 32비트 정수 곱셈 누적 */
    power_stats->imul_acc_execution += power_stats->get_intmul_accessess(1);     /* [한국어] 범용 정수 곱셈 누적 */
    power_stats->idiv_acc_execution += power_stats->get_intdiv_accessess(1);     /* [한국어] 정수 나눗셈 누적 */
    power_stats->dp_acc_execution += power_stats->get_dp_accessess(1);           /* [한국어] 배정밀도 FP 누적 */
    power_stats->dpmul_acc_execution += power_stats->get_dpmul_accessess(1);     /* [한국어] DP 곱셈 누적 */
    power_stats->dpdiv_acc_execution += power_stats->get_dpdiv_accessess(1);     /* [한국어] DP 나눗셈 누적 */
    power_stats->fp_acc_execution += power_stats->get_fp_accessess(1);           /* [한국어] 단정밀도 FP 누적 */
    power_stats->fpmul_acc_execution += power_stats->get_fpmul_accessess(1);     /* [한국어] SP 곱셈 누적 */
    power_stats->fpdiv_acc_execution += power_stats->get_fpdiv_accessess(1);     /* [한국어] SP 나눗셈 누적 */
    power_stats->sqrt_acc_execution += power_stats->get_sqrt_accessess(1);       /* [한국어] sqrt 연산 누적 */
    power_stats->log_acc_execution += power_stats->get_log_accessess(1);         /* [한국어] log 연산 누적 */
    power_stats->sin_acc_execution += power_stats->get_sin_accessess(1);         /* [한국어] sin 연산 누적 */
    power_stats->exp_acc_execution += power_stats->get_exp_accessess(1);         /* [한국어] exp 연산 누적 */
    power_stats->tensor_acc_execution += power_stats->get_tensor_accessess(1);   /* [한국어] 텐서 코어 접근 누적 */
    power_stats->tex_acc_execution += power_stats->get_tex_accessess(1);         /* [한국어] 텍스처 접근 누적 */
    power_stats->tot_fpu_acc_execution += power_stats->get_tot_fpu_accessess(1); /* [한국어] 전체 FPU 접근 누적 */
    power_stats->tot_sfu_acc_execution += power_stats->get_tot_sfu_accessess(1); /* [한국어] 전체 SFU 접근 누적 */
    power_stats->tot_threads_acc_execution +=
        power_stats->get_tot_threads_kernel(1); /* [한국어] 활성 스레드 수 누적 */
    power_stats->tot_warps_acc_execution +=
        power_stats->get_tot_warps_kernel(1);   /* [한국어] 활성 warp 수 누적 */

    /* [한국어] SP/SFU 활성 레인 누적: get_*_active_lanes()는 SM당 평균 레인 수를 반환하므로
     * 전체 SM 수와 SP 유닛 수를 곱해 절대값으로 환산 후 누적 */
    power_stats->sp_active_lanes_execution +=
        (power_stats->get_sp_active_lanes() * shdr_config->num_shader() *
         shdr_config->gpgpu_num_sp_units);   /* [한국어] SP 파이프라인 활성 레인 수 누적 */
    power_stats->sfu_active_lanes_execution +=
        (power_stats->get_sfu_active_lanes() * shdr_config->num_shader() *
         shdr_config->gpgpu_num_sp_units);   /* [한국어] SFU 파이프라인 활성 레인 수 누적 */

    /* [한국어] 누적된 실행 유닛 카운터들을 McPAT에 전달 */
    wrapper->set_int_accesses(
        power_stats->ialu_acc_execution, power_stats->imul24_acc_execution,
        power_stats->imul32_acc_execution, power_stats->imul_acc_execution,
        power_stats->idiv_acc_execution);  /* [한국어] 누적 정수 ALU/곱셈/나눗셈 */

    wrapper->set_dp_accesses(power_stats->dp_acc_execution,
                             power_stats->dpmul_acc_execution,
                             power_stats->dpdiv_acc_execution); /* [한국어] 누적 배정밀도 FP */

    wrapper->set_fp_accesses(power_stats->fp_acc_execution,
                             power_stats->fpmul_acc_execution,
                             power_stats->fpdiv_acc_execution); /* [한국어] 누적 단정밀도 FP */

    wrapper->set_trans_accesses(
        power_stats->sqrt_acc_execution, power_stats->log_acc_execution,
        power_stats->sin_acc_execution, power_stats->exp_acc_execution); /* [한국어] 누적 SFU 초월함수 */

    wrapper->set_tensor_accesses(power_stats->tensor_acc_execution);  /* [한국어] 누적 텐서 코어 */
    wrapper->set_tex_accesses(power_stats->tex_acc_execution);        /* [한국어] 누적 텍스처 접근 */

    wrapper->set_exec_unit_power(power_stats->ialu_acc_execution,
                                 power_stats->tot_fpu_acc_execution,
                                 power_stats->tot_sfu_acc_execution); /* [한국어] 누적 실행 유닛 전체 */

    /* [한국어] 평균 활성 스레드 수: 누적 스레드 수 / 누적 warp 수 = warp당 평균 활성 스레드 */
    wrapper->set_avg_active_threads(
        (double)((double)power_stats->tot_threads_acc_execution /
                 (double)power_stats->tot_warps_acc_execution));

    // Average active lanes for sp and sfu pipelines
    /* [한국어] 누적된 활성 레인 수를 전체 SM 수, SP 유닛 수, 샘플링 주기로 나눠 평균화 */
    float avg_sp_active_lanes =
        (power_stats->sp_active_lanes_execution) / shdr_config->num_shader() /
        shdr_config->gpgpu_num_sp_units / stat_sample_freq;   /* [한국어] SP 파이프라인 평균 활성 레인 수 */
    float avg_sfu_active_lanes =
        (power_stats->sfu_active_lanes_execution) / shdr_config->num_shader() /
        shdr_config->gpgpu_num_sp_units / stat_sample_freq;   /* [한국어] SFU 파이프라인 평균 활성 레인 수 */
    if (avg_sp_active_lanes > 32.0) avg_sp_active_lanes = 32.0;   /* [한국어] 32레인(warp 크기) 상한 클리핑 */
    if (avg_sfu_active_lanes > 32.0) avg_sfu_active_lanes = 32.0; /* [한국어] 32레인(warp 크기) 상한 클리핑 */
    assert(avg_sp_active_lanes <= 32);   /* [한국어] 범위 검증 */
    assert(avg_sfu_active_lanes <= 32);  /* [한국어] 범위 검증 */
    wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes); /* [한국어] McPAT에 레인 활성화율 전달 */
  } else {
    /* [한국어] 비누적 모드: 이 커널만의 Accel-Sim 실행 유닛 카운터를 McPAT에 직접 전달 */
    wrapper->set_int_accesses(power_stats->get_ialu_accessess(1),
                              power_stats->get_intmul24_accessess(1),
                              power_stats->get_intmul32_accessess(1),
                              power_stats->get_intmul_accessess(1),
                              power_stats->get_intdiv_accessess(1)); /* [한국어] 이 커널의 정수 ALU 카운터 */

    wrapper->set_dp_accesses(power_stats->get_dp_accessess(1),
                             power_stats->get_dpmul_accessess(1),
                             power_stats->get_dpdiv_accessess(1)); /* [한국어] 이 커널의 DP FP 카운터 */

    wrapper->set_fp_accesses(power_stats->get_fp_accessess(1),
                             power_stats->get_fpmul_accessess(1),
                             power_stats->get_fpdiv_accessess(1)); /* [한국어] 이 커널의 SP FP 카운터 */

    wrapper->set_trans_accesses(
        power_stats->get_sqrt_accessess(1), power_stats->get_log_accessess(1),
        power_stats->get_sin_accessess(1), power_stats->get_exp_accessess(1)); /* [한국어] 이 커널의 SFU 초월함수 */

    wrapper->set_tensor_accesses(power_stats->get_tensor_accessess(1)); /* [한국어] 이 커널의 텐서 코어 */
    wrapper->set_tex_accesses(power_stats->get_tex_accessess(1));       /* [한국어] 이 커널의 텍스처 접근 */

    wrapper->set_exec_unit_power(power_stats->get_tot_fpu_accessess(1),
                                 power_stats->get_ialu_accessess(1),
                                 power_stats->get_tot_sfu_accessess(1)); /* [한국어] 이 커널의 실행 유닛 전체 */

    /* [한국어] 이 커널의 평균 활성 스레드 수 */
    wrapper->set_avg_active_threads(power_stats->get_active_threads(1));

    // Average active lanes for sp and sfu pipelines
    /* [한국어] 이 커널의 평균 SP/SFU 활성 레인 수 계산 */
    float avg_sp_active_lanes =
        (power_stats->get_sp_active_lanes()) / stat_sample_freq;  /* [한국어] SP 파이프라인 평균 활성 레인 수 */
    float avg_sfu_active_lanes =
        (power_stats->get_sfu_active_lanes()) / stat_sample_freq; /* [한국어] SFU 파이프라인 평균 활성 레인 수 */
    if (avg_sp_active_lanes > 32.0) avg_sp_active_lanes = 32.0;   /* [한국어] 32레인 상한 클리핑 */
    if (avg_sfu_active_lanes > 32.0) avg_sfu_active_lanes = 32.0; /* [한국어] 32레인 상한 클리핑 */
    assert(avg_sp_active_lanes <= 32);   /* [한국어] 범위 검증 */
    assert(avg_sfu_active_lanes <= 32);  /* [한국어] 범위 검증 */
    wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes); /* [한국어] McPAT에 레인 활성화율 전달 */
  }

  /* [한국어] === NoC(ICNT) 트래픽 통계 ===
   * 커널 시작 기준점(noc_tr_kernel, noc_rc_kernel)을 차감하여 이 커널만의 플릿 수 계산 */
  double n_icnt_simt_to_mem =
      (double)(power_stats->get_icnt_simt_to_mem(1) -
               power_stats->noc_tr_kernel);  // # flits from SIMT clusters
                                             // to memory partitions
  /* [한국어] SIMT 클러스터 → 메모리 파티션 방향 플릿 수 증분 */
  double n_icnt_mem_to_simt =
      (double)(power_stats->get_icnt_mem_to_simt(1) -
               power_stats->noc_rc_kernel);  // # flits from memory
                                             // partitions to SIMT clusters
  /* [한국어] 메모리 파티션 → SIMT 클러스터 방향 플릿 수 증분 */
  if ((power_simulation_mode == 2) &&
      (accelwattch_hybrid_configuration[HW_NOC]))
    wrapper->set_NoC_power(
        n_icnt_mem_to_simt +
        n_icnt_simt_to_mem);  // Number of flits traversing the interconnect
                              // from Accel-Sim
    /* [한국어] 하이브리드 모드에서 NoC가 Accel-Sim 소스: 양방향 플릿 수 합산 전달 */
  else
    wrapper->set_NoC_power(
        std::stod(hw_data[HW_NOC]));  // Number of flits traversing the
                                      // interconnect from HW
    /* [한국어] HW 모드: CSV에서 읽은 NoC 플릿 수 전달 */

  wrapper->compute();                    /* [한국어] McPAT 전력 계산 실행 */
  wrapper->update_components_power();    /* [한국어] 서브컴포넌트별 전력 업데이트 */
  wrapper->power_metrics_calculations(); /* [한국어] EDP 등 파생 전력 메트릭 계산 */
  wrapper->dump();                       /* [한국어] 전력 결과를 출력 파일에 덤프 */

  /* [한국어] === 커널 기준점 갱신 ===
   * 다음 커널에서 이 커널의 카운터 증분을 계산할 수 있도록
   * 현재 누적 카운터 값을 *_kernel 필드에 저장 */
  power_stats->l1r_hits_kernel = power_stats->get_l1d_read_hits(1);     /* [한국어] L1D 읽기 히트 기준점 갱신 */
  power_stats->l1r_misses_kernel = power_stats->get_l1d_read_misses(1); /* [한국어] L1D 읽기 미스 기준점 갱신 */
  power_stats->l1w_hits_kernel = power_stats->get_l1d_write_hits(1);    /* [한국어] L1D 쓰기 히트 기준점 갱신 */
  power_stats->l1w_misses_kernel = power_stats->get_l1d_write_misses(1);/* [한국어] L1D 쓰기 미스 기준점 갱신 */
  power_stats->shared_accesses_kernel = power_stats->get_const_accessess(1); /* [한국어] 상수 캐시 접근 기준점 갱신 */
  power_stats->cc_accesses_kernel = power_stats->get_shmem_access(1);   /* [한국어] 공유 메모리 접근 기준점 갱신 */
  power_stats->dram_rd_kernel = power_stats->get_dram_rd(1);             /* [한국어] DRAM 읽기 기준점 갱신 */
  power_stats->dram_wr_kernel = power_stats->get_dram_wr(1);             /* [한국어] DRAM 쓰기 기준점 갱신 */
  power_stats->dram_pre_kernel = power_stats->get_dram_pre(1);           /* [한국어] DRAM 프리차지 기준점 갱신 */
  power_stats->l1i_hits_kernel = power_stats->get_inst_c_hits(1);        /* [한국어] L1I 히트 기준점 갱신 */
  power_stats->l1i_misses_kernel = power_stats->get_inst_c_misses(1);    /* [한국어] L1I 미스 기준점 갱신 */
  power_stats->l2r_hits_kernel = power_stats->get_l2_read_hits(1);       /* [한국어] L2 읽기 히트 기준점 갱신 */
  power_stats->l2r_misses_kernel = power_stats->get_l2_read_misses(1);   /* [한국어] L2 읽기 미스 기준점 갱신 */
  power_stats->l2w_hits_kernel = power_stats->get_l2_write_hits(1);      /* [한국어] L2 쓰기 히트 기준점 갱신 */
  power_stats->l2w_misses_kernel = power_stats->get_l2_write_misses(1);  /* [한국어] L2 쓰기 미스 기준점 갱신 */
  power_stats->noc_tr_kernel = power_stats->get_icnt_simt_to_mem(1);     /* [한국어] NoC SIMT→MEM 플릿 기준점 갱신 */
  power_stats->noc_rc_kernel = power_stats->get_icnt_mem_to_simt(1);     /* [한국어] NoC MEM→SIMT 플릿 기준점 갱신 */

  power_stats->clear(); /* [한국어] 다음 커널을 위해 power_stats의 모든 CURRENT/PREV 카운터를 0으로 리셋 */
}