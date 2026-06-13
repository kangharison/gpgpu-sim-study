/*
 * [한국어 설명] GPGPU-Sim GPU 타이밍 시뮬레이터 구현 (gpu-sim.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍 시뮬레이터의 핵심 구현체이다.
 * gpgpu_sim::cycle()은 매 코어 클럭 사이클마다 CUDA 런타임(stream_manager)이
 * 호출하는 최상위 시뮬레이션 루프 함수로, GPU 전체(모든 SM 클러스터, DRAM,
 * L2 캐시, ICNT)를 순서대로 한 사이클씩 진행시킨다.
 * 또한 gpgpusim.config 설정 파일의 파라미터를 OptionParser에 등록하는
 * reg_options() 함수들(power_config, memory_config, shader_core_config,
 * gpgpu_sim_config)과, 커널 스케줄링(launch/select_kernel/issue_block2core),
 * 통계 집계(update_stats/print_stats), 데드락 감지(deadlock_check)를 구현한다.
 * 전력 모델(AccelWattch/McPAT)과의 연동도 GPGPUSIM_POWER_MODEL ifdef 블록을
 * 통해 이 파일에서 수행된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 타이밍 시뮬레이션 계층(gpgpu-sim/)의 최상위 조율자이다.
 * 실행 흐름:
 *   CUDA app → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *     → stream_manager (gpgpu_sim::cycle 반복 호출)
 *       → [이 파일] gpgpu_sim::cycle()
 *           1. next_clock_domain(): CORE/ICNT/DRAM/L2 중 진행할 도메인 결정
 *           2. CORE tick: simt_core_cluster::icnt_cycle() + core_cycle()
 *           3. ICNT tick: mem→ICNT pop, icnt_transfer()
 *           4. DRAM tick: memory_partition_unit::dram_cycle()
 *           5. L2 tick:  mem_sub_partition::cache_cycle()
 *           6. 통계 샘플링, 데드락 검사, CDP 커널 실행
 * 실행 컨텍스트: 호스트 유저스페이스 메인 스레드 (단일 스레드 시뮬레이션).
 * GPU 내부 병렬성(수천 warp)은 시뮬레이터가 순차적으로 emulate한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈 (직접 include):
 *   - shader.cc/h: simt_core_cluster, shader_core_ctx — SM 파이프라인
 *   - dram.cc/h: memory_partition_unit — DRAM 타이밍 모델
 *   - l2cache.cc (memory_sub_partition): L2 캐시 사이클
 *   - icnt_wrapper.h: icnt_pop/push/transfer/busy — ICNT NoC 인터페이스
 *   - gpu-cache.h: cache_stats — L1/L2 캐시 통계
 *   - mem_fetch.h: mem_fetch — 메모리 요청 패킷
 *   - addrdec.h: addrdec_t, addrdec_tlx — 주소 디코딩
 *   - power_interface.h: mcpat_cycle, init_mcpat — AccelWattch 전력 모델
 *   - visualizer.h: visualizer_printstat — AerialVision 시각화
 *   - cuda-sim/ptx_ir.h, cuda-sim.h: PTX 기능 시뮬레이션 연동
 *   - libcuda/gpgpu_context.h: gpgpu_context — 전체 시뮬레이터 문맥
 * 이 파일에 의존하는 모듈:
 *   - gpgpusim_entrypoint.cc: exec_gpgpu_sim 생성 및 cycle() 루프
 *   - stream_manager.cc: kernel launch/finish 흐름에서 cycle() 구동
 * 핵심 공유 자료구조:
 *   - m_cluster[]: simt_core_cluster 포인터 배열 (SM 클러스터들)
 *   - m_memory_partition_unit[]: DRAM+L2 파티션 배열
 *   - m_running_kernels[]: 현재 실행 중인 커널 목록
 *   - gpu_sim_cycle, gpu_tot_sim_cycle: 사이클 카운터
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpu_sim::cycle()          - 1 사이클 진행; 클럭 도메인 선택 후 각 서브시스템 구동
 * sst_gpgpu_sim::SST_cycle()  - SST 모드: 코어만 사이클 진행 (메모리는 SST 담당)
 * gpgpu_sim::init()           - 커널 실행 시작 시 사이클카운터/ICNT/비주얼라이저 초기화
 * gpgpu_sim::launch()         - 커널을 m_running_kernels[] 빈 슬롯에 등록
 * gpgpu_sim::select_kernel()  - 라운드로빈으로 CTA가 남은 커널 선택
 * gpgpu_sim::issue_block2core() - 모든 클러스터에 CTA 배정 (라운드로빈)
 * gpgpu_sim::active()         - SM 미완료 warp/DRAM busy/ICNT 트래픽 여부 확인
 * gpgpu_sim::update_stats()   - 현재 커널 통계를 누적 합산하고 카운터 리셋
 * gpgpu_sim::print_stats()    - 커널 완료 시 전체 성능·캐시·전력 통계 출력
 * gpgpu_sim::deadlock_check() - 50000사이클마다 명령어 진행 없으면 데드락 판정
 * gpgpu_sim_config::reg_options() - gpgpusim.config 전체 파라미터를 OptionParser 등록
 * gpgpu_sim_config::init_clock_domains() - 코어/ICNT/DRAM/L2 클럭 주파수·주기 계산
 * gpgpu_sim::next_clock_domain() - 최소 시간 기준 다음 틱할 클럭 도메인 비트마스크 반환
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
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

#include "gpu-sim.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

// #define MAX(a, b) (((a) > (b)) ? (a) : (b)) //redefined

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

/*
 * [한국어]
 * power_config::reg_options - AccelWattch 전력 시뮬레이션 파라미터를 OptionParser에 등록
 *
 * @opp: OptionParser 인스턴스 포인터; gpgpusim.config를 파싱하는 전역 파서 객체.
 *       gpgpu_sim_config::reg_options()가 power_config::reg_options(opp)를 호출해 전달.
 * @return: void
 *
 * gpgpusim.config에서 AccelWattch 전력 모델 관련 옵션(-accelwattch_xml_file,
 * -power_simulation_enabled, -power_simulation_mode 등)을 OptionParser에 등록해
 * 파서가 설정 파일을 읽을 때 해당 멤버 변수를 채울 수 있게 한다.
 * 하이브리드 설정(accelwattch_hybrid_configuration[])은 각 성능 카운터 항목별로
 * 시뮬레이션 값 사용 여부를 개별 bool 옵션으로 등록한다.
 * 이 함수는 시뮬레이터 시작 초기 단계에서 한 번만 호출된다.
 * 실행 컨텍스트: 호스트 main 스레드, 시뮬레이션 루프 진입 전 초기화 단계.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc: gpgpu_sim_config::reg_options()
 *     → power_config::reg_options() [이 함수]
 *       → option_parser_register() (option_parser.cc)
 */
void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

/*
 * [한국어]
 * memory_config::reg_options - 메모리 서브시스템 파라미터를 OptionParser에 등록
 *
 * @opp: OptionParser 인스턴스 포인터.
 * @return: void
 *
 * gpgpusim.config에서 DRAM 스케줄러 타입, DRAM 타이밍 문자열(-gpgpu_dram_timing_opt),
 * L2 캐시 설정, 메모리 파티션 수(-gpgpu_n_mem), 서브파티션 수, SST 모드 등
 * 메모리 시스템 전체 파라미터를 OptionParser에 등록한다.
 * 등록된 옵션들은 option_parser_cmdline/cfgfile 호출 이후 각 멤버 변수에 자동으로 채워진다.
 * DRAM 타이밍 파라미터는 문자열 형식으로 등록되며, memory_config::init()에서 파싱된다.
 * 실행 컨텍스트: 호스트 main 스레드, 시뮬레이션 시작 전 초기화 단계(한 번 호출).
 *
 * 호출 체인:
 *   gpgpu_sim_config::reg_options() → memory_config::reg_options() [이 함수]
 *     → option_parser_register() 다수
 *     → m_address_mapping.addrdec_setoption(opp) (addrdec.cc: 주소 디코딩 옵션 등록)
 */
void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  // SST mode activate
  option_parser_register(opp, "-SST_mode", OPT_BOOL, &SST_mode, "SST mode",
                         "0");
  m_address_mapping.addrdec_setoption(opp);
}

/*
 * [한국어]
 * shader_core_config::reg_options - SM(Streaming Multiprocessor) 파이프라인
 *                                   파라미터를 OptionParser에 등록
 *
 * @opp: OptionParser 인스턴스 포인터.
 * @return: void
 *
 * gpgpusim.config에서 SM 파이프라인 관련 파라미터들을 OptionParser에 등록한다.
 * 등록 항목:
 *   - 스레드/warp 구성: -gpgpu_shader_core_pipeline (스레드 수:warp 크기)
 *   - L1 캐시 설정: -gpgpu_cache:dl1, -gpgpu_tex_cache:l1, -gpgpu_const_cache:l1
 *   - 공유 메모리: -gpgpu_shmem_size, -gpgpu_shmem_per_block
 *   - 레지스터 파일: -gpgpu_shader_registers, -gpgpu_num_reg_banks
 *   - operand collector: -gpgpu_operand_collector_num_units_sp 등
 *   - warp 스케줄러: -gpgpu_scheduler (lrr/gto/two_level_active)
 *   - 실행 유닛 수: -gpgpu_num_sp_units, -gpgpu_num_sfu_units 등
 * 실행 컨텍스트: 호스트 main 스레드, 시뮬레이션 시작 전 초기화(한 번 호출).
 *
 * 호출 체인:
 *   gpgpu_sim_config::reg_options() → shader_core_config::reg_options() [이 함수]
 *     → option_parser_register() 다수 (option_parser.cc)
 */
void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

/*
 * [한국어]
 * gpgpu_sim_config::reg_options - 시뮬레이터 전체 설정 파라미터를 OptionParser에 등록
 *
 * @opp: OptionParser 인스턴스 포인터.
 * @return: void
 *
 * gpgpusim.config의 최상위 시뮬레이션 제어 파라미터들을 OptionParser에 등록한다.
 * 하위 설정 객체들(functional_sim_config, shader_core_config, memory_config,
 * power_config)의 reg_options()를 먼저 위임 호출한 뒤, 아래 항목들을 추가 등록한다:
 *   - 종료 조건: -gpgpu_max_cycle, -gpgpu_max_insn, -gpgpu_max_cta
 *   - 클럭 도메인: -gpgpu_clock_domains (코어:ICNT:L2:DRAM MHz)
 *   - 캐시 플러시: -gpgpu_flush_l1_cache, -gpgpu_flush_l2_cache
 *   - 데드락 감지: -gpgpu_deadlock_detect
 *   - Compute Capability 버전: -gpgpu_compute_capability_major/minor
 *   - 비주얼라이저: -visualizer_enabled, -visualizer_outputfile
 *   - 스택/힙/동기화 디바이스 리밋
 *   - CDP(동적 병렬화): -gpgpu_cdp_enabled, -gpgpu_kernel_launch_latency
 * 실행 컨텍스트: 호스트 main 스레드, gpgpusim_entrypoint.cc 초기화 단계.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc: gpgpu_context::init() → gpgpu_sim_config::reg_options()
 *     → gpgpu_functional_sim_config::reg_options() (PTX 시뮬레이션 파라미터)
 *     → shader_core_config::reg_options()
 *     → memory_config::reg_options()
 *     → power_config::reg_options()
 *     → option_parser_register() 다수
 */
void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * increment_x_then_y_then_z - dim3 인덱스를 X→Y→Z 순서로 1씩 증가시키는 유틸리티
 *
 * @i:     현재 인덱스 dim3 (x, y, z). 참조로 전달되어 이 함수 내에서 수정된다.
 * @bound: 각 차원의 최댓값 dim3. 각 차원은 [0, bound.x), [0, bound.y), [0, bound.z) 범위.
 * @return: void
 *
 * CUDA의 CTA(thread block) ID를 (x, y, z) 3차원 그리드 상에서 선형 순서로
 * 하나씩 순회할 때 사용한다. X가 먼저 증가하고, bound.x에 달하면 X를 0으로
 * 리셋하고 Y를 증가시키는 방식이다(열 우선 순서).
 * kernel_info_t::get_next_cta_id_single()가 내부적으로 이 함수를 통해
 * 다음 CTA ID를 계산한다.
 * 실행 컨텍스트: 호스트 메인 스레드, issue_block2core() 경로.
 *
 * 호출 체인:
 *   gpgpu_sim::issue_block2core() → shader_core_ctx::issue_block2core()
 *     → kernel_info_t::get_next_cta_id_single()
 *       → increment_x_then_y_then_z() [이 함수]
 */
void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;                    // [한국어] X 차원을 먼저 증가
  if (i.x >= bound.x) {    // [한국어] X가 경계에 도달하면 X를 리셋하고 Y 증가
    i.x = 0;                // [한국어] X 리셋
    i.y++;                  // [한국어] Y 차원 증가
    if (i.y >= bound.y) {  // [한국어] Y도 경계에 도달하면 Y를 리셋하고 Z 증가
      i.y = 0;              // [한국어] Y 리셋
      if (i.z < bound.z) i.z++;  // [한국어] Z 차원 증가 (경계 초과 방지)
    }
  }
}

/*
 * [한국어]
 * gpgpu_sim::launch - 커널을 GPU 시뮬레이터에 등록(발사)하는 함수
 *
 * @kinfo: 실행할 커널의 메타데이터 포인터 (kernel_info_t).
 *         커널 이름, 그리드 크기, CTA 크기, PTX 심볼 테이블 등을 포함.
 *         stream_manager가 CUDA 스트림에서 꺼내어 전달한다.
 * @return: void
 *
 * cuLaunchKernel 인터셉트 경로에서 stream_manager를 거쳐 호출된다.
 * m_running_kernels[] 배열에서 NULL이거나 완료된(done()) 슬롯을 찾아
 * kinfo를 삽입하고, 커널 시작 사이클(kernel_time_t.start_cycle)을 기록한다.
 * 동시 실행 커널 수는 m_config.max_concurrent_kernel로 제한된다.
 * 슬롯을 찾지 못하면(배열이 전부 활성 커널로 채워진 경우) assert로 중단한다.
 * CTA 크기가 SM의 최대 스레드 수를 초과하면 즉시 abort()한다.
 * 실행 컨텍스트: 호스트 메인 스레드, stream_manager::push() 경로.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() → gpgpu_sim::launch() [이 함수]
 *     (등록 후) → gpgpu_sim::select_kernel() → issue_block2core() 경로에서 실행 시작
 */
void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned kernelID = kinfo->get_uid();  // [한국어] 커널 고유 ID 추출
  unsigned long long streamID = kinfo->get_streamID();

  kernel_time_t kernel_time = {gpu_tot_sim_cycle + gpu_sim_cycle, 0};
  if (gpu_kernel_time.find(streamID) == gpu_kernel_time.end()) {
    std::map<unsigned, kernel_time_t> new_val;
    new_val.insert(std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    gpu_kernel_time.insert(
        std::pair<unsigned long long, std::map<unsigned, kernel_time_t>>(
            streamID, new_val));
  } else {
    gpu_kernel_time.at(streamID).insert(
        std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    ////////// assume same kernel ID do not appear more than once
  }

  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

/*
 * [한국어]
 * gpgpu_sim::can_start_kernel - 새 커널을 시작할 수 있는 빈 슬롯이 있는지 확인
 *
 * @return: true  m_running_kernels[]에 NULL이거나 완료된 슬롯이 존재함
 *          false 모든 슬롯이 활성 커널로 채워져 있음
 *
 * stream_manager가 새 커널을 GPU에 launch하기 전에 이 함수를 호출해
 * 슬롯 여유를 확인한다. 슬롯 수는 gpgpusim.config의 -gpgpu_max_concurrent_kernel
 * 값으로 결정된다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() → gpgpu_sim::can_start_kernel() [이 함수]
 */
bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {  // [한국어] 모든 슬롯 순회
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;  // [한국어] 비어 있거나 완료된 슬롯 발견 → 시작 가능
  }
  return false;  // [한국어] 모든 슬롯 점유 중 → 시작 불가
}

/*
 * [한국어]
 * gpgpu_sim::hit_max_cta_count - 최대 CTA 발행 수 제한에 도달했는지 확인
 *
 * @return: true  gpu_max_cta_opt 설정값이 0이 아니고, 현재까지 발행된 CTA 합산이
 *                그 값 이상임 → 더 이상 CTA를 발행하지 않아야 함
 *          false 아직 제한에 미달 또는 제한 없음 (gpu_max_cta_opt == 0)
 *
 * -gpgpu_max_cta 옵션으로 설정된 최대 CTA 수에 도달했을 때
 * 시뮬레이션을 조기 종료하거나 추가 CTA 배정을 막기 위해 사용된다.
 * get_more_cta_left()와 kernel_more_cta_left()가 이 함수를 먼저 호출한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   get_more_cta_left() → hit_max_cta_count() [이 함수]
 *   kernel_more_cta_left() → hit_max_cta_count() [이 함수]
 */
bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {  // [한국어] 제한값이 설정된 경우에만 검사
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;  // [한국어] 누적 발행 CTA가 제한에 도달
  }
  return false;  // [한국어] 제한 없거나 아직 미달
}

/*
 * [한국어]
 * gpgpu_sim::kernel_more_cta_left - 특정 커널에 아직 발행할 CTA가 남아 있는지 확인
 *
 * @kernel: 확인할 커널의 포인터. NULL이면 false 반환.
 * @return: true  최대 CTA 제한 미달 && 커널에 실행할 CTA가 남아 있음
 *          false 최대 CTA 제한 도달 또는 커널이 NULL 또는 CTA 소진
 *
 * select_kernel()이 특정 커널에 CTA를 배정할 수 있는지 확인할 때 사용한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::select_kernel() → kernel_more_cta_left() [이 함수]
 *     → hit_max_cta_count()
 *     → kernel_info_t::no_more_ctas_to_run()
 */
bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;  // [한국어] 전역 CTA 제한 초과 시 불가

  if (kernel && !kernel->no_more_ctas_to_run()) return true;  // [한국어] CTA 남아 있음

  return false;
}

/*
 * [한국어]
 * gpgpu_sim::get_more_cta_left - 실행 중인 어느 커널이든 발행 가능한 CTA가 남아 있는지 확인
 *
 * @return: true  CTA 제한 미달 && m_running_kernels[] 중 하나 이상에 CTA가 남음
 *          false 전체 CTA 제한 도달 또는 모든 커널의 CTA 소진
 *
 * active() 함수가 시뮬레이션 종료 여부를 판정할 때 사용하고,
 * cycle() 내 core_cycle() 호출 전에도 사용된다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::active() → get_more_cta_left() [이 함수]
 *   gpgpu_sim::cycle() core 루프 조건 → get_more_cta_left() [이 함수]
 */
bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;  // [한국어] 전역 CTA 제한 초과 시 바로 false

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;  // [한국어] 아직 CTA가 남은 커널 발견
  }
  return false;  // [한국어] 모든 커널 CTA 소진
}

/*
 * [한국어]
 * gpgpu_sim::decrement_kernel_latency - 실행 중인 모든 커널의 TB 발행 지연을 1씩 감소
 *
 * @return: void
 *
 * -gpgpu_kernel_launch_latency와 -gpgpu_TB_launch_latency로 설정된 커널/TB
 * 발행 지연(m_kernel_TB_latency)을 매 CORE 사이클마다 1씩 감소시킨다.
 * 지연이 0이 되어야 select_kernel()이 해당 커널에서 CTA를 배정하기 시작한다.
 * 실행 컨텍스트: 호스트 메인 스레드, cycle() CORE 도메인 처리 끝에서 호출.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → gpgpu_sim::decrement_kernel_latency() [이 함수]
 */
void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    // [한국어] 활성 커널이고 아직 발행 지연이 남아 있으면 1 감소
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

/*
 * [한국어]
 * gpgpu_sim::select_kernel - 다음에 CTA를 배정할 커널을 선택하는 스케줄러
 *
 * @return: kernel_info_t* CTA를 배정할 커널의 포인터.
 *                         발행 가능한 커널이 없으면 NULL 반환.
 *
 * 커널 스케줄링 정책: 라운드로빈(Round-Robin) + m_last_issued_kernel 기반 우선순위.
 *   1단계: m_last_issued_kernel 슬롯의 커널이 아직 CTA를 가지고 있고
 *          TB 발행 지연이 0이면 곧바로 그 커널 반환 (동일 커널 연속 배정 우선).
 *   2단계: m_last_issued_kernel 다음 슬롯부터 순환하며 kernel_more_cta_left()
 *          &&  TB 지연 없는 첫 커널 선택.
 * 처음 선택되는 커널은 start_cycle을 기록하고 m_executed_kernel_uids에 등록한다.
 * 실행 컨텍스트: 호스트 메인 스레드, issue_block2core() → simt_core_cluster 경로.
 *
 * 호출 체인:
 *   gpgpu_sim::issue_block2core()
 *     → simt_core_cluster::issue_block2core()
 *       → gpgpu_sim::select_kernel() [이 함수]
 *         → kernel_more_cta_left()
 *         → kernel_info_t::no_more_ctas_to_run()
 */
kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&  // [한국어] 마지막 발행 슬롯이 활성이고
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

/*
 * [한국어]
 * gpgpu_sim::finished_kernel - 완료된 커널의 UID를 FIFO 순서로 반환
 *
 * @return: unsigned 완료된 커널의 UID (고유 ID). 완료된 커널이 없으면 0 반환.
 *
 * m_finished_kernel 리스트에서 맨 앞 항목을 꺼내(pop_front) 반환한다.
 * 반환값 0은 "완료된 커널 없음"을 의미한다 (UID는 1부터 시작).
 * stream_manager가 이 함수를 poll하여 커널 완료 여부를 확인하고,
 * 완료된 커널에 대해 후처리(통계 출력, 캐시 플러시 등)를 수행한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() → gpgpu_sim::finished_kernel() [이 함수]
 */
unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) {  // [한국어] 완료된 커널 없으면
    last_streamID = -1;             // [한국어] 마지막 스트림 ID 초기화
    return 0;                       // [한국어] 0 반환 (완료 없음)
  }
  unsigned result = m_finished_kernel.front();  // [한국어] FIFO 큐 맨 앞 UID 취득
  m_finished_kernel.pop_front();                // [한국어] 큐에서 제거
  return result;                                // [한국어] UID 반환
}

/*
 * [한국어]
 * gpgpu_sim::set_kernel_done - 커널 실행 완료를 기록하고 슬롯을 해제하는 함수
 *
 * @kernel: 완료된 커널의 포인터. m_running_kernels[]에 존재해야 한다.
 * @return: void
 *
 * 커널이 모든 CTA를 실행 완료했을 때 shader_core_ctx가 이 함수를 호출한다.
 * 처리 순서:
 *   1. 커널의 end_cycle을 현재 시뮬레이션 사이클로 기록
 *   2. 커널 UID를 m_finished_kernel 리스트에 추가 (finished_kernel()이 poll)
 *   3. m_running_kernels[]에서 해당 커널 슬롯을 NULL로 초기화
 * last_uid / last_streamID는 통계 출력 시 마지막 커널 정보 조회에 사용된다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   shader_core_ctx::register_cta_thread_exit()
 *     → gpgpu_sim::set_kernel_done() [이 함수]
 *   (또는) gpgpu_sim::stop_all_running_kernels()
 *     → gpgpu_sim::set_kernel_done() [이 함수]
 */
void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();  // [한국어] 커널의 고유 ID 취득
  last_uid = uid;
  unsigned long long streamID = kernel->get_streamID();
  last_streamID = streamID;
  gpu_kernel_time.at(streamID).at(uid).end_cycle =
      gpu_tot_sim_cycle + gpu_sim_cycle;
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

/*
 * [한국어]
 * gpgpu_sim::stop_all_running_kernels - 현재 실행 중인 모든 커널을 강제 종료
 *
 * @return: void
 *
 * m_running_kernels[]에서 NULL이 아닌 모든 커널에 대해 set_kernel_done()을 호출해
 * 강제 종료 처리한다. 시뮬레이션이 최대 사이클/명령어 수에 도달해 조기 종료할 때
 * 또는 데드락이 감지되어 중단할 때 호출된다.
 * 각 커널을 종료한 뒤 해당 슬롯이 NULL이 됐음을 assert로 검증한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc: gpgpusim_exit() → stop_all_running_kernels() [이 함수]
 *     → set_kernel_done()
 */
void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;  // [한국어] 커널 벡터 순회 이터레이터
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active  [한국어] 활성 슬롯이면
      set_kernel_done(*k);  // Stop the kernel         [한국어] 커널 완료 처리
      assert(*k == NULL);   // [한국어] set_kernel_done이 슬롯을 NULL로 만들었음을 검증
    }
  }
}

/*
 * [한국어]
 * exec_gpgpu_sim::createSIMTCluster - 표준 실행 모드 SIMT 클러스터 배열을 생성
 *
 * @return: void
 *
 * exec_gpgpu_sim 생성자에서 호출되는 순수 가상 함수 구현체.
 * n_simt_clusters 개수만큼 exec_simt_core_cluster 객체를 동적 할당하여
 * m_cluster[] 배열에 저장한다. 각 클러스터는 exec_simt_core_cluster 타입으로,
 * SST 없이 독립적으로 실행되는 표준 모드 SM 클러스터이다.
 * 실행 컨텍스트: 호스트 메인 스레드, 시뮬레이터 생성 초기화 단계.
 *
 * 호출 체인:
 *   exec_gpgpu_sim 생성자 → exec_gpgpu_sim::createSIMTCluster() [이 함수]
 *     → new exec_simt_core_cluster(this, i, ...) (shader.cc)
 */
void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];  // [한국어] 클러스터 포인터 배열 할당
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

/*
 * [한국어]
 * sst_gpgpu_sim::createSIMTCluster - SST 연동 모드 SIMT 클러스터 배열을 생성
 *
 * @return: void
 *
 * sst_gpgpu_sim 생성자에서 호출되는 순수 가상 함수 구현체.
 * n_simt_clusters 개수만큼 sst_simt_core_cluster 객체를 동적 할당한다.
 * sst_simt_core_cluster는 SST Balar와 연동되어 메모리 요청을 SST로 전송하고
 * SST로부터 응답을 수신하는 방식으로 동작한다.
 * 또한 SST_gpgpu_reply_buffer를 클러스터 수만큼 resize한다.
 * 실행 컨텍스트: 호스트 메인 스레드, SST Balar 컴포넌트 초기화 단계.
 *
 * 호출 체인:
 *   sst_gpgpu_sim 생성자 → sst_gpgpu_sim::createSIMTCluster() [이 함수]
 *     → new sst_simt_core_cluster(this, i, ...) (shader.cc)
 */
// SST get its own simt_cluster
void sst_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];  // [한국어] 클러스터 포인터 배열 할당
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new sst_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                  m_shader_stats, m_memory_stats);
  SST_gpgpu_reply_buffer.resize(m_shader_config->n_simt_clusters);
}

/*
 * [한국어]
 * gpgpu_sim::gpgpu_sim - GPU 타이밍 시뮬레이터 최상위 객체 생성자
 *
 * @config: gpgpu_sim_config 설정 객체의 const 참조.
 *          power/shader/memory/clock 도메인 파라미터를 모두 포함.
 *          gpgpusim.config에서 파싱된 값들이 채워져 있어야 한다.
 * @ctx:    gpgpu_context 포인터. PTX 파서, 기능 시뮬레이션, 디바이스 런타임,
 *          통계 객체 등 시뮬레이터 전체 공유 문맥에 대한 접근을 제공한다.
 * @return: (생성자)
 *
 * 타이밍 시뮬레이터의 전체 자원을 초기화한다:
 *   1. 설정 포인터(m_shader_config, m_memory_config) 저장
 *   2. PTX warp 크기 설정 (ptx_parser)
 *   3. AccelWattch 전력 래퍼 객체 생성 (GPGPUSIM_POWER_MODEL 활성 시)
 *   4. 셰이더 통계(shader_core_stats), 메모리 통계(memory_stats_t),
 *      전력 통계(power_stat_t) 객체 할당
 *   5. 파이프라인 활용률/활성SM 계수 float 버퍼 malloc
 *   6. 각종 카운터(gpu_sim_insn, gpu_tot_issued_cta 등) 0으로 초기화
 *   7. SST 모드가 아닐 때만 메모리 파티션(memory_partition_unit),
 *      서브파티션(memory_sub_partition) 배열 및 ICNT(intersim2) 초기화
 *   8. m_running_kernels 슬롯 벡터를 max_concurrent_kernel 크기로 resize
 * createSIMTCluster()는 자식 클래스(exec/sst_gpgpu_sim) 생성자에서 호출된다.
 * 실행 컨텍스트: 호스트 메인 스레드, 시뮬레이터 최초 생성 시.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → exec_gpgpu_sim::exec_gpgpu_sim()
 *     → gpgpu_sim::gpgpu_sim() [이 함수]
 *       → icnt_wrapper_init(), icnt_create() (intersim2/)
 */
gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {  // [한국어] 기반 클래스 gpgpu_t 초기화 및 설정 참조 저장
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;
  last_streamID = -1;

  gpu_kernel_time.clear();

  // TODO: somehow move this logic to the sst_gpgpu_sim constructor?
  if (!m_config.is_SST_mode()) {
    // Init memory if not in SST mode
    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i] =
          new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
      for (unsigned p = 0;
           p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned submpid =
            i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        m_memory_sub_partition[submpid] =
            m_memory_partition_unit[i]->get_sub_partition(p);
      }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition);
  }
  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

/*
 * [한국어]
 * sst_gpgpu_sim::SST_receive_mem_reply - SST로부터 메모리 응답을 수신해 버퍼에 저장
 *
 * @core_id:  응답을 받을 SIMT 클러스터(코어 그룹) 번호. n_simt_clusters 미만이어야 함.
 * @mem_req:  mem_fetch 객체에 대한 void* 포인터. SST Balar가 이 타입으로 전달한다.
 * @return:   void
 *
 * SST Balar 컴포넌트가 메모리 응답을 GPGPU-Sim으로 전달할 때 호출한다.
 * core_id에 해당하는 SST_gpgpu_reply_buffer[] deque의 뒤쪽(push_back)에 mf를 추가한다.
 * SST_cycle()에서 sst_simt_core_cluster::icnt_cycle_SST()가 이 버퍼를 비워 SM으로 전달한다.
 * 실행 컨텍스트: SST 시뮬레이션 이벤트 핸들러 (SST 스레드).
 *
 * 호출 체인:
 *   SST Balar 컴포넌트 이벤트 핸들러 → SST_receive_mem_reply() [이 함수]
 *   (소비) → sst_simt_core_cluster::icnt_cycle_SST() → SST_pop_mem_reply()
 */
void sst_gpgpu_sim::SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  assert(core_id < m_shader_config->n_simt_clusters);  // [한국어] core_id 범위 검사
  mem_fetch *mf = (mem_fetch *)mem_req;                 // [한국어] void* → mem_fetch* 캐스팅

  (SST_gpgpu_reply_buffer[core_id]).push_back(mf);  // [한국어] 해당 코어의 응답 버퍼에 추가
}

/*
 * [한국어]
 * sst_gpgpu_sim::SST_pop_mem_reply - SST 메모리 응답 버퍼에서 앞쪽 항목을 꺼냄
 *
 * @core_id:  응답을 꺼낼 SIMT 클러스터 번호.
 * @return:   mem_fetch* 버퍼 맨 앞의 응답 패킷 포인터.
 *            버퍼가 비어 있으면 NULL 반환.
 *
 * sst_simt_core_cluster::icnt_cycle_SST()가 매 사이클마다 poll하여
 * 버퍼에 쌓인 SST 메모리 응답을 SM 파이프라인으로 전달한다.
 * FIFO(선입선출) 방식으로 가장 먼저 도착한 응답부터 처리한다.
 * 실행 컨텍스트: 호스트 메인 스레드, sst_gpgpu_sim::SST_cycle() 경로.
 *
 * 호출 체인:
 *   sst_simt_core_cluster::icnt_cycle_SST() → SST_pop_mem_reply() [이 함수]
 */
mem_fetch *sst_gpgpu_sim::SST_pop_mem_reply(unsigned core_id) {
  if (SST_gpgpu_reply_buffer[core_id].size() > 0) {  // [한국어] 버퍼에 응답이 있으면
    mem_fetch *temp = SST_gpgpu_reply_buffer[core_id].front();  // [한국어] 맨 앞 항목 취득
    SST_gpgpu_reply_buffer[core_id].pop_front();                // [한국어] 버퍼에서 제거
    return temp;                                                 // [한국어] 패킷 반환
  } else
    return NULL;  // [한국어] 버퍼 비어 있으면 NULL 반환
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

/*
 * [한국어]
 * gpgpu_sim_config::init_clock_domains - 클럭 도메인 주파수와 주기를 초기화
 *
 * @return: void
 *
 * -gpgpu_clock_domains 옵션으로 지정된 "코어:ICNT:L2:DRAM" MHz 문자열을 파싱해
 * 각 도메인의 주파수(Hz)와 주기(초)를 계산하여 멤버 변수에 저장한다.
 * MhZ 매크로(*1000000)를 적용해 MHz 단위를 Hz로 변환한다.
 * 예: "1000.0:2000.0:2000.0:2000.0" → core_freq=1GHz, icnt/l2/dram=2GHz
 * 계산된 주기값들은 cycle() → next_clock_domain()에서 각 도메인의
 * 다음 상승 에지(rising edge) 시간(core_time, icnt_time 등)을 누적할 때 사용된다.
 * 실행 컨텍스트: 호스트 메인 스레드, gpgpu_sim_config::init() 내부.
 *
 * 호출 체인:
 *   gpgpu_sim_config::init() → init_clock_domains() [이 함수]
 *   (이후) → gpgpu_sim::reinit_clock_domains() → 각 domain_time = 0 리셋
 *   (이후) → gpgpu_sim::cycle() → next_clock_domain() → domain_time += domain_period
 */
void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,  // [한국어] MHz 단위로 4개 주파수 파싱
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

/*
 * [한국어]
 * gpgpu_sim::reinit_clock_domains - 클럭 도메인 타이머를 0으로 리셋
 *
 * @return: void
 *
 * 새 커널 실행이 시작될 때(gpgpu_sim::init() 내부) 각 클럭 도메인의
 * 누적 시간 카운터(core_time, dram_time, icnt_time, l2_time)를 0으로 초기화한다.
 * 이 값들은 next_clock_domain()에서 가장 빨리 도달할 도메인을 선택하는 데 사용되며,
 * 매 cycle()마다 해당 도메인의 period만큼 증가한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::init() → gpgpu_sim::reinit_clock_domains() [이 함수]
 */
void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;  // [한국어] 코어 클럭 타이머 리셋
  dram_time = 0;  // [한국어] DRAM 클럭 타이머 리셋
  icnt_time = 0;  // [한국어] ICNT 클럭 타이머 리셋
  l2_time = 0;    // [한국어] L2 캐시 클럭 타이머 리셋
}

/*
 * [한국어]
 * gpgpu_sim::active - 시뮬레이션을 계속 진행해야 하는지 여부 판정
 *
 * @return: true  아직 처리할 작업이 남아 있음 (루프 계속)
 *          false 시뮬레이션 종료 조건 충족 또는 모든 작업 완료
 *
 * 다음 순서로 종료 조건을 검사한다:
 *   1. -gpgpu_max_cycle 초과 → false
 *   2. -gpgpu_max_insn 초과 → false
 *   3. -gpgpu_max_cta 초과 → false
 *   4. -gpgpu_max_completed_cta 초과 → false
 *   5. 데드락 감지(gpu_deadlock) → false
 *   6. 미완료 SM(get_not_completed > 0) 존재 → true
 *   7. 바쁜 DRAM 파티션(busy > 0) 존재 → true
 *   8. ICNT에 패킷이 남아 있음(icnt_busy()) → true
 *   9. 발행할 CTA가 남아 있음(get_more_cta_left()) → true
 *   10. 모두 아니면 → false (시뮬레이션 완료)
 * sst_gpgpu_sim::active()는 6,7,8 조건에서 DRAM/ICNT 검사를 제외한다
 * (SST가 메모리를 관리하므로).
 * 실행 컨텍스트: 호스트 메인 스레드, stream_manager의 시뮬레이션 루프.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() 또는 메인 루프 → gpgpu_sim::active() [이 함수]
 *     → m_cluster[i]->get_not_completed() (shader.cc)
 *     → m_memory_partition_unit[i]->busy() (dram.cc)
 *     → icnt_busy() (icnt_wrapper.cc)
 *     → get_more_cta_left()
 */
bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;  // [한국어] 최대 사이클 초과 → 종료
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

/*
 * [한국어]
 * sst_gpgpu_sim::active - SST 모드 시뮬레이션 종료 여부 판정
 *
 * @return: true  처리할 SM 작업 또는 CTA가 남아 있음
 *          false 종료 조건 충족 또는 모든 작업 완료
 *
 * gpgpu_sim::active()와 동일하지만, DRAM busy 검사와 ICNT busy 검사를 제외한다.
 * SST 모드에서는 메모리 시스템(DRAM, ICNT)을 SST가 관리하므로
 * GPGPU-Sim이 직접 busy 여부를 판단할 수 없다.
 * 실행 컨텍스트: 호스트 메인 스레드 (SST Balar 이벤트 루프).
 *
 * 호출 체인:
 *   SST Balar 이벤트 루프 → sst_gpgpu_sim::active() [이 함수]
 */
bool sst_gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;  // [한국어] 최대 사이클 초과
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  if (get_more_cta_left()) return true;
  return false;
}

/*
 * [한국어]
 * gpgpu_sim::init - 커널 실행 전 시뮬레이터 상태를 초기화
 *
 * @return: void
 *
 * 새로운 CUDA 커널(그리드)이 GPU에서 실행될 때마다 한 번 호출된다.
 * 처리 순서:
 *   1. 사이클/명령어/CTA 카운터 및 병렬 요청 카운터를 0으로 리셋
 *   2. AccelWattch McPAT 초기화 (GPGPUSIM_POWER_MODEL, 최초 커널 발사 시)
 *   3. reinit_clock_domains(): CORE/ICNT/DRAM/L2 타이머를 0으로 리셋
 *   4. PTX 기능 시뮬레이션에 셰이더 수 설정
 *   5. 모든 SIMT 클러스터 reinit (warp 상태 초기화)
 *   6. 셰이더 통계 new_grid() 리셋
 *   7. 비주얼라이저가 활성화된 경우: 제어 흐름 로거(CF logger) 생성
 *   8. CTA 카운트, 명령어-warp 점유율, 메모리 접근/지연 통계 로거 생성
 *   9. ICNT 네트워크 초기화 (g_network_mode 활성 시)
 * 실행 컨텍스트: 호스트 메인 스레드, stream_manager에서 커널 발사 직전.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() → gpgpu_sim::init() [이 함수]
 *     → reinit_clock_domains()
 *     → m_cluster[i]->reinit()   (shader.cc)
 *     → icnt_init()              (icnt_wrapper.cc, ICNT 활성화 시)
 */
void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;  // [한국어] 현재 커널의 사이클 카운터 리셋
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

/*
 * [한국어]
 * gpgpu_sim::update_stats - 현재 커널 통계를 전체 누적 통계에 합산하고 리셋
 *
 * @return: void
 *
 * 커널 실행이 완료된 후(set_kernel_done 이후) stream_manager가 호출한다.
 * 처리 순서:
 *   1. memlatstat_lat_pw(): 메모리 지연 통계 정기 샘플 기록
 *   2. gpu_tot_sim_cycle += gpu_sim_cycle (전체 누적 사이클에 합산)
 *   3. gpu_tot_sim_insn  += gpu_sim_insn  (전체 누적 명령어 수에 합산)
 *   4. gpu_tot_issued_cta += m_total_cta_launched
 *   5. 파티션 병렬 요청/응답 카운터들을 누적 합산
 *   6. gpu_tot_occupancy += gpu_occupancy (점유율 누적)
 *   7. 현재 커널 카운터(gpu_sim_cycle, gpu_sim_insn 등)를 모두 0으로 리셋
 *      → 다음 커널 실행 준비
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   stream_manager::empty_stream() → gpgpu_sim::update_stats() [이 함수]
 */
void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();  // [한국어] 메모리 지연 주기별 통계 기록
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

/*
 * [한국어]
 * gpgpu_sim::print_stats - 커널 실행 완료 후 전체 성능/캐시/전력 통계를 stdout에 출력
 *
 * @streamID: 완료된 커널이 속한 CUDA 스트림 ID. 통계 레이블 출력에 사용.
 * @return:   void
 *
 * 커널이 완료될 때 stream_manager가 호출한다.
 * 출력 항목:
 *   1. PTX 파일/라인 통계 파일 기록
 *   2. gpu_print_stat(): IPC, 사이클, 명령어, 점유율, L2 대역폭 등 핵심 지표
 *   3. ICNT 통계 (g_network_mode 활성 시)
 * gpu_print_stat() 내부에서 추가로 출력하는 항목:
 *   - 스톨 카운터(dramfull, icnt2sh), 파티션 병렬성 지표
 *   - 코어 캐시 통계(L1), L2 캐시 통계
 *   - 셰이더 파이프라인/스케줄러 통계
 *   - AccelWattch 전력 통계 (POWER_MODEL 활성 시)
 *   - DRAM 파티션별 통계
 *   - 인터커넥트 패킷 카운터
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   stream_manager → gpgpu_sim::print_stats() [이 함수]
 *     → gpgpu_ctx->stats->ptx_file_line_stats_write_file()
 *     → gpu_print_stat()
 *       → shader_print_cache_stats()
 *       → m_memory_partition_unit[i]->print()
 *       → icnt_display_stats() (ICNT 활성 시)
 */
void gpgpu_sim::print_stats(unsigned long long streamID) {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();  // [한국어] PTX 파일 라인 통계 파일에 기록
  gpu_print_stat(streamID);

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}

/*
 * [한국어]
 * gpgpu_sim::deadlock_check - 데드락 감지 시 상세 진단 정보를 출력하고 abort
 *
 * @return: void
 *
 * cycle() 내에서 50000 사이클마다 gpu_deadlock 플래그가 설정된 경우 호출된다.
 * gpu_deadlock은 50000 사이클 동안 gpu_sim_insn이 변하지 않으면 true로 설정된다.
 * 처리 순서:
 *   1. 미완료 SM 클러스터와 해당 warp 수를 출력
 *   2. 바쁜 DRAM 파티션 번호를 출력
 *   3. ICNT에 트래픽이 있으면 ICNT 상태를 출력
 *   4. GDB 재실행 안내 출력
 *   5. abort()로 즉시 종료
 * -gpgpu_deadlock_detect 0으로 비활성화할 수 있다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc: 시뮬레이션 루프 또는
 *   stream_manager → gpgpu_sim::deadlock_check() [이 함수]
 */
void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {  // [한국어] 데드락 감지 활성이고 플래그가 설정된 경우
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/*
 * [한국어]
 * gpgpu_sim::executed_kernel_info_string - 실행된 커널 이름과 UID를 문자열로 조합
 *
 * @return: std::string "kernel_name = <name1> <name2> ...\n
 *                       kernel_launch_uid = <uid1> <uid2> ...\n" 형식의 문자열.
 *
 * 통계 출력 시 어떤 커널이 실행됐는지 레이블을 붙이기 위해 사용된다.
 * m_executed_kernel_names와 m_executed_kernel_uids 벡터를 공백으로 연결한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::gpu_print_stat() → executed_kernel_info_string() [이 함수]
 */
/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

/*
 * [한국어]
 * gpgpu_sim::executed_kernel_name - 실행된 커널 이름을 단일 문자열로 반환
 *
 * @return: std::string 커널이 하나이면 그 이름만, 여러 개이면 공백 구분 목록.
 *
 * AccelWattch calculate_hw_mcpat()에 전달하여 커널 이름 기반 전력 데이터를
 * 하드웨어 성능 파일에서 찾는 데 사용한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 */
std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;  // [한국어] 문자열 스트림 (문자열 조립용)
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
/*
 * [한국어]
 * gpgpu_sim::set_cache_config - 특정 커널에 대한 특수 캐시 설정을 등록
 *
 * @kernel_name: 커널 이름 (함수명 문자열).
 * @cacheConfig: 적용할 캐시 설정 (FuncCachePreferL1/PreferShared/PreferNone).
 * @return: void
 *
 * CUDA의 cudaFuncSetCacheConfig() 호출 시 libcuda 인터셉트가 이 함수를 호출한다.
 * 저장된 설정은 커널 launch 시점에 set_cache_config(kernel_name)이 읽어 적용한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 */
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;  // [한국어] 커널 이름을 키로 캐시 설정 저장
}

/*
 * [한국어]
 * gpgpu_sim::get_cache_config - 특정 커널의 특수 캐시 설정을 조회
 *
 * @kernel_name: 조회할 커널 이름.
 * @return: FuncCache 등록된 캐시 설정. 없으면 (FuncCache)0 반환.
 */
FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

/*
 * [한국어]
 * gpgpu_sim::has_special_cache_config - 특정 커널에 특수 캐시 설정이 있는지 확인
 *
 * @kernel_name: 확인할 커널 이름.
 * @return: true  m_special_cache_config에 해당 커널의 설정이 있음
 *          false 없음
 */
bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

/*
 * [한국어]
 * gpgpu_sim::set_cache_config (커널 이름 버전) - 커널 이름으로 캐시 설정을 SM에 적용
 *
 * @kernel_name: 적용할 커널의 이름.
 * @return: void
 *
 * 커널 launch 직전 stream_manager가 호출한다.
 * 특수 설정이 있으면 change_cache_config(get_cache_config())를,
 * 없으면 change_cache_config(FuncCachePreferNone)을 호출하여
 * L1D/공유메모리 비율을 기본값으로 복원한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 */
void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {  // [한국어] 특수 캐시 설정이 등록된 커널이면
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

/*
 * [한국어]
 * gpgpu_sim::change_cache_config - L1D 캐시/공유메모리 비율 설정을 실제로 변경
 *
 * @cache_config: 적용할 캐시 설정 (FuncCachePreferNone/PreferL1/PreferShared).
 * @return: void
 *
 * 현재 L1D 설정과 다른 경우 모든 SM의 L1 캐시를 먼저 무효화(invalidate)하고,
 * cache_config에 따라 L1D 설정 문자열과 공유 메모리 크기를 재초기화한다.
 *   - FuncCachePreferNone:   기본 L1D 설정, 기본 공유 메모리 크기
 *   - FuncCachePreferL1:     L1 크게/공유 메모리 작게 (m_config_stringPrefL1)
 *   - FuncCachePreferShared: 공유 메모리 크게/L1 작게 (m_config_stringPrefShared)
 * 설정 문자열이 없으면 경고를 출력하고 기본값으로 폴백한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::set_cache_config(kernel_name) → change_cache_config() [이 함수]
 *     → m_cluster[i]->cache_invalidate() (shader.cc)
 *     → m_shader_config->m_L1D_config.init()
 */
void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    // [한국어] 현재 설정과 다른 경우: L1 캐시를 모두 무효화하고 재설정
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

/*
 * [한국어]
 * gpgpu_sim::clear_executed_kernel_info - 실행된 커널 이름/UID 기록을 초기화
 *
 * @return: void
 *
 * 통계 출력(gpu_print_stat) 완료 후 m_executed_kernel_names와
 * m_executed_kernel_uids 벡터를 비운다.
 * 다음 커널 실행 시 새로운 이름/UID가 누적될 수 있도록 준비한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::gpu_print_stat() → clear_executed_kernel_info() [이 함수]
 */
void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();  // [한국어] 실행된 커널 이름 목록 초기화
  m_executed_kernel_uids.clear();   // [한국어] 실행된 커널 UID 목록 초기화
}

/*
 * [한국어]
 * gpgpu_sim::gpu_print_stat - 커널 완료 시 전체 GPU 성능 통계를 stdout에 출력
 *
 * @streamID: 완료된 커널의 CUDA 스트림 ID.
 * @return:   void
 *
 * print_stats()의 핵심 출력 로직. 출력 항목:
 *   - 커널 이름, 스트림 ID
 *   - 사이클(gpu_sim_cycle), 명령어 수(gpu_sim_insn), IPC
 *   - 누적 사이클/명령어, 누적 IPC
 *   - 총 발행 CTA 수, 점유율(occupancy)
 *   - 스톨 카운터(dramfull, icnt2sh), 파티션 병렬성 지표
 *   - L2 대역폭(GB/s)
 *   - 시뮬레이션 속도(명령어/초)
 *   - 코어 캐시 통계(L1), L2 캐시 통계(히트율, 미스율, 예약 실패)
 *   - 셰이더 통계(m_shader_stats->print)
 *   - AccelWattch 전력 통계(POWER_MODEL 활성 시)
 *   - DRAM 파티션별 상태
 *   - ICNT 인터커넥트 패킷 카운터
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::print_stats() → gpgpu_sim::gpu_print_stat() [이 함수]
 *     → shader_print_cache_stats() → m_cluster[i]->get_cache_stats()
 *     → m_memory_partition_unit[i]->print()
 *     → m_shader_stats->print()
 */
void gpgpu_sim::gpu_print_stat(unsigned long long streamID) {
  FILE *statfout = stdout;  // [한국어] 출력 대상을 표준 출력(stdout)으로 설정

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("kernel_stream_id = %llu\n", streamID);

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.core_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.core_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, streamID,
                               "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout, streamID,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, streamID, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, streamID,
                                "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
}

/*
 * [한국어]
 * gpgpu_sim::threads_per_core - SM(셰이더 코어) 하나가 지원하는 최대 스레드 수 반환
 *
 * @return: unsigned SM 하나의 최대 동시 실행 스레드 수 (= n_thread_per_shader).
 *          예: GTX 1080 = 2048 스레드/SM
 *
 * libcuda가 cudaDeviceProp.maxThreadsPerMultiProcessor를 채울 때 사용.
 * 실행 컨텍스트: 호스트 메인 스레드.
 */
// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;  // [한국어] SM당 최대 스레드 수 반환
}

/*
 * [한국어]
 * shader_core_ctx::mem_instruction_stats - 메모리 명령어 유형별 통계를 업데이트
 *
 * @inst: 실행된 warp 명령어 (warp_inst_t). 메모리 공간(space) 정보를 포함.
 * @return: void
 *
 * 메모리 명령어가 실행될 때 해당 메모리 공간 유형에 따라 통계 카운터를 증가시킨다.
 * 분류 기준 (inst.space.get_type()):
 *   - shared_space:  gpgpu_n_shmem_insn += active_count  (공유 메모리 명령)
 *   - const_space:   gpgpu_n_const_insn   (상수 메모리 명령)
 *   - tex_space:     gpgpu_n_tex_insn     (텍스처 메모리 명령)
 *   - global/local:  gpgpu_n_store_insn 또는 gpgpu_n_load_insn (글로벌/로컬 메모리)
 *   - param_space:   gpgpu_n_param_insn   (파라미터 메모리 명령)
 *   - sstarr_space:  gpgpu_n_sstarr_insn
 * active_count는 해당 warp에서 실제로 활성인 스레드 수이다.
 * 실행 컨텍스트: 호스트 메인 스레드, SM 파이프라인 실행(execute) 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::execute() → mem_instruction_stats() [이 함수]
 */
void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();  // [한국어] 이 warp에서 활성 스레드 수
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
/*
 * [한국어]
 * shader_core_ctx::can_issue_1block - 이 SM이 주어진 커널의 CTA를 하나 더 수용할 수 있는지 확인
 *
 * @kernel: 배정하려는 커널의 정보 (스레드 수, 공유 메모리, 레지스터 수 등 포함).
 * @return: true  이 SM에 CTA를 하나 더 배정할 수 있음
 *          false SM 자원(스레드/공유메모리/레지스터/CTA 슬롯) 부족
 *
 * 동시 커널 SM 지원(-gpgpu_concurrent_kernel_sm) 여부에 따라 두 가지 경로:
 *   - 비활성(기본): 현재 활성 CTA 수가 max_cta(kernel) 미만이면 true
 *   - 활성: occupy_shader_resource_1block(kernel, false)로 자원 가용성 검사
 * 실행 컨텍스트: 호스트 메인 스레드, simt_core_cluster::issue_block2core() 경로.
 *
 * 호출 체인:
 *   simt_core_cluster::issue_block2core()
 *     → shader_core_ctx::can_issue_1block() [이 함수]
 *       → occupy_shader_resource_1block() (concurrent kernel 모드)
 */
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {  // [한국어] 동시 커널 SM 지원 모드
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

/*
 * [한국어]
 * shader_core_ctx::find_available_hwtid - CTA 크기만큼의 연속 하드웨어 스레드 ID 탐색
 *
 * @cta_size: 배정할 CTA의 스레드 수 (패딩된 warp 정렬 크기).
 * @occupy:   true이면 탐색한 hwt 슬롯을 m_occupied_hwtid에 점유 표시.
 *            false이면 가용성만 확인하고 점유하지 않음.
 * @return:   int 연속으로 사용 가능한 하드웨어 스레드 ID의 시작 번호.
 *            -1이면 충분한 연속 슬롯 없음.
 *
 * m_occupied_hwtid 비트셋(bitset)을 cta_size 단위로 스캔하여
 * 모두 비어 있는 연속 슬롯의 시작점을 찾는다.
 * occupy=true이면 해당 슬롯들을 점유 표시하여 중복 배정을 방지한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   occupy_shader_resource_1block() → find_available_hwtid() [이 함수]
 *   issue_block2core() (concurrent SM 모드) → find_available_hwtid()
 */
int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;  // [한국어] cta_size 단위로 스캔할 시작 위치
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

/*
 * [한국어]
 * shader_core_ctx::occupy_shader_resource_1block - CTA 하나에 필요한 SM 자원 가용성 확인 및 점유
 *
 * @k:      배정할 커널 정보 (스레드 수, 공유 메모리 크기, 레지스터 수 포함).
 * @occupy: true이면 자원을 실제로 점유(카운터 증가 및 비트셋 설정).
 *          false이면 가용성만 확인하고 변경 없음.
 * @return: true  CTA를 수용할 수 있는 자원이 있음
 *          false 스레드/연속HWT/공유메모리/레지스터/CTA 슬롯 중 하나라도 부족
 *
 * 검사 항목:
 *   1. m_occupied_n_threads + padded_cta_size > n_thread_per_shader → false
 *   2. find_available_hwtid(padded_cta_size, false) == -1 → false
 *   3. m_occupied_shmem + kernel_info->smem > gpgpu_shmem_size → false
 *   4. m_occupied_regs + used_regs > gpgpu_shader_registers → false
 *   5. m_occupied_ctas + 1 > max_cta_per_core → false
 * occupy=true이면 각 카운터(m_occupied_n_threads 등)를 증가시키고
 * find_available_hwtid(padded_cta_size, true)로 HWT 슬롯을 점유한다.
 * 실행 컨텍스트: 호스트 메인 스레드.
 *
 * 호출 체인:
 *   can_issue_1block() → occupy_shader_resource_1block(k, false) [이 함수]
 *   issue_block2core() (concurrent SM 모드) → occupy_shader_resource_1block(k, true)
 */
bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

/*
 * [한국어]
 * shader_core_ctx::release_shader_resource_1block - CTA 완료 시 SM 자원을 해제
 *
 * @hw_ctaid: 완료된 CTA의 하드웨어 CTA ID.
 * @k:        완료된 커널 정보.
 * @return:   void
 *
 * -gpgpu_concurrent_kernel_sm 활성 시에만 실행된다.
 * 처리 순서:
 *   1. padded_cta_size 계산 (warp 크기 정렬)
 *   2. m_occupied_n_threads -= padded_cta_size
 *   3. m_occupied_hwtid의 해당 HWT 슬롯 해제(reset)
 *   4. m_occupied_cta_to_hwtid에서 hw_ctaid 항목 삭제
 *   5. m_occupied_shmem -= kernel_info->smem
 *   6. m_occupied_regs -= used_regs
 *   7. m_occupied_ctas--
 * 각 단계마다 assert로 음수 underflow를 검증한다.
 * 실행 컨텍스트: 호스트 메인 스레드, CTA 완료 처리 경로.
 *
 * 호출 체인:
 *   shader_core_ctx::register_cta_thread_exit()
 *     → release_shader_resource_1block() [이 함수]
 */
void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

/*
 * [한국어]
 * exec_shader_core_ctx::sim_init_thread - 스레드 하나를 PTX 기능 시뮬레이션으로 초기화
 *
 * @kernel:       실행할 커널 정보.
 * @thread_info:  초기화할 ptx_thread_info 포인터의 주소 (출력 인자).
 * @sid:          셰이더 코어(SM) ID.
 * @tid:          하드웨어 스레드 ID.
 * @threads_left: 이 CTA에서 아직 초기화되지 않은 스레드 수 (경계 검사용).
 * @num_threads:  SM의 최대 스레드 수.
 * @core:         이 스레드가 실행될 코어 포인터.
 * @hw_cta_id:    하드웨어 CTA ID.
 * @hw_warp_id:   하드웨어 warp ID.
 * @gpu:          시뮬레이터 포인터.
 * @return:       unsigned 초기화에 성공한 스레드 수.
 *
 * ptx_sim_init_thread()를 호출하여 PTX 기능 시뮬레이션 컨텍스트를 생성한다.
 * 기능 시뮬레이션(functional simulation)은 타이밍 없이 PTX 명령어의
 * 결과값을 계산하는 역할을 하며, 타이밍 시뮬레이션과 함께 동작한다.
 * 실행 컨텍스트: 호스트 메인 스레드, issue_block2core() 초기화 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::issue_block2core()
 *     → exec_shader_core_ctx::sim_init_thread() [이 함수]
 *       → ptx_sim_init_thread() (cuda-sim/cuda-sim.cc)
 */
unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

/*
 * [한국어]
 * shader_core_ctx::issue_block2core - SM에 CTA(스레드 블록) 하나를 배정하고 초기화
 *
 * @kernel: 배정할 CTA가 속한 커널.
 * @return: void
 *
 * SM 하나에 CTA 하나를 배정하는 핵심 함수. 처리 순서:
 *   1. set_max_cta(kernel) 또는 occupy_shader_resource_1block() 자원 점유
 *   2. kernel.inc_running() — 커널의 실행 중 CTA 수 증가
 *   3. 빈 CTA 슬롯(m_cta_status[]) 탐색
 *   4. 하드웨어 스레드 ID 범위(start_thread ~ end_thread) 결정
 *      (concurrent kernel 모드: find_available_hwtid)
 *   5. reinit(start_thread, end_thread): 해당 warp 상태 초기화
 *   6. 각 스레드에 대해 sim_init_thread() 호출 — PTX 기능 시뮬레이션 컨텍스트 생성
 *   7. 체크포인트 재개(resume) 로직 (m_gpu->resume_option == 1 시)
 *   8. m_barriers.allocate_barrier() — CTA 배리어(__syncthreads) 자원 할당
 *   9. init_warps() — SIMT 스택 초기화 및 첫 fetch PC 설정
 *   10. m_n_active_cta++ — 활성 CTA 카운터 증가
 * 실행 컨텍스트: 호스트 메인 스레드, gpgpu_sim::issue_block2core() 경로.
 *
 * 호출 체인:
 *   gpgpu_sim::issue_block2core() → simt_core_cluster::issue_block2core()
 *     → shader_core_ctx::issue_block2core() [이 함수]
 *       → sim_init_thread()
 *       → init_warps() (shader.cc)
 */
void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);  // [한국어] 기본 모드: 커널의 최대 CTA 수를 SM에 설정
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * dram_t::dram_log - DRAM 요청 큐 길이를 샘플링하거나 통계를 출력하는 함수
 *
 * @task: SAMPLELOG(222) = 현재 큐 길이를 통계 분포에 샘플 추가.
 *        DUMPLOG(333) = 수집된 분포 통계를 stdout에 출력.
 * @return: void
 *
 * DRAM 요청 큐(mrqq_Dist) 길이 분포를 주기적으로 샘플링하여
 * DRAM 포화(saturation) 정도를 통계적으로 분석하는 데 사용된다.
 * StatAddSample/StatDisp는 stat-tool.h의 통계 유틸리티 함수이다.
 * 실행 컨텍스트: 호스트 메인 스레드, DRAM 통계 수집 경로.
 *
 * 호출 체인:
 *   memory_partition_unit::visualize() 또는 통계 샘플링 경로
 *     → dram_t::dram_log() [이 함수]
 */
void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {                      // [한국어] 샘플링 태스크: 현재 큐 길이를 분포에 추가
    StatAddSample(mrqq_Dist, que_length());      // [한국어] 큐 길이를 mrqq_Dist 통계에 추가
  } else if (task == DUMPLOG) {                 // [한국어] 덤프 태스크: 수집된 분포 출력
    printf("Queue Length DRAM[%d] ", id);       // [한국어] DRAM ID와 함께 레이블 출력
    StatDisp(mrqq_Dist);                         // [한국어] 분포 통계 화면 출력
  }
}

/*
 * [한국어]
 * gpgpu_sim::next_clock_domain - 다음에 틱할 클럭 도메인을 비트마스크로 반환
 *
 * @return: int 비트마스크 (CORE=0x01, L2=0x02, DRAM=0x04, ICNT=0x08)
 *          복수의 도메인이 동시에 틱할 수 있으므로 비트 OR 조합이 가능하다.
 *
 * 각 도메인의 누적 시간(core_time, icnt_time, dram_time, l2_time) 중
 * 가장 작은 값(smallest)을 기준으로, 그 값 이하인 모든 도메인을 이번 사이클에
 * 처리 대상으로 선택하고 해당 도메인의 period만큼 시간을 증가시킨다.
 * 예: core_period=1ns, dram_period=0.5ns 이면 DRAM이 코어보다 2배 빠르게 틱.
 * 이 비트마스크를 clock_mask에 저장하여 cycle() 내에서 각 if 블록의 조건으로 사용한다.
 * 실행 컨텍스트: 호스트 메인 스레드, cycle() 최상위에서 호출.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → gpgpu_sim::next_clock_domain() [이 함수]
 */
// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);  // [한국어] 코어/ICNT/DRAM 중 최솟값 먼저 계산
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

/*
 * [한국어]
 * gpgpu_sim::issue_block2core - 모든 SIMT 클러스터에 CTA(스레드 블록)를 배정
 *
 * @return: void
 *
 * 매 CORE 도메인 사이클마다 cycle()이 호출한다.
 * m_last_cluster_issue 다음부터 라운드로빈으로 모든 클러스터를 순회하며
 * 각 클러스터의 issue_block2core()를 호출한다.
 * 클러스터가 CTA를 받아들이면 m_last_cluster_issue를 해당 클러스터로 갱신하고
 * m_total_cta_launched에 발행 수를 누적한다.
 * 각 클러스터는 내부적으로 select_kernel()을 호출하여 실행할 커널을 선택한다.
 * 실행 컨텍스트: 호스트 메인 스레드, cycle() CORE 도메인 후단.
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() → gpgpu_sim::issue_block2core() [이 함수]
 *     → simt_core_cluster::issue_block2core() (shader.cc)
 *       → gpgpu_sim::select_kernel()
 *       → shader_core_ctx::issue_block2core()
 */
void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;  // [한국어] 이전 사이클의 마지막 발행 클러스터
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    // [한국어] 라운드로빈: last_issued 다음 클러스터부터 순환
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();  // [한국어] 해당 클러스터에 CTA 배정 시도
    if (num) {
      m_last_cluster_issue = idx;           // [한국어] 발행 성공 시 마지막 발행 클러스터 갱신
      m_total_cta_launched += num;          // [한국어] 이번 커널의 총 발행 CTA 수 누적
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline
        // [한국어] gdb에서 이 값을 설정하면 해당 사이클에서 SIGTRAP이 발생해 파이프라인 단계 추적 가능

/*
 * [한국어]
 * gpgpu_sim::cycle - GPU 전체를 한 사이클 진행시키는 핵심 시뮬레이션 루프 함수
 *
 * @return: void
 *
 * GPGPU-Sim 타이밍 시뮬레이터의 심장부(heart). stream_manager가 active()가
 * true인 동안 이 함수를 반복 호출하여 GPU 전체를 사이클-레벨로 시뮬레이션한다.
 *
 * 동작 순서 (클럭 도메인 기반):
 *   [step 1] next_clock_domain()으로 이번에 틱할 도메인(CORE/ICNT/DRAM/L2) 결정.
 *            각 도메인의 period가 다르므로 한 cycle() 호출에서 복수 도메인이 동시에 틱 가능.
 *
 *   [step 2] CORE 도메인: 각 클러스터의 icnt_cycle() 호출
 *            → ICNT에서 SM으로 도착한 메모리 응답을 파이프라인에 주입
 *
 *   [step 3] ICNT 도메인 (메모리→코어 방향):
 *            m_memory_sub_partition[i]->top()으로 응답 패킷을 조회하고
 *            icnt_has_buffer() 확인 후 icnt_push()로 ICNT에 주입.
 *            버퍼 부족 시 gpu_stall_icnt2sh++ (ICNT 백프레셔 통계)
 *
 *   [step 4] DRAM 도메인: memory_partition_unit[i]->dram_cycle() 또는
 *            simple_dram_model_cycle(). DRAM 파워 카운터 업데이트.
 *
 *   [step 5] L2 도메인:
 *            5a. ICNT에서 L2로 요청 pop (icnt_pop): SM→L2 메모리 요청 전달
 *            5b. memory_sub_partition[i]->push(): L2 큐에 요청 삽입
 *            5c. memory_sub_partition[i]->cache_cycle(): L2 캐시 처리
 *            5d. L2 power stat 누적 (AccelWattch 활성 시)
 *            5e. gpu_stall_dramfull++ if full (DRAM 백프레셔 통계)
 *
 *   [step 6] ICNT transfer(): ICNT 내부 라우팅 사이클 진행
 *
 *   [step 7] CORE 도메인 (파이프라인):
 *            m_cluster[i]->core_cycle(): 각 SM의 warp fetch/decode/execute/wb
 *            active_sms 집계, AccelWattch용 ICNT/캐시 통계 업데이트
 *            warp 슬롯 점유율(gpu_occupancy) 수집
 *
 *   [step 8] 파이프라인 활용률 계산 (average_pipeline_duty_cycle)
 *
 *   [step 9] gpu_sim_cycle++ (코어 사이클 카운터 증가)
 *
 *   [step 10] 인터랙티브 디버거 진입 (g_interactive_debugger_enabled 시)
 *
 *   [step 11] McPAT 전력 사이클 (POWER_MODEL 활성 && mode==0 시)
 *
 *   [step 12] issue_block2core(): 빈 SM에 CTA 배정
 *
 *   [step 13] decrement_kernel_latency(): TB 발행 지연 카운트다운
 *
 *   [step 14] L1/L2 캐시 플러시 (gpgpu_flush_l1/l2_cache 설정 시)
 *
 *   [step 15] 통계 샘플링 (gpu_stat_sample_freq 주기마다):
 *             비주얼라이저, 메모리 지연, 런타임 통계 출력
 *             liveness 메시지 (IPC, 점유율, 경과 시간)
 *
 *   [step 16] 데드락 검사 (50000 사이클마다 명령어 진행 없으면 gpu_deadlock=true)
 *
 *   [step 17] 스냅샷/체크포인트 및 CDP 디바이스 커널 실행 (CUDA 5.0 이상)
 *
 * 실행 컨텍스트: 호스트 메인 스레드 (단일 스레드). 재진입 불가.
 * 이 함수 자체는 락을 사용하지 않으며, 시뮬레이터 전체가 단일 스레드 모델임.
 *
 * 호출 체인:
 *   stream_manager (메인 루프) → gpgpu_sim::cycle() [이 함수]
 *     → next_clock_domain()
 *     → simt_core_cluster::icnt_cycle()      (shader.cc)
 *     → simt_core_cluster::core_cycle()      (shader.cc)
 *     → memory_partition_unit::dram_cycle()  (dram.cc)
 *     → memory_sub_partition::cache_cycle()  (l2cache.cc)
 *     → icnt_push/pop/transfer/busy          (icnt_wrapper.cc → intersim2/)
 *     → issue_block2core()
 *     → decrement_kernel_latency()
 *     → gpgpu_ctx->device_runtime->launch_one_device_kernel() (CDP)
 */
void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();  // [한국어] 이번 cycle()에서 처리할 도메인 비트마스크 결정

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for AccelWattch
      if (m_config.g_power_simulation_enabled) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
        m_cluster[i]->get_cache_stats(
            m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

    issue_block2core();
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

/*
 * [한국어]
 * sst_gpgpu_sim::cycle - SST 연동 모드의 사이클 함수 (SST_cycle() 래퍼)
 *
 * @return: void
 *
 * gpgpu_sim::cycle()을 재정의(override)하여 SST_cycle()을 호출한다.
 * SST Balar 컴포넌트가 GPU를 한 사이클 진행시킬 때 이 함수를 통해
 * GPGPU-Sim SM 파이프라인만 구동한다.
 * 메모리 시스템(DRAM/L2/ICNT)은 SST가 처리하므로 SST_cycle()에는 포함되지 않는다.
 * 실행 컨텍스트: SST 이벤트 루프 (SST Balar 컴포넌트).
 *
 * 호출 체인:
 *   SST Balar 컴포넌트 clock() → sst_gpgpu_sim::cycle() → SST_cycle() [이 함수]
 */
void sst_gpgpu_sim::cycle() {
  SST_cycle();  // [한국어] SST 모드 전용 사이클 함수 호출 (SM 파이프라인만 진행)
  return;
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

/*
 * [한국어]
 * gpgpu_sim::perf_memcpy_to_gpu - cudaMemcpy 호출 시 L2 캐시를 채우는 성능 시뮬레이션
 *
 * @dst_start_addr: GPU 메모리 상의 복사 대상 시작 주소 (바이트 단위).
 * @count:          복사할 데이터 크기 (바이트 단위).
 * @return:         void
 *
 * -gpgpu_perf_sim_memcpy 1이 설정된 경우, cudaMemcpy(Host→GPU)로 전송된 데이터가
 * L2 캐시에 적재된 것처럼 시뮬레이션한다.
 * 32바이트 단위로 순회하며 각 주소에 대해 addrdec_tlx()로 DRAM 파티션을 계산하고
 * memory_partition_unit::handle_memcpy_to_gpu()를 호출해 L2에 해당 캐시 라인을 채운다.
 * 이를 통해 memcpy 이후 GPU 커널이 L2 히트를 경험하는 현실적인 시뮬레이션이 가능해진다.
 * 실행 컨텍스트: 호스트 메인 스레드, cuMemcpy/cudaMemcpy 인터셉트 경로.
 *
 * 호출 체인:
 *   libcuda: cudaMemcpy 인터셉트 → gpgpu_sim::perf_memcpy_to_gpu() [이 함수]
 *     → m_memory_config->m_address_mapping.addrdec_tlx() (addrdec.cc)
 *     → m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu() (dram.cc)
 */
void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {  // [한국어] memcpy 성능 시뮬레이션 활성화된 경우에만 처리
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

/*
 * [한국어]
 * gpgpu_sim::dump_pipeline - 파이프라인 상태를 stdout에 출력하는 디버깅 함수
 *
 * @mask: 출력할 정보를 선택하는 비트마스크.
 *        0이면 0xFFFFFFFF(모든 정보) 사용.
 *        비트 0x1: SM 파이프라인 상태
 *        비트 0x10000: DRAM/메모리 컨트롤러 상태
 *        비트 0x100000: DRAM 통계
 *        비트 0x1000000: DRAM 시각화
 *        비트 0x10000000: DRAM 전체 상태
 * @s:   출력할 특정 셰이더 코어 번호. -1이면 모든 코어.
 * @m:   출력할 특정 메모리 파티션 번호. -1이면 모든 파티션.
 * @return: void
 *
 * GDB에서 "dp 3" 명령으로 셰이더 코어 3의 파이프라인을 볼 때 사용한다.
 * .gdbinit에 "define dp / call g_the_gpu.dump_pipeline_impl(...) / end" 설정.
 * 실행 컨텍스트: 디버깅 세션 (GDB 대화형 모드).
 *
 * 호출 체인:
 *   GDB 사용자 명령 → gpgpu_sim::dump_pipeline() [이 함수]
 *     → m_cluster[i]->display_pipeline() (shader.cc)
 *     → m_memory_partition_unit[i]->print/visualize() (dram.cc)
 */
void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

/*
 * [한국어]
 * gpgpu_sim::getShaderCoreConfig - 셰이더 코어 설정 포인터 반환
 *
 * @return: const shader_core_config* SM 파이프라인 설정 객체 포인터.
 *
 * AccelWattch mcpat_cycle/calculate_hw_mcpat()에서 warp 크기, 실행 유닛 수 등
 * 전력 모델 계산에 필요한 SM 설정을 조회할 때 사용한다.
 */
const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;  // [한국어] SM 설정 포인터 반환
}

/*
 * [한국어]
 * gpgpu_sim::getMemoryConfig - 메모리 설정 포인터 반환
 *
 * @return: const memory_config* 메모리 서브시스템 설정 객체 포인터.
 *
 * 외부 모듈(AccelWattch, libcuda 속성 조회 등)이 DRAM 구성(파티션 수, 뱅크 수 등)을
 * 읽어야 할 때 사용한다.
 */
const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }  // [한국어] 메모리 설정 포인터 반환

/*
 * [한국어]
 * gpgpu_sim::getSIMTCluster - 첫 번째 SIMT 클러스터 포인터 반환
 *
 * @return: simt_core_cluster* m_cluster[0] (첫 번째 클러스터) 포인터.
 *
 * 기능 시뮬레이션(functional simulation)에서 SM 내부 상태에 접근할 때 사용한다.
 * 주의: m_cluster는 배열이므로 *m_cluster는 m_cluster[0]를 역참조한 것이다.
 */
simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }  // [한국어] m_cluster[0] 포인터 반환

/*
 * [한국어]
 * sst_gpgpu_sim::SST_gpgpusim_numcores_equal_check - SST와 GPGPU-Sim의 코어 수가 일치하는지 검증
 *
 * @sst_numcores: SST Balar 컴포넌트에서 설정한 GPU 코어 수.
 * @return: void (불일치 시 assert로 abort)
 *
 * SST Balar와 gpgpusim.config의 -gpgpu_n_clusters 값이 다르면
 * 두 시뮬레이터가 서로 다른 토폴로지를 가정하게 되어 결과가 올바르지 않게 된다.
 * 이를 방지하기 위해 시뮬레이션 시작 시 한 번 호출하여 일치 여부를 검증한다.
 * 불일치 시 assert 메시지로 설정 파일 수정 방법을 안내하고 프로세스를 종료한다.
 * 실행 컨텍스트: SST 초기화 단계.
 *
 * 호출 체인:
 *   SST Balar setup() → SST_gpgpusim_numcores_equal_check() [이 함수]
 */
void sst_gpgpu_sim::SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  if (m_shader_config->n_simt_clusters != sst_numcores) {  // [한국어] 코어 수 불일치 시 abort
    assert(
        "\nSST core is not equal the GPGPU-sim cores. Open gpgpu-sim.config "
        "file and ensure n_simt_clusters"
        "is the same as SST gpu cores.\n" &&
        0);
  } else {
    printf("\nSST GPU core is equal the GPGPU-sim cores = %d\n", sst_numcores);
  }
}

/*
 * [한국어]
 * sst_gpgpu_sim::SST_cycle - SST 모드에서 SM 파이프라인만 한 사이클 진행
 *
 * @return: void
 *
 * gpgpu_sim::cycle()의 SST 버전. 메모리 시스템(DRAM, L2, ICNT)은 SST가
 * 담당하므로 SM 파이프라인 관련 처리만 수행한다.
 * 처리 순서:
 *   1. sst_simt_core_cluster::icnt_cycle_SST(): SST 응답 버퍼에서
 *      SM으로 메모리 응답 주입 (표준 icnt_cycle 대체)
 *   2. 각 클러스터의 core_cycle(): SM warp 파이프라인 진행
 *   3. active_sms 및 파이프라인 활용률 통계 수집
 *   4. gpu_sim_cycle++ (코어 사이클 카운터)
 *   5. McPAT 전력 사이클 (POWER_MODEL 활성 시)
 *   6. issue_block2core(): 빈 SM에 CTA 배정
 *   7. 통계 샘플링 (비주얼라이저, 런타임 통계)
 *   8. 데드락 검사 (20000 사이클 주기, 표준보다 짧음)
 *   9. CDP 디바이스 커널 실행
 * 실행 컨텍스트: SST 이벤트 루프.
 *
 * 호출 체인:
 *   sst_gpgpu_sim::cycle() → sst_gpgpu_sim::SST_cycle() [이 함수]
 *     → sst_simt_core_cluster::icnt_cycle_SST() (shader.cc)
 *     → m_cluster[i]->core_cycle()
 *     → issue_block2core()
 */
void sst_gpgpu_sim::SST_cycle() {
  // shader core loading (pop from ICNT into core) follows CORE clock
  // [한국어] SST 응답 버퍼에서 각 SM으로 메모리 응답을 주입 (ICNT 대신 SST 버퍼 사용)
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    static_cast<sst_simt_core_cluster *>(m_cluster[i])->icnt_cycle_SST();

  // L1 cache + shader core pipeline stages
  m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
      *active_sms += m_cluster[i]->get_n_active_sms();
    }
    // Update core icnt/cache stats for GPUWattch
    m_cluster[i]->get_icnt_stats(
        m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
        m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
    m_cluster[i]->get_cache_stats(
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
  }
  float temp = 0;
  for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
    temp += m_shader_stats->m_pipeline_duty_cycle[i];
  }
  temp = temp / m_shader_config->num_shader();
  *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
  // cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;

  if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
    asm("int $03");
  }
  gpu_sim_cycle++;
  if (g_interactive_debugger_enabled) gpgpu_debug();

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle,
                gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn,
                m_config.g_dvfs_enabled);
  }
#endif

  issue_block2core();

  if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
    time_t days, hrs, minutes, sec;
    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
    if ((elapsed_time - last_liveness_message_time) >=
        m_config.liveness_message_freq) {
      days = elapsed_time / (3600 * 24);
      hrs = elapsed_time / 3600 - 24 * days;
      minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
      sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

      last_liveness_message_time = elapsed_time;
    }
    visualizer_printstat();
    m_memory_stats->memlatstat_lat_pw();
    if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
          m_memory_partition_unit[i]->print_stat(stdout);
        printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
        printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
      }
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
        shader_print_runtime_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
        shader_print_l1_miss_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
        shader_print_scheduler_stat(stdout, false);
    }
  }

  if (!(gpu_sim_cycle % 20000)) {
    // deadlock detection
    if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
      gpu_deadlock = true;
    } else {
      last_gpu_sim_insn = gpu_sim_insn;
    }
  }
  try_snap_shot(gpu_sim_cycle);
  spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
  // launch device kernel
  gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
}
