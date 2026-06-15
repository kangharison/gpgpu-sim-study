/*****************************************************************************
 *                                McPAT
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/
/********************************************************************
 *      Modified by:
 * Jingwen Leng, University of Texas, Austin
 * Syed Gilani, University of Wisconsin–Madison
 * Tayler Hetherington, University of British Columbia
 * Ahmed ElTantawy, University of British Columbia
 * Vijay Kandiah, Northwestern University
 ********************************************************************/

#ifndef XML_PARSE_H_
#define XML_PARSE_H_

//#ifdef WIN32
//#define _CRT_SECURE_NO_DEPRECATE
//#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include "xmlParser.h"
using namespace std;

/*
 * [한국어 설명] McPAT/AccelWattch XML 설정 구조체 헤더 (XML_Parse.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 AccelWattch 전력 모델이 사용하는 XML 설정 파일을 파싱하여
 * C 구조체(root_system)로 매핑하는 인터페이스를 정의한다.
 * gpgpusim.config의 -accelwattch_xml_file 옵션으로 지정된 XML을 읽어
 * GPU 아키텍처, 코어/L2/NoC/MC 파라미터, 스케일링 계수, 성능 카운터 등을 저장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * XML_Parse.cc::ParseXML::parse()
 * → xmlParser.h/cc의 XMLNode 트리 파싱
 * → root_system sys 필드 채움
 * → processor.h의 Processor 생성자/compute()가 sys를 참조
 *
 * === 타 모듈과의 연결 ===
 * 의존: xmlParser.h/xmlParser.cc (XML 노드 파싱 라이브러리)
 * 사용: gpgpu_sim_wrapper.h/.cc (활동 카운터 주입), processor.h (McPAT 전력 계산)
 *
 * === 주요 함수/구조체 요약 ===
 * ParseXML 클래스 — XML 파일 파싱 및 초기화
 * perf_count_t enum — GPGPU-Sim ↔ AccelWattch 성능 카운터 식별자
 * system_core/system_L2/system_NoC/system_mc 등 — GPU 하드웨어 서브시스템 구조체
 * root_system — 최상위 XML 설정 집계체
 */

/*
void myfree(char *t); // {free(t);}
ToXMLStringTool tx,tx2;
*/
// all subnodes at the level of system.core(0-n)
// cache_policy is added into cache property arrays;//0 no write or write-though
// with non-write allocate;1 write-back with write-allocate
//
// tgrogers - This was a static array declared in the header...
//           Not too sure why the authors did this, maybe they didn't understand
//           the context of the "static" keyword outside a class declaration. As
//           it was written, each object file had it's own copy of the string
//           list - which is okay, since the string is constant but wastes space
//           and causes the compiler to complain about unused vars in files
//           where this header is included but they don't use the variable.  Now
//           this is extern'd here and the storage/definition is in the
//           XML_Parse.cc file
extern const char* perf_count_label[];

enum perf_count_t {
  TOT_INST = 0,
  FP_INT,
  IC_H,
  IC_M,
  DC_RH,
  DC_RM,
  DC_WH,
  DC_WM,
  TC_H,
  TC_M,
  CC_H,
  CC_M,
  SHRD_ACC,
  REG_RD,
  REG_WR,
  NON_REG_OPs,
  INT_ACC,        // SPU
  FP_ACC,         // FPU
  DP_ACC,         // FPU
  INT_MUL24_ACC,  // SFU
  INT_MUL32_ACC,  // SFU
  INT_MUL_ACC,    // SFU
  INT_DIV_ACC,    // SFU
  FP_MUL_ACC,     // SFU
  FP_DIV_ACC,     // SFU
  FP_SQRT_ACC,    // SFU
  FP_LG_ACC,      // SFU
  FP_SIN_ACC,     // SFU
  FP_EXP_ACC,     // SFU
  DP_MUL_ACC,     // SFU
  DP_DIV_ACC,     // SFU
  TENSOR_ACC,     // SFU
  TEX_ACC,        // SFU
  MEM_RD,
  MEM_WR,
  MEM_PRE,
  L2_RH,
  L2_RM,
  L2_WH,
  L2_WM,
  NOC_A,
  PIPE_A,
  IDLE_CORE_N,
  constant_power,
  NUM_PERFORMANCE_COUNTERS
};

/*
 * [한국어] 분기 예측기 구조체 (GPU에서는 주로 미사용)
 */
typedef struct {
  int prediction_width;  // [한국어] prediction_width 필드
  char prediction_scheme[20];  // [한국어] prediction_scheme 필드
  int predictor_size;  // [한국어] predictor_size 필드
  int predictor_entries;  // [한국어] predictor_entries 필드
  int local_predictor_size[20];  // [한국어] local_predictor_size 필드
  int local_predictor_entries;  // [한국어] local_predictor_entries 필드
  int global_predictor_entries;  // [한국어] global_predictor_entries 필드
  int global_predictor_bits;  // [한국어] global_predictor_bits 필드
  int chooser_predictor_entries;  // [한국어] chooser_predictor_entries 필드
  int chooser_predictor_bits;  // [한국어] chooser_predictor_bits 필드
  double predictor_accesses;  // [한국어] predictor_accesses 필드
} predictor_systemcore;
/*
 * [한국어] 명령어 TLB 구조체
 */
typedef struct {
  int number_entries;  // [한국어] number_entries 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  double total_hits;  // [한국어] total_hits 필드
  double total_accesses;  // [한국어] total_accesses 필드
  double total_misses;  // [한국어] total_misses 필드
  double conflicts;  // [한국어] conflicts 필드
} itlb_systemcore;
/*
 * [한국어] 명령어 캐시(L1I) 파라미터 및 통계 구조체
 */
typedef struct {
  // params
  double icache_config[20];  // [한국어] icache_config 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double read_misses;  // [한국어] read_misses 필드
  double replacements;  // [한국어] replacements 필드
  double read_hits;  // [한국어] read_hits 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double miss_buffer_access;  // [한국어] miss_buffer_access 필드
  double fill_buffer_accesses;  // [한국어] fill_buffer_accesses 필드
  double prefetch_buffer_accesses;  // [한국어] prefetch_buffer_accesses 필드
  double prefetch_buffer_writes;  // [한국어] prefetch_buffer_writes 필드
  double prefetch_buffer_reads;  // [한국어] prefetch_buffer_reads 필드
  double prefetch_buffer_hits;  // [한국어] prefetch_buffer_hits 필드
  double conflicts;  // [한국어] conflicts 필드
} icache_systemcore;
/*
 * [한국어] 데이터 TLB 구조체
 */
typedef struct {
  // params
  int number_entries;  // [한국어] number_entries 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double write_hits;  // [한국어] write_hits 필드
  double read_hits;  // [한국어] read_hits 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double conflicts;  // [한국어] conflicts 필드
} dtlb_systemcore;
/*
 * [한국어] 데이터 캐시(L1D) 파라미터 및 통계 구조체
(L1D, 상수 캐시, 텍스처 캐시, 공유 메모리 모두 이 타입 재사용)
 */
typedef struct {
  // params
  double dcache_config[20];  // [한국어] dcache_config 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double read_hits;  // [한국어] read_hits 필드
  double write_hits;  // [한국어] write_hits 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double replacements;  // [한국어] replacements 필드
  double write_backs;  // [한국어] write_backs 필드
  double miss_buffer_access;  // [한국어] miss_buffer_access 필드
  double fill_buffer_accesses;  // [한국어] fill_buffer_accesses 필드
  double prefetch_buffer_accesses;  // [한국어] prefetch_buffer_accesses 필드
  double prefetch_buffer_writes;  // [한국어] prefetch_buffer_writes 필드
  double prefetch_buffer_reads;  // [한국어] prefetch_buffer_reads 필드
  double prefetch_buffer_hits;  // [한국어] prefetch_buffer_hits 필드
  double wbb_writes;  // [한국어] wbb_writes 필드
  double wbb_reads;  // [한국어] wbb_reads 필드
  double conflicts;  // [한국어] conflicts 필드
} dcache_systemcore;
/*
 * [한국어] BTB(Branch Target Buffer) 구조체
 */
typedef struct {
  // params
  int BTB_config[20];  // [한국어] BTB_config 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double read_hits;  // [한국어] read_hits 필드
  double write_hits;  // [한국어] write_hits 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double replacements;  // [한국어] replacements 필드
} BTB_systemcore;
/*
 * [한국어] SM(Streaming Multiprocessor) 하나의 전체 파라미터/통계 구조체
 */
typedef struct {
  // all params at the level of system.core(0-n)
  int clock_rate;  // [한국어] clock_rate 필드
  bool opt_local;  // [한국어] opt_local 필드
  bool x86;  // [한국어] x86 필드
  int machine_bits;  // [한국어] machine_bits 필드
  int virtual_address_width;  // [한국어] virtual_address_width 필드
  int physical_address_width;  // [한국어] physical_address_width 필드
  int opcode_width;  // [한국어] opcode_width 필드
  int micro_opcode_width;  // [한국어] micro_opcode_width 필드
  int instruction_length;  // [한국어] instruction_length 필드
  int machine_type;  // [한국어] machine_type 필드
  int internal_datapath_width;  // [한국어] internal_datapath_width 필드
  int number_hardware_threads;  // [한국어] number_hardware_threads 필드
  int fetch_width;  // [한국어] fetch_width 필드
  int number_instruction_fetch_ports;  // [한국어] number_instruction_fetch_ports 필드
  int decode_width;  // [한국어] decode_width 필드
  int issue_width;  // [한국어] issue_width 필드
  int peak_issue_width;  // [한국어] peak_issue_width 필드
  int commit_width;  // [한국어] commit_width 필드
  int pipelines_per_core[20];  // [한국어] pipelines_per_core 필드
  int pipeline_depth[20];  // [한국어] pipeline_depth 필드
  char FPU[20];  // [한국어] FPU 필드
  char divider_multiplier[20];  // [한국어] divider_multiplier 필드
  int ALU_per_core;  // [한국어] ALU_per_core 필드
  double FPU_per_core;  // [한국어] FPU_per_core 필드
  int MUL_per_core;  // [한국어] MUL_per_core 필드
  int instruction_buffer_size;  // [한국어] instruction_buffer_size 필드
  int decoded_stream_buffer_size;  // [한국어] decoded_stream_buffer_size 필드
  int instruction_window_scheme;  // [한국어] instruction_window_scheme 필드
  int instruction_window_size;  // [한국어] instruction_window_size 필드
  int fp_instruction_window_size;  // [한국어] fp_instruction_window_size 필드
  int ROB_size;  // [한국어] ROB_size 필드
  int archi_Regs_IRF_size;  // [한국어] archi_Regs_IRF_size 필드
  int archi_Regs_FRF_size;  // [한국어] archi_Regs_FRF_size 필드
  int phy_Regs_IRF_size;  // [한국어] phy_Regs_IRF_size 필드
  int phy_Regs_FRF_size;  // [한국어] phy_Regs_FRF_size 필드
  int rename_scheme;  // [한국어] rename_scheme 필드
  int register_windows_size;  // [한국어] register_windows_size 필드
  char LSU_order[20];  // [한국어] LSU_order 필드
  int store_buffer_size;  // [한국어] store_buffer_size 필드
  int load_buffer_size;  // [한국어] load_buffer_size 필드
  int memory_ports;  // [한국어] memory_ports 필드
  char Dcache_dual_pump[20];  // [한국어] Dcache_dual_pump 필드
  int RAS_size;  // [한국어] RAS_size 필드
  int fp_issue_width;  // [한국어] fp_issue_width 필드
  int prediction_width;  // [한국어] prediction_width 필드
  int number_of_BTB;  // [한국어] number_of_BTB 필드
  int number_of_BPT;  // [한국어] number_of_BPT 필드
  bool gpgpu_clock_gated_lanes;  // [한국어] gpgpu_clock_gated_lanes 필드

  // all stats at the level of system.core(0-n)
  double total_instructions;  // [한국어] total_instructions 필드
  double int_instructions;  // [한국어] int_instructions 필드
  double fp_instructions;  // [한국어] fp_instructions 필드
  double branch_instructions;  // [한국어] branch_instructions 필드
  double branch_mispredictions;  // [한국어] branch_mispredictions 필드
  double committed_instructions;  // [한국어] committed_instructions 필드
  double committed_int_instructions;  // [한국어] committed_int_instructions 필드
  double committed_fp_instructions;  // [한국어] committed_fp_instructions 필드
  double load_instructions;  // [한국어] load_instructions 필드
  double store_instructions;  // [한국어] store_instructions 필드
  double total_cycles;  // [한국어] total_cycles 필드
  double idle_cycles;  // [한국어] idle_cycles 필드
  double busy_cycles;  // [한국어] busy_cycles 필드
  double instruction_buffer_reads;  // [한국어] instruction_buffer_reads 필드
  double instruction_buffer_write;  // [한국어] instruction_buffer_write 필드
  double ROB_reads;  // [한국어] ROB_reads 필드
  double ROB_writes;  // [한국어] ROB_writes 필드
  double rename_accesses;  // [한국어] rename_accesses 필드
  double fp_rename_accesses;  // [한국어] fp_rename_accesses 필드
  double rename_reads;  // [한국어] rename_reads 필드
  double rename_writes;  // [한국어] rename_writes 필드
  double fp_rename_reads;  // [한국어] fp_rename_reads 필드
  double fp_rename_writes;  // [한국어] fp_rename_writes 필드
  double inst_window_reads;  // [한국어] inst_window_reads 필드
  double inst_window_writes;  // [한국어] inst_window_writes 필드
  double inst_window_wakeup_accesses;  // [한국어] inst_window_wakeup_accesses 필드
  double inst_window_selections;  // [한국어] inst_window_selections 필드
  double fp_inst_window_reads;  // [한국어] fp_inst_window_reads 필드
  double fp_inst_window_writes;  // [한국어] fp_inst_window_writes 필드
  double fp_inst_window_wakeup_accesses;  // [한국어] fp_inst_window_wakeup_accesses 필드
  double fp_inst_window_selections;  // [한국어] fp_inst_window_selections 필드
  double archi_int_regfile_reads;  // [한국어] archi_int_regfile_reads 필드
  double archi_float_regfile_reads;  // [한국어] archi_float_regfile_reads 필드
  double phy_int_regfile_reads;  // [한국어] phy_int_regfile_reads 필드
  double phy_float_regfile_reads;  // [한국어] phy_float_regfile_reads 필드
  double phy_int_regfile_writes;  // [한국어] phy_int_regfile_writes 필드
  double phy_float_regfile_writes;  // [한국어] phy_float_regfile_writes 필드
  double archi_int_regfile_writes;  // [한국어] archi_int_regfile_writes 필드
  double archi_float_regfile_writes;  // [한국어] archi_float_regfile_writes 필드
  double int_regfile_reads;  // [한국어] int_regfile_reads 필드
  double float_regfile_reads;  // [한국어] float_regfile_reads 필드
  double int_regfile_writes;  // [한국어] int_regfile_writes 필드
  double float_regfile_writes;  // [한국어] float_regfile_writes 필드
  double non_rf_operands;  // [한국어] non_rf_operands 필드
  double windowed_reg_accesses;  // [한국어] windowed_reg_accesses 필드
  double windowed_reg_transports;  // [한국어] windowed_reg_transports 필드
  double function_calls;  // [한국어] function_calls 필드
  double context_switches;  // [한국어] context_switches 필드
  double ialu_accesses;  // [한국어] ialu_accesses 필드
  double fpu_accesses;  // [한국어] fpu_accesses 필드
  double mul_accesses;  // [한국어] mul_accesses 필드
  double sp_average_active_lanes;  // [한국어] sp_average_active_lanes 필드
  double sfu_average_active_lanes;  // [한국어] sfu_average_active_lanes 필드
  double cdb_alu_accesses;  // [한국어] cdb_alu_accesses 필드
  double cdb_mul_accesses;  // [한국어] cdb_mul_accesses 필드
  double cdb_fpu_accesses;  // [한국어] cdb_fpu_accesses 필드
  double load_buffer_reads;  // [한국어] load_buffer_reads 필드
  double load_buffer_writes;  // [한국어] load_buffer_writes 필드
  double load_buffer_cams;  // [한국어] load_buffer_cams 필드
  double store_buffer_reads;  // [한국어] store_buffer_reads 필드
  double store_buffer_writes;  // [한국어] store_buffer_writes 필드
  double store_buffer_cams;  // [한국어] store_buffer_cams 필드
  double store_buffer_forwards;  // [한국어] store_buffer_forwards 필드
  double main_memory_access;  // [한국어] main_memory_access 필드
  double main_memory_read;  // [한국어] main_memory_read 필드
  double main_memory_write;  // [한국어] main_memory_write 필드
  double pipeline_duty_cycle;  // [한국어] pipeline_duty_cycle 필드

  double IFU_duty_cycle;  // [한국어] IFU_duty_cycle 필드
  double BR_duty_cycle;  // [한국어] BR_duty_cycle 필드
  double LSU_duty_cycle;  // [한국어] LSU_duty_cycle 필드
  double MemManU_I_duty_cycle;  // [한국어] MemManU_I_duty_cycle 필드
  double MemManU_D_duty_cycle;  // [한국어] MemManU_D_duty_cycle 필드
  double ALU_duty_cycle;  // [한국어] ALU_duty_cycle 필드
  double MUL_duty_cycle;  // [한국어] MUL_duty_cycle 필드
  double FPU_duty_cycle;  // [한국어] FPU_duty_cycle 필드
  double ALU_cdb_duty_cycle;  // [한국어] ALU_cdb_duty_cycle 필드
  double MUL_cdb_duty_cycle;  // [한국어] MUL_cdb_duty_cycle 필드
  double FPU_cdb_duty_cycle;  // [한국어] FPU_cdb_duty_cycle 필드

  double num_idle_cores;  // [한국어] num_idle_cores 필드

  int rf_banks;             // (4)
  int simd_width;           // (8)
  int collector_units;      // (4)
  double core_clock_ratio;  // (2.0)
  int warp_size;            // (32)

  // all subnodes at the level of system.core(0-n)
  predictor_systemcore predictor;  // [한국어] predictor 필드
  itlb_systemcore itlb;  // [한국어] itlb 필드
  icache_systemcore icache;  // [한국어] icache 필드
  dtlb_systemcore dtlb;  // [한국어] dtlb 필드
  dcache_systemcore dcache;  // [한국어] dcache 필드
  dcache_systemcore ccache;  // [한국어] ccache 필드
  dcache_systemcore tcache;  // [한국어] tcache 필드
  dcache_systemcore sharedmemory;  // added by Jingwen
  BTB_systemcore BTB;  // [한국어] BTB 필드

} system_core;
/*
 * [한국어] L1 디렉터리 캐시 구조체 (CPU 코히런스용, GPU에서는 미사용)
 */
typedef struct {
  // params
  int Directory_type;  // [한국어] Directory_type 필드
  double Dir_config[20];  // [한국어] Dir_config 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  int clockrate;  // [한국어] clockrate 필드
  int ports[20];  // [한국어] ports 필드
  int device_type;  // [한국어] device_type 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  char threeD_stack[20];  // [한국어] threeD_stack 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double conflicts;  // [한국어] conflicts 필드
  double duty_cycle;  // [한국어] duty_cycle 필드
} system_L1Directory;
/*
 * [한국어] L2 디렉터리 캐시 구조체 (CPU 코히런스용, GPU에서는 미사용)
 */
typedef struct {
  // params
  int Directory_type;  // [한국어] Directory_type 필드
  double Dir_config[20];  // [한국어] Dir_config 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  int clockrate;  // [한국어] clockrate 필드
  int ports[20];  // [한국어] ports 필드
  int device_type;  // [한국어] device_type 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  char threeD_stack[20];  // [한국어] threeD_stack 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double conflicts;  // [한국어] conflicts 필드
  double duty_cycle;  // [한국어] duty_cycle 필드
} system_L2Directory;
/*
 * [한국어] L2 캐시 파라미터 및 통계 구조체
 */
typedef struct {
  // params
  double L2_config[20];  // [한국어] L2_config 필드
  int clockrate;  // [한국어] clockrate 필드
  int ports[20];  // [한국어] ports 필드
  int device_type;  // [한국어] device_type 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  char threeD_stack[20];  // [한국어] threeD_stack 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double read_hits;  // [한국어] read_hits 필드
  double write_hits;  // [한국어] write_hits 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double replacements;  // [한국어] replacements 필드
  double write_backs;  // [한국어] write_backs 필드
  double miss_buffer_accesses;  // [한국어] miss_buffer_accesses 필드
  double fill_buffer_accesses;  // [한국어] fill_buffer_accesses 필드
  double prefetch_buffer_accesses;  // [한국어] prefetch_buffer_accesses 필드
  double prefetch_buffer_writes;  // [한국어] prefetch_buffer_writes 필드
  double prefetch_buffer_reads;  // [한국어] prefetch_buffer_reads 필드
  double prefetch_buffer_hits;  // [한국어] prefetch_buffer_hits 필드
  double wbb_writes;  // [한국어] wbb_writes 필드
  double wbb_reads;  // [한국어] wbb_reads 필드
  double conflicts;  // [한국어] conflicts 필드
  double duty_cycle;  // [한국어] duty_cycle 필드

  bool merged_dir;  // [한국어] merged_dir 필드
  double homenode_read_accesses;  // [한국어] homenode_read_accesses 필드
  double homenode_write_accesses;  // [한국어] homenode_write_accesses 필드
  double homenode_read_hits;  // [한국어] homenode_read_hits 필드
  double homenode_write_hits;  // [한국어] homenode_write_hits 필드
  double homenode_read_misses;  // [한국어] homenode_read_misses 필드
  double homenode_write_misses;  // [한국어] homenode_write_misses 필드
  double dir_duty_cycle;  // [한국어] dir_duty_cycle 필드
} system_L2;
/*
 * [한국어] L3 캐시 구조체 (GPU에서는 미사용)
 */
typedef struct {
  // params
  double L3_config[20];  // [한국어] L3_config 필드
  int clockrate;  // [한국어] clockrate 필드
  int ports[20];  // [한국어] ports 필드
  int device_type;  // [한국어] device_type 필드
  int cache_policy;  // 0 no write or write-though with non-write allocate;1
                     // write-back with write-allocate
  char threeD_stack[20];  // [한국어] threeD_stack 필드
  int buffer_sizes[20];  // [한국어] buffer_sizes 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double read_accesses;  // [한국어] read_accesses 필드
  double write_accesses;  // [한국어] write_accesses 필드
  double total_hits;  // [한국어] total_hits 필드
  double total_misses;  // [한국어] total_misses 필드
  double read_hits;  // [한국어] read_hits 필드
  double write_hits;  // [한국어] write_hits 필드
  double read_misses;  // [한국어] read_misses 필드
  double write_misses;  // [한국어] write_misses 필드
  double replacements;  // [한국어] replacements 필드
  double write_backs;  // [한국어] write_backs 필드
  double miss_buffer_accesses;  // [한국어] miss_buffer_accesses 필드
  double fill_buffer_accesses;  // [한국어] fill_buffer_accesses 필드
  double prefetch_buffer_accesses;  // [한국어] prefetch_buffer_accesses 필드
  double prefetch_buffer_writes;  // [한국어] prefetch_buffer_writes 필드
  double prefetch_buffer_reads;  // [한국어] prefetch_buffer_reads 필드
  double prefetch_buffer_hits;  // [한국어] prefetch_buffer_hits 필드
  double wbb_writes;  // [한국어] wbb_writes 필드
  double wbb_reads;  // [한국어] wbb_reads 필드
  double conflicts;  // [한국어] conflicts 필드
  double duty_cycle;  // [한국어] duty_cycle 필드

  bool merged_dir;  // [한국어] merged_dir 필드
  double homenode_read_accesses;  // [한국어] homenode_read_accesses 필드
  double homenode_write_accesses;  // [한국어] homenode_write_accesses 필드
  double homenode_read_hits;  // [한국어] homenode_read_hits 필드
  double homenode_write_hits;  // [한국어] homenode_write_hits 필드
  double homenode_read_misses;  // [한국어] homenode_read_misses 필드
  double homenode_write_misses;  // [한국어] homenode_write_misses 필드
  double dir_duty_cycle;  // [한국어] dir_duty_cycle 필드
} system_L3;
/*
 * [한국어] NoC 낮부 크로스바 구조체
 */
typedef struct {
  // params
  int number_of_inputs_of_crossbars;  // [한국어] number_of_inputs_of_crossbars 필드
  int number_of_outputs_of_crossbars;  // [한국어] number_of_outputs_of_crossbars 필드
  int flit_bits;  // [한국어] flit_bits 필드
  int input_buffer_entries_per_port;  // [한국어] input_buffer_entries_per_port 필드
  int ports_of_input_buffer[20];  // [한국어] ports_of_input_buffer 필드
  // stats
  double crossbar_accesses;  // [한국어] crossbar_accesses 필드
} xbar0_systemNoC;
/*
 * [한국어] Network-on-Chip 파라미터 및 통계 구조체
 */
typedef struct {
  // params
  int clockrate;  // [한국어] clockrate 필드
  bool type;  // [한국어] type 필드
  bool has_global_link;  // [한국어] has_global_link 필드
  char topology[20];  // [한국어] topology 필드
  int horizontal_nodes;  // [한국어] horizontal_nodes 필드
  int vertical_nodes;  // [한국어] vertical_nodes 필드
  int link_throughput;  // [한국어] link_throughput 필드
  int link_latency;  // [한국어] link_latency 필드
  int input_ports;  // [한국어] input_ports 필드
  int output_ports;  // [한국어] output_ports 필드
  int virtual_channel_per_port;  // [한국어] virtual_channel_per_port 필드
  int flit_bits;  // [한국어] flit_bits 필드
  int input_buffer_entries_per_vc;  // [한국어] input_buffer_entries_per_vc 필드
  int ports_of_input_buffer[20];  // [한국어] ports_of_input_buffer 필드
  int dual_pump;  // [한국어] dual_pump 필드
  int number_of_crossbars;  // [한국어] number_of_crossbars 필드
  char crossbar_type[20];  // [한국어] crossbar_type 필드
  char crosspoint_type[20];  // [한국어] crosspoint_type 필드
  xbar0_systemNoC xbar0;  // [한국어] xbar0 필드
  int arbiter_type;  // [한국어] arbiter_type 필드
  double chip_coverage;  // [한국어] chip_coverage 필드
  // stats
  double total_accesses;  // [한국어] total_accesses 필드
  double duty_cycle;  // [한국어] duty_cycle 필드
  double route_over_perc;  // [한국어] route_over_perc 필드
} system_NoC;
/*
 * [한국어] DRAM 메모리 파라미터 및 통계 구조체
 */
typedef struct {
  // params
  int mem_tech_node;  // [한국어] mem_tech_node 필드
  int device_clock;  // [한국어] device_clock 필드
  int peak_transfer_rate;  // [한국어] peak_transfer_rate 필드
  int internal_prefetch_of_DRAM_chip;  // [한국어] internal_prefetch_of_DRAM_chip 필드
  int capacity_per_channel;  // [한국어] capacity_per_channel 필드
  int number_ranks;  // [한국어] number_ranks 필드
  int num_banks_of_DRAM_chip;  // [한국어] num_banks_of_DRAM_chip 필드
  int Block_width_of_DRAM_chip;  // [한국어] Block_width_of_DRAM_chip 필드
  int output_width_of_DRAM_chip;  // [한국어] output_width_of_DRAM_chip 필드
  int page_size_of_DRAM_chip;  // [한국어] page_size_of_DRAM_chip 필드
  int burstlength_of_DRAM_chip;  // [한국어] burstlength_of_DRAM_chip 필드

  // stats
  double memory_accesses;  // [한국어] memory_accesses 필드
  double memory_reads;  // [한국어] memory_reads 필드
  double memory_writes;  // [한국어] memory_writes 필드
  double dram_pre;  // [한국어] dram_pre 필드
} system_mem;
/*
 * [한국어] 메모리 컨트롤러(GDDR5 MC) 파라미터 및 통계 구조체
 */
typedef struct {
  // params
  // Common Param for mc and fc
  double peak_transfer_rate;  // [한국어] peak_transfer_rate 필드
  int number_mcs;  // [한국어] number_mcs 필드
  bool withPHY;  // [한국어] withPHY 필드
  int type;  // [한국어] type 필드

  // FCParam
  // stats
  double duty_cycle;  // [한국어] duty_cycle 필드
  double total_load_perc;  // [한국어] total_load_perc 필드

  // McParam
  int mc_clock;  // [한국어] mc_clock 필드
  int llc_line_length;  // [한국어] llc_line_length 필드
  int memory_channels_per_mc;  // [한국어] memory_channels_per_mc 필드
  int number_ranks;  // [한국어] number_ranks 필드
  int req_window_size_per_channel;  // [한국어] req_window_size_per_channel 필드
  int IO_buffer_size_per_channel;  // [한국어] IO_buffer_size_per_channel 필드
  int databus_width;  // [한국어] databus_width 필드
  int addressbus_width;  // [한국어] addressbus_width 필드
  int PRT_entries;  // [한국어] PRT_entries 필드
  bool LVDS;  // [한국어] LVDS 필드

  // emprical DRAM coeff
  double dram_cmd_coeff;  // [한국어] dram_cmd_coeff 필드
  double dram_act_coeff;  // [한국어] dram_act_coeff 필드
  double dram_nop_coeff;  // [한국어] dram_nop_coeff 필드
  double dram_activity_coeff;  // [한국어] dram_activity_coeff 필드
  double dram_pre_coeff;  // [한국어] dram_pre_coeff 필드
  double dram_rd_coeff;  // [한국어] dram_rd_coeff 필드
  double dram_wr_coeff;  // [한국어] dram_wr_coeff 필드
  double dram_req_coeff;  // [한국어] dram_req_coeff 필드
  double dram_const_coeff;  // [한국어] dram_const_coeff 필드

  // stats
  double memory_accesses;  // [한국어] memory_accesses 필드
  double memory_reads;  // [한국어] memory_reads 필드
  double memory_writes;  // [한국어] memory_writes 필드

  // dram stats
  double dram_cmd;  // [한국어] dram_cmd 필드
  double dram_activity;  // [한국어] dram_activity 필드
  double dram_nop;  // [한국어] dram_nop 필드
  double dram_act;  // [한국어] dram_act 필드
  double dram_pre;  // [한국어] dram_pre 필드
  double dram_rd;  // [한국어] dram_rd 필드
  double dram_wr;  // [한국어] dram_wr 필드
  double dram_req;  // [한국어] dram_req 필드

} system_mc;

/*
 * [한국어] NIU(Network Interface Unit) 구조체
 */
typedef struct {
  // params
  int clockrate;  // [한국어] clockrate 필드
  int number_units;  // [한국어] number_units 필드
  int type;  // [한국어] type 필드
  // stats
  double duty_cycle;  // [한국어] duty_cycle 필드
  double total_load_perc;  // [한국어] total_load_perc 필드
} system_niu;

/*
 * [한국어] PCIe 컨트롤러 구조체
 */
typedef struct {
  // params
  int clockrate;  // [한국어] clockrate 필드
  int number_units;  // [한국어] number_units 필드
  int num_channels;  // [한국어] num_channels 필드
  int type;  // [한국어] type 필드
  bool withPHY;  // [한국어] withPHY 필드
  // stats
  double duty_cycle;  // [한국어] duty_cycle 필드
  double total_load_perc;  // [한국어] total_load_perc 필드
} system_pcie;

/*
 * [한국어] 최상위 시스템 설정 집계 구조체 (전체 GPU/XML 루트)
 */
typedef struct {
  // All number_of_* at the level of 'system' Ying 03/21/2009
  int GPU_Architecture;  // [한국어] GPU_Architecture 필드
  int number_of_cores;  // [한국어] number_of_cores 필드
  int architecture;  // [한국어] architecture 필드
  int number_of_L1Directories;  // [한국어] number_of_L1Directories 필드
  int number_of_L2Directories;  // [한국어] number_of_L2Directories 필드
  int number_of_L2s;  // [한국어] number_of_L2s 필드
  bool Private_L2;  // [한국어] Private_L2 필드
  int number_of_L3s;  // [한국어] number_of_L3s 필드
  int number_of_NoCs;  // [한국어] number_of_NoCs 필드
  int number_of_dir_levels;  // [한국어] number_of_dir_levels 필드
  int domain_size;  // [한국어] domain_size 필드
  int first_level_dir;  // [한국어] first_level_dir 필드
  // All params at the level of 'system'
  int homogeneous_cores;  // [한국어] homogeneous_cores 필드
  int homogeneous_L1Directories;  // [한국어] homogeneous_L1Directories 필드
  int homogeneous_L2Directories;  // [한국어] homogeneous_L2Directories 필드
  double core_tech_node;  // [한국어] core_tech_node 필드
  int target_core_clockrate;  // [한국어] target_core_clockrate 필드
  double modeled_chip_voltage_ref;  // [한국어] modeled_chip_voltage_ref 필드
  double static_cat1_flane;  // [한국어] static_cat1_flane 필드
  double static_cat2_flane;  // [한국어] static_cat2_flane 필드
  double static_cat3_flane;  // [한국어] static_cat3_flane 필드
  double static_cat4_flane;  // [한국어] static_cat4_flane 필드
  double static_cat5_flane;  // [한국어] static_cat5_flane 필드
  double static_cat6_flane;  // [한국어] static_cat6_flane 필드
  double static_shared_flane;  // [한국어] static_shared_flane 필드
  double static_l1_flane;  // [한국어] static_l1_flane 필드
  double static_l2_flane;  // [한국어] static_l2_flane 필드
  double static_light_flane;  // [한국어] static_light_flane 필드
  double static_intadd_flane;  // [한국어] static_intadd_flane 필드
  double static_intmul_flane;  // [한국어] static_intmul_flane 필드
  double static_geomean_flane;  // [한국어] static_geomean_flane 필드
  double static_cat1_addlane;  // [한국어] static_cat1_addlane 필드
  double static_cat2_addlane;  // [한국어] static_cat2_addlane 필드
  double static_cat3_addlane;  // [한국어] static_cat3_addlane 필드
  double static_cat4_addlane;  // [한국어] static_cat4_addlane 필드
  double static_cat5_addlane;  // [한국어] static_cat5_addlane 필드
  double static_cat6_addlane;  // [한국어] static_cat6_addlane 필드
  double static_shared_addlane;  // [한국어] static_shared_addlane 필드
  double static_l1_addlane;  // [한국어] static_l1_addlane 필드
  double static_l2_addlane;  // [한국어] static_l2_addlane 필드
  double static_light_addlane;  // [한국어] static_light_addlane 필드
  double static_intadd_addlane;  // [한국어] static_intadd_addlane 필드
  double static_intmul_addlane;  // [한국어] static_intmul_addlane 필드
  double static_geomean_addlane;  // [한국어] static_geomean_addlane 필드
  int target_chip_area;  // [한국어] target_chip_area 필드
  int temperature;  // [한국어] temperature 필드
  int number_cache_levels;  // [한국어] number_cache_levels 필드
  int L1_property;  // [한국어] L1_property 필드
  int L2_property;  // [한국어] L2_property 필드
  int homogeneous_L2s;  // [한국어] homogeneous_L2s 필드
  int L3_property;  // [한국어] L3_property 필드
  int homogeneous_L3s;  // [한국어] homogeneous_L3s 필드
  int homogeneous_NoCs;  // [한국어] homogeneous_NoCs 필드
  int homogeneous_ccs;  // [한국어] homogeneous_ccs 필드
  int Max_area_deviation;  // [한국어] Max_area_deviation 필드
  int Max_power_deviation;  // [한국어] Max_power_deviation 필드
  int device_type;  // [한국어] device_type 필드
  bool longer_channel_device;  // [한국어] longer_channel_device 필드
  bool Embedded;  // [한국어] Embedded 필드
  bool opt_dynamic_power;  // [한국어] opt_dynamic_power 필드
  bool opt_lakage_power;  // [한국어] opt_lakage_power 필드
  bool opt_clockrate;  // [한국어] opt_clockrate 필드
  bool opt_area;  // [한국어] opt_area 필드
  int interconnect_projection_type;  // [한국어] interconnect_projection_type 필드
  int machine_bits;  // [한국어] machine_bits 필드
  int virtual_address_width;  // [한국어] virtual_address_width 필드
  int physical_address_width;  // [한국어] physical_address_width 필드
  int virtual_memory_page_size;  // [한국어] virtual_memory_page_size 필드
  double idle_core_power;  // [한국어] idle_core_power 필드
  double num_idle_cores;  // [한국어] num_idle_cores 필드
  int arch;  // [한국어] arch 필드
  double total_cycles;  // [한국어] total_cycles 필드
  // system.core(0-n):3rd level
  double scaling_coefficients[64];  // [한국어] scaling_coefficients 필드
  system_core core[64];  // [한국어] core 필드
  system_L1Directory L1Directory[64];  // [한국어] L1Directory 필드
  system_L2Directory L2Directory[64];  // [한국어] L2Directory 필드
  system_L2 L2[64];  // [한국어] L2 필드
  system_L2 l2;  // [한국어] l2 필드
  system_L3 L3[64];  // [한국어] L3 필드
  system_NoC NoC[64];  // [한국어] NoC 필드
  system_mem mem;  // [한국어] mem 필드
  system_mc mc;  // [한국어] mc 필드
  system_mc flashc;  // [한국어] flashc 필드
  system_niu niu;  // [한국어] niu 필드
  system_pcie pcie;  // [한국어] pcie 필드
} root_system;

class ParseXML {
 public:
  void parse(char* filepath);
  void initialize();

 public:
  root_system sys;
};

#endif /* XML_PARSE_H_ */
