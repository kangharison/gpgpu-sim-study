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
 ** Jingwen Leng, Univeristy of Texas, Austin                   * Syed Gilani,
 *University of Wisconsin–Madison                * Tayler Hetherington,
 *University of British Columbia         * Ahmed ElTantawy, University of
 *British Columbia             *
 ********************************************************************/

/*
 * [한국어 설명] AccelWattch SM 코어 전력 모델 헤더 (core.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 AccelWattch(GPGPU-Sim 전력 모델)에서 하나의 SM(Streaming
 * Multiprocessor, 셰이더 코어)을 모델링하는 핵심 클래스들을 선언한다. SM을
 * 구성하는 명령어 페치 유닛(InstFetchU), 분기 예측기(BranchPredictor),
 * 스케줄러(SchedulerU), 레지스터 파일/실행 유닛(RegFU/EXECU), 로드/스토어
 * 유닛(LoadStoreU), 메모리 관리 유닛(MemManU), 리네이밍 유닛(RENAMINGU) 등의
 * 서브컴포넌트 클래스와 이들을 집계하는 Core 클래스를 정의한다. 각 클래스는
 * McPAT/CACTI 기반의 면적(area)과 전력(power) 필드를 상속받으며,
 * XML 설정에서 읽은 GPU 마이크로아키텍처 파라미터를 기반으로 SRAM/어레이
 * 구조를 초기화하고 동적/정적 전력을 계산한다. Processor::get_coefficient_*
 * 계수 메서드에서 이 Core의 하위 구조체 전력값을 읽어 GPGPU-Sim의 이벤트
 * 카운터와 곱해 최종 GPU 전력을 산출한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층도:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *       → accelwattch_interface::update() [accelwattch_interface.cc]
 *           → Processor::compute() [processor.cc]
 *               → Core::compute()          ← [이 파일의 핵심 진입점]
 *                   → InstFetchU::computeEnergy()
 *                   → LoadStoreU::computeEnergy()
 *                   → MemManU::computeEnergy()
 *                   → EXECU::computeEnergy()
 *                   → RENAMINGU::computeEnergy() (OOO일 때)
 *               → Processor::get_coefficient_*()
 *                   → ifu->icache, lsu->dcache/tcache/ccache/sharedmemory,
 *                     exu->scheu/rfu/exeu/mul/fp_u 등의 전력 필드 읽기
 *           → 최종 전력(Watt) 산출
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이션 후처리/주기적 전력 갱신.
 * GPU 디바이스 코드와 무관하며, 시뮬레이션된 하드웨어의 전력을 추정한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈 (이 파일이 include/사용하는 것):
 *   - XML_Parse.h (ParseXML*): gpgpusim.config → accelwattch XML 파싱 결과.
 *     system.core[*] 하위의 캐시 크기, 레지스터 파일, 클럭, duty cycle 등을
 *     읽어 서브컴포넌트를 초기화한다.
 *   - basic_components.h (Component, CoreDynParam): McPAT 기본 전력/면적
 *     컴포넌트와 GPU 코어 동적 파라미터 집합.
 *   - array.h (ArrayST): McPAT SRAM/어레이 모델 — 캐시, 버퍼, 레지스터 파일,
 *     TLB, 스케줄러 창 등을 ArrayST로 모델링한다.
 *   - cacti/crossbar.h, cacti/arbiter.h: 레지스터 파일 크로스바/아비터 전력.
 *   - interconnect.h, noc.h: bypass 버스 및 NoC 모델.
 *   - sharedcache.h (SharedCache): Private L2 캐시 전력 모델.
 *   - logic.h: inst_decoder, selection_logic 등 McPAT 로직 모듈.
 * 데이터 흐름:
 *   ParseXML → Core 생성자 → 서브컴포넌트(ArrayST/FunctionalUnit 등) 초기화
 *   → local_result.power에 단위 접근 에너지 저장
 *   → computeEnergy(false)에서 XML->sys.core[*] 카운터로 stats_t.access 설정
 *   → power_t.readOp.dynamic = 에너지 × 접근 횟수
 *   → Processor가 get_coefficient_*()로 이 값을 읽어 최종 전력 계산.
 *
 * === 주요 함수/구조체 요약 ===
 * - BranchPredictor: tournament 분기 예측기(global/local/chooser/RAS) 전력.
 * - InstFetchU: L1I 캐시(icache), BTB, 명령어 버퍼(IB), 디코더 전력.
 * - SchedulerU: 명령어 스케줄러 창(int/fp_inst_window), ROB 전력.
 * - RENAMINGU: OOO 리네이밍 유닛(FRAT/RRAT/free list/DCL) 전력.
 * - LoadStoreU: L1D(dcache), shared memory, constant cache(ccache),
 *   texture cache(tcache), LSQ/LoadQ, shared crossbar 전력.
 * - MemManU: I/D TLB 전력.
 * - RegFU: integer register file(IRF), operand collectors(OPC), FRF,
 *   RFU crossbar/arbiter 전력.
 * - EXECU: RegFU + SchedulerU + ALU(exeu) + FPU(fp_u) + MUL(mul) +
 *   bypass interconnect 집계.
 * - Core: 위 모든 서브유닛을 생성/집계. compute() / computeEnergy() /
 *   displayEnergy() / set_core_param() / get_coefficient_*() 메서드 제공.
 * - get_coefficient_*(): Processor가 호출하는 SM 낭비 에너지 계수.
 *   예: get_coefficient_icache_hits, get_coefficient_dcache_readhits,
 *   get_coefficient_regreads_accesses, get_coefficient_ialu_accesses 등.
 *
 * 관련 설정 파일:
 *   - gpgpusim.config: --power_config_name (AccelWattch XML 경로),
 *     --enable_power_simulation 등 power_config 옵션.
 *   - accelwattch XML: system.core[*] 아래의 icache/dcache/sharedmemory/
 *     tcache/ccache/BTB/predictor/RAS/itlb/dtlb/archi_Regs_*_size/
 *     rf_banks/collector_units/ALU_per_core/FPU_per_core/MUL_per_core/
 *     clock_rate/core_clock_ratio/pipeline_duty_cycle/LSU_duty_cycle 등.
 */

#ifndef CORE_H_
#define CORE_H_

#include "XML_Parse.h"
#include "array.h"
#include "basic_components.h"
#include "cacti/arbiter.h"
#include "cacti/crossbar.h"
#include "cacti/parameter.h"
#include "interconnect.h"
#include "logic.h"
#include "noc.h"
#include "sharedcache.h"

/*
 * [한국어] BranchPredictor — SM 분기 예측기 전력 모델 클래스.
 *
 * GPU에서는 warp 단위 SIMT 분기 처리가 핵심이며, AccelWattch는 McPAT의
 * tournament 분기 예측기 모델(Alpha 21264 스타일)을 GPU SM에 맞게 사용한다.
 * global predictor, local 2-level predictor, chooser, RAS로 구성되며,
 * XML->sys.core[*].predictor / BTB 설정과 prediction_width에 의해 크기/전력이
 * 결정된다. GPU SM에서 실제 분기 예측기 사용량은 제한적이지만, peak 전력
 * 추정을 위해 모델링된다.
 */
class BranchPredictor : public Component {
 public:
  ParseXML *XML;
  /* [한국어] AccelWattch XML 설정 객체 포인터.
   * predictor 관련 파라미터(global_predictor_entries, local_predictor_size,
   * chooser_predictor_bits, RAS_size 등)를 읽는다. */
  int ithCore;
  /* [한국어] 현재 코어(SM) 인덱스. XML->sys.core[ithCore] 접근용. */
  InputParameter interface_ip;
  /* [한국어] McPAT/CACTI 공정/인터페이스 파라미터 구조체.
   * ArrayST 초기화 시 전달되며, 기술 노드/전압/클록 등이 설정되어 있다. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터 (issue width, prediction width, clock rate 등). */
  double clockRate, executionTime;
  /* [한국어] 클럭 레이트(Hz) 및 총 실행 시간(s). 전력↔에너지 변환에 사용. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 소켓/칩/매크로 배치 오버헤드 계수. 면적 보정용. */
  ArrayST *globalBPT;
  /* [한국어] Global Branch Predictor Table SRAM 객체. */
  ArrayST *localBPT;
  /* [한국어] Local BPT 1차(history table) SRAM 객체. */
  ArrayST *L1_localBPT;
  /* [한국어] Local BPT 1차 예측 테이블 SRAM 객체. */
  ArrayST *L2_localBPT;
  /* [한국어] Local BPT 2차 예측 테이블 SRAM 객체. */
  ArrayST *chooser;
  /* [한국어] global/local 예측기 선택을 위한 chooser SRAM 객체. */
  ArrayST *RAS;
  /* [한국어] Return Address Stack SRAM 객체. 함수 호출/복귀 예측용. */
  bool exist;
  /* [한국어] 이 컴포넌트가 실제로 존재(활성화)하는지 여부.
   * false이면 생성자/소멸자에서 대부분 스킵. */

  BranchPredictor(ParseXML *XML_interface, int ithCore_,
                  InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
                  bool exsit = true);
  /* [한국어] 생성자 — XML 설정에 따라 분기 예측기 서브블록(global/local/RAS 등)
   * ArrayST를 생성하고 면적을 누적한다. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] TDP(peak) 또는 runtime 전력 계산.
   * is_tdp=true이면 predictionW * BR_duty_cycle 기준 peak access;
   * false이면 XML->sys.core[ithCore].branch_* 카운터 기준 runtime access. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 계산된 면적/전력 정보를 stdout에 들여쓰기 형식으로 출력. */
  ~BranchPredictor();
  /* [한국어] 소멸자 — 동적 할당한 ArrayST 객체들을 해제. */
};

/*
 * [한국어] InstFetchU — 명령어 페치 유닛 전력 모델 클래스.
 *
 * SM의 L1I 캐시(icache), miss/fill/prefetch 버퍼, 명령어 버퍼(IB),
 * branch target buffer(BTB), 분기 예측기(BPT), inst_decoder(ID_inst/operand/misc)
 * 등의 전력/면적을 모델링한다. XML->sys.core[*].icache / BTB 설정과
 * fetch_width, peak_issue_width 등이 초기화에 영향을 준다.
 */
class InstFetchU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 객체 포인터. */
  int ithCore;
  /* [한국어] 현재 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭 레이트 및 실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  enum Cache_policy cache_p;
  /* [한국어] 캐시 쓰기 정책(Write_back / Write_through). */
  InstCache icache;
  /* [한국어] L1 instruction cache 구조체(caches/missb/ifb/prefetchb/wbb). */
  ArrayST *IB;
  /* [한국어] Instruction Buffer SRAM. */
  ArrayST *BTB;
  /* [한국어] Branch Target Buffer SRAM. predictionW>0일 때 생성. */
  BranchPredictor *BPT;
  /* [한국어] Branch Predictor 객체 포인터. predictionW>0일 때 생성. */
  inst_decoder *ID_inst;
  /* [한국어] 명령어 opcode 디코더. */
  inst_decoder *ID_operand;
  /* [한국어] 피연산자 관련 디코더. */
  inst_decoder *ID_misc;
  /* [한국어] prefix/misc 디코더(x86 스타일). */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */

  InstFetchU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exsit = true);
  /* [한국어] 생성자 — icache/버퍼/BTB/BPT/디코더를 초기화하고 면적 누적. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] 명령어 페치 관련 접근 횟수를 TDP 또는 runtime 카운터로 설정하고
   * 동적 전력을 계산. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] IFU 면적/전력 출력. */
  ~InstFetchU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] SchedulerU — 명령어 스케줄러 전력 모델 클래스.
 *
 * OOO 스케줄러의 int/fp instruction window, ROB, instruction selection logic을
 * 모델링한다. GPU SM에서 AccelWattch는 issue width, ROB_size 등 XML 값을
 * 사용하지만, 실제 warp 스케줄러와는 다소 추상화된 McPAT 모델을 따른다.
 */
class SchedulerU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  double Iw_height, fp_Iw_height, ROB_height;
  /* [한국어] instruction window/ROB의 물리 높이(연결 길이 추정용). */
  ArrayST *int_inst_window;
  /* [한국어] 정수 명령어 스케줄러 창 SRAM. */
  ArrayST *fp_inst_window;
  /* [한국어] FP 명령어 스케줄러 창 SRAM. */
  ArrayST *ROB;
  /* [한국어] Reorder Buffer SRAM. */
  selection_logic *instruction_selection;
  /* [한국어] instruction selection logic(ready inst 선택). */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */

  SchedulerU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exist_ = true);
  /* [한국어] 생성자 — 스케줄러 창/ROB/selection logic 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] 스케줄러 접근 횟수를 TDP 또는 runtime 카운터로 설정 후 전력 산출. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 스케줄러 면적/전력 출력. */
  ~SchedulerU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] RENAMINGU — OOO 리네이밍 유닛 전력 모델 클래스.
 *
 * OOO 코어의 register alias table(RAT), free list, dependency check logic
 * 등을 모델링한다. GPU SM은 일반적으로 In-order SIMT 모델이므로 GPU
 * 설정에서는 대부분 활성화되지 않지만, CPU/OOO 경로 호환성을 위해 유지된다.
 */
class RENAMINGU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  ArrayST *iFRAT;
  /* [한국어] integer Front RAT. */
  ArrayST *fFRAT;
  /* [한국어] FP Front RAT. */
  ArrayST *iRRAT;
  /* [한국어] integer Retire RAT. */
  ArrayST *fRRAT;
  /* [한국어] FP Retire RAT. */
  ArrayST *ifreeL;
  /* [한국어] integer free list. */
  ArrayST *ffreeL;
  /* [한국어] FP free list. */
  dep_resource_conflict_check *idcl;
  /* [한국어] integer dependency/conflict check logic. */
  dep_resource_conflict_check *fdcl;
  /* [한국어] FP dependency/conflict check logic. */
  ArrayST *RAHT;  // register alias history table Used to store GC
  /* [한국어] Register Alias History Table (global checkpoint 저장용).
   * 현재 코드에서는 사용되지 않는 레거시 포인터. */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */

  RENAMINGU(ParseXML *XML_interface, int ithCore_,
            InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
            bool exist_ = true);
  /* [한국어] 생성자 — RAT/free list/DCL 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] 리네이밍 유닛 전력 계산. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 출력. */
  ~RENAMINGU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] LoadStoreU — 로드/스토어 유닛 전력 모델 클래스.
 *
 * SM의 L1D 캐시(dcache), shared memory, constant cache(ccache),
 * texture cache(tcache), LSQ(Load/Store Queue), shared crossbar(xbar_shared),
 * NoC 연결(nocs) 등의 전력/면적을 모델링한다. AccelWattch가 GPU 메모리
 * 접근(dcache read/write hit/miss, sharedmemory, tcache, ccache)의 주요
 * 계수를 산출하는 곳이며, XML->sys.core[*].{dcache,sharedmemory,ccache,
 * tcache}.dcache_config / buffer_sizes가 핵심 설정이다.
 */
class LoadStoreU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  enum Cache_policy cache_p;
  /* [한국어] L1D/sharedmemory/tcache/ccache 공통 쓰기 정책. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  double lsq_height;
  /* [한국어] LSQ 물리 높이(연결 길이 추정용). */
  DataCache dcache;
  /* [한국어] L1 data cache 전력 모델(caches/missb/ifb/prefetchb/wbb). */
  DataCache ccache;
  /* [한국어] constant cache 전력 모델. */
  DataCache tcache;
  /* [한국어] texture cache 전력 모델. */
  DataCache sharedmemory;
  /* [한국어] shared memory 전력 모델. */
  ArrayST *LSQ;  // it is actually the store queue but for inorder processors it
                 // serves as both loadQ and StoreQ
  /* [한국어] Load/Store Queue(실제로 store queue) SRAM.
   * GPU 설정에서는 전력을 0으로 리셋하여 비활성화한다. */
  ArrayST *LoadQ;
  /* [한국어] 별도 load queue(OOO + load_buffer_size>0일 때). */
  vector<NoC *> nocs;
  /* [한국어] LSU 관련 NoC 객체 벡터(현재 GPU 모델에서는 사용 안 됨). */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */
  Crossbar *xbar_shared;
  /* [한국어] shared memory/dcache/tcache/ccache 접근을 위한 낮춤형 크로스바.
   * Syed가 추가한 GPU shared memory interconnect 모델. */
  Component noc;
  /* [한국어] NoC 집계 전력/면적용 임시 Component. */
  LoadStoreU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exist_ = true);
  /* [한국어] 생성자 — shared xbar, sharedmemory, ccache, tcache, dcache,
   * 버퍼, LSQ/LoadQ를 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] LSU 하위 캐시/버퍼/LSQ의 runtime/TDP 전력 계산.
   * GPGPU-Sim 카운터(read_accesses, write_accesses, read_misses,
   * write_misses)를 XML에서 읽어 접근 횟수로 사용. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] LSU 면적/전력 출력. */
  void displayDeviceType(int device_type_,
                         uint32_t indent);  // Added by Syed Gilani
  /* [한국어] 디바이스 타입 출력(Syed 추가). */

  ~LoadStoreU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] MemManU — 메모리 관리 유닛(TLB) 전력 모델 클래스.
 *
 * I-TLB(itlb)와 D-TLB(dtlb)를 ArrayST로 모델링한다. GPGPU-Sim은 현재 TLB
 * 미스를 시뮬레이션하지 않지만 전력 추정을 위해 TLB 구조 크기를 XML에서
 * 읽어 모델링한다.
 */
class MemManU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  ArrayST *itlb;
  /* [한국어] Instruction TLB SRAM. */
  ArrayST *dtlb;
  /* [한국어] Data TLB SRAM. */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */

  MemManU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
          const CoreDynParam &dyn_p_, bool exist_ = false);
  /* [한국어] 생성자 — I/D TLB 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] TLB 접근 횟수로 전력 계산. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 출력. */
  ~MemManU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] RegFU — 레지스터 파일/Operand Collector 전력 모델 클래스.
 *
 * GPU SM의 integer register file(IRF), operand collectors(OPC), FP register
 * file(FRF), register window(RFWIN), 그리고 RFU crossbar/arbiter를 모델링한다.
 * AccelWattch는 GPU의 warp 단위 SIMD 실행을 반영해 IRF와 operand collector를
 * 중심으로 전력을 산출하며, FRF는 GPU 결과에서 제외(주석 처리)한다.
 * XML->sys.core[*].archi_Regs_IRF_size, rf_banks, collector_units,
 * architecture(1=Tesla, 2=Fermi 이상) 등이 초기화에 영향을 준다.
 */
class RegFU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  double int_regfile_height, fp_regfile_height;
  /* [한국어] 정수/FP 레지스터 파일 물리 높이. */
  ArrayST *IRF;
  /* [한국어] Integer Register File SRAM. */
  ArrayST *FRF;
  /* [한국어] Floating-point Register File SRAM (GPU 결과에서 제외). */
  ArrayST *RFWIN;
  /* [한국어] Register Window SRAM(In-order SPARC 스타일). */
  ArrayST *OPC;  // Operand collectors
  /* [한국어] Operand Collectors SRAM. */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */
  double exClockRate;
  /* [한국어] 실행 유닛/RFU 클럭 레이트. 코어 클럭과 다를 수 있음
   * (core_clock_ratio 반영). */
  // OC Modelling (Syed)
  Crossbar *xbar_rfu;
  /* [한국어] 레지스터 파일 뱅크↔operand collector 간 crossbar. */
  MCPAT_Arbiter *arbiter_rfu;
  /* [한국어] crossbar 포트 중재용 arbiter. */
  RegFU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
        const CoreDynParam &dyn_p_, double exClockRate, bool exist_ = true);
  /* [한국어] 생성자 — xbar/arbiter/IRF/OPC/FRF/RFWIN 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] RF/OPC/xbar/arbiter 접근 횟수로 전력 계산.
   * runtime 시 XML->sys.core[*].int_regfile_reads/writes,
   * non_rf_operands 카운터를 사용. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 출력. */
  ~RegFU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] EXECU — 실행 유닛 집계 전력 모델 클래스.
 *
 * RegFU, SchedulerU, ALU(exeu), FPU(fp_u), MUL(mul), 그리고 bypass
 * interconnect들을 묶어 SM의 실행 부분 전력/면적을 집계한다. GPU 설정에서는
 * bypass interconnect이 실제로 동작하지 않는 것으로 처리되며(주석/No conventional
 * bypassing in GPU), ALU/FPU/MUL/RFU/scheduler 전력만 합산한다.
 */
class EXECU : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 클럭/실행 시간. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  double lsq_height;
  /* [한국어] LSU LSQ 높이(bypass 길이 추정용). */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. */
  RegFU *rfu;
  /* [한국어] 레지스터 파일/operand collector 유닛. */
  SchedulerU *scheu;
  /* [한국어] 명령어 스케줄러 유닛. */
  FunctionalUnit *fp_u;
  /* [한국어] FP 실행 유닛(FPU). */
  FunctionalUnit *exeu;
  /* [한국어] 정수 ALU 실행 유닛. */
  FunctionalUnit *mul;
  /* [한국어] 정수 곱셈기(MUL/SFU). */
  interconnect *int_bypass;
  /* [한국어] 정수 bypass 데이터 버스. */
  interconnect *intTagBypass;
  /* [한국어] 정수 bypass 태그 버스. */
  interconnect *int_mul_bypass;
  /* [한국어] 곱셈기 bypass 데이터 버스. */
  interconnect *intTag_mul_Bypass;
  /* [한국어] 곱셈기 bypass 태그 버스. */
  interconnect *fp_bypass;
  /* [한국어] FP bypass 데이터 버스. */
  interconnect *fpTagBypass;
  /* [한국어] FP bypass 태그 버스. */
  bool exist;
  /* [한국어] 컴포넌트 활성화 여부. */
  double rf_fu_clockRate;
  /* [한국어] RF/FU 클럭 레이트. 코어 클럭과 다를 때 전력 스케일링에 사용. */
  Component bypass;
  /* [한국어] bypass 버스 집계 전력/면적용 Component. */

  EXECU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
        double lsq_height_, const CoreDynParam &dyn_p_, double exClockRate,
        bool exist_);
  /* [한국어] 생성자 — rfu/scheu/exeu/fp_u/mul 및 bypass interconnect 초기화. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] 하위 유닛 전력을 계산하고 집계. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 출력. */
  ~EXECU();
  /* [한국어] 소멸자. */
};

/*
 * [한국어] Core — 단일 SM의 전체 전력/면적을 집계하는 최상위 클래스.
 *
 * IFU/LSU/MMU/EXEU/RENAME(OOO) 서브유닛과 파이프라인(Pipeline), 미분류 코어
 * 로직(UndiffCore), Private L2(SharedCache)를 포함한다. Processor가 생성하는
 * 대표 SM이며, homogeneous_cores==1일 때 number_of_cores만큼 스케일링되어
 * 전체 SM 전력을 구한다.
 */
class Core : public Component {
 public:
  ParseXML *XML;
  /* [한국어] XML 설정 포인터. */
  int ithCore;
  /* [한국어] 코어 인덱스. */
  InputParameter interface_ip;
  /* [한국어] CACTI 인터페이스 파라미터. */
  double clockRate, executionTime;
  /* [한국어] 코어 클럭/실행 시간. */
  double exClockRate;
  /* [한국어] 실행 유닛 클럭 레이트. */
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 면적 보정 계수. */
  InstFetchU *ifu;
  /* [한국어] 명령어 페치 유닛. */
  LoadStoreU *lsu;
  /* [한국어] 로드/스토어 유닛. */
  MemManU *mmu;
  /* [한국어] 메모리 관리(TLB) 유닛. */
  EXECU *exu;
  /* [한국어] 실행 유닛. */
  RENAMINGU *rnu;
  /* [한국어] 리네이밍 유닛(OOO). */
  double IdleCoreEnergy;
  /* [한국어] 유휴(idle) 코어들이 소비하는 총 동적 에너지(Joule).
   * compute()에서 XML->sys.num_idle_cores × idle_core_power × executionTime
   * 로 계산. */
  double IdlePower_PerCore;
  /* [한국어] 코어당 유휴 전력(Watt). 현재 사용되지 않는 필드. */
  Pipeline *corepipe;
  /* [한국어] 파이프라인 레지스터 클록 전력 모델. */
  UndiffCore *undiffCore;
  /* [한국어] 미분류 코어 로직(pll, fuse 등) 전력 모델. */
  SharedCache *l2cache;
  /* [한국어] Private L2 캐시 전력 모델. XML->sys.Private_L2일 때 생성. */
  CoreDynParam coredynp;
  /* [한국어] 코어 동적 파라미터. set_core_param()에서 XML로부터 채워진다. */
  double Pipeline_energy;
  /* [한국어] 이번 구간 동안 파이프라인에서 소비된 에너지(Joule).
   * compute()에서 누적. */
  // full_decoder 	inst_decoder;
  // clock_network	clockNetwork;

  Core(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_);
  /* [한국어] 생성자 — set_core_param()을 호출한 뒤 IFU/LSU/MMU/EXU/UndiffCore/
   * RENAME(OOO)/Pipeline/PrivateL2를 생성하고 면적을 집계한다. */
  void set_core_param();
  /* [한국어] XML->sys.core[ithCore]의 마이크로아키텍처 파라미터를
   * coredynp에 복사하고 클럭, duty cycle, 레지스터/파이프라인 수 등을 설정. */
  void computeEnergy(bool is_tdp = true);
  /* [한국어] TDP 또는 runtime 기준으로 모든 서브유닛 전력을 집계.
   * Processor::compute() 내부에서 호출되기도 한다. */
  void compute();
  /* [한국어] AccelWattch 래퍼가 호출하는 runtime 전력 계산 진입점.
   * 하위 유닛의 computeEnergy(false)를 호출하고 파이프라인 에너지와
   * idle core 에너지를 추가한다. */
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] Core 전체 면적/전력 출력. */

  /*
   * [한국어] get_coefficient_icache_hits — L1I 읽기 히트 1회당 에너지 계수.
   * Processor::get_coefficient_tot_insts와 함께 총 명령어/페치 전력에 사용.
   */
  float get_coefficient_icache_hits() {
    // return 1.5*ifu->icache.caches->local_result.power.readOp.dynamic;
    return ifu->icache.caches->local_result.power.readOp.dynamic;
  }

  /*
   * [한국어] get_coefficient_icache_misses — L1I 읽기 미스 1회당 에너지.
   * 캐시 태그/데이터 읽기 + miss/fill/prefetch 버퍼 search/write 에너지 합산.
   */
  float get_coefficient_icache_misses() {
    float value = 0;
    value += ifu->icache.caches->local_result.power.writeOp.dynamic;
    value += ifu->icache.caches->local_result.power.readOp.dynamic;
    value += ifu->icache.missb->local_result.power.searchOp.dynamic;
    value += ifu->icache.missb->local_result.power.writeOp.dynamic;
    value += ifu->icache.ifb->local_result.power.searchOp.dynamic;
    value += ifu->icache.ifb->local_result.power.writeOp.dynamic;
    value += ifu->icache.prefetchb->local_result.power.searchOp.dynamic;
    value += ifu->icache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_tot_insts — SM에서 실행된 명령어 1개당
   * 명령어 버퍼(IB)와 디코더(ID_inst/operand/misc) 에너지 합산.
   */
  float get_coefficient_tot_insts() {
    float value = 0;
    value += ifu->IB->local_result.power.readOp.dynamic;
    value += ifu->IB->local_result.power.writeOp.dynamic;
    value += ifu->ID_inst->power_t.readOp.dynamic;
    value += ifu->ID_operand->power_t.readOp.dynamic;
    value += ifu->ID_misc->power_t.readOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_fpint_insts — 정수/FP instruction window 및
   * selection logic 접근 에너지. int_inst_window read/search/write + 
   * instruction_selection read.
   */
  float get_coefficient_fpint_insts() {
    float value = 0;
    value += exu->scheu->int_inst_window->local_result.power.readOp.dynamic;
    value +=
        2 * exu->scheu->int_inst_window->local_result.power.searchOp.dynamic;
    value += exu->scheu->int_inst_window->local_result.power.writeOp.dynamic;
    value += exu->scheu->instruction_selection->power.readOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_dcache_readhits — L1D 읽기 히트 1회당 에너지.
   * dcache data array read + shared xbar read.
   */
  float get_coefficient_dcache_readhits() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 0.5*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_dcache_readmisses — L1D 읽기 미스 1회당 에너지.
   * Write_back 정책이 아닐 때에만 missb/ifb/prefetchb search/write 추가.
   */
  float get_coefficient_dcache_readmisses() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.prefetchb->local_result.power.writeOp.dynamic;

    // return 0.5*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_dcache_writehits — L1D 쓰기 히트 1회당 에너지.
   */
  float get_coefficient_dcache_writehits() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.writeOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_dcache_writemisses — L1D 쓰기 미스 1회당 에너지.
   * Write_back일 때 dirty block write-back으로 인한 추가 에너지 포함.
   */
  float get_coefficient_dcache_writemisses() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.writeOp.dynamic;
    value += lsu->dcache.caches->local_result.tag_array2->power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.caches->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.missb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.ifb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.prefetchb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.wbb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.missb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.ifb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.prefetchb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.wbb->local_result.power.writeOp.dynamic
                 : 0;
    // return 1.6*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_tcache_readhits — texture cache 읽기 히트 에너지.
   */
  float get_coefficient_tcache_readhits() {
    float value = 0;
    value += lsu->tcache.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 0.2*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_tcache_readmisses — texture cache 읽기 미스 에너지.
   */
  float get_coefficient_tcache_readmisses() {
    float value = 0;
    value += lsu->tcache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.writeOp.dynamic;

    // return 0.2*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_tcache_readmisses1 — texture cache miss 중
   * data array 접근만 분리한 계수.
   */
  float get_coefficient_tcache_readmisses1() {
    return lsu->tcache.caches->local_result.power.readOp.dynamic;
  }

  /*
   * [한국어] get_coefficient_tcache_readmisses2 — texture cache miss 중
   * controller 버퍼 접근만 분리한 계수.
   */
  float get_coefficient_tcache_readmisses2() {
    float value = 0;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_ccache_readhits — constant cache 읽기 히트 에너지.
   */
  float get_coefficient_ccache_readhits() {
    // return 1.2*lsu->ccache.caches->local_result.power.readOp.dynamic+lsu->xbar_shared->power.readOp.dynamic;
    // return 1.2*lsu->ccache.caches->local_result.power.readOp.dynamic+lsu->xbar_shared->power.readOp.dynamic;
    return lsu->ccache.caches->local_result.power.readOp.dynamic +
           lsu->xbar_shared->power.readOp.dynamic;
  }

  /*
   * [한국어] get_coefficient_ccache_readmisses — constant cache 읽기 미스 에너지.
   */
  float get_coefficient_ccache_readmisses() {
    float value = 0;
    value += lsu->ccache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_sharedmemory_readhits — shared memory 읽기 히트 에너지.
   */
  float get_coefficient_sharedmemory_readhits() {
    float value = 0;
    value += lsu->sharedmemory.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 3*value;
    return value;
  }

  /*
   * [한국어] get_coefficient_lsq_accesses — LSQ 접근 계수.
   * GPU에서는 LSQ가 제거되었으므로 0을 반환.
   */
  float get_coefficient_lsq_accesses() {
    float value = 0;
    // Changed by Syed -- We have removed LSQ
    // value+=2*lsu->LSQ->local_result.power.searchOp.dynamic;
    // value+=2*lsu->LSQ->local_result.power.readOp.dynamic;
    // value+=2*lsu->LSQ->local_result.power.writeOp.dynamic;
    return value;
  }

  /*
   * [한국어] get_coefficient_regreads_accesses — 레지스터 파일 읽기 1회당 에너지.
   * IRF read/32 × (4×2) bank factor + xbar/arbiter/OPC read 분배.
   */
  float get_coefficient_regreads_accesses() {
    float value = 0;
    value += ((exu->rfu->IRF->local_result.power.readOp.dynamic / 32) *
              (4 * 2) /*/1.5*/);
    value += exu->rfu->xbar_rfu->power.readOp.dynamic / (32 /**1.5*/);
    value += (exu->rfu->arbiter_rfu->power.readOp.dynamic / 32 /**1.5)*/);
    value += exu->rfu->OPC->local_result.power.readOp.dynamic /*/1.5*/;
    return value;
  }

  /*
   * [한국어] get_coefficient_regwrites_accesses — 레지스터 파일 쓰기 1회당 에너지.
   */
  float get_coefficient_regwrites_accesses() {
    return ((exu->rfu->IRF->local_result.power.writeOp.dynamic / 32) *
            (4 * 2) /*/1.5*/);
  }

  /*
   * [한국어] get_coefficient_noregfileops_accesses — 레지스터 파일을 사용하지
   * 않는 피연산자 접근(operand collector/xbar/arbiter) 에너지.
   */
  float get_coefficient_noregfileops_accesses() {
    return ((exu->rfu->xbar_rfu->power.readOp.dynamic / (32 /**1.5*/)) +
            (exu->rfu->arbiter_rfu->power.readOp.dynamic / (32 /**1.5*/)) +
            (exu->rfu->OPC->local_result.power.readOp.dynamic /*/(1.5)*/));
  }

  /*
   * [한국어] get_coefficient_ialu_accesses — 정수 ALU 접근 1회당 에너지.
   */
  float get_coefficient_ialu_accesses() {
    // return 10*exu->exeu->per_access_energy*g_tp.sckt_co_eff;
    return exu->exeu->per_access_energy * g_tp.sckt_co_eff;
  }

  /*
   * [한국어] get_coefficient_sfu_accesses — 특수 함수 유닛(MUL) 접근 1회당 에너지.
   */
  float get_coefficient_sfu_accesses() {
    return exu->mul->per_access_energy * g_tp.sckt_co_eff;
    // return 2.6*exu->mul->per_access_energy*g_tp.sckt_co_eff;
  }

  /*
   * [한국어] get_coefficient_fpu_accesses — FP 유닛 접근 1회당 에너지.
   */
  float get_coefficient_fpu_accesses() {
    // return 3.2*exu->fp_u->per_access_energy*g_tp.sckt_co_eff;
    return exu->fp_u->per_access_energy * g_tp.sckt_co_eff;
  }

  /*
   * [한국어] get_coefficient_duty_cycle — 파이프라인 duty-cycle 기반 클럭
   * 에너지 계수. XML->sys.total_cycles, number_of_cores, num_pipelines 사용.
   */
  float get_coefficient_duty_cycle() {
    float value = 0;
    float num_units = 4.0;
    value = XML->sys.total_cycles * XML->sys.number_of_cores;
    value *= coredynp.num_pipelines;
    value /= num_units;
    value *= corepipe->power.readOp.dynamic;
    value *= 3;
    return value;
    // return 1.5*value;
  }

  ~Core();
  /* [한국어] 소멸자 — 하위 객체들을 해제. */
};

#endif /* CORE_H_ */
