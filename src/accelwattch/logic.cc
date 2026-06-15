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
 * [한국어 설명] AccelWattch 논리 회로/기능 유닛 전력 모델 구현 (logic.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT 스타일의 디지털 논리 블록들 — Issue Window 선택 논리,
 * 의존성/자원 충돌 검사기, DFF 셀, 파이프라인 레지스터, 기능 유닛(FPU/ALU/MUL),
 * 미분화 코어(UndiffCore), 명령어 디코더 — 의 동적/정적 전력을 트랜지스터
 * 수준에서 추정한다. CACTI의 gate_C/drain_C_ 함수와 cmos_Isub_leakage/
 * cmos_Ig_leakage 함수를 사용해 0.8um/65nm/90nm 등의 공정 기준 데이터를
 * 대상 기술 노드로 스케일링한다. AccelWattch에서 GPU SM 코어의 ALU/FPU/MUL,
 * 파이프라인, 디코더 등의 전력이 이 파일을 통해 계산된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Core 클래스가 이 파일의 클래스 객체를 멤버로 생성하여 개별 코어 전력을 구성한다.
 *   Core::Core() → selection_logic, dep_resource_conflict_check, Pipeline,
 *                   FunctionalUnit(FPU/ALU/MUL), inst_decoder, UndiffCore 등 생성
 *   Core::computeEnergy() → 각 멤버의 computeEnergy()/compute() 호출
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계 및 주기적 전력 집계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: logic.h(클스 선언), XML_Parse.h(ParseXML, CoreDynParam),
 *       basic_components.h(powerDef, Component, longer_channel_device_reduction),
 *       cacti/basic_circuit.h(cmos_Isub_leakage, gate_C, drain_C_, Decoder, Predec 등),
 *       cacti/parameter.h(InputParameter, g_tp, g_ip, init_interface).
 * 의존 받음: core.cc(Core의 구성 요소), processor.cc(최종 집계).
 * 데이터 흐름: XML/코어 파라미터 → 생성자 → 면적/누설/기준 동적 에너지 산출
 *             → computeEnergy()에서 stats_t 접근수(access)를 곱해 TDP/런타임 전력 산출.
 *
 * === 주요 함수/클례스 요약 ===
 * selection_logic::selection_power()      — issue-window 선택기의 OR/arbiter 전력.
 * dep_resource_conflict_check             — Scoreboard/CAM 스타일 비교기 충돌 검사.
 * DFFCell                                  — NAND2 기반 D 플립플롭 셀 전력.
 * Pipeline                                 — 코어/비코어 파이프라인 레지스터 전력.
 * FunctionalUnit                           — FPU/ALU/MUL/SFU 전력 (AccelWattch GPU 핵심).
 * UndiffCore                               — 코어 공통 로직(front-end/back-end 등) 전력.
 * inst_decoder                             — n-to-2^n 디코더 및 x86 시퀀서 전력.
 * leakage_feedback()                      — 온도 변화에 따른 누설 재계산.
 */
#include "logic.h"           // [한국어] logic.h 클래스 선언 (selection_logic, FunctionalUnit 등)
#define SP_BASE_POWER 0  // [한국어] AccelWattch GPU SP base power off/placeholder
#define SFU_BASE_POWER 0  // [한국어] AccelWattch GPU SFU base power off/placeholder
//.67

// extern double exClockRate;
// selection_logic
/*
 * [한국어]
 * selection_logic::selection_logic - Issue Window 선택 논리 생성자
 *
 * @configure_interface: CACTI InputParameter (공정/전압/온도).
 * @device_ty_, core_ty_: 장치/코어 유형 — 장채널 누설 보정에 사용.
 * @return: (생성자) — power.readOp에 동적/누설/게이트 누설 저장.
 *
 * 설정값을 복사하고 CACTI를 초기화한 뒤, selection_power()로 선택 논리의
 * OR/arbiter/encoder 전력을 계산한다. 소켓 효과(sckt_co_eff)와 장채널
 * 누설 보정을 적용한다.
 */
selection_logic::selection_logic(bool _is_default, int win_entries_,
                                 int issue_width_,
                                 const InputParameter *configure_interface,
                                 enum Device_ty device_ty_,
                                 enum Core_type core_ty_)
    // const ParseXML *_XML_interface)
    : is_default(_is_default),
      win_entries(win_entries_),
      issue_width(issue_width_),
      device_ty(device_ty_),
      core_ty(core_ty_) {
  // uca_org_t result2;
  l_ip = *configure_interface;
  local_result = init_interface(&l_ip);
  // init_tech_params(l_ip.F_sz_um, false);
  // win_entries=numIBEntries;//IQentries;
  // issue_width=issueWidth;
  selection_power();  // [한국어] 선택 논리(우선순위 인코더 + OR) 전력 계산
  double sckRation = g_tp.sckt_co_eff;  // [한국어] 소켓/IO 드라이버 오버헤드 계수
  power.readOp.dynamic *= sckRation;
  power.writeOp.dynamic *= sckRation;
  power.searchOp.dynamic *= sckRation;

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
}

/*
 * [한국어]
 * selection_logic::selection_power - 비용 효율적인 슈퍼스칼라 선택기 전력
 *
 * TR pp.27-31 기반: 4입력 OR(anyreq), 4비트 우선순위 인코더, enable/grant
 * 인버터 체인의 캐패시턴스를 합산하여 동적 에너지를 산출한다. win_entries가
 * 4를 초과하면 트리 arbiter 개수(num_arbiter)를 증가시킨다.
 * 동적 에너지에 2를 곱하는 이유는 arbitration 신호가 왕복(round trip)하기
 * 때문이다.
 */
void selection_logic::selection_power() {  // based on cost effective
                                           // superscalar processor TR pp27-31
  double Ctotal, Cor, Cpencode;
  int num_arbiter;
  double WSelORn, WSelORprequ, WSelPn, WSelPp, WSelEnn, WSelEnp;

  // [한국어] 아래 트랜지스터 폭은 0.8um 공정 기준값을 현재 기술 노드(F_sz_um)로 비례 스케일링한 것이다.
  // TODO: the 0.8um process data is used.
  WSelORn =
      12.5 * l_ip.F_sz_um;  // this was 10 micron for the 0.8 micron process
  WSelORprequ =
      50 * l_ip.F_sz_um;  // this was 40 micron for the 0.8 micron process
  WSelPn = 12.5 * l_ip.F_sz_um;  // this was 10mcron for the 0.8 micron process
  WSelPp =
      18.75 * l_ip.F_sz_um;  // this was 15 micron for the 0.8 micron process
  WSelEnn = 6.25 * l_ip.F_sz_um;  // this was 5 micron for the 0.8 micron
                                  // process
  WSelEnp =
      12.5 * l_ip.F_sz_um;  // this was 10 micron for the 0.8 micron process

  Ctotal = 0;  // [한국어] 누적 캐패시턴스 초기화
  num_arbiter = 1;
  // [한국어] 발행 가능 슬롯 수가 4를 초과하면 4진 트리 우선순위 arbiter 추가
  while (win_entries > 4) {
    win_entries = (int)ceil((double)win_entries / 4.0);
    num_arbiter += win_entries;
  }
  // the 4-input OR logic to generate anyreq
  Cor = 4 * drain_C_(WSelORn, NCH, 1, 1, g_tp.cell_h_def) +
        drain_C_(WSelORprequ, PCH, 1, 1, g_tp.cell_h_def);
  power.readOp.gate_leakage =
      cmos_Ig_leakage(WSelORn, WSelORprequ, 4, nor) * g_tp.peri_global.Vdd;

  // The total capacity of the 4-bit priority encoder
  Cpencode =
      drain_C_(WSelPn, NCH, 1, 1, g_tp.cell_h_def) +
      drain_C_(WSelPp, PCH, 1, 1, g_tp.cell_h_def) +
      2 * drain_C_(WSelPn, NCH, 1, 1, g_tp.cell_h_def) +
      drain_C_(WSelPp, PCH, 2, 1, g_tp.cell_h_def) +
      3 * drain_C_(WSelPn, NCH, 1, 1, g_tp.cell_h_def) +
      drain_C_(WSelPp, PCH, 3, 1, g_tp.cell_h_def) +
      4 * drain_C_(WSelPn, NCH, 1, 1, g_tp.cell_h_def) +
      drain_C_(WSelPp, PCH, 4, 1,
               g_tp.cell_h_def) +  // precompute priority logic
      2 * 4 * gate_C(WSelEnn + WSelEnp, 20.0) +
      4 * drain_C_(WSelEnn, NCH, 1, 1, g_tp.cell_h_def) +
      2 * 4 * drain_C_(WSelEnp, PCH, 1, 1, g_tp.cell_h_def) +  // enable logic
      (2 * 4 + 2 * 3 + 2 * 2 + 2) *
          gate_C(WSelPn + WSelPp, 10.0);  // requests signal

  Ctotal += issue_width * num_arbiter * (Cor + Cpencode);  // [한국어] issue_width×arbiter 수×(OR+encoder cap)

  power.readOp.dynamic =  // [한국어] 동적 에너지 = Ctotal × Vdd² × 2(왕복)
      Ctotal * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd *
      2;  // 2 means the abitration signal need to travel round trip
  power.readOp.leakage =  // [한국어] 누설 전력 = issue_width×arbiter 수 × (grant/enable/inverter NOR 누설 합) × Vdd
      issue_width * num_arbiter *
      (cmos_Isub_leakage(
           WSelPn, WSelPp, 2,
           nor) /*approximate precompute with a nor gate*/  // grant1p
       + cmos_Isub_leakage(WSelPn, WSelPp, 3, nor)          // grant2p
       + cmos_Isub_leakage(WSelPn, WSelPp, 4, nor)          // grant3p
       + cmos_Isub_leakage(WSelEnn, WSelEnp, 2, nor) * 4    // enable logic
       + cmos_Isub_leakage(WSelEnn, WSelEnp, 1, inv) * 2 *
             3  // for each grant there are two inverters, there are 3 grant
                // sIsubnals
       ) *
      g_tp.peri_global.Vdd;
  power.readOp.gate_leakage =
      issue_width * num_arbiter *
      (cmos_Ig_leakage(
           WSelPn, WSelPp, 2,
           nor) /*approximate precompute with a nor gate*/  // grant1p
       + cmos_Ig_leakage(WSelPn, WSelPp, 3, nor)            // grant2p
       + cmos_Ig_leakage(WSelPn, WSelPp, 4, nor)            // grant3p
       + cmos_Ig_leakage(WSelEnn, WSelEnp, 2, nor) * 4      // enable logic
       + cmos_Ig_leakage(WSelEnn, WSelEnp, 1, inv) * 2 *
             3  // for each grant there are two inverters, there are 3 grant
                // signals
       ) *
      g_tp.peri_global.Vdd;
}

/*
 * [한국어]
 * dep_resource_conflict_check::dep_resource_conflict_check - 의존성/자원 충돌 검사기 생성자
 *
 * @configure_interface: CACTI InputParameter.
 * @dyn_p_: 코어 동적 파라미터 (decodeW, core_ty 등).
 * @compare_bits_: 비교할 비트 수. Inorder/OOO 모두 opcode+reg_tag 비트 추가.
 * @return: (생성자) — conflict_check_power()로 전력 산출.
 *
 * 트랜지스터 폭을 0.8um 기준에서 현재 공정으로 스케일링하고, Inorder/OOO에
 * 따라 compare_bits에 opcode/tag 비트를 추가한 뒤 충돌 검사 전력을 계산한다.
 */
dep_resource_conflict_check::dep_resource_conflict_check(
    const InputParameter *configure_interface, const CoreDynParam &dyn_p_,
    int compare_bits_, bool _is_default)
    : l_ip(*configure_interface),
      coredynp(dyn_p_),
      compare_bits(compare_bits_),
      is_default(_is_default) {
  Wcompn = 25 * l_ip.F_sz_um;  // this was 20.0 micron for the 0.8 micron
                               // process
  Wevalinvp =
      25 * l_ip.F_sz_um;  // this was 20.0 micron for the 0.8 micron process
  Wevalinvn =
      100 * l_ip.F_sz_um;  // this was 80.0 mcron for the 0.8 micron process
  Wcomppreequ =
      50 * l_ip.F_sz_um;  // this was 40.0  micron for the 0.8 micron process
  WNORn = 6.75 * l_ip.F_sz_um;  // this was 5.4 micron for the 0.8 micron
                                // process
  WNORp =
      38.125 * l_ip.F_sz_um;  // this was 30.5 micron for the 0.8 micron process

  local_result = init_interface(&l_ip);

  // [한국어] Inorder/OOO 모두 opcode(16) + shared resource(8) + REG TAG(8) 비트 추가
  if (coredynp.core_ty == Inorder)
    // [한국어] 비교 비트에 opcode + shared resource + register tag 추가
    compare_bits += 16 + 8 + 8;  // TODO: opcode bits + log(shared resources) +
                                 // REG TAG BITS-->opcode comparator
  else
    compare_bits += 16 + 8 + 8;

  conflict_check_power();  // [한국어] 비교기/충돌 검사 전력 계산
  double sckRation = g_tp.sckt_co_eff;
  power.readOp.dynamic *= sckRation;
  power.writeOp.dynamic *= sckRation;
  power.searchOp.dynamic *= sckRation;
}

/*
 * [한국어]
 * dep_resource_conflict_check::conflict_check_power - Scoreboard/CAM 스타일 충돌 검사 전력
 *
 * 비교기 수는 decodeW 기준으로 source-to-dest 2(N²-N) + dest-to-dest (N²-N)로
 * 계산한다. 각 비교기는 compare_bits×2개의 NMOS를 포함하며, 누설은
 * simplified_nmos_leakage()로, 게이트 누설은 cmos_Ig_leakage()로 산출한다.
 */
void dep_resource_conflict_check::conflict_check_power() {
  double Ctotal;
  int num_comparators;
  // [한국어] 비교기 수 = 3×(decodeW² - decodeW) (source→dest 2배 + dest→dest 1배)
  num_comparators =
      3 *
      ((coredynp.decodeW) * (coredynp.decodeW) -
       coredynp.decodeW);  // 2(N*N-N) is used for source to dest comparison,
                           // (N*N-N) is used for dest to dest comparision.
  // When decode-width ==1, no dcl logic

  Ctotal = num_comparators * compare_cap();  // [한국어] 총 부하 = 비교기 수 × 단일 비교기 용량
  // printf("%i,%s\n",XML_interface->sys.core[0].predictor.predictor_entries,XML_interface->sys.core[0].predictor.prediction_scheme);

  // [한국어] 동적 에너지 = Ctotal × Vdd² (AF/주파수는 외부 적용)
  power.readOp.dynamic =
      Ctotal * /*CLOCKRATE*/ g_tp.peri_global.Vdd * g_tp.peri_global.Vdd /*AF*/;
  power.readOp.leakage = num_comparators * compare_bits * 2 *
                         simplified_nmos_leakage(Wcompn, false);

  double long_channel_device_reduction =
      longer_channel_device_reduction(Core_device, coredynp.core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
  power.readOp.gate_leakage =
      num_comparators * compare_bits * 2 * cmos_Ig_leakage(Wcompn, 0, 2, nmos);
}

/* estimate comparator power consumption (this comparator is similar
   to the tag-match structure in a CAM */
/*
 * [한국어]
 * dep_resource_conflict_check::compare_cap - CAM/tag-매치 스타일 비교기 용량
 *
 * 비교기 하단(비교 트랜지스터)과 상단(NOR match)의 드레인/게이트 용량을 합산.
 * fan-in에 따라 큰 NOR 게이트의 WNORp를 재조정한다.
 */
double dep_resource_conflict_check::compare_cap() {
  double c1, c2;

  WNORp = WNORp * compare_bits /
          2.0;  // resize the big NOR gate at the DCL according to fan in.
  /* bottom part of comparator */
  c2 = (compare_bits) * (drain_C_(Wcompn, NCH, 1, 1, g_tp.cell_h_def) +
                         drain_C_(Wcompn, NCH, 2, 1, g_tp.cell_h_def)) +
       drain_C_(Wevalinvp, PCH, 1, 1, g_tp.cell_h_def) +
       drain_C_(Wevalinvn, NCH, 1, 1, g_tp.cell_h_def);

  /* top part of comparator */
  c1 = (compare_bits) * (drain_C_(Wcompn, NCH, 1, 1, g_tp.cell_h_def) +
                         drain_C_(Wcompn, NCH, 2, 1, g_tp.cell_h_def) +
                         drain_C_(Wcomppreequ, NCH, 1, 1, g_tp.cell_h_def)) +
       gate_C(WNORn + WNORp, 10.0) +
       drain_C_(WNORp, NCH, 2, 1, g_tp.cell_h_def) +
       compare_bits * drain_C_(WNORn, NCH, 2, 1, g_tp.cell_h_def);
  return (c1 + c2);
}

/*
 * [한국어]
 * dep_resource_conflict_check::leakage_feedback - 온도 변화에 따른 누설 재계산
 *
 * @temperature: 새 절대 온도(K). 10K 단위로 반올림하여 CACTI 파라미터 갱신.
 * @return: void — power.readOp.leakage, longer_channel_leakage, gate_leakage 갱신.
 */
void dep_resource_conflict_check::leakage_feedback(double temperature) {
  l_ip.temp = (unsigned int)round(temperature / 10.0) * 10;
  uca_org_t init_result = init_interface(&l_ip);  // init_result is dummy

  // This is part of conflict_check_power()
  int num_comparators =
      3 *
      ((coredynp.decodeW) * (coredynp.decodeW) -
       coredynp.decodeW);  // 2(N*N-N) is used for source to dest comparison,
                           // (N*N-N) is used for dest to dest comparision.
  power.readOp.leakage = num_comparators * compare_bits * 2 *
                         simplified_nmos_leakage(Wcompn, false);

  double long_channel_device_reduction =
      longer_channel_device_reduction(Core_device, coredynp.core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
  power.readOp.gate_leakage =
      num_comparators * compare_bits * 2 * cmos_Ig_leakage(Wcompn, 0, 2, nmos);
}

// TODO: add inverter and transmission gate base DFF.

/*
 * [한국어]
 * DFFCell::DFFCell - NAND2 기반 D 플립플롭 셀 생성자
 *
 * 5개의 NAND2 + 1개의 NAND3 면적으로 DFF 면적을 추정한다. cell_load는
 * 다음 단에 구동할 부하 캐패시턴스이며, compute_DFF_cell()에서 동적/정적
 * 전력을 계산한다.
 */
DFFCell::DFFCell(bool _is_dram, double _WdecNANDn, double _WdecNANDp,
                 double _cell_load, const InputParameter *configure_interface)
    : is_dram(_is_dram),
      cell_load(_cell_load),
      WdecNANDn(_WdecNANDn),
      WdecNANDp(_WdecNANDp) {  // this model is based on the NAND2 based DFF.
  l_ip = *configure_interface;
  //			area.set_area(730*l_ip.F_sz_um*l_ip.F_sz_um);
  area.set_area(
      5 * compute_gate_area(NAND, 2, WdecNANDn, WdecNANDp, g_tp.cell_h_def) +
      compute_gate_area(NAND, 2, WdecNANDn, WdecNANDn, g_tp.cell_h_def));
}

/*
 * [한국어]
 * DFFCell::fpfp_node_cap - DFF 남부 노드 총 캐패시턴스
 *
 * @fan_in, fan_out: 입력/출력 팬 수.
 * @return: 드레인 캡 + fan_out×게이트 캡.
 */
double DFFCell::fpfp_node_cap(unsigned int fan_in, unsigned int fan_out) {
  double Ctotal = 0;
  // printf("WdecNANDn = %E\n", WdecNANDn);

  /* part 1: drain cap of NAND gate */
  Ctotal += drain_C_(WdecNANDn, NCH, 2, 1, g_tp.cell_h_def, is_dram) +
            fan_in * drain_C_(WdecNANDp, PCH, 1, 1, g_tp.cell_h_def, is_dram);

  /* part 2: gate cap of NAND gates */
  Ctotal += fan_out * gate_C(WdecNANDn + WdecNANDp, 0, is_dram);

  return Ctotal;
}

/*
 * [한국어]
 * DFFCell::compute_DFF_cell - DFF 남 남부/게이트/클록/유지 전력 계산
 *
 * 6개 남(NAND2×5 + NAND3×1)의 캐패시턴스를 계산하고, 0→1 전이(switch),
 * 1 유지(keep_1), 0 유지(keep_0), 클록 충전(e_clock)별 에너지를 분리한다.
 * 정적 전력은 NAND2 5개 + NAND3 1개의 누설/게이트 누설을 합산한다.
 */
void DFFCell::compute_DFF_cell() {
  double c1, c2, c3, c4, c5, c6;
  /* node 5 and node 6 are identical to node 1 in capacitance */
  c1 = c5 = c6 = fpfp_node_cap(2, 1);
  c2 = fpfp_node_cap(2, 3);
  c3 = fpfp_node_cap(3, 2);
  c4 = fpfp_node_cap(2, 2);

  // cap-load of the clock signal in each Dff, actually the clock signal only
  // connected to one NAND2
  clock_cap = 2 * gate_C(WdecNANDn + WdecNANDp, 0, is_dram);  // [한국어] 클록이 한 NAND2의 두 입력에 연결됨
  // [한국어] 스위칭 에너지 = 6개 남 총합 × 0.5 × Vdd² + 2×외부 부하
  e_switch.readOp.dynamic += (c4 + c1 + c2 + c3 + c5 + c6 + 2 * cell_load) *
                             0.5 * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
  ;

  /* no 1/2 for e_keep and e_clock because clock signal switches twice in one
   * cycle */
  e_keep_1.readOp.dynamic += c3 * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
  e_keep_0.readOp.dynamic += c2 * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
  e_clock.readOp.dynamic +=
      clock_cap * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
  ;

  /* static power */
  e_switch.readOp.leakage +=
      (cmos_Isub_leakage(WdecNANDn, WdecNANDp, 2, nand) *
           5  // 5 NAND2 and 1 NAND3 in a DFF
       + cmos_Isub_leakage(WdecNANDn, WdecNANDn, 3, nand)) *
      g_tp.peri_global.Vdd;
  e_switch.readOp.gate_leakage +=
      (cmos_Ig_leakage(WdecNANDn, WdecNANDp, 2, nand) *
           5  // 5 NAND2 and 1 NAND3 in a DFF
       + cmos_Ig_leakage(WdecNANDn, WdecNANDn, 3, nand)) *
      g_tp.peri_global.Vdd;
  // printf("leakage =%E\n",cmos_Ileak(1, is_dram) );
}

/*
 * [한국어]
 * Pipeline::Pipeline - 파이프라인 레지스터 생성자
 *
 * 코어 파이프라인(is_core_pipeline)이면 파이프라인 단계별 비트 수를
 * CoreDynParam에서 추정(compute_stage_vector). 비코어 파이프라인이면
 * XML pipeline_stages/per_stage_vector를 직접 사용한다.
 * Embedded 프로세서가 아닐 경우 고성능 트랜지스터 폭을 사용한다.
 */
Pipeline::Pipeline(const InputParameter *configure_interface,
                   const CoreDynParam &dyn_p_, enum Device_ty device_ty_,
                   bool _is_core_pipeline, bool _is_default)
    : l_ip(*configure_interface),
      coredynp(dyn_p_),
      device_ty(device_ty_),
      is_core_pipeline(_is_core_pipeline),
      is_default(_is_default),
      num_piperegs(0.0)

{
  local_result = init_interface(&l_ip);
  // [한국어] Embedded가 아니면 고성능 트랜지스터 폭 사용
  if (!coredynp.Embedded)
    process_ind = true;
  else
    process_ind = false;
  WNANDn =
      (process_ind)
          ? 25 * l_ip.F_sz_um
          : g_tp.min_w_nmos_;  // this was  20 micron for the 0.8 micron process
  WNANDp = (process_ind)
               ? 37.5 * l_ip.F_sz_um
               : g_tp.min_w_nmos_ *
                     pmos_to_nmos_sz_ratio();  // this was  30 micron for the
                                               // 0.8 micron process
  load_per_pipeline_stage = 2 * gate_C(WNANDn + WNANDp, 0, false);
  compute();  // [한국어] 파이프라인 레지스터 수 및 전력 최종 계산
}

/*
 * [한국어]
 * Pipeline::compute - 파이프라인 레지스터 면적·전력 합산
 *
 * 단일 DFF 셀 전력을 num_piperegs로 확장. McPAT은 최악의 경우를 가정해
 * switch/keep_0/keep_1 상태를 평균(1/3)으로 처리한다. 소켓 효과와
 * 매크로 레이아웃 오버헤드를 적용한다.
 */
void Pipeline::compute() {
  compute_stage_vector();
  DFFCell pipe_reg(false, WNANDn, WNANDp, load_per_pipeline_stage, &l_ip);
  pipe_reg.compute_DFF_cell();

  double clock_power_pipereg = num_piperegs * pipe_reg.e_clock.readOp.dynamic;
  //******************pipeline power: currently, we average all the
  // possibilities of the states of DFFs in the pipeline. A better way to do it
  // is to consider the harming distance of two consecutive signals, However
  // McPAT does not have plan to do this in near future as it focuses on worst
  // case power.
  // [한국어] 파이프라인 레지스터 전력 = num_piperegs×(switch+keep0+keep1)/3 + 클록
  double pipe_reg_power =
      num_piperegs *
          (pipe_reg.e_switch.readOp.dynamic + pipe_reg.e_keep_0.readOp.dynamic +
           pipe_reg.e_keep_1.readOp.dynamic) /
          3 +
      clock_power_pipereg;
  double pipe_reg_leakage = num_piperegs * pipe_reg.e_switch.readOp.leakage;
  double pipe_reg_gate_leakage =
      num_piperegs * pipe_reg.e_switch.readOp.gate_leakage;
  power.readOp.dynamic += pipe_reg_power;
  power.readOp.leakage += pipe_reg_leakage;
  power.readOp.gate_leakage += pipe_reg_gate_leakage;
  area.set_area(num_piperegs * pipe_reg.area.get_area());  // [한국어] DFF 면적 × 레지스터 수

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, coredynp.core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;

  double sckRation = g_tp.sckt_co_eff;
  power.readOp.dynamic *= sckRation;
  power.writeOp.dynamic *= sckRation;
  power.searchOp.dynamic *= sckRation;
  double macro_layout_overhead = g_tp.macro_layout_overhead;
  if (!coredynp.Embedded)
    area.set_area(area.get_area() * macro_layout_overhead);
}

/*
 * [한국어]
 * Pipeline::compute_stage_vector - 코어 파이프라인 단계별 비트 수 추정
 *
 * Inorder는 6단계, OOO는 12단계로 고정. 각 단계를 지나는 PC, 명령어,
 * 물리/논리 레지스터 번호, opcode decode 신호(2^opcode_length) 등의 비트를
 * 더하고, 제어/인터럽트 레지스터를 50% 추가 가정한 뒤 사용자가 지정한
 * pipeline_stages로 재조정한다.
 */
void Pipeline::compute_stage_vector() {
  double num_stages, tot_stage_vector, per_stage_vector;
  int opcode_length =
      coredynp.x86 ? coredynp.micro_opcode_length : coredynp.opcode_length;
  // Hthread = thread_clock_gated? 1:num_thread;

  // [한국어] 비코어 파이프라인: 사용자가 지정한 stages × per_stage_vector
  if (!is_core_pipeline) {
    num_piperegs = l_ip.pipeline_stages *
                   l_ip.per_stage_vector;  // The number of pipeline stages are
                                           // calculated based on the achievable
                                           // throughput and required throughput
  } else {
    // [한국어] Inorder 코어: 6단계 파이프라인 (IF→ID→ThreadSEL→EXE→MEM→WB)
    if (coredynp.core_ty == Inorder) {
      /* assume 6 pipe stages and try to estimate bits per pipe stage */
      /* pipe stage 0/IF */
      num_piperegs += coredynp.pc_width * 2 * coredynp.num_hthreads;
      /* pipe stage IF/ID */
      num_piperegs += coredynp.fetchW *
                      (coredynp.instruction_length + coredynp.pc_width) *
                      coredynp.num_hthreads;
      /* pipe stage IF/ThreadSEL */
      if (coredynp.multithreaded)
        num_piperegs += coredynp.num_hthreads *
                        coredynp.perThreadState;  // 8 bit thread states
      /* pipe stage ID/EXE */
      num_piperegs += coredynp.decodeW *
                      (coredynp.instruction_length + coredynp.pc_width +
                       pow(2.0, opcode_length) + 2 * coredynp.int_data_width) *
                      coredynp.num_hthreads;
      /* pipe stage EXE/MEM */
      num_piperegs +=
          coredynp.issueW *
          (3 * coredynp.arch_ireg_width + pow(2.0, opcode_length) +
           8 * 2 * coredynp.int_data_width /*+2*powers (2,reg_length)*/);
      /* pipe stage MEM/WB the 2^opcode_length means the total decoded signal
       * for the opcode*/
      num_piperegs +=
          coredynp.issueW *
          (2 * coredynp.int_data_width + pow(2.0, opcode_length) +
           8 * 2 * coredynp.int_data_width /*+2*powers (2,reg_length)*/);
      //		/* pipe stage 5/6 */
      //		num_piperegs += issueWidth*(data_width + powers
      //(2,opcode_length)/*+2*powers (2,reg_length)*/);
      //		/* pipe stage 6/7 */
      //		num_piperegs += issueWidth*(data_width + powers
      //(2,opcode_length)/*+2*powers (2,reg_length)*/);
      //		/* pipe stage 7/8 */
      //		num_piperegs += issueWidth*(data_width + powers
      //(2,opcode_length)/**2*powers (2,reg_length)*/);
      //		/* assume 50% extra in control signals (rule of thumb)
      //*/
      num_stages = 6;

    } else {
      // [한국어] OOO 코어: 12단계 파이프라인 (Fetch→Decode→Rename→IssueQ→Dispatch→RegRead→EXE→MEM→WB→CM)
      /* assume 12 stage pipe stages and try to estimate bits per pipe stage */
      /*OOO: Fetch, decode, rename, IssueQ, dispatch, regread, EXE, MEM, WB, CM
       */

      /* pipe stage 0/1F*/
      num_piperegs +=
          coredynp.pc_width * 2 * coredynp.num_hthreads;  // PC and Next PC
      /* pipe stage IF/ID */
      num_piperegs +=
          coredynp.fetchW * (coredynp.instruction_length + coredynp.pc_width) *
          coredynp.num_hthreads;  // PC is used to feed branch predictor in ID
      /* pipe stage 1D/Renaming*/
      num_piperegs +=
          coredynp.decodeW * (coredynp.instruction_length + coredynp.pc_width) *
          coredynp.num_hthreads;  // PC is for branch exe in later stage.
      /* pipe stage Renaming/wire_drive */
      num_piperegs +=
          coredynp.decodeW * (coredynp.instruction_length + coredynp.pc_width);
      /* pipe stage Renaming/IssueQ */
      num_piperegs += coredynp.issueW *
                      (coredynp.instruction_length + coredynp.pc_width +
                       3 * coredynp.phy_ireg_width) *
                      coredynp.num_hthreads;  // 3*coredynp.phy_ireg_width means
                                              // 2 sources and 1 dest
      /* pipe stage IssueQ/Dispatch */
      num_piperegs += coredynp.issueW * (coredynp.instruction_length +
                                         3 * coredynp.phy_ireg_width);
      /* pipe stage Dispatch/EXE */

      num_piperegs += coredynp.issueW *
                      (3 * coredynp.phy_ireg_width + coredynp.pc_width +
                       pow(2.0, opcode_length) /*+2*powers (2,reg_length)*/);
      /* 2^opcode_length means the total decoded signal for the opcode*/
      num_piperegs += coredynp.issueW *
                      (2 * coredynp.int_data_width +
                       pow(2.0, opcode_length) /*+2*powers (2,reg_length)*/);
      /*2 source operands in EXE; Assume 2EXE stages* since we do not really
       * distinguish OP*/
      num_piperegs += coredynp.issueW *
                      (2 * coredynp.int_data_width +
                       pow(2.0, opcode_length) /*+2*powers (2,reg_length)*/);
      /* pipe stage EXE/MEM, data need to be read/write, address*/
      num_piperegs +=
          coredynp.issueW *
          (coredynp.int_data_width + coredynp.v_address_width +
           pow(2.0,
               opcode_length) /*+2*powers (2,reg_length)*/);  // memory Opcode
                                                              // still need to
                                                              // be passed
      /* pipe stage MEM/WB; result data, writeback regs */
      num_piperegs +=
          coredynp.issueW * (coredynp.int_data_width + coredynp.phy_ireg_width /* powers (2,opcode_length) + (2,opcode_length)+2*powers (2,reg_length)*/);
      /* pipe stage WB/CM ; result data, regs need to be updated, address for
       * resolve memory ops in ROB's top*/
      num_piperegs +=
          coredynp.commitW *
          (coredynp.int_data_width + coredynp.v_address_width + coredynp.phy_ireg_width /*+ powers (2,opcode_length)*2*powers (2,reg_length)*/) *
          coredynp.num_hthreads;
      //		if (multithreaded)
      //		{
      //
      //		}
      num_stages = 12;
    }

    /* assume 50% extra in control registers and interrupt registers (rule of
     * thumb) */
    num_piperegs = num_piperegs * 1.5;  // [한국어] 제어/인터럽트 레지스터 50% 추가
    tot_stage_vector = num_piperegs;
    per_stage_vector = tot_stage_vector / num_stages;

    if (coredynp.core_ty == Inorder) {
      if (coredynp.pipeline_stages > 6)
        num_piperegs = per_stage_vector * coredynp.pipeline_stages;
    } else  // OOO
    {
      if (coredynp.pipeline_stages > 12)
        num_piperegs = per_stage_vector * coredynp.pipeline_stages;
    }
  }
}

/*
 * [한국어]
 * FunctionalUnit::FunctionalUnit - FPU/ALU/MUL 기능 유닛 생성자
 *
 * @fu_type_: FPU, ALU, MUL 중 하나. GPU SM의 SP/DP/SFU/INT ALU 모델링에 사용.
 * @exClockRate: 실행 유닛 클록(Hz).
 * @return: (생성자) — area_t, leakage, gate_leakage, per_access_energy, base_energy 초기화.
 *
 * Embedded 프로세서와 일반(고성능) 프로세서의 면적/누설 계수를 구분한다.
 * FPU의 경우 2개 DP FPU가 1개 SP FPU로 결합되도록 num_fu/=2 처리.
 * AccelWattch GPU 모드에서는 SP/ALU/SFU 전력 상수(SP_BASE_POWER,
 * SFU_BASE_POWER)가 base_energy로 사용된다.
 */
FunctionalUnit::FunctionalUnit(ParseXML *XML_interface, int ithCore_,
                               InputParameter *interface_ip_,
                               const CoreDynParam &dyn_p_,
                               enum FU_type fu_type_, double exClockRate)
    : XML(XML_interface),
      ithCore(ithCore_),
      interface_ip(*interface_ip_),
      coredynp(dyn_p_),
      fu_type(fu_type_) {
  double area_t;  //, leakage, gate_leakage;
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  clockRate = exClockRate;  // coredynp.clockRate;
  executionTime = coredynp.executionTime;
  // cout<<"FU executionTime: "<<executionTime<<endl;

  // XML_interface=_XML_interface;
  uca_org_t result2;
  result2 = init_interface(&interface_ip);
  // [한국어] Embedded CPU 모드와 고성능 CPU/GPU 모드 분기
  if (XML->sys.Embedded) {
    if (fu_type == FPU) {
      num_fu = coredynp.num_fpus;
      // area_t = 8.47*1e6*g_tp.scaling_factor.logic_scaling_co_eff;//this is
      // um^2
      area_t = 4.47 * 1e6 *
               (g_ip->F_sz_nm * g_ip->F_sz_nm / 90.0 /
                90.0);  // this is um^2 The base number
      // 4.47 contains both VFP and NEON processing unit, VFP is about 40% and
      // NEON is about 60%
      if (g_ip->F_sz_nm > 90)
        area_t = 4.47 * 1e6 *
                 g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2
      leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
                cmos_Isub_leakage(5 * g_tp.min_w_nmos_,
                                  5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r,
                                  1, inv) *
                g_tp.peri_global.Vdd / 2;  // unit W
      gate_leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
                     cmos_Ig_leakage(
                         5 * g_tp.min_w_nmos_,
                         5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1, inv) *
                     g_tp.peri_global.Vdd / 2;  // unit W
      // energy = 0.3529/10*1e-9;//this is the energy(nJ) for a FP instruction
      // in FPU usually it can have up to 20 cycles.
      //			base_energy = coredynp.core_ty==Inorder? 0:
      // 89e-3*3; //W The base energy of ALU average numbers from Intel 4G and
      // 773Mhz (Wattch) 			base_energy
      //*=(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);
      base_energy = 0;
      per_access_energy =
          1.15 / 1e9 / 4 / 1.3 / 1.3 * g_tp.peri_global.Vdd *
          g_tp.peri_global.Vdd *
          (g_ip->F_sz_nm /
           90.0);  // g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);//0.00649*1e-9;
                   // //This is per Hz energy(nJ)
      // per_access_energy*=3;
      // FPU power from Sandia's processor sizing tech report
      FU_height =
          (18667 * num_fu) * interface_ip.F_sz_um;  // FPU from Sun's data
    // [한국어] 정수 ALU: 71.85×71.85 um² 기준 × num_fu × logic scaling
    } else if (fu_type == ALU) {
      num_fu = coredynp.num_alus;
      // FIXME: The first area_t = is from updated McAPAT, the second is from
      // our changes (conflict from base) area_t =
      // 280*260*g_tp.scaling_factor.logic_scaling_co_eff;//this is um^2 ALU +
      // MUl
      area_t =
          71.85 * 71.85 * num_fu *
          g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
      leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
                cmos_Isub_leakage(20 * g_tp.min_w_nmos_,
                                  20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r,
                                  1, inv) *
                g_tp.peri_global.Vdd / 2;  // unit W
      gate_leakage =
          area_t * (g_tp.scaling_factor.core_tx_density) *
          cmos_Ig_leakage(20 * g_tp.min_w_nmos_,
                          20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1,
                          inv) *
          g_tp.peri_global.Vdd / 2;
      leakage = 0;
      //			base_energy = coredynp.core_ty==Inorder?
      // 0:89e-3; //W The base energy of ALU average numbers from Intel 4G and
      // 773Mhz (Wattch) 			base_energy
      //*=(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);
      base_energy = 0;
      // per_access_energy
      // = 1.15/3/1e9/4/1.3/1.3*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd*(g_ip->F_sz_nm/90.0);//(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);//0.00649*1e-9;
      // //This is per cycle energy(nJ)
      per_access_energy = 1.29 / 1e12 / 1.3 / 1.3 * g_tp.peri_global.Vdd *
                          g_tp.peri_global.Vdd * (g_ip->F_sz_nm / 90.0);
      // per_access_energy*=3;
      FU_height = (6222 * num_fu) * interface_ip.F_sz_um;  // integer ALU

    // [한국어] 곱셈/나눗셈 유닛: divider/mul Sun 데이터 기반
    } else if (fu_type == MUL) {
      num_fu = coredynp.num_muls;
      area_t =
          280 * 260 * 3 *
          g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
      leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
                cmos_Isub_leakage(20 * g_tp.min_w_nmos_,
                                  20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r,
                                  1, inv) *
                g_tp.peri_global.Vdd / 2;  // unit W
      gate_leakage =
          area_t * (g_tp.scaling_factor.core_tx_density) *
          cmos_Ig_leakage(20 * g_tp.min_w_nmos_,
                          20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1,
                          inv) *
          g_tp.peri_global.Vdd / 2;
      //			base_energy = coredynp.core_ty==Inorder?
      // 0:89e-3*2; //W The base energy of ALU average numbers from Intel 4G and
      // 773Mhz (Wattch) 			base_energy
      //*=(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);
      base_energy = 0;
      per_access_energy =
          1.15 * 2 / 3 / 1e9 / 1.3 / 1.3 * g_tp.peri_global.Vdd *
          g_tp.peri_global.Vdd *
          (g_ip->F_sz_nm /
           90.0);  //(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2)/24;//0.00649*1e-9;
                   ////This is per cycle energy(nJ), coefficient based on Wattch
                   //(24 is the division ny latency: Syed)
      // per_access_energy*=3;
      FU_height = (9334 * num_fu) *
                  interface_ip.F_sz_um;  // divider/mul from Sun's data
    } else {
      cout << "Unknown Functional Unit Type" << endl;
      exit(0);
    }
    per_access_energy *= 0.5;  // According to ARM data embedded processor has
                               // much lower per acc energy
  }                            /* if (XML->sys.Embedded) */
  else {
    if (fu_type == FPU) {
      num_fu = coredynp.num_fpus;

      /*
      num_fu/=2; //2 DP FPUs combine to for a SP FPU
      //area_t = 8.47*1e6*g_tp.scaling_factor.logic_scaling_co_eff;//this is
      um^2 area_t = 8.47*1e6*(g_ip->F_sz_nm*g_ip->F_sz_nm/90.0/90.0);//this is
      um^2 if (g_ip->F_sz_nm>90) area_t
      = 8.47*1e6*g_tp.scaling_factor.logic_scaling_co_eff;//this is um^2 leakage
      = area_t
      *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(5*g_tp.min_w_nmos_,
      5*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      inv)*g_tp.peri_global.Vdd/2;//unit W gate_leakage = area_t
      *(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(5*g_tp.min_w_nmos_,
      5*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      inv)*g_tp.peri_global.Vdd/2;//unit W
      //energy = 0.3529/10*1e-9;//this is the energy(nJ) for a FP instruction in
      FPU usually it can have up to 20 cycles. base_energy =
      coredynp.core_ty==Inorder? 0: 89e-3*3; //W The base energy of ALU average
      numbers from Intel 4G and 773Mhz (Wattch) base_energy
      *=(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2); per_access_energy
      = 1.15*3/1e9/4/1.3/1.3*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd*(g_ip->F_sz_nm/90.0);//g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);//0.00649*1e-9;
      //This is per op energy(nJ)
      FU_height=(38667*num_fu)*interface_ip.F_sz_um;//FPU from Sun's data
      */
      // area_t = 8.47*1e6*g_tp.scaling_factor.logic_scaling_co_eff;//this is
      // um^2
      num_fu = num_fu / 2;  // 2 DP FPUs combine to for a SP FPU
      area_t = 8.47 * 1e6 *
               (g_ip->F_sz_nm * g_ip->F_sz_nm / 90.0 / 90.0);  // this is um^2
      if (g_ip->F_sz_nm > 90)
        area_t = 8.47 * 1e6 *
                 g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2
      leakage =
          37e-3;  // area_t
                  // *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(5*g_tp.min_w_nmos_,
                  // 5*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
                  // inv)*g_tp.peri_global.Vdd/2;//unit W
      gate_leakage =
          0;  // area_t
              // *(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(5*g_tp.min_w_nmos_,
              // 5*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
              // inv)*g_tp.peri_global.Vdd/2;//unit W
      // energy = 0.3529/10*1e-9;//this is the energy(nJ) for a FP instruction
      // in FPU usually it can have up to 20 cycles.
      base_energy =
          coredynp.core_ty == Inorder
              ? 0
              : 89e-3 * 3;  // W The base energy of ALU average numbers from
                            // Intel 4G and 773Mhz (Wattch)

      base_energy *= (g_tp.peri_global.Vdd * g_tp.peri_global.Vdd / 1.2 / 1.2);

      // Base energy (if the pipeline is not clock gated)
      // TODO: add a check for clockgating enable
      base_energy = SP_BASE_POWER;

      // per_access_energy
      // = 1.15*3/1e9/4/1.3/1.3*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd*(g_ip->F_sz_nm/90.0);//g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);//0.00649*1e-9;
      // //This is per op energy(nJ)
      per_access_energy =
          3.9 * 14.91 / 1e12 / 1.08 / 1.08 * g_tp.peri_global.Vdd *
          g_tp.peri_global.Vdd *
          (g_ip->F_sz_nm /
           90.0);  //;4.34 is scaling factor based on hardware measurements
                   // ALU instrucitons are also executed on FPUs so add 30%
                   // overhead for supporting ALU instrcutions per_access_energy
                   // = 1.3*per_access_energy;

      // ALU instrucitons are also executed on FPUs so add 10% overhead for
      // supporting ALU instrcutions
      leakage = 1.1 * leakage;
      // cout<<"FPU Per access erngy: "<<per_access_energy/2;
      // Divide per access energy by 2 so we have 4 DP units capapble of
      // doing 8 SP operations
      // per_access_energy = per_access_energy/2;
      FU_height =
          (38667 * num_fu) * interface_ip.F_sz_um;  // FPU from Sun's data
      per_access_energy *= 2;
    } else if (fu_type == ALU) {
      num_fu = coredynp.num_alus;
      // FIXME: The first area_t = is from updated McPAT. The second is from our
      // McPAT changes. Fix after merge area_t =
      // 280*260*2*g_tp.scaling_factor.logic_scaling_co_eff;//this is um^2 ALU +
      // MUl leakage = area_t
      // *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(20*g_tp.min_w_nmos_,
      // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      // inv)*g_tp.peri_global.Vdd/2;//unit W gate_leakage =
      // area_t*(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(20*g_tp.min_w_nmos_,
      // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      // inv)*g_tp.peri_global.Vdd/2;
      area_t =
          71.85 * 71.85 * num_fu *
          g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
      // leakage = area_t
      // *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(20*g_tp.min_w_nmos_,
      // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      // inv)*g_tp.peri_global.Vdd/2;//unit W gate_leakage =
      // area_t*(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(20*g_tp.min_w_nmos_,
      // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      // inv)*g_tp.peri_global.Vdd/2; Following leakage numbers are based on
      // Syntheis based power-estimation (SYED)
      leakage = 2.58e-5;
      gate_leakage = 0;
      base_energy = coredynp.core_ty == Inorder
                        ? 0
                        : 89e-3;  // W The base energy of ALU average numbers
                                  // from Intel 4G and 773Mhz (Wattch)
      base_energy *= (g_tp.peri_global.Vdd * g_tp.peri_global.Vdd / 1.2 / 1.2);
      // per_access_energy
      // = 1.15/1e9/4/1.3/1.3*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd*(g_ip->F_sz_nm/90.0);//(g_tp.peri_global.Vdd*g_tp.peri_global.Vdd/1.2/1.2);//0.00649*1e-9;
      // //This is per cycle energy(nJ)
      per_access_energy = 0.8 * 1.29 / 1e12 / 1.3 / 1.3 * g_tp.peri_global.Vdd *
                          g_tp.peri_global.Vdd * (g_ip->F_sz_nm / 90.0);
      FU_height = (6222 * num_fu) * interface_ip.F_sz_um;  // integer ALU
      per_access_energy *= 2;
    } else if (fu_type == MUL) {
      num_fu = coredynp.num_muls;
      area_t =
          280 * 260 * 2 * 3 *
          g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
      // leakage = area_t
      // *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(20*g_tp.min_w_nmos_,
      // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
      // inv)*g_tp.peri_global.Vdd/2;//unit W
      leakage = 37e-3;
      gate_leakage =
          0;  // area_t*(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(20*g_tp.min_w_nmos_,
              // 20*g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1,
              // inv)*g_tp.peri_global.Vdd/2;
      base_energy =
          coredynp.core_ty == Inorder
              ? 0
              : 89e-3 * 2;  // W The base energy of ALU average numbers from
                            // Intel 4G and 773Mhz (Wattch)
      base_energy *= (g_tp.peri_global.Vdd * g_tp.peri_global.Vdd / 1.2 / 1.2);
      base_energy = SFU_BASE_POWER;
      // SFU is modelled as a double preicison FPU
      per_access_energy =
          8 * 14.91 / 1e12 / 1.08 / 1.08 * g_tp.peri_global.Vdd *
          g_tp.peri_global.Vdd *
          (g_ip->F_sz_nm /
           90.0);  // 1.5 is scaling factor based on hardware measuremetns
      FU_height = (9334 * num_fu) *
                  interface_ip.F_sz_um;  // divider/mul from Sun's data
      per_access_energy *= 2;
    }

    else {
      cout << "Unknown Functional Unit Type" << endl;
      exit(0);
    }
  }
  // IEXEU, simple ALU and FPU
  //  double C_ALU, C_EXEU, C_FPU; //Lum Equivalent capacitance of IEXEU and
  //  FPU. Based on Intel and Sun 90nm process fabracation.
  //
  //  C_ALU	  = 0.025e-9;//F
  //  C_EXEU  = 0.05e-9; //F
  //  C_FPU	  = 0.35e-9;//F
  area.set_area(area_t * num_fu);  // [한국어] 단위 FU 면적 × FU 개수
  leakage *= num_fu;  // [한국어] 단위 누설 × FU 개수
  gate_leakage *= num_fu;
  double macro_layout_overhead = g_tp.macro_layout_overhead;
  //	if (!XML->sys.Embedded)
  area.set_area(area.get_area() * macro_layout_overhead);
}

/*
 * [한국어]
 * FunctionalUnit::computeEnergy - TDP/런타임 FU 전력 계산
 *
 * @is_tdp: true이면 TDP(peak) 모드, false이면 런타임 모드.
 * @return: void — power/rt_power에 저장.
 *
 * TDP 모드에서는 FU 개수(num_fu)와 duty_cycle을, 런타임 모드에서는
 * XML->sys.core[ithCore].{fpu,ialu,mul}_accesses를 사용. GPU 모드에서
 * inactive lane 보정을 위해 FPU/SFU에 base_energy×(32-active_lanes)를
 * 추가한다.
 */
void FunctionalUnit::computeEnergy(bool is_tdp) {
  executionTime =
      XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);  // Syed
  double pppm_t[4] = {1, 1, 1, 1};
  double FU_duty_cycle;
  if (is_tdp) {
    set_pppm(pppm_t, 2, 2, 2, 2);  // [한국어] 정수 명령당 2개 소스 오퍼랜드 전달 가정  // 2 means two source operands needs to be
                                   // passed for each int instruction.
    if (fu_type == FPU) {
      stats_t.readAc.access = num_fu;
      tdp_stats = stats_t;
      // Syed: FPU power numbers are already average
      // so activity factor is already accounted for
      FU_duty_cycle = coredynp.FPU_duty_cycle;  // [한국어] FPU 활성률
    } else if (fu_type == ALU) {
      stats_t.readAc.access = 1 * num_fu;
      tdp_stats = stats_t;
      FU_duty_cycle = coredynp.ALU_duty_cycle;
    } else if (fu_type == MUL) {
      stats_t.readAc.access = num_fu;
      tdp_stats = stats_t;
      FU_duty_cycle = coredynp.MUL_duty_cycle;
    }

    // power.readOp.dynamic = base_energy/clockRate +
    // energy*stats_t.readAc.access;
    // [한국어] 피크 동적 전력 = per_access_energy×접근수 + base_energy/clockRate
    power.readOp.dynamic =
        per_access_energy * stats_t.readAc.access + base_energy / clockRate;
    double sckRation = g_tp.sckt_co_eff;
    power.readOp.dynamic *= sckRation * FU_duty_cycle;
    power.writeOp.dynamic *= sckRation;
    power.searchOp.dynamic *= sckRation;

    power.readOp.leakage = leakage;
    power.readOp.gate_leakage = gate_leakage;
    double long_channel_device_reduction =
        longer_channel_device_reduction(Core_device, coredynp.core_ty);
    power.readOp.longer_channel_leakage =
        power.readOp.leakage * long_channel_device_reduction;

  } else {
    if (fu_type == FPU) {
      // Each access activates an equililant of a double-precision unit
      // so divide accesses into half
      // [한국어] GPU FPU 실제 접근 수 → inactive lane 보정에 사용
      stats_t.readAc.access = XML->sys.core[ithCore].fpu_accesses;
      rtp_stats = stats_t;
      // cout<<"FPU: --accesses "<<stats_t.readAc.access <<endl;

    } else if (fu_type == ALU) {
      // [한국어] GPU/CPU 정수 ALU 실제 접근 수
      stats_t.readAc.access = XML->sys.core[ithCore].ialu_accesses;
      rtp_stats = stats_t;
      // cout<<"ALU: --accesses "<<stats_t.readAc.access <<endl;
    } else if (fu_type == MUL) {
      // [한국어] GPU/CPU 곱셈 유닛 실제 접근 수
      stats_t.readAc.access = XML->sys.core[ithCore].mul_accesses;
      rtp_stats = stats_t;
      // cout<<"MUL: --accesses "<<stats_t.readAc.access <<endl;
    }

    // rt_power.readOp.dynamic = base_energy*executionTime +
    // energy*stats_t.readAc.access;

    // [한국어] ALU 런타임 전력: per_access_energy×access + base_energy×실행시간
    if (fu_type == ALU) {
      rt_power.readOp.dynamic = per_access_energy * stats_t.readAc.access +
                                base_energy * executionTime;
    } else {
      rt_power.readOp.dynamic = per_access_energy * stats_t.readAc.access;
    }

    double sckRation = g_tp.sckt_co_eff;
    rt_power.readOp.dynamic *= sckRation;
    rt_power.writeOp.dynamic *= sckRation;
    rt_power.searchOp.dynamic *= sckRation;
    // cout<<"Power: "<<rt_power.readOp.dynamic<<endl;
    if (fu_type == FPU) {
      // [한국어] FPU inactive lane 보정: 32개 lane 중 비활성 lane만큼 base_energy 추가
      rt_power.readOp.dynamic +=
          base_energy * executionTime *
          (32 - XML->sys.core[ithCore].sp_average_active_lanes);
    }
    if (fu_type == MUL) {
      // [한국어] SFU inactive lane 보정
      if (XML->sys.core[ithCore].sfu_average_active_lanes >= 1)
        rt_power.readOp.dynamic +=
            base_energy * executionTime *
            (32 - XML->sys.core[ithCore].sfu_average_active_lanes);
    }

  } /* else */
}

/*
 * [한국어]
 * FunctionalUnit::displayEnergy - FPU/ALU/MUL 전력·면적 출력
 *
 * fu_type에 따라 "Floating Point Units", "Integer ALUs", "Complex ALUs"
 * 헤더를 출력하고 면적, 피크 동적 전력, 누설, 게이트 누설, 런타임 전력을
 * 보여준다.
 */
void FunctionalUnit::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  bool long_channel = XML->sys.longer_channel_device;

  //	cout << indent_str_next << "Results Broadcast Bus Area = " <<
  // bypass->area.get_area() *1e-6 << " mm^2" << endl;
  if (is_tdp) {
    if (fu_type == FPU) {
      cout << indent_str
           << "Floating Point Units (FPUs) (Count: " << coredynp.num_fpus
           << " ):" << endl;
      cout << indent_str_next << "Area = " << area.get_area() * 1e-6 << " mm^2"
           << endl;
      cout << indent_str_next
           << "Peak Dynamic = " << power.readOp.dynamic * clockRate << " W"
           << endl;
      cout << indent_str_next << "Peak Energy = " << power.readOp.dynamic
           << " J" << endl;

      //			cout << indent_str_next << "Subthreshold Leakage
      //= " << power.readOp.leakage  << " W" << endl; cout <<"clock:
      // "<<clockRate<<endl;
      cout << indent_str_next << "Subthreshold Leakage = "
           << (long_channel ? power.readOp.longer_channel_leakage
                            : power.readOp.leakage)
           << " W" << endl;
      cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage
           << " W" << endl;
      cout << indent_str_next
           << "Runtime Dynamic = " << rt_power.readOp.dynamic / executionTime
           << " W" << endl;
      cout << endl;
    } else if (fu_type == ALU) {
      cout << indent_str << "Integer ALUs (Count: " << coredynp.num_alus
           << " ):" << endl;
      cout << indent_str_next << "Area = " << area.get_area() * 1e-6 << " mm^2"
           << endl;
      cout << indent_str_next
           << "Peak Dynamic = " << power.readOp.dynamic * clockRate << " W"
           << endl;
      //			cout << indent_str_next << "Subthreshold Leakage
      //= " << power.readOp.leakage  << " W" << endl;
      cout << indent_str_next << "Subthreshold Leakage = "
           << (long_channel ? power.readOp.longer_channel_leakage
                            : power.readOp.leakage)
           << " W" << endl;
      cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage
           << " W" << endl;
      cout << indent_str_next
           << "Runtime Dynamic = " << rt_power.readOp.dynamic / executionTime
           << " W" << endl;
      cout << endl;
    } else if (fu_type == MUL) {
      cout << indent_str
           << "Complex ALUs (Mul/Div) (Count: " << coredynp.num_muls
           << " ):" << endl;
      cout << indent_str_next << "Area = " << area.get_area() * 1e-6 << " mm^2"
           << endl;
      cout << indent_str_next
           << "Peak Dynamic = " << power.readOp.dynamic * clockRate << " W"
           << endl;
      //			cout << indent_str_next << "Subthreshold Leakage
      //= " << power.readOp.leakage  << " W" << endl;
      cout << indent_str_next << "Subthreshold Leakage = "
           << (long_channel ? power.readOp.longer_channel_leakage
                            : power.readOp.leakage)
           << " W" << endl;
      cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage
           << " W" << endl;
      cout << indent_str_next
           << "Runtime Dynamic = " << rt_power.readOp.dynamic / executionTime
           << " W" << endl;
      cout << endl;
    }

  } else {
  }
}

/*
 * [한국어]
 * FunctionalUnit::leakage_feedback - 온도 변화에 따른 FU 누설 재계산
 *
 * Embedded 모드와 유사한 면적 기준으로 FPU/ALU/MUL leakage/gate_leakage를
 * 다시 계산하고 power에 저장한다.
 */
void FunctionalUnit::leakage_feedback(double temperature) {
  // Update the temperature and initialize the global interfaces.
  interface_ip.temp = (unsigned int)round(temperature / 10.0) * 10;

  uca_org_t init_result =
      init_interface(&interface_ip);  // init_result is dummy

  // This is part of FunctionalUnit()
  double area_t, leakage, gate_leakage;
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();

  if (fu_type == FPU) {
    area_t = 4.47 * 1e6 *
             (g_ip->F_sz_nm * g_ip->F_sz_nm / 90.0 /
              90.0);  // this is um^2 The base number
    if (g_ip->F_sz_nm > 90)
      area_t = 4.47 * 1e6 *
               g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2
    leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
              cmos_Isub_leakage(5 * g_tp.min_w_nmos_,
                                5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1,
                                inv) *
              g_tp.peri_global.Vdd / 2;  // unit W
    gate_leakage =
        area_t * (g_tp.scaling_factor.core_tx_density) *
        cmos_Ig_leakage(5 * g_tp.min_w_nmos_,
                        5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1, inv) *
        g_tp.peri_global.Vdd / 2;  // unit W
  } else if (fu_type == ALU) {
    area_t =
        280 * 260 * 2 * num_fu *
        g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
    leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
              cmos_Isub_leakage(20 * g_tp.min_w_nmos_,
                                20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r,
                                1, inv) *
              g_tp.peri_global.Vdd / 2;  // unit W
    gate_leakage =
        area_t * (g_tp.scaling_factor.core_tx_density) *
        cmos_Ig_leakage(20 * g_tp.min_w_nmos_,
                        20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1, inv) *
        g_tp.peri_global.Vdd / 2;
  } else if (fu_type == MUL) {
    area_t =
        280 * 260 * 2 * 3 * num_fu *
        g_tp.scaling_factor.logic_scaling_co_eff;  // this is um^2 ALU + MUl
    leakage = area_t * (g_tp.scaling_factor.core_tx_density) *
              cmos_Isub_leakage(20 * g_tp.min_w_nmos_,
                                20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r,
                                1, inv) *
              g_tp.peri_global.Vdd / 2;  // unit W
    gate_leakage =
        area_t * (g_tp.scaling_factor.core_tx_density) *
        cmos_Ig_leakage(20 * g_tp.min_w_nmos_,
                        20 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1, inv) *
        g_tp.peri_global.Vdd / 2;
  } else {
    cout << "Unknown Functional Unit Type" << endl;
    exit(1);
  }

  power.readOp.leakage = leakage * num_fu;
  power.readOp.gate_leakage = gate_leakage * num_fu;
  power.readOp.longer_channel_leakage =
      longer_channel_device_reduction(Core_device, coredynp.core_ty);
}

/*
 * [한국어]
 * UndiffCore::UndiffCore - 미분화 코어(undifferentiated core) 전력 모델
 *
 * 코어에서 ALU/FPU/레지스터 파일 등으로 명시 모델링되지 않는 나머지
 * 로직(front-end, OoO 스케줄링, 제어 로직 등)의 면적과 누설 전력을
 * 다항식/로그 피팅(Niagara, Merom, Penryn 등)으로 추정한다.
 * 존재하지 않는 코어(exist==false)면 즉시 리턴.
 */
UndiffCore::UndiffCore(ParseXML *XML_interface, int ithCore_,
                       InputParameter *interface_ip_,
                       const CoreDynParam &dyn_p_, bool exist_, bool embedded_)
    : XML(XML_interface),
      ithCore(ithCore_),
      interface_ip(*interface_ip_),
      coredynp(dyn_p_),
      core_ty(coredynp.core_ty),
      embedded(XML->sys.Embedded),
      pipeline_stage(coredynp.pipeline_stages),
      num_hthreads(coredynp.num_hthreads),
      issue_width(coredynp.issueW),
      exist(exist_)
// is_default(_is_default)
{
  if (!exist) return;  // [한국어] 해당 코어가 비활성이면 전력 계산 생략
  double undifferentiated_core = 0;
  double core_tx_density = 0;
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  double undifferentiated_core_coe;
  // XML_interface=_XML_interface;
  uca_org_t result2;
  result2 = init_interface(&interface_ip);

  // Compute undifferentiated core area at 90nm.
  // [한국어] 고성능 코어: 다이 측정 기반 피팅 공식
  if (embedded == false) {
    // Based on the results of polynomial/log curve fitting based on
    // undifferentiated core of Niagara, Niagara2, Merom, Penyrn, Prescott,
    // Opteron die measurements
    // [한국어] OOO 코어: pipeline_stage에 대한 로그 피팅
    if (core_ty == OOO) {
      // undifferentiated_core = (0.0764*pipeline_stage*pipeline_stage
      // -2.3685*pipeline_stage + 10.405);//OOO
      undifferentiated_core = (3.57 * log(pipeline_stage) - 1.2643) > 0
                                  ? (3.57 * log(pipeline_stage) - 1.2643)
                                  : 0;
    // [한국어] Inorder 코어: pipeline_stage에 대한 로그 피팅
    } else if (core_ty == Inorder) {
      // undifferentiated_core = (0.1238*pipeline_stage + 7.2572)*0.9;//inorder
      undifferentiated_core = (-2.19 * log(pipeline_stage) + 6.55) > 0
                                  ? (-2.19 * log(pipeline_stage) + 6.55)
                                  : 0;
    } else {
      cout << "invalid core type" << endl;
      exit(0);
    }
    undifferentiated_core *= (1 + logtwo(num_hthreads) * 0.0716);  // [한국어] 하드웨어 스레드 수에 따른 면적 증가
  } else {
    // Based on the results in paper "parametrized processor models" Sandia Labs
    if (XML->sys.opt_clockrate)
      undifferentiated_core_coe = 0.05;
    else
      undifferentiated_core_coe = 0;
    undifferentiated_core =
        (0.4109 * pipeline_stage - 0.776) * undifferentiated_core_coe;
    undifferentiated_core *= (1 + logtwo(num_hthreads) * 0.0426);
  }

  // [한국어] mm² → um² 변환 및 논리 면적 스케일링
  undifferentiated_core *= g_tp.scaling_factor.logic_scaling_co_eff *
                           1e6;  // change from mm^2 to um^2
  core_tx_density = g_tp.scaling_factor.core_tx_density;
  // undifferentiated_core 		    = 3*1e6;
  // undifferentiated_core			*=
  // g_tp.scaling_factor.logic_scaling_co_eff;//(g_ip->F_sz_um*g_ip->F_sz_um/0.09/0.09)*;
  // [한국어] 미분화 코어 누설 = 면적 × 트랜지스터 밀도 × 단위 누설 × Vdd
  power.readOp.leakage =
      undifferentiated_core *
      (core_tx_density)*cmos_Isub_leakage(
          5 * g_tp.min_w_nmos_, 5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1,
          inv) *
      g_tp.peri_global.Vdd;  // unit W
  power.readOp.gate_leakage =
      undifferentiated_core *
      (core_tx_density)*cmos_Ig_leakage(
          5 * g_tp.min_w_nmos_, 5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r, 1,
          inv) *
      g_tp.peri_global.Vdd;

  double long_channel_device_reduction =
      longer_channel_device_reduction(Core_device, coredynp.core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
  area.set_area(undifferentiated_core);

  scktRatio = g_tp.sckt_co_eff;  // [한국어] 소켓 오버헤드
  power.readOp.dynamic *= scktRatio;
  power.writeOp.dynamic *= scktRatio;
  power.searchOp.dynamic *= scktRatio;
  macro_PR_overhead = g_tp.macro_layout_overhead;
  area.set_area(area.get_area() * macro_PR_overhead);

  //		double vt=g_tp.peri_global.Vth;
  //		double velocity_index=1.1;
  //		double c_in=gate_C(g_tp.min_w_nmos_,
  // g_tp.min_w_nmos_*pmos_to_nmos_sizing_r , 0.0, false); 		double
  // c_out= drain_C_(g_tp.min_w_nmos_, NCH, 2, 1, g_tp.cell_h_def, false) +
  // drain_C_(g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, PCH, 1, 1,
  // g_tp.cell_h_def, false) + c_in; 		double w_nmos=g_tp.min_w_nmos_;
  // double w_pmos=g_tp.min_w_nmos_*pmos_to_nmos_sizing_r; 		double
  // i_on_n=1.0; 		double
  // i_on_p=1.0; 		double i_on_n_in=1.0; 		double
  // i_on_p_in=1; double vdd=g_tp.peri_global.Vdd;

  //		power.readOp.sc=shortcircuit_simple(vt, velocity_index, c_in,
  // c_out, w_nmos,w_pmos, i_on_n, i_on_p,i_on_n_in, i_on_p_in, vdd);
  //		power.readOp.dynamic=c_out*vdd*vdd/2;

  //		cout<<power.readOp.dynamic << "dynamic" <<endl;
  //		cout<<power.readOp.sc << "sc" << endl;

  //		power.readOp.sc=shortcircuit(vt, velocity_index, c_in, c_out,
  // w_nmos,w_pmos, i_on_n, i_on_p,i_on_n_in, i_on_p_in, vdd);
  //		power.readOp.dynamic=c_out*vdd*vdd/2;
  //
  //		cout<<power.readOp.dynamic << "dynamic" <<endl;
  //		cout<<power.readOp.sc << "sc" << endl;
}

/*
 * [한국어]
 * UndiffCore::displayEnergy - 미분화 코어 전력·면적 출력
 *
 * TDP와 런타임 모드 모두에서 area, peak dynamic, leakage, gate leakage를
 * 출력한다.
 */
void UndiffCore::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  bool long_channel = XML->sys.longer_channel_device;

  if (is_tdp) {
    cout << indent_str << "UndiffCore:" << endl;
    cout << indent_str_next << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str_next
         << "Peak Dynamic = " << power.readOp.dynamic * clockRate << " W"
         << endl;
    // cout << indent_str_next << "Subthreshold Leakage = " <<
    // power.readOp.leakage <<" W" << endl;
    cout << indent_str_next << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage
         << " W" << endl;
    // cout << indent_str_next << "Runtime Dynamic = " <<
    // rt_power.readOp.dynamic/executionTime << " W" << endl;
    cout << endl;
  } else {
    cout << indent_str << "UndiffCore:" << endl;
    cout << indent_str_next << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str_next
         << "Peak Dynamic = " << power.readOp.dynamic * clockRate << " W"
         << endl;
    cout << indent_str_next << "Subthreshold Leakage = " << power.readOp.leakage
         << " W" << endl;
    cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage
         << " W" << endl;
    // cout << indent_str_next << "Runtime Dynamic = " <<
    // rt_power.readOp.dynamic/executionTime << " W" << endl;
    cout << endl;
  }
}

/*
 * [한국어]
 * inst_decoder::inst_decoder - 명령어 디코더 생성자
 *
 * RISC 디코더는 n-to-2^n 디코더로 근사하고, x86 CISC 디코더는 시퀀서가
 * 2회 통과(squencer_passes=2)하도록 모델링한다. opcode_length가 18비트를
 * 초과하면 세그먼트를 분할(num_decoder_segments)하고, Decoder/Predec 객체를
 * 생성하여 면적/전력을 계산한다.
 */
inst_decoder::inst_decoder(bool _is_default,
                           const InputParameter *configure_interface,
                           int opcode_length_, int num_decoders_, bool x86_,
                           enum Device_ty device_ty_, enum Core_type core_ty_)
    : is_default(_is_default),
      opcode_length(opcode_length_),
      num_decoders(num_decoders_),
      x86(x86_),
      device_ty(device_ty_),
      core_ty(core_ty_) {
  /*
   * Instruction decoder is different from n to 2^n decoders
   * that are commonly used in row decoders in memory arrays.
   * The RISC instruction decoder is typically a very simple device.
   * We can decode an instruction by simply
   * separating the machine word into small parts using wire slices
   * The RISC instruction decoder can be approximate by the n to 2^n decoders,
   * although this approximation usually underestimate power since each decoded
   * instruction normally has more than 1 active signal.
   *
   * However, decoding a CISC instruction word is much more difficult
   * than the RISC case. A CISC decoder is typically set up as a state machine.
   * The machine reads the opcode field to determine
   * what type of instruction it is,
   * and where the other data values are.
   * The instruction word is read in piece by piece,
   * and decisions are made at each stage as to
   * how the remainder of the instruction word will be read.
   * (sequencer and ROM are usually needed)
   * An x86 decoder can be even more complex since
   * it involve  both decoding instructions into u-ops and
   * merge u-ops when doing micro-ops fusion.
   */
  bool is_dram = false;
  double pmos_to_nmos_sizing_r;
  double load_nmos_width, load_pmos_width;
  double C_driver_load, R_wire_load;
  Area cell;

  l_ip = *configure_interface;
  local_result = init_interface(&l_ip);
  cell.h = g_tp.cell_h_def;
  cell.w = g_tp.cell_h_def;

  num_decoder_segments = (int)ceil(opcode_length / 18.0);  // [한국어] 18비트 이상이면 디코더 세그먼트 분할
  if (opcode_length > 18) opcode_length = 18;
  num_decoded_signals = (int)pow(2.0, opcode_length);  // [한국어] 디코딩된 신호 수 = 2^opcode_length
  pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  load_nmos_width = g_tp.max_w_nmos_ / 2;
  load_pmos_width = g_tp.max_w_nmos_ * pmos_to_nmos_sizing_r;
  // [한국어] 디코더 구동 부하: 1024개 게이트로 가정 (TODO: 재검토 필요)
  C_driver_load =
      1024 * gate_C(load_nmos_width + load_pmos_width, 0,
                    is_dram);  // TODO: this number 1024 needs to be revisited
  R_wire_load = 3000 * l_ip.F_sz_um * g_tp.wire_outside_mat.R_per_um;  // [한국어] 외부 와이어 저항

  final_dec = new Decoder(num_decoded_signals, false, C_driver_load,
                          R_wire_load, false /*is_fa*/, false /*is_dram*/,
                          false /*wl_tr*/,  // to use peri device
                          cell);

  PredecBlk *predec_blk1 =
      new PredecBlk(num_decoded_signals, final_dec,
                    0,  // Assuming predec and dec are back to back
                    0,
                    1,  // Each Predec only drives one final dec
                    false /*is_dram*/, true);
  PredecBlk *predec_blk2 =
      new PredecBlk(num_decoded_signals, final_dec,
                    0,  // Assuming predec and dec are back to back
                    0,
                    1,  // Each Predec only drives one final dec
                    false /*is_dram*/, false);

  PredecBlkDrv *predec_blk_drv1 = new PredecBlkDrv(0, predec_blk1, false);
  PredecBlkDrv *predec_blk_drv2 = new PredecBlkDrv(0, predec_blk2, false);

  pre_dec = new Predec(predec_blk_drv1, predec_blk_drv2);

  // [한국어] 최종 디코더 면적 = 단위 면적 × 신호 수 × 세그먼트 × 디코더 수
  double area_decoder = final_dec->area.get_area() * num_decoded_signals *
                        num_decoder_segments * num_decoders;
  // double w_decoder    = area_decoder / area.get_h();
  double area_pre_dec =
      (predec_blk_drv1->area.get_area() + predec_blk_drv2->area.get_area() +
       predec_blk1->area.get_area() + predec_blk2->area.get_area()) *
      num_decoder_segments * num_decoders;
  area.set_area(area.get_area() + area_decoder + area_pre_dec);
  double macro_layout_overhead = g_tp.macro_layout_overhead;
  double chip_PR_overhead = g_tp.chip_layout_overhead;
  area.set_area(area.get_area() * macro_layout_overhead * chip_PR_overhead);

  inst_decoder_delay_power();  // [한국어] 디코더 동적/누설 전력 산출

  double sckRation = g_tp.sckt_co_eff;
  power.readOp.dynamic *= sckRation;
  power.writeOp.dynamic *= sckRation;
  power.searchOp.dynamic *= sckRation;

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
}

/*
 * [한국어]
 * inst_decoder::inst_decoder_delay_power - 최종/사전 디코더 전력 합산
 *
 * set_pppm()으로 Predec/Final Decoder의 접근 패턴(ppcm)을 조정한 뒤
 * power에 더한다. x86은 squencer_passes=2.
 */
void inst_decoder::inst_decoder_delay_power() {
  double pppm_t[4] = {1, 1, 1, 1};
  double squencer_passes = x86 ? 2 : 1;

  set_pppm(pppm_t, squencer_passes * num_decoder_segments, num_decoder_segments,
           squencer_passes * num_decoder_segments, num_decoder_segments);
  power = power + pre_dec->power * pppm_t;
  set_pppm(pppm_t, squencer_passes * num_decoder_segments,
           num_decoder_segments * num_decoded_signals,
           num_decoder_segments * num_decoded_signals,
           squencer_passes * num_decoder_segments);
  power = power + final_dec->power * pppm_t;
}
/*
 * [한국어]
 * inst_decoder::leakage_feedback - 온도 변화에 따른 디코더 누설 재계산
 *
 * Predec/Final Decoder의 leakage_feedback()을 호출한 뒤, TDP와 동일한
 * ppcm 가중치로 power를 재구성한다.
 */
void inst_decoder::leakage_feedback(double temperature) {
  l_ip.temp = (unsigned int)round(temperature / 10.0) * 10;
  uca_org_t init_result = init_interface(&l_ip);  // init_result is dummy

  final_dec->leakage_feedback(temperature);
  pre_dec->leakage_feedback(temperature);

  double pppm_t[4] = {1, 1, 1, 1};
  double squencer_passes = x86 ? 2 : 1;

  set_pppm(pppm_t, squencer_passes * num_decoder_segments, num_decoder_segments,
           squencer_passes * num_decoder_segments, num_decoder_segments);
  power = pre_dec->power * pppm_t;

  set_pppm(pppm_t, squencer_passes * num_decoder_segments,
           num_decoder_segments * num_decoded_signals,
           num_decoder_segments * num_decoded_signals,
           squencer_passes * num_decoder_segments);
  power = power + final_dec->power * pppm_t;

  double sckRation = g_tp.sckt_co_eff;

  power.readOp.dynamic *= sckRation;
  power.writeOp.dynamic *= sckRation;
  power.searchOp.dynamic *= sckRation;

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
}

/*
 * [한국어]
 * inst_decoder::~inst_decoder - 디코더 객체 메모리 해제
 *
 * CACTI uca_org_t cleanup과 Decoder/Predec 동적 할당 객체를 삭제한다.
 */
inst_decoder::~inst_decoder() {
  local_result.cleanup();

  delete final_dec;

  delete pre_dec->blk1;
  delete pre_dec->blk2;
  delete pre_dec->drv1;
  delete pre_dec->drv2;
  delete pre_dec;
}
