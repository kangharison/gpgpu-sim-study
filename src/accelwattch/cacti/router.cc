/*****************************************************************************
 *                                McPAT/CACTI
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



/*
 * [한국어 설명] MCPAT_Router — NUCA용 NoC 라우터 전력/면적/지연 모델 구현 (router.cc)
 *
 * === 파일의 역할 ===
 * NUCA 캐시 뱅크들을 연결하는 NoC 라우터 하나를 완전히 모델링한다.
 * 생성자 호출 시 전송 게이트 기반 크로스바, SRAM 기반 VC 버퍼,
 * VC 중재기+크로스바 중재기 각각의 전력·면적을 계산해 Component 멤버에 저장한다.
 * CACTI 6.0 Tech Report의 라우터 모델 수식을 직접 구현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Nuca::sim_nuca() → new MCPAT_Router(flit_size, vc_buf, vc_count) → [이 파일]
 * 생성자가 calc_router_parameters()를 호출해 지연/전력/면적을 즉시 계산한다.
 * 결과는 nuca_org_t에 포인터로 저장되어 최적 NUCA 탐색에 활용된다.
 * 호스트 유저스페이스, 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Wire(Cw3 — 3배 간격 배선 커패시턴스), Crossbar(크로스바 전력),
 *       MCPAT_Arbiter(VC/크로스바 중재기 전력), Mat(VC 버퍼 SRAM 전력),
 *       basic_circuit(gate_C, drain_C_, cmos_Isub_leakage, MIN)
 * 소비: Nuca::sim_nuca()가 3개 라우터(64/128/256비트 flit)를 생성해 사용
 * 공유: g_tp(기술 파라미터), g_ip->F_sz_um(공정 피처 사이즈)
 *
 * === 주요 함수/구조체 요약 ===
 * MCPAT_Router()       : 생성자 — 트랜지스터 파라미터 초기화, calc_router_parameters() 호출
 * calc_router_parameters(): 지연→전력→면적 순서로 orchestrate
 * get_router_power()   : 버퍼/크로스바/중재기 전력을 합산해 Component::power 설정
 * buffer_stats()       : Mat 클래스로 VC 버퍼(SRAM) 전력·면적 계산
 *
 * === AccelWattch XML / gpgpusim.config 연동 ===
 * 본 파일의 생성자는 Nuca::sim_nuca()가 인스턴스화할 때 호출되며, XML의
 * <param name="nuca">(0/1), <param name="cache_policy">, <param name="router">,
 * 그리고 공정/온도 관련 파라미터(technology_node, temperature)에 간접적으로
 * 의존한다. g_ip->F_sz_um, g_tp.* 등은 io.cc/parameter.cc에서 XML/구성 파일을
 * 읽어 초기화되며, gpgpusim.config의 --power_config_name <xml> 옵션이
 * AccelWattch 파워 모델 초기화를 트리거하여 이 라우터 모델까지 호출한다.
 * cb_stats()           : Crossbar 클래스로 크로스바 전력·면적 계산
 * print_router()       : 라우터 상세 통계 콘솔 출력
 */

#include "router.h"

/*
 * [한국어]
 * MCPAT_Router::MCPAT_Router - NoC 라우터 생성자
 *
 * @flit_size_: 링크 폭 [비트] (64/128/256)
 * @vc_buf    : VC당 버퍼 크기 [flit 단위] (기본 8)
 * @vc_c      : 가상 채널 수 (기본 4)
 * @dt        : CMOS 디바이스 파라미터 포인터 (기본 &g_tp.peri_global)
 * @I_        : 크로스바 입력 포트 수 (기본 5)
 * @O_        : 크로스바 출력 포트 수 (기본 5)
 * @M_        : 네트워크 부하율 (기본 0.6)
 *
 * 공정 기술 파라미터(F_sz_um)를 기반으로 트랜지스터 치수를 계산하고,
 * calc_router_parameters()를 호출해 지연·전력·면적을 즉시 산출한다.
 * 호출 체인: Nuca::sim_nuca() → [이 함수] → calc_router_parameters()
 */
MCPAT_Router::MCPAT_Router(
    double flit_size_,
    double vc_buf, /* vc size = vc_buffer_size * flit_size */
    double vc_c,
    TechnologyParameter::DeviceType *dt,
    double I_,
    double O_,
    double M_
    ):flit_size(flit_size_),
      deviceType(dt),
      I(I_),
      O(O_),
      M(M_)
{
  vc_buffer_size = vc_buf; // [한국어] VC당 버퍼 크기 설정 [flit 단위]
  vc_count = vc_c;         // [한국어] 가상 채널 수 설정
  min_w_pmos = deviceType->n_to_p_eff_curr_drv_ratio*g_tp.min_w_nmos_;
  // [한국어] 최소 PMOS 폭 = n-to-p 전류 구동비 × 최소 NMOS 폭 [m]
  double technology = g_ip->F_sz_um; // [한국어] 공정 피처 사이즈 [um] — 트랜지스터 치수 스케일링 기준

  Vdd = dt->Vdd; // [한국어] 공급 전압 [V] — 크로스바 전력 계산의 C×Vdd² 항에 사용

  /*Crossbar parameters. Transmisson gate is employed for connector*/
  NTtr = 10*technology*1e-6/2; /*Transmission gate's nmos tr. length*/
  // [한국어] 트랜스미션 게이트 NMOS 폭 = 10F/2 [m] (F=feature size)
  PTtr = 20*technology*1e-6/2; /* pmos tr. length*/
  // [한국어] 트랜스미션 게이트 PMOS 폭 = 20F/2 [m] (PMOS는 NMOS의 2배)
  wt = 15*technology*1e-6/2; /*track width*/
  // [한국어] 크로스바 트랙 폭 = 15F/2 [m]
  ht = 15*technology*1e-6/2; /*track height*/
  // [한국어] 크로스바 트랙 높이 = 15F/2 [m]
//  I = 5; /*Number of crossbar input ports*/
//  O = 5; /*Number of crossbar output ports*/
  NTi = 12.5*technology*1e-6/2; // [한국어] 입력 인버터 NMOS 폭 [m]
  PTi = 25*technology*1e-6/2;   // [한국어] 입력 인버터 PMOS 폭 [m]

  NTid = 60*technology*1e-6/2; //m // [한국어] 입력 드라이버 NMOS 폭 [m]
  PTid = 120*technology*1e-6/2; // m // [한국어] 입력 드라이버 PMOS 폭 [m]
  NTod = 60*technology*1e-6/2; // m  // [한국어] 출력 드라이버 NMOS 폭 [m]
  PTod = 120*technology*1e-6/2; // m // [한국어] 출력 드라이버 PMOS 폭 [m]

  calc_router_parameters(); // [한국어] 지연→전력→면적 순서로 라우터 파라미터 계산
}

MCPAT_Router::~MCPAT_Router(){} // [한국어] 소멸자 — 멤버가 모두 스택/값 타입이므로 별도 해제 없음


/*
 * [한국어]
 * MCPAT_Router::Cw3 - 3배 간격 글로벌 와이어 커패시턴스 [F] 반환
 *
 * @length: 와이어 길이 [m]
 * @return: 커패시턴스 [F]
 *
 * 크로스바 내부 배선은 인접 배선과의 결합 커패시턴스를 줄이기 위해
 * 3배 간격(w_scale=3, s_scale=3)을 사용한다. Wire 클래스를 임시 생성해 계산.
 * 호출 체인: crossbar_inpline/outline/ctrline() → [이 함수]
 */
double //wire cap with triple spacing
MCPAT_Router::Cw3(double length) {
  Wire wc(g_ip->wt, length, 1, 3, 3); // [한국어] 3배 폭/간격 와이어 모델 임시 생성
  return (wc.wire_cap(length));         // [한국어] 해당 길이의 총 커패시턴스 반환 [F]
}

/*
 * [한국어]
 * MCPAT_Router::gate_cap - MOSFET 게이트 커패시턴스 [F] 반환
 *
 * @w    : 트랜지스터 폭 [m] (내부에서 um로 변환)
 * @return: 게이트 커패시턴스 [F]
 *
 * 호출 체인: crossbar_inpline/ctrline() → [이 함수]
 */
/*Function to calculate the gate capacitance*/
double
MCPAT_Router::gate_cap(double w) {
  return (double) gate_C (w*1e6 /*u*/, 0); // [한국어] m→um 변환 후 gate_C() 호출 (겹침 길이 0)
}

/*
 * [한국어]
 * MCPAT_Router::diff_cap - MOSFET 드레인(확산) 커패시턴스 [F] 반환
 *
 * @w    : 트랜지스터 폭 [m]
 * @type : 0=NMOS, 1=PMOS
 * @s    : 스택 수 (직렬 연결 트랜지스터 수)
 * @return: 드레인 커패시턴스 [F]
 *
 * 호출 체인: transmission_buf_inpcap/outcap(), crossbar_inpline/outline() → [이 함수]
 */
/*Function to calculate the diffusion capacitance*/
double
MCPAT_Router::diff_cap(double w, int type /*0 for n-mos and 1 for p-mos*/,
    double s /*number of stacking transistors*/) {
  return (double) drain_C_(w*1e6 /*u*/, type, (int) s, 1, g_tp.cell_h_def);
  // [한국어] m→um 변환 후 drain_C_() 호출 — 공유 접촉(is_dram=1), 표준 셀 높이 기준
}


/*crossbar related functions */

// Model for simple transmission gate
double
MCPAT_Router::transmission_buf_inpcap() {
  return diff_cap(NTtr, 0, 1)+diff_cap(PTtr, 1, 1);
}

double
MCPAT_Router::transmission_buf_outcap() {
  return diff_cap(NTtr, 0, 1)+diff_cap(PTtr, 1, 1);
}

double
MCPAT_Router::transmission_buf_ctrcap() {
  return gate_cap(NTtr)+gate_cap(PTtr);
}

double
MCPAT_Router::crossbar_inpline() {
  return (Cw3(O*flit_size*wt) + O*transmission_buf_inpcap() + gate_cap(NTid) +
      gate_cap(PTid) + diff_cap(NTid, 0, 1) + diff_cap(PTid, 1, 1));
}

double
MCPAT_Router::crossbar_outline() {
  return (Cw3(I*flit_size*ht) + I*transmission_buf_outcap() + gate_cap(NTod) +
      gate_cap(PTod) + diff_cap(NTod, 0, 1) + diff_cap(PTod, 1, 1));
}

double
MCPAT_Router::crossbar_ctrline() {
  return (Cw3(0.5*O*flit_size*wt) + flit_size*transmission_buf_ctrcap() +
      diff_cap(NTi, 0, 1) + diff_cap(PTi, 1, 1) +
      gate_cap(NTi) + gate_cap(PTi));
}

double
MCPAT_Router::tr_crossbar_power() {
  return (crossbar_inpline()*Vdd*Vdd*flit_size/2 +
      crossbar_outline()*Vdd*Vdd*flit_size/2)*2;
}

/*
 * [한국어]
 * MCPAT_Router::buffer_stats - VC 버퍼(SRAM)의 전력·면적 계산
 *
 * VC 버퍼를 1포트 읽기, vc_count개 포트 쓰기를 지원하는 SRAM으로 모델링한다.
 * 행 수 = vc_buffer_size (엔트리 수), 열 수 = flit_size × vc_count (비트 수).
 * Mat 클래스를 사용해 비트라인/워드라인/SA/디코더 전력을 계산한다.
 * 계산 결과는 buffer.power(읽기·쓰기)와 buffer.area에 저장된다.
 * 호출 체인: get_router_power() → [이 함수]
 */
void MCPAT_Router::buffer_stats()
{
  DynamicParameter dyn_p; // [한국어] SRAM 버퍼의 동적 파라미터 구조체 — 모든 필드 수동 설정 필요
  dyn_p.is_tag      = false;
  dyn_p.pure_cam    = false;
  dyn_p.fully_assoc = false;
  dyn_p.pure_ram    = true;
  dyn_p.is_dram     = false;
  dyn_p.is_main_mem = false;
  dyn_p.num_subarrays = 1;
  dyn_p.num_mats = 1;
  dyn_p.Ndbl = 1;
  dyn_p.Ndwl = 1;
  dyn_p.Nspd = 1;
  dyn_p.deg_bl_muxing = 1;
  dyn_p.deg_senseamp_muxing_non_associativity = 1;
  dyn_p.Ndsam_lev_1 = 1;
  dyn_p.Ndsam_lev_2 = 1;
  dyn_p.Ndcm = 1;
  dyn_p.number_addr_bits_mat = 8;
  dyn_p.number_way_select_signals_mat = 1;
  dyn_p.number_subbanks_decode = 0;
  dyn_p.num_act_mats_hor_dir = 1;
  dyn_p.V_b_sense = Vdd; // FIXME check power calc.
  dyn_p.ram_cell_tech_type = 0;
  dyn_p.num_r_subarray = (int) vc_buffer_size;
  dyn_p.num_c_subarray = (int) flit_size * (int) vc_count;
  dyn_p.num_mats_h_dir = 1;
  dyn_p.num_mats_v_dir = 1;
  dyn_p.num_do_b_subbank = (int)flit_size;
  dyn_p.num_di_b_subbank = (int)flit_size;
  dyn_p.num_do_b_mat = (int) flit_size;
  dyn_p.num_di_b_mat = (int) flit_size;
  dyn_p.num_do_b_mat = (int) flit_size;
  dyn_p.num_di_b_mat = (int) flit_size;
  dyn_p.num_do_b_bank_per_port = (int) flit_size;
  dyn_p.num_di_b_bank_per_port = (int) flit_size;
  dyn_p.out_w = (int) flit_size;

  dyn_p.use_inp_params = 1;
  dyn_p.num_wr_ports = (unsigned int) vc_count;
  dyn_p.num_rd_ports = 1;//(unsigned int) vc_count;//based on Bill Dally's book
  dyn_p.num_rw_ports = 0;
  dyn_p.num_se_rd_ports =0;
  dyn_p.num_search_ports =0;



  dyn_p.cell.h = g_tp.sram.b_h + 2 * g_tp.wire_outside_mat.pitch * (dyn_p.num_wr_ports +
      dyn_p.num_rw_ports - 1 + dyn_p.num_rd_ports);
  dyn_p.cell.w = g_tp.sram.b_w + 2 * g_tp.wire_outside_mat.pitch * (dyn_p.num_rw_ports - 1 +
      (dyn_p.num_rd_ports - dyn_p.num_se_rd_ports) +
      dyn_p.num_wr_ports) + g_tp.wire_outside_mat.pitch * dyn_p.num_se_rd_ports;

  Mat buff(dyn_p);
  buff.compute_delays(0);
  buff.compute_power_energy();
  buffer.power.readOp  = buff.power.readOp;
  buffer.power.writeOp = buffer.power.readOp; //FIXME
  buffer.area = buff.area;
}



/*
 * [한국어]
 * MCPAT_Router::cb_stats - 크로스바 전력·면적 계산
 *
 * Crossbar 클래스(I×O 크로스바)를 생성해 compute_power()로 계산하고
 * crossbar 멤버에 결과를 복사한다.
 * 대안으로 전송 게이트 모델(tr_crossbar_power)도 있으나 현재는 Crossbar 클래스를 사용한다.
 * 호출 체인: get_router_power() → [이 함수] → Crossbar::compute_power()
 */
  void
MCPAT_Router::cb_stats ()
{
  if (1) { // [한국어] 항상 true — Crossbar 클래스 기반 모델 사용 (전송게이트 모델 비활성)
    Crossbar c_b(I, O, flit_size);
    c_b.compute_power();
    crossbar.delay = c_b.delay;
    crossbar.power.readOp.dynamic = c_b.power.readOp.dynamic;
    crossbar.power.readOp.leakage = c_b.power.readOp.leakage;
    crossbar.power.readOp.gate_leakage = c_b.power.readOp.gate_leakage;
    crossbar.area = c_b.area;
//  c_b.print_crossbar();
  }
  else {
    crossbar.power.readOp.dynamic = tr_crossbar_power();
    crossbar.power.readOp.leakage = flit_size * I * O *
        cmos_Isub_leakage(NTtr*g_tp.min_w_nmos_, PTtr*min_w_pmos, 1, tg);
    crossbar.power.readOp.gate_leakage = flit_size * I * O *
        cmos_Ig_leakage(NTtr*g_tp.min_w_nmos_, PTtr*min_w_pmos, 1, tg);
  }
}

/*
 * [한국어]
 * MCPAT_Router::get_router_power - 버퍼·크로스바·중재기 전력 합산
 *
 * 1) buffer_stats(): VC 버퍼(SRAM) 전력 계산
 * 2) cb_stats()    : 크로스바(Crossbar) 전력 계산
 * 3) VC 중재기(vcarb, VC_COUNT 입력)와 크로스바 중재기(cbarb, I 입력) 생성·계산
 * 4) 총 동적 전력 = (버퍼읽기+쓰기 + 크로스바 + 중재기) × MIN(I,O) × M (부하율)
 * 5) 누설 전력 = pppm_lkg 적용 (입력 포트 수 I로 스케일링)
 *
 * 호출 체인: calc_router_parameters() → [이 함수]
 */
void
MCPAT_Router::get_router_power()
{
  /* calculate buffer stats */
  buffer_stats(); // [한국어] VC 버퍼의 읽기/쓰기 전력·면적 계산

  /* calculate cross-bar stats */
  cb_stats(); // [한국어] 크로스바 전력·면적·지연 계산

  /* calculate arbiter stats */
  MCPAT_Arbiter vcarb(vc_count, flit_size, buffer.area.w);
  // [한국어] VC 중재기: vc_count개 VC 중 하나를 선택하는 중재기 — 버퍼 폭 기준
  MCPAT_Arbiter cbarb(I, flit_size, crossbar.area.w);
  // [한국어] 크로스바 중재기: I개 입력 포트 중 크로스바를 통과할 패킷 선택
  vcarb.compute_power(); // [한국어] VC 중재기 전력 계산
  cbarb.compute_power(); // [한국어] 크로스바 중재기 전력 계산

  arbiter.power.readOp.dynamic = vcarb.power.readOp.dynamic * I +
    cbarb.power.readOp.dynamic * O;
  // [한국어] 총 중재기 동적 전력 = VC중재기 × I입력포트 + 크로스바중재기 × O출력포트
  arbiter.power.readOp.leakage = vcarb.power.readOp.leakage * I +
    cbarb.power.readOp.leakage * O;
  // [한국어] 총 중재기 누설 전력 = VC중재기누설×I + 크로스바중재기누설×O
  arbiter.power.readOp.gate_leakage = vcarb.power.readOp.gate_leakage * I +
    cbarb.power.readOp.gate_leakage * O;

//  arb_stats();
  power.readOp.dynamic = ((buffer.power.readOp.dynamic+buffer.power.writeOp.dynamic) +
		  crossbar.power.readOp.dynamic +
		  arbiter.power.readOp.dynamic)*MIN(I, O)*M;
  // [한국어] 총 동적 전력 = (버퍼읽기+쓰기 + 크로스바 + 중재기) × min(I,O) × 부하율M
  // min(I,O): 실제 전달 가능한 패킷 수의 병목, M: 네트워크 부하율(0~1)
  double pppm_t[4]    = {1,I,I,1};
  // [한국어] 누설 전력 스케일링 벡터: [readLeak, writeLeak, searchLeak, gateLeak]
  // 버퍼 누설은 I포트(입력포트 수)만큼 스케일, 크로스바/중재기는 1배 그대로
  power = power + (buffer.power*pppm_t + crossbar.power + arbiter.power)*pppm_lkg;
  // [한국어] 누설 전력(pppm_lkg 적용) 합산 — pppm_lkg는 기술 파라미터 기반 누설 배율

}

/*
 * [한국어]
 * MCPAT_Router::get_router_delay - 라우터 주파수·파이프라인 스테이지 수 결정
 *
 * 고정 주파수(5 GHz)로 시작하고, 공정 기반 최대 허용 사이클 타임(17×FO4)이
 * 5 GHz 사이클보다 크면 주파수를 낮춰 재설정한다.
 * delay=4는 파이프라인 스테이지 수 (SA/CB/VC arb/output 각 1 단계).
 * 호출 체인: calc_router_parameters() → [이 함수]
 */
  void
MCPAT_Router::get_router_delay ()
{
  FREQUENCY=5; // move this to config file --TODO
  // [한국어] 라우터 주파수를 5 GHz로 초기 설정 (TODO: 설정 파일에서 읽어야 함)
  cycle_time = (1/(double)FREQUENCY)*1e3; //ps
  // [한국어] 1/5GHz = 200ps — 라우터 사이클 타임 [ps]
  delay = 4;
  // [한국어] 파이프라인 스테이지 수 = 4 (BC/VA/SA/ST 단계 각각 1 사이클)
  max_cyc = 17 * g_tp.FO4; //s
  // [한국어] 공정 기반 최대 허용 사이클 타임 = 17 × FO4 [s] (FO4: Fan-Out-of-4 지연)
  max_cyc *= 1e12; //ps
  // [한국어] 단위 변환: s → ps
  if (cycle_time < max_cyc) {
    FREQUENCY = (1/max_cyc)*1e3; //GHz
    // [한국어] 5 GHz 사이클이 공정 한계보다 짧으면 가능한 최대 주파수로 하향 조정
  }
}

/*
 * [한국어]
 * MCPAT_Router::get_router_area - 라우터 총 면적 계산
 *
 * 라우터 면적 = 버퍼 면적 + 크로스바 면적 (중재기 면적은 버퍼 내에 포함된다고 가정).
 * 높이 = I(입력 포트 수) × 버퍼 높이 (포트별 버퍼가 수직으로 쌓임).
 * 너비 = 버퍼 너비 + 크로스바 너비.
 * 호출 체인: calc_router_parameters() → [이 함수]
 */
  void
MCPAT_Router::get_router_area()
{
  area.h = I*buffer.area.h;          // [한국어] I개 입력 포트 × 버퍼 높이
  area.w = buffer.area.w+crossbar.area.w; // [한국어] 버퍼 너비 + 크로스바 너비
}

/*
 * [한국어]
 * MCPAT_Router::calc_router_parameters - 라우터 파라미터 계산 orchestrator
 *
 * 지연 → 전력 → 면적 순서로 계산을 orchestrate한다.
 * 순서가 중요: get_router_power()가 buffer.area를 읽고, get_router_area()가 이를 사용.
 * 호출 체인: MCPAT_Router() → [이 함수] → get_router_delay/power/area()
 */
  void
MCPAT_Router::calc_router_parameters()
{
  /* calculate router frequency and pipeline cycles */
  get_router_delay(); // [한국어] 1단계: 주파수·사이클 타임·파이프라인 스테이지 계산

  /* router power stats */
  get_router_power(); // [한국어] 2단계: 버퍼/크로스바/중재기 전력 계산

  /* area stats */
  get_router_area();  // [한국어] 3단계: 총 면적 계산 (buffer.area + crossbar.area 사용)
}

/*
 * [한국어]
 * MCPAT_Router::print_router - 라우터 상세 통계 콘솔 출력
 *
 * calc_router_parameters() 완료 후 면적·주파수·VC 수·파이프라인 스테이지·
 * 버퍼/크로스바/중재기 전력·면적을 사람이 읽기 좋은 형식으로 출력한다.
 * 단위 변환: 면적 um²→mm², 에너지 J→nJ, 전력 W→mW.
 * 호출 체인: Nuca::sim_nuca() → print_router() (디버그 목적)
 */
  void
MCPAT_Router::print_router()
{
  cout << "\n\nMCPAT_Router stats:\n";
  cout << "\tMCPAT_Router Area - "<< area.get_area()*1e-6<<"(mm^2)\n";
  // [한국어] 라우터 총 면적 [mm²] = area.get_area() [um²] × 1e-6
  cout << "\tMaximum possible network frequency - " << (1/max_cyc)*1e3 << "GHz\n";
  // [한국어] 공정 기반 최대 주파수 [GHz] = 1/max_cyc [ps] × 1e3
  cout << "\tNetwork frequency - " << FREQUENCY <<" GHz\n";
  // [한국어] 실제 설정된 라우터 주파수 [GHz]
  cout << "\tNo. of Virtual channels - " << vc_count << "\n";
  // [한국어] 가상 채널 수 (VC 버퍼 수)
  cout << "\tNo. of pipeline stages - " << delay << endl;
  // [한국어] 파이프라인 스테이지 수 (= 라우터 지연 [사이클])
  cout << "\tLink bandwidth - " << flit_size << " (bits)\n";
  // [한국어] 링크 폭 = flit 크기 [비트]
  cout << "\tNo. of buffer entries per virtual channel -  "<< vc_buffer_size << "\n";
  // [한국어] VC당 버퍼 엔트리 수 (SRAM 행 수)
  cout << "\tSimple buffer Area - "<< buffer.area.get_area()*1e-6<<"(mm^2)\n";
  // [한국어] VC 버퍼 면적 [mm²]
  cout << "\tSimple buffer access (Read) - " << buffer.power.readOp.dynamic * 1e9 <<" (nJ)\n";
  // [한국어] VC 버퍼 읽기 에너지 [nJ] = J × 1e9
  cout << "\tSimple buffer leakage - " << buffer.power.readOp.leakage * 1e3 <<" (mW)\n";
  // [한국어] VC 버퍼 누설 전력 [mW] = W × 1e3
  cout << "\tCrossbar Area - "<< crossbar.area.get_area()*1e-6<<"(mm^2)\n";
  // [한국어] 크로스바 면적 [mm²]
  cout << "\tCross bar access energy - " << crossbar.power.readOp.dynamic * 1e9<<" (nJ)\n";
  // [한국어] 크로스바 접근 에너지 [nJ]
  cout << "\tCross bar leakage power - " << crossbar.power.readOp.leakage * 1e3<<" (mW)\n";
  // [한국어] 크로스바 누설 전력 [mW]
  cout << "\tMCPAT_Arbiter access energy (VC arb + Crossbar arb) - "<<arbiter.power.readOp.dynamic * 1e9 <<" (nJ)\n";
  // [한국어] 중재기(VC 중재기 + 크로스바 중재기) 접근 에너지 합산 [nJ]
  cout << "\tMCPAT_Arbiter leakage (VC arb + Crossbar arb) - "<<arbiter.power.readOp.leakage * 1e3 <<" (mW)\n";
  // [한국어] 중재기 누설 전력 합산 [mW]

}

