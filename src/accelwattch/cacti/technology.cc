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
 * [한국어 설명] CACTI/McPAT 공정 기술 파라미터 초기화 (technology.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 CACTI 캐시 모델과 McPAT 전력 모델이 사용하는 반도체 공정(technology node)
 * 파라미터를 초기화하는 핵심 파일이다. 트랜지스터 전기적 특성(문턱전압, 산화막 두께,
 * 온/오프 전류 등)과 Cu 배선의 저항/커패시턴스를 지원하는 모든 공정 노드(180nm ~
 * 22nm)에 대해 ITRS(International Technology Roadmap for Semiconductors) 수치로 채운다.
 * AccelWattch가 GPU 전력을 추정할 때 이 파일에서 설정된 g_tp(전역 기술 파라미터 구조체)를
 * 기반으로 게이트/배선 커패시턴스와 누설 전류를 계산한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim AccelWattch 전력 모델의 최하단 물리 계층에 해당한다.
 * 호출 체인: cacti_interface() → init_tech_params() [이 파일]
 * cacti_interface()가 시뮬레이션 시작 시 1회 호출하며, g_tp를 채운 뒤에는
 * basic_circuit.cc 전체와 gpu-sim.cc의 전력 추정 루틴이 g_tp를 읽기 전용으로 참조한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이션 초기화 단계에서 1회만 실행된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존(읽는 것): g_ip(전역 입력 파라미터, parameter.h) — 요청 공정 노드(F_sz_um),
 *   메모리 셀 타입(data_arr_ram_cell_tech_type 등), 온도(temp), 배선 프로젝션 타입(ic_proj_type)
 * - 생성(쓰는 것): g_tp(전역 기술 파라미터 구조체) — 이후 basic_circuit.cc의 gate_C(),
 *   tr_R_on(), drain_C_() 등 모든 회로 계산 함수가 g_tp를 사용한다
 * - 핵심 자료구조: TechnologyParameter(parameter.h)의 g_tp — peri_global, sram_cell,
 *   dram_acc, dram_wl, wire_local, wire_inside_mat, wire_outside_mat 하위 구조체를 포함
 * - wire_resistance(), wire_capacitance()는 이 파일 내부에 정의된 헬퍼 함수로,
 *   배선 물리 치수로부터 저항/커패시턴스를 계산한다
 *
 * === 주요 함수/구조체 요약 ===
 * - wire_resistance(): Cu 배선 단위 길이(μm)당 저항을 계산. Ta 배리어 라이너와 CMP
 *   디싱 효과를 물리적으로 모델링한다.
 * - wire_capacitance(): Cu 배선 단위 길이(μm)당 커패시턴스를 계산. 수직(기판 대향)과
 *   측벽(Miller 계수 적용) 성분의 합으로 구한다.
 * - init_tech_params(): 메인 함수. technology(μm 단위 공정 노드)를 입력받아 g_tp의
 *   모든 필드를 채운다. 요청 노드가 지원 노드 사이에 있으면 인접 두 노드 값을 선형 보간한다.
 *   함수 내부는 두 개의 for(iter=0; iter<=1) 루프로 구성되며, 첫 번째는 트랜지스터
 *   파라미터를, 두 번째는 배선 파라미터를 각각 보간·누적한다.
 */

#include "basic_circuit.h"  /* [한국어] gate_C, tr_R_on, drain_C_, horowitz, pmos_to_nmos_sz_ratio 등 회로 계산 함수 선언 */

#include "parameter.h"  /* [한국어] TechnologyParameter(g_tp), InputParameter(g_ip), NUMBER_TECH_FLAVORS 등 전역 파라미터 선언 */

/*
 * [한국어]
 * wire_resistance - Cu 배선 단위 길이(μm)당 저항 계산
 *
 * @resistivity: Cu 비저항 (Ω·μm). CU_RESISTIVITY(실사 배선) 또는 BULK_CU_RESISTIVITY(이상적 Cu)
 * @wire_width: 배선 폭 (μm). 공정 노드별 wire_pitch / 2로 결정된다.
 * @wire_thickness: 배선 두께 (μm). aspect_ratio * wire_width로 결정된다.
 * @barrier_thickness: Ta(탄탈럼) 배리어 라이너 두께 (μm). Cu 확산 방지층으로, 실효 단면적을 줄인다.
 * @dishing_thickness: CMP(화학기계연마) 디싱으로 인한 두께 손실 (μm). 글로벌 배선처럼 넓은 선에서 발생.
 * @alpha_scatter: 입자 산란에 의한 저항 상승 계수 (>= 1.0). 나노미터 스케일에서 표면 산란이 증가한다.
 * @return: 단위 길이(μm)당 저항 (Ω/μm). 호출자(init_tech_params)가 wire_r_per_micron 배열에 저장한다.
 *
 * 실효 Cu 단면적은 (두께 - barrier - dishing) × (폭 - 2×barrier)이다.
 * 양쪽 측벽에 배리어가 존재하므로 폭에서 2배를 빼고, 바닥·천장에도 배리어가 있어 두께에서 차감한다.
 * 저항 = alpha_scatter × ρ / (실효 단면적).
 *
 * 호출 체인:
 *   init_tech_params() → [wire_resistance] (각 공정 노드·각 배선 계층별로 반복 호출)
 */
double wire_resistance(double resistivity, double wire_width, double wire_thickness,
    double barrier_thickness, double dishing_thickness, double alpha_scatter)
{
  double resistance;
  /* [한국어] 실효 Cu 단면적 = (두께 - 배리어 - 디싱) × (폭 - 2×배리어).
   * alpha_scatter: 나노 스케일 표면/입계 산란으로 인한 저항 증가 보정 계수 (22nm 이하에서 중요). */
  resistance = alpha_scatter * resistivity /((wire_thickness - barrier_thickness - dishing_thickness)*(wire_width - 2 * barrier_thickness));
  return(resistance);
}

/*
 * [한국어]
 * wire_capacitance - Cu 배선 단위 길이(μm)당 커패시턴스 계산
 *
 * @wire_width: 배선 폭 (μm)
 * @wire_thickness: 배선 두께 (μm)
 * @wire_spacing: 인접 배선과의 간격 (μm). wire_pitch - wire_width.
 * @ild_thickness: ILD(층간 유전체, Inter-Layer Dielectric) 두께 (μm). 수직 커패시턴스 결정.
 * @miller_value: Miller 효과 계수 (통상 1.5). 배선이 반전 신호를 구동할 때 측벽 커패시턴스가
 *   실효적으로 2배 증폭된다(Miller 효과). miller_value >= 1.0.
 * @horiz_dielectric_constant: 측벽 방향(수평) ILD의 비유전율 (ε_r). 공정 노드가 작을수록 낮은
 *   low-k 유전체를 사용하여 배선 간 커패시턴스를 줄인다.
 * @vert_dielectric_constant: 수직(기판 방향) ILD의 비유전율. 보통 SiO2(ε_r=3.9)에 가깝다.
 * @fringe_cap: 프린지(fringe) 커패시턴스 (F/μm). 배선 모서리에서 발생하는 기생 커패시턴스.
 * @return: 단위 길이(μm)당 총 커패시턴스 (F/μm). init_tech_params가 wire_c_per_micron에 저장.
 *
 * 총 커패시턴스 = 수직 커패시턴스(기판 대향 평행판) + 측벽 커패시턴스(인접 배선 간, Miller 보정) + 프린지.
 * 수직: 2 × ε₀ × ε_vert × W / t_ILD  (배선 위아래로 기판/상위 금속 층과의 커패시턴스)
 * 측벽: 2 × ε₀ × miller × ε_horiz × T / spacing (배선 양측벽과 인접 배선 사이)
 *
 * 호출 체인:
 *   init_tech_params() → [wire_capacitance] (각 공정 노드·각 배선 계층별로 반복 호출)
 */
double wire_capacitance(double wire_width, double wire_thickness, double wire_spacing,
    double ild_thickness, double miller_value, double horiz_dielectric_constant,
    double vert_dielectric_constant, double fringe_cap)
{
  double vertical_cap, sidewall_cap, total_cap;
  /* [한국어] 수직 커패시턴스: 배선이 기판/인접 금속 층과 이루는 평행판 커패시턴스 (F/μm).
   * C_vert = 2 × ε₀ × ε_vert × wire_width / ild_thickness. 계수 2는 배선 위아래 두 면을 합산. */
  vertical_cap = 2 * PERMITTIVITY_FREE_SPACE * vert_dielectric_constant * wire_width / ild_thickness;
  /* [한국어] 측벽 커패시턴스: 인접 배선 사이 커패시턴스에 Miller 계수를 적용한 값 (F/μm).
   * C_side = 2 × ε₀ × miller × ε_horiz × wire_thickness / wire_spacing.
   * miller_value(통상 1.5)는 스위칭 배선이 반전 신호와 마주칠 때의 전압차 증폭을 반영한다. */
  sidewall_cap = 2 * PERMITTIVITY_FREE_SPACE * miller_value * horiz_dielectric_constant * wire_thickness / wire_spacing;
  /* [한국어] 총 커패시턴스: 수직 + 측벽 + 프린지(모서리 기생 커패시턴스). */
  total_cap = vertical_cap + sidewall_cap + fringe_cap;
  return(total_cap);
}


/*
 * [한국어]
 * init_tech_params - 요청 공정 노드에 해당하는 모든 기술 파라미터를 g_tp에 초기화
 *
 * @technology: 공정 노드 크기 (μm 단위; 예: 0.045 = 45nm). 함수 내부에서 nm으로 변환한다.
 * @is_tag: 태그 배열(true) 또는 데이터 배열(false)의 파라미터를 채울지 선택. g_ip의
 *   tag_arr_* vs data_arr_* 필드에서 RAM 셀 타입과 글로벌 트랜지스터 타입을 읽는다.
 * @return: void. 결과는 전역 구조체 g_tp에 직접 기록된다.
 *
 * 이 함수는 CACTI/AccelWattch 초기화 시 cacti_interface()에서 단 1회 호출된다.
 * 지원 노드: 180nm, 90nm, 65nm, 45nm, 32nm, 22nm (ITRS 2000~2016 기반).
 * 요청 노드가 지원 노드 사이에 있으면 인접 두 노드 값을 선형 보간한다:
 *   결과 = alpha × (tech_lo 값) + (1-alpha) × (tech_hi 값), alpha = (tech - tech_hi)/(tech_lo - tech_hi).
 * 함수 구조:
 *   1단계: tech_lo, tech_hi 결정 (if-else 체인)
 *   2단계: for(iter=0; iter<=1) — 트랜지스터 파라미터 보간·누적 (g_tp.peri_global 등)
 *   3단계: 배선 무관 파생값 계산 (커패시턴스, 트랜지스터 치수 등)
 *   4단계: for(iter=0; iter<=1) — 배선 파라미터 보간·누적 (g_tp.wire_local 등)
 *   5단계: FO4(Fan-Out-of-4 인버터 지연)와 kinv(단위 인버터 지연) 계산
 *
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, 시뮬레이션 초기화 단계 1회.
 * 에러 처리: 지원하지 않는 공정 노드 또는 22nm에서 eDRAM 요청 시 exit(0)으로 종료.
 *
 * 호출 체인:
 *   cacti_interface() → [init_tech_params] → wire_resistance(), wire_capacitance(),
 *                                              horowitz(), pmos_to_nmos_sz_ratio(), tr_R_on()
 */
void init_tech_params(double technology, bool is_tag)
{
  int   iter = 0;           /* [한국어] 선형 보간 루프 인덱스: 0=tech_lo 처리, 1=tech_hi 처리 */
  int 	tech = 0;           /* [한국어] 현재 루프 이터레이션에서 처리 중인 공정 노드 (nm 정수) */
  int 	tech_lo = 0;        /* [한국어] 보간에 사용할 상위(큰) 공정 노드 (nm). 예: 90nm 요청 시 tech_lo=90 */
  int 	tech_hi = 0;        /* [한국어] 보간에 사용할 하위(작은) 공정 노드 (nm). 예: 75nm 요청 시 tech_hi=65 */
  double curr_alpha = 0;    /* [한국어] 현재 이터레이션의 선형 보간 가중치. tech_lo에 대해 (tech-tech_hi)/(tech_lo-tech_hi) */
  double curr_vpp = 0;      /* [한국어] DRAM 워드라인 부스트 전압 Vpp (V). 셀 접근 트랜지스터 완전 개통을 위해 Vdd보다 높다 */
  double wire_width = 0;    /* [한국어] 배선 폭 (μm) = wire_pitch / 2. 매 배선 계층 계산 전 갱신됨 */
  double wire_thickness =0; /* [한국어] 배선 두께 (μm) = aspect_ratio × wire_width */
  double wire_spacing = 0;  /* [한국어] 인접 배선 간격 (μm) = wire_pitch - wire_width */
  double fringe_cap = 0;    /* [한국어] 배선 모서리 프린지 커패시턴스 (F/μm). 전역 배선에서 중요 */
  double pmos_to_nmos_sizing_r = 0; /* [한국어] PMOS/NMOS 구동 전류 비. precharge/equalizer 트랜지스터 폭 결정에 사용 */
//  double aspect_ratio,ild_thickness, miller_value = 1.5, horiz_dielectric_constant, vert_dielectric_constant;
  double barrier_thickness = 0;    /* [한국어] Ta 배리어 라이너 두께 (μm). wire_resistance() 계산에 사용 */
  double dishing_thickness = 0;    /* [한국어] CMP 디싱에 의한 두께 손실 (μm). 넓은 글로벌 배선에서 발생 */
  double alpha_scatter = 0;        /* [한국어] 표면·입계 산란 저항 증가 계수 (>=1.0). 22nm 이하에서 1.05 이상 */
  double curr_vdd_dram_cell = 0;   /* [한국어] DRAM 셀 공급 전압 Vdd (V). HP 트랜지스터 Vdd와 별도로 설정됨 */
  double curr_v_th_dram_access_transistor = 0; /* [한국어] DRAM 접근 트랜지스터 문턱전압 Vth (V). 높은 Vth로 리텐션 확보 */
  double curr_I_on_dram_cell = 0;  /* [한국어] DRAM 셀 온전류 (A/μm). 셀 리프레시 속도 결정 */
  double curr_c_dram_cell = 0;     /* [한국어] DRAM 셀 커패시터 용량 (F). 리텐션 시간과 신호 크기를 결정 */

  /* [한국어] RAM 셀 타입과 주변 회로(글로벌) 트랜지스터 타입 선택.
   * is_tag=true이면 태그 배열용 파라미터, 아니면 데이터 배열용 파라미터를 사용한다.
   * ram_cell_tech_type: lp_dram(2), comm_dram(4), sram(0) 등 메모리 셀 타입 enum */
  uint32_t ram_cell_tech_type    = (is_tag) ? g_ip->tag_arr_ram_cell_tech_type : g_ip->data_arr_ram_cell_tech_type;
  uint32_t peri_global_tech_type = (is_tag) ? g_ip->tag_arr_peri_global_tech_type : g_ip->data_arr_peri_global_tech_type;

  /* [한국어] technology 파라미터를 μm에서 nm으로 변환. 이후 모든 if-else 비교는 nm 단위로 수행된다. */
  technology  = technology * 1000.0;  // in the unit of nm

  // initialize parameters
  g_tp.reset();
  double gmp_to_gmn_multiplier_periph_global = 0;

  double curr_Wmemcella_dram = 0;
  double curr_Wmemcellpmos_dram = 0;
  double curr_Wmemcellnmos_dram = 0;
  double curr_area_cell_dram = 0;
  double curr_asp_ratio_cell_dram = 0;
  double curr_Wmemcella_sram = 0;
  double curr_Wmemcellpmos_sram = 0;
  double curr_Wmemcellnmos_sram = 0;
  double curr_area_cell_sram = 0;
  double curr_asp_ratio_cell_sram = 0;
  double curr_I_off_dram_cell_worst_case_length_temp = 0;
  double curr_Wmemcella_cam = 0;
  double curr_Wmemcellpmos_cam = 0;
  double curr_Wmemcellnmos_cam = 0;
  double curr_area_cell_cam = 0;//Sheng: CAM data
  double curr_asp_ratio_cell_cam = 0;
  double SENSE_AMP_D, SENSE_AMP_P; // J
  double area_cell_dram = 0;
  double asp_ratio_cell_dram = 0;
  double area_cell_sram = 0;
  double asp_ratio_cell_sram = 0;
  double area_cell_cam = 0;
  double asp_ratio_cell_cam = 0;
  double mobility_eff_periph_global = 0;
  double Vdsat_periph_global = 0;
  double nmos_effective_resistance_multiplier;
  double width_dram_access_transistor;

  double curr_logic_scaling_co_eff = 0;//This is based on the reported numbers of Intel Merom 65nm, Penryn45nm and IBM cell 90/65/45 date
  double curr_core_tx_density = 0;//this is density per um^2; 90, ...22nm based on Intel Penryn
  double curr_chip_layout_overhead = 0;
  double curr_macro_layout_overhead = 0;
  double curr_sckt_co_eff = 0;

  /* [한국어] ===== tech_lo / tech_hi 결정 =====
   * 요청 공정 노드(technology, nm)를 CACTI가 지원하는 이산 노드(180/90/65/45/32/22nm)에 매핑한다.
   * 정확히 지원 노드에 해당하면 tech_lo = tech_hi = 해당 노드 → 보간 없이 alpha=1 사용.
   * 두 지원 노드 사이에 있으면 tech_lo(더 큰 노드), tech_hi(더 작은 노드)를 설정 →
   *   이후 for 루프에서 alpha = (technology - tech_hi) / (tech_lo - tech_hi)로 선형 보간.
   * 예: 75nm 요청 → tech_lo=90, tech_hi=65, alpha_lo=(75-65)/(90-65)=0.4, alpha_hi=0.6.
   */
  if (technology < 181 && technology > 179)
      {
        tech_lo = 180;  /* [한국어] 정확히 180nm 노드: 보간 불필요 */
        tech_hi = 180;
      }
  else if (technology < 91 && technology > 89)
  {
    tech_lo = 90;   /* [한국어] 정확히 90nm 노드: 보간 불필요 */
    tech_hi = 90;
  }
  else if (technology < 66 && technology > 64)
  {
    tech_lo = 65;   /* [한국어] 정확히 65nm 노드: 보간 불필요 */
    tech_hi = 65;
  }
  else if (technology < 46 && technology > 44)
  {
    tech_lo = 45;   /* [한국어] 정확히 45nm 노드: 보간 불필요 */
    tech_hi = 45;
  }
  else if (technology < 33 && technology > 31)
  {
    tech_lo = 32;   /* [한국어] 정확히 32nm 노드: 보간 불필요 */
    tech_hi = 32;
  }
  else if (technology < 23 && technology > 21)
  {
    tech_lo = 22;   /* [한국어] 정확히 22nm 노드: 보간 불필요 */
    tech_hi = 22;
    if (ram_cell_tech_type == 3 )
    {
       /* [한국어] 22nm에서 eDRAM(embedded DRAM, type=3)은 미지원. ITRS 데이터 부재로 인한 한계. */
       cout<<"current version does not support eDRAM technologies at 22nm"<<endl;
       exit(0);
    }
  }
//  else if (technology < 17 && technology > 15)
//  {
//    tech_lo = 16;
//    tech_hi = 16;
//  }
  /* [한국어] 이하는 두 지원 노드 사이 중간 값 요청 시의 보간 구간 설정 */
  else if (technology < 180 && technology > 90)
    {
      tech_lo = 180;  /* [한국어] 90nm~180nm 사이: 180nm와 90nm 사이를 선형 보간 */
      tech_hi = 90;
    }
  else if (technology < 90 && technology > 65)
  {
    tech_lo = 90;   /* [한국어] 65nm~90nm 사이: 90nm와 65nm 사이를 선형 보간 */
    tech_hi = 65;
  }
  else if (technology < 65 && technology > 45)
  {
    tech_lo = 65;   /* [한국어] 45nm~65nm 사이: 65nm와 45nm 사이를 선형 보간 */
    tech_hi = 45;
  }
  else if (technology < 45 && technology > 32)
  {
    tech_lo = 45;   /* [한국어] 32nm~45nm 사이: 45nm와 32nm 사이를 선형 보간 */
    tech_hi = 32;
  }
  else if (technology < 32 && technology > 22)
    {
      tech_lo = 32;  /* [한국어] 22nm~32nm 사이: 32nm와 22nm 사이를 선형 보간 */
      tech_hi = 22;
    }
//  else if (technology < 22 && technology > 16)
//    {
//      tech_lo = 22;
//      tech_hi = 16;
//    }
      else
    {
        /* [한국어] 지원 범위 밖(예: 180nm 초과 또는 22nm 미만) 요청 시 오류 출력 후 종료 */
  	  cout<<"Invalid technology nodes"<<endl;
  	  exit(0);
    }

  /* [한국어] ===== 트랜지스터 파라미터 배열 선언 =====
   * NUMBER_TECH_FLAVORS = 3: [0]=HP(High Performance), [1]=LSTP(Low Standby Power), [2]=LOP(Low Operating Power).
   * DRAM 접근/워드라인 트랜지스터는 [3]을 사용한다(별도 Vth, t_ox 값).
   * 각 배열은 루프 내 공정 노드 블록에서 채워지고, 이후 += curr_alpha × 값 패턴으로 g_tp에 누적된다.
   */
  double vdd[NUMBER_TECH_FLAVORS];              /* [한국어] 트랜지스터 공급 전압 Vdd (V). 노드 축소에 따라 감소 */
  double Lphy[NUMBER_TECH_FLAVORS];             /* [한국어] 물리적 게이트 길이 Lphy (μm). 리소그래피로 인쇄된 실제 게이트 길이 */
  double Lelec[NUMBER_TECH_FLAVORS];            /* [한국어] 전기적 게이트 길이 Lelec (μm). Lphy보다 짧음 — 도핑 프로파일로 결정되는 전류 흐름 유효 길이 */
  double t_ox[NUMBER_TECH_FLAVORS];             /* [한국어] 게이트 산화막 두께 t_ox (μm). EOT(등가 산화막 두께) 기준 */
  double v_th[NUMBER_TECH_FLAVORS];             /* [한국어] 문턱전압 Vth (V). HP는 낮고(빠르지만 누설↑), LSTP는 높다(느리지만 누설↓) */
  double c_ox[NUMBER_TECH_FLAVORS];             /* [한국어] 단위 면적당 게이트 산화막 커패시턴스 Cox = ε₀·ε_SiO2 / t_ox (F/μm²) */
  double mobility_eff[NUMBER_TECH_FLAVORS];     /* [한국어] 유효 캐리어 이동도 (μm²/V·s). 드레인 전류 및 gm 계산에 사용 */
  double Vdsat[NUMBER_TECH_FLAVORS];            /* [한국어] 채널 핀치오프 포화 전압 Vdsat (V). 단채널 효과로 장채널보다 작다 */
  double c_g_ideal[NUMBER_TECH_FLAVORS];        /* [한국어] 이상적 게이트 커패시턴스 C_g_ideal = Cox × Lphy (F/μm). 부하 커패시턴스 계산 기준 */
  double c_fringe[NUMBER_TECH_FLAVORS];         /* [한국어] 게이트 프린지 커패시턴스 (F/μm). 게이트 모서리와 소스/드레인 사이의 기생 커패시턴스 */
  double c_junc[NUMBER_TECH_FLAVORS];           /* [한국어] 접합 커패시턴스 C_junc (F/μm²). 역방향 바이어스 소스/드레인 p-n 접합 커패시턴스 */
  double I_on_n[NUMBER_TECH_FLAVORS];           /* [한국어] NMOS 온전류 I_on (A/μm). 게이트=Vdd, Vds=Vdd에서의 드레인 포화 전류 */
  double Rnchannelon[NUMBER_TECH_FLAVORS];      /* [한국어] NMOS 온저항 R_nch_on (Ω·μm) = nmos_effective_resistance_multiplier × Vdd / I_on_n */
  double Rpchannelon[NUMBER_TECH_FLAVORS];      /* [한국어] PMOS 온저항 R_pch_on (Ω·μm) = n_to_p_eff_curr_drv_ratio × R_nch_on */
  double n_to_p_eff_curr_drv_ratio[NUMBER_TECH_FLAVORS]; /* [한국어] PMOS/NMOS 유효 전류 비. PMOS 홀 이동도가 낮아 통상 2.0~2.5 */
  double I_off_n[NUMBER_TECH_FLAVORS][101];     /* [한국어] NMOS 오프 누설 전류 (A/μm). [flavor][온도-300K] 인덱스: [0][0]=300K, [0][10]=310K, ..., [0][100]=400K */
  double I_g_on_n[NUMBER_TECH_FLAVORS][101];    /* [한국어] 게이트 직접 터널링 누설 전류 (A/μm). 얇은 산화막에서 발생하며 온도 의존성이 I_off보다 낮다 */
  //double I_off_p[NUMBER_TECH_FLAVORS][101];
  double gmp_to_gmn_multiplier[NUMBER_TECH_FLAVORS]; /* [한국어] PMOS/NMOS 상호 컨덕턴스 gm 비. sense amplifier 래치 gm 계산에 사용 */
  //double curr_sckt_co_eff[NUMBER_TECH_FLAVORS];
  double long_channel_leakage_reduction[NUMBER_TECH_FLAVORS]; /* [한국어] 장채널 누설 저감 계수. 단채널(MASTAR) 누설 대비 게이트 길이 증가 시 누설 감소 비율. HP에서 1/3~1/4 */

  /* [한국어] ===== 트랜지스터 파라미터 선형 보간 루프 =====
   * iter=0: tech_lo(더 큰 노드) 처리. alpha_lo = (technology - tech_hi) / (tech_lo - tech_hi).
   *   예: 75nm 요청, tech_lo=90, tech_hi=65 → alpha_lo=(75-65)/(90-65)=0.4.
   * iter=1: tech_hi(더 작은 노드) 처리. alpha_hi = (tech_lo - technology) / (tech_lo - tech_hi) = 0.6.
   *   tech_lo==tech_hi인 경우(정확히 지원 노드) iter=1 루프는 break로 건너뛴다.
   * 각 이터레이션에서 해당 노드의 트랜지스터 파라미터를 배열에 채우고,
   * 루프 말미의 += curr_alpha × 값 블록에서 g_tp에 누적한다.
   * 최종 g_tp 값 = alpha_lo × (tech_lo 파라미터) + alpha_hi × (tech_hi 파라미터).
   */
  for (iter = 0; iter <= 1; ++iter)
  {
    // linear interpolation
    if (iter == 0)
    {
      tech = tech_lo;  /* [한국어] 첫 이터레이션: 더 큰(상위) 노드 파라미터 적재 */
      if (tech_lo == tech_hi)
      {
        curr_alpha = 1;  /* [한국어] 정확히 지원 노드: 가중치 1.0, 보간 없음 */
      }
      else
      {
        /* [한국어] tech_lo 기여 가중치: 요청 노드가 tech_lo에 가까울수록 alpha → 1.
         * curr_alpha = (technology - tech_hi) / (tech_lo - tech_hi) */
        curr_alpha = (technology - tech_hi)/(tech_lo - tech_hi);
      }
    }
    else
    {
      tech = tech_hi;  /* [한국어] 두 번째 이터레이션: 더 작은(하위) 노드 파라미터 적재 */
      if (tech_lo == tech_hi)
      {
        break;  /* [한국어] 정확히 지원 노드이면 보간 불필요 — 루프 종료 */
      }
      else
      {
        /* [한국어] tech_hi 기여 가중치 = 1 - alpha_lo. 두 가중치의 합은 항상 1이다.
         * curr_alpha = (tech_lo - technology) / (tech_lo - tech_hi) */
        curr_alpha = (tech_lo - technology)/(tech_lo - tech_hi);
      }
    }

    /* [한국어] ===== 180nm 공정 노드 파라미터 블록 (ITRS 2000, 대략 1999년 양산 세대) =====
     * 180nm는 CACTI 지원 노드 중 가장 큰(구형) 공정이다. 이 시대에는 누설 전력보다 성능이
     * 주된 관심사였으므로 HP(High Performance) 트랜지스터 파라미터만 실질적으로 사용된다.
     * LSTP/LOP는 이 노드에서 별도로 지정되지 않고 HP와 같은 값을 사용한다(인덱스 0만 채워짐).
     * 데이터 출처: ITRS 2000 업데이트 + IBM 0.18μm Cu Spice 입력 (MASTAR 미지원).
     * Aggre_proj 플래그로 공격적(낙관적) 예측과 보수적 예측 사이를 선택한다.
     */
    if (tech == 180)
    {
      //180nm technology-node. Corresponds to year 1999 in ITRS
      //Only HP transistor was of interest that 180nm since leakage power was not a big issue. Performance was the king
      //MASTAR does not contain data for 0.18um process. The following parameters are projected based on ITRS 2000 update and IBM 0.18 Cu Spice input
      bool Aggre_proj = false;
      SENSE_AMP_D = .28e-9; // s
      SENSE_AMP_P = 14.7e-15; // J
      vdd[0]   = 1.5;
      Lphy[0]  = 0.12;//Lphy is the physical gate-length. micron
      Lelec[0] = 0.10;//Lelec is the electrical gate-length. micron
      t_ox[0]  = 1.2e-3*(Aggre_proj? 1.9/1.2:2);//micron
      v_th[0]  = Aggre_proj? 0.36 : 0.4407;//V
      c_ox[0]  = 1.79e-14*(Aggre_proj? 1.9/1.2:2);//F/micron2
      mobility_eff[0] = 302.16 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
      Vdsat[0] = 0.128*2; //V
      c_g_ideal[0] = (Aggre_proj? 1.9/1.2:2)*6.64e-16;//F/micron
      c_fringe[0]  = (Aggre_proj? 1.9/1.2:2)*0.08e-15;//F/micron
      c_junc[0] = (Aggre_proj? 1.9/1.2:2)*1e-15;//F/micron2
      I_on_n[0] = 750e-6;//A/micron
      //Note that nmos_effective_resistance_multiplier, n_to_p_eff_curr_drv_ratio and gmp_to_gmn_multiplier values are calculated offline
      nmos_effective_resistance_multiplier = 1.54;
      n_to_p_eff_curr_drv_ratio[0] = 2.45;
      gmp_to_gmn_multiplier[0] = 1.22;
      /* [한국어] R_nch_on: NMOS 온저항(Ω·μm) = nmos_effective_resistance_multiplier × Vdd / I_on_n.
       * nmos_effective_resistance_multiplier는 Ieff/Idsat 비로, 실효 스위칭 저항이 Idsat 기반
       * 이상적 저항보다 큰 정도를 보정한다. 오프라인으로 SPICE 시뮬레이션으로 사전 계산됨. */
      Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];//ohm-micron
      /* [한국어] R_pch_on: PMOS 온저항 = n_to_p_eff_curr_drv_ratio × R_nch_on.
       * PMOS는 홀 이동도가 낮아 같은 폭 대비 전류가 작으므로 저항이 더 크다. */
      Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];//ohm-micron
      long_channel_leakage_reduction[0] = 1;  /* [한국어] 180nm: 단채널 효과가 크지 않아 장채널 누설 저감 계수=1 (저감 없음) */
      /* [한국어] I_off_n[0][idx]: 온도 (300+idx)K에서의 NMOS 오프 누설 전류 (A/μm).
       * 인덱스 간격 = 10K. 예: [0][0]=300K, [0][10]=310K, [0][100]=400K.
       * 온도가 높아질수록 누설 전류는 지수적으로 증가한다 (Boltzmann 분포). */
      I_off_n[0][0]  = 7e-10;//A/micron
      I_off_n[0][10] = 8.26e-10;
      I_off_n[0][20] = 9.74e-10;
      I_off_n[0][30] = 1.15e-9;
      I_off_n[0][40] = 1.35e-9;
      I_off_n[0][50] = 1.60e-9;
      I_off_n[0][60] = 1.88e-9;
      I_off_n[0][70] = 2.29e-9;
      I_off_n[0][80] = 2.70e-9;
      I_off_n[0][90] = 3.19e-9;
      I_off_n[0][100] = 3.76e-9;

      I_g_on_n[0][0]  = 1.65e-10;//A/micron
      I_g_on_n[0][10] = 1.65e-10;
      I_g_on_n[0][20] = 1.65e-10;
      I_g_on_n[0][30] = 1.65e-10;
      I_g_on_n[0][40] = 1.65e-10;
      I_g_on_n[0][50] = 1.65e-10;
      I_g_on_n[0][60] = 1.65e-10;
      I_g_on_n[0][70] = 1.65e-10;
      I_g_on_n[0][80] = 1.65e-10;
      I_g_on_n[0][90] = 1.65e-10;
      I_g_on_n[0][100] = 1.65e-10;

      //SRAM cell properties
      curr_Wmemcella_sram = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
      curr_area_cell_sram = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_sram = 1.46;
      //CAM cell properties //TODO: data need to be revisited
      curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
      curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;//360
      curr_asp_ratio_cell_cam = 2.92;//2.5
      //Empirical undifferetiated core/FU coefficient
      curr_logic_scaling_co_eff  = 1.5;//linear scaling from 90nm
      curr_core_tx_density       = 1.25*0.7*0.7*0.4;
      curr_sckt_co_eff           = 1.11;
      curr_chip_layout_overhead  = 1.0;//die measurement results based on Niagara 1 and 2
      curr_macro_layout_overhead = 1.0;//EDA placement and routing tool rule of thumb

    }

    /* [한국어] ===== 90nm 공정 노드 파라미터 블록 (ITRS 2004, 대략 2004년 양산 세대) =====
     * 90nm는 누설 전력이 처음 심각한 문제로 대두된 세대다. HP/LSTP/LOP 세 플레이버가 모두
     * 지원되며, LSTP와 LOP에서 뚜렷하게 높은 Vth와 낮은 I_off를 확인할 수 있다.
     * lp_dram(LP-DRAM)과 comm_dram(Commodity DRAM) 두 타입의 메모리 셀 파라미터를 지원한다.
     * 데이터 출처: ITRS 2004, MASTAR 시뮬레이션.
     */
    if (tech == 90)
    {
      SENSE_AMP_D = .28e-9; // s
      SENSE_AMP_P = 14.7e-15; // J
      //90nm technology-node. Corresponds to year 2004 in ITRS
      //ITRS HP device type
      vdd[0]   = 1.2;
      Lphy[0]  = 0.037;//Lphy is the physical gate-length. micron
      Lelec[0] = 0.0266;//Lelec is the electrical gate-length. micron
      t_ox[0]  = 1.2e-3;//micron
      v_th[0]  = 0.23707;//V
      c_ox[0]  = 1.79e-14;//F/micron2
      mobility_eff[0] = 342.16 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
      Vdsat[0] = 0.128; //V
      c_g_ideal[0] = 6.64e-16;//F/micron
      c_fringe[0]  = 0.08e-15;//F/micron
      c_junc[0] = 1e-15;//F/micron2
      I_on_n[0] = 1076.9e-6;//A/micron
      //I_on_p[0] = 712.6e-6;//A/micron
      //Note that nmos_effective_resistance_multiplier, n_to_p_eff_curr_drv_ratio and gmp_to_gmn_multiplier values are calculated offline
      nmos_effective_resistance_multiplier = 1.54;
      n_to_p_eff_curr_drv_ratio[0] = 2.45;
      gmp_to_gmn_multiplier[0] = 1.22;
      Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];//ohm-micron
      Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];//ohm-micron
      long_channel_leakage_reduction[0] = 1;
      I_off_n[0][0]  = 3.24e-8;//A/micron
      I_off_n[0][10] = 4.01e-8;
      I_off_n[0][20] = 4.90e-8;
      I_off_n[0][30] = 5.92e-8;
      I_off_n[0][40] = 7.08e-8;
      I_off_n[0][50] = 8.38e-8;
      I_off_n[0][60] = 9.82e-8;
      I_off_n[0][70] = 1.14e-7;
      I_off_n[0][80] = 1.29e-7;
      I_off_n[0][90] = 1.43e-7;
      I_off_n[0][100] = 1.54e-7;

      I_g_on_n[0][0]  = 1.65e-8;//A/micron
      I_g_on_n[0][10] = 1.65e-8;
      I_g_on_n[0][20] = 1.65e-8;
      I_g_on_n[0][30] = 1.65e-8;
      I_g_on_n[0][40] = 1.65e-8;
      I_g_on_n[0][50] = 1.65e-8;
      I_g_on_n[0][60] = 1.65e-8;
      I_g_on_n[0][70] = 1.65e-8;
      I_g_on_n[0][80] = 1.65e-8;
      I_g_on_n[0][90] = 1.65e-8;
      I_g_on_n[0][100] = 1.65e-8;

      //ITRS LSTP device type
      vdd[1]   = 1.3;
      Lphy[1]  = 0.075;
      Lelec[1] = 0.0486;
      t_ox[1]  = 2.2e-3;
      v_th[1]  = 0.48203;
      c_ox[1]  = 1.22e-14;
      mobility_eff[1] = 356.76 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[1] = 0.373;
      c_g_ideal[1] = 9.15e-16;
      c_fringe[1]  = 0.08e-15;
      c_junc[1] = 1e-15;
      I_on_n[1] = 503.6e-6;
      nmos_effective_resistance_multiplier = 1.92;
      n_to_p_eff_curr_drv_ratio[1] = 2.44;
      gmp_to_gmn_multiplier[1] =0.88;
      Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];
      Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];
      long_channel_leakage_reduction[1] = 1;
      I_off_n[1][0]  = 2.81e-12;
      I_off_n[1][10] = 4.76e-12;
      I_off_n[1][20] = 7.82e-12;
      I_off_n[1][30] = 1.25e-11;
      I_off_n[1][40] = 1.94e-11;
      I_off_n[1][50] = 2.94e-11;
      I_off_n[1][60] = 4.36e-11;
      I_off_n[1][70] = 6.32e-11;
      I_off_n[1][80] = 8.95e-11;
      I_off_n[1][90] = 1.25e-10;
      I_off_n[1][100] = 1.7e-10;

      I_g_on_n[1][0]  = 3.87e-11;//A/micron
      I_g_on_n[1][10] = 3.87e-11;
      I_g_on_n[1][20] = 3.87e-11;
      I_g_on_n[1][30] = 3.87e-11;
      I_g_on_n[1][40] = 3.87e-11;
      I_g_on_n[1][50] = 3.87e-11;
      I_g_on_n[1][60] = 3.87e-11;
      I_g_on_n[1][70] = 3.87e-11;
      I_g_on_n[1][80] = 3.87e-11;
      I_g_on_n[1][90] = 3.87e-11;
      I_g_on_n[1][100] = 3.87e-11;

      //ITRS LOP device type
      vdd[2] = 0.9;
      Lphy[2] = 0.053;
      Lelec[2] = 0.0354;
      t_ox[2] = 1.5e-3;
      v_th[2] = 0.30764;
      c_ox[2] = 1.59e-14;
      mobility_eff[2] = 460.39 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[2] = 0.113;
      c_g_ideal[2] = 8.45e-16;
      c_fringe[2] = 0.08e-15;
      c_junc[2] = 1e-15;
      I_on_n[2] = 386.6e-6;
      nmos_effective_resistance_multiplier = 1.77;
      n_to_p_eff_curr_drv_ratio[2] = 2.54;
      gmp_to_gmn_multiplier[2] = 0.98;
      Rnchannelon[2] = nmos_effective_resistance_multiplier * vdd[2] / I_on_n[2];
      Rpchannelon[2] = n_to_p_eff_curr_drv_ratio[2] * Rnchannelon[2];
      long_channel_leakage_reduction[2] = 1;
      I_off_n[2][0] = 2.14e-9;
      I_off_n[2][10] = 2.9e-9;
      I_off_n[2][20] = 3.87e-9;
      I_off_n[2][30] = 5.07e-9;
      I_off_n[2][40] = 6.54e-9;
      I_off_n[2][50] = 8.27e-8;
      I_off_n[2][60] = 1.02e-7;
      I_off_n[2][70] = 1.20e-7;
      I_off_n[2][80] = 1.36e-8;
      I_off_n[2][90] = 1.52e-8;
      I_off_n[2][100] = 1.73e-8;

      I_g_on_n[2][0]  = 4.31e-8;//A/micron
      I_g_on_n[2][10] = 4.31e-8;
      I_g_on_n[2][20] = 4.31e-8;
      I_g_on_n[2][30] = 4.31e-8;
      I_g_on_n[2][40] = 4.31e-8;
      I_g_on_n[2][50] = 4.31e-8;
      I_g_on_n[2][60] = 4.31e-8;
      I_g_on_n[2][70] = 4.31e-8;
      I_g_on_n[2][80] = 4.31e-8;
      I_g_on_n[2][90] = 4.31e-8;
      I_g_on_n[2][100] = 4.31e-8;

      if (ram_cell_tech_type == lp_dram)
      {
        //LP-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.2;
        Lphy[3] = 0.12;
        Lelec[3] = 0.0756;
        curr_v_th_dram_access_transistor = 0.4545;
        width_dram_access_transistor = 0.14;
        curr_I_on_dram_cell = 45e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 21.1e-12;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = 0.168;
        curr_asp_ratio_cell_dram = 1.46;
        curr_c_dram_cell = 20e-15;

        //LP-DRAM wordline transistor parameters
        curr_vpp = 1.6;
        t_ox[3] = 2.2e-3;
        v_th[3] = 0.4545;
        c_ox[3] = 1.22e-14;
        mobility_eff[3] =  323.95 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.3;
        c_g_ideal[3] = 1.47e-15;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 321.6e-6;
        nmos_effective_resistance_multiplier = 1.65;
        n_to_p_eff_curr_drv_ratio[3] = 1.95;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0] = 1.42e-11;
        I_off_n[3][10] = 2.25e-11;
        I_off_n[3][20] = 3.46e-11;
        I_off_n[3][30] = 5.18e-11;
        I_off_n[3][40] = 7.58e-11;
        I_off_n[3][50] = 1.08e-10;
        I_off_n[3][60] = 1.51e-10;
        I_off_n[3][70] = 2.02e-10;
        I_off_n[3][80] = 2.57e-10;
        I_off_n[3][90] = 3.14e-10;
        I_off_n[3][100] = 3.85e-10;
      }
      else if (ram_cell_tech_type == comm_dram)
      {
        //COMM-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.6;
        Lphy[3] = 0.09;
        Lelec[3] = 0.0576;
        curr_v_th_dram_access_transistor = 1;
        width_dram_access_transistor = 0.09;
        curr_I_on_dram_cell = 20e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 1e-15;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = 6*0.09*0.09;
        curr_asp_ratio_cell_dram = 1.5;
        curr_c_dram_cell = 30e-15;

        //COMM-DRAM wordline transistor parameters
        curr_vpp = 3.7;
        t_ox[3] = 5.5e-3;
        v_th[3] = 1.0;
        c_ox[3] = 5.65e-15;
        mobility_eff[3] =  302.2 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.32;
        c_g_ideal[3] = 5.08e-16;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 1094.3e-6;
        nmos_effective_resistance_multiplier = 1.62;
        n_to_p_eff_curr_drv_ratio[3] = 2.05;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0] = 5.80e-15;
        I_off_n[3][10] = 1.21e-14;
        I_off_n[3][20] = 2.42e-14;
        I_off_n[3][30] = 4.65e-14;
        I_off_n[3][40] = 8.60e-14;
        I_off_n[3][50] = 1.54e-13;
        I_off_n[3][60] = 2.66e-13;
        I_off_n[3][70] = 4.45e-13;
        I_off_n[3][80] = 7.17e-13;
        I_off_n[3][90] = 1.11e-12;
        I_off_n[3][100] = 1.67e-12;
      }

      //SRAM cell properties
      curr_Wmemcella_sram = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
      curr_area_cell_sram = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_sram = 1.46;
      //CAM cell properties //TODO: data need to be revisited
      curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
      curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;//360
      curr_asp_ratio_cell_cam = 2.92;//2.5
      //Empirical undifferetiated core/FU coefficient
      curr_logic_scaling_co_eff  = 1;
      curr_core_tx_density       = 1.25*0.7*0.7;
      curr_sckt_co_eff           = 1.1539;
      curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
      curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb


    }

    /* [한국어] ===== 65nm 공정 노드 파라미터 블록 (ITRS 2007, 대략 2007년 양산 세대) =====
     * 65nm는 Intel Merom/Penryn, IBM Cell 프로세서 등 주요 CPU/GPU가 채택한 세대다.
     * long_channel_leakage_reduction이 1/3.74(HP), 1/2.82(LSTP), 1/2.05(LOP)로 처음
     * 1보다 작아지는 노드 — 단채널 효과로 인해 짧은 게이트의 누설이 급증함을 반영한다.
     * curr_logic_scaling_co_eff=0.7: Intel 65nm 실측 기반, 면적 제곱 비 대신 선형 스케일링.
     * 데이터 출처: ITRS 2007, MASTAR 시뮬레이션 (380K에서 Lgate 증가 실험 포함).
     */
    if (tech == 65)
    { //65nm technology-node. Corresponds to year 2007 in ITRS
      //ITRS HP device type
      SENSE_AMP_D = .2e-9; // s
      SENSE_AMP_P = 5.7e-15; // J
      vdd[0] = 1.1;
      Lphy[0] = 0.025;
      Lelec[0] = 0.019;
      t_ox[0] = 1.1e-3;
      v_th[0] = .19491;
      c_ox[0] = 1.88e-14;
      mobility_eff[0] = 436.24 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[0] = 7.71e-2;
      c_g_ideal[0] = 4.69e-16;
      c_fringe[0] = 0.077e-15;
      c_junc[0] = 1e-15;
      I_on_n[0] = 1197.2e-6;
      nmos_effective_resistance_multiplier = 1.50;
      n_to_p_eff_curr_drv_ratio[0] = 2.41;
      gmp_to_gmn_multiplier[0] = 1.38;
      Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];
      Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];
      long_channel_leakage_reduction[0] = 1/3.74;
      //Using MASTAR, @380K, increase Lgate until Ion reduces to 90% or Lgate increase by 10%, whichever comes first
      //Ioff(Lgate normal)/Ioff(Lgate long)= 3.74.
      I_off_n[0][0] = 1.96e-7;
      I_off_n[0][10] = 2.29e-7;
      I_off_n[0][20] = 2.66e-7;
      I_off_n[0][30] = 3.05e-7;
      I_off_n[0][40] = 3.49e-7;
      I_off_n[0][50] = 3.95e-7;
      I_off_n[0][60] = 4.45e-7;
      I_off_n[0][70] = 4.97e-7;
      I_off_n[0][80] = 5.48e-7;
      I_off_n[0][90] = 5.94e-7;
      I_off_n[0][100] = 6.3e-7;
      I_g_on_n[0][0]  = 4.09e-8;//A/micron
      I_g_on_n[0][10] = 4.09e-8;
      I_g_on_n[0][20] = 4.09e-8;
      I_g_on_n[0][30] = 4.09e-8;
      I_g_on_n[0][40] = 4.09e-8;
      I_g_on_n[0][50] = 4.09e-8;
      I_g_on_n[0][60] = 4.09e-8;
      I_g_on_n[0][70] = 4.09e-8;
      I_g_on_n[0][80] = 4.09e-8;
      I_g_on_n[0][90] = 4.09e-8;
      I_g_on_n[0][100] = 4.09e-8;

      //ITRS LSTP device type
      vdd[1] = 1.2;
      Lphy[1] = 0.045;
      Lelec[1] = 0.0298;
      t_ox[1] = 1.9e-3;
      v_th[1] = 0.52354;
      c_ox[1] = 1.36e-14;
      mobility_eff[1] = 341.21 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[1] = 0.128;
      c_g_ideal[1] = 6.14e-16;
      c_fringe[1] = 0.08e-15;
      c_junc[1] = 1e-15;
      I_on_n[1] = 519.2e-6;
      nmos_effective_resistance_multiplier = 1.96;
      n_to_p_eff_curr_drv_ratio[1] = 2.23;
      gmp_to_gmn_multiplier[1] = 0.99;
      Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];
      Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];
      long_channel_leakage_reduction[1] = 1/2.82;
      I_off_n[1][0] = 9.12e-12;
      I_off_n[1][10] = 1.49e-11;
      I_off_n[1][20] = 2.36e-11;
      I_off_n[1][30] = 3.64e-11;
      I_off_n[1][40] = 5.48e-11;
      I_off_n[1][50] = 8.05e-11;
      I_off_n[1][60] = 1.15e-10;
      I_off_n[1][70] = 1.59e-10;
      I_off_n[1][80] = 2.1e-10;
      I_off_n[1][90] = 2.62e-10;
      I_off_n[1][100] = 3.21e-10;

      I_g_on_n[1][0]  = 1.09e-10;//A/micron
      I_g_on_n[1][10] = 1.09e-10;
      I_g_on_n[1][20] = 1.09e-10;
      I_g_on_n[1][30] = 1.09e-10;
      I_g_on_n[1][40] = 1.09e-10;
      I_g_on_n[1][50] = 1.09e-10;
      I_g_on_n[1][60] = 1.09e-10;
      I_g_on_n[1][70] = 1.09e-10;
      I_g_on_n[1][80] = 1.09e-10;
      I_g_on_n[1][90] = 1.09e-10;
      I_g_on_n[1][100] = 1.09e-10;

      //ITRS LOP device type
      vdd[2] = 0.8;
      Lphy[2] = 0.032;
      Lelec[2] = 0.0216;
      t_ox[2] = 1.2e-3;
      v_th[2] = 0.28512;
      c_ox[2] = 1.87e-14;
      mobility_eff[2] = 495.19 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[2] = 0.292;
      c_g_ideal[2] = 6e-16;
      c_fringe[2] = 0.08e-15;
      c_junc[2] = 1e-15;
      I_on_n[2] = 573.1e-6;
      nmos_effective_resistance_multiplier = 1.82;
      n_to_p_eff_curr_drv_ratio[2] = 2.28;
      gmp_to_gmn_multiplier[2] = 1.11;
      Rnchannelon[2] = nmos_effective_resistance_multiplier * vdd[2] / I_on_n[2];
      Rpchannelon[2] = n_to_p_eff_curr_drv_ratio[2] * Rnchannelon[2];
      long_channel_leakage_reduction[2] = 1/2.05;
      I_off_n[2][0] = 4.9e-9;
      I_off_n[2][10] = 6.49e-9;
      I_off_n[2][20] = 8.45e-9;
      I_off_n[2][30] = 1.08e-8;
      I_off_n[2][40] = 1.37e-8;
      I_off_n[2][50] = 1.71e-8;
      I_off_n[2][60] = 2.09e-8;
      I_off_n[2][70] = 2.48e-8;
      I_off_n[2][80] = 2.84e-8;
      I_off_n[2][90] = 3.13e-8;
      I_off_n[2][100] = 3.42e-8;

      I_g_on_n[2][0]  = 9.61e-9;//A/micron
      I_g_on_n[2][10] = 9.61e-9;
      I_g_on_n[2][20] = 9.61e-9;
      I_g_on_n[2][30] = 9.61e-9;
      I_g_on_n[2][40] = 9.61e-9;
      I_g_on_n[2][50] = 9.61e-9;
      I_g_on_n[2][60] = 9.61e-9;
      I_g_on_n[2][70] = 9.61e-9;
      I_g_on_n[2][80] = 9.61e-9;
      I_g_on_n[2][90] = 9.61e-9;
      I_g_on_n[2][100] = 9.61e-9;

      if (ram_cell_tech_type == lp_dram)
      {
        //LP-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.2;
        Lphy[3] = 0.12;
        Lelec[3] = 0.0756;
        curr_v_th_dram_access_transistor = 0.43806;
        width_dram_access_transistor = 0.09;
        curr_I_on_dram_cell = 36e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 19.6e-12;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = 0.11;
        curr_asp_ratio_cell_dram = 1.46;
        curr_c_dram_cell = 20e-15;

        //LP-DRAM wordline transistor parameters
        curr_vpp = 1.6;
        t_ox[3] = 2.2e-3;
        v_th[3] = 0.43806;
        c_ox[3] = 1.22e-14;
        mobility_eff[3] =  328.32 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.43806;
        c_g_ideal[3] = 1.46e-15;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15 ;
        I_on_n[3] = 399.8e-6;
        nmos_effective_resistance_multiplier = 1.65;
        n_to_p_eff_curr_drv_ratio[3] = 2.05;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0]  = 2.23e-11;
        I_off_n[3][10] = 3.46e-11;
        I_off_n[3][20] = 5.24e-11;
        I_off_n[3][30] = 7.75e-11;
        I_off_n[3][40] = 1.12e-10;
        I_off_n[3][50] = 1.58e-10;
        I_off_n[3][60] = 2.18e-10;
        I_off_n[3][70] = 2.88e-10;
        I_off_n[3][80] = 3.63e-10;
        I_off_n[3][90] = 4.41e-10;
        I_off_n[3][100] = 5.36e-10;
      }
      else if (ram_cell_tech_type == comm_dram)
      {
        //COMM-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.3;
        Lphy[3] = 0.065;
        Lelec[3] = 0.0426;
        curr_v_th_dram_access_transistor = 1;
        width_dram_access_transistor = 0.065;
        curr_I_on_dram_cell = 20e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 1e-15;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = 6*0.065*0.065;
        curr_asp_ratio_cell_dram = 1.5;
        curr_c_dram_cell = 30e-15;

        //COMM-DRAM wordline transistor parameters
        curr_vpp = 3.3;
        t_ox[3] = 5e-3;
        v_th[3] = 1.0;
        c_ox[3] = 6.16e-15;
        mobility_eff[3] =  303.44 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.385;
        c_g_ideal[3] = 4e-16;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15 ;
        I_on_n[3] = 1031e-6;
        nmos_effective_resistance_multiplier = 1.69;
        n_to_p_eff_curr_drv_ratio[3] = 2.39;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0]  = 1.80e-14;
        I_off_n[3][10] = 3.64e-14;
        I_off_n[3][20] = 7.03e-14;
        I_off_n[3][30] = 1.31e-13;
        I_off_n[3][40] = 2.35e-13;
        I_off_n[3][50] = 4.09e-13;
        I_off_n[3][60] = 6.89e-13;
        I_off_n[3][70] = 1.13e-12;
        I_off_n[3][80] = 1.78e-12;
        I_off_n[3][90] = 2.71e-12;
        I_off_n[3][100] = 3.99e-12;
      }

      //SRAM cell properties
      curr_Wmemcella_sram = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
      curr_area_cell_sram = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_sram = 1.46;
      //CAM cell properties //TODO: data need to be revisited
      curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
      curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_cam = 2.92;
      //Empirical undifferetiated core/FU coefficient
      curr_logic_scaling_co_eff = 0.7; //Rather than scale proportionally to square of feature size, only scale linearly according to IBM cell processor
      curr_core_tx_density      = 1.25*0.7;
      curr_sckt_co_eff           = 1.1359;
      curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
      curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb
    }

    /* [한국어] ===== 45nm 공정 노드 파라미터 블록 (ITRS 2010, 대략 2010년 양산 세대) =====
     * 45nm는 Intel Penryn(Tick), IBM 45nm SOI 등에 해당한다. t_ox가 0.65nm까지 얇아져
     * Cox가 크게 증가하지만(3.77e-14 F/μm²) 게이트 누설이 심화되는 세대다.
     * HP 파라미터는 ITRS PMOS MASTAR 데이터 이상으로 인해 65nm 값을 일부 재사용한다
     * (n_to_p_eff_curr_drv_ratio, gmp_to_gmn_multiplier를 65nm와 동일하게 설정).
     * 데이터 출처: ITRS 2010, MASTAR (HP는 @380K Lgate 실험 포함).
     */
    if (tech == 45)
    { //45nm technology-node. Corresponds to year 2010 in ITRS
      //ITRS HP device type
      SENSE_AMP_D = .04e-9; // s
      SENSE_AMP_P = 2.7e-15; // J
      vdd[0] = 1.0;
      Lphy[0] = 0.018;
      Lelec[0] = 0.01345;
      t_ox[0] = 0.65e-3;
      v_th[0] = .18035;
      c_ox[0] = 3.77e-14;
      mobility_eff[0] = 266.68 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[0] = 9.38E-2;
      c_g_ideal[0] = 6.78e-16;
      c_fringe[0] = 0.05e-15;
      c_junc[0] = 1e-15;
      I_on_n[0] = 2046.6e-6;
      //There are certain problems with the ITRS PMOS numbers in MASTAR for 45nm. So we are using 65nm values of
      //n_to_p_eff_curr_drv_ratio and gmp_to_gmn_multiplier for 45nm
      nmos_effective_resistance_multiplier = 1.51;
      n_to_p_eff_curr_drv_ratio[0] = 2.41;
      gmp_to_gmn_multiplier[0] = 1.38;
      Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];
      Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];
      long_channel_leakage_reduction[0] = 1/3.546;//Using MASTAR, @380K, increase Lgate until Ion reduces to 90%, Ioff(Lgate normal)/Ioff(Lgate long)= 3.74
      I_off_n[0][0] = 2.8e-7;
      I_off_n[0][10] = 3.28e-7;
      I_off_n[0][20] = 3.81e-7;
      I_off_n[0][30] = 4.39e-7;
      I_off_n[0][40] = 5.02e-7;
      I_off_n[0][50] = 5.69e-7;
      I_off_n[0][60] = 6.42e-7;
      I_off_n[0][70] = 7.2e-7;
      I_off_n[0][80] = 8.03e-7;
      I_off_n[0][90] = 8.91e-7;
      I_off_n[0][100] = 9.84e-7;

      I_g_on_n[0][0]  = 3.59e-8;//A/micron
      I_g_on_n[0][10] = 3.59e-8;
      I_g_on_n[0][20] = 3.59e-8;
      I_g_on_n[0][30] = 3.59e-8;
      I_g_on_n[0][40] = 3.59e-8;
      I_g_on_n[0][50] = 3.59e-8;
      I_g_on_n[0][60] = 3.59e-8;
      I_g_on_n[0][70] = 3.59e-8;
      I_g_on_n[0][80] = 3.59e-8;
      I_g_on_n[0][90] = 3.59e-8;
      I_g_on_n[0][100] = 3.59e-8;

      //ITRS LSTP device type
      vdd[1] = 1.1;
      Lphy[1] =  0.028;
      Lelec[1] = 0.0212;
      t_ox[1] = 1.4e-3;
      v_th[1] = 0.50245;
      c_ox[1] = 2.01e-14;
      mobility_eff[1] =  363.96 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[1] = 9.12e-2;
      c_g_ideal[1] = 5.18e-16;
      c_fringe[1] = 0.08e-15;
      c_junc[1] = 1e-15;
      I_on_n[1] = 666.2e-6;
      nmos_effective_resistance_multiplier = 1.99;
      n_to_p_eff_curr_drv_ratio[1] = 2.23;
      gmp_to_gmn_multiplier[1] = 0.99;
      Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];
      Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];
      long_channel_leakage_reduction[1] = 1/2.08;
      I_off_n[1][0] = 1.01e-11;
      I_off_n[1][10] = 1.65e-11;
      I_off_n[1][20] = 2.62e-11;
      I_off_n[1][30] = 4.06e-11;
      I_off_n[1][40] = 6.12e-11;
      I_off_n[1][50] = 9.02e-11;
      I_off_n[1][60] = 1.3e-10;
      I_off_n[1][70] = 1.83e-10;
      I_off_n[1][80] = 2.51e-10;
      I_off_n[1][90] = 3.29e-10;
      I_off_n[1][100] = 4.1e-10;

      I_g_on_n[1][0]  = 9.47e-12;//A/micron
      I_g_on_n[1][10] = 9.47e-12;
      I_g_on_n[1][20] = 9.47e-12;
      I_g_on_n[1][30] = 9.47e-12;
      I_g_on_n[1][40] = 9.47e-12;
      I_g_on_n[1][50] = 9.47e-12;
      I_g_on_n[1][60] = 9.47e-12;
      I_g_on_n[1][70] = 9.47e-12;
      I_g_on_n[1][80] = 9.47e-12;
      I_g_on_n[1][90] = 9.47e-12;
      I_g_on_n[1][100] = 9.47e-12;

      //ITRS LOP device type
      vdd[2] = 0.7;
      Lphy[2] = 0.022;
      Lelec[2] = 0.016;
      t_ox[2] = 0.9e-3;
      v_th[2] = 0.22599;
      c_ox[2] = 2.82e-14;//F/micron2
      mobility_eff[2] = 508.9 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[2] = 5.71e-2;
      c_g_ideal[2] = 6.2e-16;
      c_fringe[2] = 0.073e-15;
      c_junc[2] = 1e-15;
      I_on_n[2] = 748.9e-6;
      nmos_effective_resistance_multiplier = 1.76;
      n_to_p_eff_curr_drv_ratio[2] = 2.28;
      gmp_to_gmn_multiplier[2] = 1.11;
      Rnchannelon[2] = nmos_effective_resistance_multiplier * vdd[2] / I_on_n[2];
      Rpchannelon[2] = n_to_p_eff_curr_drv_ratio[2] * Rnchannelon[2];
      long_channel_leakage_reduction[2] = 1/1.92;
      I_off_n[2][0] = 4.03e-9;
      I_off_n[2][10] = 5.02e-9;
      I_off_n[2][20] = 6.18e-9;
      I_off_n[2][30] = 7.51e-9;
      I_off_n[2][40] = 9.04e-9;
      I_off_n[2][50] = 1.08e-8;
      I_off_n[2][60] = 1.27e-8;
      I_off_n[2][70] = 1.47e-8;
      I_off_n[2][80] = 1.66e-8;
      I_off_n[2][90] = 1.84e-8;
      I_off_n[2][100] = 2.03e-8;

      I_g_on_n[2][0]  = 3.24e-8;//A/micron
      I_g_on_n[2][10] = 4.01e-8;
      I_g_on_n[2][20] = 4.90e-8;
      I_g_on_n[2][30] = 5.92e-8;
      I_g_on_n[2][40] = 7.08e-8;
      I_g_on_n[2][50] = 8.38e-8;
      I_g_on_n[2][60] = 9.82e-8;
      I_g_on_n[2][70] = 1.14e-7;
      I_g_on_n[2][80] = 1.29e-7;
      I_g_on_n[2][90] = 1.43e-7;
      I_g_on_n[2][100] = 1.54e-7;

      if (ram_cell_tech_type == lp_dram)
      {
        //LP-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.1;
        Lphy[3] = 0.078;
        Lelec[3] = 0.0504;// Assume Lelec is 30% lesser than Lphy for DRAM access and wordline transistors.
        curr_v_th_dram_access_transistor = 0.44559;
        width_dram_access_transistor = 0.079;
        curr_I_on_dram_cell = 36e-6;//A
        curr_I_off_dram_cell_worst_case_length_temp = 19.5e-12;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram  = 0;
        curr_area_cell_dram = width_dram_access_transistor * Lphy[3] * 10.0;
        curr_asp_ratio_cell_dram = 1.46;
        curr_c_dram_cell = 20e-15;

        //LP-DRAM wordline transistor parameters
        curr_vpp = 1.5;
        t_ox[3] = 2.1e-3;
        v_th[3] = 0.44559;
        c_ox[3] = 1.41e-14;
        mobility_eff[3] =   426.30 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.181;
        c_g_ideal[3] = 1.10e-15;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 456e-6;
        nmos_effective_resistance_multiplier = 1.65;
        n_to_p_eff_curr_drv_ratio[3] = 2.05;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0] = 2.54e-11;
        I_off_n[3][10] = 3.94e-11;
        I_off_n[3][20] = 5.95e-11;
        I_off_n[3][30] = 8.79e-11;
        I_off_n[3][40] = 1.27e-10;
        I_off_n[3][50] = 1.79e-10;
        I_off_n[3][60] = 2.47e-10;
        I_off_n[3][70] = 3.31e-10;
        I_off_n[3][80] = 4.26e-10;
        I_off_n[3][90] = 5.27e-10;
        I_off_n[3][100] = 6.46e-10;
      }
      else if (ram_cell_tech_type == comm_dram)
      {
        //COMM-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.1;
        Lphy[3] = 0.045;
        Lelec[3] = 0.0298;
        curr_v_th_dram_access_transistor = 1;
        width_dram_access_transistor = 0.045;
        curr_I_on_dram_cell = 20e-6;//A
        curr_I_off_dram_cell_worst_case_length_temp = 1e-15;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram  = 0;
        curr_area_cell_dram = 6*0.045*0.045;
        curr_asp_ratio_cell_dram = 1.5;
        curr_c_dram_cell = 30e-15;

        //COMM-DRAM wordline transistor parameters
        curr_vpp = 2.7;
        t_ox[3] = 4e-3;
        v_th[3] = 1.0;
        c_ox[3] = 7.98e-15;
        mobility_eff[3] = 368.58 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.147;
        c_g_ideal[3] = 3.59e-16;
        c_fringe[3] = 0.08e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 999.4e-6;
        nmos_effective_resistance_multiplier = 1.69;
        n_to_p_eff_curr_drv_ratio[3] = 1.95;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0] = 1.31e-14;
        I_off_n[3][10] = 2.68e-14;
        I_off_n[3][20] = 5.25e-14;
        I_off_n[3][30] = 9.88e-14;
        I_off_n[3][40] = 1.79e-13;
        I_off_n[3][50] = 3.15e-13;
        I_off_n[3][60] = 5.36e-13;
        I_off_n[3][70] = 8.86e-13;
        I_off_n[3][80] = 1.42e-12;
        I_off_n[3][90] = 2.20e-12;
        I_off_n[3][100] = 3.29e-12;
      }


      //SRAM cell properties
      curr_Wmemcella_sram = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
      curr_area_cell_sram = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_sram = 1.46;
      //CAM cell properties //TODO: data need to be revisited
      curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
      curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_cam = 2.92;
      //Empirical undifferetiated core/FU coefficient
      curr_logic_scaling_co_eff = 0.7*0.7;
      curr_core_tx_density      = 1.25;
      curr_sckt_co_eff           = 1.1387;
      curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
      curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb
    }

    /* [한국어] ===== 32nm 공정 노드 파라미터 블록 (ITRS 2013, 대략 2013년 양산 세대) =====
     * 32nm는 MPU/ASIC M1 half-pitch 32nm에 해당하며 SOI(Silicon-On-Insulator) 공정 수치를
     * HP와 LSTP에 사용한다. SOI 구조는 접합 누설(c_junc)이 감소하며 단채널 특성이 개선된다.
     * HP의 I_off는 300K~340K 구간에서 거의 일정하다가 고온에서 급격히 증가하는 특이 패턴을 보인다
     * (MASTAR @300K에서 Lgate 증가 5% 제한 조건).
     * LOP/LSTP/HP 모두 t_ox가 0.5~1.2nm 수준으로 gate direct tunneling 누설이 중요해진다.
     * 데이터 출처: ITRS 2012, MASTAR SOI (HP는 @300K Lgate 실험).
     */
    if (tech == 32)
    {
      SENSE_AMP_D = .03e-9; // s
      SENSE_AMP_P = 2.16e-15; // J
      //For 2013, MPU/ASIC stagger-contacted M1 half-pitch is 32 nm (so this is 32 nm
      //technology i.e. FEATURESIZE = 0.032). Using the SOI process numbers for
      //HP and LSTP.
      vdd[0] = 0.9;
      Lphy[0] = 0.013;
      Lelec[0] = 0.01013;
      t_ox[0] = 0.5e-3;
      v_th[0] = 0.21835;
      c_ox[0] = 4.11e-14;
      mobility_eff[0] = 361.84 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[0] = 5.09E-2;
      c_g_ideal[0] = 5.34e-16;
      c_fringe[0] = 0.04e-15;
      c_junc[0] = 1e-15;
      I_on_n[0] =  2211.7e-6;
      nmos_effective_resistance_multiplier = 1.49;
      n_to_p_eff_curr_drv_ratio[0] = 2.41;
      gmp_to_gmn_multiplier[0] = 1.38;
      Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];//ohm-micron
      Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];//ohm-micron
      long_channel_leakage_reduction[0] = 1/3.706;
      //Using MASTAR, @300K (380K does not work in MASTAR), increase Lgate until Ion reduces to 95% or Lgate increase by 5% (DG device can only increase by 5%),
      //whichever comes first
      I_off_n[0][0] = 1.52e-7;
      I_off_n[0][10] = 1.55e-7;
      I_off_n[0][20] = 1.59e-7;
      I_off_n[0][30] = 1.68e-7;
      I_off_n[0][40] = 1.90e-7;
      I_off_n[0][50] = 2.69e-7;
      I_off_n[0][60] = 5.32e-7;
      I_off_n[0][70] = 1.02e-6;
      I_off_n[0][80] = 1.62e-6;
      I_off_n[0][90] = 2.73e-6;
      I_off_n[0][100] = 6.1e-6;

      I_g_on_n[0][0]  = 6.55e-8;//A/micron
      I_g_on_n[0][10] = 6.55e-8;
      I_g_on_n[0][20] = 6.55e-8;
      I_g_on_n[0][30] = 6.55e-8;
      I_g_on_n[0][40] = 6.55e-8;
      I_g_on_n[0][50] = 6.55e-8;
      I_g_on_n[0][60] = 6.55e-8;
      I_g_on_n[0][70] = 6.55e-8;
      I_g_on_n[0][80] = 6.55e-8;
      I_g_on_n[0][90] = 6.55e-8;
      I_g_on_n[0][100] = 6.55e-8;

//      32 DG
//      I_g_on_n[0][0]  = 2.71e-9;//A/micron
//      I_g_on_n[0][10] = 2.71e-9;
//      I_g_on_n[0][20] = 2.71e-9;
//      I_g_on_n[0][30] = 2.71e-9;
//      I_g_on_n[0][40] = 2.71e-9;
//      I_g_on_n[0][50] = 2.71e-9;
//      I_g_on_n[0][60] = 2.71e-9;
//      I_g_on_n[0][70] = 2.71e-9;
//      I_g_on_n[0][80] = 2.71e-9;
//      I_g_on_n[0][90] = 2.71e-9;
//      I_g_on_n[0][100] = 2.71e-9;

      //LSTP device type
      vdd[1] = 1;
      Lphy[1] = 0.020;
      Lelec[1] = 0.0173;
      t_ox[1] = 1.2e-3;
      v_th[1] = 0.513;
      c_ox[1] = 2.29e-14;
      mobility_eff[1] =  347.46 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[1] = 8.64e-2;
      c_g_ideal[1] = 4.58e-16;
      c_fringe[1] = 0.053e-15;
      c_junc[1] = 1e-15;
      I_on_n[1] = 683.6e-6;
      nmos_effective_resistance_multiplier = 1.99;
      n_to_p_eff_curr_drv_ratio[1] = 2.23;
      gmp_to_gmn_multiplier[1] = 0.99;
      Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];
      Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];
      long_channel_leakage_reduction[1] = 1/1.93;
      I_off_n[1][0] = 2.06e-11;
      I_off_n[1][10] = 3.30e-11;
      I_off_n[1][20] = 5.15e-11;
      I_off_n[1][30] = 7.83e-11;
      I_off_n[1][40] = 1.16e-10;
      I_off_n[1][50] = 1.69e-10;
      I_off_n[1][60] = 2.40e-10;
      I_off_n[1][70] = 3.34e-10;
      I_off_n[1][80] = 4.54e-10;
      I_off_n[1][90] = 5.96e-10;
      I_off_n[1][100] = 7.44e-10;

      I_g_on_n[1][0]  = 3.73e-11;//A/micron
      I_g_on_n[1][10] = 3.73e-11;
      I_g_on_n[1][20] = 3.73e-11;
      I_g_on_n[1][30] = 3.73e-11;
      I_g_on_n[1][40] = 3.73e-11;
      I_g_on_n[1][50] = 3.73e-11;
      I_g_on_n[1][60] = 3.73e-11;
      I_g_on_n[1][70] = 3.73e-11;
      I_g_on_n[1][80] = 3.73e-11;
      I_g_on_n[1][90] = 3.73e-11;
      I_g_on_n[1][100] = 3.73e-11;


      //LOP device type
      vdd[2] = 0.6;
      Lphy[2] = 0.016;
      Lelec[2] = 0.01232;
      t_ox[2] = 0.9e-3;
      v_th[2] = 0.24227;
      c_ox[2] = 2.84e-14;
      mobility_eff[2] =  513.52 * (1e-2 * 1e6 * 1e-2 * 1e6);
      Vdsat[2] = 4.64e-2;
      c_g_ideal[2] = 4.54e-16;
      c_fringe[2] = 0.057e-15;
      c_junc[2] = 1e-15;
      I_on_n[2] = 827.8e-6;
      nmos_effective_resistance_multiplier = 1.73;
      n_to_p_eff_curr_drv_ratio[2] = 2.28;
      gmp_to_gmn_multiplier[2] = 1.11;
      Rnchannelon[2] = nmos_effective_resistance_multiplier * vdd[2] / I_on_n[2];
      Rpchannelon[2] = n_to_p_eff_curr_drv_ratio[2] * Rnchannelon[2];
      long_channel_leakage_reduction[2] = 1/1.89;
      I_off_n[2][0] = 5.94e-8;
      I_off_n[2][10] = 7.23e-8;
      I_off_n[2][20] = 8.7e-8;
      I_off_n[2][30] = 1.04e-7;
      I_off_n[2][40] = 1.22e-7;
      I_off_n[2][50] = 1.43e-7;
      I_off_n[2][60] = 1.65e-7;
      I_off_n[2][70] = 1.90e-7;
      I_off_n[2][80] = 2.15e-7;
      I_off_n[2][90] = 2.39e-7;
      I_off_n[2][100] = 2.63e-7;

      I_g_on_n[2][0]  = 2.93e-9;//A/micron
      I_g_on_n[2][10] = 2.93e-9;
      I_g_on_n[2][20] = 2.93e-9;
      I_g_on_n[2][30] = 2.93e-9;
      I_g_on_n[2][40] = 2.93e-9;
      I_g_on_n[2][50] = 2.93e-9;
      I_g_on_n[2][60] = 2.93e-9;
      I_g_on_n[2][70] = 2.93e-9;
      I_g_on_n[2][80] = 2.93e-9;
      I_g_on_n[2][90] = 2.93e-9;
      I_g_on_n[2][100] = 2.93e-9;

      if (ram_cell_tech_type == lp_dram)
      {
        //LP-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.0;
        Lphy[3] = 0.056;
        Lelec[3] = 0.0419;//Assume Lelec is 30% lesser than Lphy for DRAM access and wordline transistors.
        curr_v_th_dram_access_transistor = 0.44129;
        width_dram_access_transistor = 0.056;
        curr_I_on_dram_cell = 36e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 18.9e-12;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = width_dram_access_transistor * Lphy[3] * 10.0;
        curr_asp_ratio_cell_dram = 1.46;
        curr_c_dram_cell = 20e-15;

        //LP-DRAM wordline transistor parameters
        curr_vpp = 1.5;
        t_ox[3] = 2e-3;
        v_th[3] = 0.44467;
        c_ox[3] = 1.48e-14;
        mobility_eff[3] =  408.12 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.174;
        c_g_ideal[3] = 7.45e-16;
        c_fringe[3] = 0.053e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 1055.4e-6;
        nmos_effective_resistance_multiplier = 1.65;
        n_to_p_eff_curr_drv_ratio[3] = 2.05;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0]  = 3.57e-11;
        I_off_n[3][10] = 5.51e-11;
        I_off_n[3][20] = 8.27e-11;
        I_off_n[3][30] = 1.21e-10;
        I_off_n[3][40] = 1.74e-10;
        I_off_n[3][50] = 2.45e-10;
        I_off_n[3][60] = 3.38e-10;
        I_off_n[3][70] = 4.53e-10;
        I_off_n[3][80] = 5.87e-10;
        I_off_n[3][90] = 7.29e-10;
        I_off_n[3][100] = 8.87e-10;
      }
      else if (ram_cell_tech_type == comm_dram)
      {
        //COMM-DRAM cell access transistor technology parameters
        curr_vdd_dram_cell = 1.0;
        Lphy[3] = 0.032;
        Lelec[3] = 0.0205;//Assume Lelec is 30% lesser than Lphy for DRAM access and wordline transistors.
        curr_v_th_dram_access_transistor = 1;
        width_dram_access_transistor = 0.032;
        curr_I_on_dram_cell = 20e-6;
        curr_I_off_dram_cell_worst_case_length_temp = 1e-15;
        curr_Wmemcella_dram = width_dram_access_transistor;
        curr_Wmemcellpmos_dram = 0;
        curr_Wmemcellnmos_dram = 0;
        curr_area_cell_dram = 6*0.032*0.032;
        curr_asp_ratio_cell_dram = 1.5;
        curr_c_dram_cell = 30e-15;

        //COMM-DRAM wordline transistor parameters
        curr_vpp = 2.6;
        t_ox[3] = 4e-3;
        v_th[3] = 1.0;
        c_ox[3] = 7.99e-15;
        mobility_eff[3] =  380.76 * (1e-2 * 1e6 * 1e-2 * 1e6);
        Vdsat[3] = 0.129;
        c_g_ideal[3] = 2.56e-16;
        c_fringe[3] = 0.053e-15;
        c_junc[3] = 1e-15;
        I_on_n[3] = 1024.5e-6;
        nmos_effective_resistance_multiplier = 1.69;
        n_to_p_eff_curr_drv_ratio[3] = 1.95;
        gmp_to_gmn_multiplier[3] = 0.90;
        Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp / I_on_n[3];
        Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];
        long_channel_leakage_reduction[3] = 1;
        I_off_n[3][0]  = 3.63e-14;
        I_off_n[3][10] = 7.18e-14;
        I_off_n[3][20] = 1.36e-13;
        I_off_n[3][30] = 2.49e-13;
        I_off_n[3][40] = 4.41e-13;
        I_off_n[3][50] = 7.55e-13;
        I_off_n[3][60] = 1.26e-12;
        I_off_n[3][70] = 2.03e-12;
        I_off_n[3][80] = 3.19e-12;
        I_off_n[3][90] = 4.87e-12;
        I_off_n[3][100] = 7.16e-12;
      }

      //SRAM cell properties
      curr_Wmemcella_sram    = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
      curr_area_cell_sram    = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_sram = 1.46;
      //CAM cell properties //TODO: data need to be revisited
      curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
      curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
      curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
      curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;
      curr_asp_ratio_cell_cam = 2.92;
      //Empirical undifferetiated core/FU coefficient
      curr_logic_scaling_co_eff = 0.7*0.7*0.7;
      curr_core_tx_density      = 1.25/0.7;
      curr_sckt_co_eff           = 1.1111;
      curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
      curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb
    }

    /* [한국어] ===== 22nm 공정 노드 파라미터 블록 (ITRS 2016, 대략 2016년 양산 세대) =====
     * 22nm는 DG(Double-Gate, FinFET의 초기 형태) SOI 공정 수치를 사용하는 CACTI 지원 최소 노드.
     * HP: Vdd=0.8V, Lphy=9nm, c_junc=0 (SOI/DG 구조에서 접합 커패시턴스 소멸).
     * I_off는 MASTAR 버그로 인해 ITRS 보고 수치에서 직접 추출하고 32nm 대비 1.2/1.5 배율 적용.
     * eDRAM(ram_cell_tech_type==3)은 22nm 미지원 — 초기에 체크하여 exit(0)으로 종료.
     * comm_dram(type==4)만 22nm에서 지원하며, Commodity DRAM 파라미터는 22nm MASTAR 수치 사용.
     * 데이터 출처: ITRS 2013, MASTAR DG (HP는 직접 ITRS 보고 수치).
     */
    if(tech == 22){
        SENSE_AMP_D = .03e-9; // s
	SENSE_AMP_P = 2.16e-15; // J
    	//For 2016, MPU/ASIC stagger-contacted M1 half-pitch is 22 nm (so this is 22 nm
    	//technology i.e. FEATURESIZE = 0.022). Using the DG process numbers for HP.
    	//22 nm HP
    	vdd[0] = 0.8;
    	Lphy[0] = 0.009;//Lphy is the physical gate-length.
    	Lelec[0] = 0.00468;//Lelec is the electrical gate-length.
    	t_ox[0] = 0.55e-3;//micron
    	v_th[0] = 0.1395;//V
    	c_ox[0] = 3.63e-14;//F/micron2
    	mobility_eff[0] = 426.07 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
    	Vdsat[0] = 2.33e-2; //V/micron
    	c_g_ideal[0] = 3.27e-16;//F/micron
    	c_fringe[0] = 0.06e-15;//F/micron
    	c_junc[0] = 0;//F/micron2
    	I_on_n[0] =  2626.4e-6;//A/micron
    	//I_on_p[0] = I_on_n[0] / 2;//A/micron //This value for I_on_p is not really used.
        nmos_effective_resistance_multiplier = 1.45;
        n_to_p_eff_curr_drv_ratio[0] = 2; //Wpmos/Wnmos = 2 in 2007 MASTAR. Look in
    	//"Dynamic" tab of Device workspace.
        gmp_to_gmn_multiplier[0] = 1.38; //Just using the 32nm SOI value.
        Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];//ohm-micron
        Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];//ohm-micron
        long_channel_leakage_reduction[0] = 1/3.274;
        I_off_n[0][0] = 1.52e-7/1.5*1.2;//From 22nm, leakage current are directly from ITRS report rather than MASTAR, since MASTAR has serious bugs there.
        I_off_n[0][10] = 1.55e-7/1.5*1.2;
        I_off_n[0][20] = 1.59e-7/1.5*1.2;
        I_off_n[0][30] = 1.68e-7/1.5*1.2;
        I_off_n[0][40] = 1.90e-7/1.5*1.2;
        I_off_n[0][50] = 2.69e-7/1.5*1.2;
        I_off_n[0][60] = 5.32e-7/1.5*1.2;
        I_off_n[0][70] = 1.02e-6/1.5*1.2;
        I_off_n[0][80] = 1.62e-6/1.5*1.2;
        I_off_n[0][90] = 2.73e-6/1.5*1.2;
        I_off_n[0][100] = 6.1e-6/1.5*1.2;
        //for 22nm DG HP
        I_g_on_n[0][0]  = 1.81e-9;//A/micron
        I_g_on_n[0][10] = 1.81e-9;
        I_g_on_n[0][20] = 1.81e-9;
        I_g_on_n[0][30] = 1.81e-9;
        I_g_on_n[0][40] = 1.81e-9;
        I_g_on_n[0][50] = 1.81e-9;
        I_g_on_n[0][60] = 1.81e-9;
        I_g_on_n[0][70] = 1.81e-9;
        I_g_on_n[0][80] = 1.81e-9;
        I_g_on_n[0][90] = 1.81e-9;
        I_g_on_n[0][100] = 1.81e-9;

    	//22 nm LSTP DG
    	vdd[1] = 0.8;
    	Lphy[1] = 0.014;
    	Lelec[1] = 0.008;//Lelec is the electrical gate-length.
    	t_ox[1] = 1.1e-3;//micron
    	v_th[1] = 0.40126;//V
    	c_ox[1] = 2.30e-14;//F/micron2
    	mobility_eff[1] =  738.09 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
    	Vdsat[1] = 6.64e-2; //V/micron
    	c_g_ideal[1] = 3.22e-16;//F/micron
    	c_fringe[1] = 0.08e-15;
    	c_junc[1] = 0;//F/micron2
    	I_on_n[1] = 727.6e-6;//A/micron
    	nmos_effective_resistance_multiplier = 1.99;
    	n_to_p_eff_curr_drv_ratio[1] = 2;
    	gmp_to_gmn_multiplier[1] = 0.99;
    	Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];//ohm-micron
    	Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];//ohm-micron
    	long_channel_leakage_reduction[1] = 1/1.89;
    	I_off_n[1][0] = 2.43e-11;
    	I_off_n[1][10] = 4.85e-11;
    	I_off_n[1][20] = 9.68e-11;
    	I_off_n[1][30] = 1.94e-10;
    	I_off_n[1][40] = 3.87e-10;
    	I_off_n[1][50] = 7.73e-10;
    	I_off_n[1][60] = 3.55e-10;
    	I_off_n[1][70] = 3.09e-9;
    	I_off_n[1][80] = 6.19e-9;
    	I_off_n[1][90] = 1.24e-8;
    	I_off_n[1][100]= 2.48e-8;

    	I_g_on_n[1][0]  = 4.51e-10;//A/micron
    	I_g_on_n[1][10] = 4.51e-10;
    	I_g_on_n[1][20] = 4.51e-10;
    	I_g_on_n[1][30] = 4.51e-10;
    	I_g_on_n[1][40] = 4.51e-10;
    	I_g_on_n[1][50] = 4.51e-10;
    	I_g_on_n[1][60] = 4.51e-10;
    	I_g_on_n[1][70] = 4.51e-10;
    	I_g_on_n[1][80] = 4.51e-10;
    	I_g_on_n[1][90] = 4.51e-10;
    	I_g_on_n[1][100] = 4.51e-10;

    	//22 nm LOP
    	vdd[2] = 0.6;
    	Lphy[2] = 0.011;
    	Lelec[2] = 0.00604;//Lelec is the electrical gate-length.
    	t_ox[2] = 0.8e-3;//micron
    	v_th[2] = 0.2315;//V
    	c_ox[2] = 2.87e-14;//F/micron2
    	mobility_eff[2] =  698.37 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
    	Vdsat[2] = 1.81e-2; //V/micron
    	c_g_ideal[2] = 3.16e-16;//F/micron
    	c_fringe[2] = 0.08e-15;
    	c_junc[2] = 0;//F/micron2 This is Cj0 not Cjunc in MASTAR results->Dynamic Tab
    	I_on_n[2] = 916.1e-6;//A/micron
    	nmos_effective_resistance_multiplier = 1.73;
    	n_to_p_eff_curr_drv_ratio[2] = 2;
    	gmp_to_gmn_multiplier[2] = 1.11;
    	Rnchannelon[2] = nmos_effective_resistance_multiplier * vdd[2] / I_on_n[2];//ohm-micron
    	Rpchannelon[2] = n_to_p_eff_curr_drv_ratio[2] * Rnchannelon[2];//ohm-micron
    	long_channel_leakage_reduction[2] = 1/2.38;

    	I_off_n[2][0] = 1.31e-8;
    	I_off_n[2][10] = 2.60e-8;
    	I_off_n[2][20] = 5.14e-8;
    	I_off_n[2][30] = 1.02e-7;
    	I_off_n[2][40] = 2.02e-7;
    	I_off_n[2][50] = 3.99e-7;
    	I_off_n[2][60] = 7.91e-7;
    	I_off_n[2][70] = 1.09e-6;
    	I_off_n[2][80] = 2.09e-6;
    	I_off_n[2][90] = 4.04e-6;
    	I_off_n[2][100]= 4.48e-6;

    	I_g_on_n[2][0]  = 2.74e-9;//A/micron
    	I_g_on_n[2][10] = 2.74e-9;
    	I_g_on_n[2][20] = 2.74e-9;
    	I_g_on_n[2][30] = 2.74e-9;
    	I_g_on_n[2][40] = 2.74e-9;
    	I_g_on_n[2][50] = 2.74e-9;
    	I_g_on_n[2][60] = 2.74e-9;
    	I_g_on_n[2][70] = 2.74e-9;
    	I_g_on_n[2][80] = 2.74e-9;
    	I_g_on_n[2][90] = 2.74e-9;
    	I_g_on_n[2][100] = 2.74e-9;



        if (ram_cell_tech_type == 3)
              {}
        else if (ram_cell_tech_type == 4)
        {
    	//22 nm commodity DRAM cell access transistor technology parameters.
    		//parameters
        	curr_vdd_dram_cell = 0.9;//0.45;//This value has reduced greatly in 2007 ITRS for all technology nodes. In
    		//2005 ITRS, the value was about twice the value in 2007 ITRS
    		Lphy[3] = 0.022;//micron
    		Lelec[3] = 0.0181;//micron.
    		curr_v_th_dram_access_transistor = 1;//V
    		width_dram_access_transistor = 0.022;//micron
    		curr_I_on_dram_cell = 20e-6; //This is a typical value that I have always
    		//kept constant. In reality this could perhaps be lower
    		curr_I_off_dram_cell_worst_case_length_temp = 1e-15;//A
    		curr_Wmemcella_dram = width_dram_access_transistor;
    		curr_Wmemcellpmos_dram = 0;
    		curr_Wmemcellnmos_dram = 0;
    		curr_area_cell_dram = 6*0.022*0.022;//micron2.
    		curr_asp_ratio_cell_dram = 0.667;
    		curr_c_dram_cell = 30e-15;//This is a typical value that I have alwaus
    		//kept constant.

    	//22 nm commodity DRAM wordline transistor parameters obtained using MASTAR.
    		curr_vpp = 2.3;//vpp. V
    		t_ox[3] = 3.5e-3;//micron
    		v_th[3] = 1.0;//V
    		c_ox[3] = 9.06e-15;//F/micron2
    		mobility_eff[3] =  367.29 * (1e-2 * 1e6 * 1e-2 * 1e6);//micron2 / Vs
    		Vdsat[3] = 0.0972; //V/micron
    		c_g_ideal[3] = 1.99e-16;//F/micron
    		c_fringe[3] = 0.053e-15;//F/micron
    		c_junc[3] = 1e-15;//F/micron2
    		I_on_n[3] = 910.5e-6;//A/micron
    		nmos_effective_resistance_multiplier = 1.69;//Using the value from 32nm.
    		//
    		n_to_p_eff_curr_drv_ratio[3] = 1.95;//Using the value from 32nm
    		gmp_to_gmn_multiplier[3] = 0.90;
    		Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp  / I_on_n[3];//ohm-micron
    		Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];//ohm-micron
    		long_channel_leakage_reduction[3] = 1;
    		I_off_n[3][0] = 1.1e-13; //A/micron
    		I_off_n[3][10] = 2.11e-13;
    		I_off_n[3][20] = 3.88e-13;
    		I_off_n[3][30] = 6.9e-13;
    		I_off_n[3][40] = 1.19e-12;
    		I_off_n[3][50] = 1.98e-12;
    		I_off_n[3][60] = 3.22e-12;
    		I_off_n[3][70] = 5.09e-12;
    		I_off_n[3][80] = 7.85e-12;
    		I_off_n[3][90] = 1.18e-11;
    		I_off_n[3][100] = 1.72e-11;

    	}
        else
        {
      	  //some error handler
        }

        //SRAM cell properties
        curr_Wmemcella_sram    = 1.31 * g_ip->F_sz_um;
        curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
        curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
        curr_area_cell_sram    = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
        curr_asp_ratio_cell_sram = 1.46;
        //CAM cell properties //TODO: data need to be revisited
        curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
        curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
        curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
        curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;
        curr_asp_ratio_cell_cam = 2.92;
        //Empirical undifferetiated core/FU coefficient
        curr_logic_scaling_co_eff = 0.7*0.7*0.7*0.7;
        curr_core_tx_density      = 1.25/0.7/0.7;
        curr_sckt_co_eff           = 1.1296;
        curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
        curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb
    	}

    /* [한국어] ===== 16nm 공정 노드 파라미터 블록 (미완성 노드, 실질적으로 도달 불가) =====
     * 16nm는 22nm와 동일한 DG(Double-Gate) SOI 공정을 가정하며, ITRS 2016 예측 데이터를 사용한다.
     * 그러나 tech_lo/tech_hi 결정 if-else 체인에서 16nm 경우는 주석 처리되어 있어 이 블록은
     * 현재 코드 경로에서 실제로 도달하지 않는다 (dead code). 향후 확장을 위한 플레이스홀더.
     * I_off 값은 22nm 값에 1.07 배율을 추가로 적용하여 추정한다.
     */
    if(tech == 16){
    	//For 2019, MPU/ASIC stagger-contacted M1 half-pitch is 16 nm (so this is 16 nm
    	//technology i.e. FEATURESIZE = 0.016). Using the DG process numbers for HP.
    	//16 nm HP
    	vdd[0] = 0.7;
    	Lphy[0] = 0.006;//Lphy is the physical gate-length.
    	Lelec[0] = 0.00315;//Lelec is the electrical gate-length.
    	t_ox[0] = 0.5e-3;//micron
    	v_th[0] = 0.1489;//V
    	c_ox[0] = 3.83e-14;//F/micron2 Cox_elec in MASTAR
    	mobility_eff[0] = 476.15 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
    	Vdsat[0] = 1.42e-2; //V/micron calculated in spreadsheet
    	c_g_ideal[0] = 2.30e-16;//F/micron
    	c_fringe[0] = 0.06e-15;//F/micron MASTAR inputdynamic/3
    	c_junc[0] = 0;//F/micron2 MASTAR result dynamic
    	I_on_n[0] =  2768.4e-6;//A/micron
        nmos_effective_resistance_multiplier = 1.48;//nmos_effective_resistance_multiplier  is the ratio of Ieff to Idsat where Ieff is the effective NMOS current and Idsat is the saturation current.
        n_to_p_eff_curr_drv_ratio[0] = 2; //Wpmos/Wnmos = 2 in 2007 MASTAR. Look in
    	//"Dynamic" tab of Device workspace.
        gmp_to_gmn_multiplier[0] = 1.38; //Just using the 32nm SOI value.
        Rnchannelon[0] = nmos_effective_resistance_multiplier * vdd[0] / I_on_n[0];//ohm-micron
        Rpchannelon[0] = n_to_p_eff_curr_drv_ratio[0] * Rnchannelon[0];//ohm-micron
        long_channel_leakage_reduction[0] = 1/2.655;
        I_off_n[0][0] = 1.52e-7/1.5*1.2*1.07;
        I_off_n[0][10] = 1.55e-7/1.5*1.2*1.07;
        I_off_n[0][20] = 1.59e-7/1.5*1.2*1.07;
        I_off_n[0][30] = 1.68e-7/1.5*1.2*1.07;
        I_off_n[0][40] = 1.90e-7/1.5*1.2*1.07;
        I_off_n[0][50] = 2.69e-7/1.5*1.2*1.07;
        I_off_n[0][60] = 5.32e-7/1.5*1.2*1.07;
        I_off_n[0][70] = 1.02e-6/1.5*1.2*1.07;
        I_off_n[0][80] = 1.62e-6/1.5*1.2*1.07;
        I_off_n[0][90] = 2.73e-6/1.5*1.2*1.07;
        I_off_n[0][100] = 6.1e-6/1.5*1.2*1.07;
        //for 16nm DG HP
        I_g_on_n[0][0]  = 1.07e-9;//A/micron
        I_g_on_n[0][10] = 1.07e-9;
        I_g_on_n[0][20] = 1.07e-9;
        I_g_on_n[0][30] = 1.07e-9;
        I_g_on_n[0][40] = 1.07e-9;
        I_g_on_n[0][50] = 1.07e-9;
        I_g_on_n[0][60] = 1.07e-9;
        I_g_on_n[0][70] = 1.07e-9;
        I_g_on_n[0][80] = 1.07e-9;
        I_g_on_n[0][90] = 1.07e-9;
        I_g_on_n[0][100] = 1.07e-9;

//    	//16 nm LSTP DG
//    	vdd[1] = 0.8;
//    	Lphy[1] = 0.014;
//    	Lelec[1] = 0.008;//Lelec is the electrical gate-length.
//    	t_ox[1] = 1.1e-3;//micron
//    	v_th[1] = 0.40126;//V
//    	c_ox[1] = 2.30e-14;//F/micron2
//    	mobility_eff[1] =  738.09 * (1e-2 * 1e6 * 1e-2 * 1e6); //micron2 / Vs
//    	Vdsat[1] = 6.64e-2; //V/micron
//    	c_g_ideal[1] = 3.22e-16;//F/micron
//    	c_fringe[1] = 0.008e-15;
//    	c_junc[1] = 0;//F/micron2
//    	I_on_n[1] = 727.6e-6;//A/micron
//    	I_on_p[1] = I_on_n[1] / 2;
//    	nmos_effective_resistance_multiplier = 1.99;
//    	n_to_p_eff_curr_drv_ratio[1] = 2;
//    	gmp_to_gmn_multiplier[1] = 0.99;
//    	Rnchannelon[1] = nmos_effective_resistance_multiplier * vdd[1] / I_on_n[1];//ohm-micron
//    	Rpchannelon[1] = n_to_p_eff_curr_drv_ratio[1] * Rnchannelon[1];//ohm-micron
//    	I_off_n[1][0] = 2.43e-11;
//    	I_off_n[1][10] = 4.85e-11;
//    	I_off_n[1][20] = 9.68e-11;
//    	I_off_n[1][30] = 1.94e-10;
//    	I_off_n[1][40] = 3.87e-10;
//    	I_off_n[1][50] = 7.73e-10;
//    	I_off_n[1][60] = 3.55e-10;
//    	I_off_n[1][70] = 3.09e-9;
//    	I_off_n[1][80] = 6.19e-9;
//    	I_off_n[1][90] = 1.24e-8;
//    	I_off_n[1][100]= 2.48e-8;
//
//    	//    for 22nm LSTP HP
//    	I_g_on_n[1][0]  = 4.51e-10;//A/micron
//    	I_g_on_n[1][10] = 4.51e-10;
//    	I_g_on_n[1][20] = 4.51e-10;
//    	I_g_on_n[1][30] = 4.51e-10;
//    	I_g_on_n[1][40] = 4.51e-10;
//    	I_g_on_n[1][50] = 4.51e-10;
//    	I_g_on_n[1][60] = 4.51e-10;
//    	I_g_on_n[1][70] = 4.51e-10;
//    	I_g_on_n[1][80] = 4.51e-10;
//    	I_g_on_n[1][90] = 4.51e-10;
//    	I_g_on_n[1][100] = 4.51e-10;


        if (ram_cell_tech_type == 3)
              {}
        else if (ram_cell_tech_type == 4)
        {
    	//22 nm commodity DRAM cell access transistor technology parameters.
    		//parameters
        	curr_vdd_dram_cell = 0.9;//0.45;//This value has reduced greatly in 2007 ITRS for all technology nodes. In
    		//2005 ITRS, the value was about twice the value in 2007 ITRS
    		Lphy[3] = 0.022;//micron
    		Lelec[3] = 0.0181;//micron.
    		curr_v_th_dram_access_transistor = 1;//V
    		width_dram_access_transistor = 0.022;//micron
    		curr_I_on_dram_cell = 20e-6; //This is a typical value that I have always
    		//kept constant. In reality this could perhaps be lower
    		curr_I_off_dram_cell_worst_case_length_temp = 1e-15;//A
    		curr_Wmemcella_dram = width_dram_access_transistor;
    		curr_Wmemcellpmos_dram = 0;
    		curr_Wmemcellnmos_dram = 0;
    		curr_area_cell_dram = 6*0.022*0.022;//micron2.
    		curr_asp_ratio_cell_dram = 0.667;
    		curr_c_dram_cell = 30e-15;//This is a typical value that I have alwaus
    		//kept constant.

    	//22 nm commodity DRAM wordline transistor parameters obtained using MASTAR.
    		curr_vpp = 2.3;//vpp. V
    		t_ox[3] = 3.5e-3;//micron
    		v_th[3] = 1.0;//V
    		c_ox[3] = 9.06e-15;//F/micron2
    		mobility_eff[3] =  367.29 * (1e-2 * 1e6 * 1e-2 * 1e6);//micron2 / Vs
    		Vdsat[3] = 0.0972; //V/micron
    		c_g_ideal[3] = 1.99e-16;//F/micron
    		c_fringe[3] = 0.053e-15;//F/micron
    		c_junc[3] = 1e-15;//F/micron2
    		I_on_n[3] = 910.5e-6;//A/micron
    		nmos_effective_resistance_multiplier = 1.69;//Using the value from 32nm.
    		//
    		n_to_p_eff_curr_drv_ratio[3] = 1.95;//Using the value from 32nm
    		gmp_to_gmn_multiplier[3] = 0.90;
    		Rnchannelon[3] = nmos_effective_resistance_multiplier * curr_vpp  / I_on_n[3];//ohm-micron
    		Rpchannelon[3] = n_to_p_eff_curr_drv_ratio[3] * Rnchannelon[3];//ohm-micron
    		long_channel_leakage_reduction[3] = 1;
    		I_off_n[3][0] = 1.1e-13; //A/micron
    		I_off_n[3][10] = 2.11e-13;
    		I_off_n[3][20] = 3.88e-13;
    		I_off_n[3][30] = 6.9e-13;
    		I_off_n[3][40] = 1.19e-12;
    		I_off_n[3][50] = 1.98e-12;
    		I_off_n[3][60] = 3.22e-12;
    		I_off_n[3][70] = 5.09e-12;
    		I_off_n[3][80] = 7.85e-12;
    		I_off_n[3][90] = 1.18e-11;
    		I_off_n[3][100] = 1.72e-11;

    	}
        else
        {
      	  //some error handler
        }

        //SRAM cell properties
        curr_Wmemcella_sram    = 1.31 * g_ip->F_sz_um;
        curr_Wmemcellpmos_sram = 1.23 * g_ip->F_sz_um;
        curr_Wmemcellnmos_sram = 2.08 * g_ip->F_sz_um;
        curr_area_cell_sram    = 146 * g_ip->F_sz_um * g_ip->F_sz_um;
        curr_asp_ratio_cell_sram = 1.46;
        //CAM cell properties //TODO: data need to be revisited
        curr_Wmemcella_cam = 1.31 * g_ip->F_sz_um;
        curr_Wmemcellpmos_cam = 1.23 * g_ip->F_sz_um;
        curr_Wmemcellnmos_cam = 2.08 * g_ip->F_sz_um;
        curr_area_cell_cam = 292 * g_ip->F_sz_um * g_ip->F_sz_um;
        curr_asp_ratio_cell_cam = 2.92;
        //Empirical undifferetiated core/FU coefficient
        curr_logic_scaling_co_eff = 0.7*0.7*0.7*0.7*0.7;
        curr_core_tx_density      = 1.25/0.7/0.7/0.7;
        curr_sckt_co_eff           = 1.1296;
        curr_chip_layout_overhead  = 1.2;//die measurement results based on Niagara 1 and 2
        curr_macro_layout_overhead = 1.1;//EDA placement and routing tool rule of thumb
    	}


    /* [한국어] ===== g_tp 누적 블록: 선형 보간 결과를 g_tp에 += curr_alpha × 값으로 축적 =====
     * iter=0에서 tech_lo 기여분, iter=1에서 tech_hi 기여분을 누적한다.
     * 최종 g_tp 값 = alpha_lo × (tech_lo 파라미터) + alpha_hi × (tech_hi 파라미터).
     * peri_global: 주변 회로(워드라인 드라이버, SA, 디코더 등) 트랜지스터에 사용.
     * sram_cell: SRAM 셀 트랜지스터(6T SRAM)에 사용.
     * dram_acc / dram_wl: DRAM 접근 트랜지스터 및 워드라인 트랜지스터에 사용.
     * cam_cell: CAM(Content-Addressable Memory) 셀 트랜지스터에 사용.
     * 인덱스 [g_ip->temp - 300]: 시뮬레이션 온도에서의 누설 전류 룩업 (300K=인덱스 0).
     */

    /* [한국어] 주변 회로(Peripheral Global) 트랜지스터 파라미터 누적.
     * peri_global_tech_type: 0=HP, 1=LSTP, 2=LOP. 캐시 주변 회로에 사용되는 플레이버. */
    g_tp.peri_global.Vdd       += curr_alpha * vdd[peri_global_tech_type];
    g_tp.peri_global.t_ox      += curr_alpha * t_ox[peri_global_tech_type];
    g_tp.peri_global.Vth       += curr_alpha * v_th[peri_global_tech_type];
    g_tp.peri_global.C_ox      += curr_alpha * c_ox[peri_global_tech_type];
    g_tp.peri_global.C_g_ideal += curr_alpha * c_g_ideal[peri_global_tech_type];
    g_tp.peri_global.C_fringe  += curr_alpha * c_fringe[peri_global_tech_type];
    g_tp.peri_global.C_junc    += curr_alpha * c_junc[peri_global_tech_type];
    g_tp.peri_global.C_junc_sidewall = 0.25e-15;  // F/micron  /* [한국어] 측벽 접합 커패시턴스: 공정 독립 상수값 */
    g_tp.peri_global.l_phy     += curr_alpha * Lphy[peri_global_tech_type];
    g_tp.peri_global.l_elec    += curr_alpha * Lelec[peri_global_tech_type];
    g_tp.peri_global.I_on_n    += curr_alpha * I_on_n[peri_global_tech_type];
    g_tp.peri_global.R_nch_on  += curr_alpha * Rnchannelon[peri_global_tech_type];
    g_tp.peri_global.R_pch_on  += curr_alpha * Rpchannelon[peri_global_tech_type];
    g_tp.peri_global.n_to_p_eff_curr_drv_ratio
      += curr_alpha * n_to_p_eff_curr_drv_ratio[peri_global_tech_type];
    g_tp.peri_global.long_channel_leakage_reduction
      += curr_alpha * long_channel_leakage_reduction[peri_global_tech_type];
    /* [한국어] 온도 인덱스 = g_ip->temp - 300: 300K 기준 오프셋. 예: 350K → 인덱스 50.
     * I_off_p는 별도 PMOS 수치가 없어 I_off_n 값을 재사용한다. */
    g_tp.peri_global.I_off_n   += curr_alpha * I_off_n[peri_global_tech_type][g_ip->temp - 300];
    g_tp.peri_global.I_off_p   += curr_alpha * I_off_n[peri_global_tech_type][g_ip->temp - 300];
    g_tp.peri_global.I_g_on_n   += curr_alpha * I_g_on_n[peri_global_tech_type][g_ip->temp - 300];
    g_tp.peri_global.I_g_on_p   += curr_alpha * I_g_on_n[peri_global_tech_type][g_ip->temp - 300];
    gmp_to_gmn_multiplier_periph_global += curr_alpha * gmp_to_gmn_multiplier[peri_global_tech_type];

    /* [한국어] SRAM 셀 트랜지스터 파라미터 누적.
     * ram_cell_tech_type: 0=ITRS-HP, 1=ITRS-LSTP, 2=ITRS-LOP. 6T SRAM 셀의 pull-down/pass 트랜지스터 플레이버. */
    g_tp.sram_cell.Vdd       += curr_alpha * vdd[ram_cell_tech_type];
    g_tp.sram_cell.l_phy     += curr_alpha * Lphy[ram_cell_tech_type];
    g_tp.sram_cell.l_elec    += curr_alpha * Lelec[ram_cell_tech_type];
    g_tp.sram_cell.t_ox      += curr_alpha * t_ox[ram_cell_tech_type];
    g_tp.sram_cell.Vth       += curr_alpha * v_th[ram_cell_tech_type];
    g_tp.sram_cell.C_g_ideal += curr_alpha * c_g_ideal[ram_cell_tech_type];
    g_tp.sram_cell.C_fringe  += curr_alpha * c_fringe[ram_cell_tech_type];
    g_tp.sram_cell.C_junc    += curr_alpha * c_junc[ram_cell_tech_type];
    g_tp.sram_cell.C_junc_sidewall = 0.25e-15;  // F/micron
    g_tp.sram_cell.I_on_n    += curr_alpha * I_on_n[ram_cell_tech_type];
    g_tp.sram_cell.R_nch_on  += curr_alpha * Rnchannelon[ram_cell_tech_type];
    g_tp.sram_cell.R_pch_on  += curr_alpha * Rpchannelon[ram_cell_tech_type];
    g_tp.sram_cell.n_to_p_eff_curr_drv_ratio += curr_alpha * n_to_p_eff_curr_drv_ratio[ram_cell_tech_type];
    g_tp.sram_cell.long_channel_leakage_reduction += curr_alpha * long_channel_leakage_reduction[ram_cell_tech_type];
    g_tp.sram_cell.I_off_n   += curr_alpha * I_off_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.sram_cell.I_off_p   += curr_alpha * I_off_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.sram_cell.I_g_on_n   += curr_alpha * I_g_on_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.sram_cell.I_g_on_p   += curr_alpha * I_g_on_n[ram_cell_tech_type][g_ip->temp - 300];

    /* [한국어] DRAM 셀 및 접근/워드라인 트랜지스터 파라미터 누적.
     * dram_cell_tech_flavor(=3): DRAM 접근 트랜지스터 전용 인덱스. lp_dram/comm_dram 블록에서 채워진다.
     * dram_acc: DRAM 셀 접근 트랜지스터 (행 어드레스 선택 시 비트라인 충전).
     * dram_wl: DRAM 워드라인 드라이버 트랜지스터 (Vpp로 구동, 높은 Vth 필요).
     * g_tp.vpp: DRAM 워드라인 부스트 전압 — 접근 트랜지스터를 완전 개통하기 위해 Vdd보다 높게 설정 */
    g_tp.dram_cell_Vdd      += curr_alpha * curr_vdd_dram_cell;
    g_tp.dram_acc.Vth       += curr_alpha * curr_v_th_dram_access_transistor;
    g_tp.dram_acc.l_phy     += curr_alpha * Lphy[dram_cell_tech_flavor];
    g_tp.dram_acc.l_elec    += curr_alpha * Lelec[dram_cell_tech_flavor];
    g_tp.dram_acc.C_g_ideal += curr_alpha * c_g_ideal[dram_cell_tech_flavor];
    g_tp.dram_acc.C_fringe  += curr_alpha * c_fringe[dram_cell_tech_flavor];
    g_tp.dram_acc.C_junc    += curr_alpha * c_junc[dram_cell_tech_flavor];
    g_tp.dram_acc.C_junc_sidewall = 0.25e-15;  // F/micron
    g_tp.dram_cell_I_on     += curr_alpha * curr_I_on_dram_cell;
    g_tp.dram_cell_I_off_worst_case_len_temp += curr_alpha * curr_I_off_dram_cell_worst_case_length_temp;
    g_tp.dram_acc.I_on_n    += curr_alpha * I_on_n[dram_cell_tech_flavor];
    g_tp.dram_cell_C        += curr_alpha * curr_c_dram_cell;
    g_tp.vpp                += curr_alpha * curr_vpp;
    g_tp.dram_wl.l_phy      += curr_alpha * Lphy[dram_cell_tech_flavor];
    g_tp.dram_wl.l_elec     += curr_alpha * Lelec[dram_cell_tech_flavor];
    g_tp.dram_wl.C_g_ideal  += curr_alpha * c_g_ideal[dram_cell_tech_flavor];
    g_tp.dram_wl.C_fringe   += curr_alpha * c_fringe[dram_cell_tech_flavor];
    g_tp.dram_wl.C_junc     += curr_alpha * c_junc[dram_cell_tech_flavor];
    g_tp.dram_wl.C_junc_sidewall = 0.25e-15;  // F/micron
    g_tp.dram_wl.I_on_n     += curr_alpha * I_on_n[dram_cell_tech_flavor];
    g_tp.dram_wl.R_nch_on   += curr_alpha * Rnchannelon[dram_cell_tech_flavor];
    g_tp.dram_wl.R_pch_on   += curr_alpha * Rpchannelon[dram_cell_tech_flavor];
    g_tp.dram_wl.n_to_p_eff_curr_drv_ratio += curr_alpha * n_to_p_eff_curr_drv_ratio[dram_cell_tech_flavor];
    g_tp.dram_wl.long_channel_leakage_reduction += curr_alpha * long_channel_leakage_reduction[dram_cell_tech_flavor];
    g_tp.dram_wl.I_off_n    += curr_alpha * I_off_n[dram_cell_tech_flavor][g_ip->temp - 300];
    g_tp.dram_wl.I_off_p    += curr_alpha * I_off_n[dram_cell_tech_flavor][g_ip->temp - 300];

    g_tp.cam_cell.Vdd       += curr_alpha * vdd[ram_cell_tech_type];
    g_tp.cam_cell.l_phy     += curr_alpha * Lphy[ram_cell_tech_type];
    g_tp.cam_cell.l_elec    += curr_alpha * Lelec[ram_cell_tech_type];
    g_tp.cam_cell.t_ox      += curr_alpha * t_ox[ram_cell_tech_type];
    g_tp.cam_cell.Vth       += curr_alpha * v_th[ram_cell_tech_type];
    g_tp.cam_cell.C_g_ideal += curr_alpha * c_g_ideal[ram_cell_tech_type];
    g_tp.cam_cell.C_fringe  += curr_alpha * c_fringe[ram_cell_tech_type];
    g_tp.cam_cell.C_junc    += curr_alpha * c_junc[ram_cell_tech_type];
    g_tp.cam_cell.C_junc_sidewall = 0.25e-15;  // F/micron
    g_tp.cam_cell.I_on_n    += curr_alpha * I_on_n[ram_cell_tech_type];
    g_tp.cam_cell.R_nch_on  += curr_alpha * Rnchannelon[ram_cell_tech_type];
    g_tp.cam_cell.R_pch_on  += curr_alpha * Rpchannelon[ram_cell_tech_type];
    g_tp.cam_cell.n_to_p_eff_curr_drv_ratio += curr_alpha * n_to_p_eff_curr_drv_ratio[ram_cell_tech_type];
    g_tp.cam_cell.long_channel_leakage_reduction += curr_alpha * long_channel_leakage_reduction[ram_cell_tech_type];
    g_tp.cam_cell.I_off_n   += curr_alpha * I_off_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.cam_cell.I_off_p   += curr_alpha * I_off_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.cam_cell.I_g_on_n   += curr_alpha * I_g_on_n[ram_cell_tech_type][g_ip->temp - 300];
    g_tp.cam_cell.I_g_on_p   += curr_alpha * I_g_on_n[ram_cell_tech_type][g_ip->temp - 300];

    g_tp.dram.cell_a_w    += curr_alpha * curr_Wmemcella_dram;
    g_tp.dram.cell_pmos_w += curr_alpha * curr_Wmemcellpmos_dram;
    g_tp.dram.cell_nmos_w += curr_alpha * curr_Wmemcellnmos_dram;
    area_cell_dram        += curr_alpha * curr_area_cell_dram;
    asp_ratio_cell_dram   += curr_alpha * curr_asp_ratio_cell_dram;

    g_tp.sram.cell_a_w    += curr_alpha * curr_Wmemcella_sram;
    g_tp.sram.cell_pmos_w += curr_alpha * curr_Wmemcellpmos_sram;
    g_tp.sram.cell_nmos_w += curr_alpha * curr_Wmemcellnmos_sram;
    area_cell_sram += curr_alpha * curr_area_cell_sram;
    asp_ratio_cell_sram += curr_alpha * curr_asp_ratio_cell_sram;

    g_tp.cam.cell_a_w    += curr_alpha * curr_Wmemcella_cam;//sheng
    g_tp.cam.cell_pmos_w += curr_alpha * curr_Wmemcellpmos_cam;
    g_tp.cam.cell_nmos_w += curr_alpha * curr_Wmemcellnmos_cam;
    area_cell_cam += curr_alpha * curr_area_cell_cam;
    asp_ratio_cell_cam += curr_alpha * curr_asp_ratio_cell_cam;

    //Sense amplifier latch Gm calculation
    mobility_eff_periph_global += curr_alpha * mobility_eff[peri_global_tech_type];
    Vdsat_periph_global += curr_alpha * Vdsat[peri_global_tech_type];

    //Empirical undifferetiated core/FU coefficient
    g_tp.scaling_factor.logic_scaling_co_eff += curr_alpha * curr_logic_scaling_co_eff;
    g_tp.scaling_factor.core_tx_density += curr_alpha * curr_core_tx_density;
    g_tp.chip_layout_overhead  += curr_alpha * curr_chip_layout_overhead;
    g_tp.macro_layout_overhead += curr_alpha * curr_macro_layout_overhead;
    g_tp.sckt_co_eff           += curr_alpha * curr_sckt_co_eff;
  }


  /* [한국어] ===== 트랜지스터 파라미터 보간 완료 후: 공정-연속 파생값 계산 =====
   * 아래 값들은 공정 노드의 이산 룩업테이블이 아닌 F_sz_um(피처 사이즈)에 비례하는
   * 연속 함수로 결정된다. 따라서 선형 보간이 불필요하며 보간 루프 밖에서 한 번만 계산한다.
   * 비교기(comparator) 인버터 체인 트랜지스터 폭, sense amplifier 트랜지스터 폭,
   * 최소/최대 NMOS 폭, 디코더 제약 등을 F_sz_um 배수로 설정한다.
   */

  //Currently we are not modeling the resistance/capacitance of poly anywhere.
  //Continuous function (or date have been processed) does not need linear interpolation
  /* [한국어] 비교기 인버터 체인(3단 테이퍼드 인버터) PMOS/NMOS 폭 (μm). F_sz_um에 비례 스케일 */
  g_tp.w_comp_inv_p1 = 12.5 * g_ip->F_sz_um;//this was 10 micron for the 0.8 micron process
  g_tp.w_comp_inv_n1 =  7.5 * g_ip->F_sz_um;//this was  6 micron for the 0.8 micron process
  g_tp.w_comp_inv_p2 =   25 * g_ip->F_sz_um;//this was 20 micron for the 0.8 micron process
  g_tp.w_comp_inv_n2 =   15 * g_ip->F_sz_um;//this was 12 micron for the 0.8 micron process
  g_tp.w_comp_inv_p3 =   50 * g_ip->F_sz_um;//this was 40 micron for the 0.8 micron process
  g_tp.w_comp_inv_n3 =   30 * g_ip->F_sz_um;//this was 24 micron for the 0.8 micron process
  g_tp.w_eval_inv_p  =  100 * g_ip->F_sz_um;//this was 80 micron for the 0.8 micron process
  g_tp.w_eval_inv_n  =   50 * g_ip->F_sz_um;//this was 40 micron for the 0.8 micron process
  g_tp.w_comp_n     = 12.5 * g_ip->F_sz_um;//this was 10 micron for the 0.8 micron process
  g_tp.w_comp_p     = 37.5 * g_ip->F_sz_um;//this was 30 micron for the 0.8 micron process

  g_tp.MIN_GAP_BET_P_AND_N_DIFFS = 5 * g_ip->F_sz_um;
  g_tp.MIN_GAP_BET_SAME_TYPE_DIFFS = 1.5 * g_ip->F_sz_um;
  g_tp.HPOWERRAIL = 2 * g_ip->F_sz_um;
  g_tp.cell_h_def = 50 * g_ip->F_sz_um;
  g_tp.w_poly_contact = g_ip->F_sz_um;
  g_tp.spacing_poly_to_contact = g_ip->F_sz_um;
  g_tp.spacing_poly_to_poly = 1.5 * g_ip->F_sz_um;
  g_tp.ram_wl_stitching_overhead_ = 7.5 * g_ip->F_sz_um;

  g_tp.min_w_nmos_ = 3 * g_ip->F_sz_um / 2;
  g_tp.max_w_nmos_ = 100 * g_ip->F_sz_um;
  g_tp.w_iso       = 12.5*g_ip->F_sz_um;//was 10 micron for the 0.8 micron process
  g_tp.w_sense_n   = 3.75*g_ip->F_sz_um; // sense amplifier N-trans; was 3 micron for the 0.8 micron process
  g_tp.w_sense_p   = 7.5*g_ip->F_sz_um; // sense amplifier P-trans; was 6 micron for the 0.8 micron process
  g_tp.w_sense_en  = 5*g_ip->F_sz_um; // Sense enable transistor of the sense amplifier; was 4 micron for the 0.8 micron process
  g_tp.w_nmos_b_mux  = 6 * g_tp.min_w_nmos_;
  g_tp.w_nmos_sa_mux = 6 * g_tp.min_w_nmos_;

  if (ram_cell_tech_type == comm_dram)
  {
    g_tp.max_w_nmos_dec = 8 * g_ip->F_sz_um;
    g_tp.h_dec          = 8;  // in the unit of memory cell height
  }
  else
  {
    g_tp.max_w_nmos_dec = g_tp.max_w_nmos_;
    g_tp.h_dec          = 4;  // in the unit of memory cell height
  }

  /* [한국어] C_overlap: 게이트-소스/드레인 오버랩 커패시턴스 = 0.2 × C_g_ideal.
   * 이상적 게이트 커패시턴스의 20%를 오버랩 커패시턴스로 경험적으로 추정한다. */
  g_tp.peri_global.C_overlap = 0.2 * g_tp.peri_global.C_g_ideal;
  g_tp.sram_cell.C_overlap   = 0.2 * g_tp.sram_cell.C_g_ideal;
  g_tp.cam_cell.C_overlap    = 0.2 * g_tp.cam_cell.C_g_ideal;

  g_tp.dram_acc.C_overlap = 0.2 * g_tp.dram_acc.C_g_ideal;
  /* [한국어] DRAM 접근 트랜지스터 온저항: R_nch_on = Vdd_cell / I_on (A/μm). Rnchannelon과 달리
   * nmos_effective_resistance_multiplier 없이 단순 비율로 계산 — DRAM 셀 접근 속도 결정. */
  g_tp.dram_acc.R_nch_on = g_tp.dram_cell_Vdd / g_tp.dram_acc.I_on_n;
  //g_tp.dram_acc.R_pch_on = g_tp.dram_cell_Vdd / g_tp.dram_acc.I_on_p;

  g_tp.dram_wl.C_overlap = 0.2 * g_tp.dram_wl.C_g_ideal;

  /* [한국어] Sense amplifier 래치 상호 컨덕턴스 gm 계산.
   * gmn = (μ_eff/2) × Cox × (W/Lelec) × Vdsat  (long-channel MOSFET gm 공식).
   * gmp = gmp_to_gmn_multiplier × gmn (PMOS gm은 오프라인 계산 비율 적용).
   * g_tp.gm_sense_amp_latch = gmn + gmp: 양분 래치(cross-coupled inverter SA)의 총 gm.
   * 이 값은 sense amplifier 지연 및 최소 입력 차동 전압 계산에 사용된다. */
  double gmn_sense_amp_latch = (mobility_eff_periph_global / 2) * g_tp.peri_global.C_ox * (g_tp.w_sense_n / g_tp.peri_global.l_elec) * Vdsat_periph_global;
  double gmp_sense_amp_latch = gmp_to_gmn_multiplier_periph_global * gmn_sense_amp_latch;
  g_tp.gm_sense_amp_latch = gmn_sense_amp_latch + gmp_sense_amp_latch;

  /* [한국어] DRAM/SRAM/CAM 비트셀 물리 치수 계산.
   * b_w = sqrt(cell_area / asp_ratio): 비트셀 폭 (μm). asp_ratio = b_h / b_w.
   * b_h = asp_ratio × b_w: 비트셀 높이 (μm).
   * 이 치수는 어레이 면적 추정(bank_height, mat_width 계산)에 사용된다. */
  g_tp.dram.b_w = sqrt(area_cell_dram / (asp_ratio_cell_dram));
  g_tp.dram.b_h = asp_ratio_cell_dram * g_tp.dram.b_w;
  g_tp.sram.b_w = sqrt(area_cell_sram / (asp_ratio_cell_sram));
  g_tp.sram.b_h = asp_ratio_cell_sram * g_tp.sram.b_w;
  g_tp.cam.b_w =  sqrt(area_cell_cam / (asp_ratio_cell_cam));//Sheng
  g_tp.cam.b_h = asp_ratio_cell_cam * g_tp.cam.b_w;

  g_tp.dram.Vbitpre = g_tp.dram_cell_Vdd;
  g_tp.sram.Vbitpre = vdd[ram_cell_tech_type];
  g_tp.cam.Vbitpre = vdd[ram_cell_tech_type];//Sheng
  pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  g_tp.w_pmos_bl_precharge = 6 * pmos_to_nmos_sizing_r * g_tp.min_w_nmos_;
  g_tp.w_pmos_bl_eq = pmos_to_nmos_sizing_r * g_tp.min_w_nmos_;


  /* [한국어] ===== 배선(Interconnect) 파라미터 배열 선언 =====
   * 인덱스 구조: [ic_proj_type][wire_layer_type]
   *   ic_proj_type: 0=Aggressive(공격적, 낙관적 ITRS 예측), 1=Conservative(보수적 예측)
   *   wire_layer_type: 0=Local(최소 피치, 셀 내부 배선), 1=Semi-global(중간 피치),
   *                    2=Global(최대 피치, 뱅크 간 배선), 3=DRAM commodity(DRAM 워드라인/비트라인)
   * wire_r_per_micron[i][j]: 해당 배선 계층의 단위 길이당 저항 (Ω/μm), wire_resistance()로 계산
   * wire_c_per_micron[i][j]: 해당 배선 계층의 단위 길이당 커패시턴스 (F/μm), wire_capacitance()로 계산
   */
  double wire_pitch       [NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         wire_r_per_micron[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         wire_c_per_micron[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         horiz_dielectric_constant[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         vert_dielectric_constant[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         aspect_ratio[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         miller_value[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES],
         ild_thickness[NUMBER_INTERCONNECT_PROJECTION_TYPES][NUMBER_WIRE_TYPES];

  /* [한국어] ===== 배선 파라미터 선형 보간 루프 =====
   * 트랜지스터 파라미터와 동일한 보간 메커니즘을 사용한다.
   * 각 공정 노드 블록에서 wire_r_per_micron, wire_c_per_micron을 wire_resistance()와
   * wire_capacitance()를 호출하여 계산한 뒤, 루프 말미의 += curr_alpha 블록으로 g_tp에 누적.
   * Aggressive 예측: barrier_thickness=0, BULK_CU_RESISTIVITY 사용 (이상적 Cu 가정).
   * Conservative 예측: 실제 barrier_thickness, CMP 디싱, alpha_scatter > 1.0 적용.
   */
  for (iter=0; iter<=1; ++iter)
  {
    // linear interpolation
    if (iter == 0)
    {
      tech = tech_lo;
      if (tech_lo == tech_hi)
      {
        curr_alpha = 1;
      }
      else
      {
        curr_alpha = (technology - tech_hi)/(tech_lo - tech_hi);
      }
    }
    else
    {
      tech = tech_hi;
      if (tech_lo == tech_hi)
      {
        break;
      }
      else
      {
        curr_alpha = (tech_lo - technology)/(tech_lo - tech_hi);
      }
    }

    /* [한국어] ===== 180nm 배선 파라미터 블록 =====
     * Aggressive(공격적): 낮은 유전율(k=2.709) low-k ILD, 이상적 Cu(CU_RESISTIVITY, barrier=17nm).
     * Conservative(보수적): 더 높은 k(k=3.038), 동일 barrier. 두 예측 모두 180nm에서 비슷한 값.
     * Local(0): 2.5F 피치 (가장 촘촘), Semi-global(1): 4F 피치, Global(2): 8F 피치.
     * DRAM commodity[3]: pitch=2×0.18μm, R/C는 256비트 워드라인/비트라인 실측 기반 직접 상수.
     */
    if (tech == 180)
    {
    	//Aggressive projections
    	/* [한국어] Local 배선 (인덱스 [0][0]): 셀 내부 최소 피치 배선. 가장 저항 높고 커패시턴스 낮음. */
    	wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;//micron
    	aspect_ratio[0][0] = 2.0;
    	wire_width = wire_pitch[0][0] / 2; //micron   /* [한국어] 폭 = 피치/2 (50% 선폭, 50% 간격) */
    	wire_thickness = aspect_ratio[0][0] * wire_width;//micron  /* [한국어] 두께 = aspect_ratio × 폭 */
    	wire_spacing = wire_pitch[0][0] - wire_width;//micron  /* [한국어] 간격 = 피치 - 폭 */
    	barrier_thickness = 0.017;//micron  /* [한국어] Ta 배리어 라이너 두께 17nm (실사 공정) */
    	dishing_thickness = 0;//micron  /* [한국어] 로컬 배선은 폭이 좁아 CMP 디싱 무시 */
    	alpha_scatter = 1;  /* [한국어] 180nm에서는 표면 산란 보정 없음 (bulk Cu에 가까움) */
    	/* [한국어] wire_resistance(): 물리 치수로부터 단위 길이당 저항 계산 (Ω/μm) */
    	wire_r_per_micron[0][0] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);//ohm/micron
    	ild_thickness[0][0] = 0.75;//micron  /* [한국어] ILD 두께 (기판 방향 수직 커패시턴스 결정) */
    	miller_value[0][0] = 1.5;  /* [한국어] Miller 계수: 반전 신호와의 측벽 커패시턴스 증폭 (1.5배) */
    	horiz_dielectric_constant[0][0] = 2.709;  /* [한국어] 측벽 방향 low-k ILD 비유전율 (공격적 예측) */
    	vert_dielectric_constant[0][0] = 3.9;  /* [한국어] 수직 방향 SiO2 비유전율 (보수적으로 SiO2 가정) */
    	fringe_cap = 0.115e-15; //F/micron  /* [한국어] 배선 모서리 프린지 커패시턴스 */
    	/* [한국어] wire_capacitance(): 수직+측벽+프린지 성분 합산 (F/μm) */
        wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][0], miller_value[0][0], horiz_dielectric_constant[0][0],
          vert_dielectric_constant[0][0],
          fringe_cap);//F/micron.

    	wire_pitch[0][1] = 4 * g_ip->F_sz_um;
    	wire_width = wire_pitch[0][1] / 2;
    	aspect_ratio[0][1] = 2.4;
    	wire_thickness = aspect_ratio[0][1] * wire_width;
    	wire_spacing = wire_pitch[0][1] - wire_width;
    	wire_r_per_micron[0][1] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
    	ild_thickness[0][1] = 0.75;//micron
    	miller_value[0][1] = 1.5;
    	horiz_dielectric_constant[0][1] = 2.709;
    	vert_dielectric_constant[0][1] = 3.9;
    	fringe_cap = 0.115e-15; //F/micron
        wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1],
          vert_dielectric_constant[0][1],
          fringe_cap);

    	wire_pitch[0][2] = 8 * g_ip->F_sz_um;
    	aspect_ratio[0][2] = 2.2;
    	wire_width = wire_pitch[0][2] / 2;
    	wire_thickness = aspect_ratio[0][2] * wire_width;
    	wire_spacing = wire_pitch[0][2] - wire_width;
    	wire_r_per_micron[0][2] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
    	ild_thickness[0][2] = 1.5;
    	miller_value[0][2] = 1.5;
        horiz_dielectric_constant[0][2] = 2.709;
        vert_dielectric_constant[0][2] = 3.9;
        wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
          fringe_cap);

    	//Conservative projections
    	wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
    	aspect_ratio[1][0]= 2.0;
    	wire_width = wire_pitch[1][0] / 2;
    	wire_thickness = aspect_ratio[1][0] * wire_width;
    	wire_spacing = wire_pitch[1][0] - wire_width;
    	barrier_thickness = 0.017;
    	dishing_thickness = 0;
    	alpha_scatter = 1;
    	wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
    	ild_thickness[1][0] = 0.75;
    	miller_value[1][0] = 1.5;
    	horiz_dielectric_constant[1][0] = 3.038;
    	vert_dielectric_constant[1][0] = 3.9;
    	fringe_cap = 0.115e-15;
        wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0],
          vert_dielectric_constant[1][0],
          fringe_cap);

    	wire_pitch[1][1] = 4 * g_ip->F_sz_um;
    	wire_width = wire_pitch[1][1] / 2;
    	aspect_ratio[1][1] = 2.0;
    	wire_thickness = aspect_ratio[1][1] * wire_width;
    	wire_spacing = wire_pitch[1][1] - wire_width;
    	wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
    	ild_thickness[1][1] = 0.75;
    	miller_value[1][1] = 1.5;
    	horiz_dielectric_constant[1][1] = 3.038;
    	vert_dielectric_constant[1][1] = 3.9;
        wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1],
          vert_dielectric_constant[1][1],
          fringe_cap);

    	wire_pitch[1][2] = 8 * g_ip->F_sz_um;
    	aspect_ratio[1][2] = 2.2;
    	wire_width = wire_pitch[1][2] / 2;
    	wire_thickness = aspect_ratio[1][2] * wire_width;
    	wire_spacing = wire_pitch[1][2] - wire_width;
    	dishing_thickness = 0.1 *  wire_thickness;
    	wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
    			wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
    	ild_thickness[1][2]  = 1.98;
    	miller_value[1][2]  = 1.5;
        horiz_dielectric_constant[1][2]  = 3.038;
        vert_dielectric_constant[1][2]  = 3.9;
        wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][2] , miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
          fringe_cap);
    	/* [한국어] Commodity DRAM 워드라인/비트라인 배선 파라미터 (인덱스 [1][3]).
    	 * wire_c = 60fF / (256열 × 2μm 피치 × 배선 길이): 256비트 DRAM 배열의 비트라인 1μm당 커패시턴스.
    	 * wire_r = 12Ω/μm: 폴리실리콘 워드라인 저항 경험치 (Cu 배선보다 훨씬 높음). */
    	//Nominal projections for commodity DRAM wordline/bitline
    	wire_pitch[1][3] = 2 * 0.18;
    	wire_c_per_micron[1][3] = 60e-15 / (256 * 2 * 0.18);
    	wire_r_per_micron[1][3] = 12 / 0.18;
    }
    /* [한국어] ===== 90nm 배선 파라미터 블록 =====
     * 공격적: k=2.709, barrier=10nm, ILD=0.48μm (180nm 대비 축소).
     * 보수적: k=3.038, barrier=8nm. 글로벌 배선 보수적 예측에서 CMP 디싱(10% 두께 손실) 적용.
     */
    else if (tech == 90)
    {
      //Aggressive projections
      wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;//micron
      aspect_ratio[0][0] = 2.4;
      wire_width = wire_pitch[0][0] / 2; //micron
      wire_thickness = aspect_ratio[0][0] * wire_width;//micron
      wire_spacing = wire_pitch[0][0] - wire_width;//micron
      barrier_thickness = 0.01;//micron
      dishing_thickness = 0;//micron
      alpha_scatter = 1;
      wire_r_per_micron[0][0] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);//ohm/micron
      ild_thickness[0][0] = 0.48;//micron
      miller_value[0][0] = 1.5;
      horiz_dielectric_constant[0][0] = 2.709;
      vert_dielectric_constant[0][0] = 3.9;
      fringe_cap = 0.115e-15; //F/micron
      wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][0], miller_value[0][0], horiz_dielectric_constant[0][0],
          vert_dielectric_constant[0][0],
          fringe_cap);//F/micron.

      wire_pitch[0][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[0][1] / 2;
      aspect_ratio[0][1] = 2.4;
      wire_thickness = aspect_ratio[0][1] * wire_width;
      wire_spacing = wire_pitch[0][1] - wire_width;
      wire_r_per_micron[0][1] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][1] = 0.48;//micron
      miller_value[0][1] = 1.5;
      horiz_dielectric_constant[0][1] = 2.709;
      vert_dielectric_constant[0][1] = 3.9;
      wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1],
          vert_dielectric_constant[0][1],
          fringe_cap);

      wire_pitch[0][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[0][2] = 2.7;
      wire_width = wire_pitch[0][2] / 2;
      wire_thickness = aspect_ratio[0][2] * wire_width;
      wire_spacing = wire_pitch[0][2] - wire_width;
      wire_r_per_micron[0][2] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][2] = 0.96;
      miller_value[0][2] = 1.5;
      horiz_dielectric_constant[0][2] = 2.709;
      vert_dielectric_constant[0][2] = 3.9;
      wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
          fringe_cap);

      //Conservative projections
      wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[1][0]  = 2.0;
      wire_width = wire_pitch[1][0] / 2;
      wire_thickness = aspect_ratio[1][0] * wire_width;
      wire_spacing = wire_pitch[1][0] - wire_width;
      barrier_thickness = 0.008;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][0]  = 0.48;
      miller_value[1][0]  = 1.5;
      horiz_dielectric_constant[1][0]  = 3.038;
      vert_dielectric_constant[1][0]  = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0],
          vert_dielectric_constant[1][0],
          fringe_cap);

      wire_pitch[1][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[1][1] / 2;
      aspect_ratio[1][1] = 2.0;
      wire_thickness = aspect_ratio[1][1] * wire_width;
      wire_spacing = wire_pitch[1][1] - wire_width;
      wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][1]  = 0.48;
      miller_value[1][1]  = 1.5;
      horiz_dielectric_constant[1][1]  = 3.038;
      vert_dielectric_constant[1][1]  = 3.9;
      wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1],
          vert_dielectric_constant[1][1],
          fringe_cap);

      wire_pitch[1][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[1][2]  = 2.2;
      wire_width = wire_pitch[1][2] / 2;
      wire_thickness = aspect_ratio[1][2] * wire_width;
      wire_spacing = wire_pitch[1][2] - wire_width;
      dishing_thickness = 0.1 *  wire_thickness;
      wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][2]  = 1.1;
      miller_value[1][2]  = 1.5;
      horiz_dielectric_constant[1][2]  = 3.038;
      vert_dielectric_constant[1][2]  = 3.9;
      wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][2] , miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
          fringe_cap);
      //Nominal projections for commodity DRAM wordline/bitline
      wire_pitch[1][3] = 2 * 0.09;
      wire_c_per_micron[1][3] = 60e-15 / (256 * 2 * 0.09);
      wire_r_per_micron[1][3] = 12 / 0.09;
    }
    /* [한국어] ===== 65nm 배선 파라미터 블록 =====
     * 공격적: barrier=0 (BULK_CU_RESISTIVITY, 이상적 Cu), k=2.303, ILD=0.405μm.
     * 보수적: barrier=6nm, k=2.734 (여전히 low-k). 65nm부터 barrier 없는 공격적 예측이 도입됨.
     * aspect_ratio가 2.7~2.8로 증가: 높이/폭 비율 증가로 저항은 낮아지지만 측벽 커패시턴스 증가.
     */
    else if (tech == 65)
    {
      //Aggressive projections
      wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[0][0]  = 2.7;
      wire_width = wire_pitch[0][0] / 2;
      wire_thickness = aspect_ratio[0][0]  * wire_width;
      wire_spacing = wire_pitch[0][0] - wire_width;
      barrier_thickness = 0;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[0][0] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][0]  = 0.405;
      miller_value[0][0]   = 1.5;
      horiz_dielectric_constant[0][0]  = 2.303;
      vert_dielectric_constant[0][0]   = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][0] , miller_value[0][0] , horiz_dielectric_constant[0][0] , vert_dielectric_constant[0][0] ,
          fringe_cap);

      wire_pitch[0][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[0][1] / 2;
      aspect_ratio[0][1]  = 2.7;
      wire_thickness = aspect_ratio[0][1]  * wire_width;
      wire_spacing = wire_pitch[0][1] - wire_width;
      wire_r_per_micron[0][1] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][1]  = 0.405;
      miller_value[0][1]   = 1.5;
      horiz_dielectric_constant[0][1]  = 2.303;
      vert_dielectric_constant[0][1]   = 3.9;
      wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1],
          vert_dielectric_constant[0][1],
          fringe_cap);

      wire_pitch[0][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[0][2] = 2.8;
      wire_width = wire_pitch[0][2] / 2;
      wire_thickness = aspect_ratio[0][2] * wire_width;
      wire_spacing = wire_pitch[0][2] - wire_width;
      wire_r_per_micron[0][2] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][2] = 0.81;
      miller_value[0][2]   = 1.5;
      horiz_dielectric_constant[0][2]  = 2.303;
      vert_dielectric_constant[0][2]   = 3.9;
      wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
          fringe_cap);

      //Conservative projections
      wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[1][0] = 2.0;
      wire_width = wire_pitch[1][0] / 2;
      wire_thickness = aspect_ratio[1][0] * wire_width;
      wire_spacing = wire_pitch[1][0] - wire_width;
      barrier_thickness = 0.006;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][0] = 0.405;
      miller_value[1][0] = 1.5;
      horiz_dielectric_constant[1][0] = 2.734;
      vert_dielectric_constant[1][0] = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0], vert_dielectric_constant[1][0],
          fringe_cap);

      wire_pitch[1][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[1][1] / 2;
      aspect_ratio[1][1] = 2.0;
      wire_thickness = aspect_ratio[1][1] * wire_width;
      wire_spacing = wire_pitch[1][1] - wire_width;
      wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][1] = 0.405;
      miller_value[1][1] = 1.5;
      horiz_dielectric_constant[1][1] = 2.734;
      vert_dielectric_constant[1][1] = 3.9;
      wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1], vert_dielectric_constant[1][1],
          fringe_cap);

      wire_pitch[1][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[1][2] = 2.2;
      wire_width = wire_pitch[1][2] / 2;
      wire_thickness = aspect_ratio[1][2] * wire_width;
      wire_spacing = wire_pitch[1][2] - wire_width;
      dishing_thickness = 0.1 *  wire_thickness;
      wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][2] = 0.77;
      miller_value[1][2] = 1.5;
      horiz_dielectric_constant[1][2] = 2.734;
      vert_dielectric_constant[1][2] = 3.9;
      wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][2], miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
          fringe_cap);
      //Nominal projections for commodity DRAM wordline/bitline
      wire_pitch[1][3] = 2 * 0.065;
      wire_c_per_micron[1][3] = 52.5e-15 / (256 * 2 * 0.065);
      wire_r_per_micron[1][3] = 12 / 0.065;
    }
    /* [한국어] ===== 45nm 배선 파라미터 블록 =====
     * 공격적: barrier=0, BULK_CU_RESISTIVITY, k=1.958 (ultra-low-k ILD), ILD=0.315μm.
     * 보수적: barrier=4nm, k=2.46. aspect_ratio=3.0(공격적)으로 배선이 더 세워져 저항 감소.
     * 45nm부터 ultra-low-k(k<2) 유전체가 공격적 예측에 도입된다.
     */
    else if (tech == 45)
    {
      //Aggressive projections.
      wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[0][0]  = 3.0;
      wire_width = wire_pitch[0][0] / 2;
      wire_thickness = aspect_ratio[0][0]  * wire_width;
      wire_spacing = wire_pitch[0][0] - wire_width;
      barrier_thickness = 0;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[0][0] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][0]  = 0.315;
      miller_value[0][0]  = 1.5;
      horiz_dielectric_constant[0][0]  = 1.958;
      vert_dielectric_constant[0][0]  = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][0] , miller_value[0][0] , horiz_dielectric_constant[0][0] , vert_dielectric_constant[0][0] ,
          fringe_cap);

      wire_pitch[0][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[0][1] / 2;
      aspect_ratio[0][1]  = 3.0;
      wire_thickness = aspect_ratio[0][1] * wire_width;
      wire_spacing = wire_pitch[0][1] - wire_width;
      wire_r_per_micron[0][1] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][1]  = 0.315;
      miller_value[0][1]  = 1.5;
      horiz_dielectric_constant[0][1]  = 1.958;
      vert_dielectric_constant[0][1]  = 3.9;
      wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1], vert_dielectric_constant[0][1],
          fringe_cap);

      wire_pitch[0][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[0][2] = 3.0;
      wire_width = wire_pitch[0][2] / 2;
      wire_thickness = aspect_ratio[0][2] * wire_width;
      wire_spacing = wire_pitch[0][2] - wire_width;
      wire_r_per_micron[0][2] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][2] = 0.63;
      miller_value[0][2]  = 1.5;
      horiz_dielectric_constant[0][2]  = 1.958;
      vert_dielectric_constant[0][2]  = 3.9;
      wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
          fringe_cap);

      //Conservative projections
      wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[1][0] = 2.0;
      wire_width = wire_pitch[1][0] / 2;
      wire_thickness = aspect_ratio[1][0] * wire_width;
      wire_spacing = wire_pitch[1][0] - wire_width;
      barrier_thickness = 0.004;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][0] = 0.315;
      miller_value[1][0] = 1.5;
      horiz_dielectric_constant[1][0] = 2.46;
      vert_dielectric_constant[1][0] = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0], vert_dielectric_constant[1][0],
          fringe_cap);

      wire_pitch[1][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[1][1] / 2;
      aspect_ratio[1][1] = 2.0;
      wire_thickness = aspect_ratio[1][1] * wire_width;
      wire_spacing = wire_pitch[1][1] - wire_width;
      wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][1] = 0.315;
      miller_value[1][1] = 1.5;
      horiz_dielectric_constant[1][1] = 2.46;
      vert_dielectric_constant[1][1] = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1], vert_dielectric_constant[1][1],
          fringe_cap);

      wire_pitch[1][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[1][2] = 2.2;
      wire_width = wire_pitch[1][2] / 2;
      wire_thickness = aspect_ratio[1][2] * wire_width;
      wire_spacing = wire_pitch[1][2] - wire_width;
      dishing_thickness = 0.1 * wire_thickness;
      wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][2] = 0.55;
      miller_value[1][2] = 1.5;
      horiz_dielectric_constant[1][2] = 2.46;
      vert_dielectric_constant[1][2] = 3.9;
      wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][2], miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
          fringe_cap);
      //Nominal projections for commodity DRAM wordline/bitline
      wire_pitch[1][3] = 2 * 0.045;
      wire_c_per_micron[1][3] = 37.5e-15 / (256 * 2 * 0.045);
      wire_r_per_micron[1][3] = 12 / 0.045;
    }
    /* [한국어] ===== 32nm 배선 파라미터 블록 =====
     * 공격적: barrier=0, BULK_CU_RESISTIVITY, k=1.664, ILD=0.21μm.
     * 보수적: barrier=3nm, k=2.214. 배선이 더 좁아져 저항이 급격히 증가하는 세대.
     * aspect_ratio=3.0(공격적)/2.0~2.2(보수적). ILD 두께가 큰 폭으로 감소하여 수직 커패시턴스 증가.
     */
    else if (tech == 32)
    {
      //Aggressive projections.
      wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[0][0] = 3.0;
      wire_width = wire_pitch[0][0] / 2;
      wire_thickness = aspect_ratio[0][0] * wire_width;
      wire_spacing = wire_pitch[0][0] - wire_width;
      barrier_thickness = 0;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[0][0] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][0] = 0.21;
      miller_value[0][0] = 1.5;
      horiz_dielectric_constant[0][0] = 1.664;
      vert_dielectric_constant[0][0] = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][0], miller_value[0][0], horiz_dielectric_constant[0][0], vert_dielectric_constant[0][0],
          fringe_cap);

      wire_pitch[0][1] = 4 * g_ip->F_sz_um;
      wire_width = wire_pitch[0][1] / 2;
      aspect_ratio[0][1] = 3.0;
      wire_thickness = aspect_ratio[0][1] * wire_width;
      wire_spacing = wire_pitch[0][1] - wire_width;
      wire_r_per_micron[0][1] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][1] = 0.21;
      miller_value[0][1] = 1.5;
      horiz_dielectric_constant[0][1] = 1.664;
      vert_dielectric_constant[0][1] = 3.9;
      wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1], vert_dielectric_constant[0][1],
          fringe_cap);

      wire_pitch[0][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[0][2] = 3.0;
      wire_width = wire_pitch[0][2] / 2;
      wire_thickness = aspect_ratio[0][2] * wire_width;
      wire_spacing = wire_pitch[0][2] - wire_width;
      wire_r_per_micron[0][2] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[0][2] = 0.42;
      miller_value[0][2] = 1.5;
      horiz_dielectric_constant[0][2] = 1.664;
      vert_dielectric_constant[0][2] = 3.9;
      wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
          fringe_cap);

      //Conservative projections
      wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
      aspect_ratio[1][0] = 2.0;
      wire_width = wire_pitch[1][0] / 2;
      wire_thickness = aspect_ratio[1][0] * wire_width;
      wire_spacing = wire_pitch[1][0] - wire_width;
      barrier_thickness = 0.003;
      dishing_thickness = 0;
      alpha_scatter = 1;
      wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][0] = 0.21;
      miller_value[1][0] = 1.5;
      horiz_dielectric_constant[1][0] = 2.214;
      vert_dielectric_constant[1][0] = 3.9;
      fringe_cap = 0.115e-15;
      wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0], vert_dielectric_constant[1][0],
          fringe_cap);

      wire_pitch[1][1] = 4 * g_ip->F_sz_um;
      aspect_ratio[1][1] = 2.0;
      wire_width = wire_pitch[1][1] / 2;
      wire_thickness = aspect_ratio[1][1] * wire_width;
      wire_spacing = wire_pitch[1][1] - wire_width;
      wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][1] = 0.21;
      miller_value[1][1] = 1.5;
      horiz_dielectric_constant[1][1] = 2.214;
      vert_dielectric_constant[1][1] = 3.9;
      wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1], vert_dielectric_constant[1][1],
          fringe_cap);

      wire_pitch[1][2] = 8 * g_ip->F_sz_um;
      aspect_ratio[1][2] = 2.2;
      wire_width = wire_pitch[1][2] / 2;
      wire_thickness = aspect_ratio[1][2] * wire_width;
      wire_spacing = wire_pitch[1][2] - wire_width;
      dishing_thickness = 0.1 *  wire_thickness;
      wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
          wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
      ild_thickness[1][2] = 0.385;
      miller_value[1][2] = 1.5;
      horiz_dielectric_constant[1][2] = 2.214;
      vert_dielectric_constant[1][2] = 3.9;
      wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
          ild_thickness[1][2], miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
          fringe_cap);
      //Nominal projections for commodity DRAM wordline/bitline
      wire_pitch[1][3] = 2 * 0.032;//micron
      wire_c_per_micron[1][3] = 31e-15 / (256 * 2 * 0.032);//F/micron
      wire_r_per_micron[1][3] = 12 / 0.032;//ohm/micron
    }
    /* [한국어] ===== 22nm 배선 파라미터 블록 =====
     * 공격적: barrier=0, BULK_CU_RESISTIVITY, k=1.414 (극저유전율), ILD=0.15μm.
     * 보수적: barrier=3nm, alpha_scatter=1.05 (표면 산란 보정 시작), k=2.104.
     * 22nm에서 표면 산란(alpha_scatter>1)이 처음 보수적 예측에 반영된다 — 배선 폭이 Cu 평균
     * 자유 경로(~40nm)에 근접하여 표면 산란 저항이 무시 불가해진다.
     * DRAM commodity [1][3]: pitch=2×22nm. wire_c/r는 31fF/256셀 기반 직접 상수.
     */
    else if (tech == 22)
        {
          //Aggressive projections.
          wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;//local
          aspect_ratio[0][0] = 3.0;
          wire_width = wire_pitch[0][0] / 2;
          wire_thickness = aspect_ratio[0][0] * wire_width;
          wire_spacing = wire_pitch[0][0] - wire_width;
          barrier_thickness = 0;
          dishing_thickness = 0;
          alpha_scatter = 1;
          wire_r_per_micron[0][0] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][0] = 0.15;
          miller_value[0][0] = 1.5;
          horiz_dielectric_constant[0][0] = 1.414;
          vert_dielectric_constant[0][0] = 3.9;
          fringe_cap = 0.115e-15;
          wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[0][0], miller_value[0][0], horiz_dielectric_constant[0][0], vert_dielectric_constant[0][0],
            fringe_cap);

          wire_pitch[0][1] = 4 * g_ip->F_sz_um;//semi-global
          wire_width = wire_pitch[0][1] / 2;
          aspect_ratio[0][1] = 3.0;
          wire_thickness = aspect_ratio[0][1] * wire_width;
          wire_spacing = wire_pitch[0][1] - wire_width;
          wire_r_per_micron[0][1] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][1] = 0.15;
          miller_value[0][1] = 1.5;
          horiz_dielectric_constant[0][1] = 1.414;
          vert_dielectric_constant[0][1] = 3.9;
          wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1], vert_dielectric_constant[0][1],
            fringe_cap);

          wire_pitch[0][2] = 8 * g_ip->F_sz_um;//global
          aspect_ratio[0][2] = 3.0;
          wire_width = wire_pitch[0][2] / 2;
          wire_thickness = aspect_ratio[0][2] * wire_width;
          wire_spacing = wire_pitch[0][2] - wire_width;
          wire_r_per_micron[0][2] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][2] = 0.3;
          miller_value[0][2] = 1.5;
          horiz_dielectric_constant[0][2] = 1.414;
          vert_dielectric_constant[0][2] = 3.9;
          wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
        		  ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
        		  fringe_cap);

//          //*************************
//          wire_pitch[0][4] = 16 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][4] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][4] - wire_width;
//          wire_r_per_micron[0][4] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][4] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
//
//          wire_pitch[0][5] = 24 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][5] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][5] - wire_width;
//          wire_r_per_micron[0][5] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][5] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
//
//          wire_pitch[0][6] = 32 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][6] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][6] - wire_width;
//          wire_r_per_micron[0][6] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][6] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
          //*************************

          //Conservative projections
          wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
          aspect_ratio[1][0] = 2.0;
          wire_width = wire_pitch[1][0] / 2;
          wire_thickness = aspect_ratio[1][0] * wire_width;
          wire_spacing = wire_pitch[1][0] - wire_width;
          barrier_thickness = 0.003;
          dishing_thickness = 0;
          alpha_scatter = 1.05;
          wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[1][0] = 0.15;
          miller_value[1][0] = 1.5;
          horiz_dielectric_constant[1][0] = 2.104;
          vert_dielectric_constant[1][0] = 3.9;
          fringe_cap = 0.115e-15;
          wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0], vert_dielectric_constant[1][0],
            fringe_cap);

          wire_pitch[1][1] = 4 * g_ip->F_sz_um;
          wire_width = wire_pitch[1][1] / 2;
          aspect_ratio[1][1] = 2.0;
          wire_thickness = aspect_ratio[1][1] * wire_width;
          wire_spacing = wire_pitch[1][1] - wire_width;
          wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[1][1] = 0.15;
          miller_value[1][1] = 1.5;
          horiz_dielectric_constant[1][1] = 2.104;
          vert_dielectric_constant[1][1] = 3.9;
          wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1], vert_dielectric_constant[1][1],
            fringe_cap);

            wire_pitch[1][2] = 8 * g_ip->F_sz_um;
            aspect_ratio[1][2] = 2.2;
            wire_width = wire_pitch[1][2] / 2;
            wire_thickness = aspect_ratio[1][2] * wire_width;
            wire_spacing = wire_pitch[1][2] - wire_width;
            dishing_thickness = 0.1 *  wire_thickness;
            wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
            ild_thickness[1][2] = 0.275;
            miller_value[1][2] = 1.5;
            horiz_dielectric_constant[1][2] = 2.104;
            vert_dielectric_constant[1][2] = 3.9;
            wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            		ild_thickness[1][2], miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
            		fringe_cap);
            //Nominal projections for commodity DRAM wordline/bitline
            wire_pitch[1][3] = 2 * 0.022;//micron
            wire_c_per_micron[1][3] = 31e-15 / (256 * 2 * 0.022);//F/micron
            wire_r_per_micron[1][3] = 12 / 0.022;//ohm/micron

            //******************
//            wire_pitch[1][4] = 16 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][4] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][4] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][4] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][4] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
//
//            wire_pitch[1][5] = 24 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][5] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][5] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][5] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][5] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
//
//            wire_pitch[1][6] = 32 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][6] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][6] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][6] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][6] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
        }

    /* [한국어] ===== 16nm 배선 파라미터 블록 (dead code: 현재 경로에서 도달 불가) =====
     * 22nm와 동일한 구조이지만 ILD=0.108μm, k=1.202(공격적)/1.998(보수적)으로 더 작아진 값.
     * alpha_scatter=1.05(보수적): 16nm에서 표면 산란이 더 강하게 반영.
     * 이 블록은 tech_lo/tech_hi if-else에서 16nm가 주석 처리되어 실제 실행되지 않는다.
     */
    else if (tech == 16)
        {
          //Aggressive projections.
          wire_pitch[0][0] = 2.5 * g_ip->F_sz_um;//local
          aspect_ratio[0][0] = 3.0;
          wire_width = wire_pitch[0][0] / 2;
          wire_thickness = aspect_ratio[0][0] * wire_width;
          wire_spacing = wire_pitch[0][0] - wire_width;
          barrier_thickness = 0;
          dishing_thickness = 0;
          alpha_scatter = 1;
          wire_r_per_micron[0][0] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][0] = 0.108;
          miller_value[0][0] = 1.5;
          horiz_dielectric_constant[0][0] = 1.202;
          vert_dielectric_constant[0][0] = 3.9;
          fringe_cap = 0.115e-15;
          wire_c_per_micron[0][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[0][0], miller_value[0][0], horiz_dielectric_constant[0][0], vert_dielectric_constant[0][0],
            fringe_cap);

          wire_pitch[0][1] = 4 * g_ip->F_sz_um;//semi-global
          aspect_ratio[0][1] = 3.0;
          wire_width = wire_pitch[0][1] / 2;
          wire_thickness = aspect_ratio[0][1] * wire_width;
          wire_spacing = wire_pitch[0][1] - wire_width;
          wire_r_per_micron[0][1] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][1] = 0.108;
          miller_value[0][1] = 1.5;
          horiz_dielectric_constant[0][1] = 1.202;
          vert_dielectric_constant[0][1] = 3.9;
          wire_c_per_micron[0][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[0][1], miller_value[0][1], horiz_dielectric_constant[0][1], vert_dielectric_constant[0][1],
            fringe_cap);

          wire_pitch[0][2] = 8 * g_ip->F_sz_um;//global
          aspect_ratio[0][2] = 3.0;
          wire_width = wire_pitch[0][2] / 2;
          wire_thickness = aspect_ratio[0][2] * wire_width;
          wire_spacing = wire_pitch[0][2] - wire_width;
          wire_r_per_micron[0][2] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[0][2] = 0.216;
          miller_value[0][2] = 1.5;
          horiz_dielectric_constant[0][2] = 1.202;
          vert_dielectric_constant[0][2] = 3.9;
          wire_c_per_micron[0][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
        		  ild_thickness[0][2], miller_value[0][2], horiz_dielectric_constant[0][2], vert_dielectric_constant[0][2],
        		  fringe_cap);

//          //*************************
//          wire_pitch[0][4] = 16 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][4] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][4] - wire_width;
//          wire_r_per_micron[0][4] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][4] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
//
//          wire_pitch[0][5] = 24 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][5] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][5] - wire_width;
//          wire_r_per_micron[0][5] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][5] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
//
//          wire_pitch[0][6] = 32 * g_ip.F_sz_um;//global
//          aspect_ratio = 3.0;
//          wire_width = wire_pitch[0][6] / 2;
//          wire_thickness = aspect_ratio * wire_width;
//          wire_spacing = wire_pitch[0][6] - wire_width;
//          wire_r_per_micron[0][6] = wire_resistance(BULK_CU_RESISTIVITY, wire_width,
//        		  wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//          ild_thickness = 0.3;
//          wire_c_per_micron[0][6] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//        		  ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//        		  fringe_cap);
          //*************************

          //Conservative projections
          wire_pitch[1][0] = 2.5 * g_ip->F_sz_um;
          aspect_ratio[1][0] = 2.0;
          wire_width = wire_pitch[1][0] / 2;
          wire_thickness = aspect_ratio[1][0] * wire_width;
          wire_spacing = wire_pitch[1][0] - wire_width;
          barrier_thickness = 0.002;
          dishing_thickness = 0;
          alpha_scatter = 1.05;
          wire_r_per_micron[1][0] = wire_resistance(CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[1][0] = 0.108;
          miller_value[1][0] = 1.5;
          horiz_dielectric_constant[1][0] = 1.998;
          vert_dielectric_constant[1][0] = 3.9;
          fringe_cap = 0.115e-15;
          wire_c_per_micron[1][0] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[1][0], miller_value[1][0], horiz_dielectric_constant[1][0], vert_dielectric_constant[1][0],
            fringe_cap);

          wire_pitch[1][1] = 4 * g_ip->F_sz_um;
          wire_width = wire_pitch[1][1] / 2;
          aspect_ratio[1][1] = 2.0;
          wire_thickness = aspect_ratio[1][1] * wire_width;
          wire_spacing = wire_pitch[1][1] - wire_width;
          wire_r_per_micron[1][1] = wire_resistance(CU_RESISTIVITY, wire_width,
            wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
          ild_thickness[1][1] = 0.108;
          miller_value[1][1] = 1.5;
          horiz_dielectric_constant[1][1] = 1.998;
          vert_dielectric_constant[1][1] = 3.9;
            wire_c_per_micron[1][1] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            ild_thickness[1][1], miller_value[1][1], horiz_dielectric_constant[1][1], vert_dielectric_constant[1][1],
            fringe_cap);

            wire_pitch[1][2] = 8 * g_ip->F_sz_um;
            aspect_ratio[1][2] = 2.2;
            wire_width = wire_pitch[1][2] / 2;
            wire_thickness = aspect_ratio[1][2] * wire_width;
            wire_spacing = wire_pitch[1][2] - wire_width;
            dishing_thickness = 0.1 *  wire_thickness;
            wire_r_per_micron[1][2] = wire_resistance(CU_RESISTIVITY, wire_width,
            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
            ild_thickness[1][2] = 0.198;
            miller_value[1][2] = 1.5;
            horiz_dielectric_constant[1][2] = 1.998;
            vert_dielectric_constant[1][2] = 3.9;
            wire_c_per_micron[1][2] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
            		ild_thickness[1][2], miller_value[1][2], horiz_dielectric_constant[1][2], vert_dielectric_constant[1][2],
            		fringe_cap);
            //Nominal projections for commodity DRAM wordline/bitline
            wire_pitch[1][3] = 2 * 0.016;//micron
            wire_c_per_micron[1][3] = 31e-15 / (256 * 2 * 0.016);//F/micron
            wire_r_per_micron[1][3] = 12 / 0.016;//ohm/micron

            //******************
//            wire_pitch[1][4] = 16 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][4] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][4] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][4] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][4] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
//
//            wire_pitch[1][5] = 24 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][5] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][5] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][5] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][5] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
//
//            wire_pitch[1][6] = 32 * g_ip.F_sz_um;
//            aspect_ratio = 2.2;
//            wire_width = wire_pitch[1][6] / 2;
//            wire_thickness = aspect_ratio * wire_width;
//            wire_spacing = wire_pitch[1][6] - wire_width;
//            dishing_thickness = 0.1 *  wire_thickness;
//            wire_r_per_micron[1][6] = wire_resistance(CU_RESISTIVITY, wire_width,
//            		wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter);
//            ild_thickness = 0.275;
//            wire_c_per_micron[1][6] = wire_capacitance(wire_width, wire_thickness, wire_spacing,
//            		ild_thickness, miller_value, horiz_dielectric_constant, vert_dielectric_constant,
//            		fringe_cap);
        }
    /* [한국어] ===== 배선 파라미터 g_tp 누적 블록 =====
     * wire_local: 셀 내부 Local 배선. comm_dram이면 인덱스 3(DRAM 전용), 아니면 인덱스 0(Local).
     * wire_inside_mat: 매트(Mat) 내부 배선. g_ip->wire_is_mat_type으로 선택 (Local/Semi-global/Global).
     * wire_outside_mat: 매트 간(뱅크 간) 배선. g_ip->wire_os_mat_type으로 선택.
     * ic_proj_type: 0=Aggressive, 1=Conservative — g_ip->ic_proj_type으로 선택.
     * 두 이터레이션(tech_lo, tech_hi)에서 += curr_alpha 로 누적하여 최종 보간값이 g_tp에 저장된다.
     */

    /* [한국어] wire_local: comm_dram이면 DRAM 워드라인/비트라인 전용 인덱스(3), 아니면 Local 배선(0) */
    g_tp.wire_local.pitch    += curr_alpha * wire_pitch[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.R_per_um += curr_alpha * wire_r_per_micron[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.C_per_um += curr_alpha * wire_c_per_micron[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.aspect_ratio  += curr_alpha * aspect_ratio[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.ild_thickness += curr_alpha * ild_thickness[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.miller_value   += curr_alpha * miller_value[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.horiz_dielectric_constant += curr_alpha* horiz_dielectric_constant[g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];
    g_tp.wire_local.vert_dielectric_constant  += curr_alpha* vert_dielectric_constant [g_ip->ic_proj_type][(ram_cell_tech_type == comm_dram)?3:0];

    /* [한국어] wire_inside_mat: 매트 내부 배선. g_ip->wire_is_mat_type(0/1/2)으로 Local/Semi-global/Global 선택 */
    g_tp.wire_inside_mat.pitch     += curr_alpha * wire_pitch[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.R_per_um  += curr_alpha* wire_r_per_micron[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.C_per_um  += curr_alpha* wire_c_per_micron[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.aspect_ratio  += curr_alpha * aspect_ratio[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.ild_thickness += curr_alpha * ild_thickness[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.miller_value   += curr_alpha * miller_value[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.horiz_dielectric_constant += curr_alpha* horiz_dielectric_constant[g_ip->ic_proj_type][g_ip->wire_is_mat_type];
    g_tp.wire_inside_mat.vert_dielectric_constant  += curr_alpha* vert_dielectric_constant [g_ip->ic_proj_type][g_ip->wire_is_mat_type];

    /* [한국어] wire_outside_mat: 매트 간 배선. g_ip->wire_os_mat_type(0/1/2)으로 계층 선택 */
    g_tp.wire_outside_mat.pitch    += curr_alpha * wire_pitch[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.R_per_um += curr_alpha*wire_r_per_micron[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.C_per_um += curr_alpha*wire_c_per_micron[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.aspect_ratio  += curr_alpha * aspect_ratio[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.ild_thickness += curr_alpha * ild_thickness[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.miller_value   += curr_alpha * miller_value[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.horiz_dielectric_constant += curr_alpha* horiz_dielectric_constant[g_ip->ic_proj_type][g_ip->wire_os_mat_type];
    g_tp.wire_outside_mat.vert_dielectric_constant  += curr_alpha* vert_dielectric_constant [g_ip->ic_proj_type][g_ip->wire_os_mat_type];

    /* [한국어] unit_len_wire_del: 배선 RC 지연 (s/μm²). Elmore 지연 모델: t = R·C/2.
     * 배선 길이의 제곱에 비례하는 지연으로, 긴 배선일수록 지연이 급격히 증가한다. */
    g_tp.unit_len_wire_del = g_tp.wire_inside_mat.R_per_um * g_tp.wire_inside_mat.C_per_um / 2;

    /* [한국어] Sense amplifier 지연(SENSE_AMP_D, s)과 동적 소비 에너지(SENSE_AMP_P, J) 보간·누적.
     * 이 값들은 각 공정 노드 블록에서 SPICE 시뮬레이션 기반으로 직접 설정된 경험 수치이다. */
    g_tp.sense_delay               += curr_alpha *SENSE_AMP_D;
    g_tp.sense_dy_power            += curr_alpha *SENSE_AMP_P;
//    g_tp.horiz_dielectric_constant += horiz_dielectric_constant;
//    g_tp.vert_dielectric_constant  += vert_dielectric_constant;
//    g_tp.aspect_ratio              += aspect_ratio;
//    g_tp.miller_value              += miller_value;
//    g_tp.ild_thickness             += ild_thickness;

  }
  g_tp.fringe_cap = fringe_cap;  /* [한국어] 최종 배선 루프의 마지막 fringe_cap 값을 g_tp에 저장. 배선 커패시턴스 계산에 재사용 */

  /* [한국어] ===== kinv: 단위 인버터 지연 계산 (Fan-out-1 단자 인버터 지연) =====
   * rd: 최소 폭 NMOS(min_w_nmos_)의 온저항 (Ω). tr_R_on()으로 트랜지스터 파라미터 기반 계산.
   * p_to_n_sizing_r: PMOS/NMOS 구동 전류 비 — 인버터에서 PMOS 폭 = p_to_n_sizing_r × NMOS 폭.
   * c_load: 게이트 커패시턴스만 포함 (Fan-out-1: 다음 단 인버터 게이트 1개의 커패시턴스).
   * tf = rd × c_load: RC 시상수.
   * kinv = horowitz(0, tf, 0.5, 0.5, RISE): Horowitz 모델로 50%→50% 상승 지연 계산.
   *   Horowitz 모델: t ≈ tf × sqrt(0.69 + ...) — 실제 MOSFET 전달 특성을 선형 근사.
   */
  double rd = tr_R_on(g_tp.min_w_nmos_, NCH, 1);
  double p_to_n_sizing_r = pmos_to_nmos_sz_ratio();
  double c_load = gate_C(g_tp.min_w_nmos_ * (1 + p_to_n_sizing_r), 0.0);
  double tf = rd * c_load;
  g_tp.kinv = horowitz(0, tf, 0.5, 0.5, RISE);

  /* [한국어] ===== FO4: Fan-Out-4 인버터 지연 계산 (표준 공정 성능 지표) =====
   * FO4는 공정 노드 성능을 나타내는 핵심 지표: 단위 인버터가 자신의 4배 크기 인버터 4개를
   * 구동할 때의 지연 시간. CACTI에서 파이프라인 단계 지연의 기준 단위로 사용된다.
   * c_load: FO4 부하 커패시턴스 = KLOAD × (NMOS 드레인 C + PMOS 드레인 C + 4× 게이트 C).
   *   drain_C_(): 드레인 접합 커패시턴스 계산 (c_junc + c_fringe 기반).
   *   gate_C(): 게이트 커패시턴스 계산 (C_g_ideal + C_overlap + C_fringe 기반).
   *   4×(1+p_to_n_sizing_r)는 팬아웃-4 부하: NMOS 4개 + PMOS 4개의 게이트.
   * g_tp.FO4 = horowitz(): 최종 FO4 지연 (s). GPU 파이프라인 설계 지연 추정의 기준이 된다.
   */
  double KLOAD = 1;
  c_load = KLOAD * (drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) +
                    drain_C_(g_tp.min_w_nmos_ * p_to_n_sizing_r, PCH, 1, 1, g_tp.cell_h_def) +
                    gate_C(g_tp.min_w_nmos_ * 4 * (1 + p_to_n_sizing_r), 0.0));
  tf = rd * c_load;
  g_tp.FO4 = horowitz(0, tf, 0.5, 0.5, RISE);
}

