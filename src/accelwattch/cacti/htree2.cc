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
 * [한국어 설명] H-tree 와이어 모델 구현 (htree2.cc)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 전력/지연 모델에서 H-tree 배선 구조의 생성자, in_htree(), out_htree(),
 * input_nand(), output_buffer() 함수를 구현한다. H-tree는 캐시 뱅크 내 복수의 mat을
 * 균일한 RC 지연으로 연결하는 프랙탈 이진 트리 배선 위상이다.
 * 각 함수는 Wire 객체로 배선 RC 지연·전력을 구하고, 노드마다 NAND/tristate 버퍼 모델을
 * 삽입한 뒤 Component::delay, power, area에 결과를 누적한다.
 * AccelWattch가 GPU 캐시의 배선 에너지를 추정할 때 이 구현이 핵심적으로 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 계산 흐름: UCA → Htree2 (배선 지연/전력) + Mat (어레이 지연/전력)
 * Htree2는 UCA::UCA() 생성자 안에서 5가지 H-tree 타입별로 독립적으로 생성된다.
 * 결과(delay, power, area)는 UCA 레벨에서 Mat 결과와 합산되어 최종 캐시 특성값이 된다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, AccelWattch 초기화 단계 (시뮬레이션 시작 시 1회).
 *
 * === 타 모듈과의 연결 ===
 * 의존: Wire(배선 RC), basic_circuit(tr_R_on/gate_C/drain_C_/horowitz/cmos_Isub_leakage/cmos_Ig_leakage),
 *       parameter.h(g_tp 전역 기술 파라미터, DeviceType::Vth/Vdd/n_to_p_eff_curr_drv_ratio)
 * 상위: UCA::UCA() — Htree2 객체를 생성하고 delay/power를 읽어 bank 수준 결과에 합산
 * 데이터 흐름: mat_width/mat_height × ndbl/ndwl → 트리 링크 길이 계산 →
 *              루프: Wire(길이) → RC 지연 누적 + NAND/output_buffer 노드 전력 누적 →
 *              Component::delay, power, area 에 최종 저장
 *
 * === 주요 함수/구조체 요약 ===
 * Htree2()       : 생성자 — htree_type 스위치로 wire_bw 초기화 후 in/out_htree() 즉시 호출
 * input_nand()   : H-tree 입력 노드의 NAND2 리피터 지연·동적/누설 전력 계산
 * output_buffer(): H-tree 출력 노드의 tristate 버퍼(NOT+NAND+NOR+출력 tr) 전력 계산
 * in_htree()     : 루트→리프 브로드캐스트 H-tree의 링크별 지연·전력 루프 (불균형 트리 처리 포함)
 * out_htree()    : 리프→루트 수집 H-tree의 링크별 지연·전력 루프 (tristate fanin 수집)
 */

#include "htree2.h"  // [한국어] Htree2 클래스 선언 및 관련 헤더 포함
#include "wire.h"    // [한국어] Wire 클래스 — 배선 RC 지연, 리피터 크기/간격 계산
#include <assert.h>  // [한국어] 전제 조건 검증 (ndbl/ndwl >= 2 등)
#include <iostream>  // [한국어] 디버그 출력 (현재 주석 처리된 cout 코드에서 사용)

/*
 * [한국어]
 * Htree2::Htree2 - H-tree 배선 모델 생성자
 *
 * @wire_model    : 배선 층 종류 (Wire_type enum) — Wire 객체 파라미터 선택
 * @mat_w, mat_h  : 단일 mat의 가로/세로 (미터) — H-tree 링크 길이의 기본 단위
 * @a_bits        : 어드레스 비트 수
 * @d_inbits      : 데이터 입력 비트 수
 * @search_data_in: CAM 검색 입력 비트 수
 * @d_outbits     : 데이터 출력 비트 수
 * @search_data_out: CAM 검색 출력 비트 수
 * @bl, wl        : ndbl(수직 mat 분할 수), ndwl(수평 mat 분할 수)
 * @htree_type    : H-tree 종류 — wire_bw 초깃값과 in/out_htree() 선택을 결정
 * @uca_tree_     : interbank 트리 모드 여부
 * @search_tree_  : CAM 검색 트리 모드 여부
 * @dt            : 사용할 DeviceType 포인터 (기본: g_tp.peri_global)
 * @return        : (생성자) — delay, power, area가 계산된 상태로 반환
 *
 * 초기화 리스트에서 모든 멤버를 설정한 뒤, htree_type 스위치로 wire_bw를 초기화하고
 * in_htree() 또는 out_htree()를 즉시 호출하여 지연·전력을 계산한다.
 * 계산 후 power_bit에 1비트당 전력을 저장하고, 전체 버스 전력은 init_wire_bw를 곱하여 확정.
 *
 * 호출 체인:
 *   UCA::UCA() → [Htree2::Htree2()] → in_htree() 또는 out_htree()
 */
Htree2::Htree2(
    enum Wire_type wire_model, double mat_w, double mat_h,
    int a_bits, int d_inbits, int search_data_in, int d_outbits, int search_data_out, int bl, int wl, enum Htree_type htree_type,
    bool uca_tree_, bool search_tree_, TechnologyParameter::DeviceType *dt)
 :in_rise_time(0), out_rise_time(0),          // [한국어] 신호 상승 시간 0으로 초기화 (이상적 step 입력 가정)
  tree_type(htree_type), mat_width(mat_w), mat_height(mat_h), // [한국어] 트리 타입과 mat 크기 설정
  add_bits(a_bits), data_in_bits(d_inbits), search_data_in_bits(search_data_in), data_out_bits(d_outbits),
  search_data_out_bits(search_data_out), ndbl(bl), ndwl(wl), // [한국어] 버스 비트 폭과 분할 수 초기화
  uca_tree(uca_tree_), search_tree(search_tree_), wt(wire_model), deviceType(dt) // [한국어] 모드 플래그와 배선 종류 초기화
{
  assert(ndbl >= 2 && ndwl >= 2); // [한국어] H-tree는 최소 2분할(한 레벨) 이상이어야 동작 가능; 위반 시 중단

  // [한국어] 아래 주석 처리된 블록은 ndbl=1 또는 ndwl=1인 단일 mat 예외처리 코드였으나
  //          현재는 assert로 2 이상을 강제하므로 비활성화됨.

  max_unpipelined_link_delay = 0; //TODO // [한국어] 파이프라이닝 없는 최대 링크 지연 — 미구현(항상 0)
  min_w_nmos = g_tp.min_w_nmos_;  // [한국어] 전역 기술 파라미터에서 NMOS 최소 폭 가져오기
  min_w_pmos = deviceType->n_to_p_eff_curr_drv_ratio * min_w_nmos; // [한국어] PMOS 최소 폭 = PMOS/NMOS 전류 구동비 × NMOS 최소폭

  switch (htree_type) // [한국어] H-tree 타입에 따라 버스 폭(wire_bw)을 초기화하고 계산 함수 선택
  {
    case Add_htree: // [한국어] 어드레스 신호 배포 트리 — 루트→리프 브로드캐스트
      wire_bw = init_wire_bw = add_bits; // [한국어] 버스 폭 = 어드레스 비트 수
      in_htree(); // [한국어] 입력 H-tree 지연·전력 계산 실행
      break;
    case Data_in_htree: // [한국어] 데이터 입력 신호 배포 트리 — 루트→리프 브로드캐스트
      wire_bw = init_wire_bw = data_in_bits; // [한국어] 버스 폭 = 데이터 입력 비트 수
      in_htree();
      break;
    case Data_out_htree: // [한국어] 데이터 출력 수집 트리 — 리프→루트 fanin
      wire_bw = init_wire_bw = data_out_bits; // [한국어] 버스 폭 = 데이터 출력 비트 수
      out_htree(); // [한국어] 출력 H-tree(tristate 버퍼 기반) 계산 실행
      break;
    case Search_in_htree: // [한국어] CAM 검색 입력 브로드캐스트 트리
      wire_bw = init_wire_bw = search_data_in_bits; // [한국어] 버스 폭 = CAM 검색 입력 비트 수 (in_search_tree는 브로드캐스트)
      in_htree();
      break;
    case Search_out_htree: // [한국어] CAM 검색 결과 수집 트리
      wire_bw = init_wire_bw = search_data_out_bits; // [한국어] 버스 폭 = CAM 검색 출력 비트 수
      out_htree();
      break;
    default:
      assert(0); // [한국어] 정의되지 않은 H-tree 타입 — 프로그래밍 오류, 즉시 중단
      break;
  }

  power_bit = power; // [한국어] init_wire_bw 곱셈 전 상태의 전력을 power_bit에 저장 (1비트당 전력)
  power.readOp.dynamic *= init_wire_bw; // [한국어] 총 동적 전력 = 1비트 전력 × 전체 버스 폭 (전체 버스 에너지로 확장)

  assert(power.readOp.dynamic >= 0); // [한국어] 동적 전력은 반드시 0 이상 (음수이면 계산 오류)
  assert(power.readOp.leakage >= 0); // [한국어] 누설 전력도 반드시 0 이상
}



/*
 * [한국어]
 * Htree2::input_nand - 입력 H-tree 노드의 NAND2 리피터 지연·전력 계산
 *
 * @s1    : 현재 배선 리피터 크기 (이전 Wire 객체의 repeater_size에서 유도)
 * @s2    : 다음 레벨 리피터 크기 (다음 Wire 객체의 repeater_size)
 * @l_eff : 유효 배선 길이 (µm 단위) — Wire 리피터 간격과 비교하여 실제 s1을 스케일링
 * @return: (void) — Component::delay 및 power에 이 노드의 기여분을 누적
 *
 * H-tree 입력 노드에서 신호를 재생하는 NAND2 게이트의 지연과 전력을 모델링한다.
 * NAND2를 리피터로 사용하는 이유: 단순 인버터와 달리 두 입력 중 하나를 Enable로 쓸 수 있어
 * H-tree 노드에서 선택적 활성화(예: 어드레스 디코딩)에 적합하기 때문이다.
 * 지연은 horowitz 모델(Elmore RC 지연 공식), 전력은 0.5*C*V^2*f 공식을 사용한다.
 * 누설 전력은 wire_bw(현재 레벨 버스 폭)를 곱하여 전체 배선의 누설을 반영한다.
 *
 * 호출 체인:
 *   in_htree() → [input_nand(s1, s2, l_eff)]
 */
// nand gate sizing calculation
void Htree2::input_nand(double s1, double s2, double l_eff)
{
  Wire w1(wt, l_eff); // [한국어] 유효 배선 길이 l_eff에 대한 Wire 객체 생성 — out_rise_time(출력 상승시간) 계산
  double pton_size = deviceType->n_to_p_eff_curr_drv_ratio; // [한국어] PMOS/NMOS 유효 전류 구동비 (공정 파라미터)
  // input capacitance of a repeater  = input capacitance of nand.
  // [한국어] 리피터 입력 커패시턴스 = NAND 입력 커패시턴스 (NMOS 기여 + PMOS 기여)
  double nsize = s1*(1 + pton_size)/(2 + pton_size); // [한국어] NAND 내 NMOS 트랜지스터 크기 — PMOS/NMOS 비율로 정규화
  nsize = (nsize < 1) ? 1 : nsize; // [한국어] 최소 크기 클램핑: 최솟값은 1 (최소 트랜지스터 폭)

  // [한국어] tc: NAND2 출력 노드의 RC 시정수 계산
  // 2 × NMOS 온-저항 × (NAND 내부 중간 노드 커패시턴스×2 + 다음 단 게이트 커패시턴스×2)
  double tc = 2*tr_R_on(nsize*min_w_nmos, NCH, 1) *
    (drain_C_(nsize*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)*2 + // [한국어] NMOS 드레인 커패시턴스 × 2 (직렬 NMOS 두 개)
     2 * gate_C(s2*(min_w_nmos + min_w_pmos), 0)); // [한국어] 다음 레벨 리피터 게이트 커패시턴스 × 2 (두 팬아웃)

  delay+= horowitz (w1.out_rise_time, tc, // [한국어] horowitz 모델로 NAND 지연 계산: 입력 상승시간 + RC 시정수 기반
      deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, RISE); // [한국어] Vth/Vdd 비율 — 논리 전환 임계점(입력/출력 동일)

  // [한국어] 동적 전력(readOp): 0.5 × C_total × Vdd² — NAND2 스위칭 1회당 에너지
  // C_total = PMOS 드레인×2 + NMOS 드레인×1 + 다음 단 게이트×2
  power.readOp.dynamic += 0.5 *
    (2*drain_C_(pton_size * nsize*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) // [한국어] PMOS 드레인 커패시턴스 × 2
     + drain_C_(nsize*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)             // [한국어] NMOS 드레인 커패시턴스
     + 2*gate_C(s2*(min_w_nmos + min_w_pmos), 0)) *                       // [한국어] 다음 단 입력 게이트 커패시턴스 × 2
    deviceType->Vdd * deviceType->Vdd; // [한국어] Vdd² 항 — CMOS 스위칭 에너지 공식

  // [한국어] 검색 동적 전력(searchOp): CAM 검색 시 전체 wire_bw 비트 동시 스위칭 가정 → ×wire_bw
  power.searchOp.dynamic += 0.5 *
    (2*drain_C_(pton_size * nsize*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
     + drain_C_(nsize*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)
     + 2*gate_C(s2*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd * wire_bw; // [한국어] 현재 H-tree 레벨의 버스 폭으로 스케일링

  // [한국어] 서브스레숄드 누설 전력: wire_bw개 NAND2 게이트의 누설 합산 × Vdd
  power.readOp.leakage += (wire_bw*cmos_Isub_leakage(min_w_nmos*(nsize*2), min_w_pmos * nsize * 2, 2, nand))*deviceType->Vdd;
  // [한국어] 게이트 터널링 누설 전력: 동일 구조 × Vdd
  power.readOp.gate_leakage += (wire_bw*cmos_Ig_leakage(min_w_nmos*(nsize*2), min_w_pmos * nsize * 2, 2, nand))*deviceType->Vdd;
}



/*
 * [한국어]
 * Htree2::output_buffer - 출력 H-tree 노드의 tristate 버퍼 지연·전력 계산
 *
 * @s1    : 현재 레벨 tristate 버퍼 크기 (이전 Wire repeater_size에서 유도)
 * @s2    : 다음 레벨 버퍼 크기 (다음 Wire repeater_size)
 * @l_eff : 유효 배선 길이 (µm) — Wire 리피터 간격과 비교하여 실제 크기 스케일링
 * @return: (void) — Component::delay 및 power에 이 노드의 기여분을 누적
 *
 * H-tree 출력 노드에서 복수의 리프(mat)가 공유 버스로 데이터를 합쳐 보내는 fanin을 처리한다.
 * tristate 버퍼 구조: NOT + NAND2(제어 입력 생성) + NOR2(드라이버 제어) + PMOS/NMOS 출력 트랜지스터 쌍.
 * 지연은 NOR의 저항 × NAND 출력 커패시턴스 + (NOR+출력tr 저항) × 출력 트랜지스터 커패시턴스로 계산.
 * 동적 전력은 4단(NAND, NOT, NOR, 출력 tr) 각각의 0.5*C*Vdd²를 별도로 계산하여 합산한다.
 * searchOp.dynamic은 init_wire_bw(루트 버스 폭)를 곱하여 전체 검색 버스 에너지로 확장한다.
 * 누설 전력은 uca_tree 여부와 관계없이 현재 같은 수식이 적용됨 (TODO: 분기 코드가 동일).
 *
 * 호출 체인:
 *   out_htree() → [output_buffer(s1, s2, l_eff)]
 */
// tristate buffer model consisting of not, nand, nor, and driver transistors
void Htree2::output_buffer(double s1, double s2, double l_eff)
{
  Wire w1(wt, l_eff); // [한국어] 유효 배선 길이에 대한 Wire 객체 — out_rise_time 및 wire_cap 제공
  double pton_size = deviceType->n_to_p_eff_curr_drv_ratio; // [한국어] PMOS/NMOS 전류 구동비 (공정 파라미터)
  // input capacitance of repeater = input capacitance of nand + nor.
  // [한국어] tristate 버퍼의 입력 커패시턴스 = NAND 입력 + NOR 입력 (두 병렬 경로)
  double size = s1*(1 + pton_size)/(2 + pton_size + 1 + 2*pton_size); // [한국어] NAND/NOR 내 트랜지스터 크기 계산 (PMOS/NMOS 비율로 정규화)
  double s_eff =  //stage eff of a repeater in a wire
    // [한국어] s_eff: Wire 리피터의 스테이지 팩터 = (게이트 부하 + 배선 커패시턴스) / 게이트 부하
    (gate_C(s2*(min_w_nmos + min_w_pmos), 0) + w1.wire_cap(l_eff*1e-6,true))/
    gate_C(s2*(min_w_nmos + min_w_pmos), 0);
  double tr_size = gate_C(s1*(min_w_nmos + min_w_pmos), 0) * 1/2/(s_eff*gate_C(min_w_pmos, 0));
  // [한국어] tr_size: 출력 tristate 트랜지스터 크기 — 이전 단의 구동 능력(s1 게이트 커패시턴스)과 스테이지 팩터에서 유도
  size = (size < 1) ? 1 : size; // [한국어] 최소 크기 클램핑 (1 미만이면 1로 고정)

  double res_nor = 2*tr_R_on(size*min_w_pmos, PCH, 1); // [한국어] NOR 내 직렬 PMOS 두 개의 온-저항 합
  double res_ptrans = tr_R_on(tr_size*min_w_nmos, NCH, 1); // [한국어] 출력 tristate NMOS 트랜지스터의 온-저항
  double cap_nand_out = drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] NAND 출력 NMOS 드레인 커패시턴스
                        drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)*2 + // [한국어] NAND 출력 PMOS 드레인 커패시턴스 × 2
                        gate_C(tr_size*min_w_pmos, 0); // [한국어] 출력 PMOS 트랜지스터의 게이트 커패시턴스 (NAND 출력이 구동하는 부하)
  double cap_ptrans_out = 2 *(drain_C_(tr_size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 출력 PMOS+NMOS 드레인 커패시턴스 × 2 (양쪽 병렬)
                              drain_C_(tr_size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)) +
                          gate_C(s1*(min_w_nmos + min_w_pmos), 0); // [한국어] 다음 단 인버터 입력 게이트 커패시턴스

  // [한국어] tc: tristate 버퍼의 총 RC 시정수
  // = NOR 저항 × NAND 출력 커패시턴스 + (NOR 저항 + 출력 tr 저항) × 출력 트랜지스터 커패시턴스
  double tc = res_nor * cap_nand_out + (res_nor + res_ptrans) * cap_ptrans_out;

  delay += horowitz (w1.out_rise_time, tc, // [한국어] tristate 버퍼 전체 지연 계산 (Wire RC 지연 이후의 버퍼 지연)
      deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, RISE);

  //nand
  // [한국어] NAND2 단의 동적 전력: PMOS×2 드레인 + NMOS×1 드레인 + 출력 tr 게이트 커패시턴스
  power.readOp.dynamic += 0.5 *
    (2*drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) +
       drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def) +
     gate_C(tr_size*(min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd;

  // [한국어] NAND2 단의 CAM 검색 동적 전력 — init_wire_bw(루트 버스 폭) 배수로 전체 버스 에너지 확장
  power.searchOp.dynamic += 0.5 *
    (2*drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) +
       drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def) +
     gate_C(tr_size*(min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd*init_wire_bw;

  //not
  // [한국어] NOT(인버터) 단의 동적 전력: PMOS+NMOS 드레인 + 다음 단 입력 게이트 커패시턴스
  power.readOp.dynamic += 0.5 *
    (drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
     +drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)
     +gate_C(size*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd;

  // [한국어] NOT 단의 CAM 검색 동적 전력 — init_wire_bw 배수 확장
  power.searchOp.dynamic += 0.5 *
    (drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
     +drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)
     +gate_C(size*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd*init_wire_bw;

  //nor
  // [한국어] NOR2 단의 동적 전력: PMOS×1 드레인 + NMOS×2 드레인 + 출력 tr 게이트 커패시턴스
  power.readOp.dynamic += 0.5 *
    (drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
     + 2*drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)
     +gate_C(tr_size*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd;

  // [한국어] NOR2 단의 CAM 검색 동적 전력
  power.searchOp.dynamic += 0.5 *
    (drain_C_(size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
     + 2*drain_C_(size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def)
     +gate_C(tr_size*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd*init_wire_bw;

  //output transistor
  // [한국어] 출력 tristate 트랜지스터 단의 동적 전력
  // (PMOS+NMOS 드레인)×2 + 다음 단 입력 게이트 커패시턴스
  power.readOp.dynamic += 0.5 *
    ((drain_C_(tr_size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
      +drain_C_(tr_size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def))*2
     + gate_C(s1*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd;

  // [한국어] 출력 tr 단의 CAM 검색 동적 전력
  power.searchOp.dynamic += 0.5 *
    ((drain_C_(tr_size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)
      +drain_C_(tr_size*min_w_nmos, NCH, 1, 1, g_tp.cell_h_def))*2
     + gate_C(s1*(min_w_nmos + min_w_pmos), 0)) *
    deviceType->Vdd * deviceType->Vdd*init_wire_bw;

  if(uca_tree) { // [한국어] UCA interbank 트리 모드: 누설은 wire_bw 기준으로 계산
    // [한국어] 서브스레숄드 누설: 인버터+출력tr, NAND, NOR 각 단의 누설 전류 × Vdd × wire_bw
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*tr_size*2, min_w_pmos*tr_size*2, 1, inv)*deviceType->Vdd*wire_bw;/*inverter + output tr*/
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nand)*deviceType->Vdd*wire_bw;//nand
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nor)*deviceType->Vdd*wire_bw;//nor

    // [한국어] 게이트 터널링 누설: 동일 구조의 게이트 누설 × Vdd × wire_bw
	power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*tr_size*2, min_w_pmos*tr_size*2, 1, inv)*deviceType->Vdd*wire_bw;/*inverter + output tr*/
    power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nand)*deviceType->Vdd*wire_bw;//nand
    power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nor)*deviceType->Vdd*wire_bw;//nor
    //power.readOp.gate_leakage *=;
  }
  else { // [한국어] 뱅크 내부 트리 모드: 동일한 수식 적용 (현재 두 분기가 동일 — 향후 분리 예정)
    // [한국어] 서브스레숄드 누설
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*tr_size*2, min_w_pmos*tr_size*2, 1, inv)*deviceType->Vdd*wire_bw;/*inverter + output tr*/
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nand)*deviceType->Vdd*wire_bw;//nand
	power.readOp.leakage += cmos_Isub_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nor)*deviceType->Vdd*wire_bw;//nor

    // [한국어] 게이트 터널링 누설
	power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*tr_size*2, min_w_pmos*tr_size*2, 1, inv)*deviceType->Vdd*wire_bw;/*inverter + output tr*/
    power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nand)*deviceType->Vdd*wire_bw;//nand
    power.readOp.gate_leakage += cmos_Ig_leakage(min_w_nmos*size*3, min_w_pmos*size*3, 2, nor)*deviceType->Vdd*wire_bw;//nor
    //power.readOp.gate_leakage *=deviceType->Vdd*wire_bw;
  }
}



/* calculates the input h-tree delay/power
 * A nand gate is used at each node to
 * limit the signal
 * The area of an unbalanced htree (rows != columns)
 * depends on how data is traversed.
 * In the following function, if ( no. of rows < no. of columns),
 * then data first traverse in excess hor. links until vertical
 * and horizontal nodes are same.
 * If no. of rows is bigger, then data traverse in
 * a hor. link followed by a ver. link in a repeated
 * fashion (similar to a balanced tree) until there are no
 * hor. links left. After this it goes through the remaining vertical
 * links.
 */
/*
 * [한국어]
 * Htree2::in_htree - 입력(루트→리프 브로드캐스트) H-tree 지연·전력 계산
 *
 * @return: (void) — Component::delay, power, area에 계산 결과를 직접 저장
 *
 * 어드레스/데이터 입력 신호가 루트에서 모든 mat 리프로 브로드캐스트되는 H-tree를 모델링한다.
 * 알고리즘:
 *   1) H-tree 전체 크기(area.h, area.w)를 mat_height/mat_width × ndbl/ndwl로 결정
 *   2) ht_temp, len_temp: 각 레벨에서 처리할 배선의 절반 길이 (루트에서 한쪽 자식까지)
 *   3) while(v>0 || h>0) 루프: 수평(h)과 수직(v) 레벨을 번갈아가며 처리
 *      - h > v  (option 0): 수평 링크만 처리 (불균형 트리 — 수평이 더 많을 때)
 *      - h > 0 && v > 0 (option 1): 수평 + 수직 링크를 한 번에 처리 (균형 진행)
 *      - h == 0 (option 2): 수직 링크만 처리 (불균형 트리 — 수직이 더 많을 때)
 *   4) 각 링크마다 Wire 객체로 RC 지연·전력 누적, input_nand()로 노드 리피터 전력 누적
 *   5) 수직 링크를 만날 때마다(option 2 또는 search_tree) wire_bw *= 2 (팬아웃 증가)
 *
 * 불균형 트리 처리 근거: ndwl ≠ ndbl이면 수평/수직 레벨 수가 다르므로
 * excess_part = |log2(ndwl/2) - log2(ndbl/2)| 만큼 한 방향 링크를 먼저 소진한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드, AccelWattch 초기화 중 한 번만 실행.
 *
 * 호출 체인:
 *   Htree2::Htree2() → [in_htree()] → Wire(), input_nand()
 */
  void
Htree2::in_htree()
{
  //temp var
  double s1 = 0, s2 = 0, s3 = 0; // [한국어] 리피터 크기: s1=현재 링크, s2=다음 링크, s3=그 다음 링크
  double l_eff = 0;              // [한국어] 유효 배선 길이 (리피터 간격 또는 실제 링크 길이 중 작은 쪽)
  Wire *wtemp1 = 0, *wtemp2 = 0, *wtemp3 = 0; // [한국어] 임시 Wire 객체: 각 레벨 링크의 RC 지연/전력 계산용
  double len = 0, ht = 0;       // [한국어] 현재 처리 레벨의 수평(len)/수직(ht) 링크 길이
  int option = 0;                // [한국어] 처리 모드: 0=수평만, 1=수평+수직, 2=수직만

  int h = (int) _log2(ndwl/2); // horizontal nodes // [한국어] H-tree 수평 레벨 수 = log2(ndwl/2)
  int v = (int) _log2(ndbl/2); // vertical nodes   // [한국어] H-tree 수직 레벨 수 = log2(ndbl/2)
  double len_temp; // [한국어] 현재 처리 중인 수평 링크 길이 임시 저장
  double ht_temp;  // [한국어] 현재 처리 중인 수직 링크 길이 임시 저장

  if (uca_tree) // [한국어] UCA interbank 트리: mat_height는 bank 전체 높이로 해석
  {//Sheng: this computation do not consider the wires that route from edge to middle.
    // [한국어] ht_temp: 수직 방향 H-tree 절반 높이
    // = (뱅크 전체 높이 + 데이터/주소 버스 배선이 추가하는 pitch 보정) / 2
    ht_temp = (mat_height*ndbl/2 +/* since uca_tree models interbank tree, mat_height => bank height */
        ((add_bits + data_in_bits + data_out_bits + (search_data_in_bits + search_data_out_bits)) * g_tp.wire_outside_mat.pitch *
         2 * (1-pow(0.5,h))))/2;
    // [한국어] len_temp: 수평 방향 H-tree 절반 너비 — 유사한 구조로 계산
    len_temp = (mat_width*ndwl/2 +
        ((add_bits + data_in_bits + data_out_bits + (search_data_in_bits + search_data_out_bits)) * g_tp.wire_outside_mat.pitch *
         2 * (1-pow(0.5,v))))/2;
  }
  else // [한국어] 뱅크 내부 트리: ndwl/ndbl 관계에 따라 세 가지 케이스로 면적 계산
  {
    if (ndwl == ndbl) { // [한국어] 균형 트리: 수평/수직 레벨 수가 동일
      // [한국어] ht_temp: mat 배열 세로 + 어드레스/검색 버스 pitch 보정(레벨마다) + 데이터 버스 pitch(수평 레벨 수)
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits + (search_data_in_bits + search_data_out_bits))* (ndbl/2-1) * g_tp.wire_outside_mat.pitch) +
          ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * h)
          )/2;
      len_temp = (mat_width*ndwl/2 +
        ((add_bits + (search_data_in_bits + search_data_out_bits)) * (ndwl/2-1) * g_tp.wire_outside_mat.pitch) +
        ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * v))/2;
    }
    else if (ndwl > ndbl) { // [한국어] 수평 레벨이 더 많은 불균형 트리: excess_part 만큼 수평 초과
      double excess_part = (_log2(ndwl/2) - _log2(ndbl/2)); // [한국어] 수평 초과 레벨 수
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits + + (search_data_in_bits + search_data_out_bits)) * ((ndbl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          (data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch *
          (2*(1 - pow(0.5, h-v)) + pow(0.5, v-h) * v))/2; // [한국어] 불균형 보정 항
      len_temp = (mat_width*ndwl/2 +
        ((add_bits + (search_data_in_bits + search_data_out_bits))* (ndwl/2-1) * g_tp.wire_outside_mat.pitch) +
        ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * v))/2;
    }
    else { // [한국어] 수직 레벨이 더 많은 불균형 트리: excess_part 만큼 수직 초과
       double excess_part = (_log2(ndbl/2) - _log2(ndwl/2)); // [한국어] 수직 초과 레벨 수
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits + (search_data_in_bits + search_data_out_bits))* ((ndwl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * h)
          )/2;
      len_temp = (mat_width*ndwl/2 +
          ((add_bits + (search_data_in_bits + search_data_out_bits)) * ((ndwl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          (data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * (h + 2*(1-pow(0.5, v-h))))/2; // [한국어] 수직 초과분 보정 항
    }
  }

  area.h   = ht_temp * 2; // [한국어] H-tree 전체 높이 = 절반 높이 × 2 (대칭 구조)
  area.w   = len_temp * 2; // [한국어] H-tree 전체 너비 = 절반 너비 × 2
  delay = 0;               // [한국어] 누적 지연 초기화 — 이후 루프에서 링크·노드 지연을 더함
  power.readOp.dynamic = 0; // [한국어] 읽기 동적 전력 초기화
  power.readOp.leakage = 0; // [한국어] 누설 전력 초기화
  power.searchOp.dynamic = 0; // [한국어] 검색 동적 전력 초기화
  len = len_temp;           // [한국어] 루프 시작 수평 길이 = 루트에서 첫 분기까지의 절반 너비
  ht  = ht_temp/2;          // [한국어] 루프 시작 수직 길이 = 루트 수직 절반의 절반

  while (v > 0 || h > 0) // [한국어] 모든 트리 레벨을 소진할 때까지 반복
  {
    if (wtemp1) delete wtemp1; // [한국어] 이전 반복의 Wire 객체 해제 (메모리 누수 방지)
    if (wtemp2) delete wtemp2;
    if (wtemp3) delete wtemp3;

    if (h > v) // [한국어] option 0: 수평 레벨이 남아있고 수직보다 많음 — 수평 링크만 처리
    {
      //the iteration considers only one horizontal link
      wtemp1 = new Wire(wt, len); // hor // [한국어] 현재 수평 링크의 Wire 객체 생성
      wtemp2 = new Wire(wt, len/2);  // ver // [한국어] 다음 레벨 절반 길이의 Wire (노드 크기 계산용)
      len_temp = len;  // [한국어] 현재 링크 길이 저장 (input_nand 호출 시 리피터 크기 스케일링 기준)
      len /= 2;        // [한국어] 다음 레벨로 내려가면 링크 길이 절반
      wtemp3 = 0;      // [한국어] 세 번째 Wire 불필요 (단일 링크 처리)
      h--;             // [한국어] 수평 레벨 카운트 감소
      option = 0;      // [한국어] 처리 모드: 수평만
    }
    else if (v>0 && h>0) // [한국어] option 1: 수평·수직 모두 남아있음 — 두 링크를 한 번에 처리
    {
      //considers one horizontal link and one vertical link
      wtemp1 = new Wire(wt, len); // hor // [한국어] 수평 링크
      wtemp2 = new Wire(wt, ht);  // ver // [한국어] 수직 링크
      wtemp3 = new Wire(wt, len/2);  // next hor // [한국어] 다음 레벨 수평 링크 (노드 크기 계산용)
      len_temp = len;  // [한국어] 수평 링크 길이 저장
      ht_temp = ht;    // [한국어] 수직 링크 길이 저장
      len /= 2;        // [한국어] 다음 수평 레벨 길이 절반으로 갱신
      ht  /= 2;        // [한국어] 다음 수직 레벨 길이 절반으로 갱신
      v--;             // [한국어] 수직 레벨 카운트 감소
      h--;             // [한국어] 수평 레벨 카운트 감소
      option = 1;      // [한국어] 처리 모드: 수평+수직
    }
    else // [한국어] option 2: 수평 레벨 소진, 수직만 남음 — 수직 링크만 처리
    {
      // considers only one vertical link
      assert(h == 0); // [한국어] 이 분기에서는 h가 반드시 0이어야 함 (프로그래밍 불변 조건)
      wtemp1 = new Wire(wt, ht); // ver // [한국어] 수직 링크
      wtemp2 = new Wire(wt, ht/2);  // hor // [한국어] 다음 레벨 절반 수직 링크
      ht_temp = ht;  // [한국어] 수직 링크 길이 저장
      ht /= 2;       // [한국어] 다음 수직 레벨 길이 절반으로 갱신
      wtemp3 = 0;    // [한국어] 세 번째 Wire 불필요
      v--;           // [한국어] 수직 레벨 카운트 감소
      option = 2;    // [한국어] 처리 모드: 수직만
    }

    delay += wtemp1->delay; // [한국어] 첫 번째 링크(수평 또는 수직)의 RC 배선 지연 누적
    power.readOp.dynamic += wtemp1->power.readOp.dynamic; // [한국어] 첫 번째 링크 동적 전력 누적 (1비트 기준)
    power.searchOp.dynamic += wtemp1->power.readOp.dynamic*wire_bw; // [한국어] 검색 동적 전력: 현재 버스 폭 배수 적용
    power.readOp.leakage += wtemp1->power.readOp.leakage*wire_bw;    // [한국어] 배선 누설 전력: 버스 폭 배수 적용
    power.readOp.gate_leakage += wtemp1->power.readOp.gate_leakage*wire_bw; // [한국어] 게이트 누설 전력 누적

    if ((uca_tree == false && option == 2) || search_tree==true)
    { // [한국어] 수직 링크 처리 완료 후 (뱅크 내부 트리의 수직 분기 or CAM 검색 트리 전체):
      wire_bw*=2;  // wire bandwidth doubles only for vertical branches
      // [한국어] wire_bw 두 배: 수직 링크 아래에서는 mat가 두 배 많아 신호 개수도 두 배가 됨
    }

    if (uca_tree == false) // [한국어] 뱅크 내부 트리에서만 NAND 노드 전력 계산 (UCA 트리는 노드 없음)
    {
      if (len_temp > wtemp1->repeater_spacing)
      { // [한국어] 링크 길이가 리피터 간격보다 길면: 최적 크기의 리피터를 한 개 삽입
        s1 = wtemp1->repeater_size;         // [한국어] 최적 리피터 크기 사용
        l_eff = wtemp1->repeater_spacing;   // [한국어] 유효 배선 길이 = 리피터 간격 (한 구간)
      }
      else
      { // [한국어] 링크 길이가 리피터 간격보다 짧으면: 비율로 스케일 다운한 크기 사용
        s1 = (len_temp/wtemp1->repeater_spacing) * wtemp1->repeater_size; // [한국어] 짧은 링크에 비례한 리피터 크기
        l_eff = len_temp; // [한국어] 유효 배선 길이 = 실제 링크 길이 전체
      }

      if (ht_temp > wtemp2->repeater_spacing)
      { // [한국어] 다음 단 링크도 리피터 간격보다 길면: 최적 크기 사용
        s2 = wtemp2->repeater_size; // [한국어] 다음 레벨 리피터 크기
      }
      else
      { // [한국어] 짧으면: 비율로 스케일 다운 (len_temp 기준 — 불균형 트리 보정)
        s2 = (len_temp/wtemp2->repeater_spacing) * wtemp2->repeater_size;
      }
      // first level
      input_nand(s1, s2, l_eff); // [한국어] 첫 번째 링크 노드에 NAND 리피터 전력·지연 적용
    }


    if (option != 1) // [한국어] option 0(수평만) 또는 2(수직만)이면 두 번째 링크 처리 불필요
    {
      continue; // [한국어] 다음 루프 반복으로 이동
    }

    // second level // [한국어] option 1(수평+수직)일 때만 수직 링크(두 번째 링크) 처리
    delay += wtemp2->delay; // [한국어] 수직 링크 RC 지연 누적
    power.readOp.dynamic += wtemp2->power.readOp.dynamic; // [한국어] 수직 링크 동적 전력
    power.searchOp.dynamic += wtemp2->power.readOp.dynamic*wire_bw; // [한국어] 검색 동적 전력: 업데이트된 wire_bw 적용
    power.readOp.leakage += wtemp2->power.readOp.leakage*wire_bw;   // [한국어] 수직 링크 누설 전력
    power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw; // [한국어] 수직 링크 게이트 누설

    if (uca_tree) // [한국어] UCA interbank 트리: 수직 링크 누설만 추가 (NAND 노드 없음)
    {
      power.readOp.leakage += (wtemp2->power.readOp.leakage*wire_bw); // [한국어] 수직 배선 누설 중복 적산 (UCA 모드 보정)
      power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw;
    }
    else // [한국어] 뱅크 내부 트리: 수직 링크 이후 wire_bw 두 배 후 NAND 노드 전력 계산
    {
      power.readOp.leakage += (wtemp2->power.readOp.leakage*wire_bw);
      power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw;
      wire_bw*=2; // [한국어] 수직 링크 통과 후 wire_bw 두 배 — 아래쪽 팬아웃 증가 반영

      if (ht_temp > wtemp3->repeater_spacing)
      { // [한국어] 다음 수평 링크가 리피터 간격보다 길면: 최적 리피터 크기 사용
        s3    = wtemp3->repeater_size;
        l_eff = wtemp3->repeater_spacing; // [한국어] 다음 링크의 유효 배선 길이
      }
      else
      { // [한국어] 짧으면: 비율 스케일 다운 (ht_temp 기준)
        s3    = (len_temp/wtemp3->repeater_spacing) * wtemp3->repeater_size;
        l_eff = ht_temp; // [한국어] 수직 링크 길이를 유효 길이로 사용
      }

      input_nand(s2, s3, l_eff); // [한국어] 수직 링크 노드에 NAND 리피터 전력·지연 적용
    }
  }

  if (wtemp1) delete wtemp1; // [한국어] 루프 종료 후 마지막 Wire 객체 해제 (메모리 누수 방지)
  if (wtemp2) delete wtemp2;
  if (wtemp3) delete wtemp3;
}



/* a tristate buffer is used to handle fan-ins
 * The area of an unbalanced htree (rows != columns)
 * depends on how data is traversed.
 * In the following function, if ( no. of rows < no. of columns),
 * then data first traverse in excess hor. links until vertical
 * and horizontal nodes are same.
 * If no. of rows is bigger, then data traverse in
 * a hor. link followed by a ver. link in a repeated
 * fashion (similar to a balanced tree) until there are no
 * hor. links left. After this it goes through the remaining vertical
 * links.
 */
/*
 * [한국어]
 * Htree2::out_htree - 출력(리프→루트 수집) H-tree 지연·전력 계산
 *
 * @return: (void) — Component::delay, power, area에 계산 결과를 직접 저장
 *
 * 각 mat 리프의 데이터를 tristate 버퍼를 통해 루트로 수집하는 fanin H-tree를 모델링한다.
 * in_htree()와 동일한 트리 구조(불균형 처리 포함)를 사용하지만, 노드마다 NAND 대신
 * output_buffer()(NOT+NAND+NOR+출력 tr 4단 tristate)를 삽입하는 점이 다르다.
 * searchOp.dynamic 계산에 wire_bw 대신 init_wire_bw(루트 버스 폭)를 사용하는데,
 * 이는 출력 트리에서는 각 리프가 독립적으로 구동하므로 전체 버스 폭이 고정되기 때문이다.
 *
 * 알고리즘: in_htree()와 동일한 option 0/1/2 루프 구조 사용.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드, AccelWattch 초기화 중 한 번만 실행.
 *
 * 호출 체인:
 *   Htree2::Htree2() → [out_htree()] → Wire(), output_buffer()
 */
void Htree2::out_htree()
{
  //temp var
  double s1 = 0, s2 = 0, s3 = 0; // [한국어] tristate 버퍼 크기: s1=현재, s2=다음, s3=그 다음
  double l_eff = 0;               // [한국어] 유효 배선 길이
  Wire *wtemp1 = 0, *wtemp2 = 0, *wtemp3 = 0; // [한국어] 각 레벨 링크의 Wire 객체
  double len = 0, ht = 0;        // [한국어] 현재 레벨의 수평/수직 링크 길이
  int option = 0;                 // [한국어] 처리 모드: 0=수평, 1=수평+수직, 2=수직

  int h = (int) _log2(ndwl/2); // [한국어] 수평 레벨 수 = log2(ndwl/2)
  int v = (int) _log2(ndbl/2); // [한국어] 수직 레벨 수 = log2(ndbl/2)
  double len_temp; // [한국어] 현재 수평 링크 길이 임시 저장
  double ht_temp;  // [한국어] 현재 수직 링크 길이 임시 저장

  if (uca_tree) // [한국어] UCA interbank 트리: mat_height는 bank 전체 높이로 해석
  {
    // [한국어] ht_temp, len_temp 계산: in_htree()와 동일한 공식 (전체 버스 pitch 보정 포함)
    ht_temp = (mat_height*ndbl/2 +/* since uca_tree models interbank tree, mat_height => bank height */
        ((add_bits + data_in_bits + data_out_bits + (search_data_in_bits + search_data_out_bits)) * g_tp.wire_outside_mat.pitch *
         2 * (1-pow(0.5,h))))/2;
    len_temp = (mat_width*ndwl/2 +
        ((add_bits + data_in_bits + data_out_bits + (search_data_in_bits + search_data_out_bits)) * g_tp.wire_outside_mat.pitch *
         2 * (1-pow(0.5,v))))/2;
  }
  else // [한국어] 뱅크 내부 트리: ndwl/ndbl 비교로 균형/불균형 분기
  {
    if (ndwl == ndbl) { // [한국어] 균형 트리
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits+ (search_data_in_bits + search_data_out_bits)) * (ndbl/2-1) * g_tp.wire_outside_mat.pitch) +
          ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * h)
          )/2;
      len_temp = (mat_width*ndwl/2 +
        ((add_bits + (search_data_in_bits + search_data_out_bits)) * (ndwl/2-1) * g_tp.wire_outside_mat.pitch) +
        ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * v))/2;

    }
    else if (ndwl > ndbl) { // [한국어] 수평 레벨 초과 불균형
      double excess_part = (_log2(ndwl/2) - _log2(ndbl/2)); // [한국어] 초과 수평 레벨 수
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits + (search_data_in_bits + search_data_out_bits)) * ((ndbl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          (data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch *
          (2*(1 - pow(0.5, h-v)) + pow(0.5, v-h) * v))/2;
      len_temp = (mat_width*ndwl/2 +
        ((add_bits + (search_data_in_bits + search_data_out_bits))* (ndwl/2-1) * g_tp.wire_outside_mat.pitch) +
        ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * v))/2;
    }
    else { // [한국어] 수직 레벨 초과 불균형
      double excess_part = (_log2(ndbl/2) - _log2(ndwl/2)); // [한국어] 초과 수직 레벨 수
      ht_temp = ((mat_height*ndbl/2) +
          ((add_bits + (search_data_in_bits + search_data_out_bits))* ((ndwl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          ((data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * h)
          )/2;
      len_temp = (mat_width*ndwl/2 +
          ((add_bits + (search_data_in_bits + search_data_out_bits))* ((ndwl/2-1) + excess_part) * g_tp.wire_outside_mat.pitch) +
          (data_in_bits + data_out_bits) * g_tp.wire_outside_mat.pitch * (h + 2*(1-pow(0.5, v-h))))/2;
    }
  }
  area.h = ht_temp * 2; // [한국어] H-tree 전체 높이
  area.w = len_temp * 2; // [한국어] H-tree 전체 너비
  delay = 0;             // [한국어] 지연 초기화
  power.readOp.dynamic = 0;  // [한국어] 읽기 동적 전력 초기화
  power.readOp.leakage = 0;  // [한국어] 누설 전력 초기화
  power.readOp.gate_leakage = 0; // [한국어] 게이트 터널링 누설 초기화
  len = len_temp;        // [한국어] 루프 시작 수평 링크 길이
  ht = ht_temp/2;        // [한국어] 루프 시작 수직 링크 길이

  while (v > 0 || h > 0) // [한국어] 모든 트리 레벨을 소진할 때까지 반복
  { //finds delay/power of each link in the tree
    if (wtemp1) delete wtemp1; // [한국어] 이전 반복 Wire 해제
    if (wtemp2) delete wtemp2;
    if (wtemp3) delete wtemp3;

    if(h > v) { // [한국어] option 0: 수평 링크만 처리 (수평 레벨 초과)
      //the iteration considers only one horizontal link
      wtemp1 = new Wire(wt, len); // hor // [한국어] 현재 수평 링크
      wtemp2 = new Wire(wt, len/2);  // ver // [한국어] 다음 레벨 절반 길이 (버퍼 크기 계산용)
      len_temp = len;  // [한국어] 현재 링크 길이 저장
      len /= 2;        // [한국어] 다음 레벨 수평 링크 절반으로 갱신
      wtemp3 = 0;
      h--;
      option = 0;
    }
    else if (v>0 && h>0) { // [한국어] option 1: 수평+수직 링크 동시 처리
      //considers one horizontal link and one vertical link
      wtemp1 = new Wire(wt, len); // hor // [한국어] 수평 링크
      wtemp2 = new Wire(wt, ht);  // ver // [한국어] 수직 링크
      wtemp3 = new Wire(wt, len/2);  // next hor // [한국어] 다음 수평 링크 (버퍼 크기 계산용)
      len_temp = len;
      ht_temp = ht;
      len /= 2;
      ht /= 2;
      v--;
      h--;
      option = 1;
    }
    else { // [한국어] option 2: 수직 링크만 처리 (수평 레벨 소진)
      // considers only one vertical link
      assert(h == 0); // [한국어] 수평 레벨이 0임을 확인
      wtemp1 = new Wire(wt, ht); // hor // [한국어] 수직 링크
      wtemp2 = new Wire(wt, ht/2);  // ver // [한국어] 다음 레벨 절반 수직 링크
      ht_temp = ht;
      ht /= 2;
      wtemp3 = 0;
      v--;
      option = 2;
    }

    delay += wtemp1->delay; // [한국어] 첫 번째 링크 RC 지연 누적
    power.readOp.dynamic += wtemp1->power.readOp.dynamic; // [한국어] 읽기 동적 전력 누적 (1비트 기준)
    power.searchOp.dynamic += wtemp1->power.readOp.dynamic*init_wire_bw; // [한국어] 검색 동적 전력: init_wire_bw 고정 배수 (출력 트리는 루트 버스 폭 기준)
    power.readOp.leakage += wtemp1->power.readOp.leakage*wire_bw;         // [한국어] 배선 누설 전력: 현재 wire_bw 배수
    power.readOp.gate_leakage += wtemp1->power.readOp.gate_leakage*wire_bw; // [한국어] 게이트 누설 누적

    if ((uca_tree == false && option == 2) || search_tree==true)
    { // [한국어] 수직 링크 처리 완료 또는 CAM 검색 트리: wire_bw 두 배 (팬아웃 반영)
      wire_bw*=2;
    }

    if (uca_tree == false) // [한국어] 뱅크 내부 트리에서만 tristate 버퍼 노드 전력 계산
    {
      if (len_temp > wtemp1->repeater_spacing)
      { // [한국어] 링크가 리피터 간격보다 길면: 최적 크기 사용
        s1 = wtemp1->repeater_size;
        l_eff = wtemp1->repeater_spacing;
      }
      else
      { // [한국어] 짧으면: 비율로 스케일 다운
        s1 = (len_temp/wtemp1->repeater_spacing) * wtemp1->repeater_size;
        l_eff = len_temp;
      }
      if (ht_temp > wtemp2->repeater_spacing)
      { // [한국어] 다음 링크 크기: 최적 사용
        s2 = wtemp2->repeater_size;
      }
      else
      { // [한국어] 다음 링크 크기: 비율 스케일 다운
        s2 = (len_temp/wtemp2->repeater_spacing) * wtemp2->repeater_size;
      }
      // first level
      output_buffer(s1, s2, l_eff); // [한국어] 첫 번째 링크 노드에 tristate 버퍼 전력·지연 적용
    }


    if (option != 1) // [한국어] option 0 또는 2이면 두 번째 링크 처리 불필요
    {
      continue;
    }

    // second level // [한국어] option 1: 수직 링크(두 번째 링크) 처리
    delay += wtemp2->delay; // [한국어] 수직 링크 지연 누적
    power.readOp.dynamic += wtemp2->power.readOp.dynamic; // [한국어] 수직 링크 동적 전력
    power.searchOp.dynamic += wtemp2->power.readOp.dynamic*init_wire_bw; // [한국어] 검색 동적 전력: 출력 트리는 init_wire_bw 고정
    power.readOp.leakage += wtemp2->power.readOp.leakage*wire_bw;        // [한국어] 수직 링크 누설 전력
    power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw;

    if (uca_tree) // [한국어] UCA 트리: 수직 링크 누설만 추가 (tristate 노드 없음)
    {
      power.readOp.leakage += (wtemp2->power.readOp.leakage*wire_bw);
      power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw;
    }
    else // [한국어] 뱅크 내부 트리: 수직 링크 이후 wire_bw 두 배 + tristate 버퍼 노드 처리
    {
      power.readOp.leakage += (wtemp2->power.readOp.leakage*wire_bw);
      power.readOp.gate_leakage += wtemp2->power.readOp.gate_leakage*wire_bw;
      wire_bw*=2; // [한국어] 수직 링크 통과 후 wire_bw 두 배 갱신

      if (ht_temp > wtemp3->repeater_spacing)
      { // [한국어] 다음 수평 링크 크기: 최적 사용
        s3 = wtemp3->repeater_size;
        l_eff = wtemp3->repeater_spacing;
      }
      else
      { // [한국어] 다음 수평 링크 크기: 비율 스케일 다운 (ht_temp 기준)
        s3 = (len_temp/wtemp3->repeater_spacing) * wtemp3->repeater_size;
        l_eff = ht_temp;
      }

      output_buffer(s2, s3, l_eff); // [한국어] 수직 링크 노드에 tristate 버퍼 전력·지연 적용
    }
  }

  if (wtemp1) delete wtemp1; // [한국어] 루프 종료 후 Wire 객체 해제
  if (wtemp2) delete wtemp2;
  if (wtemp3) delete wtemp3;
}

