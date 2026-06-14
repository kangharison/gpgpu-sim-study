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
 * [한국어 설명] UCA(Uniform Cache Access) 캐시 타이밍/전력 모델 헤더 (uca.h)
 *
 * === 파일의 역할 ===
 * 단일 뱅크 또는 균등 접근 지연을 갖는 멀티뱅크 캐시(UCA)를 모델링한다.
 * 생성자에서 Bank와 H-tree 배선을 인스턴스화한 뒤, compute_delays()와 compute_power_energy()를
 * 호출해 접근 지연(access_time)·사이클 타임(cycle_time)·전력(power)·면적(area)을 모두 산출한다.
 * CACTI의 핵심 출력 객체이며, Ucache.cc의 calculate_time()이 UCA를 생성해 mem_array에 결과를 복사한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * solve() → calculate_time() → new UCA(dyn_p) → [이 모듈] → mem_array에 결과 저장
 * UCA는 뱅크(Bank) 위에 전역 H-tree 라우팅(Htree2)을 얹은 구조를 시뮬레이션한다.
 * AccelWattch는 cacti_interface()를 통해 최종 선택된 UCA 구성의 power/area 값을 GPU 전력 모델에 반영한다.
 * 호스트 유저스페이스, 멀티 스레드(pthread) 환경에서 calc_time_mt_wrapper 내에서 생성.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Bank(뱅크 구조), DynamicParameter(설계 공간 파라미터), Htree2(H-tree 배선 모델),
 *       parameter.h(g_ip, g_tp), component.h(powerDef, Component 기반 클래스)
 * 소비: calculate_time()이 UCA를 heap 할당 후 결과 복사, 즉시 delete
 * 공유: g_ip->nbanks(뱅크 수), g_ip->burst_len(버스트 길이), g_ip->rpters_in_htree
 *
 * === 주요 함수/구조체 요약 ===
 * UCA(dyn_p)              : 생성자 — Bank/Htree2 생성, 지연·전력 계산 완료
 * compute_delays(inrise)  : H-tree·디코더·비트라인·SA 경로별 지연 계산, access_time 확정
 * compute_power_energy()  : 읽기·쓰기·누설 전력 분리 계산, DRAM 리프레시 전력 포함
 * access_time             : 총 캐시 접근 지연 [s] — CACTI 핵심 출력값
 * cycle_time              : 사이클 타임 [s] — 최대 처리율 결정
 * power                   : powerDef — readOp/writeOp/searchOp 동적·누설 전력
 */

#ifndef __UCA_H__
#define __UCA_H__

#include "area.h"
#include "bank.h"
#include "component.h"
#include "parameter.h"
#include "htree2.h"


class UCA : public Component
{
  public:
    UCA(const DynamicParameter & dyn_p);
    ~UCA();
    double compute_delays(double inrisetime);  // returns outrisetime
    void   compute_power_energy();

    DynamicParameter dp;
    /* [한국어] 이 UCA 인스턴스의 설계 파라미터 복사본.
     * 설정자: 생성자 초기화 리스트에서 dyn_p로 초기화.
     * 읽는 자: compute_delays/compute_power_energy 전반에서 num_act_mats_hor_dir, num_subarrays 등 접근.
     * 동기화: 생성 후 불변. */

    Bank   bank;
    /* [한국어] 단일 뱅크 구조 — Mat 배열과 내부 H-tree를 포함하는 최하위 물리 캐시 단위.
     * 설정자: 생성자에서 Bank(dp)로 초기화; 내부에서 Mat/Subarray/Htree2를 재귀 생성.
     * 읽는 자: compute_delays()에서 bank.mat.* 지연값, compute_power_energy()에서 bank.power 사용.
     * 동기화: 단일 스레드 내에서 compute_delays 후 compute_power_energy 순서로 사용. */

    Htree2   * htree_in_add;
    /* [한국어] 주소 입력 H-tree — 캐시 컨트롤러에서 뱅크까지 주소를 분배하는 배선 트리.
     * 설정자: 생성자에서 new Htree2(Add_htree) 로 생성.
     * 읽는 자: compute_delays()에서 delay_array_to_mat = htree_in_add->delay + bank.htree_in_add->delay.
     * 값 범위: non-null; 소멸자에서 delete.
     * 동기화: 단일 스레드. */

    Htree2   * htree_in_data;
    /* [한국어] 데이터 입력(쓰기) H-tree — 쓰기 데이터를 뱅크에 분배하는 배선 트리.
     * 설정자: 생성자에서 new Htree2(Data_in_htree)로 생성.
     * 읽는 자: compute_power_energy()에서 쓰기 경로 전력 계산; area.w = htree_in_data->area.w.
     * 동기화: 단일 스레드. */

    Htree2   * htree_out_data;
    /* [한국어] 데이터 출력(읽기) H-tree — 뱅크에서 캐시 컨트롤러로 읽기 데이터를 전달하는 트리.
     * 설정자: 생성자에서 new Htree2(Data_out_htree)로 생성.
     * 읽는 자: delay_from_subarray_out_drv_to_out 계산에 htree_out_data->delay 포함.
     * 동기화: 단일 스레드. */

    Htree2   * htree_in_search;
    /* [한국어] FA/CAM 캐시 전용 검색 입력 H-tree — fully_assoc 또는 pure_cam일 때만 생성.
     * 설정자: 생성자에서 조건부 new Htree2(Data_in_htree, search용)로 생성.
     * 읽는 자: compute_power_energy()에서 searchOp 전력 계산.
     * 값 범위: FA/CAM이 아니면 미초기화 — 접근 전 dp.fully_assoc || dp.pure_cam 확인 필요.
     * 동기화: 단일 스레드. */

    Htree2   * htree_out_search;
    /* [한국어] FA/CAM 캐시 전용 검색 결과 출력 H-tree.
     * 설정자: 생성자에서 조건부 생성.
     * 읽는 자: compute_power_energy()에서 searchOp 누설 전력 계산.
     * 동기화: 단일 스레드. */

    powerDef power_routing_to_bank;
    /* [한국어] 뱅크로의 라우팅(전역 H-tree) 전력만을 분리한 결과.
     * 설정자: compute_power_energy()에서 htree_in_add + htree_out_data 동적/누설 합산.
     * 읽는 자: 쓰기/읽기 에너지 계산 시 htree 경로 전력을 별도로 조정하기 위해 사용.
     * 값 범위: readOp(읽기 경로), writeOp(쓰기 경로) 동적·누설·gate_leakage 각각.
     * 동기화: compute_power_energy() 내에서만 설정. */

    uint32_t nbanks;
    /* [한국어] 이 UCA에서 관리하는 총 뱅크 수 = g_ip->nbanks.
     * 설정자: 생성자 초기화 리스트에서 g_ip->nbanks로 초기화.
     * 읽는 자: 뱅크 분산 방향(수직/수평 뱅크 수) 계산에 사용.
     * 값 범위: 1 이상의 2의 거듭제곱.
     * 동기화: 불변. */

    int   num_addr_b_bank;
    /* [한국어] 뱅크당 주소 비트 수 (포트별 곱셈 적용).
     * 설정자: (number_addr_bits_mat + number_subbanks_decode) × (RWP+ERP+EWP).
     * 읽는 자: Htree2 생성 시 num_addr_b 인자로 전달.
     * 동기화: 생성 시 한 번 계산. */

    int   num_di_b_bank;
    /* [한국어] 뱅크당 데이터 입력 비트 수 (RWP + EWP 포트 합산). */
    int   num_do_b_bank;
    /* [한국어] 뱅크당 데이터 출력 비트 수 (RWP + ERP 포트 합산). */
    int   num_si_b_bank;
    /* [한국어] FA/CAM 검색 입력 비트 수 (SCHP 포트 합산). */
    int   num_so_b_bank;
    /* [한국어] FA/CAM 검색 출력 비트 수 (SCHP 포트 합산). */

    int   RWP, ERP, EWP,SCHP;
    /* [한국어] 포트 수: RWP(읽기/쓰기 공용), ERP(전용 읽기), EWP(전용 쓰기), SCHP(검색).
     * 설정자: dp.use_inp_params 여부에 따라 dp 또는 g_ip에서 읽음.
     * 읽는 자: num_*_b_bank 계산에 사용.
     * 동기화: 불변. */

    double area_all_dataramcells;
    /* [한국어] 전체 데이터 RAM 셀 면적 [um²] — 면적 효율(area efficiency) 계산에 사용.
     * 설정자: bank.mat.subarray.get_total_cell_area() × num_subarrays × g_ip->nbanks.
     * 읽는 자: calculate_time()에서 ptr_array->area_ram_cells/area_efficiency 계산.
     * 동기화: 생성 시 한 번 계산. */

    double dyn_read_energy_from_closed_page;
    /* [한국어] 닫힌 페이지(precharge 후 첫 접근) 읽기 동적 에너지 [J] — DRAM 모델용.
     * 행 활성화(activate) + 열 읽기(CAS) 에너지를 모두 포함. */
    double dyn_read_energy_from_open_page;
    /* [한국어] 열린 페이지(이미 활성화된 행) 읽기 동적 에너지 [J] — 행 디코더/SA 에너지 제외. */
    double dyn_read_energy_remaining_words_in_burst;
    /* [한국어] 버스트 읽기에서 첫 단어 이후 나머지 단어들의 추가 에너지 [J].
     * = (burst_len / int_prefetch_w - 1) × SA mux + 출력 드라이버 + H-tree 에너지. */

    double refresh_power;  // only for DRAM
    /* [한국어] DRAM 리프레시 전력 [W] — 비DRAM이면 0.
     * 설정자: compute_power_energy()에서 r_predec + row_dec 에너지 / dram_refresh_period.
     * 읽는 자: 총 누설 전력에 더해져 DRAM 정적 소비 전력에 기여.
     * 동기화: compute_power_energy() 내에서 설정. */

    double activate_energy;
    /* [한국어] DRAM 행 활성화(ACTIVATE) 에너지 [J] — htree_in_add + r_predec + row_dec + SA. */
    double read_energy;
    /* [한국어] DRAM 읽기(READ) 에너지 [J] — CAS 단계의 SA mux + 출력 드라이버 × burst_len. */
    double write_energy;
    /* [한국어] DRAM 쓰기(WRITE) 에너지 [J] — htree_in_data + SA mux × burst_len. */
    double precharge_energy;
    /* [한국어] DRAM 프리차지(PRECHARGE) 에너지 [J] — bitline + precharge 드라이버. */

    double leak_power_subbank_closed_page;
    /* [한국어] 닫힌 페이지 상태에서 서브뱅크 누설 전력 [W] — 프리디코더+디코더+SA(비활성) 합산. */
    double leak_power_subbank_open_page;
    /* [한국어] 열린 페이지 상태에서 서브뱅크 누설 전력 [W] — 닫힌 페이지 + 활성 SA 누설. */
    double leak_power_request_and_reply_networks;
    /* [한국어] 요청(주소/데이터 입력)·응답(데이터 출력) H-tree 네트워크의 누설 전력 [W]. */

    double delay_array_to_sa_mux_lev_1_decoder;
    /* [한국어] 배열 입력(H-tree) → SA mux 1단 디코더까지의 경로 지연 [s]. */
    double delay_array_to_sa_mux_lev_2_decoder;
    /* [한국어] 배열 입력(H-tree) → SA mux 2단 디코더까지의 경로 지연 [s]. */
    double delay_before_subarray_output_driver;
    /* [한국어] 행 디코더→비트라인→SA 경로와 열 mux 경로 중 최대 지연 [s] — SA 출력 직전. */
    double delay_from_subarray_out_drv_to_out;
    /* [한국어] 서브어레이 출력 드라이버 → (내부 H-tree + 전역 H-tree) → 최종 출력까지 지연 [s]. */

    double access_time;
    /* [한국어] 총 캐시 접근 지연 [s] — CACTI의 핵심 출력 파라미터.
     * = delay_before_subarray_output_driver + delay_from_subarray_out_drv_to_out (비FA 경우).
     * 설정자: compute_delays()에서 계산. 읽는 자: calculate_time()이 ptr_array->access_time으로 복사.
     * 동기화: compute_delays() → compute_power_energy() 순서 보장. */

    double precharge_delay;
    /* [한국어] 프리차지 지연 [s] — DRAM일 때 htree + writeback + wl_reset + bl_restore 합산.
     * 비DRAM이면 0. cycle_time에 더해져 DRAM 사이클 타임을 결정. */

    double multisubbank_interleave_cycle_time;
    /* [한국어] 멀티 서브뱅크 인터리빙 사이클 타임 [s] — 요청·응답 네트워크 지연 중 최대값.
     * DRAM 메인 메모리일 때는 htree_in_add->delay만으로 결정. */
};

#endif

