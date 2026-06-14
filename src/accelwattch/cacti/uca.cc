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
 * [한국어 설명] UCA(Uniform Cache Array) 캐시 타이밍·전력·면적 모델 구현 (uca.cc)
 *
 * === 파일의 역할 ===
 * UCA 캐시(단일 뱅크 또는 균일 다중 뱅크 구조) 하나의 접근 지연, 사이클 타임,
 * 동적/누설 전력을 완전히 계산한다. UCA 생성자가 Bank → Mat → Subarray 계층을 통해
 * 물리적 타이밍을 계산하고, H-tree 네트워크 지연·전력을 추가해 최종 캐시 성능지표를 도출한다.
 * DRAM 리프레시 전력, FA/CAM 서치 경로, 버스트 에너지 계산도 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * calculate_time() → UCA(dp) → [이 파일] → Bank → Mat → Subarray
 * UCA 생성자가 완료되면 access_time, cycle_time, power 등이 설정되고,
 * calculate_time()이 이 값을 mem_array 결과 구조체에 복사해 solve()에 반환한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 pthread 스레드 내에서 동작.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Bank(bank.h), Htree2(htree2.h), DynamicParameter, cacti_interface.h(g_ip)
 * 소비: calculate_time()이 UCA를 생성해 mem_array 결과 저장 후 삭제
 * 공유: g_ip(전역 캐시 입력 파라미터), g_tp(전역 기술 파라미터)
 *
 * === 주요 함수/구조체 요약 ===
 * UCA()              : 생성자 — H-tree 생성, 면적·지연·전력 전체 계산
 * ~UCA()             : 소멸자 — H-tree 포인터 delete
 * compute_delays()   : Bank/Mat/H-tree 지연을 합산해 access_time/cycle_time 설정
 * compute_power_energy(): 동적/누설/리프레시/버스트 전력 계산 및 합산
 */

#include <iostream>
#include <math.h>

#include "uca.h"


/*
 * [한국어]
 * UCA::UCA - UCA 캐시 생성자 — 전체 계층 계산 진입점
 *
 * @dyn_p: DynamicParameter — Ndwl/Ndbl/Nspd/캐시 크기 등 한 조합의 설계 파라미터
 * @return: (생성자)
 *
 * 1) Bank(dp)를 생성해 mat 계층까지 타이밍 사전 계산
 * 2) nbanks 수에 따라 뱅크 배치(num_banks_ver_dir × num_banks_hor_dir) 결정
 * 3) 포트 수에 따라 H-tree (htree_in_add/data/out_data, FA이면 search도) 생성
 * 4) 전체 면적 = H-tree 면적 (가장 외부 트리가 전체 캐시 경계를 결정)
 * 5) compute_delays(0) → compute_power_energy() 순서로 호출
 * 호출 체인: calculate_time() → [이 함수] → compute_delays() → compute_power_energy()
 */
UCA::UCA(const DynamicParameter & dyn_p)
 :dp(dyn_p),
  bank(dp),                    // [한국어] Bank 생성자 — Mat/Subarray 계층까지 면적·기본 파라미터 계산
  nbanks(g_ip->nbanks),       // [한국어] 이 UCA가 포함하는 전체 뱅크 수
  refresh_power(0)             // [한국어] DRAM 리프레시 전력 — compute_power_energy()에서 계산
{
  int num_banks_ver_dir = 1 << ((bank.area.h > bank.area.w) ? _log2(nbanks)/2 : (_log2(nbanks) - _log2(nbanks)/2));
  // [한국어] 수직 방향 뱅크 수: 뱅크가 세로로 긴 경우 log2(nbanks)/2, 그 외 나머지로 결정
  int num_banks_hor_dir = nbanks/num_banks_ver_dir;
  // [한국어] 수평 방향 뱅크 수 = 전체 뱅크 수 / 수직 뱅크 수

  if (dp.use_inp_params)
  // [한국어] 외부 입력 파라미터(dp)에서 포트 수를 직접 사용 — 라우터 버퍼 모델링 시
  {
	  RWP  = dp.num_rw_ports;    // [한국어] 읽기/쓰기 공용 포트 수
	  ERP  = dp.num_rd_ports;    // [한국어] 읽기 전용 포트 수
	  EWP  = dp.num_wr_ports;    // [한국어] 쓰기 전용 포트 수
	  SCHP = dp.num_search_ports; // [한국어] 서치 포트 수 (CAM/FA 전용)
  }
  else
  // [한국어] 전역 입력(g_ip)에서 포트 수 가져오기 — cacti_interface() 일반 호출 시
  {
	  RWP  = g_ip->num_rw_ports;
	  ERP  = g_ip->num_rd_ports;
	  EWP  = g_ip->num_wr_ports;
	  SCHP = g_ip->num_search_ports;
  }

  num_addr_b_bank = (dp.number_addr_bits_mat + dp.number_subbanks_decode)*(RWP+ERP+EWP);
  // [한국어] 뱅크 레벨 주소 비트 수 = (mat 주소 + 서브뱅크 디코드 비트) × 총 읽기+쓰기 포트 수
  num_di_b_bank   = dp.num_di_b_bank_per_port * (RWP + EWP);
  // [한국어] 뱅크 데이터 입력 비트 = 포트당 입력 × (RW+W 포트 수)
  num_do_b_bank   = dp.num_do_b_bank_per_port * (RWP + ERP);
  // [한국어] 뱅크 데이터 출력 비트 = 포트당 출력 × (RW+R 포트 수)
  num_si_b_bank   = dp.num_si_b_bank_per_port * SCHP;
  // [한국어] 서치 입력 비트 (CAM/FA 전용)
  num_so_b_bank   = dp.num_so_b_bank_per_port * SCHP;
  // [한국어] 서치 출력 비트 (CAM/FA 전용)

  if (!dp.fully_assoc && !dp.pure_cam)
  // [한국어] 일반 캐시(set-associative 또는 direct-mapped): 서치 H-tree 불필요
  {

	  if (g_ip->fast_access && dp.is_tag == false)
	  // [한국어] fast_access 모드: 데이터 배열이 연관도(assoc) 배만큼 출력 비트 확장
	  {
		  num_do_b_bank *= g_ip->data_assoc; // [한국어] fast_access: 한 번에 모든 way 데이터를 출력
	  }

	  htree_in_add   = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank,0, num_do_b_bank,0,num_banks_ver_dir*2, num_banks_hor_dir*2, Add_htree, true);
	  // [한국어] 주소 입력 H-tree: 상위 캐시 레벨에서 뱅크까지 주소 배포
	  htree_in_data  = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank, 0, num_do_b_bank, 0, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_in_htree, true);
	  // [한국어] 데이터 입력 H-tree: 쓰기 데이터를 뱅크까지 배포
	  htree_out_data = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank, 0, num_do_b_bank, 0, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_out_htree, true);
	  // [한국어] 데이터 출력 H-tree: 읽기 데이터를 뱅크에서 상위로 전달
  }

  else
  // [한국어] FA 또는 순수 CAM: 서치 입출력 H-tree 추가 생성
  {

	  htree_in_add   = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank, num_si_b_bank, num_do_b_bank, num_so_b_bank, num_banks_ver_dir*2, num_banks_hor_dir*2, Add_htree, true);
	  // [한국어] 주소 입력 H-tree (서치 비트 포함)
	  htree_in_data  = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank,num_si_b_bank, num_do_b_bank, num_so_b_bank, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_in_htree, true);
	  // [한국어] 데이터 입력 H-tree (서치 비트 포함)
	  htree_out_data = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank,num_si_b_bank, num_do_b_bank, num_so_b_bank, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_out_htree, true);
	  // [한국어] 데이터 출력 H-tree (서치 비트 포함)
	  htree_in_search  = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank,num_si_b_bank, num_do_b_bank, num_so_b_bank, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_in_htree, true);
	  // [한국어] 서치 입력 H-tree — CAM 태그를 각 서브어레이 서치라인으로 브로드캐스트
	  htree_out_search = new Htree2(g_ip->wt, bank.area.w, bank.area.h,
			  num_addr_b_bank, num_di_b_bank,num_si_b_bank, num_do_b_bank, num_so_b_bank, num_banks_ver_dir*2, num_banks_hor_dir*2, Data_out_htree, true);
	  // [한국어] 서치 출력 H-tree — 매치 결과를 상위로 전달
  }

  area.w = htree_in_data->area.w; // [한국어] UCA 전체 너비 = 가장 외부 H-tree 너비
  area.h = htree_in_data->area.h; // [한국어] UCA 전체 높이 = 가장 외부 H-tree 높이

  area_all_dataramcells = bank.mat.subarray.get_total_cell_area() * dp.num_subarrays * g_ip->nbanks;
  // [한국어] 전체 SRAM 셀 면적 = 서브어레이 셀 면적 × 서브어레이 수 × 뱅크 수 (면적 효율 계산용)
//  cout<<"area cell"<<area_all_dataramcells<<endl;
//  cout<<area.get_area()<<endl;
  // delay calculation
  double inrisetime = 0.0; // [한국어] 입력 라이즈 타임 초기값 0 (이상적 스텝 입력 가정)
  compute_delays(inrisetime); // [한국어] 모든 경로의 지연을 계산해 access_time/cycle_time 설정
  compute_power_energy();     // [한국어] 동적/누설/리프레시 전력 계산
}



// [한국어] 소멸자 — H-tree 포인터 3개 해제 (FA/CAM이면 htree_in_search/out_search 누락 주의)
UCA::~UCA()
{
  delete htree_in_add;    // [한국어] 주소 입력 H-tree 해제
  delete htree_in_data;   // [한국어] 데이터 입력 H-tree 해제
  delete htree_out_data;  // [한국어] 데이터 출력 H-tree 해제
}



/*
 * [한국어]
 * UCA::compute_delays - 캐시 전체 접근 지연 및 사이클 타임 계산
 *
 * @inrisetime: 입력 신호 라이즈 타임 [s] (통상 0.0 — 이상적 스텝 입력)
 * @return    : 출력 라이즈 타임 [s] — 다음 단계에 전달 가능하나 현재 미사용
 *
 * 지연을 여러 경로로 분해하고 MAX를 취해 임계 경로를 결정한다:
 *   - row_path: 주소→mat→행디코더→비트라인→SA
 *   - col_path: 주소→mat→비트MUX 예비디코더→비트MUX→SA
 *   - sa_mux_lev_1/2_path: SA MUX 예비디코더 경로
 * DRAM(is_main_mem): tRCD(행 활성화) + CAS 지연으로 access_time 계산
 * FA: H-tree 입력 + 비트라인 + 매치라인 지연으로 access_time 계산
 * cycle_time: 프리차지·리셋 포함 랜덤 사이클 타임 (랜덤 접근 처리 간격)
 * 호출 체인: UCA() → [이 함수] → bank.compute_delays() → Mat.compute_delays()
 */
double UCA::compute_delays(double inrisetime)
{
  double outrisetime = bank.compute_delays(inrisetime);
  // [한국어] Bank 계층 지연 계산 — Mat/Subarray/H-tree 내부 타이밍 설정

  double delay_array_to_mat = htree_in_add->delay + bank.htree_in_add->delay;
  // [한국어] 외부 H-tree + 뱅크 내부 H-tree 주소 전달 지연 합산
  double max_delay_before_row_decoder = delay_array_to_mat + bank.mat.r_predec->delay;
  // [한국어] 행 디코더 앞까지의 최대 지연 = H-tree 지연 + 행 예비디코더 지연
  delay_array_to_sa_mux_lev_1_decoder = delay_array_to_mat +
    bank.mat.sa_mux_lev_1_predec->delay +
    bank.mat.sa_mux_lev_1_dec->delay;
  // [한국어] SA MUX 레벨1 디코더까지의 지연 = H-tree + SA MUX 예비디코더 + 디코더
  delay_array_to_sa_mux_lev_2_decoder = delay_array_to_mat +
    bank.mat.sa_mux_lev_2_predec->delay +
    bank.mat.sa_mux_lev_2_dec->delay;
  // [한국어] SA MUX 레벨2 디코더까지의 지연
  double delay_inside_mat = bank.mat.row_dec->delay + bank.mat.delay_bitline + bank.mat.delay_sa;
  // [한국어] mat 내부 지연 = 행디코더 + 비트라인 방전 + 센스앰프 감지

  delay_before_subarray_output_driver =
    MAX(MAX(max_delay_before_row_decoder + delay_inside_mat,  // row_path
            delay_array_to_mat + bank.mat.b_mux_predec->delay + bank.mat.bit_mux_dec->delay + bank.mat.delay_sa),  // col_path
        MAX(delay_array_to_sa_mux_lev_1_decoder,    // sa_mux_lev_1_path
            delay_array_to_sa_mux_lev_2_decoder));  // sa_mux_lev_2_path
  // [한국어] 서브어레이 출력 드라이버 이전까지의 임계 경로 지연 = 4가지 경로의 MAX
  delay_from_subarray_out_drv_to_out = bank.mat.delay_subarray_out_drv_htree +
                                       bank.htree_out_data->delay + htree_out_data->delay;
  // [한국어] 서브어레이 출력 드라이버 → 뱅크 출력 H-tree → 외부 H-tree 지연 합산
  access_time                        = bank.mat.delay_comparator;
  // [한국어] 비교기 지연 초기값 (일반 캐시는 이후 덮어씀)

  double ram_delay_inside_mat;
  if (dp.fully_assoc)
  // [한국어] FA 캐시: CAM 태그 서치 + RAM 데이터 읽기 경로를 순차 계산
  {
    //delay of FA contains both CAM tag and RAM data
    { //delay of CAM
      ram_delay_inside_mat = bank.mat.delay_bitline + bank.mat.delay_matchchline;
      // [한국어] ram_delay_inside_mat = FA내 비트라인 + 매치라인 지연
      access_time = htree_in_add->delay + bank.htree_in_add->delay;
      // [한국어] 서치 address H-tree 지연 (CAM에 태그 브로드캐스트)
      //delay of fully-associative data array
      access_time += ram_delay_inside_mat + delay_from_subarray_out_drv_to_out;
      // [한국어] FA access_time = H-tree입력 + (비트라인+매치라인) + 출력드라이버→출력
    }
  }
  else
  // [한국어] 일반 set-associative/direct-mapped: 임계 경로 지연으로 access_time 결정
  {
    access_time = delay_before_subarray_output_driver + delay_from_subarray_out_drv_to_out; //data_acc_path
    // [한국어] access_time = 임계 경로 지연 + 출력 H-tree 지연
  }

  if (dp.is_main_mem)
  // [한국어] 메인 메모리(DRAM): tRCD + CAS 지연 모델로 access_time 재계산
  {
    double t_rcd       = max_delay_before_row_decoder + delay_inside_mat;
    // [한국어] tRCD(Row Cycle Delay): 행 활성화 지연 = H-tree + 행디코더 + 비트라인 + SA
    double cas_latency = MAX(delay_array_to_sa_mux_lev_1_decoder, delay_array_to_sa_mux_lev_2_decoder) +
                         delay_from_subarray_out_drv_to_out;
    // [한국어] CAS 지연: SA MUX 디코더 지연 + 출력 H-tree 지연
    access_time = t_rcd + cas_latency; // [한국어] DRAM access_time = tRCD + CAS
  }

  double temp;

  if (!dp.fully_assoc)
  // [한국어] 일반 캐시: 프리차지·워드라인 리셋 포함 사이클 타임 계산
  {
    temp = delay_inside_mat + bank.mat.delay_wl_reset + bank.mat.delay_bl_restore;//TODO: Sheng: revisit
    // [한국어] temp = mat 내부 지연 + 워드라인 리셋 + 비트라인 복원 지연
   if (dp.is_dram)
    {
      temp += bank.mat.delay_writeback;  // temp stores random cycle time
      // [한국어] DRAM: 쓰기 백 지연 추가 — 행을 닫기 전에 데이터 복원 필요
    }


  temp = MAX(temp, bank.mat.r_predec->delay);            // [한국어] 행 예비디코더 지연과 비교
  temp = MAX(temp, bank.mat.b_mux_predec->delay);        // [한국어] 비트 MUX 예비디코더 지연과 비교
  temp = MAX(temp, bank.mat.sa_mux_lev_1_predec->delay); // [한국어] SA MUX lv1 예비디코더와 비교
  temp = MAX(temp, bank.mat.sa_mux_lev_2_predec->delay); // [한국어] SA MUX lv2 예비디코더와 비교
  }
  else
  // [한국어] FA/CAM: 서치라인 복원·매치라인 리셋 포함 사이클 타임 계산
   {
	  ram_delay_inside_mat = bank.mat.delay_bitline + bank.mat.delay_matchchline;
	  temp = ram_delay_inside_mat + bank.mat.delay_cam_sl_restore + bank.mat.delay_cam_ml_reset + bank.mat.delay_bl_restore
	         + bank.mat.delay_hit_miss_reset + bank.mat.delay_wl_reset;
	  // [한국어] temp = 비트라인+매치라인 + CAM 서치라인 복원 + 매치라인 리셋 + 비트라인 복원 + 히트/미스 리셋 + 워드라인 리셋

	  temp = MAX(temp, bank.mat.b_mux_predec->delay);//TODO: Sheng revisit whether distinguish cam and ram bitline etc.
	  temp = MAX(temp, bank.mat.sa_mux_lev_1_predec->delay); // [한국어] SA MUX lv1 예비디코더와 비교
	  temp = MAX(temp, bank.mat.sa_mux_lev_2_predec->delay); // [한국어] SA MUX lv2 예비디코더와 비교
   }

  // The following is true only if the input parameter "repeaters_in_htree" is set to false --Nav
  if (g_ip->rpters_in_htree == false)
  // [한국어] H-tree에 반복기를 삽입하지 않는 경우: 비파이프라인 링크 지연을 사이클 타임에 포함
  {
    temp = MAX(temp, bank.htree_in_add->max_unpipelined_link_delay);
    // [한국어] 비파이프라인 H-tree 링크 최대 지연과 비교
  }
  cycle_time = temp; // [한국어] cycle_time = 가장 긴 경로의 복원·리셋 시간

  double delay_req_network = max_delay_before_row_decoder;
  // [한국어] 요청 네트워크 지연 = 행 디코더 이전까지 지연 (H-tree + 예비디코더)
  double delay_rep_network = delay_from_subarray_out_drv_to_out;
  // [한국어] 응답 네트워크 지연 = 서브어레이 출력 → 최종 출력까지
  multisubbank_interleave_cycle_time = MAX(delay_req_network, delay_rep_network);
  // [한국어] 서브뱅크 인터리브 사이클 타임 = 요청/응답 중 긴 것

  if (dp.is_main_mem)
  // [한국어] DRAM 메인 메모리: 프리차지 지연과 사이클 타임 재계산
  {
    multisubbank_interleave_cycle_time = htree_in_add->delay;
    // [한국어] 메인 메모리의 인터리브 사이클 = 외부 H-tree 주소 지연만
    precharge_delay = htree_in_add->delay +
                      bank.htree_in_add->delay + bank.mat.delay_writeback +
                      bank.mat.delay_wl_reset + bank.mat.delay_bl_restore;
    // [한국어] 프리차지 지연 = 주소 H-tree + 쓰기백 + 워드라인 리셋 + 비트라인 복원
    cycle_time = access_time + precharge_delay;
    // [한국어] DRAM 사이클 타임 = 접근 지연 + 프리차지 지연
  }
  else
  {
    precharge_delay = 0; // [한국어] SRAM은 프리차지가 사이클 임계 경로에 없음
  }
  return outrisetime; // [한국어] 출력 라이즈 타임 반환 (현재 호출자는 미사용)
}



/*
 * [한국어]
 * UCA::compute_power_energy - UCA 전체 전력·에너지 계산
 *
 * @return: (없음) — power, activate/read/write/precharge_energy 등 멤버 직접 설정
 *
 * 1) bank.compute_power_energy() — Bank/Mat/Subarray 전력 계산
 * 2) H-tree 라우팅 전력(power_routing_to_bank) 계산 — 읽기/쓰기/서치별 분리
 * 3) 누설 전력: 닫힌 페이지/열린 페이지/요청·응답 네트워크 누설 분리
 * 4) DRAM 리프레시 전력: 리프레시 주기로 나누어 등가 누설 전력에 추가
 * 5) 데이터 배열(is_tag==false)은 closed/open page read 에너지를 직접 power.readOp에 설정
 * 실행 컨텍스트: UCA() 생성자 내에서 compute_delays() 직후 호출
 * 호출 체인: UCA() → [이 함수]
 */
// note: currently, power numbers are for a bank of an array
void UCA::compute_power_energy()
{
  bank.compute_power_energy(); // [한국어] Bank 계층 전력 계산 — Mat/SA/H-tree 내부
  power = bank.power;          // [한국어] bank 전력을 UCA power의 초기값으로 복사

  power_routing_to_bank.readOp.dynamic  = htree_in_add->power.readOp.dynamic + htree_out_data->power.readOp.dynamic;
  // [한국어] 읽기 H-tree 동적 전력 = 주소 입력 H-tree + 데이터 출력 H-tree
  power_routing_to_bank.writeOp.dynamic = htree_in_add->power.readOp.dynamic + htree_in_data->power.readOp.dynamic;
  // [한국어] 쓰기 H-tree 동적 전력 = 주소 입력 H-tree + 데이터 입력 H-tree
  if (dp.fully_assoc || dp.pure_cam)
      power_routing_to_bank.searchOp.dynamic= htree_in_search->power.searchOp.dynamic + htree_out_search->power.searchOp.dynamic;
  // [한국어] FA/CAM 서치 H-tree 동적 전력 = 서치 입력 + 서치 출력 H-tree

  power_routing_to_bank.readOp.leakage += htree_in_add->power.readOp.leakage +
                                          htree_in_data->power.readOp.leakage +
                                          htree_out_data->power.readOp.leakage;
  // [한국어] H-tree 3개의 누설 전력 합산

  power_routing_to_bank.readOp.gate_leakage += htree_in_add->power.readOp.gate_leakage +
                                          htree_in_data->power.readOp.gate_leakage +
                                          htree_out_data->power.readOp.gate_leakage;
  // [한국어] H-tree 3개의 게이트 누설 전력 합산
  if (dp.fully_assoc || dp.pure_cam)
  // [한국어] FA/CAM: 서치 H-tree 누설도 추가
  {
	power_routing_to_bank.readOp.leakage += htree_in_search->power.readOp.leakage + htree_out_search->power.readOp.leakage;
	power_routing_to_bank.readOp.gate_leakage += htree_in_search->power.readOp.gate_leakage + htree_out_search->power.readOp.gate_leakage;
  }

  power.searchOp.dynamic += power_routing_to_bank.searchOp.dynamic;
  // [한국어] 서치 동적 전력에 서치 H-tree 전력 추가
  power.readOp.dynamic += power_routing_to_bank.readOp.dynamic;
  // [한국어] 읽기 동적 전력에 H-tree 전력 추가
  power.readOp.leakage += power_routing_to_bank.readOp.leakage;
  // [한국어] 읽기 누설 전력에 H-tree 누설 추가
  power.readOp.gate_leakage += power_routing_to_bank.readOp.gate_leakage;
  // [한국어] 읽기 게이트 누설에 H-tree 게이트 누설 추가

  // calculate total write energy per access
  power.writeOp.dynamic = power.readOp.dynamic
                        - bank.mat.power_bitline.readOp.dynamic * dp.num_act_mats_hor_dir
                        // [한국어] 읽기용 비트라인 전력 제거
                        + bank.mat.power_bitline.writeOp.dynamic * dp.num_act_mats_hor_dir
                        // [한국어] 쓰기용 비트라인 전력 추가
                        - power_routing_to_bank.readOp.dynamic
                        // [한국어] 읽기 H-tree 전력 제거
                        + power_routing_to_bank.writeOp.dynamic
                        // [한국어] 쓰기 H-tree 전력 추가 (데이터 입력 H-tree)
                        + bank.htree_in_data->power.readOp.dynamic
                        // [한국어] 뱅크 내 데이터 입력 H-tree 전력 추가 (쓰기 시 사용)
                        - bank.htree_out_data->power.readOp.dynamic;
                        // [한국어] 뱅크 내 데이터 출력 H-tree 전력 제거 (쓰기 시 미사용)

  if (dp.is_dram == false)
  // [한국어] SRAM: 쓰기 시 SA를 사용하지 않으므로 SA 전력 제거
  {
    power.writeOp.dynamic -= bank.mat.power_sa.readOp.dynamic * dp.num_act_mats_hor_dir;
    // [한국어] SRAM 쓰기는 SA가 비활성 — 수평 방향 활성 mat 수만큼 SA 전력 감산
  }

  dyn_read_energy_from_closed_page = power.readOp.dynamic;
  // [한국어] 닫힌 페이지 읽기 에너지 = 전체 읽기 동적 전력 (행 활성화 포함)
  dyn_read_energy_from_open_page   = power.readOp.dynamic -
                                     (bank.mat.r_predec->power.readOp.dynamic +
                                      bank.mat.power_row_decoders.readOp.dynamic +
                                      bank.mat.power_bl_precharge_eq_drv.readOp.dynamic +
                                      bank.mat.power_sa.readOp.dynamic +
                                      bank.mat.power_bitline.readOp.dynamic) * dp.num_act_mats_hor_dir;
  // [한국어] 열린 페이지 읽기 에너지 = 전체 - (행 예비디코더+행디코더+BL프리차지+SA+비트라인) × 활성 mat 수
  // 열린 페이지: 행이 이미 활성화된 상태 — 행 관련 전력 불필요

  dyn_read_energy_remaining_words_in_burst =
    (MAX((g_ip->burst_len / g_ip->int_prefetch_w), 1) - 1) *
    // [한국어] 버스트 내 첫 번째 단어 이후 추가 단어 수 = max(버스트/프리페치, 1) - 1
    ((bank.mat.sa_mux_lev_1_predec->power.readOp.dynamic +
      bank.mat.sa_mux_lev_2_predec->power.readOp.dynamic +
      bank.mat.power_sa_mux_lev_1_decoders.readOp.dynamic +
      bank.mat.power_sa_mux_lev_2_decoders.readOp.dynamic +
      bank.mat.power_subarray_out_drv.readOp.dynamic)     * dp.num_act_mats_hor_dir +
     bank.htree_out_data->power.readOp.dynamic +
     power_routing_to_bank.readOp.dynamic);
  // [한국어] 버스트 잔여 단어 에너지 = (SA MUX + 서브어레이 출력드라이버 전력 × 활성 mat) + 출력 H-tree 전력
  dyn_read_energy_from_closed_page += dyn_read_energy_remaining_words_in_burst;
  // [한국어] 닫힌 페이지 총 에너지 = 첫 단어 + 잔여 버스트 단어 에너지
  dyn_read_energy_from_open_page   += dyn_read_energy_remaining_words_in_burst;
  // [한국어] 열린 페이지 총 에너지 = 첫 단어 + 잔여 버스트 단어 에너지

  activate_energy = htree_in_add->power.readOp.dynamic +
                    bank.htree_in_add->power_bit.readOp.dynamic * bank.num_addr_b_routed_to_mat_for_act +
                    (bank.mat.r_predec->power.readOp.dynamic +
                     bank.mat.power_row_decoders.readOp.dynamic +
                     bank.mat.power_sa.readOp.dynamic) * dp.num_act_mats_hor_dir;
  // [한국어] 행 활성화(ACTIVATE) 에너지 = 주소 H-tree + 행 관련 mat 전력 × 활성 mat 수
  read_energy    = (htree_in_add->power.readOp.dynamic +
                    bank.htree_in_add->power_bit.readOp.dynamic * bank.num_addr_b_routed_to_mat_for_rd_or_wr +
                    (bank.mat.sa_mux_lev_1_predec->power.readOp.dynamic  +
                     bank.mat.sa_mux_lev_2_predec->power.readOp.dynamic  +
                     bank.mat.power_sa_mux_lev_1_decoders.readOp.dynamic +
                     bank.mat.power_sa_mux_lev_2_decoders.readOp.dynamic +
                     bank.mat.power_subarray_out_drv.readOp.dynamic) * dp.num_act_mats_hor_dir +
                    bank.htree_out_data->power.readOp.dynamic +
                    htree_in_data->power.readOp.dynamic) * g_ip->burst_len;
  // [한국어] 읽기 에너지 = (주소H-tree + SA MUX + 출력드라이버 + 출력H-tree + 데이터입력H-tree) × 버스트 길이
  write_energy   = (htree_in_add->power.readOp.dynamic +
                    bank.htree_in_add->power_bit.readOp.dynamic * bank.num_addr_b_routed_to_mat_for_rd_or_wr +
                    htree_in_data->power.readOp.dynamic +
                    bank.htree_in_data->power.readOp.dynamic +
                    (bank.mat.sa_mux_lev_1_predec->power.readOp.dynamic  +
                     bank.mat.sa_mux_lev_2_predec->power.readOp.dynamic  +
                     bank.mat.power_sa_mux_lev_1_decoders.readOp.dynamic +
                     bank.mat.power_sa_mux_lev_2_decoders.readOp.dynamic) * dp.num_act_mats_hor_dir) * g_ip->burst_len;
  // [한국어] 쓰기 에너지 = (주소H-tree + 데이터입력H-tree × 2 + SA MUX) × 버스트 길이
  precharge_energy = (bank.mat.power_bitline.readOp.dynamic +
                      bank.mat.power_bl_precharge_eq_drv.readOp.dynamic) * dp.num_act_mats_hor_dir;
  // [한국어] 프리차지 에너지 = (비트라인 전력 + BL 프리차지 드라이버 전력) × 활성 mat 수

  leak_power_subbank_closed_page =
    (bank.mat.r_predec->power.readOp.leakage +
     bank.mat.b_mux_predec->power.readOp.leakage +
     bank.mat.sa_mux_lev_1_predec->power.readOp.leakage +
     bank.mat.sa_mux_lev_2_predec->power.readOp.leakage +
     bank.mat.power_row_decoders.readOp.leakage +
     bank.mat.power_bit_mux_decoders.readOp.leakage +
     bank.mat.power_sa_mux_lev_1_decoders.readOp.leakage +
     bank.mat.power_sa_mux_lev_2_decoders.readOp.leakage +
     bank.mat.leak_power_sense_amps_closed_page_state) * dp.num_act_mats_hor_dir;
  // [한국어] 닫힌 페이지 뱅크 누설: 예비디코더+디코더+SA(닫힌 상태) 누설 × 활성 mat 수

  leak_power_subbank_closed_page +=
    (bank.mat.r_predec->power.readOp.gate_leakage +
     bank.mat.b_mux_predec->power.readOp.gate_leakage +
     bank.mat.sa_mux_lev_1_predec->power.readOp.gate_leakage +
     bank.mat.sa_mux_lev_2_predec->power.readOp.gate_leakage +
     bank.mat.power_row_decoders.readOp.gate_leakage +
     bank.mat.power_bit_mux_decoders.readOp.gate_leakage +
     bank.mat.power_sa_mux_lev_1_decoders.readOp.gate_leakage +
     bank.mat.power_sa_mux_lev_2_decoders.readOp.gate_leakage) * dp.num_act_mats_hor_dir; //+
     //bank.mat.leak_power_sense_amps_closed_page_state) * dp.num_act_mats_hor_dir;
  // [한국어] 닫힌 페이지 게이트 누설 추가 누산

  leak_power_subbank_open_page =
    (bank.mat.r_predec->power.readOp.leakage +
     bank.mat.b_mux_predec->power.readOp.leakage +
     bank.mat.sa_mux_lev_1_predec->power.readOp.leakage +
     bank.mat.sa_mux_lev_2_predec->power.readOp.leakage +
     bank.mat.power_row_decoders.readOp.leakage +
     bank.mat.power_bit_mux_decoders.readOp.leakage +
     bank.mat.power_sa_mux_lev_1_decoders.readOp.leakage +
     bank.mat.power_sa_mux_lev_2_decoders.readOp.leakage +
     bank.mat.leak_power_sense_amps_open_page_state) * dp.num_act_mats_hor_dir;
  // [한국어] 열린 페이지 뱅크 누설: 닫힌 페이지와 다른 점 = SA가 열린 상태 누설 사용

  leak_power_subbank_open_page +=
    (bank.mat.r_predec->power.readOp.gate_leakage +
     bank.mat.b_mux_predec->power.readOp.gate_leakage +
     bank.mat.sa_mux_lev_1_predec->power.readOp.gate_leakage +
     bank.mat.sa_mux_lev_2_predec->power.readOp.gate_leakage +
     bank.mat.power_row_decoders.readOp.gate_leakage +
     bank.mat.power_bit_mux_decoders.readOp.gate_leakage +
     bank.mat.power_sa_mux_lev_1_decoders.readOp.gate_leakage +
     bank.mat.power_sa_mux_lev_2_decoders.readOp.gate_leakage ) * dp.num_act_mats_hor_dir;
     //bank.mat.leak_power_sense_amps_open_page_state) * dp.num_act_mats_hor_dir;
  // [한국어] 열린 페이지 게이트 누설 추가 누산

  leak_power_request_and_reply_networks =
    power_routing_to_bank.readOp.leakage +
    bank.htree_in_add->power.readOp.leakage +
    bank.htree_in_data->power.readOp.leakage +
    bank.htree_out_data->power.readOp.leakage;
  // [한국어] 요청·응답 네트워크 누설 = 외부+내부 H-tree 3개 누설 합산

  leak_power_request_and_reply_networks +=
    power_routing_to_bank.readOp.gate_leakage +
    bank.htree_in_add->power.readOp.gate_leakage +
    bank.htree_in_data->power.readOp.gate_leakage +
    bank.htree_out_data->power.readOp.gate_leakage;
  // [한국어] H-tree 게이트 누설 추가 누산

  if (dp.fully_assoc || dp.pure_cam)
  // [한국어] FA/CAM: 서치 H-tree 누설도 네트워크 누설에 추가
  {
	leak_power_request_and_reply_networks += htree_in_search->power.readOp.leakage + htree_out_search->power.readOp.leakage;
	leak_power_request_and_reply_networks += htree_in_search->power.readOp.gate_leakage + htree_out_search->power.readOp.gate_leakage;
  }


  if (dp.is_dram)
  { // if DRAM, add contribution of power spent in row predecoder drivers, blocks and decoders to refresh power
    refresh_power  = (bank.mat.r_predec->power.readOp.dynamic * dp.num_act_mats_hor_dir +
                      bank.mat.row_dec->power.readOp.dynamic) * dp.num_r_subarray * dp.num_subarrays;
    // [한국어] 리프레시 행 디코딩 에너지 = (행 예비디코더 × 활성 mat + 행 디코더) × 행 수 × 서브어레이 수
    refresh_power += bank.mat.per_bitline_read_energy * dp.num_c_subarray * dp.num_r_subarray * dp.num_subarrays;
    // [한국어] 비트라인 읽기 에너지 × 전체 셀 수 (열 수 × 행 수 × 서브어레이 수)
    refresh_power += bank.mat.power_bl_precharge_eq_drv.readOp.dynamic * dp.num_act_mats_hor_dir;
    // [한국어] 비트라인 프리차지 드라이버 에너지 × 활성 mat 수
    refresh_power += bank.mat.power_sa.readOp.dynamic * dp.num_act_mats_hor_dir;
    // [한국어] SA 전력 × 활성 mat 수 (리프레시 시 SA 활성화 필요)
    refresh_power /= dp.dram_refresh_period;
    // [한국어] 리프레시 주기[s]로 나눠 평균 리프레시 전력[W] 변환
  }


  if (dp.is_tag == false)
  // [한국어] 데이터 배열: closed/open page 에너지를 power.readOp/writeOp에 직접 설정
  {
    power.readOp.dynamic  = dyn_read_energy_from_closed_page;
    // [한국어] 데이터 배열 읽기 에너지 = 닫힌 페이지 에너지 (최악 시나리오)
    power.writeOp.dynamic = dyn_read_energy_from_closed_page
      - dyn_read_energy_remaining_words_in_burst
      // [한국어] 버스트 잔여 에너지 제거 (쓰기는 단일 단어만)
      - bank.mat.power_bitline.readOp.dynamic * dp.num_act_mats_hor_dir
      + bank.mat.power_bitline.writeOp.dynamic * dp.num_act_mats_hor_dir
      // [한국어] 읽기 비트라인 에너지 대신 쓰기 비트라인 에너지로 교체
      + (power_routing_to_bank.writeOp.dynamic -
         power_routing_to_bank.readOp.dynamic -
         bank.htree_out_data->power.readOp.dynamic +
         bank.htree_in_data->power.readOp.dynamic) *
        (MAX((g_ip->burst_len / g_ip->int_prefetch_w), 1) - 1); //FIXME
    // [한국어] 버스트 잔여 단어의 쓰기/읽기 H-tree 에너지 차이 보정 (FIXME: 검토 필요)

    if (dp.is_dram == false)
    // [한국어] SRAM: 쓰기 시 SA 불필요 — SA 에너지 감산
    {
      power.writeOp.dynamic -= bank.mat.power_sa.readOp.dynamic * dp.num_act_mats_hor_dir;
    }
  }

  // if DRAM, add refresh power to total leakage
  if (dp.is_dram)
  // [한국어] DRAM: 리프레시 전력을 누설 전력에 추가 (항상 소비되는 정적 전력으로 간주)
  {
    power.readOp.leakage += refresh_power; // [한국어] 등가 누설 전력에 리프레시 전력 추가
  }

  // TODO: below should be  avoided.
  /*if (dp.is_main_mem)
  {
    power.readOp.leakage += MAIN_MEM_PER_CHIP_STANDBY_CURRENT_mA * 1e-3 * g_tp.peri_global.Vdd / g_ip->nbanks;
  }*/

  assert(power.readOp.dynamic  > 0); // [한국어] 읽기 동적 전력 양수 확인
  assert(power.writeOp.dynamic > 0); // [한국어] 쓰기 동적 전력 양수 확인
  assert(power.readOp.leakage  > 0); // [한국어] 읽기 누설 전력 양수 확인
}

