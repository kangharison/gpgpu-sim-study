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
 * [한국어 설명] CACTI 캐시 뱅크 구현 (bank.cc)
 *
 * === 파일의 역할 ===
 * Bank 클래스의 생성자, 소멸자, compute_delays(), compute_power_energy()를 구현한다.
 * 생성자는 DynamicParameter를 기반으로 포트 수와 비트 폭을 계산하여 주소·데이터·검색
 * H-tree(Htree2) 객체를 동적 할당하고 뱅크 면적을 확정한다. compute_power_energy()는
 * 매트 레벨 전력과 H-tree 전력을 합산하여 Bank::power에 저장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 최적화 루프(find_optimal_conf) → Bank 생성 → Mat 초기화 → Htree2 생성.
 * Bank가 CACTI 계층 최상위이며, 상위의 AccelWattch/cacti_interface.cc가 Bank 객체를
 * 직접 생성하여 compute_delays()와 compute_power_energy()를 호출한 뒤 결과를 읽는다.
 *
 * === 타 모듈과의 연결 ===
 * - bank.h: Bank 클래스 선언, DynamicParameter 참조, Htree2 포인터 선언.
 * - mat.h: Bank 내부에 Mat 멤버를 포함. mat.compute_delays/compute_power_energy 위임.
 * - htree2.h: Htree2 — 주소/데이터/검색 H-tree 모델. Bank가 5개(또는 3개) 포인터 소유.
 * - parameter.h: g_ip(num_rw_ports, fast_access, data_assoc 등), _log2() 유틸리티.
 * - component.h: Component 기반 클래스를 통해 power, area 필드를 보유.
 *
 * === 주요 함수/구조체 요약 ===
 * - Bank(): 포트 수 결정, 비트 폭 계산, Htree2 할당, 뱅크 면적 설정, 주소 비트 분할.
 * - ~Bank(): Htree2 동적 메모리 해제 (FA/CAM 조건부 검색 H-tree 포함).
 * - compute_delays(): Mat::compute_delays()에 위임하여 출력 rise time 반환.
 * - compute_power_energy(): Mat 전력 + H-tree 전력 합산 → power.readOp/searchOp 갱신.
 */

#include "bank.h"     // [한국어] Bank 클래스 선언 및 멤버 정의
#include <iostream>   // [한국어] cout — 디버그 출력 (현재 파일에서 직접 미사용)


/*
 * [한국어]
 * Bank - 생성자: 포트 수 결정, H-tree 생성, 뱅크 면적 확정
 *
 * @dyn_p: CACTI 최적화 루프에서 계산된 동적 캐시 파라미터 (캐시 크기, 연관도,
 *         포트 수, 매트 구성, FA/CAM 여부 등).
 * @return: 없음 (생성자)
 *
 * 초기화 단계:
 * 1. dp(dyn_p), mat(dp) — 동적 파라미터와 대표 매트 초기화.
 * 2. num_addr_b_mat, num_mats_hor_dir, num_mats_ver_dir 복사.
 * 3. use_inp_params 여부에 따라 포트 수(RWP/ERP/EWP/SCHP) 결정.
 * 4. 총 주소 비트 수 및 데이터 입출력 비트 수 계산.
 * 5. 일반 캐시: 3개 H-tree 생성 (Add, Data_in, Data_out).
 *    FA/CAM 캐시: 5개 H-tree 생성 (+ 검색 in/out H-tree).
 * 6. 뱅크 면적 = htree_in_data 면적.
 * 7. 주소 비트 분할: num_addr_b_row_dec, _for_act, _for_rd_or_wr.
 *
 * 호출 체인: CACTI 최적화 루프 → [Bank()] → Mat() → Htree2()
 */
Bank::Bank(const DynamicParameter & dyn_p):
  dp(dyn_p), mat(dp),                                        // [한국어] dp=동적 파라미터 참조 바인딩, mat=대표 매트 초기화
  num_addr_b_mat(dyn_p.number_addr_bits_mat),                // [한국어] 매트당 주소 비트 수 복사
  num_mats_hor_dir(dyn_p.num_mats_h_dir), num_mats_ver_dir(dyn_p.num_mats_v_dir) // [한국어] 수평/수직 매트 수 복사
{
  int RWP;   // [한국어] 읽기-쓰기 겸용 포트(Read-Write Port) 수
  int ERP;   // [한국어] 전용 읽기 포트(Exclusive Read Port) 수
  int EWP;   // [한국어] 전용 쓰기 포트(Exclusive Write Port) 수
  int SCHP;  // [한국어] 검색 포트(Search Port) 수 (CAM/FA 캐시 전용)

  if (dp.use_inp_params) // [한국어] DynamicParameter 자체에 포트 정보가 있으면 dp에서 읽음
  {
    RWP  = dp.num_rw_ports;     // [한국어] dp가 직접 지정한 RW 포트 수
    ERP  = dp.num_rd_ports;     // [한국어] dp가 직접 지정한 읽기 전용 포트 수
    EWP  = dp.num_wr_ports;     // [한국어] dp가 직접 지정한 쓰기 전용 포트 수
    SCHP = dp.num_search_ports; // [한국어] dp가 직접 지정한 검색 포트 수
  }
  else // [한국어] dp에 포트 정보가 없으면 사용자 입력(g_ip)에서 가져옴
  {
    RWP  = g_ip->num_rw_ports;     // [한국어] 전역 입력 파라미터(g_ip)에서 RW 포트 수
    ERP  = g_ip->num_rd_ports;     // [한국어] 전역 입력 파라미터에서 읽기 전용 포트 수
    EWP  = g_ip->num_wr_ports;     // [한국어] 전역 입력 파라미터에서 쓰기 전용 포트 수
    SCHP = g_ip->num_search_ports; // [한국어] 전역 입력 파라미터에서 검색 포트 수 (CAM 전용)
  }

  int total_addrbits = (dp.number_addr_bits_mat + dp.number_subbanks_decode)*(RWP+ERP+EWP);
  // [한국어] 총 주소 비트 수 = (매트 주소 비트 + 서브뱅크 디코드 비트) × 총 포트 수(RWP+ERP+EWP).
  //         서브뱅크 디코드 비트: 뱅크 내 어느 매트를 선택할지 결정하는 추가 주소 비트.
  int datainbits     = dp.num_di_b_bank_per_port * (RWP + EWP);
  // [한국어] 데이터 입력 총 비트 수 = 포트당 데이터 입력 비트 × (RW + 쓰기 전용) 포트 수.
  int dataoutbits    = dp.num_do_b_bank_per_port * (RWP + ERP);
  // [한국어] 데이터 출력 총 비트 수 = 포트당 데이터 출력 비트 × (RW + 읽기 전용) 포트 수.
  int searchinbits;  // [한국어] 검색 입력 비트 수 (FA/CAM 캐시만 사용)
  int searchoutbits; // [한국어] 검색 출력 비트 수 (FA/CAM 캐시만 사용)

  if (dp.fully_assoc || dp.pure_cam) // [한국어] FA(완전 연관)/순수 CAM 캐시인 경우 검색 비트 계산
  {
	  datainbits   = dp.num_di_b_bank_per_port * (RWP + EWP);
	  // [한국어] FA/CAM에서도 동일한 데이터 입력 비트 계산 (위와 동일하지만 명시적 재계산).
	  dataoutbits  = dp.num_do_b_bank_per_port * (RWP + ERP);
	  // [한국어] FA/CAM에서도 동일한 데이터 출력 비트 계산.
	  searchinbits    = dp.num_si_b_bank_per_port * SCHP;
	  // [한국어] 검색 입력 비트 = 포트당 검색 키 비트 × 검색 포트 수.
	  searchoutbits   = dp.num_so_b_bank_per_port * SCHP;
	  // [한국어] 검색 출력 비트 = 포트당 검색 결과 비트 × 검색 포트 수.
  }

  if (!(dp.fully_assoc || dp.pure_cam)) // [한국어] 일반 캐시(직접 사상/세트 연관) 경우
    {
    if (g_ip->fast_access && dp.is_tag == false) // [한국어] 빠른 접근 모드이며 데이터 배열인 경우
    {
        dataoutbits *= g_ip->data_assoc;
        // [한국어] 빠른 접근: 연관도만큼 데이터 출력 비트를 확장 (태그 비교 병렬화로 모든 way 동시 출력).
    }

  htree_in_add   = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
      total_addrbits, datainbits, 0,dataoutbits,0, num_mats_ver_dir*2, num_mats_hor_dir*2, Add_htree);
  // [한국어] 주소 입력 H-tree 생성: 배선 타입=g_ip->wt, 리프 노드 크기=mat면적,
  //         주소/데이터 비트 수, 검색 비트=0, 수직/수평 노드 수=num_mats*2, 타입=Add_htree.
  htree_in_data  = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
      total_addrbits, datainbits, 0,dataoutbits,0, num_mats_ver_dir*2, num_mats_hor_dir*2, Data_in_htree);
  // [한국어] 데이터 입력(쓰기) H-tree 생성: 타입=Data_in_htree. 뱅크 면적은 이 H-tree의 면적.
  htree_out_data = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
      total_addrbits, datainbits, 0,dataoutbits,0, num_mats_ver_dir*2, num_mats_hor_dir*2, Data_out_htree);
  // [한국어] 데이터 출력(읽기) H-tree 생성: 타입=Data_out_htree.

//  htree_out_data = new Htree2 (g_ip->wt,(double) 100, (double)100,
//		  total_addrbits, datainbits, 0,dataoutbits,0, num_mats_ver_dir*2, num_mats_hor_dir*2, Data_out_htree);
  // [한국어] 위 라인은 디버그용으로 고정 100µm×100µm 면적을 사용하던 코드 — 현재 비활성(주석 처리).

  area.w = htree_in_data->area.w; // [한국어] 뱅크 가로 폭 = 데이터 입력 H-tree 가로 폭 (H-tree가 뱅크 전체를 덮음)
  area.h = htree_in_data->area.h; // [한국어] 뱅크 세로 높이 = 데이터 입력 H-tree 세로 높이
  }
  else // [한국어] FA/CAM 캐시: 5개 H-tree 생성 (검색 in/out 추가)
  {
	  htree_in_add   = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
			  total_addrbits, datainbits, searchinbits,dataoutbits,searchoutbits, num_mats_ver_dir*2, num_mats_hor_dir*2, Add_htree);
	  // [한국어] 주소 입력 H-tree (FA/CAM: 검색 비트 포함).
	  htree_in_data  = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
			  total_addrbits, datainbits,searchinbits, dataoutbits, searchoutbits, num_mats_ver_dir*2, num_mats_hor_dir*2, Data_in_htree);
	  // [한국어] 데이터 입력 H-tree (FA/CAM: 검색 비트 포함).
	  htree_out_data = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
			  total_addrbits, datainbits,searchinbits, dataoutbits, searchoutbits,num_mats_ver_dir*2, num_mats_hor_dir*2, Data_out_htree);
	  // [한국어] 데이터 출력 H-tree (FA/CAM: 검색 비트 포함).
	  htree_in_search  = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
			  total_addrbits, datainbits,searchinbits, dataoutbits, searchoutbits, num_mats_ver_dir*2, num_mats_hor_dir*2, Data_in_htree,true, true);
	  // [한국어] 검색 키 입력 H-tree: is_search=true, is_input=true — CAM 검색 키 분배.
	  htree_out_search = new Htree2 (g_ip->wt,(double) mat.area.w, (double)mat.area.h,
			  total_addrbits, datainbits,searchinbits, dataoutbits, searchoutbits,num_mats_ver_dir*2, num_mats_hor_dir*2, Data_out_htree,true);
	  // [한국어] 검색 결과 출력 H-tree: is_search=true — CAM 히트/데이터 수집.

      area.w = htree_in_data->area.w; // [한국어] FA/CAM 뱅크 가로 폭 = 데이터 입력 H-tree 폭
      area.h = htree_in_data->area.h; // [한국어] FA/CAM 뱅크 세로 높이 = 데이터 입력 H-tree 높이
  }

  num_addr_b_row_dec = _log2(mat.subarray.num_rows);
  // [한국어] 행 디코더 주소 비트 수 = log2(서브어레이 행 수).
  //         행 디코더는 이 비트 수로 특정 워드라인(WL)을 선택한다.
  num_addr_b_routed_to_mat_for_act = num_addr_b_row_dec;
  // [한국어] 매트 활성화에 필요한 주소 비트 수 = 행 디코더 비트 수 (동일).
  num_addr_b_routed_to_mat_for_rd_or_wr = num_addr_b_mat - num_addr_b_row_dec;
  // [한국어] 읽기/쓰기에 사용되는 열(column) 주소 비트 수 = 총 매트 주소 비트 - 행 디코더 비트.
}



/*
 * [한국어]
 * ~Bank - 소멸자: H-tree 동적 메모리 해제
 *
 * @return: void
 *
 * 생성자에서 new로 할당된 htree_in_add, htree_out_data, htree_in_data를 항상 해제한다.
 * FA(완전 연관)/순수 CAM 캐시인 경우 추가로 htree_in_search, htree_out_search도 해제한다.
 * 일반 캐시에서는 검색 H-tree가 생성되지 않으므로 조건 검사 후 해제한다.
 *
 * 호출 체인: CACTI 최적화 루프 종료 시 자동 호출
 */
Bank::~Bank()
{
  delete htree_in_add;   // [한국어] 주소 입력 H-tree 해제
  delete htree_out_data; // [한국어] 데이터 출력 H-tree 해제
  delete htree_in_data;  // [한국어] 데이터 입력 H-tree 해제
  if (dp.fully_assoc || dp.pure_cam) // [한국어] FA/CAM 캐시인 경우에만 검색 H-tree 해제
  {
	  delete htree_in_search;  // [한국어] 검색 키 입력 H-tree 해제 (FA/CAM 전용)
	  delete htree_out_search; // [한국어] 검색 결과 출력 H-tree 해제 (FA/CAM 전용)
  }
}


/*
 * [한국어]
 * compute_delays - 뱅크 전체 지연 계산 (Mat에 위임)
 *
 * @inrisetime: 입력 신호 상승 시간(s). 첫 번째 게이트/배선의 초기 조건.
 * @return: 출력 신호 상승 시간(s). 상위 계층이 다음 단의 inrisetime으로 사용.
 *
 * Bank 수준에서 추가 지연 계산 없이 Mat::compute_delays()에 완전히 위임한다.
 * 매트 내부의 디코더→배선→센스앰프 경로가 지배적 지연이므로 Bank 자체 추가 지연은 없다.
 *
 * 호출 체인: CACTI 지연 계산 루틴 → [compute_delays(inrisetime)] → Mat::compute_delays()
 */
double Bank::compute_delays(double inrisetime)
{
  return mat.compute_delays(inrisetime); // [한국어] 대표 매트의 지연 계산 결과를 그대로 반환
}


/*
 * [한국어]
 * compute_power_energy - 뱅크 전체 전력·에너지 합산
 *
 * @return: void. power.readOp.dynamic/leakage/gate_leakage 및 (FA/CAM의 경우)
 *          power.searchOp.dynamic에 결과를 누적한다.
 *
 * 1단계: mat.compute_power_energy()로 대표 매트의 전력 계산.
 * 2단계(일반 캐시):
 *   - 동적 전력: 매트 동적 전력 × num_act_mats_hor_dir (수평 방향 활성화 매트 수).
 *   - 누설/게이트 누설: 매트 전력 × dp.num_mats (전체 매트, 항상 누설 발생).
 *   - H-tree 동적: htree_in_add + htree_out_data 합산.
 *   - H-tree 누설/게이트 누설: 세 H-tree 모두 합산.
 * 3단계(FA/CAM 캐시):
 *   - readOp 동적은 수평 활성 매트=1개로 고정.
 *   - searchOp 동적: 검색 관련 매트 전력(BL 프리차지, SA, BL, 서브어레이 출력 드라이버,
 *     ml_to_ram_wl 드라이버) + 검색 H-tree 전력 합산.
 *
 * 호출 체인: CACTI 전력 최적화 루프 → [compute_power_energy()] → Mat::compute_power_energy()
 */
void Bank::compute_power_energy()
{
  mat.compute_power_energy(); // [한국어] 대표 매트의 전력 계산 (Decoder, Bitline, SA, 서브어레이 포함)

  if (!(dp.fully_assoc || dp.pure_cam)) // [한국어] 일반 캐시(직접 사상/세트 연관) 전력 집계
  {
	  power.readOp.dynamic += mat.power.readOp.dynamic * dp.num_act_mats_hor_dir;
	  // [한국어] 동적 전력: 수평 방향 활성 매트 수(num_act_mats_hor_dir) × 매트 1개 동적 전력.
	  //         읽기/쓰기 시 수평 방향 매트들이 동시 활성화된다.
	  power.readOp.leakage += mat.power.readOp.leakage * dp.num_mats;
	  // [한국어] 누설 전력: 모든 매트(num_mats)에서 항상 누설 발생 → 전체 매트 수 곱함.
	  power.readOp.gate_leakage += mat.power.readOp.gate_leakage * dp.num_mats;
	  // [한국어] 게이트 누설: 모든 매트에서 항상 발생 → 전체 매트 수 곱함.

	  power.readOp.dynamic += htree_in_add->power.readOp.dynamic;
	  // [한국어] 주소 입력 H-tree 동적 전력 합산.
	  power.readOp.dynamic += htree_out_data->power.readOp.dynamic;
	  // [한국어] 데이터 출력 H-tree 동적 전력 합산.

	  power.readOp.leakage += htree_in_add->power.readOp.leakage;
	  // [한국어] 주소 H-tree 누설 전력 합산.
	  power.readOp.leakage += htree_in_data->power.readOp.leakage;
	  // [한국어] 데이터 입력 H-tree 누설 전력 합산.
	  power.readOp.leakage += htree_out_data->power.readOp.leakage;
	  // [한국어] 데이터 출력 H-tree 누설 전력 합산.
	  power.readOp.gate_leakage += htree_in_add->power.readOp.gate_leakage;
	  // [한국어] 주소 H-tree 게이트 누설 전력 합산.
	  power.readOp.gate_leakage += htree_in_data->power.readOp.gate_leakage;
	  // [한국어] 데이터 입력 H-tree 게이트 누설 전력 합산.
	  power.readOp.gate_leakage += htree_out_data->power.readOp.gate_leakage;
	  // [한국어] 데이터 출력 H-tree 게이트 누설 전력 합산.
  }
  else // [한국어] FA(완전 연관)/CAM 캐시 전력 집계
  {

	  power.readOp.dynamic += mat.power.readOp.dynamic ;//for fa and cam num_act_mats_hor_dir is 1 for plain r/w
	  // [한국어] FA/CAM에서 일반 읽기/쓰기 동적 전력: num_act_mats_hor_dir=1이므로 ×1.
	  power.readOp.leakage += mat.power.readOp.leakage * dp.num_mats;
	  // [한국어] FA/CAM 누설: 전체 매트 수 × 대표 매트 누설.
	  power.readOp.gate_leakage += mat.power.readOp.gate_leakage * dp.num_mats;
	  // [한국어] FA/CAM 게이트 누설: 전체 매트 수 × 대표 매트 게이트 누설.

	  power.searchOp.dynamic += mat.power.searchOp.dynamic * dp.num_mats;
	  // [한국어] 검색 동적 전력: CAM 검색은 모든 매트를 동시에 활성화 → 전체 매트 수 곱함.
	  power.searchOp.dynamic += mat.power_bl_precharge_eq_drv.searchOp.dynamic +
	  	                        mat.power_sa.searchOp.dynamic +
	  	                        mat.power_bitline.searchOp.dynamic +
	  	                        mat.power_subarray_out_drv.searchOp.dynamic+
	  	                        mat.ml_to_ram_wl_drv->power.readOp.dynamic;
	  // [한국어] 검색 동적 전력 세부 항목 합산:
	  //   power_bl_precharge_eq_drv: 비트라인(BL) 프리차지 이퀄라이저 드라이버.
	  //   power_sa: 센스앰프(Sense Amplifier) 전력.
	  //   power_bitline: 비트라인 스윙 전력.
	  //   power_subarray_out_drv: 서브어레이 출력 드라이버 전력.
	  //   ml_to_ram_wl_drv: 매치 라인(ML)에서 RAM 워드라인(WL) 드라이버 전력.

	  power.readOp.dynamic += htree_in_add->power.readOp.dynamic;
	  // [한국어] FA/CAM 주소 H-tree 동적 전력 합산 (일반 읽기/쓰기 경로).
	  power.readOp.dynamic += htree_out_data->power.readOp.dynamic;
	  // [한국어] FA/CAM 데이터 출력 H-tree 동적 전력 합산.

	  power.searchOp.dynamic += htree_in_search->power.searchOp.dynamic;
	  // [한국어] 검색 키 입력 H-tree 동적 전력 합산.
	  power.searchOp.dynamic += htree_out_search->power.searchOp.dynamic;
	  // [한국어] 검색 결과 출력 H-tree 동적 전력 합산.

	  power.readOp.leakage += htree_in_add->power.readOp.leakage;
	  // [한국어] 주소 H-tree 누설 합산 (FA/CAM).
	  power.readOp.leakage += htree_in_data->power.readOp.leakage;
	  // [한국어] 데이터 입력 H-tree 누설 합산 (FA/CAM).
	  power.readOp.leakage += htree_out_data->power.readOp.leakage;
	  // [한국어] 데이터 출력 H-tree 누설 합산 (FA/CAM).
	  power.readOp.leakage += htree_in_search->power.readOp.leakage;
	  // [한국어] 검색 입력 H-tree 누설 합산.
	  power.readOp.leakage += htree_out_search->power.readOp.leakage;
	  // [한국어] 검색 출력 H-tree 누설 합산.


	  power.readOp.gate_leakage += htree_in_add->power.readOp.gate_leakage;
	  // [한국어] 주소 H-tree 게이트 누설 합산 (FA/CAM).
	  power.readOp.gate_leakage += htree_in_data->power.readOp.gate_leakage;
	  // [한국어] 데이터 입력 H-tree 게이트 누설 합산 (FA/CAM).
	  power.readOp.gate_leakage += htree_out_data->power.readOp.gate_leakage;
	  // [한국어] 데이터 출력 H-tree 게이트 누설 합산 (FA/CAM).
	  power.readOp.gate_leakage += htree_in_search->power.readOp.gate_leakage;
	  // [한국어] 검색 입력 H-tree 게이트 누설 합산.
	  power.readOp.gate_leakage += htree_out_search->power.readOp.gate_leakage;
	  // [한국어] 검색 출력 H-tree 게이트 누설 합산.

  }

}

