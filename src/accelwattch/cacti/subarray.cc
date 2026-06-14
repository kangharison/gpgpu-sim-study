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
 * [한국어 설명] CACTI SRAM/DRAM/CAM 서브어레이 면적·커패시턴스 모델 구현 (subarray.cc)
 *
 * === 파일의 역할 ===
 * SRAM/DRAM/CAM 중 하나인 물리 서브어레이(subarray) 하나를 모델링한다.
 * 생성자에서 행 수·열 수·셀 크기로 서브어레이 면적을 계산하고,
 * compute_C()에서 워드라인(C_wl/R_wl)과 비트라인(C_bl) 커패시턴스를 계산한다.
 * 이 값들은 Mat 클래스가 센스앰프·드라이버 지연·전력 계산에 사용한다.
 * FA(Fully Associative) / pure_cam / pure_ram 세 가지 조직 모드를 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Mat → Subarray → [이 파일] — Mat이 자신의 서브어레이를 생성할 때 호출된다.
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드 (pthread 내부 포함).
 * 호출 체인: calculate_time() → UCA → Mat → Subarray(dp, is_fa)
 *
 * === 타 모듈과의 연결 ===
 * 의존: DynamicParameter(dp — 배열 구성 정보), basic_circuit(gate_C_pass, drain_C_),
 *       parameter.h(g_tp — 공정 파라미터, 셀 치수, 배선 파라미터)
 * 소비: Mat 클래스가 Subarray 인스턴스를 생성해 C_wl/C_bl/R_wl과 area를 읽음
 * 공유: g_ip(전역 입력, add_ecc_b_ 플래그), g_tp(sram/dram/cam 셀 치수, wire_local)
 *
 * === 주요 함수/구조체 요약 ===
 * Subarray()           : 생성자 — 행/열 수 결정, ECC 열 추가, 면적 계산, compute_C() 호출
 * get_total_cell_area(): 순수 셀 면적 계산 (서브어레이 전체 면적에서 배선 오버헤드 제외)
 * compute_C()          : C_wl/R_wl(워드라인) + C_bl(비트라인) 커패시턴스/저항 계산
 */

#include <iostream>
#include <math.h>
#include <assert.h>

#include "subarray.h"


/*
 * [한국어]
 * Subarray::Subarray - 서브어레이 생성자 — 면적·커패시턴스 계산 진입점
 *
 * @dp_  : DynamicParameter — 배열 구성(행/열, 셀 타입, FA/CAM 플래그 등)
 * @is_fa_: true이면 FA(완전 연관) 캐시 모드 (CAM + RAM 병렬 구조)
 * @return: (생성자이므로 반환값 없음)
 *
 * 1) 멤버 초기화: dp, num_rows, num_cols, cell, cam_cell, is_fa 설정
 * 2) ECC 열 추가: add_ecc_b_ 플래그에 따라 ECC 비트 열을 num_cols에 추가
 * 3) 셀 타입(SRAM/DRAM/FA-CAM)에 따라 분기해 area.h × area.w 계산
 *    - SRAM/DRAM: area.h = cell.h × num_rows, area.w = cell.w × num_cols + stitching overhead
 *    - FA/CAM: area.h = cam_cell.h × (num_rows+1), area.w = CAM+RAM 폭 합 + NAND/드라이버 오버헤드
 * 4) compute_C(): 워드라인·비트라인 커패시턴스 계산
 * 실행 컨텍스트: Mat 생성자 내에서 호출 (단일 스레드 또는 pthread 내부)
 * 호출 체인: Mat() → [이 함수] → compute_C()
 */
Subarray::Subarray(const DynamicParameter & dp_, bool is_fa_):
  dp(dp_),
  num_rows(dp.num_r_subarray), // [한국어] 서브어레이 행 수 (워드라인 수)
  num_cols(dp.num_c_subarray), // [한국어] 서브어레이 열 수 (비트라인 수, ECC 추가 전)
  num_cols_fa_cam(dp.tag_num_c_subarray),  // [한국어] FA 모드 CAM 부분 열 수
  num_cols_fa_ram(dp.data_num_c_subarray), // [한국어] FA 모드 RAM 부분 열 수
  cell(dp.cell),         // [한국어] SRAM 셀 치수 (높이/너비)
  cam_cell(dp.cam_cell), // [한국어] CAM 셀 치수 — FA 모드에서 높이 결정에 사용
  is_fa(is_fa_)          // [한국어] true: 완전 연관(Fully Associative) 캐시 모드
{
	//num_cols=7;
	//cout<<"num_cols ="<< num_cols <<endl;
  if (!(is_fa || dp.pure_cam))
  {
	  num_cols +=(g_ip->add_ecc_b_ ? (int)ceil(num_cols / num_bits_per_ecc_b_) : 0);   // ECC overhead
	  uint32_t ram_num_cells_wl_stitching =
		  (dp.ram_cell_tech_type == lp_dram)   ? dram_num_cells_wl_stitching_ :
	  (dp.ram_cell_tech_type == comm_dram) ? comm_dram_num_cells_wl_stitching_ : sram_num_cells_wl_stitching_;

	  area.h = cell.h * num_rows;

	  area.w = cell.w * num_cols +
	  ceil(num_cols / ram_num_cells_wl_stitching) * g_tp.ram_wl_stitching_overhead_;  // stitching overhead
  }
  else  //cam fa
  {

	  //should not add dummy row here since the dummy row do not need decoder
	  if (is_fa)// fully associative cache
	  {
		  num_cols_fa_cam  += g_ip->add_ecc_b_ ? (int)ceil(num_cols_fa_cam / num_bits_per_ecc_b_) : 0;
		  num_cols_fa_ram  += (g_ip->add_ecc_b_ ? (int)ceil(num_cols_fa_ram / num_bits_per_ecc_b_) : 0);
		  num_cols = num_cols_fa_cam + num_cols_fa_ram;
	  }
	  else
	  {
		  num_cols_fa_cam  += g_ip->add_ecc_b_ ? (int)ceil(num_cols_fa_cam / num_bits_per_ecc_b_) : 0;
		  num_cols_fa_ram  = 0;
		  num_cols = num_cols_fa_cam;
	  }

	  area.h = cam_cell.h * (num_rows + 1);//height of subarray is decided by CAM array. blank space in sram array are filled with dummy cells
	  area.w = cam_cell.w * num_cols_fa_cam + cell.w * num_cols_fa_ram
	  + ceil((num_cols_fa_cam + num_cols_fa_ram) / sram_num_cells_wl_stitching_)*g_tp.ram_wl_stitching_overhead_
	  + 16*g_tp.wire_local.pitch //the overhead for the NAND gate to connect the two halves
	  + 128*g_tp.wire_local.pitch;//the overhead for the drivers from matchline to wordline of RAM
  }

  assert(area.h>0);
  assert(area.w>0);
  compute_C();
}



// [한국어] 소멸자 — 모든 멤버가 값 타입이므로 별도 해제 없음
Subarray::~Subarray()
{
}



/*
 * [한국어]
 * Subarray::get_total_cell_area - 순수 셀 면적 계산 [um²]
 *
 * @return: 서브어레이 내 실제 SRAM/CAM 셀이 차지하는 면적 [um²]
 *          (배선 오버헤드, 스티칭, 드라이버 면적 제외)
 *
 * 면적 효율(efficiency) 계산의 분자로 사용된다.
 * SRAM: cell.get_area() × num_rows × num_cols
 * FA  : cam_cell.h × (num_rows+1) × (CAM셀폭×CAM열 + SRAM셀폭×RAM열)
 * pure_cam: cam_cell.get_area() × (num_rows+1) × num_cols_fa_cam
 * 호출 체인: UCA / Mat → [이 함수]
 */
double Subarray::get_total_cell_area()
{
//  return (is_fa==false? cell.get_area() * num_rows * num_cols
//		  //: cam_cell.h*(num_rows+1)*(num_cols_fa_cam + sram_cell.get_area()*num_cols_fa_ram));
//		  : cam_cell.get_area()*(num_rows+1)*(num_cols_fa_cam + num_cols_fa_ram));
//		  //: cam_cell.get_area()*(num_rows+1)*num_cols_fa_cam + sram_cell.get_area()*(num_rows+1)*num_cols_fa_ram);//for FA, this area does not include the dummy cells in SRAM arrays.

    if (!(is_fa || dp.pure_cam))
      // [한국어] 일반 SRAM/DRAM: 셀 면적 × 행 × 열
	  return (cell.get_area() * num_rows * num_cols);
    else if (is_fa)
    { //for FA, this area includes the dummy cells in SRAM arrays.
      // [한국어] FA: CAM과 RAM 두 부분의 실제 셀 면적 합산 (더미 행 포함)
      //return (cam_cell.get_area()*(num_rows+1)*(num_cols_fa_cam + num_cols_fa_ram));
      //cout<<"diff" <<cam_cell.get_area()*(num_rows+1)*(num_cols_fa_cam + num_cols_fa_ram)- cam_cell.h*(num_rows+1)*(cam_cell.w*num_cols_fa_cam + cell.w*num_cols_fa_ram)<<endl;
      return (cam_cell.h*(num_rows+1)*(cam_cell.w*num_cols_fa_cam + cell.w*num_cols_fa_ram));
      // [한국어] CAM셀높이 × (행+1더미) × (CAM열×CAM폭 + RAM열×SRAM폭)
    }
    else
      return (cam_cell.get_area()*(num_rows+1)*num_cols_fa_cam );
      // [한국어] 순수 CAM: CAM 셀 면적 × (행+1더미) × CAM 열 수

}



/*
 * [한국어]
 * Subarray::compute_C - 워드라인/비트라인 커패시턴스·저항 계산
 *
 * @return: (없음) — C_wl, R_wl, C_bl, C_wl_cam, R_wl_cam 등 멤버를 직접 설정
 *
 * SRAM/DRAM 배열의 경우:
 *   C_wl = (게이트 커패시턴스 × 2 + 금속 배선 커패시턴스) × num_cols
 *   C_bl = num_rows × (셀 드레인 커패시턴스/2 + 금속 배선 커패시턴스)
 * FA/CAM 배열의 경우:
 *   CAM과 RAM 부분 각각 별도로 계산 후 합산
 *   워드라인(WL)과 매치라인(ML)을 독립적으로 모델링
 * 이 값들은 Mat::compute_delays()에서 워드라인 드라이브 지연, SA 지연 계산에 사용된다.
 * 호출 체인: Subarray() → [이 함수]
 */
void Subarray::compute_C()
{
  double c_w_metal = cell.w * g_tp.wire_local.C_per_um;
  // [한국어] SRAM 셀 1개 너비당 로컬 금속 배선 커패시턴스 [F/셀]
  double r_w_metal = cell.w * g_tp.wire_local.R_per_um;
  // [한국어] SRAM 셀 1개 너비당 로컬 금속 배선 저항 [Ω/셀]
  double C_b_metal = cell.h * g_tp.wire_local.C_per_um;
  // [한국어] SRAM 셀 1개 높이당 로컬 금속 배선 커패시턴스 [F/셀] — 비트라인 방향
  double C_b_row_drain_C; // [한국어] 한 행의 드레인 커패시턴스 (공유 접촉으로 /2)

  if (dp.is_dram)
  // [한국어] DRAM 배열 — pass 트랜지스터 1개 구조
  {
    C_wl = (gate_C_pass(g_tp.dram.cell_a_w, g_tp.dram.b_w, true, true) + c_w_metal) * num_cols;
    // [한국어] C_wl = (DRAM pass 게이트 커패시턴스 + 금속 배선 커패시턴스) × 열 수 [F]
    // gate_C_pass: DRAM pass 트랜지스터의 게이트 커패시턴스

    if (dp.ram_cell_tech_type == comm_dram)
    // [한국어] 상업용 DRAM: 비트라인이 매우 길어 셀 드레인 커패시턴스 무시 가능
    {
      C_bl = num_rows * C_b_metal; // [한국어] C_bl = 행 수 × 금속 배선 커패시턴스만
    }
    else
    // [한국어] 임베디드 DRAM (lp_dram): 셀 드레인 커패시턴스 포함
    {
      C_b_row_drain_C = drain_C_(g_tp.dram.cell_a_w, NCH, 1, 0, cell.w, true, true) / 2.0;  // due to shared contact
      // [한국어] DRAM 셀 NMOS 드레인 커패시턴스 / 2 (공유 접촉 구조)
      C_bl = num_rows * (C_b_row_drain_C + C_b_metal); // [한국어] C_bl = 행 수 × (드레인+금속) [F]
    }
  }
  else
  // [한국어] SRAM/FA/CAM 배열
  {
	  if (!(is_fa ||dp.pure_cam))
	  // [한국어] 일반 SRAM (6T 셀)
	  {
		  C_wl = (gate_C_pass(g_tp.sram.cell_a_w, (g_tp.sram.b_w-2*g_tp.sram.cell_a_w)/2.0, false, true)*2 +
				  c_w_metal) * num_cols;
		  // [한국어] C_wl = (양쪽 2개 pass 게이트 커패시턴스 + 금속 커패시턴스) × 열 수
		  // SRAM 6T 셀에는 워드라인에 연결된 pass 트랜지스터가 2개(각 비트라인 1개)
		  C_b_row_drain_C = drain_C_(g_tp.sram.cell_a_w, NCH, 1, 0, cell.w, false, true) / 2.0;  // due to shared contact
		  // [한국어] SRAM pass 트랜지스터 드레인 커패시턴스 / 2 (공유 접촉)
		  C_bl = num_rows * (C_b_row_drain_C + C_b_metal); // [한국어] C_bl = 행 수 × (드레인+금속) [F]
	  }
	  else
	  // [한국어] FA(완전 연관) 또는 순수 CAM 배열 — 워드라인(WL)을 CAM부분과 RAM부분으로 분리 계산
	  {
		 //Following is wordline not matchline
		 //CAM portion
		 c_w_metal = cam_cell.w * g_tp.wire_local.C_per_um;
		 // [한국어] CAM 셀 너비 기준 금속 배선 커패시턴스 — CAM 셀이 SRAM보다 넓을 수 있음
		 r_w_metal = cam_cell.w * g_tp.wire_local.R_per_um;
		 // [한국어] CAM 셀 너비 기준 금속 배선 저항
         C_wl_cam = (gate_C_pass(g_tp.cam.cell_a_w, (g_tp.cam.b_w-2*g_tp.cam.cell_a_w)/2.0, false, true)*2 +
				  c_w_metal) * num_cols_fa_cam;
         // [한국어] C_wl_cam = (CAM pass 게이트×2 + 금속 커패시턴스) × CAM 열 수 [F]
         R_wl_cam = (r_w_metal) * num_cols_fa_cam;
         // [한국어] R_wl_cam = 금속 배선 저항 × CAM 열 수 [Ω]

         if (!dp.pure_cam)
         // [한국어] FA: CAM 외에 RAM 부분도 있는 경우 RAM 워드라인 파라미터 계산
         {
        	 //RAM portion
        	 c_w_metal = cell.w * g_tp.wire_local.C_per_um;
        	 // [한국어] SRAM 셀 너비 기준 금속 배선 커패시턴스로 재설정
        	 r_w_metal = cell.w * g_tp.wire_local.R_per_um;
        	 // [한국어] SRAM 셀 너비 기준 금속 배선 저항으로 재설정
        	 C_wl_ram = (gate_C_pass(g_tp.sram.cell_a_w, (g_tp.sram.b_w-2*g_tp.sram.cell_a_w)/2.0, false, true)*2 +
        			 c_w_metal) * num_cols_fa_ram;
        	 // [한국어] C_wl_ram = (SRAM pass 게이트×2 + 금속 커패시턴스) × RAM 열 수 [F]
        	 R_wl_ram = (r_w_metal) * num_cols_fa_ram;
        	 // [한국어] R_wl_ram = 금속 저항 × RAM 열 수 [Ω]
         }
         else
         // [한국어] 순수 CAM: RAM 부분 없음 — WL RAM 커패시턴스/저항 0
         {
        	 C_wl_ram = R_wl_ram =0; // [한국어] RAM 부분 커패시턴스·저항 = 0
         }
         C_wl = C_wl_cam + C_wl_ram; // [한국어] 전체 워드라인 커패시턴스 = CAM부분 + RAM부분
         C_wl += (16+128)*g_tp.wire_local.pitch*g_tp.wire_local.C_per_um;
         // [한국어] NAND 게이트(16배치)와 드라이버(128배치) 배선 오버헤드의 커패시턴스 추가

         R_wl = R_wl_cam + R_wl_ram; // [한국어] 전체 워드라인 저항 = CAM부분 + RAM부분
         R_wl += (16+128)*g_tp.wire_local.pitch*g_tp.wire_local.R_per_um;
         // [한국어] NAND/드라이버 배선 오버헤드 저항 추가

         //there are two ways to write to a FA,
         //1) Write to CAM array then force a match on match line to active the corresponding wordline in RAM;
         //2) using separate wordline for read/write and search in RAM.
         //We are using the second approach.
         // [한국어] FA 쓰기 방법 선택: 방법2(RAM에 독립적인 WL 사용)를 채택
         // 이는 매치라인→RAM 워드라인 활성화 경로와 독립적으로 read/write 가능

         //Bitline CAM portion This is bitline not searchline. We assume no sharing between bitline and searchline according to SUN's implementations.
         // [한국어] CAM 부분 비트라인 커패시턴스 계산 (서치라인과 분리 — SUN 구현 기준)
         C_b_metal = cam_cell.h * g_tp.wire_local.C_per_um;
         // [한국어] CAM 셀 높이 기준 금속 배선 커패시턴스 [F/셀]
         C_b_row_drain_C = drain_C_(g_tp.cam.cell_a_w, NCH, 1, 0, cam_cell.w, false, true) / 2.0;  // due to shared contact
         // [한국어] CAM pass 트랜지스터 드레인 커패시턴스 / 2 (공유 접촉)
         C_bl_cam = (num_rows+1) * (C_b_row_drain_C + C_b_metal);
         // [한국어] CAM 비트라인 커패시턴스 = (행 수+1더미) × (드레인+금속) [F]
         //height of subarray is decided by CAM array. blank space in sram array are filled with dummy cells
         C_b_row_drain_C = drain_C_(g_tp.sram.cell_a_w, NCH, 1, 0, cell.w, false, true) / 2.0;  // due to shared contact
         // [한국어] SRAM RAM 부분 pass 트랜지스터 드레인 커패시턴스 / 2
         C_bl = (num_rows +1) * (C_b_row_drain_C + C_b_metal);
         // [한국어] RAM 비트라인 커패시턴스 = (행 수+1더미) × (드레인+CAM높이 금속) [F]
         // 주의: C_b_metal은 CAM 셀 높이 기준으로 재계산됨 — FA 구조에서 높이는 CAM이 결정

	  }
  }
}


