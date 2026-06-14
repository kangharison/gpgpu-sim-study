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
 * [한국어 설명] SRAM 서브어레이 면적·커패시턴스 모델 헤더 (subarray.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 계층의 최하위 물리 단위인 서브어레이(Subarray)를 모델링한다.
 * 서브어레이는 실제 6T SRAM 셀이 행(num_rows)×열(num_cols) 배치된 메모리 셀 배열이다.
 * 생성자에서 면적(area.h × area.w)을 계산하고, compute_C()로 워드라인·비트라인 커패시턴스를 구한다.
 * ECC 오버헤드, 워드라인 스티칭(WL stitching), DRAM/SRAM/FA-CAM 분기를 처리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Bank → Mat → Subarray [이 모듈] (CACTI 계층 최하위)
 * Mat 생성자 → Subarray(dp, is_fa) → area/C_wl/C_bl 계산 완료
 * 계산된 커패시턴스(C_wl, C_bl)는 Mat에서 비트라인/워드라인 지연 계산에 사용된다.
 * 호스트 유저스페이스, 단일 스레드, GPU 시뮬 루프 외부에서 실행.
 *
 * === 타 모듈과의 연결 ===
 * 의존: DynamicParameter(dp — 행/열 수, 셀 타입, ECC 등 파라미터 집합),
 *       Area(area.h/area.w — 면적 계산 결과 저장), parameter.h(g_tp 기술 상수),
 *       basic_circuit(gate_C_pass, drain_C_ — SRAM/DRAM 셀 커패시턴스 수식)
 * 소비: Mat 클래스가 Subarray를 멤버로 보유하며 C_wl/C_bl을 읽음
 * 공유: g_tp.sram/dram/cam (셀 치수), g_tp.wire_local (로컬 와이어 파라미터)
 *
 * === 주요 함수/구조체 요약 ===
 * Subarray(dp, is_fa)   : 생성자 — 행/열 수 확정, ECC 오버헤드 적용, area 계산, compute_C() 호출
 * get_total_cell_area() : 전체 셀 면적 반환 (SRAM/FA-CAM/순수 CAM 분기)
 * compute_C()           : 워드라인(C_wl)·비트라인(C_bl) 커패시턴스 계산 (DRAM/SRAM/FA 분기)
 * num_rows, num_cols    : 서브어레이 행·열 수 (ECC 포함된 최종값)
 * C_wl, C_bl            : 워드라인·비트라인 총 커패시턴스 [F] — 비트라인 지연 계산에 사용
 */

#ifndef __SUBARRAY_H__
#define __SUBARRAY_H__

#include "area.h"
#include "component.h"
#include "parameter.h"

using namespace std;


class Subarray : public Component
{
  public:
    Subarray(const DynamicParameter & dp, bool is_fa_);
    ~Subarray();

    const DynamicParameter & dp;
    /* [한국어] 이 서브어레이의 설계 파라미터 참조 (행/열 수, 셀 타입, 포트 수 등).
     * 설정자: 생성자 초기화 리스트에서 dp_ 참조 바인딩.
     * 읽는 자: compute_C(), get_total_cell_area()에서 is_dram, ram_cell_tech_type, num_r_subarray 등 접근.
     * 값 범위: DynamicParameter가 유효한 파라미터 조합을 갖고 있어야 함.
     * 동기화: 참조이므로 const — 수정 불가. */

    double  get_total_cell_area();
    /* [한국어] 전체 셀 면적 [um²] 반환 — SRAM/FA-CAM/순수 CAM 분기 처리 */

    unsigned int num_rows;
    /* [한국어] 서브어레이의 실제 행 수 (ECC 추가 없이 dp.num_r_subarray 그대로).
     * 설정자: 생성자 초기화 리스트에서 dp.num_r_subarray로 초기화.
     * 읽는 자: compute_C()에서 C_wl/C_bl 계산의 row 개수 팩터로 사용; area.h = cell.h × num_rows.
     * 값 범위: 1 이상의 양의 정수.
     * 동기화: 생성 후 불변. */

    unsigned int num_cols;
    /* [한국어] 서브어레이의 실제 열 수 (ECC 비트 포함, FA인 경우 CAM+RAM 열 합계).
     * 설정자: 생성자에서 dp.num_c_subarray를 기반으로 ECC 오버헤드 추가 후 설정.
     * 읽는 자: area.w = cell.w × num_cols + 스티칭 오버헤드 계산에 사용.
     * 값 범위: dp.num_c_subarray 이상 (ECC 추가로 늘어남).
     * 동기화: 생성 후 불변. */

    int32_t num_cols_fa_cam;
    /* [한국어] FA(Fully-Associative) 캐시에서 CAM(Content-Addressable Memory) 부분의 열 수.
     * 설정자: 생성자에서 dp.tag_num_c_subarray를 기반으로 ECC 추가 후 설정; FA가 아니면 미사용.
     * 읽는 자: compute_C()에서 C_wl_cam 계산, area.w에서 CAM 부분 폭 계산.
     * 값 범위: FA일 때 양수, 아닐 때 dp.tag_num_c_subarray 그대로.
     * 동기화: 생성 후 불변. */

    int32_t num_cols_fa_ram;
    /* [한국어] FA 캐시에서 RAM(데이터) 부분의 열 수 — CAM 태그와 함께 FA 서브어레이 구성.
     * 설정자: FA이면 dp.data_num_c_subarray + ECC; 순수 CAM이면 0.
     * 읽는 자: compute_C()에서 C_wl_ram 계산, area.w에서 RAM 부분 폭 계산.
     * 값 범위: FA이면 양수, 순수 CAM이면 0.
     * 동기화: 생성 후 불변. */

    Area    cell, cam_cell;
    /* [한국어] SRAM 셀 치수(cell)와 CAM 셀 치수(cam_cell) [um].
     * cell    : 표준 6T SRAM 셀의 h(높이) × w(너비).
     * cam_cell: FA-CAM 셀의 h × w (SRAM보다 큰 면적).
     * 설정자: 생성자 초기화 리스트에서 dp.cell/dp.cam_cell로 초기화.
     * 읽는 자: area 계산, compute_C()에서 c_w_metal(단위 길이당 커패시턴스의 기준 길이).
     * 동기화: 불변. */

    bool    is_fa;
    /* [한국어] 완전 연관(Fully-Associative) 캐시 서브어레이 여부.
     * 설정자: 생성자 is_fa_ 인자로 초기화.
     * 읽는 자: compute_C()에서 SRAM/FA-CAM 분기 선택, get_total_cell_area()에서 계산 방식 결정.
     * 값 범위: true(FA) 또는 false(직접 매핑/집합 연관).
     * 동기화: 불변. */

    double  C_wl, C_wl_cam, C_wl_ram;
    /* [한국어] 워드라인 총 커패시턴스 [F].
     * C_wl    : 전체 워드라인 커패시턴스 (SRAM인 경우) 또는 CAM+RAM 합산(FA인 경우).
     * C_wl_cam: FA에서 CAM 워드라인 커패시턴스.
     * C_wl_ram: FA에서 RAM 워드라인 커패시턴스.
     * 설정자: compute_C()에서 gate_C_pass() × num_cols + c_w_metal × num_cols로 계산.
     * 읽는 자: Mat에서 비트라인 방전 지연(Elmore 지연) 계산 시 R_wl × C_wl 항.
     * 동기화: 생성 시 compute_C()에서 한 번 계산. */

    double  R_wl, R_wl_cam, R_wl_ram;
    /* [한국어] 워드라인 총 저항 [Ω].
     * FA 캐시에서 CAM·RAM 워드라인 저항을 분리 추적; 비FA는 사용하지 않음.
     * 설정자: compute_C()에서 r_w_metal × num_cols로 계산.
     * 읽는 자: Mat에서 RC 지연 계산에 사용.
     * 동기화: 생성 시 한 번 계산. */

    double  C_bl, C_bl_cam;
    /* [한국어] 비트라인 총 커패시턴스 [F].
     * C_bl    : SRAM/DRAM 비트라인 커패시턴스 (drain_C_ × num_rows + C_b_metal × num_rows).
     * C_bl_cam: FA에서 CAM 비트라인 커패시턴스 (검색라인과 구분).
     * 설정자: compute_C()에서 계산.
     * 읽는 자: Mat에서 비트라인 방전 지연 계산에 사용.
     * 동기화: 생성 시 한 번 계산. */
  private:

    void compute_C();  // compute bitline and wordline capacitance
};



#endif

