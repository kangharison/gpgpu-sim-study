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
 * [한국어 설명] CACTI 캐시 뱅크(Bank) 헤더 (bank.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 모델에서 하나의 뱅크(Bank)를 표현하는 클래스를 선언한다. 뱅크는 여러 개의
 * 매트(Mat)로 구성되며, 각 매트는 서브어레이(Subarray)를 포함한다. Bank 클래스는
 * 매트 배열 구성(수평/수직 방향 매트 수), H-tree 글로벌 배선 네트워크(주소·데이터 입출력),
 * 그리고 지연·전력 집계 함수를 제공한다. CACTI가 최적 캐시 구성을 탐색할 때 Bank 객체가
 * 생성되어 면적·지연·전력을 평가받는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI 최적화 루프(cacti_interface.cc) → Bank 생성 → Mat → Subarray.
 * Bank는 CACTI 계층의 최상위 단위이며, 그 아래로 Mat (mat.cc) → Decoder/Bitline/Sense Amp
 * → Subarray가 계단식으로 존재한다. Htree2는 Bank 수준의 글로벌 H-tree 배선(주소/데이터)을
 * 모델링한다.
 *
 * === 타 모듈과의 연결 ===
 * - component.h: Component 기반 클래스 상속. area, power, delay 필드를 부여받음.
 * - mat.h: Bank가 Mat mat 멤버를 포함. 매트 레벨 지연·전력이 Bank에 합산됨.
 * - htree2.h: H-tree 글로벌 배선 모델 포인터 5개 (주소/데이터 입출력, 검색 입출력).
 * - decoder.h: Decoder 포함 관계 (mat.h를 통해 간접 포함).
 * - parameter.h: g_ip, g_tp 전역 파라미터, DynamicParameter 구조체 참조.
 *
 * === 주요 함수/구조체 요약 ===
 * - Bank(const DynamicParameter &): 생성자. 매트 배열 크기와 H-tree 생성.
 * - ~Bank(): 소멸자. H-tree 포인터 5개 동적 해제.
 * - compute_delays(double): 입력 rise time을 받아 출력 rise time을 반환. 매트 레벨 지연 위임.
 * - compute_power_energy(): 매트 전력과 H-tree 전력을 Bank 총 전력에 합산.
 */

#ifndef __BANK_H__
#define __BANK_H__

#include "component.h" // [한국어] Component 기반 클래스 — area, power, delay, cycle_time 필드 제공
#include "decoder.h"   // [한국어] Decoder 모델 (mat.h에서 간접 사용; Bank가 mat을 포함하므로 필요)
#include "mat.h"       // [한국어] Mat 클래스 — 캐시 매트 단위 모델 (Bank가 포함하는 핵심 서브 컴포넌트)
#include "htree2.h"    // [한국어] Htree2 — H-tree 글로벌 배선 모델 (주소·데이터 입출력 H-tree)

/*
 * [한국어]
 * Bank - CACTI 캐시 뱅크 전체를 표현하는 최상위 컴포넌트 클래스
 *
 * 하나의 Bank는 num_mats_ver_dir × num_mats_hor_dir 개의 Mat으로 구성된다.
 * 매트 간 주소/데이터 배선은 H-tree (Htree2) 구조로 모델링되어 뱅크 전체의
 * 지연과 전력에 합산된다. FA(완전 연관)/CAM 캐시는 추가로 검색 H-tree를 사용한다.
 *
 * 상속: Component (area, power, delay, cycle_time 필드를 부여받음)
 */
class Bank : public Component
{
  public:
    /*
     * [한국어]
     * Bank - 생성자: 매트 배열 구성 및 H-tree 생성
     *
     * @dyn_p: 동적 캐시 파라미터 구조체 (캐시 크기, 연관도, 포트 수 등 포함).
     *         DynamicParameter는 g_ip와 g_tp를 기반으로 CACTI 최적화 루프에서 계산됨.
     * @return: 없음 (생성자)
     *
     * 포트 수(RWP/ERP/EWP/SCHP)를 결정하고, 주소/데이터 총 비트 수를 계산한 뒤
     * htree_in_add, htree_in_data, htree_out_data (FA/CAM이면 추가로 htree_in/out_search)를
     * 동적 할당한다. 뱅크 면적은 htree_in_data의 면적으로 설정된다.
     *
     * 호출 체인: CACTI 최적화 루프(find_optimal_conf) → [Bank()] → Mat() → Htree2()
     */
    Bank(const DynamicParameter & dyn_p);

    /*
     * [한국어]
     * ~Bank - 소멸자: H-tree 동적 메모리 해제
     *
     * htree_in_add, htree_out_data, htree_in_data를 항상 해제하고,
     * FA/CAM 캐시인 경우 htree_in_search, htree_out_search도 해제한다.
     *
     * 호출 체인: CACTI 최적화 루프 종료 시 자동 호출
     */
    ~Bank();

    /*
     * [한국어]
     * compute_delays - 뱅크 전체 지연 계산
     *
     * @inrisetime: 입력 신호의 상승 시간(s). 배선/게이트 지연 계산의 초기 조건.
     * @return: 출력 신호의 상승 시간(s). 상위 계층이 이 값을 다음 단의 inrisetime으로 사용.
     *
     * Mat::compute_delays()에 위임한다. Bank 수준에서 추가 계산 없이 매트 레벨의
     * 지연 결과를 그대로 반환한다.
     *
     * 호출 체인: CACTI 지연 계산 루틴 → [compute_delays()] → Mat::compute_delays()
     */
    double compute_delays(double inrisetime);  // return outrisetime

    /*
     * [한국어]
     * compute_power_energy - 뱅크 전체 전력·에너지 계산
     *
     * @return: void. power.readOp.*(dynamic/leakage/gate_leakage) 및
     *          power.searchOp.* (FA/CAM의 경우)에 결과를 누적한다.
     *
     * Mat::compute_power_energy()를 먼저 호출한 뒤, 활성화된 매트 수(num_act_mats_hor_dir)에
     * 비례하는 동적 전력, 전체 매트 수(num_mats)에 비례하는 누설/게이트 누설,
     * H-tree (htree_in_add, htree_in_data, htree_out_data) 전력을 합산한다.
     * FA/CAM 캐시는 searchOp와 검색 H-tree 전력도 포함한다.
     *
     * 호출 체인: CACTI 전력 계산 → [compute_power_energy()] → Mat::compute_power_energy()
     */
    void   compute_power_energy();

    const DynamicParameter & dp;
    /* dp: 이 뱅크가 모델링하는 캐시 구성의 동적 파라미터 참조.
     * 캐시 크기, 연관도, 포트 수, 매트 구성(num_mats_h_dir, num_mats_v_dir 등)을 담는다.
     * 설정자: 생성자 초기화 리스트에서 const 참조로 바인딩.
     * 읽는 자: compute_power_energy() — dp.num_act_mats_hor_dir, dp.num_mats, dp.fully_assoc 등.
     * 동기화: const 참조이므로 변경 불가; CACTI 최적화 루프가 소유. */

    Mat   mat;
    /* mat: 이 뱅크를 구성하는 단일 매트(Mat) 인스턴스.
     * CACTI는 뱅크 내 모든 매트가 동일하다고 가정하므로 대표 매트 1개만 모델링하고,
     * 전력 계산 시 매트 수(num_mats)를 곱하여 총 전력을 산출한다.
     * 설정자: 생성자 초기화 리스트에서 dp를 전달하여 초기화.
     * 읽는 자: compute_delays(), compute_power_energy().
     * 동기화: 단일 스레드 CACTI 계산. */

    Htree2 *htree_in_add;
    /* htree_in_add: 주소(Address) 입력 H-tree 포인터.
     * 상위(컨트롤러)에서 뱅크로 주소 비트를 분배하는 H-tree 배선 모델.
     * 설정자: 생성자에서 new Htree2(Add_htree)로 생성.
     * 읽는 자: compute_power_energy() — power.readOp.dynamic/leakage/gate_leakage 누적.
     * 동기화: 소멸자에서 delete 처리. */

    Htree2 *htree_in_data;
    /* htree_in_data: 데이터 입력(쓰기) H-tree 포인터.
     * 상위에서 뱅크로 쓰기 데이터를 분배하는 H-tree 배선 모델.
     * 설정자: 생성자에서 new Htree2(Data_in_htree)로 생성.
     * 읽는 자: compute_power_energy() — leakage/gate_leakage 누적 (면적은 area.w/h 기준).
     * 동기화: 소멸자에서 delete 처리. */

    Htree2 *htree_out_data;
    /* htree_out_data: 데이터 출력(읽기) H-tree 포인터.
     * 뱅크에서 상위로 읽기 데이터를 수집하는 H-tree 배선 모델.
     * 설정자: 생성자에서 new Htree2(Data_out_htree)로 생성.
     * 읽는 자: compute_power_energy() — dynamic/leakage/gate_leakage 누적.
     * 동기화: 소멸자에서 delete 처리. */

    Htree2 *htree_in_search;
    /* htree_in_search: 검색 입력 H-tree 포인터 (FA/CAM 캐시 전용).
     * CAM/FA 캐시의 검색 키 분배 H-tree. 일반 캐시에서는 생성·해제 안 함.
     * 설정자: 생성자에서 dp.fully_assoc || dp.pure_cam인 경우에만 생성.
     * 읽는 자: compute_power_energy() — searchOp.dynamic, leakage/gate_leakage 누적.
     * 동기화: 소멸자에서 FA/CAM 조건부 delete 처리. */

    Htree2 *htree_out_search;
    /* htree_out_search: 검색 출력 H-tree 포인터 (FA/CAM 캐시 전용).
     * CAM/FA 캐시에서 검색 결과(히트/미스 + 데이터)를 수집하는 H-tree.
     * 설정자: 생성자에서 FA/CAM 조건부 생성. 읽는 자: compute_power_energy().
     * 동기화: 소멸자에서 FA/CAM 조건부 delete 처리. */

    int  num_addr_b_mat;
    /* num_addr_b_mat: 매트 하나에 라우팅되는 주소 비트 수.
     * 뱅크 내 매트 개수에 따른 주소 분할 깊이를 결정한다.
     * 설정자: 생성자에서 dyn_p.number_addr_bits_mat 복사.
     * 읽는 자: 생성자 내 num_addr_b_row_dec 등 파생 계산. */

    int  num_mats_hor_dir;
    /* num_mats_hor_dir: 뱅크 내 수평 방향 매트 수.
     * Htree2 생성 시 수평 방향 노드 수(num_mats_hor_dir*2) 계산에 사용.
     * 설정자: 생성자에서 dyn_p.num_mats_h_dir 복사. */

    int  num_mats_ver_dir;
    /* num_mats_ver_dir: 뱅크 내 수직 방향 매트 수.
     * Htree2 생성 시 수직 방향 노드 수(num_mats_ver_dir*2) 계산에 사용.
     * 설정자: 생성자에서 dyn_p.num_mats_v_dir 복사. */

    int  num_addr_b_row_dec;
    /* num_addr_b_row_dec: 행 디코더(row decoder)에 사용되는 주소 비트 수.
     * log2(서브어레이 행 수)로 계산된다.
     * 설정자: 생성자 마지막 단계에서 _log2(mat.subarray.num_rows)로 설정. */

    int  num_addr_b_routed_to_mat_for_act;
    /* num_addr_b_routed_to_mat_for_act: 매트 활성화(activation)에 라우팅되는 주소 비트 수.
     * 행 디코더 주소 비트 수(num_addr_b_row_dec)와 동일하게 설정된다.
     * 설정자: 생성자에서 num_addr_b_row_dec = num_addr_b_routed_to_mat_for_act 로 초기화. */

    int  num_addr_b_routed_to_mat_for_rd_or_wr;
    /* num_addr_b_routed_to_mat_for_rd_or_wr: 읽기/쓰기 시 매트에 라우팅되는 주소 비트 수.
     * 총 매트 주소 비트에서 행 디코더 비트를 뺀 나머지 (열 주소).
     * 설정자: 생성자에서 num_addr_b_mat - num_addr_b_row_dec 로 계산. */
};



#endif
