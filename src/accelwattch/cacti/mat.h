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
 * [한국어 설명] CACTI 캐시 Mat(매트) 모델 헤더 (mat.h)
 *
 * === 파일의 역할 ===
 * CACTI에서 캐시 뱅크의 기본 타이밍/전력 단위인 "mat"를 모델링하는 Mat 클래스를 선언한다.
 * mat은 여러 개의 서브어레이(Subarray)를 포함하는 2차원 배열로, 행 디코더·예비디코더·
 * 비트라인 멀티플렉서·센스앰프 멀티플렉서·출력 드라이버 등 캐시 접근 경로의 모든 단계를 담는다.
 * compute_delays()는 mat 전체의 타이밍 경로를 계산하고, compute_power_energy()는 각
 * 서브회로의 동적·누설·게이트 누설 전력을 계산한다. AccelWattch는 이 결과를 GPU L1/L2
 * 캐시 에너지 추정의 핵심 입력으로 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 모델 계층: UCA → Mat (타이밍/전력 핵심 단위) → Subarray → (셀 어레이)
 * UCA::UCA() 생성자에서 Mat 객체를 생성하고, compute_delays()→compute_power_energy() 순서로 호출.
 * Htree2가 뱅크 간 배선을 담당하는 반면, Mat는 배선 내부의 어레이 접근 지연·전력을 담당한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, AccelWattch 초기화 단계 (시뮬레이션 시작 시 1회).
 * 호출 체인: UCA::UCA() → Mat::Mat() → compute_delays() → compute_power_energy()
 *
 * === 타 모듈과의 연결 ===
 * 의존: Decoder(행/비트 디코더), Predec(예비디코더), PredecBlk/PredecBlkDrv(예비디코더 블록),
 *       Wire(서브어레이 출력 배선), Driver(프리차지/드라이버), Subarray(셀 어레이 크기/지연),
 *       parameter.h(DynamicParameter — Ndwl/Ndbl/Nspd 등 파티션 파라미터)
 * 상위: UCA — Mat 결과(delay, power, area)를 읽어 뱅크 수준 결과에 합산
 * 데이터 흐름: DynamicParameter(캐시 구조 파라미터) → Mat 서브회로 초기화 →
 *              compute_delays() → 각 단계 지연 누적 → compute_power_energy() → 전력 누적
 *
 * === 주요 함수/구조체 요약 ===
 * Mat()                  : 생성자 — DynamicParameter로 모든 서브회로 객체 초기화
 * compute_delays()       : mat 전체 타이밍 경로 계산 (행 디코더→비트라인→SA→출력 드라이버)
 * compute_power_energy() : mat 내 모든 서브회로의 동적·누설 전력 계산
 * compute_bitline_delay(): 비트라인 RC 지연 계산 (내부 private 함수)
 * compute_sa_delay()     : 센스앰프 지연 계산 (내부 private 함수)
 */

#ifndef __MAT_H__
#define __MAT_H__

#include "component.h" // [한국어] Component 기본 클래스 (delay, power, area 필드)
#include "decoder.h"   // [한국어] Decoder, Predec, PredecBlk, PredecBlkDrv — 디코더 계층
#include "wire.h"      // [한국어] Wire 클래스 — 서브어레이 출력 배선 RC 모델
#include "subarray.h"  // [한국어] Subarray 클래스 — 셀 어레이 크기·비트라인 커패시턴스

class Mat : public Component
/* [한국어] mat 타이밍/전력 모델 클래스.
 * Component를 상속하여 delay, power, area를 최종 결과로 저장한다.
 * 하나의 Mat 객체는 캐시 뱅크 내 하나의 mat(서브어레이 2D 배열)을 나타내며,
 * UCA에서 뱅크 전체 결과를 산출할 때 Mat 결과를 기반으로 한다. */
{
  public:
    /*
     * [한국어]
     * Mat::Mat - mat 타이밍/전력 모델 생성자
     *
     * @dyn_p : DynamicParameter const 참조 — Ndwl/Ndbl/Nspd/num_mats/is_dram 등 캐시 구조 파라미터
     * @return: (생성자) — 모든 서브회로 객체(디코더, 예비디코더, 드라이버 등) 초기화 완료
     *
     * DynamicParameter에서 캐시 파티션 정보를 받아 행 디코더, 비트 멀티플렉서 디코더,
     * 센스앰프 멀티플렉서 디코더, 예비디코더, 서브어레이 출력 배선, 프리차지 드라이버 등을
     * new로 할당하여 초기화한다. 소멸자에서 delete로 해제.
     *
     * 호출 체인:
     *   UCA::UCA() → [Mat::Mat()] → (서브회로 객체들 생성)
     */
    Mat(const DynamicParameter & dyn_p);
    ~Mat();

    /*
     * [한국어]
     * Mat::compute_delays - mat 전체 타이밍 경로 계산
     *
     * @inrisetime : mat 입력 신호의 상승 시간(초) — H-tree 출력 rise time에서 전달
     * @return     : mat 출력 신호의 상승 시간(초) — 다음 단(H-tree 출력) 입력으로 사용
     *
     * 행 디코더 → 예비디코더 → 비트라인 → 센스앰프 → SA 멀티플렉서 → 출력 드라이버 순서로
     * 각 단의 지연을 계산하여 누적한다. is_dram 여부와 is_fa(완전 연관) 여부에 따라
     * 추가적인 매치라인/서치라인 지연 경로를 포함할 수 있다.
     * 결과는 delay_before_decoder, delay_bitline, delay_sa, delay_subarray_out_drv 등
     * 각 단계 필드와 Component::delay에 저장된다.
     *
     * 호출 체인:
     *   UCA::compute_delays() → [Mat::compute_delays()] → compute_bitline_delay(), compute_sa_delay() 등
     */
    double compute_delays(double inrisetime);  // return outrisetime

    /*
     * [한국어]
     * Mat::compute_power_energy - mat 내 모든 서브회로의 전력 계산
     *
     * @return: (void) — power_row_decoders, power_bitline, power_sa 등 각 필드와
     *                   Component::power에 동적·누설·게이트 누설 전력을 저장
     *
     * compute_delays() 완료 후 호출되어 각 서브회로의 스위칭 에너지와 누설 전력을 계산한다.
     * 행 디코더, 비트 멀티플렉서 디코더, SA 멀티플렉서 디코더, 비트라인, 센스앰프,
     * 서브어레이 출력 드라이버, (CAM이면) 서치라인/매치라인 전력을 포함한다.
     *
     * 호출 체인:
     *   UCA::compute_power_energy() → [Mat::compute_power_energy()]
     */
    void compute_power_energy();

    const DynamicParameter & dp;
    /* [한국어] DynamicParameter const 참조 — 이 mat이 속한 캐시의 구조 파라미터.
     * 설정자: 생성자 초기화 리스트에서 dyn_p 인수로 한 번만 바인딩.
     * 읽는 자: compute_delays(), compute_power_energy(), 각 private 함수에서 Ndwl/Ndbl/
     *          deg_bl_muxing/is_dram/is_fa/num_mats 등 파라미터를 참조.
     * 값 범위: UCA가 최적 파라미터로 결정한 DynamicParameter 객체 (유효한 상태).
     * 동기화: const 참조이므로 수정 불가. 단일 스레드 접근. */

    // TODO: clean up pointers and powerDefs below
    Decoder * row_dec;
    /* [한국어] 행 디코더(Row Decoder) 객체 포인터.
     * 설정자: 생성자에서 new Decoder(...)로 초기화.
     * 읽는 자: compute_delays()에서 delay_before_decoder 계산 시 row_dec->delay 참조.
     *          compute_power_energy()에서 power_row_decoders에 전력 합산.
     * 값 범위: 유효한 Decoder 포인터 (NULL 불가). 소멸자에서 delete.
     * 동기화: 단일 스레드 접근. */

    Decoder * bit_mux_dec;
    /* [한국어] 비트 멀티플렉서 디코더(Bit Mux Decoder) 객체 포인터.
     * 설정자: 생성자에서 new Decoder(...)로 초기화.
     * 읽는 자: compute_delays()에서 비트 mux 경로 지연 계산.
     *          compute_power_energy()에서 power_bit_mux_decoders에 전력 합산.
     * 값 범위: 유효한 Decoder 포인터. 동기화: 단일 스레드. */

    Decoder * sa_mux_lev_1_dec;
    /* [한국어] 센스앰프 멀티플렉서 레벨 1 디코더 포인터.
     * SA 멀티플렉서는 여러 서브어레이의 sense amp 출력 중 하나를 선택한다.
     * 설정자: 생성자에서 초기화. 읽는 자: compute_power_energy()에서 레벨 1 전력 합산.
     * 동기화: 단일 스레드. */

    Decoder * sa_mux_lev_2_dec;
    /* [한국어] 센스앰프 멀티플렉서 레벨 2 디코더 포인터.
     * Ndsam_lev_2 단계의 SA mux를 담당한다.
     * 설정자: 생성자에서 초기화. 읽는 자: compute_power_energy()에서 레벨 2 전력 합산.
     * 동기화: 단일 스레드. */

    PredecBlk * dummy_way_sel_predec_blk1;
    /* [한국어] 웨이 선택 예비디코더 블록 1 (더미 — 실제 연산에서 미사용하는 경우가 있음).
     * 설정자: 생성자에서 초기화. 읽는 자: Predec 구성 시 파라미터로 전달.
     * 동기화: 단일 스레드. */

    PredecBlk * dummy_way_sel_predec_blk2;
    /* [한국어] 웨이 선택 예비디코더 블록 2 (더미). 동일한 역할.
     * 설정자: 생성자에서 초기화. 동기화: 단일 스레드. */

    PredecBlkDrv * way_sel_drv1;
    /* [한국어] 웨이 선택 예비디코더 블록 드라이버 1.
     * 예비디코더 블록의 출력을 구동하는 드라이버 회로.
     * 설정자: 생성자에서 초기화. 읽는 자: compute_delays()에서 웨이 선택 경로 지연 계산.
     * 동기화: 단일 스레드. */

    PredecBlkDrv * dummy_way_sel_predec_blk_drv2;
    /* [한국어] 웨이 선택 예비디코더 블록 드라이버 2 (더미).
     * 설정자: 생성자에서 초기화. 동기화: 단일 스레드. */

    Predec * r_predec;
    /* [한국어] 행 예비디코더(Row Predecoder) 객체 포인터.
     * 전체 행 어드레스를 행 디코더가 받기 전에 미리 부분 디코딩하는 회로.
     * 설정자: 생성자에서 new Predec(...)로 초기화.
     * 읽는 자: compute_delays()에서 예비디코더 지연(delay_before_decoder의 일부) 계산.
     *          compute_power_energy()에서 power_row_decoders에 합산.
     * 동기화: 단일 스레드. */

    Predec * b_mux_predec;
    /* [한국어] 비트 멀티플렉서 예비디코더 포인터.
     * 비트 mux 선택 신호를 미리 디코딩. 설정자: 생성자 초기화.
     * 읽는 자: compute_power_energy()에서 power_bit_mux_decoders에 합산.
     * 동기화: 단일 스레드. */

    Predec * sa_mux_lev_1_predec;
    /* [한국어] SA 멀티플렉서 레벨 1 예비디코더. 설정자: 생성자 초기화.
     * 읽는 자: compute_power_energy()에서 power_sa_mux_lev_1_decoders에 합산.
     * 동기화: 단일 스레드. */

    Predec * sa_mux_lev_2_predec;
    /* [한국어] SA 멀티플렉서 레벨 2 예비디코더. 설정자: 생성자 초기화.
     * 읽는 자: compute_power_energy()에서 power_sa_mux_lev_2_decoders에 합산.
     * 동기화: 단일 스레드. */

    Wire   * subarray_out_wire;
    /* [한국어] 서브어레이 출력 배선(Wire) 객체 포인터.
     * SA 출력 신호가 mat 출력 드라이버까지 이동하는 배선의 RC 지연·전력을 모델링.
     * 설정자: 생성자에서 new Wire(...)로 초기화.
     * 읽는 자: compute_delays()에서 delay_subarray_out_drv_htree 계산 시 사용.
     *          compute_power_energy()에서 power_subarray_out_drv에 합산.
     * 동기화: 단일 스레드. */

    Driver * bl_precharge_eq_drv;
    /* [한국어] 비트라인 프리차지/이퀄라이즈 드라이버 포인터.
     * 비트라인을 Vdd/2 또는 Vdd로 프리차지하는 회로를 모델링.
     * 설정자: 생성자에서 초기화.
     * 읽는 자: compute_power_energy()에서 power_bl_precharge_eq_drv에 합산.
     * 동기화: 단일 스레드. */

    Driver * cam_bl_precharge_eq_drv; //bitline pre-charge circuit is separated for CAM and RAM arrays.
    /* [한국어] CAM 비트라인 프리차지 드라이버 포인터.
     * CAM(Content Addressable Memory)용 비트라인 프리차지 — RAM과 별도 모델링.
     * 설정자: 생성자에서 초기화 (is_fa가 true일 때 유효).
     * 읽는 자: compute_power_energy()에서 CAM 비트라인 전력 계산 시 사용.
     * 동기화: 단일 스레드. */

    Driver * ml_precharge_drv; //matchline prechange driver
    /* [한국어] 매치라인(Matchline) 프리차지 드라이버 포인터 (CAM 전용).
     * CAM에서 검색 결과 비교에 사용하는 매치라인을 Vdd로 프리차지하는 드라이버.
     * 설정자: 생성자에서 초기화 (is_fa true 시 유효).
     * 읽는 자: compute_delays()에서 delay_cam_ml_reset, compute_power_energy()에서 전력 합산.
     * 동기화: 단일 스레드. */

    Driver * sl_precharge_eq_drv; //searchline prechage driver
    /* [한국어] 서치라인(Searchline) 프리차지 드라이버 포인터 (CAM 전용).
     * CAM 검색 시 서치라인을 프리차지하는 드라이버.
     * 설정자: 생성자에서 초기화 (is_fa true 시 유효).
     * 읽는 자: compute_power_energy()에서 power_searchline_precharge에 합산.
     * 동기화: 단일 스레드. */

    Driver * sl_data_drv; //search line data driver
    /* [한국어] 서치라인 데이터 드라이버 포인터 (CAM 전용).
     * 검색 키 데이터를 서치라인으로 구동하는 드라이버.
     * 설정자: 생성자에서 초기화. 읽는 자: compute_delays()에서 delay_searchline 계산.
     * 동기화: 단일 스레드. */

    Driver * ml_to_ram_wl_drv; //search line data driver
    /* [한국어] 매치라인→RAM 워드라인 드라이버 포인터 (FA CAM 전용).
     * 완전 연관 캐시에서 매치라인 신호를 RAM 섹션의 워드라인으로 변환하는 드라이버.
     * 설정자: 생성자에서 초기화 (is_fa true 시 유효).
     * 읽는 자: compute_delays()에서 delay_fa_ram_wl, compute_power_energy()에서 전력 합산.
     * 동기화: 단일 스레드. */


    powerDef power_row_decoders;
    /* [한국어] 행 디코더(예비디코더 포함) 전력 합계 (동적 + 누설 + 게이트 누설).
     * 설정자: compute_power_energy()에서 r_predec 및 row_dec 전력을 합산하여 저장.
     * 읽는 자: UCA::compute_power_energy()에서 뱅크 전체 전력 계산 시 참조.
     * 값 범위: 0 이상 실수 (J, W 단위). 동기화: 단일 스레드. */

    powerDef power_bit_mux_decoders;
    /* [한국어] 비트 멀티플렉서 디코더 전력 합계.
     * 설정자: compute_power_energy()에서 b_mux_predec 및 bit_mux_dec 전력 합산.
     * 읽는 자: UCA에서 뱅크 수준 전력 집계 시 참조.
     * 동기화: 단일 스레드. */

    powerDef power_sa_mux_lev_1_decoders;
    /* [한국어] SA 멀티플렉서 레벨 1 디코더 전력 합계.
     * 설정자: compute_power_energy()에서 sa_mux_lev_1_predec 및 sa_mux_lev_1_dec 전력 합산.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    powerDef power_sa_mux_lev_2_decoders;
    /* [한국어] SA 멀티플렉서 레벨 2 디코더 전력 합계.
     * 설정자: compute_power_energy()에서 합산.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    powerDef power_fa_cam;  // TODO: leakage power is not computed yet
    /* [한국어] 완전 연관 CAM 전력 합계 (동적 전력만; 누설은 미구현 상태).
     * 설정자: compute_power_energy()에서 CAM 관련 구성 요소 전력 합산.
     * 읽는 자: UCA 집계. 값 범위: 0 이상 (누설은 0으로 고정).
     * 동기화: 단일 스레드. */

    powerDef power_bl_precharge_eq_drv;
    /* [한국어] 비트라인 프리차지/이퀄라이즈 드라이버 전력.
     * 설정자: compute_power_energy()에서 bl_precharge_eq_drv 전력 저장.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    powerDef power_subarray_out_drv;
    /* [한국어] 서브어레이 출력 드라이버(및 출력 배선) 전력.
     * 설정자: compute_power_energy()에서 subarray_out_wire 등 전력 합산.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    powerDef power_cam_all_active;
    /* [한국어] CAM 전체 활성 상태(모든 mat 동시 검색)에서의 전력.
     * 설정자: compute_power_energy()에서 CAM 구성 시 계산.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    powerDef power_searchline_precharge;
    /* [한국어] 서치라인 프리차지 전력 (CAM 전용).
     * 설정자: compute_power_energy()에서 sl_precharge_eq_drv 전력 저장.
     * 동기화: 단일 스레드. */

    powerDef power_matchline_precharge;
    /* [한국어] 매치라인 프리차지 전력 (CAM 전용).
     * 설정자: compute_power_energy()에서 ml_precharge_drv 전력 저장.
     * 동기화: 단일 스레드. */

    powerDef power_ml_to_ram_wl_drv;
    /* [한국어] 매치라인→RAM 워드라인 드라이버 전력 (FA CAM 전용).
     * 설정자: compute_power_energy()에서 ml_to_ram_wl_drv 전력 저장.
     * 동기화: 단일 스레드. */

    double   delay_fa_tag, delay_cam;
    /* [한국어] delay_fa_tag : 완전 연관 캐시에서 태그 비교(CAM 검색) 전 단계 지연.
     * delay_cam   : CAM 전체 접근 지연 (서치라인 구동 + 매치라인 평가).
     * 설정자: compute_delays()에서 compute_cam_delay() 호출 결과로 저장.
     * 읽는 자: compute_delays()에서 전체 mat 지연 경로에 합산.
     * 동기화: 단일 스레드. */

    double   delay_before_decoder;
    /* [한국어] 예비디코더까지의 지연 (H-tree → 예비디코더 입력).
     * 설정자: compute_delays()에서 r_predec->delay로 초기화.
     * 읽는 자: compute_delays() 내 전체 경로 합산.
     * 동기화: 단일 스레드. */

    double   delay_bitline;
    /* [한국어] 비트라인 방전/충전 지연 (워드라인 활성화 → 비트라인 충분히 스윙).
     * 설정자: compute_bitline_delay() 반환값.
     * 읽는 자: compute_delays()에서 전체 경로에 합산.
     * 동기화: 단일 스레드. */

    double   delay_wl_reset;
    /* [한국어] 워드라인 리셋(비활성화) 지연 — 접근 후 워드라인을 내리는 데 걸리는 시간.
     * 설정자: compute_delays()에서 계산.
     * 읽는 자: 전체 사이클 타임 계산 시 참조.
     * 동기화: 단일 스레드. */

    double   delay_bl_restore;
    /* [한국어] 비트라인 복원(프리차지) 지연 — 접근 완료 후 비트라인을 다시 프리차지하는 시간.
     * 설정자: compute_delays()에서 계산.
     * 읽는 자: 사이클 타임 계산. 동기화: 단일 스레드. */

    double   delay_searchline;
    /* [한국어] CAM 서치라인 구동 지연 (서치라인 드라이버 → 서치라인 끝까지).
     * 설정자: compute_cam_delay()에서 sl_data_drv 지연으로 초기화.
     * 읽는 자: compute_delays()에서 CAM 경로에 합산.
     * 동기화: 단일 스레드. */

    double   delay_matchchline;
    /* [한국어] CAM 매치라인 평가 지연 (서치라인 구동 → 매치라인 결과 확정).
     * 설정자: compute_cam_delay()에서 계산.
     * 읽는 자: compute_delays()에서 CAM 경로에 합산.
     * 동기화: 단일 스레드. */

    double   delay_cam_sl_restore;
    /* [한국어] CAM 서치라인 복원(프리차지) 지연.
     * 설정자: compute_cam_delay(). 읽는 자: 사이클 타임 계산. 동기화: 단일 스레드. */

    double   delay_cam_ml_reset;
    /* [한국어] CAM 매치라인 리셋 지연 — 매치라인을 다시 프리차지하는 데 걸리는 시간.
     * 설정자: compute_cam_delay(). 읽는 자: 사이클 타임 계산. 동기화: 단일 스레드. */

    double   delay_fa_ram_wl;
    /* [한국어] 완전 연관 캐시에서 매치라인 히트 후 RAM 섹션 워드라인 활성화까지의 지연.
     * 설정자: compute_delays()에서 ml_to_ram_wl_drv->delay로 초기화.
     * 읽는 자: FA 캐시 전체 접근 지연 계산. 동기화: 단일 스레드. */

    double   delay_hit_miss_reset;
    /* [한국어] CAM 히트/미스 결과 리셋 지연.
     * 설정자: compute_delays(). 동기화: 단일 스레드. */

    double   delay_hit_miss;
    /* [한국어] CAM 히트/미스 결과를 확정하는 지연.
     * 설정자: compute_delays(). 읽는 자: 최종 mat 지연 경로. 동기화: 단일 스레드. */

    Subarray subarray;
    /* [한국어] 이 mat에 포함된 서브어레이 모델 객체 (비포인터 멤버 — mat 소멸 시 자동 해제).
     * Subarray는 셀 어레이 크기(b_w, b_h), 비트라인 커패시턴스(C_bl), 워드라인 저항 등을 제공.
     * 설정자: Mat 생성자 초기화 리스트에서 Subarray(dp) 형태로 초기화.
     * 읽는 자: compute_delays()에서 비트라인 지연/전력 계산 시 subarray.C_bl, subarray.delay 참조.
     * 동기화: 단일 스레드. */

    powerDef power_bitline, power_searchline, power_matchline;
    /* [한국어] power_bitline   : 비트라인 스위칭 전력 (읽기/쓰기 시 셀→비트라인 전하 이동).
     * power_searchline : CAM 서치라인 구동 전력.
     * power_matchline  : CAM 매치라인 평가 전력.
     * 설정자: compute_power_energy()에서 각 경로별로 계산하여 저장.
     * 읽는 자: UCA에서 뱅크 전력 집계 시 참조.
     * 동기화: 단일 스레드. */

    double   per_bitline_read_energy;
    /* [한국어] 비트라인 1개(1비트)를 읽을 때 소비되는 에너지 (J 단위).
     * 설정자: compute_power_energy()에서 0.5*C_bl*Vdd² 형태로 계산.
     * 읽는 자: power_bitline 계산 시 num_sa_subarray를 곱하여 전체 비트라인 전력 산출.
     * 동기화: 단일 스레드. */

    int      deg_bl_muxing;
    /* [한국어] 비트라인 멀티플렉싱 차수 (Degree of Bit-Line Muxing).
     * 몇 개의 비트라인이 하나의 SA(Sense Amplifier)를 공유하는지를 나타낸다.
     * 설정자: 생성자에서 dp.deg_bl_muxing으로 초기화.
     * 읽는 자: compute_bit_mux_sa_precharge_sa_mux_wr_drv_wr_mux_h()에서 높이 계산.
     * 값 범위: 1, 2, 4, 8 등 2의 거듭제곱.
     * 동기화: 단일 스레드. */

    int      num_act_mats_hor_dir;
    /* [한국어] 수평 방향으로 동시에 활성화되는 mat 수.
     * 동시 활성 mat 수가 클수록 출력 비트 수가 증가하고 데이터 버스 폭이 넓어진다.
     * 설정자: 생성자에서 dp.num_act_mats_hor_dir으로 초기화.
     * 읽는 자: compute_delays()에서 SA 멀티플렉서 구성 계산 시 사용.
     * 동기화: 단일 스레드. */

    double   delay_writeback;
    /* [한국어] 쓰기 동작 시 비트라인 복원까지 포함한 추가 지연 (DRAM 쓰기 백 등).
     * 설정자: compute_delays()에서 계산. 읽는 자: 사이클 타임 계산.
     * 동기화: 단일 스레드. */

    Area     cell, cam_cell;
    /* [한국어] cell    : SRAM 셀 1개의 면적 (w × h, 미터 단위).
     * cam_cell : CAM 셀(SRAM + 비교기) 1개의 면적.
     * 설정자: 생성자에서 g_tp.sram / g_tp.cam 파라미터로 초기화.
     * 읽는 자: compute_delays()에서 비트라인 길이 = num_r_subarray × cell.h 계산.
     * 동기화: 단일 스레드. */

    bool     is_dram, is_fa, pure_cam, camFlag;
    /* [한국어] is_dram  : DRAM 셀 타입 여부 (true이면 DRAM 셀 RC 파라미터 사용).
     * is_fa    : 완전 연관(Fully Associative) 캐시 여부.
     * pure_cam : CAM 전용 구조 여부 (데이터 배열 없이 태그만 있는 경우).
     * camFlag  : CAM 경로 계산 활성화 플래그.
     * 설정자: 생성자에서 dp.is_dram, dp.fully_assoc, dp.pure_cam으로 초기화.
     * 읽는 자: compute_delays()/compute_power_energy()에서 경로 분기 조건으로 사용.
     * 동기화: 단일 스레드. */

    int      num_mats;
    /* [한국어] 이 뱅크 내 mat의 총 개수 (= num_mats_h_dir × num_mats_v_dir).
     * 설정자: 생성자에서 dp.num_mats로 초기화.
     * 읽는 자: compute_power_energy()에서 누설 전력 계산 시 전체 mat 수를 고려.
     * 값 범위: 1 이상 정수. 동기화: 단일 스레드. */

    powerDef power_sa;
    /* [한국어] 센스앰프(SA) 전력 합계 (동적 + 누설).
     * 설정자: compute_power_energy()에서 SA 스위칭 에너지와 누설을 계산하여 저장.
     * 읽는 자: UCA에서 뱅크 전력 집계.
     * 동기화: 단일 스레드. */

    double   delay_sa;
    /* [한국어] 센스앰프 지연 (비트라인 스윙 검출 → SA 출력 확정까지).
     * 설정자: compute_sa_delay() 반환값.
     * 읽는 자: compute_delays()에서 전체 mat 지연에 합산.
     * 동기화: 단일 스레드. */

    double   leak_power_sense_amps_closed_page_state;
    /* [한국어] 페이지 미활성(closed page) 상태에서 SA의 누설 전력.
     * 설정자: compute_power_energy()에서 is_dram 기반 계산.
     * 읽는 자: UCA에서 대기 모드 전력 산출 시 참조.
     * 동기화: 단일 스레드. */

    double   leak_power_sense_amps_open_page_state;
    /* [한국어] 페이지 활성(open page) 상태에서 SA의 누설 전력.
     * 설정자: compute_power_energy()에서 계산.
     * 읽는 자: UCA에서 활성 모드 대기 전력 산출.
     * 동기화: 단일 스레드. */

    double   delay_subarray_out_drv;
    /* [한국어] 서브어레이 출력 드라이버 지연 (SA 출력 → mat 출력 드라이버 통과 후).
     * 설정자: compute_subarray_out_drv() 반환값.
     * 읽는 자: compute_delays()에서 전체 지연에 합산.
     * 동기화: 단일 스레드. */

    double   delay_subarray_out_drv_htree;
    /* [한국어] 서브어레이 출력 드라이버 + H-tree 배선 지연 합계.
     * 설정자: compute_delays()에서 delay_subarray_out_drv + subarray_out_wire->delay로 계산.
     * 읽는 자: UCA에서 mat 출력 지연으로 참조.
     * 동기화: 단일 스레드. */

    double   delay_comparator;
    /* [한국어] 태그 비교기 지연 (FA 캐시에서 태그 검색 결과 비교에 사용).
     * 설정자: compute_comparator_delay() 반환값.
     * 읽는 자: compute_delays()에서 FA 캐시 경로에 합산.
     * 동기화: 단일 스레드. */

    powerDef power_comparator;
    /* [한국어] 태그 비교기 전력 (FA 캐시 전용).
     * 설정자: compute_power_energy()에서 계산.
     * 읽는 자: UCA 집계. 동기화: 단일 스레드. */

    int      num_do_b_mat;
    /* [한국어] 이 mat에서 출력되는 데이터 비트 수 (Data Output bits per mat).
     * 설정자: 생성자에서 dp.num_do_b_mat으로 초기화.
     * 읽는 자: compute_power_energy()에서 SA 전력 계산 시 SA 개수 기준으로 사용.
     * 동기화: 단일 스레드. */

    int      num_so_b_mat;
    /* [한국어] 이 mat에서 출력되는 검색(Search Output) 비트 수 (CAM 전용).
     * 설정자: 생성자에서 dp.num_so_b_mat으로 초기화.
     * 읽는 자: compute_power_energy()에서 CAM SA 전력 계산 시 사용.
     * 동기화: 단일 스레드. */

    int      num_sa_subarray;
    /* [한국어] 서브어레이 1개당 SA(센스앰프) 수 = num_c_subarray / deg_bl_muxing.
     * 설정자: 생성자에서 계산하여 초기화.
     * 읽는 자: compute_power_energy()에서 SA 동적 전력 = per_SA_energy × num_sa_subarray 계산.
     * 동기화: 단일 스레드. */

    int      num_sa_subarray_search;
    /* [한국어] CAM 서브어레이 1개당 검색용 SA 수 (CAM 전용).
     * 설정자: 생성자에서 계산. 읽는 자: compute_power_energy()에서 CAM SA 전력 계산.
     * 동기화: 단일 스레드. */

    double   C_bl;
    /* [한국어] 비트라인 총 커패시턴스 (F 단위) = 셀 커패시턴스 × 행 수 + 배선 커패시턴스.
     * 설정자: 생성자에서 subarray.C_bl로 초기화.
     * 읽는 자: compute_bitline_delay()에서 RC 지연 계산, compute_power_energy()에서 스위칭 에너지 계산.
     * 값 범위: 0 이상 실수 (fF ~ pF 범위).
     * 동기화: 단일 스레드. */

    uint32_t num_subarrays_per_mat;  // the number of subarrays in a mat
    /* [한국어] mat 1개 내의 서브어레이 수 = num_subarrays_per_row × (mat 세로 서브어레이 수).
     * 설정자: 생성자에서 dp 파라미터로 계산하여 초기화.
     * 읽는 자: compute_power_energy()에서 누설 전력 = 서브어레이당 누설 × 전체 서브어레이 수.
     * 값 범위: 1 이상 정수. 동기화: 단일 스레드. */

    uint32_t num_subarrays_per_row;  // the number of subarrays in a row of a mat
    /* [한국어] mat 가로 1행에 배치된 서브어레이 수.
     * 설정자: 생성자에서 dp.Ndwl / dp.num_mats_h_dir 등으로 계산.
     * 읽는 자: 서브어레이 출력 배선 길이 계산 시 사용.
     * 값 범위: 1 이상 정수. 동기화: 단일 스레드. */


  private:
    /*
     * [한국어]
     * compute_bit_mux_sa_precharge_sa_mux_wr_drv_wr_mux_h - 비트 mux/SA/프리차지/쓰기 드라이버 높이 계산
     *
     * @return: 해당 회로들의 높이 합 (미터 단위) — 서브어레이 레이아웃 면적 계산에 사용
     *
     * 비트라인 멀티플렉서, SA, 프리차지 회로, SA 멀티플렉서, 쓰기 드라이버/mux의
     * 물리적 높이(cell 단위)를 계산한다. 면적 예측과 배선 길이 계산에 활용.
     *
     * 호출 체인: compute_delays() 또는 Mat() 생성자 → [compute_bit_mux_sa_precharge_sa_mux_wr_drv_wr_mux_h()]
     */
    double compute_bit_mux_sa_precharge_sa_mux_wr_drv_wr_mux_h();

    /*
     * [한국어]
     * width_write_driver_or_write_mux - 쓰기 드라이버/쓰기 멀티플렉서 폭 계산
     *
     * @return: 쓰기 드라이버 또는 쓰기 mux의 물리적 폭 (미터) — 면적 계산에 사용
     *
     * 호출 체인: compute_bit_mux_sa_precharge_sa_mux_wr_drv_wr_mux_h() → [width_write_driver_or_write_mux()]
     */
    double width_write_driver_or_write_mux();

    /*
     * [한국어]
     * compute_comparators_height - 태그 비교기 배열 높이 계산 (FA 캐시 전용)
     *
     * @tagbits                 : 태그 비트 수
     * @number_ways_in_mat      : 이 mat 내 웨이(way) 수
     * @subarray_mem_cell_area_w: 서브어레이 셀 면적의 폭
     * @return: 비교기 회로 배열의 총 높이 (미터)
     *
     * 호출 체인: compute_delays() → [compute_comparators_height()]
     */
    double compute_comparators_height(int tagbits, int number_ways_in_mat, double subarray_mem_cell_area_w);

    /*
     * [한국어]
     * compute_cam_delay - CAM 접근 경로(서치라인+매치라인) 지연 계산
     *
     * @inrisetime: 입력 신호 상승 시간 (초)
     * @return    : CAM 접근 경로의 출력 rise time (초)
     *
     * 서치라인 구동 지연 + 매치라인 평가 지연을 단계적으로 계산한다.
     * delay_searchline, delay_matchchline 등 CAM 관련 지연 필드를 채운다.
     *
     * 호출 체인: compute_delays() → [compute_cam_delay()]
     */
    double compute_cam_delay(double inrisetime);

    /*
     * [한국어]
     * compute_bitline_delay - 비트라인 RC 지연 계산
     *
     * @inrisetime: 워드라인 활성화 후 비트라인 입력 rise time (초)
     * @return    : 비트라인 충분히 스윙 후의 rise time (초)
     *
     * 비트라인 저항(R_bl), 커패시턴스(C_bl), 셀 전류를 이용하여 비트라인 지연을 계산.
     * delay_bitline 필드를 채운다.
     *
     * 호출 체인: compute_delays() → [compute_bitline_delay()]
     */
    double compute_bitline_delay(double inrisetime);

    /*
     * [한국어]
     * compute_sa_delay - 센스앰프 지연 계산
     *
     * @inrisetime: 비트라인 충분히 스윙 후 SA 입력 rise time (초)
     * @return    : SA 출력 확정 후의 rise time (초)
     *
     * 래치형 SA의 재생(regeneration) 지연을 horowitz 모델로 계산.
     * delay_sa 필드를 채운다.
     *
     * 호출 체인: compute_delays() → [compute_sa_delay()]
     */
    double compute_sa_delay(double inrisetime);

    /*
     * [한국어]
     * compute_subarray_out_drv - 서브어레이 출력 드라이버 지연 계산
     *
     * @inrisetime: SA 출력 rise time (초)
     * @return    : 서브어레이 출력 드라이버 출력 rise time (초)
     *
     * SA 출력을 mat 출력 버스로 구동하는 드라이버의 RC 지연을 계산.
     * delay_subarray_out_drv 필드를 채운다.
     *
     * 호출 체인: compute_delays() → [compute_subarray_out_drv()]
     */
    double compute_subarray_out_drv(double inrisetime);

    /*
     * [한국어]
     * compute_comparator_delay - 태그 비교기 지연 계산 (FA 캐시 전용)
     *
     * @inrisetime: 비교기 입력 rise time (초)
     * @return    : 비교기 출력 rise time (초)
     *
     * 완전 연관 캐시에서 태그 비트를 병렬로 비교하는 회로의 지연을 계산.
     * delay_comparator 필드를 채운다.
     *
     * 호출 체인: compute_delays() → [compute_comparator_delay()]
     */
    double compute_comparator_delay(double inrisetime);

    int RWP;
    /* [한국어] 읽기/쓰기 포트(Read-Write Port) 수.
     * 설정자: 생성자에서 g_ip->num_rw_ports로 초기화.
     * 읽는 자: compute_delays() 및 compute_power_energy()에서 포트 수에 비례한 전력/면적 계산.
     * 값 범위: 0 이상 정수. 동기화: 단일 스레드. */

    int ERP;
    /* [한국어] 읽기 전용 포트(Extra Read Port) 수.
     * 설정자: 생성자에서 g_ip->num_rd_ports로 초기화.
     * 읽는 자: 포트 수에 비례한 면적/전력 계산. 동기화: 단일 스레드. */

    int EWP;
    /* [한국어] 쓰기 전용 포트(Extra Write Port) 수.
     * 설정자: 생성자에서 g_ip->num_wr_ports로 초기화.
     * 읽는 자: 포트 수에 비례한 면적/전력 계산. 동기화: 단일 스레드. */

    int SCHP;
    /* [한국어] 검색 포트(Search Port) 수 (CAM 전용).
     * 설정자: 생성자에서 g_ip->num_search_ports로 초기화.
     * 읽는 자: CAM 검색 경로 면적/전력 계산 시 사용. 동기화: 단일 스레드. */
};



#endif
