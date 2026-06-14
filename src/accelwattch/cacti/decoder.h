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
 * [한국어 설명] CACTI 디코더/프리디코더 모델 선언 (decoder.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 어레이의 행/열 디코더(Decoder), 프리디코더 블록(PredecBlk),
 * 프리디코더 블록 드라이버(PredecBlkDrv), 프리디코더(Predec), 드라이버(Driver)
 * 클래스를 선언한다. 이 클래스들은 캐시 주소를 워드라인 선택 신호로 변환하는
 * 회로 계층을 단계별로 모델링하며, 각각의 게이트 크기/면적/지연/누설을 분석한다.
 * 디코더 계층: PredecBlkDrv → PredecBlk → Predec → Decoder → 워드라인 구동.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI → decoder.h (디코더/프리디코더 계층 선언).
 * 호출 체인: Mat(캐시 어레이 타일) → Predec → PredecBlkDrv + PredecBlk → Decoder.
 * 각 클래스는 Component를 상속하여 area/power/delay를 누적.
 * 실행 컨텍스트: 호스트 유저스페이스 — 캐시 설계 공간 탐색 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: area.h (Area 클래스), component.h (Component 기반 클래스), parameter.h (g_tp),
 *   basic_circuit.h의 gate_C/drain_C_/tr_R_on/horowitz/cmos_Isub_leakage (구현에서 사용).
 * 이 파일에 의존: mat.cc (Mat 클래스가 Predec/Decoder 객체 생성),
 *   decoder.cc (구현), uca.cc (결과 집계).
 *
 * === 주요 함수/구조체 요약 ===
 * Decoder       : NAND2/3 + 인버터 체인으로 워드라인 구동; logical_effort 기반 게이트 크기 결정
 * PredecBlk     : 두 단계 프리디코더 블록 (L1=NAND2/3 경로, L2=OR 출력)
 * PredecBlkDrv  : 프리디코더 블록 앞의 버퍼 체인 드라이버
 * Predec        : PredecBlkDrv + PredecBlk 쌍을 하나로 묶는 프리디코더 스테이지
 * Driver        : 용량성 부하를 구동하는 범용 버퍼 체인
 */

#ifndef __DECODER_H__
#define __DECODER_H__

#include "area.h"      // [한국어] Area 클래스 — 면적 추적 (h, w 필드)
#include "component.h" // [한국어] Component 기반 클래스 — area, power, delay 필드 보유
#include "parameter.h" // [한국어] g_tp (TechnologyParameter 전역 인스턴스)
#include <vector>      // [한국어] 디코더 구성 목록 (미사용이나 포함됨)

using namespace std;


/*
 * [한국어] Decoder — 워드라인 디코더 (NAND2/NAND3 게이트 + 인버터 체인)
 * 캐시 행/열 주소의 최하위 비트를 실제 워드라인 선택 신호로 변환한다.
 * 주소 비트 수가 4비트 미만이면 프리디코더만으로 직접 워드라인을 구동하고,
 * 4비트 이상이면 NAND2 또는 NAND3 게이트 + 인버터 체인으로 구성된 디코더를 사용한다.
 * logical_effort() 알고리즘으로 최소 지연 게이트 크기를 자동 결정한다.
 */
class Decoder : public Component
{
  public:
    /*
     * [한국어]
     * Decoder 생성자 — 디코더 설계 파라미터 초기화 및 게이트 크기/면적 계산
     * @_num_dec_signals       : 디코더 출력 신호 수 (워드라인 수)
     * @flag_way_select        : 웨이 선택 신호가 추가 입력으로 있는지 여부
     * @_C_ld_dec_out          : 디코더 출력 부하 커패시턴스 (F) — 워드라인 커패시턴스
     * @_R_wire_dec_out        : 디코더 출력 와이어 저항 (ohm)
     * @fully_assoc_           : 완전 결합 캐시 여부
     * @is_dram_               : DRAM 여부 (트랜지스터 파라미터 선택)
     * @is_wl_tr_              : 워드라인 트랜지스터 여부 (Vpp 사용)
     * @cell_                  : 셀 높이/폭 (레이아웃 기준)
     * 주소 비트 수를 계산하여 exist 여부와 num_in_signals(2 또는 3)를 결정.
     * compute_widths(), compute_area()를 호출하여 초기화 완료.
     * 호출 체인: Mat → [Decoder()] → compute_widths → compute_area
     */
    Decoder(
        int _num_dec_signals,
        bool flag_way_select,
        double _C_ld_dec_out,
        double _R_wire_dec_out,
        bool fully_assoc_,
        bool is_dram_,
        bool is_wl_tr_,
        const Area & cell_);

    bool   exist;
    /* 이 디코더가 실제로 필요한지 여부
     * 설정자: 생성자에서 num_addr_bits_dec < 4이고 way_select 없으면 false.
     * 읽는 자: compute_widths/compute_area/compute_delays에서 exist 체크 후 처리.
     * 값 범위: true(디코더 존재) / false(프리디코더만으로 충분).
     * 동기화: 생성 시 1회 설정, 이후 읽기 전용. */

    int    num_in_signals;
    /* 디코더 첫 단 게이트 입력 수 (2=NAND2, 3=NAND3)
     * 설정자: 생성자에서 num_addr_bits_dec와 flag_way_select로 결정.
     * 읽는 자: compute_widths — 게이트 유형(NAND2/NAND3) 선택.
     * 값 범위: 0(비존재), 2(NAND2), 3(NAND3). */

    double C_ld_dec_out;
    /* 디코더 출력 부하 커패시턴스 (F) — 워드라인을 구동하는 커패시턴스
     * 설정자: 생성자 파라미터로 전달.
     * 읽는 자: logical_effort() — 게이트 체인의 전기적 노력 계산.
     * 값 범위: 수 fF ~ 수십 fF (워드라인 길이에 비례). */

    double R_wire_dec_out;
    /* 디코더 출력 와이어 저항 (ohm) — 워드라인 배선 저항
     * 설정자: 생성자 파라미터로 전달.
     * 읽는 자: compute_delays — 마지막 인버터의 RC 시정수에 추가.
     * 값 범위: 0 이상 (워드라인 길이에 비례). */

    int    num_gates;
    /* 디코더 게이트 체인 총 단 수 (logical_effort()가 결정)
     * 설정자: logical_effort()가 반환.
     * 읽는 자: compute_area/compute_delays에서 게이트 루프 범위.
     * 값 범위: num_gates_min(2) 이상. */

    int    num_gates_min;
    /* 최소 게이트 단 수 (기본값 2: NAND + 인버터)
     * 설정자: 생성자에서 2로 초기화.
     * 읽는 자: logical_effort()에 전달.
     * 값 범위: 항상 2. */

    double w_dec_n[MAX_NUMBER_GATES_STAGE];
    /* NMOS 게이트 폭 배열 (각 단계별) — w_dec_n[0]이 첫 단 NAND2/3의 NMOS 폭
     * 설정자: compute_widths()의 logical_effort() 호출 결과.
     * 읽는 자: compute_area(cmos_Isub_leakage), compute_delays(gate_C, drain_C_, tr_R_on).
     * 값 범위: g_tp.min_w_nmos_ × fanin 이상. */

    double w_dec_p[MAX_NUMBER_GATES_STAGE];
    /* PMOS 게이트 폭 배열 (각 단계별)
     * 설정자: compute_widths()의 logical_effort() 호출 결과.
     * 읽는 자: compute_area, compute_delays.
     * 값 범위: p_to_n_sz_ratio × g_tp.min_w_nmos_ 이상. */

    double delay;
    /* 누적 전파 지연 (초) — 첫 입력부터 워드라인까지의 총 지연
     * 설정자: compute_delays()에서 각 게이트 horowitz() 결과를 누적.
     * 읽는 자: Mat/Predec 지연 합산.
     * 값 범위: 수십~수백 ps. */

    //powerDef power; // [한국어] (주석됨) Component 기반 클래스의 power 필드를 사용

    bool   fully_assoc; // [한국어] 완전 결합 캐시 여부 — compute_widths에서 NAND2 강제 사용 조건
    bool   is_dram;     // [한국어] DRAM 여부 — 트랜지스터 파라미터 선택
    bool   is_wl_tr;    // [한국어] 워드라인 트랜지스터 여부 — Vpp 사용 시 true
    const  Area & cell; /* 셀 높이/폭 참조 — 디코더 드라이버 셀 높이 기준
                         * 설정자: 생성자 파라미터로 전달.
                         * 읽는 자: compute_area에서 area.h = g_tp.h_dec * cell.h 계산. */


    /* [한국어]
     * compute_widths - logical_effort 기반 게이트 체인 크기 결정
     * NAND2 또는 NAND3 첫 단 게이트 + 인버터 체인의 각 게이트 폭을 계산.
     * 전기적 노력(F = C_ld / C_in_first)과 논리적 노력(gnand2/gnand3)을 바탕으로
     * logical_effort()가 num_gates와 w_dec_n/p 배열을 채운다.
     * 호출 체인: Decoder() → [compute_widths] → logical_effort */
    void   compute_widths();

    /* [한국어]
     * compute_area - 디코더 게이트 면적과 누설전류 계산
     * 첫 단(NAND2 또는 NAND3)과 이후 인버터 체인의 면적을 합산하고
     * cmos_Isub_leakage, cmos_Ig_leakage로 누설 전력을 power.readOp에 저장.
     * 호출 체인: Decoder() → [compute_area] → cmos_Isub/Ig_leakage */
    void   compute_area();

    /* [한국어]
     * compute_delays - 디코더 단별 전파 지연 계산
     * @inrisetime : 입력 신호 상승 시간 (s)
     * @return     : 출력 신호 상승 시간 (s)
     * 각 게이트의 rd, c_load, c_intrinsic, tf를 계산하고 horowitz()로 단별 지연을 누적.
     * 마지막 인버터에 R_wire_dec_out을 추가하여 워드라인 배선 지연 포함.
     * 호출 체인: Mat → Predec → [compute_delays] → horowitz */
    double compute_delays(double inrisetime);  // return outrisetime

    /* [한국어]
     * leakage_feedback - 온도에 따른 누설전류 재계산 (온도 변화 시 피드백 루프)
     * @temperature : 동작 온도 (K)
     * power.readOp.leakage와 gate_leakage를 현재 온도 기준으로 재계산.
     * 호출 체인: 온도 피드백 루프 → [leakage_feedback] */
    void leakage_feedback(double temperature);
};



/*
 * [한국어] PredecBlk — 2단 프리디코더 블록 (L1 NAND2/3 경로 + L2 OR 출력)
 * 캐시 주소 비트를 L1(NAND2 경로 또는 NAND3 경로)로 디코딩한 후,
 * L2 OR 게이트로 최종 프리디코더 출력을 생성한다.
 * 입력 주소 비트 수에 따라 4/8/16 신호 중 적합한 구조를 선택하며,
 * nand2/nand3 두 경로의 지연과 전력을 독립적으로 추적한다.
 */
class PredecBlk : public Component
{
 public:
  /* [한국어]
   * PredecBlk 생성자 — 프리디코더 블록 파라미터 초기화
   * @num_dec_signals          : 디코더 출력 신호 수 (주소 비트 수 결정에 사용)
   * @dec                      : 연결된 Decoder 포인터 (num_in_signals 참조)
   * @C_wire_predec_blk_out   : L2 출력 와이어 커패시턴스 (F)
   * @R_wire_predec_blk_out   : L2 출력 와이어 저항 (ohm)
   * @num_dec_per_predec       : 이 PredecBlk에 대응하는 디코더 수
   * @is_dram_                 : DRAM 여부
   * @is_blk1                 : 블록1(하위 비트) 또는 블록2(상위 비트) 여부
   * 내부에서 compute_widths, compute_area를 호출하여 초기화.
   * 호출 체인: Predec/Mat → [PredecBlk()] → compute_widths → compute_area */
  PredecBlk(
      int num_dec_signals,
      Decoder * dec,
      double C_wire_predec_blk_out,
      double R_wire_predec_blk_out,
      int    num_dec_per_predec,
      bool   is_dram_,
      bool   is_blk1);

  Decoder * dec;
  /* 연결된 Decoder 포인터 — num_in_signals(2=NAND2, 3=NAND3) 참조
   * 설정자: 생성자 파라미터.
   * 읽는 자: compute_widths()에서 NAND2/NAND3 경로 선택.
   * 동기화: 읽기 전용. */

  bool exist;
  /* 이 프리디코더 블록이 실제로 필요한지 여부
   * 설정자: 생성자에서 number_input_addr_bits가 0이면 false.
   * 읽는 자: PredecBlkDrv 생성 시, compute_widths 시작 시 체크.
   * 값 범위: true/false. */

  int number_input_addr_bits;
  /* 이 PredecBlk이 처리하는 주소 비트 수
   * 설정자: 생성자에서 num_dec_signals와 is_blk1로 계산.
   * 읽는 자: compute_widths — 4비트면 L2 불필요, 5~8비트면 L2 NAND2, 9~12비트면 L2 NAND3.
   * 값 범위: 0~12. */

  double C_ld_predec_blk_out;
  /* L2 출력 부하 커패시턴스 (F) = C_wire + 디코더 입력 커패시턴스
   * 설정자: 생성자 파라미터 C_wire_predec_blk_out에서 초기화.
   * 읽는 자: compute_widths의 logical_effort()에 전달. */

  double R_wire_predec_blk_out;
  /* L2 출력 와이어 저항 (ohm)
   * 설정자: 생성자 파라미터.
   * 읽는 자: compute_delays — 출력 드라이버 RC 계산. */

  int branch_effort_nand2_gate_output;
  /* NAND2 L1 게이트 출력의 분기 노력(branch effort)
   * 설정자: compute_widths()에서 num_L1_active_nand2_path 기반 계산.
   * 읽는 자: logical_effort()에서 전기적 노력 계산. */

  int branch_effort_nand3_gate_output;
  /* NAND3 L1 게이트 출력의 분기 노력
   * 설정자: compute_widths()에서 num_L1_active_nand3_path 기반 계산. */

  bool   flag_two_unique_paths;
  /* NAND2와 NAND3 두 경로가 모두 존재하는지 여부
   * 설정자: compute_widths()에서 nand2/nand3 경로 조합 분석.
   * 읽는 자: compute_delays에서 두 경로를 모두 계산할지 결정. */

  int flag_L2_gate;
  /* L2 게이트 유형 플래그 (0=없음, 2=NAND2, 3=NAND3)
   * 설정자: compute_widths()에서 number_input_addr_bits에 따라 설정.
   * 읽는 자: compute_widths/area/delays에서 L2 게이트 크기 선택. */

  int number_inputs_L1_gate;
  /* L1 게이트 입력 수 (2=NAND2, 3=NAND3)
   * 설정자: compute_widths()에서 dec->num_in_signals 참조.
   * 읽는 자: w_L1_nand2_n/p 배열 인덱싱. */

  int number_gates_L1_nand2_path;
  /* L1 NAND2 경로의 총 게이트 단 수
   * 설정자: logical_effort() 반환값.
   * 읽는 자: compute_area/compute_delays 루프 범위. */

  int number_gates_L1_nand3_path;
  /* L1 NAND3 경로의 총 게이트 단 수
   * 설정자: logical_effort() 반환값. */

  int number_gates_L2;
  /* L2 OR 경로의 총 게이트 단 수
   * 설정자: logical_effort() 반환값. 0이면 L2 없음. */

  int min_number_gates_L1;
  /* L1 경로 최소 게이트 단 수 (기본 2)
   * 설정자: 생성자에서 2로 초기화.
   * 읽는 자: logical_effort()에 전달. */

  int min_number_gates_L2;
  /* L2 경로 최소 게이트 단 수 (기본 2)
   * 설정자: 생성자에서 2로 초기화. */

  int num_L1_active_nand2_path;
  /* NAND2 경로에서 활성화된 L1 출력 신호 수
   * 설정자: compute_widths()에서 address 비트 수로 계산.
   * 읽는 자: branch_effort_nand2_gate_output 계산. */

  int num_L1_active_nand3_path;
  /* NAND3 경로에서 활성화된 L1 출력 신호 수
   * 설정자: compute_widths()에서 address 비트 수로 계산. */

  double w_L1_nand2_n[MAX_NUMBER_GATES_STAGE];
  /* L1 NAND2 경로 각 단의 NMOS 폭 (m)
   * 설정자: logical_effort()가 채움.
   * 읽는 자: compute_area/compute_delays. */

  double w_L1_nand2_p[MAX_NUMBER_GATES_STAGE];
  /* L1 NAND2 경로 각 단의 PMOS 폭 (m) */

  double w_L1_nand3_n[MAX_NUMBER_GATES_STAGE];
  /* L1 NAND3 경로 각 단의 NMOS 폭 (m) */

  double w_L1_nand3_p[MAX_NUMBER_GATES_STAGE];
  /* L1 NAND3 경로 각 단의 PMOS 폭 (m) */

  double w_L2_n[MAX_NUMBER_GATES_STAGE];
  /* L2 OR 경로 각 단의 NMOS 폭 (m) */

  double w_L2_p[MAX_NUMBER_GATES_STAGE];
  /* L2 OR 경로 각 단의 PMOS 폭 (m) */

  double delay_nand2_path;
  /* NAND2 경로 총 전파 지연 (s)
   * 설정자: compute_delays()에서 L1+L2 단별 horowitz() 누적.
   * 읽는 자: Predec::get_max_delay_before_decoder()에서 최대값 선택. */

  double delay_nand3_path;
  /* NAND3 경로 총 전파 지연 (s) */

  powerDef power_nand2_path;
  /* NAND2 경로 전력 (dynamic + leakage + gate_leakage)
   * 설정자: compute_area()에서 cmos_Isub/Ig_leakage로 누설 계산.
   * 읽는 자: Predec::block_power 집계. */

  powerDef power_nand3_path;
  /* NAND3 경로 전력 */

  powerDef power_L2;
  /* L2 OR 게이트 전력 */

  bool is_dram_;
  /* DRAM 여부 — 트랜지스터 파라미터 선택
   * 설정자: 생성자 파라미터.
   * 읽는 자: compute_widths — min_w_nmos/pmos 선택. */

  /* [한국어]
   * compute_widths - L1/L2 게이트 체인 크기 결정 (logical_effort 기반)
   * nand2/nand3 두 경로와 L2 OR 경로의 게이트 폭을 각각 계산.
   * 호출 체인: PredecBlk() → [compute_widths] → logical_effort */
  void compute_widths();

  /* [한국어]
   * compute_area - L1/L2 각 경로의 면적과 누설전류 계산
   * 호출 체인: PredecBlk() → [compute_area] → cmos_Isub/Ig_leakage */
  void compute_area();

  /* [한국어]
   * leakage_feedback - 온도에 따른 누설전류 재계산
   * @temperature : 동작 온도 (K)
   * 호출 체인: 온도 피드백 루프 → [leakage_feedback] */
  void leakage_feedback(double temperature);

  /* [한국어]
   * compute_delays - nand2/nand3 두 경로의 지연을 각각 계산
   * @inrisetime : pair<nand2 입력 상승시간, nand3 입력 상승시간> (s)
   * @return     : pair<nand2 출력 상승시간, nand3 출력 상승시간> (s)
   * 각 경로의 단별 horowitz() 지연을 delay_nand2_path / delay_nand3_path에 누적.
   * 호출 체인: Predec → PredecBlkDrv → [PredecBlk::compute_delays] → horowitz */
  pair<double, double> compute_delays(pair<double, double> inrisetime); // <nand2, nand3>
  // return <outrise_nand2, outrise_nand3>
};


/*
 * [한국어] PredecBlkDrv — 프리디코더 블록 입력 버퍼 드라이버
 * PredecBlk 앞에서 주소 신호를 충분한 구동력으로 증폭하는 버퍼 체인.
 * NAND2 경로(1/2/4 팬아웃 구분)와 NAND3 경로(2/8 팬아웃 구분)를
 * 각각 독립적으로 버퍼링하여 최소 지연으로 PredecBlk에 전달.
 */
class PredecBlkDrv : public Component
{
 public:
  /* [한국어]
   * PredecBlkDrv 생성자 — 드라이버 파라미터 초기화
   * @way_select : 웨이 선택 신호 사용 여부 (1이면 별도 드라이버 생성)
   * @blk_       : 이 드라이버가 구동하는 PredecBlk 포인터
   * @is_dram    : DRAM 여부
   * 내부에서 compute_widths, compute_area 호출.
   * 호출 체인: Predec/Mat → [PredecBlkDrv()] → compute_widths → compute_area */
  PredecBlkDrv(
      int   way_select,
      PredecBlk * blk_,
      bool  is_dram);

  int flag_driver_exists;
  /* 이 드라이버가 실제로 필요한지 여부 (blk->exist 기반)
   * 설정자: 생성자에서 blk->exist로 결정.
   * 읽는 자: compute_widths/compute_delays 시작 시 체크. */

  int number_input_addr_bits;
  /* 이 드라이버가 처리하는 주소 비트 수 (blk->number_input_addr_bits)
   * 설정자: 생성자에서 blk->number_input_addr_bits 복사.
   * 읽는 자: num_buffers_driving_* 계산. */

  int number_gates_nand2_path;
  /* NAND2 경로의 버퍼 체인 게이트 단 수
   * 설정자: compute_widths()의 logical_effort() 결과.
   * 읽는 자: compute_area/compute_delays 루프 범위. */

  int number_gates_nand3_path;
  /* NAND3 경로의 버퍼 체인 게이트 단 수 */

  int min_number_gates;
  /* 최소 게이트 단 수 (기본 2)
   * 설정자: 생성자에서 2로 초기화.
   * 읽는 자: logical_effort()에 전달. */

  int num_buffers_driving_1_nand2_load;
  /* NAND2 경로에서 팬아웃 1(단일 NAND2)을 구동하는 버퍼 수
   * 설정자: compute_widths()에서 number_input_addr_bits 분석.
   * 읽는 자: num_addr_bits_nand2_path() 합산. */

  int num_buffers_driving_2_nand2_load;
  /* NAND2 경로에서 팬아웃 2(두 NAND2)를 구동하는 버퍼 수 */

  int num_buffers_driving_4_nand2_load;
  /* NAND2 경로에서 팬아웃 4(네 NAND2)를 구동하는 버퍼 수 */

  int num_buffers_driving_2_nand3_load;
  /* NAND3 경로에서 팬아웃 2를 구동하는 버퍼 수 */

  int num_buffers_driving_8_nand3_load;
  /* NAND3 경로에서 팬아웃 8을 구동하는 버퍼 수 */

  int num_buffers_nand3_path;
  /* NAND3 경로의 총 버퍼 수
   * 설정자: compute_widths()에서 계산.
   * 읽는 자: compute_area에서 누설 계산 루프. */

  double c_load_nand2_path_out;
  /* NAND2 경로 출력 부하 커패시턴스 (F)
   * 설정자: compute_widths()에서 PredecBlk 입력 커패시턴스 + 와이어.
   * 읽는 자: logical_effort()에 전달. */

  double c_load_nand3_path_out;
  /* NAND3 경로 출력 부하 커패시턴스 (F) */

  double r_load_nand2_path_out;
  /* NAND2 경로 출력 와이어 저항 (ohm) */

  double r_load_nand3_path_out;
  /* NAND3 경로 출력 와이어 저항 (ohm) */

  double width_nand2_path_n[MAX_NUMBER_GATES_STAGE];
  /* NAND2 경로 각 단의 NMOS 폭 (m) */

  double width_nand2_path_p[MAX_NUMBER_GATES_STAGE];
  /* NAND2 경로 각 단의 PMOS 폭 (m) */

  double width_nand3_path_n[MAX_NUMBER_GATES_STAGE];
  /* NAND3 경로 각 단의 NMOS 폭 (m) */

  double width_nand3_path_p[MAX_NUMBER_GATES_STAGE];
  /* NAND3 경로 각 단의 PMOS 폭 (m) */

  double delay_nand2_path;
  /* NAND2 경로 드라이버 총 지연 (s)
   * 설정자: compute_delays()에서 horowitz() 누적.
   * 읽는 자: Predec::compute_delays()에서 PredecBlk 지연에 합산. */

  double delay_nand3_path;
  /* NAND3 경로 드라이버 총 지연 (s) */

  powerDef power_nand2_path;
  /* NAND2 경로 드라이버 전력 (dynamic + leakage + gate_leakage) */

  powerDef power_nand3_path;
  /* NAND3 경로 드라이버 전력 */

  PredecBlk * blk;
  /* 이 드라이버가 구동하는 PredecBlk 포인터
   * 설정자: 생성자 파라미터.
   * 읽는 자: compute_widths()에서 blk->number_input_addr_bits 참조. */

  Decoder   * dec;
  /* 연결된 Decoder 포인터 (blk->dec 복사)
   * 설정자: 생성자에서 blk->dec 복사.
   * 읽는 자: compute_widths()에서 dec->num_in_signals 참조. */

  bool  is_dram_;
  /* DRAM 여부 — 트랜지스터 파라미터 선택 */

  int   way_select;
  /* 웨이 선택 신호 사용 여부 (0=없음, 1=있음)
   * 설정자: 생성자 파라미터.
   * 읽는 자: compute_widths()에서 웨이 선택 경로 추가 여부 결정. */

  /* [한국어]
   * compute_widths - NAND2/3 경로 버퍼 체인 게이트 크기 결정
   * 팬아웃 분류별로 logical_effort()를 호출하여 width_nand2/3_path_n/p 배열 채움.
   * 호출 체인: PredecBlkDrv() → [compute_widths] → logical_effort */
  void compute_widths();

  /* [한국어]
   * compute_area - 드라이버 면적 및 누설전류 계산
   * 호출 체인: PredecBlkDrv() → [compute_area] → cmos_Isub/Ig_leakage */
  void compute_area();

  /* [한국어]
   * leakage_feedback - 온도에 따른 누설전류 재계산
   * 호출 체인: 온도 피드백 루프 → [leakage_feedback] */
  void leakage_feedback(double temperature);

  /* [한국어]
   * compute_delays - nand2/nand3 경로 드라이버 지연 계산
   * @inrisetime_nand2_path : NAND2 경로 입력 상승 시간 (s)
   * @inrisetime_nand3_path : NAND3 경로 입력 상승 시간 (s)
   * @return : pair<nand2 출력 상승시간, nand3 출력 상승시간>
   * 각 경로의 버퍼 체인 단별 horowitz()를 호출하여 지연 누적.
   * 호출 체인: Predec::compute_delays → [PredecBlkDrv::compute_delays] → horowitz */
  pair<double, double> compute_delays(
      double inrisetime_nand2_path,
      double inrisetime_nand3_path);  // return <outrise_nand2, outrise_nand3>

  /* [한국어]
   * num_addr_bits_nand2_path - NAND2 경로의 총 주소 비트 수 반환 (인라인)
   * = 팬아웃1 버퍼 수 + 팬아웃2 버퍼 수 + 팬아웃4 버퍼 수
   * 설계: 각 버퍼 그룹이 담당하는 주소 비트 수를 합산.
   * 호출 체인: compute_widths, compute_area → [num_addr_bits_nand2_path] */
  inline int num_addr_bits_nand2_path()
  {
    return num_buffers_driving_1_nand2_load + // [한국어] 팬아웃 1 버퍼가 담당하는 주소 비트
           num_buffers_driving_2_nand2_load + // [한국어] 팬아웃 2 버퍼가 담당하는 주소 비트
           num_buffers_driving_4_nand2_load;  // [한국어] 팬아웃 4 버퍼가 담당하는 주소 비트
  }

  /* [한국어]
   * num_addr_bits_nand3_path - NAND3 경로의 총 주소 비트 수 반환 (인라인)
   * = 팬아웃2 버퍼 수 + 팬아웃8 버퍼 수
   * 호출 체인: compute_widths, compute_area → [num_addr_bits_nand3_path] */
  inline int num_addr_bits_nand3_path()
  {
    return num_buffers_driving_2_nand3_load + // [한국어] 팬아웃 2 버퍼가 담당하는 주소 비트
           num_buffers_driving_8_nand3_load;  // [한국어] 팬아웃 8 버퍼가 담당하는 주소 비트
  }

  /* [한국어]
   * get_rdOp_dynamic_E - 읽기 동작 동적 에너지 계산
   * @num_act_mats_hor_dir : 수평 방향 활성화된 MAT 수
   * @return               : 이 드라이버의 읽기 동적 에너지 (J)
   * 호출 체인: Mat/uca → [get_rdOp_dynamic_E] */
  double get_rdOp_dynamic_E(int num_act_mats_hor_dir);
};



/*
 * [한국어] Predec — 프리디코더 스테이지 (PredecBlkDrv × 2 + PredecBlk × 2 쌍)
 * 두 PredecBlkDrv와 두 PredecBlk를 하나의 프리디코더 스테이지로 묶는다.
 * 두 블록 경로(blk1/blk2)의 지연 중 최대값을 Decoder 입력 지연으로 사용.
 * block_power와 driver_power를 분리하여 각 계층의 전력 기여를 추적한다.
 */
class Predec : public Component
{
  public:
    /* [한국어]
     * Predec 생성자 — 두 PredecBlkDrv를 연결하여 프리디코더 스테이지 초기화
     * @drv1 : 하위 주소 비트 블록의 드라이버 (blk1 연결)
     * @drv2 : 상위 주소 비트 블록의 드라이버 (blk2 연결)
     * drv1/drv2에서 blk1/blk2를 추출하여 area/power를 집계.
     * 호출 체인: Mat → [Predec()] → PredecBlkDrv::compute_delays, PredecBlk::compute_delays */
    Predec(
        PredecBlkDrv * drv1,
        PredecBlkDrv * drv2);

    /* [한국어]
     * compute_delays - 두 프리디코더 블록 경로의 최대 지연 계산
     * @inrisetime : 입력 신호 상승 시간 (s)
     * @return     : 출력 신호 상승 시간 (s)
     * drv1, drv2의 compute_delays()와 blk1, blk2의 compute_delays()를 순서대로 호출.
     * get_max_delay_before_decoder()로 두 경로 중 최대 지연을 선택.
     * 호출 체인: Mat → [Predec::compute_delays] → PredecBlkDrv/PredecBlk::compute_delays */
    double compute_delays(double inrisetime);  // return outrisetime

    /* [한국어]
     * leakage_feedback - 온도에 따른 누설전류 재계산 (blk1/2, drv1/2 위임)
     * @temperature : 동작 온도 (K)
     * 호출 체인: 온도 피드백 루프 → [Predec::leakage_feedback] */
    void leakage_feedback(double temperature);

    PredecBlk    * blk1;
    /* 하위 주소 비트 프리디코더 블록 포인터
     * 설정자: 생성자에서 drv1->blk 복사.
     * 읽는 자: compute_delays, leakage_feedback, block_power 집계. */

    PredecBlk    * blk2;
    /* 상위 주소 비트 프리디코더 블록 포인터 */

    PredecBlkDrv * drv1;
    /* 하위 블록 드라이버 포인터
     * 설정자: 생성자 파라미터.
     * 읽는 자: compute_delays → drv1->compute_delays() 호출. */

    PredecBlkDrv * drv2;
    /* 상위 블록 드라이버 포인터 */

    powerDef block_power;
    /* 두 PredecBlk의 전력 합산 (dynamic + leakage + gate_leakage)
     * 설정자: Predec 생성자에서 blk1/blk2의 power_nand2/3_path + power_L2 합산.
     * 읽는 자: Mat/uca — 전체 캐시 전력 집계. */

    powerDef driver_power;
    /* 두 PredecBlkDrv의 전력 합산
     * 설정자: Predec 생성자에서 drv1/drv2의 power_nand2/3_path 합산.
     * 읽는 자: Mat/uca — 전체 캐시 전력 집계. */

  private:
    // returns <delay, risetime>
    /* [한국어]
     * get_max_delay_before_decoder - 두 경로 쌍 중 지연이 큰 경로 선택
     * @input_pair1 : blk1의 <delay_nand2_path, delay_nand3_path>
     * @input_pair2 : blk2의 <delay_nand2_path, delay_nand3_path>
     * @return      : pair<최대 지연, 해당 경로의 출력 상승 시간>
     * Decoder 입력 전 마지막 지연 병목을 결정하는 함수.
     * 호출 체인: Predec::compute_delays → [get_max_delay_before_decoder] */
    pair<double, double> get_max_delay_before_decoder(
        pair<double, double> input_pair1,
        pair<double, double> input_pair2);
};



/*
 * [한국어] Driver — 범용 버퍼 체인 드라이버
 * 임의의 게이트+와이어 부하를 구동하는 인버터 체인.
 * c_gate_load, c_wire_load, r_wire_load를 기반으로 logical_effort()로
 * 최적 단 수와 각 단의 게이트 폭을 결정한다.
 * PredecBlkDrv와 달리 단일 경로만 처리하는 단순화된 드라이버.
 */
class Driver : public Component
{
 public:
  /* [한국어]
   * Driver 생성자 — 게이트/와이어 부하 파라미터 초기화
   * @c_gate_load_ : 구동할 게이트 커패시턴스 (F)
   * @c_wire_load_ : 와이어 커패시턴스 (F)
   * @r_wire_load_ : 와이어 저항 (ohm)
   * @is_dram      : DRAM 여부
   * 내부에서 compute_widths()를 호출하여 게이트 폭 결정.
   * 호출 체인: Mat/분석 코드 → [Driver()] → compute_widths */
  Driver(double c_gate_load_, double c_wire_load_, double r_wire_load_, bool is_dram);

  int    number_gates;
  /* 인버터 체인 총 단 수 (logical_effort() 결정)
   * 설정자: compute_widths()의 logical_effort() 반환값.
   * 읽는 자: compute_delay() 루프 범위, compute_widths의 면적 루프. */

  int    min_number_gates;
  /* 최소 게이트 단 수 (기본 2)
   * 설정자: 생성자에서 2로 초기화. */

  double width_n[MAX_NUMBER_GATES_STAGE];
  /* 각 단의 NMOS 폭 (m)
   * 설정자: logical_effort() 결과.
   * 읽는 자: compute_delay → gate_C, drain_C_, tr_R_on. */

  double width_p[MAX_NUMBER_GATES_STAGE];
  /* 각 단의 PMOS 폭 (m) */

  double c_gate_load;
  /* 게이트 부하 커패시턴스 (F) — logical_effort()에 사용 */

  double c_wire_load;
  /* 와이어 부하 커패시턴스 (F) — 마지막 단 출력에 추가 */

  double r_wire_load;
  /* 와이어 부하 저항 (ohm) — 마지막 단 Elmore 지연에 추가 */

  double delay;
  /* 버퍼 체인 총 전파 지연 (s)
   * 설정자: compute_delay()에서 단별 horowitz() 누적.
   * 읽는 자: 상위 회로의 경로 지연 합산. */

  powerDef power;
  /* 동적/누설/게이트누설 전력
   * 설정자: compute_widths()에서 cmos_Isub/Ig_leakage로 계산. */

  bool   is_dram_;
  /* DRAM 여부 — 트랜지스터 파라미터 선택 */

  /* [한국어]
   * compute_widths - logical_effort 기반 인버터 체인 게이트 크기 결정
   * 호출 체인: Driver() → [compute_widths] → logical_effort */
  void   compute_widths();

  /* [한국어]
   * compute_delay - 인버터 체인 단별 전파 지연 계산
   * @inrisetime : 입력 신호 상승 시간 (s)
   * @return     : 출력 신호 상승 시간 (s)
   * 각 단의 rd, c_load, c_intrinsic, tf를 horowitz()에 전달하여 delay 누적.
   * 호출 체인: Mat → [Driver::compute_delay] → horowitz */
  double compute_delay(double inrisetime);
};


#endif
