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
 * [한국어 설명] CACTI 기본 회로 모델 함수 선언 (basic_circuit.h)
 *
 * === 파일의 역할 ===
 * CACTI(Cache Access and Cycle Time Information)에서 사용하는 트랜지스터/회로 수준의
 * 핵심 물리 모델 함수들을 선언한다. 게이트 커패시턴스(gate_C), 드레인 커패시턴스(drain_C_),
 * 트랜지스터 온 저항(tr_R_on), Horowitz 타이밍 모델(horowitz), 서브임계 누설전류
 * (cmos_Isub_leakage), 게이트 터널링 누설전류(cmos_Ig_leakage), 단락전류(shortcircuit)
 * 등 회로 분석에 필요한 모든 저수준 물리 함수가 여기에 선언된다.
 * 이 파일의 함수들은 CACTI의 타이밍/전력/면적 모델링 계층의 최하단부를 구성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch(GPU 전력 모델) → CACTI(캐시 에너지/면적/타이밍) → basic_circuit.h(물리 회로 계층).
 * CACTI 내부에서 decoder.cc, crossbar.cc, wire.cc, mat.cc 등 상위 회로 모델 파일들이
 * 이 파일에 선언된 함수들을 직접 호출하여 게이트 크기를 결정하고 지연/전력을 계산한다.
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 시뮬레이션 전 CACTI 초기화 단계에서 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈: const.h (RISE/FALL, NCH/PCH 등 상수), cacti_interface.h (InputParameter 등),
 *   parameter.h (g_tp: TechnologyParameter 전역 객체 — 공정별 C_g_ideal, R_nch_on 등 수치 포함).
 * 이 파일에 의존하는 모듈: decoder.cc, crossbar.cc, wire.cc, mat.cc, uca.cc 등 CACTI 전체.
 * 공유 핵심 자료구조: TechnologyParameter::DeviceType (g_tp.peri_global, g_tp.dram_acc,
 *   g_tp.sram_cell 등) — 공정 노드별 트랜지스터 파라미터를 보관하는 구조체.
 *
 * === 주요 함수/구조체 요약 ===
 * gate_C()           : NMOS/PMOS 게이트 커패시턴스 계산 (C_g_ideal + C_overlap + 3*C_fringe)*W
 * drain_C_()         : 드레인 커패시턴스 계산 (접합면적 + 측벽 + 프린지/오버랩 + 폴딩 보정)
 * tr_R_on()          : 트랜지스터 온(on) 저항 = stack * R_nch_on / width
 * horowitz()         : Horowitz 1984 논문 기반 RC 타이밍 모델
 * cmos_Isub_leakage(): CMOS 게이트 유형별 서브임계 누설전류 통계적 평균 계산
 * cmos_Ig_leakage()  : CMOS 게이트 유형별 게이트 터널링 누설전류 계산
 * set_pppm()         : 전력 포인트 프로덕트 마스크(power point product mask) 설정
 */

#ifndef __BASIC_CIRCUIT_H__
#define __BASIC_CIRCUIT_H__

#include "const.h"           // [한국어] RISE/FALL, NCH/PCH, MAX_NUMBER_GATES_STAGE 등 CACTI 전역 상수
#include "cacti_interface.h" // [한국어] InputParameter, powerDef, mem_array, uca_org_t 등 인터페이스 타입

using namespace std;

/* [한국어] UNI_LEAK_STACK_FACTOR = 0.43
 * 직렬 스택(series stack) 트랜지스터에서 하위 트랜지스터가 완전히 오프(off)되어 있을 때
 * 스택 효과(stack effect)로 인해 누설전류가 감소하는 비율.
 * 경험적(empirical) 값으로, NAND/NOR 게이트의 직렬 NMOS/PMOS 스택에서
 * 각 OFF 트랜지스터를 추가할 때마다 이 인수만큼 누설이 줄어든다고 모델링한다. */
#define UNI_LEAK_STACK_FACTOR 0.43

/* [한국어] 수학 유틸리티 함수 선언 — 디코더/프리디코더 계산에 사용 */
int powers (int base, int n);   // [한국어] base^n 정수 지수승
bool is_pow2(int64_t val);      // [한국어] val이 2의 거듭제곱인지 판별
uint32_t _log2(uint64_t num);   // [한국어] 이진 로그 (정수, 비트 이동 방식)
int factorial(int n, int m = 1);// [한국어] n!/m! (부분 팩토리얼, m에서 n까지의 곱)
int combination(int n, int m);  // [한국어] 이항 계수 C(n,m) = n! / (m! * (n-m)!)

/* [한국어] DBG 조건부 컴파일 — 디버그 빌드일 때만 PRINTDW(a) 매크로가 코드를 실행.
 * 릴리즈 빌드에서는 완전히 제거되어 성능 영향 없음. */
//#define DBG
#ifdef DBG
    #define PRINTDW(a);\
    a;
#else
    #define PRINTDW(a);\

#endif


/* [한국어] Wire_placement — 와이어가 배치되는 위치를 구분하는 enum.
 * CACTI는 배치 위치에 따라 다른 RC 파라미터(pitch, C_per_um, R_per_um)를 적용한다. */
enum Wire_placement {
    outside_mat,  // [한국어] MAT 외부 (H-tree, 서브뱅크 간 배선) — wire_outside_mat 파라미터 사용
    inside_mat,   // [한국어] MAT 내부 (비트라인, 센스앰프 등) — wire_inside_mat 파라미터 사용
    local_wires   // [한국어] 로컬 와이어 (셀 내부 연결 등) — wire_local 파라미터 사용
};



/* [한국어] Htree_type — H-tree 네트워크의 신호 방향과 종류를 구분하는 enum.
 * H-tree는 주소/데이터를 캐시 서브어레이까지 균등하게 분배하는 이진 트리 네트워크이다. */
enum Htree_type {
    Add_htree,        // [한국어] 주소(address) H-tree — 행/열 주소를 서브어레이까지 전달
    Data_in_htree,    // [한국어] 데이터 입력 H-tree — 쓰기 데이터를 서브어레이로 분배
    Data_out_htree,   // [한국어] 데이터 출력 H-tree — 읽기 데이터를 출력 드라이버로 수집
    Search_in_htree,  // [한국어] CAM(Content Addressable Memory) 검색 입력 H-tree
    Search_out_htree, // [한국어] CAM 검색 결과 출력 H-tree (매치라인 결과 수집)
};

/* [한국어] Gate_type — 논리 게이트 유형을 구분하는 enum.
 * cmos_Isub_leakage() 및 cmos_Ig_leakage()에서 게이트 유형별 평균 누설전류 계산에 사용. */
enum Gate_type {
    nmos, // [한국어] NMOS 단독 패스 트랜지스터 또는 NMOS 단일 게이트
    pmos, // [한국어] PMOS 단독 패스 트랜지스터 또는 PMOS 단일 게이트
	inv,  // [한국어] CMOS 인버터 (NMOS + PMOS 각 1개)
    nand, // [한국어] NAND 게이트 (PMOS 병렬 + NMOS 직렬 스택)
    nor,  // [한국어] NOR 게이트 (PMOS 직렬 스택 + NMOS 병렬)
    tri,  // [한국어] 트라이스테이트(tri-state) 버퍼 — 인버터 + 패스 게이트 제어
    tg    // [한국어] 트랜스미션 게이트(transmission gate) — NMOS + PMOS 병렬 패스 트랜지스터
};

/* [한국어] Half_net_topology — NMOS 또는 PMOS 하프 네트워크의 트랜지스터 연결 방식.
 * 누설전류 계산 시 병렬(parallel) 연결과 직렬(series) 연결의 스택 효과를 다르게 처리. */
enum Half_net_topology {
    parallel, // [한국어] 병렬 연결 — 하나라도 ON이면 전류가 흐름; 모두 OFF여야 누설이 최소화
    series    // [한국어] 직렬 연결 — 스택 효과로 누설이 지수적으로 감소 (UNI_LEAK_STACK_FACTOR 적용)
};

/* [한국어]
 * logtwo - 밑이 2인 로그(log₂) 계산
 * @x    : 입력값 (양수여야 하며, assert로 검사됨)
 * @return: log₂(x) 값 (double)
 * 디코더 비트 수 계산 등 여러 곳에서 사용되는 부동소수점 이진 로그.
 * 호출 체인: 다수의 CACTI 상위 모듈 → [logtwo] → log()/log(2.0) */
double logtwo (double x);

/* [한국어]
 * gate_C - MOSFET 게이트 커패시턴스(farads) 계산
 * @width      : 게이트 폭 (um 단위)
 * @wirelength : 게이트로 들어가는 폴리 와이어 길이 (lambda 단위, 현재 사용 안 됨)
 * @_is_dram   : DRAM 셀 접근 트랜지스터 여부 (g_tp.dram_acc 사용)
 * @_is_sram   : SRAM 셀 접근 트랜지스터 여부 (g_tp.sram_cell 사용)
 * @_is_wl_tr  : 워드라인 트랜지스터 여부 (g_tp.dram_wl 사용)
 * @return     : 게이트 커패시턴스 값 (F)
 * 수식: (C_g_ideal + C_overlap + 3*C_fringe)*width + l_phy*Cpolywire.
 * 트랜지스터 유형에 따라 DeviceType을 선택 후 동일 수식을 적용한다.
 * 호출 체인: decoder/predecoder/crossbar 등 → [gate_C] → g_tp.{peri_global,dram_acc,...} */
double gate_C(
    double width,
    double wirelength,
    bool _is_dram = false,
    bool _is_sram = false,
    bool _is_wl_tr = false);

/* [한국어]
 * gate_C_pass - 패스 트랜지스터의 게이트 커패시턴스 계산 (gate_C와 동일 구현)
 * @width      : 게이트 폭 (um 단위)
 * @wirelength : 폴리 와이어 길이 (lambda 단위, 현재 사용 안 됨)
 * @_is_dram / @_is_sram / @_is_wl_tr : 트랜지스터 유형 선택 플래그
 * @return     : 게이트 커패시턴스 값 (F)
 * v5.0부터 gate_C()와 동일한 수식을 사용한다. 히스토리적으로 패스 트랜지스터를
 * 별도 처리하려던 흔적이 남아 있음. 호출 체인: 일부 레거시 모듈 → [gate_C_pass] */
double gate_C_pass(
    double width,
    double wirelength,
    bool   _is_dram = false,
    bool   _is_sram = false,
    bool   _is_wl_tr = false);

/* [한국어]
 * drain_C_ - MOSFET 드레인 커패시턴스(farads) 계산 (폴딩 포함)
 * @width       : 트랜지스터 게이트 폭 (um)
 * @nchannel    : 1이면 NMOS, 0이면 PMOS
 * @stack       : 직렬 스택 트랜지스터 수 (NAND: stack=fanin)
 * @next_arg_thresh_folding_width_or_height_cell : 0이면 fold_dimension을 폴딩 임계폭으로,
 *               1이면 셀 높이로 해석
 * @fold_dimension : 폴딩 임계폭(um) 또는 셀 높이(um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return      : 드레인 커패시턴스 (F) — 접합 면적 + 측벽 + 프린지/오버랩 + 폴딩 메탈 기여 합산
 * 트랜지스터가 임계폭을 초과하면 폴딩(folding)하여 드레인 공유를 고려한다.
 * 폴딩 메탈 연결 커패시턴스(drain_C_metal_connecting_folded_tr)를 추가로 포함.
 * 호출 체인: decoder/predecoder/mat/wire → [drain_C_] → g_tp.DeviceType */
double drain_C_(
    double width,
    int nchannel,
    int stack,
    int next_arg_thresh_folding_width_or_height_cell,
    double fold_dimension,
    bool _is_dram = false,
    bool _is_sram = false,
    bool _is_wl_tr = false);

/* [한국어]
 * tr_R_on - 트랜지스터 온(on) 저항(ohm) 계산
 * @width     : 게이트 폭 (um) — 폭이 클수록 저항 감소
 * @nchannel  : 1이면 NMOS(R_nch_on 사용), 0이면 PMOS(R_pch_on 사용)
 * @stack     : 직렬 스택 수 — 직렬 연결 시 저항이 stack배로 증가
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형 선택
 * @return    : 온 저항 = stack * R_{n/p}ch_on / width (ohm)
 * 타이밍 모델(horowitz)에서 tf = R_on * C_load 계산의 핵심 입력으로 사용된다.
 * 호출 체인: decoder/predecoder/crossbar → [tr_R_on] → g_tp.DeviceType.R_nch_on */
double tr_R_on(
    double width,
    int nchannel,
    int stack,
    bool _is_dram = false,
    bool _is_sram = false,
    bool _is_wl_tr = false);

/* [한국어]
 * R_to_w - 목표 온 저항으로부터 트랜지스터 폭 역산
 * @res      : 목표 저항값 (ohm)
 * @nchannel : 1이면 NMOS, 0이면 PMOS
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return   : 트랜지스터 폭 (um) = R_nch_on / res
 * 데이터 워드라인 드라이버 크기 결정에 사용 (tr_R_on의 역연산).
 * 호출 체인: wordline 드라이버 크기 계산 → [R_to_w] */
double R_to_w(
    double res,
    int nchannel,
    bool _is_dram = false,
    bool _is_sram = false,
    bool _is_wl_tr = false);

/* [한국어]
 * horowitz - Horowitz(1984) RC 타이밍 모델로 게이트 전파 지연 계산
 * @inputramptime : 입력 신호 상승/하강 시간 (s) — 이전 게이트의 출력 슬루율
 * @tf            : 게이트 시정수 = R_on * C_load (s)
 * @vs1           : 입력 임계전압 비율 (Vth/Vdd)
 * @vs2           : 출력 임계전압 비율 (Vth/Vdd)
 * @rise          : RISE(1)이면 출력 상승, FALL(0)이면 출력 하강
 * @return        : 게이트 전파 지연 td (s)
 * 수식(RISE): td = tf*sqrt(ln(vs1)^2 + 2*a*b*(1-vs1)) + tf*(ln(vs1) - ln(vs2)), b=0.5
 * 수식(FALL): 대칭적 형태, b=0.4. inputramptime==0이면 단순 tf*ln(1/vs1) 사용.
 * 호출 체인: decoder/predecoder/crossbar/mat → [horowitz] (타이밍 누적에 핵심) */
double horowitz (
    double inputramptime,
    double tf,
    double vs1,
    double vs2,
    int rise);

/* [한국어]
 * pmos_to_nmos_sz_ratio - PMOS/NMOS 전류 구동력 비율(sizing ratio) 반환
 * @_is_dram  : DRAM 여부
 * @_is_wl_tr : 워드라인 트랜지스터 여부
 * @return    : n_to_p_eff_curr_drv_ratio (PMOS 폭 = ratio * NMOS 폭으로 대칭 구동력 확보)
 * 게이트 크기 결정(logical_effort) 시 PMOS 크기를 NMOS 기준으로 결정하기 위해 사용.
 * 호출 체인: decoder/predecoder compute_widths → [pmos_to_nmos_sz_ratio] */
double pmos_to_nmos_sz_ratio(
    bool _is_dram = false,
    bool _is_wl_tr = false);

/* [한국어]
 * simplified_nmos_leakage - NMOS 단위 폭당 서브임계 누설전류 반환 (I_off_n * width)
 * @nwidth    : NMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : nwidth * I_off_n (A)
 * cmos_Isub_leakage() 내부에서 단위 누설전류 기준값으로 호출된다.
 * 호출 체인: cmos_Isub_leakage → [simplified_nmos_leakage] */
double simplified_nmos_leakage(
    double nwidth,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr = false);

/* [한국어]
 * simplified_pmos_leakage - PMOS 단위 폭당 서브임계 누설전류 반환 (I_off_p * width)
 * @pwidth    : PMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : pwidth * I_off_p (A)
 * 호출 체인: cmos_Isub_leakage → [simplified_pmos_leakage] */
double simplified_pmos_leakage(
    double pwidth,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr = false);


/* [한국어]
 * cmos_Ileak - CMOS 인버터/게이트의 총 서브임계 누설전류 (단순 합산)
 * @nWidth / @pWidth : NMOS/PMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return : nWidth*I_off_n + pWidth*I_off_p (A) — 단순 합산, 상태 평균 없음
 * cmos_Isub_leakage()보다 단순한 모델; 일부 모듈에서 빠른 추정에 사용.
 * 호출 체인: 일부 면적/전력 추정 모듈 → [cmos_Ileak] */
double cmos_Ileak(
    double nWidth,
    double pWidth,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr = false);

/* [한국어]
 * cmos_Ig_n - NMOS 게이트 터널링 누설전류 반환 (I_g_on_n * nWidth)
 * @nWidth    : NMOS 게이트 폭 (um)
 * @return    : nWidth * I_g_on_n (A)
 * 극소 공정 노드에서 게이트 산화막이 얇아져 발생하는 터널링 전류를 모델링.
 * 호출 체인: cmos_Ig_leakage → [cmos_Ig_n] */
double cmos_Ig_n(
    double nWidth,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr= false);

/* [한국어]
 * cmos_Ig_p - PMOS 게이트 터널링 누설전류 반환 (I_g_on_p * pWidth)
 * @pWidth    : PMOS 게이트 폭 (um)
 * @return    : pWidth * I_g_on_p (A)
 * 호출 체인: cmos_Ig_leakage → [cmos_Ig_p] */
double cmos_Ig_p(
    double pWidth,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr= false);


/* [한국어]
 * cmos_Isub_leakage - 게이트 유형별 평균 서브임계 누설전류 계산
 * @nWidth / @pWidth : NMOS/PMOS 게이트 폭 (um)
 * @fanin     : 게이트 입력 수 (1~N)
 * @g_type    : 게이트 유형 (inv/nand/nor/tri/tg/nmos/pmos)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @topo      : 하프 네트워크 토폴로지 (series/parallel)
 * @return    : 통계적 평균 서브임계 누설전류 (A)
 * 가능한 모든 입력 상태(2^fanin)에 대해 누설전류를 계산하고 평균한다.
 * UNI_LEAK_STACK_FACTOR로 직렬 스택 효과를 보정하고 이항계수로 상태 조합수를 반영.
 * 호출 체인: decoder/predecoder/crossbar compute_area → [cmos_Isub_leakage] */
double cmos_Isub_leakage(
    double nWidth,
    double pWidth,
    int    fanin,
    enum Gate_type g_type,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr = false,
    enum Half_net_topology topo = series);

/* [한국어]
 * cmos_Ig_leakage - 게이트 유형별 평균 게이트 터널링 누설전류 계산
 * @nWidth / @pWidth : NMOS/PMOS 게이트 폭 (um)
 * @fanin     : 게이트 입력 수
 * @g_type    : 게이트 유형
 * @topo      : 하프 네트워크 토폴로지
 * @return    : 통계적 평균 게이트 터널링 누설전류 (A)
 * ON 상태 트랜지스터에서 발생하는 게이트 터널링을 모든 입력 상태에 대해 평균.
 * 호출 체인: decoder/predecoder/crossbar compute_area → [cmos_Ig_leakage] */
double cmos_Ig_leakage(
    double nWidth,
    double pWidth,
    int    fanin,
    enum Gate_type g_type,
    bool _is_dram = false,
    bool _is_cell = false,
    bool _is_wl_tr = false,
    enum Half_net_topology topo = series);

/* [한국어]
 * shortcircuit - CMOS 인버터 단락전류(short-circuit) 에너지 계산 (완전 모델)
 * @vt            : 임계전압 (V)
 * @velocity_index: 속도 포화 지수
 * @c_in/c_out    : 입출력 커패시턴스 (F)
 * @w_nmos/w_pmos : NMOS/PMOS 폭 (um)
 * @i_on_n/i_on_p : 출력 NMOS/PMOS 드라이브 전류 (A/um)
 * @i_on_n_in/i_on_p_in : 입력측 드라이브 전류 (A/um)
 * @vdd           : 공급 전압 (V)
 * @return        : 단락전류 에너지 (J) — 현재 구현은 0 반환(미완성)
 * 호출 체인: 전력 모델 → [shortcircuit] (현재 shortcircuit_simple로 대체됨) */
double shortcircuit(
    double vt,
    double velocity_index,
    double c_in,
    double c_out,
    double w_nmos,
    double w_pmos,
    double i_on_n,
    double i_on_p,
    double i_on_n_in,
    double i_on_p_in,
    double vdd);

/* [한국어]
 * shortcircuit_simple - CMOS 인버터 단락전류 에너지 간소화 모델
 * @vt / @velocity_index / @c_in / @c_out / @w_nmos / @w_pmos : 동일
 * @i_on_n/p / @i_on_n_in/p_in / @vdd : 동일
 * @return : 단락전류 에너지 (J) — 방전/충전 경로 단락전류의 평균
 * 수식: p_short_circuit = (p_discharge + p_charge) / 2
 * fo_n/fo_p (fanout 비율), beta_ratio (PMOS/NMOS 전류비)로 정규화하여 계산.
 * 호출 체인: 단락전류 전력 추정 → [shortcircuit_simple] */
double shortcircuit_simple(
    double vt,
    double velocity_index,
    double c_in,
    double c_out,
    double w_nmos,
    double w_pmos,
    double i_on_n,
    double i_on_p,
    double i_on_n_in,
    double i_on_p_in,
    double vdd);

/* [한국어]
 * set_pppm - 전력 포인트 프로덕트 마스크(power point product mask) 벡터 설정
 * @pppv : 4원소 double 배열 포인터 — [dynamic, leakage, gate_leakage, short_circuit] 인수
 * @a,b,c,d : 각 전력 성분에 곱할 가중치 (기본값 1.0)
 * 특정 사이클/상태에서 전력 기여분을 스케일링하기 위한 마스크 벡터.
 * 인라인 함수로 컴파일러 최적화(코드 삽입)가 가능하다.
 * 호출 체인: 전력 누적 코드 → [set_pppm] */
//set power point product mask; strictly speaking this is not real point product
inline void set_pppm(
	double * pppv,
	double a=1,
    double b=1,
    double c=1,
    double d=1
    ){
		pppv[0]= a; // [한국어] dynamic 전력 가중치
		pppv[1]= b; // [한국어] leakage 전력 가중치
		pppv[2]= c; // [한국어] gate_leakage 전력 가중치
		pppv[3]= d; // [한국어] short_circuit 전력 가중치

}

/* [한국어]
 * set_sppm - 검색 연산(search) 전력 포인트 프로덕트 마스크 설정 (3원소)
 * @sppv : 3원소 double 배열 포인터
 * @a,b,c : 각 전력 성분 가중치 (기본값 1.0)
 * CAM(Content Addressable Memory) 검색 연산에서 3가지 전력 성분을 스케일링.
 * 호출 체인: CAM 전력 계산 → [set_sppm] */
inline void set_sppm(
	double * sppv,
	double a=1,
    double b=1,
    double c=1,
    double d=1
    ){
		sppv[0]= a; // [한국어] 첫 번째 전력 성분 가중치
		sppv[1]= b; // [한국어] 두 번째 전력 성분 가중치
		sppv[2]= c; // [한국어] 세 번째 전력 성분 가중치
}

#endif
