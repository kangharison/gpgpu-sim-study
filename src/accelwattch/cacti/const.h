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
 * [한국어 설명] CACTI 전역 상수 정의 헤더 (const.h)
 *
 * === 파일의 역할 ===
 * CACTI 전체에서 공유하는 수치 상수, 매크로, 열거형, 전역 배열을 정의한다.
 * 캐시 주소 비트 수(ADDRESS_BITS), 서브어레이 크기 한계(MAXSUBARRAYS), 타이밍 모델 상수
 * (fopt, VTHFA* 임계 전압), 게이트 타입(INV/NOR/NAND), 누설 스택 팩터, 물리 상수
 * (구리 비저항, 유전율), 전력 분류 배열(pppm_*)을 포함한다. 이 파일의 상수들은
 * CACTI 레이아웃·전력·타이밍 계산의 모든 경로에서 참조된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * const.h는 CACTI 전체 헤더 의존 트리의 최하위 기반 파일이다. parameter.h,
 * basic_circuit.h, component.h 등 모든 핵심 헤더가 이 파일에 정의된 상수를 직간접
 * 참조한다. 실행 흐름: AccelWattch → cacti_interface → parameter.h → const.h.
 * 모든 CACTI 함수는 fopt, MAX_NUMBER_GATES_STAGE, INV/NOR/NAND 등을 이 파일에서 얻는다.
 *
 * === 타 모듈과의 연결 ===
 * - parameter.h: g_ip, g_tp 정의에 const.h 상수 활용 (NUMBER_TECH_FLAVORS 등).
 * - basic_circuit.h: INV/NOR/NAND 게이트 타입 상수, 누설 스택 팩터 사용.
 * - component.cc: fopt, MAX_NUMBER_GATES_STAGE, INV/NOR/NAND 사용.
 * - wire.h, htree2.h: WIRE_TYPES, ROUTER_TYPES 등 사용.
 * - 모든 CACTI .cc 파일: MAX/MIN 매크로, BIGNUM, INF, RISE/FALL/NCH/PCH 상수 사용.
 *
 * === 주요 함수/구조체 요약 ===
 * - ADDRESS_BITS(42): 모델링하는 물리 주소 비트 수 (Power4 이상 시스템 기준).
 * - fopt(4.0): Logical Effort 최적 팬아웃 상수 (RC 지연 최소화).
 * - ram_cell_tech_type_num: SRAM/DRAM 셀 기술 종류 열거형 (itrs_hp/lstp/lop/lp_dram/comm_dram).
 * - pppm[]/pppm_lkg[]/pppm_dyn[] 등: 전력 컴포넌트 선택 마스크 배열.
 * - MAX_NUMBER_GATES_STAGE(20): logical_effort() 결과 배열 최대 크기.
 */

#ifndef __CONST_H__
#define __CONST_H__

#include <stdint.h>  // [한국어] uint32_t 등 고정 폭 정수 타입 (sram_num_cells_wl_stitching_ 등)
#include <stdlib.h>  // [한국어] exit() — 치명적 오류 시 종료 (compute_gate_area 미지원 타입)
#include <string.h>  // [한국어] 문자열 함수 (const.h 자체에서 직접 미사용, 포함 체인)
#include <stdio.h>   // [한국어] printf 등 (const.h 자체에서 직접 미사용)
#include <math.h>    // [한국어] log(), pow() 등 수학 함수 (logical_effort 등에서 사용)

/*  The following are things you might want to change
 *  when compiling
 */

/*
 * Address bits in a word, and number of output bits from the cache
 */

/*
was: #define ADDRESS_BITS 32
now: I'm using 42 bits as in the Power4,
since that's bigger then the 36 bits on the Pentium 4
and 40 bits on the Opteron
*/
const int ADDRESS_BITS = 42;
// [한국어] 물리 주소 비트 수 = 42. IBM Power4 기준 (Pentium4의 36비트, Opteron의 40비트보다 큰 값).
//         캐시 태그 계산의 기준이 된다: 태그 비트 = ADDRESS_BITS - 인덱스 비트 - 오프셋 비트.

/*dt: In addition to the tag bits, the tags also include 1 valid bit, 1 dirty bit, 2 bits for a 4-state
  cache coherency protocoll (MESI), 1 bit for MRU (change this to log(ways) for full LRU).
  So in total we have 1 + 1 + 2 + 1 = 5 */
const int EXTRA_TAG_BITS = 5;
// [한국어] 추가 태그 비트 수 = 5. 주소 태그 외에 다음 정보를 저장:
//         valid(1) + dirty(1) + MESI 코히런시 상태(2) + MRU 비트(1) = 5비트.
//         완전 LRU를 구현하려면 log2(ways)비트가 필요하지만, 여기서는 근사로 1비트 사용.

/* limits on the various N parameters */

const unsigned int MAXDATAN     = 512;      // maximum for Ndwl and Ndbl
// [한국어] Ndwl(데이터 서브어레이 수평 분할)과 Ndbl(수직 분할)의 최대값 = 512.
const unsigned int MAXSUBARRAYS = 1048576;  // maximum subarrays for data and tag arrays
// [한국어] 데이터/태그 배열의 최대 서브어레이 수 = 2^20. 탐색 공간의 상한.
const unsigned int MAXDATASPD   = 256;      // maximum for Nspd
// [한국어] Nspd(데이터 열 다중화 비율)의 최대값 = 256.
const unsigned int MAX_COL_MUX  = 256;
// [한국어] 열 다중화(column mux) 비율의 최대값 = 256. 센스앰프 공유 비율 상한.



#define ROUTER_TYPES 3
// [한국어] 지원하는 라우터 타입 수 = 3 (NUCA 라우터 구성 옵션 수).
#define WIRE_TYPES 6
// [한국어] 지원하는 배선 타입 수 = 6 (local, semi-global, global + aggressive/conservative 조합).

const double Cpolywire = 0;
// [한국어] poly 배선 커패시턴스 = 0. 현재 모델에서 poly 배선을 별도로 모델링하지 않음 (근사).


/* Threshold voltages (as a proportion of Vdd)
   If you don't know them, set all values to 0.5 */
#define VTHFA1         0.452
// [한국어] FA(완전 연관) 캐시 관련 임계 전압 1 = Vdd의 45.2%. 비교기/평가 인버터 전환점.
#define VTHFA2         0.304
// [한국어] FA 임계 전압 2 = Vdd의 30.4%. 다른 게이트 단의 전환점.
#define VTHFA3         0.420
// [한국어] FA 임계 전압 3 = Vdd의 42.0%.
#define VTHFA4         0.413
// [한국어] FA 임계 전압 4 = Vdd의 41.3%.
#define VTHFA5         0.405
// [한국어] FA 임계 전압 5 = Vdd의 40.5%.
#define VTHFA6         0.452
// [한국어] FA 임계 전압 6 = Vdd의 45.2%.
#define VSINV          0.452
// [한국어] 인버터 전환 전압(Switching voltage of an Inverter) = Vdd의 45.2%.
//         Horowitz 지연 모델에서 입력 임계값으로 사용.
#define VTHCOMPINV     0.437
// [한국어] 비교기 인버터(Comparator Inverter) 전환 전압 = Vdd의 43.7%.
#define VTHMUXNAND     0.548  // TODO : this constant must be revisited
// [한국어] MUX NAND 게이트 전환 전압 = Vdd의 54.8%. TODO: 재검토 필요 (주석 참조).
#define VTHEVALINV     0.452
// [한국어] 평가 인버터(Evaluate Inverter) 전환 전압 = Vdd의 45.2%. 동적 게이트 평가 단계.
#define VTHSENSEEXTDRV 0.438
// [한국어] 센스앰프 외부 드라이버 전환 전압 = Vdd의 43.8%.


//WmuxdrvNANDn and WmuxdrvNANDp are no longer being used but it's part of the old
//delay_comparator function which we are using exactly as it used to be, so just setting these to 0
const double WmuxdrvNANDn = 0;
// [한국어] MUX 드라이버 NAND NMOS 폭 = 0. 구형 delay_comparator() 함수에서 사용하던 값이나
//         현재는 더 이상 사용하지 않아 0으로 설정 (하위 호환 유지).
const double WmuxdrvNANDp = 0;
// [한국어] MUX 드라이버 NAND PMOS 폭 = 0. WmuxdrvNANDn과 동일한 이유로 0.


/*===================================================================*/
/*
 * The following are things you probably wouldn't want to change.
 */

#define BIGNUM 1e30
// [한국어] 사실상 무한대를 나타내는 큰 수 (1e30). 지연/전력 초기값으로 사용하여 최소값 탐색.
#define INF 9999999
// [한국어] 정수형 무한대 값. 배열 인덱스나 정수 비교에서 "도달 불가" 상태를 표현.
#define MAX(a,b) (((a)>(b))?(a):(b))
// [한국어] 두 값 중 큰 값 반환 매크로. CACTI 전반에서 최솟값 경계 적용에 사용.
#define MIN(a,b) (((a)<(b))?(a):(b))
// [한국어] 두 값 중 작은 값 반환 매크로.

/* Used to communicate with the horowitz model */
#define RISE 1
// [한국어] 신호 상승 엣지(Rising edge) = 1. horowitz() 타이밍 모델에 입력.
#define FALL 0
// [한국어] 신호 하강 엣지(Falling edge) = 0.
#define NCH  1
// [한국어] NMOS 채널 타입 = 1. drain_C_() 등 회로 함수에서 트랜지스터 타입 지정.
#define PCH  0
// [한국어] PMOS 채널 타입 = 0.


#define EPSILON 0.5 //v4.1: This constant is being used in order to fix floating point -> integer
//conversion problems that were occuring within CACTI. Typical problem that was occuring was
//that with different compilers a floating point number like 3.0 would get represented as either
//2.9999....or 3.00000001 and then the integer part of the floating point number (3.0) would
//be computed differently depending on the compiler. What we are doing now is to replace
//int (x) with (int) (x+EPSILON) where EPSILON is 0.5. This would fix such problems. Note that
//this works only when x is an integer >= 0.
// [한국어] EPSILON = 0.5: 부동소수점 → 정수 변환 시 컴파일러 간 차이를 보정.
//         예: 3.0이 2.9999로 표현될 때 int(3.0+0.5)=3으로 올바르게 변환.
//         단, CACTI 6.5 이후에는 (int)ceil()을 권장 (아래 Sheng 주석 참조).
/*
 * Sheng thinks this is more a solution to solve the simple truncate problem
 * (http://www.cs.tut.fi/~jkorpela/round.html) rather than the problem mentioned above.
 * Unfortunately, this solution causes nasty bugs (different results when using O0 and O3).
 * Moreover, round is not correct in CACTI since when an extra fraction of bit/line is needed,
 * we need to provide a complete bit/line even the fraction is just 0.01.
 * So, in later version than 6.5 we use (int)ceil() to get double to int conversion.
 */

#define EPSILON2 0.1
// [한국어] 보조 엡실론 = 0.1. 일부 경계 조건 판별에 사용.
#define EPSILON3 0.6
// [한국어] 보조 엡실론 = 0.6. ceil() 대체 패턴의 변형.


#define MINSUBARRAYROWS 16 //For simplicity in modeling, for the row decoding structure, we assume
//that each row predecode block is composed of at least one 2-4 decoder. When the outputs from the
//row predecode blocks are combined this means that there are at least 4*4=16 row decode outputs
// [한국어] 서브어레이 최소 행 수 = 16. 행 프리디코드 블록이 최소 2-to-4 디코더(4출력)×2단 구조를
//         가정하므로 최소 4×4=16개의 행 디코드 출력이 존재한다.
#define MAXSUBARRAYROWS 262144 //Each row predecode block produces a max of 2^9 outputs. So
//the maximum number of row decode outputs will be 2^9*2^9
// [한국어] 서브어레이 최대 행 수 = 2^18 = 262144. 각 프리디코드 블록이 최대 2^9 출력 → 2^9×2^9.
#define MINSUBARRAYCOLS 2
// [한국어] 서브어레이 최소 열 수 = 2. 비트라인 쌍(BL/BLB) 최소 1쌍.
#define MAXSUBARRAYCOLS 262144
// [한국어] 서브어레이 최대 열 수 = 2^18. 탐색 공간 상한.


#define INV 0
// [한국어] 인버터(Inverter) 게이트 타입 식별자. compute_gate_area() switch 분기에 사용.
#define NOR 1
// [한국어] NOR 게이트 타입 식별자. 아비터(arbiter)의 NOR 체인과 게이트 면적 계산에 사용.
#define NAND 2
// [한국어] NAND 게이트 타입 식별자.


#define NUMBER_TECH_FLAVORS 4
// [한국어] 지원 기술 변형 수 = 4 (itrs_hp, itrs_lstp, itrs_lop, lp_dram).
//         parameter.h의 TechnologyParameter 배열 크기에 사용.

#define NUMBER_INTERCONNECT_PROJECTION_TYPES 2 //aggressive and conservative
//0 = Aggressive projections, 1 = Conservative projections
// [한국어] 배선 예측 타입 수 = 2. 0=공격적(aggressive, 낙관적 RC), 1=보수적(conservative, 비관적 RC).
#define NUMBER_WIRE_TYPES 4 //local, semi-global and global
//1 = 'Semi-global' wire type, 2 = 'Global' wire type
// [한국어] 배선 타입 수 = 4: local(셀 내), semi-global(블록 간), global(칩 전체), 기타.


const int dram_cell_tech_flavor = 3;
// [한국어] DRAM 셀 기술 타입 인덱스 = 3 = lp_dram (저전력 DRAM).
//         parameter.h의 TechnologyParameter 배열에서 DRAM 공정 파라미터 행을 선택.


#define VBITSENSEMIN 0.08 //minimum bitline sense voltage is fixed to be 80 mV.
// [한국어] 비트라인 최소 센싱 전압 = 0.08V(80mV). 센스앰프가 감지할 수 있는 최소 신호 스윙.
//         이보다 작은 비트라인 전압 차이는 신뢰성 있게 판별할 수 없다.

#define fopt 4.0
// [한국어] Logical Effort 최적 팬아웃 상수 = 4.0.
//         각 버퍼/인버터 단의 팬아웃이 4.0일 때 RC 지연이 최소화된다 (Sutherland 1999 이론).
//         logical_effort()에서 최적 게이트 단 수(num_gates) 계산의 기준.

#define INPUT_WIRE_TO_INPUT_GATE_CAP_RATIO 0
// [한국어] 입력 배선 커패시턴스 대 게이트 커패시턴스 비율 = 0. 현재 모델에서 배선 커패시턴스 무시.
#define BUFFER_SEPARATION_LENGTH_MULTIPLIER 1
// [한국어] 버퍼 간격 길이 배율 = 1. 배선 길이에 곱하여 버퍼 삽입 간격을 결정.
#define NUMBER_MATS_PER_REDUNDANT_MAT 8
// [한국어] 여분 매트(redundant mat) 1개당 일반 매트 수 = 8. 에러 복구용 여분 자원 비율.

#define NUMBER_STACKED_DIE_LAYERS 1
// [한국어] 적층 다이(stacked die) 레이어 수 = 1. 3D 집적 미사용 시 기본값.

// this variable can be set to carry out solution optimization for
// a maximum area allocation.
#define STACKED_DIE_LAYER_ALLOTED_AREA_mm2 0 //6.24 //6.21//71.5
// [한국어] 적층 다이 레이어당 허용 면적(mm²) = 0 (비활성). 면적 제한 최적화 시 설정.

// this variable can also be employed when solution optimization
// with maximum area allocation is carried out.
#define MAX_PERCENT_AWAY_FROM_ALLOTED_AREA 50
// [한국어] 허용 면적에서 최대 허용 편차(%) = 50%. 면적 제한 최적화의 허용 오차 범위.

// this variable can also be employed when solution optimization
// with maximum area allocation is carried out.
#define MIN_AREA_EFFICIENCY 20
// [한국어] 최소 면적 효율(%) = 20%. 사용 면적 / 할당 면적 비율의 하한.

// this variable can be employed when solution with a desired
// aspect ratio is required.
#define STACKED_DIE_LAYER_ASPECT_RATIO 1
// [한국어] 목표 가로/세로 비율 = 1 (정사각형). 종횡비 제약 최적화 시 사용.

// this variable can be employed when solution with a desired
// aspect ratio is required.
#define MAX_PERCENT_AWAY_FROM_ASPECT_RATIO 101
// [한국어] 종횡비 편차 최대 허용(%) = 101% (사실상 무제한). 종횡비 제약 미사용 시 기본값.

// this variable can be employed to carry out solution optimization
// for a certain target random cycle time.
#define TARGET_CYCLE_TIME_ns 1000000000
// [한국어] 목표 사이클 타임(나노초) = 1e9 ns (사실상 무제한). 사이클 타임 제약 최적화 시 설정.

#define NUMBER_PIPELINE_STAGES 4
// [한국어] 파이프라인 단 수 = 4. cycle_time = delay / NUMBER_PIPELINE_STAGES로 계산되어
//         캐시가 파이프라인화될 때 클럭 주기를 결정.

// this can be used to model the length of interconnect
// between a bank and a crossbar
#define LENGTH_INTERCONNECT_FROM_BANK_TO_CROSSBAR 0 //3791 // 2880//micron
// [한국어] 뱅크-크로스바 간 인터커넥트 배선 길이(µm) = 0 (비활성). NUCA 라우터 지연 모델링 시 사용.

#define IS_CROSSBAR 0
// [한국어] 크로스바 모드 활성화 플래그 = 0 (비활성). 1로 설정 시 크로스바 전용 계산 경로 사용.
#define NUMBER_INPUT_PORTS_CROSSBAR 8
// [한국어] 크로스바 입력 포트 수 = 8. IS_CROSSBAR=1일 때 Crossbar 생성에 사용.
#define NUMBER_OUTPUT_PORTS_CROSSBAR 8
// [한국어] 크로스바 출력 포트 수 = 8.
#define NUMBER_SIGNALS_PER_PORT_CROSSBAR 256
// [한국어] 크로스바 포트당 신호 수 = 256비트 (플릿 크기).


#define MAT_LEAKAGE_REDUCTION_DUE_TO_SLEEP_TRANSISTORS_FACTOR 1
// [한국어] 슬립 트랜지스터로 인한 매트 누설 감소 계수 = 1 (감소 없음). 슬립 모드 미사용 시 기본값.
#define LEAKAGE_REDUCTION_DUE_TO_LONG_CHANNEL_HP_TRANSISTORS_FACTOR 1
// [한국어] 긴 채널 HP 트랜지스터로 인한 누설 감소 계수 = 1 (감소 없음).

#define PAGE_MODE 0
// [한국어] 페이지 모드 활성화 = 0 (비활성). DRAM 페이지 모드 접근 최적화 시 1로 설정.

#define MAIN_MEM_PER_CHIP_STANDBY_CURRENT_mA 60
// We are actually not using this variable in the CACTI code. We just want to acknowledge that
// this current should be multiplied by the DDR(n) system VDD value to compute the standby power
// consumed during precharge.
// [한국어] 칩당 메인 메모리 스탠바이 전류 = 60mA. 프리차지 시 대기 전력 참조용.
//         CACTI 코드에서 직접 사용하지 않음 — 참고 문서 목적.


const double VDD_STORAGE_LOSS_FRACTION_WORST = 0.125;
// [한국어] DRAM 셀의 최악 케이스 전하 손실 비율 = 12.5%. 리프레시 전 최소 잔류 전압 비율.
const double CU_RESISTIVITY = 0.022; //ohm-micron
// [한국어] 구리(Cu) 배선 비저항 = 0.022 Ω·µm. 배선 저항 계산에 사용: R = CU_RESISTIVITY × 길이 / 단면적.
const double BULK_CU_RESISTIVITY = 0.018; //ohm-micron
// [한국어] 벌크 구리 비저항 = 0.018 Ω·µm. 실제 배선에서는 입자 경계 산란 등으로 0.022 Ω·µm 사용.
const double PERMITTIVITY_FREE_SPACE = 8.854e-18; //F/micron
// [한국어] 진공 유전율 = 8.854×10⁻¹⁸ F/µm (SI 단위 8.854×10⁻¹² F/m를 µm 단위로 변환).
//         배선 커패시턴스 계산의 기본 상수: C = ε₀ × εr × 면적 / 거리.

const static uint32_t sram_num_cells_wl_stitching_ = 16;
// [한국어] SRAM 워드라인 스티칭(stitching) 당 셀 수 = 16. 긴 워드라인을 분할하는 간격.
//         스티칭은 워드라인을 여러 세그먼트로 나눠 RC 지연을 줄이는 기법.
const static uint32_t dram_num_cells_wl_stitching_ = 64;
// [한국어] LP-DRAM 워드라인 스티칭 당 셀 수 = 64. SRAM보다 셀 간격이 크므로 더 긴 세그먼트.
const static uint32_t comm_dram_num_cells_wl_stitching_ = 256;
// [한국어] 상용 DRAM(comm_dram) 워드라인 스티칭 당 셀 수 = 256.
const static double num_bits_per_ecc_b_          = 8.0;
// [한국어] ECC(오류 수정 코드) 비트당 데이터 비트 수 = 8. SECDED(단일 비트 수정, 이중 비트 검출) 기준.

const double    bit_to_byte  = 8.0;
// [한국어] 1바이트 = 8비트. 비트·바이트 단위 변환 상수.

#define MAX_NUMBER_GATES_STAGE 20
// [한국어] logical_effort()의 w_n[], w_p[] 배열 최대 크기 = 20단.
//         버퍼 체인이 20단을 넘을 경우 assert 실패로 설계 오류를 감지.
#define MAX_NUMBER_HTREE_NODES 20
// [한국어] H-tree 최대 노드 수 = 20. Htree2 재귀 분할의 최대 깊이.
#define NAND2_LEAK_STACK_FACTOR 0.2
// [한국어] NAND2 게이트의 누설 스택 인수 = 0.2. 직렬 스택 트랜지스터의 누설 감소 효과:
//         직렬 2개 연결 시 누설 = 단일 트랜지스터 누설 × 0.2 (바디 효과로 Vth 상승).
#define NAND3_LEAK_STACK_FACTOR 0.2
// [한국어] NAND3 게이트 누설 스택 인수 = 0.2 (NAND2와 동일 근사).
#define NOR2_LEAK_STACK_FACTOR 0.2
// [한국어] NOR2 게이트 누설 스택 인수 = 0.2. PMOS 직렬 스택 누설 감소.
#define INV_LEAK_STACK_FACTOR  0.5
// [한국어] 인버터 누설 스택 인수 = 0.5. NMOS/PMOS 각 1개이므로 스택 감소 효과가 적음.
#define MAX_NUMBER_ARRAY_PARTITIONS 1000000
// [한국어] 배열 분할 최대 경우의 수 = 10^6. 탐색 공간 상한으로 메모리 과다 사용 방지.

// abbreviations used in this project
// ----------------------------------
//
//  num  : number
//  rw   : read/write
//  rd   : read
//  wr   : write
//  se   : single-ended
//  sz   : size
//  F    : feature
//  w    : width
//  h    : height or horizontal
//  v    : vertical or velocity
// [한국어] 프로젝트 내 약어 사전 — 변수명 이해를 위한 참고 (F=feature/피처, sz=size 등).


/*
 * [한국어]
 * ram_cell_tech_type_num - SRAM/DRAM 셀 기술 종류 열거형
 *
 * CACTI가 지원하는 메모리 셀 공정 기술을 식별하는 열거형.
 * parameter.h의 TechnologyParameter 배열을 인덱싱하는 데 사용된다.
 * itrs_hp/lstp/lop는 ITRS(International Technology Roadmap for Semiconductors) 2006 기준.
 */
enum ram_cell_tech_type_num
{
  itrs_hp   = 0, // [한국어] ITRS HP(High Performance): 고성능, 낮은 Vth, 높은 누설. 서버/데스크톱 캐시.
  itrs_lstp = 1, // [한국어] ITRS LSTP(Low Standby Power): 저대기 전력, 높은 Vth, 낮은 누설. 모바일/임베디드.
  itrs_lop  = 2, // [한국어] ITRS LOP(Low Operating Power): HP와 LSTP 중간 특성.
  lp_dram   = 3, // [한국어] LP-DRAM(Low Power DRAM): 저전력 DRAM 셀 기술.
  comm_dram = 4  // [한국어] 상용 DRAM(Commercial DRAM): DDR3/4 등 일반 DRAM 셀 기술.
};

const double pppm[4]      = {1,1,1,1};
// [한국어] 전체 전력 선택 마스크 — [동적, 누설, 게이트 누설, 단락 회로] 모두 포함.
//         powerDef에 곱하면 total_power = dynamic + leakage + gate_leakage + short_circuit.
const double pppm_lkg[4]  = {0,1,1,0};
// [한국어] 누설 전력만 선택 — 서브-스레시홀드 + 게이트 누설만 합산.
const double pppm_dyn[4]  = {1,0,0,0};
// [한국어] 동적 전력만 선택 — 스위칭 에너지만 합산.
const double pppm_Isub[4] = {0,1,0,0};
// [한국어] 서브-스레시홀드 누설(Isub)만 선택.
const double pppm_Ig[4]   = {0,0,1,0};
// [한국어] 게이트 산화막 누설(Ig)만 선택.
const double pppm_sc[4]   = {0,0,0,1};
// [한국어] 단락 회로 전류(short circuit) 전력만 선택.



#endif
