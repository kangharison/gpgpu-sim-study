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
 * [한국어 설명] H-tree 와이어 모델 헤더 (htree2.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 모델에서 H-tree 배선 구조를 시뮬레이션하는 Htree2 클래스를 선언한다.
 * H-tree(H자형 프랙탈 트리 레이아웃)는 캐시 뱅크 내 mat 배열 전체에 주소/데이터 신호를
 * 균일한 지연 시간으로 분배하거나 수집하는 데 사용하는 배선 위상이다.
 * 이 파일은 입력 H-tree(루트→리프 브로드캐스트)와 출력 H-tree(리프→루트 수집) 두 방향을
 * 모두 모델링하며, 각 트리 노드에서 NAND 리피터 또는 tristate 버퍼를 삽입한다.
 * AccelWattch에서 GPU L1/L2 캐시의 배선 지연·전력을 추정할 때 이 클래스가 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 모델 계층: InputParameter → DynamicParameter → UCA(Bank) → Mat → Subarray
 *                                                     └─ Htree2 (뱅크 간/내부 배선)
 * Htree2는 UCA(Uniform Cache Architecture) 뱅크 수준에서 mat 배열 사이를 연결하는
 * 배선 지연·전력을 계산한다. Mat 클래스가 단일 mat의 지연/전력을 담당하는 반면,
 * Htree2는 복수의 mat을 가로지르는 버스 배선 전체의 RC 지연과 전력을 담당한다.
 * 실행 컨텍스트: AccelWattch 전력 추정 단계 (호스트 유저스페이스, 시뮬레이션 초기화 시)
 * 호출 체인: UCA::UCA() → Htree2::Htree2() → in_htree()/out_htree()
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈: Wire (배선 RC 파라미터), basic_circuit (tr_R_on, gate_C, horowitz 등 회로 함수),
 *            parameter.h (g_tp 전역 기술 파라미터, DeviceType), subarray.h, cacti_interface.h
 * 상위 모듈: UCA (UniformCacheArray) — Htree2 객체를 생성하여 배선 결과를 bank 지연/전력에 합산
 * 데이터 흐름: mat_width/mat_height × ndbl/ndwl(뱅크 분할 수) → 트리 길이 계산 →
 *              각 레벨 Wire 객체 생성 → delay/power 누적 → Component.delay/power 에 저장
 *
 * === 주요 함수/구조체 요약 ===
 * Htree2()      : 생성자. htree_type에 따라 in_htree() 또는 out_htree() 호출하여 결과 계산
 * in_htree()    : 주소/데이터 입력 H-tree 지연·전력 계산 (루트→리프 브로드캐스트)
 * out_htree()   : 데이터 출력 H-tree 지연·전력 계산 (리프→루트 수집, tristate 버퍼 사용)
 * input_nand()  : 입력 H-tree 각 노드에 삽입되는 NAND2 리피터의 지연·전력 모델
 * output_buffer(): 출력 H-tree 각 노드에 삽입되는 tristate 버퍼(not+nand+nor+드라이버)의 모델
 */

#ifndef __HTREE2_H__
#define __HTREE2_H__

#include "basic_circuit.h" // [한국어] tr_R_on, gate_C, drain_C_, horowitz 등 소자 수준 회로 계산 함수
#include "component.h"     // [한국어] Component 기본 클래스 (delay, power, area 필드 포함)
#include "parameter.h"     // [한국어] g_tp(기술 파라미터), DeviceType, InterconnectType 정의
#include "assert.h"        // [한국어] 조건 검증용 assert 매크로
#include "subarray.h"      // [한국어] Subarray 클래스 (mat 내부 서브어레이 구조)
#include "cacti_interface.h" // [한국어] uca_org_t, InputParameter 등 CACTI 공개 인터페이스
#include "wire.h"          // [한국어] Wire 클래스 — 배선 RC 지연·전력·리피터 간격 계산

// leakge power includes entire htree in a bank (when uca_tree == false)
// leakge power includes only part to one bank when uca_tree == true
/* [한국어] uca_tree == false : 누설 전력이 뱅크 내 전체 H-tree를 포함
 *          uca_tree == true  : 하나의 뱅크까지 이르는 경로만 포함 (뱅크 간 interbank 트리) */

class Htree2 : public Component
/* [한국어] H-tree 배선 모델 클래스.
 * Component를 상속하여 delay, power(readOp/searchOp), area를 최종 결과로 저장한다.
 * 생성자에서 htree_type에 따라 in_htree() 또는 out_htree()를 즉시 호출하므로
 * 객체 생성 완료 시 결과가 모두 계산된 상태가 된다. */
{
  public:
    /*
     * [한국어]
     * Htree2 - H-tree 배선 모델 생성자
     *
     * @wire_model    : 배선 종류 (예: Global_30, Local 등) — Wire 객체 초기화에 사용
     * @mat_w, mat_h  : 단일 mat의 가로/세로 크기 (미터 단위) — H-tree 링크 길이 계산의 기준
     * @a_bits        : 어드레스 버스 비트 수 — Add_htree 타입일 때 wire_bw로 설정
     * @d_inbits      : 데이터 입력 버스 비트 수 — Data_in_htree 타입일 때 wire_bw로 설정
     * @search_data_in: CAM 검색 데이터 입력 비트 수
     * @d_outbits     : 데이터 출력 버스 비트 수 — Data_out_htree 타입일 때 wire_bw로 설정
     * @search_data_out: CAM 검색 결과 출력 비트 수
     * @bl            : ndbl — 수직 방향 mat 분할 수 (Ndbl, 블록 라인 방향)
     * @wl            : ndwl — 수평 방향 mat 분할 수 (Ndwl, 워드 라인 방향)
     * @htree_type    : H-tree 종류 (Add_htree/Data_in_htree/Data_out_htree/Search_in_htree/Search_out_htree)
     * @uca_tree_     : true이면 뱅크 간 interbank 트리 모드 (leakage 계산 범위가 달라짐)
     * @search_tree_  : true이면 CAM 검색 트리 모드 (wire_bw 배증 정책이 달라짐)
     * @dt            : 사용할 디바이스 타입 파라미터 (기본: g_tp.peri_global)
     * @return        : (생성자) — 결과는 Component::delay, Component::power, Component::area에 저장
     *
     * htree_type에 따라 in_htree() 또는 out_htree()를 호출한 후 wire_bw를 곱하여
     * 버스 전체의 동적 전력(power.readOp.dynamic *= init_wire_bw)을 산출한다.
     * 실행 컨텍스트: 호스트 CPU, AccelWattch 초기화 중 (UCA 생성자에서 호출)
     *
     * 호출 체인:
     *   UCA::UCA() → [Htree2::Htree2()] → in_htree() 또는 out_htree()
     */
    Htree2(enum Wire_type wire_model,
        double mat_w, double mat_h, int add, int data_in, int search_data_in, int data_out, int search_data_out, int bl, int wl,
        enum Htree_type h_type, bool uca_tree_ = false, bool search_tree_ = false,
        TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~Htree2() {};

    /*
     * [한국어]
     * in_htree - 입력 H-tree(루트→리프 브로드캐스트) 지연·전력 계산
     *
     * 주소/데이터 입력 신호가 루트에서 모든 mat 리프로 브로드캐스트되는 경로를 모델링한다.
     * 각 H-tree 레벨마다 Wire 객체로 배선 RC 지연을 계산하고, NAND2 리피터(input_nand)를
     * 삽입하여 신호 재생 지연과 전력을 누적한다.
     * ndwl이 ndbl보다 크면 수평 링크를 먼저 처리하는 불균형 트리 전략을 사용한다.
     *
     * 호출 체인:
     *   Htree2::Htree2() → [in_htree()] → Wire(), input_nand()
     */
    void in_htree();

    /*
     * [한국어]
     * out_htree - 출력 H-tree(리프→루트 수집) 지연·전력 계산
     *
     * 각 mat 리프의 데이터가 tristate 버퍼를 통해 루트로 합쳐지는 경로를 모델링한다.
     * in_htree와 동일한 트리 구조지만 노드마다 output_buffer(tristate)를 삽입한다.
     * CAM 검색 결과 또는 데이터 출력 버스에 대한 fanin 수집 동작을 나타낸다.
     *
     * 호출 체인:
     *   Htree2::Htree2() → [out_htree()] → Wire(), output_buffer()
     */
    void out_htree();

    // repeaters only at h-tree nodes
    /* [한국어] limited_in_htree / limited_out_htree: H-tree 노드에만 리피터를 두는 제한적 모델
     * (현재 CACTI 버전에서는 미구현 / 미사용 상태로 남아 있음) */
    void limited_in_htree();
    void limited_out_htree();

    /*
     * [한국어]
     * input_nand - 입력 H-tree 노드의 NAND2 리피터 지연·전력 모델
     *
     * @s1    : 현재 리피터 크기(게이트 스케일) — 이전 Wire 리피터 사이즈에서 유도
     * @s2    : 다음 레벨 리피터 크기 — 출력 부하 계산에 사용
     * @l_eff : 유효 배선 길이(µm) — 리피터 간격과 비교하여 s1을 스케일링하는 데 사용
     *
     * NAND2 게이트를 리피터로 사용할 때의 지연(horowitz 모델)과 동적/누설/게이트 누설 전력을
     * 계산하여 Component::delay 및 power에 누적한다.
     * 입력 신호가 브로드캐스트되므로 wire_bw(현재 레벨의 버스 폭)를 곱하여 총 누설 전력 산출.
     *
     * 호출 체인:
     *   in_htree() → [input_nand()]
     */
    void input_nand(double s1, double s2, double l);

    /*
     * [한국어]
     * output_buffer - 출력 H-tree 노드의 tristate 버퍼 지연·전력 모델
     *
     * @s1    : 현재 tristate 버퍼 크기
     * @s2    : 다음 레벨 버퍼 크기
     * @l_eff : 유효 배선 길이(µm)
     *
     * tristate 버퍼는 not + nand + nor + 출력 트랜지스터 4단으로 구성된다.
     * 각 단의 동적 전력을 readOp.dynamic과 searchOp.dynamic에 각각 누적하며,
     * uca_tree 여부에 따라 누설 전력 계산 범위(wire_bw 배수)가 동일하게 처리된다.
     *
     * 호출 체인:
     *   out_htree() → [output_buffer()]
     */
    void output_buffer(double s1, double s2, double l);

    double in_rise_time;
    /* [한국어] 입력 신호의 상승 시간(rise time, 초 단위).
     * 설정자: set_in_rise_time()으로 외부에서 주입, 또는 생성자 초기화 시 0으로 설정.
     * 읽는 자: Wire 객체 생성 시 wire.in_rise_time으로 전달되어 RC 지연 계산에 반영.
     * 값 범위: 0 이상의 실수 (초). 0이면 이상적인 step 입력으로 간주.
     * 동기화: 단일 스레드에서만 접근하므로 별도 동기화 불필요. */

    double out_rise_time;
    /* [한국어] H-tree 출력 신호의 상승 시간(초 단위).
     * 설정자: in_htree()/out_htree() 계산 완료 후 마지막 링크의 Wire.out_rise_time으로 갱신.
     * 읽는 자: UCA가 이 값을 Mat 또는 다음 단의 delay 계산에 초기 rise time으로 전달.
     * 값 범위: 0 이상의 실수 (초).
     * 동기화: 단일 스레드 접근, 락 불필요. */

    void set_in_rise_time(double rt)
    /* [한국어] in_rise_time 설정자 함수. 외부(UCA)에서 트리 입력 신호의 rise time을 주입할 때 사용. */
    {
      in_rise_time = rt; // [한국어] 주어진 rt 값을 in_rise_time 멤버에 저장
    }

    double max_unpipelined_link_delay;
    /* [한국어] 파이프라이닝 없이 H-tree 전체를 한 번에 통과하는 최대 지연 시간(초).
     * 설정자: 현재는 생성자에서 0으로 초기화되며 실제 계산이 구현되지 않음(TODO 주석 참고).
     * 읽는 자: UCA에서 파이프라인 단계 설계 시 참고값으로 사용 가능.
     * 값 범위: 0 이상 (미구현 상태이므로 항상 0).
     * 동기화: 단일 스레드 접근. */

    powerDef power_bit;
    /* [한국어] 버스 폭 1비트당 동적 전력 (단위: J/access).
     * 설정자: 생성자에서 power_bit = power 를 통해 wire_bw 곱셈 전 값을 저장.
     * 읽는 자: UCA가 비트별 전력을 별도로 참조할 때 사용.
     * 값 범위: 0 이상의 실수 (전력 에너지 단위 J).
     * 동기화: 단일 스레드 접근. */


  private:
    double wire_bw;
    /* [한국어] 현재 H-tree 레벨의 버스 폭(비트 수).
     * 설정자: 생성자에서 init_wire_bw로 초기화 후, 트리를 내려갈수록 ×2 또는 /2로 변화.
     *         in_htree(): 수직 링크를 만날 때마다 wire_bw *= 2 (팬아웃 증가).
     * 읽는 자: input_nand() / output_buffer()에서 누설 전력에 wire_bw를 곱하는 데 사용.
     * 값 범위: init_wire_bw ~ init_wire_bw * (ndbl * ndwl / 4) (정수배).
     * 동기화: 단일 스레드. */

    double init_wire_bw;  // bus width at root
    /* [한국어] H-tree 루트(최상위) 레벨의 버스 폭 (비트 수). 계산 시작 시의 wire_bw 초깃값.
     * 설정자: 생성자에서 htree_type에 따른 비트 수(add_bits, data_in_bits 등)로 한 번만 설정.
     * 읽는 자: output_buffer()에서 searchOp.dynamic 계산 시 전체 버스 폭 기준으로 사용.
     *          생성자 마지막에 power.readOp.dynamic *= init_wire_bw 로 총 동적 전력 산출.
     * 값 범위: 1 이상 정수 (add_bits, data_in_bits, data_out_bits, search bits 중 하나).
     * 동기화: 읽기 전용 (생성자 이후 변경 없음). */

    enum Htree_type tree_type;
    /* [한국어] H-tree 종류 — Add_htree / Data_in_htree / Data_out_htree / Search_in_htree / Search_out_htree.
     * 설정자: 생성자 초기화 리스트에서 htree_type 인수로 설정.
     * 읽는 자: 생성자의 switch 문에서 wire_bw 초깃값과 호출 경로(in_htree/out_htree) 결정.
     * 값 범위: Htree_type enum 값 5종.
     * 동기화: 읽기 전용. */

    double htree_hnodes;
    /* [한국어] H-tree의 수평 방향 노드 수 (현재 미사용 — 예약 필드).
     * 설정자: 미설정 (계산 미구현).
     * 읽는 자: 미사용.
     * 값 범위: 정의 없음. */

    double htree_vnodes;
    /* [한국어] H-tree의 수직 방향 노드 수 (현재 미사용 — 예약 필드).
     * 설정자: 미설정.
     * 읽는 자: 미사용. */

    double mat_width;
    /* [한국어] 단일 mat의 가로(수평) 크기 (미터 단위).
     * 설정자: 생성자 인수 mat_w에서 초기화.
     * 읽는 자: in_htree()/out_htree()에서 len_temp 계산 시 mat_width * ndwl / 2 공식에 사용.
     * 값 범위: 공정 기술 파라미터와 서브어레이 크기에 의존 (양수 실수).
     * 동기화: 읽기 전용. */

    double mat_height;
    /* [한국어] 단일 mat의 세로(수직) 크기 (미터 단위).
     * 설정자: 생성자 인수 mat_h에서 초기화.
     * 읽는 자: in_htree()/out_htree()에서 ht_temp 계산 시 mat_height * ndbl / 2 공식에 사용.
     * 값 범위: 양수 실수.
     * 동기화: 읽기 전용. */

    int add_bits, data_in_bits, search_data_in_bits, data_out_bits, search_data_out_bits;
    /* [한국어] 각 신호 버스의 비트 폭.
     * add_bits           : 어드레스 버스 비트 수
     * data_in_bits       : 데이터 입력 버스 비트 수
     * search_data_in_bits: CAM 검색 입력 비트 수
     * data_out_bits      : 데이터 출력 버스 비트 수
     * search_data_out_bits: CAM 검색 결과 출력 비트 수
     * 설정자: 생성자 초기화 리스트.
     * 읽는 자: in_htree()/out_htree()에서 배선 간 pitch 보정 계산(버스 폭 × pitch × 트리 깊이).
     * 값 범위: 0 이상 정수. search 관련 값은 CAM 구성 시에만 0 이상.
     * 동기화: 읽기 전용. */

    int ndbl, ndwl;
    /* [한국어] 뱅크 분할 수.
     * ndbl (Number of Data BLock Lines): 수직 방향 mat 분할 수 — H-tree 수직 레벨 = log2(ndbl/2)
     * ndwl (Number of Data Word Lines) : 수평 방향 mat 분할 수 — H-tree 수평 레벨 = log2(ndwl/2)
     * 설정자: 생성자 인수 bl, wl에서 초기화.
     * 읽는 자: in_htree()/out_htree()에서 h = log2(ndwl/2), v = log2(ndbl/2) 루프 변수 초기화.
     * 값 범위: 2 이상 짝수 (assert(ndbl >= 2 && ndwl >= 2) 강제).
     * 동기화: 읽기 전용. */

    bool uca_tree; // should have full bandwidth to access all banks in the array simultaneously
    /* [한국어] UCA 뱅크 간 interbank 트리 모드 플래그.
     * true이면 여러 뱅크를 동시에 접근하기 위한 최대 대역폭 경로를 모델링한다.
     * 이 경우 H-tree 길이 계산에 bank 전체 높이(mat_height * ndbl/2)를 사용하며
     * wire_bw 배증 로직과 누설 전력 계산 방식이 달라진다.
     * 설정자: 생성자 인수 uca_tree_.
     * 읽는 자: in_htree()/out_htree()에서 ht_temp/len_temp 분기 및 wire_bw 정책 결정.
     * 값 범위: true / false.
     * 동기화: 읽기 전용. */

    bool search_tree;
    /* [한국어] CAM 검색 트리 모드 플래그.
     * true이면 Search_in/out_htree처럼 모든 레벨(수평+수직)에서 wire_bw를 2배로 늘린다.
     * 설정자: 생성자 인수 search_tree_.
     * 읽는 자: in_htree()/out_htree() 내부 wire_bw 배증 조건.
     * 값 범위: true / false.
     * 동기화: 읽기 전용. */

    enum Wire_type wt;
    /* [한국어] 배선 종류 (Wire_type enum — 예: Global_30, Semi_global, Local 등).
     * Wire 객체 생성 시 해당 층의 저항·커패시턴스 파라미터를 선택하는 데 사용.
     * 설정자: 생성자 인수 wire_model에서 초기화.
     * 읽는 자: in_htree()/out_htree()에서 new Wire(wt, len) 호출마다 전달.
     * 동기화: 읽기 전용. */

    double min_w_nmos;
    /* [한국어] NMOS 최소 트랜지스터 폭 (g_tp.min_w_nmos_, 미터 단위).
     * 설정자: 생성자에서 g_tp.min_w_nmos_ 값으로 초기화.
     * 읽는 자: input_nand()/output_buffer()에서 nsize, size 계산 및 tr_R_on / drain_C_ / gate_C 호출 시 기준폭.
     * 동기화: 읽기 전용. */

    double min_w_pmos;
    /* [한국어] PMOS 최소 트랜지스터 폭 = n_to_p_eff_curr_drv_ratio * min_w_nmos.
     * 설정자: 생성자에서 deviceType->n_to_p_eff_curr_drv_ratio * min_w_nmos로 초기화.
     * 읽는 자: input_nand()/output_buffer()에서 PMOS drain 커패시턴스 계산에 사용.
     * 동기화: 읽기 전용. */

    TechnologyParameter::DeviceType *deviceType;
    /* [한국어] 이 H-tree에 사용되는 트랜지스터 디바이스 파라미터 포인터.
     * 설정자: 생성자 인수 dt로 설정. 기본값은 &g_tp.peri_global (주변부 글로벌 디바이스).
     * 읽는 자: input_nand()/output_buffer()에서 n_to_p_eff_curr_drv_ratio, Vth, Vdd 참조.
     * 값 범위: g_tp의 sram_cell / dram_acc / peri_global 등 중 하나 (NULL 불가).
     * 동기화: 읽기 전용 포인터, 원본 g_tp는 시뮬레이션 초기에 한 번 설정된 후 불변. */

};

#endif
