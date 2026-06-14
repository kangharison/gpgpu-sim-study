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
 * [한국어 설명] MCPAT_Router — NUCA용 NoC 라우터 전력/면적/지연 모델 헤더 (router.h)
 *
 * === 파일의 역할 ===
 * NUCA 캐시 뱅크들을 연결하는 NoC(Network-on-Chip) 라우터 하나를 모델링한다.
 * 라우터는 크로스바(Crossbar), 중재기(Arbiter), 버퍼(VC buffer) 세 서브컴포넌트로 구성되며,
 * 각 서브컴포넌트의 동적 전력·누설 전력·면적을 계산해 Component 멤버로 저장한다.
 * CACTI 6.0 기술 보고서의 라우터 모델을 구현하며 flit 크기·VC 수·입출력 포트 수를 파라미터로 받는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Nuca::sim_nuca() → new MCPAT_Router(flit_size, vc_buf, vc_count) → [이 모듈]
 * 생성자 호출 시 calc_router_parameters()가 지연·전력·면적을 즉시 계산한다.
 * 결과(power, delay, area)는 nuca_org_t에 포인터로 저장되어 최적 NUCA 탐색에 활용된다.
 * 호스트 유저스페이스에서 실행(단일 스레드, GPU 시뮬 루프 밖 초기화 단계).
 *
 * === 타 모듈과의 연결 ===
 * 의존: Wire(Cw3 — 와이어 커패시턴스), Crossbar(cb_stats), Arbiter(arb_power),
 *       basic_circuit(gate_C, drain_C_, cmos_Isub_leakage 등 트랜지스터 수식),
 *       parameter.h(g_tp 기술 파라미터), cacti_interface.h(g_ip 입력 파라미터)
 * 소비: Nuca::sim_nuca()가 3개의 라우터 인스턴스(64/128/256비트 flit)를 생성해 사용
 * 공유: g_tp.peri_global(주변 회로 CMOS 파라미터), g_ip->F_sz_um(공정 피처 사이즈)
 *
 * === 주요 함수/구조체 요약 ===
 * MCPAT_Router()          : 생성자 — 파라미터 초기화 후 calc_router_parameters() 호출
 * calc_router_parameters(): 지연·전력·면적을 한 번에 계산하는 orchestrator
 * get_router_power()      : 버퍼/크로스바/중재기 전력을 각각 계산해 합산
 * get_router_delay()      : 라우터 클럭 주파수와 파이프라인 스테이지 수 결정
 * get_router_area()       : 버퍼 면적 + 크로스바 면적 합산
 * cb_stats()              : Crossbar 클래스를 이용해 크로스바 전력/면적 계산
 * buffer_stats()          : Mat 클래스를 이용해 VC 버퍼(SRAM) 전력/면적 계산
 */

#ifndef __ROUTER_H__
#define __ROUTER_H__

#include <assert.h>
#include <iostream>
#include "basic_circuit.h"
#include "cacti_interface.h"
#include "component.h"
#include "mat.h"
#include "parameter.h"
#include "wire.h"
#include "crossbar.h"
#include "arbiter.h"



class MCPAT_Router : public Component
{
  public:
    MCPAT_Router(
        double flit_size_,
        double vc_buf, /* vc size = vc_buffer_size * flit_size */
        double vc_count,
        TechnologyParameter::DeviceType *dt = &(g_tp.peri_global),
        double I_ = 5,
        double O_ = 5,
        double M_ = 0.6);
    ~MCPAT_Router();


    void print_router();

    Component arbiter, crossbar, buffer;
    /* [한국어] 라우터 서브컴포넌트 3개의 전력·면적 결과.
     * arbiter  : VC 중재기 + 크로스바 중재기 전력 합산 결과.
     * crossbar : 크로스바 스위치 전력·면적 결과 (Crossbar 클래스 기반).
     * buffer   : VC 버퍼(SRAM 기반) 전력·면적 결과 (Mat 클래스 기반).
     * 설정자: get_router_power()가 각 서브컴포넌트를 계산해 이 멤버에 저장.
     * 읽는 자: get_router_area()가 buffer/crossbar.area를 읽어 라우터 총 면적 계산;
     *          get_router_power()가 3개를 합산해 Component::power에 저장.
     * 값 범위: dynamic [J/flit], leakage [W].
     * 동기화: 생성자에서 단 한 번 계산되며 이후 읽기 전용. */

    double cycle_time, max_cyc;
    /* [한국어] 라우터 클럭 주기 [ps] 와 허용 최대 사이클 타임 [ps].
     * cycle_time: FREQUENCY(GHz)의 역수 × 1000 — 라우터 동작 주기.
     * max_cyc   : 17 × g_tp.FO4 [ps] — 공정 FO4 기반 최대 허용 주기.
     * 설정자: get_router_delay()에서 설정.
     * 읽는 자: Nuca::sim_nuca()에서 nuca_pda.cycle_time 초기화에 사용.
     * 동기화: 불변. */

    double flit_size;
    /* [한국어] 라우터 링크의 flit 크기 [비트] — 생성자 인자로 받는 핵심 파라미터.
     * 설정자: 생성자 초기화 리스트에서 flit_size_ 인자로 설정.
     * 읽는 자: buffer_stats()에서 VC 버퍼 열 수 계산; cb_stats()에서 Crossbar 생성; print_router().
     * 값 범위: 64, 128, 256 (비트) — Nuca::sim_nuca()에서 세 가지 값으로 생성.
     * 동기화: 불변. */

    double vc_count;
    /* [한국어] 가상 채널(Virtual Channel) 수 — 버퍼 용량과 중재기 크기 결정.
     * 설정자: 생성자에서 vc_c 인자로 초기화.
     * 읽는 자: buffer_stats()에서 num_wr_ports 설정; arb_power()에서 VC 중재기 크기.
     * 값 범위: 4 (기본값, Nuca::sim_nuca()에서 생성).
     * 동기화: 불변. */

    double vc_buffer_size; /* vc size = vc_buffer_size * flit_size */
    /* [한국어] VC당 버퍼 크기 [flit 단위] — VC 버퍼의 행 수에 해당.
     * 설정자: 생성자에서 vc_buf 인자로 초기화.
     * 읽는 자: buffer_stats()에서 dyn_p.num_r_subarray = (int)vc_buffer_size로 사용.
     * 값 범위: 8 (기본값) — 즉 8개 flit을 한 VC에 버퍼링.
     * 동기화: 불변. */

  private:
	TechnologyParameter::DeviceType *deviceType;
    /* [한국어] CMOS 디바이스 파라미터 포인터 (Vdd, Vth, n_to_p 비율 등).
     * 설정자: 생성자 초기화 리스트.
     * 읽는 자: min_w_pmos 계산, Vdd 초기화, gate_C/drain_C 호출 시.
     * 값 범위: 보통 &g_tp.peri_global.
     * 동기화: 불변. */

	double FREQUENCY; // move this to config file --TODO
    /* [한국어] 라우터 동작 주파수 [GHz] — 현재 하드코딩(5 GHz), config 파일 이동 예정.
     * 설정자: get_router_delay()에서 5로 초기화; 공정 한계 초과 시 낮춰 재설정.
     * 읽는 자: cycle_time 계산, print_router() 출력.
     * 값 범위: 0 < FREQUENCY ≤ 5 GHz.
     * 동기화: 불변 (생성 시 확정). */

    double Cw3(double len);
    /* [한국어] 3배 간격 글로벌 와이어 커패시턴스 계산 [F] — 크로스바 배선 모델용 */
    double gate_cap(double w);
    /* [한국어] 게이트 커패시턴스 [F] — 트랜지스터 폭 w[m]에 대해 gate_C 호출 래퍼 */
    double diff_cap(double w, int type /*0 for n-mos and 1 for p-mos*/, double stack);
    /* [한국어] 드레인 커패시턴스 [F] — drain_C_ 호출 래퍼 (N/P형, 스택 수 지정) */
    enum Wire_type wtype;
    /* [한국어] 크로스바 배선에 사용할 와이어 타입 — 현재 코드에서는 g_ip->wt 참조 */
    enum Wire_placement wire_placement;
    /* [한국어] 와이어 배치 위치 (outside_mat/inside_mat/local) */
    //corssbar
    double NTtr, PTtr, wt, ht, I, O, NTi, PTi, NTid, PTid, NTod, PTod, TriS1, TriS2;
    /* [한국어] 크로스바 트랜지스터 치수 파라미터 [m]:
     * NTtr/PTtr: 트랜스미션 게이트 NMOS/PMOS 폭
     * wt/ht    : 크로스바 트랙 폭/높이
     * I/O      : 크로스바 입력/출력 포트 수 (기본 5/5)
     * NTi/PTi  : 입력 인버터 NMOS/PMOS 폭
     * NTid/PTid: 입력 드라이버 NMOS/PMOS 폭
     * NTod/PTod: 출력 드라이버 NMOS/PMOS 폭 */

    double M; //network load
    /* [한국어] 네트워크 부하율(activity factor) — 전력 계산 시 실효 트래픽 비율.
     * 설정자: 생성자에서 M_ 인자(기본 0.6)로 초기화.
     * 읽는 자: get_router_power()에서 동적 전력 × MIN(I,O) × M으로 실효 전력 산출.
     * 값 범위: 0 < M ≤ 1.0.
     * 동기화: 불변. */

    double transmission_buf_inpcap();
    /* [한국어] 트랜스미션 게이트 입력측 커패시턴스 [F] */
    double transmission_buf_outcap();
    /* [한국어] 트랜스미션 게이트 출력측 커패시턴스 [F] */
    double transmission_buf_ctrcap();
    /* [한국어] 트랜스미션 게이트 제어선 커패시턴스 [F] */
    double crossbar_inpline();
    /* [한국어] 크로스바 입력 배선의 총 커패시턴스 [F] */
    double crossbar_outline();
    /* [한국어] 크로스바 출력 배선의 총 커패시턴스 [F] */
    double crossbar_ctrline();
    /* [한국어] 크로스바 제어 배선의 총 커패시턴스 [F] */
    double tr_crossbar_power();
    /* [한국어] 트랜스미션 게이트 기반 크로스바 동적 전력 [J/flit] 계산 (대안 모델) */
    void  cb_stats ();
    /* [한국어] Crossbar 클래스로 크로스바 전력·면적 계산 후 crossbar 멤버에 저장 */
    double arb_power();
    /* [한국어] (미사용) 중재기 전력 계산 — get_router_power()가 직접 MCPAT_Arbiter를 사용 */
    void  arb_stats ();
    /* [한국어] (미사용) 중재기 통계 출력 */
    double buffer_params();
    /* [한국어] (미사용) 버퍼 파라미터 반환 */
    void buffer_stats();
    /* [한국어] Mat 클래스로 VC 버퍼의 전력·면적을 계산하고 buffer 멤버에 저장 */


    //arbiter

    //buffer

    //router params
    double Vdd;
    /* [한국어] 공급 전압 [V] — deviceType->Vdd에서 복사.
     * 설정자: 생성자에서 dt->Vdd로 초기화.
     * 읽는 자: 크로스바 전력 계산의 C×Vdd² 항에서 사용.
     * 값 범위: 공정 노드에 따라 다름 (예: 65nm → 1.1V).
     * 동기화: 불변. */

    void calc_router_parameters();
    /* [한국어] 지연·전력·면적 계산을 순서대로 호출하는 orchestrator */
    void get_router_area();
    /* [한국어] buffer + crossbar 면적 합산해 area 멤버 설정 */
    void get_router_power();
    /* [한국어] 버퍼·크로스바·중재기 전력을 계산하고 Component::power에 합산 */
    void get_router_delay();
    /* [한국어] FREQUENCY, cycle_time, delay(파이프라인 스테이지 수) 결정 */

    double min_w_pmos;
    /* [한국어] 최소 PMOS 트랜지스터 폭 [m] = n_to_p_eff 비율 × min_w_nmos.
     * 설정자: 생성자에서 deviceType->n_to_p_eff_curr_drv_ratio × g_tp.min_w_nmos_ 로 계산.
     * 읽는 자: 누설 전력 계산(cmos_Isub_leakage), 게이트/드레인 커패시턴스 계산에 사용.
     * 동기화: 불변. */


};

#endif
