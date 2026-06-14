/*------------------------------------------------------------
 *                              CACTI 6.5
 *         Copyright 2008 Hewlett-Packard Development Corporation
 *                         All Rights Reserved
 *
 * Permission to use, copy, and modify this software and its documentation is
 * hereby granted only under the following terms and conditions.  Both the
 * above copyright notice and this permission notice must appear in all copies
 * of the software, derivative works or modified versions, and any portions
 * thereof, and both notices must appear in supporting documentation.
 *
 * Users of this software agree to the terms and conditions set forth herein, and
 * hereby grant back to Hewlett-Packard Company and its affiliated companies ("HP")
 * a non-exclusive, unrestricted, royalty-free right and license under any changes, 
 * enhancements or extensions  made to the core functions of the software, including 
 * but not limited to those affording compatibility with other hardware or software
 * environments, but excluding applications which incorporate this software.
 * Users further agree to use their best efforts to return to HP any such changes,
 * enhancements or extensions that they make and inform HP of noteworthy uses of
 * this software.  Correspondence should be provided to HP at:
 *
 *                       Director of Intellectual Property Licensing
 *                       Office of Strategy and Technology
 *                       Hewlett-Packard Company
 *                       1501 Page Mill Road
 *                       Palo Alto, California  94304
 *
 * This software may be distributed (but not offered for sale or transferred
 * for compensation) to third parties, provided such third parties agree to
 * abide by the terms and conditions of this notice.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND HP DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL HP 
 * CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *------------------------------------------------------------*/

/*
 * [한국어 설명] 고기수(High-Radix) NoC 라우터 전력·면적 모델 헤더 (highradix.h)
 *
 * === 파일의 역할 ===
 * MCPAT_Router와 달리 고기수(Valiant/Butterfly 토폴로지) 라우터를 모델링한다.
 * 라우터를 SUB_SWITCH_SZ × ROWS × COLUMNS 서브스위치 구조로 분해하고,
 * 크로스바(cb, out_cb)·중재기(vc_arb, c_arb, cb_arb)·버퍼(inp_buff, r_buff, c_buff)·
 * 버스(hor_bus, ver_bus) 각각의 전력을 계산해 합산한다.
 * CACTI 6.5 기반이며 NUCA 대규모 네트워크 시뮬레이션에서 대안 라우터 모델로 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * HighRadix 생성자 → compute_power() → [이 모듈의 서브컴포넌트 계산]
 * MCPAT_Router보다 세밀한 서브스위치 분해 모델이며, 독립적으로도 사용 가능하다.
 * Waveguide 클래스도 이 파일에 선언되어 있으나 현재 구현은 stub 수준이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Crossbar(크로스바 전력), MCPAT_Arbiter(중재기), Wire(버스 배선), Mat(버퍼 SRAM),
 *       ROUTER.def(기본값 상수 DEF_RADIX 등), basic_circuit, parameter.h
 * 소비: NUCA 또는 독립 실행(highradix standalone) 시 생성
 * 공유: g_ip->F_sz_nm(공정 노드), g_tp.peri_global(주변 회로 CMOS 파라미터)
 *
 * === 주요 함수/구조체 요약 ===
 * HighRadix()          : 생성자 — 공정별 면적 스케일링, 서브스위치 수 계산, area 초기화
 * compute_power()      : 크로스바·버퍼·버스·중재기 모두를 인스턴스화하고 전력 합산
 * sub_switch_power()   : 서브스위치 1개의 전력 계산 (r_buff + c_buff + cb + out_cb + arb)
 * buffer_(block_sz, sz): Mat(SRAM) 기반 버퍼 생성 헬퍼
 * print_router()       : 상세 전력·면적 통계 출력
 */

#ifndef __HIGHRADIX__
#define __HIGHRADIX__

#include <iostream>
#include "basic_circuit.h"
#include "component.h"
#include "parameter.h"
#include "assert.h"
#include "cacti_interface.h"
#include "wire.h"
#include "mat.h"
#include "crossbar.h"
#include "arbiter.h"
#include "ROUTER.def"

#define FLIP_FLOP_L 0 //W leakage
#define FLIP_FLOP_D 0 //J dynamic
#define ROUTE_LOGIC_D 0 //J
#define ROUTE_LOGIC_L 0 //W

class HighRadix : public Component
{
  public:
    HighRadix(
    double SUB_SWITCH_SZ_ = DEF_SUB_SWITCH_SZ,
    double ROWS_ = DEF_ROWS,
    double FREQUENCY_ = DEF_FREQUENCY, // GHz
    double RADIX_ = DEF_RADIX,
    double VC_COUNT_ = DEF_VC_COUNT,
    double FLIT_SZ_ = DEF_FLIT_SZ,
    double AF_ = DEF_AF,// activity factor
    double DIE_LEN_ = DEF_DIE_LEN,//u
    double DIE_HT_ = DEF_DIE_HT,//u
    double INP_BUFF_ENT_ = DEF_INP_BUFF_ENT, 
    double ROW_BUFF_ENT_ = DEF_ROW_BUFF_ENT, 
    double COL_BUFF_ENT_ = DEF_COL_BUFF_ENT,
    TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~HighRadix();


// Params
    double SUB_SWITCH_SZ;
    /* [한국어] 서브스위치 크기 — 서브스위치 내 크로스바의 입출력 포트 수.
     * 설정자: 생성자 SUB_SWITCH_SZ_ 인자.
     * 읽는 자: compute_power()에서 Crossbar(SUB_SWITCH_SZ, ...) 크기 지정, 면적 계산.
     * 동기화: 불변. */

    double ROWS;
    /* [한국어] 서브스위치 행 수 — ROWS × COLUMNS = 총 서브스위치 수.
     * 설정자: 생성자 ROWS_ 인자.
     * 읽는 자: COLUMNS = (RADIX/SUB_SWITCH_SZ)² / ROWS 계산, 버스 면적/전력 계산.
     * 동기화: 불변. */

    double FREQUENCY;// GHz
    /* [한국어] 라우터 동작 주파수 [GHz].
     * 설정자: 생성자 FREQUENCY_ 인자.
     * 읽는 자: print_router()에서 W 단위 전력 = J × FREQUENCY × 1e9.
     * 동기화: 불변. */

    double RADIX;
    /* [한국어] 라우터의 총 기수(입출력 포트 수) — 고기수 라우터의 핵심 파라미터.
     * 설정자: 생성자 RADIX_ 인자.
     * 읽는 자: COLUMNS 계산, 입력 버퍼 × RADIX 전력 계산.
     * 동기화: 불변. */

    double VC_COUNT;
    /* [한국어] 가상 채널(Virtual Channel) 수.
     * 설정자: 생성자 VC_COUNT_ 인자.
     * 읽는 자: vc_arb 크기, r_buff/c_buff 수 × VC_COUNT로 전력/면적 계산.
     * 동기화: 불변. */

    double FLIT_SZ;
    /* [한국어] flit 크기 [비트] — 라우터 링크 폭.
     * 설정자: 생성자 FLIT_SZ_ 인자.
     * 읽는 자: INP_BUFF_SZ = FLIT_SZ × INP_BUFF_ENT; 버스 전력 × FLIT_SZ.
     * 동기화: 불변. */

    double AF;// activity factor
    /* [한국어] 활성화 인수(Activity Factor) — 실제 트래픽 부하율 (0~1).
     * 설정자: 생성자 AF_ 인자.
     * 읽는 자: print_router()에서 동적 전력 × AF로 실효 전력 계산.
     * 동기화: 불변. */

    double DIE_LEN;//u
    /* [한국어] 칩 한 변의 길이 [um] — 생성자에서 DIE_LEN × DIE_HT / area_scale의 제곱근으로 재계산.
     * 설정자: 생성자에서 sqrt(DIE_LEN_×DIE_HT_/area_scale)으로 정사각형 근사.
     * 읽는 자: hor_bus = new Wire(wt, DIE_LEN)으로 수평 버스 길이 결정.
     * 동기화: 불변. */

    double DIE_HT;//u
    /* [한국어] 칩 높이 [um] — 생성자에서 DIE_LEN과 동일하게 설정 (정사각형 근사).
     * 읽는 자: ver_bus 길이 계산 (ROWS×(ROWS+1)/2 × DIE_HT/ROWS).
     * 동기화: 불변. */

    double INP_BUFF_ENT;
    /* [한국어] 입력 버퍼 엔트리 수 [flit 단위].
     * 설정자: 생성자 INP_BUFF_ENT_ 인자.
     * 읽는 자: INP_BUFF_SZ = FLIT_SZ × INP_BUFF_ENT 계산.
     * 동기화: 불변. */

    double ROW_BUFF_ENT;
    /* [한국어] 행 버퍼 엔트리 수 [flit 단위] — 서브스위치 내 행 간 중간 버퍼. */
    double COL_BUFF_ENT;
    /* [한국어] 열 버퍼 엔트리 수 [flit 단위] — 서브스위치 내 열 방향 중간 버퍼. */

    void print_router();
    /* [한국어] 라우터 전력·면적 상세 통계를 콘솔에 출력 */

    double INP_BUFF_SZ;
    /* [한국어] 입력 버퍼 총 크기 [비트] = FLIT_SZ × INP_BUFF_ENT.
     * 설정자: 생성자에서 계산. 읽는 자: buffer_(FLIT_SZ, INP_BUFF_SZ) 호출. */
    double COLUMNS;
    /* [한국어] 서브스위치 열 수 = (RADIX/SUB_SWITCH_SZ)² / ROWS.
     * 설정자: 생성자에서 pow(RADIX/SUB_SWITCH_SZ, 2) / ROWS로 계산.
     * 읽는 자: num_sub = ROWS × COLUMNS; 버스 전력·면적 계산. */
    double ROW_BUFF_SZ;
    /* [한국어] 행 버퍼 크기 [비트] = ROW_BUFF_ENT × FLIT_SZ. */
    double COL_BUFF_SZ;
    /* [한국어] 열 버퍼 크기 [비트] = COL_BUFF_ENT × FLIT_SZ. */

    void compute_power();
    /* [한국어] 전체 전력 계산 — 크로스바/버퍼/버스/중재기를 인스턴스화하고 합산 */
    void compute_arb_power();
    /* [한국어] 중재기 전력을 arb_tot에 집계 */
    void compute_crossbar_power();
    /* [한국어] 크로스바 전력을 crossbar_tot에 집계 */
    void compute_buff_power();
    /* [한국어] 버퍼 전력을 buff_tot에 집계 */
    void compute_bus_power();
    /* [한국어] 버스(와이어) 전력을 wire_tot에 집계 */
    void print_buffer(Component *r);
    /* [한국어] 버퍼 컴포넌트의 전력·면적을 콘솔에 출력 */
    void sub_switch_power();
    /* [한국어] 서브스위치 1개의 전력 합산 — r_buff + c_buff + cb + out_cb + arb */
    Mat * buffer_(double block_sz, double sz);
    /* [한국어] SRAM Mat 기반 버퍼 생성 헬퍼 — block_sz(비트 너비), sz(총 크기) */

    Crossbar *cb, *out_cb;
    /* [한국어] cb: 서브스위치 내부 정방형 크로스바 (SUB_SWITCH_SZ × SUB_SWITCH_SZ).
     *          out_cb: 서브스위치 출력 크로스바 (1 × SUB_SWITCH_SZ — 한 입력에서 여러 출력).
     * 설정자: compute_power()에서 new Crossbar(...)로 생성. 읽는 자: print_router() 출력.
     * 동기화: compute_power() 내 순차 생성. */

    MCPAT_Arbiter *cb_arb, *vc_arb, *c_arb;
    /* [한국어] 세 종류의 중재기:
     * vc_arb : VC(가상 채널) 중재기 — VC_COUNT 입력.
     * c_arb  : 열(column) 중재기 — COLUMNS 입력.
     * cb_arb : 크로스바 중재기 — RADIX/ROWS 입력.
     * 설정자: compute_power()에서 new MCPAT_Arbiter(...)로 생성.
     * 동기화: compute_power() 내 순차 생성. */

    Mat *inp_buff, *r_buff, *c_buff;
    /* [한국어] 세 종류의 SRAM 버퍼:
     * inp_buff: 입력 버퍼 (포트당 하나, RADIX 개).
     * r_buff  : 행 버퍼 (서브스위치 내 행 방향 중간 저장).
     * c_buff  : 열 버퍼 (서브스위치 내 열 방향 중간 저장).
     * 설정자: compute_power()에서 buffer_() 헬퍼로 생성.
     * 동기화: 소멸자에서 delete. */

    Component sub_sw;
    /* [한국어] 서브스위치 1개의 전력·면적 누산 컴포넌트.
     * 설정자: sub_switch_power()에서 r_buff + c_buff + cb + out_cb + arb 합산.
     * 읽는 자: compute_power()에서 × num_sub으로 총 전력·면적 계산. */

    Component wire_tot, buff_tot, crossbar_tot, arb_tot;
    /* [한국어] 전체 배선/버퍼/크로스바/중재기 전력·면적 집계 컴포넌트.
     * 설정자: compute_bus_power/buff_power/crossbar_power/arb_power()에서 각각 설정.
     * 읽는 자: print_router()에서 각 컴포넌트 비율(%) 출력. */

    Wire *hor_bus, *ver_bus;
    /* [한국어] 수평·수직 버스 배선 모델 포인터.
     * hor_bus: 수평 버스 (DIE_LEN 길이), ver_bus: 수직 버스 (유효 높이 = 삼각형 합 × 셀 높이).
     * 설정자: compute_power()에서 new Wire(wt, len)으로 생성.
     * 동기화: 소멸자 없음 (HighRadix가 소유하지 않으므로 주의). */

  private:
    double min_w_pmos;
    /* [한국어] 최소 PMOS 폭 [m] = n_to_p 비율 × min_w_nmos. */
    TechnologyParameter::DeviceType *deviceType;
    /* [한국어] CMOS 디바이스 파라미터 포인터 (Vdd 등). */
    double num_sub;
    /* [한국어] 총 서브스위치 수 = ROWS × COLUMNS.
     * 설정자: compute_power() 첫 줄에서 계산.
     * 읽는 자: 총 전력/면적 계산 시 서브스위치 1개 결과를 num_sub로 스케일링. */

};

class Waveguide : public Component
{
  public:
    Waveguide(TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~Waveguide();
};

#endif
