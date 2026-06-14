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
 * [한국어 설명] CACTI 아비터(중재기) 헤더 (arbiter.h)
 *
 * === 파일의 역할 ===
 * MCPAT_Arbiter 클래스를 선언하는 헤더 파일이다. CACTI 내 NUCA(Non-Uniform Cache
 * Architecture) 라우터 구성 요소 중 R-입력 라운드-로빈(Round-Robin) 중재기의 전력 및
 * 면적 모델을 제공한다. 아비터는 여러 입력 포트가 동일한 출력 포트(크로스바)에 동시에
 * 접근하려 할 때 하나를 선택하는 하드웨어 회로이며, 이 파일은 NOR 게이트 체인으로
 * 구현된 중재기의 커패시턴스/누설 전력을 수치 모델링한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 → CACTI 캐시/네트워크 파워 모델 → 아비터 컴포넌트.
 * CACTI는 캐시 배열(Bank→Mat→Subarray)의 면적·전력·타이밍을 계산하고, 그 위에
 * NUCA 라우터(아비터 + 크로스바)의 전력까지 포함한다. MCPAT_Arbiter는 Component
 * 기반 클래스를 상속하며, compute_power()를 통해 readOp.dynamic/leakage/
 * gate_leakage 값을 채운다.
 * 호출 흐름: AccelWattch GPU 전력 계산 → (필요 시) NUCA 라우터 모델 → MCPAT_Arbiter
 *
 * === 타 모듈과의 연결 ===
 * - Component (component.h): 상속 기반 클래스. area, power, delay 필드를 제공한다.
 * - basic_circuit.h: gate_C(), drain_C_(), cmos_Isub_leakage(), cmos_Ig_leakage()
 *   등 트랜지스터 레벨 RC 계산 함수들을 제공한다.
 * - parameter.h: g_tp(글로벌 기술 파라미터), g_ip(글로벌 입력 파라미터)를 통해
 *   공정 노드별 Vdd, 최소 트랜지스터 폭, 셀 높이 등을 참조한다.
 * - Wire (wire.h): Cw3()에서 3배 간격(triple-spaced) 배선 커패시턴스를 계산한다.
 * - Mat (mat.h): 캐시 배열 매트(Mat) — 아비터는 라우터 레벨에서 매트 위에 올라간다.
 *
 * === 주요 함수/구조체 요약 ===
 * - MCPAT_Arbiter(): 생성자. 공정 파라미터로 트랜지스터 폭(NTn1/PTn1/NTn2/PTn2 등)을 초기화.
 * - compute_power(): 동적 전력(arb_req+arb_pri+arb_grant+arb_int) 및 누설·게이트 누설 계산.
 * - arb_req(): 요청(request) 신호가 스위칭할 때 구동하는 커패시턴스 반환.
 * - arb_grant(): 그랜트(grant) 신호 출력 커패시턴스 (크로스바 제어선 포함) 반환.
 * - Cw3(): 3배 간격 배선의 커패시턴스를 Wire 모델로 계산.
 */

#ifndef __ARBITER__
#define __ARBITER__

#include <assert.h>        // [한국어] assert() — 내부 불변식 검증용
#include <iostream>        // [한국어] cout — print_arbiter() 출력용
#include "basic_circuit.h" // [한국어] gate_C(), drain_C_(), cmos_Isub_leakage() 등 CMOS 기본 회로 계산 함수
#include "cacti_interface.h" // [한국어] g_ip(InputParameter 전역 포인터), g_tp(TechnologyParameter 전역 구조체)
#include "component.h"     // [한국어] Component 기반 클래스 — area, power, delay 필드 제공
#include "parameter.h"     // [한국어] TechnologyParameter::DeviceType — 공정별 디바이스 특성(Vdd, n_to_p 비율 등)
#include "mat.h"           // [한국어] Mat — 캐시 매트 단위 모델 (아비터가 속하는 상위 캐시 구조)
#include "wire.h"          // [한국어] Wire — 배선 커패시턴스/저항 모델 (Cw3 계산에 사용)

/*
 * [한국어]
 * MCPAT_Arbiter - R-입력 라운드-로빈 아비터(중재기) 전력·면적 모델
 *
 * NOR 게이트 체인으로 구성된 R-입력 아비터를 CMOS 레벨에서 모델링한다.
 * 아비터는 NUCA 라우터의 크로스바 앞단에서 여러 입력 포트 중 하나를
 * 선택(Grant)하는 역할을 담당한다. 이 클래스는 요청·우선순위·그랜트·
 * 내부 노드 신호에서 발생하는 동적 전력과 서브-스레시홀드/게이트 누설
 * 전력을 수치적으로 계산한다.
 *
 * 상속: Component (area, power, delay, cycle_time 필드를 부여받음)
 */
class MCPAT_Arbiter : public Component
{
  public:
    /*
     * [한국어]
     * MCPAT_Arbiter - 생성자
     *
     * @Req:        아비터에 연결되는 입력 요청 수 (R). NOR 게이트 체인의 팬인.
     * @flit_sz:    플릿(flit) 크기(비트). 아비터가 서비스하는 데이터 폭.
     * @output_len: 크로스바 출력 방향 배선 길이(µm). 제어선 커패시턴스 계산에 사용.
     * @dt:         공정 노드별 디바이스 파라미터 포인터. 기본값 g_tp.peri_global.
     *
     * 공정 피처 사이즈(F_sz_um)에 비례하여 NOR1, NOR2, NOT 게이트의 NMOS/PMOS
     * 트랜지스터 폭(NTn1/PTn1/NTn2/PTn2/NTi/PTi)과 전송 게이트 폭(NTtr/PTtr)을
     * 초기화한다. 이 값들은 이후 arb_req/arb_pri/arb_grant/arb_int에서 커패시턴스
     * 계산의 입력으로 사용된다.
     *
     * 호출 체인: NUCA 라우터 생성 → [MCPAT_Arbiter()] → (트랜지스터 폭 초기화)
     */
    MCPAT_Arbiter(
      double Req,
      double flit_sz,
      double output_len,
      TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));

    /*
     * [한국어]
     * ~MCPAT_Arbiter - 소멸자
     *
     * 동적 할당 자원이 없으므로 빈 소멸자. Component 소멸자가 자동 호출된다.
     */
    ~MCPAT_Arbiter();

    /*
     * [한국어]
     * print_arbiter - 아비터 통계 출력
     *
     * @return: void
     *
     * 입력 수(R), 플릿 크기, 동적 전력(nJ), 누설 전력(mW)을 콘솔에 출력한다.
     * 디버깅·결과 보고용이며, AccelWattch가 최종 전력을 상위로 전달한 후 호출된다.
     *
     * 호출 체인: (디버그/출력 루틴) → [print_arbiter()]
     */
    void print_arbiter();

    /*
     * [한국어]
     * arb_req - 요청(request) 신호 스위칭 커패시턴스 계산
     *
     * @return: 요청 신호가 NOR1·NOR2·NOT 게이트를 구동할 때의 총 커패시턴스(F)
     *
     * (R-1)개의 NOR1 게이트 게이트 커패시턴스 + NOR2 + NOT 인버터의 게이트·드레인
     * 커패시턴스를 합산한다. compute_power()가 이 값에 Vdd²/2를 곱해 동적 에너지를 산출한다.
     *
     * 호출 체인: compute_power() → [arb_req()]
     */
    double arb_req();

    /*
     * [한국어]
     * arb_pri - 우선순위(priority) 플립플롭 스위칭 커패시턴스 계산
     *
     * @return: 우선순위 상태 갱신 시 NOR1 게이트들이 부담하는 커패시턴스(F)
     *
     * 라운드-로빈 우선순위 테이블을 갱신할 때 스위칭되는 NOR1 게이트 2개의
     * 게이트 커패시턴스를 반환한다. 플립플롭 내부 스위칭은 근사적으로 무시(주석 참조).
     *
     * 호출 체인: compute_power() → [arb_pri()]
     */
    double arb_pri();

    /*
     * [한국어]
     * arb_grant - 그랜트(grant) 출력 노드 커패시턴스 계산
     *
     * @return: 그랜트 출력선 + 크로스바 제어선에 달린 총 커패시턴스(F)
     *
     * NOR1 드레인 커패시턴스(2개)와 crossbar_ctrline()의 커패시턴스를 합산한다.
     * 이 노드는 매 사이클 1회 스위칭한다고 가정하므로 Vdd²(×1)을 곱한다.
     *
     * 호출 체인: compute_power() → [arb_grant()] → crossbar_ctrline()
     */
    double arb_grant();

    /*
     * [한국어]
     * arb_int - 아비터 내부 노드 커패시턴스 계산
     *
     * @return: NOR1 출력 → NOR2 입력 사이 내부 배선의 총 커패시턴스(F)
     *
     * NOR1 드레인(×2) + NOR2 게이트 커패시턴스를 합산한다. 0.5 스위칭 인수(activity)
     * 를 가정하여 compute_power()에서 Vdd²×0.5를 곱한다.
     *
     * 호출 체인: compute_power() → [arb_int()]
     */
    double arb_int();

    /*
     * [한국어]
     * compute_power - 아비터 동적 전력 및 누설 전력 계산
     *
     * @return: void (power.readOp.dynamic/leakage/gate_leakage 에 결과 저장)
     *
     * arb_req/arb_pri/arb_grant/arb_int를 통해 각 노드의 커패시턴스를 얻은 뒤
     * 동적 에너지(½CV²)를 합산한다. 누설 전류는 cmos_Isub_leakage()로, 게이트
     * 누설은 cmos_Ig_leakage()로 계산한다. 결과는 Component::power.readOp.*에 저장된다.
     *
     * 호출 체인: NUCA 라우터 전력 계산 → [compute_power()] → arb_req/arb_pri/arb_grant/arb_int
     */
    void compute_power();

    /*
     * [한국어]
     * Cw3 - 3배 간격(triple-spaced) 배선 커패시턴스 계산
     *
     * @length: 배선 길이 (µm 단위 입력, 함수 내부에서 m로 변환)
     * @return: 해당 길이의 배선 커패시턴스(F)
     *
     * Wire 모델을 이용해 배선 간격을 3배로 넓혀 배치한 경우의 커패시턴스를 반환한다.
     * 라우터 제어선처럼 노이즈 마진이 중요한 신호에 적용된다.
     *
     * 호출 체인: crossbar_ctrline() → [Cw3(length)] → Wire::wire_cap()
     */
    double Cw3(double len);

    /*
     * [한국어]
     * crossbar_ctrline - 크로스바 제어선 커패시턴스 계산
     *
     * @return: 크로스바 제어선에 연결된 총 커패시턴스(F)
     *
     * 배선 커패시턴스(Cw3) + 제어 인버터(NTi/PTi) 드레인·게이트 커패시턴스를 합산한다.
     * 크로스바 출력 방향으로 뻗는 제어선의 부하 커패시턴스를 모델링한다.
     *
     * 호출 체인: arb_grant() → [crossbar_ctrline()] → Cw3()
     */
    double crossbar_ctrline();

    /*
     * [한국어]
     * transmission_buf_ctrcap - 전송 게이트(transmission gate) 제어 커패시턴스
     *
     * @return: 전송 게이트의 NMOS+PMOS 게이트 커패시턴스 합계(F)
     *
     * 크로스바 데이터 경로의 전송 게이트를 온/오프 제어하는 신호가 구동해야 하는
     * 커패시턴스를 반환한다. NTtr(NMOS), PTtr(PMOS) 게이트 커패시턴스 합산.
     *
     * 호출 체인: 크로스바(Crossbar) 모델 → [transmission_buf_ctrcap()]
     */
    double transmission_buf_ctrcap();



  private:
    double NTn1, PTn1;
    /* NTn1: NOR1 게이트의 NMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 13.5/2 로 초기화.
     * PTn1: NOR1 게이트의 PMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 76/2 로 초기화.
     * 설정자: 생성자에서 g_ip->F_sz_um 기반으로 1회 설정.
     * 읽는 자: arb_req(), arb_pri(), arb_grant(), arb_int(), compute_power().
     * 값 범위: 공정 노드에 따라 수 nm~수십 nm 수준.
     * 동기화: 생성 후 불변이므로 별도 동기화 불필요. */
    double NTn2, PTn2;
    /* NTn2: NOR2 게이트의 NMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 13.5/2.
     * PTn2: NOR2 게이트의 PMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 76/2.
     * 설정자: 생성자에서 1회 설정. 읽는 자: arb_req(), arb_int(), compute_power().
     * 동기화: 불변. */
    double R;
    /* R: 아비터의 입력 요청 수 (팬인). NOR2 게이트의 스택 깊이 결정에 사용.
     * 설정자: 생성자 파라미터 n_req 로 초기화.
     * 읽는 자: arb_req(), arb_int(), compute_power().
     * 값 범위: 양의 정수 (통상 4~16).
     * 동기화: 불변. */
    double PTi, NTi;
    /* PTi: 크로스바 제어 인버터의 PMOS 폭 (µm). 공정 피처 사이즈 × 25/2.
     * NTi: 크로스바 제어 인버터의 NMOS 폭 (µm). 공정 피처 사이즈 × 12.5/2.
     * 설정자: 생성자에서 1회 설정. 읽는 자: arb_req(), crossbar_ctrline().
     * 동기화: 불변. */
    double flit_size;
    /* flit_size: 이 아비터가 서비스하는 플릿(flit) 크기(비트).
     * 설정자: 생성자 파라미터 flit_size_. 읽는 자: print_arbiter().
     * 값 범위: 일반적으로 64~512비트 (NoC 구성에 따라 상이).
     * 동기화: 불변. */
    double NTtr, PTtr;
    /* NTtr: 전송 게이트 NMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 10/2.
     * PTtr: 전송 게이트 PMOS 트랜지스터 폭 (µm). 공정 피처 사이즈 × 20/2.
     * 설정자: 생성자에서 1회 설정. 읽는 자: transmission_buf_ctrcap().
     * 동기화: 불변. */
    double o_len;
    /* o_len: 크로스바 출력 방향 배선 길이 (µm).
     * 설정자: 생성자 파라미터 output_len. 읽는 자: crossbar_ctrline() → Cw3().
     * 값 범위: 수백~수천 µm (캐시 배열 크기에 따라 상이).
     * 동기화: 불변. */
    TechnologyParameter::DeviceType *deviceType;
    /* deviceType: 현재 공정의 디바이스 특성 포인터 (Vdd, n_to_p_eff_curr_drv_ratio 등 포함).
     * 설정자: 생성자 파라미터 dt (기본값 g_tp.peri_global 주변 회로 기술).
     * 읽는 자: 생성자에서 min_w_pmos, Vdd 추출에 사용.
     * 동기화: 전역 g_tp를 가리키므로 멀티스레드 접근 시 외부에서 보호 필요. */
    double TriS1, TriS2;
    /* TriS1, TriS2: 트라이-스테이트 버퍼 크기 파라미터 (현재 구현에서는 미사용/예약).
     * 설정자: 미사용 — 선언만 존재.
     * 동기화: 해당 없음. */
    double min_w_pmos, Vdd;
    /* min_w_pmos: 최소 PMOS 폭 = n_to_p_eff_curr_drv_ratio × g_tp.min_w_nmos_ (µm).
     *   PMOS는 전자 이동도가 낮아 동일 구동력을 얻으려면 NMOS보다 넓어야 한다.
     *   설정자: 생성자. 읽는 자: compute_power() — cmos_Isub_leakage/cmos_Ig_leakage 호출 시.
     * Vdd: 공급 전압 (V). 동적 에너지(½CV²) 및 누설 전력(I×Vdd) 계산에 사용.
     *   설정자: 생성자에서 deviceType->Vdd 복사. 읽는 자: compute_power().
     * 동기화: 불변. */

};

#endif
