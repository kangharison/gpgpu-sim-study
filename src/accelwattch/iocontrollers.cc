/*****************************************************************************
 *                                McPAT
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
 * [한국어 설명] McPAT/AccelWattch 온칩 I/O 컨트롤러 구현 (iocontrollers.cc)
 *
 * === 파일의 역할 ===
 * iocontrollers.h에서 선언한 NIUController, PCIeController, FlashController의
 * 생성자와 computeEnergy(), displayEnergy(), set_*_param() 함수들을 구현한다.
 * 각 컨트롤러는 Cadence ChipEstimator나 Niagara 2 다이 포토 기반의 65nm
 * 경험적 데이터를 기술 노드(F_sz_nm)와 전압(Vdd)으로 스케일링하여 면적과
 * 전력을 추정한다. 서브스레숄드 누설과 게이트 누설은 cmos_Isub_leakage() /
 * cmos_Ig_leakage()를 사용하며, longer_channel_device_reduction()으로
 * 장채널 소자 보정을 적용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 Uncore 영역에 속한다. Processor 클래스가 GPU 설정에
 * 따라 이 컨트롤러들을 선택적으로 생성하고 전체 전력에 합산한다.
 *   Processor::Processor() → [new NIUController/PCIeController/FlashController]
 *                            → computeEnergy() → Processor::power/rt_power
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계(사이클 루프 밖).
 *
 * === 타 모듈과의 연결 ===
 * 의존: iocontrollers.h, XML_Parse.h(ParseXML, 파라미터 구조체),
 *       basic_components.h(Component, powerDef, longer_channel_device_reduction),
 *       cacti/basic_circuit.h(cmos_Isub_leakage, cmos_Ig_leakage),
 *       cacti/parameter.h(InputParameter, init_interface, g_tp),
 *       io.h, logic.h, array.h 등.
 * 의존 받음: processor.cc(Processor의 생성/소멸/전력 집계).
 * 데이터 흐름: XML 파라미터 → set_*_param() → 면적/동적/누설 계산 → power_t
 *             → computeEnergy(is_tdp) → power/rt_power.
 *
 * === 주요 함수/구조체 요약 ===
 * NIUController::NIUController() — NIU MAC/PCS/SerDes 면적·전력 초기화.
 * NIUController::computeEnergy() — duty_cycle(TDP) 또는 perc_load(런타임) 적용.
 * NIUController::displayEnergy() — NIU 전력/면적 결과 출력.
 * NIUController::set_niu_param() — XML->sys.niu 파라미터 복사.
 * PCIeController::PCIeController() — PCIe 컨트롤러 + PHY 면적·전력 초기화.
 * PCIeController::computeEnergy()/displayEnergy()/set_pcie_param() — 동일 패턴.
 * FlashController::FlashController() — Flash 컨트롤러 + PHY 면적·전력 초기화.
 * FlashController::computeEnergy()/displayEnergy()/set_fc_param() — 동일 패턴.
 */
#include "iocontrollers.h"           // [한국어] NIUController, PCIeController, FlashController 클래스 선언
#include <assert.h>                  // [한국어] assert() — 방어적 검사 (현재는 직접 사용되지 않음)
#include <algorithm>                 // [한국어] std::min/max 등 — 현재 직접 사용되지 않으나 포함
#include <cmath>                     // [한국어] sqrt() — SerDes 전력 스케일링에 사용
#include <iostream>                  // [한국어] cout — 경고 메시지 및 결과 출력
#include <string>                    // [한국어] std::string — indent 문자열 생성 등
#include "XML_Parse.h"               // [한국어] ParseXML 및 niup/pciep/fcp 파라미터 구조체
#include "basic_components.h"        // [한국어] Component, powerDef, longer_channel_device_reduction()
#include "cacti/basic_circuit.h"     // [한국어] cmos_Isub_leakage(), cmos_Ig_leakage(), pmos_to_nmos_sz_ratio()
#include "const.h"                   // [한국어] McPAT 전역 상수 (nand, inv 등 회류 유형)
#include "io.h"                      // [한국어] McPAT I/O 출력 유틸리티
#include "logic.h"                   // [한국어] selection_logic 등 로직 모델 (간접 의존)
#include "parameter.h"               // [한국어] init_interface(), g_tp 전역 기술 파라미터

/*
 * SUN Niagara 2 I/O power analysis:
 * total signal bits: 711
 * Total FBDIMM bits: (14+10)*2*8= 384
 * PCIe bits:         (8 + 8)*2 = 32
 * 10Gb NIC:          (4*2+4*2)*2 = 32
 * Debug I/Os:        168
 * Other I/Os:        711- 32-32 - 384 - 168 = 95
 *
 * According to "Implementation of an 8-Core, 64-Thread, Power-Efficient SPARC
 * Server on a Chip" 90% of I/Os are SerDers (the calucaltion is
 * 384+64/(711-168)=83% about the same as the 90% reported in the paper)
 * --> around 80Pins are common I/Os.
 * Common I/Os consumes 71mW/Gb/s according to Cadence ChipEstimate @65nm
 * Niagara 2 I/O clock is 1/4 of core clock. --> 87pin (<--((711-168)*17%)) *
 * 71mW/Gb/s *0.25*1.4Ghz = 2.17W
 *
 * Total dynamic power of FBDIMM, NIC, PCIe = 84*0.132 + 84*0.049*0.132 = 11.14
 * - 2.17 = 8.98 Further, if assuming I/O logic power is about 50% of I/Os then
 * Total energy of FBDIMM, NIC, PCIe = 11.14 - 2.17*1.5 = 7.89
 */
// [한국어] 위 주석 블록은 Niagara 2 프로세서의 I/O 전력 분석 원문이다.
//          FBDIMM, PCIe, 10Gb NIC, Debug/Other I/O 비트 수와 SerDes 비율,
//          Cadence ChipEstimator의 65nm common I/O 전력(71mW/Gb/s)을 바탕으로
//          NIU/SerDes 동적 전력 추정의 근거를 설명한다.

/*
 * A bug in Cadence ChipEstimator: After update the clock rate in the clock tab,
 * a user need to re-select the IP clock (the same clk) and then click Estimate.
 * if not reselect the new clock rate may not be propogate into the IPs.
 *
 */
// [한국어] Cadence ChipEstimator 사용 시 클럭 설정 갱신 후 동일 IP 클럭을
//          재선택하지 않으면 새 클럭이 IP에 반영되지 않는 버그에 대한 주석.
//          McPAT의 경험적 모델 데이터 해석 시 참고용이다.

/*
 * [한국어]
 * NIUController::NIUController - 10Gb Ethernet NIU 면적·전력 초기화 생성자
 *
 * @XML_interface: GPU/시스템 설정 XML 파싱 객체 포인터.
 * @interface_ip_: CACTI 공정 파라미터 (기술 노드, 전압, 온도 등).
 * @return: (생성자) — area, power_t, local_result 초기화.
 *
 * NIU는 MAC, 프론트엔드 PCS, SerDes PHY 세 부분으로 구성된다.
 *   - 고성능 NIU(niup.type == 0): Niagara 2/Cadence 65nm 기반 면적/전력 추정.
 *   - 저전력 NIU(niup.type != 0): Cadence ChipEstimator 저전력 IP 기반 추정.
 * 면적은 65nm 기준 값에서 (F_sz_um/0.065)^2 또는 (F_sz_um/0.065)로 스케일하고,
 * 동적 전력은 Vdd^2, 공정 비율(F_sz_nm/65), 클럭 주기(1/clockRate)로 스케일한다.
 * 누설 전력은 gate 수 × cmos_Isub_leakage × Vdd로 계산한다.
 *
 * 호출 체인:
 *   Processor::Processor() → [NIUController::NIUController]
 *       → set_niu_param() → init_interface() → 면적/전력 계산
 */
NIUController::NIUController(ParseXML* XML_interface,
                             InputParameter* interface_ip_)
    : XML(XML_interface), interface_ip(*interface_ip_) {
  local_result = init_interface(&interface_ip);
  // [한국어] CACTI 전역 기술 파라미터(g_tp)를 interface_ip 기반으로 초기화.
  //          local_result는 init_interface의 반환값이며, 이후 직접 사용하지는 않고
  //          부수 효과로 g_tp가 설정되는 것이 목적이다.

  double frontend_area, mac_area, SerDer_area;
  // [한국어] PCS 프론트엔드, MAC, SerDes PHY 각각의 면적 임시 변수 (단위: mm² 또는 um²)

  double frontend_dyn, mac_dyn, SerDer_dyn;
  // [한국어] 각 서브블록의 동적 에너지 임시 변수 (단위: J/cycle)

  double frontend_gates, mac_gates;
  // [한국어] 각 서브블록의 게이트 수 — 누설 전력 계산에 사용

  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  // [한국어] PMOS 대 NMOS 전류 구동비 — 누설 계산 시 PMOS 폭 산출에 사용

  double NMOS_sizing, PMOS_sizing;
  // [한국어] 누설 전류 계산용 NMOS/PMOS 트랜지스터 폭

  set_niu_param();
  // [한국어] XML->sys.niu에서 클럭, 유닛 수, duty_cycle, 부하율, 타입을 읽어 niup에 저장.

  if (niup.type == 0)  // high performance NIU
  {
    // [한국어] 고성능 NIU 모드: Niagara 2 10Gb Ethernet 기준.

    // Area estimation based on average of die photo from Niagara 2 and Cadence
    // ChipEstimate using 65nm.
    // [한국어] MAC 면적: Niagara 2 다이 포토(1.53)와 Cadence(0.3)의 평균을 65nm 기준에서
    //          현재 공정(F_sz_um)으로 면적 비례 스케일링. (F_sz_um/0.065)^2
    mac_area = (1.53 + 0.3) / 2 * (interface_ip.F_sz_um / 0.065) *
               (interface_ip.F_sz_um / 0.065);

    // Area estimation based on average of die photo from Niagara 2, ISSCC "An
    // 800mW 10Gb Ethernet Transceiver in 0.13μm CMOS" and"A 1.2-V-Only 900-mW
    // 10 Gb Ethernet Transceiver and XAUI Interface With Robust VCO Tuning
    // Technique" Frontend is PCS
    // [한국어] PCS 프론트엔드 면적: 65nm 기준 여러 출처의 평균값을 현재 공정으로 스케일링.
    frontend_area = (9.8 + (6 + 18) * 65 / 130 * 65 / 130) / 3 *
                    (interface_ip.F_sz_um / 0.065) *
                    (interface_ip.F_sz_um / 0.065);

    // Area estimation based on average of die photo from Niagara 2 and Cadence
    // ChipEstimate hard IP @65nm. SerDer is very hard to scale
    // [한국어] SerDes PHY 면적: 면적이 아닌 선형 길이 위주로 스케일링하므로
    //          (F_sz_um/0.065) 한 번만 곱함. 면적 스케일이 어려운 아날로그 회로 특성 반영.
    SerDer_area = (1.39 + 0.36) * (interface_ip.F_sz_um /
                                   0.065);  //* (interface_ip.F_sz_um/0.065);

    // total area
    // [한국어] 위 세 면적의 합에 1e6을 곱해 mm² → um² 단위로 변환하여 area에 저장.
    area.set_area((mac_area + frontend_area + SerDer_area) * 1e6);

    // Power
    // Cadence ChipEstimate using 65nm (mac, front_end are all energy. E=P*T =
    // P/F = 1.37/1Ghz = 1.37e-9);
    // [한국어] MAC 동적 에너지: 65nm 1GHz 기준 2.19W를 현재 전압과 공정으로 스케일링.
    //          에너지 = 전력/주파수 이므로 2.19e-9 J가 기준값.
    mac_dyn = 2.19e-9 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
              1.1 *
              (interface_ip.F_sz_nm /
               65.0);  // niup.clockRate; //2.19W@1GHz fully active according to
                       // Cadence ChipEstimate @65nm

    // Cadence ChipEstimate using 65nm soft IP;
    // [한국어] PCS 프론트엔드 동적 에너지: 0.27W@1GHz 기준을 동일하게 스케일링.
    frontend_dyn = 0.27e-9 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
                   1.1 * (interface_ip.F_sz_nm / 65.0);  // niup.clockRate;

    // according to "A 100mW 9.6Gb/s Transceiver in 90nm CMOS..." ISSCC 2006
    // SerDer_dyn is power not energy, scaling from 10mw/Gb/s @90nm
    // [한국어] SerDes 동적 전력: 90nm 기준 10mW/Gb/s × 10Gb/s = 100mW를
    //          현재 공정과 전압으로 스케일링한 뒤, 1/clockRate로 에너지(J/cycle) 변환.
    SerDer_dyn = 0.01 * 10 * sqrt(interface_ip.F_sz_um / 0.09) *
                 g_tp.peri_global.Vdd / 1.2 * g_tp.peri_global.Vdd / 1.2;
    SerDer_dyn /=
        niup.clockRate;  // covert to energy per clock cycle of whole NIU

    // Cadence ChipEstimate using 65nm
    // [한국어] 각 서브블록의 게이트 수 — 누설 전력 계산의 기준.
    mac_gates = 111700;
    frontend_gates = 320000;

    // [한국어] 누설 계산용 트랜지스터 폭: 고성능 모드에서는 5배 최소 폭 사용.
    NMOS_sizing = 5 * g_tp.min_w_nmos_;
    PMOS_sizing = 5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;

  } else {  // Low power implementations are mostly from Cadence ChipEstimator;
            // Ignore the multiple IP effect
    // ---When there are multiple IP (same kind or not) selected, Cadence
    // ChipEstimator results are not a simple summation of all IPs. Ignore this
    // effect
    // [한국어] 저전력 NIU 모드: Cadence ChipEstimator 저전력 IP 기준.
    //          면적/전력 추정 방식은 고성능과 동일하나 기준 계수가 더 작음.
    mac_area =
        0.24 * (interface_ip.F_sz_um / 0.065) * (interface_ip.F_sz_um / 0.065);
    frontend_area =
        0.1 * (interface_ip.F_sz_um / 0.065) *
        (interface_ip.F_sz_um / 0.065);  // Frontend is the PCS layer
    SerDer_area =
        0.35 * (interface_ip.F_sz_um / 0.065) * (interface_ip.F_sz_um / 0.065);

    // Compare 130um implementation in "A 1.2-V-Only 900-mW 10 Gb Ethernet
    // Transceiver and XAUI Interface With Robust VCO Tuning Technique" and the
    // ChipEstimator XAUI PHY hard IP, confirm that even PHY can scale perfectly
    // with the technology total area
    // [한국어] 저전력 PHY는 디지털 비율이 높아 면적 스케일링이 가능하므로
    //          (F_sz_um/0.065)^2로 확장.
    area.set_area((mac_area + frontend_area + SerDer_area) * 1e6);

    // Power
    // [한국어] 저전력 MAC/프론트엔드/SerDes 동적 에너지 추정.
    mac_dyn = 1.257e-9 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
              1.1 *
              (interface_ip.F_sz_nm /
               65.0);  // niup.clockRate; //2.19W@1GHz fully active according to
                       // Cadence ChipEstimate @65nm
    frontend_dyn = 0.6e-9 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
                   1.1 * (interface_ip.F_sz_nm / 65.0);  // niup.clockRate;

    // SerDer_dyn is power not energy, scaling from 216mw/10Gb/s @130nm
    SerDer_dyn = 0.0216 * 10 * (interface_ip.F_sz_um / 0.13) *
                 g_tp.peri_global.Vdd / 1.2 * g_tp.peri_global.Vdd / 1.2;
    SerDer_dyn /=
        niup.clockRate;  // covert to energy per clock cycle of whole NIU

    // [한국어] 저전력 모드 게이트 수 및 트랜지스터 폭(최소 폭 사용).
    mac_gates = 111700;
    frontend_gates = 52000;

    NMOS_sizing = g_tp.min_w_nmos_;
    PMOS_sizing = g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;
  }

  // [한국어] NIU 전체 동적 에너지 = MAC + PCS + SerDes 동적 에너지 합.
  power_t.readOp.dynamic = mac_dyn + frontend_dyn + SerDer_dyn;

  // [한국어] 서브스레숄드 누설 전력: 총 게이트 수 × 단일 게이트 누설 × Vdd.
  //          (mac_gates + frontend_gates + frontend_gates)에서 frontend_gates가
  //          두 번 더해지는 것은 원본 McPAT의 PCS+SerDes 게이트 추정 관행을 따름.
  power_t.readOp.leakage =
      (mac_gates + frontend_gates + frontend_gates) *
      cmos_Isub_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W

  // [한국어] 장채널 소자 사용 시 누설 보정 계수를 Uncore_device 기준으로 계산.
  double long_channel_device_reduction =
      longer_channel_device_reduction(Uncore_device);

  // [한국어] 보정된 장채널 누설 전력 저장 — displayEnergy에서 long_channel=true일 때 출력.
  power_t.readOp.longer_channel_leakage =
      power_t.readOp.leakage * long_channel_device_reduction;

  // [한국어] 게이트 산화막 터널링 누설 전력 계산.
  power_t.readOp.gate_leakage =
      (mac_gates + frontend_gates + frontend_gates) *
      cmos_Ig_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W
}

/*
 * [한국어]
 * NIUController::computeEnergy - TDP 또는 런타임 모드로 NIU 동적 전력 재조정
 *
 * @is_tdp: true이면 TDP 모드(peak), false이면 런타임 모드.
 * @return: void — power 또는 rt_power에 결과를 저장.
 *
 * 생성자에서 구한 power_t(100% 활성 기준)에 대해:
 *   - TDP: duty_cycle(활성 사이클 비율)을 곱함.
 *   - 런타임: perc_load(실제 부하 비율)을 곱함.
 * 누설 전력은 동적 전력과 달리 활동률과 무관하므로 그대로 복사한다.
 *
 * 호출 체인:
 *   Processor::computeEnergy() → NIUController::computeEnergy(is_tdp)
 */
void NIUController::computeEnergy(bool is_tdp) {
  if (is_tdp) {
    // [한국어] TDP(peak) 모드: power_t를 그대로 복사한 뒤 동적 전력에 duty_cycle 적용.
    power = power_t;
    power.readOp.dynamic *= niup.duty_cycle;

  } else {
    // [한국어] 런타임 모드: rt_power에 power_t를 복사하고 실제 부하 비율(perc_load) 적용.
    rt_power = power_t;
    rt_power.readOp.dynamic *= niup.perc_load;
  }
}

/*
 * [한국어]
 * NIUController::displayEnergy - NIU 전력·면적 결과를 stdout에 출력
 *
 * @indent: 들여쓰기 공백 수.
 * @plevel: 출력 상세 수준 (현재 사용되지 않음).
 * @is_tdp: true이면 TDP 결과, false이면 런타임 결과(현재 else 분기는 비어 있음).
 * @return: void.
 *
 * Area(mm²), Peak Dynamic(W), Subthreshold Leakage(W), Gate Leakage(W),
 * Runtime Dynamic(W)를 출력한다. longer_channel_device 플래그에 따라
 * 누설 출력 시 longer_channel_leakage 또는 기본 leakage를 선택한다.
 *
 * 호출 체인:
 *   Processor::displayEnergy() → NIUController::displayEnergy()
 */
void NIUController::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  bool long_channel = XML->sys.longer_channel_device;

  if (is_tdp) {
    cout << "NIU:" << endl;
    cout << indent_str << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str
         << "Peak Dynamic = " << power.readOp.dynamic * niup.clockRate << " W"
         << endl;
    cout << indent_str << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    // cout << indent_str<< "Subthreshold Leakage = " <<
    // power.readOp.longer_channel_leakage <<" W" << endl;
    cout << indent_str << "Gate Leakage = " << power.readOp.gate_leakage << " W"
         << endl;
    cout << indent_str
         << "Runtime Dynamic = " << rt_power.readOp.dynamic * niup.clockRate
         << " W" << endl;
    cout << endl;
  } else {
  }
}

/*
 * [한국어]
 * NIUController::set_niu_param - XML에서 NIU 파라미터를 읽어 niup에 저장
 *
 * @return: void.
 *
 * XML->sys.niu의 클럭(MHz), 유닛 수, duty_cycle, 부하 비율, 타입을 복사한다.
 * 클럭은 MHz → Hz로 1e6을 곱해 변환한다.
 *
 * 호출 체인:
 *   NIUController::NIUController() → set_niu_param()
 */
void NIUController::set_niu_param() {
  niup.clockRate = XML->sys.niu.clockrate;
  niup.clockRate *= 1e6;  // [한국어] MHz → Hz 변환

  niup.num_units = XML->sys.niu.number_units;
  // [한국어] 칩 내 NIU 유닛 수 — Processor에서 최종적으로 이 수만큼 면적/전력 확장.

  niup.duty_cycle = XML->sys.niu.duty_cycle;
  // [한국어] TDP 모드에서 사용하는 활성 사이클 비율.

  niup.perc_load = XML->sys.niu.total_load_perc;
  // [한국어] 런타임 모드에서 사용하는 실제 네트워크 부하 비율.

  niup.type = XML->sys.niu.type;
  // [한국어] 0=고성능 NIU, 1=저전력 NIU.

  //	  niup.executionTime   =
  // XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);
}

/*
 * [한국어]
 * PCIeController::PCIeController - PCIe 컨트롤러 면적·전력 초기화 생성자
 *
 * @XML_interface: GPU/시스템 설정 XML 파싱 객체 포인터.
 * @interface_ip_: CACTI 공정 파라미터.
 * @return: (생성자) — area, power_t 초기화.
 *
 * PCIe는 컨트롤러 로직(ctrl)과 PHY(SerDes)로 구성되며, bit-slice 기반으로
 * per-lane 값을 계산한 뒤 num_channels로 확장한다. 고성능(type==0)과
 * 저전력(type==1) 두 모드를 지원한다. withPHY=false이면 PHY 면적/전력을 0으로
 * 처리할 수 있다.
 *
 * 호출 체인:
 *   Processor::Processor() → [PCIeController::PCIeController]
 *       → set_pcie_param() → init_interface() → 면적/전력 계산
 */
PCIeController::PCIeController(ParseXML* XML_interface,
                               InputParameter* interface_ip_)
    : XML(XML_interface), interface_ip(*interface_ip_) {
  local_result = init_interface(&interface_ip);
  // [한국어] CACTI 전역 기술 파라미터 초기화 (부수 효과용).

  double ctrl_area, SerDer_area;
  // [한국어] 컨트롤러 로직 면적, PHY 면적 임시 변수

  double ctrl_dyn, SerDer_dyn;
  // [한국어] 컨트롤러 로직 동적 에너지, PHY 동적 에너지 임시 변수

  double ctrl_gates, SerDer_gates;
  // [한국어] 각 부분의 게이트 수 — 누설 전력 계산용

  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  // [한국어] PMOS/NMOS 구동비

  double NMOS_sizing, PMOS_sizing;
  // [한국어] 누설 계산용 트랜지스터 폭

  /* Assuming PCIe is bit-slice based architecture
   * This is the reason for /8 in both area and power calculation
   * to get per lane numbers
   */
  // [한국어] PCIe는 bit-slice 구조로 가정: area/power 계산에서 /8은 per-lane 값을
  //          얻기 위한 것이며, 이후 num_channels로 전체 채널 수만큼 확장한다.

  set_pcie_param();
  // [한국어] XML->sys.pcie에서 파라미터를 읽어 pciep에 저장.

  if (pciep.type == 0)  // high performance NIU
  {
    // [한국어] 고성능 PCIe 모드.

    // Area estimation based on average of die photo from Niagara 2 and Cadence
    // ChipEstimate @ 65nm.
    ctrl_area = (5.2 + 0.5) / 2 * (interface_ip.F_sz_um / 0.065) *
                (interface_ip.F_sz_um / 0.065);

    // Area estimation based on average of die photo from Niagara 2, and Cadence
    // ChipEstimate @ 65nm. Area estimation based on average of die photo from
    // Niagara 2 and Cadence ChipEstimate hard IP @65nm. SerDer is very hard to
    // scale
    SerDer_area = (3.03 + 0.36) * (interface_ip.F_sz_um /
                                   0.065);  //* (interface_ip.F_sz_um/0.065);

    // total area
    // Power
    // Cadence ChipEstimate using 65nm the controller includes everything: the
    // PHY, the data link and transaction layer
    ctrl_dyn = 3.75e-9 / 8 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
               1.1 * (interface_ip.F_sz_nm / 65.0);

    //	  //Cadence ChipEstimate using 65nm soft IP;
    //	  frontend_dyn =
    // 0.27e-9/8*g_tp.peri_global.Vdd/1.1*g_tp.peri_global.Vdd/1.1*(interface_ip.F_sz_nm/65.0);

    // SerDer_dyn is power not energy, scaling from 10mw/Gb/s @90nm
    SerDer_dyn = 0.01 * 4 * (interface_ip.F_sz_um / 0.09) *
                 g_tp.peri_global.Vdd / 1.2 * g_tp.peri_global.Vdd /
                 1.2;               // PCIe 2.0 max per lane speed is 4Gb/s
    SerDer_dyn /= pciep.clockRate;  // covert to energy per clock cycle

    // power_t.readOp.dynamic = (ctrl_dyn)*pciep.num_channels;
    // Cadence ChipEstimate using 65nm
    ctrl_gates = 900000 / 8 * pciep.num_channels;
    //	  frontend_gates   = 120000/8;
    //	  SerDer_gates     = 200000/8;
    NMOS_sizing = 5 * g_tp.min_w_nmos_;
    PMOS_sizing = 5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;
  } else {
    // [한국어] 저전력 PCIe 모드.
    ctrl_area =
        0.412 * (interface_ip.F_sz_um / 0.065) * (interface_ip.F_sz_um / 0.065);

    // Area estimation based on average of die photo from Niagara 2, and Cadence
    // ChipEstimate @ 65nm.
    SerDer_area =
        0.36 * (interface_ip.F_sz_um / 0.065) * (interface_ip.F_sz_um / 0.065);

    // total area
    // Power
    // Cadence ChipEstimate using 65nm the controller includes everything: the
    // PHY, the data link and transaction layer
    ctrl_dyn = 2.21e-9 / 8 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd /
               1.1 * (interface_ip.F_sz_nm / 65.0);

    //	  //Cadence ChipEstimate using 65nm soft IP;
    //	  frontend_dyn =
    // 0.27e-9/8*g_tp.peri_global.Vdd/1.1*g_tp.peri_global.Vdd/1.1*(interface_ip.F_sz_nm/65.0);

    // SerDer_dyn is power not energy, scaling from 10mw/Gb/s @90nm
    SerDer_dyn = 0.01 * 4 * (interface_ip.F_sz_um / 0.09) *
                 g_tp.peri_global.Vdd / 1.2 * g_tp.peri_global.Vdd /
                 1.2;               // PCIe 2.0 max per lane speed is 4Gb/s
    SerDer_dyn /= pciep.clockRate;  // covert to energy per clock cycle

    // Cadence ChipEstimate using 65nm
    ctrl_gates = 200000 / 8 * pciep.num_channels;
    //	  frontend_gates   = 120000/8;
    SerDer_gates = 200000 / 8 * pciep.num_channels;
    NMOS_sizing = g_tp.min_w_nmos_;
    PMOS_sizing = g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;
  }

  // [한국어] withPHY 플래그에 따라 PHY 면적/동적 전력을 포함하거나 0으로 처리한 뒤
  //          num_channels로 확장. ctrl 부분은 이미 per-lane × num_channels로 계산됨.
  area.set_area(((ctrl_area + (pciep.withPHY ? SerDer_area : 0)) / 8 *
                 pciep.num_channels) *
                1e6);

  power_t.readOp.dynamic =
      (ctrl_dyn + (pciep.withPHY ? SerDer_dyn : 0)) * pciep.num_channels;

  // [한국어] 누설 전력: ctrl_gates + (PHY 게이트 수) × 단위 누설 × Vdd.
  power_t.readOp.leakage =
      (ctrl_gates + (pciep.withPHY ? SerDer_gates : 0)) *
      cmos_Isub_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W

  double long_channel_device_reduction =
      longer_channel_device_reduction(Uncore_device);
  power_t.readOp.longer_channel_leakage =
      power_t.readOp.leakage * long_channel_device_reduction;

  power_t.readOp.gate_leakage =
      (ctrl_gates + (pciep.withPHY ? SerDer_gates : 0)) *
      cmos_Ig_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W
}

/*
 * [한국어]
 * PCIeController::computeEnergy - TDP/런타임 모드로 PCIe 동적 전력 재조정
 *
 * @is_tdp: true이면 TDP, false이면 런타임.
 * @return: void — power/rt_power에 저장.
 *
 * 생성자의 power_t에 duty_cycle(TDP) 또는 perc_load(런타임)을 곱한다.
 *
 * 호출 체인:
 *   Processor::computeEnergy() → PCIeController::computeEnergy(is_tdp)
 */
void PCIeController::computeEnergy(bool is_tdp) {
  if (is_tdp) {
    power = power_t;
    power.readOp.dynamic *= pciep.duty_cycle;

  } else {
    rt_power = power_t;
    rt_power.readOp.dynamic *= pciep.perc_load;
  }
}

/*
 * [한국어]
 * PCIeController::displayEnergy - PCIe 전력·면적 결과를 stdout에 출력
 *
 * @indent: 들여쓰기 공백 수.
 * @plevel: 출력 상세 수준 (미사용).
 * @is_tdp: true이면 TDP 결과, false이면 런타임(else 분기 비어 있음).
 * @return: void.
 *
 * NIUController::displayEnergy()와 동일한 항목을 "PCIe:" 헤더로 출력한다.
 *
 * 호출 체인:
 *   Processor::displayEnergy() → PCIeController::displayEnergy()
 */
void PCIeController::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  bool long_channel = XML->sys.longer_channel_device;

  if (is_tdp) {
    cout << "PCIe:" << endl;
    cout << indent_str << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str
         << "Peak Dynamic = " << power.readOp.dynamic * pciep.clockRate << " W"
         << endl;
    cout << indent_str << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    // cout << indent_str<< "Subthreshold Leakage = " <<
    // power.readOp.longer_channel_leakage <<" W" << endl;
    cout << indent_str << "Gate Leakage = " << power.readOp.gate_leakage << " W"
         << endl;
    cout << indent_str
         << "Runtime Dynamic = " << rt_power.readOp.dynamic * pciep.clockRate
         << " W" << endl;
    cout << endl;
  } else {
  }
}

/*
 * [한국어]
 * PCIeController::set_pcie_param - XML에서 PCIe 파라미터를 읽어 pciep에 저장
 *
 * @return: void.
 *
 * XML->sys.pcie의 클럭, 유닛 수, 채널 수, duty_cycle, 부하 비율, 타입,
 * PHY 사용 여부를 복사한다. 클럭은 MHz → Hz로 변환한다.
 *
 * 호출 체인:
 *   PCIeController::PCIeController() → set_pcie_param()
 */
void PCIeController::set_pcie_param() {
  pciep.clockRate = XML->sys.pcie.clockrate;
  pciep.clockRate *= 1e6;  // [한국어] MHz → Hz

  pciep.num_units = XML->sys.pcie.number_units;
  // [한국어] PCIe 컨트롤러 유닛 수

  pciep.num_channels = XML->sys.pcie.num_channels;
  // [한국어] PCIe 채널(레인) 수 — 면적/전력 확장 인수

  pciep.duty_cycle = XML->sys.pcie.duty_cycle;
  // [한국어] TDP 모드 활성률

  pciep.perc_load = XML->sys.pcie.total_load_perc;
  // [한국어] 런타임 모드 실제 부하 비율

  pciep.type = XML->sys.pcie.type;
  // [한국어] 0=고성능, 1=저전력

  pciep.withPHY = XML->sys.pcie.withPHY;
  // [한국어] PHY(SerDes) 면적/전력 포함 여부

  //	  pciep.executionTime   =
  // XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);
}

/*
 * [한국어]
 * FlashController::FlashController - Flash/SSD 컨트롤러 면적·전력 초기화 생성자
 *
 * @XML_interface: GPU/시스템 설정 XML 파싱 객체 포인터.
 * @interface_ip_: CACTI 공정 파라미터.
 * @return: (생성자) — area, power_t 초기화.
 *
 * Flash 컨트롤러는 컨트롤러 로직(ctrl)과 PHY(SerDes)로 구성된다.
 * 고성능 플래시 컨트롤러(type==0)는 현재 지원하지 않으므로 즉시 exit(0)로
 * 종료한다. 저전력 모드(type==1)만 정상 동작한다.
 *
 * 호출 체인:
 *   Processor::Processor() → [FlashController::FlashController]
 *       → set_fc_param() → init_interface() → 면적/전력 계산
 */
FlashController::FlashController(ParseXML* XML_interface,
                                 InputParameter* interface_ip_)
    : XML(XML_interface), interface_ip(*interface_ip_) {
  local_result = init_interface(&interface_ip);
  // [한국어] CACTI 전역 기술 파라미터 초기화 (부수 효과용).

  double ctrl_area, SerDer_area;
  double ctrl_dyn, SerDer_dyn;
  double ctrl_gates, SerDer_gates;
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  double NMOS_sizing, PMOS_sizing;

  /* Assuming PCIe is bit-slice based architecture
   * This is the reason for /8 in both area and power calculation
   * to get per lane numbers
   */
  // [한국어] Flash PHY도 bit-slice 구조로 가정하여 /8 per-lane 계산 후
  //          채널 수(number_channel)로 확장. 주석이 PCIe로 되어 있으나
  //          원본 McPAT의 FlashController에서 동일한 구조를 재사용한다.

  set_fc_param();
  // [한국어] XML->sys.flashc에서 파라미터를 읽어 fcp에 저장.

  if (fcp.type == 0)  // high performance NIU
  {
    // [한국어] 고성능 Flash 컨트롤러는 지원하지 않음.
    cout << "Current McPAT does not support high performance flash contorller "
            "since even low power designs are enough for maintain throughput"
         << endl;
    exit(0);
    NMOS_sizing = 5 * g_tp.min_w_nmos_;
    PMOS_sizing = 5 * g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;
  } else {
    // [한국어] 저전력 Flash 컨트롤러 모드.
    ctrl_area =
        0.243 * (interface_ip.F_sz_um / 0.065) * (interface_ip.F_sz_um / 0.065);

    // Area estimation based on Cadence ChipEstimate @ 65nm: NANDFLASH-CTRL from
    // CAST
    SerDer_area = 0.36 / 8 * (interface_ip.F_sz_um / 0.065) *
                  (interface_ip.F_sz_um / 0.065);

    // based On PCIe PHY TSMC65GP from Cadence ChipEstimate @ 65nm, it support
    // 8x lanes with each lane speed up to 250MB/s (PCIe1.1x) This is already
    // saturate the 200MB/s of the flash controller core above.
    // [한국어] PHY는 PCIe1.1x 8레인 250MB/s 기준으로, 플래시 컨트롤러 코어의
    //          200MB/s를 이미 충족하므로 별도 추가 대역폭 불필요.

    ctrl_gates = 129267;
    SerDer_gates = 200000 / 8;
    NMOS_sizing = g_tp.min_w_nmos_;
    PMOS_sizing = g_tp.min_w_nmos_ * pmos_to_nmos_sizing_r;

    // Power
    // Cadence ChipEstimate using 65nm the controller 125mW for every 200MB/s
    // This is power not energy!
    // [한국어] 컨트롤러 동적 전력: 65nm 125mW@200MB/s 기준을 현재 전압/공정으로 스케일링.
    ctrl_dyn = 0.125 * g_tp.peri_global.Vdd / 1.1 * g_tp.peri_global.Vdd / 1.1 *
               (interface_ip.F_sz_nm / 65.0);

    // SerDer_dyn is power not energy, scaling from 10mw/Gb/s @90nm
    SerDer_dyn = 0.01 * 1.6 * (interface_ip.F_sz_um / 0.09) *
                 g_tp.peri_global.Vdd / 1.2 * g_tp.peri_global.Vdd / 1.2;
    // max  Per controller speed is 1.6Gb/s (200MB/s)
  }

  // [한국어] 채널 수에 대한 경험적 확장: 1채널 이후 추가 채널은 20%씩만 면적/전력 증가.
  double number_channel = 1 + (fcp.num_channels - 1) * 0.2;

  area.set_area((ctrl_area + (fcp.withPHY ? SerDer_area : 0)) * 1e6 *
                number_channel);
  power_t.readOp.dynamic =
      (ctrl_dyn + (fcp.withPHY ? SerDer_dyn : 0)) * number_channel;

  power_t.readOp.leakage =
      ((ctrl_gates + (fcp.withPHY ? SerDer_gates : 0)) * number_channel) *
      cmos_Isub_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W

  double long_channel_device_reduction =
      longer_channel_device_reduction(Uncore_device);
  power_t.readOp.longer_channel_leakage =
      power_t.readOp.leakage * long_channel_device_reduction;

  power_t.readOp.gate_leakage =
      ((ctrl_gates + (fcp.withPHY ? SerDer_gates : 0)) * number_channel) *
      cmos_Ig_leakage(NMOS_sizing, PMOS_sizing, 2, nand) *
      g_tp.peri_global.Vdd;  // unit W
}

/*
 * [한국어]
 * FlashController::computeEnergy - TDP/런타임 모드로 Flash 동적 전력 재조정
 *
 * @is_tdp: true이면 TDP, false이면 런타임.
 * @return: void — power/rt_power에 저장.
 *
 * power_t에 duty_cycle(TDP) 또는 perc_load(런타임)을 곱한다.
 *
 * 호출 체인:
 *   Processor::computeEnergy() → FlashController::computeEnergy(is_tdp)
 */
void FlashController::computeEnergy(bool is_tdp) {
  if (is_tdp) {
    power = power_t;
    power.readOp.dynamic *= fcp.duty_cycle;

  } else {
    rt_power = power_t;
    rt_power.readOp.dynamic *= fcp.perc_load;
  }
}

/*
 * [한국어]
 * FlashController::displayEnergy - Flash Controller 전력·면적 결과를 stdout에 출력
 *
 * @indent: 들여쓰기 공백 수.
 * @plevel: 출력 상세 수준 (미사용).
 * @is_tdp: true이면 TDP 결과, false이면 런타임(else 분기 비어 있음).
 * @return: void.
 *
 * Flash 컨트롤러의 면적, 피크 동적 전력, 누설, 게이트 누설, 런타임 동적 전력을
 * 출력한다. PCIe/NIU와 달리 피크 동적 전력에 clockRate를 곱하지 않는다
 * (ctrl_dyn이 이미 전력 단위로 계산되었으므로).
 *
 * 호출 체인:
 *   Processor::displayEnergy() → FlashController::displayEnergy()
 */
void FlashController::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  bool long_channel = XML->sys.longer_channel_device;

  if (is_tdp) {
    cout << "Flash Controller:" << endl;
    cout << indent_str << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str << "Peak Dynamic = " << power.readOp.dynamic << " W"
         << endl;  // no multiply of clock since this is power already
    cout << indent_str << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    // cout << indent_str<< "Subthreshold Leakage = " <<
    // power.readOp.longer_channel_leakage <<" W" << endl;
    cout << indent_str << "Gate Leakage = " << power.readOp.gate_leakage << " W"
         << endl;
    cout << indent_str << "Runtime Dynamic = " << rt_power.readOp.dynamic
         << " W" << endl;
    cout << endl;
  } else {
  }
}

/*
 * [한국어]
 * FlashController::set_fc_param - XML에서 Flash 컨트롤러 파라미터를 읽어 fcp에 저장
 *
 * @return: void.
 *
 * XML->sys.flashc에서 피크 전송률, 채널 수, 컨트롤러 수, duty_cycle, 부하 비율,
 * 타입, PHY 사용 여부를 복사한다. num_channels는 피크 전송률/200으로 올림
 * 계산한다.
 *
 * 호출 체인:
 *   FlashController::FlashController() → set_fc_param()
 */
void FlashController::set_fc_param() {
  //	  fcp.clockRate       = XML->sys.flashc.mc_clock;
  //	  fcp.clockRate       *= 1e6;
  // [한국어] Flash 컨트롤러는 별도 클럭 필드가 없고 피크 전송률 기반으로 채널 수를 계산.

  fcp.peakDataTransferRate = XML->sys.flashc.peak_transfer_rate;
  // [한국어] 플래시 컨트롤러 피크 데이터 전송률 [MB/s]

  fcp.num_channels = ceil(fcp.peakDataTransferRate / 200);
  // [한국어] 채널 수 = 피크 전송률 / 200MB/s 올림 — 각 채널이 200MB/s 처리 가정.

  fcp.num_mcs = XML->sys.flashc.number_mcs;
  // [한국어] 플래시 컨트롤러 수

  fcp.duty_cycle = XML->sys.flashc.duty_cycle;
  // [한국어] TDP 모드 활성률

  fcp.perc_load = XML->sys.flashc.total_load_perc;
  // [한국어] 런타임 모드 실제 부하 비율

  fcp.type = XML->sys.flashc.type;
  // [한국어] 0=고성능(미지원), 1=저전력

  fcp.withPHY = XML->sys.flashc.withPHY;
  // [한국어] PHY 면적/전력 포함 여부

  //	  flashcp.executionTime   =
  // XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);
}
