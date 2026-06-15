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
 * [한국어 설명] McPAT/AccelWattch 온칩 I/O 컨트롤러 헤더 (iocontrollers.h)
 *
 * === 파일의 역할 ===
 * 이 헤더는 GPU/CPU 칩 외부와 고속 인터페이스를 담당하는 세 가지 온칩
 * I/O 컨트롤러(NIU, PCIe, Flash)의 클래스 선언을 담는다. AccelWattch에서
 * 이들은 Uncore 전력의 일부로 모델링되며, XML 설정값(XML->sys.niu,
 * XML->sys.pcie, XML->sys.flashc)을 기반으로 면적·누설·동적 전력을 추정한다.
 * 각 클래스는 Component 기반 클래스를 상속받아 power/area 필드를 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 최상위 Processor 클래스가 필요 시 이 컨트롤러들을
 * 생성하여 전체 칩 전력에 합산한다.
 *   Processor::Processor() → new NIUController / PCIeController / FlashController
 *                          → computeEnergy() → Processor::power/rt_power 합산
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계(사이클 루프 밖).
 *
 * === 타 모듈과의 연결 ===
 * 의존: XML_Parse.h(ParseXML, NIUParam, PCIeParam, MCParam),
 *       cacti/parameter.h(InputParameter, uca_org_t),
 *       array.h(ArrayST), basic_components.h(Component, powerDef).
 * 의존 받음: processor.cc(Processor 생성자/소멸자, computeEnergy/displayEnergy).
 * 데이터 흐름: XML 파라미터 → set_*_param() → power_t → computeEnergy() → power/rt_power.
 *
 * === 주요 함수/구조체 요약 ===
 * NIUController     : 10Gb Ethernet 네트워크 인터페이스 컨트롤러 전력 모델.
 * PCIeController    : PCIe 2.0/1.1 레인 기반 고속 I/O 컨트롤러 전력 모델.
 * FlashController   : NAND 플래시/SSD 컨트롤러 전력 모델.
 * 각 클래스는 생성자에서 면적·전력을 계산하고, computeEnergy()에서 TDP/런타임
 * 모드를 분기하며, displayEnergy()로 결과를 출력한다.
 */
#ifndef IOCONTROLLERS_H_
#define IOCONTROLLERS_H_

#endif /* IOCONTROLLERS_H_ */

#include "XML_Parse.h"          // [한국어] ParseXML 및 NIUParam/PCIeParam/MCParam 파라미터 구조체
#include "cacti/parameter.h"    // [한국어] CACTI InputParameter, uca_org_t 결과 구조체
//#include "io.h"
#include "array.h"              // [한국어] ArrayST 등 CACTI SRAM 모델 (직접 사용하지 않으나 McPAT 일관성을 위해 포함)
//#include "Undifferentiated_Core_Area.h"
#include <vector>               // [한국어] std::vector — 현재 직접 사용하지 않으나 McPAT 원본 포함
#include "basic_components.h"   // [한국어] Component 기반 클래스, powerDef, Device_ty enum

/*
 * [한국어]
 * NIUController — 10Gb Ethernet Network Interface Unit 전력·면적 모델
 *
 * NIC MAC, 프론트엔드 PCS, SerDes PHY 세 부분의 면적/전력을 분리 추정한다.
 * XML->sys.niu.* 파라미터에서 클럭, 유닛 수, duty_cycle, 부하율, 고성능/저전력
 * 타입을 읽어오며, Niagara 2/Cadence ChipEstimator 65nm 데이터를 기술 노드로
 * 스케일링하여 전력을 계산한다.
 *
 * 호출 체인:
 *   Processor::Processor() → [new NIUController] → set_niu_param() → 면적/전력 초기화
 *   Processor::computeEnergy() → NIUController::computeEnergy()
 */
class NIUController : public Component {
 public:
  ParseXML *XML;
  // [한국어] GPU/시스템 XML 파싱 객체 포인터 — sys.niu.* 설정값과 시뮬레이션 통계 접근용

  InputParameter interface_ip;
  // [한국어] CACTI 공정 파라미터 복사본 (기술 노드, 전압, 온도 등)

  NIUParam niup;
  // [한국어] NIU 특화 파라미터 집합 (clockRate, num_units, duty_cycle, perc_load, type)

  powerDef power_t;
  // [한국어] 생성자에서 계산한 기준 전력 (스케일링 전 TDP 기준)

  uca_org_t local_result;
  // [한국어] CACTI 초기화 결과 (init_interface의 부수 효과용 더미 결과)

  NIUController(ParseXML *XML_interface, InputParameter *interface_ip_);
  // [한국어] 생성자 — XML에서 파라미터를 읽어 NIU 면적과 전력을 초기화한다.

  void set_niu_param();
  // [한국어] XML->sys.niu 필드를 niup 구조체로 복사하고 클럭 단위를 MHz→Hz로 변환한다.

  void computeEnergy(bool is_tdp = true);
  // [한국어] TDP(true) 또는 런타임(false) 모드로 동적 전력을 재조정하여 power/rt_power에 저장.

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  // [한국어] NIU 면적, 피크 동적 전력, 누설 전력, 게이트 누설, 런타임 동적 전력을 stdout에 출력.

  ~NIUController(){};
  // [한국어] 소멸자 — 동적 할당이 없으므로 빈 구현.
};

/*
 * [한국어]
 * PCIeController — PCIe 고속 I/O 컨트롤러 전력·면적 모델
 *
 * PCIe 컨트롤러는 컨트롤러 로직(ctrl)과 물리 계층 PHY(SerDes) 두 부분으로
 * 구성된다. XML->sys.pcie.*에서 채널 수(num_channels), withPHY 플래그, 타입,
 * 클럭 등을 읽어와 bit-slice 방식으로 per-lane 면적/전력을 계산한 후
 * num_channels로 확장한다. 고성능(type==0)과 저전력(type==1) 두 가지 모델을
 * 지원한다.
 *
 * 호출 체인:
 *   Processor::Processor() → [new PCIeController] → set_pcie_param() → 면적/전력 초기화
 *   Processor::computeEnergy() → PCIeController::computeEnergy()
 */
class PCIeController : public Component {
 public:
  ParseXML *XML;
  // [한국어] GPU/시스템 XML 파싱 객체 포인터 — sys.pcie.* 설정값 접근용

  InputParameter interface_ip;
  // [한국어] CACTI 공정 파라미터 복사본

  PCIeParam pciep;
  // [한국어] PCIe 특화 파라미터 집합 (clockRate, num_channels, num_units,
  //          duty_cycle, perc_load, type, withPHY)

  powerDef power_t;
  // [한국어] 생성자에서 계산한 기준 전력

  uca_org_t local_result;
  // [한국어] CACTI 초기화 결과 (init_interface의 부수 효과용 더미)

  PCIeController(ParseXML *XML_interface, InputParameter *interface_ip_);
  // [한국어] 생성자 — XML에서 PCIe 파라미터를 읽어 면적과 전력을 초기화한다.

  void set_pcie_param();
  // [한국어] XML->sys.pcie 필드를 pciep 구조체로 복사하고 클럭 단위를 변환한다.

  void computeEnergy(bool is_tdp = true);
  // [한국어] TDP/런타임 모드로 동적 전력을 재조정하여 power/rt_power에 저장.

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  // [한국어] PCIe 면적 및 전력 항목을 stdout에 출력.

  ~PCIeController(){};
  // [한국어] 소멸자 — 동적 할당이 없으므로 빈 구현.
};

/*
 * [한국어]
 * FlashController — NAND 플래시/SSD 컨트롤러 전력·면적 모델
 *
 * SSD 컨트롤러의 핵심 로직(ctrl)과 PCIe/SATA PHY(SerDes) 면적/전력을 추정한다.
 * 고성능 플래시 컨트롤러(type==0)는 현재 McPAT에서 지원하지 않으며, 저전력
 * 모델(type==1)만 사용 가능하다. XML->sys.flashc.*에서 채널 수, PHY 사용 여부,
 * 부하율 등을 읽어온다.
 *
 * 호출 체인:
 *   Processor::Processor() → [new FlashController] → set_fc_param() → 면적/전력 초기화
 *   Processor::computeEnergy() → FlashController::computeEnergy()
 */
class FlashController : public Component {
 public:
  ParseXML *XML;
  // [한국어] GPU/시스템 XML 파싱 객체 포인터 — sys.flashc.* 설정값 접근용

  InputParameter interface_ip;
  // [한국어] CACTI 공정 파라미터 복사본

  MCParam fcp;
  // [한국어] Flash 컨트롤러 파라미터 집합 (peakDataTransferRate, num_channels,
  //          num_mcs, duty_cycle, perc_load, type, withPHY).
  //          MCParam을 재사용하며, 실제로는 플래시 컨트롤러 전용 값들이 채워진다.

  powerDef power_t;
  // [한국어] 생성자에서 계산한 기준 전력

  uca_org_t local_result;
  // [한국어] CACTI 초기화 결과 (init_interface의 부수 효과용 더미)

  FlashController(ParseXML *XML_interface, InputParameter *interface_ip_);
  // [한국어] 생성자 — XML에서 플래시 파라미터를 읽어 면적과 전력을 초기화한다.

  void set_fc_param();
  // [한국어] XML->sys.flashc 필드를 fcp 구조체로 복사하고 채널 수를 계산한다.

  void computeEnergy(bool is_tdp = true);
  // [한국어] TDP/런타임 모드로 동적 전력을 재조정하여 power/rt_power에 저장.

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  // [한국어] Flash Controller 면적 및 전력 항목을 stdout에 출력.

  ~FlashController(){};
  // [한국어] 소멸자 — 동적 할당이 없으므로 빈 구현.
};
