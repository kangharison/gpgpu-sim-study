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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/
/********************************************************************
 *      Modified by:
 ** Jingwen Leng, Univeristy of Texas, Austin                   * Syed Gilani,
 *University of Wisconsin–Madison                * Tayler Hetherington,
 *University of British Columbia         * Ahmed ElTantawy, University of
 *British Columbia             *
 ********************************************************************/

/*
 * [한국어 설명] 메모리 컨트롤러 전력 모델 헤더 (memoryctrl.h)
 *
 * === 파일의 역할 ===
 * AccelWattch(McPAT 기반) GPU 전력 모델에서 메모리 컨트롤러(MC)의 전력·면적을
 * 추정하기 위한 클래스 선언 파일이다. GDDR5/DDR3 메모리 시스템에 존재하는
 * 4개의 서브컴포넌트를 각각 독립 클래스로 분리하여 계층적으로 모델링한다:
 * (1) MCFrontEnd — 트랜잭션 큐·코얼레싱 로직,
 * (2) MCBackend — DRAM 컨트롤러 백엔드(DDR2/DDR3-Lite 프로토콜),
 * (3) MCPHY — 물리 계층(SerDes, I/O 버퍼),
 * (4) DRAM — 실제 GDDR5/GDDR3 디바이스 전력(IDD 전류 모델).
 * MemoryController는 이 4개를 하나로 묶는 최상위 래퍼 클래스이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 "언코어(Uncore)" 컴포넌트 중 하나로,
 * Processor 클래스가 GPU의 전체 구성 요소를 순회할 때
 * MemoryController를 생성하고 computeEnergy()를 호출한다.
 * GPGPU-Sim의 타이밍 시뮬레이터(dram.cc, gpu-sim.cc)에서 수집된
 * 메모리 접근 통계(읽기/쓰기 횟수 등)가 XML 구조체를 통해 이 모델로 전달된다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이션 종료 후 통계 집계 단계.
 *
 * === 타 모듈과의 연결 ===
 * - XML_Parse.h (ParseXML): GPU 설정·통계를 담는 최상위 구조체.
 *   XML->sys.L2[], XML->sys.mc.* 필드를 읽어 MCParam을 채운다.
 * - basic_components.h: Component 기반 클래스, statsDef, powerDef, MCParam,
 *   DRAMParam, selection_logic, Pipeline 등 공통 타입 정의.
 * - array.h (ArrayST): CACTI 기반 SRAM 배열 전력·면적 모델.
 *   MCFrontEnd의 frontendBuffer, readBuffer, writeBuffer, PRT, threadMasks,
 *   PRC가 ArrayST 인스턴스를 생성하여 큐 SRAM을 모델링한다.
 * - cacti/parameter.h (InputParameter, uca_org_t): CACTI 입력 파라미터와
 *   결과 구조체. MCBackend·MCPHY는 CACTI를 직접 호출하여 면적/전력을 추정한다.
 * - MemoryController는 gpgpu_sim_wrapper.cc에서 rd_coeff, wr_coeff를 읽어
 *   DRAM 동적 전력을 GPU 사이클 단위로 환산한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - MCBackend: DDR2/DDR3-Lite 백엔드 컨트롤러 전력 모델.
 *   compute()로 CACTI를 통한 면적 추정, computeEnergy()로 활동 통계 적용.
 * - MCPHY: PHY 물리 계층(SerDes) 전력 모델.
 *   ISSCC 2006/2007 90nm CMOS 트랜시버 데이터 기반 경험적 모델.
 * - MCFrontEnd: MC 프론트엔드 큐 로직 + GPU 코얼레싱(warp 병합) 전력 모델.
 *   PRT(Pending Request Table)와 threadMasks가 GPU 고유 확장이다.
 * - DRAM: GDDR5/GDDR3 디바이스 IDD 전류 모델.
 *   rd_coeff, wr_coeff는 gpgpu_sim_wrapper가 읽어 런타임 전력에 곱한다.
 * - MemoryController: 위 4개를 합산하는 최상위 래퍼.
 *   set_mc_param()으로 XML 파싱, computeEnergy()로 TDP/런타임 전력 계산.
 */

#ifndef MEMORYCTRL_H_
#define MEMORYCTRL_H_

#include "XML_Parse.h"        // [한국어] GPU 설정·통계 XML 파싱 구조체 (ParseXML, sys.mc.* 등)
#include "cacti/parameter.h"  // [한국어] CACTI 입력 파라미터(InputParameter)와 결과(uca_org_t) 타입
//#include "io.h"
#include "array.h"            // [한국어] CACTI 기반 SRAM 배열 전력·면적 모델 (ArrayST)
//#include "Undifferentiated_Core_Area.h"
#include <vector>             // [한국어] std::vector — 서브컴포넌트 목록 관리용 (현재 미사용이나 포함됨)
#include "basic_components.h" // [한국어] Component, statsDef, powerDef, MCParam, DRAMParam, selection_logic, Pipeline 등 공통 타입

/*
 * [한국어]
 * MCBackend - DRAM 컨트롤러 백엔드(DDR2/DDR3-Lite 프로토콜) 전력·면적 모델
 *
 * DDR 메모리 프로토콜을 구현하는 컨트롤러 백엔드 로직의 면적과 전력을 추정한다.
 * Cadence ChipEstimator 실측 데이터를 기반으로 한 경험적 모델이며,
 * CACTI를 통해 내부 SRAM 버퍼의 면적/전력을 계산한다.
 * mc_type이 MC이면 일반 DRAM 컨트롤러, FLASHC이면 플래시 컨트롤러로 동작한다.
 * MemoryController가 이 객체를 transecEngine 멤버로 보유하고 computeEnergy()를 호출한다.
 *
 * 호출 체인:
 *   MemoryController::MemoryController() → [new MCBackend] → MCBackend::compute()
 *   MemoryController::computeEnergy() → [MCBackend::computeEnergy()]
 */
class MCBackend : public Component {
 public:
  InputParameter l_ip;
  /* [한국어] CACTI에 전달하는 입력 파라미터 구조체.
   * 설정자: MCBackend 생성자에서 interface_ip를 복사하여 초기화.
   * 읽는 자: compute() 내부에서 CACTI 호출 시 사용.
   * 값 범위: 공정 노드(F_sz_nm), 전압(Vdd), 포트 수 등 CACTI 요구 항목 전체.
   * 동기화: 단일 스레드에서만 접근하므로 별도 락 불필요. */

  uca_org_t local_result;
  /* [한국어] CACTI가 반환하는 배열 구성(UCA: Uniform Cache Architecture) 결과.
   * 설정자: compute() 내 CACTI 호출 후 채워짐.
   * 읽는 자: area, power 계산 시 local_result.area, local_result.power 참조.
   * 값 범위: 면적(mm²), 동적/누설 전력(W), 접근 시간(ns) 등.
   * 동기화: 불필요 (단일 스레드). */

  enum MemoryCtrl_type mc_type;
  /* [한국어] 메모리 컨트롤러 종류 — MC(일반 DRAM 컨트롤러) 또는 FLASHC(플래시).
   * 설정자: MCBackend 생성자 인자 mc_type_로 초기화.
   * 읽는 자: compute()/computeEnergy()에서 타입별 분기 결정.
   * 값 범위: basic_components.h의 enum MemoryCtrl_type {MC, FLASHC}.
   * 동기화: 불필요. */

  MCParam mcp;
  /* [한국어] 메모리 컨트롤러 파라미터 묶음 (클럭 주파수, 버스 폭, 채널 수 등).
   * 설정자: MemoryController::set_mc_param()에서 XML을 파싱하여 채운 뒤
   *         MCBackend 생성자 인자로 전달됨.
   * 읽는 자: compute()에서 mcp.clockRate, mcp.dataBusWidth 등 참조.
   * 값 범위: basic_components.h의 MCParam 구조체 정의 참조.
   * 동기화: 불필요. */

  statsDef tdp_stats;
  /* [한국어] TDP(열설계전력) 계산을 위한 활동 통계 (피크 접근률 가정).
   * 설정자: computeEnergy(is_tdp=true) 내에서 피크 duty_cycle로 설정.
   * 읽는 자: power 필드 계산 시 참조.
   * 값 범위: readAc.access, writeAc.access 등 — 피크 시 포트 수 × duty_cycle.
   * 동기화: 불필요. */

  statsDef rtp_stats;
  /* [한국어] 런타임 실제 접근 통계 (XML의 시뮬레이션 결과에서 복사).
   * 설정자: computeEnergy(is_tdp=false) 내에서 XML->sys.mc.* 값으로 채워짐.
   * 읽는 자: rt_power 계산 시 참조.
   * 값 범위: 실제 시뮬레이션 중 관측된 읽기/쓰기 접근 횟수.
   * 동기화: 불필요. */

  statsDef stats_t;
  /* [한국어] tdp_stats와 rtp_stats 중 현재 계산 모드에 따라 선택된 통계.
   * 설정자: computeEnergy() 내에서 is_tdp 플래그에 따라 tdp_stats 또는 rtp_stats를 복사.
   * 읽는 자: 전력 계산 식에서 직접 stats_t.readAc.access 등을 참조.
   * 값 범위: tdp 또는 rtp 통계와 동일.
   * 동기화: 불필요. */

  powerDef power_t;
  /* [한국어] 전력 계산의 중간 결과 — 스케일링 전 기준 전력.
   * 설정자: compute() 또는 computeEnergy() 내에서 계산식으로 채워짐.
   * 읽는 자: computeEnergy()에서 power 또는 rt_power로 복사할 때 사용.
   * 값 범위: readOp.dynamic(J/접근), readOp.leakage(W), gate_leakage(W).
   * 동기화: 불필요. */

  MCBackend(InputParameter *interface_ip_, const MCParam &mcp_,
            enum MemoryCtrl_type mc_type_);
  /* [한국어] MCBackend 생성자 — CACTI를 호출해 백엔드 면적·전력을 초기화한다. */

  void compute();
  /* [한국어] compute - CACTI를 통해 백엔드 SRAM 버퍼 면적·전력을 계산한다. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] computeEnergy - TDP 또는 런타임 모드로 전력을 계산한다.
   * is_tdp=true: power 필드에 피크 전력을 계산.
   * is_tdp=false: rt_power 필드에 실제 시뮬레이션 접근 수 기반 전력을 계산. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] displayEnergy - 계산된 전력·면적 결과를 stdout에 출력한다. */

  ~MCBackend(){};
};

/*
 * [한국어]
 * MCPHY - 메모리 컨트롤러 PHY(물리 계층, SerDes) 전력·면적 모델
 *
 * GDDR5/DDR3 인터페이스의 물리 계층(SerDes: Serializer/Deserializer,
 * I/O 버퍼, PLL 등)을 모델링한다. ISSCC 2006/2007 논문의 90nm CMOS
 * 트랜시버 측정 데이터를 기반으로 약 100mW/채널 @9.6Gb/s를 기준값으로 사용한다.
 * mcp.withPHY 플래그가 true일 때만 MemoryController가 이 객체를 생성하고
 * 전력에 포함시킨다. GPU에서 GDDR5의 고속 I/O 전력 추정에 핵심적이다.
 *
 * 호출 체인:
 *   MemoryController::MemoryController() → [new MCPHY (if withPHY)] → MCPHY::compute()
 *   MemoryController::computeEnergy() → [PHY->computeEnergy()]
 */
class MCPHY : public Component {
 public:
  InputParameter l_ip;
  /* [한국어] CACTI에 전달하는 입력 파라미터.
   * 설정자: MCPHY 생성자에서 interface_ip_를 복사하여 초기화.
   * 읽는 자: compute() 내부에서 공정 노드(F_sz_nm), 전압(Vdd) 스케일링에 사용.
   * 값 범위: InputParameter의 전체 필드 (공정, 온도, 포트 수 등).
   * 동기화: 불필요. */

  uca_org_t local_result;
  /* [한국어] CACTI I/O 모델 계산 결과.
   * 설정자: compute()에서 init_interface() 호출 후 채워짐.
   * 읽는 자: area 및 power 합산 시 참조.
   * 값 범위: 면적, 접근 시간, 동적/정적 전력.
   * 동기화: 불필요. */

  enum MemoryCtrl_type mc_type;
  /* [한국어] MC 또는 FLASHC — PHY 모델 분기 결정.
   * 설정자: MCPHY 생성자 인자 mc_type_로 설정.
   * 읽는 자: compute()에서 타입별 경험적 계수 선택.
   * 값 범위: enum MemoryCtrl_type 참조.
   * 동기화: 불필요. */

  MCParam mcp;
  /* [한국어] 메모리 컨트롤러 파라미터 (클럭, 버스 폭, 채널 수, withPHY 플래그 등).
   * 설정자: MCPHY 생성자 인자로 전달된 mcp_에서 복사.
   * 읽는 자: compute()에서 mcp.clockRate, mcp.num_channels 등 참조.
   * 값 범위: MCParam 구조체 전체 (basic_components.h 참조).
   * 동기화: 불필요. */

  statsDef tdp_stats;
  /* [한국어] PHY TDP 통계 — I/O 활성률 기반 피크 접근 가정값.
   * 설정자: computeEnergy(is_tdp=true)에서 duty_cycle 기반으로 설정.
   * 읽는 자: power 계산 시.
   * 값 범위: 채널 수 × duty_cycle 이내.
   * 동기화: 불필요. */

  statsDef rtp_stats;
  /* [한국어] PHY 런타임 통계 — 실제 I/O 전송 횟수 (XML->sys.mc.*에서 파생).
   * 설정자: computeEnergy(is_tdp=false)에서 채워짐.
   * 읽는 자: rt_power 계산 시.
   * 값 범위: 실제 채널 이용률 범위.
   * 동기화: 불필요. */

  statsDef stats_t;
  /* [한국어] 현재 모드(TDP/RTP)에 따라 선택된 통계.
   * 설정자: computeEnergy()에서 is_tdp 분기 후 복사.
   * 읽는 자: 전력 계산식에서 직접 참조.
   * 값 범위: tdp_stats 또는 rtp_stats와 동일.
   * 동기화: 불필요. */

  powerDef power_t;
  /* [한국어] PHY 계층 전력 중간 결과 (스케일링 전).
   * 설정자: compute()에서 ISSCC 경험적 모델로 계산.
   * 읽는 자: computeEnergy()에서 power/rt_power로 복사.
   * 값 범위: 채널당 ~100mW @90nm, 9.6Gb/s 기준값에서 공정/전압 스케일링.
   * 동기화: 불필요. */

  MCPHY(InputParameter *interface_ip_, const MCParam &mcp_,
        enum MemoryCtrl_type mc_type_);
  /* [한국어] MCPHY 생성자 — I/O 버퍼/SerDes 면적과 전력을 경험적으로 계산한다. */

  void compute();
  /* [한국어] compute - ISSCC 데이터 기반 PHY 면적·전력을 계산한다.
   * 공정 노드(F_sz_nm)와 공급 전압(Vdd)으로 스케일링한다. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] computeEnergy - PHY TDP 또는 런타임 전력을 설정한다. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] displayEnergy - PHY 전력·면적 결과를 stdout에 출력한다. */

  ~MCPHY(){};
};

/*
 * [한국어]
 * MCFrontEnd - 메모리 컨트롤러 프론트엔드 큐 로직 + GPU 코얼레싱 전력 모델
 *
 * 메모리 컨트롤러의 프론트엔드는 CPU/GPU 코어에서 오는 메모리 트랜잭션을
 * 수신하고 스케줄링 큐에 넣는 역할을 한다. GPU 워크로드의 특성 상
 * 동일 warp의 여러 스레드가 같은 캐시 라인에 접근하는 코얼레싱(coalescing)이
 * 빈번하므로, AccelWattch는 PRT(Pending Request Table)와 threadMasks를
 * GPU 고유 확장으로 추가하였다. coalesce_scale은 코얼레싱 전력 기여 비율을
 * 전체 전력에 반영하는 스케일링 인수이다.
 *
 * 호출 체인:
 *   MemoryController::MemoryController() → [new MCFrontEnd] → ArrayST 생성 5~7개
 *   MemoryController::computeEnergy() → [frontend->computeEnergy()]
 */
class MCFrontEnd : public Component {
 public:
  ParseXML *XML;
  /* [한국어] GPU 설정·통계 XML 최상위 파서 포인터.
   * 설정자: MCFrontEnd 생성자 인자 XML_interface로 초기화.
   * 읽는 자: computeEnergy()에서 XML->sys.mc.* 통계를 읽어 활동 계수 계산.
   * 값 범위: 유효한 ParseXML 포인터 (NULL 불가).
   * 동기화: 읽기 전용 접근이므로 별도 락 불필요. */

  InputParameter interface_ip;
  /* [한국어] CACTI 입력 파라미터 — 공정 노드, 전압, 포트 구성 등.
   * 설정자: 생성자에서 interface_ip_를 역참조하여 복사.
   * 읽는 자: ArrayST 생성 시 각 큐 SRAM의 CACTI 파라미터로 전달.
   * 값 범위: F_sz_nm, Vdd, num_rw_ports 등 전체.
   * 동기화: 불필요. */

  enum MemoryCtrl_type mc_type;
  /* [한국어] 컨트롤러 종류 (MC 또는 FLASHC).
   * 설정자: 생성자 인자 mc_type_로 초기화.
   * 읽는 자: computeEnergy()에서 MC/FLASHC별 전력 계산 분기.
   * 값 범위: enum MemoryCtrl_type {MC, FLASHC}.
   * 동기화: 불필요. */

  MCParam mcp;
  /* [한국어] MC 파라미터 구조체 (클럭, 버스 폭, 채널 수, 읽기/쓰기 비율 등).
   * 설정자: 생성자 인자 mcp_로 복사 초기화.
   * 읽는 자: computeEnergy()에서 mcp.clockRate, mcp.reads, mcp.writes 참조.
   * 값 범위: MCParam 전체 필드 (basic_components.h 참조).
   * 동기화: 불필요. */

  selection_logic *MC_arb;
  /* [한국어] 읽기/쓰기 큐 간 중재(arbitration) 로직 전력 모델.
   * 설정자: 생성자에서 new selection_logic(...)으로 생성.
   * 읽는 자: computeEnergy()에서 중재 전력을 합산.
   * 값 범위: 유효한 포인터 — 생성자에서 반드시 초기화됨.
   * 동기화: 불필요 (단일 스레드 접근). */

  ArrayST *frontendBuffer;
  /* [한국어] 프론트엔드 트랜잭션 큐 SRAM 모델 (incoming request queue).
   * MC가 수신하는 메모리 트랜잭션을 일시적으로 저장하는 큐의 SRAM 전력을 추정한다.
   * 설정자: 생성자에서 ArrayST(&interface_ip, ...) 로 생성.
   * 읽는 자: computeEnergy()에서 frontendBuffer->local_result.power 참조.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  ArrayST *readBuffer;
  /* [한국어] 읽기 데이터 반환 버퍼 SRAM 모델.
   * DRAM에서 읽어온 데이터를 코어로 반환하기 전 임시 저장하는 버퍼의 전력 추정.
   * 설정자: 생성자에서 ArrayST로 생성.
   * 읽는 자: computeEnergy()에서 readBuffer->local_result.power 합산.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  ArrayST *writeBuffer;
  /* [한국어] 쓰기 데이터 버퍼 SRAM 모델.
   * write-combining 또는 write-back 데이터를 DRAM 전송 전 보관하는 버퍼.
   * 설정자: 생성자에서 ArrayST로 생성.
   * 읽는 자: computeEnergy()에서 writeBuffer->local_result.power 합산.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  ArrayST *PRT;
  /* [한국어] PRT(Pending Request Table) — GPU 코얼레싱용 미결 요청 추적 테이블.
   * GPU에서는 같은 warp 내 여러 스레드의 메모리 요청이 동일 캐시 라인으로
   * 병합(coalesce)될 수 있다. PRT는 각 warp의 미결 메모리 요청 상태를 추적하여
   * 코얼레싱 판단 근거를 제공하는 CAM/SRAM 구조이다.
   * 설정자: 생성자에서 ArrayST로 생성 (GPU mc_type일 때만 유효).
   * 읽는 자: computeEnergy()에서 PRT 전력 × coalesce_scale로 합산.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  ArrayST *threadMasks;
  /* [한국어] 스레드 액티브 마스크 추적 SRAM — warp별 활성 스레드 비트마스크 저장.
   * 코얼레싱 결정 시 어느 스레드가 현재 요청에 참여하는지 비트마스크로 추적한다.
   * warp 크기(32비트)에 해당하는 비트마스크를 PRT와 연계하여 저장한다.
   * 설정자: 생성자에서 ArrayST로 생성.
   * 읽는 자: computeEnergy()에서 threadMasks 전력 × coalesce_scale로 합산.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  ArrayST *PRC;
  /* [한국어] PRC(Pending Request Controller) — 코얼레싱 FSM 상태 저장 SRAM.
   * PRT/threadMasks를 관리하는 컨트롤러 로직의 상태 레지스터 전력을 모델링한다.
   * 설정자: 생성자에서 ArrayST로 생성.
   * 읽는 자: computeEnergy()에서 PRC 전력 × coalesce_scale로 합산.
   * 값 범위: 유효한 ArrayST 포인터.
   * 동기화: 불필요. */

  double coalesce_scale;
  /* [한국어] GPU 코얼레싱 관련 전력(PRT+threadMasks+PRC)의 전체 대비 기여 비율.
   * 설정자: 생성자에서 XML->sys.mc.* 파라미터 및 GPU warp 크기로 계산.
   * 읽는 자: computeEnergy()에서 PRT/threadMasks/PRC 전력에 곱함.
   * 값 범위: 0.0 ~ 1.0 (기여 비율).
   * 동기화: 불필요. */

  MCFrontEnd(ParseXML *XML_interface, InputParameter *interface_ip_,
             const MCParam &mcp_, enum MemoryCtrl_type mc_type_);
  /* [한국어] MCFrontEnd 생성자 — 큐 SRAM, PRT, threadMasks, PRC, MC_arb를 생성하고 초기화한다. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] computeEnergy - 프론트엔드 버퍼, 중재기, 코얼레싱 로직 전력을 합산한다.
   * is_tdp=true: 피크 duty_cycle 기반 TDP 전력 계산.
   * is_tdp=false: 실제 읽기/쓰기 통계 기반 런타임 전력 계산. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] displayEnergy - 프론트엔드 전력·면적을 stdout에 출력한다. */

  ~MCFrontEnd();
  /* [한국어] 소멸자 — new로 생성한 ArrayST 및 selection_logic 포인터를 해제한다. */
};

/*
 * [한국어]
 * DRAM - GDDR5/GDDR3 DRAM 디바이스 전력 모델 (IDD 전류 기반)
 *
 * 실제 GDDR5/GDDR3 메모리 칩의 소비 전력을 데이터시트의 IDD 전류 파라미터
 * (idd0~idd7: active precharge, active standby, precharge standby 등)로 모델링한다.
 * rd_coeff(읽기당 에너지 계수)와 wr_coeff(쓰기당 에너지 계수)를 계산하여
 * gpgpu_sim_wrapper.cc가 시뮬레이션 중 접근 횟수에 곱해 런타임 DRAM 전력을 산출한다.
 * MemoryController가 이 객체를 dram 멤버로 소유한다.
 *
 * 호출 체인:
 *   MemoryController::MemoryController() → [new DRAM] → DRAM::set_dram_param()
 *   MemoryController::computeEnergy() → [dram->computeEnergy()]
 *   gpgpu_sim_wrapper::computePower() → dram->rd_coeff, dram->wr_coeff 참조
 */
class DRAM : public Component {
 public:
  ParseXML *XML;
  /* [한국어] GPU 설정·통계 XML 파서 포인터.
   * 설정자: DRAM 생성자 인자 XML_interface로 초기화.
   * 읽는 자: set_dram_param()에서 XML->sys.mc.DRAM_* 파라미터 읽기.
   * 값 범위: 유효한 ParseXML 포인터.
   * 동기화: 읽기 전용, 불필요. */

  InputParameter interface_ip;
  /* [한국어] 공정 노드·전압 정보를 담는 CACTI 입력 파라미터.
   * 설정자: DRAM 생성자에서 interface_ip_를 역참조하여 복사.
   * 읽는 자: computeEnergy()에서 전압(Vdd) 스케일링에 사용.
   * 값 범위: F_sz_nm, Vdd 등.
   * 동기화: 불필요. */

  enum Dram_type dram_type;
  /* [한국어] DRAM 종류 — GDDR5, GDDR3, LPDDR2 등을 구분하는 열거형.
   * 설정자: DRAM 생성자 인자 dram_type_로 초기화.
   * 읽는 자: set_dram_param()에서 타입별 IDD 계수 로드, computeEnergy()에서 전력 식 분기.
   * 값 범위: basic_components.h의 enum Dram_type 참조.
   * 동기화: 불필요. */

  DRAMParam dramp;
  /* [한국어] DRAM 파라미터 묶음 (idd0~idd7 전류값, VDD, 버스 주파수, 채널 수 등).
   * 설정자: set_dram_param()에서 XML 및 dram_type 기반으로 채워짐.
   * 읽는 자: computeEnergy()에서 IDD 기반 전력 계산식에 사용.
   * 값 범위: DRAMParam 구조체 (basic_components.h 참조); idd 단위는 mA.
   * 동기화: 불필요. */

  powerDef power_t;
  /* [한국어] DRAM 전력 중간 결과 (rd_coeff, wr_coeff 포함).
   * 설정자: computeEnergy()에서 IDD 기반 전력 계산 후 채워짐.
   * 읽는 자: MemoryController::computeEnergy()에서 합산,
   *          gpgpu_sim_wrapper가 rd_coeff, wr_coeff를 직접 참조.
   * 값 범위: 읽기당 동적 에너지(J), 누설 전력(W).
   * 동기화: 불필요. */

  DRAM(ParseXML *XML_interface, InputParameter *interface_ip_,
       enum Dram_type dram_type_);
  /* [한국어] DRAM 생성자 — set_dram_param()을 호출하고 초기 IDD 전력을 계산한다. */

  void set_dram_param();
  /* [한국어] set_dram_param - XML에서 DRAM 종류별 IDD 전류 파라미터를 로드한다.
   * GDDR5이면 GDDR5 데이터시트 IDD 기본값, GDDR3이면 GDDR3 기본값 등을 설정한다. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] computeEnergy - IDD 전류와 VDD 전압으로 DRAM 디바이스 전력을 계산한다.
   * rd_coeff = 읽기 1회당 에너지(J), wr_coeff = 쓰기 1회당 에너지(J). */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] displayEnergy - DRAM 전력·면적을 stdout에 출력한다. */

  ~DRAM();
  /* [한국어] 소멸자 — DRAM 관련 동적 할당 자원을 해제한다. */
};

/*
 * [한국어]
 * MemoryController - GPU 메모리 컨트롤러 전체를 묶는 최상위 래퍼 클래스
 *
 * MCFrontEnd, MCBackend, MCPHY, DRAM, Pipeline 다섯 서브컴포넌트를 생성하고
 * 전력·면적을 합산하는 최상위 컴포넌트이다. Processor 클래스가 GPU 설정의
 * mc 섹션(XML->sys.mc.*)을 파싱하여 MemoryController를 인스턴스화한다.
 * set_mc_param()이 XML 파싱을 담당하고, computeEnergy()가 각 서브컴포넌트에
 * 위임하여 전체 MC 전력을 집계한다. GPU의 총 전력 보고(displayEnergy)에
 * "Memory Controller" 섹션으로 포함된다.
 *
 * 호출 체인:
 *   Processor::Processor() → [new MemoryController] → set_mc_param() → 서브컴포넌트 생성
 *   Processor::computePower() → [MemoryController::computeEnergy()] → 각 서브컴포넌트 위임
 *   gpgpu_sim_wrapper::computePower() → MemoryController::computeEnergy() 직접 호출 가능
 */
class MemoryController : public Component {
 public:
  ParseXML *XML;
  /* [한국어] GPU 설정·통계 XML 파서 포인터.
   * 설정자: 생성자 인자 XML_interface로 초기화.
   * 읽는 자: set_mc_param()에서 XML->sys.mc.*, XML->sys.L2[*] 등 파싱.
   * 값 범위: 유효한 ParseXML 포인터.
   * 동기화: 읽기 전용, 불필요. */

  InputParameter interface_ip;
  /* [한국어] CACTI 공통 입력 파라미터 (공정, 전압, 온도 등).
   * 설정자: 생성자에서 interface_ip_를 역참조하여 복사.
   * 읽는 자: 서브컴포넌트 생성 시 전달.
   * 값 범위: InputParameter 구조체 전체.
   * 동기화: 불필요. */

  enum MemoryCtrl_type mc_type;
  /* [한국어] MC 또는 FLASHC — 전체 컨트롤러 종류.
   * 설정자: 생성자 인자 mc_type_로 초기화.
   * 읽는 자: 서브컴포넌트 생성 시 전달, set_mc_param()에서 분기.
   * 값 범위: enum MemoryCtrl_type {MC, FLASHC}.
   * 동기화: 불필요. */

  MCParam mcp;
  /* [한국어] 메모리 컨트롤러 파라미터 전체 (클럭, 버스 폭, 채널 수, 읽기/쓰기 비율 등).
   * 설정자: set_mc_param()에서 XML->sys.mc.* 파싱 후 채워짐.
   * 읽는 자: 각 서브컴포넌트 생성자에 const 참조로 전달.
   * 값 범위: MCParam 구조체 전체 (basic_components.h 참조).
   * 동기화: 불필요. */

  DRAM *dram;
  /* [한국어] GDDR5/GDDR3 DRAM 디바이스 전력 모델 포인터.
   * 설정자: 생성자에서 new DRAM(...)으로 생성.
   * 읽는 자: computeEnergy()에서 dram->computeEnergy()를 호출하고 전력 합산.
   * 값 범위: 유효한 DRAM 포인터 (NULL 불가).
   * 동기화: 불필요. */

  MCFrontEnd *frontend;
  /* [한국어] 프론트엔드 큐·코얼레싱 로직 전력 모델 포인터.
   * 설정자: 생성자에서 new MCFrontEnd(...)으로 생성.
   * 읽는 자: computeEnergy()에서 frontend->computeEnergy() 호출 후 전력 합산.
   * 값 범위: 유효한 MCFrontEnd 포인터 (NULL 불가).
   * 동기화: 불필요. */

  MCBackend *transecEngine;
  /* [한국어] DDR 백엔드 트랜잭션 엔진 전력 모델 포인터.
   * "transec"는 transaction engine의 약어.
   * 설정자: 생성자에서 new MCBackend(...)으로 생성.
   * 읽는 자: computeEnergy()에서 transecEngine->computeEnergy() 호출 후 합산.
   * 값 범위: 유효한 MCBackend 포인터 (NULL 불가).
   * 동기화: 불필요. */

  MCPHY *PHY;
  /* [한국어] PHY(물리 계층) 전력 모델 포인터 — mcp.withPHY가 true일 때만 생성.
   * 설정자: 생성자에서 mcp.withPHY가 true일 경우 new MCPHY(...)으로 생성, 아니면 nullptr.
   * 읽는 자: computeEnergy()에서 PHY != nullptr이면 PHY->computeEnergy() 호출 후 합산.
   * 값 범위: 유효한 MCPHY 포인터 또는 nullptr.
   * 동기화: 불필요. */

  Pipeline *pipeLogic;
  /* [한국어] MC 내부 파이프라인 래치 전력 모델 포인터.
   * MC가 여러 파이프라인 스테이지를 가질 경우 래치 전력을 추정한다.
   * 설정자: 생성자에서 new Pipeline(...)으로 생성 (파이프라인 스테이지 수 기반).
   * 읽는 자: computeEnergy()에서 pipeLogic->power 합산.
   * 값 범위: 유효한 Pipeline 포인터.
   * 동기화: 불필요. */

  // Add coalescing logic related modules with each memory controller --Syed
  // Gilani

  // clock_network clockNetwork;
  MemoryController(ParseXML *XML_interface, InputParameter *interface_ip_,
                   enum MemoryCtrl_type mc_type_, enum Dram_type dram_type_);
  /* [한국어] MemoryController 생성자 — set_mc_param()을 호출한 뒤
   * MCFrontEnd, MCBackend, MCPHY, DRAM, Pipeline을 순서대로 생성하고 면적을 합산한다. */

  void set_mc_param();
  /* [한국어] set_mc_param - XML->sys.mc.* 필드를 파싱하여 mcp 구조체를 채운다.
   * mc_type에 따라 DRAM 채널 수, 클럭, 버스 폭, IDD 전류 등을 로드한다. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] computeEnergy - 모든 서브컴포넌트 전력을 합산하여 전체 MC 전력을 계산한다.
   * is_tdp=true: TDP 모드 (피크 전력), is_tdp=false: 런타임 모드 (실제 접근 기반). */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] displayEnergy - 서브컴포넌트별 전력과 면적을 들여쓰기와 함께 stdout에 출력한다. */

  ~MemoryController();
  /* [한국어] 소멸자 — dram, frontend, transecEngine, PHY, pipeLogic 포인터를 delete한다. */
};
#endif /* MEMORYCTRL_H_ */
