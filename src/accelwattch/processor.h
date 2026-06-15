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
 * [한국어 설명] AccelWattch 전력 모델 최상위 프로세서 클래스 (processor.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 AccelWattch(GPGPU-Sim의 GPU 전력 모델, MICRO 2021)에서 가장
 * 상위 수준의 전력 계산 객체인 `Processor` 클래스를 선언한다. `Processor`는
 * GPU 전체를 구성하는 모든 하드웨어 서브컴포넌트(SM 코어, L2 캐시, NoC,
 * 메모리 컨트롤러, NIU, PCIe, Flash)를 집계하고, 각 컴포넌트별 전력 계수를
 * 계산하는 인터페이스를 제공한다. GPGPU-Sim의 cycle() 루프가 매 시뮬레이션
 * 구간마다 이 클래스의 계수 메서드를 호출하여 실제 동적 전력을 산출한다.
 * 원본 McPAT CPU 전력 모델을 GPU 아키텍처(GDDR5 DRAM, NoC, 코얼레싱 등)에
 * 맞게 확장한 파일이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층도:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *       → accelwattch_interface::update() [accelwattch_interface.cc]
 *           → Processor::compute()          ← [이 파일의 핵심 진입점]
 *               ├── Core::compute()          (SM 1개의 전력 계산)
 *               ├── SharedCache::compute()   (L2 캐시 전력 계산)
 *               ├── NoC::compute()           (NoC 라우터 전력 계산)
 *               └── MemoryController::compute() (DRAM 컨트롤러 전력)
 *           → Processor::get_coefficient_*() ← [이 파일의 계수 메서드들]
 *               → accelwattch_interface가 GPGPU-Sim 이벤트 카운터와 곱하여
 *                  최종 전력(Watt) 산출
 * 실행 컨텍스트: 호스트 CPU 유저스페이스 (시뮬레이터 메인 스레드).
 * GPU 디바이스 코드와 무관하며, 시뮬레이션 후처리 단계에서 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈 (이 파일이 include/사용하는 것):
 *   - core.h (Core*): SM 하나의 실행 유닛/레지스터 파일 전력 모델
 *   - sharedcache.h (SharedCache*): L2/L3 캐시 어레이 전력 모델
 *   - noc.h (NoC*): GPU 내부 NoC 라우터(버퍼/크로스바/아비터) 전력 모델
 *   - memoryctrl.h (MemoryController*): GDDR5 프론트엔드/PHY/DRAM 전력 모델
 *   - iocontrollers.h (NIU/PCIe/Flash): 입출력 컨트롤러 전력 모델
 *   - XML_Parse.h (ParseXML*): gpgpusim.config에서 파싱된 XML 설정
 *   - basic_components.h (Component, ProcParam): McPAT 기본 전력 컴포넌트
 *   - cacti/router.h: CACTI 기반 라우터 전력 파라미터
 *   - ../gpgpu-sim/visualizer.h: gzip 압축 시각화 파일 출력용
 * 데이터 흐름:
 *   ParseXML → Processor 생성자 → 각 서브컴포넌트 초기화 및 compute()
 *   → get_coefficient_*() 반환값 → accelwattch_interface가 이벤트 수와 곱셈
 *   → 최종 동적 전력(Joule/cycle or Watt)을 GPGPU-Sim visualizer에 기록
 *
 * === 주요 함수/구조체 요약 ===
 * - Processor(ParseXML*): 생성자. XML 설정을 읽어 모든 서브컴포넌트를 초기화
 *   하고 McPAT 전력 파라미터 테이블을 설정한다.
 * - compute(): 모든 서브컴포넌트의 전력을 한꺼번에 계산하는 메인 메서드.
 *   TDP(최대 설계 전력)와 동적 전력을 모두 산출한다.
 * - get_const_dynamic_power(): 항상 소비되는 상수 동적 전력 반환. DRAM
 *   컨트롤러 10% 기저 부하 + 실행 유닛 기저 에너지로 구성된다.
 * - get_coefficient_readcoalescing() / get_coefficient_writecoalescing():
 *   메모리 읽기/쓰기 코얼레싱 1회당 소비되는 에너지 계수(Joule)를 반환한다.
 *   PRT, threadMasks, PRC SRAM 접근 에너지와 CV² 코얼레싱 에너지의 합산이다.
 * - get_coefficient_noc_accesses(): NoC 라우터 접근 1회당 에너지 계수 반환.
 *   버퍼 읽기·쓰기 + 크로스바 + 아비터 에너지의 합이다.
 * - get_coefficient_l2_read_hits/misses/write_hits/misses(): L2 캐시 접근
 *   유형별(읽기 히트/미스, 쓰기 히트/미스) 에너지 계수를 반환한다. 쓰기
 *   미스의 경우 MSHR 버퍼(missb, ifb, prefetchb, wbb)까지 모두 합산한다.
 * - get_coefficient_mem_reads() / get_coefficient_mem_writes(): DRAM 읽기/
 *   쓰기 트랜잭션 1회당 전체 메모리 계층(프론트엔드 버퍼 + DRAM 칩 + PHY)
 *   에너지 계수를 반환한다.
 * - get_coefficient_mem_pre(): DRAM precharge(프리차지) 1회당 에너지 계수.
 * - nonlinear_scale() / coefficient_scale() / iterative_lse(): 비선형 전력
 *   모델 피팅 메서드. 실측 전력과 모델 전력의 오차를 최소제곱법으로 보정한다.
 * - visualizer_print(): GPGPU-Sim gzip 시각화 파일에 전력 데이터를 기록한다.
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include <vector>
#include "../gpgpu-sim/visualizer.h" // [한국어] GPGPU-Sim gzip 시각화 파일 출력용 (gzFile 타입 정의)
#include "XML_Parse.h"               // [한국어] gpgpusim.config → ParseXML 구조체 (전력 파라미터 공급원)
#include "array.h"                   // [한국어] McPAT SRAM 어레이 전력 계산 유틸리티
#include "basic_components.h"        // [한국어] Component, ProcParam 등 McPAT 기본 빌딩블록
#include "cacti/arbiter.h"           // [한국어] CACTI 아비터(중재기) 전력 파라미터
#include "cacti/area.h"              // [한국어] CACTI 면적 모델 (면적→전력 연관)
#include "cacti/decoder.h"           // [한국어] CACTI 디코더 전력 파라미터
#include "cacti/parameter.h"         // [한국어] CACTI 글로벌 파라미터 (g_tp.peri_global.Vdd 등)
#include "cacti/router.h"            // [한국어] CACTI 라우터(NoC 버퍼/크로스바) 전력 파라미터
#include "core.h"                    // [한국어] Core 클래스 — SM 하나의 실행 유닛/레지스터 파일 전력 모델
#include "iocontrollers.h"           // [한국어] NIUController, PCIeController, FlashController 선언
#include "memoryctrl.h"              // [한국어] MemoryController 클래스 — GDDR5 프론트엔드/PHY/DRAM 전력
#include "noc.h"                     // [한국어] NoC 클래스 — GPU 내부 네트워크온칩 라우터 전력 모델
#include "sharedcache.h"             // [한국어] SharedCache 클래스 — L2/L3 캐시 어레이 전력 모델

/*
 * [한국어] Processor — AccelWattch 전력 모델 최상위 집계 클래스
 *
 * GPU 전체 하드웨어를 구성하는 모든 서브컴포넌트의 전력 모델 객체를 보유하고,
 * 이를 집계하여 시스템 전체 전력을 산출한다. McPAT의 CPU 전력 모델을 GPU에
 * 맞게 확장한 구조체이며, GPGPU-Sim의 accelwattch_interface가 이 클래스의
 * 메서드를 호출하여 시뮬레이션 이벤트 카운터와 에너지 계수를 곱한 뒤 최종
 * 동적 전력을 산출한다.
 *
 * 상속 관계: Processor → Component (McPAT 기본 컴포넌트, 전력·면적 필드 보유)
 */
class Processor : public Component {
 public:
  ParseXML *XML;
  /* [한국어] gpgpusim.config에서 파싱된 XML 설정 객체 포인터.
   * 설정자: Processor(ParseXML*) 생성자가 인자로 받아 이 필드에 저장한다.
   * 읽는 자: set_proc_param(), compute(), 각 서브컴포넌트 생성자가 XML에서
   *   코어 수, 캐시 크기, 클럭 레이트, DRAM 파라미터 등을 읽는다.
   * 값 범위: NULL이 아닌 유효한 ParseXML 포인터여야 하며, 시뮬레이터 기동
   *   시점에 이미 완전히 파싱된 상태로 전달된다.
   * 동기화: 시뮬레이터 초기화 시 단 한 번 설정되며, 이후 읽기 전용으로
   *   사용된다. 멀티스레드 안전성은 보장되나 쓰기 접근은 없다. */

  vector<Core *> cores;
  /* [한국어] SM(Streaming Multiprocessor) 전력 모델 객체 벡터.
   * 설정자: Processor 생성자에서 procdynp.numCore 개수만큼 Core 객체를 생성하여
   *   push_back한다. GPU 시뮬레이션에서는 모든 SM을 동질적(homogeneous)으로
   *   가정하므로 실제로는 index 0 하나만 생성되는 경우가 많다.
   * 읽는 자: get_const_dynamic_power()가 cores[0]->exu->exeu 등의 기저 에너지를
   *   읽고, compute()가 전체 SM 동적 전력 합산에 cores[0] 값을 스케일링한다.
   * 값 범위: 최소 1개 이상. procdynp.numCore(= XML에서 읽은 SM 수)와 동일.
   *   GPU 모델에서 index 0이 대표 SM 역할을 한다.
   * 동기화: 생성 이후 읽기 전용. 소멸자에서 각 포인터를 delete한다. */

  vector<SharedCache *> l2array;
  /* [한국어] L2 캐시 어레이 전력 모델 객체 벡터.
   * 설정자: Processor 생성자에서 XML->sys.number_of_L2s 개수만큼 SharedCache
   *   객체를 생성한다. GPU에서는 단일 통합 L2가 존재하므로 l2array[0]이 유일한
   *   L2 전력 모델이다. CPU 모델과의 코드 공유를 위해 벡터로 선언되어 있다.
   * 읽는 자: get_coefficient_l2_read_hits/misses/write_hits/misses()가
   *   l2array[0]->unicache.caches→local_result.power.*를 읽어 L2 접근 계수를
   *   반환한다. compute()가 전체 L2 동적 전력 집계에 사용한다.
   * 값 범위: GPU에서는 항상 크기 1(l2array[0]만 유효). XML 설정에서
   *   number_of_L2s == 0이면 계수 메서드들은 0을 반환한다.
   * 동기화: 생성 이후 읽기 전용. 소멸자에서 delete한다. */

  vector<SharedCache *> l3array;
  /* [한국어] L3 캐시 어레이 전력 모델 객체 벡터.
   * 설정자: Processor 생성자에서 XML->sys.number_of_L3s 개수만큼 생성.
   * 읽는 자: compute()에서 L3 동적 전력 집계 시 사용. GPU 시뮬레이션에서는
   *   일반적으로 L3가 없으므로 크기 0인 빈 벡터로 남는다.
   * 값 범위: GPU 구성에서는 대개 0(비어 있음). CPU 전력 모델 재사용 목적.
   * 동기화: 생성 이후 읽기 전용. */

  vector<SharedCache *> l1dirarray;
  /* [한국어] L1 디렉터리 캐시 전력 모델 벡터 (CPU 디렉터리 캐시 코히런스용).
   * 설정자: Processor 생성자에서 XML->sys.number_of_L1Directories 개수만큼 생성.
   * 읽는 자: compute()에서 L1dir 동적 전력 집계. GPU 모델에서는 사용 안 됨.
   * 값 범위: GPU 구성에서는 대개 0(비어 있음). McPAT CPU 코드와의 호환성 유지.
   * 동기화: 생성 이후 읽기 전용. */

  vector<SharedCache *> l2dirarray;
  /* [한국어] L2 디렉터리 캐시 전력 모델 벡터 (CPU 디렉터리 캐시 코히런스용).
   * 설정자: Processor 생성자에서 XML->sys.number_of_L2Directories 개수만큼 생성.
   * 읽는 자: compute()에서 L2dir 동적 전력 집계. GPU 모델에서는 사용 안 됨.
   * 값 범위: GPU 구성에서는 대개 0(비어 있음). McPAT CPU 코드와의 호환성 유지.
   * 동기화: 생성 이후 읽기 전용. */

  vector<NoC *> nocs;
  /* [한국어] GPU 내부 NoC(Network-on-Chip) 라우터 전력 모델 객체 벡터.
   * 설정자: Processor 생성자에서 XML->sys.number_of_NoCs 개수만큼 NoC 객체를
   *   생성한다. GPU에서는 SM-to-L2 인터커넥트가 단일 NoC 레벨로 모델링되므로
   *   nocs[0]이 대표 라우터 전력 모델이다 (intersim2와 대응).
   * 읽는 자: get_coefficient_noc_accesses()가 nocs[0]->router의 버퍼/크로스바/
   *   아비터 동적 에너지를 읽는다. compute()가 NoC 전체 동적 전력 집계에 사용.
   * 값 범위: GPU에서는 항상 크기 1(nocs[0]만 유효). 크기 0이면 NoC 전력 0.
   * 동기화: 생성 이후 읽기 전용. 소멸자에서 delete한다. */

  MemoryController *mc;
  /* [한국어] DRAM 메모리 컨트롤러(GDDR5) 전력 모델 객체 포인터.
   * 설정자: Processor 생성자에서 new MemoryController(XML, &interface_ip)로
   *   생성. GDDR5 메모리 컨트롤러(MC)의 프론트엔드 버퍼, 트랜잭션 엔진,
   *   PHY, DRAM 칩 모델을 모두 포함하는 복합 객체이다.
   * 읽는 자: get_const_dynamic_power()가 mc->frontend/transecEngine/PHY의
   *   기저 전력을 읽고, get_coefficient_mem_reads/writes/pre()가 접근 에너지
   *   계수를 읽으며, get_coefficient_readcoalescing/writecoalescing()이
   *   mc->frontend의 PRT/threadMasks/PRC 에너지를 읽는다.
   * 값 범위: NULL이 아닌 유효한 MemoryController 포인터. XML에서 MC 설정이
   *   누락되어도 기본값으로 생성된다.
   * 동기화: 생성 이후 읽기 전용. 소멸자에서 delete한다. */

  NIUController *niu;
  /* [한국어] NIU(Network Interface Unit) 컨트롤러 전력 모델 포인터.
   * 설정자: Processor 생성자에서 XML->sys.niu.number_units > 0이면 생성.
   * 읽는 자: compute()에서 NIU 동적 전력 집계. GPU GPGPU-Sim에서는 일반적으로
   *   NIU가 없으므로 NULL이거나 전력 기여가 0이다.
   * 값 범위: NULL 가능 (NIU가 없는 GPU 구성). 소멸자에서 NULL 체크 후 delete.
   * 동기화: 생성 이후 읽기 전용. */

  PCIeController *pcie;
  /* [한국어] PCIe 컨트롤러 전력 모델 포인터.
   * 설정자: Processor 생성자에서 XML->sys.pcie.number_units > 0이면 생성.
   * 읽는 자: compute()에서 PCIe 동적 전력 집계. GPU에서는 호스트↔디바이스
   *   데이터 전송 전력에 해당하나, GPGPU-Sim은 메모리 트랜잭션 레벨 모델이므로
   *   주로 설정값에 따라 활성화 여부가 결정된다.
   * 값 범위: NULL 가능. 소멸자에서 NULL 체크 후 delete.
   * 동기화: 생성 이후 읽기 전용. */

  FlashController *flashcontroller;
  /* [한국어] Flash 컨트롤러 전력 모델 포인터.
   * 설정자: Processor 생성자에서 XML->sys.flashc.number_units > 0이면 생성.
   * 읽는 자: compute()에서 Flash 동적 전력 집계. GPU 전력 모델에서는 일반적으로
   *   비활성화 상태이다.
   * 값 범위: NULL 가능. 소멸자에서 NULL 체크 후 delete.
   * 동기화: 생성 이후 읽기 전용. */

  InputParameter interface_ip;
  /* [한국어] McPAT/CACTI 인터페이스 파라미터 구조체.
   * 설정자: set_proc_param()에서 XML 설정으로부터 공정 기술 파라미터
   *   (feature size, Vdd, 클럭 레이트 등)를 채운 후, 각 서브컴포넌트 생성자에
   *   레퍼런스로 전달된다.
   * 읽는 자: Core, SharedCache, NoC, MemoryController 생성자들이 공정 기술
   *   파라미터를 읽어 면적·전력 모델을 초기화한다.
   * 값 범위: CACTI가 지원하는 공정 기술 범위(22nm ~ 180nm). Vdd는 공정에
   *   따라 자동 결정된다.
   * 동기화: 서브컴포넌트 생성 후에는 읽기 전용. 이후 수정 금지. */

  double exClockRate;
  /* [한국어] 실행 유닛(functional unit) 클럭 레이트 (Hz 단위).
   * GPU 컨텍스트에서는 SM 코어 클럭 레이트에 해당한다 (예: 1.5 GHz → 1.5e9).
   * 설정자: set_proc_param()에서 XML->sys.core[0].clock_rate로부터 설정.
   *   GPU의 경우 모든 SM이 동일한 코어 클럭을 사용하므로 index 0 값을 사용.
   * 읽는 자: get_const_dynamic_power()에서 기저 에너지(Joule)를 전력(Watt)으로
   *   변환할 때 이 클럭 레이트를 나누는 데 사용된다. 또한 compute()에서
   *   동적 전력 계산의 분모로 사용된다.
   * 값 범위: 양의 실수 (Hz). 전형적인 GPU에서 수백 MHz ~ 수 GHz.
   * 동기화: 초기화 이후 읽기 전용. 멀티스레드 접근 없음. */

  ProcParam procdynp;
  /* [한국어] 프로세서 동적 파라미터 구조체 (McPAT ProcParam 타입).
   * GPU 구성의 핵심 파라미터(SM 수, 클럭, Vdd 등)를 보유한다.
   * 설정자: set_proc_param()에서 XML 설정을 읽어 numCore(SM 수), clockRate,
   *   Vdd, homogeneous_cores 등을 설정한다.
   * 읽는 자: Processor 생성자와 compute()에서 SM 수(procdynp.numCore)를
   *   반복 횟수로 사용하고, 전력 계산의 스케일링 인수로 활용한다.
   * 값 범위: numCore ≥ 1 (GPU에서는 SM 수, 전형적으로 1~128).
   * 동기화: 초기화 이후 읽기 전용. */

  // for debugging nonlinear model
  double dyn_power_before_scaling;
  /* [한국어] 비선형 스케일링 적용 전 동적 전력값 (디버깅용 임시 보관 필드).
   * 설정자: nonlinear_scale() 또는 compute() 내에서 스케일링을 적용하기 직전에
   *   현재 동적 전력을 이 필드에 저장한다.
   * 읽는 자: 디버깅 출력 코드 또는 visualizer_print()에서 스케일링 전후 비교에
   *   활용될 수 있다. 정상 전력 산출 경로에는 영향을 주지 않는다.
   * 값 범위: 0.0 이상의 실수 (Watt). 초기값 0.0.
   * 동기화: 단일 스레드에서 compute()→nonlinear_scale() 순으로 기록되므로
   *   별도 동기화 불필요. */

  // wire	globalInterconnect;
  // clock_network globalClock;

  Component core, l2, l3, l1dir, l2dir, noc, mcs, cc, nius, pcies,
      flashcontrollers;
  /* [한국어] 각 서브시스템의 집계 전력/면적을 보관하는 Component 객체들.
   * Component는 McPAT의 기본 빌딩블록으로 power(동적/정적), area 필드를 보유.
   * - core: 모든 SM 코어의 집계 전력 (cores 벡터 합산 결과 저장)
   * - l2: L2 캐시 집계 전력 (l2array 벡터 합산)
   * - l3: L3 캐시 집계 전력 (l3array 합산, GPU에서는 0)
   * - l1dir / l2dir: L1/L2 디렉터리 캐시 집계 (GPU에서는 0)
   * - noc: NoC 집계 전력 (nocs 벡터 합산)
   * - mcs: 메모리 컨트롤러 집계 전력 (mc 결과)
   * - cc: 코히런스 컨트롤러 전력 (GPU에서는 보통 0)
   * - nius: NIU 집계 전력
   * - pcies: PCIe 집계 전력
   * - flashcontrollers: Flash 컨트롤러 집계 전력
   * 설정자: compute()에서 각 서브컴포넌트의 전력을 합산하여 이 필드들에 기록.
   * 읽는 자: displayEnergy()와 visualizer_print()에서 최종 보고에 사용.
   * 동기화: compute() 호출 후 읽기 전용. */

  int numCore, numL2, numL3, numNOC, numL1Dir, numL2Dir;
  /* [한국어] 각 서브시스템의 인스턴스 수 (편의 캐시 필드).
   * 설정자: set_proc_param()에서 XML->sys.number_of_cores 등으로부터 설정.
   * 읽는 자: compute(), displayEnergy() 등에서 루프 범위나 스케일링 계수로 사용.
   * - numCore: SM 수 (procdynp.numCore와 동일해야 함)
   * - numL2: L2 캐시 인스턴스 수 (GPU에서는 1)
   * - numL3: L3 캐시 인스턴스 수 (GPU에서는 0)
   * - numNOC: NoC 인스턴스 수 (GPU에서는 1)
   * - numL1Dir / numL2Dir: 디렉터리 캐시 수 (GPU에서는 0)
   * 값 범위: 0 이상의 정수.
   * 동기화: 초기화 이후 읽기 전용. */

  Processor(ParseXML *XML_interface);

  /*
   * [한국어] compute — 모든 서브컴포넌트의 전력을 일괄 계산하는 메인 메서드.
   *
   * 동작 과정:
   *   1) 각 Core::compute()를 호출하여 SM 파이프라인 전력을 산출한다.
   *   2) L2/L3 SharedCache::compute()로 캐시 전력을 산출한다.
   *   3) NoC::compute()로 라우터 전력을 산출한다.
   *   4) MemoryController::compute()로 DRAM 컨트롤러 전력을 산출한다.
   *   5) 각 서브컴포넌트 결과를 core/l2/noc/mcs Component 집계 객체에 합산.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [compute()] → Core::compute()
   *                                                  → SharedCache::compute()
   *                                                  → NoC::compute()
   *                                                  → MemoryController::compute()
   */
  void compute();

  /*
   * [한국어] set_proc_param — XML 설정으로부터 프로세서 파라미터를 초기화.
   *
   * 동작 과정:
   *   XML->sys 하위의 코어 수, 클럭 레이트, Vdd, 캐시 크기 등을 읽어
   *   procdynp, interface_ip, exClockRate 등의 멤버 필드를 채운다.
   *   서브컴포넌트 생성자에 전달할 interface_ip 구조체를 완성하는 역할이다.
   *
   * 호출 체인:
   *   Processor 생성자 → [set_proc_param()] → procdynp / interface_ip 설정
   */
  void set_proc_param();

  /*
   * [한국어] visualizer_print — GPGPU-Sim gzip 시각화 파일에 전력 데이터 기록.
   *
   * @visualizer_file: GPGPU-Sim이 열어둔 gzip 압축 visualizer 파일 핸들.
   *
   * 동작 과정:
   *   compute() 후 집계된 전력 데이터(코어/L2/NoC/MC별 동적·정적 전력)를
   *   gzprintf()로 visualizer 파일에 기록한다. AerialVision 등의 시각화 도구가
   *   이 데이터를 읽어 GPU 전력 타임라인을 표시한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() → accelwattch_interface::update()
   *       → [visualizer_print()] → gzprintf()
   */
  void visualizer_print(gzFile visualizer_file);

  /*
   * [한국어] displayEnergy — 계산된 전력/에너지 정보를 stdout에 출력.
   *
   * @indent: 들여쓰기 칸 수 (중첩 출력 시 계층 표시용).
   * @plevel: 출력 상세도 레벨 (100 = 모두 출력).
   * @is_tdp_parm: true이면 TDP(설계 최대 전력) 파라미터 기준으로 출력.
   *
   * 호출 체인:
   *   main() 또는 accelwattch_interface → [displayEnergy()] → 각 서브컴포넌트
   *   displayEnergy()
   */
  void displayEnergy(uint32_t indent = 0, int plevel = 100,
                     bool is_tdp_parm = true);

  /*
   * [한국어] displayDeviceType — McPAT 디바이스 타입 이름을 stdout에 출력.
   *
   * @device_type_: McPAT device_type 정수 (0=HP, 1=LSTP, 2=LOP 등).
   * @indent: 들여쓰기 칸 수.
   *
   * 호출 체인:
   *   displayEnergy() → [displayDeviceType()]
   */
  void displayDeviceType(int device_type_, uint32_t indent = 0);

  /*
   * [한국어] displayInterconnectType — McPAT 인터커넥트 타입 이름을 stdout에 출력.
   *
   * @interconnect_type_: McPAT interconnect_type 정수 (0=global, 1=semi-global 등).
   * @indent: 들여쓰기 칸 수.
   *
   * 호출 체인:
   *   displayEnergy() → [displayInterconnectType()]
   */
  void displayInterconnectType(int interconnect_type_, uint32_t indent = 0);

  double l2_power;
  /* [한국어] L2 캐시 전체 전력의 캐시 필드 (동적 + 정적, Watt 단위).
   * 설정자: compute()에서 l2array 전체 전력 합산 후 이 필드에 저장.
   *   L2 전력을 여러 곳에서 중복 계산하지 않도록 한 번만 계산해 보관한다.
   * 읽는 자: visualizer_print()와 displayEnergy()에서 L2 전력 보고에 사용.
   *   accelwattch_interface가 최종 전력 분류 리포트 생성 시 참조할 수 있다.
   * 값 범위: 0.0 이상의 실수 (Watt). 초기값 0.0.
   * 동기화: compute() 이후 읽기 전용. 단일 스레드 접근. */

  double idle_core_power;
  /* [한국어] 아이들(유휴) 상태 SM 코어들의 정적+기저 전력 합 (Watt 단위).
   * 실행 중이지 않은 SM들도 누설 전류로 인해 정적 전력을 소비한다.
   * 설정자: compute()에서 사용 중이지 않은 SM의 정적 전력을 합산하여 저장.
   *   procdynp.numCore에서 실제 활성 SM 수를 빼서 유휴 SM 수를 구한 뒤 곱산.
   * 읽는 자: visualizer_print()와 displayEnergy()에서 유휴 전력 보고에 사용.
   *   최종 전력 = 동적 전력 + idle_core_power + 기타 정적 전력.
   * 값 범위: 0.0 이상의 실수 (Watt). 모든 SM이 활성이면 0에 가까워짐.
   * 동기화: compute() 이후 읽기 전용. 단일 스레드 접근. */

  /*
   * [한국어] get_const_dynamic_power — 항상 소비되는 상수 동적 전력 반환.
   *
   * @return: 상수 동적 전력 (Joule/cycle 또는 Watt. 계산식에 따라 단위 확인 필요).
   *
   * 이 함수는 GPU가 실행 중일 때 이벤트 카운터와 무관하게 항상 소비되는
   * 전력 성분을 반환한다. 두 가지 성분으로 구성된다:
   *   1) DRAM 메모리 컨트롤러(MC)의 프론트엔드/트랜잭션 엔진/PHY 기저 전력:
   *      각 서브모듈의 동적 에너지에 10%(= 0.1, 기저 부하 가정) 이용률과
   *      클럭 레이트 및 실행 시간을 곱하여 전력으로 변환한다.
   *   2) SM 실행 유닛(정수 ALU exeu, 곱셈기 mul, FP 유닛 fp_u)의 기저 에너지:
   *      base_energy를 clockRate으로 나누어 사이클당 에너지(= Watt에 해당)를
   *      구한다. rf_fu_clockRate / clockRate는 레지스터 파일-FU 클럭 비율 보정.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_const_dynamic_power()]
   *       → mc->frontend->power / mc->transecEngine->power / mc->PHY->power
   *       → cores[0]->exu->exeu->base_energy / cores[0]->exu->mul->base_energy
   *         / cores[0]->exu->fp_u->base_energy
   */
  double get_const_dynamic_power() {
    double constpart = 0; // [한국어] 상수 동적 전력 누산 변수 초기화

    // [한국어] MC 프론트엔드의 10% 기저 부하 동적 전력 계산.
    //   동적 에너지(readOp.dynamic, Joule/access) × 10% 이용률 × 클럭 레이트
    //   × MC 수 × 실행 시간 = 전력 기여분(Watt). mc->frontend는 DRAM 컨트롤러의
    //   명령어 큐, PRT, threadMasks, PRC 등 프론트엔드 로직 모듈이다.
    constpart += (mc->frontend->power.readOp.dynamic * 0.1 *
                  mc->frontend->mcp.clockRate * mc->frontend->mcp.num_mcs *
                  mc->frontend->mcp.executionTime);

    // [한국어] MC 트랜잭션 엔진(transecEngine)의 10% 기저 부하 동적 전력 계산.
    //   transecEngine은 DRAM 트랜잭션 스케줄러 로직으로, 읽기/쓰기 큐를 관리.
    //   MC 전력 공식은 프론트엔드와 동일하게 0.1 × clockRate × num_mcs ×
    //   executionTime을 곱한다.
    constpart +=
        (mc->transecEngine->power.readOp.dynamic * 0.1 *
         mc->transecEngine->mcp.clockRate * mc->transecEngine->mcp.num_mcs *
         mc->transecEngine->mcp.executionTime);

    // [한국어] MC PHY(Physical Layer)의 10% 기저 부하 동적 전력 계산.
    //   PHY는 GDDR5 I/O 인터페이스 물리 레이어로, 데이터 버스 드라이버를 포함.
    //   동일한 0.1 × clockRate × num_mcs × executionTime 공식 적용.
    constpart += (mc->PHY->power.readOp.dynamic * 0.1 * mc->PHY->mcp.clockRate *
                  mc->PHY->mcp.num_mcs * mc->PHY->mcp.executionTime);

    // [한국어] SM 정수 ALU 실행 유닛(exeu) 기저 에너지를 사이클당 전력으로 변환.
    //   base_energy(Joule/사이클) / clockRate(Hz) = 전력(Watt). 여기에
    //   rf_fu_clockRate / clockRate 비율을 곱하여 레지스터 파일 클럭 도메인과
    //   FU 클럭 도메인의 차이를 보정한다. GPU에서는 cores[0]이 대표 SM이다.
    constpart +=
        (cores[0]->exu->exeu->base_energy / cores[0]->exu->exeu->clockRate) *
        (cores[0]->exu->rf_fu_clockRate / cores[0]->exu->clockRate);

    // [한국어] SM 정수 곱셈기(mul) 기저 에너지를 사이클당 전력으로 변환.
    //   base_energy / clockRate = 사이클당 Watt. 곱셈기는 분기 유닛과 별도
    //   클럭 도메인을 가지지 않으므로 추가 비율 보정 없이 단순 나눗셈.
    constpart +=
        (cores[0]->exu->mul->base_energy / cores[0]->exu->mul->clockRate);

    // [한국어] SM FP(부동소수점) 유닛(fp_u) 기저 에너지를 사이클당 전력으로 변환.
    //   CUDA GPU에서 FP 유닛은 SIMD 방식으로 warp 단위로 동작하므로 이 값은
    //   32개 레인 전체의 기저 전력을 포함한다.
    constpart +=
        (cores[0]->exu->fp_u->base_energy / cores[0]->exu->fp_u->clockRate);

    return constpart; // [한국어] 합산된 상수 동적 전력 반환
  }

  /*
   * [한국어] COALESCE_SCALE — 코얼레싱 에너지 스케일링 상수.
   * 현재 값 1 (= 스케일링 없음). 향후 보정 계수가 필요할 경우 이 매크로를
   * 수정하면 get_coefficient_readcoalescing/writecoalescing()에 일괄 적용된다.
   * CV² 공식의 커패시턴스 및 전압 파라미터 보정에 사용하도록 설계됨.
   */
#define COALESCE_SCALE 1

  /*
   * [한국어] get_coefficient_readcoalescing — 읽기 코얼레싱 1회당 에너지 계수 반환.
   *
   * @return: 메모리 읽기 코얼레싱 1회에 소비되는 에너지 (Joule/access).
   *
   * GPU 메모리 코얼레싱은 동일 warp 내 여러 스레드의 메모리 접근을 단일
   * 트랜잭션으로 병합하는 과정이다. 이 과정에서 다음 SRAM/로직 구조가 접근됨:
   *   - PRT(Pending Request Table): warp의 미완료 메모리 요청을 추적하는 SRAM.
   *     읽기 코얼레싱 시 readOp(조회 연산) 에너지 소비.
   *   - threadMasks: 어떤 스레드가 해당 요청에 참여했는지 추적하는 비트맵 SRAM.
   *     읽기 코얼레싱 시 readOp 에너지 소비.
   *   - PRC(Pending Request Controller): 코얼레싱 완료 신호를 각 스레드에 전달
   *     하는 컨트롤 로직 SRAM. readOp 에너지 소비.
   *   - perAccessCoalescingEnergy: CV² 공식 기반 코얼레싱 로직 자체 에너지.
   *     공식: COALESCE_SCALE × (0.443e-3 F) × (0.5e-9 H) × Vdd² / (1×1)
   *     0.443e-3은 커패시턴스(F), 0.5e-9는 인덕턴스 또는 에너지 보정 상수(H).
   *     g_tp.peri_global.Vdd는 CACTI 글로벌 파라미터에서 읽은 주변 회로 전원 전압.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_readcoalescing()]
   *       → mc->frontend->PRT/threadMasks/PRC->local_result.power.readOp.dynamic
   */
  double get_coefficient_readcoalescing() {
    double value = 0; // [한국어] 읽기 코얼레싱 에너지 계수 누산 변수 초기화

    // [한국어] CV² 공식으로 코얼레싱 로직 자체의 1회 접근 에너지를 계산.
    //   0.443e-3: 코얼레싱 로직의 등가 커패시턴스(F).
    //   0.5e-9: 에너지 보정 계수(단위 맥락상 인덕턴스 등, 실험적 상수).
    //   g_tp.peri_global.Vdd²: 주변 회로 전원 전압의 제곱 (CV² 에너지 공식).
    //   COALESCE_SCALE (=1): 현재 스케일링 없음, 향후 보정 여지.
    //   분모 (1 * 1): 정규화 상수 (현재는 1).
    double perAccessCoalescingEnergy =
        COALESCE_SCALE *
        ((0.443e-3) * (0.5e-9) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd) /
        (1 * 1);

    // [한국어] PRT(Pending Request Table) 읽기 동작 에너지 추가.
    //   PRT는 warp의 미완료 메모리 요청을 저장하는 SRAM으로,
    //   읽기 코얼레싱 시 조회(read) 연산이 발생하여 이 에너지가 소비된다.
    value += mc->frontend->PRT->local_result.power.readOp.dynamic;

    // [한국어] threadMasks(스레드 마스크 배열) 읽기 동작 에너지 추가.
    //   어떤 스레드가 해당 메모리 요청에 참여하는지 추적하는 비트맵 SRAM의
    //   read 접근 에너지. 코얼레싱 결과를 스레드별로 전달하는 데 필요.
    value += mc->frontend->threadMasks->local_result.power.readOp.dynamic;

    // [한국어] PRC(Pending Request Controller) 읽기 동작 에너지 추가.
    //   코얼레싱 완료 시 대기 스레드들에게 완료 신호를 전달하는 컨트롤러 SRAM의
    //   read 접근 에너지. PRT, threadMasks와 함께 MC 프론트엔드를 구성한다.
    value += mc->frontend->PRC->local_result.power.readOp.dynamic;

    // [한국어] CV² 기반 코얼레싱 로직 자체 에너지를 누산.
    value += perAccessCoalescingEnergy;

    return value; // [한국어] 읽기 코얼레싱 1회당 총 에너지 계수 반환 (Joule)
  }

  /*
   * [한국어] get_coefficient_writecoalescing — 쓰기 코얼레싱 1회당 에너지 계수 반환.
   *
   * @return: 메모리 쓰기 코얼레싱 1회에 소비되는 에너지 (Joule/access).
   *
   * 읽기 코얼레싱(get_coefficient_readcoalescing)과 동일한 하드웨어 구조(PRT,
   * threadMasks, PRC)를 사용하지만, 쓰기 방향 동작(writeOp)의 에너지를 반환한다.
   * GPU에서 쓰기 코얼레싱은 store 명령어 수행 시 동일 캐시 라인을 대상으로 하는
   * 여러 스레드의 쓰기를 병합하는 과정이다.
   * perAccessCoalescingEnergy 계산 공식은 읽기와 동일하다 (회로는 공유됨).
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_writecoalescing()]
   *       → mc->frontend->PRT/threadMasks/PRC->local_result.power.writeOp.dynamic
   */
  double get_coefficient_writecoalescing() {
    double value = 0; // [한국어] 쓰기 코얼레싱 에너지 계수 누산 변수 초기화

    // [한국어] CV² 공식으로 코얼레싱 로직 1회 접근 에너지를 계산.
    //   공식은 읽기 코얼레싱과 동일하다: 코얼레싱 회로 자체는 읽기/쓰기 방향에
    //   관계없이 동일한 하드웨어이므로 같은 커패시턴스·전압 파라미터를 사용.
    double perAccessCoalescingEnergy =
        COALESCE_SCALE *
        ((0.443e-3) * (0.5e-9) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd) /
        (1 * 1);

    // [한국어] PRT 쓰기 동작 에너지 추가.
    //   쓰기 코얼레싱 시 PRT에 새 엔트리를 기록(write)하는 에너지. 읽기 코얼레싱의
    //   readOp와 달리 여기서는 writeOp.dynamic을 사용한다.
    value += (mc->frontend->PRT->local_result.power.writeOp.dynamic);

    // [한국어] threadMasks 쓰기 동작 에너지 추가.
    //   쓰기 요청에 참여하는 스레드 마스크를 PRT에 기록하는 write 접근 에너지.
    value += mc->frontend->threadMasks->local_result.power.writeOp.dynamic;

    // [한국어] PRC 쓰기 동작 에너지 추가.
    //   쓰기 완료 컨트롤 정보를 PRC에 기록하는 write 접근 에너지.
    value += mc->frontend->PRC->local_result.power.writeOp.dynamic;

    // [한국어] CV² 기반 코얼레싱 로직 자체 에너지를 누산.
    value += perAccessCoalescingEnergy;

    return value; // [한국어] 쓰기 코얼레싱 1회당 총 에너지 계수 반환 (Joule)
  }

  /*
   * [한국어] get_coefficient_noc_accesses — NoC 라우터 접근 1회당 에너지 계수 반환.
   *
   * @return: NoC(SM-to-L2 인터커넥트) 라우터 접근 1회에 소비되는 에너지
   *   (Joule/flit 또는 Joule/packet, 설정에 따라 다름).
   *
   * GPU의 NoC는 SM과 L2 캐시 사이의 온칩 네트워크(intersim2와 대응)이다.
   * 라우터 접근 1회에 다음 4가지 하드웨어 구조가 동작한다:
   *   1) buffer.readOp: 입력 버퍼(Virtual Channel 버퍼)에서 플릿 읽기
   *   2) buffer.writeOp: 입력 버퍼에 플릿 쓰기 (수신 시)
   *   3) crossbar.readOp: 크로스바 스위치 통과 에너지 (입력→출력 포트 연결)
   *   4) arbiter.readOp: 중재기(VC 중재 + 스위치 중재) 동작 에너지
   * 주석(32/4)은 원래 NoC 접근 카운터에 적용하던 스케일링 팩터였으나,
   * 현재는 accelwattch_interface의 이벤트 카운터 쪽에서 처리하므로 여기서는
   * 제거되어 있다.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_noc_accesses()]
   *       → nocs[0]->router->buffer/crossbar/arbiter 전력 필드
   */
  double get_coefficient_noc_accesses() {
    double read_coef = 0; // [한국어] NoC 접근 에너지 계수 누산 변수 초기화

    // the 32/4 is applied to the NoC access counters (32/4*L2 cache access)
    // [한국어] (주석 보존) 원래 32/4 스케일링은 NoC 접근 카운터 쪽에 적용됨.
    //   현재 이 함수에서는 스케일링 없이 라우터 1회 접근의 순수 에너지만 반환.

    // [한국어] NoC 라우터 입력 버퍼 읽기 에너지 추가.
    //   Virtual Channel(VC) 버퍼에서 플릿을 읽어 크로스바로 보내는 에너지.
    //   nocs[0]는 GPU 내 단일 NoC 레이어(SM-to-L2)의 대표 라우터이다.
    read_coef += nocs[0]->router->buffer.power.readOp.dynamic;

    // [한국어] NoC 라우터 입력 버퍼 쓰기 에너지 추가.
    //   새 플릿이 라우터에 도착하여 VC 버퍼에 저장될 때의 write 에너지.
    read_coef += nocs[0]->router->buffer.power.writeOp.dynamic;

    // [한국어] NoC 크로스바 스위치 통과 에너지 추가.
    //   입력 포트에서 출력 포트로 플릿을 전송하는 크로스바 횡단(crossbar
    //   traversal) 에너지. readOp으로 모델링되며 스위치 배선 커패시턴스 충전.
    read_coef += nocs[0]->router->crossbar.power.readOp.dynamic;

    // [한국어] NoC 중재기(arbiter) 동작 에너지 추가.
    //   VC 중재(VC allocation)와 스위치 중재(switch allocation)를 수행하는
    //   중재기 로직의 에너지. 매 라우터 접근마다 중재기가 활성화된다.
    read_coef += nocs[0]->router->arbiter.power.readOp.dynamic;

    return read_coef; // [한국어] NoC 라우터 접근 1회당 총 에너지 계수 반환
  }

  /*
   * [한국어] get_coefficient_l2_read_hits — L2 캐시 읽기 히트 1회당 에너지 계수 반환.
   *
   * @return: L2 캐시 읽기 히트 1회에 소비되는 에너지 (Joule/access).
   *   L2가 없으면(number_of_L2s == 0) 0을 반환한다.
   *
   * L2 읽기 히트 시에는 태그 어레이와 데이터 어레이 모두 접근하지만,
   * McPAT에서 readOp.dynamic은 태그+데이터 배열 전체 접근 에너지를 포함한다.
   * GPU에서 L2는 모든 SM이 공유하는 통합 캐시이며, l2array[0]이 이를 모델링.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_l2_read_hits()]
   *       → l2array[0]->unicache.caches->local_result.power.readOp.dynamic
   */
  double get_coefficient_l2_read_hits() {
    double read_coef = 0; // [한국어] L2 읽기 히트 에너지 계수 초기화

    // [한국어] L2가 존재하는 경우에만 계수 계산 (number_of_L2s > 0 조건).
    //   GPU에서는 항상 L2가 1개 존재하므로 이 조건은 항상 참이다.
    if (XML->sys.number_of_L2s > 0)
      // [한국어] l2array[0]의 unicache(통합 캐시) 구조 내 caches SRAM 어레이의
      //   readOp 동적 에너지를 읽기 히트 계수로 반환. McPAT local_result에는
      //   CACTI가 계산한 최종 데이터+태그 어레이 접근 에너지가 담겨 있다.
      read_coef =
          l2array[0]->unicache.caches->local_result.power.readOp.dynamic;

    return read_coef; // [한국어] L2 읽기 히트 1회 에너지 계수 반환 (없으면 0)
  }

  /*
   * [한국어] get_coefficient_l2_read_misses — L2 캐시 읽기 미스 1회당 에너지 계수 반환.
   *
   * @return: L2 캐시 읽기 미스 1회에 소비되는 에너지 (Joule/access).
   *   L2가 없으면 0을 반환한다.
   *
   * 읽기 미스 시에는 태그 어레이만 접근하고(히트 판정 후 미스 확인),
   * 데이터 어레이는 접근하지 않는다. 따라서 tag_array2의 readOp 에너지만
   * 반환한다. tag_array2는 McPAT에서 2웨이 세트 어소시어티브 태그 배열에 해당.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_l2_read_misses()]
   *       → l2array[0]->unicache.caches->local_result.tag_array2->power.readOp.dynamic
   */
  double get_coefficient_l2_read_misses() {
    double read_coef = 0; // [한국어] L2 읽기 미스 에너지 계수 초기화

    // [한국어] L2 존재 여부 확인 후 태그 배열 읽기 에너지만 반환.
    if (XML->sys.number_of_L2s > 0)
      // [한국어] tag_array2→power.readOp.dynamic: 태그 어레이 읽기(히트/미스 판정)
      //   에너지. 미스일 때는 데이터 어레이 접근이 없으므로 태그 에너지만 계산.
      //   태그 어레이 접근은 히트 시에도 발생하지만, 여기서는 미스 시의
      //   추가 에너지(상위 메모리 계층 접근 트리거 에너지)를 근사한다.
      read_coef =
          l2array[0]
              ->unicache.caches->local_result.tag_array2->power.readOp.dynamic;

    return read_coef; // [한국어] L2 읽기 미스 1회 에너지 계수 반환 (없으면 0)
  }

  /*
   * [한국어] get_coefficient_l2_write_hits — L2 캐시 쓰기 히트 1회당 에너지 계수 반환.
   *
   * @return: L2 캐시 쓰기 히트 1회에 소비되는 에너지 (Joule/access).
   *   L2가 없으면 0을 반환한다.
   *
   * 쓰기 히트 시에는 태그 어레이 읽기(히트 확인) 후 데이터 어레이에 쓰기가
   * 발생한다. McPAT의 writeOp.dynamic은 이 전체 과정의 에너지를 포함한다.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_l2_write_hits()]
   *       → l2array[0]->unicache.caches->local_result.power.writeOp.dynamic
   */
  double get_coefficient_l2_write_hits() {
    double read_coef = 0; // [한국어] L2 쓰기 히트 에너지 계수 초기화 (변수명은 read_coef이나 쓰기 히트 계수임)

    // [한국어] L2 존재 여부 확인 후 캐시 어레이 쓰기 에너지 반환.
    if (XML->sys.number_of_L2s > 0)
      // [한국어] unicache.caches→power.writeOp.dynamic: 쓰기 히트 시 데이터
      //   어레이 및 태그 어레이에 쓰기하는 전체 에너지. 읽기 히트(readOp)와
      //   달리 쓰기 동작(writeOp)의 에너지는 일반적으로 더 크다.
      read_coef =
          l2array[0]->unicache.caches->local_result.power.writeOp.dynamic;

    return read_coef; // [한국어] L2 쓰기 히트 1회 에너지 계수 반환 (없으면 0)
  }

  /*
   * [한국어] get_coefficient_l2_write_misses — L2 캐시 쓰기 미스 1회당 에너지 계수 반환.
   *
   * @return: L2 캐시 쓰기 미스 1회에 소비되는 에너지 (Joule/access).
   *   L2가 없으면 0을 반환한다.
   *
   * 쓰기 미스는 가장 복잡한 L2 접근 유형이다. McPAT에서 쓰기 미스 시에는
   * 태그 어레이 읽기 + 데이터 어레이 쓰기 외에도 MSHR(Miss Status Holding
   * Register) 관련 4개 버퍼가 모두 접근된다:
   *   1) tag_array2 writeOp: 새 블록 할당 시 태그 어레이 업데이트
   *   2) caches writeOp: 데이터 어레이에 새 블록 쓰기 (write-allocate 정책)
   *   3) missb(Miss Buffer): 미스 처리 중인 요청 추적 SRAM
   *      - searchOp: 중복 미스(secondary miss) 확인을 위한 탐색
   *      - writeOp: 새 미스 엔트리 기록
   *   4) ifb(In-Flight Buffer): 상위 메모리로 전송 중인 요청 추적
   *      - searchOp + writeOp
   *   5) prefetchb(Prefetch Buffer): 프리패치 요청 추적 SRAM
   *      - searchOp + writeOp
   *   6) wbb(Write-Back Buffer): 교체 시 쓰기백 대기 블록 저장 SRAM
   *      - searchOp + writeOp
   * 원래 32/4 스케일링(Jingwen에 의해 제거됨)은 태그 어레이 접근에만 적용됐으나
   * McPAT 내부에서 처리되므로 현재는 제거 상태이다.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_l2_write_misses()]
   *       → l2array[0]->unicache.caches/missb/ifb/prefetchb/wbb 전력 필드
   */
  double get_coefficient_l2_write_misses() {
    double read_coef = 0; // [한국어] L2 쓰기 미스 에너지 계수 누산 변수 초기화

    // [한국어] L2 존재 여부 확인. GPU에서는 항상 L2가 있으므로 항상 진입.
    if (XML->sys.number_of_L2s > 0) {
      // [한국어] tag_array2 쓰기 에너지: 쓰기 미스 시 새 블록을 캐시에 할당할 때
      //   태그 어레이에 새 태그/상태를 기록하는 에너지. 원주석에 따르면 이전에는
      //   32/4 스케일을 곱했으나 McPAT 내부 처리 방식 변경으로 제거됨.
      read_coef = l2array[0]
                      ->unicache.caches->local_result.tag_array2->power.writeOp
                      .dynamic;  //*(32/4); // removed by Jingwen, the scaling
                                 // of 32/4 is not used in the mcpat

      // [한국어] 데이터 어레이 쓰기 에너지: write-allocate 정책에 따라 미스된
      //   캐시 라인을 상위 메모리에서 가져온 후 L2 데이터 어레이에 기록하는 에너지.
      read_coef +=
          l2array[0]->unicache.caches->local_result.power.writeOp.dynamic;

      // [한국어] missb(Miss Buffer) 탐색 에너지: 같은 캐시 라인에 대한 중복 미스
      //   (secondary miss)가 있는지 missb를 검색(search)하는 에너지.
      //   MSHR(Miss Status Holding Register)이 이 역할을 수행한다.
      read_coef +=
          l2array[0]->unicache.missb->local_result.power.searchOp.dynamic;

      // [한국어] missb 쓰기 에너지: 새 미스 요청을 missb에 기록하는 에너지.
      //   미스가 완료될 때까지 이 엔트리가 유지되어 중복 요청을 병합한다.
      read_coef +=
          l2array[0]->unicache.missb->local_result.power.writeOp.dynamic;

      // [한국어] ifb(In-Flight Buffer) 탐색 에너지: 상위 메모리(DRAM)로 이미
      //   전송 요청이 나간 블록에 대한 중복 접근을 확인하는 검색 에너지.
      read_coef +=
          l2array[0]->unicache.ifb->local_result.power.searchOp.dynamic;

      // [한국어] ifb 쓰기 에너지: 상위 메모리로 전송 중인 요청을 ifb에 기록하는
      //   에너지. 전송 완료 시 ifb 엔트리가 제거되고 대기 요청이 처리된다.
      read_coef += l2array[0]->unicache.ifb->local_result.power.writeOp.dynamic;

      // [한국어] prefetchb(Prefetch Buffer) 탐색 에너지: 프리패치로 가져온 블록이
      //   이미 버퍼에 있는지 확인하는 검색 에너지. 쓰기 미스 시에도 프리패치
      //   버퍼를 확인하여 중복 요청을 방지한다.
      read_coef +=
          l2array[0]->unicache.prefetchb->local_result.power.searchOp.dynamic;

      // [한국어] prefetchb 쓰기 에너지: 새 프리패치 요청을 prefetchb에 기록하는
      //   에너지. L2 쓰기 미스가 하드웨어 프리패처를 트리거할 수 있다.
      read_coef +=
          l2array[0]->unicache.prefetchb->local_result.power.writeOp.dynamic;

      // [한국어] wbb(Write-Back Buffer) 탐색 에너지: 교체될 캐시 라인이 dirty인
      //   경우 wbb에 저장되어 있는지 확인하는 검색 에너지. wbb는 쓰기백 대기열.
      read_coef +=
          l2array[0]->unicache.wbb->local_result.power.searchOp.dynamic;

      // [한국어] wbb 쓰기 에너지: 교체되는 dirty 캐시 라인을 wbb에 기록하는
      //   에너지. DRAM으로의 실제 쓰기백은 wbb에서 비동기적으로 수행된다.
      read_coef += l2array[0]->unicache.wbb->local_result.power.writeOp.dynamic;
    }

    return read_coef; // [한국어] L2 쓰기 미스 1회 총 에너지 계수 반환 (없으면 0)
  }

  /*
   * [한국어] get_coefficient_mem_reads — DRAM 읽기 트랜잭션 1회당 에너지 계수 반환.
   *
   * @return: GDDR5 DRAM 읽기 1회(llcBlockSize 크기의 캐시 라인 읽기)에 소비되는
   *   전체 메모리 계층 에너지 (Joule/transaction).
   *
   * DRAM 읽기 트랜잭션은 다음 5개 하드웨어 계층에서 에너지를 소비한다:
   *   1) MC 프론트엔드 버퍼(frontendBuffer): L2 → MC 사이의 요청 큐 SRAM.
   *      llcBlockSize × 8 / dataBusWidth × dataBusWidth/72 회만큼 접근.
   *      (하나의 캐시 블록을 dataBusWidth 비트씩 나누어 전송하는 횟수)
   *      - searchOp: 중복/충돌 확인
   *      - readOp: 데이터 읽기
   *   2) MC readBuffer: 읽기 데이터가 DRAM에서 도착할 때까지 대기하는 버퍼.
   *      llcBlockSize × 8 / dataBusWidth 회만큼 readOp + writeOp.
   *   3) DRAM 칩 읽기(dramp.rd_coeff): DRAM 칩 자체의 읽기 에너지 계수.
   *      CACTI DRAM 모델에서 계산된 단위 에너지 (Joule/access).
   *   4) 트랜잭션 엔진(transecEngine): DRAM 명령 스케줄러 로직.
   *      llcBlockSize × 8 / dataBusWidth × power_t.readOp.dynamic.
   *   5) PHY(물리 레이어): GDDR5 I/O 물리 드라이버. 버스 전송 에너지.
   *      PHY 전력 × llcBlockSize × 8(비트) / 1e9(GHz) / executionTime × executionTime
   *      = PHY 전력 × llcBlockSize × 8(비트) / 1e9 (단순화된 에너지 계산).
   *
   * 주석 처리된 NoC 접근 에너지는 get_coefficient_noc_accesses()로 분리됨.
   * PRT/threadMasks/PRC 에너지도 get_coefficient_readcoalescing()으로 분리됨.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_mem_reads()]
   *       → mc->frontend->frontendBuffer/readBuffer/PRT 전력 필드
   *       → mc->dram->dramp.rd_coeff
   *       → mc->transecEngine->power_t / mc->PHY->power_t
   */
  double get_coefficient_mem_reads() {
    double value = 0; // [한국어] DRAM 읽기 에너지 계수 누산 변수 초기화

    // [한국어] MC 프론트엔드 버퍼 탐색 에너지 추가.
    //   llcBlockSize(LLC 블록 크기, 바이트) × 8(비트/바이트)를 dataBusWidth로 나누면
    //   하나의 캐시 라인을 전송하는 데 필요한 버스 전송 횟수가 된다.
    //   여기에 dataBusWidth / 72를 곱하는 것은 72비트(64비트 데이터 + 8비트 ECC)
    //   기준으로 정규화하여 ECC 오버헤드를 제외한 실제 데이터 버스 너비 비율을 보정.
    //   searchOp: frontendBuffer에서 이미 처리 중인 요청과 충돌하는지 탐색하는 에너지.
    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.searchOp.dynamic);

    // [한국어] MC 프론트엔드 버퍼 읽기 에너지 추가.
    //   동일한 스케일링 팩터(llcBlockSize × 8 / dataBusWidth × dataBusWidth/72)를
    //   적용한 frontendBuffer readOp 에너지. 버퍼에서 요청 항목을 읽어
    //   트랜잭션 엔진으로 전달하는 에너지에 해당한다.
    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.readOp.dynamic);

    // TODO: Jingwen this should only compute for one time?
    // value+=(mc->frontend->mcp.llcBlockSize*8.0/mc->frontend->mcp.dataBusWidth*mc->frontend->mcp.dataBusWidth/72)
    //*(mc->frontend->frontendBuffer->local_result.power.readOp.dynamic);
    // [한국어] 원주석: 이 frontendBuffer readOp를 한 번만 계산해야 하는지 검토 필요.
    //   현재는 위에서 한 번만 계산하므로 이 주석 처리된 코드는 중복 계산 방지.

    // [한국어] MC readBuffer 읽기 에너지 추가.
    //   readBuffer는 DRAM에서 데이터가 돌아올 때까지 읽기 요청을 보관하는 큐.
    //   dataBusWidth 기준으로 접근 횟수를 스케일링 (ECC 보정 없이 단순 비율).
    //   readOp: 버퍼에서 완료된 데이터 읽기.
    value += (mc->frontend->mcp.llcBlockSize * 8.0 / mc->mcp.dataBusWidth) *
             (mc->frontend->readBuffer->local_result.power.readOp.dynamic);

    // [한국어] MC readBuffer 쓰기 에너지 추가.
    //   DRAM에서 데이터가 도착하여 readBuffer에 기록하는 writeOp 에너지.
    //   읽기 트랜잭션 완료 시 DRAM → readBuffer → L2 순서로 데이터가 흐른다.
    value += (mc->frontend->mcp.llcBlockSize * 8.0 / mc->mcp.dataBusWidth) *
             (mc->frontend->readBuffer->local_result.power.writeOp.dynamic);

    // [한국어] DRAM 칩 읽기 에너지 계수 추가.
    //   mc->dram->dramp.rd_coeff는 CACTI DRAM 모델이 계산한 DRAM 칩 자체의
    //   단위 읽기 에너지(row 활성화 + 열 읽기 + 프리차지 등 포함 가능).
    //   단위: Joule/transaction (llcBlockSize 크기 기준).
    value += mc->dram->dramp.rd_coeff;

    /*
    [한국어] 주석 처리된 PRT/threadMasks/PRC/coalescing 에너지:
    이 성분들은 get_coefficient_readcoalescing()으로 분리되었으므로
    여기서 중복 계산하지 않는다. 코얼레싱은 DRAM 접근과 별개 이벤트로 카운트.
            value+=mc->frontend->PRT->local_result.power.readOp.dynamic;
            value+=mc->frontend->threadMasks->local_result.power.readOp.dynamic;
            value+=mc->frontend->PRC->local_result.power.readOp.dynamic;
            value+=perAccessCoalescingEnergy;
            */

    // [한국어] 트랜잭션 엔진(transecEngine) 읽기 에너지 추가.
    //   transecEngine은 DRAM 명령(RAS/CAS/PRE)을 생성하고 스케줄링하는 로직.
    //   llcBlockSize × 8 / dataBusWidth 횟수만큼 power_t.readOp 에너지 소비.
    //   power_t는 TDP 기준 전력이 아닌 실제 동작 기준 전력(per-access)을 나타냄.
    value += (mc->transecEngine->mcp.llcBlockSize * 8.0 /
              mc->transecEngine->mcp.dataBusWidth *
              mc->transecEngine->power_t.readOp.dynamic);

    // if mcp.type ==1 TODO: add this check here
    // [한국어] PHY(물리 레이어) 읽기 에너지 추가.
    //   PHY는 GDDR5 직렬 I/O 인터페이스의 물리적 드라이버/수신기 회로.
    //   계산식: PHY power_t.readOp(Watt) × llcBlockSize(byte) × 8(bits/byte)
    //           / 1e9 / executionTime × executionTime
    //   = PHY power_t.readOp × llcBlockSize × 8 / 1e9 (Joule/transaction)
    //   executionTime의 분모/분자 약분으로 최종적으로 1e9(GHz 단위 보정)만 남음.
    //   TODO: mcp.type == 1 (GDDR5)인지 확인 후에만 적용해야 할 수도 있음.
    value += (mc->PHY->power_t.readOp.dynamic) * (mc->PHY->mcp.llcBlockSize) *
             8 / 1e9 / mc->PHY->mcp.executionTime *
             (mc->PHY->mcp.executionTime);

    // printf("MC PHY read power coeff:
    // %f\n",(mc->PHY->power_t.readOp.dynamic)*(mc->PHY->mcp.llcBlockSize)*8/1e9/mc->PHY->mcp.executionTime*(mc->PHY->mcp.executionTime));
    // printf("MC trans read power coeff:
    // %f\n",(mc->transecEngine->mcp.llcBlockSize*8.0/mc->transecEngine->mcp.dataBusWidth*mc->transecEngine->power_t.readOp.dynamic));
    // [한국어] 위 printf 주석들: PHY 및 transecEngine 계수를 디버깅할 때 활성화.

    // TODO: Jingwen nocs stats should not be here
    //		value+= nocs[0]->router->buffer.power.readOp.dynamic*(32/4);
    //		value+= nocs[0]->router->buffer.power.writeOp.dynamic*(32/4);
    //		value+= nocs[0]->router->crossbar.power.readOp.dynamic*(32/4);
    //		value+= nocs[0]->router->arbiter.power.readOp.dynamic*(32/4);
    // [한국어] 주석 처리된 NoC 성분: get_coefficient_noc_accesses()로 분리됨.
    //   NoC 접근은 DRAM 트랜잭션과 별개 이벤트 카운터로 계산해야 하므로 제거됨.

    // return 0.4*value;
    // [한국어] 이전 0.4 스케일링 코드 (제거됨): 실측 대비 과대 추정을 보정하던
    //   임시 계수였으나 AccelWattch 검증 후 제거되었다.
    return value; // [한국어] DRAM 읽기 1회 총 에너지 계수 반환 (Joule/transaction)
  }

  /*
   * [한국어] get_coefficient_mem_writes — DRAM 쓰기 트랜잭션 1회당 에너지 계수 반환.
   *
   * @return: GDDR5 DRAM 쓰기 1회(llcBlockSize 크기의 캐시 라인 쓰기)에 소비되는
   *   전체 메모리 계층 에너지 (Joule/transaction).
   *
   * 읽기(get_coefficient_mem_reads)와 구조가 동일하나, 다음 차이점이 있다:
   *   - frontendBuffer: searchOp + writeOp (읽기는 searchOp + readOp)
   *     writeOp는 L2에서 쓰기 요청이 frontendBuffer에 기록되는 에너지.
   *   - readBuffer 대신 writeBuffer: 쓰기 데이터가 DRAM으로 전송되기 전
   *     대기하는 쓰기 전용 버퍼. readOp + writeOp 에너지.
   *   - DRAM 칩 쓰기(dramp.wr_coeff): 읽기(rd_coeff)와 별개의 쓰기 에너지 계수.
   *   - transecEngine, PHY: 읽기와 동일한 공식 적용 (방향 무관한 공유 회로).
   *
   * 주석 처리된 PRT/threadMasks/PRC/coalescing 에너지: 쓰기 코얼레싱은
   * get_coefficient_writecoalescing()으로 분리됨.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_mem_writes()]
   *       → mc->frontend->frontendBuffer/writeBuffer 전력 필드
   *       → mc->dram->dramp.wr_coeff
   *       → mc->transecEngine->power_t / mc->PHY->power_t
   */
  double get_coefficient_mem_writes() {
    double value = 0; // [한국어] DRAM 쓰기 에너지 계수 누산 변수 초기화

    // [한국어] MC 프론트엔드 버퍼 탐색 에너지 추가 (쓰기 방향).
    //   읽기와 동일한 스케일링 팩터 적용. searchOp: 쓰기 요청 도착 시
    //   frontendBuffer에서 충돌/중복 요청이 있는지 탐색하는 에너지.
    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.searchOp.dynamic);

    // [한국어] MC 프론트엔드 버퍼 쓰기 에너지 추가.
    //   쓰기 요청을 frontendBuffer에 기록하는 writeOp 에너지.
    //   읽기의 readOp 대신 writeOp를 사용하는 것이 읽기와의 유일한 차이.
    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.writeOp.dynamic);

    // value+=(mc->frontend->mcp.llcBlockSize*8.0/mc->frontend->mcp.dataBusWidth*mc->frontend->mcp.dataBusWidth/72)*
    // (mc->frontend->frontendBuffer->local_result.power.writeOp.dynamic);
    // [한국어] 중복 writeOp 계산 방지를 위해 주석 처리된 이전 코드.

    // [한국어] MC writeBuffer 읽기 에너지 추가.
    //   writeBuffer는 쓰기 데이터를 DRAM 컨트롤러로 전달하기 전 임시 저장하는 큐.
    //   readOp: writeBuffer에서 쓰기 데이터를 읽어 DRAM 명령으로 변환하는 에너지.
    value += (mc->frontend->mcp.llcBlockSize * 8.0 /
              mc->frontend->mcp.dataBusWidth) *
             (mc->frontend->writeBuffer->local_result.power.readOp.dynamic);

    // [한국어] MC writeBuffer 쓰기 에너지 추가.
    //   L2 캐시에서 내려온 쓰기 데이터를 writeBuffer에 기록하는 writeOp 에너지.
    //   DRAM 쓰기는 쓰기 데이터 → writeBuffer → DRAM 칩 순서로 진행된다.
    value += (mc->frontend->mcp.llcBlockSize * 8.0 /
              mc->frontend->mcp.dataBusWidth) *
             (mc->frontend->writeBuffer->local_result.power.writeOp.dynamic);

    // [한국어] DRAM 칩 쓰기 에너지 계수 추가.
    //   mc->dram->dramp.wr_coeff: CACTI DRAM 모델이 계산한 DRAM 칩 쓰기 에너지.
    //   읽기(rd_coeff)와 달리 쓰기는 row 활성화 + 열 쓰기 에너지로 구성되며
    //   일반적으로 rd_coeff와 유사하거나 약간 다른 값을 가진다.
    value += mc->dram->dramp.wr_coeff;

    /*
    [한국어] 주석 처리된 PRT/threadMasks/PRC/coalescing 에너지:
    쓰기 코얼레싱은 get_coefficient_writecoalescing()으로 분리 계산.
            value+=(mc->frontend->PRT->local_result.power.writeOp.dynamic);
            value+=mc->frontend->threadMasks->local_result.power.writeOp.dynamic;
            value+=mc->frontend->PRC->local_result.power.writeOp.dynamic;
            value+=perAccessCoalescingEnergy;
            */

    // [한국어] 트랜잭션 엔진 쓰기 에너지 추가.
    //   transecEngine은 읽기/쓰기 공유 로직이므로 동일한 power_t.readOp을 사용.
    //   (쓰기 방향이라도 transecEngine의 전력 모델은 readOp으로 표현된다.)
    value += (mc->transecEngine->mcp.llcBlockSize * 8.0 /
              mc->transecEngine->mcp.dataBusWidth *
              mc->transecEngine->power_t.readOp.dynamic);

    // if mcp.type ==1 TODO: add this check here
    // [한국어] PHY 쓰기 에너지 추가.
    //   공식은 읽기와 동일: PHY power_t.readOp × llcBlockSize × 8 / 1e9.
    //   PHY는 I/O 방향(읽기/쓰기)에 관계없이 동일한 드라이버 회로를 사용하므로
    //   readOp 에너지로 모델링. TODO: mcp.type 확인 필요.
    value += (mc->PHY->power_t.readOp.dynamic) * (mc->PHY->mcp.llcBlockSize) *
             8 / 1e9 / mc->PHY->mcp.executionTime *
             (mc->PHY->mcp.executionTime);

    // TODO: Jingwen nocs stats should not be here
    //		value+= nocs[0]->router->buffer.power.readOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->buffer.power.writeOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->crossbar.power.readOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->arbiter.power.readOp.dynamic*(32/4);
    // [한국어] 주석 처리된 NoC 성분: get_coefficient_noc_accesses()로 분리됨.

    //
    // return 0.4*value;
    // [한국어] 이전 0.4 스케일링 코드 (제거됨): 검증 후 불필요하여 제거.
    return value; // [한국어] DRAM 쓰기 1회 총 에너지 계수 반환 (Joule/transaction)
  }

  /*
   * [한국어] get_coefficient_mem_pre — DRAM 프리차지(precharge) 1회당 에너지 계수 반환.
   *
   * @return: DRAM precharge 명령 1회에 소비되는 에너지 (Joule/precharge).
   *
   * DRAM 동작은 행 활성화(RAS) → 열 읽기/쓰기(CAS) → 프리차지(PRE) 순서로
   * 진행된다. 프리차지는 활성화된 행을 닫고 비트라인을 VDD/2로 충전하는 동작.
   * AccelWattch에서는 DRAM 접근 이벤트를 읽기(rd_coeff), 쓰기(wr_coeff),
   * 프리차지(pre_coeff) 세 가지로 분리하여 계산한다.
   * dramp.pre_coeff는 CACTI DRAM 모델이 계산한 PRE 명령 1회 에너지이다.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [get_coefficient_mem_pre()]
   *       → mc->dram->dramp.pre_coeff
   */
  double get_coefficient_mem_pre() {
    double value = 0; // [한국어] 프리차지 에너지 계수 초기화

    // [한국어] DRAM 칩 프리차지(PRE 명령) 에너지 계수 할당.
    //   dramp.pre_coeff는 CACTI DRAM 타이밍 모델에서 계산된 PRE 1회 에너지.
    //   비트라인 프리차지 전력은 DRAM 전체 전력의 상당 부분을 차지하므로
    //   읽기/쓰기와 별도로 추적하는 것이 AccelWattch의 설계 결정이다.
    value += mc->dram->dramp.pre_coeff;

    // return 0.4*value;
    // [한국어] 이전 0.4 스케일링 코드 (제거됨): 검증 후 불필요하여 제거.
    return value; // [한국어] DRAM 프리차지 1회 에너지 계수 반환 (Joule/precharge)
  }

  // nonlinear scale
  /*
   * [한국어] nonlinear_scale — 비선형 전력 모델 스케일링 적용.
   *
   * @param1: int — 반복 횟수 또는 스케일링 모드 선택자.
   * @param2: double — 스케일링 타깃 값 또는 기준 전력.
   * @param3: int — 추가 제어 파라미터 (구현체에서 결정).
   *
   * 실측 GPU 전력과 McPAT 기반 모델 전력 사이의 오차를 비선형 보정 함수로
   * 줄이는 메서드. AccelWattch의 핵심 기여 중 하나로, 단순 선형 스케일이
   * 아닌 전력 성분별 비선형 관계를 모델링하여 검증 오차를 최소화한다.
   * dyn_power_before_scaling 필드에 보정 전 값을 저장한 뒤 적용한다.
   *
   * 호출 체인:
   *   accelwattch_interface::update() → [nonlinear_scale()] → coefficient_scale()
   */
  void nonlinear_scale(int, double, int);

  /*
   * [한국어] coefficient_scale — 에너지 계수 전체에 스케일링 인수 적용.
   *
   * nonlinear_scale()에서 결정된 스케일링 인수를 각 get_coefficient_*() 반환값에
   * 일괄 적용하여 최종 에너지 계수를 보정하는 메서드. iterative_lse()가 계산한
   * 최소제곱 해를 바탕으로 각 에너지 성분의 비중을 조정한다.
   *
   * 호출 체인:
   *   nonlinear_scale() → [coefficient_scale()]
   */
  void coefficient_scale();

  /*
   * [한국어] iterative_lse — 최소제곱법(LSE) 반복 피팅으로 스케일 계수 계산.
   *
   * @param1: double* — 관측값 벡터 (실측 전력 데이터 포인터).
   * @param2: double* — 모델 예측값 벡터 (계수 × 이벤트 카운터 결과 포인터).
   *
   * 실측 GPU 전력(AccelWattch 벤치마크 측정값)과 AccelWattch 모델 예측값 사이의
   * 오차를 최소화하는 스케일 계수를 최소제곱법으로 반복 계산한다. 수렴하면
   * 각 에너지 성분의 최적 스케일 계수가 결정된다.
   *
   * 호출 체인:
   *   nonlinear_scale() → [iterative_lse()] → 스케일 계수 → coefficient_scale()
   */
  void iterative_lse(double *, double *);

  /*
   * [한국어] ~Processor — 소멸자. 동적으로 할당된 모든 서브컴포넌트 해제.
   *
   * cores, l2array, l3array, l1dirarray, l2dirarray 벡터 내 각 포인터와
   * nocs 벡터, mc, niu, pcie, flashcontroller 포인터를 delete하여 메모리 누수를
   * 방지한다. ParseXML(XML)은 외부에서 소유하므로 여기서 해제하지 않는다.
   *
   * 호출 시점: accelwattch_interface가 소멸되거나 GPGPU-Sim 시뮬레이션 종료 시.
   */
  ~Processor();
};

#endif /* PROCESSOR_H_ */
