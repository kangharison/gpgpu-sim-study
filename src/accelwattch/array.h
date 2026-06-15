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

/*
 * [한국어 설명] SRAM 배열 전력/면적 모델 헤더 (array.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 AccelWattch(McPAT 기반 GPU 전력 모델)에서 모든 SRAM 배열 구조를
 * 표현하는 핵심 클래스를 선언한다. ArrayST는 CACTI(Cache And Memory Circuit/Technology
 * Interface) 도구를 래핑하여 임의의 SRAM 배열에 대해 동적 에너지(dynamic energy),
 * 누설 전력(leakage power), 면적(area)을 계산하는 단일 인터페이스를 제공한다.
 * InstCache와 DataCache는 이를 기반으로 I-cache / D-cache 전체를 구성하는
 * 상위 컨테이너 클래스다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 가장 하위 레벨 빌딩 블록이다. GPU 내 모든 SRAM 구조
 * (L1 I-cache, L1 D-cache, Shared Memory, Register File, L2 Cache, TLB 등)는
 * ArrayST 인스턴스를 하나 이상 포함한다. 호출 체인은 다음과 같다:
 *   GPGPU-Sim 전력 계산 루프
 *     → ProcessorTemplate (accelwattch/) 또는 각 GPU 서브유닛(shader, l2, tlb 등)
 *         → ArrayST 생성자 → optimize_array() → CACTI(cacti_interface)
 *   시뮬레이션 중 온도 갱신:
 *     → leakage_feedback(temperature) → reconfigure() → CACTI 내부
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드 (시뮬레이션 초기화 및 전력 집계 단계)
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cacti/cacti_interface.h : CACTI 공개 API (cacti_interface(), reconfigure())
 *   - cacti/component.h       : 상위 클래스 Component (power, area 필드 보유)
 *   - cacti/const.h           : 상수 (BIGNUM 등)
 *   - cacti/parameter.h       : InputParameter, uca_org_t 구조체 (CACTI 입출력)
 *   - basic_components.h      : statsDef, powerDef, longer_channel_device_reduction()
 *   - globalvar.h             : g_tp (전역 기술 파라미터), opt_for_clk 플래그
 * 이 파일에 의존하는 모듈:
 *   - GPU 서브유닛 구현 파일들 (l2cache.cc, shader.cc 관련 전력 모델 등)
 *   - processor.cc (McPAT/AccelWattch 최상위 조립)
 * 데이터 흐름:
 *   InputParameter(l_ip) → CACTI → uca_org_t(local_result) → 전력/면적 스케일링
 *   → statsDef(tdp_stats, rtp_stats, stats_t), powerDef(power_t)
 *
 * === 주요 함수/구조체 요약 ===
 * ArrayST::ArrayST()        - SRAM 배열 초기화: l_ip 복사 → 에러 검사 → optimize_array()
 * ArrayST::optimize_array() - CACTI를 반복 호출해 타이밍 제약을 만족하면서
 *                             동적 에너지가 최소인 SRAM 설정을 탐색
 * ArrayST::compute_base_power() - CACTI를 단 한 번 호출해 현재 l_ip 설정의 전력/면적 계산
 * ArrayST::leakage_feedback()   - 시뮬레이션 중 온도 변화에 따라 누설 전력을 재계산
 * InstCache / DataCache     - 캐시 계층 전체를 구성하는 ArrayST 포인터 컨테이너
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <iostream>   // [한국어] cout (경고 출력용)
#include <string>     // [한국어] std::string (배열 이름 저장용)
#include "basic_components.h"        // [한국어] statsDef, powerDef, longer_channel_device_reduction 등 AccelWattch 공통 컴포넌트
#include "cacti/cacti_interface.h"   // [한국어] CACTI 공개 진입점: cacti_interface(), reconfigure()
#include "cacti/component.h"         // [한국어] Component 기반 클래스: power(powerDef), area(double) 필드 제공
#include "cacti/const.h"             // [한국어] BIGNUM 등 CACTI 내부 상수
#include "cacti/parameter.h"         // [한국어] InputParameter(CACTI 입력), uca_org_t(CACTI 출력 결과)

using namespace std;

/*
 * [한국어]
 * ArrayST - 임의 SRAM 배열의 전력/면적 모델 클래스
 *
 * CACTI를 래핑하여 GPU 내 모든 SRAM 구조(캐시, 레지스터 파일, TLB 등)의
 * 동적 에너지, 누설 전력, 면적을 계산한다. AccelWattch 전력 모델의
 * 원자적 빌딩 블록으로, 상위 GPU 서브유닛들이 이를 인스턴스로 보유한다.
 *
 * 상속: Component (cacti/component.h) — power(powerDef), area 필드 제공
 *
 * 생명주기:
 *   1. 생성자에서 l_ip 복사 → error_checking() → optimize_array() 호출
 *   2. 시뮬레이션 중 leakage_feedback()으로 온도별 누설 전력 갱신
 *   3. 소멸자에서 local_result.cleanup()으로 CACTI 동적 할당 메모리 해제
 */
class ArrayST : public Component {
 public:
  ArrayST(){};
  /* [한국어] 기본 생성자. 빈 ArrayST 객체를 생성한다.
   * 사용처: InstCache/DataCache 멤버 포인터 초기화 전 임시 사용 또는
   *         서브클래스가 별도 초기화 경로를 갖는 경우. */

  ArrayST(const InputParameter* configure_interface, string _name,
          enum Device_ty device_ty_, bool opt_local_ = true,
          enum Core_type core_ty_ = Inorder, bool _is_default = true);
  /* [한국어] 매개변수 있는 생성자. SRAM 배열을 완전히 초기화한다.
   * @configure_interface: CACTI 입력 파라미터 구조체 포인터.
   *   호출자(GPU 서브유닛)가 캐시 크기, 연관도, 뱅크 수 등 하드웨어 파라미터를
   *   설정하여 전달한다. l_ip에 복사본을 저장한다.
   * @_name: 이 배열의 이름 문자열 (경고 출력 및 디버깅용, 예: "L1_icache").
   * @device_ty_: 공정 기술 유형 (ITRS HP/LSTP/LOP 등). 누설 전력 축소 계수
   *   longer_channel_device_reduction()에 영향을 준다.
   * @opt_local_: true면 이 ArrayST 내부에서 타이밍 최적화 루프를 수행한다.
   *   false면 단순히 현재 l_ip로만 전력을 계산한다 (외부에서 이미 최적화된 경우).
   * @core_ty_: 코어 유형 (Inorder/OOO). 누설 전력 스케일링 계수에 영향.
   * @_is_default: 기본 설정 여부 플래그. McPAT 내부 분기에서 참조.
   * 내부 흐름: l_ip 복사 → cache_sz 하한 보정 → error_checking() → optimize_array() */

  InputParameter l_ip;
  /* [한국어] CACTI 입력 파라미터 구조체 (local input parameter).
   * 설정자: 생성자 초기화 리스트에서 *configure_interface로부터 복사 설정됨.
   *   optimize_array() 내부에서 weight/deviation 필드를 반복적으로 수정하며
   *   CACTI 탐색 공간을 조정한다.
   * 읽는 자: compute_base_power()가 cacti_interface(&l_ip)에 전달.
   *   leakage_feedback()이 l_ip.temp를 갱신한 후 reconfigure(&l_ip, ...)에 전달.
   * 값 범위: cache_sz는 최소 64바이트. nbanks, assoc, line_sz, throughput,
   *   latency 등 다양한 SRAM 구조 파라미터를 포함.
   * 동기화: 단일 스레드 환경에서만 접근 (시뮬레이션 초기화 / 전력 집계 단계). */

  string name;
  /* [한국어] 이 SRAM 배열의 식별 이름 문자열.
   * 설정자: 생성자 초기화 리스트에서 _name으로부터 복사.
   * 읽는 자: optimize_array()가 타이밍 제약 미충족 시 경고 메시지 출력에 사용.
   * 값 범위: 임의 문자열 (예: "L1_icache", "RF", "L2_cache").
   * 동기화: 읽기 전용으로만 사용되므로 별도 락 불필요. */

  enum Device_ty device_ty;
  /* [한국어] 반도체 공정 기술 유형 (Device Technology).
   * 설정자: 생성자 인자 device_ty_로부터 초기화 리스트에서 설정.
   * 읽는 자: optimize_array() 및 leakage_feedback()에서
   *   longer_channel_device_reduction(device_ty, core_ty) 호출 시 전달.
   * 값 범위: ITRS HP(High Performance), LSTP(Low Standby Power), LOP 등
   *   basic_components.h의 Device_ty enum 값.
   * 동기화: 초기화 후 읽기 전용. */

  bool opt_local;
  /* [한국어] 이 ArrayST 객체 내부에서 CACTI 타이밍 최적화 루프를 수행할지 여부.
   * 설정자: 생성자 인자 opt_local_로부터 초기화 리스트에서 설정.
   * 읽는 자: optimize_array()에서 opt_for_clk && opt_local 조건을 평가할 때 참조.
   * 값 범위: true(최적화 수행) / false(현재 l_ip 설정으로 단순 계산).
   * 동기화: 초기화 후 읽기 전용. */

  enum Core_type core_ty;
  /* [한국어] 이 SRAM 배열이 속한 코어 유형 (Core Type).
   * 설정자: 생성자 인자 core_ty_로부터 초기화 리스트에서 설정.
   * 읽는 자: optimize_array() 및 leakage_feedback()에서
   *   longer_channel_device_reduction(device_ty, core_ty) 호출 시 전달.
   * 값 범위: Inorder(순서형), OOO(비순서형) 등 Core_type enum 값.
   * 동기화: 초기화 후 읽기 전용. */

  bool is_default;
  /* [한국어] 이 배열이 기본(default) 설정을 사용하는지 여부.
   * 설정자: 생성자 인자 _is_default로부터 초기화 리스트에서 설정.
   * 읽는 자: McPAT/AccelWattch 내부의 조건 분기 로직에서 참조.
   * 값 범위: true(기본 설정) / false(사용자 지정 설정).
   * 동기화: 초기화 후 읽기 전용. */

  uca_org_t local_result;
  /* [한국어] CACTI 출력 결과 구조체 (Unified Cache Array organization result).
   * cacti_interface() 호출 결과를 저장하며, 전력/면적/타이밍 정보를 담는다.
   * 설정자: compute_base_power()가 cacti_interface(&l_ip) 반환값으로 설정.
   *   optimize_array()가 최적 후보 중 동적 에너지 최솟값을 찾아 덮어씀.
   *   leakage_feedback()에서 reconfigure() 후 전력 스케일링 결과로 갱신됨.
   * 읽는 자: optimize_array()에서 cycle_time, access_time, area,
   *   power.readOp.dynamic 등을 읽어 최적화 판단에 사용.
   *   상위 GPU 서브유닛이 power(dynamic/leakage) 및 area를 읽어 전력 집계.
   * 주요 내부 필드:
   *   .valid          : CACTI 결과가 유효한지 여부 (최적화 성공 시 true)
   *   .cycle_time     : SRAM 사이클 타임 (초 단위, throughput 제약 비교 대상)
   *   .access_time    : SRAM 접근 시간 (초 단위, latency 제약 비교 대상)
   *   .area           : 배열 면적 (m², 레이아웃 오버헤드 스케일링 후 최종값)
   *   .power          : 전체 배열 전력 (readOp/writeOp/searchOp × dynamic/leakage)
   *   .data_array2    : 데이터 배열 전력/면적 (포인터, CACTI 내부 할당)
   *   .tag_array2     : 태그 배열 전력/면적 (캐시 구조일 때만 유효, 포인터)
   * 동기화: 단일 스레드에서만 접근. 소멸자에서 cleanup()으로 내부 동적 메모리 해제. */

  statsDef tdp_stats;
  /* [한국어] TDP(Thermal Design Power) 기준 통계 카운터.
   * 설정자: 상위 GPU 서브유닛이 TDP 기준 활동도(activity factor)로 채움.
   * 읽는 자: 전력 집계 루틴이 TDP 전력 계산 시 stats_t와 함께 참조.
   * 값 범위: 읽기/쓰기 횟수 등 양의 정수 카운터.
   * 동기화: 단일 집계 스레드에서 접근. */

  statsDef rtp_stats;
  /* [한국어] RTP(Runtime Power) 기준 통계 카운터.
   * 설정자: GPGPU-Sim 시뮬레이션 사이클에서 실제 접근 횟수로 채움.
   * 읽는 자: 전력 집계 루틴이 런타임 동적 전력 계산 시 참조.
   * 값 범위: 읽기/쓰기 횟수 등 양의 정수 카운터.
   * 동기화: 단일 집계 스레드에서 접근. */

  statsDef stats_t;
  /* [한국어] 최종 전력 계산에 사용되는 통합 통계 카운터.
   * 설정자: 전력 집계 루틴이 tdp_stats 또는 rtp_stats를 선택적으로 복사.
   * 읽는 자: power_t 계산 루틴이 stats_t의 카운터를 기반으로 동적 전력을 스케일링.
   * 값 범위: tdp_stats 또는 rtp_stats와 동일 형식.
   * 동기화: 단일 집계 스레드에서 접근. */

  powerDef power_t;
  /* [한국어] 이 SRAM 배열의 최종 전력 값 (동적 + 누설, 단위: W).
   * 설정자: 전력 집계 루틴이 local_result.power를 stats_t로 스케일링하여 계산.
   * 읽는 자: 상위 GPU 서브유닛이 자신의 total_power에 누적.
   * 값 범위: 양수 (동적 전력은 접근 횟수 × 에너지/접근, 누설은 상시 소비).
   * 동기화: 단일 집계 스레드에서 접근. */

  virtual void optimize_array();
  /* [한국어] CACTI를 반복 호출해 타이밍 제약을 만족하는 최소 동적 에너지 설정을 찾는다.
   * 상세 설명은 array.cc 구현부 주석 참조. */

  virtual void compute_base_power();
  /* [한국어] 현재 l_ip 설정으로 CACTI를 한 번 호출해 전력/면적을 계산한다.
   * optimize_array()의 내부 루프에서 매 이터레이션마다 호출된다. */

  virtual ~ArrayST();
  /* [한국어] 소멸자. local_result.cleanup()을 호출해 CACTI가 동적 할당한
   * data_array2, tag_array2 등의 내부 메모리를 해제한다. */

  void leakage_feedback(double temperature);
  /* [한국어] 시뮬레이션 중 온도 변화에 따라 누설 전력을 재계산한다.
   * 상세 설명은 array.cc 구현부 주석 참조. */
};

/*
 * [한국어]
 * InstCache - 명령어 캐시(I-cache) 전체를 구성하는 ArrayST 컨테이너
 *
 * 실제 GPU I-cache 하드웨어는 주 캐시 배열 외에도 미스 상태 보유 레지스터(MSHR),
 * 인플라이트 버퍼(IFB), 프리페치 버퍼(prefetch buffer) 등 보조 SRAM 구조를 포함한다.
 * InstCache는 이들을 각각 ArrayST 포인터로 묶어 하나의 I-cache 전력 모델로 관리한다.
 * 상속: Component (power, area 필드 제공)
 */
class InstCache : public Component {
 public:
  ArrayST* caches;
  /* [한국어] 주 I-cache 데이터/태그 배열에 대한 ArrayST 포인터.
   * 설정자: 상위 GPU 서브유닛(I-cache 전력 모델 초기화 코드)에서 new ArrayST()로 생성.
   * 읽는 자: 전력 집계 루틴이 power_t 계산 시 caches->local_result.power를 참조.
   * 값 범위: 유효한 ArrayST 포인터, 또는 미초기화 시 nullptr(0).
   * 동기화: 초기화 후 읽기 전용. 소멸자에서 delete. */

  ArrayST* missb;
  /* [한국어] 미스 상태 보유 레지스터(MSHR, Miss Status Holding Register) 배열 포인터.
   * MSHR은 캐시 미스 발생 시 해당 미스의 메모리 요청 정보를 임시 보관하는
   * 소형 SRAM 구조다. 병렬 미스 처리(outstanding miss tracking)를 가능하게 한다.
   * 설정자: 상위 GPU 서브유닛에서 new ArrayST()로 생성.
   * 읽는 자: 전력 집계 루틴이 missb->local_result.power를 참조.
   * 값 범위: 유효한 ArrayST 포인터, 또는 nullptr(0).
   * 동기화: 초기화 후 읽기 전용. 소멸자에서 delete. */

  ArrayST* ifb;
  /* [한국어] 인플라이트 버퍼(IFB, In-Flight Buffer) 배열 포인터.
   * IFB는 캐시 미스로 메모리 계층에 내려간 요청이 돌아오기를 기다리는 동안
   * 해당 캐시 라인 위치 정보를 보관하는 소형 SRAM 구조다.
   * 설정자: 상위 GPU 서브유닛에서 new ArrayST()로 생성.
   * 읽는 자: 전력 집계 루틴이 ifb->local_result.power를 참조.
   * 값 범위: 유효한 ArrayST 포인터, 또는 nullptr(0).
   * 동기화: 초기화 후 읽기 전용. 소멸자에서 delete. */

  ArrayST* prefetchb;
  /* [한국어] 프리페치 버퍼(Prefetch Buffer) 배열 포인터.
   * 프리페치된 캐시 라인이 실제 수요 접근 전에 임시로 저장되는 소형 SRAM 구조다.
   * 설정자: 상위 GPU 서브유닛에서 new ArrayST()로 생성.
   * 읽는 자: 전력 집계 루틴이 prefetchb->local_result.power를 참조.
   * 값 범위: 유효한 ArrayST 포인터, 또는 nullptr(0).
   * 동기화: 초기화 후 읽기 전용. 소멸자에서 delete. */

  powerDef power_t;  // temp value holder for both (max) power and runtime power
  /* [한국어] I-cache 전체 전력(TDP 기준 또는 런타임 기준)을 임시로 보관하는 필드.
   * caches + missb + ifb + prefetchb 각각의 전력을 합산하여 저장한다.
   * 설정자: 전력 집계 루틴이 각 하위 ArrayST의 power_t를 합산해 이 필드에 저장.
   * 읽는 자: 상위 GPU 전력 모델이 I-cache 전체 전력으로 이 값을 참조.
   * 값 범위: 양수 (단위: W).
   * 동기화: 단일 집계 스레드에서 접근. */

  InstCache() {
    // [한국어] 기본 생성자: 모든 ArrayST 포인터를 nullptr(0)로 초기화한다.
    // 이후 상위 서브유닛이 필요한 포인터만 new ArrayST()로 채운다.
    caches = 0;    // [한국어] 주 캐시 배열 포인터 null 초기화
    missb = 0;     // [한국어] MSHR 배열 포인터 null 초기화
    ifb = 0;       // [한국어] 인플라이트 버퍼 포인터 null 초기화
    prefetchb = 0; // [한국어] 프리페치 버퍼 포인터 null 초기화
  };

  ~InstCache() {
    // [한국어] 소멸자: 초기화된 포인터만 골라 delete하고 nullptr로 다시 설정한다.
    // cleanup() 주석은 이전에 개별 cleanup이 필요했던 흔적이나 현재는 delete만 수행.
    if (caches) {  // caches->local_result.cleanup();
      // [한국어] caches가 유효(non-null)하면 ArrayST 소멸자가 local_result.cleanup()을 호출함
      delete caches;  // [한국어] 주 캐시 배열 ArrayST 해제 (내부적으로 CACTI 동적 메모리 해제 포함)
      caches = 0;     // [한국어] 댕글링 포인터 방지를 위해 nullptr로 재설정
    }
    if (missb) {  // missb->local_result.cleanup();
      // [한국어] missb가 유효하면 delete
      delete missb;  // [한국어] MSHR 배열 ArrayST 해제
      missb = 0;     // [한국어] 댕글링 포인터 방지
    }
    if (ifb) {  // ifb->local_result.cleanup();
      // [한국어] ifb가 유효하면 delete
      delete ifb;  // [한국어] 인플라이트 버퍼 ArrayST 해제
      ifb = 0;     // [한국어] 댕글링 포인터 방지
    }
    if (prefetchb) {  // prefetchb->local_result.cleanup();
      // [한국어] prefetchb가 유효하면 delete
      delete prefetchb;  // [한국어] 프리페치 버퍼 ArrayST 해제
      prefetchb = 0;     // [한국어] 댕글링 포인터 방지
    }
  };
};

/*
 * [한국어]
 * DataCache - 데이터 캐시(D-cache) 전체를 구성하는 ArrayST 컨테이너
 *
 * InstCache를 상속하여 caches/missb/ifb/prefetchb에 더해 쓰기 백 버퍼(WBB,
 * Write-Back Buffer)를 추가로 관리한다. D-cache는 쓰기 연산이 발생하므로
 * 더티 라인(dirty line)을 퇴출할 때 하위 메모리로 기록하기 위한 WBB가 필요하다.
 * 상속: InstCache → Component
 */
class DataCache : public InstCache {
 public:
  ArrayST* wbb;
  /* [한국어] 쓰기 백 버퍼(WBB, Write-Back Buffer) 배열 포인터.
   * D-cache에서 더티 캐시 라인이 퇴출될 때 하위 메모리(L2 등)로 기록될 때까지
   * 임시로 보관하는 소형 SRAM 구조다. I-cache에는 없고 D-cache에만 존재한다.
   * 설정자: 상위 GPU 서브유닛에서 new ArrayST()로 생성.
   * 읽는 자: 전력 집계 루틴이 wbb->local_result.power를 참조.
   * 값 범위: 유효한 ArrayST 포인터, 또는 nullptr(0).
   * 동기화: 초기화 후 읽기 전용. 소멸자에서 delete. */

  DataCache() { wbb = 0; };
  /* [한국어] 기본 생성자: wbb 포인터를 nullptr(0)로 초기화한다.
   * InstCache() 기본 생성자도 자동 호출되어 caches/missb/ifb/prefetchb도 0으로 초기화됨. */

  ~DataCache() {
    // [한국어] 소멸자: wbb가 유효하면 delete하고 nullptr로 재설정한다.
    // InstCache 소멸자는 자동으로 이어서 호출되어 나머지 포인터들을 해제함.
    if (wbb) {  // wbb->local_result.cleanup();
      // [한국어] wbb가 유효(non-null)하면 ArrayST 소멸자가 local_result.cleanup()을 호출함
      delete wbb;  // [한국어] 쓰기 백 버퍼 ArrayST 해제 (CACTI 동적 메모리 포함)
      wbb = 0;     // [한국어] 댕글링 포인터 방지를 위해 nullptr로 재설정
    }
  };
};

#endif /* TLB_H_ */
