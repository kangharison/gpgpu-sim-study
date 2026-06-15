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
 * [한국어 설명] 공유 캐시 및 캐시 코히런스 디렉토리 전력 모델 헤더 (sharedcache.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU L2 공유 캐시(SharedCache)와 캐시 코히런스 디렉토리(CCdir)의
 * 전력/면적 모델을 정의하는 헤더이다. McPAT(Multi-Core Power, Area, and Timing)
 * 프레임워크의 일부로, SRAM 배열(DataCache)과 CacheDynParam을 조합하여
 * 캐시 접근당 동적 에너지와 누설 전력을 계산한다. AccelWattch에서는 GPU의
 * 통합 L2 캐시(all SM이 공유하는 온칩 SRAM) 전력 추정에 SharedCache를 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층:
 *   gpgpu_sim::cycle() → AccelWattch 카운터 수집
 *     → gpgpu_sim_wrapper::compute() [gpgpu_sim_wrapper.cc]
 *         → Processor::Processor() [processor.cc]
 *             → [이 파일] SharedCache 객체 생성
 *                 → SharedCache::computeEnergy(is_tdp)  (전력 계산)
 *                 → SharedCache::displayEnergy()         (결과 출력)
 * GPU 실행 컨텍스트: 호스트 유저스페이스 (시뮬레이터 사이클 단위 전력 추정).
 * SharedCache는 타이밍 시뮬레이션과 별도로 에너지를 사후 추정한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - XML_Parse.h / ParseXML  : gpgpusim.config에서 읽은 L2 파라미터 트리 제공
 *   - array.h / DataCache     : unicache(L2 SRAM 배열 — caches/missb/ifb/prefetchb/wbb) 보유
 *   - basic_components.h / statsDef : TDP/런타임 통계 구조체 (homenode 디렉토리 오버헤드)
 *   - cacti/parameter.h / InputParameter : CACTI SRAM 크기·연상도·라인 크기 파라미터
 *   - cacti/area.h            : 면적 계산 결과 저장
 *   - logic.h / CacheDynParam : XML에서 파싱된 캐시 동적 파라미터 (크기, 연상도, 대역폭)
 * 이 파일에 의존하는 모듈:
 *   - sharedcache.cc          : SharedCache/CCdir 구현체
 *   - processor.cc            : Processor가 SharedCache 객체를 직접 생성·소유
 *
 * === 주요 함수/구조체 요약 ===
 * SharedCache            : GPU L2 공유 캐시 전력 모델 클래스. unicache(DataCache)로
 *                          SRAM 배열을 모델링하고, computeEnergy()로 접근당 에너지 산출.
 * SharedCache::set_cache_param()   : XML에서 L2 캐시 파라미터(크기/연상도/라인)를 cachep에 로드.
 * SharedCache::computeEnergy()     : TDP 또는 런타임 통계를 이용해 동적 에너지 + 누설 전력 계산.
 * SharedCache::displayEnergy()     : 계산된 전력/면적 결과를 indent 수준에 맞춰 출력.
 * CCdir                  : 캐시 코히런스 디렉토리 전력 모델 클래스. GPU에서는 사용되지
 *                          않으며, CPU MOESI 디렉토리 구현 시에만 활성화된다.
 */

#ifndef SHAREDCACHE_H_
#define SHAREDCACHE_H_
#include <vector>
#include "XML_Parse.h"
#include "array.h"
#include "basic_components.h"
#include "cacti/area.h"
#include "cacti/parameter.h"
#include "logic.h"

class SharedCache : public Component {
 public:
  ParseXML* XML;
  /* [한국어] McPAT XML 설정 파싱 결과 트리 포인터.
   * 설정자: SharedCache 생성자에서 XML_interface 인자로 전달받아 저장.
   * 읽는 자: set_cache_param()이 XML->sys.L2[ithCache] 등으로 캐시 파라미터를 읽음.
   * 값 범위: 유효한 ParseXML 포인터 (NULL 불가). 소유권은 호출자(Processor)가 가짐.
   * 동기화: 단일 스레드(McPAT 전력 계산 컨텍스트)에서만 접근하므로 락 불필요. */

  int ithCache;
  /* [한국어] XML sys.L2[] 배열에서 이 SharedCache 인스턴스가 몇 번째 캐시인지를 나타내는 인덱스.
   * 설정자: 생성자 인자 ithCache_로 초기화.
   * 읽는 자: set_cache_param()이 XML->sys.L2[ithCache] 접근 시 사용.
   * 값 범위: 0 이상의 정수. GPU 모델에서는 L2 캐시가 보통 하나이므로 0이 일반적.
   * 동기화: 읽기 전용 (생성 후 불변). 별도 락 불필요. */

  InputParameter interface_ip;
  /* [한국어] CACTI SRAM 모델로 전달할 입력 파라미터 구조체.
   * 설정자: set_cache_param()이 XML에서 읽은 캐시 크기, 연상도(associativity),
   *         라인 크기, 읽기/쓰기 포트 수, 클럭 등을 채워 넣음.
   * 읽는 자: DataCache(unicache) 생성자 및 ArrayST 생성자가 SRAM 타이밍/면적 계산에 사용.
   * 값 범위: CACTI InputParameter 스펙에 따름 (cache_sz, nbanks, assoc 등).
   * 동기화: set_cache_param() 호출 후 unicache 초기화까지만 수정되며, 이후 읽기 전용. */

  enum cache_level cacheL;
  /* [한국어] 이 SharedCache가 모델링하는 캐시 계층 수준을 나타내는 열거형.
   * 설정자: 생성자 인자 cacheL_ (기본값 L2)로 초기화.
   * 읽는 자: computeEnergy()/set_cache_param()이 계층별 파라미터 분기에 사용.
   *          예: L2, L3, L1Directory, L2Directory 중 하나.
   * 값 범위: enum cache_level { L1, L2, L3, L1Directory, L2Directory } 범위.
   * 동기화: 읽기 전용 (생성 후 불변). 별도 락 불필요. */

  DataCache unicache;  // Shared cache
  /* [한국어] L2 공유 캐시의 SRAM 배열 모델을 담는 DataCache 객체.
   * 설정자: SharedCache 생성자에서 interface_ip와 cachep를 기반으로 초기화.
   *         내부에 caches(주 SRAM 배열), missb(미스 버퍼), ifb(채우기 버퍼),
   *         prefetchb(프리페치 버퍼), wbb(쓰기 버퍼) 등의 ArrayST 인스턴스를 포함.
   * 읽는 자: computeEnergy()가 unicache.caches->local_result에서 접근당 에너지,
   *          누설 전력, 면적을 읽어 전체 캐시 전력을 집계.
   * 값 범위: ArrayST 계산 결과(local_result.power, .area 등)로 채워진 상태.
   * 동기화: 단일 스레드 전력 계산 컨텍스트에서만 사용. 별도 락 불필요. */

  CacheDynParam cachep;
  /* [한국어] XML에서 파싱된 캐시 동적 파라미터 집합 (크기, 연상도, 라인 크기, 대역폭 등).
   * 설정자: set_cache_param()이 XML->sys.L2[ithCache]를 읽어 필드를 채움.
   * 읽는 자: computeEnergy()가 cachep.duty_cycle, cachep.throughput 등을 이용해
   *          런타임 접근 횟수와 에너지를 환산. unicache 초기화에도 전달됨.
   * 값 범위: XML 설정 파일(XML input)에 기재된 L2 크기/성능 파라미터 범위.
   * 동기화: set_cache_param() 완료 후 읽기 전용. 별도 락 불필요. */

  statsDef homenode_tdp_stats;
  /* [한국어] 캐시 코히런스 홈 노드(home node) 역할 수행 시의 TDP(Thermal Design Power) 통계.
   * 설정자: computeEnergy(is_tdp=true) 경로에서 디렉토리 오버헤드 계산 시 채워짐.
   * 읽는 자: displayEnergy()가 TDP 모드 결과 출력 시 참조.
   * 값 범위: 접근 횟수·에너지의 double 필드들. GPU에서는 dir_overhead=0이므로 실질적으로 0.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요. */

  statsDef homenode_rtp_stats;
  /* [한국어] 캐시 코히런스 홈 노드 역할 수행 시의 런타임 동적 전력(Runtime Power) 통계.
   * 설정자: computeEnergy(is_tdp=false) 경로에서 실제 접근 카운터 기반으로 채워짐.
   * 읽는 자: displayEnergy()가 런타임 전력 모드 결과 출력 시 참조.
   * 값 범위: GPGPU-Sim이 AccelWattch를 통해 전달한 실제 L2 접근 횟수 기반.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요. */

  statsDef homenode_stats_t;
  /* [한국어] TDP와 런타임 통계를 혼합한 임시 통계 버퍼 (계산 중간 상태 저장).
   * 설정자: computeEnergy() 내부에서 homenode_tdp_stats와 homenode_rtp_stats를
   *         조합할 때 임시로 사용.
   * 읽는 자: computeEnergy() 내에서만 참조. displayEnergy()는 직접 참조하지 않음.
   * 값 범위: double 필드들. 중간 계산 결과로만 의미를 가짐.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요. */

  double dir_overhead;
  /* [한국어] 디렉토리 프로토콜 오버헤드로 인한 추가 캐시 용량 비율.
   * 설정자: set_cache_param() 또는 생성자에서 XML의 dir_overhead 필드로 초기화.
   * 읽는 자: computeEnergy()가 홈 노드 디렉토리 에너지를 산출할 때 참조.
   * 값 범위: 0.0(디렉토리 없음/GPU 기본) ~ 1.0+ (디렉토리 태그 저장을 위한 추가 용량 비율).
   * 동기화: set_cache_param() 완료 후 읽기 전용. 별도 락 불필요. */

  //	cache_processor llCache,directory, directory1, inv_dir;

  // pipeline pipeLogicCache, pipeLogicDirectory;
  // clock_network				clockNetwork;
  double scktRatio, executionTime;
  /* [한국어] scktRatio: 소켓(칩) 단위 면적 스케일링 비율. 멀티소켓 시스템에서 에너지를
   *          전체 소켓 수로 나눠 per-socket 값을 구할 때 사용.
   *          executionTime: 전체 GPU 커널 실행 시간(초). 누설 전력(W)을 에너지(J)로
   *          변환할 때 곱해지는 값으로, GPGPU-Sim 시뮬레이션 종료 시 전달됨.
   * 설정자: 생성자 또는 computeEnergy()에서 XML의 sys.total_cycles / clock_rate 등으로 산출.
   * 읽는 자: computeEnergy()의 누설 에너지 계산 경로. displayEnergy()는 직접 참조하지 않음.
   * 값 범위: scktRatio > 0; executionTime > 0 (단위: 초).
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요. */

  //   Component L2Tot, cc, cc1, ccTot;

  SharedCache(ParseXML* XML_interface, int ithCache_,
              InputParameter* interface_ip_, enum cache_level cacheL_ = L2);
  void set_cache_param();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, bool is_tdp = true);
  ~SharedCache(){};
};

class CCdir : public Component {
 public:
  ParseXML* XML;
  /* [한국어] McPAT XML 설정 파싱 결과 트리 포인터.
   * 설정자: CCdir 생성자에서 XML_interface 인자로 전달받아 저장.
   * 읽는 자: computeEnergy()가 XML에서 디렉토리 파라미터를 읽을 때 사용.
   * 값 범위: 유효한 ParseXML 포인터 (NULL 불가). 소유권은 호출자(Processor)가 가짐.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요.
   * 참고: GPU에서는 CCdir이 사용되지 않으므로 이 포인터가 참조하는 데이터는
   *       GPU 시뮬레이션 경로에서 실질적으로 접근되지 않는다. */

  int ithCache;
  /* [한국어] XML sys.L2[] 배열에서 이 CCdir 인스턴스가 가리키는 캐시 인덱스.
   * 설정자: CCdir 생성자 인자 ithCache_로 초기화.
   * 읽는 자: computeEnergy() 내부에서 XML 캐시 파라미터 접근 시 사용.
   * 값 범위: 0 이상의 정수.
   * 동기화: 읽기 전용 (생성 후 불변). 별도 락 불필요. */

  InputParameter interface_ip;
  /* [한국어] CACTI SRAM 배열 모델링을 위한 입력 파라미터 구조체.
   * 설정자: CCdir 생성자에서 interface_ip_ 인자를 복사하여 초기화.
   * 읽는 자: dc(DataCache)와 shadow_dir(ArrayST) 초기화 시 전달됨.
   * 값 범위: CACTI InputParameter 스펙 범위 내.
   * 동기화: 생성 후 변경 없음. 별도 락 불필요. */

  DataCache dc;  // Shared cache
  /* [한국어] 캐시 코히런스 디렉토리의 태그 배열(tag array)을 모델링하는 DataCache 객체.
   * 설정자: CCdir 생성자에서 interface_ip와 XML 파라미터로 초기화.
   * 읽는 자: computeEnergy()가 dc.caches->local_result에서 접근당 에너지와 면적을 읽음.
   * 값 범위: CACTI가 계산한 SRAM 타이밍/에너지 결과로 채워진 DataCache 구조.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요.
   * 참고: GPU 시뮬레이션에서는 CCdir 자체가 사용되지 않으므로 dc도 비활성 상태. */

  ArrayST* shadow_dir;
  /* [한국어] 섀도우 디렉토리(shadow directory) — MOESI 프로토콜에서 태그 중복 저장을
   *          위해 추가되는 SRAM 배열 포인터.
   * 설정자: CCdir 생성자에서 new ArrayST(...)로 동적 할당.
   * 읽는 자: computeEnergy()가 shadow_dir->local_result에서 에너지/면적을 읽음.
   *          ~CCdir() 소멸자에서 delete로 해제.
   * 값 범위: 유효한 ArrayST 포인터 (생성 후 NULL 불가).
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요.
   * 참고: GPU에서는 CCdir을 사용하지 않으므로 shadow_dir도 실질적으로 비활성. */

  //	cache_processor llCache,directory, directory1, inv_dir;

  // pipeline pipeLogicCache, pipeLogicDirectory;
  // clock_network				clockNetwork;
  double scktRatio, clockRate, executionTime;
  /* [한국어] scktRatio: 소켓 단위 면적/전력 스케일링 비율 (멀티소켓 환경에서 per-socket 분리용).
   *          clockRate: 캐시 디렉토리 동작 클럭 주파수 (Hz). CACTI 타이밍 계산의 기준값.
   *          executionTime: 전체 실행 시간 (초). 누설 에너지 = 누설전력(W) × executionTime.
   * 설정자: CCdir 생성자 또는 computeEnergy()에서 XML의 clock_rate, total_cycles 등으로 산출.
   * 읽는 자: computeEnergy()의 누설 에너지 환산 경로.
   * 값 범위: 모두 양수 실수.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요. */

  Component L2Tot, cc, cc1, ccTot;
  /* [한국어] 전력/면적 집계용 Component 객체들.
   *   L2Tot : L2 캐시 전체 전력/면적 합계 (태그 배열 + 데이터 배열 합산 결과).
   *   cc    : 캐시 코히런스 디렉토리 1단계 전력/면적 소계.
   *   cc1   : 캐시 코히런스 디렉토리 2단계(예: L2Directory 상위 계층) 전력/면적 소계.
   *   ccTot : cc + cc1을 합산한 코히런스 오버헤드 총합 Component.
   * 설정자: computeEnergy()가 각 ArrayST::local_result에서 읽어 누적.
   * 읽는 자: displayEnergy()가 인덴트 수준에 맞춰 출력.
   * 값 범위: Component::power(W), Component::area(mm^2) 등 double 필드.
   * 동기화: 단일 스레드 전력 계산 컨텍스트. 별도 락 불필요.
   * 참고: GPU 시뮬레이션에서는 CCdir 자체가 사용되지 않으므로 이 필드들도 비활성. */

  CCdir(ParseXML* XML_interface, int ithCache_, InputParameter* interface_ip_);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, bool is_tdp = true);
  ~CCdir();
};

#endif /* SHAREDCACHE_H_ */
