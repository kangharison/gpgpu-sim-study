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
 * [한국어 설명] AccelWattch 전력 모델 기본 컴포넌트 헤더 (basic_components.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 AccelWattch GPU 전력 모델의 공통 데이터 타입과 파라미터 클래스를 정의한다.
 * McPAT(Hewlett-Packard) 전력 모델에서 파생되어 GPU 아키텍처(GDDR5/GDDR3, SM, NoC 등)에
 * 맞게 확장된 버전이다. 런타임 활동 통계(statsComponents, statsDef)와 하드웨어
 * 컴포넌트별 파라미터(CoreDynParam, CacheDynParam, DRAMParam, MCParam, NoCParam, ProcParam 등)를
 * 선언한다. 이 클래스들은 XML 설정 파일에서 읽은 하드웨어 스펙을 전력 계산 컴포넌트에
 * 전달하는 데이터 버스 역할을 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이션 계층:
 *   GPU 시뮬레이션 (gpgpu-sim/) → 성능 카운터 수집
 *       → accelwattch/gpuwattch.* (AccelWattch 진입점)
 *           → [이 파일에서 정의된 파라미터 클래스로 XML 파싱 후 전달]
 *               → ArrayST / MemorySystem / CoreDynamic (McPAT 기반 전력 계산)
 *                   → 최종 전력 소비 수치 출력
 * 이 파일의 클래스는 XML_Parse(XML_Parse.h)를 통해 gpgpusim.config / XML 설정을
 * 읽어 각 컴포넌트 전력 모델에 하드웨어 스펙을 제공한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 및 매 통계 출력 시점에 사용.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - XML_Parse.h / XML_Parse.cc: ParseXML 클래스로 gpgpusim config XML을 파싱
 *   - cacti/parameter.h: g_tp(전역 기술 파라미터, TechnologyParameter) — 특히
 *     g_tp.peri_global.long_channel_leakage_reduction 을 longer_channel_device_reduction()에서 참조
 * 이 파일에 의존하는 모듈:
 *   - basic_components.cc: 이 헤더에서 선언된 클래스/함수의 구현
 *   - gpuwattch.*, core.*, memorySystem.*, noc.*, etc.: 전력 계산 각 모듈이
 *     CoreDynParam / CacheDynParam / DRAMParam / MCParam / NoCParam / ProcParam 등을 사용
 * 데이터 흐름:
 *   XML 설정 → ParseXML → {CoreDynParam, CacheDynParam, ...} → 전력 계산 모듈
 *   시뮬레이터 카운터 → {statsComponents, statsDef} → 전력 계산 모듈 (동적 전력 활동 인수)
 *
 * === 주요 함수/구조체 요약 ===
 * - statsComponents: 캐시/메모리 구조에 대한 access/hit/miss 카운트를 저장하는 집계 구조체
 * - statsDef: readAc/writeAc/searchAc (각각 statsComponents)를 묶어 하나의 메모리 통계 단위를 구성
 * - longer_channel_device_reduction(): 긴 채널 소자(longer-channel device)의 누설 전류
 *   감소 인수를 계산한다. OOO 코어 56%, Inorder 80%, Uncore 82%, LLC 100% 적용.
 * - CoreDynParam: SM(코어) 동적 파라미터 — 클럭, 파이프라인 폭, 레지스터 파일 크기, duty cycle
 * - CacheDynParam: 캐시 지오메트리·타이밍 파라미터 (용량, 연관도, 뱅크 수, 처리량, 레이턴시)
 * - DRAMParam: GDDR5/GDDR3 IDD 전류·전압·타이밍 파라미터 (데이터시트 기반)
 * - MCParam: 메모리 컨트롤러 파라미터 (버스 폭, 채널 수, 데이터 레이트)
 * - NoCParam: 네트워크온칩 파라미터 (플릿 크기, 포트, 가상 채널, 토폴로지 노드 수)
 * - ProcParam: 전체 프로세서 파라미터 (코어 수, L2/L3/NoC 수, 동질성 플래그)
 * - NIUParam / PCIeParam: I/O 컨트롤러(NIU: Network Interface Unit, PCIe) 파라미터
 */

#ifndef BASIC_COMPONENTS_H_
#define BASIC_COMPONENTS_H_

#include <vector>       // [한국어] std::vector — 일부 파라미터 리스트에 사용
#include "XML_Parse.h"  // [한국어] ParseXML 클래스 — gpgpusim config XML 파싱용
#include "cacti/parameter.h"  // [한국어] g_tp (TechnologyParameter 전역) — 기술 노드별 물리 파라미터 접근

const double cdb_overhead = 1.1;
/* [한국어] CDB(Common Data Bus, 결과 버스) 오버헤드 계수.
 * 실제 CDB 회선이 차지하는 전력/면적을 10% 추가로 반영하기 위해 1.1을 곱한다.
 * 설정자: 이 파일 초기화 시 정적으로 설정 (상수).
 * 읽는 자: CoreDynamic 등 파이프라인 CDB 전력 계산 모듈.
 * 값 범위: 고정 1.1 (McPAT 원본에서 유래한 경험적 값).
 * 동기화: 상수이므로 동기화 불필요. */

/* [한국어] FU_type — 파이프라인 기능 유닛(Functional Unit) 종류 열거형.
 * FPU: 부동소수점 유닛 (Floating-Point Unit) — float/double 연산 담당.
 * ALU: 정수 산술·논리 유닛 (Arithmetic Logic Unit) — 정수 add/sub/shift/bitwise 담당.
 * MUL: 정수 곱셈 유닛 (Multiply Unit) — ALU와 별도 파이프라인으로 구현되는 경우 구분.
 * 읽는 자: CoreDynParam 생성자가 XML에서 파싱 후, CoreDynamic 전력 계산 모듈이 유닛별 duty cycle 적용에 사용. */
enum FU_type { FPU, ALU, MUL };

/* [한국어] Core_type — CPU/GPU 코어의 파이프라인 종류 열거형.
 * OOO(Out-Of-Order): 비순서 실행 코어 — 더 많은 예측·재명명 하드웨어 → 더 큰 누설 전류.
 *   AccelWattch에서 OOO SM: longer_channel_device_percentage_core = 0.56 (Xeon Tulsa 기준).
 * Inorder: 순서 실행 코어 — 단순한 파이프라인 → 긴 채널 소자 비율 높음 (0.80, Niagara 기준).
 * 읽는 자: longer_channel_device_reduction() 함수가 누설 전류 감소 인수 계산 시 분기에 사용. */
enum Core_type { OOO, Inorder };

/* [한국어] Renaming_type — 레지스터 재명명(Register Renaming) 구현 방식 열거형.
 * RAMbased: RAM 기반 재명명 테이블 — 정적 배열로 구현, 면적 효율적.
 * CAMbased: CAM(Content Addressable Memory) 기반 — 태그 검색 가능, 더 유연하나 전력 소모 큼.
 * 읽는 자: CoreDynParam, OOO 코어 재명명 하드웨어 전력 계산 모듈. */
enum Renaming_type { RAMbased, CAMbased };

/* [한국어] Scheduler_type — 이슈 스케줄러 구현 방식 열거형.
 * PhysicalRegFile: 물리 레지스터 파일 방식 — 실제 레지스터에 결과 저장, 재명명 테이블 사용.
 * ReservationStation: 예약 스테이션 방식 — Tomasulo 알고리즘 기반, 결과를 RS 내부에 저장.
 * 읽는 자: CoreDynParam, 이슈 큐 전력 계산 시 구현 방식 분기에 사용. */
enum Scheduler_type { PhysicalRegFile, ReservationStation };

/* [한국어] cache_level — 캐시 계층(Cache Hierarchy) 레벨 열거형.
 * L2: 2차 캐시 (Last Level Cache보다 상위, SM 로컬 또는 공유 L2).
 * L3: 3차 캐시 (LLC, Last Level Cache).
 * L1Directory: L1 수준 디렉토리 캐시 (코히런스 디렉토리).
 * L2Directory: L2 수준 디렉토리 캐시.
 * 읽는 자: CacheDynParam 생성자, 캐시 레벨별 전력 모델 파라미터 설정에 사용. */
enum cache_level { L2, L3, L1Directory, L2Directory };

/* [한국어] MemoryCtrl_type — 메모리 컨트롤러 종류 열거형.
 * MC: 일반 DRAM 메모리 컨트롤러 (GDDR5/GDDR3 등).
 * FLASHC: 플래시 메모리 컨트롤러 (NAND Flash 등 비휘발성 메모리 제어).
 * 읽는 자: MCParam 생성자, 컨트롤러 종류에 따른 전력 계산 분기에 사용. */
enum MemoryCtrl_type {
  MC,     // memory controller
  FLASHC  // flash controller
};

/* [한국어] Dram_type — DRAM 종류 열거형.
 * GDDR5: Graphics DDR5 — 최신 GPU 메모리 타입, 더 높은 대역폭과 전력 소모.
 *   detailed_dram_model = 1일 때 idd 전류 기반 상세 모델 적용 가능.
 * GDDR3: Graphics DDR3 — 이전 세대 GPU 메모리 타입, 경험적(empirical) 모델 사용.
 * 읽는 자: DRAMParam 생성자, DRAM 전력 계산 모듈에서 모델 선택 시 사용. */
enum Dram_type { GDDR5, GDDR3 };

/* [한국어] Dir_type — 캐시 디렉토리 구현 방식 열거형.
 * ST(Shadowed Tag): 별도 태그 배열로 디렉토리 구현 — 태그 중복 저장.
 * DC(Directory Cache): 전용 디렉토리 캐시 구조.
 * SBT(Static Bank Tag): 정적 뱅크 태그 방식 디렉토리.
 * NonDir: 디렉토리 없음 — 스누핑(snooping) 프로토콜 등 비디렉토리 코히런스.
 * 읽는 자: CacheDynParam, 디렉토리 전력 계산 모듈. */
enum Dir_type {
  ST,   // shadowed tag
  DC,   // directory cache
  SBT,  // static bank tag
  NonDir
};

/* [한국어] Cache_policy — 캐시 쓰기 정책 열거형.
 * Write_through: 쓰기 관통 — 캐시와 하위 메모리를 동시에 갱신, 쓰기 버퍼 불필요하나 대역폭 소모 큼.
 * Write_back: 쓰기 후 기록 — 캐시에만 쓰고 교체 시 하위 메모리로 반영, dirty bit 필요.
 * 읽는 자: CacheDynParam, 쓰기 버퍼(write buffer) 전력 계산 시 분기에 사용. */
enum Cache_policy { Write_through, Write_back };

/* [한국어] Device_ty — 소자(Device) 위치/역할 범주 열거형.
 * AccelWattch에서 longer_channel_device_reduction() 함수가 이 값을 기반으로
 * 누설 전류 감소 인수를 다르게 계산한다.
 * Core_device: SM(코어) 내부 소자 — OOO 56% / Inorder 80% 긴 채널 소자 비율.
 * Uncore_device: L2 캐시, NoC 등 코어 외부 공유 자원 — 82% 긴 채널 소자 비율.
 * LLC_device: LLC(Last Level Cache) — 100% 긴 채널 소자 비율 (완전 적용).
 * 읽는 자: longer_channel_device_reduction(), 각 컴포넌트 누설 전력 보정에 사용. */
enum Device_ty { Core_device, Uncore_device, LLC_device };

/*
 * [한국어] statsComponents — 캐시/메모리 구조의 접근 활동 통계 컴포넌트.
 *
 * 이 클래스는 하나의 메모리 접근 유형(읽기, 쓰기, 검색 중 하나)에 대해
 * access(총 접근), hit(캐시 적중), miss(캐시 미스) 세 카운터를 묶어 관리한다.
 * AccelWattch에서 각 메모리 구조의 동적 전력(dynamic power)을 계산할 때
 * 활동 인수(activity factor)로 사용된다. operator+ 로 여러 SM의 카운터를 합산하고,
 * operator* 로 각 카운터에 독립적인 가중치(에너지 계수 배열)를 곱한다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 매 통계 출력 주기(stat dump interval)마다
 *   GPGPU-Sim 성능 카운터로부터 값이 채워진다.
 */
class statsComponents {
 public:
  double access;
  /* [한국어] 해당 메모리 구조에 대한 총 접근 횟수 (access count).
   * 설정자: GPGPU-Sim 성능 카운터에서 매 stat dump 시 복사된다.
   *   AccelWattch 진입점(gpuwattch.*)이 시뮬레이터 내부 stat을 이 필드에 기록.
   * 읽는 자: 전력 계산 모듈(ArrayST, MemorySystem 등)이 동적 전력 계산 시
   *   에너지 계수와 곱하는 활동 인수로 사용.
   * 값 범위: 0 이상 실수 (누적 카운트 또는 평균 접근률).
   * 동기화: 단일 스레드 (AccelWattch 전력 계산은 시뮬레이션과 순차 실행). */

  double hit;
  /* [한국어] 해당 메모리 구조에서 캐시 적중(hit) 횟수.
   * 설정자: GPGPU-Sim 캐시 통계(L1/L2 hit 카운터)로부터 AccelWattch 진입점이 기록.
   * 읽는 자: 캐시 동적 전력 = hit_energy * hit + miss_energy * miss 형태로 계산.
   * 값 범위: 0 이상, access 이하 실수.
   * 동기화: 단일 스레드. */

  double miss;
  /* [한국어] 해당 메모리 구조에서 캐시 미스(miss) 횟수.
   * 설정자: GPGPU-Sim 캐시 통계(L1/L2 miss 카운터)로부터 AccelWattch 진입점이 기록.
   *   일반적으로 miss = access - hit 관계가 성립하나, 별도로 추적되기도 한다.
   * 읽는 자: 캐시 미스 시 하위 계층 접근 에너지 계산에 사용.
   * 값 범위: 0 이상, access 이하 실수.
   * 동기화: 단일 스레드. */

  statsComponents() : access(0), hit(0), miss(0) {}
  /* [한국어] 기본 생성자 — 모든 카운터를 0으로 초기화한다.
   * 호출 시점: statsDef 생성 시 멤버 초기화, reset() 호출 대신 새 객체 생성 시. */

  statsComponents(const statsComponents &obj) { *this = obj; }
  /* [한국어] 복사 생성자 — 대입 연산자를 재사용해 깊은 복사를 수행한다.
   * operator+, operator* 의 반환값(값 의미론) 복사에 사용된다. */

  statsComponents &operator=(const statsComponents &rhs) {
    /* [한국어] 대입 연산자 — rhs의 세 카운터를 this에 복사한다. */
    access = rhs.access;  // [한국어] 총 접근 횟수 복사
    hit = rhs.hit;        // [한국어] 적중 횟수 복사
    miss = rhs.miss;      // [한국어] 미스 횟수 복사
    return *this;         // [한국어] 연쇄 대입(a = b = c)을 지원하기 위해 자신 반환
  }

  void reset() {
    /* [한국어] 모든 카운터를 0으로 리셋한다.
     * 매 stat dump 주기 시작 시 이전 구간 누적값을 지우고 새로 측정하기 위해 호출된다. */
    access = 0;  // [한국어] 총 접근 횟수 초기화
    hit = 0;     // [한국어] 적중 횟수 초기화
    miss = 0;    // [한국어] 미스 횟수 초기화
  }

  friend statsComponents operator+(const statsComponents &x,
                                   const statsComponents &y);
  /* [한국어] 두 statsComponents를 더하는 전역 friend 연산자.
   * 여러 SM 또는 여러 캐시 뱅크의 카운터를 합산할 때 사용된다.
   * 구현: basic_components.cc의 operator+(statsComponents, statsComponents). */

  friend statsComponents operator*(const statsComponents &x,
                                   double const *const y);
  /* [한국어] statsComponents에 double 배열(크기 3)을 곱하는 전역 friend 연산자.
   * y[0]=access 가중치, y[1]=hit 가중치, y[2]=miss 가중치 를 각 카운터에 곱한다.
   * 에너지 계수 배열과 활동 카운터를 곱해 컴포넌트별 에너지를 계산할 때 사용된다.
   * 구현: basic_components.cc의 operator*(statsComponents, double*). */
};

/*
 * [한국어] statsDef — 메모리 구조의 읽기/쓰기/검색 접근 통계를 묶는 컨테이너 클래스.
 *
 * 하나의 메모리 구조(캐시 뱅크, 레지스터 파일, CAM 등)에 대한 세 가지 접근 유형—
 * 읽기(readAc), 쓰기(writeAc), 검색(searchAc) — 을 각각 statsComponents로 보유한다.
 * ArrayST, MemorySystem, NoC 등 전력 계산 클래스가 이 통계를 입력으로 받아
 * 에너지 = Σ (접근횟수 × 에너지계수) 형태의 동적 전력을 산출한다.
 * operator+ 로 여러 인스턴스의 통계를 합산하고, operator* 로 에너지 계수 배열을 적용한다.
 * 실행 컨텍스트: 호스트 유저스페이스, AccelWattch 전력 계산 흐름 내에서 사용.
 */
class statsDef {
 public:
  statsComponents readAc;
  /* [한국어] 읽기 접근(read access) 통계 — access/hit/miss 세 카운터 보유.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim L1/L2 읽기 카운터를 여기에 기록.
   * 읽는 자: ArrayST::computeEnergy() 등이 읽기 에너지 계산 시 사용.
   * 값 범위: 각 필드 0 이상 실수.
   * 동기화: 단일 스레드 (AccelWattch 전력 계산과 시뮬레이터 메인 루프는 순차 실행). */

  statsComponents writeAc;
  /* [한국어] 쓰기 접근(write access) 통계 — access/hit/miss 세 카운터 보유.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim L1/L2 쓰기 카운터를 여기에 기록.
   * 읽는 자: ArrayST::computeEnergy() 등이 쓰기 에너지 계산 시 사용.
   * 값 범위: 각 필드 0 이상 실수.
   * 동기화: 단일 스레드. */

  statsComponents searchAc;
  /* [한국어] 검색 접근(search access) 통계 — CAM(Content Addressable Memory),
   *   TLB, 분기 예측기 등 연관 검색 구조의 검색 횟수를 보유.
   * 설정자: AccelWattch 진입점이 해당 구조의 검색 카운터를 여기에 기록.
   *   캐시처럼 hit/miss 구분이 의미 없는 경우 searchAc.access 만 사용될 수 있다.
   * 읽는 자: ArrayST::computeEnergy() 검색 에너지 계산 시 사용.
   * 값 범위: 각 필드 0 이상 실수.
   * 동기화: 단일 스레드. */

  statsDef() : readAc(), writeAc(), searchAc() {}
  /* [한국어] 기본 생성자 — readAc/writeAc/searchAc 모두 0으로 초기화.
   * 각 statsComponents의 기본 생성자가 자동 호출된다. */

  void reset() {
    /* [한국어] 세 접근 유형의 모든 카운터를 0으로 리셋한다.
     * 매 stat dump 주기 초기화 또는 새 측정 구간 시작 시 호출된다. */
    readAc.reset();   // [한국어] 읽기 통계 초기화
    writeAc.reset();  // [한국어] 쓰기 통계 초기화
    searchAc.reset(); // [한국어] 검색 통계 초기화
  }

  friend statsDef operator+(const statsDef &x, const statsDef &y);
  /* [한국어] 두 statsDef를 더하는 전역 friend 연산자.
   * readAc, writeAc, searchAc 각각을 statsComponents::operator+로 합산한다.
   * 여러 SM의 전체 통계를 하나로 합칠 때 사용된다. */

  friend statsDef operator*(const statsDef &x, double const *const y);
  /* [한국어] statsDef에 double 배열을 곱하는 전역 friend 연산자.
   * readAc, writeAc, searchAc 각각에 동일한 y 배열을 statsComponents::operator*로 적용한다.
   * 에너지 계수 배열을 전체 statsDef에 한꺼번에 적용할 때 사용된다. */
};

/*
 * [한국어]
 * longer_channel_device_reduction - 긴 채널 소자(longer-channel device)의 누설 전류 감소 인수 계산.
 *
 * @device_ty: 소자가 속한 위치 범주 (Core_device / Uncore_device / LLC_device).
 *   기본값 Core_device — SM 내부 소자.
 * @core_ty: 코어 파이프라인 종류 (OOO / Inorder). device_ty == Core_device 일 때만 유효.
 *   기본값 Inorder — 순서 실행 코어.
 * @return: 누설 전류 감소 인수 (0.0~1.0 실수).
 *   반환값이 작을수록 긴 채널 소자 비율이 높아 누설 전류가 많이 감소함을 의미한다.
 *   호출자(CoreDynamic, MemorySystem 등)는 이 값을 기준 누설 전력에 곱해 보정된 누설 전력을 얻는다.
 *
 * 긴 채널 소자(longer-channel device)는 표준 최소 채널 길이보다 긴 트랜지스터로,
 * 임계 전압이 높아 누설 전류가 적다. 고성능 소자(OOO 코어)일수록 짧은 채널(빠른) 소자
 * 비율이 높고, 순서 실행/공유 자원일수록 긴 채널 소자 비율이 높다.
 * g_tp.peri_global.long_channel_leakage_reduction은 기술 노드별 긴 채널 소자의
 * 누설 감소 비율을 CACTI 파라미터로 제공한다.
 * 최종 인수 = (짧은 채널 비율) + (긴 채널 비율 × 긴채널_누설_감소비율)
 *   Core OOO:   56% 긴 채널 → longer_channel_device_percentage_core = 0.56
 *   Core Inorder: 80% 긴 채널 → longer_channel_device_percentage_core = 0.80
 *   Uncore:     82% 긴 채널 → longer_channel_device_percentage_uncore = 0.82
 *   LLC:        100% 긴 채널 → longer_channel_device_percentage_llc = 1.0
 *
 * 호출 체인:
 *   CoreDynamic::computeStaticPower() / MemorySystem::computeStaticPower()
 *   → [longer_channel_device_reduction()] → g_tp.peri_global.long_channel_leakage_reduction
 */
double longer_channel_device_reduction(enum Device_ty device_ty = Core_device,
                                       enum Core_type core_ty = Inorder);

/*
 * [한국어] CoreDynParam — SM(Streaming Multiprocessor) 코어의 동적 파라미터 집합 클래스.
 *
 * 이 클래스는 AccelWattch의 CoreDynamic 전력 모델이 필요로 하는 SM 하드웨어 스펙을 보유한다.
 * ParseXML을 통해 gpgpusim config XML에서 파싱된 값으로 초기화되며,
 * 클럭 속도, 파이프라인 폭(fetch/decode/issue/commit), 레지스터 파일 크기,
 * 기능 유닛 수, 각 유닛의 duty cycle 등을 포함한다.
 * 이 파라미터들은 SM의 IFU(명령어 패치 유닛), ALU, FPU, MUL, LSU 각 유닛의
 * 동적 전력과 누설 전력을 계산하는 데 사용된다.
 * GPGPU-Sim의 SM에 대응하며, McPAT의 CPU Core 파라미터 구조를 GPU용으로 확장한 형태이다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성 후 읽기 전용.
 */
class CoreDynParam {
 public:
  CoreDynParam(){};
  /* [한국어] 기본 생성자 — 필드 초기화 없이 빈 객체 생성. CoreDynParam(ParseXML*, int)로 초기화 필요. */

  CoreDynParam(ParseXML *XML_interface, int ithCore_);
  /* [한국어] XML 파싱 생성자 — XML_interface에서 ithCore_번째 코어의 파라미터를 읽어 필드를 초기화.
   * 구현: basic_components.cc (AccelWattch 측 구현 또는 gpuwattch.cc 측 호출). */

  //    :XML(XML_interface),
  //     ithCore(ithCore_)
  //     core_ty(inorder),
  //     rm_ty(CAMbased),
  //     scheu_ty(PhysicalRegFile),
  //     clockRate(1e9),//1GHz
  //     arch_ireg_width(32),
  //     arch_freg_width(32),
  //     phy_ireg_width(128),
  //     phy_freg_width(128),
  //     perThreadState(8),
  //     globalCheckpoint(32),
  //     instructionLength(32){};
  // ParseXML * XML;
  /* [한국어] 위 주석은 생성자 초기화 목록 예시 (비활성화된 주석). 실제 초기화는 구현부에서 수행. */

  bool opt_local;
  /* [한국어] 로컬 최적화 활성화 플래그.
   * 설정자: ParseXML 생성자가 XML의 opt_local 필드에서 읽어 설정.
   * 읽는 자: CoreDynamic — 로컬 최적화 적용 여부에 따라 전력 모델 선택.
   * 값 범위: true(최적화 활성) / false(비활성).
   * 동기화: 초기화 후 읽기 전용. */

  bool x86;
  /* [한국어] x86 ISA 사용 여부 플래그.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic — x86의 가변 길이 명령어 디코더 전력 모델 적용 분기.
   *   GPGPU-Sim에서는 PTX/SASS 기반이므로 일반적으로 false.
   * 값 범위: true(x86) / false(비x86).
   * 동기화: 초기화 후 읽기 전용. */

  bool Embedded;
  /* [한국어] 임베디드 코어 여부 플래그.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic — 임베디드 코어용 소형 전력 모델 선택 시 참조.
   * 값 범위: true(임베디드) / false(데스크톱/서버급).
   * 동기화: 초기화 후 읽기 전용. */

  enum Core_type core_ty;
  /* [한국어] 코어 파이프라인 종류 (OOO / Inorder).
   * 설정자: ParseXML 생성자 — XML의 pipeline_type 필드에서 파싱.
   * 읽는 자: longer_channel_device_reduction() 누설 인수 계산, CoreDynamic 전력 모델 분기.
   * 값 범위: OOO(비순서 실행, GPU 텐서코어 등) / Inorder(순서 실행, 단순 SM).
   * 동기화: 초기화 후 읽기 전용. */

  enum Renaming_type rm_ty;
  /* [한국어] 레지스터 재명명 방식 (RAMbased / CAMbased).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic 재명명 하드웨어 전력 계산 — RAM vs CAM 면적/전력 차이 반영.
   * 값 범위: RAMbased / CAMbased.
   * 동기화: 초기화 후 읽기 전용. */

  enum Scheduler_type scheu_ty;
  /* [한국어] 이슈 스케줄러 구현 방식 (PhysicalRegFile / ReservationStation).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic 이슈 큐 전력 계산 — 구현 방식에 따른 전력 모델 선택.
   * 값 범위: PhysicalRegFile / ReservationStation.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate, executionTime;
  /* [한국어] clockRate: SM 코어 클럭 주파수 (Hz 단위, 예: 1e9 = 1GHz).
   * 설정자: ParseXML 생성자 — XML의 clock_rate를 GHz 단위에서 Hz로 변환해 저장.
   * 읽는 자: CoreDynamic — 동적 전력 계산 시 스위칭 활동(α·C·V²·f)의 f(클럭)에 사용.
   * 값 범위: 양수 실수 (전형적으로 700e6~2e9, 700MHz~2GHz).
   * 동기화: 초기화 후 읽기 전용.
   *
   * executionTime: 시뮬레이션 총 실행 시간 (초 단위).
   * 설정자: ParseXML 생성자 — total_cycles / clockRate 로 계산.
   * 읽는 자: CoreDynamic — 전력(W) = 에너지(J) / 시간(s) 변환에 사용.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  int arch_ireg_width, arch_freg_width, phy_ireg_width, phy_freg_width;
  /* [한국어] 아키텍처/물리 레지스터 파일 엔트리 비트 폭.
   * arch_ireg_width: 아키텍처 정수 레지스터 폭 (비트, 예: 32).
   * arch_freg_width: 아키텍처 부동소수점 레지스터 폭 (비트, 예: 32 또는 64).
   * phy_ireg_width: 물리 정수 레지스터 파일 폭 (재명명 후, 예: 128).
   * phy_freg_width: 물리 부동소수점 레지스터 파일 폭 (예: 128).
   * 설정자: ParseXML 생성자 — XML의 각 register_width 필드.
   * 읽는 자: CoreDynamic 레지스터 파일 배열 크기 계산 → CACTI ArrayST 호출 시 사용.
   * 값 범위: 양의 정수 (비트 단위).
   * 동기화: 초기화 후 읽기 전용. */

  int num_IRF_entry, num_FRF_entry, num_ifreelist_entries,
      num_ffreelist_entries;
  /* [한국어] 레지스터 파일 엔트리 수 및 프리리스트(free list) 크기.
   * num_IRF_entry: 정수 레지스터 파일(IRF) 엔트리 수 (물리 레지스터 개수).
   * num_FRF_entry: 부동소수점 레지스터 파일(FRF) 엔트리 수.
   * num_ifreelist_entries: 정수 레지스터 프리리스트 엔트리 수 (사용 가능한 물리 레지스터 추적).
   * num_ffreelist_entries: 부동소수점 레지스터 프리리스트 엔트리 수.
   * 설정자: ParseXML 생성자 — XML의 register_file_size 계열 필드.
   * 읽는 자: CoreDynamic 레지스터 파일 + 프리리스트 배열 면적/전력 계산.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int fetchW, decodeW, issueW, peak_issueW, commitW, peak_commitW, predictionW,
      fp_issueW, fp_decodeW;
  /* [한국어] SM 파이프라인 단계별 폭(width, 한 사이클에 처리하는 명령어 수).
   * fetchW: 명령어 패치(fetch) 폭 — 한 사이클에 패치하는 명령어 수.
   * decodeW: 디코드(decode) 폭.
   * issueW: 정수 이슈(issue) 폭 — 한 사이클에 정수 유닛으로 디스패치하는 명령어 수.
   * peak_issueW: 최대 이슈 폭 (피크 성능 기준).
   * commitW: 커밋(commit/retire) 폭.
   * peak_commitW: 최대 커밋 폭.
   * predictionW: 분기 예측 폭.
   * fp_issueW: 부동소수점 이슈 폭.
   * fp_decodeW: 부동소수점 디코드 폭.
   * 설정자: ParseXML 생성자 — XML의 pipeline_width 계열 필드.
   * 읽는 자: CoreDynamic — IFU/Scheduler/FPU 배열 크기 → CACTI 호출 파라미터.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int perThreadState, globalCheckpoint, instruction_length, pc_width,
      opcode_length, micro_opcode_length;
  /* [한국어] 스레드 상태 및 명령어 포맷 파라미터.
   * perThreadState: 스레드당 HW 상태 레지스터 크기 (바이트, SMT에서 컨텍스트 스위치 비용).
   * globalCheckpoint: 전체 체크포인트(ROB 스냅샷) 수 — OOO 복구 구조 크기.
   * instruction_length: 명령어 길이 (비트, PTX/SASS는 보통 32 또는 64).
   * pc_width: 프로그램 카운터(PC) 비트 폭.
   * opcode_length: 오피코드 필드 길이 (비트).
   * micro_opcode_length: 마이크로-오피코드 길이 (비트, 복잡 명령어 분해 시).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic IFU, 디코더, ROB 배열 크기 계산.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int num_hthreads, pipeline_stages, fp_pipeline_stages, num_pipelines,
      num_fp_pipelines;
  /* [한국어] 하드웨어 스레드 수 및 파이프라인 구조.
   * num_hthreads: 하드웨어 스레드(SMT) 수 — GPU SM에서는 동시 실행 warp 수에 대응.
   * pipeline_stages: 정수 파이프라인 단계 수 (CPI 계산에 사용).
   * fp_pipeline_stages: 부동소수점 파이프라인 단계 수 (FP 레이턴시).
   * num_pipelines: 정수 연산 파이프라인 수 (병렬 ALU 수).
   * num_fp_pipelines: 부동소수점 파이프라인 수 (병렬 FPU 수).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic 파이프라인 전력 계산 — 스테이지·파이프라인 수에 비례.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int num_alus, num_muls;
  /* [한국어] 정수 ALU 수 및 곱셈기(MUL) 수.
   * num_alus: SM 내 정수 ALU(덧셈/논리/비교) 유닛 수.
   * num_muls: SM 내 정수 곱셈기(MUL) 유닛 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic 정수 실행 유닛 전력 = 유닛당 전력 × num_alus/num_muls.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  double num_fpus;
  /* [한국어] 부동소수점 유닛(FPU) 수 (실수형 — 부분 FPU 지원 시 소수점 사용 가능).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic FPU 전력 = FPU당 전력 × num_fpus.
   * 값 범위: 양수 실수 (일반적으로 양의 정수이지만 double로 선언).
   * 동기화: 초기화 후 읽기 전용. */

  int int_data_width, fp_data_width, v_address_width, p_address_width;
  /* [한국어] 데이터 및 주소 버스 비트 폭.
   * int_data_width: 정수 데이터 경로 폭 (비트, 예: 32 또는 64).
   * fp_data_width: 부동소수점 데이터 경로 폭 (비트, 예: 32 또는 64).
   * v_address_width: 가상 주소(virtual address) 비트 폭.
   * p_address_width: 물리 주소(physical address) 비트 폭.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic 인터커넥트, 로드/스토어 유닛 버스 전력 계산.
   * 값 범위: 양의 정수 (비트 단위).
   * 동기화: 초기화 후 읽기 전용. */

  double pipeline_duty_cycle, total_cycles, busy_cycles, idle_cycles;
  /* [한국어] 파이프라인 활동 사이클 통계.
   * pipeline_duty_cycle: 파이프라인이 실제로 활성화된 비율 (0~1).
   * total_cycles: 시뮬레이션 총 클럭 사이클 수.
   * busy_cycles: 파이프라인이 유효 명령어를 실행한 사이클 수.
   * idle_cycles: 파이프라인이 유휴 상태(bubble/stall)였던 사이클 수.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim 성능 카운터로부터 기록.
   * 읽는 자: CoreDynamic — duty cycle × 피크 전력 = 평균 동적 전력 계산.
   * 값 범위: total_cycles = busy + idle (부동소수점 오차 허용).
   * 동기화: 단일 스레드. */

  bool regWindowing, multithreaded;
  /* [한국어] 레지스터 윈도잉 및 멀티스레드 지원 플래그.
   * regWindowing: SPARC 스타일 레지스터 윈도우 지원 여부 — GPU에선 일반적으로 false.
   * multithreaded: SMT(동시 멀티스레딩) 지원 여부 — GPU SM은 warp 다중 실행이므로 true.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CoreDynamic — 멀티스레드 전력 오버헤드(pppm_lkg_multhread) 적용 분기.
   * 값 범위: true/false.
   * 동기화: 초기화 후 읽기 전용. */

  double pppm_lkg_multhread[4];
  /* [한국어] 멀티스레드 누설 전류 보정 계수 배열 (크기 4).
   * PPPM(Per-Phase Power Model) 누설 전력에 멀티스레드 오버헤드를 곱하기 위한 계수.
   * [0]: 동적 전력 멀티플라이어, [1]: 누설 전력 멀티플라이어,
   * [2]: 면적 멀티플라이어, [3]: 예비.
   * 설정자: CoreDynamic 생성자가 num_hthreads 기반으로 계산해 저장.
   * 읽는 자: CoreDynamic 누설 전력 계산 시 스레드 수에 따른 보정에 사용.
   * 값 범위: 1.0 이상 실수 (멀티스레드이면 단일보다 누설이 더 큼).
   * 동기화: 초기화 후 읽기 전용. */

  double IFU_duty_cycle, BR_duty_cycle, LSU_duty_cycle, MemManU_I_duty_cycle,
      MemManU_D_duty_cycle, ALU_duty_cycle, MUL_duty_cycle, FPU_duty_cycle,
      ALU_cdb_duty_cycle, MUL_cdb_duty_cycle, FPU_cdb_duty_cycle;
  /* [한국어] 각 파이프라인 유닛의 duty cycle (활성화 비율, 0~1).
   * IFU_duty_cycle: 명령어 패치 유닛(IFU, Instruction Fetch Unit) 활성 비율.
   * BR_duty_cycle: 분기 예측 유닛(Branch Predictor) 활성 비율.
   * LSU_duty_cycle: 로드/스토어 유닛(LSU, Load/Store Unit) 활성 비율.
   * MemManU_I_duty_cycle: 명령어 측 메모리 관리 유닛(ITLB 등) 활성 비율.
   * MemManU_D_duty_cycle: 데이터 측 메모리 관리 유닛(DTLB 등) 활성 비율.
   * ALU_duty_cycle: 정수 ALU 활성 비율.
   * MUL_duty_cycle: 정수 곱셈기 활성 비율.
   * FPU_duty_cycle: 부동소수점 유닛 활성 비율.
   * ALU_cdb_duty_cycle: ALU 결과 버스(CDB) 활성 비율.
   * MUL_cdb_duty_cycle: MUL 결과 버스 활성 비율.
   * FPU_cdb_duty_cycle: FPU 결과 버스 활성 비율.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim 성능 카운터(SM별 유닛 활성 사이클)로 계산해 기록.
   * 읽는 자: CoreDynamic — 각 유닛 피크 전력 × duty cycle = 평균 동적 전력.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 단일 스레드. */

  ~CoreDynParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] CacheDynParam — 캐시 계층(L1/L2/LLC/디렉토리) 동적 파라미터 클래스.
 *
 * AccelWattch의 CacheDynamic 전력 모델이 필요로 하는 캐시 지오메트리, 타이밍,
 * 활동 파라미터를 보유한다. ParseXML로 XML에서 파싱한 값으로 초기화되며,
 * CACTI(SRAM 배열 모델)에 캐시 용량/연관도/뱅크/처리량/레이턴시를 전달하여
 * 캐시 구조의 접근 에너지와 누설 전력을 계산하는 데 사용된다.
 * GPGPU-Sim의 L1 데이터 캐시, L1 텍스처 캐시, L2 캐시 각각에 대해
 * 별도의 CacheDynParam 인스턴스가 생성된다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class CacheDynParam {
 public:
  CacheDynParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. CacheDynParam(ParseXML*, int)로 초기화 필요. */

  CacheDynParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 캐시의 파라미터를 XML에서 읽어 필드 초기화. */

  string name;
  /* [한국어] 캐시 인스턴스 이름 문자열 (예: "L1_Data_Cache", "L2_Cache").
   * 설정자: ParseXML 생성자 — XML의 name 태그에서 파싱.
   * 읽는 자: 전력 리포트 출력 시 식별자로 사용.
   * 값 범위: 비어 있지 않은 문자열.
   * 동기화: 초기화 후 읽기 전용. */

  enum Dir_type dir_ty;
  /* [한국어] 캐시 디렉토리 구현 방식 (ST/DC/SBT/NonDir).
   * 설정자: ParseXML 생성자.
   * 읽는 자: CacheDynamic — 디렉토리 캐시 전력 계산 분기 여부 결정.
   * 값 범위: Dir_type 열거형 중 하나.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate, executionTime;
  /* [한국어] 캐시 클럭 주파수 (Hz) 및 총 실행 시간 (초).
   * clockRate: 캐시가 동작하는 클럭 (SM 클럭 또는 메모리 클럭).
   * executionTime: 시뮬레이션 총 실행 시간 — 전력(W) = 에너지(J) / 시간(s) 계산에 사용.
   * 설정자: ParseXML 생성자.
   * 읽는 자: CacheDynamic 동적 전력 계산.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double capacity, blockW, assoc, nbanks;
  /* [한국어] 캐시 지오메트리 파라미터.
   * capacity: 캐시 용량 (바이트, 예: 32768 = 32KB).
   * blockW: 캐시 블록(캐시 라인) 크기 (바이트, 예: 128).
   * assoc: 연관도(associativity) — 1=직접 매핑, 4=4-way 세트 연관, 0=완전 연관.
   * nbanks: 캐시 뱅크 수 — 병렬 접근을 위해 배열을 물리적으로 분할한 수.
   * 설정자: ParseXML 생성자 — XML의 cache_size, block_size, associativity, nbanks.
   * 읽는 자: CACTI ArrayST — 이 값으로 SRAM 배열 크기와 접근 에너지를 계산.
   * 값 범위: 양수 실수 (capacity는 바이트, blockW는 바이트, assoc/nbanks는 양의 정수).
   * 동기화: 초기화 후 읽기 전용. */

  double throughput, latency;
  /* [한국어] 캐시 처리량과 레이턴시 (사이클 단위 또는 ns 단위).
   * throughput: 캐시 접근 처리량 — 단위 시간당 처리 가능한 접근 수.
   * latency: 캐시 접근 레이턴시 — 사이클 또는 ns 단위 접근 시간.
   * 설정자: ParseXML 생성자 또는 CACTI 추정 결과에서 설정.
   * 읽는 자: 캐시 타이밍 검증 및 파이프라인 레이턴시 모델에 사용.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double duty_cycle, dir_duty_cycle;
  /* [한국어] 캐시 및 디렉토리 접근 duty cycle (활성화 비율, 0~1).
   * duty_cycle: 캐시 데이터 배열 접근 duty cycle — (캐시 접근 횟수) / (총 사이클).
   * dir_duty_cycle: 디렉토리 캐시 접근 duty cycle.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim 캐시 접근 카운터로 계산해 기록.
   * 읽는 자: CacheDynamic — 피크 전력 × duty cycle = 평균 동적 전력.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 단일 스레드. */

  // double duty_cycle;
  /* [한국어] (비활성 주석) 이전 버전의 duty_cycle 선언 — 현재 위의 duty_cycle로 통합됨. */

  int missb_size, fu_size, prefetchb_size, wbb_size;
  /* [한국어] 캐시 보조 구조(auxiliary structure) 크기 (엔트리 수).
   * missb_size: 미스 버퍼(Miss Status Holding Registers, MSHR) 크기 — 동시 처리 미스 수.
   * fu_size: 필업 버퍼(Fill/Line-fill Buffer) 크기 — 하위 계층에서 가져온 데이터 임시 보관.
   * prefetchb_size: 프리패치 버퍼 크기 — 하드웨어 프리패처가 미리 가져온 라인 보관.
   * wbb_size: 쓰기 버퍼(Write Back Buffer) 크기 — Write_back 정책에서 교체된 dirty 라인 보관.
   * 설정자: ParseXML 생성자 — XML의 miss_buffer_size, fill_buffer_size 등.
   * 읽는 자: CacheDynamic — 보조 구조 배열(CAM/RAM) 전력 계산 시 CACTI에 전달.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  ~CacheDynParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] DRAMParam — GDDR5/GDDR3 DRAM 전력 모델 파라미터 클래스.
 *
 * GPU 메모리(GDDR5/GDDR3) 전력 계산에 필요한 데이터시트 기반 전류(IDD), 전압(VDD),
 * 타이밍 파라미터, I/O 신호 폭, 경험적 계수 등을 모두 보유한다.
 * detailed_dram_model = 1 이면 IDD 전류 기반 상세 GDDR5 전력 모델을 사용하고,
 * 0 이면 cmd_coeff/activity_coeff 등 경험적(empirical) 계수를 사용한다.
 * IDD 전류 표기법은 JEDEC GDDR5 데이터시트 표준을 따른다.
 * GPGPU-Sim의 dram.cc(DRAM 타이밍 모델)와는 독립적으로 동작하며,
 * AccelWattch가 시뮬레이션 후 전력 계산 단계에서만 이 파라미터를 참조한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class DRAMParam {
 public:
  DRAMParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. DRAMParam(ParseXML*, int)로 초기화 필요. */

  DRAMParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 DRAM의 파라미터를 XML에서 읽어 필드 초기화. */

  string name;
  /* [한국어] DRAM 인스턴스 이름 (예: "GDDR5_Main_Memory").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 값 범위: 비어 있지 않은 문자열.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate;
  /* [한국어] DRAM 인터페이스 클럭 주파수 (Hz).
   * actual_operating_clock(MHz)에서 변환해 저장된다.
   * 설정자: ParseXML 생성자 — actual_operating_clock × 1e6.
   * 읽는 자: DRAM 전력 계산 — 전류×전압×주파수 계산의 f.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double executionTime;
  /* [한국어] 시뮬레이션 총 실행 시간 (초).
   * 설정자: ParseXML 생성자 (total_cycles / clockRate).
   * 읽는 자: DRAM 전력 계산 — 에너지(J) / 시간(s) = 전력(W).
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double cmd_coeff;
  /* [한국어] 커맨드(command) 발행당 경험적 전력 계수 (mW/cmd 또는 상대 단위).
   * 경험적 모델(detailed_dram_model=0)에서 커맨드 전력 = cmd_coeff × 커맨드 수.
   * 설정자: ParseXML 생성자 — XML의 empirical 계수 필드.
   * 읽는 자: DRAM 전력 계산 모듈(경험적 모드).
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double activity_coeff;
  /* [한국어] 메모리 활동(행 활성화, 열 읽기/쓰기)당 경험적 전력 계수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double nop_coeff;
  /* [한국어] NOP(No-Operation) 커맨드당 경험적 전력 계수.
   * DRAM이 활성이지만 실제 데이터 전송이 없는 대기 상태 전력 추정에 사용.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double act_coeff;
  /* [한국어] ACT(Row Activate) 커맨드당 경험적 전력 계수.
   * 행 활성화(Row Activation) 시 충전 전력을 추정한다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double pre_coeff;
  /* [한국어] PRE(Precharge) 커맨드당 경험적 전력 계수.
   * 프리차지 동작 전력 추정에 사용.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double rd_coeff;
  /* [한국어] READ 커맨드당 경험적 전력 계수.
   * 열 읽기(Column Read) 동작 전력 추정.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double wr_coeff;
  /* [한국어] WRITE 커맨드당 경험적 전력 계수.
   * 열 쓰기(Column Write) 동작 전력 추정.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double req_coeff;
  /* [한국어] 요청(request)당 경험적 전력 계수.
   * 전체 DRAM 요청(ACT+CAS+PRE 시퀀스)에 대한 통합 계수로 사용될 수 있다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  double const_coeff;
  /* [한국어] 상수(constant) 전력 계수 — 동작 상태와 무관한 고정 배경 전력.
   * DRAM이 활성 상태에서 항상 소모하는 정적 구성 요소를 모델링한다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 경험적 DRAM 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  int detailed_dram_model;  // 1 - to use newly added DRAM model (GDDR5 only), 0
                            // - use empirical model
  /* [한국어] 상세 DRAM 전력 모델 사용 여부 플래그.
   * 1: IDD 전류 기반 상세 GDDR5 전력 모델 사용 — idd0~idd7, 전압, 타이밍 파라미터 활용.
   * 0: 경험적 계수(cmd_coeff 등) 기반 단순 모델 사용.
   * 설정자: ParseXML 생성자 — XML의 detailed_dram_model 필드.
   * 읽는 자: DRAM 전력 계산 모듈 — 모델 선택 분기.
   * 값 범위: 0 또는 1.
   * 동기화: 초기화 후 읽기 전용. */

  // the following are the current specified by DATA SHEET
  // unit: mA
  /* [한국어] 아래 idd* 필드들은 JEDEC GDDR5 데이터시트의 IDD(전류 소비) 파라미터이다.
   * 단위: mA. 상세 모델(detailed_dram_model=1)에서만 사용된다.
   * IDD는 "Input Device current Draw"로 DRAM 동작 상태별 전류 소비를 정의한다. */

  int idd0;
  /* [한국어] IDD0: 활성 대기 전류 (Active Precharge Current, mA).
   * 단일 ACT-PRE 사이클에서 소모되는 전류.
   * 설정자: ParseXML 생성자 — XML의 idd0 필드.
   * 읽는 자: 상세 DRAM 전력 모델 — 행 활성화당 에너지 계산.
   * 동기화: 초기화 후 읽기 전용. */

  int idd1;
  /* [한국어] IDD1: 활성 읽기/쓰기 후 프리차지 전류 (Active Read/Write then Precharge, mA).
   * ACT-READ(또는 WRITE)-PRE 전체 사이클의 평균 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd2p;
  /* [한국어] IDD2P: 프리차지 파워다운 전류 (Precharge Power Down, mA).
   * 모든 뱅크가 프리차지된 파워다운 상태의 전류 — 저전력 대기 상태.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd2n;
  /* [한국어] IDD2N: 프리차지 스탠바이 전류 (Precharge Standby, Normal, mA).
   * 모든 뱅크 프리차지 상태에서 CKE 활성, NOP 커맨드 상태의 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd3p;
  /* [한국어] IDD3P: 활성 파워다운 전류 (Active Power Down, mA).
   * 하나 이상의 뱅크가 활성화된 파워다운 상태의 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd3n;
  /* [한국어] IDD3N: 활성 스탠바이 전류 (Active Standby, Normal, mA).
   * 하나 이상의 뱅크가 활성화된 NOP 커맨드 상태의 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd4r;
  /* [한국어] IDD4R: 버스트 읽기 전류 (Burst Read, mA).
   * 연속 CAS-READ 커맨드 실행 중 데이터 I/O 전류 포함 최대 읽기 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델 — 읽기 대역폭 전력 계산.
   * 동기화: 초기화 후 읽기 전용. */

  int idd4w;
  /* [한국어] IDD4W: 버스트 쓰기 전류 (Burst Write, mA).
   * 연속 CAS-WRITE 커맨드 실행 중 데이터 I/O 전류 포함 최대 쓰기 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델 — 쓰기 대역폭 전력 계산.
   * 동기화: 초기화 후 읽기 전용. */

  int idd5;
  /* [한국어] IDD5: 자동 리프레시 전류 (Auto Refresh, mA).
   * DRAM 셀 데이터 유지를 위한 주기적 리프레시 동작 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델 — 리프레시 전력 계산.
   * 동기화: 초기화 후 읽기 전용. */

  int idd6;
  /* [한국어] IDD6: 셀프 리프레시 전류 (Self Refresh, mA).
   * 외부 클럭 없이 DRAM 내부적으로 리프레시하는 최저전력 대기 상태 전류.
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  int idd7;
  /* [한국어] IDD7: 풀 페이지 버스트 읽기 전류 (Full-Speed Burst Read, mA).
   * 최대 버스트 길이로 연속 읽기 시의 전류 (IDD4R보다 높은 피크).
   * 설정자: ParseXML 생성자.
   * 읽는 자: 상세 DRAM 전력 모델.
   * 동기화: 초기화 후 읽기 전용. */

  // the following are the vdd specified by DATA SHEET; NOT the actual VDD
  double datasheet_vdd;
  /* [한국어] 데이터시트 공칭 VDD(전원 전압, V) — 실제 동작 전압이 아닌 스펙 기준 전압.
   * IDD 전류 수치가 이 전압 기준으로 명시된다.
   * 설정자: ParseXML 생성자 — XML의 datasheet_vdd 필드.
   * 읽는 자: DRAM 전력 = IDD × VDD 계산 시 기준 전압으로 사용.
   * 값 범위: 양수 실수 (예: GDDR5 = 1.5V).
   * 동기화: 초기화 후 읽기 전용. */

  double actual_vdd;
  /* [한국어] 실제 동작 VDD(전압, V) — 실제 GPU 보드에서 DRAM에 인가되는 전압.
   * datasheet_vdd와 다를 수 있으며, 전력 = IDD × actual_vdd 로 보정한다.
   * 설정자: ParseXML 생성자 — XML의 actual_vdd 필드.
   * 읽는 자: 상세 DRAM 전력 모델 — 실제 전압으로 전력 재계산.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  // the following are the timing parameters specified by DATA SHEET
  // unit: ns
  /* [한국어] 아래 t_* 필드들은 JEDEC GDDR5 데이터시트의 타이밍 파라미터이다.
   * 단위: ns(나노초). 상세 전력 모델에서 IDD 전류의 적용 시간(t_RAS, t_RP 등)을 결정한다. */

  int t_ccd;
  /* [한국어] tCCD: CAS-to-CAS Delay (ns) — 연속 CAS 커맨드 사이 최소 간격.
   * 버스트 읽기/쓰기 전력 계산 시 IDD4R/IDD4W 적용 시간 결정.
   * 동기화: 초기화 후 읽기 전용. */

  int t_rrd;
  /* [한국어] tRRD: Row-to-Row Delay (ns) — 같은 뱅크 그룹 내 연속 ACT 사이 최소 간격.
   * 행 활성화 전력(IDD0) 적용 빈도 계산에 사용.
   * 동기화: 초기화 후 읽기 전용. */

  int t_rcd;
  /* [한국어] tRCD: RAS-to-CAS Delay (ns) — ACT 커맨드 후 CAS 커맨드 발행 가능까지 대기 시간.
   * 행 활성화 후 열 접근까지의 레이턴시.
   * 동기화: 초기화 후 읽기 전용. */

  int t_ras;
  /* [한국어] tRAS: Row Active Time (ns) — ACT 후 PRE 발행 가능 최소 시간.
   * 행이 활성 상태(IDD3N/IDD4R/IDD4W)를 유지하는 최소 지속 시간.
   * 동기화: 초기화 후 읽기 전용. */

  int t_rp;
  /* [한국어] tRP: Row Precharge Time (ns) — PRE 커맨드 후 다음 ACT 발행 가능까지 대기 시간.
   * 프리차지 전류(IDD0의 일부) 적용 시간 계산에 사용.
   * 동기화: 초기화 후 읽기 전용. */

  int t_rc;
  /* [한국어] tRC: Row Cycle Time (ns) = tRAS + tRP — ACT-to-ACT (same bank) 최소 간격.
   * 동일 뱅크에 대한 전체 활성화-프리차지 사이클 시간.
   * 동기화: 초기화 후 읽기 전용. */

  int t_cl;
  /* [한국어] tCL: CAS Latency (ns) — CAS 커맨드 후 데이터 출력까지 지연 시간.
   * 읽기 레이턴시의 핵심 파라미터.
   * 동기화: 초기화 후 읽기 전용. */

  int t_cdlr;
  /* [한국어] tCDLR: CAS Delay for Last Read (ns) — 마지막 읽기 CAS 후 쓰기 전환 지연.
   * 읽기→쓰기 전환 시 버스 방향 전환 시간.
   * 동기화: 초기화 후 읽기 전용. */

  int t_wr;
  /* [한국어] tWR: Write Recovery Time (ns) — 마지막 쓰기 후 PRE 가능 최소 대기 시간.
   * 쓰기 데이터가 DRAM 셀에 완전히 기록될 때까지 필요한 시간.
   * 동기화: 초기화 후 읽기 전용. */

  // the following are the DRAM clocks
  // unit: MHz
  /* [한국어] 아래 클럭 파라미터는 MHz 단위이다. */

  int datasheet_operating_clock;  // this is specified by DATA SHEET. This is
                                  // NOT the actual DRAM clock
  /* [한국어] datasheet_operating_clock: 데이터시트 공칭 동작 클럭 (MHz).
   * IDD 전류 수치가 이 클럭 기준으로 명시된다. 실제 GPU 동작 클럭과 다를 수 있다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: IDD 전류를 실제 클럭으로 스케일링 시 기준값으로 사용.
   * 동기화: 초기화 후 읽기 전용. */

  int actual_operating_clock;
  /* [한국어] actual_operating_clock: 실제 GPU DRAM 동작 클럭 (MHz).
   * 이 값이 clockRate(Hz) 계산의 기준이 된다 (clockRate = actual_operating_clock × 1e6).
   * 설정자: ParseXML 생성자 — XML의 actual_operating_clock 필드.
   * 읽는 자: DRAM 전력 계산 시 실제 동작 주파수.
   * 동기화: 초기화 후 읽기 전용. */

  // the following are each DRAM bank's IO info
  int bank_width;                   // in bits
  /* [한국어] bank_width: DRAM 뱅크 하나의 데이터 I/O 폭 (비트).
   * 데이터 핀(DQ) 수에 해당하며, 한 번의 READ/WRITE로 전송되는 비트 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: DRAM I/O 전력 계산 — per_dq_read/write_power × bank_width.
   * 값 범위: 양의 정수 (비트 단위, 예: 32).
   * 동기화: 초기화 후 읽기 전용. */

  int dqs_signal_width;             // in bits
  /* [한국어] dqs_signal_width: DQS(Data Strobe) 신호 수 (비트).
   * DQS는 데이터 샘플링 타이밍을 제공하는 차동 신호 쌍이다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: DRAM I/O 전력 계산 — DQS 신호 전력 기여.
   * 동기화: 초기화 후 읽기 전용. */

  int extra_dq_write_signal_width;  // in bits
  /* [한국어] extra_dq_write_signal_width: 쓰기 시 추가 DQ 신호 폭 (비트).
   * 쓰기 마스킹(Write Mask) 등 추가 신호에 사용되는 DQ 비트 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: DRAM 쓰기 I/O 전력 보정.
   * 동기화: 초기화 후 읽기 전용. */

  int per_dq_read_power;            // in mW
  /* [한국어] per_dq_read_power: DQ 핀 하나당 읽기 전력 (mW).
   * 총 읽기 I/O 전력 = per_dq_read_power × bank_width 로 계산된다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: DRAM 전력 계산 모듈.
   * 값 범위: 양의 정수 (mW).
   * 동기화: 초기화 후 읽기 전용. */

  int per_dq_write_power;           // in mW
  /* [한국어] per_dq_write_power: DQ 핀 하나당 쓰기 전력 (mW).
   * 총 쓰기 I/O 전력 = per_dq_write_power × (bank_width + extra_dq_write_signal_width).
   * 설정자: ParseXML 생성자.
   * 읽는 자: DRAM 전력 계산 모듈.
   * 값 범위: 양의 정수 (mW).
   * 동기화: 초기화 후 읽기 전용. */

  ~DRAMParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] MCParam — 메모리 컨트롤러(Memory Controller) 동적 파라미터 클래스.
 *
 * GPU DRAM 메모리 컨트롤러의 전력 계산에 필요한 파라미터를 보유한다.
 * 버스 폭(dataBusWidth), 채널 수(num_channels), 데이터 전송률, 읽기/쓰기 횟수,
 * PHY(물리 계층) 포함 여부 등을 포함하며, McPAT의 MemoryController 클래스에 전달된다.
 * GPGPU-Sim의 DRAM 컨트롤러(dram.cc)가 처리한 요청 통계를 AccelWattch가 이 파라미터로
 * 변환해 메모리 컨트롤러 전력을 추정한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class MCParam {
 public:
  MCParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. MCParam(ParseXML*, int)로 초기화 필요. */

  MCParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 메모리 컨트롤러 파라미터를 XML에서 파싱. */

  string name;
  /* [한국어] 메모리 컨트롤러 인스턴스 이름 (예: "MC_GDDR5").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate, num_mcs, peakDataTransferRate, num_channels;
  /* [한국어] 메모리 컨트롤러 기본 성능 파라미터.
   * clockRate: 메모리 컨트롤러 클럭 (Hz). 프론트엔드/백엔드 클럭 도메인에 따라 다를 수 있음.
   * num_mcs: 메모리 컨트롤러 수 (GPU에는 여러 MC가 있을 수 있음).
   * peakDataTransferRate: 피크 데이터 전송률 (Gbit/s 또는 GB/s 단위).
   * num_channels: DRAM 채널 수 — 각 채널이 독립적인 MC를 가질 수 있음.
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 전력 모델 — 채널 수에 비례한 전력 스케일링.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  //  double mcTEPowerperGhz;
  //	double mcPHYperGbit;
  //	double area;
  /* [한국어] (비활성 주석) 이전 버전의 MC 전력/면적 파라미터 — 현재 미사용. */

  int llcBlockSize, dataBusWidth, addressBusWidth;
  /* [한국어] 버스 및 캐시 블록 폭 파라미터.
   * llcBlockSize: LLC(Last Level Cache) 블록 크기 (바이트) — MC가 처리하는 최소 전송 단위.
   * dataBusWidth: DRAM 데이터 버스 폭 (비트, 예: 256 = 256-bit 버스).
   * addressBusWidth: DRAM 주소 버스 폭 (비트).
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 전력 모델 — 버스 스위칭 전력 = 버스 폭 × 스위칭 활동 × 에너지계수.
   * 값 범위: 양의 정수 (비트 또는 바이트 단위).
   * 동기화: 초기화 후 읽기 전용. */

  int opcodeW;
  /* [한국어] DRAM 커맨드 오피코드(opcode) 비트 폭.
   * MC→DRAM 커맨드 버스의 커맨드 코드 필드 크기.
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 커맨드 버스 전력 계산.
   * 값 범위: 양의 정수 (비트 단위).
   * 동기화: 초기화 후 읽기 전용. */

  int memAccesses;
  /* [한국어] 총 메모리 접근 횟수 (시뮬레이션 전체 구간).
   * 설정자: AccelWattch 진입점이 GPGPU-Sim DRAM 요청 카운터로부터 기록.
   * 읽는 자: MC 전력 계산 — 접근당 에너지 × 접근 수.
   * 값 범위: 양의 정수.
   * 동기화: 단일 스레드. */

  int memRank;
  /* [한국어] DRAM 랭크(rank) 수 — 동일 채널에서 CS(Chip Select) 신호로 구분되는 DRAM 그룹.
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 전력 모델 — 랭크 수에 따른 CS/CKE 신호 전력 스케일링.
   * 값 범위: 양의 정수 (통상 1~4).
   * 동기화: 초기화 후 읽기 전용. */

  int type;
  /* [한국어] 메모리 컨트롤러 종류 (MemoryCtrl_type 열거형 정수값).
   * 0 = MC(DRAM 컨트롤러), 1 = FLASHC(플래시 컨트롤러).
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 전력 모델 선택 분기.
   * 값 범위: 0 또는 1.
   * 동기화: 초기화 후 읽기 전용. */

  double frontend_duty_cycle, duty_cycle, perc_load;
  /* [한국어] MC 활동 비율 파라미터.
   * frontend_duty_cycle: MC 프론트엔드(요청 수신/디코딩) 활성 비율.
   * duty_cycle: MC 백엔드(DRAM 커맨드 발행) 활성 비율.
   * perc_load: MC 부하율(load percentage, 0~1) — 최대 대역폭 대비 실제 사용 비율.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim 통계로 계산.
   * 읽는 자: MC 동적 전력 계산.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 단일 스레드. */

  double executionTime, reads, writes;
  /* [한국어] 시뮬레이션 시간 및 읽기/쓰기 요청 수.
   * executionTime: 시뮬레이션 총 실행 시간 (초) — 전력(W) = 에너지(J) / 시간(s).
   * reads: MC가 처리한 총 읽기 요청 수.
   * writes: MC가 처리한 총 쓰기 요청 수.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim DRAM 통계에서 기록.
   * 읽는 자: MC 동적 전력 계산 — 읽기/쓰기별 에너지가 다를 수 있음.
   * 동기화: 단일 스레드. */

  bool LVDS, withPHY;
  /* [한국어] I/O 인터페이스 옵션 플래그.
   * LVDS(Low Voltage Differential Signaling): LVDS 방식 신호 사용 여부.
   *   LVDS는 저전압 차동 신호로 LPDDR 등에 사용 — 전력 효율적.
   * withPHY: PHY(Physical Layer) 칩 포함 여부.
   *   PHY가 포함되면 serdes/equalizer 전력을 별도로 계산한다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: MC 전력 모델 — PHY 전력 추가 여부 결정.
   * 동기화: 초기화 후 읽기 전용. */

  ~MCParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] NoCParam — NoC(Network-on-Chip, 네트워크온칩) 전력 모델 파라미터 클래스.
 *
 * GPU 내부 인터커넥트(NoC)의 전력 계산에 필요한 파라미터를 보유한다.
 * 플릿 크기(flit_size), 포트 수, 가상 채널(VC) 수, 토폴로지 노드 수,
 * 링크 처리량/레이턴시, duty cycle 등을 포함한다.
 * GPGPU-Sim의 intersim2(ICNT, Booksim 기반 NoC 시뮬레이터)가 처리한
 * 패킷 통계를 AccelWattch가 이 파라미터로 변환해 NoC 전력을 추정한다.
 * McPAT의 NoC 클래스에 이 파라미터를 전달해 라우터+링크 전력을 계산한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class NoCParam {
 public:
  NoCParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. NoCParam(ParseXML*, int)로 초기화 필요. */

  NoCParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 NoC 파라미터를 XML에서 파싱. */

  string name;
  /* [한국어] NoC 인스턴스 이름 (예: "GPU_Interconnect").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate;
  /* [한국어] NoC 클럭 주파수 (Hz).
   * 설정자: ParseXML 생성자.
   * 읽는 자: NoC 전력 모델 — 라우터 동적 전력 계산의 f(주파수).
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  int flit_size;
  /* [한국어] 플릿(flit, flow control unit) 크기 (비트).
   * NoC에서 라우터 간 한 번에 전송되는 최소 데이터 단위.
   * 설정자: ParseXML 생성자 — XML의 flit_size 필드.
   * 읽는 자: NoC 전력 모델 — 플릿 크기에 비례한 링크/버퍼 전력 계산.
   * 값 범위: 양의 정수 (비트 단위, 예: 128, 256).
   * 동기화: 초기화 후 읽기 전용. */

  int input_ports, output_ports, min_ports, global_linked_ports;
  /* [한국어] 라우터 포트 수 파라미터.
   * input_ports: 라우터당 입력 포트 수.
   * output_ports: 라우터당 출력 포트 수.
   * min_ports: 최소 포트 수 (토폴로지 경계 노드의 포트).
   * global_linked_ports: 글로벌 링크(장거리 연결)를 통해 연결된 포트 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: NoC 전력 모델 — 포트 수에 비례한 크로스바 전력 계산.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int virtual_channel_per_port, input_buffer_entries_per_vc;
  /* [한국어] 가상 채널(Virtual Channel) 파라미터.
   * virtual_channel_per_port: 포트당 가상 채널 수.
   *   VC는 서로 다른 트래픽 클래스를 분리해 데드락 방지에 사용된다.
   * input_buffer_entries_per_vc: VC당 입력 버퍼(FIFO) 엔트리 수.
   *   플릿 버퍼링 깊이 → 버퍼 SRAM 전력 계산에 영향.
   * 설정자: ParseXML 생성자.
   * 읽는 자: NoC 라우터 버퍼 전력 모델.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int horizontal_nodes, vertical_nodes, total_nodes;
  /* [한국어] NoC 토폴로지 노드 수 (메시/토러스 등 2D 배열 기준).
   * horizontal_nodes: 수평 방향 노드 수.
   * vertical_nodes: 수직 방향 노드 수.
   * total_nodes: 전체 라우터 노드 수 = horizontal × vertical (단순 메시 기준).
   * 설정자: ParseXML 생성자.
   * 읽는 자: NoC 전력 모델 — 총 노드 수에 비례한 전체 NoC 전력 스케일링.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  double executionTime, total_access, link_throughput, link_latency, duty_cycle,
      chip_coverage, route_over_perc;
  /* [한국어] NoC 동적 활동 통계 파라미터.
   * executionTime: 시뮬레이션 총 실행 시간 (초).
   * total_access: NoC를 통과한 총 플릿/패킷 수 (AccelWattch 진입점이 ICNT 통계에서 기록).
   * link_throughput: 링크 처리량 (플릿/사이클 또는 GB/s).
   * link_latency: 링크 전파 레이턴시 (사이클 또는 ns).
   * duty_cycle: NoC 활성 비율 (0~1) — 링크가 실제 플릿을 전송 중인 비율.
   * chip_coverage: 글로벌 링크가 커버하는 칩 면적 비율 (0~1).
   * route_over_perc: 오버헤드 라우팅(redundant path) 비율.
   * 설정자: AccelWattch 진입점이 GPGPU-Sim ICNT(intersim2) 통계에서 계산해 기록.
   * 읽는 자: NoC 전력 모델 — duty_cycle × 피크 전력 = 평균 동적 전력.
   * 값 범위: 양수 실수.
   * 동기화: 단일 스레드. */

  bool has_global_link, type;
  /* [한국어] NoC 구조 플래그.
   * has_global_link: 글로벌 링크(long-range wire) 존재 여부.
   *   true이면 글로벌 링크 전력을 별도로 계산.
   * type: NoC 라우터 구현 방식 (true=이진, false=아날로그 또는 다른 방식).
   *   McPAT의 NoC 유형 선택 파라미터.
   * 설정자: ParseXML 생성자.
   * 읽는 자: NoC 전력 모델 — 글로벌 링크/라우터 유형 분기.
   * 동기화: 초기화 후 읽기 전용. */

  ~NoCParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] ProcParam — 전체 프로세서(GPU 칩) 수준의 파라미터 클래스.
 *
 * GPU 칩 전체를 구성하는 컴포넌트의 수와 동질성(homogeneity)을 기술한다.
 * numCore(SM 수), numL2(L2 캐시 수), numNOC(NoC 수), numMC(메모리 컨트롤러 수) 등
 * 각 서브컴포넌트가 몇 개 인스턴스 존재하는지 정의하고,
 * homoCore/homoL2 등 동질성 플래그로 동일한 설정의 컴포넌트를 반복 계산할지를 결정한다.
 * AccelWattch 최상위 계층이 이 파라미터를 보고 각 코어/캐시/MC에 대해
 * CoreDynParam/CacheDynParam/MCParam 인스턴스를 numCore/numL2/numMC 개수만큼 생성한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class ProcParam {
 public:
  ProcParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. ProcParam(ParseXML*, int)로 초기화 필요. */

  ProcParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — GPU 칩 전체 컴포넌트 수 파라미터를 XML에서 파싱. */

  string name;
  /* [한국어] 프로세서 이름 문자열 (예: "GPU").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 동기화: 초기화 후 읽기 전용. */

  int numCore, numL2, numL3, numNOC, numL1Dir, numL2Dir, numMC, numMCChannel;
  /* [한국어] 컴포넌트 인스턴스 수.
   * numCore: SM(Streaming Multiprocessor) 수 — GPU 코어 수.
   * numL2: L2 캐시 인스턴스 수 (GPU에서는 통상 1개 공유 L2 또는 파티션별 L2).
   * numL3: L3 캐시 인스턴스 수 (GPU에 L3가 있는 경우).
   * numNOC: NoC 인스턴스 수 (SM↔L2 연결 인터커넥트 수).
   * numL1Dir: L1 디렉토리 캐시 수.
   * numL2Dir: L2 디렉토리 캐시 수.
   * numMC: 메모리 컨트롤러 수.
   * numMCChannel: MC 채널 수.
   * 설정자: ParseXML 생성자 — XML의 number_of_cores, number_of_L2s 등 필드.
   * 읽는 자: AccelWattch 최상위 — 각 컴포넌트 파라미터 인스턴스를 해당 수만큼 생성.
   * 값 범위: 0 이상 정수.
   * 동기화: 초기화 후 읽기 전용. */

  bool homoCore, homoL2, homoL3, homoNOC, homoL1Dir, homoL2Dir;
  /* [한국어] 컴포넌트 동질성(homogeneity) 플래그.
   * homoCore: 모든 코어(SM)가 동일한 설정인지 여부.
   *   true이면 하나의 SM 전력 × numCore 로 계산 (반복 계산 생략).
   *   false이면 각 SM 별도 파라미터로 개별 계산.
   * homoL2/homoL3/homoNOC/homoL1Dir/homoL2Dir: L2 캐시/L3/NoC/디렉토리 각각의 동질성.
   * 설정자: ParseXML 생성자 — XML의 homogeneous_cores 등 필드.
   * 읽는 자: AccelWattch 최상위 — true이면 단일 인스턴스 결과를 전체에 곱함.
   * 값 범위: true/false.
   * 동기화: 초기화 후 읽기 전용. */

  ~ProcParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] NIUParam — NIU(Network Interface Unit, 네트워크 인터페이스 유닛) 파라미터 클래스.
 *
 * GPU 칩의 외부 네트워크 인터페이스(이더넷 등) 전력 계산 파라미터를 보유한다.
 * McPAT의 NIU 전력 모델에 전달되어 I/O 인터페이스 유닛의 전력을 추정한다.
 * GPGPU-Sim에서는 NIU가 GPU 시뮬레이션에서 직접 사용되지 않을 수 있으나,
 * McPAT 호환성을 위해 전체 프로세서 전력 계산에 포함된다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class NIUParam {
 public:
  NIUParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. NIUParam(ParseXML*, int)로 초기화 필요. */

  NIUParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 NIU 파라미터를 XML에서 파싱. */

  string name;
  /* [한국어] NIU 인스턴스 이름 (예: "NetworkInterfaceUnit").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate;
  /* [한국어] NIU 클럭 주파수 (Hz).
   * 설정자: ParseXML 생성자.
   * 읽는 자: NIU 전력 모델.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  int num_units;
  /* [한국어] NIU 유닛 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: NIU 전력 모델 — 유닛 수에 비례한 전력 스케일링.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  int type;
  /* [한국어] NIU 종류 (구현 방식 또는 프로토콜 타입).
   * 설정자: ParseXML 생성자.
   * 읽는 자: NIU 전력 모델 선택 분기.
   * 값 범위: 정수.
   * 동기화: 초기화 후 읽기 전용. */

  double duty_cycle, perc_load;
  /* [한국어] NIU 활동 파라미터.
   * duty_cycle: NIU 활성 비율 (0~1).
   * perc_load: NIU 부하율 (0~1) — 최대 처리량 대비 실제 사용량.
   * 설정자: AccelWattch 진입점 또는 XML에서 설정.
   * 읽는 자: NIU 전력 모델 — duty_cycle × 피크 전력.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 초기화 후 읽기 전용. */

  ~NIUParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};

/*
 * [한국어] PCIeParam — PCIe(PCI Express) 인터페이스 전력 모델 파라미터 클래스.
 *
 * GPU와 호스트 CPU 사이의 PCIe 링크 전력 계산 파라미터를 보유한다.
 * 채널 수, PHY 포함 여부, duty cycle, 부하율 등을 포함하며,
 * McPAT의 PCIe 전력 모델에 전달된다.
 * GPGPU-Sim에서 PCIe 전송(cudaMemcpy 등)의 전력을 추정할 때 사용된다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 시 한 번 생성.
 */
class PCIeParam {
 public:
  PCIeParam(){};
  /* [한국어] 기본 생성자 — 빈 객체 생성. PCIeParam(ParseXML*, int)로 초기화 필요. */

  PCIeParam(ParseXML *XML_interface, int ithCache_);
  /* [한국어] XML 파싱 생성자 — ithCache_번째 PCIe 파라미터를 XML에서 파싱. */

  string name;
  /* [한국어] PCIe 인스턴스 이름 (예: "PCIe_x16").
   * 설정자: ParseXML 생성자.
   * 읽는 자: 전력 리포트 출력 식별자.
   * 동기화: 초기화 후 읽기 전용. */

  double clockRate;
  /* [한국어] PCIe 클럭 주파수 (Hz).
   * 설정자: ParseXML 생성자.
   * 읽는 자: PCIe 전력 모델 — 동적 스위칭 전력 계산의 주파수.
   * 값 범위: 양수 실수.
   * 동기화: 초기화 후 읽기 전용. */

  int num_channels, num_units;
  /* [한국어] PCIe 채널 수 및 유닛 수.
   * num_channels: PCIe 레인(lane) 수 (예: x16 = 16레인).
   *   레인 수에 비례해 대역폭과 전력이 증가.
   * num_units: PCIe 컨트롤러 유닛 수.
   * 설정자: ParseXML 생성자.
   * 읽는 자: PCIe 전력 모델 — 채널/유닛 수에 비례한 전력 스케일링.
   * 값 범위: 양의 정수.
   * 동기화: 초기화 후 읽기 전용. */

  bool withPHY;
  /* [한국어] PCIe PHY(Physical Layer) 포함 여부.
   * true이면 SerDes(직렬화/역직렬화) 회로 전력을 추가로 계산한다.
   * 설정자: ParseXML 생성자.
   * 읽는 자: PCIe 전력 모델 — PHY 전력 추가 분기.
   * 동기화: 초기화 후 읽기 전용. */

  int type;
  /* [한국어] PCIe 세대/종류 (예: Gen1/Gen2/Gen3 구분 정수).
   * 설정자: ParseXML 생성자.
   * 읽는 자: PCIe 전력 모델 선택 분기.
   * 값 범위: 정수.
   * 동기화: 초기화 후 읽기 전용. */

  double duty_cycle, perc_load;
  /* [한국어] PCIe 활동 파라미터.
   * duty_cycle: PCIe 링크 활성 비율 (0~1).
   * perc_load: PCIe 부하율 (0~1) — 최대 대역폭 대비 실제 사용량.
   * 설정자: AccelWattch 진입점이 cudaMemcpy 전송량 통계로 계산.
   * 읽는 자: PCIe 전력 모델 — duty_cycle × 피크 전력.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 초기화 후 읽기 전용. */

  ~PCIeParam(){};
  /* [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 소멸자. */
};
#endif /* BASIC_COMPONENTS_H_ */
