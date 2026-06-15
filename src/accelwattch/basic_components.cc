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
 * [한국어 설명] AccelWattch 전력 모델 기본 컴포넌트 구현 (basic_components.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 basic_components.h에서 선언된 클래스와 전역 함수들의 구현을 제공한다.
 * 핵심 함수인 longer_channel_device_reduction()은 GPU 누설 전류 계산의 핵심으로,
 * 하드웨어 위치(Core/Uncore/LLC)와 코어 종류(OOO/Inorder)에 따라 긴 채널 소자 비율을
 * 적용해 누설 전류 감소 인수를 반환한다. statsComponents와 statsDef의 operator+/operator*
 * 구현으로 여러 SM의 활동 통계를 합산하고 에너지 계수를 곱하는 연산을 제공한다.
 * 이 파일의 함수들은 AccelWattch 전력 계산의 가장 낮은 수준의 산술 연산을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 전력 계산 흐름:
 *   gpgpu-sim/gpu-sim.cc (cycle()) → 성능 카운터 누적
 *       → accelwattch/gpuwattch.* (AccelWattch 진입점, stat dump 시 호출)
 *           → {CoreDynParam, CacheDynParam, ...} 초기화 (basic_components.cc 사용)
 *               → longer_channel_device_reduction() (누설 인수 계산)
 *               → statsDef/statsComponents 산술 연산 (활동 통계 합산)
 *                   → McPAT 기반 전력 계산 모듈 (core.*, memorySystem.*, noc.*)
 * 실행 컨텍스트: 호스트 유저스페이스, AccelWattch 전력 리포트 생성 시점.
 * 사이클-레벨 모델링과는 분리되어, 매 통계 출력 주기(stat dump)마다 한 번 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - basic_components.h: 선언부 (statsComponents, statsDef, Device_ty, Core_type 등)
 *   - cacti/parameter.h: g_tp (TechnologyParameter 전역 객체)
 *     특히 g_tp.peri_global.long_channel_leakage_reduction 을 longer_channel_device_reduction()에서 참조
 *   - <assert.h>: 방어적 단언문 (현재 코드에서는 직접 사용되지 않으나 헤더 포함)
 *   - <cmath>: 수학 함수 (현재 코드에서는 직접 사용되지 않으나 헤더 포함)
 *   - <iostream>: cout (오류 메시지 출력)
 * 이 파일에 의존하는 모듈:
 *   - accelwattch/gpuwattch.*: AccelWattch 진입점이 statsDef 연산과 longer_channel_device_reduction()을 사용
 *   - accelwattch/core.*, memorySystem.*, noc.*: 전력 계산 모듈들이 이 함수를 누설 보정에 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - longer_channel_device_reduction(): Core/Uncore/LLC 소자 종류와 코어 타입에 따라
 *   CACTI g_tp.peri_global.long_channel_leakage_reduction 을 보간해 누설 감소 인수 반환
 * - statsComponents operator+: 두 statsComponents의 access/hit/miss를 각각 합산
 * - statsComponents operator*: statsComponents의 각 필드에 double 배열[3]의 가중치를 곱함
 * - statsDef operator+: readAc/writeAc/searchAc 각각을 statsComponents::operator+로 합산
 * - statsDef operator*: readAc/writeAc/searchAc 각각에 동일한 double 배열을 적용
 */

#include "basic_components.h"  // [한국어] statsComponents, statsDef, Device_ty, Core_type 등 선언
#include <assert.h>            // [한국어] 방어적 단언문 (assert) — 현재 코드에서 직접 사용 없음
#include <cmath>               // [한국어] 수학 함수 (pow, sqrt 등) — 현재 직접 사용 없으나 포함
#include <iostream>            // [한국어] cout/endl — 오류 메시지 출력에 사용

/*
 * [한국어]
 * longer_channel_device_reduction - 긴 채널 소자의 누설 전류 감소 인수를 계산한다.
 *
 * @device_ty: 소자 위치 범주 (Core_device / Uncore_device / LLC_device).
 * @core_ty: 코어 파이프라인 종류 (OOO / Inorder). device_ty==Core_device 일 때만 의미 있음.
 * @return: 누설 전류 감소 인수 (0.0~1.0 실수).
 *   반환값 = (짧은 채널 소자 비율) + (긴 채널 소자 비율 × g_tp.peri_global.long_channel_leakage_reduction)
 *   이 값을 기준 누설 전력에 곱하면 긴 채널 소자 혼합 효과가 반영된 보정 누설 전력을 얻는다.
 *   반환값이 낮을수록 누설 전류가 많이 감소함을 의미한다.
 *
 * 긴 채널(longer-channel) 소자는 표준 최소 채널 길이보다 긴 트랜지스터로,
 * 임계 전압(Vth)이 높아 서브스레시홀드 누설이 훨씬 적다.
 * 고성능 OOO 코어는 빠른 속도를 위해 짧은 채널 소자 비율이 높고(44% 짧은 채널),
 * 순서 실행 코어와 Uncore/LLC는 긴 채널 소자 비율이 높아 누설이 더 적다.
 * g_tp.peri_global.long_channel_leakage_reduction은 CACTI의 기술 노드별 파라미터로,
 * 긴 채널 소자가 표준 소자 대비 누설 전류를 몇 배 줄이는지를 나타낸다.
 *
 * 수식:
 *   감소_인수 = (1 - p) + p × long_channel_leakage_reduction
 *   (p = 긴 채널 소자 비율, long_channel_leakage_reduction ≤ 1.0)
 *
 * 호출 체인:
 *   CoreDynamic::computeStaticPower() / MemorySystem::computeStaticPower()
 *   → [longer_channel_device_reduction()] → g_tp.peri_global.long_channel_leakage_reduction
 */
double longer_channel_device_reduction(enum Device_ty device_ty,
                                       enum Core_type core_ty) {
  double longer_channel_device_percentage_core;    // [한국어] 코어 내 긴 채널 소자 비율 (0~1)
  double longer_channel_device_percentage_uncore;  // [한국어] Uncore 내 긴 채널 소자 비율 (0~1)
  double longer_channel_device_percentage_llc;     // [한국어] LLC 내 긴 채널 소자 비율 (0~1)

  double long_channel_device_reduction;  // [한국어] 최종 반환할 누설 전류 감소 인수

  longer_channel_device_percentage_llc = 1.0;
  /* [한국어] LLC는 100% 긴 채널 소자로 구성 — LLC는 성능보다 누설 절감이 우선이므로
   * 모든 트랜지스터를 긴 채널 소자로 구현한다 (완전 적용). */

  longer_channel_device_percentage_uncore = 0.82;
  /* [한국어] Uncore(L2, NoC 등)는 82% 긴 채널 소자.
   * 공유 자원은 지연 허용 범위가 넓어 긴 채널 소자 비율을 높게 설정해 누설을 줄인다.
   * 나머지 18%는 성능/타이밍 관련 짧은 채널 소자로 구현된다. */

  if (core_ty == OOO) {
    /* [한국어] OOO(비순서 실행) 코어인 경우 — 더 높은 성능을 위해 짧은 채널 소자 비율이 높다. */
    longer_channel_device_percentage_core =
        0.56;  // 0.54 Xeon Tulsa //0.58 Nehelam
    /* [한국어] OOO 코어는 56% 긴 채널 소자 (Xeon Tulsa 기준 0.54, Nehalem 기준 0.58).
     * 44%는 빠른 짧은 채널 소자 — 비순서 실행 로직(ROB, 이슈 큐 등)에 필요한 속도 때문.
     * 현재 값 0.56은 Xeon Tulsa와 Nehalem 사이의 근사값으로 사용된다. */
    // longer_channel_device_percentage_uncore = 0.76;//0.85 Nehelam
    /* [한국어] (비활성 주석) 이전 실험값 — Nehalem Uncore 기준 0.85, 현재 0.82로 통일. */

  } else {
    /* [한국어] Inorder(순서 실행) 코어인 경우 — 단순한 파이프라인, 긴 채널 소자 비율이 높다. */
    longer_channel_device_percentage_core = 0.8;  // 0.8;//Niagara
    /* [한국어] Inorder 코어는 80% 긴 채널 소자 (Niagara 기반 Sun UltraSPARC T1 기준 0.80).
     * 20%만 짧은 채널 소자 — 단순 파이프라인이므로 성능보다 전력 효율 우선. */
    // longer_channel_device_percentage_uncore = 0.9;//Niagara
    /* [한국어] (비활성 주석) Niagara Uncore 기준 0.90 — 현재 0.82로 통일. */
  }

  if (device_ty == Core_device) {
    /* [한국어] 소자가 코어(SM) 내부에 있는 경우 — longer_channel_device_percentage_core 적용. */
    long_channel_device_reduction =
        (1 - longer_channel_device_percentage_core) +
        longer_channel_device_percentage_core *
            g_tp.peri_global.long_channel_leakage_reduction;
    /* [한국어] 누설 감소 인수 계산:
     *   = (짧은 채널 비율 × 1.0) + (긴 채널 비율 × long_channel_leakage_reduction)
     * 짧은 채널 소자는 감소 없이 100% 누설, 긴 채널 소자는 long_channel_leakage_reduction 배로 감소.
     * g_tp.peri_global.long_channel_leakage_reduction: CACTI 기술 파라미터, 일반적으로 1보다 작음
     * (예: 0.3이면 긴 채널 소자가 표준 대비 30% 누설만 발생). */
  } else if (device_ty == Uncore_device) {
    /* [한국어] 소자가 Uncore(L2 캐시, NoC 등) 내부에 있는 경우. */
    long_channel_device_reduction =
        (1 - longer_channel_device_percentage_uncore) +
        longer_channel_device_percentage_uncore *
            g_tp.peri_global.long_channel_leakage_reduction;
    /* [한국어] 82% 긴 채널 소자 비율로 Core_device보다 더 큰 누설 감소 효과를 얻는다. */
  } else if (device_ty == LLC_device) {
    /* [한국어] 소자가 LLC(Last Level Cache) 내부에 있는 경우. */
    long_channel_device_reduction =
        (1 - longer_channel_device_percentage_llc) +
        longer_channel_device_percentage_llc *
            g_tp.peri_global.long_channel_leakage_reduction;
    /* [한국어] 100% 긴 채널 소자이므로:
     *   감소_인수 = 0 + 1.0 × long_channel_leakage_reduction = long_channel_leakage_reduction
     * LLC는 완전히 긴 채널 소자로만 구성 → 최대 누설 감소. */
  } else {
    /* [한국어] 알 수 없는 device_ty 값이 전달된 경우 — 치명적 오류로 처리. */
    cout << "unknown device category" << endl;  // [한국어] 오류 메시지를 표준 출력에 출력
    exit(0);  // [한국어] 즉시 프로세스 종료 — 잘못된 소자 분류는 전력 계산 전체를 무효화하므로 강제 종료
  }

  return long_channel_device_reduction;
  /* [한국어] 계산된 누설 전류 감소 인수 반환.
   * 호출자(CoreDynamic 등)는 이 값을 기준 누설 전력에 곱해 최종 누설 전력을 산출한다. */
}

/*
 * [한국어]
 * operator+(statsComponents, statsComponents) - 두 statsComponents 카운터를 합산한다.
 *
 * @x: 첫 번째 피연산자 statsComponents (const 참조).
 * @y: 두 번째 피연산자 statsComponents (const 참조).
 * @return: x와 y의 access/hit/miss 각각을 더한 새 statsComponents 객체.
 *
 * 여러 SM(Streaming Multiprocessor) 또는 여러 캐시 파티션의 활동 통계를
 * 하나의 statsComponents로 합산할 때 사용된다.
 * 예: 전체 GPU의 L1 캐시 hit 횟수 = Σ(각 SM의 L1 hit 횟수).
 * 값 의미론(value semantics)을 사용해 새 객체를 반환하므로 피연산자를 수정하지 않는다.
 *
 * 호출 체인:
 *   statsDef::operator+(statsDef, statsDef) → [이 함수] (readAc/writeAc/searchAc 각각)
 */
statsComponents operator+(const statsComponents& x, const statsComponents& y) {
  statsComponents z;  // [한국어] 결과를 담을 새 statsComponents 객체 (access=hit=miss=0으로 초기화)

  z.access = x.access + y.access;  // [한국어] 총 접근 횟수 합산 — x의 접근 수 + y의 접근 수
  z.hit = x.hit + y.hit;           // [한국어] 적중 횟수 합산
  z.miss = x.miss + y.miss;        // [한국어] 미스 횟수 합산

  return z;  // [한국어] 합산 결과 반환 (값 복사)
}

/*
 * [한국어]
 * operator*(statsComponents, double*) - statsComponents에 에너지 계수 배열을 곱한다.
 *
 * @x: 피연산자 statsComponents (const 참조) — access/hit/miss 활동 카운터.
 * @y: 크기 3의 double 배열 포인터 — y[0]=access 계수, y[1]=hit 계수, y[2]=miss 계수.
 * @return: x.access*y[0], x.hit*y[1], x.miss*y[2]를 담은 새 statsComponents 객체.
 *
 * 각 이벤트 유형(접근/적중/미스)마다 독립적인 에너지 계수를 적용해
 * 이벤트당 에너지 소비를 계산할 때 사용된다.
 * 예: 캐시 접근 에너지 = hit_count × hit_energy + miss_count × miss_energy.
 * y[0], y[1], y[2]가 각각 다른 에너지 계수이므로, access/hit/miss에 다른 가중치를 적용.
 *
 * 호출 체인:
 *   statsDef::operator*(statsDef, double*) → [이 함수] (readAc/writeAc/searchAc 각각)
 */
statsComponents operator*(const statsComponents& x, double const* const y) {
  statsComponents z;  // [한국어] 결과를 담을 새 statsComponents 객체

  z.access = x.access * y[0];  // [한국어] 총 접근 횟수 × y[0] (access 이벤트 에너지 계수)
  z.hit = x.hit * y[1];        // [한국어] 적중 횟수 × y[1] (hit 이벤트 에너지 계수)
  z.miss = x.miss * y[2];      // [한국어] 미스 횟수 × y[2] (miss 이벤트 에너지 계수)

  return z;  // [한국어] 에너지 계수 적용 결과 반환
}

/*
 * [한국어]
 * operator+(statsDef, statsDef) - 두 statsDef의 읽기/쓰기/검색 통계를 각각 합산한다.
 *
 * @x: 첫 번째 피연산자 statsDef (const 참조).
 * @y: 두 번째 피연산자 statsDef (const 참조).
 * @return: x와 y의 readAc/writeAc/searchAc 각각을 statsComponents::operator+로 합산한 새 statsDef.
 *
 * 여러 SM 또는 캐시 인스턴스의 읽기/쓰기/검색 통계를 하나로 합칠 때 사용된다.
 * 내부적으로 statsComponents::operator+를 세 번 호출한다.
 *
 * 호출 체인:
 *   AccelWattch 통계 합산 루프 → [이 함수] → statsComponents::operator+
 */
statsDef operator+(const statsDef& x, const statsDef& y) {
  statsDef z;  // [한국어] 결과를 담을 새 statsDef 객체 (모든 카운터 0으로 초기화)

  z.readAc = x.readAc + y.readAc;      // [한국어] 읽기 접근 통계 합산 (statsComponents::operator+ 호출)
  z.writeAc = x.writeAc + y.writeAc;   // [한국어] 쓰기 접근 통계 합산
  z.searchAc = x.searchAc + y.searchAc; // [한국어] 검색 접근 통계 합산
  return z;  // [한국어] 합산 결과 반환
}

/*
 * [한국어]
 * operator*(statsDef, double*) - statsDef의 모든 접근 유형에 동일한 에너지 계수 배열을 적용한다.
 *
 * @x: 피연산자 statsDef (const 참조) — readAc/writeAc/searchAc 통계.
 * @y: 크기 3의 double 배열 포인터 — 동일한 y가 readAc/writeAc/searchAc 각각에 전달된다.
 *   y[0]=access 계수, y[1]=hit 계수, y[2]=miss 계수.
 * @return: 에너지 계수가 적용된 새 statsDef 객체.
 *
 * statsDef 전체에 하나의 에너지 계수 배열을 일괄 적용할 때 사용된다.
 * 읽기/쓰기/검색 각각에 동일한 y 배열을 적용하므로, 세 접근 유형이 같은 에너지 계수를 공유할 때 적합.
 * 내부적으로 statsComponents::operator*를 세 번 호출한다.
 *
 * 호출 체인:
 *   AccelWattch 에너지 계산 루프 → [이 함수] → statsComponents::operator*
 */
statsDef operator*(const statsDef& x, double const* const y) {
  statsDef z;  // [한국어] 결과를 담을 새 statsDef 객체

  z.readAc = x.readAc * y;     // [한국어] 읽기 통계에 y 배열 적용 (statsComponents::operator* 호출)
  z.writeAc = x.writeAc * y;   // [한국어] 쓰기 통계에 y 배열 적용
  z.searchAc = x.searchAc * y; // [한국어] 검색 통계에 y 배열 적용
  return z;  // [한국어] 에너지 계수 적용 결과 반환
}
