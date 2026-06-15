#!/usr/bin/env python3
"""
[한국어] accelwattch/sharedcache.cc 한국어 주석 삽입 스크립트
"""
from pathlib import Path


def insert_before(content: str, anchor: str, text: str, occurrence: int = 0) -> str:
    start = 0
    for _ in range(occurrence):
        idx = content.find(anchor, start)
        if idx == -1:
            raise ValueError(f"occurrence {occurrence} not found for anchor: {anchor[:60]}")
        start = idx + len(anchor)
    idx = content.find(anchor, start)
    if idx == -1:
        raise ValueError(f"anchor not found: {anchor[:80]}")
    line_start = content.rfind("\n", 0, idx) + 1
    return content[:line_start] + text + "\n" + content[line_start:]


def replace_first(content: str, old: str, new: str, occurrence: int = 0) -> str:
    start = 0
    for _ in range(occurrence):
        idx = content.find(old, start)
        if idx == -1:
            raise ValueError(f"occurrence {occurrence} not found for old: {old[:60]}")
        start = idx + len(old)
    idx = content.find(old, start)
    if idx == -1:
        raise ValueError(f"old not found: {old[:80]}")
    return content[:idx] + new + content[idx + len(old):]


def apply(path: Path, insertions, replacements):
    content = path.read_text()
    for anchor, text, occ in insertions:
        content = insert_before(content, anchor, text, occ)
    for old, new, occ in replacements:
        content = replace_first(content, old, new, occ)
    path.write_text(content)
    print(f"annotated {path}")


ROOT = Path("/home/harison/company/gpgpu-sim-study/src/accelwattch")

SHARED_TOP = """/*
 * [한국어 설명] AccelWattch 공유 캐시(L2/L3/Directory) 전력 모델 구현 (sharedcache.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT/AccelWattch에서 GPU/CPU 공유 캐시(SharedCache)의 전력과 면적을
 * 모델링한다. SharedCache는 L2 SRAM(데이터/태그 배열)뿐 아니라 미스 버퍼(MissB),
 * 채우기 버퍼(FillB), 프리페치 버퍼(PrefetchB), 쓰기 버퍼(WBB) 등의
 * 보조 SRAM 구조물을 ArrayST(DataCache)로 모델링한다. computeEnergy()는
 * TDP(peak) 모드와 런타임 모드를 모두 지원하며, XML에서 수집된 실제 L2/L3/
 * Directory 접근 횟수를 기반으로 동적 에너지(dynamic energy)를 산출하고,
 * CACTI가 계산한 local_result.power 누설 전력과 결합해 최종 power/rt_power를
 * 완성한다. GPU 모델에서는 통합 L2 캐시(all SM 공유) 전력 추정에 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층:
 *   gpgpu_sim::cycle() → AccelWattch 카운터 수집
 *     → gpgpu_sim_wrapper::compute() [gpgpu_sim_wrapper.cc]
 *         → Processor::Processor() / compute() [processor.cc]
 *             → [이 파일] SharedCache 객체 생성 및 computeEnergy(false)
 *                 → SharedCache::displayEnergy() (결과 출력)
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 종료 후 전력 보고서 생성 단계.
 * 타이밍 시뮬레이션과는 별개로, McPAT이 사후적으로 에너지를 추정한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - sharedcache.h           : SharedCache/CCdir 클래스 선언, CacheDynParam,
 *                               statsDef, DataCache 필드 정의
 *   - XML_Parse.h / ParseXML  : XML 설정 트리(sys.L2[], sys.L3[], sys.l2 등)
 *   - array.h / DataCache     : SRAM 배열 모델(ArrayST)과 캐시 보조 버퍼 집합
 *   - cacti/parameter.h       : CACTI InputParameter (용량, 연상도, 라인 크기,
 *                               포트 수, 클럭 등)
 *   - logic.h / CacheDynParam : XML에서 파싱된 캐시 동적 파라미터(duty_cycle,
 *                               throughput, latency, buffer 크기 등)
 * 이 파일에 의존하는 모듈:
 *   - processor.cc            : Processor가 SharedCache 객체를 생성·소유
 *   - processor.h             : SharedCache 전방 선언/사용
 *
 * === 주요 함수 요약 ===
 * SharedCache::SharedCache()   : XML 파라미터를 읽어 캐시/버퍼 ArrayST를 생성하고
 *                                면적을 누적한다.
 * SharedCache::computeEnergy() : TDP 또는 런타임 통계를 이용해 동적 에너지와
 *                                누설 전력을 계산하여 power/rt_power에 반영한다.
 * SharedCache::displayEnergy() : 계산된 면적/Peak Dynamic/누설/Runtime Dynamic을
 *                                출력한다.
 * SharedCache::set_cache_param() : cacheL(L2/L3/L1Directory/L2Directory)에 따라
 *                                  cachep 필드를 XML에서 채운다.
 */"""

shared_insertions = [
    ('#include "sharedcache.h"', SHARED_TOP, 0),

    ("SharedCache::SharedCache(ParseXML* XML_interface, int ithCache_,", """/*
 * [한국어]
 * SharedCache::SharedCache - 공유 캐시 전력 모델 생성자
 *
 * @XML_interface: ParseXML 객체 포인터 (시스템 설정 및 시뮬레이션 통계).
 * @ithCache_:    XML sys.L2[]/sys.L3[] 등에서 이 인스턴스의 인덱스.
 * @interface_ip_: CACTI InputParameter 초기값 포인터.
 * @cacheL_:      모델링할 캐시 계층 (L2/L3/L1Directory/L2Directory).
 *
 * 동작 순서:
 *   1. Private L2 여부에 따라 장치 유형(Core_device vs LLC_device)과 코어 유형 설정.
 *   2. Embedded/GPU 환경에 따른 와이어 타입(Global_30 vs Global) 선택.
 *   3. set_cache_param()으로 cachep(용량/연상도/라인/버퍼 크기 등) 초기화.
 *   4. 주 캐시 SRAM 배열(unicache.caches)과 missb/ifb/prefetchb/wbb를 생성.
 *   5. 각 ArrayST의 local_result.area를 unicache.area 및 Component::area에 누적.
 */""", 0),

    ("void SharedCache::computeEnergy(bool is_tdp) {", """/*
 * [한국어]
 * SharedCache::computeEnergy - 공유 캐시 동적 에너지 및 누설 전력 계산
 *
 * @is_tdp: true이면 TDP(Peak) 모드, false이면 런타임 통계 기반 모드.
 *
 * 동작 순서:
 *   1. TDP 모드: cachep.duty_cycle과 포트 수로부터 read/write 접근 횟수를 추정.
 *      Directory(ST)인 경우 search port 기반 접근만 계산.
 *   2. 런타임 모드: XML->sys.L2[], sys.L3[], sys.L1Directory[], sys.L2Directory[]의
 *      실제 read_accesses/write_accesses/misses/hits를 stats_t에 복사.
 *   3. homenode_data_access 비율로 SBT(디렉토리 병합) 캐시의 홈 노드 부하를 조정.
 *   4. 주 캐시 + missb/ifb/prefetchb/wbb의 동적 에너지를 local_result.power에
 *      stats_t를 곱해 누적(unicache.power_t).
 *   5. 누설 전력(local_result.power * pppm_lkg)을 더해 power(TDP) 또는
 *      rt_power(런타임)에 반영.
 */""", 0),

    ("void SharedCache::displayEnergy(uint32_t indent, bool is_tdp) {", """/*
 * [한국어]
 * SharedCache::displayEnergy - 공유 캐시 전력/면적 결과 출력
 *
 * @indent: 출력 들여쓰기 칸 수.
 * @is_tdp: true이면 TDP 모드 결과 출력(현재 TDP 블록만 구현되어 있음).
 *
 * 출력 항목:
 *   - 캐시 이름(Private L2 여부에 따라 indent 차이)
 *   - Area (um^2 → mm^2)
 *   - Peak Dynamic (W)
 *   - Subthreshold Leakage (W, longer_channel_device 선택)
 *   - Gate Leakage (W)
 *   - Runtime Dynamic (W)
 */""", 0),

    ("void SharedCache::set_cache_param() {", """/*
 * [한국어]
 * SharedCache::set_cache_param - XML에서 캐시 파라미터를 cachep로 복사
 *
 * cacheL 값에 따라 L2, L3, L1Directory, L2Directory 중 하나를 선택하고,
 * 각 계층의 clockRate, executionTime, device_type, capacity, blockW, assoc,
 * nbanks, throughput, latency, buffer_sizes, duty_cycle, 디렉토리 병합 여부 등을
 * XML->sys.* 에서 읽어 cachep 및 interface_ip 관련 필드를 채운다.
 */""", 0),
]

shared_replacements = [
    ('#include "sharedcache.h"', '#include "sharedcache.h"          // [한국어] SharedCache 클래스 선언', 0),
    ('#include <assert.h>', '#include <assert.h>              // [한국어] assert', 0),
    ('#include <string.h>', '#include <string.h>              // [한국어] string 함수', 0),
    ('#include <algorithm>', '#include <algorithm>             // [한국어] std::min/max 등', 0),
    ('#include <cmath>', '#include <cmath>                 // [한국어] ceil/log2 등', 0),
    ('#include <iostream>', '#include <iostream>              // [한국어] cout/cerr', 0),
    ('#include "XML_Parse.h"', '#include "XML_Parse.h"            // [한국어] ParseXML 및 시스템 파라미터', 0),
    ('#include "array.h"', '#include "array.h"              // [한국어] ArrayST/DataCache', 0),
    ('#include "cacti/arbiter.h"', '#include "cacti/arbiter.h"        // [한국어] CACTI arbiter', 0),
    ('#include "cacti/basic_circuit.h"', '#include "cacti/basic_circuit.h" // [한국어] cmos 누설 함수', 0),
    ('#include "cacti/parameter.h"', '#include "cacti/parameter.h"      // [한국어] CACTI InputParameter', 0),
    ('#include "const.h"', '#include "const.h"               // [한국어] McPAT 상수', 0),
    ('#include "io.h"', '#include "io.h"                  // [한국어] CACTI 입출력', 0),
    ('#include "logic.h"', '#include "logic.h"               // [한국어] CacheDynParam 등', 0),
]

apply(ROOT / "sharedcache.cc", shared_insertions, shared_replacements)
