#!/usr/bin/env python3
"""
[한국어] accelwattch/processor.cc 한국어 주석 삽입 스크립트
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

PROC_TOP = """/*
 * [한국어 설명] AccelWattch 최상위 프로세서 전력 모델 구현 (processor.cc)
 *
 * === 파일의 역할 ===
 * Processor 클래스는 McPAT/AccelWattch 전력 모델의 최상위 집계자이다.
 * XML에서 GPU/CPU 시스템 파라미터를 읽어 Core, SharedCache(L2/L3/L1Dir/L2Dir),
 * MemoryController, NIUController, PCIeController, FlashController, NoC 등의
 * 구성요소 객체를 생성하고, 각각의 면적과 전력을 합산하여 전체 프로세서의
 * power(TDP)와 rt_power(런타임)을 산출한다. 동질(homogeneous) 구성요소는
 * set_pppm()으로 단일 객체 결과를 전체 개수만큼 확장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 루트. main.cc에서 Processor 객체를 생성한 뒤
 * compute()/displayEnergy()를 호출하여 전체 전력 보고서를 얻는다.
 *   main() → new Processor(XML) → computeEnergy()/compute() → displayEnergy()
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 및 종료 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: processor.h, XML_Parse.h, Core, SharedCache, MemoryController,
 *       NIUController, PCIeController, FlashController, NoC, array.h,
 *       cacti/parameter.h, version.h.
 * 데이터 흐름: XML → set_proc_param() → procdynp/interface_ip
 *             → 구성요소 생성 → computeEnergy(false) → power/rt_power 집계
 *             → displayEnergy() 출력.
 *
 * === 주요 함수 요약 ===
 * Processor::Processor()   — XML 파싱, 모든 구성요소 생성/초기화/전력 집계.
 * Processor::compute()     — 런타임 전력 재계산 (사이클 통계 기반).
 * Processor::displayEnergy() — 전체 및 하위 구성요소 전력/면적 출력.
 * Processor::set_proc_param() — XML->sys.*를 procdynp/interface_ip로 복사.
 * Processor::~Processor()  — 동적 할당 객체 해제.
 */"""

proc_insertions = [
    ('#include "processor.h"', PROC_TOP, 0),

    ("Processor::Processor(ParseXML *XML_interface)", """/*
 * [한국어]
 * Processor::Processor - 최상위 프로세서 전력 모델 생성자
 *
 * @XML_interface: ParseXML 객체 포인터 (GPU/CPU 시스템 설정 및 시뮬레이션 통계).
 * @return: (생성자) — power/rt_power에 전체 전력을 누적.
 *
 * 동작 순서:
 *   1. set_proc_param()으로 프로세서 전역 파라미터 초기화.
 *   2. 동질(homo*) 플래그에 따라 numCore/numL2/numL3/numNOC 등 결정.
 *   3. Core, SharedCache(L2/L3/Directory), MemoryController, Flash, NIU,
 *      PCIe, NoC 객체를 생성하고 computeEnergy() 호출.
 *   4. 각 구성요소의 면적/전력을 power/rt_power에 합산. 동질 구성요소는
 *      set_pppm()으로 numXX 배 확장.
 */""", 0),
    ("  rt_power.reset();", "  // [한국어] 전체 런타임 전력 누적값 초기화", 0),
    ("  set_proc_param();", "  // [한국어] XML에서 프로세서 전역 파라미터 복사", 0),
    ("  if (procdynp.homoCore)", "  // [한국어] 동질 코어일 때 numCore를 0/1로 축소", 0),
    ("  if (procdynp.homoL2)", "  // [한국어] 동질 L2 캐시일 때 numL2를 0/1로 축소", 0),
    ("  if (procdynp.homoL3)", "  // [한국어] 동질 L3 캐시일 때 numL3를 0/1로 축소", 0),
    ("  if (procdynp.homoNOC)", "  // [한국어] 동질 NoC일 때 numNOC를 0/1로 축소", 0),
    ("  for (i = 0; i < numCore; i++) {", "  // [한국어] 코어 생성 및 TDP/런타임 전력 집계", 0),
    ("    cores.push_back(new Core(XML, i, &interface_ip));", "    // [한국어] i번째 Core 객체 생성", 0),
    ("    cores[i]->computeEnergy();", "    // [한국어] Core TDP 전력 계산", 0),
    ("    cores[i]->computeEnergy(false);", "    // [한국어] Core 런타임 전력 계산", 0),
    ("      core.area.set_area(core.area.get_area() +\n                         cores[i]->area.get_area() * procdynp.numCore);", "      // [한국어] 동질 코어: 단일 코어 면적 × 코어 수", 0),
    ("      core.power = core.power + cores[i]->power * pppm_t;", "      // [한국어] 동질 코어 TDP 전력 확장", 0),
    ("      core.rt_power = core.rt_power + cores[i]->rt_power * pppm_t;", "      // [한국어] 동질 코어 런타임 전력 확장", 0),
    ("      power = power + core.power;", "      // [한국어] 전체 전력에 코어 합산", 0),
    ("      rt_power = rt_power + core.rt_power;", "      // [한국어] 전체 런타임 전력에 코어 합산", 0),

    ("  if (!XML->sys.Private_L2) {", "  // [한국어] Private L2가 아닐 경우 별도 L2 캐시 집계", 0),
    ("        l2array.push_back(new SharedCache(XML, i, &interface_ip));", "        // [한국어] i번째 Shared L2 생성", 0),
    ("  if (numL3 > 0)", "  // [한국어] L3 캐시 생성 및 집계", 0),
    ("  if (numL1Dir > 0)", "  // [한국어] L1 Directory 생성 및 집계", 0),
    ("  if (numL2Dir > 0)", "  // [한국어] L2 Directory 생성 및 집계", 0),

    ("  if (XML->sys.mc.number_mcs > 0 && XML->sys.mc.memory_channels_per_mc > 0) {", "  // [한국어] 메모리 컨트롤러 생성 및 집계", 0),
    ("    if (XML->sys.architecture == 1)  // 1 for fermi", "    // [한국어] Fermi 아키텍처: GDDR5 메모리 컨트롤러", 0),
    ("    else if (XML->sys.architecture == 2)  // 2 for quadro", "    // [한국어] Quadro 아키텍처: GDDR3 메모리 컨트롤러", 0),
    ("    mcs.power = mc->power * pppm_t;", "    // [한국어] MC TDP 전력 × MC 개수 확장", 0),

    ("  if (XML->sys.flashc.number_mcs > 0)  // flash controller", "  // [한국어] Flash/SSD 컨트롤러 생성 및 집계", 0),
    ("  if (XML->sys.niu.number_units > 0) {", "  // [한국어] NIU(네트워크 인터페이스) 생성 및 집계", 0),
    ("  if (XML->sys.pcie.number_units > 0 && XML->sys.pcie.num_channels > 0) {", "  // [한국어] PCIe 컨트롤러 생성 및 집계", 0),

    ("  if (numNOC > 0) {", "  // [한국어] NoC/버스 인터커넥트 생성 및 집계", 0),
    ("      if (XML->sys.NoC[i].type) {  // First add up area of routers if NoC is", "      // [한국어] NoC(라우터 기반) 인터커넥트", 0),
    ("      } else {  // Bus based interconnect", "      // [한국어] 버스 기반 인터커넥트", 0),

    ("void Processor::compute()", """/*
 * [한국어]
 * Processor::compute - 런타임 전력 재계산
 *
 * Processor::Processor에서 이미 TDP/초기 런타임 전력을 집계했지만,
 * 이 함수는 시뮬레이션 중 수집된 실제 사이클 통계를 바탕으로 각 구성요소의
 * 런타임 전력을 다시 계산한다. core[0]의 클록을 기준으로 executionTime을
 * 설정하고, 동질 구성요소에 대해 set_pppm()으로 확장한다.
 */""", 0),
    ("  core.rt_power.reset();", "  // [한국어] 코어 런타임 전력 초기화", 0),
    ("  for (i = 0; i < numCore; i++) {", "  // [한국어] 코어 런타임 전력 재계산", 1),
    ("  if (!XML->sys.Private_L2) {", "  // [한국어] Shared L2 런타임 전력 재계산", 1),
    ("  l3.rt_power.reset();", "  // [한국어] L3 런타임 전력 초기화", 0),
    ("  l1dir.rt_power.reset();", "  // [한국어] L1 Directory 런타임 전력 초기화", 0),
    ("  l2dir.rt_power.reset();", "  // [한국어] L2 Directory 런타임 전력 초기화", 0),
    ("  mcs.rt_power.reset();", "  // [한국어] MC 런타임 전력 초기화", 0),
    ("  noc.rt_power.reset();", "  // [한국어] NoC 런타임 전력 초기화", 0),

    ("void Processor::displayDeviceType(int device_type_, uint32_t indent)", """/*
 * [한국어]
 * Processor::displayDeviceType - ITRS 장치 유형 출력
 *
 * @device_type_: 0=HP, 1=LSTP, 2=LOP, 3=LP-DRAM, 4=COMM-DRAM.
 */""", 0),
    ("void Processor::displayInterconnectType(int interconnect_type_,", """/*
 * [한국어]
 * Processor::displayInterconnectType - 인터커넥트 공정 전망 출력
 */""", 0),
    ("void Processor::displayEnergy(uint32_t indent, int plevel, bool is_tdp_parm)", """/*
 * [한국어]
 * Processor::displayEnergy - 전체 프로세서 및 하위 구성요소 전력/면적 출력
 *
 * @plevel: 출력 상세 수준. plevel>1이면 코어/캐시/MC/NoC 등의 상세 결과도
 *          재귀적으로 출력한다.
 * @is_tdp_parm: true이면 TDP(peak) 결과, false이면 런타임 결과.
 */""", 0),
    ("  bool long_channel = XML->sys.longer_channel_device;", "  // [한국어] 장채널 소자 사용 여부", 0),
    ("  if (is_tdp_parm) {", "  // [한국어] TDP/런타임 출력 분기", 0),
    ("    if (plevel < 5) {", "    // [한국어] 낮은 출력 레벨일 때 요약 메시지", 0),
    ("    cout << indent_str << \"Area = \" << area.get_area() * 1e-6 << \" mm^2\"", "    // [한국어] 전체 칩 면적", 0),
    ("    cout << indent_str << \"Peak Power = \"", "    // [한국어] 전체 피크 전력 (동적 + 누설)", 0),
    ("    cout << indent_str << \"Runtime Dynamic = \" << rt_power.readOp.dynamic", "    // [한국어] 전체 런타임 동적 전력", 0),

    ("void Processor::set_proc_param()", """/*
 * [한국어]
 * Processor::set_proc_param - XML에서 프로세서 전역 파라미터 초기화
 *
 * homogeneous_* 플래그, 코어/캐시/NoC/Directory/MC 개수, CACTI InputParameter
 * (기술 노드, 온도, 와이어 타입, 설계 가중치 등)를 XML->sys.*에서 복사한다.
 */""", 0),
    ("  procdynp.homoCore = bool(debug ? 1 : XML->sys.homogeneous_cores);", "  // [한국어] 코어 동질성 여부", 0),
    ("  procdynp.numCore = XML->sys.number_of_cores;", "  // [한국어] 총 코어 수", 0),
    ("  procdynp.numMC = XML->sys.mc.number_mcs;", "  // [한국어] 메모리 컨트롤러 수", 0),
    ("  interface_ip.F_sz_nm =", "  // [한국어] 핵심 기술 노드 [nm]", 0),
    ("  interface_ip.F_sz_um = interface_ip.F_sz_nm / 1000;", "  // [한국어] 기술 노드 [um] 변환", 0),

    ("Processor::~Processor() {", """/*
 * [한국어]
 * Processor::~Processor - 동적 할당한 구성요소 객체 해제
 */""", 0),
]

proc_replacements = [
    ('#include "processor.h"', '#include "processor.h"          // [한국어] Processor 클래스 선언', 0),
    ('#include <assert.h>', '#include <assert.h>              // [한국어] assert', 0),
    ('#include <stdio.h>', '#include <stdio.h>               // [한국어] printf', 0),
    ('#include <string.h>', '#include <string.h>              // [한국어] string 함수', 0),
    ('#include <algorithm>', '#include <algorithm>             // [한국어] std::min/max 등', 0),
    ('#include <cmath>', '#include <cmath>                 // [한국어] sqrt 등', 0),
    ('#include <fstream>', '#include <fstream>               // [한국어] 파일 입출력', 0),
    ('#include <iostream>', '#include <iostream>              // [한국어] cout/cerr', 0),
    ('#include "XML_Parse.h"', '#include "XML_Parse.h"            // [한국어] ParseXML 및 시스템 파라미터', 0),
    ('#include "array.h"', '#include "array.h"              // [한국어] ArrayST', 0),
    ('#include "cacti/basic_circuit.h"', '#include "cacti/basic_circuit.h" // [한국어] cmos 누설 함수', 0),
    ('#include "const.h"', '#include "const.h"               // [한국어] McPAT 상수', 0),
    ('#include "parameter.h"', '#include "parameter.h"           // [한국어] CACTI InputParameter', 0),
    ('#include "version.h"', '#include "version.h"            // [한국어] McPAT 버전 정보', 0),
]

apply(ROOT / "processor.cc", proc_insertions, proc_replacements)
