#!/usr/bin/env python3
"""
[한국어] accelwattch/memoryctrl.cc 한국어 주석 삽입 스크립트
- 원본 들여쓰기를 그대로 두고, 파일 상단/함수 상단/주요 분기 앞에
  한국어 블록 주석을 추가한다.
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

MEM_TOP = """/*
 * [한국어 설명] AccelWattch 메모리 컨트롤러 전력 모델 구현 (memoryctrl.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT/AccelWattch의 메모리 컨트롤러(MC)를 모델링한다.
 * MC는 FrontEnd(리오더 버퍼/읽기 버퍼/쓰기 버퍼/PRT/ThreadMasks/PRC),
 * Backend(transaction engine), PHY(SerDes 기반 물리 계층), 그리고 DRAM
 * 전력(DRAM IDD 계수 기반) 네 부분으로 구성된다. GPU 메모리 공동 발행
 *(coalescing)을 위한 PRT(Pending Request Table), ThreadMasks, PRC 등의
 * SRAM 구조물도 이 파일에서 ArrayST로 생성된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Processor 클래스가 GPU/CPU 설정에 따라 MemoryController 객체를 생성하고,
 * 이는 다시 MCFrontEnd, MCBackend, MCPHY, DRAM 객체를 구성한다.
 *   Processor::Processor() → new MemoryController()
 *                            → frontend(MCFrontEnd), transecEngine(MCBackend),
 *                               PHY(MCPHY), dram(DRAM)
 *   Processor::computeEnergy() → MemoryController::computeEnergy()
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 및 주기적 전력 집계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: memoryctrl.h, XML_Parse.h, basic_components.h, logic.h,
 *       array.h(ArrayST), parameter.h, cacti/basic_circuit.h.
 * 의존 받음: processor.cc(Processor의 생성/소멸/전력 집계).
 * 데이터 흐름: XML->sys.mc.* → set_mc_param() → mcp
 *             → MCFrontEnd/MCBackend/MCPHY/DRAM 생성 → computeEnergy()
 *             → power/rt_power.
 *
 * === 주요 함수/클례스 요약 ===
 * MCBackend       — transaction engine/backend 전력 (경험적 curve fitting).
 * MCPHY           — 메모리 PHY 전력 (mW/Gb/s 기반 스케일링).
 * MCFrontEnd      — 리오더 버퍼, 읽기/쓰기 버퍼, GPU coalescing 자료구조.
 * DRAM            — 시뮬레이션에서 수집된 메모리 통계에 기반한 DRAM 동적 전력.
 * MemoryController — 위 네 구성요소를 생성·연결·집계하는 최상위 MC 클래스.
 */"""

mem_insertions = [
    ("#include \"io.h\"", """/*""", 0),
    ("MCBackend::MCBackend(", """/*""", 0),
    ("void MCBackend::compute()", """/*""", 0),
    ("  if (mc_type == MC) {", """  // [한국어] MC/Flash 분기 (현재 MC만 지원)""", 0),
    ("    if (mcp.type == 0) {", """    // [한국어] 고성능 MC (Niagara 기반)""", 0),
    ("    } else {
      NMOS_sizing = g_tp.min_w_nmos_;", """    // [한국어] 저전력 MC (Cadence 기반)""", 0),
    ("  } else {  // skip old model", """  // [한국어] MC/Flash 이외 타입은 종료""", 0),
    ("void MCBackend::computeEnergy(bool is_tdp)", """/*""", 0),
    ("  if (is_tdp) {
    power.reset();", """  // [한국어] TDP/런타임 분기""", 0),
    ("    stats_t.readAc.access = 0.5 * mcp.num_channels;", """    // [한국어] TDP peak: 채널당 0.5 access 가정""", 0),
    ("    stats_t.readAc.access = mcp.reads;", """    // [한국어] 런타임: XML에서 수집한 read/write 통계 사용""", 0),
    ("    power.readOp.dynamic = (stats_t.readAc.access + stats_t.writeAc.access) *", """    // [한국어] TDP 동적 전력 = (read+write) × per-access energy""", 0),
    ("    rt_power.readOp.dynamic = (stats_t.readAc.access + stats_t.writeAc.access) *", """    // [한국어] 런타임 동적 전력 = access × blockSize×8 / dataBusWidth × per-access energy""", 0),
    ("    rt_power = rt_power + power_t * pppm_lkg;", """    // [한국어] 런타임 누설 전력 추가""", 0),
    ("    rt_power.readOp.dynamic =
        rt_power.readOp.dynamic + power.readOp.dynamic * 0.1 * mcp.clockRate *", """    // [한국어] refresh/scrubbing 등 루틴 작업 10% 추가""", 0),
    ("MCPHY::MCPHY(", """/*""", 0),
    ("void MCPHY::compute()", """/*""", 0),
    ("    if (mcp.type == 0) {", """    // [한국어] 고성능 PHY""", 1),
    ("      power_t.readOp.dynamic = power_per_gb_per_s * sqrt(l_ip.F_sz_um / 0.09) *", """      // [한국어] PHY 동적 전력 = mW/Gb/s × sqrt(공정비) × (Vdd/1.2)²""", 0),
    ("      double non_IO_percentage = 0.2;", """      // [한국어] PHY 면적 중 IO를 제외한 로직 비율 20%""", 0),
    ("      area.set_area(1.3 * non_IO_percentage / 2133.0e6 * mcp.clockRate / 17066 *", """      // [한국어] DesignWare 16bit DDR3 PHY 1.3mm²@40nm 기준 스케일링""", 0),
    ("void MCPHY::computeEnergy(bool is_tdp)", """/*""", 0),
    ("    double data_transfer_unit = (mc_type == MC) ? 72 : 16; /*DIMM data width*/", """    // [한국어] DIMM 데이터 폭: MC=72bit, Flash=16bit""", 0),
    ("    power.readOp.dynamic =
        power.readOp.dynamic *
        (mcp.peakDataTransferRate * 8 * 1e6 / 1e9 /*change to Gbs*/) *", """    // [한국어] TDP PHY 동적 전력 = peak 대역폭 × bus 폭 비율 × 채널 수 / clockRate""", 0),
    ("    rt_power.readOp.dynamic = power_t.readOp.dynamic *", """    // [한국어] 런타임 PHY 동적 전력""", 0),
    ("MCFrontEnd::MCFrontEnd(", """/*""", 0),
    ("  tag = mcp.addressBusWidth + EXTRA_TAG_BITS + mcp.opcodeW;", """  // [한국어] 리오더 버퍼 CAM 태그 폭 = 주소 + 여유 태그 + opcode""", 0),
    ("  data = int(ceil((XML->sys.physical_address_width + mcp.opcodeW) / 8.0));", """  // [한국어] 리오더 버퍼 데이터 폭 = 물리주소 + opcode [byte]""", 0),
    ("  frontendBuffer =
      new ArrayST(&interface_ip, \"MC ReorderBuffer\", Uncore_device);", """  // [한국어] 메모리 요청 리오더 버퍼 (CAM+RAM)""", 0),
    ("  MC_arb =
      new selection_logic(is_default, XML->sys.mc.req_window_size_per_channel,", """  // [한국어] MC 요청 선택/조정(selection_logic)""", 0),
    ("  readBuffer = new ArrayST(&interface_ip, \"MC ReadBuffer\", Uncore_device);", """  // [한국어] 읽기 버퍼""", 0),
    ("  writeBuffer = new ArrayST(&interface_ip, \"MC writeBuffer\", Uncore_device);", """  // [한국어] 쓰기 버퍼""", 0),
    ("  // SRAM structures for memory coalescing --Syed Gilani", """  // [한국어] GPU 메모리 coalescing을 위한 SRAM 구조물 (Syed Gilani)""", 0),
    ("  PRT = new ArrayST(&interface_ip, \"MC PRT\", Uncore_device);", """  // [한국어] Pending Request Table (coalescing 기본 주소/오프셋/TID 저장)""", 0),
    ("  threadMasks = new ArrayST(&interface_ip, \"MC ThreadMasks\", Uncore_device);", """  // [한국어] coalescing된 스레드 마스크 저장""", 0),
    ("  PRC = new ArrayST(&interface_ip, \"MC PendingRequestCount\", Uncore_device);", """  // [한국어] PRT 항목당 pending request 개수 저장""", 0),
    ("void DRAM::computeEnergy(bool is_tdp)", """/*""", 0),
    ("  power_t.readOp.dynamic += XML->sys.mc.memory_reads * dramp.rd_coeff;", """  // [한국어] DRAM read 동적 전력 = memory_reads × rd_coeff""", 0),
    ("  power_t.readOp.dynamic += XML->sys.mc.memory_writes * dramp.wr_coeff;", """  // [한국어] DRAM write 동적 전력 = memory_writes × wr_coeff""", 0),
    ("  power_t.readOp.dynamic += XML->sys.mc.dram_pre * dramp.pre_coeff;", """  // [한국어] DRAM precharge 동적 전력 = dram_pre × pre_coeff""", 0),
    ("void MCFrontEnd::computeEnergy(bool is_tdp)", """/*""", 0),
    ("    frontendBuffer->stats_t.readAc.access =
        frontendBuffer->l_ip.num_search_ports;", """    // [한국어] TDP: 리오더 버퍼 read = search 포트 수""", 0),
    ("    frontendBuffer->stats_t.readAc.access =
        XML->sys.mc.memory_reads * mcp.llcBlockSize * 8.0 / mcp.dataBusWidth *
        mcp.dataBusWidth / 72;", """    // [한국어] 런타임: 리오더 버퍼 read access""", 0),
    ("    PRT->stats_t.readAc.access = XML->sys.core[0].dcache.read_accesses +", """    // [한국어] PRT access = core[0] dcache/ccache/tcache 접근 합""", 0),
    ("    threadMasks->stats_t.readAc.access = XML->sys.core[0].dcache.read_accesses +", """    // [한국어] threadMasks access = core[0] 캐시 접근 합""", 0),
    ("    PRC->stats_t.readAc.access = XML->sys.core[0].dcache.read_accesses +", """    // [한국어] PRC access = core[0] 캐시 접근 합""", 0),
    ("  double perAccessCoalescingEnergy =", """  // [한국어] Verilog/Synopsys PowerCompiler 기반 coalescing 논리 에너지""", 0),
    ("  if (is_tdp) {
    power =
        power + frontendBuffer->power_t + readBuffer->power_t +", """  // [한국어] TDP: 프론트엔드 구조물 동적+누설 전력 집계""", 0),
    ("  } else {
    rt_power =
        rt_power + frontendBuffer->power_t + readBuffer->power_t +", """  } else {
    // [한국어] 런타임: 프론트엔드 구조물 동적+누설 전력 집계""", 0),
    ("void MCFrontEnd::displayEnergy(uint32_t indent, int plevel, bool is_tdp)", """/*""", 0),
    ("DRAM::DRAM(", """/*""", 0),
    ("MemoryController::MemoryController(", """/*""", 0),
    ("  set_mc_param();", """  // [한국어] XML에서 MC 파라미터 복사""", 0),
    ("  frontend = new MCFrontEnd(XML, &interface_ip, mcp, mc_type);", """  // [한국어] MC 프론트엔드 생성""", 0),
    ("  dram = new DRAM(XML, &interface_ip, dram_type_);", """  // [한국어] DRAM 전력 모델 생성""", 0),
    ("  transecEngine = new MCBackend(&interface_ip, mcp, mc_type);", """  // [한국어] MC backend(transaction engine) 생성""", 0),
    ("  if (mcp.type == 0 || (mcp.type == 1 && mcp.withPHY)) {", """  // [한국어] 고성능 MC 또는 저전력 MC+PHY 옵션일 때 PHY 생성""", 0),
    ("    PHY = new MCPHY(&interface_ip, mcp, mc_type);", """    // [한국어] 메모리 PHY 생성""", 0),
    ("void MemoryController::computeEnergy(bool is_tdp)", """/*""", 0),
    ("  if (is_tdp) {
    power = power + frontend->power + transecEngine->power;", """  // [한국어] TDP: frontend/backend/PHY(존재 시) 전력 집계""", 0),
    ("  } else {
    rt_power = rt_power + frontend->rt_power + transecEngine->rt_power +
               dram->rt_power;", """  } else {
    // [한국어] 런타임: frontend/backend/DRAM/PHY(존재 시) 전력 집계""", 0),
    ("void MemoryController::displayEnergy(uint32_t indent, int plevel, bool is_tdp)", """/*""", 0),
    ("void DRAM::set_dram_param()", """/*""", 0),
    ("void MemoryController::set_mc_param()", """/*""", 0),
    ("    mcp.clockRate = XML->sys.mc.mc_clock * 2;  // DDR double pumped", """    // [한국어] DDR 더블 펌프: 실제 클록 = 2×mc_clock""", 0),
    ("    mcp.llcBlockSize = int(ceil(XML->sys.mc.llc_line_length / 8.0)) +
                       XML->sys.mc.llc_line_length;  // ecc overhead", """    // [한국어] LLC block size [byte] + ECC 오버헤드""", 0),
    ("    mcp.dataBusWidth =
        int(ceil(XML->sys.mc.databus_width / 8.0)) + XML->sys.mc.databus_width;", """    // [한국어] 데이터 버스 폭 [bit] + ECC 비트""", 0),
    ("MCFrontEnd ::~MCFrontEnd()", """/*""", 0),
    ("MemoryController ::~MemoryController()", """/*""", 0),
]
mem_replacements = [
    ('#include "io.h"', '#include "io.h"                 // [한국어] McPAT I/O 출력 유틸리티', 0),
    ('#include "parameter.h"', '#include "parameter.h"           // [한국어] CACTI InputParameter, init_interface', 0),
    ('#include "const.h"', '#include "const.h"               // [한국어] McPAT 회로 상수 (nand, inv 등)', 0),
    ('#include "logic.h"', '#include "logic.h"               // [한국어] selection_logic 등 논리 회로', 0),
    ('#include "cacti/basic_circuit.h"', '#include "cacti/basic_circuit.h" // [한국어] cmos_Isub_leakage, gate_C 등', 0),
    ('#include <iostream>', '#include <iostream>              // [한국어] cout/cerr', 0),
    ('#include <algorithm>', '#include <algorithm>             // [한국어] std::min/max 등', 0),
    ('#include "XML_Parse.h"', '#include "XML_Parse.h"            // [한국어] ParseXML 및 sys.mc 파라미터', 0),
    ('#include <string>', '#include <string>                // [한국어] std::string', 0),
    ('#include <cmath>', '#include <cmath>                 // [한국어] log, ceil 등', 0),
    ('#include <assert.h>', '#include <assert.h>              // [한국어] assert', 0),
    ('#include "memoryctrl.h"', '#include "memoryctrl.h"          // [한국어] MC 클래스 선언', 0),
    ('#include "basic_components.h"', '#include "basic_components.h"    // [한국어] Component, powerDef, longer_channel_device_reduction', 0),
]

apply(ROOT / "memoryctrl.cc", mem_insertions, mem_replacements)
