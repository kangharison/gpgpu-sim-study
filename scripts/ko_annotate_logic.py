#!/usr/bin/env python3
"""
[한국어] accelwattch/logic.cc 한국어 주석 삽입 스크립트
- 원본 코드/들여쓰기를 보존하고, 파일 상단에 4-섹션 블록, 함수 상단에
  한국어 설명 블록, 주요 계산/분기 라인 끝에 인라인 주석을 추가한다.
"""
from pathlib import Path


def insert_before(content: str, anchor: str, text: str, occurrence: int = 0) -> str:
    """anchor가 포함된 행 앞에 text(주석 블록)를 삽입한다."""
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
    """old를 occurrence 번째에 new로 치환한다(단일 라인 끝 주석용)."""
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

# -----------------------------------------------------------------------------
# logic.cc
# -----------------------------------------------------------------------------
LOGIC_TOP = """/*
 * [한국어 설명] AccelWattch 논리 회로/기능 유닛 전력 모델 구현 (logic.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT 스타일의 디지털 논리 블록들 — Issue Window 선택 논리,
 * 의존성/자원 충돌 검사기, DFF 셀, 파이프라인 레지스터, 기능 유닛(FPU/ALU/MUL),
 * 미분화 코어(UndiffCore), 명령어 디코더 — 의 동적/정적 전력을 트랜지스터
 * 수준에서 추정한다. CACTI의 gate_C/drain_C_ 함수와 cmos_Isub_leakage/
 * cmos_Ig_leakage 함수를 사용해 0.8um/65nm/90nm 등의 공정 기준 데이터를
 * 대상 기술 노드로 스케일링한다. AccelWattch에서 GPU SM 코어의 ALU/FPU/MUL,
 * 파이프라인, 디코더 등의 전력이 이 파일을 통해 계산된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Core 클래스가 이 파일의 클래스 객체를 멤버로 생성하여 개별 코어 전력을 구성한다.
 *   Core::Core() → selection_logic, dep_resource_conflict_check, Pipeline,
 *                   FunctionalUnit(FPU/ALU/MUL), inst_decoder, UndiffCore 등 생성
 *   Core::computeEnergy() → 각 멤버의 computeEnergy()/compute() 호출
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계 및 주기적 전력 집계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: logic.h(클스 선언), XML_Parse.h(ParseXML, CoreDynParam),
 *       basic_components.h(powerDef, Component, longer_channel_device_reduction),
 *       cacti/basic_circuit.h(cmos_Isub_leakage, gate_C, drain_C_, Decoder, Predec 등),
 *       cacti/parameter.h(InputParameter, g_tp, g_ip, init_interface).
 * 의존 받음: core.cc(Core의 구성 요소), processor.cc(최종 집계).
 * 데이터 흐름: XML/코어 파라미터 → 생성자 → 면적/누설/기준 동적 에너지 산출
 *             → computeEnergy()에서 stats_t 접근수(access)를 곱해 TDP/런타임 전력 산출.
 *
 * === 주요 함수/클례스 요약 ===
 * selection_logic::selection_power()      — issue-window 선택기의 OR/arbiter 전력.
 * dep_resource_conflict_check             — Scoreboard/CAM 스타일 비교기 충돌 검사.
 * DFFCell                                  — NAND2 기반 D 플립플롭 셀 전력.
 * Pipeline                                 — 코어/비코어 파이프라인 레지스터 전력.
 * FunctionalUnit                           — FPU/ALU/MUL/SFU 전력 (AccelWattch GPU 핵심).
 * UndiffCore                               — 코어 공통 로직(front-end/back-end 등) 전력.
 * inst_decoder                             — n-to-2^n 디코더 및 x86 시퀀서 전력.
 * leakage_feedback()                      — 온도 변화에 따른 누설 재계산.
 */"""

logic_insertions = [
    ('#include "logic.h"', LOGIC_TOP, 0),

    # selection_logic
    ("selection_logic::selection_logic(", """/*
 * [한국어]
 * selection_logic::selection_logic - Issue Window 선택 논리 생성자
 *
 * @configure_interface: CACTI InputParameter (공정/전압/온도).
 * @device_ty_, core_ty_: 장치/코어 유형 — 장채널 누설 보정에 사용.
 * @return: (생성자) — power.readOp에 동적/누설/게이트 누설 저장.
 *
 * 설정값을 복사하고 CACTI를 초기화한 뒤, selection_power()로 선택 논리의
 * OR/arbiter/encoder 전력을 계산한다. 소켓 효과(sckt_co_eff)와 장채널
 * 누설 보정을 적용한다.
 */""", 0),
    ("void selection_logic::selection_power()", """/*
 * [한국어]
 * selection_logic::selection_power - 비용 효율적인 슈퍼스칼라 선택기 전력
 *
 * TR pp.27-31 기반: 4입력 OR(anyreq), 4비트 우선순위 인코더, enable/grant
 * 인버터 체인의 캐패시턴스를 합산하여 동적 에너지를 산출한다. win_entries가
 * 4를 초과하면 트리 arbiter 개수(num_arbiter)를 증가시킨다.
 * 동적 에너지에 2를 곱하는 이유는 arbitration 신호가 왕복(round trip)하기
 * 때문이다.
 */""", 0),

    # dep_resource_conflict_check
    ("dep_resource_conflict_check::dep_resource_conflict_check(", """/*
 * [한국어]
 * dep_resource_conflict_check::dep_resource_conflict_check - 의존성/자원 충돌 검사기 생성자
 *
 * @configure_interface: CACTI InputParameter.
 * @dyn_p_: 코어 동적 파라미터 (decodeW, core_ty 등).
 * @compare_bits_: 비교할 비트 수. Inorder/OOO 모두 opcode+reg_tag 비트 추가.
 * @return: (생성자) — conflict_check_power()로 전력 산출.
 *
 * 트랜지스터 폭을 0.8um 기준에서 현재 공정으로 스케일링하고, Inorder/OOO에
 * 따라 compare_bits에 opcode/tag 비트를 추가한 뒤 충돌 검사 전력을 계산한다.
 */""", 0),
    ("void dep_resource_conflict_check::conflict_check_power()", """/*
 * [한국어]
 * dep_resource_conflict_check::conflict_check_power - Scoreboard/CAM 스타일 충돌 검사 전력
 *
 * 비교기 수는 decodeW 기준으로 source-to-dest 2(N²-N) + dest-to-dest (N²-N)로
 * 계산한다. 각 비교기는 compare_bits×2개의 NMOS를 포함하며, 누설은
 * simplified_nmos_leakage()로, 게이트 누설은 cmos_Ig_leakage()로 산출한다.
 */""", 0),
    ("double dep_resource_conflict_check::compare_cap()", """/*
 * [한국어]
 * dep_resource_conflict_check::compare_cap - CAM/tag-매치 스타일 비교기 용량
 *
 * 비교기 하단(비교 트랜지스터)과 상단(NOR match)의 드레인/게이트 용량을 합산.
 * fan-in에 따라 큰 NOR 게이트의 WNORp를 재조정한다.
 */""", 0),
    ("void dep_resource_conflict_check::leakage_feedback(double temperature)", """/*
 * [한국어]
 * dep_resource_conflict_check::leakage_feedback - 온도 변화에 따른 누설 재계산
 *
 * @temperature: 새 절대 온도(K). 10K 단위로 반올림하여 CACTI 파라미터 갱신.
 * @return: void — power.readOp.leakage, longer_channel_leakage, gate_leakage 갱신.
 */""", 0),

    # DFFCell
    ("DFFCell::DFFCell(", """/*
 * [한국어]
 * DFFCell::DFFCell - NAND2 기반 D 플립플롭 셀 생성자
 *
 * 5개의 NAND2 + 1개의 NAND3 면적으로 DFF 면적을 추정한다. cell_load는
 * 다음 단에 구동할 부하 캐패시턴스이며, compute_DFF_cell()에서 동적/정적
 * 전력을 계산한다.
 */""", 0),
    ("double DFFCell::fpfp_node_cap(", """/*
 * [한국어]
 * DFFCell::fpfp_node_cap - DFF 남부 노드 총 캐패시턴스
 *
 * @fan_in, fan_out: 입력/출력 팬 수.
 * @return: 드레인 캡 + fan_out×게이트 캡.
 */""", 0),
    ("void DFFCell::compute_DFF_cell()", """/*
 * [한국어]
 * DFFCell::compute_DFF_cell - DFF 남 남부/게이트/클록/유지 전력 계산
 *
 * 6개 남(NAND2×5 + NAND3×1)의 캐패시턴스를 계산하고, 0→1 전이(switch),
 * 1 유지(keep_1), 0 유지(keep_0), 클록 충전(e_clock)별 에너지를 분리한다.
 * 정적 전력은 NAND2 5개 + NAND3 1개의 누설/게이트 누설을 합산한다.
 */""", 0),

    # Pipeline
    ("Pipeline::Pipeline(", """/*
 * [한국어]
 * Pipeline::Pipeline - 파이프라인 레지스터 생성자
 *
 * 코어 파이프라인(is_core_pipeline)이면 파이프라인 단계별 비트 수를
 * CoreDynParam에서 추정(compute_stage_vector). 비코어 파이프라인이면
 * XML pipeline_stages/per_stage_vector를 직접 사용한다.
 * Embedded 프로세서가 아닐 경우 고성능 트랜지스터 폭을 사용한다.
 */""", 0),
    ("void Pipeline::compute()", """/*
 * [한국어]
 * Pipeline::compute - 파이프라인 레지스터 면적·전력 합산
 *
 * 단일 DFF 셀 전력을 num_piperegs로 확장. McPAT은 최악의 경우를 가정해
 * switch/keep_0/keep_1 상태를 평균(1/3)으로 처리한다. 소켓 효과와
 * 매크로 레이아웃 오버헤드를 적용한다.
 */""", 0),
    ("void Pipeline::compute_stage_vector()", """/*
 * [한국어]
 * Pipeline::compute_stage_vector - 코어 파이프라인 단계별 비트 수 추정
 *
 * Inorder는 6단계, OOO는 12단계로 고정. 각 단계를 지나는 PC, 명령어,
 * 물리/논리 레지스터 번호, opcode decode 신호(2^opcode_length) 등의 비트를
 * 더하고, 제어/인터럽트 레지스터를 50% 추가 가정한 뒤 사용자가 지정한
 * pipeline_stages로 재조정한다.
 */""", 0),

    # FunctionalUnit
    ("FunctionalUnit::FunctionalUnit(", """/*
 * [한국어]
 * FunctionalUnit::FunctionalUnit - FPU/ALU/MUL 기능 유닛 생성자
 *
 * @fu_type_: FPU, ALU, MUL 중 하나. GPU SM의 SP/DP/SFU/INT ALU 모델링에 사용.
 * @exClockRate: 실행 유닛 클록(Hz).
 * @return: (생성자) — area_t, leakage, gate_leakage, per_access_energy, base_energy 초기화.
 *
 * Embedded 프로세서와 일반(고성능) 프로세서의 면적/누설 계수를 구분한다.
 * FPU의 경우 2개 DP FPU가 1개 SP FPU로 결합되도록 num_fu/=2 처리.
 * AccelWattch GPU 모드에서는 SP/ALU/SFU 전력 상수(SP_BASE_POWER,
 * SFU_BASE_POWER)가 base_energy로 사용된다.
 */""", 0),
    ("void FunctionalUnit::computeEnergy(bool is_tdp)", """/*
 * [한국어]
 * FunctionalUnit::computeEnergy - TDP/런타임 FU 전력 계산
 *
 * @is_tdp: true이면 TDP(peak) 모드, false이면 런타임 모드.
 * @return: void — power/rt_power에 저장.
 *
 * TDP 모드에서는 FU 개수(num_fu)와 duty_cycle을, 런타임 모드에서는
 * XML->sys.core[ithCore].{fpu,ialu,mul}_accesses를 사용. GPU 모드에서
 * inactive lane 보정을 위해 FPU/SFU에 base_energy×(32-active_lanes)를
 * 추가한다.
 */""", 0),
    ("void FunctionalUnit::displayEnergy(uint32_t indent, int plevel, bool is_tdp)", """/*
 * [한국어]
 * FunctionalUnit::displayEnergy - FPU/ALU/MUL 전력·면적 출력
 *
 * fu_type에 따라 "Floating Point Units", "Integer ALUs", "Complex ALUs"
 * 헤더를 출력하고 면적, 피크 동적 전력, 누설, 게이트 누설, 런타임 전력을
 * 보여준다.
 */""", 0),
    ("void FunctionalUnit::leakage_feedback(double temperature)", """/*
 * [한국어]
 * FunctionalUnit::leakage_feedback - 온도 변화에 따른 FU 누설 재계산
 *
 * Embedded 모드와 유사한 면적 기준으로 FPU/ALU/MUL leakage/gate_leakage를
 * 다시 계산하고 power에 저장한다.
 */""", 0),

    # UndiffCore
    ("UndiffCore::UndiffCore(", """/*
 * [한국어]
 * UndiffCore::UndiffCore - 미분화 코어(undifferentiated core) 전력 모델
 *
 * 코어에서 ALU/FPU/레지스터 파일 등으로 명시 모델링되지 않는 나머지
 * 로직(front-end, OoO 스케줄링, 제어 로직 등)의 면적과 누설 전력을
 * 다항식/로그 피팅(Niagara, Merom, Penryn 등)으로 추정한다.
 * 존재하지 않는 코어(exist==false)면 즉시 리턴.
 */""", 0),
    ("void UndiffCore::displayEnergy(uint32_t indent, int plevel, bool is_tdp)", """/*
 * [한국어]
 * UndiffCore::displayEnergy - 미분화 코어 전력·면적 출력
 *
 * TDP와 런타임 모드 모두에서 area, peak dynamic, leakage, gate leakage를
 * 출력한다.
 */""", 0),

    # inst_decoder
    ("inst_decoder::inst_decoder(", """/*
 * [한국어]
 * inst_decoder::inst_decoder - 명령어 디코더 생성자
 *
 * RISC 디코더는 n-to-2^n 디코더로 근사하고, x86 CISC 디코더는 시퀀서가
 * 2회 통과(squencer_passes=2)하도록 모델링한다. opcode_length가 18비트를
 * 초과하면 세그먼트를 분할(num_decoder_segments)하고, Decoder/Predec 객체를
 * 생성하여 면적/전력을 계산한다.
 */""", 0),
    ("void inst_decoder::inst_decoder_delay_power()", """/*
 * [한국어]
 * inst_decoder::inst_decoder_delay_power - 최종/사전 디코더 전력 합산
 *
 * set_pppm()으로 Predec/Final Decoder의 접근 패턴(ppcm)을 조정한 뒤
 * power에 더한다. x86은 squencer_passes=2.
 */""", 0),
    ("void inst_decoder::leakage_feedback(double temperature)", """/*
 * [한국어]
 * inst_decoder::leakage_feedback - 온도 변화에 따른 디코더 누설 재계산
 *
 * Predec/Final Decoder의 leakage_feedback()을 호출한 뒤, TDP와 동일한
 * ppcm 가중치로 power를 재구성한다.
 */""", 0),
    ("inst_decoder::~inst_decoder()", """/*
 * [한국어]
 * inst_decoder::~inst_decoder - 디코더 객체 메모리 해제
 *
 * CACTI uca_org_t cleanup과 Decoder/Predec 동적 할당 객체를 삭제한다.
 */""", 0),
]

logic_replacements = [
    ('#include "logic.h"', '#include "logic.h"           // [한국어] logic.h 클래스 선언 (selection_logic, FunctionalUnit 등)', 0),
    ('#define SP_BASE_POWER 0', '#define SP_BASE_POWER 0  // [한국어] AccelWattch GPU SP base power off/placeholder', 0),
    ('#define SFU_BASE_POWER 0', '#define SFU_BASE_POWER 0  // [한국어] AccelWattch GPU SFU base power off/placeholder', 0),

    ('  selection_power();', '  selection_power();  // [한국어] 선택 논리(우선순위 인코더 + OR) 전력 계산', 0),
    ('  double sckRation = g_tp.sckt_co_eff;', '  double sckRation = g_tp.sckt_co_eff;  // [한국어] 소켓/IO 드라이버 오버헤드 계수', 0),

    ('  // TODO: the 0.8um process data is used.', '  // [한국어] 아래 트랜지스터 폭은 0.8um 공정 기준값을 현재 기술 노드(F_sz_um)로 비례 스케일링한 것이다.\n  // TODO: the 0.8um process data is used.', 0),
    ('  Ctotal = 0;', '  Ctotal = 0;  // [한국어] 누적 캐패시턴스 초기화', 0),
    ('  while (win_entries > 4) {', '  // [한국어] 발행 가능 슬롯 수가 4를 초과하면 4진 트리 우선순위 arbiter 추가\n  while (win_entries > 4) {', 0),
    ('  Ctotal += issue_width * num_arbiter * (Cor + Cpencode);', '  Ctotal += issue_width * num_arbiter * (Cor + Cpencode);  // [한국어] issue_width×arbiter 수×(OR+encoder cap)', 0),
    ('  power.readOp.dynamic =', '  power.readOp.dynamic =  // [한국어] 동적 에너지 = Ctotal × Vdd² × 2(왕복)', 0),
    ('  power.readOp.leakage =', '  power.readOp.leakage =  // [한국어] 누설 전력 = issue_width×arbiter 수 × (grant/enable/inverter NOR 누설 합) × Vdd', 0),

    ('  conflict_check_power();', '  conflict_check_power();  // [한국어] 비교기/충돌 검사 전력 계산', 0),
    ('  num_comparators =', '  // [한국어] 비교기 수 = 3×(decodeW² - decodeW) (source→dest 2배 + dest→dest 1배)\n  num_comparators =', 0),
    ('  Ctotal = num_comparators * compare_cap();', '  Ctotal = num_comparators * compare_cap();  // [한국어] 총 부하 = 비교기 수 × 단일 비교기 용량', 0),
    ('  power.readOp.dynamic =\n      Ctotal * /*CLOCKRATE*/ g_tp.peri_global.Vdd * g_tp.peri_global.Vdd /*AF*/;',
     '  // [한국어] 동적 에너지 = Ctotal × Vdd² (AF/주파수는 외부 적용)\n  power.readOp.dynamic =\n      Ctotal * /*CLOCKRATE*/ g_tp.peri_global.Vdd * g_tp.peri_global.Vdd /*AF*/;', 0),

    ('  if (coredynp.core_ty == Inorder)', '  // [한국어] Inorder/OOO 모두 opcode(16) + shared resource(8) + REG TAG(8) 비트 추가\n  if (coredynp.core_ty == Inorder)', 0),
    ('    compare_bits += 16 + 8 + 8;  // TODO: opcode bits + log(shared resources) +\n                                 // REG TAG BITS-->opcode comparator',
     '    // [한국어] 비교 비트에 opcode + shared resource + register tag 추가\n    compare_bits += 16 + 8 + 8;  // TODO: opcode bits + log(shared resources) +\n                                 // REG TAG BITS-->opcode comparator', 0),

    ('  clock_cap = 2 * gate_C(WdecNANDn + WdecNANDp, 0, is_dram);', '  clock_cap = 2 * gate_C(WdecNANDn + WdecNANDp, 0, is_dram);  // [한국어] 클록이 한 NAND2의 두 입력에 연결됨', 0),
    ('  e_switch.readOp.dynamic += (c4 + c1 + c2 + c3 + c5 + c6 + 2 * cell_load) *',
     '  // [한국어] 스위칭 에너지 = 6개 남 총합 × 0.5 × Vdd² + 2×외부 부하\n  e_switch.readOp.dynamic += (c4 + c1 + c2 + c3 + c5 + c6 + 2 * cell_load) *', 0),

    ('  if (!coredynp.Embedded)', '  // [한국어] Embedded가 아니면 고성능 트랜지스터 폭 사용\n  if (!coredynp.Embedded)', 0),
    ('  compute();', '  compute();  // [한국어] 파이프라인 레지스터 수 및 전력 최종 계산', 0),
    ('  double pipe_reg_power =', '  // [한국어] 파이프라인 레지스터 전력 = num_piperegs×(switch+keep0+keep1)/3 + 클록\n  double pipe_reg_power =', 0),
    ('  area.set_area(num_piperegs * pipe_reg.area.get_area());', '  area.set_area(num_piperegs * pipe_reg.area.get_area());  // [한국어] DFF 면적 × 레지스터 수', 0),
    ('  if (!is_core_pipeline) {', '  // [한국어] 비코어 파이프라인: 사용자가 지정한 stages × per_stage_vector\n  if (!is_core_pipeline) {', 0),
    ('    if (coredynp.core_ty == Inorder) {', '    // [한국어] Inorder 코어: 6단계 파이프라인 (IF→ID→ThreadSEL→EXE→MEM→WB)\n    if (coredynp.core_ty == Inorder) {', 0),
    ('    } else {\n      /* assume 12 stage pipe stages', '    } else {\n      // [한국어] OOO 코어: 12단계 파이프라인 (Fetch→Decode→Rename→IssueQ→Dispatch→RegRead→EXE→MEM→WB→CM)\n      /* assume 12 stage pipe stages', 0),
    ('    num_piperegs = num_piperegs * 1.5;', '    num_piperegs = num_piperegs * 1.5;  // [한국어] 제어/인터럽트 레지스터 50% 추가', 0),

    ('  if (XML->sys.Embedded) {', '  // [한국어] Embedded CPU 모드와 고성능 CPU/GPU 모드 분기\n  if (XML->sys.Embedded) {', 0),
    ('    } else if (fu_type == ALU) {', '    // [한국어] 정수 ALU: 71.85×71.85 um² 기준 × num_fu × logic scaling\n    } else if (fu_type == ALU) {', 0),
    ('    } else if (fu_type == MUL) {', '    // [한국어] 곱셈/나눗셈 유닛: divider/mul Sun 데이터 기반\n    } else if (fu_type == MUL) {', 0),
    ('  area.set_area(area_t * num_fu);', '  area.set_area(area_t * num_fu);  // [한국어] 단위 FU 면적 × FU 개수', 0),
    ('  leakage *= num_fu;', '  leakage *= num_fu;  // [한국어] 단위 누설 × FU 개수', 0),

    ('    set_pppm(pppm_t, 2, 2, 2, 2);', '    set_pppm(pppm_t, 2, 2, 2, 2);  // [한국어] 정수 명령당 2개 소스 오퍼랜드 전달 가정', 0),
    ('    FU_duty_cycle = coredynp.FPU_duty_cycle;', '    FU_duty_cycle = coredynp.FPU_duty_cycle;  // [한국어] FPU 활성률', 0),
    ('    power.readOp.dynamic =\n        per_access_energy * stats_t.readAc.access + base_energy / clockRate;',
     '    // [한국어] 피크 동적 전력 = per_access_energy×접근수 + base_energy/clockRate\n    power.readOp.dynamic =\n        per_access_energy * stats_t.readAc.access + base_energy / clockRate;', 0),
    ('      stats_t.readAc.access = XML->sys.core[ithCore].fpu_accesses;', '      // [한국어] GPU FPU 실제 접근 수 → inactive lane 보정에 사용\n      stats_t.readAc.access = XML->sys.core[ithCore].fpu_accesses;', 0),
    ('      stats_t.readAc.access = XML->sys.core[ithCore].ialu_accesses;', '      // [한국어] GPU/CPU 정수 ALU 실제 접근 수\n      stats_t.readAc.access = XML->sys.core[ithCore].ialu_accesses;', 0),
    ('      stats_t.readAc.access = XML->sys.core[ithCore].mul_accesses;', '      // [한국어] GPU/CPU 곱셈 유닛 실제 접근 수\n      stats_t.readAc.access = XML->sys.core[ithCore].mul_accesses;', 0),
    ('    if (fu_type == ALU) {\n      rt_power.readOp.dynamic = per_access_energy * stats_t.readAc.access +\n                                base_energy * executionTime;',
     '    // [한국어] ALU 런타임 전력: per_access_energy×access + base_energy×실행시간\n    if (fu_type == ALU) {\n      rt_power.readOp.dynamic = per_access_energy * stats_t.readAc.access +\n                                base_energy * executionTime;', 0),
    ('      rt_power.readOp.dynamic +=\n          base_energy * executionTime *\n          (32 - XML->sys.core[ithCore].sp_average_active_lanes);',
     '      // [한국어] FPU inactive lane 보정: 32개 lane 중 비활성 lane만큼 base_energy 추가\n      rt_power.readOp.dynamic +=\n          base_energy * executionTime *\n          (32 - XML->sys.core[ithCore].sp_average_active_lanes);', 0),
    ('      if (XML->sys.core[ithCore].sfu_average_active_lanes >= 1)', '      // [한국어] SFU inactive lane 보정\n      if (XML->sys.core[ithCore].sfu_average_active_lanes >= 1)', 0),

    ('  if (!exist) return;', '  if (!exist) return;  // [한국어] 해당 코어가 비활성이면 전력 계산 생략', 0),
    ('  if (embedded == false) {', '  // [한국어] 고성능 코어: 다이 측정 기반 피팅 공식\n  if (embedded == false) {', 0),
    ('    if (core_ty == OOO) {', '    // [한국어] OOO 코어: pipeline_stage에 대한 로그 피팅\n    if (core_ty == OOO) {', 0),
    ('    } else if (core_ty == Inorder) {', '    // [한국어] Inorder 코어: pipeline_stage에 대한 로그 피팅\n    } else if (core_ty == Inorder) {', 0),
    ('    undifferentiated_core *= (1 + logtwo(num_hthreads) * 0.0716);', '    undifferentiated_core *= (1 + logtwo(num_hthreads) * 0.0716);  // [한국어] 하드웨어 스레드 수에 따른 면적 증가', 0),
    ('  undifferentiated_core *= g_tp.scaling_factor.logic_scaling_co_eff *\n                           1e6;',
     '  // [한국어] mm² → um² 변환 및 논리 면적 스케일링\n  undifferentiated_core *= g_tp.scaling_factor.logic_scaling_co_eff *\n                           1e6;', 0),
    ('  power.readOp.leakage =\n      undifferentiated_core *', '  // [한국어] 미분화 코어 누설 = 면적 × 트랜지스터 밀도 × 단위 누설 × Vdd\n  power.readOp.leakage =\n      undifferentiated_core *', 0),
    ('  scktRatio = g_tp.sckt_co_eff;', '  scktRatio = g_tp.sckt_co_eff;  // [한국어] 소켓 오버헤드', 0),

    ('  num_decoder_segments = (int)ceil(opcode_length / 18.0);', '  num_decoder_segments = (int)ceil(opcode_length / 18.0);  // [한국어] 18비트 이상이면 디코더 세그먼트 분할', 0),
    ('  num_decoded_signals = (int)pow(2.0, opcode_length);', '  num_decoded_signals = (int)pow(2.0, opcode_length);  // [한국어] 디코딩된 신호 수 = 2^opcode_length', 0),
    ('  C_driver_load =\n      1024 * gate_C(load_nmos_width + load_pmos_width, 0,', '  // [한국어] 디코더 구동 부하: 1024개 게이트로 가정 (TODO: 재검토 필요)\n  C_driver_load =\n      1024 * gate_C(load_nmos_width + load_pmos_width, 0,', 0),
    ('  R_wire_load = 3000 * l_ip.F_sz_um * g_tp.wire_outside_mat.R_per_um;', '  R_wire_load = 3000 * l_ip.F_sz_um * g_tp.wire_outside_mat.R_per_um;  // [한국어] 외부 와이어 저항', 0),
    ('  double area_decoder = final_dec->area.get_area() * num_decoded_signals *\n                        num_decoder_segments * num_decoders;',
     '  // [한국어] 최종 디코더 면적 = 단위 면적 × 신호 수 × 세그먼트 × 디코더 수\n  double area_decoder = final_dec->area.get_area() * num_decoded_signals *\n                        num_decoder_segments * num_decoders;', 0),
    ('  inst_decoder_delay_power();', '  inst_decoder_delay_power();  // [한국어] 디코더 동적/누설 전력 산출', 0),
]

apply(ROOT / "logic.cc", logic_insertions, logic_replacements)
