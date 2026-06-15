#!/usr/bin/env python3
"""
[한국어] accelwattch/sharedcache.cc 인라인 주석 추가 (기본 주석 삽입 후 실행)
"""
from pathlib import Path


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


def apply_replacements(path: Path, replacements):
    content = path.read_text()
    for old, new, occ in replacements:
        content = replace_first(content, old, new, occ)
    path.write_text(content)
    print(f"annotated {path}")


ROOT = Path("/home/harison/company/gpgpu-sim-study/src/accelwattch")

replacements = [
    # ---- SharedCache 생성자 ----
    ("  double size, line, assoc, banks;\n  if (cacheL == L2 && XML->sys.Private_L2) {",
     "  double size, line, assoc, banks;\n  // [한국어] Private L2일 때는 코어 장치/유형으로 모델링\n  if (cacheL == L2 && XML->sys.Private_L2) {", 0),
    ("  } else {\n    device_t = LLC_device;\n    core_t = Inorder;\n  }",
     "  // [한국어] 공유 LLC(L3) 또는 Directory는 LLC 장치, Inorder 코어 가정\n  } else {\n    device_t = LLC_device;\n    core_t = Inorder;\n  }", 0),
    ("  if (XML->sys.Embedded) {",
     "  // [한국어] 임베디드/GPU 환경: 저전력 글로벌 와이어 모델 사용\n  if (XML->sys.Embedded) {", 0),
    ("  set_cache_param();",
     "  // [한국어] XML에서 L2/L3/Directory 파라미터를 cachep로 로드\n  set_cache_param();", 0),
    ("  // All lower level cache are physically indexed and tagged.\n  size = cachep.capacity;",
     "  // All lower level cache are physically indexed and tagged.\n  // [한국어] CACTI용 캐시 용량/라인/연상도/뱅크 수를 cachep에서 복사\n  size = cachep.capacity;", 0),
    ("  if ((cachep.dir_ty == ST && cacheL == L1Directory) ||",
     "  // [한국어] ST(포인터+태그) Directory: CAM 구조로 모델링(assoc=0, search port=1)\n  if ((cachep.dir_ty == ST && cacheL == L1Directory) ||", 0),
    ("  } else {\n    idx = debug ? 9 : int(ceil(log2(size / line / assoc)));",
     "  } else {\n    // [한국어] 일반 캐시: 인덱스 비트 = log2(용량/라인/연상도)\n    idx = debug ? 9 : int(ceil(log2(size / line / assoc)));", 0),
    ("    interface_ip.num_search_ports = 0;\n    if (cachep.dir_ty == SBT) {",
     "    interface_ip.num_search_ports = 0;\n    // [한국어] SBT(흩어진 디렉토리 비트)는 데이터 블록에 디렉토리 비트를 병합\n    if (cachep.dir_ty == SBT) {", 0),
    ("  interface_ip.specific_tag = 1;",
     "  // [한국어] CACTI에 태그 폭을 명시적으로 지정\n  interface_ip.specific_tag = 1;", 0),
    ("  unicache.caches =\n      new ArrayST(&interface_ip, cachep.name + \"cache\", device_t, true, core_t);",
     "  // [한국어] 주 캐시 SRAM 배열(ArrayST) 생성: 태그+데이터 또는 Directory\n  unicache.caches =\n      new ArrayST(&interface_ip, cachep.name + \"cache\", device_t, true, core_t);", 0),
    ("  unicache.area.set_area(unicache.area.get_area() +\n                         unicache.caches->local_result.area);",
     "  // [한국어] unicache 총면적에 주 캐시 면적 누적\n  unicache.area.set_area(unicache.area.get_area() +\n                         unicache.caches->local_result.area);", 0),
    ("  area.set_area(area.get_area() + unicache.caches->local_result.area);",
     "  // [한국어] SharedCache Component 전체 면적에 주 캐시 면적 추가\n  area.set_area(area.get_area() + unicache.caches->local_result.area);", 0),
    ("  if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||\n        (cachep.dir_ty == ST && cacheL == L2Directory))) {",
     "  // [한국어] 일반 캐시/Directory가 아닌 경우에만 missb/ifb/prefetchb/wbb 생성\n  if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||\n        (cachep.dir_ty == ST && cacheL == L2Directory))) {", 0),
    ("    unicache.missb = new ArrayST(&interface_ip, cachep.name + \"MissB\", device_t,\n                                 true, core_t);",
     "    // [한국어] Miss Buffer(MSHR): 태그+데이터 상태를 CAM+RAM으로 모델링\n    unicache.missb = new ArrayST(&interface_ip, cachep.name + \"MissB\", device_t,\n                                 true, core_t);", 0),
    ("    unicache.ifb = new ArrayST(&interface_ip, cachep.name + \"FillB\", device_t,\n                               true, core_t);",
     "    // [한국어] Fill Buffer: 하위 메모리에서 읽어온 라인을 임시 저장\n    unicache.ifb = new ArrayST(&interface_ip, cachep.name + \"FillB\", device_t,\n                               true, core_t);", 0),
    ("    unicache.prefetchb = new ArrayST(&interface_ip, cachep.name + \"PrefetchB\",\n                                     device_t, true, core_t);",
     "    // [한국어] Prefetch Buffer: 미리 가져온 캐시 라인 큐\n    unicache.prefetchb = new ArrayST(&interface_ip, cachep.name + \"PrefetchB\",\n                                     device_t, true, core_t);", 0),
    ("    unicache.wbb =\n        new ArrayST(&interface_ip, cachep.name + \"WBB\", device_t, true, core_t);",
     "    // [한국어] Write Back Buffer: 더티 라인을 하위 계층에 쓰기 전 임시 보관\n    unicache.wbb =\n        new ArrayST(&interface_ip, cachep.name + \"WBB\", device_t, true, core_t);", 0),
    ("  interface_ip.force_cache_config = false;",
     "  // [한국어] 이후 보조 버퍼들은 CACTI가 자동 최적화하도록 강제 설정 해제\n  interface_ip.force_cache_config = false;", 0),

    # ---- computeEnergy ----
    ("  double homenode_data_access = (cachep.dir_ty == SBT) ? 0.9 : 1.0;",
     "  // [한국어] SBT 디렉토리 병합 캐시: 데이터 접근의 90%가 로컬, 10%가 홈 노드\n  double homenode_data_access = (cachep.dir_ty == SBT) ? 0.9 : 1.0;", 0),
    ("  if (is_tdp) {",
     "  // [한국어] TDP 모드: Peak 부하 가정(읽기 67%, 쓰기 33%)으로 접근 수 추정\n  if (is_tdp) {", 0),
    ("      // init stats for Peak\n      unicache.caches->stats_t.readAc.access =",
     "      // init stats for Peak\n      // [한국어] TDP 읽기 접근 = 0.67 * rw_port * duty_cycle * homenode_data_factor\n      unicache.caches->stats_t.readAc.access =", 0),
    ("      unicache.caches->stats_t.writeAc.access =",
     "      // [한국어] TDP 쓰기 접근 = 0.33 * rw_port * duty_cycle * homenode_data_factor\n      unicache.caches->stats_t.writeAc.access =", 0),
    ("      unicache.caches->tdp_stats = unicache.caches->stats_t;",
     "      // [한국어] TDP 통계 복사본 저장\n      unicache.caches->tdp_stats = unicache.caches->stats_t;", 0),
    ("      if (cachep.dir_ty == SBT) {",
     "      // [한국어] SBT 캐시: 나머지 10% 디렉토리 홈 노드 접근에 대한 TDP 통계\n      if (cachep.dir_ty == SBT) {", 0),
    ("      unicache.missb->stats_t.readAc.access =\n          unicache.missb->l_ip.num_search_ports;",
     "      // [한국어] Miss Buffer: search port당 한 번의 CAM 검색과 한 번의 RAM 쓰기\n      unicache.missb->stats_t.readAc.access =\n          unicache.missb->l_ip.num_search_ports;", 0),
    ("      unicache.ifb->stats_t.readAc.access = unicache.ifb->l_ip.num_search_ports;",
     "      // [한국어] Fill Buffer: search port 기반 읽기/쓰기 접근\n      unicache.ifb->stats_t.readAc.access = unicache.ifb->l_ip.num_search_ports;", 0),
    ("      unicache.prefetchb->stats_t.readAc.access =",
     "      // [한국어] Prefetch Buffer: search port 기반 접근\n      unicache.prefetchb->stats_t.readAc.access =", 0),
    ("      unicache.wbb->stats_t.readAc.access = unicache.wbb->l_ip.num_search_ports;",
     "      // [한국어] Write Back Buffer: search port 기반 접근\n      unicache.wbb->stats_t.readAc.access = unicache.wbb->l_ip.num_search_ports;", 0),
    ("    } else {\n      unicache.caches->stats_t.readAc.access =\n          unicache.caches->l_ip.num_search_ports * cachep.duty_cycle;",
     "    } else {\n      // [한국어] Directory(ST) TDP: search port와 duty cycle로 접근 추정\n      unicache.caches->stats_t.readAc.access =\n          unicache.caches->l_ip.num_search_ports * cachep.duty_cycle;", 0),
    ("  } else {\n    // init stats for runtime power (RTP)",
     "  } else {\n  // [한국어] 런타임 모드: GPGPU-Sim/AccelWattch가 채운 실제 접근 통계 사용\n    // init stats for runtime power (RTP)", 0),
    ("      // Copy stats from l1 to L1[0]\n      XML->sys.L2[ithCache].total_accesses = XML->sys.l2.total_accesses;",
     "      // Copy stats from l1 to L1[0]\n      // [한국어] GPU 통합 L2 통계를 XML->sys.L2[0]로 복사(Processor가 읽음)\n      XML->sys.L2[ithCache].total_accesses = XML->sys.l2.total_accesses;", 0),
    ("    } else if (cacheL == L3) {",
     "    // [한국어] CPU/GPU L3 캐시: sys.L3[] 통계를 직접 사용\n    } else if (cacheL == L3) {", 0),
    ("    } else if (cacheL == L1Directory) {",
     "    // [한국어] L1 Directory: sys.L1Directory[] 통계 사용\n    } else if (cacheL == L1Directory) {", 0),
    ("    } else if (cacheL == L2Directory) {  // cout<<\"L2 directory\"<<endl;",
     "    // [한국어] L2 Directory: sys.L2Directory[] 통계 사용\n    } else if (cacheL == L2Directory) {  // cout<<\"L2 directory\"<<endl;", 0),
    ("      unicache.missb->stats_t.readAc.access =\n          unicache.caches->stats_t.writeAc.miss;",
     "      // [한국어] W-back/W-allocate 가정: write miss 수만큼 missb/ifb/wbb 접근\n      unicache.missb->stats_t.readAc.access =\n          unicache.caches->stats_t.writeAc.miss;", 0),
    ("      if (cachep.dir_ty == SBT) {",
     "      // [한국어] SBT 홈 노드 write miss도 버퍼 접근에 추가\n      if (cachep.dir_ty == SBT) {", 1),
    ("  unicache.power_t.reset();",
     "  // [한국어] 동적 에너지 누적값 초기화\n  unicache.power_t.reset();", 0),
    ("  unicache.rt_power.reset();",
     "  // [한국어] 런타임 전력 누적값 초기화\n  unicache.rt_power.reset();", 0),
    ("    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||\n          (cachep.dir_ty == ST &&\n           cacheL ==\n               L2Directory))) {  // Assuming write back and write-allocate cache",
     "    // [한국어] 일반 캐시/Directory가 아닐 때만 missb/ifb/prefetchb/wbb 에너지 포함\n    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||\n          (cachep.dir_ty == ST &&\n           cacheL ==\n               L2Directory))) {  // Assuming write back and write-allocate cache", 0),
    ("    unicache.power_t.readOp.dynamic +=\n        (unicache.caches->stats_t.readAc.hit *",
     "    // [한국어] 주 캐시 동적 에너지: read hit + read miss(tag) + write miss(tag) + write access\n    unicache.power_t.readOp.dynamic +=\n        (unicache.caches->stats_t.readAc.hit *", 0),
    ("    if (cachep.dir_ty == SBT) {",
     "    // [한국어] SBT 홈 노드 접근에 대한 추가 동적 에너지(dir_overhead 반영)\n    if (cachep.dir_ty == SBT) {", 1),
    ("    unicache.power_t.readOp.dynamic +=\n        unicache.missb->stats_t.readAc.access *",
     "    // [한국어] Miss Buffer 동적 에너지: searchOp(전체 CAM 검색) + writeOp(상태 갱신)\n    unicache.power_t.readOp.dynamic +=\n        unicache.missb->stats_t.readAc.access *", 0),
    ("    unicache.power_t.readOp.dynamic +=\n        unicache.ifb->stats_t.readAc.access *",
     "    // [한국어] Fill Buffer 동적 에너지\n    unicache.power_t.readOp.dynamic +=\n        unicache.ifb->stats_t.readAc.access *", 0),
    ("    unicache.power_t.readOp.dynamic +=\n        unicache.prefetchb->stats_t.readAc.access *",
     "    // [한국어] Prefetch Buffer 동적 에너지\n    unicache.power_t.readOp.dynamic +=\n        unicache.prefetchb->stats_t.readAc.access *", 0),
    ("    unicache.power_t.readOp.dynamic +=\n        unicache.wbb->stats_t.readAc.access *",
     "    // [한국어] Write Back Buffer 동적 에너지\n    unicache.power_t.readOp.dynamic +=\n        unicache.wbb->stats_t.readAc.access *", 0),
    ("  } else {\n    unicache.power_t.readOp.dynamic +=\n        (unicache.caches->stats_t.readAc.access *",
     "  } else {\n  // [한국어] Directory(ST) 구조: searchOp + writeOp 기반 동적 에너지\n    unicache.power_t.readOp.dynamic +=\n        (unicache.caches->stats_t.readAc.access *", 0),
    ("  if (is_tdp) {",
     "  // [한국어] TDP: 동적 에너지 + 주 캐시 누설 전력을 power에 합산\n  if (is_tdp) {", 1),
    ("    unicache.power =\n        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;",
     "    // [한국어] power_t(TDP 동적) + caches 누설(pppm_lkg 마스크 적용)\n    unicache.power =\n        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;", 0),
    ("      unicache.power =\n          unicache.power + (unicache.missb->local_result.power +",
     "      // [한국어] 보조 버퍼들의 누설 전력도 추가\n      unicache.power =\n          unicache.power + (unicache.missb->local_result.power +", 0),
    ("    power = power + unicache.power;",
     "    // [한국어] SharedCache의 TDP 전력을 Processor 전체 power에 누적\n    power = power + unicache.power;", 0),
    ("  } else {\n    unicache.rt_power =\n        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;",
     "  } else {\n  // [한국어] 런타임: 동적 에너지 + 주 캐시 누설 전력을 rt_power에 합산\n    unicache.rt_power =\n        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;", 0),
    ("      unicache.rt_power =\n          unicache.rt_power + (unicache.missb->local_result.power +",
     "      // [한국어] 보조 버퍼들의 누설 전력도 추가\n      unicache.rt_power =\n          unicache.rt_power + (unicache.missb->local_result.power +", 0),
    ("    rt_power = rt_power + unicache.rt_power;",
     "    // [한국어] SharedCache의 런타임 전력을 Processor 전체 rt_power에 누적\n    rt_power = rt_power + unicache.rt_power;", 0),

    # ---- displayEnergy ----
    ("  bool long_channel = XML->sys.longer_channel_device;",
     "  // [한국어] 장채널 소자 사용 여부에 따라 누설 전력 출력 분기\n  bool long_channel = XML->sys.longer_channel_device;", 0),
    ("  if (is_tdp) {",
     "  // [한국어] TDP 모드 결과 출력(런타임 모드 출력은 현재 비어 있음)\n  if (is_tdp) {", 1),
    ("    cout << (XML->sys.Private_L2 ? indent_str : \"\") << cachep.name << endl;",
     "    // [한국어] Private L2가 아니면 상위 Processor에서 이미 인덴트를 출력\n    cout << (XML->sys.Private_L2 ? indent_str : \"\") << cachep.name << endl;", 0),
    ("         << \"Peak Dynamic = \" << power.readOp.dynamic * cachep.clockRate << \" W\"",
     "    // [한국어] Peak 동적 전력 = 동적 에너지 × 클럭주파수\n         << \"Peak Dynamic = \" << power.readOp.dynamic * cachep.clockRate << \" W\"", 0),
    ("    cout << indent_str << \"Runtime Dynamic = \"",
     "    // [한국어] 런타임 동적 전력 = 런타임 에너지 / 실행 시간\n    cout << indent_str << \"Runtime Dynamic = \"", 0),

    # ---- set_cache_param ----
    ("  if (cacheL == L2) {",
     "  // [한국어] GPU 통합 L2 파라미터 로드\n  if (cacheL == L2) {", 1),
    ("    cachep.clockRate *= 1e6;",
     "    // [한국어] MHz → Hz 변환\n    cachep.clockRate *= 1e6;", 0),
    ("    cachep.executionTime =\n        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);",
     "    // [한국어] 전체 실행 시간(초) = 사이클 수 / 목표 코어 클럭\n    cachep.executionTime =\n        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);", 0),
    ("    if (!XML->sys.L2[ithCache].merged_dir) {",
     "    // [한국어] merged_dir=false면 일반 캐시(NonDir)\n    if (!XML->sys.L2[ithCache].merged_dir) {", 0),
    ("    } else {\n      cachep.dir_ty = SBT;",
     "    } else {\n      // [한국어] merged_dir=true면 SBT(디렉토리 비트 병합) 캐시\n      cachep.dir_ty = SBT;", 0),
    ("  } else if (cacheL == L3) {",
     "  // [한국어] L3(Last-Level Cache) 파라미터 로드\n  } else if (cacheL == L3) {", 0),
    ("  } else if (cacheL == L1Directory) {",
     "  // [한국어] L1 Directory 파라미터 로드\n  } else if (cacheL == L1Directory) {", 0),
    ("  } else if (cacheL == L2Directory) {",
     "  // [한국어] L2 Directory 파라미터 로드\n  } else if (cacheL == L2Directory) {", 0),
]

apply_replacements(ROOT / "sharedcache.cc", replacements)
