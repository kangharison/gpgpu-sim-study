# CLAUDE.md — gpgpu-sim-study 프로젝트 지침

> 주석 작성 방법론(파일/함수/인라인/구조체 포맷, 핵심 원칙, 공통 작업 원칙, 커밋 규칙)은
> 상위 디렉토리의 [`../CLAUDE.md`](../CLAUDE.md)에서 공통으로 관리한다.
> 이 파일은 **GPGPU-Sim 도메인에 특화된 지식과 작업 현황**만 정의한다.

## 프로젝트 개요

이 저장소는 [GPGPU-Sim 4.0](https://github.com/accel-sim/gpgpu-sim_distribution) 소스 코드를 분석하고 학습하기 위한 스터디 프로젝트이다. GPGPU-Sim은 CUDA/OpenCL GPU 워크로드를 사이클-레벨로 시뮬레이션하는 컴퓨터 구조 연구용 시뮬레이터이다. AccelWattch(전력 모델)와 AerialVision(성능 가시화 도구)을 포함하며, Accel-Sim 프레임워크와 호환되어 NVBit SASS 트레이스 기반 시뮬레이션도 지원한다.

- **원본**: GPGPU-Sim 4.0 (UBC Computer Architecture Lab / Accel-Sim)
- **참고 논문**: ISCA 2020 — "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling"
- **브랜치**: master

## GPGPU-Sim 핵심 아키텍처

### 실행 흐름 (execution-driven)

```
CUDA Application
  → libcuda (libcuda/) / libopencl — 런타임 인터셉트
      → gpgpusim_entrypoint.cc (시뮬레이터 진입)
          → cuda-sim/ (PTX/SASS 함수 시뮬레이션: fetch/decode/execute)
          → gpgpu-sim/ (마이크로아키텍처 타이밍 모델)
              ├── shader.cc (SM: SIMT stack, warp scheduler, exec units)
              ├── gpu-cache.cc (L1/L2/텍스처/상수 캐시)
              ├── dram.cc (DRAM 타이밍 모델)
              ├── mem_fetch.cc (메모리 요청 패킷)
              └── intersim2/ (NoC 라우팅 시뮬레이션)
          → accelwattch/ (전력 모델, McPAT 기반)
          → 결과 리포트 출력 (stats)
```

### 시뮬레이션 계층 구조

```
┌──────────────────────────────────────────────┐
│          CUDA/OpenCL 애플리케이션               │
├──────────────────────────────────────────────┤
│      libcuda / libopencl 인터셉트 레이어        │
│   (cuLaunchKernel → 시뮬레이터로 리다이렉트)     │
├──────────────────────────────────────────────┤
│   기능 시뮬레이션 (cuda-sim/)                   │
│   PTX/PTXPlus/SASS 인스트럭션 실행 모델           │
├──────────────────────────────────────────────┤
│   타이밍 모델 (gpgpu-sim/)                      │
│   SM, 워프 스케줄러, 캐시, DRAM, NoC 사이클 모델   │
├──────────────────────────────────────────────┤
│   전력 모델 (accelwattch/)                      │
│   McPAT 기반 동적/정적 전력 추정                  │
└──────────────────────────────────────────────┘
```

### 주요 모듈

```
src/
├── abstract_hardware_model.{cc,h}  - warp/SIMT 스택/스레드 블록 등 상위 추상
├── cuda-sim/                       - 기능 시뮬레이션 (PTX 파싱/실행)
│   ├── ptx.l, ptx.y               - PTX Lex/Yacc 파서
│   ├── ptx_ir.{cc,h}              - PTX IR
│   ├── instructions.cc            - PTX 명령어 시맨틱
│   └── cuda-math.h                - CUDA 수학 함수 에뮬레이션
├── gpgpu-sim/                      - 타이밍 시뮬레이션
│   ├── shader.{cc,h}              - SM 파이프라인, 워프 스케줄러
│   ├── gpu-sim.{cc,h}             - 최상위 시뮬레이션 루프 (cycle-by-cycle)
│   ├── gpu-cache.{cc,h}           - 캐시 계층
│   ├── dram.{cc,h}                - DRAM 컨트롤러
│   ├── mem_fetch.{cc,h}           - 메모리 요청 객체
│   ├── scoreboard.{cc,h}          - RAW 해저드 감지
│   └── addrdec.{cc,h}             - 메모리 주소 디코딩
├── intersim2/                      - NoC(네트워크온칩) 시뮬레이터
├── accelwattch/                    - 전력 모델 (McPAT 기반)
├── gpgpusim_entrypoint.{cc,h}      - 시뮬레이터 진입점
├── stream_manager.{cc,h}           - CUDA 스트림 관리
└── option_parser.{cc,h}            - 설정 파일 파싱
```

## GPGPU-Sim 특화 주석 요구사항

GPGPU-Sim 코드의 특성상, 주석에 다음 사항을 반드시 포함한다:

1. **GPU 마이크로아키텍처 용어**: SM(Streaming Multiprocessor), warp, SIMT stack, scoreboard, operand collector 등 CUDA HW 용어의 역할을 시뮬레이터 관점에서 설명
2. **사이클-레벨 모델링**: 각 함수가 1 사이클 단위로 불리는지, 여러 사이클에 걸쳐 상태를 유지하는지 명시
3. **PTX vs SASS vs PTXPlus**: 함수가 처리하는 명령어 레이어가 무엇인지 명시
4. **기능/타이밍 모델 분리**: 이 파일이 기능 시뮬레이션(cuda-sim)인지 타이밍 시뮬레이션(gpgpu-sim)인지 명시
5. **워프 수준 동작**: warp divergence, SIMT 재합류(reconvergence), `__syncthreads` 구현 등 워프 제어 메커니즘 설명
6. **메모리 계층 경로**: L1/L2/DRAM 경로와 mem_fetch 전달 흐름, miss/hit 분기 설명
7. **NoC 라우팅**: intersim2 경유 시 라우팅/플릿/가상 채널 개념 부연
8. **설정 파일 연동**: `gpgpusim.config`의 어떤 옵션이 이 코드의 동작을 바꾸는지 명시
9. **AccelWattch 연동**: 전력 계산을 위한 counter 업데이트 지점 설명

## 디테일 수준 가이드 (GPGPU-Sim 구체화)

| 파일 유형 | 상단 블록 | 함수 주석 | 인라인 주석 | 구조체 필드 |
|----------|----------|----------|------------|-----------|
| 핵심 타이밍 모델 (gpgpu-sim/shader, gpu-sim, gpu-cache, dram) | 매우 상세 | 모든 함수 | **모든 라인** | 모든 필드 |
| 기능 시뮬레이션 (cuda-sim/ptx_ir, instructions, cuda-math) | 매우 상세 | 모든 함수 | **모든 라인** | 모든 필드 |
| 추상 모델 (abstract_hardware_model.*) | 매우 상세 | 모든 함수 | **모든 라인** | 모든 필드 |
| NoC (intersim2/) | 상세 | 모든 함수 | **모든 라인** | 모든 필드 |
| 전력 모델 (accelwattch/) | 상세 | 모든 함수 | **모든 라인** | 모든 필드 |
| 진입점/옵션/스트림 (entrypoint, stream_manager, option_parser) | 상세 | 모든 함수 | **모든 라인** | 주요 구조체 |
| 런타임 인터셉트 (libcuda/, libopencl/) | 상세 | 모든 함수 | **모든 라인** | - |
| 파서 생성 파일 (ptx.l, ptx.y) | 간결 | 주요 규칙만 | 문법 핵심 부분 | - |
| 디버그/트레이스 (debug.*, trace.*) | 상세 | 모든 함수 | **모든 라인** | - |

## 주요 용어 사전

- **SM (Streaming Multiprocessor)**: NVIDIA GPU의 코어 유닛, 여러 warp를 동시에 실행
- **warp**: 32개 스레드의 SIMT 실행 단위
- **SIMT stack**: warp 내 분기 처리를 위한 active mask + PC 스택
- **scoreboard**: 레지스터 의존성 추적 구조 (RAW 해저드 방지)
- **operand collector**: 레지스터 파일 뱅크 충돌을 처리하는 교환기
- **mem_fetch**: 시뮬레이터 내부 메모리 요청 패킷
- **shader core**: GPGPU-Sim에서 SM을 부르는 이름
- **PTX / PTXPlus / SASS**: NVIDIA GPU의 중간 표현 계층 (PTX는 가상 ISA, SASS는 실제 ISA)
- **execution-driven**: 실제 CUDA 바이너리를 런타임에 가로채어 실행하며 시뮬
- **trace-driven**: 사전에 캡처한 SASS 트레이스(NVBit 등)를 입력으로 실행
- **ICNT (intersim2)**: Booksim 기반 GPU 내부 NoC 시뮬레이터
- **cycle**: 시뮬레이터의 기본 시간 단위 (코어 클럭 1 사이클)

## 작업 제외 대상

- `accelwattch/mcpat/` 등 외부 도입 McPAT 코드 — 핵심 연동부만 주석, 내부는 제외
- `cuobjdump_to_ptxplus/` 자동 변환 툴 — 14개 소스/헤더 파일 전체 주석 완료
- `bin/`, `build/` 자동 생성/빌드 산출물
- `doc/doxygen/` 등 자동 생성 문서
- `.l`, `.y` 파일 자체의 생성된 C 출력물

## 주석 작업 진행 현황

### 완료 (218 파일/범위)

| 파일 | 설명 | 완료일 |
|------|------|--------|
| src/gpgpu-sim/gpu-cache.h | 캐시 계층 헤더 전체 (cache_t, baseline_cache, read_only_cache, data_cache, l1_cache, l2_cache, tex_cache 및 내부 FIFO/ROB/fragment 구조체, 필드/메서드 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/gpu-cache.cc | 캐시 계층 구현 전체 (cache_config, tag_array, mshr_table, baseline_cache cycle/fill, data_cache 쓰기/읽기 정책 10종, read_only_cache, l1_cache, l2_cache, tex_cache cycle/fill/display_state 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/gpgpu-sim/l2cache.h | L2 캐시/메모리 파티션 헤더 전체 (partition_mf_allocator, memory_partition_unit, memory_sub_partition, arbitration_metadata, L2interface 클래스 및 모든 필드/메서드 주석) | 2026-06-14 |
| src/gpgpu-sim/l2cache.cc | L2 캐시/메모리 파티션 구현 전체 (alloc 생성자, dram_cycle/simple_dram_model_cycle, cache_cycle, push/pop/top, sector 분해, ROP 지연 큐, 크레딧 중재, flush/invalidate 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/gpgpu-sim/mem_latency_stat.h | 메모리 레이턴시 통계 헤더 (memory_stats_t 클래스, memlatstat_* 메서드, 히스토그램/뱅크 접근/행 지역성/AerialVision L2 통계 필드 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/mem_latency_stat.cc | 메모리 레이턴시 통계 구현 전체 (생성자, memlatstat_done/read_done/dram_access/icnt2mem_pop/lat_pw/print, visualizer_print, clear_L2_stats_pw, print_dram_stats 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/gpgpu-sim/addrdec.h | 주소 디코딩 헤더 (enum, struct, class 선언) | 2026-06-13 |
| src/gpgpu-sim/addrdec.cc | 주소 디코딩 구현 (전체 함수 + 인라인 주석) | 2026-06-13 |
| src/gpgpu-sim/mem_fetch.h | 메모리 요청 패킷 헤더 (mf_type, mem_fetch_status, mem_fetch 클래스) | 2026-06-13 |
| src/gpgpu-sim/mem_fetch.cc | 메모리 요청 패킷 구현 (생성자/소멸자/set_status/print/flits) | 2026-06-13 |
| src/gpgpu-sim/scoreboard.h | 스코어보드 헤더 (Scoreboard 클래스 선언 및 필드) | 2026-06-13 |
| src/gpgpu-sim/scoreboard.cc | 스코어보드 구현 (reserveRegisters/releaseRegisters/checkCollision 등) | 2026-06-13 |
| src/option_parser.h | 옵션 파서 C 헤더 (option_dtype enum, C 인터페이스 선언) | 2026-06-13 |
| src/option_parser.cc | 옵션 파서 C++ 구현 (OptionRegistry 템플릿, OptionParser, C 래퍼, UNIT_TEST) | 2026-06-13 |
| src/gpgpu-sim/gpu-misc.h | GPU 공통 유틸 헤더 (LOGB2 선언, DEBUGL1MISS, gs_min2, min3 매크로) | 2026-06-13 |
| src/gpgpu-sim/gpu-misc.cc | GPU 공통 유틸 구현 (LOGB2 비트 조작 5단계 이진 탐색 알고리즘) | 2026-06-13 |
| src/gpgpu-sim/hashing.h | 메모리 뱅크/셋 해시 함수 헤더 (ipoly/bitwise/PAE 선언) | 2026-06-13 |
| src/gpgpu-sim/hashing.cc | 메모리 뱅크/셋 해시 함수 구현 (IPOLY GF(2), XOR, PAE) | 2026-06-13 |
| src/gpgpu-sim/delayqueue.h | 파이프라인 레이턴시 지연 큐 템플릿 (fifo_pipeline<T>, fifo_data<T>) | 2026-06-13 |
| src/gpgpu-sim/stack.h | SIMT 분기 처리용 주소 스택 선언 (Stack 구조체 및 함수 선언) | 2026-06-13 |
| src/gpgpu-sim/stack.cc | SIMT 분기 처리용 주소 스택 구현 (push/pop/top/new/free/reset 등) | 2026-06-13 |
| src/tr1_hash_map.h | C++03/TR1 unordered_map 컴파일러 호환성 추상화 (GCC 버전/디버그 STL 분기) | 2026-06-13 |
| src/gpgpu-sim/dram.h | DRAM 타이밍 헤더 (dram_req_t, bankgrp_t, bank_t, dram_t 클래스 전체 필드 주석) | 2026-06-13 |
| src/gpgpu-sim/dram.cc | DRAM 타이밍 구현 (생성자/push/cycle/issue_col_command/issue_row_command 등) | 2026-06-13 |

| src/abstract_hardware_model.h | GPU 추상 하드웨어 모델 헤더 전체 (inst_t, warp_inst_t, core_t, register_set, 메모리 상수/구조체) | 2026-06-13 |
| src/abstract_hardware_model.cc | GPU 추상 하드웨어 모델 구현 (generate_mem_accesses, coalescing, simt_stack::update, kernel_info_t, core_t) | 2026-06-13 |
| src/gpgpu-sim/dram_sched.h | FR-FCFS DRAM 스케줄러 헤더 (frfcfs_scheduler 클래스, memory_mode enum, 모든 private 필드) | 2026-06-13 |
| src/gpgpu-sim/dram_sched.cc | FR-FCFS DRAM 스케줄러 구현 (생성자, add_req, data_collection, schedule, dram_t::scheduler_frfcfs) | 2026-06-13 |
| src/gpgpu-sim/icnt_wrapper.h | ICNT 래퍼 헤더 (함수 포인터 타입, 전역 포인터 extern, network_mode enum, icnt_wrapper_init/icnt_reg_options 선언) | 2026-06-13 |
| src/gpgpu-sim/icnt_wrapper.cc | ICNT 래퍼 구현 (intersim2_*, LocalInterconnect_* 래퍼 static 함수, icnt_reg_options, icnt_wrapper_init) | 2026-06-13 |
| src/gpgpu-sim/local_interconnect.h | 로컬 인터커넥트 헤더 (inct_config, xbar_router, LocalInterconnect 전체 선언 및 필드) | 2026-06-13 |
| src/gpgpu-sim/local_interconnect.cc | 로컬 인터커넥트 구현 (xbar_router 생성자, Push/Pop/Advance, RR_Advance, iSLIP_Advance, LocalInterconnect 전체) | 2026-06-13 |
| src/gpgpusim_entrypoint.h | 시뮬레이터 진입점 헤더 (GPGPUsim_ctx 클래스, 전체 필드 주석) | 2026-06-13 |
| src/gpgpusim_entrypoint.cc | 시뮬레이터 진입점 구현 (init/thread/sync/SST 전체 함수 주석) | 2026-06-13 |
| src/stream_manager.h | CUDA 스트림 관리 헤더 (CUevent_st, stream_operation, CUstream_st, stream_manager) | 2026-06-13 |
| src/stream_manager.cc | CUDA 스트림 관리 구현 (push/front/operation/register_finished_kernel 등 전체 주석) | 2026-06-13 |

| src/gpgpu-sim/gpu-sim.h | 최상위 GPU 시뮬레이터 헤더 (gpgpu_sim, gpgpu_sim_config, memory_config, power_config, 필수 4섹션 상단 블록, 모든 필드 멀티라인 주석 보강) | 2026-06-14 |
| src/gpgpu-sim/gpu-sim.cc | 최상위 GPU 시뮬레이터 구현 (cycle(), init(), launch(), issue_block2core(), 전체 함수 주석) | 2026-06-13 |
| src/cuda-sim/cuda-math.h | CUDA 수학 에뮬레이션 (정수→float 변환 16종, float2int/uint, __saturatef, __powf, macOS 보완 함수, 상단 4섹션 블록) | 2026-06-13 |
| src/cuda-sim/ptx_ir.h | PTX IR 헤더 (type_info_key, symbol, symbol_table, operand_info, basic_block_t, gpgpu_recon_t, ptx_instruction, param_info, function_info, arg_buffer_t 전체 — 모든 필드/메서드 멀티라인 주석) | 2026-06-14 |
| src/cuda-sim/ptx_ir.cc | PTX IR 구현 (symbol_table, function_info CFG 분석 — create_basic_blocks, find_dominators/postdominators/ipostdominators, do_pdom, ptx_instruction 생성자, ptx_assemble, copy_arg_to_buffer 등 전체 함수 주석) | 2026-06-14 |
| src/cuda-sim/opcodes.h | PTX opcode 열거형 (opcode_t X-매크로, special_regs 전체 필드 주석, wmma_type Tensor Core 연산 분류) | 2026-06-13 |
| src/cuda-sim/ptx_sim.cc | PTX 기능 시뮬레이션 CTA/warp/스레드 상태 관리 (ptx_cta_info/ptx_warp_info/ptx_thread_info 전체 함수 + 인라인 주석, 콜스택/레지스터 프레임/특수 레지스터 구현) | 2026-06-14 |
| src/gpgpu-sim/shader_trace.h | 셰이더 코어 전용 디버그 트레이스 매크로 (SHADER_DTRACE/DPRINTF/SCHED_DPRINTF, TRACING_ON 분기, no-op 정의) | 2026-06-13 |
| src/gpgpu-sim/l2cache_trace.h | L2 캐시/메모리 파티션 전용 디버그 트레이스 매크로 (MEMPART_DTRACE/DPRINTF, MEM_SUBPART_DTRACE/DPRINTF) | 2026-06-13 |
| src/gpgpu-sim/stats.h | 메모리 파이프라인 통계 열거형 (mem_stage_access_type 7종, tlb_request_status 3종, mem_stage_stall_type 9종 전체 필드 주석) | 2026-06-13 |
| src/cuda-sim/cuda-sim.h | PTX 기능 시뮬레이션 공개 인터페이스 (functionalCoreSim 클래스, cuda_sim 클래스 전체 필드/멤버함수, ptx_sim_init_thread/ptx_sim_kernel_info/get_converge_point 선언, 상수 매크로 주석) | 2026-06-14 |
| src/cuda-sim/cuda-sim.cc | PTX 기능 시뮬레이션 구현 (ptx_exec_inst 핵심 실행 루프, ptx_assemble PC할당, pre_decode 타이밍준비, ptx_sim_init_thread CTA초기화, gpgpu_cuda_ptx_sim_main_func 최상위 루프, 텍스처/메모리 API 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/shader.h | SM 파이프라인 헤더 전체 (scheduler_unit 계층, opndcoll_rfu_t, ldst_unit, barrier_set_t, shader_core_config, shader_core_stats, shader_core_ctx, exec_shader_core_ctx, simt_core_cluster, exec_simt_core_cluster, sst_simt_core_cluster, shader_memory_interface 등 모든 클래스 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/shader.cc | SM 파이프라인 구현 전체 (shader_core_ctx::cycle 핵심 루프, warp 스케줄러 6종 GTO/LRR/RRR/TwoLevel/OldestFirst/SWL, operand collector wavefront 알고리즘, ldst_unit writeback/cycle, barrier_set __syncthreads 구현, simt_core_cluster icnt_cycle/issue_block2core, exec_shader_core_ctx::checkExecutionStatusAndUpdate 등 모든 함수 + 인라인 주석) | 2026-06-14 |
| src/intersim2/booksim.hpp | BookSim 전역 기반 헤더 (표준 헤더 일괄 포함, Win32 경고 억제, using namespace std 전역 선언) | 2026-06-14 |
| src/intersim2/booksim_config.hpp | BookSim 설정 클래스 헤더 (BookSimConfig, PowerConfig 선언, 상속 구조 주석) | 2026-06-14 |
| src/intersim2/booksim_config.cpp | BookSim 설정 파라미터 구현 (BookSimConfig 생성자 70여 개 파라미터 6범주 전체, PowerConfig 공정 파라미터 전체) | 2026-06-14 |
| src/intersim2/batchtrafficmanager.hpp | 배치 트래픽 매니저 헤더 (BatchTrafficManager 클래스, 모든 protected 필드 멀티라인 주석, 가상 함수 선언) | 2026-06-14 |
| src/intersim2/batchtrafficmanager.cpp | 배치 트래픽 매니저 구현 (생성자/소멸자, _IssuePacket 주입 결정, _SingleSim 배치 루프, _UpdateOverallStats, DisplayStats/DisplayOverallStats 전체 주석) | 2026-06-14 |
| src/intersim2/networks/anynet.hpp | AnyNet 임의 토폴로지 헤더 (클래스 선언, node_list/router_list/routing_table 멀티라인 필드 주석, min_anynet 함수 선언, global_routing_table 전역 포인터 설계 의도 주석) | 2026-06-14 |
| src/intersim2/networks/anynet.cpp | AnyNet 임의 토폴로지 구현 (readFile 5단계 상태 머신 파싱, _ComputeSize/_BuildNet 채널 연결, buildRoutingTable/route Dijkstra 알고리즘, min_anynet VC 범위 선택, 전체 함수+인라인 주석) | 2026-06-14 |
| src/intersim2/networks/qtree.hpp | QTree 4진 간접 트리 헤더 (k=4/n=3 고정 설계, _k/_n 멀티라인 필드 주석, 인덱싱 함수 3종 프로토타입, HeightFromID/PosFromID 선언) | 2026-06-14 |
| src/intersim2/networks/qtree.cpp | QTree 4진 간접 트리 구현 (_ComputeSize prefix-sum, _BuildNet 라우터 생성+inject/eject+UP/DOWN 채널, _RouterIndex/_InputIndex/_OutputIndex 인덱스 계산, HeightFromID/PosFromID, 전체 주석) | 2026-06-14 |
| src/intersim2/networks/tree4.hpp | Tree4 4진 트리 헤더 (64노드 28라우터 3레벨, (4>>h)*k^h 공식 설명, _k/_n/_channelWidth 멀티라인 필드 주석, _Router/_WireLatency/HeightFromID/PosFromID/SpeedUp 선언) | 2026-06-14 |
| src/intersim2/networks/tree4.cpp | Tree4 4진 트리 구현 (_ComputeSize 28라우터/_channels 계산, _BuildNet h=1↔h=2/h=0↔h=1 채널 연결, _Router 접두합 참조 반환, _WireLatency 물리 레이아웃 레거시 로직, 전체 주석) | 2026-06-14 |
| src/cuda-sim/instructions.cc | PTX 명령어 시맨틱 전체 (lines 1-8677, 상단 4섹션 블록 + 모든 PTX 명령어 구현 함수 및 공통 헬퍼, 인라인 주석) | 2026-06-14 |
| src/gpgpu-sim/power_interface.h | AccelWattch 전력 모델 인터페이스 헤더 (init_mcpat/mcpat_cycle/calculate_hw_mcpat/parse_hw_file/mcpat_reset_perf_count 함수 선언 및 상단 4섹션 블록) | 2026-06-14 |
| src/gpgpu-sim/power_interface.cc | AccelWattch 전력 모델 인터페이스 구현 (전력 시뮬레이션 모드 0/1/2, McPAT 래퍼 카운터 전달, CSV 파싱, 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/gpgpu-sim/power_stat.h | AccelWattch 전력 통계 카운터 헤더 (stat_idx, shader_core_power_stats_pod, mem_power_stats_pod, power_core_stat_t, power_mem_stat_t, power_stat_t 클래스 및 모든 get_*() 쿼리 메서드 주석) | 2026-06-14 |
| src/gpgpu-sim/power_stat.cc | AccelWattch 전력 통계 카운터 구현 (init/save_stats/print/clear, CURRENT/PREV 슬롯 관리, 전체 함수 + 인라인 주석) | 2026-06-14 |

| src/intersim2/traffic.hpp | NoC 합성 트래픽 패턴 클래스 계층 선언 (TrafficPattern 팩토리 및 18종 패턴, 모든 클래스/필드/메서드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/traffic.cpp | NoC 합성 트래픽 패턴 구현 (TrafficPattern::New 팩토리, 18종 dest() 구현, 모든 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/trafficmanager.hpp | BookSim2 트래픽 매니저 기반 클래스 헤더 (New/Run/_Step/_GeneratePacket/_RetireFlit/_ClearStats/_ComputeStats/_SingleSim/WriteStats/DisplayStats 등 모든 멤버 + 필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/trafficmanager.cpp | BookSim2 트래픽 매니저 기반 클래스 구현 (생성자 파라미터 초기화, _Step/_Inject/_IssuePacket/_GeneratePacket/_RetireFlit/_PacketsOutstanding/_SingleSim/Run/통계 출력 등 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/gputrafficmanager.hpp | GPU 특화 트래픽 매니저 헤더 (GPUTrafficManager 클래스, _input_queue, _GeneratePacket/_Step/_RetireFlit/Init 선언, 모든 필드/메서드 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/gputrafficmanager.cpp | GPU 특화 트래픽 매니저 구현 (생성자/소멸자, Init, _GeneratePacket mem_fetch→flit 변환, _RetireFlit 통계, _Step 1사이클 파이프라인 전체 주석) | 2026-06-14 |
| src/intersim2/injection.hpp | NoC 트래픽 주입 프로세스 선언 (InjectionProcess, BernoulliInjectionProcess, OnOffInjectionProcess 클래스 및 모든 필드/가상 함수 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/injection.cpp | NoC 트래픽 주입 프로세스 구현 (InjectionProcess::New 팩토리, Bernoulli test, OnOff 상태 전이 + 패킷 생성, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/routefunc.hpp | NoC 라우팅 함수 헤더 (tRoutingFunction 타입, gRoutingFunctionMap, gNumVCs/요청응답 VC 범위 전역 변수, 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/routefunc.cpp | NoC 라우팅 알고리즘 구현 (dor_mesh/valiant_mesh/min_adapt_mesh/ planar_adapt/ dim_order_torus/ fattree_nca/ chaos 등 20여 종 + InitializeRoutingMap 등록, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/vc.hpp | 가상 채널(Virtual Channel) 헤더 (VC 클래스, eVCState, AddFlit/RemoveFlit/Route/SetOutput/UpdatePriority 등 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/vc.cpp | 가상 채널 구현 (VCSTATE 배열, 생성자, AddFlit/RemoveFlit/Route/SetOutput/UpdatePriority, 우선순위 기증 메커니즘, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/buffer.hpp | 라우터 입력 포트 버퍼 헤더 (Buffer 클래스, num_vcs/buf_size/vc_buf_size 기반 VC 배열, AddFlit/RemoveFlit/FrontFlit/Full/Route 등 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/buffer.cpp | 라우터 입력 포트 버퍼 구현 (생성자/소멸자, AddFlit 오버플로 검사, Display, TRACK_BUFFERS 클래스별 occupancy, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/buffer_state.hpp | 다운스트림 버퍼 상태 추적 헤더 (BufferState + 7가지 BufferPolicy 전략 패턴, private/shared/limited/dynamic/shifting/feedback/simplefeedback, ProcessCredit/SendingFlit/TakeBuffer/IsFullFor 등 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/buffer_state.cpp | 다운스트림 버퍼 상태 추적 구현 (BufferPolicy::New 팩토리, 7가지 정책 구현, ProcessCredit/SendingFlit/TakeBuffer/Display, RTT 기반 피드백 로직, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/credit.hpp | NoC 크레딧 패킷 헤더 (Credit 클래스, vc 집합, head/tail/id, 풀 기반 New/Free/FreeAll/OutStanding, 모든 필드/메서드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/credit.cpp | NoC 크레딧 패킷 구현 (정적 _all/_free 스택, New/Free/FreeAll/OutStanding, 풀 기반 재사용, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/flit.hpp | NoC 전송 기본 단위 Flit 헤더 (Flit 클래스, FlitType enum, vc/cl/head/tail/ctime/itime/atime/id/pid/record/src/dest/pri/hops/watch/subnetwork/intm/ph/data/la_route_set, 풀 기반 New/Free/FreeAll, 모든 필드/메서드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/flit.cpp | NoC 전송 기본 단위 Flit 구현 (operator<<, Reset, New/Free/FreeAll, 풀 기반 재사용, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/flitchannel.hpp | NoC 물리 링크(Flit 채널) 헤더 (FlitChannel 클래스, Channel<Flit> 상속, SetSource/SetSink/Send/ReadInputs/WriteOutputs, 활동/유휴 통계, 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/flitchannel.cpp | NoC 물리 링크(Flit 채널) 구현 (생성자, SetSource/SetSink, Send 시 active/idle 카운터, ReadInputs/WriteOutputs watch 로그, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/outputset.hpp | 라우터 출력 포트/VC 후보 집합 헤더 (OutputSet 클래스, sSetElement 구조체, Clear/Add/AddRange/OutputEmpty/NumVCs/GetSet/GetVC/GetPortVC, operator< 우선순위 정렬, 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/outputset.cpp | 라우터 출력 포트/VC 후보 집합 구현 (Clear/Add/AddRange/NumVCs/OutputEmpty/GetSet/GetVC/GetPortVC, 레거시 지원 함수, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/packet_reply_info.hpp | 요청 패킷 응답 추적 헤더 (PacketReplyInfo 클래스, source/time/record/type, 풀 기반 New/Free/FreeAll, 모든 필드/메서드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/packet_reply_info.cpp | 요청 패킷 응답 추적 구현 (정적 _all/_free 스택, New/Free/FreeAll, 풀 기반 재사용, 전체 함수 + 인라인 주석 + 상단 4섹션 블록) | 2026-06-14 |
| src/intersim2/pipefifo.hpp | 파이프라인 FIFO 템플릿 헤더 (PipelineFIFO<T> 클래스, _lanes/_depth/_pipe_len/_pipe_ptr/_data 링 버퍼, Write/WriteAll/Read/Advance, 모든 메서드/필드 멀티라인 주석 + 상단 4섹션 블록) | 2026-06-14 |

| src/debug.h | PTX 인터랙티브 디버거 브레이크포인트/워치포인트 헤더 (brk_pt 클래스, 생성자, is_equal, thread_at_brkpt, 모든 필드 멀티라인 주석) | 2026-06-14 |
| src/debug.cc | PTX 인터랙티브 디버거 구현 (hit_watchpoint, gpgpu_debug 대화형 루프, thread_at_brkpt, 명령어별 분기 및 인라인 주석) | 2026-06-14 |
| src/trace.h | GPGPU-Sim 트레이스/디버그 출력 시스템 헤더 (Trace 네임스페이스, trace_streams_type enum, DTRACE/DPRINTF/DPRINTFG 매크로, 전역 변수 주석) | 2026-06-14 |
| src/trace.cc | GPGPU-Sim 트레이스 시스템 구현 (Trace::init config_str 파싱, trace_streams_str[] X-매크로 생성, 전역 변수 정의 주석) | 2026-06-14 |
| src/statwrapper.h | intersim2 Stats 클래스 C 래퍼 헤더 (StatCreate/Clear/AddSample/Average/Max/Min/Disp 함수 선언 및 주석) | 2026-06-14 |
| src/statwrapper.cc | intersim2 Stats 클래스 C 래퍼 구현 (StatCreate/AddSample/Average/Max/Min/Disp, void*→Stats* 캐스팅, 단위 테스트 주석) | 2026-06-14 |
| src/cuda-sim/cuda_device_printf.h | GPU 디바이스 printf() 에뮬레이션 인터페이스 (gpgpusim_cuda_vprintf 선언) | 2026-06-14 |
| src/cuda-sim/cuda_device_printf.cc | GPU 디바이스 printf() 에뮬레이션 구현 (my_cuda_printf, gpgpusim_cuda_vprintf) | 2026-06-14 |
| src/cuda-sim/cuda_device_runtime.h | CUDA Dynamic Parallelism(CDP) 디바이스 런타임 인터페이스 (device_launch_config_t/operation_t, cuda_device_runtime 클래스 및 필드) | 2026-06-14 |
| src/cuda-sim/cuda_device_runtime.cc | CDP 런타임 API 에뮬레이션 구현 (getParameterBufferV2, launchDeviceV2, streamCreateWithFlags, launch_all/one_device_kernel) | 2026-06-14 |
| src/cuda-sim/decuda_pred_table/decuda_pred_table.h | decuda predicate 조회 테이블 헤더 (pred_lookup 선언) | 2026-06-14 |
| src/cuda-sim/decuda_pred_table/decuda_pred_table.cc | decuda predicate 조회 테이블 구현 (G80 기반 정적 pred_table 및 pred_lookup) | 2026-06-14 |
| src/cuda-sim/ptx_loader.h | PTX/ptxinfo 로딩 및 PTXPlus 변환 인터페이스 (ptxinfo_data 클래스 및 필드) | 2026-06-14 |
| src/cuda-sim/ptx_loader.cc | PTX/ptxinfo 로딩 및 PTXPlus 변환 구현 (load_ptx_from_string, ptxinfo_load_from_string, convert_ptx_and_sass_to_ptxplus, fix_duplicate_errors) | 2026-06-14 |
| src/cuda-sim/ptx_parser.h | PTX 파서 semantic recognizer 헤더 (ptx_recognizer 클래스, 모든 멤버/필드/메서드 멀티라인 주석) | 2026-06-14 |
| src/cuda-sim/ptx_parser.cc | PTX 파서 semantic action 구현 (init_parser, init_directive/instruction_state, add_identifier/add_instruction/end_function, pad_address 등 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/cuda-sim/ptx.l | PTX Lex/Flex 어휘 분석기 (opcode/directive/타입/옵션/피연산자 토큰, START 조건, ptx_error) | 2026-06-14 |
| src/cuda-sim/ptx.y | PTX Bison/Yacc 문법 파서 (함수/변수/명령어/피연산자/옵션 규칙, WMMA/Tex/Vector/Prototype 문법) | 2026-06-14 |
| src/cuda-sim/ptxinfo.l | ptxas 리소스 정보 Lex/Flex 어휘 분석기 (ptxinfo_lex, ptxinfo_error) | 2026-06-14 |
| src/cuda-sim/ptxinfo.y | ptxas 리소스 정보 Bison/Yacc 문법 파서 (function_name, function_info, tuple, duplicate 규칙) | 2026-06-14 |
| src/cuda-sim/ptx-stats.h | PTX 소스 라인 단위 통계 인터페이스 (ptx_stats 클래스, 자유 함수, 옵션 등록) | 2026-06-14 |
| src/cuda-sim/ptx-stats.cc | PTX 소스 라인 단위 통계 구현 (ptx_file_line_stats, inflight tracker, add_* 함수군, write_file) | 2026-06-14 |
| src/cuda-sim/memory.h | PTX 시뮬레이션용 메모리 주소 공간 추상화 (mem_storage, memory_space, memory_space_impl) | 2026-06-14 |
| src/cuda-sim/memory.cc | PTX 시뮬레이션용 메모리 주소 공간 구현 (read/write fast/slow path, watchpoint, UNIT_TEST) | 2026-06-14 |
| src/gpgpu-sim/stat-tool.h | GPGPU-Sim 통계 도구 헤더 (snap_shot_trigger/spill_log_interface, thread_CFlocality, insn_warp_occ_logger, linear_histogram_logger, SM별 워프/메모리/레이턴시/캐시/CTA 로거 인터페이스 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/stat-tool.cc | GPGPU-Sim 통계 도구 구현 (스냅샷/스필 인프라, thread_CFlocality/linear_histogram_logger 멤버 함수, SM별 Create/Log/Snapshot/Print 로거 전체 주석) | 2026-06-14 |
| src/gpgpu-sim/histogram.h | 히스토그램 헤더 (binned_histogram, pow2_histogram, linear_histogram 클래스 및 bin 분류/출력 메서드 주석) | 2026-06-14 |
| src/gpgpu-sim/histogram.cc | 히스토그램 구현 (binned_histogram 동적 메모리/출력, pow2_histogram 비트연산 log2, linear_histogram stride 분류 주석) | 2026-06-14 |
| src/gpgpu-sim/traffic_breakdown.h | ICNT 네트워크 트래픽 분류 헤더 (traffic_breakdown 클래스, record_traffic/classify_memfetch/packet_size 주석) | 2026-06-14 |
| src/gpgpu-sim/traffic_breakdown.cc | ICNT 네트워크 트래픽 분류 구현 (print/record_traffic/classify_memfetch 구현 및 mem_access_type 분기 주석) | 2026-06-14 |
| src/gpgpu-sim/visualizer.h | AerialVision 시각화 지원 헤더 (time_vector_create/update/check/print 함수 및 LD/ST 레이턴시 추적 인터페이스 주석) | 2026-06-14 |
| src/gpgpu-sim/visualizer.cc | AerialVision 시각화 구현 (gpgpu_sim::visualizer_printstat, my_time_vector 타임스탬프/레이턴시 분포/출력 전체 주석) | 2026-06-14 |
| src/intersim2/arbiters/arbiter.hpp | Arbiter 기반 클래스 헤더 (entry_t, _request, _selected, AddRequest/Arbitrate/UpdateState/Clear/NewArbiter 선언, BookSimConfig 연동) | 2026-06-14 |
| src/intersim2/arbiters/arbiter.cpp | Arbiter 기반 클래스 구현 (생성자, AddRequest, Arbitrate, Clear, NewArbiter 팩토리, arb_type 연동) | 2026-06-14 |
| src/intersim2/arbiters/matrix_arb.hpp | MatrixArbiter 헤더 (_matrix 우선순위 행렬, _last_req, Arbitrate/UpdateState/Clear 선언) | 2026-06-14 |
| src/intersim2/arbiters/matrix_arb.cpp | MatrixArbiter 구현 (생성자 하삼각 초기화, UpdateState 승자 강등, O(N^2) 행렬 탐색 중재) | 2026-06-14 |
| src/intersim2/arbiters/prio_arb.hpp | PriorityArbiter 헤더 (sRequest, _requests, _rr_ptr, _match, AddRequest/RemoveRequest/Arbitrate/Update 선언) | 2026-06-14 |
| src/intersim2/arbiters/prio_arb.cpp | PriorityArbiter 구현 (in 오름차순 리스트 삽입, _rr_ptr 기준 순환 탐색 최고 우선순위 선택, spec_sw_allocator 연동) | 2026-06-14 |
| src/intersim2/arbiters/roundrobin_arb.hpp | RoundRobinArbiter 헤더 (_pointer, Supersedes, AddRequest O(1) 후보 갱신, UpdateState 선언) | 2026-06-14 |
| src/intersim2/arbiters/roundrobin_arb.cpp | RoundRobinArbiter 구현 (AddRequest에서 _best_input 즉시 갱신, Arbitrate O(1) 위임, UpdateState 포인터 전진) | 2026-06-14 |
| src/intersim2/arbiters/tree_arb.hpp | TreeArbiter 헤더 (_group_size, _group_arbiters, _global_arbiter, _group_reqs, 2단계 트리 중재 선언) | 2026-06-14 |
| src/intersim2/arbiters/tree_arb.cpp | TreeArbiter 구현 (groups개 그룹+전역 중재자 생성, 2단계 중재, UpdateState 승리 그룹만 갱신) | 2026-06-14 |
| src/intersim2/config.l | BookSim 설정 파일 Flex/Lex 어휘 분석기 (토큰 정의, 정수/실수/문자열/구분자 규칙, yyerror/yywrap) | 2026-06-14 |
| src/intersim2/config.y | BookSim 설정 파일 Bison/Yacc 문법 분석기 (command 재귀, STR=STR/NUM/FNUM 할당 액션) | 2026-06-14 |
| src/intersim2/config_utils.hpp | BookSim Configuration 클래스 헤더 (_int_map/_str_map/_float_map, Get/Add, ParseFile, WriteFile) | 2026-06-14 |
| src/intersim2/config_utils.cpp | BookSim Configuration 클래스 구현 (파라미터 읽기/쓰기, 설정 파일 파싱/출력) | 2026-06-14 |
| src/intersim2/globals.hpp | BookSim 전역 변수/함수 선언 (GetSimTime, GetStats, gK/gN/gC/gNodes, gPrintActivity/gTrace, gWatchOut) | 2026-06-14 |
| src/intersim2/interconnect_interface.hpp | GPGPU-Sim ↔ BookSim NoC 어댑터 API 헤더 (InterconnectInterface, Push/Pop/Advance/HasBuffer, _BoundaryBufferItem) | 2026-06-14 |
| src/intersim2/interconnect_interface.cpp | GPGPU-Sim ↔ BookSim NoC 어댑터 구현 (CreateInterconnect, Push mem_fetch→flit, Pop, Busy, DisplayStats) | 2026-06-14 |
| src/intersim2/intersim_config.hpp | GPGPU-Sim 전용 NoC 설정 클래스 헤더 (IntersimConfig, BookSimConfig 상속) | 2026-06-14 |
| src/intersim2/intersim_config.cpp | GPGPU-Sim 전용 NoC 설정 클래스 구현 (perfect_icnt/fixed_lat_per_hop/use_map/flit_size/buffer_size 등 기본값 등록) | 2026-06-14 |
| src/intersim2/main.cpp | BookSim 독립 실행형 메인 (Simulate, main, Network/TrafficManager 생성 및 Run, power_module) | 2026-06-14 |
| src/intersim2/misc_utils.hpp | BookSim 정수 수학 유틸리티 헤더 (log_two, powi) | 2026-06-14 |
| src/intersim2/misc_utils.cpp | BookSim 정수 수학 유틸리티 구현 (log_two 비트 시프트, powi 반복 곱셈) | 2026-06-14 |
| src/intersim2/module.hpp | BookSim 모듈 계층 기반 클래스 헤더 (Module, _name/_fullname/_children, DisplayHierarchy/Error/Debug) | 2026-06-14 |
| src/intersim2/module.cpp | BookSim 모듈 계층 기반 클래스 구현 (생성자, _AddChild, DisplayHierarchy, Error/Debug/Display) | 2026-06-14 |
| src/intersim2/random_utils.hpp | BookSim 난수 생성 유틸리티 헤더 (RandomSeed, RandomInt, RandomFloat, ran_/ranf_ 래퍼) | 2026-06-14 |
| src/intersim2/rng.c | Knuth RANARRAY 정수 난수 생성기 C 구현 (ran_x, ran_array, ran_start, ran_arr_next) | 2026-06-14 |
| src/intersim2/rng-double.c | Knuth RANARRAY 부동소수점 난수 생성기 C 구현 (ran_u, ranf_array, ranf_start, ranf_arr_next) | 2026-06-14 |
| src/intersim2/rng_wrapper.cpp | 정수 RNG C++ 래퍼 (rng.c #include, main→rng_main rename, ran_next) | 2026-06-14 |
| src/intersim2/rng_double_wrapper.cpp | 실수 RNG C++ 래퍼 (rng-double.c #include, main→rng_double_main rename, ranf_next) | 2026-06-14 |
| src/intersim2/stats.hpp | BookSim 통계 수집 클래스 헤더 (Stats, AddSample, Average, Variance, histogram, operator<<) | 2026-06-14 |
| src/intersim2/stats.cpp | BookSim 통계 수집 클래스 구현 (Clear, AddSample, Average, Variance, Display, operator<<) | 2026-06-14 |
| src/intersim2/timed_module.hpp | 사이클 기반 동기 모듈 기반 클래스 헤더 (TimedModule, ReadInputs/Evaluate/WriteOutputs) | 2026-06-14 |
| src/intersim2/channel.hpp | NoC 물리 링크(채널) 템플릿 헤더 (Channel<T>, Send/Receive, ReadInputs/WriteOutputs, _wait_queue) | 2026-06-14 |
| src/intersim2/routers/iq_router.hpp | IQ(Input-Queued) 라우터 헤더 (IQRouter 클래스, VC 정책/파이프라인 deque/할당자/NOQ/스위치 홀드 필드 및 public 메서드 전체 주석) | 2026-06-14 |
| src/intersim2/routers/iq_router.cpp | IQ(Input-Queued) 라우터 구현 (생성자/소멸자, RC/VA/SA/ST/LT 5단계 파이프라인 Evaluate/Update, 투기적 SA, 스위치 홀드, NOQ 룩어헤드, piggyback VC 할당, 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/intersim2/routers/router.hpp | 라우터 추상 기반 클래스 헤더 (Router, NewRouter 팩토리, Evaluate fractional-cycle 누적, 채널 연결, stall sentinel, fault 채널 헬퍼 전체 주석) | 2026-06-14 |
| src/intersim2/routers/router.cpp | 라우터 추상 기반 클래스 구현 (NewRouter "iq"/"event"/"chaos" 분기, Credit/Flit 채널 연결, Evaluate/WriteOutputs, 디버그 Display 전체 주석) | 2026-06-14 |
| src/intersim2/routers/event_router.hpp | 이벤트 기반 라우터 및 출력 VC 상태 머신 헤더 (EventNextVCState, EventRouter, arrival/transport 이벤트 큐, credit/transport/가상 출력 파이프라인 전체 주석) | 2026-06-14 |
| src/intersim2/routers/event_router.cpp | 이벤트 기반 라우터 구현 (생성자, ReadInputs/_InternalStep/WriteOutputs, _ReceiveFlits/_ProcessWaiting/_SendTransport/_ArrivalArb/_TransportArb/_OutputQueuing, EventNextVCState 상태 머신 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/intersim2/routers/chaos_router.hpp | Chaos(adaptive multi-queue) 라우터 헤더 (ChaosRouter, eQState 상태, multi-queue, _NextInterestingChannel/_OutputAdvance 필드/메서드 전체 주석) | 2026-06-14 |
| src/intersim2/routers/chaos_router.cpp | Chaos(adaptive multi-queue) 라우터 구현 (생성자, multi-queue 버퍼/후면 상태, _NextInterestingChannel 스케줄링, _OutputAdvance crossbar/MQ 이동 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/intersim2/allocators/allocator.hpp | Allocator 기반 클래스 및 DenseAllocator/SparseAllocator 선언 (sRequest, AddRequest/Clear/Allocate, NewAllocator 팩토리, vc_allocator/sw_allocator/alloc_iters 연동) | 2026-06-14 |
| src/intersim2/allocators/allocator.cpp | Allocator 기반 클래스 구현 (DenseAllocator/SparseAllocator 생성자·소멸자·AddRequest·Clear, NewAllocator "maxsize"/"pim"/"islip"/"loa"/"wavefront"/"selalloc"/"separable_*" 팩토리 분기) | 2026-06-14 |
| src/intersim2/allocators/islip.hpp | iSLIP 할당기 헤더 (DenseAllocator 상속, _gptrs/_aptrs 라운드로빈 포인터, Allocate() Grant-Accept 반복 선언, alloc_iters 연동) | 2026-06-14 |
| src/intersim2/allocators/islip.cpp | iSLIP 할당기 구현 (Grant-Accept 2단계 반복 매칭, 포인터 갱신으로 공정성 보장) | 2026-06-14 |
| src/intersim2/allocators/loa.hpp | LOA(Lonely Output Arbiter) 할당기 헤더 (Count-Request-Grant 3단계, _counts/_req/_rptr/_gptr 필드 주석) | 2026-06-14 |
| src/intersim2/allocators/loa.cpp | LOA 할당기 구현 (외로운 출력 우선 3단계 매칭, 부하 분산 중심) | 2026-06-14 |
| src/intersim2/allocators/maxsize.hpp | MaxSizeMatch(최대 이분 매칭) 헤더 (BFS 최단 증가 경로, _from/_s/_ns/_prio 필드 주석) | 2026-06-14 |
| src/intersim2/allocators/maxsize.cpp | MaxSizeMatch 구현 (O(N^3) 최단 증가 경로 반복, _prio 라운드로빈 공정성) | 2026-06-14 |
| src/intersim2/allocators/pim.hpp | PIM(Parallel Iterative Matching) 헤더 (RandomInt 기반 무작위 Grant-Accept, _PIM_iter 필드) | 2026-06-14 |
| src/intersim2/allocators/pim.cpp | PIM 구현 (무작위 반복 매칭, 포인터 미갱신) | 2026-06-14 |
| src/intersim2/allocators/selalloc.hpp | SelAlloc 우선순위 선택 할당기 헤더 (SparseAllocator 상속, _iter/_aptrs/_gptrs/_outmask, MaskOutput/PrintRequests) | 2026-06-14 |
| src/intersim2/allocators/selalloc.cpp | SelAlloc 구현 (우선순위 최대 Grant-Accept, 마스크된 출력 제외) | 2026-06-14 |
| src/intersim2/allocators/separable.hpp | SeparableAllocator 기반 클래스 헤더 (입출력 Arbiter 배열, Clear, InputFirst/OutputFirst 상속 기반) | 2026-06-14 |
| src/intersim2/allocators/separable.cpp | SeparableAllocator 구현 (Arbiter::NewArbiter 생성, Clear 최적화) | 2026-06-14 |
| src/intersim2/allocators/separable_input_first.hpp | 입력 먼저 분리 할당기 헤더 (SeparableAllocator 상속, Allocate() 선언) | 2026-06-14 |
| src/intersim2/allocators/separable_input_first.cpp | 입력 먼저 분리 할당기 구현 (입력 중재 → 출력 중재 2단계) | 2026-06-14 |
| src/intersim2/allocators/separable_output_first.hpp | 출력 먼저 분리 할당기 헤더 (SeparableAllocator 상속, Allocate() 선언) | 2026-06-14 |
| src/intersim2/allocators/separable_output_first.cpp | 출력 먼저 분리 할당기 구현 (출력 중재 → 입력 중재 2단계) | 2026-06-14 |
| src/intersim2/allocators/wavefront.hpp | Wavefront(파면) 할당기 헤더 (DenseAllocator 상속, _priorities/_skip_diags/_pri/_square, AddRequest/Allocate) | 2026-06-14 |
| src/intersim2/allocators/wavefront.cpp | Wavefront 구현 (대각선 순환 탐색 매칭, rr_wavefront 동적 우선순위) | 2026-06-14 |
| src/intersim2/power/buffer_monitor.hpp | NoC 입력 버퍼 모니터 헤더 (BufferMonitor, 입력 포트×클줄별 read/write 카운터, Power_Module::calcBuffer 연동, 상단 4섹션 블록 + 모든 필드/메서드 주석) | 2026-06-14 |
| src/intersim2/power/buffer_monitor.cpp | NoC 입력 버퍼 모니터 구현 (생성자, index/cycle/write/read/display, operator<<, 인라인 주석) | 2026-06-14 |
| src/intersim2/power/switch_monitor.hpp | NoC 크로스바 스위칭 모니터 헤더 (SwitchMonitor, 입력×출력×클줄별 traversal 카운터, Power_Module::calcSwitch 연동, 상단 4섹션 블록 + 모든 필드/메서드 주석) | 2026-06-14 |
| src/intersim2/power/switch_monitor.cpp | NoC 크로스바 스위칭 모니터 구현 (생성자, index/cycle/traversal/display, operator<<, 인라인 주석) | 2026-06-14 |
| src/intersim2/power/power_module.hpp | BookSim2 NoC 전력/면적 추정 모듈 헤더 (Power_Module, struct wire, 공정/NoC 설정 상수, 채널/버퍼/스위치/출력/면적 계산 메서드, 상단 4섹션 블록 + 모든 필드 주석) | 2026-06-14 |
| src/intersim2/power/power_module.cpp | BookSim2 NoC 전력/면적 추정 모듈 구현 (생성자 공정 파라미터 초기화, wireOptimize 반복기 탐색, calcChannel/calcBuffer/calcSwitch, powerCrossbar/powerMemory/area 공식, run() 전력/면적 보고서, 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/cacti_interface.h | CACTI 공개 인터페이스 헤더 (InputParameter, results_mem_array, uca_org_t, mem_array, cacti_interface 오버로드 전체 주석) | 2026-06-14 |
| src/accelwattch/cacti/cacti_interface.cc | CACTI 결과 후처리 구현 (uca_org_t find_delay/energy/area/cyc, mem_array::lt, adjust_area 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/parameter.h | CACTI 기술/동적 파라미터 헤더 (TechnologyParameter/DeviceType/InterconnectType/MemoryType/DynamicParameter 모든 필드 멀티라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/parameter.cc | DynamicParameter 생성자 전체 주석 (서브어레이/mat/어드레스 비트/ECC 계산 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/technology.cc | 공정 기술 파라미터 초기화 (wire_resistance, wire_capacitance, init_tech_params 모든 공정 노드 보간 블록 주석) | 2026-06-14 |
| src/accelwattch/cacti/uca.h | UCA 캐시 모델 헤더 (UCA 클래스, compute_delays/compute_power_energy, 면적/전력/지연 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/uca.cc | UCA 캐시 모델 구현 (생성자, compute_delays, compute_power_energy 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/nuca.h | NUCA 캐시 모델 헤더 (nuca_org_t, Nuca 클래스, sim_nuca/find_optimal_nuca/calc_cycles 선언 주석) | 2026-06-14 |
| src/accelwattch/cacti/nuca.cc | NUCA 캐시 모델 구현 (sim_nuca, find_optimal_nuca, check_nuca_org, calculate_nuca_area, print_nuca 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/Ucache.h | UCA 설계 공간 탐색 헤더 (min_values_t, solution, calculate_time, solve, calc_time_mt_wrapper_struct 모든 필드/함수 주석) | 2026-06-14 |
| src/accelwattch/cacti/Ucache.cc | UCA 설계 공간 탐색 구현 (calc_time_mt_wrapper, calculate_time, check_uca_org, check_mem_org, find_optimal_uca, filter_tag/data_arr, solve, update 전체 함수 + 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/component.h | CACTI Component 기반 클래스 헤더 (area/power/delay/cycle_time, compute_gate_area, compute_tr_width_after_folding, height_sense_amplifier, logical_effort 모든 필드/함수 주석) | 2026-06-14 |
| src/accelwattch/cacti/component.cc | CACTI Component 기반 클래스 구현 (생성자/소멸자, 게이트 면적/폴드/SA 높이/논리노력 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/mat.h | CACTI Mat 클래스 헤더 (서브회로 포인터, delay/power 필드, compute_delays/compute_power_energy 및 private 지연 계산 함수 전체 주석) | 2026-06-14 |
| src/accelwattch/cacti/mat.cc | CACTI Mat 구현 (생성자/소멸자, compute_delays, compute_bitline/sa/subarray_out/comparator/cam, compute_power_energy 모든 함수 + 핵심 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/subarray.h | CACTI Subarray 헤더 (SRAM/DRAM/CAM 셀, C_wl/C_bl/R_wl, num_rows/cols 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/subarray.cc | CACTI Subarray 구현 (생성자 면적/ECC, compute_C 워드라인·비트라인 커패시턴스 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/bank.h | CACTI Bank 헤더 (Mat/H-tree 구성, compute_delays/compute_power_energy, 뱅크 파라미터 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/bank.cc | CACTI Bank 구현 (생성자 매트/H-tree 생성, 지연/전력 집계 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/decoder.h | CACTI Decoder/PredecBlk/PredecBlkDrv/Predec/Driver 헤더 (디코더 계층 모든 클래스/필드/함수 주석) | 2026-06-14 |
| src/accelwattch/cacti/decoder.cc | CACTI Decoder/PredecBlk/PredecBlkDrv/Predec/Driver 구현 (논리노력/면적/지연/누설 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/crossbar.h | CACTI Crossbar 헤더 (MCPAT_Crossbar 클래스, compute_power/delays 선언 및 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/crossbar.cc | CACTI Crossbar 구현 (생성자, 면적/지연/동적·누설 전력 계산 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/arbiter.h | CACTI Arbiter 헤더 (MCPAT_Arbiter 클래스, compute_power/delays 선언 및 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/arbiter.cc | CACTI Arbiter 구현 (생성자, request/Grant/ arbitration 지연/전력 계산 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/wire.h | CACTI Wire 헤더 (반복기 배선 정적 멤버, delay_optimal_wire/low_swing_model, Wire_type/Placement 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/wire.cc | CACTI Wire 구현 (Global/5/10/20/30/Low_swing 모델, wire_cap/wire_res, init_wire/update_fullswing 모든 함수 + 핵심 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/htree2.h | CACTI Htree2 헤더 (H-tree 글로벌 배선, in_out_data/address/search 트리, 지연/전력 필드 주석) | 2026-06-14 |
| src/accelwattch/cacti/htree2.cc | CACTI Htree2 구현 (H-tree 생성, 와이어/드라이버/버퍼 지연·전력 집계 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/basic_circuit.h | CACTI 기본 회로 헤더 (tr_R_on, gate_C, drain_C_, horowitz, leakage 함수 및 gate 타입 enum 주석) | 2026-06-14 |
| src/accelwattch/cacti/basic_circuit.cc | CACTI 기본 회로 구현 (트랜지스터 저항/커패시턴스/누설/Horowitz 공식 인라인 주석) | 2026-06-14 |
| src/accelwattch/core.h | AccelWattch SM 코어 전력 모델 헤더 (BranchPredictor, InstFetchU, SchedulerU, RENAMINGU, LoadStoreU, MemManU, RegFU, EXECU, Core 클래스 및 get_coefficient_* 계수 메서드 전체 주석) | 2026-06-14 |
| src/accelwattch/core.cc | AccelWattch SM 코어 전력 모델 구현 (모든 생성자/computeEnergy/compute/displayEnergy/소멸자 + XML 파라미터/캐시/RFU/실행유닛/idle energy 인라인 주석) | 2026-06-14 |
| src/accelwattch/gpgpu_sim_wrapper.h | AccelWattch 전력 모델 래퍼 헤더 (gpgpu_sim_wrapper 클래스, 생성자/소멸자, set_*_power, compute, update_components_power, power_metrics_calculations, print_power_kernel_stats 등 모든 public 메서드/필드 4섹션 상단 블록 + 한국어 주석) | 2026-06-14 |
| src/accelwattch/gpgpu_sim_wrapper.cc | AccelWattch 전력 모델 래퍼 구현 (상단 4섹션 블록, 모든 set_*_power/compute/update_components_power/calculate_static_power/print_power_kernel_stats 함수 주석, McPAT 활동 카운터 주입 및 스케일링 계수 인라인 주석) | 2026-06-14 |
| src/accelwattch/XML_Parse.h | AccelWattch XML 설정 구조체 헤더 (상단 4섹션 블록, ParseXML/perf_count_t/system_*/root_system 구조체/enum 주석, 모든 구조체 필드 한국어 주석) | 2026-06-14 |
| src/accelwattch/XML_Parse.cc | AccelWattch XML 설정 파싱 구현 (상단 4섹션 블록, ParseXML::parse/initialize 함수 주석, sys.* XML 파라미터/통계 할당 라인 인라인 주석) | 2026-06-14 |
| src/accelwattch/xmlParser.h | AccelWattch XML 파서 라이브러리 헤더 (상단 4섹션 블록, XMLError/XMLElementType/XMLNode/XMLAttribute/XMLClear/XMLResults/XMLNodeContents/ToXMLStringTool/XMLParserBase64Tool 구조체/enum/클래스 주석) | 2026-06-14 |
| src/accelwattch/xmlParser.cc | AccelWattch XML 파서 라이브러리 구현 (상단 4섹션 블록, parseString/parseFile/openFileHelper/ParseXMLElement/getChildNode/getAttribute/createXMLString 등 핵심 함수 주석, 제어문/반환 인라인 주석) | 2026-06-14 |
| src/accelwattch/cacti/area.h | CACTI 면적 표현 헤더 (Area 클래스 w/h/area/get_area/set/get, 상단 4섹션 + AccelWattch XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/area.cc | CACTI 면적 모델 .cc 자리표시자 (Area 인라인 구현 안내, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/const.h | CACTI 전역 상수/마스크 헤더 (ADDRESS_BITS, fopt, pppm_* 마스크, ram_cell_tech_type_num, MAX_NUMBER_GATES_STAGE 등 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/io.h | CACTI 결과 출력 함수 헤더 (output_UCA/output_data_csv 선언, 상단 4섹션 + AccelWattch XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/io.cc | CACTI 입출력 및 최상위 인터페이스 구현 (parse_cfg, display_ip, powerComponents/powerDef operator+/operator*, cacti_interface 4종 오버로드, error_checking, output_UCA, output_data_csv, InputParameter 생성자, init_interface, reconfigure 전체 주석 + 인라인 주석) | 2026-06-15 |
| src/accelwattch/cacti/main.cc | CACTI 독립 실행형 main() (cfg/52인수/54인수 경로, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/router.h | MCPAT_Router NUCA NoC 라우터 모델 헤더 (Crossbar/Arbiter/VC buffer 서브컴포넌트, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/router.cc | MCPAT_Router 구현 (calc_router_parameters/get_router_power/delay/area/buffer_stats/cb_stats, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/highradix.h | 고기수 NoC 라우터 모델 헤더 (HighRadix/Waveguide, 서브스위치 분해, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/cacti/highradix.cc | 고기수 NoC 라우터 모델 구현 (HighRadix 생성자/compute_power/sub_switch_power/buffer_*/print_router, 상단 4섹션 + XML/config 연동 언급) | 2026-06-15 |
| src/accelwattch/arch_const.h | McPAT 아키텍처 상수 헤더 (ISA/인스트럭션/데이터 폭, 포트 수 등) | 2026-06-15 |
| src/accelwattch/array.h | ArrayST/DataCache 헤더 (SRAM 배열 및 캐시 보조 버퍼 구조체) | 2026-06-15 |
| src/accelwattch/array.cc | ArrayST/DataCache 구현 (생성자, computeEnergy, 누설 스케일링) | 2026-06-15 |
| src/accelwattch/basic_components.h | statsDef, powerComponents, powerDef, Component 기반 클래스 헤더 | 2026-06-15 |
| src/accelwattch/basic_components.cc | Component 연산자 및 누설 전력 스케일링 구현 | 2026-06-15 |
| src/accelwattch/globalvar.h | McPAT 전역 변수/상수 헤더 | 2026-06-15 |
| src/accelwattch/interconnect.h | interconnect(와이어/버스) 전력/면적 모델 헤더 | 2026-06-15 |
| src/accelwattch/interconnect.cc | interconnect 구현 (와이어/반복기/버스 파라미터 계산) | 2026-06-15 |
| src/accelwattch/iocontrollers.h | NIU/PCIe/Flash 컨트롤러 전력/면적 헤더 | 2026-06-15 |
| src/accelwattch/iocontrollers.cc | NIU/PCIe/Flash 컨트롤러 구현 | 2026-06-15 |
| src/accelwattch/logic.h | 디지털 논리/기능 유닛(FU) 및 CacheDynParam 헤더 | 2026-06-15 |
| src/accelwattch/logic.cc | 디지털 논리/FunctionalUnit/inst_decoder 구현 | 2026-06-15 |
| src/accelwattch/main.cc | McPAT/AccelWattch 독립 실행형 main() | 2026-06-15 |
| src/accelwattch/memoryctrl.h | 메모리 컨트롤러(MCFrontend/MCBackend/MCPHY/DRAM) 헤더 | 2026-06-15 |
| src/accelwattch/memoryctrl.cc | 메모리 컨트롤러 전력 모델 구현 | 2026-06-15 |
| src/accelwattch/noc.h | NoC(네트워크온칩) 전력 모델 헤더 | 2026-06-15 |
| src/accelwattch/noc.cc | NoC 라우터/링크 전력 모델 구현 | 2026-06-15 |
| src/accelwattch/processor.h | 최상위 Processor 전력/면적 집계 헤더 | 2026-06-15 |
| src/accelwattch/processor.cc | 최상위 Processor 전력 모델 구현 | 2026-06-15 |
| src/accelwattch/sharedcache.h | 공유 캐시(L2/L3/Directory) 전력 모델 헤더 | 2026-06-15 |
| src/accelwattch/sharedcache.cc | 공유 캐시 전력 모델 구현 | 2026-06-15 |
| src/accelwattch/version.h | McPAT 버전 정보 헤더 | 2026-06-15 |
| libopencl/opencl_runtime_api.cc | OpenCL 런타임 API 구현 전체 (GPGPUSim_Init, _cl_program Build/CreateKernel/get_ptx, _cl_kernel bind_args/get_workgroup_size, _cl_mem/ _cl_context/ _cl_command_queue/ _cl_device_id 구조체, clEnqueueNDRangeKernel 핵심 OpenCL→CUDA 변환, clGetDeviceInfo/ProgramInfo/KernelWorkGroupInfo 등 전체 함수 + 인라인 주석) | 2026-06-14 |
| libopencl/nvopencl_wrapper.cc | OpenCL 소스→PTX 변환 래퍼 실행 파일 전체 (vmyexit/myexit/main, NVIDIA 플랫폼 탐색/컨텍스트 생성/컴파일/바이너리 추출/PTX 파일 기록, 전체 함수 + 인라인 주석) | 2026-06-14 |
| libcuda/cuda_api_object.h | CUDA API 객체 헤더 전체 (glbmap_entry, _cuda_device_id, CUctx_st, kernel_config, cuda_runtime_api 클래스 및 모든 필드/메서드 주석) | 2026-06-14 |
| libcuda/gpgpu_context.h | GPGPU-Sim 전역 컨텍스트 헤더 전체 (gpgpu_context 생성자, UID 카운터, 서브시스템 포인터, PTX/시뮬레이션 진입 함수 주석) | 2026-06-14 |
| libcuda/cuobjdump.h | cuobjdump 파싱 자료구조 헤더 전체 (cuobjdump_parser, cuobjdumpSection, cuobjdumpELFSection, cuobjdumpPTXSection 클래스 및 필드/메서드 주석) | 2026-06-14 |
| libcuda/cuobjdump.l | cuobjdump 출력용 Flex/Lex 어휘 분석기 전체 (스타트 조건, 토큰 규칙, PTX/ELF/SASS 본문 분리, cuobjdump_error 주석) | 2026-06-14 |
| libcuda/cuobjdump.y | cuobjdump 출력용 Bison/Yacc 문법 파서 전체 (section/headerinfo/identifier/ptxcode/elfcode/sasscode 규칙 및 파일 추출 액션 주석) | 2026-06-14 |
| libcuda/cuda_api.h | CUDA 드라이버 API 헤더 전체 (버전 매크로/ABI 이름 리매핑, CUdevice/CUcontext/CUmodule 등 핸들과 enum/struct 타입 정의, 모든 CUresult CUDAAPI 함수 선언에 한국어 주석) | 2026-06-14 |
| libcuda/cuda_runtime_api.cc | CUDA 런타임 API 인터셉트 구현 전체 (GPGPUSim_Init/Context, fat binary/커널/변수 등록, 납부 Internal API, public CUDA/Driver API 래퍼, cuobjdump 추출/파싱, SST 통합 — 함수 및 주요 라인 한국어 주석) | 2026-06-14 |
| aerialvision/configs.py | AerialVision 사용자 설정(config.rc) 로더 (AerialVisionConfig 클래스, get_value, avconfig 전역 인스턴스, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/variableclasses.py | AerialVision 변수/라인 통계 메타데이터 클래스 (variable, bookmark, cudaLineNo, ptxLineNo, loadLineStatName, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/lexyaccbookmark.py | AerialVision 즐겨찾기(bookmarks.txt) PLY Lex/Yacc 파서 전체 (토큰/문법/파싱/기록, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/lexyacctexteditor.py | AerialVision Source Code View용 .stats/.ptx 파서 (textEditorParseMe, ptxToCudaMapping, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/lexyacc.py | GPGPU-Sim 로그 Lex/Yacc 파서 전체 (parseMe, skipCFLOGParsing, 사용자 정의 변수 로드, CFLOG 파싱, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/organizedata.py | AerialVision 파싱 데이터 재배열 모듈 (OrganizeScalar, nullOrganized*, CFLOGOrganize*, organizedata, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/startup.py | AerialVision 메인 GUI 진입점 (fileInput, submitClicked, startup, graphAddTab, manageFiles, textAddTab, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| aerialvision/guiclasses.py | AerialVision GUI/플로팅 클래스 전체 (formEntry, subplotInstance, PlotFormatInfo, graphManager, NaviPlotInfo, newTextTab, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/cuobjdumpInst.h | SASS 단일 명령어 표현 클래스 헤더 (cuobjdumpInst, m_label/m_predicate/m_base/m_baseModifiers/m_typeModifiers/m_operands/m_predicateModifiers, printCuobjdumpPtxPlus/Operand 선언, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/cuobjdumpInst.cc | SASS → PTXPlus 변환 구현 (printCuobjdumpPtxPlus 분기, 산술/논리/메모리/분기/텍스처/원子 명령어 변환, printCuobjdumpOperand/TypeModifiers/BaseModifiers, 상단 4섹션 블록 + 모든 함수/분기 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/cuobjdumpInstList.h | SASS → PTXPlus IR 컨테이너 헤더 (constMemory/constMemory2/constMemoryPtr/globalMemory/localMemory/cuobjdumpEntry/cuobjdumpInstList, 상단 4섹션 블록 + 모든 구조체/필드/메서드 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/cuobjdumpInstList.cc | SASS → PTXPlus IR 컨테이너 구현 (addEntry/add/register/memory/label, printMemory/RegNames/PredNames, printCuobjdumpPtxPlusList 매칭/출력, 상단 4섹션 블록 + 모든 함수 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/cuobjdump_to_ptxplus.cc | cuobjdump_to_ptxplus 독립 실행형 main() (ptx/sass/elf 입력 → ptxplus 출력, elf_parse/ptx_parse/sass_parse 호출, output/fileToString, 상단 4섹션 블록 + 모든 함수 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/ptx_parser.h | 원본 PTX 헤더 파서용 시맨틱 액션 (더미 함수군, add_version_info/target_header/func_header/add_space_spec/add_scalar_type_spec, g_headerList 조작, 상단 4섹션 블록 + 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/ptx.l | PTX(BIN) Flex/Lex 어휘 분석기 (PTX 키워드/메모리공간/타입/벡터/레지스터/특수레지스터/구두점 토큰, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/ptx.y | PTX(BIN) Bison/Yacc 문법 파서 (version/target/address_size/file/declaration/entry/func/param 문법, ptx_parser.h 액션 호출, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/header.l | PTX 헤더 전용 Flex/Lex 어휘 분석기 (.version/.target/.entry/.param/타입 토큰, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/header.y | PTX 헤더 전용 Bison/Yacc 문법 파서 (program/version/target/entry/param/type 문법, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/elf.l | cuobjdump ELF 출력 Flex/Lex 어휘 분석기 (cmem/cmem14/symtab START 조건, 상수/전역/로컬 심볼 토큰, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/elf.y | cuobjdump ELF 출력 Bison/Yacc 문법 파서 (SYMTAB/constant0/constant1/constant14/local 메모리 문법, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/sass.l | SASS 디스어셈블리 Flex/Lex 어휘 분석기 (SASS 명령어/수정자/레지스터/메모리위치/프레디케이트/라벨 토큰, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |
| cuobjdump_to_ptxplus/sass.y | SASS 디스어셈블리 Bison/Yacc 문법 파서 (Function/entry/statement/label/branch/memory/operand 문법, IR 구축, 상단 4섹션 블록 + 규칙별 한국어 주석) | 2026-06-14 |

### 미완료 (우선순위 순)

| 디렉토리 | 설명 | 우선순위 |
|---------|------|---------|
| (없음) | Wave 1~5 병렬 에이전트 작업으로 모든 주석 대상 파일이 완료되었습니다. | - |

## 빌드 방법

```bash
source setup_environment
make -j$(nproc)
```

CUDA Toolkit이 설치되어 있어야 하며, `CUDA_INSTALL_PATH` 환경변수가 설정되어야 한다.

## 참고

- Accel-Sim / GPGPU-Sim 4.0 논문: ISCA 2020
- AccelWattch 논문: MICRO 2021
- 공식 사이트: https://accel-sim.github.io/
- CUDA PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA SASS 레퍼런스: NVBit / cuobjdump
