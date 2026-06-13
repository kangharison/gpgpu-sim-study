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
- `cuobjdump_to_ptxplus/` 자동 변환 툴 — 주요 함수만
- `bin/`, `build/` 자동 생성/빌드 산출물
- `doc/doxygen/` 등 자동 생성 문서
- `.l`, `.y` 파일 자체의 생성된 C 출력물

## 주석 작업 진행 현황

### 완료 (20 파일)

| 파일 | 설명 | 완료일 |
|------|------|--------|
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

### 미완료 (우선순위 순)

| 디렉토리 | 설명 | 우선순위 |
|---------|------|---------|
| src/gpgpu-sim/shader.* | SM 파이프라인 (핵심) | P0 |
| src/gpgpu-sim/gpu-sim.* | cycle-by-cycle 시뮬레이션 루프 | P0 |
| src/gpgpu-sim/gpu-cache.* | 캐시 계층 | P0 |
| src/cuda-sim/ptx_ir.*, instructions.cc | PTX IR 및 명령어 시맨틱 | P1 |
| src/cuda-sim/cuda-math.h | CUDA 수학 에뮬레이션 | P1 |
| src/intersim2/ | NoC 시뮬레이터 | P2 |
| src/accelwattch/ | 전력 모델 | P2 |
| libcuda/, libopencl/ | 런타임 인터셉트 | P2 |
| src/gpgpusim_entrypoint.*, stream_manager.* | 진입점/스트림 관리 | P1 |

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
