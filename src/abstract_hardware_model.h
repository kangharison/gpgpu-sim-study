/*
 * [한국어 설명] GPGPU-Sim 핵심 하드웨어 추상 모델 헤더 (abstract_hardware_model.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍/기능 시뮬레이터 전반에서 공유되는 GPU 하드웨어 추상 자료구조와
 * 인터페이스를 정의하는 핵심 헤더이다. warp(32개 스레드 SIMT 실행 단위), SIMT 스택,
 * 명령어(inst_t/warp_inst_t), 메모리 접근(mem_access_t), 커널 정보(kernel_info_t),
 * GPU 코어(core_t), 파이프라인 레지스터(register_set) 등 시뮬레이터가 GPU 하드웨어를
 * 모델링하는 데 필요한 모든 기반 클래스/열거형/구조체를 담고 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 계층 구조에서 이 파일은 "기반 추상 레이어"에 해당한다.
 * 호출 체인: CUDA 런타임(libcuda/) → gpgpusim_entrypoint.cc → cuda-sim/(PTX 기능 시뮬레이션)
 *            및 gpgpu-sim/(타이밍 시뮬레이션) → 이 파일의 자료구조를 공유 사용.
 * shader.cc(SM 파이프라인), gpu-sim.cc(cycle-by-cycle 루프), gpu-cache.cc(캐시),
 * mem_fetch.cc(메모리 패킷), scoreboard.cc(레지스터 해저드 추적) 등 거의 모든 타이밍 모델
 * 파일이 이 헤더를 직접 의존한다.
 * 실행 컨텍스트: 호스트 유저스페이스(시뮬레이터 메인 루프), 사이클-레벨 타이밍 모델.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 헤더: stream_manager.h(스트림 관리), vector_types.h(dim3),
 *   builtin_types.h(CUDA 타입), option_parser.h(설정 파싱)
 * - 이 파일에 의존하는 모듈: shader.{cc,h}, gpu-sim.{cc,h}, gpu-cache.{cc,h},
 *   mem_fetch.{cc,h}, scoreboard.{cc,h}, addrdec.{cc,h}, cuda-sim/ 전체
 * - 핵심 데이터 흐름: kernel_info_t(커널 정보) → core_t(코어 실행) →
 *   warp_inst_t(명령어 발행) → mem_access_t(메모리 요청) → mem_fetch(캐시/DRAM 전달)
 * - GPU 메모리 공간 레이아웃 상수(GLOBAL_HEAP_START 등)가 cuda-sim과 gpgpu-sim 양쪽에서
 *   동일하게 참조된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - kernel_info_t: 커널의 그리드/블록 크기, CTA/스레드 ID 관리, CDP 부모-자식 관계 추적
 * - simt_stack: 워프 분기 발산·재수렴을 포스트 도미네이터 방식으로 관리하는 스택
 * - inst_t / warp_inst_t: 개별 명령어 속성(inst_t)과 워프 수준 실행 정보(warp_inst_t)
 * - mem_access_t: 하나의 메모리 접근 요청(주소·크기·스레드 마스크·섹터 마스크 포함)
 * - core_t: GPU 코어(SM) 기능 시뮬레이션 기반 추상 클래스 (SIMT 스택·스레드·리덕션 관리)
 * - register_set: 파이프라인 단계 간 명령어 이동용 레지스터 세트 (서브코어 모델 지원)
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Vijay Kandiah,
// 저작권 표시: 이 코드를 만든 사람들의 이름입니다 (2009~2021년)
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// 이 코드를 만든 대학교들: 브리티시컬럼비아 대학, 노스웨스턴 대학, 퍼듀 대학
// All rights reserved.
// 모든 권리 보유 - 이 코드의 권리는 위 사람들/기관에게 있다는 뜻
//
// Redistribution and use in source and binary forms, with or without
// 재배포 및 사용은 소스 코드와 바이너리(실행 파일) 형태로, 수정 여부에 관계없이
// modification, are permitted provided that the following conditions are met:
// 다음 조건을 충족하면 허용됩니다:
//
// 1. Redistributions of source code must retain the above copyright notice,
// 1. 소스 코드를 재배포할 때는 위의 저작권 표시를 유지해야 합니다
// this
//    list of conditions and the following disclaimer;
//    이 조건 목록과 아래의 면책 조항도 함께요
// 2. Redistributions in binary form must reproduce the above copyright notice,
// 2. 바이너리(실행 파일) 형태로 재배포할 때는 저작권 표시를 문서에 포함해야 합니다
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
// 3. 대학교 이름을 허가 없이 홍보에 사용할 수 없습니다
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// 이 소프트웨어는 "있는 그대로" 제공됩니다 (보증 없음)
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 위 내용은 "BSD 라이선스"라는 오픈소스 라이선스입니다.
// 쉽게 말해, "이 코드를 자유롭게 쓸 수 있지만, 문제가 생겨도 우리 책임이 아닙니다"라는 뜻

/*
 * #ifndef / #define / #endif는 "인클루드 가드(include guard)"라고 합니다.
 * 같은 헤더 파일이 여러 번 포함(include)되는 것을 방지합니다.
 * 마치 "이미 읽은 책은 다시 안 읽는다"와 같은 원리입니다.
 */
#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED  // 이 파일이 아직 포함되지 않았다면
#define ABSTRACT_HARDWARE_MODEL_INCLUDED  // 이 파일이 포함되었다고 표시

/*
 * [한국어] 전방 선언(Forward declarations):
 * 이 헤더 자체가 아직 완전히 정의되지 않은 클래스들을 포인터/참조 형태로 사용하기 때문에
 * 컴파일러에게 "이 이름의 클래스가 존재한다"고 미리 알린다. 완전한 정의는 각 클래스의
 * 전용 헤더(gpu-sim.h, stream_manager.h 등)에 있다. 순환 의존성을 끊기 위한 표준 기법.
 */
// Forward declarations
class gpgpu_sim;
/* [한국어] GPU 전체 타이밍 시뮬레이터 최상위 클래스.
 * gpu-sim.h에 선언되며, gpgpu-sim/gpu-sim.cc에서 사이클-레벨 루프를 구현한다.
 * 이 헤더를 포함하는 거의 모든 파일이 gpgpu_sim 포인터를 통해 GPU 전역 상태
 * (클럭 카운터, 통계, SM 배열, 메모리 컨트롤러 등)에 접근한다.
 * 완전 정의 없이 포인터만 사용하므로 전방 선언으로 충분하다. */
class kernel_info_t;
/* [한국어] CUDA 커널 하나의 실행 정보(그리드/블록 크기, CTA/스레드 ID, 실행 상태)를
 * 담는 클래스. gpgpusim_entrypoint.cc에서 cuLaunchKernel() 인터셉트 시 생성되어
 * gpu-sim을 거쳐 각 SM(shader_core_ctx)에 전달된다.
 * 이 파일 내 core_t와 warp_inst_t가 kernel_info_t 포인터를 멤버로 갖는다. */
class gpgpu_context;
/* [한국어] GPGPU-Sim 시뮬레이터 전체의 전역 컨텍스트(싱글턴 유사 구조).
 * 설정 파싱 결과(gpgpu_sim_config), 통계 카운터, 전역 상태 등을 한 곳에서 관리한다.
 * 여러 시뮬레이션 인스턴스가 공존할 때 각각의 독립적 상태를 보장하기 위해 도입되었다. */

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
/* [한국어] SM(Streaming Multiprocessor) 하나에 동시에 상주할 수 있는
 * CTA(Cooperative Thread Array, CUDA 용어로 "블록")의 상한선.
 * 실제 CUDA HW 제한은 세대마다 다르지만(Fermi: 8, Kepler~Volta: 32) 시뮬레이터에서는
 * 32를 하드 리밋으로 설정해 최신 GPU까지 커버한다.
 * 사용처: core_t::reduction_storage[MAX_CTA_PER_SHADER][...] 배열 크기,
 *         shader_core_ctx의 CTA 슬롯 배열 크기 결정. */
#define MAX_BARRIERS_PER_CTA 16
/* [한국어] CTA 하나에서 동시에 사용할 수 있는 배리어(barrier)의 최대 개수.
 * PTX의 bar.sync 명령어는 0~15번 배리어 ID를 지정할 수 있다(PTX ISA 2.1 §8.7.12).
 * 16개 배리어를 넘으면 컴파일러가 에러를 내므로 이 값은 PTX 스펙에서 도출된 상수이다.
 * 사용처: core_t::reduction_storage[...][MAX_BARRIERS_PER_CTA] 배열의 두 번째 차원. */

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
/* [한국어] 하나의 명령어(warp_inst_t)가 읽을 수 있는 최대 입력 피연산자 수.
 * PTX 벡터 피연산자(예: .v4)가 스칼라 레지스터로 확장된 후의 최댓값이다.
 * 24는 4-원소 벡터 × 최대 6개 소스 피연산자를 고려한 여유 있는 상한선이다.
 * 사용처: inst_t::in[MAX_INPUT_VALUES] 배열 크기,
 *         scoreboard가 읽는 소스 레지스터 목록 크기. */
#define MAX_OUTPUT_VALUES 8
/* [한국어] 하나의 명령어(warp_inst_t)가 쓸 수 있는 최대 출력 피연산자 수.
 * 대부분의 명령어는 목적지 레지스터가 1~2개이고, 벡터 스토어의 경우도 최대 4개이므로
 * MAX_INPUT_VALUES(24)보다 작은 8로 충분하다.
 * 사용처: inst_t::out[MAX_OUTPUT_VALUES] 배열 크기,
 *         scoreboard가 추적하는 목적지 레지스터 목록 크기. */

/*
 * [한국어] _memory_space_t: GPU의 다양한 메모리 공간(종류)을 열거하는 타입.
 * PTX ISA에서 각 메모리 명령어는 .space 한정자로 어느 메모리 공간에 접근하는지 명시한다
 * (예: ld.global, st.shared, ld.const 등). 시뮬레이터는 이 enum으로 명령어가
 * 접근하는 메모리 공간을 분류하고, 해당 공간의 레이턴시/캐시 경로를 결정한다.
 * 메모리 합치기(coalescing), 캐시 태그 계산, mem_fetch 타입 분류 등에 두루 사용된다.
 */
enum _memory_space_t {
  undefined_space = 0,
  /* [한국어] 메모리 공간이 아직 결정되지 않은 초기 상태.
   * PTX 파싱/디코딩 전 또는 메모리 접근이 없는 명령어의 경우 이 값으로 초기화된다.
   * 설정자: inst_t 기본 생성자 및 ptx_ir.cc 파서 초기화 경로.
   * 읽는 자: 메모리 합치기(coalescing) 모듈이 공간 타입을 검증할 때.
   * 값이 undefined_space로 남아 있으면 메모리 접근 처리 경로에서 assert 실패로 잡힌다. */
  reg_space,
  /* [한국어] 레지스터 파일 공간: GPU에서 가장 빠른 저장소(SM 내부 레지스터 파일).
   * PTX 명령어에서 .reg 선언된 변수가 이 공간에 할당된다.
   * 레지스터 접근은 메모리 계층(L1/L2/DRAM)을 거치지 않아 1사이클 레이턴시.
   * 설정자: ptx_ir.cc의 operand_info 파싱 단계.
   * 읽는 자: operand_collector(레지스터 뱅크 중재 단계)와 scoreboard(RAW 해저드 추적). */
  local_space,
  /* [한국어] 로컬 메모리 공간: 각 스레드 전용 스택/자동 배열 저장소.
   * GPU DRAM의 스레드별 전용 파티션으로, 물리적으로는 글로벌 메모리와 같은 DRAM에 위치해
   * 접근 레이턴시가 글로벌 메모리와 동일하게 느리다.
   * 주소 변환: LOCAL_GENERIC_START + (sm_id * MAX_THREAD + thread_id) * LOCAL_MEM_SIZE_MAX + offset.
   * 설정자: 함수 지역 배열 선언(.local) 또는 레지스터 스필(spill).
   * 읽는 자: 메모리 합치기 시 LOCAL_ACC_R/LOCAL_ACC_W 타입으로 분류되어 L1 캐시 경유. */
  shared_space,
  /* [한국어] 공유 메모리(Shared Memory) 공간: 같은 CTA 내 스레드들이 공유하는 온칩 메모리.
   * L1 캐시와 같은 SRAM 물리 자원을 나눠 쓰며(configurable split), 최대 48~96KB(아키텍처마다 다름).
   * 뱅크 충돌(bank conflict) 시 직렬화되므로 gpgpusim.config의 shared_mem_bank_size 설정이 중요.
   * 설정자: PTX .shared 변수 선언 또는 동적 공유 메모리(extern __shared__).
   * 읽는 자: shader_core_ctx의 공유 메모리 접근 처리 루틴. */
  sstarr_space,
  /* [한국어] 공유 메모리 배열(sstarr) 공간: PTXPlus에서 사용되는 특수 공유 메모리 타입.
   * 일반 shared_space와 달리 배열 접근 패턴에 최적화된 별도 경로를 사용한다.
   * GPGPU-Sim의 PTXPlus 확장(cuobjdump_to_ptxplus 변환 결과)에서만 등장한다.
   * 설정자: PTXPlus 파서. 읽는 자: 공유 메모리 접근 분류 로직. */
  param_space_unclassified,
  /* [한국어] 분류되지 않은 파라미터 공간: PTX 파싱 초기 단계에서 커널/로컬 파라미터를
   * 구분하기 전의 임시 상태. 심볼 테이블 구축 중 사용되며, 이후 반드시
   * param_space_kernel 또는 param_space_local 중 하나로 확정된다.
   * 설정자: ptx_ir.cc 파서 초기화 단계. 읽는 자: 파싱 후 타입 검증 단계. */
  param_space_kernel,  /* global to all threads in a kernel : read-only */
  /* [한국어] 커널 파라미터 공간: cuLaunchKernel()로 전달된 커널 인자가 저장되는 공간.
   * 모든 스레드가 읽을 수 있고 쓸 수 없다(읽기 전용). 상수 캐시(constant cache)로 캐시된다.
   * 브로드캐스트 접근 패턴이 일반적이므로 상수 캐시를 통해 빠르게 서빙된다.
   * 설정자: gpgpusim_entrypoint.cc의 cuLaunchKernel() 인터셉트 경로.
   * 읽는 자: PTX 명령어 실행 시 ld.param.kernel 경로. */
  param_space_local,  /* local to a thread : read-writable */
  /* [한국어] 로컬 파라미터 공간: 디바이스 함수 호출 시 전달되는 인자 공간.
   * 각 스레드가 독립적으로 읽고 쓸 수 있으며, 로컬 메모리처럼 스레드 전용이다.
   * 디바이스 함수 호출(call 명령어) 시 인자 전달에 사용되고,
   * 함수 반환 후에는 유효하지 않으므로 수명 범위(scope)가 제한적이다.
   * 설정자: call 명령어 실행 시 인자 복사. 읽는 자: callee 함수 내 ld.param.local. */
  const_space,
  /* [한국어] 상수 메모리 공간(.const): GPU 전체에서 읽기 전용으로 접근되는 특수 메모리.
   * 전용 상수 캐시(constant cache, 일반적으로 8KB)로 캐시되어, 모든 스레드가 같은 주소를
   * 읽을 때 단 1회의 캐시 접근으로 브로드캐스트된다. 서로 다른 주소를 읽으면 직렬화됨.
   * PTX에서 .const[n] 형식으로 뱅크 번호(n, 0~10)를 지정할 수 있다(PTX ISA §5.1.3).
   * 설정자: 호스트에서 cuMemcpyToSymbol() 또는 컴파일러가 정적 상수 배열 배치.
   * 읽는 자: shader_core_ctx의 상수 캐시 접근 경로. */
  tex_space,
  /* [한국어] 텍스처 메모리 공간: 2D 공간 지역성을 활용하는 전용 캐시 경로.
   * 이미지/행렬 데이터 접근에 최적화된 텍스처 캐시(L1 텍스처, 일반적으로 SM당 독립)를 사용.
   * mem_fetch 생성 시 TEXTURE_ACC_R 타입으로 분류되어 텍스처 캐시 경로를 탄다.
   * 설정자: PTX tex.* 명령어 실행 경로. 읽는 자: gpu-cache.cc의 텍스처 캐시 처리. */
  surf_space,
  /* [한국어] 서피스 메모리 공간: 텍스처와 유사하지만 읽기·쓰기 모두 지원하는 공간.
   * CUDA Surface API(cudaBindSurfaceToArray 등)를 통해 접근되며, surf.load/surf.store 명령어로 사용.
   * GPGPU-Sim에서는 tex_space와 유사한 경로로 처리되지만 쓰기 경로가 추가된다.
   * 설정자: PTX suld/sust 명령어 파싱 경로. 읽는 자: 서피스 메모리 접근 처리 루틴. */
  global_space,
  /* [한국어] 글로벌 메모리 공간: GPU DRAM에 위치하는 가장 크고 가장 느린 메모리.
   * 모든 스레드(및 모든 SM)에서 접근 가능하며, L1→L2→DRAM 계층을 거쳐 접근된다.
   * mem_fetch 생성 시 GLOBAL_ACC_R 또는 GLOBAL_ACC_W 타입으로 분류된다.
   * 메모리 합치기(coalescing)의 주 대상이며, 동일 워프 내 32개 스레드의 접근이
   * 연속적(coalesced)이면 단 1개의 트랜잭션으로 처리되어 성능이 크게 향상된다.
   * 설정자: PTX ld.global/st.global 명령어. 읽는 자: L1 캐시 → L2 캐시 → DRAM 경로. */
  generic_space,
  /* [한국어] 제네릭 주소 공간: 컴파일 시점에 메모리 종류가 결정되지 않는 포인터.
   * 런타임에 주소 범위(SHARED_GENERIC_START, LOCAL_GENERIC_START, GLOBAL_HEAP_START)를
   * 비교하여 실제 공간(공유/로컬/글로벌)으로 해석된다. 디바이스 함수에 포인터를 전달할 때
   * 자주 등장하며, abstract_hardware_model.h에 정의된 주소 범위 상수를 기반으로 분류된다.
   * 설정자: PTX ld/st 명령어(공간 한정자 없음). 읽는 자: coalescing 모듈의 주소 분류기. */
  instruction_space
  /* [한국어] 명령어 공간: GPU 프로그램의 명령어(PTX/SASS 코드)가 저장되는 공간.
   * INST_ACC_R 타입의 메모리 접근으로 처리되며, 명령어 캐시(I-Cache)를 통해 페치된다.
   * shader.cc의 fetch 단계에서 이 공간에 대한 접근이 발생하며, I-Cache 미스 시
   * 메모리 계층을 타고 올라간다(L1 I-Cache → L2 → DRAM).
   * 설정자: 커널 로드 시 PTX/SASS 바이너리 배치. 읽는 자: fetch 단계의 I-Cache 경로. */
};

/*
 * [한국어] COEFF_STRUCT 인클루드 가드:
 * PowerscalingCoefficients 구조체가 accelwattch/ 헤더와 이 파일 양쪽에서 정의될 수 있어
 * 중복 정의를 방지하기 위해 별도의 가드 매크로를 사용한다.
 */
#ifndef COEFF_STRUCT
#define COEFF_STRUCT

/*
 * [한국어] PowerscalingCoefficients: AccelWattch 전력 모델에서 사용하는 연산별 전력 스케일링 계수 구조체.
 * 각 필드는 "사이클당 해당 연산 횟수 × 계수 = 동적 전력 기여분(W)"의 형태로 사용된다.
 * 계수 값은 McPAT(accelwattch/mcpat/)로 추출하거나 gpgpusim.config의 파라미터로 주어진다.
 * AccelWattch가 각 사이클의 전력을 추산하기 위해 shader_core_ctx의 연산 카운터에 곱한다.
 * 동기화: 전력 계산은 시뮬레이션 단일 스레드에서 순차적으로 수행되므로 별도 락 불필요.
 */
struct PowerscalingCoefficients {
  double int_coeff;
  /* [한국어] 정수(Integer) ALU 일반 연산의 전력 스케일링 계수.
   * AccelWattch 전력 모델(accelwattch/)에서 사이클당 정수 ALU 연산 횟수에 곱해져
   * 동적 전력(dynamic power)을 추정한다.
   * 설정자: accelwattch/ 내 McPAT 추출 결과 또는 gpgpusim.config 파라미터 파싱.
   * 읽는 자: accelwattch의 전력 계산 루프(각 사이클 종료 후 호출).
   * 값 범위: 양수 실수(GPU 모델마다 다르며, 일반적으로 수십~수백 mW 단위 기여). */
  double int_mul_coeff;
  /* [한국어] 정수 곱셈(IMUL) 연산의 전력 계수.
   * 정수 덧셈보다 회로 복잡도가 높아 전력 소모가 크며, int_coeff와 별도로 추적한다.
   * special_operations_t::INT_MUL_OP 타입의 연산 카운터에 곱해진다.
   * 설정자·읽는 자·값 범위: int_coeff와 동일한 패턴. */
  double int_mul24_coeff;
  /* [한국어] 24비트 정수 곱셈(IMUL24)의 전력 계수.
   * PTX의 mul24.lo/mul24.hi 명령어에 해당하며, 32비트 곱셈보다 회로가 단순해 전력이 낮다.
   * special_operations_t::INT_MUL24_OP 카운터에 곱해진다.
   * Fermi 이전 GPU에서 자주 사용되었으나 최신 아키텍처에서는 mul.lo와 동일하게 처리됨. */
  double int_mul32_coeff;
  /* [한국어] 32비트 정수 곱셈(IMUL32)의 전력 계수.
   * int_mul_coeff와 별도로 추적하는 이유: 일부 PTX 명령어는 명시적으로 mul.wide.s32 등
   * 32비트 곱셈을 지정하므로 전력 기여를 세분화하기 위해 분리한다.
   * special_operations_t::INT_MUL32_OP 카운터에 곱해진다. */
  double int_div_coeff;
  /* [한국어] 정수 나눗셈(IDIV)의 전력 계수.
   * 나눗셈은 곱셈보다 훨씬 복잡한 회로를 사용하며, GPU에서 SFU를 통해 처리된다.
   * special_operations_t::INT_DIV_OP 카운터에 곱해진다.
   * 값이 클수록 나눗셈 집중 워크로드에서 전력 추정값이 높아진다. */
  double fp_coeff;
  /* [한국어] 단정밀도 부동소수점(FP32, float) 일반 ALU 연산의 전력 계수.
   * 덧셈·뺄셈·비교 등 FP32 기본 연산에 적용된다.
   * special_operations_t::FP__OP 카운터에 곱해진다.
   * SM 내 CUDA 코어(FP32 ALU)의 동적 전력을 추정하는 핵심 계수. */
  double dp_coeff;
  /* [한국어] 배정밀도 부동소수점(FP64, double) 일반 연산의 전력 계수.
   * FP64는 FP32보다 약 2배의 회로 자원을 사용하며, 전력 소모도 그에 비례해 크다.
   * special_operations_t::DP___OP 카운터에 곱해진다.
   * GPU 모델에 따라 FP64 유닛 수가 크게 다르므로 이 계수도 모델마다 상이하다. */
  double fp_mul_coeff;
  /* [한국어] 단정밀도 부동소수점 곱셈(FMUL)의 전력 계수.
   * FMUL은 FADD보다 MAC(Multiply-Accumulate) 구조로 처리되며 전력 기여가 다르다.
   * special_operations_t::FP_MUL_OP 카운터에 곱해진다.
   * FMAD(Fused Multiply-Add)는 별도로 계수화되지 않고 이 값으로 근사한다. */
  double fp_div_coeff;
  /* [한국어] 단정밀도 부동소수점 나눗셈(FDIV)의 전력 계수.
   * GPU에서 FDIV는 SFU(Special Function Unit)를 통해 역수(rcp) 후 곱셈으로 구현된다.
   * special_operations_t::FP_DIV_OP 카운터에 곱해진다. */
  double dp_mul_coeff;
  /* [한국어] 배정밀도 부동소수점 곱셈(DMUL)의 전력 계수.
   * FP64 전용 곱셈 유닛을 사용하며 fp_mul_coeff보다 높은 값을 가진다.
   * special_operations_t::DP_MUL_OP 카운터에 곱해진다. */
  double dp_div_coeff;
  /* [한국어] 배정밀도 부동소수점 나눗셈(DDIV)의 전력 계수.
   * FP64 역수 계산 후 곱셈으로 구현되며 가장 전력 소모가 큰 FP 연산 중 하나.
   * special_operations_t::DP_DIV_OP 카운터에 곱해진다. */
  double sqrt_coeff;
  /* [한국어] 제곱근(SQRT, 루트) 연산의 전력 계수.
   * GPU에서 SFU(Special Function Unit)가 SQRT를 처리하며, FP32 ALU보다 전력이 크다.
   * special_operations_t::FP_SQRT_OP 카운터에 곱해진다.
   * rsqrt(역제곱근) 후 역수를 취하는 방식으로 구현될 수 있다. */
  double log_coeff;
  /* [한국어] 로그(LG2, 자연 로그 포함) 연산의 전력 계수.
   * SFU가 처리하는 초월 함수 중 하나이며, 테이블 룩업 + 보간 방식으로 구현된다.
   * special_operations_t::FP_LG_OP 카운터에 곱해진다. */
  double sin_coeff;
  /* [한국어] 삼각함수(SIN, COS 포함) 연산의 전력 계수.
   * SFU가 처리하며, PTX의 sin.approx.f32/cos.approx.f32에 해당한다.
   * special_operations_t::FP_SIN_OP 카운터에 곱해진다.
   * 근사 정밀도는 약 2 ulp 오차 이내(PTX ISA §9.7.3). */
  double exp_coeff;
  /* [한국어] 지수 함수(EX2, e^x 포함) 연산의 전력 계수.
   * SFU가 처리하며, PTX의 ex2.approx.f32에 해당한다(2의 거듭제곱 근사).
   * special_operations_t::FP_EXP_OP 카운터에 곱해진다. */
  double tensor_coeff;
  /* [한국어] 텐서 코어(Tensor Core) 연산의 전력 계수.
   * Volta 이상 GPU에 탑재된 텐서 코어는 4×4 행렬 곱셈을 1사이클에 수행하며
   * 일반 FP32 ALU보다 훨씬 높은 전력을 소비한다.
   * special_operations_t::TENSOR__OP 카운터에 곱해진다.
   * 딥러닝 워크로드에서 전체 전력의 상당 부분을 차지할 수 있다. */
  double tex_coeff;
  /* [한국어] 텍스처 유닛 접근 연산의 전력 계수.
   * 텍스처 캐시 접근 및 샘플링 하드웨어의 동적 전력을 추정한다.
   * special_operations_t::TEX__OP 카운터에 곱해진다.
   * 이미지 처리 워크로드에서 의미 있는 전력 기여를 한다. */
};
#endif  // COEFF_STRUCT 인클루드 가드 끝

/*
 * [한국어] FuncCache: CUDA 커널별 L1 캐시와 공유 메모리 간 분배 방식 선택 enum.
 * Fermi/Kepler 이후 GPU는 SM 내 온칩 SRAM(예: 64KB)을 L1 캐시와 공유 메모리로
 * 동적으로 분배할 수 있다. cudaFuncSetCacheConfig() API 또는 PTX .pragma "nounroll" 등으로
 * 커널별로 설정 가능하다. gpgpusim.config의 캐시 크기 파라미터와 연동된다.
 */
enum FuncCache {
  FuncCachePreferNone = 0,
  /* [한국어] 캐시 분배 선호 없음(기본값): 런타임이 적절한 분배를 결정한다.
   * 설정자: cudaFuncSetCacheConfig() 호출 없이 커널을 실행하는 경우.
   * 읽는 자: shader_core_ctx의 캐시 설정 초기화 경로.
   * GPGPU-Sim에서는 gpgpusim.config의 기본 캐시 설정이 그대로 적용된다. */
  FuncCachePreferShared = 1,
  /* [한국어] 공유 메모리 선호: L1 캐시보다 공유 메모리에 더 많은 SRAM을 할당한다.
   * 예시: 48KB 공유 메모리 + 16KB L1 캐시 분할(Fermi 기준).
   * 공유 메모리를 많이 사용하는 커널(예: 행렬 곱셈 타일링)에서 선택한다.
   * 읽는 자: shader_core_ctx가 캐시 파티션 크기를 결정할 때 이 값을 확인. */
  FuncCachePreferL1 = 2
  /* [한국어] L1 캐시 선호: 공유 메모리보다 L1 캐시에 더 많은 SRAM을 할당한다.
   * 예시: 16KB 공유 메모리 + 48KB L1 캐시 분할(Fermi 기준).
   * 공유 메모리 사용이 적고 글로벌 메모리 접근이 많은 커널에서 효과적이다.
   * Kepler 이후에는 L1 캐시가 기본적으로 비활성화되는 경우도 있어 효과가 제한적. */
};

/*
 * [한국어] AdaptiveCache: 캐시 크기를 고정으로 유지할지, 워크로드에 따라 적응적으로 조절할지 선택.
 * ADAPTIVE_CACHE 모드에서는 시뮬레이터가 캐시 미스율 등 런타임 통계를 바탕으로
 * L1/공유 메모리 분배를 동적으로 변경할 수 있다(연구용 확장 기능).
 * 설정자: gpgpusim.config의 adaptive_cache 파라미터.
 * 읽는 자: shader_core_ctx의 캐시 크기 조정 루틴.
 */
enum AdaptiveCache {
  FIXED = 0,
  /* [한국어] 고정 캐시 크기: 시뮬레이션 전체에 걸쳐 캐시 분배를 변경하지 않는다.
   * 기본값이며, FuncCache 설정에 따른 고정 분배가 유지된다.
   * 재현성이 중요한 실험에서는 FIXED를 사용한다. */
  ADAPTIVE_CACHE = 1
  /* [한국어] 적응형 캐시 크기: 실행 중 워크로드 특성에 따라 캐시 분배를 동적으로 조절한다.
   * 연구 목적의 캐시 최적화 실험에 사용되며, 기본 설정에서는 비활성화된다.
   * 읽는 자: 캐시 파티션 동적 조정 로직(shader_core_ctx의 캐시 재설정 경로). */
};

/*
 * #ifdef __cplusplus: C++ 컴파일러로 컴파일할 때만 아래 코드를 포함합니다.
 * C와 C++은 비슷하지만 다른 프로그래밍 언어이고,
 * 이 부분은 C++에서만 사용 가능한 기능(클래스 등)을 포함합니다.
 */
#ifdef __cplusplus

#include <stdio.h>    // 화면에 출력하거나 파일을 읽고 쓰는 기능 (printf, fprintf 등)
#include <string.h>   // 문자열(텍스트) 처리 기능 (memset, strcmp 등)
#include <set>        // std::set 컨테이너: 중복 없이 값을 정렬해서 저장하는 자료구조

/*
 * typedef는 "타입에 별명을 붙이는 것"입니다.
 * unsigned long long은 아주 큰 양수를 저장할 수 있는 숫자 타입입니다 (최소 64비트).
 * 메모리 주소를 나타낼 때 사용합니다.
 */
typedef unsigned long long new_addr_type;        // 새로운 주소 타입: 메모리 주소를 나타냄
typedef unsigned long long cudaTextureObject_t;  // CUDA 텍스처 객체 식별자
typedef unsigned long long address_type;         // 주소 타입: 프로그램 카운터(PC) 등에 사용
typedef unsigned long long addr_t;               // 주소 타입 (짧은 이름)

/*
 * 타이밍 모델(시간 시뮬레이션)에서 볼 수 있는 연산(operation) 종류들
 * SPECIALIZED_UNIT_NUM: 특수 실행 유닛의 개수 (8개)
 * SPEC_UNIT_START_ID: 특수 유닛 ID가 시작하는 번호 (100번부터)
 */
// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8     // 특수 연산 유닛 최대 8개
#define SPEC_UNIT_START_ID 100     // 특수 유닛의 시작 ID 번호

/*
 * uarch_op_t: 마이크로아키텍처(uarch) 연산 타입
 * "마이크로아키텍처"란 CPU/GPU 내부의 실제 하드웨어 구조를 의미합니다.
 *
 * GPU가 실행할 수 있는 모든 종류의 명령어(연산)를 나열한 것입니다.
 * 마치 계산기의 버튼 종류(+, -, x, ÷ 등)를 나열한 것과 비슷합니다.
 */
enum uarch_op_t {
  NO_OP = -1,              // 연산 없음: 아무것도 하지 않는 명령어
  ALU_OP = 1,              // ALU 연산: 산술논리장치(더하기, 빼기, 비교 등 기본 계산)
  SFU_OP,                  // SFU 연산: 특수함수장치(sin, cos, 제곱근 등 복잡한 수학 계산)
  TENSOR_CORE_OP,          // 텐서 코어 연산: 행렬 곱셈 등 AI/딥러닝에 사용되는 특수 연산
  DP_OP,                   // 배정밀도(Double Precision) 연산: 64비트 실수 계산
  SP_OP,                   // 단정밀도(Single Precision) 연산: 32비트 실수 계산
  INTP_OP,                 // 정수(Integer) 연산: 정수 계산
  ALU_SFU_OP,              // ALU와 SFU를 함께 사용하는 연산
  LOAD_OP,                 // 로드 연산: 메모리에서 데이터를 읽어오는 것 (메모리 → 레지스터)
  TENSOR_CORE_LOAD_OP,     // 텐서 코어 로드: 텐서 코어 연산을 위해 데이터를 읽어옴
  TENSOR_CORE_STORE_OP,    // 텐서 코어 스토어: 텐서 코어 연산 결과를 메모리에 저장
  STORE_OP,                // 스토어 연산: 데이터를 메모리에 저장하는 것 (레지스터 → 메모리)
  BRANCH_OP,               // 분기 연산: 조건에 따라 다른 코드로 점프 (if/else와 비슷)
  BARRIER_OP,              // 배리어 연산: 스레드들이 서로 기다리는 동기화 명령
  MEMORY_BARRIER_OP,       // 메모리 배리어: 메모리 접근 순서를 보장하는 명령
  CALL_OPS,                // 함수 호출 연산: 다른 함수를 부르는 명령
  RET_OPS,                 // 리턴 연산: 함수에서 돌아오는 명령
  EXIT_OPS,                // 종료 연산: 스레드가 실행을 끝내는 명령
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,  // 특수 유닛 1 연산 (ID: 100)
  SPECIALIZED_UNIT_2_OP,   // 특수 유닛 2 연산 (ID: 101)
  SPECIALIZED_UNIT_3_OP,   // 특수 유닛 3 연산 (ID: 102)
  SPECIALIZED_UNIT_4_OP,   // 특수 유닛 4 연산 (ID: 103)
  SPECIALIZED_UNIT_5_OP,   // 특수 유닛 5 연산 (ID: 104)
  SPECIALIZED_UNIT_6_OP,   // 특수 유닛 6 연산 (ID: 105)
  SPECIALIZED_UNIT_7_OP,   // 특수 유닛 7 연산 (ID: 106)
  SPECIALIZED_UNIT_8_OP    // 특수 유닛 8 연산 (ID: 107)
};
typedef enum uarch_op_t op_type;  // uarch_op_t에 "op_type"이라는 짧은 별명을 붙임

/*
 * uarch_bar_t: 배리어(barrier) 종류
 * 배리어란 "여기서 다 같이 기다려!"라는 동기화 명령입니다.
 * 학교에서 모든 학생이 도착할 때까지 출발하지 않는 것과 비슷합니다.
 */
enum uarch_bar_t {
  NOT_BAR = -1,  // 배리어가 아님
  SYNC = 1,      // 동기화(SYNC): 모든 스레드가 이 지점에 도달할 때까지 기다림
  ARRIVE,        // 도착(ARRIVE): "나는 도착했어"라고 알리지만 기다리지는 않음
  RED            // 리덕션(Reduction): 여러 스레드의 값을 하나로 합치면서 동기화
};
typedef enum uarch_bar_t barrier_type;  // "barrier_type"이라는 별명

/*
 * uarch_red_t: 리덕션(reduction) 연산 종류
 * 리덕션이란 여러 값을 하나의 값으로 줄이는 연산입니다.
 * 예: 100명의 점수를 모두 더해서 합계를 구하는 것
 */
enum uarch_red_t {
  NOT_RED = -1,    // 리덕션이 아님
  POPC_RED = 1,    // 팝카운트(POPC) 리덕션: 1인 비트의 개수를 셈
  AND_RED,         // AND 리덕션: 모든 값을 AND(논리곱) 연산으로 합침
  OR_RED           // OR 리덕션: 모든 값을 OR(논리합) 연산으로 합침
};
typedef enum uarch_red_t reduction_type;  // "reduction_type"이라는 별명

/*
 * uarch_operand_type_t: 피연산자(operand)의 데이터 타입
 * 피연산자란 연산에 사용되는 값입니다. 예: "3 + 5"에서 3과 5가 피연산자
 */
enum uarch_operand_type_t {
  UN_OP = -1,   // 알 수 없는 타입
  INT_OP,       // 정수(Integer) 타입 연산
  FP_OP         // 부동소수점(Floating Point, 실수) 타입 연산
};
typedef enum uarch_operand_type_t types_of_operands;  // "types_of_operands"라는 별명

/*
 * special_operations_t: 특수 연산의 세부 종류
 * GPU 전력 모델(power model)에서 어떤 종류의 연산인지 구분하기 위해 사용합니다.
 * 연산 종류에 따라 소비하는 전력이 다르기 때문입니다.
 */
enum special_operations_t {
  OTHER_OP,        // 기타 연산
  INT__OP,         // 정수 일반 연산
  INT_MUL24_OP,    // 24비트 정수 곱셈
  INT_MUL32_OP,    // 32비트 정수 곱셈
  INT_MUL_OP,      // 정수 곱셈 (일반)
  INT_DIV_OP,      // 정수 나눗셈
  FP_MUL_OP,       // 단정밀도 부동소수점 곱셈
  FP_DIV_OP,       // 단정밀도 부동소수점 나눗셈
  FP__OP,          // 단정밀도 부동소수점 일반 연산
  FP_SQRT_OP,      // 제곱근(루트) 연산
  FP_LG_OP,        // 로그 연산
  FP_SIN_OP,       // 사인 연산
  FP_EXP_OP,       // 지수 연산
  DP_MUL_OP,       // 배정밀도 부동소수점 곱셈
  DP_DIV_OP,       // 배정밀도 부동소수점 나눗셈
  DP___OP,         // 배정밀도 부동소수점 일반 연산
  TENSOR__OP,      // 텐서 코어 연산
  TEX__OP          // 텍스처 연산
};

typedef enum special_operations_t
    special_ops;  // "special_ops"라는 별명 - 전력 모델에서 연산 종류를 식별하는 데 필요

/*
 * operation_pipeline_t: 연산이 사용하는 파이프라인(처리 경로)
 * GPU 내부에는 여러 종류의 파이프라인이 있고, 명령어 종류에 따라
 * 어떤 파이프라인을 사용할지 결정됩니다.
 * 파이프라인이란 세탁기처럼 여러 단계를 거쳐 작업을 처리하는 방식입니다.
 */
enum operation_pipeline_t {
  UNKOWN_OP,         // 알 수 없는 파이프라인
  SP__OP,            // SP(Single Precision) 파이프라인: 단정밀도 실수 계산용
  DP__OP,            // DP(Double Precision) 파이프라인: 배정밀도 실수 계산용
  INTP__OP,          // INT 파이프라인: 정수 계산용
  SFU__OP,           // SFU 파이프라인: 특수 함수 계산용 (sin, cos, 루트 등)
  TENSOR_CORE__OP,   // 텐서 코어 파이프라인: AI 행렬 연산용
  MEM__OP,           // 메모리 파이프라인: 메모리 읽기/쓰기 처리용
  SPECIALIZED__OP,   // 특수 파이프라인: 사용자 정의 특수 연산용
};
typedef enum operation_pipeline_t operation_pipeline;  // "operation_pipeline"이라는 별명

/*
 * mem_operation_t: 메모리 연산이 텍스처 관련인지 아닌지 구분
 */
enum mem_operation_t {
  NOT_TEX,   // 텍스처가 아닌 일반 메모리 연산
  TEX        // 텍스처 메모리 연산
};
typedef enum mem_operation_t mem_operation;  // "mem_operation"이라는 별명

/*
 * _memory_op_t: 메모리 연산의 종류 (읽기/쓰기/없음)
 */
enum _memory_op_t {
  no_memory_op = 0,   // 메모리 연산 없음
  memory_load,        // 메모리 로드(읽기): 메모리에서 데이터를 가져옴
  memory_store        // 메모리 스토어(쓰기): 메모리에 데이터를 저장함
};

/*
 * 표준 C/C++ 라이브러리 헤더 파일들을 포함합니다.
 * 라이브러리란 미리 만들어진 유용한 기능 모음입니다.
 */
#include <assert.h>     // assert: 조건이 거짓이면 프로그램을 멈추는 디버깅(오류 찾기) 도구
#include <stdlib.h>     // 메모리 할당(malloc, free), 난수 생성 등의 기본 기능
#include <algorithm>    // 정렬, 검색 등 알고리즘 모음 (std::sort, std::find 등)
#include <bitset>       // std::bitset: 비트(0 또는 1) 배열을 다루는 자료구조. 워프 마스크에 사용
#include <deque>        // std::deque: 양쪽 끝에서 추가/삭제할 수 있는 자료구조 (덱)
#include <list>         // std::list: 연결 리스트. 중간에 요소를 추가/삭제하기 쉬운 자료구조
#include <map>          // std::map: 키-값 쌍으로 데이터를 저장하는 사전(딕셔너리) 자료구조
#include <vector>       // std::vector: 크기가 자동으로 변하는 배열 (가장 많이 쓰는 자료구조)

/*
 * vector_types.h: CUDA의 dim3 같은 벡터 타입을 정의하는 헤더
 * dim3은 3차원 좌표(x, y, z)를 나타내는 구조체입니다.
 * GPU 프로그래밍에서 그리드(grid)와 블록(block)의 크기를 지정할 때 사용합니다.
 */
#if !defined(__VECTOR_TYPES_H__)  // vector_types.h가 아직 포함되지 않았다면
#include "vector_types.h"         // dim3 등의 벡터 타입 정의를 포함
#endif

/*
 * [한국어] dim3comp: dim3 구조체를 std::map의 키(key)로 사용하기 위한 비교 함수 객체(functor).
 *
 * dim3은 CUDA에서 그리드/블록 크기를 3차원(x, y, z)으로 표현하는 구조체이다.
 * kernel_info_t에서 std::map<dim3, ...>을 선언할 때 키 비교기가 필요하며, 이 구조체가 그 역할을 한다.
 * C++ STL의 std::map은 키에 strict weak ordering(완전 순서)을 요구하므로,
 * 다차원 키는 사전식(lexicographic) 비교를 구현해야 한다.
 *
 * z → y → x 순서로 비교하는 이유:
 *   CUDA 그리드는 일반적으로 x 방향이 가장 빠르게 변하는 메이저 축이므로,
 *   z를 최상위 키로, x를 최하위 키로 배치해야 블록 ID의 선형화 순서와
 *   일치하는 정렬이 된다. (blockIdx = z * gridDim.y * gridDim.x + y * gridDim.x + x)
 *
 * 사용처: kernel_info_t 내부의 std::map<dim3, std::list<CUstream_st*>, dim3comp> m_cta_streams.
 *
 * 호출 체인:
 *   kernel_info_t(CTA 스트림 맵 조회) → dim3comp::operator() → bool 반환
 */
struct dim3comp {
  /*
   * [한국어]
   * operator() - 두 dim3 값의 크기를 사전식으로 비교하는 함수 호출 연산자.
   *
   * @a: 비교 대상 첫 번째 dim3 (왼쪽 피연산자)
   * @b: 비교 대상 두 번째 dim3 (오른쪽 피연산자)
   * @return: a < b 이면 true, 그렇지 않으면 false.
   *          std::map의 Compare 요구사항인 strict weak ordering을 만족시킨다.
   *
   * z → y → x 순서로 순차 비교하여 3차원 블록 인덱스의 사전식 순서를 구현한다.
   * const 한정자: 이 연산자는 comparator 객체의 상태를 변경하지 않는다(순수 비교 함수).
   *
   * 호출 체인:
   *   std::map<dim3, ..., dim3comp> 내부 트리 연산 → [operator()] → bool
   */
  bool operator()(const dim3 &a, const dim3 &b) const {
    if (a.z < b.z)       // [한국어] z 축을 최상위 비교 키로 사용 — z가 다르면 z만으로 순서 결정
      return true;        // [한국어] a.z < b.z: a가 b보다 "작음"을 의미
    else if (a.y < b.y)  // [한국어] z가 같을 때 y 축을 차상위 비교 키로 사용
      return true;        // [한국어] a.y < b.y: y만으로 순서 결정
    else if (a.x < b.x)  // [한국어] z, y가 모두 같을 때 x 축을 최하위 비교 키로 사용
      return true;        // [한국어] a.x < b.x: x만으로 순서 결정
    else
      return false;       // [한국어] z·y·x 모두 동일하거나 a가 큰 경우 — a < b가 아님
  }
};

/*
 * [한국어]
 * increment_x_then_y_then_z - dim3 좌표를 x→y→z 순서로 증가시키는 유틸리티
 *
 * @i: 증가시킬 dim3 좌표 (in/out). CTA ID 또는 스레드 ID를 나타냄.
 * @bound: 각 차원의 상한 (grid_dim 또는 block_dim). 이 값에 도달하면 다음 차원이 증가.
 * @return: 없음. i가 직접 수정된다.
 *
 * CUDA의 블록 선형화 규칙에 따라 x가 가장 빠르게 변하고 z가 가장 느리게 변한다.
 * kernel_info_t::increment_cta_id()와 increment_thread_id()에서 호출된다.
 * x가 bound.x에 도달하면 y를 1 증가시키고 x를 0으로 리셋,
 * y가 bound.y에 도달하면 z를 1 증가시키고 y를 0으로 리셋한다.
 * 구현 위치: abstract_hardware_model.cc
 *
 * 호출 체인:
 *   kernel_info_t::increment_cta_id() → [increment_x_then_y_then_z]
 *   kernel_info_t::increment_thread_id() → [increment_x_then_y_then_z]
 */
void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

// Jin: child kernel information for CDP
// Jin이라는 개발자가 추가한 부분: CDP(CUDA Dynamic Parallelism) 지원
// CDP란 GPU 프로그램 안에서 또 다른 GPU 프로그램을 실행하는 기능입니다.
// 마치 수업 중에 또 다른 수업을 시작하는 것과 비슷합니다.
#include "stream_manager.h"     // 스트림 관리자 헤더 포함 (스트림: GPU 작업의 대기열)
class stream_manager;           // 스트림 관리자 클래스 전방 선언
struct CUstream_st;             // CUDA 스트림 구조체 전방 선언
// extern stream_manager * g_stream_manager;  // 주석 처리됨: 전역 스트림 관리자 (사용 안 함)

/*
 * extern: "이 변수는 다른 파일에서 정의되어 있어"라고 알려주는 키워드
 * pinned_memory: 고정된(pinned) 메모리의 매핑 정보
 * "고정 메모리"란 운영체제가 위치를 바꾸지 않는 메모리로, CPU↔GPU 간 데이터 전송이 빠릅니다.
 */
// support for pinned memories added
extern std::map<void *, void **> pinned_memory;       // 고정 메모리 주소 매핑 (가상 주소 → 실제 주소)
extern std::map<void *, size_t> pinned_memory_size;   // 고정 메모리 크기 매핑 (주소 → 크기)

/*
 * ==========================================================================
 * kernel_info_t - CUDA 커널 하나의 실행 정보 관리 클래스
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * cuLaunchKernel() 호출 시 생성되어 해당 커널의 실행에 필요한 모든 정보를 담는다.
 * 그리드/블록 크기(gridDim, blockDim), 다음 실행할 CTA/스레드 ID, 실행 중인 SM 수,
 * 텍스처 바인딩 스냅샷, CDP 부모-자식 관계, 타이밍 측정값을 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpu-sim.cc의 사이클 루프에서 no_more_ctas_to_run()이 false인 동안 SM(shader_core_ctx)에
 * 새 CTA를 할당하고 increment_cta_id()로 다음 CTA로 진행한다.
 * SM이 CTA를 완료하면 dec_running()을 호출하고, done()이 true가 되면 커널 종료.
 *
 * === 타 모듈과의 연결 ===
 * - 생성: gpgpusim_entrypoint.cc의 gpgpu_cuda_ptx_sim_main_func()에서 생성.
 * - 소비: gpu-sim.cc가 CTA를 배정하고, shader.cc(SM)가 CTA를 실행.
 * - 기능 시뮬레이션: cuda-sim/ptx_thread_info.cc가 m_param_mem, m_active_threads 접근.
 * - CDP: stream_manager.cc가 자식 커널 생성 시 set_parent()/set_child() 호출.
 *
 * === 스레드 안전성 ===
 * m_next_cta, m_next_tid는 gpu-sim의 사이클 루프에서만 접근하므로 별도 락 불필요.
 * m_num_cores_running은 사이클 루프에서 단일 스레드로 관리.
 * m_child_kernels, m_cta_streams는 CDP 실행 시 GPU 내부에서 접근 — 동기화 주의.
 */
class kernel_info_t {
 public:  // 외부에서 접근 가능한 영역
  //   kernel_info_t()              // 주석 처리된 기본 생성자 (사용 안 함)
  //   {
  //      m_valid=false;
  //      m_kernel_entry=NULL;
  //      m_uid=0;
  //      m_num_cores_running=0;
  //      m_param_mem=NULL;
  //   }

  /*
   * [한국어]
   * kernel_info_t - 기본 커널 정보 생성자
   *
   * @gridDim: 그리드 차원 (CTA의 3D 배치 크기). cuLaunchKernel의 gridDim에 해당.
   * @blockDim: 블록 차원 (CTA당 스레드 3D 배치). cuLaunchKernel의 blockDim에 해당.
   * @entry: 커널 함수의 PTX function_info 포인터. cuda-sim/ptx_ir.h에 정의.
   * @streamID: 이 커널이 실행되는 CUDA 스트림 ID. 스트림별 순서 보장에 사용.
   *            0 = 기본 스트림(NULL stream), 양수 = 명시적 스트림.
   * @return: (생성자이므로 반환값 없음)
   *
   * gpgpusim_entrypoint.cc의 gpgpu_cuda_ptx_sim_main_func()에서 호출된다.
   * m_next_cta, m_next_tid를 (0,0,0)으로 초기화하여 첫 CTA부터 순서대로 실행되도록 한다.
   * m_num_cores_running = 0으로 시작하며, SM이 CTA를 받을 때마다 inc_running()으로 증가.
   * 텍스처를 사용하지 않는 커널에 사용하는 버전 (텍스처 매핑 인자 없음).
   *
   * 호출 체인:
   *   gpgpusim_entrypoint.cc::gpgpu_cuda_ptx_sim_main_func() → [kernel_info_t()]
   *   → gpu-sim.cc::cycle() → shader.cc::issue_block2core()
   */
  kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry,
                unsigned long long streamID);
  /*
   * [한국어]
   * kernel_info_t - 텍스처 바인딩 스냅샷 포함 생성자
   *
   * @gridDim, @blockDim, @entry: 기본 생성자와 동일.
   * @nameToCudaArray: 커널 launch 시점의 텍스처 이름→CUDA 배열 매핑 스냅샷.
   *   런타임에 텍스처 바인딩이 변경되어도 이 커널 실행 중에는 launch 시점 상태를 유지.
   * @nameToTextureInfo: 텍스처 이름→텍스처 속성(크기·채널 형식) 매핑 스냅샷.
   *   텍스처 캐시(const/tex L1)는 이 매핑을 기준으로 히트/미스를 판정한다.
   * @return: (생성자이므로 반환값 없음)
   *
   * 텍스처를 사용하는 커널에서 호출된다. 스냅샷을 m_NameToCudaArray, m_NameToTextureInfo에
   * 복사 저장하므로, 이후 런타임에서 텍스처 바인딩이 변경되어도 이 커널은 영향을 받지 않는다.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaLaunchKernel() → [kernel_info_t()]
   */
  kernel_info_t(
      dim3 gridDim, dim3 blockDim, class function_info *entry,
      std::map<std::string, const struct cudaArray *> nameToCudaArray,       // 텍스처 이름 → CUDA 배열 매핑
      std::map<std::string, const struct textureInfo *> nameToTextureInfo);  // 텍스처 이름 → 텍스처 정보 매핑
  ~kernel_info_t();  // 소멸자(destructor): 객체가 삭제될 때 m_param_mem 등 할당된 메모리를 해제

  /*
   * [한국어]
   * inc_running / dec_running / running / done - 커널 실행 상태 추적 함수들
   *
   * inc_running():
   *   SM이 이 커널의 CTA를 받아 실행을 시작할 때 호출. m_num_cores_running++.
   *   caller: shader.cc::issue_block2core().
   *
   * dec_running():
   *   SM이 이 커널의 마지막 warp를 완료할 때 호출. m_num_cores_running--.
   *   assert로 0 미만 감소를 방지 (언더플로우 방어 코드).
   *   caller: shader.cc::warp_exit() 또는 gpu-sim.cc의 커널 완료 처리.
   *
   * running():
   *   아직 실행 중인 SM이 하나라도 있으면 true. done() 판정에 사용.
   *
   * done():
   *   더 실행할 CTA가 없고(no_more_ctas_to_run() == true) 실행 중인 SM도 없으면 true.
   *   gpu-sim.cc의 커널 완료 감지 조건으로 사용됨.
   *   주의: is_finished()는 done() 외에 CDP 자식 커널 완료도 확인한다.
   *
   * 사이클 컨텍스트: gpu-sim의 단일 사이클 루프 스레드에서 호출 — 락 불필요.
   *
   * 호출 체인:
   *   shader.cc::issue_block2core() → [inc_running()]
   *   shader.cc::warp_exit() → [dec_running()]
   *   gpu-sim.cc::cycle() → [done()]
   */
  void inc_running() { m_num_cores_running++; }  // [한국어] SM이 CTA를 받을 때마다 1 증가
  void dec_running() {                            // [한국어] SM이 커널의 마지막 CTA를 완료할 때 1 감소
    assert(m_num_cores_running > 0);              // [한국어] 음수 방지: 실행 중인 SM이 0 이상이어야 함
    m_num_cores_running--;                        // [한국어] 실행 완료된 SM 수 반영
  }
  bool running() const { return m_num_cores_running > 0; }  // [한국어] 아직 실행 중인 SM이 하나라도 있는지 확인
  bool done() const { return no_more_ctas_to_run() && !running(); }  // [한국어] 모든 CTA가 배정되고 실행 중인 SM도 없으면 커널 완료

  /*
   * [한국어]
   * entry - 이 커널이 실행하는 PTX 함수의 function_info 포인터를 반환
   *
   * @return: m_kernel_entry. cuda-sim/ptx_ir.h의 function_info 객체 포인터.
   *   이 포인터를 통해 PTX 명령어 목록, 파라미터 목록, 지역 변수 목록에 접근 가능.
   *
   * const 버전: 커널 정보를 읽기 전용으로 참조할 때 사용 (const kernel_info_t* 컨텍스트).
   *
   * 호출 체인:
   *   cuda-sim/ptx_sim.cc → [entry()] → function_info의 명령어 실행
   */
  class function_info *entry() {
    return m_kernel_entry;                        // [한국어] PTX 함수 진입점 포인터 반환
  }
  const class function_info *entry() const { return m_kernel_entry; }  // [한국어] 읽기 전용 버전 — const 컨텍스트에서 사용

  /*
   * [한국어]
   * num_blocks - 커널의 총 CTA(블록) 수를 계산
   *
   * @return: m_grid_dim.x * y * z. 3차원 그리드의 전체 CTA 개수.
   *   예: gridDim = (4, 2, 1) → 4*2*1 = 8개 CTA.
   *
   * SM 배정 루프에서 "아직 배정할 CTA가 남았는가"를 판단하는 기준이 된다.
   * GPU가 지원하는 최대 CTA 수보다 적어야 시뮬레이션이 올바르게 동작한다.
   *
   * 호출 체인:
   *   gpu-sim.cc::cycle() 또는 통계 출력 → [num_blocks()]
   */
  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;  // [한국어] x*y*z 곱으로 전체 CTA 수 계산
  }

  /*
   * [한국어]
   * threads_per_cta - CTA 하나의 스레드 수를 계산
   *
   * @return: m_block_dim.x * y * z. CUDA의 blockDim.x * y * z에 해당.
   *   예: blockDim = (256, 1, 1) → threads_per_cta() = 256.
   *
   * SM의 점유율(occupancy) 계산과 공유 메모리/레지스터 할당량 결정에 사용된다.
   * 값이 클수록 SM당 동시 실행 워프 수(occupancy)가 줄어들 수 있다.
   *
   * 호출 체인:
   *   shader.cc::issue_block2core() → [threads_per_cta()] → 레지스터/공유메모리 할당량 결정
   */
  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;  // [한국어] x*y*z 곱으로 CTA 내 스레드 총 수 계산
  }

  dim3 get_grid_dim() const { return m_grid_dim; }   // [한국어] 그리드 크기(3D dim3)를 반환 — 통계·스케줄링 참조용
  dim3 get_cta_dim() const { return m_block_dim; }   // [한국어] CTA(블록) 크기(3D dim3)를 반환 — 스레드 초기화 참조용

  /*
   * [한국어]
   * increment_cta_id - 다음 실행할 CTA의 3D 좌표를 x→y→z 순으로 진행
   *
   * 내부적으로 increment_x_then_y_then_z()를 호출한다.
   * CTA가 바뀌므로 m_next_tid를 (0,0,0)으로 리셋한다 — 새 CTA에서 첫 스레드부터 시작.
   *
   * 사이클 컨텍스트: gpu-sim의 사이클 루프에서만 호출 — 단일 스레드, 락 불필요.
   *
   * 호출 체인:
   *   gpu-sim.cc::cycle() → shader.cc::issue_block2core() → [increment_cta_id()]
   */
  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);  // [한국어] x→y→z 순으로 다음 CTA 좌표 계산
    m_next_tid.x = 0;  // [한국어] 새 CTA의 첫 스레드부터 시작 — x 리셋
    m_next_tid.y = 0;  // [한국어] y 리셋
    m_next_tid.z = 0;  // [한국어] z 리셋
  }
  dim3 get_next_cta_id() const { return m_next_cta; }  // [한국어] 다음에 SM에 배정할 CTA의 3D ID 반환

  /*
   * [한국어]
   * get_next_cta_id_single - 현재 m_next_cta의 3D 좌표를 1D 선형 인덱스로 변환
   *
   * @return: x + gridDim.x * y + gridDim.x * gridDim.y * z.
   *   CUDA의 blockIdx 선형화 공식과 동일한 방식.
   *
   * 체크포인트 저장/복원 시 특정 CTA를 식별하는 1D ID로 사용된다.
   * num_blocks() 범위 내의 값을 반환하므로 배열 인덱스로도 활용 가능.
   */
  unsigned get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +   // [한국어] x + gridDim.x*y 항
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;    // [한국어] + gridDim.x*gridDim.y*z 항 — 3D→1D 선형화
  }

  /*
   * [한국어]
   * no_more_ctas_to_run - 모든 CTA가 이미 배정되었는지(실행 완료 여부 무관) 확인
   *
   * @return: m_next_cta의 어느 차원이라도 grid_dim 이상이면 true.
   *
   * gpu-sim.cc의 CTA 배정 루프 종료 조건 및 done() 판정에 사용.
   * 주의: "모든 CTA가 완료됨"이 아니라 "모든 CTA가 배정됨"을 의미한다.
   *       실제 완료는 running()==false 조건을 추가로 확인해야 한다.
   *
   * 호출 체인:
   *   gpu-sim.cc::cycle() → [no_more_ctas_to_run()] — CTA 배정 루프 탈출 여부 판단
   *   done() → [no_more_ctas_to_run()] — 커널 완료 조건 구성
   */
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||  // [한국어] x 또는 y 차원이 경계 초과 시 배정 완료
            m_next_cta.z >= m_grid_dim.z);                                    // [한국어] z 차원이 경계 초과 시도 배정 완료
  }

  /*
   * [한국어]
   * increment_thread_id / get_next_thread_id_3d / get_next_thread_id /
   * more_threads_in_cta - CTA 내 스레드 순회 함수들
   *
   * increment_thread_id():
   *   현재 CTA 내에서 다음 스레드로 진행. increment_x_then_y_then_z()를 block_dim 기준으로 호출.
   *   cuda-sim의 스레드 초기화 루프에서 매 스레드마다 호출됨.
   *
   * get_next_thread_id_3d():
   *   다음 스레드의 3D 좌표(threadIdx.x, y, z에 대응)를 반환.
   *   ptx_thread_info 초기화 시 threadIdx 설정에 사용.
   *
   * get_next_thread_id():
   *   3D 스레드 ID를 1D 선형 인덱스로 변환.
   *   공식: x + blockDim.x*y + blockDim.x*blockDim.y*z.
   *
   * more_threads_in_cta():
   *   현재 CTA에 아직 초기화되지 않은 스레드가 있는지 확인. 스레드 순회 루프 종료 조건.
   *
   * 호출 체인:
   *   cuda-sim/ptx_sim.cc::launch() → [more_threads_in_cta() / increment_thread_id()]
   */
  void increment_thread_id() {                                  // [한국어] 현재 CTA 내에서 다음 스레드 좌표로 진행
    increment_x_then_y_then_z(m_next_tid, m_block_dim);        // [한국어] blockDim을 경계로 x→y→z 순 증가
  }
  dim3 get_next_thread_id_3d() const { return m_next_tid; }     // [한국어] 다음 스레드의 3D ID 반환 — threadIdx 초기화에 사용
  unsigned get_next_thread_id() const {                         // [한국어] 다음 스레드의 1D 선형 인덱스 반환
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +       // [한국어] x + blockDim.x*y 항
           m_block_dim.x * m_block_dim.y * m_next_tid.z;       // [한국어] + blockDim.x*blockDim.y*z 항
  }
  bool more_threads_in_cta() const {                            // [한국어] 현재 CTA에 아직 순회하지 않은 스레드가 남았는지 확인
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;                       // [한국어] 모든 차원이 경계 미만이어야 스레드가 남은 것
  }
  unsigned get_uid() const { return m_uid; }                    // [한국어] 이 커널 인스턴스의 전역 고유 ID 반환 — 통계·체크포인트 키
  unsigned long long get_streamID() const { return m_streamID; }  // [한국어] 커널이 속한 CUDA 스트림 ID 반환
  std::string get_name() const { return name(); }               // [한국어] 커널 이름 반환 (name()으로 위임)
  std::string name() const;                                     // [한국어] 커널 이름을 반환하는 실제 구현 — abstract_hardware_model.cc에 정의

  /*
   * [한국어]
   * active_threads - 현재 기능 시뮬레이션에서 활성 중인 PTX 스레드 목록의 참조를 반환
   *
   * @return: m_active_threads에 대한 참조. ptx_thread_info* 포인터들의 연결 리스트.
   *
   * cuda-sim의 스레드 초기화 루프에서 새 ptx_thread_info를 추가하고,
   * 스레드 완료 시 목록에서 제거한다.
   * 비레퍼런스 복사가 아닌 참조(ref)를 반환하므로 직접 수정 가능.
   *
   * 호출 체인:
   *   cuda-sim/ptx_sim.cc → [active_threads()] → ptx_thread_info 순회
   */
  std::list<class ptx_thread_info *> &active_threads() {
    return m_active_threads;  // [한국어] 활성 스레드 목록의 레퍼런스 반환 — 수정 허용
  }
  /*
   * [한국어]
   * get_param_memory - 커널에 전달된 파라미터가 저장된 메모리 공간 포인터를 반환
   *
   * @return: m_param_mem. PTX 실행 시 커널 인자(cuLaunchKernel의 args[])가 저장된
   *   memory_space 객체 포인터. cuda-sim/ptx_thread_info.cc에서 .param 공간 접근 시 사용.
   *
   * PTX의 .param 공간은 커널 파라미터를 읽는 전용 주소 공간이다.
   * ld.param 명령어를 실행할 때 이 포인터가 참조된다.
   *
   * 호출 체인:
   *   cuda-sim/ptx_thread_info.cc::ld_impl() → [get_param_memory()] → .param 공간 읽기
   */
  class memory_space *get_param_memory() {
    return m_param_mem;  // [한국어] .param 공간의 memory_space 포인터 반환
  }

  // The following functions access texture bindings present at the kernel's
  // launch
  // [한국어] 아래 함수들은 커널 launch 시점의 텍스처 바인딩 스냅샷에 접근한다.
  // launch 이후 cudaBindTexture()가 호출되어도 이미 실행 중인 커널의 바인딩은 변하지 않는다.

  /*
   * [한국어]
   * get_texarray - 텍스처 이름으로 커널 launch 시점의 CUDA 배열 포인터를 반환
   *
   * @texname: PTX tex 명령어에서 참조하는 텍스처 심볼 이름.
   * @return: m_NameToCudaArray[texname]. cudaArray의 실제 데이터 포인터.
   *   assert: 이름이 스냅샷에 없으면 abort — 텍스처 바인딩 없이 tex 접근 시도를 방지.
   *
   * PTX의 tex.2d.v4.f32 등의 명령어를 실행할 때 텍스처 데이터 소스를 결정하기 위해 호출된다.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texarray()] → cudaArray 데이터 접근
   */
  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =  // [한국어] 이름→배열 스냅샷 맵에서 검색
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());  // [한국어] 찾지 못하면 abort — 미등록 텍스처 접근은 프로그래밍 오류
    return t->second;                       // [한국어] 찾은 cudaArray 포인터 반환
  }

  /*
   * [한국어]
   * get_texinfo - 텍스처 이름으로 커널 launch 시점의 텍스처 속성을 반환
   *
   * @texname: 텍스처 심볼 이름.
   * @return: m_NameToTextureInfo[texname]. 텍스처의 채널 형식, 크기 등의 메타데이터.
   *   assert: 이름이 스냅샷에 없으면 abort.
   *
   * tex 명령어에서 텍스처 데이터 타입(float4, int4 등)과 크기를 결정하는 데 사용된다.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texinfo()] → 채널 형식 결정
   */
  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);           // [한국어] 이름→텍스처 정보 스냅샷 맵에서 검색
    assert(t != m_NameToTextureInfo.end());          // [한국어] 미등록 텍스처 접근 방어
    return t->second;                                // [한국어] textureInfo 포인터 반환
  }

 private:  // [한국어] 비공개 영역: kernel_info_t 클래스 외부에서 직접 접근 불가
  kernel_info_t(const kernel_info_t &);
  /* [한국어] 복사 생성자 비활성화.
   * 커널 정보는 단일 소유권(unique ownership)으로 관리된다.
   * 실수로 커널 객체를 복사하면 m_param_mem, m_active_threads 등의 이중 해제(double-free)가 발생.
   * 선언만 하고 정의하지 않아 컴파일 에러로 복사를 원천 차단한다. */
  void operator=(const kernel_info_t &);
  /* [한국어] 복사 대입 연산자 비활성화. 복사 생성자와 동일한 이중 해제 위험을 방지한다. */

  class function_info *m_kernel_entry;
  /* [한국어] 이 커널이 실행하는 PTX 함수의 정보 포인터.
   * 설정자: 생성자에서 인자로 받아 저장.
   * 읽는 자: entry()를 통해 cuda-sim의 PTX 실행 엔진이 명령어를 페치할 때.
   * 값 범위: 유효한 function_info 포인터 (NULL 불가 — 생성자 단계에서 검증됨).
   * 동기화: 커널 생성 후 읽기 전용 — 별도 락 불필요. */

  unsigned m_uid;
  /* [한국어] 이 커널 인스턴스의 전역 고유 ID.
   * 설정자: gpgpu_sim에서 커널 카운터를 1씩 증가시켜 할당.
   * 읽는 자: 통계 출력, 체크포인트 키, get_uid() 반환값.
   * 값 범위: 1부터 시작하는 단조 증가 정수. 0은 미초기화를 의미. */
  unsigned long long m_streamID;
  /* [한국어] 이 커널이 실행되는 CUDA 스트림의 ID.
   * 설정자: 생성자 인자 streamID.
   * 읽는 자: get_streamID()를 통해 warp_inst_t 발행 시 스트림 ID 전파, 통계 분리.
   * 값 범위: 0 = 기본 스트림(NULL stream), 양수 = 명시적 스트림. */

  // These maps contain the snapshot of the texture mappings at kernel launch
  // [한국어] 커널 launch 시점의 텍스처 매핑 정보를 스냅샷으로 보관 — 실행 중 바인딩 변경 격리
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  /* [한국어] 커널 launch 시점의 텍스처 이름 → CUDA 배열 포인터 스냅샷.
   * 설정자: 텍스처 생성자 또는 gpgpu_t::getNameArrayMapping() 경유로 복사 초기화.
   * 읽는 자: get_texarray()를 통해 PTX tex 명령어 실행 시 데이터 소스 결정.
   * 값 범위: 커널 launch 시 등록된 모든 텍스처 항목.
   * 동기화: launch 후 읽기 전용 — 별도 락 불필요. */
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
  /* [한국어] 커널 launch 시점의 텍스처 이름 → 텍스처 속성(크기·채널 형식) 스냅샷.
   * m_NameToCudaArray와 쌍으로 관리된다.
   * 설정자: 텍스처 생성자 또는 getNameInfoMapping() 경유로 복사 초기화.
   * 읽는 자: get_texinfo()를 통해 tex 명령어 실행 시 채널 형식과 크기 결정.
   * 동기화: launch 후 읽기 전용 — 별도 락 불필요. */

  dim3 m_grid_dim;
  /* [한국어] 그리드 크기: CTA가 3차원으로 몇 개인지(gridDim).
   * cuLaunchKernel의 gridDim 인자에서 초기화됨.
   * 값 범위: 각 차원 ≥ 1. 총 CTA 수 = x * y * z.
   * no_more_ctas_to_run()의 경계값으로 사용. 변경되지 않는다. */
  dim3 m_block_dim;
  /* [한국어] 블록 크기: CTA 하나에 스레드가 3차원으로 몇 개인지(blockDim).
   * 총 스레드 수 = x * y * z. SM 점유율 계산 및 threads_per_cta() 반환값.
   * 변경되지 않는다 — 커널 실행 중 블록 크기는 고정. */
  dim3 m_next_cta;
  /* [한국어] 다음에 SM에 배정할 CTA의 3D 좌표.
   * 설정자: increment_cta_id()로 x→y→z 순 진행. 초기값: (0,0,0).
   * 읽는 자: gpu-sim의 CTA 배정 루프, get_next_cta_id_single().
   * 값 범위: (0,0,0) ~ (grid_dim.x-1, grid_dim.y-1, grid_dim.z-1).
   * 이 값이 grid_dim 경계를 초과하면 no_more_ctas_to_run() = true. */
  dim3 m_next_tid;
  /* [한국어] 현재 CTA 내에서 다음에 초기화할 스레드의 3D 좌표.
   * 설정자: increment_thread_id(). CTA가 바뀌면 increment_cta_id()에 의해 (0,0,0)으로 리셋.
   * 읽는 자: cuda-sim의 스레드 초기화 루프에서 threadIdx 설정에 사용. */

  unsigned m_num_cores_running;
  /* [한국어] 현재 이 커널의 CTA를 실행 중인 SM(shader_core_ctx) 수.
   * 설정자: inc_running() — SM이 CTA를 받을 때 1 증가,
   *         dec_running() — SM이 마지막 warp를 완료할 때 1 감소.
   * 읽는 자: running(), done().
   * 값 범위: 0 ~ (총 SM 수). 0이고 no_more_ctas_to_run()이면 커널 완료.
   * 동기화: 사이클 루프의 단일 스레드에서만 접근 — 별도 락 불필요. */

  std::list<class ptx_thread_info *> m_active_threads;
  /* [한국어] 현재 기능 시뮬레이션에서 활성 상태인 PTX 스레드들의 목록.
   * 설정자: cuda-sim의 스레드 초기화 시 push_back() 으로 추가.
   * 읽는 자: active_threads() getter, 스레드 완료 처리 루프.
   * 값 범위: 커널 실행 중에는 비어있지 않음. 모든 스레드가 완료되면 비워짐.
   * 동기화: cuda-sim 기능 시뮬레이션 컨텍스트 내에서만 접근. */
  class memory_space *m_param_mem;
  /* [한국어] 이 커널에 전달된 파라미터(cuLaunchKernel의 args[])가 저장된 메모리 공간.
   * 설정자: 커널 생성자에서 파라미터를 복사하여 memory_space 객체 생성 후 저장.
   * 읽는 자: get_param_memory()를 통해 PTX ld.param 명령어 실행 시 .param 공간 접근.
   * 값 범위: 유효한 memory_space 포인터. ~kernel_info_t() 소멸자에서 delete로 해제. */

 public:  // [한국어] CDP(CUDA Dynamic Parallelism) 관련 공개 인터페이스
  // Jin: parent and child kernel management for CDP
  // [한국어] CDP(CUDA Dynamic Parallelism) 지원: 부모-자식 커널 관계 관리.
  // CDP란 GPU 커널이 내부에서 cuLaunchKernel()을 호출하여 또 다른 GPU 커널을 실행하는 기능이다.
  // 부모 커널은 자식 커널들이 모두 완료될 때까지 is_finished()가 false를 반환한다.

  /*
   * [한국어]
   * set_parent - CDP: 이 커널을 실행시킨 부모 커널 정보를 등록
   *
   * @parent: 부모 커널의 kernel_info_t 포인터.
   * @parent_ctaid: 부모 커널에서 이 커널을 실행시킨 CTA의 3D ID.
   * @parent_tid: 부모 커널에서 이 커널을 실행시킨 스레드의 3D ID.
   *
   * m_parent_kernel, m_parent_ctaid, m_parent_tid를 설정한다.
   * 동시에 부모의 set_child(this)도 호출하여 양방향 관계를 구성한다.
   *
   * 호출 체인:
   *   stream_manager.cc::launch() → [set_parent()]
   */
  void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);
  /*
   * [한국어]
   * set_child - CDP: 이 커널이 실행시킨 자식 커널을 m_child_kernels 목록에 추가
   *
   * @child: 등록할 자식 커널 포인터.
   *
   * 자식 커널 추적은 children_all_finished()와 is_finished() 판정에 필요하다.
   * 자식 커널이 완료되면 remove_child()로 목록에서 제거된다.
   *
   * 호출 체인:
   *   set_parent() 내부에서 parent->set_child(this) 형태로 호출
   */
  void set_child(kernel_info_t *child);
  /*
   * [한국어]
   * remove_child - CDP: 완료된 자식 커널을 m_child_kernels 목록에서 제거
   *
   * @child: 제거할 자식 커널 포인터.
   *
   * 자식 커널의 notify_parent_finished()에서 부모의 remove_child()를 호출한다.
   * 목록이 비면 children_all_finished() = true가 되어 부모 커널도 완료될 수 있다.
   *
   * 호출 체인:
   *   child->notify_parent_finished() → parent->[remove_child(child)]
   */
  void remove_child(kernel_info_t *child);
  /*
   * [한국어]
   * is_finished - 이 커널(+모든 자식 커널)이 완전히 완료됐는지 확인
   *
   * @return: done()이 true이고 children_all_finished()이면 true.
   *
   * gpu-sim.cc의 커널 완료 감지 루프에서 사용된다.
   * done()만으로는 자식 커널이 아직 실행 중인 경우를 놓칠 수 있으므로
   * CDP 환경에서는 이 함수를 사용해야 한다.
   *
   * 호출 체인:
   *   gpu-sim.cc::cycle() → [is_finished()] — 커널 제거 여부 판단
   */
  bool is_finished();
  /*
   * [한국어]
   * children_all_finished - 이 커널의 모든 자식 커널이 완료됐는지 확인
   *
   * @return: m_child_kernels 목록이 비어있으면 true.
   *
   * is_finished()의 내부 조건으로 사용된다.
   */
  bool children_all_finished();
  /*
   * [한국어]
   * notify_parent_finished - 이 커널 완료 시 부모 커널에 완료를 통보
   *
   * m_parent_kernel->remove_child(this)를 호출하여 부모의 자식 목록에서 제거한다.
   * m_parent_kernel이 null이면 (최상위 커널이면) 아무것도 하지 않는다.
   *
   * 호출 체인:
   *   gpu-sim.cc 커널 완료 처리 → [notify_parent_finished()] → parent->remove_child()
   */
  void notify_parent_finished();
  /*
   * [한국어]
   * create_stream_cta - CDP: 특정 CTA가 새 CUDA 스트림을 생성할 때 등록
   *
   * @ctaid: 스트림을 생성한 CTA의 3D ID.
   * @return: 새로 생성된 CUstream_st 포인터.
   *
   * m_cta_streams[ctaid]에 새 스트림을 추가한다.
   * destroy_cta_streams()에서 커널 종료 시 모두 해제된다.
   *
   * 호출 체인:
   *   cuda-sim::cuStreamCreateWithFlags() → [create_stream_cta()]
   */
  CUstream_st *create_stream_cta(dim3 ctaid);
  /*
   * [한국어]
   * get_default_stream_cta - CDP: 특정 CTA의 기본 스트림을 반환
   *
   * @ctaid: 스트림을 조회할 CTA의 3D ID.
   * @return: 해당 CTA의 첫 번째(기본) 스트림 포인터.
   *
   * CTA가 명시적 스트림 없이 자식 커널을 실행할 때 사용하는 기본 스트림이다.
   */
  CUstream_st *get_default_stream_cta(dim3 ctaid);
  /*
   * [한국어]
   * cta_has_stream - 특정 CTA에 해당 스트림이 등록되어 있는지 확인
   *
   * @ctaid: 조회할 CTA의 3D ID.
   * @stream: 존재 여부를 확인할 스트림 포인터.
   * @return: m_cta_streams[ctaid] 목록에 stream이 있으면 true.
   */
  bool cta_has_stream(dim3 ctaid, CUstream_st *stream);
  /*
   * [한국어]
   * destroy_cta_streams - 커널 종료 시 모든 CTA의 스트림을 해제
   *
   * m_cta_streams의 모든 항목에 대해 스트림 소멸자를 호출하고 목록을 비운다.
   * 메모리 누수 방지를 위해 kernel_info_t 소멸자 또는 커널 완료 처리에서 호출된다.
   *
   * 호출 체인:
   *   gpu-sim.cc 커널 완료 처리 또는 ~kernel_info_t() → [destroy_cta_streams()]
   */
  void destroy_cta_streams();
  /*
   * [한국어]
   * print_parent_info - 부모 커널 정보를 stdout에 출력 (디버깅/로그용)
   *
   * m_parent_kernel, m_parent_ctaid, m_parent_tid 값을 출력한다.
   * CDP 실행 경로 추적 및 디버깅에 사용된다.
   */
  void print_parent_info();
  /*
   * [한국어]
   * get_parent - 이 커널을 실행시킨 부모 커널의 포인터를 반환
   *
   * @return: m_parent_kernel. 최상위 커널이면 null.
   *
   * CDP 계층 탐색에 사용된다.
   */
  kernel_info_t *get_parent() { return m_parent_kernel; }  // [한국어] 부모 커널 포인터 반환 — null이면 호스트에서 직접 실행된 커널

 private:  // [한국어] CDP 관련 비공개 상태 — kernel_info_t 클래스 외부에서 직접 접근 불가
  kernel_info_t *m_parent_kernel;
  /* [한국어] CDP: 이 커널을 실행시킨 부모 커널의 포인터.
   * 설정자: set_parent()에서 초기화.
   * 읽는 자: notify_parent_finished()에서 부모에게 완료 통보, print_parent_info(), get_parent().
   * 값 범위: 유효한 kernel_info_t 포인터 또는 null(최상위 커널, 즉 호스트에서 직접 실행).
   * 동기화: 커널 생성 시 1회 설정 후 읽기 전용. */
  dim3 m_parent_ctaid;
  /* [한국어] CDP: 부모 커널에서 이 커널을 실행한 CTA의 3D ID.
   * 설정자: set_parent()에서 parent_ctaid 인자로 초기화.
   * 읽는 자: print_parent_info()에서 디버그 출력.
   * CDP 실행 경로 추적과 디버깅 목적으로만 사용되는 메타데이터. */
  dim3 m_parent_tid;
  /* [한국어] CDP: 부모 커널에서 이 커널을 실행한 스레드의 3D ID.
   * 설정자: set_parent()에서 parent_tid 인자로 초기화.
   * 읽는 자: print_parent_info()에서 디버그 출력. */
  std::list<kernel_info_t *> m_child_kernels;
  /* [한국어] CDP: 이 커널이 실행시킨 자식 커널들의 목록.
   * 설정자: set_child()에서 push_back()으로 추가.
   * 읽는 자: children_all_finished()에서 목록이 비었는지 확인, remove_child()에서 삭제.
   * is_finished()는 이 목록이 빌 때까지 커널을 완료로 처리하지 않는다.
   * 동기화: gpu-sim의 단일 사이클 루프에서 관리 — 별도 락 없음. */
  std::map<dim3, std::list<CUstream_st *>, dim3comp>
      m_cta_streams;
  /* [한국어] CDP: 각 CTA가 생성한 CUDA 스트림들의 목록.
   * 키: CTA의 3D ID (dim3comp로 z→y→x 사전 정렬). 값: 해당 CTA에서 생성된 스트림 포인터 목록.
   * 설정자: create_stream_cta()에서 새 스트림 추가.
   * 읽는 자: get_default_stream_cta(), cta_has_stream(), destroy_cta_streams().
   * 커널 종료 시 destroy_cta_streams()에서 모든 스트림을 해제하고 맵을 비운다. */

  // Jin: kernel timing
  // [한국어] 커널 실행 타이밍 측정용 공개 필드들.
  // GPU 사이클(gpu_cycle) 단위로 측정되며, 통계 리포트에서 커널 성능 분석에 사용된다.
 public:
  unsigned long long launch_cycle;
  /* [한국어] 이 커널의 실행 명령(dispatch)이 내려진 GPU 시뮬레이션 사이클.
   * 설정자: gpu-sim.cc에서 커널을 실행 큐에 넣을 때 현재 gpu_cycle 값으로 설정.
   * 읽는 자: 타이밍 통계 출력, start_cycle과의 차이로 dispatch latency 계산.
   * start_cycle과의 차이: 커널이 실행 큐에서 대기한 시간 (launch latency). */
  unsigned long long start_cycle;
  /* [한국어] 이 커널의 첫 CTA가 SM에 배정된 사이클(실제 실행 시작 시점).
   * 설정자: gpu-sim.cc에서 첫 CTA를 SM에 배정할 때 현재 gpu_cycle 값으로 설정.
   * 읽는 자: 타이밍 통계 출력. end_cycle - start_cycle = 커널 실행 사이클 수. */
  unsigned long long end_cycle;
  /* [한국어] 이 커널의 마지막 CTA가 완료된 사이클(실제 실행 종료 시점).
   * 설정자: gpu-sim.cc에서 done()이 true가 될 때 현재 gpu_cycle 값으로 설정.
   * 읽는 자: 타이밍 통계 출력. end_cycle - start_cycle = 총 실행 사이클 수. */
  unsigned m_launch_latency;
  /* [한국어] 커널이 실행 큐에 들어간 시점부터 SM 배정까지의 지연 사이클 수.
   * 설정자: gpgpusim.config의 launch latency 파라미터에서 파싱.
   * 읽는 자: gpu-sim.cc의 커널 배정 로직에서 배정 딜레이 구현에 사용. */

  mutable bool cache_config_set;
  /* [한국어] 이 커널의 캐시 설정(FuncCache 선호도: L1 선호 / 공유 메모리 선호)이 SM에 적용됐는지 여부.
   * mutable이므로 const kernel_info_t* 컨텍스트에서도 수정 가능 (lazy 초기화 패턴).
   * 설정자: shader_core_ctx의 캐시 설정 적용 함수에서 true로 설정.
   * 읽는 자: adaptive_cache_config 로직에서 재적용 필요 여부 판단. */

  unsigned m_kernel_TB_latency;
  /* [한국어] CPU→GPU 커널 전달 지연(Thread Block launch overhead), gpu_cycle 단위.
   * cuLaunchKernel() 호출부터 GPU가 실제로 CTA를 받기까지의 소프트웨어 오버헤드를 모델링.
   * 설정자: gpgpusim.config 파싱 시 설정. 읽는 자: gpu-sim.cc의 CTA 배정 타이밍 로직.
   * this used for any CPU-GPU kernel latency and counted in the gpu_cycle */
};

/*
 * ==========================================================================
 * core_config - GPU 코어(SM) 하드웨어 파라미터 설정 클래스
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * shader_core_ctx(SM 시뮬레이터)가 초기화될 때 사용하는 하드웨어 파라미터를 담는다.
 * gpgpusim.config 파일에서 파싱한 값들이 이 클래스의 멤버 변수에 저장된다.
 * warp_size, 공유 메모리 크기/뱅크 수, 캐시 라인 크기, 명령어 발행 제한 등을 정의한다.
 * 순수 가상 함수 init()을 가지므로 직접 인스턴스화할 수 없고, shader_core_config가 상속.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 설정값은 시뮬레이션 시작 시 1회 초기화되고 이후 읽기 전용으로 사용된다.
 * shmem_bank_func()는 매 사이클 공유 메모리 뱅크 충돌 감지 시 호출된다.
 * shader.cc의 shader_core_ctx 생성자에서 core_config의 파생 클래스인 shader_core_config를 참조.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: option_parser.h(gpgpusim.config 파싱), gpgpu_context(역방향 포인터)
 * - 피의존: shader.cc(SM 파이프라인), warp_inst_t(warp_size 참조), gpu-cache.cc(캐시 설정)
 * - 파생 클래스: shader_core_config (shader.h에 정의) — init()을 실제 구현하는 서브클래스
 *
 * === 주요 함수/구조체 요약 ===
 * core_config():       기본 생성자. m_valid=false, num_shmem_bank=16으로 초기화.
 * init():              순수 가상 함수. 파생 클래스에서 config 파싱 결과를 멤버에 반영.
 * shmem_bank_func():   주소에서 공유 메모리 뱅크 번호 계산 — (addr/4) % num_shmem_bank.
 * warp_size:           SIMT 실행 단위 크기 (NVIDIA GPU 표준 32). active_mask_t 크기와 일치해야 함.
 * gpgpu_coalesce_arch: 메모리 합치기(coalescing) 아키텍처 버전 선택자.
 */
class core_config {
 public:
  /*
   * [한국어]
   * core_config - 코어 설정 기본 생성자
   *
   * @ctx: GPU 시뮬레이터 전체 컨텍스트(gpgpu_context). 역방향 포인터로 저장(gpgpu_ctx).
   *
   * 모든 설정값을 "미설정" 또는 "기본" 상태로 초기화한다.
   * m_valid = false이므로 init()이 호출되기 전까지 설정이 완료되지 않은 상태임을 나타낸다.
   * gpgpu_shmem_sizeDefault 등을 (unsigned)-1로 초기화하는 이유:
   *   0은 "공유 메모리 없음"을 의미하는 유효한 값이므로, "미설정"과 구별하기 위해 -1 사용.
   *
   * 호출 체인:
   *   shader_core_config 생성자 → [core_config()]
   */
  core_config(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;                           // [한국어] GPU 시뮬레이터 컨텍스트 역방향 포인터 저장
    m_valid = false;                           // [한국어] init() 호출 전까지 설정 미완료 상태
    num_shmem_bank = 16;                       // [한국어] 공유 메모리 뱅크 수 기본값: 16 (Kepler 이전 아키텍처 기준)
    shmem_limited_broadcast = false;           // [한국어] 브로드캐스트 제한 비활성화 (기본 동작)
    gpgpu_shmem_sizeDefault = (unsigned)-1;    // [한국어] 기본 공유 메모리 크기 미설정 — init()에서 config 파일로 설정
    gpgpu_shmem_sizePrefL1 = (unsigned)-1;     // [한국어] L1 선호 모드 공유 메모리 크기 미설정
    gpgpu_shmem_sizePrefShared = (unsigned)-1; // [한국어] 공유 메모리 선호 모드 크기 미설정
  }
  /*
   * [한국어]
   * init - 설정 파일(gpgpusim.config)의 파싱 결과를 멤버 변수에 반영하는 초기화 함수
   *
   * 순수 가상 함수(pure virtual) — 이 클래스를 직접 인스턴스화 불가.
   * 파생 클래스 shader_core_config에서 반드시 구현해야 한다.
   * 구현 완료 시 m_valid = true로 설정하여 유효한 설정임을 표시.
   *
   * 호출 체인:
   *   gpgpu_sim::init() → shader_core_config::init() → [core_config 멤버 초기화]
   */
  virtual void init() = 0;

  bool m_valid;
  /* [한국어] 이 설정이 완전히 초기화되었는지 여부.
   * 설정자: init() 구현 완료 시 true로 설정.
   * 읽는 자: 설정 사용 전 유효성 검사 assert(m_valid).
   * false 상태에서 설정을 사용하면 정의되지 않은 동작 발생 — 반드시 init() 후 사용. */
  unsigned warp_size;
  /* [한국어] 워프(warp) 크기: SIMT 실행 단위의 스레드 수.
   * NVIDIA GPU에서 표준값은 32. 이 값이 active_mask_t(bitset<MAX_WARP_SIZE>)의 크기와 일치해야 한다.
   * 설정자: gpgpusim.config의 -gpgpu_shader_core_pipeline 옵션에서 파싱.
   * 읽는 자: warp_inst_t 생성/발행, simt_stack 초기화, register_set 슬롯 수 결정.
   * MAX_WARP_SIZE(32)를 초과하면 warp_inst_t 생성자에서 assert 실패. */

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] GPU 시뮬레이터 전체 컨텍스트를 가리키는 역방향 포인터.
   * 설정자: 생성자에서 인자로 받아 저장.
   * 읽는 자: init()에서 설정 파싱 컨텍스트 접근, 하위 설정 초기화.
   * 역방향(backward) 포인터: 설정 객체가 자신을 포함하는 상위 컨텍스트를 참조하는 패턴. */

  // off-chip memory request architecture parameters
  // [한국어] 오프칩(off-chip) 메모리 요청 아키텍처 파라미터 — GPU 칩 외부의 DRAM/L2 관련 설정
  int gpgpu_coalesce_arch;
  /* [한국어] 메모리 합치기(memory coalescing) 아키텍처 버전.
   * 설정자: gpgpusim.config의 -gpgpu_coalesce_arch 옵션.
   * 읽는 자: warp_inst_t::memory_coalescing_arch()에서 합치기 알고리즘 선택.
   * 값 범위: 13(Fermi 이전, 128B 세그먼트 기반), 20(Fermi 이후), 기타 아키텍처 코드.
   * 이 값에 따라 128B/256B 세그먼트 기반 합치기 또는 섹터 기반 합치기가 선택됨. */

  // shared memory bank conflict checking parameters
  // [한국어] 공유 메모리 뱅크 충돌(bank conflict) 검사 파라미터.
  // 뱅크 충돌: 같은 사이클에 여러 스레드가 동일 뱅크에 접근하면 직렬화되어 레이턴시 증가.
  bool shmem_limited_broadcast;
  /* [한국어] 공유 메모리 브로드캐스트 제한 여부.
   * true: 동일 주소에 대한 브로드캐스트를 1회로 제한 (뱅크 충돌 시뮬레이션 보수적 모델).
   * false: 브로드캐스트는 충돌로 계산하지 않음 (기본값, 하드웨어 실제 동작에 가까움).
   * 설정자: gpgpusim.config. 읽는 자: 공유 메모리 뱅크 충돌 감지 로직. */
  static const address_type WORD_SIZE = 4;
  /* [한국어] 공유 메모리 워드(word) 크기: 4바이트 (32비트).
   * shmem_bank_func()에서 바이트 주소를 워드 인덱스로 변환하는 제수(divisor)로 사용.
   * CUDA 공유 메모리는 4바이트 경계로 뱅크가 나뉘므로 (addr/4)%N이 뱅크 번호가 됨. */
  unsigned num_shmem_bank;
  /* [한국어] 공유 메모리의 뱅크 수.
   * 표준: Kepler+(sm_30+) = 32, 이전 아키텍처 = 16. 생성자 기본값: 16.
   * 설정자: gpgpusim.config의 -gpgpu_shmem_num_banks.
   * 읽는 자: shmem_bank_func()에서 뱅크 번호 계산, 뱅크 충돌 감지.
   * 뱅크 수가 많을수록 충돌 가능성 낮아지고 공유 메모리 대역폭이 높아진다. */

  /*
   * [한국어]
   * shmem_bank_func - 공유 메모리 주소에서 뱅크 번호를 계산
   *
   * @addr: 공유 메모리의 바이트 주소.
   * @return: 0 ~ (num_shmem_bank-1) 범위의 뱅크 번호.
   *
   * CUDA 공유 메모리는 4바이트(WORD_SIZE) 경계로 num_shmem_bank개의 뱅크에 인터리빙된다.
   * 공식: (addr / 4) % num_shmem_bank. 이 값이 같은 스레드들이 동시에 접근하면 뱅크 충돌.
   *
   * 사이클 컨텍스트: 매 사이클 공유 메모리 접근 명령어 처리 시 호출됨.
   *
   * 호출 체인:
   *   shader.cc 공유 메모리 접근 처리 → [shmem_bank_func()] → 뱅크 충돌 감지
   */
  unsigned shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);  // [한국어] (바이트주소/워드크기) % 뱅크수 = 뱅크 번호
  }
  unsigned mem_warp_parts;
  /* [한국어] 메모리 워프 분할 수(memory warp partition count).
   * 하나의 워프 메모리 요청을 몇 개의 파트로 분할하여 처리할지 결정.
   * 설정자: gpgpusim.config. 읽는 자: 메모리 접근 스케줄링 로직. */
  mutable unsigned gpgpu_shmem_size;
  /* [한국어] 현재 SM에서 실제로 사용하는 공유 메모리 크기(바이트).
   * mutable: adaptive_cache_config 모드에서 const 함수에서도 동적으로 변경 가능.
   * 설정자: init() 또는 adaptive cache config 적용 시.
   * 읽는 자: SM의 공유 메모리 할당 로직, 점유율 계산. */
  char *gpgpu_shmem_option;
  /* [한국어] 공유 메모리 크기 옵션 문자열 (gpgpusim.config 파싱용).
   * 형식: 쉼표로 구분된 크기 목록 (예: "16384,32768,49152").
   * 설정자: option_parser에서 -gpgpu_shmem_size 옵션 파싱. 읽는 자: init()에서 shmem_opt_list 생성. */
  std::vector<unsigned> shmem_opt_list;
  /* [한국어] 사용 가능한 공유 메모리 크기 옵션 목록.
   * gpgpu_shmem_option 문자열을 파싱하여 init()에서 생성됨.
   * adaptive_cache_config 모드에서 커널의 FuncCache 선호도에 따라 이 목록 중 하나를 선택. */
  unsigned gpgpu_shmem_sizeDefault;
  /* [한국어] 기본 공유 메모리 크기(바이트). FuncCache 선호도 미설정 시 사용.
   * 설정자: gpgpusim.config 파싱. (unsigned)-1이면 미설정 상태. */
  unsigned gpgpu_shmem_sizePrefL1;
  /* [한국어] L1 캐시 선호(cudaFuncCachePreferL1) 시 공유 메모리 크기(바이트).
   * L1을 크게 하면 그만큼 공유 메모리가 줄어드는 교환(trade-off) 관계.
   * 설정자: gpgpusim.config. (unsigned)-1이면 미설정. */
  unsigned gpgpu_shmem_sizePrefShared;
  /* [한국어] 공유 메모리 선호(cudaFuncCachePreferShared) 시 공유 메모리 크기(바이트).
   * L1을 줄이고 공유 메모리를 크게 할당하는 모드.
   * 설정자: gpgpusim.config. (unsigned)-1이면 미설정. */
  unsigned mem_unit_ports;
  /* [한국어] 메모리 유닛의 포트(접속 통로) 수.
   * 한 사이클에 메모리 유닛이 처리할 수 있는 최대 요청 수를 제한.
   * 설정자: gpgpusim.config. 읽는 자: 메모리 유닛 스케줄링 로직. */

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  // [한국어] 텍스처/상수 캐시 라인 크기 — 메모리 요청 수 계산 시 접근 단위를 결정
  unsigned gpgpu_cache_texl1_linesize;
  /* [한국어] 텍스처 L1 캐시(tex$)의 캐시 라인 크기(바이트).
   * 설정자: gpgpusim.config의 텍스처 캐시 파라미터.
   * 읽는 자: mem_access_t 생성 시 텍스처 접근이 몇 개의 캐시 라인에 걸치는지 계산. */
  unsigned gpgpu_cache_constl1_linesize;
  /* [한국어] 상수 L1 캐시(const$)의 캐시 라인 크기(바이트).
   * 설정자: gpgpusim.config의 상수 캐시 파라미터.
   * 읽는 자: 상수 메모리 접근의 mem_access_t 생성 시 라인 수 계산. */

  unsigned gpgpu_max_insn_issue_per_warp;
  /* [한국어] 워프당 한 사이클에 발행(issue)할 수 있는 최대 명령어 수.
   * 표준: 1 (in-order 발행). 2 이상이면 한 사이클에 2개 명령어 동시 발행(dual-issue) 가능.
   * 설정자: gpgpusim.config의 -gpgpu_max_insn_issue_per_warp.
   * 읽는 자: shader_core_ctx의 issue stage — 워프당 발행 횟수 제한. */
  bool gmem_skip_L1D;
  /* [한국어] 글로벌 메모리 접근 시 L1 데이터 캐시를 항상 우회할지 여부.
   * true: 모든 글로벌 메모리 접근이 L1 캐시를 거치지 않고 L2로 직행 (bypass).
   * PTX의 .cg(cache global) 또는 .cs(cache streaming) 힌트와 유사한 효과.
   * false: 기본 동작 — 글로벌 메모리 접근이 L1을 거침.
   * 설정자: gpgpusim.config의 -gpgpu_gmem_skip_L1D. */
  // on = global memory access always skip the L1 cache

  bool adaptive_cache_config;
  /* [한국어] 적응형 캐시 설정 사용 여부.
   * true: 커널의 FuncCache 선호도(PreferShared/PreferL1/PreferNone)에 따라
   *   gpgpu_shmem_size를 동적으로 gpgpu_shmem_sizeDefault/PrefL1/PrefShared 중 하나로 설정.
   * false: gpgpusim.config에 고정된 캐시 크기만 사용.
   * 설정자: gpgpusim.config의 adaptive_cache_config 옵션. */
};

/*
 * ==========================================================================
 * SIMT 스택 관련 타입 정의
 * ==========================================================================
 * SIMT(Single Instruction, Multiple Threads): 하나의 명령어로 여러 스레드를 동시에 실행.
 * GPU의 핵심 실행 방식으로, 워프 내 32개 스레드가 동일한 PC에서 동일한 명령어를 실행한다.
 *
 * 워프 내 스레드들이 if/else, switch 같은 분기(branch)를 만나면 일부만 활성화되어
 * 실행 경로가 갈라진다(diverge). SIMT 스택은 각 경로의 PC·활성 마스크를 저장하고,
 * 포스트 도미네이터(post-dominator) PC에서 모든 스레드를 다시 합친다(reconverge).
 * 이 알고리즘은 MICRO 2007 논문 "Dynamic Warp Formation and Scheduling"에서 기원한다.
 *
 * active_mask_t:   현재 활성인 스레드를 나타내는 32비트 비트마스크 (bitset<32>).
 * simt_mask_t:     SIMT 스택 내부에서 사용하는 스레드 마스크 타입 (MAX_WARP_SIZE_SIMT_STACK 크기).
 * addr_vector_t:   스레드별 다음 PC 주소 배열 — update()에서 발산 경로 분석에 사용.
 */

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
// [한국어] MICRO'07 논문의 포스트 도미네이터(pdom) 재수렴 메커니즘 기반 SIMT 스택
const unsigned MAX_WARP_SIZE = 32;
/* [한국어] 최대 워프 크기: 32 스레드 (NVIDIA GPU 아키텍처 표준).
 * active_mask_t(bitset<MAX_WARP_SIZE>)의 크기와 일치해야 한다.
 * core_config::warp_size도 이 값을 초과할 수 없다. */

/*
 * [한국어] active_mask_t - 워프 내 활성 스레드를 나타내는 32비트 비트마스크
 *
 * bitset<32>의 각 비트가 하나의 스레드에 대응한다 (1=활성, 0=비활성).
 * 예: 0xFFFFFFFF(all 1s) → 32개 스레드 모두 활성 (분기 발산 없음).
 *     0x0000FFFF → 하위 16개 스레드만 활성 (발산 상태).
 *
 * 사용처:
 * - simt_stack::get_active_mask(): 현재 사이클에서 실행할 스레드 집합 반환.
 * - warp_inst_t::active_mask: 명령어 발행 시 활성 스레드 집합.
 * - shader.cc의 fetch/decode/execute 단계에서 매 사이클 참조됨.
 */
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;

#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE
/* [한국어] SIMT 스택 내부에서 사용하는 워프 크기 상수.
 * MAX_WARP_SIZE와 동일한 값(32)이지만, SIMT 스택 전용 매크로로 분리하여
 * 향후 스택 크기를 독립적으로 변경할 수 있도록 추상화. */
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
/* [한국어] simt_mask_t - SIMT 스택 항목 내에서 사용하는 스레드 비트마스크 타입.
 * active_mask_t와 동일한 bitset<32>이지만, SIMT 스택 전용 타입으로 분리.
 * simt_stack_entry::m_active_mask, update()의 @thread_done 인자 등에 사용. */
typedef std::vector<address_type> addr_vector_t;
/* [한국어] addr_vector_t - 스레드별 주소(PC) 배열 타입.
 * 크기: warp_size (32개 원소).
 * simt_stack::update()의 @next_pc 인자로 전달되며, 발산 시 각 스레드의 다음 PC를 담는다.
 * 모든 원소가 같은 값이면 발산 없음(수렴 상태). 다른 값이 있으면 발산(diverge) 상태. */

/*
 * ==========================================================================
 * simt_stack - 워프의 SIMT 분기 발산·재수렴 관리 스택
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * 포스트 도미네이터(post-dominator) 기반 SIMT 재수렴 메커니즘을 구현한다.
 * (참고 논문: MICRO 2007 — "Dynamic Warp Formation and Scheduling", pdom 알고리즘)
 * 워프 내 스레드들이 조건 분기(if/else, switch)를 만나면 일부 스레드만 활성화되어
 * 실행 경로가 갈라진다(diverge). 스택에 재수렴 지점(reconvergence PC)을 저장하고,
 * 각 경로를 순서대로 실행한 뒤 재수렴 지점에서 모든 스레드를 다시 합친다(reconverge).
 *
 * === 전체 아키텍처에서의 위치 ===
 * - 매 사이클 shader_core_ctx::fetch()에서 get_active_mask()/get_pdom_stack_top_info()로
 *   현재 실행할 PC와 활성 마스크를 조회한다.
 * - 분기 명령어 실행 후 shader_core_ctx::execute()에서 update()를 호출해 스택을 갱신한다.
 * - launch()는 커널 시작 시 1회 호출, reset()은 재초기화 시 사용.
 * - 하나의 SM에 MAX_WARP_PER_SM개 simt_stack 인스턴스가 할당됨.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: gpgpu_sim(m_gpu 역방향 포인터, 사이클/통계 접근)
 * - 피의존: shader.cc(fetch/execute 단계), core_t::initilizeSIMTStack()
 * - deque(덱) 자료구조로 m_stack을 구현 — 앞쪽이 스택 맨 위(top), 뒤쪽이 맨 아래(bottom).
 *
 * === 주요 함수/구조체 요약 ===
 * launch():            커널 시작 시 스택 초기화. start_pc와 활성 마스크로 첫 항목 설정.
 * update():            분기 명령어 실행 후 발산·재수렴 처리. 핵심 pdom 알고리즘 구현.
 * get_active_mask():   현재 사이클에서 실행할 스레드 마스크 반환 (스택 top의 m_active_mask).
 * get_pdom_stack_top_info(): 스택 top의 PC와 재수렴 PC를 반환.
 * simt_stack_entry:    스택의 각 항목 — PC, 활성 마스크, 재수렴 PC, 분기 사이클, 타입.
 */
class simt_stack {
 public:
  /*
   * [한국어]
   * simt_stack - SIMT 스택 생성자
   *
   * @wid: 이 스택이 관리할 워프의 ID (0 ~ MAX_WARP_PER_SM-1).
   * @warpSize: 워프 크기 (보통 32). core_config::warp_size와 일치해야 함.
   * @gpu: GPU 시뮬레이터 객체 포인터. 사이클 번호 조회와 통계 접근에 사용.
   *
   * m_stack을 비우고 m_warp_id, m_warp_size, m_gpu를 초기화한다.
   *
   * 호출 체인:
   *   shader.cc::shader_core_ctx 생성자 → core_t::initilizeSIMTStack() → [simt_stack()]
   */
  simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);

  /*
   * [한국어]
   * reset - SIMT 스택을 초기 빈 상태로 리셋
   *
   * m_stack을 clear()하여 모든 항목을 제거한다.
   * 워프가 재사용될 때(새 CTA 배정) 이전 분기 상태를 제거하기 위해 호출됨.
   *
   * 호출 체인:
   *   shader.cc::shader_core_ctx::init_warps() → [reset()]
   */
  void reset();

  /*
   * [한국어]
   * launch - 커널 시작 시 SIMT 스택을 초기화하고 시작 상태를 설정
   *
   * @start_pc: 커널의 시작 프로그램 카운터 주소 (PTX/SASS의 첫 명령어 주소).
   * @active_mask: 워프 내 활성 스레드 마스크. 커널 시작 시 보통 모든 비트가 1(all active).
   *
   * 스택을 리셋하고 (start_pc, active_mask, recvg_pc=-1)을 첫 항목으로 스택에 쌓는다.
   * 재수렴 PC는 유효하지 않은 값(-1)으로 초기화 (최상위 레벨은 재수렴 없음).
   *
   * 호출 체인:
   *   shader.cc::issue_block2core() → core_t::initilizeSIMTStack() → [launch()]
   */
  void launch(address_type start_pc, const simt_mask_t &active_mask);

  /*
   * [한국어]
   * update - 분기 명령어 실행 후 SIMT 스택 상태를 갱신 (핵심 pdom 알고리즘)
   *
   * @thread_done: 이 사이클에서 EXIT 명령어를 실행하여 완료된 스레드 마스크.
   * @next_pc: 각 스레드의 다음 PC 주소 벡터 (스레드마다 다를 수 있음 — 발산 상태).
   *   크기: warp_size (32개 원소).
   * @recvg_pc: 이 분기의 포스트 도미네이터 PC (재수렴 지점).
   *   컴파일러/런타임이 계산하여 PTX 분기 명령어에 포함.
   * @next_inst_op: 다음 명령어의 연산 타입. EXIT_OPS이면 스레드 종료 처리.
   * @next_inst_size: 다음 명령어 크기 (바이트).
   * @next_inst_pc: 다음 명령어 PC.
   * @return: 없음. m_stack이 직접 수정된다.
   *
   * 알고리즘 개요:
   * 1. thread_done에 해당하는 스레드를 현재 마스크에서 제거.
   * 2. next_pc 벡터를 분석하여 몇 개의 발산 경로가 있는지 파악.
   * 3. 재수렴 PC(recvg_pc)를 스택에 push (합류 항목).
   * 4. 각 발산 경로를 역순으로 스택에 push (활성 마스크 포함).
   * 5. 다음 사이클부터 스택 top의 경로가 실행됨 — 재수렴 PC 도달 시 스택 pop & 마스크 합침.
   *
   * 잘못된 구현은 라이브록(livelock) 또는 잘못된 활성 마스크로 이어질 수 있다.
   *
   * 호출 체인:
   *   shader.cc::execute() → core_t::updateSIMTStack() → [update()]
   */
  void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
              address_type recvg_pc, op_type next_inst_op,
              unsigned next_inst_size, address_type next_inst_pc);

  /*
   * [한국어]
   * get_active_mask - 현재 사이클에서 실행할 스레드 마스크를 반환
   *
   * @return: m_stack 맨 위 항목(front)의 m_active_mask. 이 비트마스크가 1인 스레드들만
   *   이번 사이클에 명령어를 실행한다.
   *
   * 매 사이클 shader_core_ctx의 fetch/decode 단계에서 호출된다.
   *
   * 호출 체인:
   *   shader.cc::fetch() → [get_active_mask()]
   */
  const simt_mask_t &get_active_mask() const;
  /*
   * [한국어]
   * get_pdom_stack_top_info - 스택 맨 위 항목의 PC와 재수렴 PC를 출력 매개변수로 반환
   *
   * @pc: 현재 실행할 명령어 PC (out).
   * @rpc: 현재 분기의 재수렴 PC (out). -1이면 최상위 레벨.
   *
   * shader.cc의 fetch 단계에서 fetch할 PC를 결정하기 위해 호출된다.
   *
   * 호출 체인:
   *   shader.cc::fetch() → [get_pdom_stack_top_info()]
   */
  void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;
  /*
   * [한국어]
   * get_rp - 현재 스택 top의 재수렴 PC(reconvergence point)를 반환
   *
   * @return: m_stack front의 m_recvg_pc. -1이면 재수렴 없음(최상위).
   *
   * 통계 수집 및 분기 발산 분석에 사용된다.
   */
  unsigned get_rp() const;
  /*
   * [한국어]
   * print - 스택 내용을 파일에 출력 (디버깅·로그용)
   *
   * @fp: 출력 대상 파일 포인터.
   * 각 스택 항목의 PC, 재수렴 PC, 활성 마스크, 타입을 출력한다.
   */
  void print(FILE *fp) const;
  /*
   * [한국어]
   * resume - 체크포인트 파일에서 스택 상태를 복원
   *
   * @fname: 체크포인트 파일 경로.
   * print_checkpoint()로 저장된 상태를 다시 m_stack에 로드한다.
   * 시뮬레이션 체크포인트/재시작 기능에 사용.
   */
  void resume(char *fname);
  /*
   * [한국어]
   * print_checkpoint - 체크포인트 저장용으로 스택 상태를 파일에 출력
   *
   * @fout: 출력 대상 파일 포인터.
   * resume()이 읽을 수 있는 포맷으로 m_stack의 전체 내용을 직렬화한다.
   */
  void print_checkpoint(FILE *fout) const;

 protected:  // [한국어] 보호 영역: simt_stack 및 파생 클래스에서만 접근 가능
  unsigned m_warp_id;
  /* [한국어] 이 스택이 관리하는 워프의 ID.
   * 설정자: 생성자 인자. 변경되지 않는다.
   * 읽는 자: print(), resume(), print_checkpoint()에서 워프 식별에 사용. */
  unsigned m_warp_size;
  /* [한국어] 워프 크기 (보통 32). active_mask_t/simt_mask_t의 비트 수와 일치해야 함.
   * 설정자: 생성자 인자(core_config::warp_size로부터). 변경되지 않는다.
   * 읽는 자: update()에서 next_pc 벡터 크기 검증 및 마스크 처리. */

  /*
   * [한국어]
   * stack_entry_type - SIMT 스택 항목의 종류를 구분하는 enum
   *
   * STACK_ENTRY_TYPE_NORMAL: 일반 조건 분기(if/else, predication)를 위한 항목.
   *   update()에서 분기 발산·재수렴 처리 경로로 진입.
   * STACK_ENTRY_TYPE_CALL: PTX CALL 명령어를 처리하기 위한 항목.
   *   반환 주소(return address)를 저장하고, RET 명령어 시 pop 됨.
   *   타입에 따라 update()의 스택 조작 로직이 달라진다.
   */
  enum stack_entry_type {
    STACK_ENTRY_TYPE_NORMAL = 0,  // [한국어] 일반 조건 분기(if/else, predication) 항목
    STACK_ENTRY_TYPE_CALL         // [한국어] 함수 호출(CALL 명령어) 반환 주소 저장 항목
  };

  /*
   * [한국어]
   * simt_stack_entry - SIMT 스택의 각 항목에 저장되는 정보
   *
   * 분기가 발생할 때마다 새 항목이 스택 앞(front)에 추가된다.
   * 재수렴 PC에 도달하면 스택 front에서 pop되어 상위 항목의 마스크와 합쳐진다.
   */
  struct simt_stack_entry {
    address_type m_pc;
    /* [한국어] 이 스택 항목에서 실행할 명령어의 PC.
     * 분기 발산 시 각 경로의 시작 PC. 재수렴 항목에서는 합류 지점 PC.
     * 초기값: -1 (유효하지 않음 — launch() 전까지는 실행 불가). */

    unsigned int m_calldepth;
    /* [한국어] 함수 호출 깊이 카운터.
     * CALL 명령어를 만날 때마다 증가, RET 시 감소.
     * 재귀 호출이나 중첩 함수 호출에서 올바른 스택 항목을 식별하는 데 사용.
     * 초기값: 0 (커널 진입점 수준). */

    simt_mask_t m_active_mask;
    /* [한국어] 이 스택 항목에서 활성인 스레드 비트마스크(32비트).
     * get_active_mask()가 스택 top 항목의 이 값을 반환한다.
     * 분기 발산 후 각 경로에는 해당 경로를 따르는 스레드만 1로 설정됨.
     * 초기값: 모든 비트 0 (빈 마스크). launch() 또는 update()에서 실제 값 설정. */

    address_type m_recvg_pc;
    /* [한국어] 이 분기 항목의 재수렴(reconvergence) PC.
     * 이 PC에 도달한 스레드들은 스택에서 pop되어 상위 항목에 합류한다.
     * 초기값: -1 (최상위 레벨, 재수렴 없음 — 커널 완료까지 실행).
     * 컴파일러가 포스트 도미네이터 분석으로 계산하여 PTX에 포함시킨 값. */

    unsigned long long m_branch_div_cycle;
    /* [한국어] 이 분기 발산이 발생한 GPU 사이클 번호.
     * 통계 수집: 재수렴 사이클 - m_branch_div_cycle = 발산 지속 사이클 수.
     * 0은 발산이 없거나 측정하지 않음을 의미. */

    stack_entry_type m_type;
    /* [한국어] 이 스택 항목의 타입.
     * STACK_ENTRY_TYPE_NORMAL: 일반 조건 분기 처리.
     * STACK_ENTRY_TYPE_CALL: 함수 호출 반환 주소 저장.
     * 타입에 따라 update()의 스택 조작 로직(재수렴 vs 함수 복귀)이 달라진다. */

    simt_stack_entry()                       // [한국어] 기본 생성자: 모든 필드를 "미초기화" 상태로 설정
        : m_pc(-1),                          // [한국어] PC = -1 (유효하지 않은 주소 — 실수로 실행 방지)
          m_calldepth(0),                    // [한국어] 호출 깊이 0 (커널 진입점)
          m_active_mask(),                   // [한국어] 활성 마스크 = 0 (모든 스레드 비활성 초기화)
          m_recvg_pc(-1),                    // [한국어] 재수렴 PC = -1 (최상위 레벨, 재수렴 없음)
          m_branch_div_cycle(0),             // [한국어] 발산 사이클 = 0 (미측정)
          m_type(STACK_ENTRY_TYPE_NORMAL){}; // [한국어] 타입 = 일반 분기 (기본값)
  };

  std::deque<simt_stack_entry> m_stack;
  /* [한국어] SIMT 스택의 실제 저장소 — deque(덱)로 구현.
   * deque의 front()가 스택 top(현재 실행 중인 항목), back()이 스택 bottom.
   * 왜 deque인가: 스택 항목을 front에서 push_front/pop_front하면서 동시에
   *   전체 항목 순회(디버그 출력, 체크포인트)도 필요하기 때문.
   * 분기 발산 시: push_front()로 새 경로 항목 추가.
   * 재수렴 시: pop_front()로 완료된 항목 제거 + front의 마스크에 합류. */

  class gpgpu_sim *m_gpu;
  /* [한국어] GPU 시뮬레이터 객체를 가리키는 포인터.
   * 설정자: 생성자 인자. 변경되지 않는다.
   * 읽는 자: update()에서 현재 gpu_cycle 번호 조회 (m_branch_div_cycle 기록),
   *           통계 카운터 업데이트.
   * 역방향 포인터: simt_stack이 자신을 포함하는 gpu-sim 컨텍스트를 참조하는 패턴. */
};

/*
 * ==========================================================================
 * GPU 가상 주소 공간 레이아웃 상수들
 * ==========================================================================
 *
 * [한국어] GPGPU-Sim이 GPU 메모리 주소 공간을 분할하기 위해 사용하는 컴파일 시간 상수들.
 * GPU의 단일 32비트 주소 공간(4GB)을 글로벌·공유·로컬 메모리 영역으로 나눈다.
 *
 * 주소 공간 레이아웃 (주소 낮은 곳 → 높은 곳):
 *   [0x00000000 ~ STATIC_ALLOC_LIMIT]  : PTX .global 변수 정적 할당 영역
 *   [LOCAL_GENERIC_START ~ SHARED_GENERIC_START) : 로컬 메모리 제네릭 주소 범위
 *   [SHARED_GENERIC_START ~ GLOBAL_HEAP_START)   : 공유 메모리 제네릭 주소 범위
 *   [GLOBAL_HEAP_START ~ 0xFFFFFFFF]             : 동적 글로벌 메모리 (gpu_malloc)
 *
 * "제네릭(generic) 주소"란 PTX의 cvta 명령어로 변환된 단일 주소 공간 포인터를 말한다.
 * 포인터의 값이 어느 범위에 속하느냐에 따라 어느 메모리 공간인지 자동 판별된다.
 *
 * 읽는 자: gpgpu_t::gpu_malloc(), cuda-sim 주소 공간 분류 로직,
 *          ptx_thread_info의 로컬/공유 메모리 주소 계산.
 */

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;
/* [한국어] 글로벌 힙(동적 메모리 할당) 시작 가상 주소 = 0xC0000000(3GB).
 * gpu_malloc()이 이 주소부터 시작하여 위로 선형 할당(bump allocator)한다.
 * 이 값 아래 공간은 .ptx 파일의 .global 변수 정적 할당에 사용된다.
 * 설정자: 컴파일 시 상수. 읽는 자: gpgpu_t::gpu_malloc(), 주소 공간 분류 로직.
 * 값 선택 이유: 0xC0000000(3GB)은 32비트 가상 주소 공간의 상위 1/4로,
 *   PTX 전역 변수(정적)와 동적 할당 영역을 분리하기 위한 경계이다. */

// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
/* [한국어] SM당 최대 공유 메모리 크기 = 96KB (Volta GPU 기준, 96 * 1024바이트).
 * SHARED_GENERIC_START 계산의 기준값. 공유 메모리의 제네릭 주소 범위 크기를 결정.
 * Volta(V100/Titan V) 이전 GPU는 더 작은 값(Fermi: 48KB, Kepler: 48KB 등)을 가지지만,
 * 최대 지원 아키텍처 기준으로 보수적으로 96KB로 설정하여 주소 공간을 충분히 확보. */

// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;
/* [한국어] 스레드당 최대 로컬 메모리 크기 = 16384바이트(16KB, 1<<14).
 * 각 스레드의 로컬 메모리 영역 크기. TOTAL_LOCAL_MEM_PER_SM 계산에 사용.
 * PTX .local 변수나 레지스터 스필(spill) 데이터가 이 공간에 저장된다.
 * 실제 사용 가능 크기는 ptxinfo의 lmem 필드로 커널마다 다를 수 있다. */

// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;
/* [한국어] 시뮬레이터가 지원하는 최대 SM 수 = 80 (Volta Titan V 기준).
 * TOTAL_SHARED_MEM, TOTAL_LOCAL_MEM 배열/상수 크기 결정에 사용.
 * 실제 시뮬레이션 SM 수는 gpgpusim.config의 -gpgpu_n_cores 옵션으로 결정된다.
 * 이 값보다 많은 SM을 가진 GPU를 시뮬레이션하려면 이 상수와 관련 배열 크기 재조정 필요. */

// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;
/* [한국어] SM당 동시에 상주할 수 있는 최대 스레드 수 = 2048 (1<<11).
 * TOTAL_LOCAL_MEM_PER_SM = 2048 * 16KB = 32MB 로컬 메모리 공간 크기 결정.
 * 실제 점유 스레드 수(occupancy)는 레지스터/공유 메모리 사용량에 따라 달라진다.
 * MAX_WARP_PER_SM = MAX_THREAD_PER_SM / 32 = 64로 일관성을 유지해야 한다. */

// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;
/* [한국어] SM당 최대 워프 수 = 64 (= MAX_THREAD_PER_SM / 32 = 2048 / 32, 1<<6).
 * simt_stack 배열 크기, scoreboard 크기, register_set 크기 등 SM 내부 자료구조의
 * 상한으로 사용. 이 값을 초과하는 워프를 생성하면 배열 오버플로우 발생. */

/*
 * [한국어] 전체 메모리 크기 및 제네릭 주소 범위 계산.
 * 각 상수는 위에서 정의한 기본 상수들의 곱/차로 컴파일 시간에 계산된다.
 * 이 값들로 시뮬레이터의 메모리 주소 공간 분류 로직이 동작한다.
 */
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
/* [한국어] SM 하나에서 모든 스레드의 로컬 메모리 합계 = 2048 * 16KB = 32MB.
 * 로컬 메모리 주소 변환 시 SM별 기준 오프셋 계산에 사용된다.
 * 스레드의 로컬 메모리 주소 = LOCAL_GENERIC_START + sm_id * TOTAL_LOCAL_MEM_PER_SM
 *   + warp_id * warp_size * LOCAL_MEM_SIZE_MAX + thread_id * LOCAL_MEM_SIZE_MAX + offset. */

const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
/* [한국어] 전체 GPU의 공유 메모리 총합 = 80 * 96KB = 7,864,320바이트(≈7.5MB).
 * 제네릭 주소 공간에서 공유 메모리 영역의 총 크기를 결정한다.
 * SHARED_GENERIC_START = GLOBAL_HEAP_START - TOTAL_SHARED_MEM으로 시작 주소 계산. */

const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
/* [한국어] 전체 GPU의 로컬 메모리 총합 = 80 * 2048 * 16KB = 약 2.5GB.
 * LOCAL_GENERIC_START 계산에서 이 크기만큼 공유 메모리 시작점 아래에 할당한다.
 * 이 크기가 크기 때문에 STATIC_ALLOC_LIMIT가 주소 공간 하단으로 밀려 있어
 * PTX 전역 변수가 너무 많으면 충돌이 발생할 수 있다. */

const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
/* [한국어] 공유 메모리의 제네릭 가상 주소 시작점 = GLOBAL_HEAP_START - 7.5MB.
 * 제네릭 포인터 역참조 시 이 범위에 속하는 주소는 공유 메모리로 해석된다.
 * PTX의 cvta.to.global/shared 명령어로 변환되는 주소 범위.
 * 읽는 자: cuda-sim의 generic_to_shared() 주소 변환 함수. */

const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
/* [한국어] 로컬 메모리의 제네릭 가상 주소 시작점 = SHARED_GENERIC_START - 2.5GB.
 * 제네릭 포인터 역참조 시 이 범위에 속하는 주소는 로컬 메모리로 해석된다.
 * 읽는 자: cuda-sim의 generic_to_local() 주소 변환 함수. */

const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);
/* [한국어] PTX 전역 변수 정적 할당의 상한 주소(= LOCAL_GENERIC_START).
 * 이 주소 이하에 PTX .global 변수들이 정적으로 배치된다.
 * gpu_malloc()은 GLOBAL_HEAP_START부터 시작하므로 정적/동적 영역이 겹치지 않는다.
 * 커널의 정적 글로벌 변수 총합이 STATIC_ALLOC_LIMIT를 초과하면
 * 로컬/공유 메모리 주소 공간과 충돌하여 시뮬레이션 오류 발생. */

/*
 * [한국어] CUDA 런타임 API 헤더(__CUDA_RUNTIME_API_H__)가 이미 포함되어 있으면
 * cudaArray가 이미 정의되어 있으므로 중복 정의를 피하기 위해 조건부 컴파일한다.
 * 시뮬레이터 빌드 시에는 실제 CUDA 런타임이 없을 수 있으므로 자체 정의가 필요하다.
 */
#if !defined(__CUDA_RUNTIME_API_H__)

#include "builtin_types.h"
/* [한국어] CUDA 기본 타입(cudaChannelFormatDesc, cudaTextureReadMode 등)을 정의한 헤더.
 * CUDA 런타임 API 헤더 없이 빌드할 때 필요한 타입들을 제공한다.
 * 설치된 CUDA Toolkit의 include/builtin_types.h 또는 시뮬레이터 동봉 버전을 사용. */

/*
 * [한국어]
 * cudaArray - CUDA 텍스처·서피스 메모리에 사용되는 배열 구조체
 *
 * GPU 텍스처 메모리(tex$)에서 접근하는 2D/3D 이미지 데이터나 1D 버퍼 데이터를 담는다.
 * cudaBindTextureToArray() 또는 cudaBindSurfaceToArray()를 통해 텍스처/서피스 참조에 연결된다.
 * GPGPU-Sim에서는 gpgpu_t::gpu_mallocarray()로 할당되고 m_NameToCudaArray에 등록된다.
 *
 * 설정 흐름:
 *   libcuda/cuda_runtime_api.cc::cudaMallocArray() → gpu_mallocarray() → devPtr 설정
 *   libcuda/cuda_runtime_api.cc::cudaMemcpy2DToArray() → memcpy_to_gpu() → devPtr 위치에 데이터 기록
 *   libcuda/cuda_runtime_api.cc::cudaBindTextureToArray() → m_NameToCudaArray에 이름과 함께 등록
 */
struct cudaArray {
  void *devPtr;
  /* [한국어] GPU(디바이스) 메모리에서 이 배열 데이터의 시작 포인터.
   * 설정자: gpgpu_t::gpu_mallocarray()에서 gpu_malloc()으로 할당한 주소.
   * 읽는 자: cuda-sim의 tex 명령어 실행 시 실제 데이터 접근, memcpy 함수.
   * 값 범위: GLOBAL_HEAP_START 이상의 유효한 GPU 가상 주소 (NULL 불가). */

  int devPtr32;
  /* [한국어] 32비트 디바이스 포인터 표현 (레거시 32비트 CUDA 호환성용).
   * 64비트 포인터가 표준인 현재는 대부분 devPtr(64비트)을 사용한다.
   * 32비트 CUDA 드라이버 또는 일부 구형 API와의 호환성을 위해 유지된다. */

  struct cudaChannelFormatDesc desc;
  /* [한국어] 채널 형식 설명자: 각 채널(R, G, B, A)의 비트 수와 데이터 타입.
   * 설정자: cudaMallocArray() 호출 시 cudaCreateChannelDesc()로 생성한 값.
   * 읽는 자: 텍스처 페치 시 데이터 해석(정수/부동소수점, 정규화 여부) 결정.
   * 예: cudaCreateChannelDesc<float4>() → 각 채널 32비트, 부동소수점 타입. */

  int width;
  /* [한국어] 배열의 너비(픽셀 또는 원소 수). 1D/2D/3D 모두에서 첫 번째 차원.
   * 텍스처 좌표 계산 시 [0, width) 범위의 x 인덱스 경계값으로 사용.
   * 설정자: cudaMallocArray()의 width 인자. */

  int height;
  /* [한국어] 배열의 높이(픽셀 수). 2D/3D 배열에서만 의미 있음. 1D이면 0 또는 1.
   * 텍스처 좌표 계산 시 [0, height) 범위의 y 인덱스 경계값으로 사용.
   * 설정자: cudaMallocArray()의 height 인자. */

  int size;  // in bytes
  /* [한국어] 배열 전체 크기(바이트) = width * height * (채널 비트 수 합 / 8).
   * 메모리 할당(gpu_mallocarray)과 복사(memcpy) 크기 결정에 사용.
   * 설정자: gpu_mallocarray()에서 count 인자로 직접 전달. */

  unsigned dimensions;
  /* [한국어] 배열의 차원 수: 1(1D 텍스처), 2(2D 텍스처), 3(3D 텍스처).
   * 텍스처 좌표 계산(1D: x만, 2D: x+y, 3D: x+y+z)과
   * textureReferenceAttr::m_dim과 일치해야 정상 동작.
   * 설정자: cudaMallocArray() 또는 cudaMalloc3DArray()에서 결정. */
};

#endif

/*
 * [한국어]
 * textureReferenceAttr - textureReference 선언의 추가 속성을 담는 구조체
 *
 * __cudaRegisterTexture() 함수를 통해 전달되는 텍스처 차원/읽기모드/확장 속성들을
 * textureReference 포인터와 함께 묶어서 관리한다.
 * gpgpu_t::m_NameToAttribute에 텍스처 이름과 매핑되어 저장된다.
 *
 * 설정 흐름:
 *   CUDA 런타임 초기화 시 __cudaRegisterTexture() → gpgpu_ptx_sim_bindNameToTexture()
 *   → textureReferenceAttr 생성 → m_NameToAttribute에 저장
 *   읽는 자: cuda-sim의 tex 명령어 실행 시 get_texattr()로 차원/읽기모드 조회.
 */
// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
  const struct textureReference *m_texref;
  /* [한국어] 이 속성이 연결된 textureReference 포인터.
   * __cudaRegisterTexture()를 통해 등록된 PTX .tex 선언에 대응하는 구조체.
   * 설정자: textureReferenceAttr 생성자의 texref 인자.
   * 읽는 자: get_texattr() 후 m_texref로 실제 필터/주소 모드 접근. */

  int m_dim;
  /* [한국어] 텍스처 차원 (1, 2, 3).
   * PTX tex.1d / tex.2d / tex.3d 명령어 선택 및 텍스처 좌표 계산 기준.
   * 설정자: __cudaRegisterTexture()의 dim 인자.
   * cudaArray::dimensions 와 일치해야 올바른 텍스처 접근이 된다. */

  enum cudaTextureReadMode m_readmode;
  /* [한국어] 텍스처 읽기 모드.
   * cudaReadModeElementType: 텍스처 데이터를 원래 타입(int, float 등)으로 반환.
   * cudaReadModeNormalizedFloat: 정수형 데이터를 0.0~1.0 범위의 float으로 정규화하여 반환.
   * 설정자: __cudaRegisterTexture()의 readmode 인자.
   * 읽는 자: cuda-sim의 tex_impl()에서 결과값 타입 변환 방식 결정. */

  int m_ext;
  /* [한국어] 확장 플래그. 레이어드 텍스처(layered texture)나 CUDA 확장 기능 사용 여부.
   * 현재는 __cudaRegisterTexture()에서 전달되는 미래 확장용 예비 필드.
   * 설정자: __cudaRegisterTexture()의 ext 인자. 현재 대부분 0. */

  /*
   * [한국어]
   * textureReferenceAttr - 텍스처 속성 생성자
   *
   * @texref: 대응하는 textureReference 포인터.
   * @dim: 텍스처 차원 (1/2/3).
   * @readmode: 읽기 모드 (ElementType 또는 NormalizedFloat).
   * @ext: 확장 플래그 (보통 0).
   *
   * 초기화 리스트(member initializer list)로 모든 필드를 즉시 초기화한다.
   * 생성자 본문 내 대입보다 효율적이며, const 멤버를 포함하는 경우 필수이다.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_bindNameToTexture() → [new textureReferenceAttr(texref, dim, readmode, ext)]
   */
  textureReferenceAttr(const struct textureReference *texref, int dim,
                       enum cudaTextureReadMode readmode, int ext)
      : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};

/*
 * ==========================================================================
 * gpgpu_functional_sim_config - GPU 기능 시뮬레이션 설정 클래스
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * PTX 기능 시뮬레이션(cuda-sim/)에 필요한 설정값들을 관리한다.
 * PTX→PTXPlus 변환 여부, cuobjdump 사용, 최대 컴퓨트 능력, 텍스처 캐시 라인 크기,
 * 체크포인트/재개 옵션, PTX 디버그 파일 경로 설정 등을 포함한다.
 * gpgpu_t의 m_function_model_config 참조 멤버로 보유되어 기능 시뮬레이션 전반에서 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 시뮬레이션 초기화 단계에서 1회 파싱되어 이후 읽기 전용으로 사용된다.
 * 기능 시뮬레이션 레이어(cuda-sim/)와 타이밍 시뮬레이션 레이어(gpgpu-sim/) 모두에서 참조된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: option_parser.h — reg_options()로 gpgpusim.config 옵션과 멤버 변수 바인딩.
 * - 피의존: gpgpu_t(m_function_model_config 참조), cuda-sim/ptx_sim.cc(PTX 실행 설정 조회).
 *
 * === 주요 함수 요약 ===
 * reg_options():              gpgpusim.config 옵션을 option_parser에 등록 (시뮬레이터 초기화 시 1회).
 * ptx_set_tex_cache_linesize(): 텍스처 캐시 라인 크기를 동적으로 설정.
 * get_forced_max_capability(): PTX 컴파일 능력 강제값 반환.
 * convert_to_ptxplus():       PTXPlus 변환 여부 반환.
 * use_cuobjdump():            cuobjdump 사용 여부 반환.
 * get_checkpoint_option():    체크포인트 저장/복원 모드 반환.
 */
class gpgpu_functional_sim_config {
 public:
  /*
   * [한국어]
   * reg_options - 기능 시뮬레이션 관련 옵션을 option_parser에 등록
   *
   * @opp: gpgpusim.config 파일을 파싱하는 OptionParser 포인터.
   *
   * -ptx_convert_to_ptxplus, -gpgpu_ptx_use_cuobjdump, -gpgpu_ptx_force_max_capability,
   * -checkpoint_option, -g_ptx_inst_debug_to_file 등의 옵션을 등록하여
   * 설정 파일 파싱 시 해당 멤버 변수에 자동으로 값이 채워지도록 한다.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint.cc::GPGPUSim_Init() → [reg_options()]
   */
  void reg_options(class OptionParser *opp);

  /*
   * [한국어]
   * ptx_set_tex_cache_linesize - 텍스처 캐시 라인 크기를 동적으로 설정
   *
   * @linesize: 새 텍스처 캐시 라인 크기(바이트). core_config에서 파싱된 값으로 호출됨.
   *
   * m_texcache_linesize를 설정한다. 기능 시뮬레이션에서 텍스처 접근 시
   * 몇 개의 캐시 라인에 걸치는지 계산하는 데 사용된다.
   *
   * 호출 체인:
   *   shader_core_config::init() → [ptx_set_tex_cache_linesize()]
   */
  void ptx_set_tex_cache_linesize(unsigned linesize);

  // [한국어] 설정값 조회용 getter 함수들. 모두 const이므로 객체 상태를 변경하지 않는다.

  unsigned get_forced_max_capability() const {
    /* [한국어] PTX 컴파일 타겟 GPU 능력(compute capability)의 강제 설정값을 반환.
     * 예: 70 → Volta sm_70. 0이면 자동 감지. */
    return m_ptx_force_max_capability;
  }
  bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
  /* [한국어] PTX를 PTXPlus(GPGPU-Sim 내부 SASS 근사 형식)로 변환할지 여부를 반환.
   * PTXPlus는 레지스터 할당 정보를 포함하여 SASS 수준 실행에 가까운 시뮬레이션을 가능하게 함. */

  bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
  /* [한국어] CUDA 바이너리에서 PTX를 추출할 때 cuobjdump 도구를 사용할지 여부.
   * false이면 런타임 인터셉트 방식(execution-driven), true이면 cuobjdump 출력 파일 방식. */

  bool experimental_lib_support() const { return m_experimental_lib_support; }
  /* [한국어] 실험적 라이브러리 지원 여부. 특정 CUDA 라이브러리 함수의 시뮬레이션 활성화. */

  int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
  /* [한국어] PTX 명령어 실행 디버그 로그를 파일에 출력할지 여부(0=비활성, 1=활성).
   * 활성화 시 모든 PTX 명령어 실행 이력을 g_ptx_inst_debug_file에 기록한다. */

  const char *get_ptx_inst_debug_file() const { return g_ptx_inst_debug_file; }
  /* [한국어] PTX 명령어 디버그 로그 파일 경로 문자열 반환. */

  int get_ptx_inst_debug_thread_uid() const {
    /* [한국어] 디버그 로그를 출력할 특정 스레드의 고유 UID(thread_uid).
     * 특정 스레드만 필터링하여 디버그 출력. -1이면 모든 스레드. */
    return g_ptx_inst_debug_thread_uid;
  }
  unsigned get_texcache_linesize() const { return m_texcache_linesize; }
  /* [한국어] 텍스처 캐시(tex$) 라인 크기(바이트) 반환.
   * 텍스처 접근의 mem_access_t 생성 시 몇 개의 캐시 라인에 걸치는지 계산하는 데 사용. */

  // [한국어] 체크포인트/재개 설정 getter들.
  // 체크포인트: 특정 커널/CTA 실행 시점의 메모리 상태를 파일에 저장하는 기능.
  // 재개(resume): 저장된 체크포인트에서 시뮬레이션을 이어서 실행하는 기능.
  int get_checkpoint_option() const { return checkpoint_option; }
  /* [한국어] 체크포인트 동작 모드: 0=없음, 1=저장, 2=저장+로드. */

  int get_checkpoint_kernel() const { return checkpoint_kernel; }
  /* [한국어] 체크포인트를 저장할 커널 번호 (0-based). */

  int get_checkpoint_CTA() const { return checkpoint_CTA; }
  /* [한국어] 체크포인트를 저장할 CTA의 선형 ID. */

  int get_resume_option() const { return resume_option; }
  /* [한국어] 재개 모드: 0=처음부터 실행, 1=체크포인트에서 재개. */

  int get_resume_kernel() const { return resume_kernel; }
  /* [한국어] 재개를 시작할 커널 번호. */

  int get_resume_CTA() const { return resume_CTA; }
  /* [한국어] 재개를 시작할 CTA ID. */

  int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }
  /* [한국어] 체크포인트 저장 시 CTA 내 실행 단계(transaction) 번호. */

  int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }
  /* [한국어] 체크포인트 저장 시 기준이 되는 명령어 카운트(Y 번째 명령어 이후 저장). */

 private:  // [한국어] 비공개 멤버: option_parser를 통해서만 설정되며 getter로만 읽힌다.

  // PTX options
  // [한국어] PTX 기능 시뮬레이션 동작 방식 옵션들
  int m_ptx_convert_to_ptxplus;
  /* [한국어] PTX를 PTXPlus로 변환할지 여부 (0=비활성, 1=활성).
   * 설정자: gpgpusim.config의 -ptx_convert_to_ptxplus.
   * 읽는 자: convert_to_ptxplus() getter → cuda-sim 초기화 코드. */

  int m_ptx_use_cuobjdump;
  /* [한국어] cuobjdump 도구를 사용해 바이너리에서 PTX를 추출할지 여부 (0/1).
   * 설정자: gpgpusim.config의 -gpgpu_ptx_use_cuobjdump. */

  int m_experimental_lib_support;
  /* [한국어] 실험적 CUDA 라이브러리 함수 지원 여부 (0/1).
   * 설정자: gpgpusim.config의 -gpgpu_experimental_lib_support. */

  unsigned m_ptx_force_max_capability;
  /* [한국어] PTX 컴파일 타겟 GPU 능력(compute capability)을 강제 설정.
   * 예: 70 = Volta(sm_70). 0이면 자동 감지.
   * 더 낮은 값으로 설정하면 해당 세대의 PTX 명령어 집합으로 제한된다.
   * 설정자: gpgpusim.config의 -gpgpu_ptx_force_max_capability. */

  int checkpoint_option;
  /* [한국어] 체크포인트 동작 모드: 0=없음, 1=저장, 2=저장+로드.
   * 설정자: gpgpusim.config의 checkpoint_option. */

  int checkpoint_kernel;
  /* [한국어] 체크포인트를 저장할 커널의 번호(0-based).
   * 이 번호의 커널 시작 시 메모리 상태를 파일에 저장한다. */

  int checkpoint_CTA;
  /* [한국어] 체크포인트를 저장할 CTA의 선형 ID.
   * 특정 CTA 시작 시 메모리 상태를 덤프(dump)한다. */

  unsigned resume_option;
  /* [한국어] 재개 모드: 0=처음부터 실행, 1=체크포인트에서 재개.
   * 체크포인트 파일에서 메모리 상태를 복원하고 resume_kernel/CTA부터 실행. */

  unsigned resume_kernel;
  /* [한국어] 체크포인트 재개를 시작할 커널 번호. */

  unsigned resume_CTA;
  /* [한국어] 체크포인트 재개를 시작할 CTA ID. */

  unsigned checkpoint_CTA_t;
  /* [한국어] 체크포인트 저장 시 CTA 내 실행 단계 번호. */

  int checkpoint_insn_Y;
  /* [한국어] 체크포인트 저장 기준 명령어 카운트. Y번째 명령어 이후 상태 저장. */

  int g_ptx_inst_debug_to_file;
  /* [한국어] PTX 명령어 실행 이력을 파일에 출력할지 여부 (0=비활성, 1=활성).
   * 설정자: gpgpusim.config의 -gpgpu_ptx_inst_debug_to_file. */

  char *g_ptx_inst_debug_file;
  /* [한국어] PTX 명령어 디버그 로그 파일 경로(문자열 포인터).
   * 설정자: gpgpusim.config의 -gpgpu_ptx_inst_debug_file.
   * 읽는 자: cuda-sim 초기화 시 디버그 파일 열기(fopen). */

  int g_ptx_inst_debug_thread_uid;
  /* [한국어] 디버그 로그를 출력할 특정 스레드 UID. -1이면 모든 스레드.
   * 특정 스레드의 실행 이력만 추적할 때 유용. */

  unsigned m_texcache_linesize;
  /* [한국어] 텍스처 캐시 라인 크기(바이트).
   * 설정자: ptx_set_tex_cache_linesize() — shader_core_config::init()에서 호출.
   * 읽는 자: get_texcache_linesize() → 텍스처 접근 mem_access_t 생성 시 캐시 라인 수 계산. */
};

/*
 * ==========================================================================
 * gpgpu_t - GPU 시뮬레이터 기반 클래스 (기능 시뮬레이션 담당)
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * GPU 메모리 공간(글로벌·텍스처·서피스 메모리) 관리와 텍스처 바인딩 테이블을 담는
 * 기반 클래스이다. gpgpu_sim이 이 클래스를 상속하여 타이밍 시뮬레이션 기능을 추가한다.
 * GPU 메모리 할당(gpu_malloc), CPU↔GPU 메모리 복사, 텍스처 이름↔데이터 매핑을 담당한다.
 * "기능 시뮬레이션" 맥락에서 실행된다 — 단일 메인 스레드에서만 접근.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인: libcuda/cuda_runtime_api.cc(CUDA API 인터셉트)
 *              → gpgpu_t(메모리 API, 텍스처 바인딩)
 *              → memory_space(실제 저장)
 *            gpgpu_sim(gpgpu_t 상속)
 *              → shader.cc → mem_fetch → gpu-cache → dram
 *
 * === 타 모듈과의 연결 ===
 * - 의존: gpgpu_functional_sim_config(설정 참조), memory_space(메모리 구현), gpgpu_context
 * - 피의존: gpgpu_sim(상속), libcuda/cuda_runtime_api.cc(CUDA API)
 * - 공유 자료: m_NameToCudaArray 등 텍스처 맵을 kernel_info_t 생성 시 스냅샷으로 복사
 *
 * === 주요 함수 요약 ===
 * gpu_malloc():                  글로벌 메모리 bump 할당자 (GLOBAL_HEAP_START부터 선형 증가).
 * memcpy_to_gpu():               CPU→GPU 메모리 복사 (cudaMemcpyHostToDevice 구현).
 * memcpy_from_gpu():             GPU→CPU 메모리 복사 (cudaMemcpyDeviceToHost 구현).
 * gpgpu_ptx_sim_bindTextureToArray(): 텍스처 참조를 CUDA 배열에 연결.
 * gpgpu_ptx_sim_bindNameToTexture():  텍스처 이름을 텍스처 참조+속성에 연결.
 * get_texref()/get_texarray():   이름으로 텍스처 메타데이터 조회.
 */
class gpgpu_t {
 public:
  /*
   * [한국어]
   * gpgpu_t - 기능 시뮬레이션 기반 클래스 생성자
   *
   * @config: 기능 시뮬레이션 설정 const 참조. m_function_model_config에 저장됨.
   * @ctx: GPU 시뮬레이터 전역 컨텍스트 포인터.
   *
   * m_global_mem, m_tex_mem, m_surf_mem을 각 메모리 공간에 맞는 크기로 생성한다.
   * m_dev_malloc을 GLOBAL_HEAP_START(0xC0000000)로 초기화하여 bump allocator를 준비.
   * checkpoint/resume 설정값을 config에서 복사하여 멤버 변수에 저장한다.
   *
   * 호출 체인:
   *   gpgpu_sim 생성자 → [gpgpu_t()]
   */
  gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] GPU 시뮬레이터 전역 컨텍스트를 가리키는 역방향 포인터.
   * 설정자: 생성자에서 ctx 인자로 초기화. 변경되지 않는다.
   * 읽는 자: mem_fetch 생성 시 컨텍스트 접근, uid 카운터 조회. */

  // [한국어] 체크포인트/재개 설정값들 — gpgpu_functional_sim_config에서 복사되어 직접 접근 가능.
  // 복사 이유: const 참조(m_function_model_config)를 통하지 않고 빠르게 접근하기 위함.
  int checkpoint_option;
  /* [한국어] 체크포인트 동작 모드: 0=없음, 1=저장, 2=저장+로드.
   * 설정자: gpgpu_t 생성자에서 config에서 복사. 읽는 자: cuda-sim 체크포인트 저장 로직. */
  int checkpoint_kernel;
  /* [한국어] 체크포인트를 저장할 커널 번호. config에서 복사된 값. */
  int checkpoint_CTA;
  /* [한국어] 체크포인트를 저장할 CTA ID. config에서 복사된 값. */
  unsigned resume_option;
  /* [한국어] 재개 모드 (0=처음부터, 1=체크포인트에서). config에서 복사된 값. */
  unsigned resume_kernel;
  /* [한국어] 재개를 시작할 커널 번호. config에서 복사된 값. */
  unsigned resume_CTA;
  /* [한국어] 재개를 시작할 CTA ID. config에서 복사된 값. */
  unsigned checkpoint_CTA_t;
  /* [한국어] 체크포인트 저장 CTA 내 실행 단계 번호. config에서 복사된 값. */
  int checkpoint_insn_Y;
  /* [한국어] 체크포인트 저장 기준 명령어 카운트. config에서 복사된 값. */

  // Move some cycle core stats here instead of being global
  // [한국어] 전역 변수 대신 이 클래스에 사이클 통계를 저장 — 멀티-GPU 시뮬레이션을 위한 인스턴스 분리.
  unsigned long long gpu_sim_cycle;
  /* [한국어] 현재 커널의 GPU 시뮬레이션 사이클 수.
   * 설정자: gpu-sim.cc의 cycle()에서 매 사이클 1씩 증가. 커널 시작 시 0으로 리셋.
   * 읽는 자: 타이밍 통계, simt_stack::update()에서 branch_div_cycle 기록. */
  unsigned long long gpu_tot_sim_cycle;
  /* [한국어] 모든 커널의 누적 GPU 시뮬레이션 사이클 수.
   * 설정자: 커널 완료 시 gpu_sim_cycle을 누적. 읽는 자: 전체 실행 시간 통계. */

  /*
   * [한국어]
   * gpu_malloc - GPU 글로벌 메모리에 size 바이트를 할당 (bump allocator)
   *
   * @size: 할당할 바이트 수.
   * @return: 할당된 GPU 메모리의 가상 시작 주소(void *로 캐스팅됨).
   *
   * m_dev_malloc에서 size만큼 선형으로 증가시키는 간단한 bump allocator.
   * GLOBAL_HEAP_START(0xC0000000)부터 시작하여 위로 증가한다.
   * free() 없음 — 전체 시뮬레이션이 하나의 CUDA 프로세스 수명으로 취급.
   * 정렬: 접근 효율성을 위해 할당 전 m_dev_malloc을 size 크기로 정렬한다.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaMalloc() → [gpu_malloc()]
   */
  void *gpu_malloc(size_t size);

  /*
   * [한국어]
   * gpu_mallocarray - GPU 텍스처/서피스 배열 메모리를 할당
   *
   * @count: 할당할 바이트 수 (cudaArray::size에 해당).
   * @return: 할당된 GPU 메모리의 가상 시작 주소.
   *
   * gpu_malloc()과 동일한 bump allocator를 사용하지만, 텍스처 배열 전용.
   * 반환된 주소는 cudaArray::devPtr에 저장된다.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaMallocArray() → [gpu_mallocarray()]
   */
  void *gpu_mallocarray(size_t count);

  /*
   * [한국어]
   * gpu_memset - GPU 메모리를 특정 값으로 채움 (cudaMemset 구현)
   *
   * @dst_start_addr: GPU 측 목적지 시작 주소.
   * @c: 채울 값 (int이지만 unsigned char로 변환하여 1바이트씩 채움).
   * @count: 채울 바이트 수.
   *
   * m_global_mem에 c값을 count 바이트 기록한다.
   * cudaMemset()의 시뮬레이터 구현.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaMemset() → [gpu_memset()]
   */
  void gpu_memset(size_t dst_start_addr, int c, size_t count);

  /*
   * [한국어]
   * memcpy_to_gpu - CPU(호스트) → GPU(디바이스) 메모리 복사
   *
   * @dst_start_addr: GPU 측 목적지 가상 주소(gpu_malloc으로 받은 값).
   * @src: CPU 측 소스 버퍼 포인터.
   * @count: 복사할 바이트 수.
   *
   * m_global_mem에 src 버퍼의 데이터를 count 바이트 기록한다.
   * cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice)의 시뮬레이터 구현.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaMemcpy() → [memcpy_to_gpu()]
   */
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);

  /*
   * [한국어]
   * memcpy_from_gpu - GPU(디바이스) → CPU(호스트) 메모리 복사
   *
   * @dst: CPU 측 목적지 버퍼 포인터.
   * @src_start_addr: GPU 측 소스 가상 주소.
   * @count: 복사할 바이트 수.
   *
   * m_global_mem에서 count 바이트를 읽어 dst 버퍼에 기록한다.
   * cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost)의 시뮬레이터 구현.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaMemcpy() → [memcpy_from_gpu()]
   */
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);

  /*
   * [한국어]
   * memcpy_gpu_to_gpu - GPU → GPU 메모리 복사 (동일 시뮬레이터 내에서)
   *
   * @dst: GPU 측 목적지 가상 주소.
   * @src: GPU 측 소스 가상 주소.
   * @count: 복사할 바이트 수.
   *
   * m_global_mem에서 읽어 m_global_mem에 쓰는 방식으로 구현.
   * cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice)의 시뮬레이터 구현.
   */
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

  // [한국어] 각 메모리 공간 객체 포인터를 반환하는 getter 함수들.
  class memory_space *get_global_memory() {
    /* [한국어] 글로벌 메모리 객체 포인터 반환. gpu_malloc/memcpy/cuda-sim ld.global에 사용. */
    return m_global_mem;
  }
  class memory_space *get_tex_memory() {
    /* [한국어] 텍스처 메모리 객체 포인터 반환. 텍스처 배열 데이터 접근에 사용. */
    return m_tex_mem;
  }
  class memory_space *get_surf_memory() {
    /* [한국어] 서피스 메모리 객체 포인터 반환. 서피스 읽기/쓰기에 사용. */
    return m_surf_mem;
  }

  /*
   * [한국어]
   * gpgpu_ptx_sim_bindTextureToArray - 텍스처 참조를 CUDA 배열 데이터에 바인딩
   *
   * @texref: 바인딩할 textureReference 포인터 (PTX 소스의 .tex 선언에 대응).
   * @array: 실제 데이터가 저장된 cudaArray 포인터 (gpu_mallocarray로 할당).
   *
   * m_TextureRefToName으로 texref의 이름을 찾아 m_NameToCudaArray[name] = array로 등록.
   * 이후 tex 명령어 실행 시 get_texarray()로 이 매핑에서 데이터 소스를 조회한다.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaBindTextureToArray() → [gpgpu_ptx_sim_bindTextureToArray()]
   */
  void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                        const struct cudaArray *array);

  /*
   * [한국어]
   * gpgpu_ptx_sim_bindNameToTexture - 텍스처 이름을 textureReference와 속성에 연결
   *
   * @name: PTX에서 사용하는 텍스처 심볼 이름 (예: "texData").
   * @texref: 이름과 연결할 textureReference 포인터.
   * @dim: 텍스처 차원 (1/2/3).
   * @readmode: 텍스처 읽기 모드.
   * @ext: 확장 플래그.
   *
   * m_NameToTextureRef[name]에 texref를 추가하고,
   * m_TextureRefToName[texref] = name으로 역방향 매핑도 저장.
   * textureReferenceAttr를 새로 생성하여 m_NameToAttribute[name]에 저장.
   *
   * 호출 체인:
   *   CUDA 런타임 초기화 __cudaRegisterTexture() → [gpgpu_ptx_sim_bindNameToTexture()]
   */
  void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                       const struct textureReference *texref,
                                       int dim, int readmode, int ext);

  /*
   * [한국어]
   * gpgpu_ptx_sim_unbindTexture - 텍스처 참조의 바인딩을 해제
   *
   * @texref: 바인딩을 해제할 textureReference 포인터.
   *
   * m_TextureRefToName으로 이름을 찾아 m_NameToCudaArray에서 해당 항목을 제거.
   * cudaUnbindTexture()의 시뮬레이터 구현.
   *
   * 호출 체인:
   *   libcuda/cuda_runtime_api.cc::cudaUnbindTexture() → [gpgpu_ptx_sim_unbindTexture()]
   */
  void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);

  /*
   * [한국어]
   * gpgpu_ptx_sim_findNamefromTexture - textureReference 포인터로 텍스처 이름을 역조회
   *
   * @texref: 이름을 찾을 textureReference 포인터.
   * @return: m_TextureRefToName[texref]. 등록되지 않은 경우 assert 실패.
   *
   * gpgpu_ptx_sim_bindTextureToArray()의 내부에서 이름 조회에 사용된다.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_bindTextureToArray() → [gpgpu_ptx_sim_findNamefromTexture()]
   */
  const char *gpgpu_ptx_sim_findNamefromTexture(
      const struct textureReference *texref);

  /*
   * [한국어]
   * get_texref - 텍스처 이름으로 textureReference 포인터를 반환
   *
   * @texname: 조회할 텍스처 심볼 이름.
   * @return: m_NameToTextureRef[texname]의 첫 번째 포인터.
   *   assert: 이름이 맵에 없으면 abort — 미등록 텍스처 접근 방지.
   *
   * cuda-sim의 tex 명령어에서 필터/주소 모드 등 textureReference의 속성에 접근하기 위해 호출.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texref()] → 필터/주소 모드 조회
   */
  const struct textureReference *get_texref(const std::string &texname) const {
    std::map<std::string,
             std::set<const struct textureReference *> >::const_iterator t =
        m_NameToTextureRef.find(texname);   // [한국어] 이름→텍스처 참조 집합 맵에서 검색
    assert(t != m_NameToTextureRef.end());  // [한국어] 미등록 텍스처 접근 방어 — 실패 시 abort
    return *(t->second.begin());            // [한국어] 집합의 첫 번째(유일한) 참조 포인터 반환
  }

  /*
   * [한국어]
   * get_texarray - 텍스처 이름으로 CUDA 배열 포인터를 반환
   *
   * @texname: 조회할 텍스처 심볼 이름.
   * @return: m_NameToCudaArray[texname]. 현재 바인딩된 cudaArray 포인터.
   *   assert: 이름이 맵에 없으면 abort.
   *
   * tex 명령어 실행 시 실제 데이터 소스(cudaArray::devPtr)를 찾기 위해 호출.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texarray()] → devPtr로 데이터 접근
   */
  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);    // [한국어] 이름→CUDA 배열 맵에서 검색
    assert(t != m_NameToCudaArray.end());   // [한국어] 미등록 텍스처 접근 방어
    return t->second;                       // [한국어] cudaArray 포인터 반환
  }

  /*
   * [한국어]
   * get_texinfo - 텍스처 이름으로 텍스처 속성(크기·채널 형식)을 반환
   *
   * @texname: 조회할 텍스처 심볼 이름.
   * @return: m_NameToTextureInfo[texname]. textureInfo 포인터.
   *
   * tex 명령어에서 텍스처 데이터 타입(float4, int4 등)과 크기 결정에 사용.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texinfo()] → 채널 형식/크기 조회
   */
  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);   // [한국어] 이름→텍스처 정보 맵에서 검색
    assert(t != m_NameToTextureInfo.end());  // [한국어] 미등록 텍스처 접근 방어
    return t->second;                        // [한국어] textureInfo 포인터 반환
  }

  /*
   * [한국어]
   * get_texattr - 텍스처 이름으로 차원/읽기모드 등 추가 속성을 반환
   *
   * @texname: 조회할 텍스처 심볼 이름.
   * @return: m_NameToAttribute[texname]. textureReferenceAttr 포인터.
   *
   * tex 명령어 실행 시 차원(1D/2D/3D)과 읽기 모드(ElementType/NormalizedFloat) 조회에 사용.
   *
   * 호출 체인:
   *   cuda-sim/instructions.cc::tex_impl() → [get_texattr()] → 차원/읽기모드 결정
   */
  const struct textureReferenceAttr *get_texattr(
      const std::string &texname) const {
    std::map<std::string, const struct textureReferenceAttr *>::const_iterator
        t = m_NameToAttribute.find(texname);  // [한국어] 이름→텍스처 속성 맵에서 검색
    assert(t != m_NameToAttribute.end());     // [한국어] 미등록 텍스처 접근 방어
    return t->second;                         // [한국어] textureReferenceAttr 포인터 반환
  }

  /*
   * [한국어]
   * get_config - 기능 시뮬레이션 설정 const 참조를 반환
   *
   * @return: m_function_model_config const 참조. 복사 없이 설정에 접근.
   *
   * 읽는 자: cuda-sim 코드에서 PTX 실행 옵션(convert_to_ptxplus, use_cuobjdump 등) 조회.
   */
  const gpgpu_functional_sim_config &get_config() const {
    return m_function_model_config;
  }

  FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }
  /* [한국어] PTX 명령어 디버그 로그 파일 포인터 반환.
   * g_ptx_inst_debug_to_file이 활성화된 경우에만 유효한 파일 포인터를 반환.
   * 읽는 자: cuda-sim의 PTX 실행 루프에서 디버그 로그 기록. */

  //  These maps return the current texture mappings for the GPU at any given
  //  time.
  // [한국어] 현재 시점의 텍스처 매핑을 복사하여 반환하는 함수들.
  // kernel_info_t 생성 시 launch 시점 스냅샷으로 사용된다.
  std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
    /* [한국어] 텍스처 이름→CUDA 배열 맵의 복사본을 반환.
     * kernel_info_t 생성자에서 m_NameToCudaArray를 스냅샷으로 복사할 때 호출. */
    return m_NameToCudaArray;
  }
  std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
    /* [한국어] 텍스처 이름→텍스처 정보 맵의 복사본을 반환.
     * kernel_info_t 생성자에서 m_NameToTextureInfo를 스냅샷으로 복사할 때 호출. */
    return m_NameToTextureInfo;
  }

  /*
   * [한국어]
   * ~gpgpu_t - 가상 소멸자
   *
   * virtual 선언으로 파생 클래스(gpgpu_sim) 소멸자가 올바르게 호출됨을 보장.
   * 상속 계층에서 기반 클래스 포인터로 delete할 때 파생 클래스 소멸자가 누락되지 않도록
   * 반드시 virtual로 선언해야 한다.
   */
  virtual ~gpgpu_t() {}

 protected:  // [한국어] 보호 멤버: gpgpu_t와 파생 클래스(gpgpu_sim)에서만 직접 접근 가능.

  const gpgpu_functional_sim_config &m_function_model_config;
  /* [한국어] 기능 시뮬레이션 설정 const 참조.
   * 생성자에서 config 인자로 초기화되며 이후 읽기 전용.
   * const 참조이므로 gpgpu_t 생명주기 동안 참조 대상 객체가 살아있어야 한다.
   * 설정자: 생성자 초기화 리스트. 읽는 자: get_config() getter, 내부 함수들. */

  FILE *ptx_inst_debug_file;
  /* [한국어] PTX 명령어 실행 디버그 로그 파일 포인터.
   * 설정자: gpgpu_t 생성자에서 g_ptx_inst_debug_to_file이 1이면 fopen으로 열어 설정.
   * 읽는 자: get_ptx_inst_debug_file() getter, cuda-sim 실행 루프.
   * NULL이면 디버그 출력 비활성. */

  class memory_space *m_global_mem;
  /* [한국어] GPU 글로벌 메모리 객체 포인터.
   * 모든 스레드가 접근할 수 있는 대용량 메모리를 소프트웨어로 시뮬레이션한다.
   * 설정자: 생성자에서 new memory_space()로 할당.
   * 읽는 자: cuda-sim의 ld.global/st.global, memcpy_to_gpu/from_gpu.
   * 값 범위: GLOBAL_HEAP_START 이상의 GPU 가상 주소 공간.
   * 동기화: 단일 메인 스레드에서만 접근 — 별도 락 불필요. */

  class memory_space *m_tex_mem;
  /* [한국어] GPU 텍스처 메모리 객체 포인터.
   * 텍스처 바인딩된 데이터를 저장한다. 텍스처 캐시(tex$)를 통해 접근된다.
   * 설정자: 생성자에서 할당. gpgpu_ptx_sim_bindTextureToArray()로 데이터 연결.
   * 읽는 자: cuda-sim의 tex.nd 명령어 실행 경로. */

  class memory_space *m_surf_mem;
  /* [한국어] GPU 서피스 메모리 객체 포인터.
   * 텍스처와 유사하지만 읽기·쓰기 모두 지원하는 메모리 공간.
   * suld(surface load)/sust(surface store) PTX 명령어에서 사용. */

  unsigned long long m_dev_malloc;
  /* [한국어] 다음 gpu_malloc() 호출 시 할당할 주소 (bump allocator 포인터).
   * GLOBAL_HEAP_START(0xC0000000)로 초기화, gpu_malloc() 호출마다 size 만큼 증가.
   * 설정자: 생성자(초기화 = GLOBAL_HEAP_START), gpu_malloc()(증가).
   * 읽는 자: gpu_malloc()에서 현재 위치 반환 후 증가.
   * 동기화: 단일 메인 스레드에서만 호출 — 별도 락 불필요. */

  //  These maps contain the current texture mappings for the GPU at any given
  //  time.
  // [한국어] 현재 GPU의 텍스처 매핑 정보를 저장하는 맵들.
  // 커널 launch 시 이 맵들의 스냅샷이 kernel_info_t에 복사되어 실행 중 바인딩 변경을 격리한다.

  std::map<std::string, std::set<const struct textureReference *> >
      m_NameToTextureRef;
  /* [한국어] 텍스처 이름 → textureReference 포인터 집합 매핑.
   * 같은 이름에 여러 textureReference가 바인딩될 수 있어 set으로 관리.
   * 설정자: gpgpu_ptx_sim_bindNameToTexture(). 읽는 자: get_texref().
   * kernel_info_t 생성 시 이 맵의 스냅샷이 m_NameToTextureRef로 복사됨. */

  std::map<const struct textureReference *, std::string> m_TextureRefToName;
  /* [한국어] textureReference 포인터 → 텍스처 이름 역방향 매핑.
   * 설정자: gpgpu_ptx_sim_bindNameToTexture().
   * 읽는 자: gpgpu_ptx_sim_findNamefromTexture(), gpgpu_ptx_sim_bindTextureToArray(). */

  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  /* [한국어] 텍스처 이름 → 현재 바인딩된 CUDA 배열 포인터 매핑.
   * 설정자: gpgpu_ptx_sim_bindTextureToArray(), gpgpu_ptx_sim_unbindTexture().
   * 읽는 자: get_texarray(), getNameArrayMapping() — kernel_info_t 스냅샷 생성.
   * 동시성: CUDA 사양상 커널 실행 중 텍스처 바인딩 변경은 정의되지 않은 동작. */

  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
  /* [한국어] 텍스처 이름 → 텍스처 크기·채널 형식 정보 매핑.
   * 설정자: gpgpu_ptx_sim_bindNameToTexture() 또는 관련 초기화 함수.
   * 읽는 자: get_texinfo(), getNameInfoMapping() — kernel_info_t 스냅샷 생성. */

  std::map<std::string, const struct textureReferenceAttr *> m_NameToAttribute;
  /* [한국어] 텍스처 이름 → textureReferenceAttr 포인터 매핑.
   * __cudaRegisterTexture()로 등록된 텍스처 차원/읽기모드/확장 속성 저장.
   * 설정자: gpgpu_ptx_sim_bindNameToTexture()에서 new textureReferenceAttr()로 생성.
   * 읽는 자: get_texattr() → cuda-sim의 tex_impl()에서 차원/읽기모드 결정. */
};

/*
 * ==========================================================================
 * gpgpu_ptx_sim_info - 커널의 자원 사용량 정보 구조체
 * ==========================================================================
 *
 * [한국어] PTX 컴파일러가 생성한 .ptxinfo 파일에서 읽어온 커널별 자원 사용량 정보.
 * SM 점유율(occupancy) 계산의 핵심 입력값이다: SM당 동시에 실행 가능한 CTA 수는
 * smem, regs의 제약과 SM 하드웨어 용량에 의해 결정된다.
 *
 * 설정 흐름:
 *   ptxinfo 파일 → cuda-sim/ptx_parser.cc의 ptxinfo 파서
 *   → gpgpu_ptx_sim_info 필드 설정 → kernel_info_t에 연결
 *   → shader_core_ctx::issue_block2core()의 점유율 계산에 사용
 *
 * ptxinfo 파일이 없으면 모두 0으로 설정됨 (보수적으로 자원 0으로 가정).
 */
struct gpgpu_ptx_sim_info {
  // Holds properties of the kernel (Kernel's resource use).
  // These will be set to zero if a ptxinfo file is not present.
  int lmem;
  /* [한국어] 스레드당 사용하는 로컬 메모리 바이트 수.
   * ptxinfo 파일의 "lmem = N" 항목에서 파싱.
   * 로컬 메모리는 레지스터 스필(spill) 데이터와 PTX .local 변수를 저장한다.
   * SM 점유율 계산: 총 로컬 메모리 = lmem * threads_per_block * 동시 블록 수 ≤ TOTAL_LOCAL_MEM_PER_SM.
   * 설정자: cuda-sim ptxinfo 파서. 읽는 자: shader_core_ctx의 점유율 계산. */

  int smem;
  /* [한국어] CTA당 사용하는 공유 메모리 바이트 수.
   * ptxinfo 파일의 "smem = N" 항목에서 파싱.
   * SM 점유율 계산의 핵심 제약: 총 공유 메모리 = smem * 동시 실행 블록 수 ≤ gpgpu_shmem_size.
   * 공유 메모리 사용량이 많을수록 SM당 동시 실행 가능한 블록 수가 감소한다.
   * 설정자: cuda-sim ptxinfo 파서. 읽는 자: shader_core_ctx::issue_block2core(). */

  int cmem;
  /* [한국어] 커널이 사용하는 상수 메모리 바이트 수(커널 파라미터 크기 포함).
   * 상수 메모리는 SM당 상수 캐시(const$)에 캐시된다.
   * 현재 점유율 계산에서는 직접 제약으로 작용하지 않지만 통계에 기록됨. */

  int gmem;
  /* [한국어] 커널이 사용하는 정적 글로벌 메모리 바이트 수.
   * .global 변수들의 합계. 동적 할당(gpu_malloc)과는 다른 값.
   * 현재 점유율 계산에서는 직접 제약으로 작용하지 않지만 통계에 기록됨. */

  int regs;
  /* [한국어] 스레드당 사용하는 레지스터 수.
   * ptxinfo 파일의 "registers = N" 항목에서 파싱.
   * SM 점유율 계산의 핵심 제약: 총 레지스터 = regs * warp_size * 동시 워프 수 ≤ reg_file_size.
   * NVIDIA GPU의 레지스터 파일은 제한적(예: Volta = 65536 레지스터/SM)이므로
   * 이 값이 크면 동시 실행 워프 수가 감소한다.
   * 설정자: cuda-sim ptxinfo 파서. 읽는 자: shader_core_ctx::issue_block2core(). */

  unsigned maxthreads;
  /* [한국어] 이 커널이 지원하는 CTA당 최대 스레드 수.
   * __launch_bounds__(maxthreads) 어트리뷰트에서 설정된다.
   * 이 값을 초과하는 blockDim으로 커널을 실행하면 런타임 오류 발생.
   * 설정자: ptxinfo 파서 또는 launch_bounds 어트리뷰트. */

  unsigned ptx_version;
  /* [한국어] 사용된 PTX ISA 버전 번호 (예: 70 = PTX 7.0, 64 = PTX 6.4).
   * 버전에 따라 지원되는 명령어 집합이 다르다 (예: Volta의 독립 스레드 스케줄링).
   * 설정자: ptxinfo 파서. 읽는 자: 버전 호환성 검사. */

  unsigned sm_target;
  /* [한국어] 이 커널의 타겟 GPU 아키텍처(compute capability, 예: 70 = Volta sm_70, 75 = Turing).
   * m_ptx_force_max_capability보다 높으면 다운그레이드 또는 에러 처리.
   * 설정자: ptxinfo 파서. 읽는 자: PTX 능력 검사. */
};

/*
 * [한국어]
 * gpgpu_ptx_sim_arg - 커널에 전달하는 인자(파라미터) 하나를 표현하는 구조체
 *
 * cuLaunchKernel() 또는 CUDA 런타임이 커널 파라미터를 전달할 때 각 인자의
 * 데이터 포인터, 크기, 파라미터 메모리 내 오프셋을 묶어 관리한다.
 * gpgpu_ptx_sim_arg_list_t에 저장되어 kernel_info_t 생성 시 .param 메모리로 복사된다.
 *
 * 설정 흐름:
 *   libcuda/cuda_runtime_api.cc::cudaLaunch() / cuLaunchKernel()
 *   → gpgpu_ptx_sim_arg 생성(인자마다)
 *   → gpgpu_ptx_sim_arg_list_t에 push
 *   → kernel_info_t 생성자에서 m_param_mem에 복사
 */
struct gpgpu_ptx_sim_arg {
  gpgpu_ptx_sim_arg() { m_start = NULL; }
  /* [한국어] 기본 생성자: m_start를 NULL로 초기화. 유효하지 않은 빈 인자 슬롯. */

  gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {
    /* [한국어] 커널 인자 정보를 받는 생성자.
     * @arg: CPU 측 인자 데이터의 시작 포인터.
     * @size: 인자 데이터의 크기(바이트). sizeof(인자 타입)에 해당.
     * @offset: 파라미터 메모리 내에서 이 인자의 위치(오프셋). */
    m_start = arg;      // [한국어] 인자 데이터의 CPU 측 시작 포인터 설정
    m_nbytes = size;    // [한국어] 인자 크기(바이트) 설정
    m_offset = offset;  // [한국어] .param 공간 내 오프셋 설정
  }

  const void *m_start;
  /* [한국어] 커널 인자 데이터의 CPU 측 시작 포인터.
   * cuLaunchKernel의 kernelParams 배열에서 각 인자의 포인터.
   * 설정자: gpgpu_ptx_sim_arg 생성자의 arg 인자.
   * 읽는 자: kernel_info_t 생성 시 m_param_mem에 m_nbytes 바이트 복사. */

  size_t m_nbytes;
  /* [한국어] 인자 데이터의 크기(바이트). sizeof(인자 타입)에 해당.
   * 예: float* 인자 = 8바이트(64비트 포인터), int 인자 = 4바이트, float4 = 16바이트.
   * 설정자: gpgpu_ptx_sim_arg 생성자의 size 인자.
   * 읽는 자: kernel_info_t에서 m_param_mem에 복사할 크기 결정. */

  size_t m_offset;
  /* [한국어] PTX .param 공간 내에서 이 인자의 바이트 오프셋.
   * cuParamSetv() 또는 cuLaunchKernel에서 인자들을 순서대로 쌓을 때 누적 계산.
   * PTX의 ld.param.u64 [param0+0] 같은 명령어에서의 오프셋 값과 일치해야 한다.
   * 설정자: gpgpu_ptx_sim_arg 생성자의 offset 인자.
   * 읽는 자: kernel_info_t에서 m_param_mem의 어느 위치에 복사할지 결정. */
};

/*
 * [한국어]
 * gpgpu_ptx_sim_arg_list_t - 커널 인자들의 목록 타입
 *
 * 하나의 커널 실행에 전달되는 모든 인자를 순서대로 저장하는 연결 리스트.
 * libcuda에서 cudaLaunch() 직전까지 gpgpu_ptx_sim_arg들을 쌓고,
 * gpgpusim_entrypoint.cc에서 kernel_info_t 생성 시 이 목록에서 m_param_mem으로 복사한다.
 */
typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

/*
 * ==========================================================================
 * memory_space_t - 메모리 공간 타입 + 뱅크 번호를 담는 복합 타입 클래스
 * ==========================================================================
 *
 * [한국어] _memory_space_t enum보다 풍부한 정보를 담는 클래스.
 * PTX의 ".const[n]"처럼 메모리 공간 타입과 뱅크 번호를 함께 관리한다.
 * 메모리 공간을 std::map의 키로 사용하기 위해 비교 연산자(<, ==, !=)를 제공한다.
 * inst_t::space 필드로 각 명령어가 접근하는 메모리 공간을 표현하는 데 주로 사용된다.
 *
 * PTX 메모리 공간 종류 (_memory_space_t enum 값들):
 *   undefined_space, reg_space, local_space, shared_space,
 *   param_space_unclassified, param_space_kernel, param_space_local,
 *   const_space, tex_space, surf_space, global_space, generic_space, instruction_space
 *
 * === 타 모듈과의 연결 ===
 * - inst_t::space: 명령어가 접근하는 메모리 공간을 이 타입으로 저장.
 * - warp_inst_t::generate_mem_accesses(): space.get_type()으로 캐시 접근 경로 결정.
 * - mem_access_t 생성 시: is_global(), is_local(), is_const()로 mem_access_type 분류.
 */
class memory_space_t {
 public:
  memory_space_t() {
    /* [한국어] 기본 생성자: 메모리 타입을 undefined_space(미정의)로, 뱅크를 0으로 초기화.
     * 미정의 상태에서 메모리 접근을 시도하면 오류 — 반드시 타입 설정 후 사용. */
    m_type = undefined_space;     // [한국어] 메모리 타입: 미정의
    m_bank = 0;                   // [한국어] 뱅크 번호: 0 (상수 메모리의 기본 뱅크)
  }
  memory_space_t(const enum _memory_space_t &from) {
    /* [한국어] _memory_space_t enum에서 변환하는 생성자.
     * @from: 대응하는 enum 값. 뱅크는 항상 0으로 초기화.
     * 사용 예: memory_space_t sp(global_space); */
    m_type = from;                // [한국어] 전달받은 메모리 타입 저장
    m_bank = 0;                   // [한국어] 뱅크 번호: 기본 0 (= .const[0])
  }

  // [한국어] 비교 연산자들: std::map 키 사용 및 조건 비교를 위해 필요.
  bool operator==(const memory_space_t &x) const {
    /* [한국어] 타입과 뱅크가 모두 같아야 동일한 메모리 공간으로 판정. */
    return (m_bank == x.m_bank) && (m_type == x.m_type);
  }
  bool operator!=(const memory_space_t &x) const { return !(*this == x); }
  /* [한국어] ==의 논리 반전. 다른 메모리 공간인지 확인. */

  bool operator<(const memory_space_t &x) const {
    /* [한국어] std::map 키로 사용하기 위한 엄격한 약한 순서(strict weak ordering) 제공.
     * 정렬 기준: 타입(m_type) 우선, 타입이 같으면 뱅크(m_bank) 비교. */
    if (m_type < x.m_type)       // [한국어] 타입이 작으면 먼저 → true
      return true;
    else if (m_type > x.m_type)  // [한국어] 타입이 크면 나중 → false
      return false;
    else if (m_bank < x.m_bank)  // [한국어] 타입이 같으면 뱅크로 비교
      return true;
    return false;                 // [한국어] 타입과 뱅크가 모두 같거나 크면 false
  }

  enum _memory_space_t get_type() const { return m_type; }
  /* [한국어] 메모리 공간 타입 반환. inst_t::space.get_type()으로 접근 경로 분류에 사용. */
  void set_type(enum _memory_space_t t) { m_type = t; }
  /* [한국어] 메모리 공간 타입 설정. PTX 파서에서 명령어 디코딩 시 호출. */
  unsigned get_bank() const { return m_bank; }
  /* [한국어] 상수 메모리 뱅크 번호 반환. .const[n]에서 n 값. */
  void set_bank(unsigned b) { m_bank = b; }
  /* [한국어] 상수 메모리 뱅크 번호 설정. PTX 파서에서 .const[n] 파싱 시 호출. */

  // [한국어] 메모리 공간 분류 편의 함수들. inst_t::space에서 접근 타입 판별에 사용.
  bool is_const() const {
    /* [한국어] 상수 메모리 공간인지 확인.
     * const_space: PTX .const 변수 공간.
     * param_space_kernel: 커널 파라미터 공간 (상수 캐시로 접근되므로 동일 처리).
     * 상수 캐시(const$)를 통해 접근되는 두 가지 경우를 모두 포함. */
    return (m_type == const_space) || (m_type == param_space_kernel);
  }
  bool is_local() const {
    /* [한국어] 로컬 메모리 공간인지 확인.
     * local_space: PTX .local 변수 공간 (스레드 전용, 레지스터 스필에도 사용).
     * param_space_local: 함수 파라미터의 로컬 공간 버전.
     * 로컬 메모리는 물리적으로 글로벌 DRAM에 위치하지만 각 스레드마다 독립적. */
    return (m_type == local_space) || (m_type == param_space_local);
  }
  bool is_global() const { return (m_type == global_space); }
  /* [한국어] 글로벌 메모리 공간인지 확인. PTX .global 변수 공간.
   * 모든 스레드가 접근 가능하며, L1/L2/DRAM 경로를 통해 처리된다. */

 private:
  enum _memory_space_t m_type;
  /* [한국어] 메모리 공간 타입 (global, shared, local, const 등).
   * 설정자: 생성자 또는 set_type(). 읽는 자: get_type(), is_global/const/local(). */

  unsigned m_bank;
  // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
  /* [한국어] 상수 메모리의 뱅크 번호. PTX ".const[n]"에서 n에 해당.
   * .const는 .const[0]과 동일 (PTX 2.1 스펙 5.1.3절).
   * 뱅크 0은 커널 파라미터용으로 예약되는 경우가 많다.
   * 설정자: 생성자(기본 0), set_bank(). 읽는 자: get_bank(). */
};

/*
 * ==========================================================================
 * 메모리 접근(Memory Access) 관련 타입 정의
 * ==========================================================================
 *
 * [한국어] 메모리 접근 시 어떤 바이트/섹터에 접근하는지 추적하기 위한 비트마스크 타입들.
 * GPGPU-Sim은 캐시 라인(128B) 단위로 메모리를 관리하며,
 * 각 접근이 128B 라인 내에서 정확히 어떤 바이트와 32B 섹터에 닿는지 기록한다.
 * 섹터 기반 캐시(Volta+)에서 partial write/fill 최적화에 활용된다.
 */
const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
/* [한국어] 최대 메모리 접근 크기 = 128바이트 (캐시 라인 크기와 동일).
 * mem_access_byte_mask_t의 비트 수를 결정한다. 읽는 자: mem_access_t 생성. */

typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
/* [한국어] 128B 메모리 접근 범위에서 실제로 접근되는 바이트들의 비트마스크.
 * 비트 n = 1이면 오프셋 n번 바이트에 접근. 0이면 해당 바이트 미접근.
 * 섹터 캐시에서 partial fill(필요한 바이트만 채우기) 구현에 사용.
 * NO_PARTIAL_WRITE: 빈 마스크(전체 라인 쓰기 또는 마스크 없음). */

const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
/* [한국어] 캐시 라인(128B)을 구성하는 섹터 수 = 4.
 * 128B 라인을 4개의 32B 섹터로 나눈 것. mem_access_sector_mask_t의 비트 수. */

const unsigned SECTOR_SIZE = 32;        // sector is 32 bytes width
/* [한국어] 섹터 크기 = 32바이트 (GPU L1 캐시의 섹터/서브라인 단위).
 * 섹터 기반 캐시(Volta+의 Sector Cache)는 캐시 라인을 4개의 32B 섹터로 관리하여
 * 필요한 섹터만 이동하는 "sector miss" 최적화를 지원한다. */

typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
/* [한국어] 128B 캐시 라인 내 4개의 32B 섹터 중 어떤 섹터에 접근하는지의 비트마스크(4비트).
 * 비트 n = 1이면 섹터 n(오프셋 n*32 ~ n*32+31 바이트)에 접근.
 * 설정자: warp_inst_t::memory_coalescing_arch()에서 섹터 계산.
 * 읽는 자: gpu-cache.cc의 섹터 기반 캐시 태그 비교 및 데이터 이동. */

#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())
/* [한국어] 부분 쓰기(partial write) 없음을 나타내는 빈 바이트 마스크.
 * 전체 캐시 라인을 쓰거나 바이트 마스크가 필요 없는 경우의 기본값.
 * 사용 예: mem_access_t 생성자에서 일반 write-allocate 요청 시. */

/*
 * [한국어]
 * MEM_ACCESS_TYPE_TUP_DEF - 메모리 접근 타입 enum을 정의하는 X-매크로(X-macro) 패턴.
 *
 * X-매크로 패턴이란: 동일한 데이터 목록을 여러 목적(enum 정의, 문자열 변환 등)으로
 * 재사용하는 C 프리프로세서 기법이다. MA_TUP_BEGIN/MA_TUP/MA_TUP_END를 다르게 정의하면
 * 같은 MEM_ACCESS_TYPE_TUP_DEF 매크로로 enum 생성과 문자열 배열 생성을 모두 할 수 있다.
 *
 * 각 메모리 접근 타입의 의미:
 *   GLOBAL_ACC_R:    글로벌 메모리 읽기 (L1/L2/DRAM 경로)
 *   LOCAL_ACC_R:     로컬 메모리 읽기 (스레드 전용, 물리적으로는 DRAM)
 *   CONST_ACC_R:     상수 메모리 읽기 (상수 캐시const$ 경유)
 *   TEXTURE_ACC_R:   텍스처 메모리 읽기 (텍스처 캐시tex$ 경유)
 *   GLOBAL_ACC_W:    글로벌 메모리 쓰기
 *   LOCAL_ACC_W:     로컬 메모리 쓰기
 *   L1_WRBK_ACC:     L1 캐시 쓰기 되돌리기(write-back) 요청
 *   L2_WRBK_ACC:     L2 캐시 쓰기 되돌리기 요청
 *   INST_ACC_R:      명령어 캐시(I$) 읽기
 *   L1_WR_ALLOC_R:   L1 쓰기 시 write-allocate를 위한 read 요청
 *   L2_WR_ALLOC_R:   L2 쓰기 시 write-allocate를 위한 read 요청
 *   NUM_MEM_ACCESS_TYPE: 타입 수 (경계값, 배열 크기에 사용)
 *
 * 이 타입에 따라 mem_fetch가 어느 캐시(L1/L2/const$/tex$)를 통과할지 결정된다.
 */
#define MEM_ACCESS_TYPE_TUP_DEF                                         \
  MA_TUP_BEGIN(mem_access_type)                                         \
  MA_TUP(GLOBAL_ACC_R), MA_TUP(LOCAL_ACC_R), MA_TUP(CONST_ACC_R),       \
      MA_TUP(TEXTURE_ACC_R), MA_TUP(GLOBAL_ACC_W), MA_TUP(LOCAL_ACC_W), \
      MA_TUP(L1_WRBK_ACC), MA_TUP(L2_WRBK_ACC), MA_TUP(INST_ACC_R),     \
      MA_TUP(L1_WR_ALLOC_R), MA_TUP(L2_WR_ALLOC_R),                     \
      MA_TUP(NUM_MEM_ACCESS_TYPE) MA_TUP_END(mem_access_type)

// [한국어] X-매크로를 enum 정의용으로 재정의:
// MA_TUP_BEGIN(X) → "enum X {"
// MA_TUP(X) → "X" (enum 값 이름)
// MA_TUP_END(X) → "};"
#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X) \
  }                   \
  ;
MEM_ACCESS_TYPE_TUP_DEF
/* [한국어] 위 매크로를 전개하여 mem_access_type enum을 생성.
 * 결과: enum mem_access_type { GLOBAL_ACC_R, LOCAL_ACC_R, ..., NUM_MEM_ACCESS_TYPE }; */

#undef MA_TUP_BEGIN  // [한국어] 매크로 정의 해제 — mem_access_type_str() 생성 시 다시 정의할 수 있도록
#undef MA_TUP
#undef MA_TUP_END

/*
 * [한국어]
 * mem_access_type_str - mem_access_type enum 값을 사람이 읽을 수 있는 문자열로 변환
 *
 * @access_type: 변환할 mem_access_type 값.
 * @return: "GLOBAL_ACC_R" 같은 const 문자열 포인터.
 *
 * 디버그 출력, 통계 레이블, 시뮬레이션 로그에서 메모리 접근 타입을 식별하는 데 사용.
 * 구현은 abstract_hardware_model.cc에서 MEM_ACCESS_TYPE_TUP_DEF를 다시 정의하여 생성.
 *
 * 호출 체인:
 *   mem_access_t::print() → [mem_access_type_str()] → 타입 문자열 출력
 */
const char *mem_access_type_str(enum mem_access_type access_type);

/*
 * [한국어]
 * cache_operator_type - PTX 메모리 명령어의 캐시 동작 힌트(cache operator)
 *
 * PTX 명령어의 접미사(.ca, .cg, .cv 등)로 지정되며, 캐시를 어떻게 사용할지 지시한다.
 * GPGPU-Sim의 캐시 모델은 이 값을 참고하여 L1 우회(bypass), write-back/through 정책을 결정.
 *
 * 읽는 자:
 * - warp_inst_t::generate_mem_accesses(): cache_op에 따라 mem_access_type 분류 보정.
 * - gpu-cache.cc::baseline_cache::access(): 캐시 lookup 및 교체 정책 결정.
 *
 * PTX ISA 참고: https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
 */
enum cache_operator_type {
  CACHE_UNDEFINED,
  /* [한국어] 캐시 연산 미정의. 기본 상태 — 메모리 명령어가 아닌 경우 또는 초기화 전. */

  // loads (로드/읽기 명령어에만 적용되는 캐시 힌트)
  CACHE_ALL,
  /* [한국어] .ca (cache all) — 모든 캐시 레벨(L1/L2)에 캐시.
   * load 명령어의 기본 동작. 재사용 가능성이 높은 데이터에 적합. */

  CACHE_LAST_USE,
  /* [한국어] .lu (last use) — 이 접근이 마지막 사용임을 힌트.
   * 해당 캐시 라인을 캐시에서 조기에 evict해도 된다는 신호.
   * 스트리밍 패턴에서 캐시 오염(cache pollution) 감소용. */

  CACHE_VOLATILE,
  /* [한국어] .cv (cache volatile) — 캐시를 완전히 무시하고 항상 메모리에서 직접 읽기.
   * volatile 변수(다른 스레드가 변경 가능) 접근에 사용.
   * L1 캐시 bypass + coherence를 강제하는 효과. */

  CACHE_L1,
  /* [한국어] .nc (non-coherent / L1 cache) — L1 캐시에만 캐시.
   * 읽기 전용 데이터(예: __ldg() 내장 함수)에 사용. 일관성 검사 없이 L1에 캐시. */

  // loads and stores (로드/스토어 모두에 적용)
  CACHE_STREAMING,
  /* [한국어] .cs (cache streaming) — 스트리밍 접근 최적화.
   * 한 번만 접근하는 데이터에 사용. L1 캐시에서 가장 낮은 우선순위로 처리. */

  CACHE_GLOBAL,
  /* [한국어] .cg (cache global) — L2에만 캐시, L1은 우회(bypass).
   * 글로벌 메모리 접근 시 L1 캐시를 거치지 않아 L1 용량 절약.
   * gmem_skip_L1D와 유사한 효과를 개별 명령어 단위로 지정. */

  // stores (스토어/쓰기 명령어에만 적용)
  CACHE_WRITE_BACK,
  /* [한국어] .wb (write back) — 쓰기 캐시 정책: 캐시에만 쓰고 나중에 메모리에 반영.
   * 쓰기 지연(deferred writeback)으로 대역폭 절약. 캐시 라인이 evict될 때 DRAM 갱신. */

  CACHE_WRITE_THROUGH
  /* [한국어] .wt (write through) — 쓰기 관통 정책: 캐시와 메모리에 동시에 쓰기.
   * 캐시와 메모리 간 일관성 보장. 쓰기 대역폭이 중요한 경우에 유용. */
};

/*
 * ==========================================================================
 * mem_access_t - 하나의 메모리 접근 요청 정보 클래스
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * warp_inst_t가 메모리 합치기(coalescing) 과정에서 생성하는 메모리 접근 요청 단위.
 * 접근 주소, 크기, 읽기/쓰기 여부, 접근 타입(GLOBAL_ACC_R 등),
 * 참여 스레드 마스크, 바이트/섹터 마스크를 하나의 객체로 묶어 m_accessq에 저장된다.
 *
 * === 사이클 레벨 동작 ===
 * warp_inst_t::generate_mem_accesses() → mem_access_t 생성 → m_accessq에 push_back()
 * shader.cc의 memory 단계에서 m_accessq.front()를 꺼내 mem_fetch_allocator로 mem_fetch 생성.
 * 이 mem_fetch가 L1→L2 캐시 계층을 통과하여 DRAM에 도달한다.
 *
 * === 타 모듈과의 연결 ===
 * - 생성: warp_inst_t::memory_coalescing_arch() → mem_access_t 생성 → m_accessq
 * - 소비: shader.cc → mem_fetch_allocator::alloc() → mem_fetch → gpu-cache.cc
 * - 컨텍스트: gpgpu_ctx로 전역 uid 카운터 접근 (init()에서 사용)
 */
class mem_access_t {
 public:
  mem_access_t(gpgpu_context *ctx) { init(ctx); }
  /* [한국어] 기본 생성자: gpgpu_context로 uid를 할당하고 나머지 필드를 기본값으로 초기화.
   * 이후 setter로 필드를 채워야 유효한 접근 요청이 된다. */

  /*
   * [한국어]
   * mem_access_t(type, address, size, wr, ctx) - 간략 생성자
   *
   * @type: 메모리 접근 타입 (GLOBAL_ACC_R, GLOBAL_ACC_W 등).
   * @address: 접근할 메모리의 가상 주소(합치기 후 정렬된 주소).
   * @size: 요청 크기(바이트). 보통 캐시 라인(128B) 또는 섹터(32B) 정렬.
   * @wr: true=쓰기(store), false=읽기(load).
   * @ctx: gpgpu_context 포인터. uid 할당에 사용.
   *
   * 워프 마스크·바이트 마스크·섹터 마스크 없이 타입·주소·크기만으로 생성.
   * 단순한 캐시 내부 write-back/write-allocate 요청 생성 시 사용.
   *
   * 호출 체인:
   *   gpu-cache.cc의 write-back/write-allocate 처리 → [mem_access_t()]
   */
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, gpgpu_context *ctx) {
    init(ctx);          // [한국어] uid 할당 및 필드 기본 초기화
    m_type = type;      // [한국어] 메모리 접근 타입 설정 (L1 캐시 경로 결정에 사용)
    m_addr = address;   // [한국어] 합치기(coalescing) 후 정렬된 접근 주소
    m_req_size = size;  // [한국어] 요청 크기(바이트)
    m_write = wr;       // [한국어] 읽기(false)/쓰기(true) 방향
  }

  /*
   * [한국어]
   * mem_access_t(type, address, size, wr, active_mask, byte_mask, sector_mask, ctx)
   * - 완전 생성자 (워프 마스크·바이트 마스크·섹터 마스크 포함)
   *
   * @type: 메모리 접근 타입.
   * @address: 접근 주소.
   * @size: 요청 크기(바이트).
   * @wr: 읽기/쓰기 방향.
   * @active_mask: 이 접근에 참여하는 스레드들의 32비트 비트마스크.
   * @byte_mask: 요청 내에서 실제로 접근되는 바이트들의 128비트 마스크.
   * @sector_mask: 접근하는 32B 섹터들의 4비트 마스크.
   * @ctx: gpgpu_context. uid 할당.
   *
   * 워프 단위 메모리 합치기(coalescing) 결과로 생성하는 일반적인 경로.
   * active_mask로 MSHR 완료 시 어떤 스레드를 깨울지 추적할 수 있다.
   *
   * 호출 체인:
   *   warp_inst_t::memory_coalescing_arch() → [mem_access_t()]
   */
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),    // [한국어] 참여 스레드 마스크 초기화
        m_byte_mask(byte_mask),      // [한국어] 접근 바이트 마스크 초기화
        m_sector_mask(sector_mask) { // [한국어] 접근 섹터 마스크 초기화
    init(ctx);             // [한국어] uid 할당
    m_type = type;         // [한국어] 메모리 접근 타입
    m_addr = address;      // [한국어] 접근 주소
    m_req_size = size;     // [한국어] 요청 크기
    m_write = wr;          // [한국어] 읽기/쓰기 방향
  }

  // [한국어] 멤버 변수 조회용 getter 함수들 — 모두 const.

  new_addr_type get_addr() const { return m_addr; }
  /* [한국어] 접근할 메모리 가상 주소 반환. 캐시 태그 비교, DRAM 주소 디코딩에 사용. */

  void set_addr(new_addr_type addr) { m_addr = addr; }
  /* [한국어] 접근 주소 설정. 주소 재정렬(alignment) 보정 시 사용. */

  unsigned get_size() const { return m_req_size; }
  /* [한국어] 요청 크기(바이트) 반환. 메모리 대역폭 통계, mem_fetch 크기 결정에 사용. */

  const active_mask_t &get_warp_mask() const { return m_warp_mask; }
  /* [한국어] 이 접근에 참여하는 스레드들의 비트마스크 const 참조 반환.
   * MSHR에서 이 접근 완료 시 깨울 스레드 집합 추적에 사용. */

  bool is_write() const { return m_write; }
  /* [한국어] 쓰기 연산(store)인지 여부. true=store, false=load.
   * 캐시 write policy(write-back/write-through/write-allocate) 결정에 사용. */

  enum mem_access_type get_type() const { return m_type; }
  /* [한국어] 메모리 접근 타입(GLOBAL_ACC_R 등) 반환.
   * 이 값으로 어느 캐시(L1/L2/const$/tex$)를 통과할지 결정된다. */

  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
  /* [한국어] 요청 내 접근 바이트들의 비트마스크 반환 (값 복사).
   * 섹터 캐시에서 partial fill 처리 시 필요한 바이트 집합 파악에 사용. */

  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }
  /* [한국어] 접근하는 32B 섹터들의 비트마스크 반환 (값 복사).
   * 섹터 기반 L1 캐시(Volta+)에서 sector miss 감지에 사용. */

  /*
   * [한국어]
   * print - 이 메모리 접근 정보를 파일에 출력 (디버깅·로그용)
   *
   * @fp: 출력 대상 파일 포인터.
   *
   * 주소, 읽기/쓰기, 크기, 접근 타입을 문자열로 출력.
   * 메모리 시스템 디버깅 및 시뮬레이션 트레이스 기록에 사용.
   *
   * 호출 체인:
   *   warp_inst_t::print() → [mem_access_t::print()] 또는 독립적인 디버그 경로
   */
  void print(FILE *fp) const {
    fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
            m_write ? "store" : "load ", m_req_size);  // [한국어] 주소, 읽기/쓰기, 크기 출력
    switch (m_type) {  // [한국어] 접근 타입에 따라 레이블 문자열 출력
      case GLOBAL_ACC_R:
        fprintf(fp, "GLOBAL_R");    // [한국어] 글로벌 메모리 읽기
        break;
      case LOCAL_ACC_R:
        fprintf(fp, "LOCAL_R ");    // [한국어] 로컬 메모리 읽기
        break;
      case CONST_ACC_R:
        fprintf(fp, "CONST   ");    // [한국어] 상수 메모리 읽기 (상수 캐시 경유)
        break;
      case TEXTURE_ACC_R:
        fprintf(fp, "TEXTURE ");    // [한국어] 텍스처 메모리 읽기 (텍스처 캐시 경유)
        break;
      case GLOBAL_ACC_W:
        fprintf(fp, "GLOBAL_W");    // [한국어] 글로벌 메모리 쓰기
        break;
      case LOCAL_ACC_W:
        fprintf(fp, "LOCAL_W ");    // [한국어] 로컬 메모리 쓰기
        break;
      case L2_WRBK_ACC:
        fprintf(fp, "L2_WRBK ");   // [한국어] L2 캐시 write-back 요청
        break;
      case INST_ACC_R:
        fprintf(fp, "INST    ");    // [한국어] 명령어 캐시(I$) 읽기
        break;
      case L1_WRBK_ACC:
        fprintf(fp, "L1_WRBK ");   // [한국어] L1 캐시 write-back 요청
        break;
      default:
        fprintf(fp, "unknown ");    // [한국어] 알 수 없는 타입 (새 타입 추가 시 여기서 감지)
        break;
    }
  }

  gpgpu_context *gpgpu_ctx;
  /* [한국어] GPU 시뮬레이터 전역 컨텍스트 포인터.
   * 설정자: init()에서 ctx 인자로 초기화.
   * 읽는 자: init()에서 전역 uid 카운터(gpgpu_ctx->the_gpgpusim->uid_counter) 접근. */

 private:
  /*
   * [한국어]
   * init - mem_access_t 내부 초기화 함수
   *
   * @ctx: gpgpu_context 포인터. 전역 uid 카운터에 접근.
   *
   * m_uid를 전역 카운터에서 원자적으로 할당하고, 나머지 멤버를 기본값으로 초기화.
   * 모든 생성자에서 첫 번째로 호출된다.
   *
   * 호출 체인:
   *   mem_access_t 생성자 → [init()]
   */
  void init(gpgpu_context *ctx);

  unsigned m_uid;
  /* [한국어] 이 메모리 접근 요청의 전역 고유 ID.
   * 설정자: init()에서 gpgpu_context의 전역 카운터로 자동 할당(단조 증가).
   * 읽는 자: 디버그 출력, 통계. 값 범위: 1부터 단조 증가. */

  new_addr_type m_addr;
  // request address
  /* [한국어] 접근할 메모리의 가상 주소 (합치기 후 캐시 라인/섹터 정렬된 시작 주소).
   * 설정자: 생성자의 address 인자, set_addr().
   * 읽는 자: 캐시 태그 비교(get_addr()), DRAM 주소 디코딩(addrdec.cc). */

  bool m_write;
  /* [한국어] 쓰기(store) 여부. true=store, false=load.
   * 설정자: 생성자 wr 인자.
   * 읽는 자: is_write() → 캐시 write policy 결정, mem_fetch 방향 설정. */

  unsigned m_req_size;
  // bytes
  /* [한국어] 이 메모리 요청의 크기(바이트). 보통 캐시 라인(128B) 또는 섹터(32B) 단위로 정렬됨.
   * 설정자: 생성자. 읽는 자: get_size() → 메모리 대역폭 통계, mem_fetch 크기 필드. */

  mem_access_type m_type;
  /* [한국어] 메모리 접근 타입 (GLOBAL_ACC_R, LOCAL_ACC_W 등).
   * 설정자: 생성자 type 인자.
   * 읽는 자: get_type() → 캐시 접근 경로 결정(L1/L2/const$/tex$).
   * 이 타입에 따라 mem_fetch가 통과할 캐시 레벨이 결정된다. */

  active_mask_t m_warp_mask;
  /* [한국어] 이 메모리 접근에 참여하는 스레드들의 비트마스크(32비트).
   * 설정자: 완전 생성자의 active_mask 인자 (기본 생성자에서는 0).
   * 읽는 자: get_warp_mask() → MSHR에서 완료 시 깨울 스레드 집합 결정.
   * 비트 n = 1이면 스레드 n이 이 메모리 접근에 참여한다. */

  mem_access_byte_mask_t m_byte_mask;
  /* [한국어] 요청 내에서 실제로 접근되는 바이트들의 비트마스크(최대 128비트).
   * 섹터 기반 캐시에서 partial write 처리 및 fill 최적화에 사용.
   * NO_PARTIAL_WRITE(빈 마스크)면 전체 라인 쓰기 또는 마스크 없는 접근.
   * 설정자: 완전 생성자의 byte_mask 인자 (기본 생성자에서는 0). */

  mem_access_sector_mask_t m_sector_mask;
  /* [한국어] 요청이 접근하는 32B 섹터들의 비트마스크(4비트).
   * 128B 캐시 라인을 4개의 32B 섹터로 나눈 것 중 어떤 섹터가 필요한지 표시.
   * 섹터 기반 캐시(Volta+)에서 sector miss 단위의 데이터 이동에 사용.
   * 설정자: 완전 생성자의 sector_mask 인자 (기본 생성자에서는 0). */
};

/*
 * [한국어]
 * class mem_fetch - 메모리 페치(memory fetch) 패킷 전방 선언(forward declaration).
 *
 * mem_fetch는 mem_access_t에서 생성되어 L1→L2→DRAM 캐시 계층을 이동하는 실제 패킷이다.
 * 전방 선언만 하고 구체 정의는 src/gpgpu-sim/mem_fetch.h에 있다.
 * mem_fetch_interface::push(), mem_fetch_allocator::alloc()에서 포인터로만 사용되므로 전방 선언으로 충분.
 */
class mem_fetch;

/*
 * ==========================================================================
 * mem_fetch_interface - 메모리 페치 패킷을 받는 버퍼의 추상 인터페이스
 * ==========================================================================
 *
 * [한국어] 메모리 요청 패킷(mem_fetch)을 수신하는 버퍼의 순수 가상 인터페이스.
 * SM→L1, L1→L2, L2→DRAM 등 계층 간 전달 지점마다 이 인터페이스를 구현하는 버퍼가 존재한다.
 * 순수 가상 함수만 있으므로 직접 인스턴스화 불가 — 반드시 파생 클래스에서 구현해야 한다.
 *
 * 사용 예:
 *   shader.cc의 메모리 단계에서 mem_fetch_interface::push()로 L1 캐시에 요청을 전달.
 *   full() 확인 후 공간이 있으면 push() — 없으면 다음 사이클에 재시도(stall).
 */
class mem_fetch_interface {
 public:
  virtual bool full(unsigned size, bool write) const = 0;
  /* [한국어] 이 인터페이스 뒤의 버퍼가 가득 찼는지 확인 (순수 가상).
   * @size: 전달하려는 mem_fetch의 크기(바이트).
   * @write: true=쓰기 요청, false=읽기 요청.
   * @return: true이면 현재 수용 불가 — 호출자는 스톨(stall) 처리.
   * 사이클 컨텍스트: 매 사이클 push() 전에 호출하여 역압력(backpressure) 처리. */

  virtual void push(mem_fetch *mf) = 0;
  /* [한국어] 메모리 페치 요청을 버퍼에 추가 (순수 가상).
   * @mf: 전달할 mem_fetch 포인터. full()이 false인 상태에서만 호출해야 한다.
   * full()이 true인데 push()하면 버퍼 오버플로우 — 반드시 full() 확인 후 호출.
   * 소유권: mf의 소유권이 이 인터페이스 뒤의 큐로 이전된다. */
};

/*
 * ==========================================================================
 * mem_fetch_allocator - mem_fetch 객체 생성 팩토리 추상 인터페이스
 * ==========================================================================
 *
 * [한국어] mem_fetch 패킷을 생성하는 팩토리 메서드 패턴의 추상 인터페이스.
 * 다양한 호출 컨텍스트(SM의 메모리 단계, 캐시 내부 write-back 등)에서
 * 적절한 mem_fetch를 생성하기 위해 여러 오버로드(overload)를 제공한다.
 * 구체 구현은 shader.cc의 shader_core_mem_fetch_allocator에 있다.
 *
 * 읽는 자:
 *   shader.cc의 메모리 단계: m_accessq에서 mem_access_t를 꺼내 alloc()으로 mem_fetch 생성.
 *   gpu-cache.cc의 write-back/write-allocate: 내부 접근용 mem_fetch 생성.
 */
class mem_fetch_allocator {
 public:
  /*
   * [한국어]
   * alloc(addr, type, size, wr, cycle, streamID) - 단순 정보로 mem_fetch 생성
   *
   * @addr: 접근 주소. @type: 접근 타입. @size: 크기. @wr: 읽기/쓰기.
   * @cycle: 생성 사이클 번호. @streamID: 소속 스트림 ID.
   * @return: 새로 할당된 mem_fetch 포인터.
   *
   * 워프 정보 없이 단순한 주소+타입만으로 생성하는 경우에 사용 (캐시 내부 요청 등).
   */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const = 0;

  /*
   * [한국어]
   * alloc(inst, access, cycle) - warp_inst_t와 mem_access_t로 mem_fetch 생성
   *
   * @inst: 이 메모리 접근을 유발한 warp_inst_t 참조. 워프 ID, 스트림 ID 추출.
   * @access: mem_access_t 참조. 주소, 타입, 마스크 정보.
   * @cycle: 생성 사이클 번호.
   * @return: 새로 할당된 mem_fetch 포인터.
   *
   * shader.cc의 메모리 단계에서 가장 일반적으로 사용하는 오버로드.
   */
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const = 0;

  /*
   * [한국어]
   * alloc(addr, type, active_mask, byte_mask, sector_mask, ...) - 상세 정보 포함 생성
   *
   * @addr: 접근 주소. @type: 접근 타입. @active_mask: 참여 스레드 마스크.
   * @byte_mask: 바이트 마스크. @sector_mask: 섹터 마스크.
   * @size: 크기. @wr: 읽기/쓰기. @cycle: 사이클. @wid: 워프 ID.
   * @sid: SM ID. @tpc: TPC(텍스처 프로세싱 클러스터) ID.
   * @original_mf: 이 요청을 유발한 원본 mem_fetch (write-allocate read 등에서 사용).
   * @streamID: 스트림 ID.
   * @return: 새로 할당된 mem_fetch 포인터.
   *
   * 섹터 기반 캐시의 partial write-allocate나 세분화된 통계가 필요한 경우에 사용.
   */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32
/* [한국어] 하나의 명령어에서 사용할 수 있는 최대 레지스터 피연산자(operand) 수 = 32.
 * inst_t::out[] = 8개(목적지), inst_t::in[] = 24개(소스), 합계 32.
 * scoreboard에서 레지스터 의존성 추적 배열의 크기로 사용된다. */

/*
 * [한국어]
 * dram_callback_t - 메모리 요청 완료 시 호출할 콜백 함수 정보 구조체
 *
 * GPU의 원자적 연산(atomicAdd, atomicCAS 등)은 "글로벌 메모리에서 값 읽기 →
 * 수정 → 쓰기"를 단일 트랜잭션으로 보장해야 한다. 이 과정에서 DRAM에서 값을
 * 읽어온 뒤 수정 결과를 레지스터에 쓰는 동작을 콜백으로 구현한다.
 *
 * 설정자: warp_inst_t::add_callback() — 원자적 연산 명령어에서 호출.
 * 읽는 자: mem_fetch 완료 처리 경로에서 function(instruction, thread) 호출.
 */
struct dram_callback_t {
  dram_callback_t() {
    /* [한국어] 기본 생성자: 모든 포인터를 NULL로 초기화.
     * has_callback() = function != NULL로 콜백 존재 여부를 판정. */
    function = NULL;     // [한국어] 콜백 함수 포인터: 미등록 상태
    instruction = NULL;  // [한국어] 관련 명령어: 없음
    thread = NULL;       // [한국어] 관련 스레드: 없음
  }

  void (*function)(const class inst_t *, class ptx_thread_info *);
  /* [한국어] 메모리 요청 완료 시 호출할 콜백 함수 포인터.
   * 함수 시그니처: (const inst_t *명령어, ptx_thread_info *스레드) → void.
   * 원자적 연산의 결과값을 레지스터에 기록하는 함수(execute_atom_op 계열)를 가리킨다.
   * NULL이면 콜백 없음(일반 로드/스토어). */

  const class inst_t *instruction;
  /* [한국어] 이 콜백과 연결된 PTX 명령어 포인터.
   * 콜백 함수의 첫 번째 인자로 전달된다.
   * 원자적 연산의 종류(atomicAdd vs atomicCAS 등)와 피연산자 정보 접근에 사용. */

  class ptx_thread_info *thread;
  /* [한국어] 이 콜백과 연결된 PTX 스레드 정보 포인터.
   * 콜백 함수의 두 번째 인자로 전달된다.
   * 레지스터 파일에 결과값을 쓰기 위해 스레드 컨텍스트에 접근해야 한다. */
};

/*
 * ==========================================================================
 * inst_t - GPU 명령어 추상 기반 클래스 (개별 스레드/정적 수준)
 * ==========================================================================
 *
 * === 파일의 역할 ===
 * PTX/SASS 명령어 하나의 정적 속성을 담는 기반 클래스.
 * 명령어 종류(op), 소스/목적지 레지스터 번호, 실행 지연(latency),
 * 파이프라인(op_pipe), 메모리 공간(space), 캐시 힌트(cache_op),
 * 배리어 정보(bar_type, bar_id) 등을 포함한다.
 * warp_inst_t가 이를 상속하여 워프 단위 실행 정보(활성 마스크, 사이클, m_accessq)를 추가한다.
 * cuda-sim의 ptx_instruction도 이를 상속하여 PTX IR 정보를 추가한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * cuda-sim/ptx_ir.cc에서 PTX 파싱 시 inst_t 파생 클래스(ptx_instruction) 인스턴스가 생성.
 * warp_inst_t::issue() 시 inst_t 필드들(latency, initiation_interval, op, space 등)을
 * 파이프라인 스케줄링과 실행에 사용한다.
 *
 * === 타 모듈과의 연결 ===
 * - 파생 클래스: warp_inst_t(워프 실행), ptx_instruction(PTX IR)
 * - 읽는 모듈: shader.cc(issue/execute 단계), scoreboard.cc(레지스터 의존성 추적),
 *              accelwattch(전력 계산용 연산 분류), simt_stack(reconvergence_pc 참조)
 *
 * === 주요 필드 요약 ===
 * pc:                    명령어 프로그램 카운터 (PTX 주소).
 * op:                    실행 유닛 선택 코드 (LOAD_OP, ALU_OP, SFU_OP 등).
 * out[8] / in[24]:       목적지/소스 레지스터 번호. scoreboard 의존성 추적.
 * latency:               연산 완료까지 사이클 수. scoreboard 해제 타이밍.
 * initiation_interval:   같은 유닛에 다음 명령어 보낼 수 있는 최소 간격.
 * space:                 접근 메모리 공간 (global/shared/local/const/tex).
 * reconvergence_pc:      분기 재수렴 PC. -1=분기 아님, -2=함수 반환 주소.
 */
class inst_t {
 public:
  /*
   * [한국어]
   * inst_t - 명령어 기본 생성자
   *
   * 모든 멤버를 "유효하지 않음" 또는 "기본값" 상태로 초기화한다.
   * m_decoded = false 상태에서 valid()는 false를 반환한다.
   * PTX 파서가 명령어를 파싱하여 필드를 채우면 m_decoded = true로 설정된다.
   *
   * 호출 체인:
   *   ptx_instruction 생성자 → [inst_t()] 또는
   *   warp_inst_t 생성자 → [inst_t()]
   */
  inst_t() {
    m_decoded = false;                    // [한국어] 아직 디코딩되지 않은 상태 — valid() = false
    pc = (address_type)-1;                // [한국어] PC = -1 (유효하지 않은 주소 — 실수 실행 방지)
    reconvergence_pc = (address_type)-1;  // [한국어] 재수렴 PC = -1 (분기 아님 초기값)
    op = NO_OP;                           // [한국어] 연산 코드 = NO_OP (실행 유닛 없음)
    bar_type = NOT_BAR;                   // [한국어] 배리어 타입 = NOT_BAR (배리어 명령어 아님)
    red_type = NOT_RED;                   // [한국어] 리덕션 타입 = NOT_RED
    bar_id = (unsigned)-1;                // [한국어] 배리어 ID = -1 (미설정)
    bar_count = (unsigned)-1;             // [한국어] 배리어 참여 스레드 수 = -1 (미설정)
    oprnd_type = UN_OP;                   // [한국어] 피연산자 타입 = UN_OP (미분류)
    sp_op = OTHER_OP;                     // [한국어] 특수 연산 = OTHER_OP (기타)
    op_pipe = UNKOWN_OP;                  // [한국어] 파이프라인 = UNKOWN_OP (미결정)
    mem_op = NOT_TEX;                     // [한국어] 메모리 연산 = NOT_TEX (일반 메모리)
    const_cache_operand = 0;              // [한국어] 상수 캐시 피연산자 없음
    num_operands = 0;                     // [한국어] 피연산자 수 = 0
    num_regs = 0;                         // [한국어] 레지스터 수 = 0
    memset(out, 0, sizeof(unsigned));     // [한국어] 목적지 레지스터 배열 0 초기화 (주의: sizeof(out) 이어야 하지만 원본 유지)
    memset(in, 0, sizeof(unsigned));      // [한국어] 소스 레지스터 배열 0 초기화
    is_vectorin = 0;                      // [한국어] 벡터 입력 없음
    is_vectorout = 0;                     // [한국어] 벡터 출력 없음
    space = memory_space_t();             // [한국어] 메모리 공간 = undefined_space
    cache_op = CACHE_UNDEFINED;           // [한국어] 캐시 힌트 = 미정의
    latency = 1;                          // [한국어] 지연 = 1 사이클 (최소값, 파서에서 덮어씀)
    initiation_interval = 1;              // [한국어] 발행 간격 = 1 사이클 (최소값)
    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
      arch_reg.src[i] = -1;              // [한국어] 아키텍처 소스 레지스터 = -1 (미사용)
      arch_reg.dst[i] = -1;             // [한국어] 아키텍처 목적지 레지스터 = -1 (미사용)
    }
    isize = 0;                            // [한국어] 명령어 크기 = 0바이트 (파서에서 설정)
  }

  bool valid() const { return m_decoded; }
  /* [한국어] 이 명령어가 유효하게 디코딩되었는지 반환.
   * true이면 PTX 파서가 모든 필드를 채운 상태. false이면 미초기화 상태.
   * 파이프라인에서 유효성 검사에 사용. */

  /*
   * [한국어]
   * print_insn - 명령어 정보를 파일에 출력 (디버깅용, 가상 함수)
   *
   * @fp: 출력 대상 파일 포인터.
   *
   * inst_t 기본 구현은 PC만 출력. warp_inst_t가 오버라이드하여 활성 마스크도 출력.
   * ptx_instruction은 전체 PTX 어셈블리 문자열을 출력.
   *
   * 호출 체인:
   *   디버그 출력 경로 → [print_insn()] — virtual이므로 실제 타입의 구현 호출
   */
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);  // [한국어] 명령어의 PC 주소를 16진수로 출력
  }

  // [한국어] 명령어 종류 분류 판별 함수들 — 실행 유닛 및 파이프라인 선택에 사용.

  bool is_load() const {
    /* [한국어] 메모리 로드(읽기) 명령어인지 확인.
     * LOAD_OP: PTX ld 명령어. TENSOR_CORE_LOAD_OP: 텐서 코어 로드.
     * memory_load: PTXPlus의 메모리 로드 타입. */
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  bool is_store() const {
    /* [한국어] 메모리 스토어(쓰기) 명령어인지 확인.
     * STORE_OP: PTX st 명령어. TENSOR_CORE_STORE_OP: 텐서 코어 스토어. */
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); }
  /* [한국어] 단정밀도(FP32) 부동소수점 연산인지. SP 파이프라인에서 처리. */
  bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }
  /* [한국어] 단정밀도 나눗셈인지. SFU 파이프라인에서 처리. */
  bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }
  /* [한국어] 단정밀도 곱셈인지. SP 파이프라인에서 처리. */
  bool is_dp() const { return ((sp_op == DP___OP)); }
  /* [한국어] 배정밀도(FP64) 부동소수점 연산인지. DP 파이프라인에서 처리. */
  bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }
  /* [한국어] 배정밀도 나눗셈인지. */
  bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }
  /* [한국어] 배정밀도 곱셈인지. */
  bool is_imul() const { return ((sp_op == INT_MUL_OP)); }
  /* [한국어] 정수 곱셈인지. */
  bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }
  /* [한국어] 24비트 정수 곱셈인지. PTX mul24.lo/hi 명령어. */
  bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }
  /* [한국어] 32비트 정수 곱셈인지. */
  bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }
  /* [한국어] 정수 나눗셈인지. */
  bool is_sfu() const {
    /* [한국어] SFU(Special Function Unit, 특수 함수 유닛) 연산인지.
     * SFU: sqrt, log2, sin, exp, 텐서 코어 연산을 처리하는 전용 유닛.
     * SFU는 SP보다 낮은 처리량(throughput)을 가지며 initiation_interval이 더 크다. */
    return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
            (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
            (sp_op == TENSOR__OP));
  }
  bool is_alu() const { return (sp_op == INT__OP); }
  /* [한국어] 정수 ALU 연산인지. SP 파이프라인의 정수 연산(add, sub, and, or 등). */

  unsigned get_num_operands() const { return num_operands; }
  /* [한국어] 피연산자(operand) 수 반환. AccelWattch 전력 계산에 사용. */
  unsigned get_num_regs() const { return num_regs; }
  /* [한국어] 레지스터 피연산자 수 반환 (벡터 피연산자는 1개로 셈). */
  void set_num_regs(unsigned num) { num_regs = num; }
  /* [한국어] 레지스터 수 설정. PTX 파서에서 호출. */
  void set_num_operands(unsigned num) { num_operands = num; }
  /* [한국어] 피연산자 수 설정. PTX 파서에서 호출. */
  void set_bar_id(unsigned id) { bar_id = id; }
  /* [한국어] 배리어 ID 설정. PTX bar.sync/arrive/red 명령어 파싱 시 호출. */
  void set_bar_count(unsigned count) { bar_count = count; }
  /* [한국어] 배리어 참여 스레드 수 설정. bar.sync가 몇 스레드를 기다릴지. */

  // [한국어] 명령어의 주요 속성 필드들 (public — 파서와 실행 엔진에서 직접 접근).

  address_type pc;
  // program counter address of instruction
  /* [한국어] 이 명령어의 프로그램 카운터(PC) 가상 주소.
   * PTX 명령어 스트림에서의 고유 위치. SIMT 스택이 이 값으로 명령어를 식별한다.
   * 설정자: PTX 파서. 읽는 자: SIMT 스택 update(), fetch 단계, print_insn(). */

  unsigned isize;
  // size of instruction in bytes
  /* [한국어] 명령어의 바이트 크기. PTX에서 보통 8바이트(64비트). SASS는 16바이트.
   * 분기 명령어 후 다음 명령어 PC = pc + isize. */

  op_type op;
  // opcode (uarch visible)
  /* [한국어] 마이크로아키텍처에서 관찰 가능한 연산 코드(opcode).
   * 어떤 실행 유닛(ALU_OP, LOAD_OP, STORE_OP, BRANCH_OP, SFU_OP 등)을 사용할지 결정.
   * 설정자: PTX 파서 또는 디코더. 읽는 자: shader.cc의 issue stage에서 유닛 선택. */

  barrier_type bar_type;
  /* [한국어] 배리어 명령어 종류 (NOT_BAR, SYNC, ARRIVE, RED).
   * op == BARRIER_OP인 경우에만 유효. core_t::warp_waiting_at_barrier() 판정에 사용. */

  reduction_type red_type;
  /* [한국어] 배리어 리덕션 연산 종류 (NOT_RED, POPC_RED, AND_RED, OR_RED).
   * bar_type == RED인 경우에만 유효. 리덕션 결과를 계산하는 방식 결정. */

  unsigned bar_id;
  /* [한국어] 배리어 식별자 (0 ~ MAX_BARRIERS_PER_CTA-1).
   * 같은 bar_id를 가진 스레드들만 동기화된다.
   * core_t::reduction_storage[bar_id]의 인덱스. 설정자: set_bar_id(). */

  unsigned bar_count;
  /* [한국어] 이 배리어에 참여해야 하는 스레드 수. 이 수만큼 도착하면 배리어 완료.
   * PTX bar.arrive N에서 N에 해당. 설정자: set_bar_count(). */

  types_of_operands oprnd_type;
  // code (uarch visible) identify if the operation is an integer or a floating point
  /* [한국어] 피연산자 타입: 정수(INT_OP) 또는 부동소수점(FP_OP) 구분.
   * AccelWattch 전력 계산에서 정수/FP 연산 카운터를 분리하는 데 사용. */

  special_ops sp_op;
  // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
  /* [한국어] 세분화된 특수 연산 종류: INT__OP, FP__OP, FP_MUL_OP, FP_DIV_OP, DP___OP 등.
   * is_sfu(), is_fp(), is_dp() 등의 분류 함수에서 이 값으로 실행 유닛을 더 세밀하게 결정.
   * AccelWattch에서 각 유닛별 전력 카운터에 사용. */

  operation_pipeline op_pipe;
  // code (uarch visible) identify the pipeline of the operation (SP, SFU or MEM)
  /* [한국어] 이 명령어가 통과할 파이프라인: SP(스칼라 프로세서), SFU, MEM, DP, INT, TENSOR 등.
   * shader.cc의 issue stage에서 어느 파이프라인 레지스터에 명령어를 넣을지 결정.
   * 설정자: PTX 디코더. 읽는 자: shader_core_ctx::execute()의 파이프라인 선택. */

  mem_operation mem_op;
  // code (uarch visible) identify memory type
  /* [한국어] 메모리 연산 타입: NOT_TEX(일반), TEX(텍스처).
   * 텍스처 접근인 경우 텍스처 캐시(tex$)를 사용한다는 것을 나타낸다.
   * PTXPlus 실행 경로에서 사용됨. */

  bool const_cache_operand;
  // has a load from constant memory as an operand
  /* [한국어] 이 명령어가 상수 메모리에서 로드하는 피연산자를 가지는지 여부.
   * true이면 상수 캐시(const$)를 통해 피연산자를 읽어야 한다.
   * operand collector에서 상수 메모리 접근 경로를 별도로 처리하는 데 사용. */

  _memory_op_t memory_op;
  // memory_op used by ptxplus
  /* [한국어] PTXPlus에서 사용하는 메모리 연산 타입 (memory_load, memory_store, memory_none).
   * is_load(), is_store() 판별에서 op_type과 함께 사용. */

  unsigned num_operands;
  /* [한국어] 이 명령어의 총 피연산자(소스+목적지) 수.
   * 설정자: PTX 파서의 set_num_operands(). 읽는 자: AccelWattch 전력 계산. */

  unsigned num_regs;
  // count vector operand as one register operand
  /* [한국어] 레지스터 파일에 접근하는 피연산자 수 (벡터를 1개로 셈).
   * operand collector에서 레지스터 파일 뱅크 충돌 분석에 사용. */

  address_type reconvergence_pc;
  // -1 => not a branch, -2 => use function return address
  /* [한국어] SIMT 분기 명령어인 경우의 재수렴(reconvergence) PC.
   * -1: 분기 명령어가 아님 (직선 코드).
   * -2: PTX CALL 명령어 — 함수 반환 주소를 재수렴 PC로 사용.
   * 양수: 실제 재수렴 지점 PC (컴파일러가 계산한 포스트 도미네이터).
   * 설정자: PTX 파서. 읽는 자: simt_stack::update()의 recvg_pc 인자. */

  unsigned out[8];
  /* [한국어] 목적지(결과를 쓰는) 레지스터 번호 배열. 최대 8개(MAX_OUTPUT_VALUES).
   * 설정자: PTX 파서. 읽는 자: scoreboard::reserveRegisters() — 이 레지스터들에 "사용 중" 표시. */

  unsigned outcount;
  /* [한국어] 유효한 목적지 레지스터 수. out[0..outcount-1]이 유효. */

  unsigned in[24];
  /* [한국어] 소스(값을 읽는) 레지스터 번호 배열. 최대 24개(MAX_INPUT_VALUES).
   * 설정자: PTX 파서. 읽는 자: scoreboard::checkCollision() — RAW 해저드 검사. */

  unsigned incount;
  /* [한국어] 유효한 소스 레지스터 수. in[0..incount-1]이 유효. */

  unsigned char is_vectorin;
  /* [한국어] 소스 피연산자가 벡터 레지스터인지 여부. 1이면 벡터(float4, int4 등). */

  unsigned char is_vectorout;
  /* [한국어] 목적지 피연산자가 벡터 레지스터인지 여부. */

  int pred;
  // predicate register number
  /* [한국어] 조건(predicate) 레지스터 번호. -1이면 predication 없음.
   * @P 레지스터에 저장된 조건값이 false인 스레드는 이 명령어를 실행하지 않는다.
   * 결과는 m_warp_active_mask에 반영된다. */

  int ar1, ar2;
  /* [한국어] 주소 레지스터 번호. ar1: 베이스 주소 레지스터, ar2: 오프셋 레지스터.
   * 메모리 접근 주소 계산: effective_addr = reg[ar1] + reg[ar2] + immediate_offset. */

  // register number for bank conflict evaluation
  // [한국어] 레지스터 파일 뱅크 충돌 분석을 위한 아키텍처 레지스터 번호 구조체.
  // operand collector가 레지스터 파일의 각 뱅크에 동시 접근을 시도할 때 충돌 감지에 사용.
  struct {
    int dst[MAX_REG_OPERANDS];
    /* [한국어] 목적지 아키텍처 레지스터 번호 배열 (최대 MAX_REG_OPERANDS=32개).
     * -1이면 해당 슬롯 미사용. operand collector의 뱅크 충돌 분석에 사용. */
    int src[MAX_REG_OPERANDS];
    /* [한국어] 소스 아키텍처 레지스터 번호 배열 (최대 MAX_REG_OPERANDS=32개).
     * -1이면 해당 슬롯 미사용. operand collector의 뱅크 충돌 분석에 사용. */
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // [한국어] 이전 단일 배열 버전 (현재 dst/src 구조체로 교체됨)

  unsigned latency;
  // operation latency
  /* [한국어] 이 연산의 완료까지 걸리는 사이클 수(operation latency).
   * scoreboard에서 이 사이클 수 이후에 결과 레지스터를 "완료" 표시로 해제한다.
   * 예: FP add = 4사이클, INT = 1사이클, SFU = 16사이클 (GPU 세대마다 다름).
   * 설정자: PTX 파서에서 GPU 세대별 레이턴시 테이블로 설정. */

  unsigned initiation_interval;
  /* [한국어] 같은 파이프라인 유닛에 다음 명령어를 보낼 수 있는 최소 사이클 간격.
   * SP 파이프라인은 보통 initiation_interval = 1 (매 사이클 발행 가능).
   * SFU는 일부 연산에서 initiation_interval > latency (완전 파이프라인이 아님).
   * warp_inst_t::dispatch_delay()가 cycles를 이 값으로 초기화하고 매 사이클 감소.
   * 설정자: PTX 파서에서 GPU 세대별 테이블로 설정. */

  unsigned data_size;
  // what is the size of the word being operated on?
  /* [한국어] 연산하는 데이터 원소의 크기(바이트). 예: 4=float/int, 8=double, 2=half.
   * 메모리 접근 크기와 레지스터 파일 접근 단위 결정에 사용. */

  memory_space_t space;
  /* [한국어] 이 명령어가 접근하는 메모리 공간(global, shared, local, const, tex 등).
   * 메모리 접근 명령어(LOAD_OP, STORE_OP)에서 어느 캐시를 통과할지 결정하는 핵심 필드.
   * 설정자: PTX 파서. 읽는 자: warp_inst_t::generate_mem_accesses(). */

  cache_operator_type cache_op;
  /* [한국어] PTX 명령어의 캐시 동작 힌트(.ca, .cg, .cv, .wb 등).
   * CACHE_GLOBAL이면 L1 우회, CACHE_VOLATILE이면 캐시 무시.
   * 설정자: PTX 파서. 읽는 자: gpu-cache.cc의 캐시 접근 정책 결정. */

 protected:  // [한국어] 보호 영역: inst_t와 파생 클래스에서만 직접 접근.

  bool m_decoded;
  /* [한국어] 이 명령어가 유효하게 디코딩(파싱+필드 채움) 되었는지 여부.
   * 설정자: PTX 파서에서 모든 필드 채운 후 true로 설정.
   * 읽는 자: valid() getter → 파이프라인 유효성 검사. */

  virtual void pre_decode() {}
  /* [한국어] 디코딩 전 처리 훅(hook) 함수. 파생 클래스에서 재정의 가능.
   * 현재 기본 구현은 빈 함수. ptx_instruction이 오버라이드할 수 있다. */
};

enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };
/* [한국어] GPU SIMT 분기 발산(divergence) 처리 방식 열거형.
 * POST_DOMINATOR: 포스트 도미네이터(post-dominator) 기반 재수렴 방식.
 *   warp가 분기할 때 컴파일러가 계산한 포스트 도미네이터(양쪽 경로가 다시 합류하는 점)를
 *   reconvergence_pc에 저장하고, SIMT 스택에 push한다.
 *   이 방식은 Hardware-based post-dominator reconvergence (Fung et al., MICRO'07)에서 제안됨.
 * NUM_SIMD_MODEL: 모델 개수 카운터 (배열 크기 등에 사용).
 * 읽는 자: simt_stack 구현부, gpgpusim.config의 분기 발산 설정. */

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;
/* [한국어] 하나의 명령어에서 하나의 스레드가 수행할 수 있는 최대 메모리 접근 횟수.
 * 128B 캐시 라인을 16B 섹터(Volta+) 기준으로 최대 8개 섹터에 접근 가능.
 * per_thread_info::memreqaddr[] 배열 크기를 이 값으로 결정한다.
 * 예: struct.x, .y, .z, .w가 각기 다른 캐시 라인에 흩어진 경우 4번 접근 발생.
 * 값 범위: 항상 8. 변경 시 per_thread_info 구조체와 set_addr()도 맞춰야 함. */

/*
 * [한국어 설명] warp_inst_t - 워프 단위 명령어 실행 컨텍스트 클래스
 *
 * === 파일의 역할 ===
 * inst_t를 상속받아 32개 스레드(1개 warp)가 함께 실행하는 명령어의
 * 타이밍/상태 정보를 포함하는 핵심 클래스.
 * 개별 스레드의 메모리 접근 주소(per_thread_info), 원자적 연산 콜백,
 * 활성 스레드 마스크(m_warp_active_mask), 발행 타이밍(issue_cycle),
 * 발행 간격 카운터(cycles), 생성된 메모리 접근 큐(m_accessq) 등을 보유한다.
 * 파이프라인의 한 슬롯에 이 객체가 들어가 이슈(issue) → 실행(execute) → 완료(complete)
 * 사이클 전체를 추적한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 타이밍 모델(gpgpu-sim/)에서 SM 내부 파이프라인 레지스터의 기본 단위.
 * warp_inst_t 포인터 배열이 issue→decode→execute→writeback 단계들 간 이동한다.
 * move_warp()로 슬롯 간 포인터 스왑. shader_core_ctx가 소유하는 배열에서 관리.
 *
 * 호출 체인(워프 발행):
 *   shader_core_ctx::issue_warp() → warp_inst_t::issue() [활성 마스크·사이클 기록]
 *     → generate_mem_accesses() [메모리 명령어이면 m_accessq 채움]
 *       → memory_coalescing_arch*() → mem_access_t 생성 및 m_accessq push
 *
 * === 타 모듈과의 연결 ===
 * - inst_t (부모): pc, op, latency, space, cache_op 등 정적 명령어 속성.
 * - scoreboard.cc: warp_inst_t의 out[] 레지스터를 reserveRegisters()로 예약.
 *   완료 시 releaseRegisters()로 해제.
 * - mem_fetch.cc: m_accessq의 mem_access_t를 소비하여 mem_fetch 생성.
 * - simt_stack: reconvergence_pc를 읽어 분기 처리.
 * - dram_callback_t: per_thread_info의 callback을 원자적 연산 완료 시 호출.
 * - accelwattch: op, sp_op, m_isatomic 등을 읽어 전력 카운터 업데이트.
 *
 * === 주요 함수/구조체 요약 ===
 * issue():                    활성 마스크·워프ID·사이클을 기록하고 슬롯을 '유효' 상태로 설정.
 * generate_mem_accesses():    스레드별 주소로부터 코얼레스드 mem_access_t를 m_accessq에 생성.
 * memory_coalescing_arch():   아키텍처별(Fermi/Volta) 합치기 알고리즘 구현.
 * dispatch_delay():           cycles 카운터로 initiation_interval 경과 여부 추적.
 * per_thread_info:            스레드당 memreqaddr[] + dram_callback_t 묶음.
 * m_accessq:                  생성된 mem_access_t 목록. ld_st 유닛이 팝(pop)하여 처리.
 */
class warp_inst_t : public inst_t {  // [한국어] inst_t를 public으로 상속 — 정적 명령어 속성 전부 물려받음
 public:
  // constructors

  /*
   * [한국어]
   * warp_inst_t() - 기본 생성자 (core_config 없이 빈 슬롯으로 초기화)
   *
   * m_config = NULL이므로 warp_size 등 코어 설정에 의존하는 함수(set_addr, add_callback 등)는
   * 이 생성자로 만든 객체에서 사용하면 안 된다. NULL 역참조 발생.
   * 주로 파이프라인 슬롯 배열의 초기값(플레이스홀더)으로 사용.
   * 실제 시뮬레이션에서는 core_config를 받는 생성자를 사용.
   *
   * 호출 체인:
   *   파이프라인 슬롯 배열 초기화 → [warp_inst_t()]
   */
  warp_inst_t() {
    m_uid = 0;                            // [한국어] 고유 명령어 ID = 0 (미설정)
    m_streamID = (unsigned long long)-1;  // [한국어] CUDA 스트림 ID = -1 (유효하지 않음)
    m_empty = true;                       // [한국어] 슬롯 비어있음 — 유효 명령어 없음
    m_config = NULL;                      // [한국어] 코어 설정 없음 — 크기 의존 함수 사용 불가

    // Ni: (Ni 개발자가 Ampere/Hopper 비동기 복사 명령어 지원을 위해 추가)
    m_is_ldgsts = false;     // [한국어] ldgsts(비동기 글로벌→공유 메모리 복사 명령어) 아님
    m_is_ldgdepbar = false;  // [한국어] ldgdepbar(로드 의존성 배리어 체크 명령어) 아님
    m_is_depbar = false;     // [한국어] depbar(의존성 배리어 대기 명령어) 아님

    m_depbar_group_no = 0;   // [한국어] 의존성 배리어 그룹 번호 = 0 (미사용)
  }

  /*
   * [한국어]
   * warp_inst_t(config) - core_config를 받는 생성자 (실제 시뮬레이션용)
   *
   * @config: shader_core_ctx가 보유한 core_config 포인터. warp_size, 캐시 설정 등 포함.
   *
   * 실제 파이프라인 슬롯에 사용되는 warp_inst_t 인스턴스를 초기화한다.
   * m_per_scalar_thread 벡터는 issue() 시점에 lazy 할당(resize)한다.
   * should_do_atomic = true로 초기화 — 원자적 연산 명령어인 경우
   * execute 단계에서 실제로 atomic을 수행할지 결정.
   *
   * 호출 체인:
   *   shader_core_ctx 생성자 → [warp_inst_t(config)] (파이프라인 슬롯 배열 할당)
   */
  warp_inst_t(const core_config *config) {
    m_uid = 0;                                              // [한국어] 고유 명령어 ID = 0 (issue 시 설정)
    m_streamID = (unsigned long long)-1;                    // [한국어] CUDA 스트림 ID = -1 (미설정)
    assert(config->warp_size <= MAX_WARP_SIZE);             // [한국어] 워프 크기가 최대(MAX_WARP_SIZE=1024) 이하인지 검증
    m_config = config;                                      // [한국어] 코어 설정 저장 — warp_size 등 접근 가능
    m_empty = true;                                         // [한국어] 슬롯 비어있음 (issue 전까지 true)
    m_isatomic = false;                                     // [한국어] 원자적 연산 명령어 아님 (add_callback 시 설정)
    m_per_scalar_thread_valid = false;                      // [한국어] 스레드별 정보 미초기화 — lazy alloc
    m_mem_accesses_created = false;                         // [한국어] 메모리 접근 큐 미생성 (generate_mem_accesses 호출 전)
    m_cache_hit = false;                                    // [한국어] 캐시 히트 아님 (초기값)
    m_is_printf = false;                                    // [한국어] PTX vprintf 명령어 아님
    m_is_cdp = 0;                                          // [한국어] CDP(CUDA Dynamic Parallelism) 명령어 아님
    should_do_atomic = true;                                // [한국어] 원자적 연산 실행 허용 플래그 초기값

    // Ni: (Ni 개발자가 추가한 Ampere/Hopper 비동기 명령어 지원 필드)
    m_is_ldgsts = false;     // [한국어] ldgsts 아님
    m_is_ldgdepbar = false;  // [한국어] ldgdepbar 아님
    m_is_depbar = false;     // [한국어] depbar 아님

    m_depbar_group_no = 0;   // [한국어] 의존성 배리어 그룹 번호 = 0
  }
  virtual ~warp_inst_t() {}
  /* [한국어] 가상 소멸자 — ptx_instruction 등 파생 클래스 객체를 warp_inst_t* 포인터로
   * delete할 때 파생 클래스 소멸자가 올바르게 호출되도록 보장. */

  // modifiers

  void broadcast_barrier_reduction(const active_mask_t &access_mask);
  /* [한국어] 배리어 리덕션 결과를 access_mask의 모든 스레드에 브로드캐스트.
   * bar.red (POPC/AND/OR) 명령어 완료 시 결과를 참여 스레드에 전파.
   * 호출자: shader.cc execute 단계의 배리어 리덕션 완료 핸들러. */

  void do_atomic(bool forceDo = false);
  /* [한국어] 워프의 모든 활성 스레드에 대해 원자적(atomic) 연산을 수행.
   * @forceDo: true이면 should_do_atomic 무관하게 강제 실행. 기본 false.
   * 원자적 연산(atomicAdd, atomicCAS 등)은 전역 메모리에서 RMW를 보장해야 하므로
   * DRAM 응답 시점에 콜백으로 per_thread_info의 callback을 호출한다.
   * 호출자: shader.cc ld_st 유닛의 원자적 연산 완료 처리. */

  void do_atomic(const active_mask_t &access_mask, bool forceDo = false);
  /* [한국어] access_mask에 지정된 특정 스레드들에 대해서만 원자적 연산 수행.
   * 배리어 리덕션과 달리 일부 스레드만 원자적 연산을 하는 경우에 사용. */

  void clear() { m_empty = true; }
  /* [한국어] 명령어 슬롯을 비워 재사용 가능 상태로 표시.
   * m_empty = true로 설정. 이 슬롯을 점유했던 명령어가 파이프라인에서 제거될 때 호출.
   * m_accessq나 m_per_scalar_thread는 clear하지 않으므로 issue()에서 덮어써진다. */

  /*
   * [한국어]
   * issue - 명령어를 실행 파이프라인에 발행(Issue)
   *
   * @mask:            실행할 스레드들의 활성 비트마스크 (warp_size 비트, 1=활성).
   * @warp_id:         이 명령어를 발행하는 워프의 정적 ID (0 ~ max_warps_per_core-1).
   * @cycle:           현재 시뮬레이션 사이클. issue_cycle에 기록됨.
   * @dynamic_warp_id: 이 명령어 실행에 사용된 동적 워프 슬롯 번호.
   * @sch_id:          이 명령어를 발행한 warp scheduler ID (다중 스케줄러 추적용).
   * @streamID:        이 워프가 속하는 CUDA 스트림 ID.
   *
   * 1) m_warp_issued_mask = m_warp_active_mask = mask 설정.
   * 2) m_warp_id, m_dynamic_warp_id, issue_cycle, m_scheduler_id 기록.
   * 3) cycles = initiation_interval (dispatch_delay 카운터 초기화).
   * 4) m_empty = false로 슬롯 유효 표시.
   * 5) scoreboard에 출력 레지스터(out[]) 예약.
   *
   * 호출 체인:
   *   shader_core_ctx::issue_warp() → [issue()] → scoreboard::reserveRegisters()
   */
  void issue(const active_mask_t &mask, unsigned warp_id,
             unsigned long long cycle, int dynamic_warp_id, int sch_id,
             unsigned long long streamID);

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }
  /* [한국어] 현재 활성 스레드 마스크(m_warp_active_mask) 반환.
   * predication 적용 후의 실제 실행 스레드 집합. scoreboard, accelwattch, 통계에서 참조. */

  void completed(unsigned long long cycle) const;
  // stat collection: called when the instruction is completed
  /* [한국어] 명령어 완료 시 통계 수집을 위해 호출되는 함수.
   * @cycle: 완료된 시뮬레이션 사이클 번호.
   * issue_cycle과 비교하여 실행 지연(latency) 통계를 기록한다.
   * 호출자: shader.cc writeback 단계. const이므로 상태는 수정하지 않음. */

  /*
   * [한국어]
   * set_addr - 특정 스레드(lane)의 메모리 접근 주소를 1개 설정
   *
   * @n:    워프 내 스레드 번호 (lane index, 0 ~ warp_size-1).
   * @addr: 이 스레드가 접근할 유효 주소(effective address).
   *
   * m_per_scalar_thread를 아직 할당하지 않은 경우 lazy하게 resize().
   * memreqaddr[0]에 주소를 저장. 단일 접근 명령어(일반 ld/st)에서 호출.
   *
   * 호출 체인:
   *   ptx_thread_info::execute_instruction() → [set_addr(n, addr)]
   */
  void set_addr(unsigned n, new_addr_type addr) {
    if (!m_per_scalar_thread_valid) {                  // [한국어] 스레드별 info 배열이 아직 미할당이면
      m_per_scalar_thread.resize(m_config->warp_size); // [한국어] warp_size 크기로 lazy 할당
      m_per_scalar_thread_valid = true;                // [한국어] 이후 중복 resize 방지
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;       // [한국어] 첫 번째(유일한) 접근 주소 설정
  }

  /*
   * [한국어]
   * set_addr - 특정 스레드의 메모리 접근 주소를 num_addrs개 설정 (다중 접근용)
   *
   * @n:         워프 내 스레드 번호.
   * @addr:      주소 배열 포인터 (길이 num_addrs).
   * @num_addrs: 접근 주소 수. MAX_ACCESSES_PER_INSN_PER_THREAD(8) 이하여야 함.
   *
   * 벡터 로드(ld.v4) 또는 여러 메모리 공간에 걸친 접근 시 여러 주소를 한꺼번에 설정.
   *
   * 호출 체인:
   *   ptx_thread_info::execute_instruction() → [set_addr(n, addr, num_addrs)]
   */
  void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size); // [한국어] lazy 할당
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // [한국어] 배열 오버플로우 방지 검증
    for (unsigned i = 0; i < num_addrs; i++)
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];       // [한국어] 각 접근 주소를 순서대로 저장
  }

  /*
   * [한국어]
   * print_m_accessq - 생성된 메모리 접근 큐(m_accessq)를 stdout에 출력 (디버깅용)
   *
   * m_accessq가 비어있으면 아무것도 출력하지 않는다.
   * 비어있지 않으면 각 mem_access_t의 타입 문자열, 주소(16진수), 크기를 출력.
   * generate_mem_accesses() 후 코얼레싱 결과를 확인할 때 유용.
   *
   * 호출 체인:
   *   디버그/검증 코드 → [print_m_accessq()]
   */
  void print_m_accessq() {
    if (accessq_empty())  // [한국어] 큐가 비어있으면 출력 없이 반환
      return;
    else {
      printf("Printing mem access generated\n");  // [한국어] 디버그 헤더 출력
      std::list<mem_access_t>::iterator it;        // [한국어] m_accessq 순회용 이터레이터
      for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {  // [한국어] 모든 mem_access_t 순회
        printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
               mem_access_type_str(it->get_type()), it->get_addr(),  // [한국어] 접근 타입/주소/크기 출력
               it->get_size());
      }
    }
  }

  /*
   * [한국어 설명] transaction_info - 메모리 코얼레싱 과정의 단일 트랜잭션 정보
   *
   * memory_coalescing_arch() 내부에서 워프 스레드들의 접근 패턴을
   * 하나의 캐시 라인 트랜잭션으로 묶을 때 사용하는 임시 구조체.
   * 128B 캐시 라인을 32B 청크 4개로 나눠 어느 청크에 접근하는지 추적하고,
   * 바이트 단위 접근 마스크(bytes)와 참여 스레드 마스크(active)를 함께 저장한다.
   * memory_coalescing_arch_reduce_and_send()에 전달되어 mem_access_t를 생성한다.
   */
  struct transaction_info {
    std::bitset<4> chunks;
    // bitmask: 32-byte chunks accessed
    /* [한국어] 128B 캐시 라인을 32B 청크 4개로 나눈 비트마스크.
     * 비트 i가 1이면 i번째 32B 청크([i*32, i*32+31])에 접근.
     * Fermi 아키텍처에서 캐시 라인이 4개 섹터로 나뉠 때의 트랜잭션 수 계산에 사용. */

    mem_access_byte_mask_t bytes;
    /* [한국어] 128B 캐시 라인 내에서 실제로 접근하는 바이트들의 마스크.
     * 비트 i가 1이면 캐시 라인 내 i번째 바이트에 접근.
     * mem_access_t 생성 시 byte_mask 필드에 그대로 전달된다. */

    active_mask_t active;
    // threads in this transaction
    /* [한국어] 이 트랜잭션에 참여하는(같은 캐시 라인에 접근하는) 스레드 비트마스크.
     * warp 내 일부 스레드만 특정 캐시 라인에 접근할 때 해당 스레드만 1.
     * mem_access_t 생성 시 warp_mask 필드에 전달된다. */

    /*
     * [한국어]
     * test_bytes - start_bit부터 end_bit까지 범위에 접근된 바이트가 있는지 검사
     *
     * @start_bit: 검사 시작 바이트 인덱스.
     * @end_bit:   검사 종료 바이트 인덱스 (포함).
     * @return:    범위 내 접근된 바이트가 하나 이상이면 true.
     *
     * 트랜잭션이 특정 32B 청크를 실제로 포함하는지 판단하는 데 사용.
     */
    bool test_bytes(unsigned start_bit, unsigned end_bit) {
      for (unsigned i = start_bit; i <= end_bit; i++)
        if (bytes.test(i)) return true;  // [한국어] 범위 내 접근 바이트 발견 시 즉시 true 반환
      return false;                       // [한국어] 범위 내 접근 없으면 false
    }
  };

  /*
   * [한국어]
   * generate_mem_accesses - 워프의 스레드별 주소에서 코얼레스드 mem_access_t 목록 생성
   *
   * 기능 시뮬레이션 단계에서 각 스레드의 memreqaddr[]이 이미 설정된 후,
   * 타이밍 모델에서 이 함수를 호출하여 실제 메모리 트랜잭션을 생성한다.
   * 내부에서 space.get_type()으로 메모리 공간을 판별하고
   * memory_coalescing_arch() 또는 memory_coalescing_arch_atomic()을 호출한다.
   * 결과는 m_accessq에 push된다. m_mem_accesses_created = true로 표시.
   *
   * 호출 체인:
   *   shader_core_ctx::ld_st_unit_pipeline() → [generate_mem_accesses()]
   *     → memory_coalescing_arch() → memory_coalescing_arch_reduce_and_send()
   *       → m_accessq에 mem_access_t push
   */
  void generate_mem_accesses();

  /*
   * [한국어]
   * memory_coalescing_arch - GPU 아키텍처 방식의 메모리 코얼레싱 구현
   *
   * @is_write:    true이면 스토어(write), false이면 로드(read).
   * @access_type: 생성할 mem_access_t의 타입 (GLOBAL_ACC_R/W, LOCAL_ACC_R/W 등).
   *
   * 각 스레드의 memreqaddr[0]을 기준으로 동일 128B 캐시 라인에 접근하는
   * 스레드들을 묶어 transaction_info를 구성하고 reduce_and_send()로 전달한다.
   * Volta+ 섹터 캐시(32B 섹터 단위 코얼레싱)도 이 함수에서 분기하여 처리.
   *
   * 호출 체인:
   *   generate_mem_accesses() → [memory_coalescing_arch()]
   *     → memory_coalescing_arch_reduce_and_send()
   */
  void memory_coalescing_arch(bool is_write, mem_access_type access_type);

  /*
   * [한국어]
   * memory_coalescing_arch_atomic - 원자적 연산의 메모리 코얼레싱
   *
   * @is_write:    true이면 CAS/exch 스토어 경로, false이면 로드 경로.
   * @access_type: GLOBAL_ACC_R/W 등.
   *
   * 원자적 연산은 스레드별로 독립적인 RMW를 수행해야 하므로
   * 일반 코얼레싱과 달리 각 스레드의 접근을 별도 트랜잭션으로 분리.
   * (또는 동일 주소 접근을 serialization하는 방식으로 처리.)
   *
   * 호출 체인:
   *   generate_mem_accesses() [m_isatomic == true] → [memory_coalescing_arch_atomic()]
   */
  void memory_coalescing_arch_atomic(bool is_write,
                                     mem_access_type access_type);

  /*
   * [한국어]
   * memory_coalescing_arch_reduce_and_send - transaction_info로부터 mem_access_t 생성 및 큐에 삽입
   *
   * @is_write:     스토어 여부.
   * @access_type:  mem_access_t 타입.
   * @info:         코얼레싱 계산으로 구성된 transaction_info.
   * @addr:         이 트랜잭션의 기준 주소(캐시 라인 정렬 주소).
   * @segment_size: 한 트랜잭션의 크기 (예: 128B 또는 32B 섹터).
   *
   * info로부터 mem_access_t(access_type, addr, size, warp_mask, byte_mask, sector_mask)를
   * 생성하여 m_accessq.push_front()로 삽입한다.
   * accessq_count()가 증가하며, ld_st 유닛이 accessq_pop_back()으로 꺼내 mem_fetch 생성.
   *
   * 호출 체인:
   *   memory_coalescing_arch() → [memory_coalescing_arch_reduce_and_send()]
   *     → m_accessq.push_front(mem_access_t)
   */
  void memory_coalescing_arch_reduce_and_send(bool is_write,
                                              mem_access_type access_type,
                                              const transaction_info &info,
                                              new_addr_type addr,
                                              unsigned segment_size);

  /*
   * [한국어]
   * add_callback - 특정 스레드(lane)에 원자적 연산 완료 콜백 등록
   *
   * @lane_id:  워프 내 스레드 번호 (0 ~ warp_size-1).
   * @function: DRAM에서 원자적 RMW 완료 후 호출할 함수 포인터.
   * @inst:     이 원자적 연산을 기술하는 inst_t 포인터.
   * @thread:   실행 컨텍스트 스레드 (ptx_thread_info — 레지스터 파일 접근에 사용).
   * @atomic:   true이면 m_isatomic = true로 플래그 설정.
   *
   * per_thread_info[lane_id].callback을 채운다. m_per_scalar_thread가 미할당이면
   * 이 시점에 lazy 할당한다.
   * DRAM 완료 후 dram_callback_t::function(instruction, thread)를 호출하여
   * 결과를 레지스터에 쓴다.
   *
   * 호출 체인:
   *   ptx_thread_info::execute_atomic() → [add_callback(lane_id, fn, inst, thread, true)]
   *     → 이후 dram_callback_t::function() 이 DRAM 응답 시 호출됨
   */
  void add_callback(unsigned lane_id,
                    void (*function)(const class inst_t *,
                                     class ptx_thread_info *),
                    const inst_t *inst, class ptx_thread_info *thread,
                    bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);  // [한국어] lazy 할당
      m_per_scalar_thread_valid = true;
      if (atomic) m_isatomic = true;  // [한국어] 원자적 연산임을 표시 — do_atomic() 호출 트리거
    }
    m_per_scalar_thread[lane_id].callback.function = function;    // [한국어] 완료 콜백 함수 등록
    m_per_scalar_thread[lane_id].callback.instruction = inst;     // [한국어] 원자적 연산 명령어 포인터
    m_per_scalar_thread[lane_id].callback.thread = thread;        // [한국어] 실행 스레드 컨텍스트
  }

  void set_active(const active_mask_t &active);
  /* [한국어] 활성 마스크를 외부에서 지정한 마스크로 설정.
   * predication 처리 후 실제 실행 스레드 집합을 업데이트할 때 사용.
   * 호출자: shader.cc의 issue 단계에서 predicate 레지스터 처리 후. */

  void clear_active(const active_mask_t &inactive);
  /* [한국어] inactive 마스크에 해당하는 스레드를 활성 마스크에서 제거(AND NOT).
   * 분기 발산 시 한쪽 경로에서 비활성화될 스레드 집합을 제거하는 데 사용.
   * 호출자: simt_stack 분기 처리 코드. */

  void set_not_active(unsigned lane_id);
  /* [한국어] lane_id 스레드 하나를 활성 마스크에서 제거.
   * 개별 스레드 예외 처리(예: 캐시 히트로 특정 스레드만 조기 완료) 시 사용. */

  // accessors

  /*
   * [한국어]
   * print_insn - PC와 워프 활성 마스크를 파일에 출력 (inst_t 가상 함수 오버라이드)
   *
   * @fp: 출력 파일 포인터.
   *
   * PC를 16진수로 출력한 뒤, warp_size개 비트를 MSB(높은 스레드 번호)부터 출력.
   * 활성 스레드는 '1', 비활성은 '0'. 예: "[inst @ pc=0x0080] 11110000...0000"
   * 디버그 및 파이프라인 trace 출력에서 사용.
   */
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);  // [한국어] PC 주소 출력
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)  // [한국어] MSB(높은 스레드번호)부터 출력
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));  // [한국어] 활성='1', 비활성='0'
  }

  bool active(unsigned thread) const { return m_warp_active_mask.test(thread); }
  /* [한국어] 워프 내 thread번 스레드가 이 명령어에서 활성인지 반환.
   * predication 적용 후의 실제 실행 여부. scoreboard, accelwattch에서 사용. */

  unsigned active_count() const { return m_warp_active_mask.count(); }
  /* [한국어] 이 명령어에서 실제로 실행하는 활성 스레드 수 반환.
   * 32이면 full warp. 32보다 작으면 partial warp (divergence 또는 CTA 경계). */

  unsigned issued_count() const {  // for instruction counting
    /* [한국어] 발행 시점의 활성 스레드 수 반환 (predication 적용 전 m_warp_issued_mask 기준).
     * 통계 카운터에서 "발행된 스레드 수" 계산에 사용. m_empty == false 보장. */
    assert(m_empty == false);  // [한국어] 빈 슬롯에서 호출 시 assert — 유효한 명령어여야 함
    return m_warp_issued_mask.count();
  }

  bool empty() const { return m_empty; }
  /* [한국어] 이 명령어 슬롯이 비어있는지(유효 명령어가 없는지) 반환.
   * true이면 clear() 호출 후 또는 미초기화 상태. 파이프라인 단계 체크에 사용. */

  unsigned warp_id() const {
    /* [한국어] 이 명령어가 속한 워프의 정적 ID 반환.
     * m_empty == true이면 assert 실패. 유효한 명령어 슬롯에서만 호출해야 함. */
    assert(!m_empty);  // [한국어] 빈 슬롯에서 warp_id 읽기 방지
    return m_warp_id;
  }

  unsigned warp_id_func() const  // to be used in functional simulations only
  /* [한국어] 기능 시뮬레이션 전용 warp_id getter — assert 없음.
   * 타이밍 모델과 달리 기능 시뮬레이션에서는 m_empty 상태 무관하게 접근할 수 있다.
   * (기능 시뮬레이션은 슬롯 유효성 검사 없이 warp_id를 추적함.) */
  {
    return m_warp_id;
  }

  unsigned dynamic_warp_id() const {
    /* [한국어] 동적 워프 ID 반환. 스케줄러가 실제로 이 명령어를 발행할 때 결정되는 ID.
     * 정적 warp_id와 달리 라운드로빈/GTO 등 스케줄 정책에 따라 변할 수 있다. */
    assert(!m_empty);  // [한국어] 빈 슬롯에서 접근 방지
    return m_dynamic_warp_id;
  }

  bool has_callback(unsigned n) const {
    /* [한국어] 스레드 n에 원자적 연산 콜백이 등록되어 있는지 확인.
     * 활성 스레드이면서 per_scalar_thread가 유효하고 콜백 함수 포인터가 NULL이 아닌 경우.
     * do_atomic()에서 호출하여 실제 콜백 호출 여부를 결정. */
    return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
           (m_per_scalar_thread[n].callback.function != NULL);
  }

  new_addr_type get_addr(unsigned n) const {
    /* [한국어] 스레드 n의 메모리 접근 유효 주소(memreqaddr[0]) 반환.
     * m_per_scalar_thread_valid가 false이면 assert 실패.
     * 기능 시뮬레이션에서 스레드별 주소를 확인할 때 사용. */
    assert(m_per_scalar_thread_valid);  // [한국어] 주소 배열이 할당된 상태여야 함
    return m_per_scalar_thread[n].memreqaddr[0];
  }

  bool isatomic() const { return m_isatomic; }
  /* [한국어] 이 명령어가 원자적 연산(atomicAdd, atomicCAS 등)인지 반환.
   * add_callback(atomic=true) 호출 시 m_isatomic = true로 설정됨.
   * ld_st 유닛에서 atomic 경로와 일반 메모리 경로를 분기할 때 사용. */

  unsigned warp_size() const { return m_config->warp_size; }
  /* [한국어] 이 코어의 워프 크기 반환. 보통 32. active_mask_t 비트 수와 일치.
   * m_config->warp_size에서 읽음. */

  // [한국어] 메모리 접근 큐(m_accessq) 관련 accessor 함수들.
  // m_accessq는 generate_mem_accesses()에서 채워지고 ld_st 유닛이 소비하는 std::list.

  bool accessq_empty() const { return m_accessq.empty(); }
  /* [한국어] m_accessq가 비어있는지 반환. ld_st 유닛이 처리할 트랜잭션이 없는지 확인. */

  unsigned accessq_count() const { return m_accessq.size(); }
  /* [한국어] m_accessq에 대기 중인 mem_access_t 수 반환. 코얼레싱 비율 통계에 활용. */

  const mem_access_t &accessq_back() { return m_accessq.back(); }
  /* [한국어] m_accessq의 마지막 요소(가장 오래된 트랜잭션)를 const ref로 반환.
   * ld_st 유닛이 pop_back()으로 꺼내기 전에 내용을 확인하는 데 사용. */

  void accessq_pop_back() { m_accessq.pop_back(); }
  /* [한국어] m_accessq에서 마지막 요소 제거. ld_st 유닛이 mem_fetch 생성 후 호출. */

  /*
   * [한국어]
   * dispatch_delay - initiation_interval 카운터(cycles) 관리
   *
   * @return: 아직 지연 사이클이 남아있으면 true (같은 파이프라인에 다음 명령어 못 보냄).
   *
   * issue() 시 cycles = initiation_interval으로 초기화된다.
   * 파이프라인의 execute 단계마다 dispatch_delay()를 호출하면 cycles가 1씩 감소.
   * cycles가 0이 되면 false를 반환하여 같은 파이프라인에 다음 명령어 발행 허용.
   * SFU처럼 initiation_interval > 1인 유닛에서 throughput 제한을 구현.
   */
  bool dispatch_delay() {
    if (cycles > 0) cycles--;  // [한국어] 잔여 발행 지연 사이클 1 감소
    return cycles > 0;          // [한국어] 아직 지연 중이면 true (다음 명령어 발행 차단)
  }

  bool has_dispatch_delay() { return cycles > 0; }
  /* [한국어] 발행 지연이 남아있는지 확인 (cycles > 0이면 true).
   * dispatch_delay()를 호출하지 않고 현재 상태만 조회할 때 사용. */

  void print(FILE *fout) const;
  /* [한국어] 명령어의 전체 정보(PC, op, 레지스터, 활성 마스크 등)를 fout에 출력.
   * 파이프라인 덤프, 디버그 출력에서 사용. print_insn보다 더 상세한 정보. */

  unsigned get_uid() const { return m_uid; }
  /* [한국어] 이 명령어 인스턴스의 전역 고유 ID 반환.
   * 시뮬레이터 전체에서 명령어를 유일하게 식별. 통계, trace에서 사용. */

  unsigned long long get_streamID() const { return m_streamID; }
  /* [한국어] 이 명령어가 속하는 CUDA 스트림 ID 반환.
   * 멀티스트림 시뮬레이션에서 스트림별 통계 수집에 사용. */

  unsigned get_schd_id() const { return m_scheduler_id; }
  /* [한국어] 이 명령어를 발행한 warp scheduler의 ID 반환.
   * SM에 여러 스케줄러가 있을 때(예: 4개) 어느 스케줄러가 발행했는지 추적. */

  active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }
  /* [한국어] 활성 스레드 마스크를 값 복사로 반환.
   * get_active_mask()는 const ref 반환, 이 함수는 값 복사. 수정 가능한 복사본 필요 시 사용. */

 protected:  // [한국어] 보호 영역: warp_inst_t와 파생 클래스(ptx_instruction 등)에서만 직접 접근.

  unsigned m_uid;
  /* [한국어] 이 명령어 인스턴스의 전역 고유 ID (Unique ID).
   * 시뮬레이터 시작부터 단조 증가하는 카운터로 할당.
   * 설정자: issue() 내부 또는 warp_inst_t 생성 시. 읽는 자: get_uid(), 통계/trace. */

  unsigned long long m_streamID;
  /* [한국어] 이 명령어가 속하는 CUDA 스트림의 ID.
   * 멀티스트림 시뮬레이션에서 스트림별 성능 분리 분석에 사용.
   * 설정자: issue(). 읽는 자: get_streamID(). */

  bool m_empty;
  /* [한국어] 이 슬롯에 유효한 명령어가 없으면 true.
   * 생성 직후 및 clear() 호출 후 true. issue()에서 false로 전환.
   * empty(), warp_id(), dynamic_warp_id()에서 유효성 검사에 사용. */

  bool m_cache_hit;
  /* [한국어] 이 명령어의 메모리 접근이 캐시 히트(L1 또는 L2)인지.
   * 설정자: gpu-cache.cc에서 접근 응답 시 설정. 읽는 자: 통계 수집. */

  unsigned long long issue_cycle;
  /* [한국어] 이 명령어가 issue()된 시뮬레이션 사이클 번호.
   * completed() 호출 시 (current_cycle - issue_cycle)로 실제 실행 시간 계산.
   * 설정자: issue(). 읽는 자: completed() 통계 수집. */

  unsigned cycles;
  // used for implementing initiation interval delay
  /* [한국어] initiation_interval 지연을 구현하는 카운터.
   * issue() 시 cycles = inst_t::initiation_interval 값으로 초기화.
   * dispatch_delay()가 매 사이클 1씩 감소시키며, 0이 되면 같은 파이프라인에 다음 명령어 허용.
   * SFU(initiation_interval > 1)에서 throughput 제한을 이 카운터로 구현. */

  bool m_isatomic;
  /* [한국어] 이 명령어가 원자적 메모리 연산(atomicAdd, atomicCAS, atomicExch 등)인지.
   * add_callback(atomic=true) 호출 시 true로 설정.
   * isatomic() getter → ld_st 유닛이 atomic 경로 분기. */

  bool should_do_atomic;
  /* [한국어] 원자적 연산을 실제로 실행해야 하는지 여부.
   * 기본값 true. 특정 시뮬레이션 모드에서 atomic을 스킵하기 위해 false로 설정 가능.
   * do_atomic(forceDo) 호출 시 forceDo=false이면 이 플래그를 확인. */

  bool m_is_printf;
  /* [한국어] 이 명령어가 PTX vprintf(CUDA printf) 명령어인지.
   * true이면 ld_st 유닛이 printf 버퍼에 출력을 기록하는 특수 처리 경로로 분기. */

  unsigned m_warp_id;
  /* [한국어] 이 명령어를 실행하는 워프의 정적 ID (SM 내에서 고정된 번호).
   * issue() 시 warp_id 인자로 설정. 읽는 자: warp_id(), scoreboard, 통계. */

  unsigned m_dynamic_warp_id;
  /* [한국어] 이 명령어 발행 시 사용된 동적 워프 슬롯 번호.
   * 워프 스케줄러가 발행할 때마다 결정되는 동적 번호. 스케줄링 분석에 사용. */

  const core_config *m_config;
  /* [한국어] 이 워프가 실행되는 SM의 core_config 포인터.
   * warp_size, 캐시 설정, 파이프라인 수 등 접근. NULL이면 기본 생성자로 생성된 슬롯.
   * warp_size() getter, set_addr(), add_callback() 등에서 참조. */

  active_mask_t m_warp_active_mask;
  // dynamic active mask for timing model (after predication)
  /* [한국어] 타이밍 모델에서 실제 실행하는 스레드 집합 (predication 적용 후).
   * issue()에서 mask 인자로 초기화. predicate 레지스터 처리 후 clear_active()로 갱신.
   * active(), active_count(), get_active_mask()로 접근. */

  active_mask_t m_warp_issued_mask;
  // active mask at issue (prior to predication test) -- for instruction counting
  /* [한국어] 발행 시점의 활성 마스크 (predication 적용 전).
   * m_warp_active_mask와 달리 predication 후 변경되지 않는 불변 참조값.
   * issued_count()에서 사용 — 발행된 스레드 수 통계 카운터. */

  /*
   * [한국어 설명] per_thread_info - 워프 내 각 스칼라 스레드의 실행 정보
   *
   * 각 스레드의 메모리 접근 유효 주소(memreqaddr[])와
   * 원자적 연산 완료 콜백(callback)을 묶어 저장.
   * m_per_scalar_thread[warp_size] 배열로 스레드 수만큼 유지.
   * issue 전에는 lazy 할당(m_per_scalar_thread_valid = false).
   */
  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;  // [한국어] 모든 주소 슬롯을 0으로 초기화 (유효하지 않은 주소)
    }

    dram_callback_t callback;
    /* [한국어] 원자적 연산(atomicAdd 등) 완료 시 호출할 콜백 정보.
     * 설정자: warp_inst_t::add_callback(). 읽는 자: do_atomic() → callback.function(inst, thread).
     * 비원자적 명령어는 callback.function == NULL. */

    new_addr_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];
    // effective address, upto 8 different requests
    // (to support 32B access in 8 chunks of 4B each)
    /* [한국어] 이 스레드의 유효 메모리 주소 배열. 최대 MAX_ACCESSES_PER_INSN_PER_THREAD(8)개.
     * 설정자: set_addr(). 읽는 자: memory_coalescing_arch()가 각 스레드 주소를 읽어 코얼레싱.
     * 벡터 로드/스토어나 128B 라인을 여러 4B 청크로 나눠 접근하는 경우 여러 주소 설정.
     * 인덱스 0: 첫 번째(또는 유일한) 접근 주소. */
  };

  bool m_per_scalar_thread_valid;
  /* [한국어] m_per_scalar_thread 벡터가 할당(resize)되었는지.
   * false이면 set_addr/add_callback이 lazy 할당을 수행. true이면 이미 warp_size 크기 배열 존재.
   * 설정자: set_addr(), add_callback()의 lazy 할당 경로. 읽는 자: get_addr(), has_callback(). */

  std::vector<per_thread_info> m_per_scalar_thread;
  /* [한국어] 워프 내 각 스레드의 per_thread_info 배열 (크기: warp_size).
   * lazy 할당 — m_per_scalar_thread_valid가 false이면 빈 벡터. set_addr/add_callback에서 resize.
   * 인덱스: 워프 내 스레드 번호 (lane index, 0 ~ warp_size-1). */

  bool m_mem_accesses_created;
  /* [한국어] generate_mem_accesses()가 이미 호출되어 m_accessq가 채워졌는지.
   * false이면 ld_st 유닛이 generate_mem_accesses()를 호출해야 함.
   * 중복 호출 방지용 플래그. */

  std::list<mem_access_t> m_accessq;
  /* [한국어] 코얼레싱된 메모리 접근 요청(mem_access_t) 대기 큐.
   * generate_mem_accesses()에서 push_front()로 채워진다.
   * ld_st 유닛이 accessq_back()으로 접근하고 accessq_pop_back()으로 제거하면서 mem_fetch 생성.
   * std::list라서 front/back pop이 O(1). 큐 크기 = 코얼레싱 트랜잭션 수. */

  unsigned m_scheduler_id;
  // the scheduler that issues this inst
  /* [한국어] 이 명령어를 발행한 warp scheduler의 인덱스 (0 ~ num_schedulers_per_core-1).
   * SM에 복수의 스케줄러가 있을 때 어느 스케줄러가 발행했는지 추적. 통계 분리에 사용.
   * 설정자: issue(). 읽는 자: get_schd_id(). */

  // Jin: cdp support
 public:
  int m_is_cdp;
  /* [한국어] CDP(CUDA Dynamic Parallelism) 관련 명령어 플래그.
   * 0: CDP 아님. 양수: CDP 커널 launch 관련 명령어(cudaLaunchDevice 등).
   * Jin 개발자가 CDP 지원을 위해 추가. shader.cc의 CDP 분기 처리에서 참조.
   * CDP: GPU 커널 내부에서 새로운 GPU 커널을 동적으로 launch하는 기능. */

  // Ni: add boolean to indicate whether the instruction is ldgsts
  /* [한국어] Ni 개발자가 추가한 Ampere/Hopper 비동기 메모리 명령어 지원 필드들.
   * ldgsts: 글로벌 메모리 → 공유 메모리 비동기 복사 명령어 (PTX cp.async).
   * ldgdepbar: 비동기 복사 완료를 기다리는 의존성 배리어 로드 명령어.
   * depbar: 비동기 명령어 완료 의존성 배리어. */

  bool m_is_ldgsts;
  /* [한국어] ldgsts(load global store shared — 비동기 글로벌→공유 메모리 복사) 명령어인지.
   * Ampere cp.async에 해당. 비동기이므로 완료 대기 없이 다음 명령어 진행.
   * ld_st 유닛이 비동기 복사 경로로 분기할 때 확인. */

  bool m_is_ldgdepbar;
  /* [한국어] ldgdepbar(load global dependency barrier) 명령어인지.
   * 이전 ldgsts가 완료됐는지 확인하는 배리어 점검 명령어.
   * 완료되지 않은 비동기 복사가 있으면 스톨(stall). */

  bool m_is_depbar;
  /* [한국어] depbar(dependency barrier) 명령어인지.
   * 비동기 명령어들의 완료를 기다리는 범용 의존성 배리어.
   * m_depbar_group_no로 어느 비동기 명령어 그룹을 기다릴지 지정. */

  unsigned int m_depbar_group_no;
  /* [한국어] 의존성 배리어가 기다려야 하는 비동기 명령어 그룹 번호.
   * m_is_depbar == true인 경우에만 유효. 같은 그룹 번호의 비동기 명령어 완료 시까지 대기. */
};

/*
 * [한국어]
 * move_warp - 파이프라인 슬롯 간 warp_inst_t 포인터 이동 (포인터 스왑)
 *
 * @dst: 대상 슬롯 포인터 레퍼런스 (이동 후 이 포인터가 명령어를 가리킴).
 * @src: 원본 슬롯 포인터 레퍼런스 (이동 후 NULL 또는 빈 슬롯 포인터로 됨).
 *
 * 파이프라인 단계(fetch→decode→issue→execute→writeback) 간 명령어를 이동할 때
 * 객체 복사 없이 포인터만 스왑하여 효율적으로 처리한다.
 * 내부적으로 dst와 src의 포인터를 맞교환하는 swap 패턴으로 구현.
 *
 * 호출 체인:
 *   shader_core_ctx 파이프라인 advance 코드 → [move_warp(dst, src)]
 */
void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

/*
 * [한국어]
 * get_kernel_code_size - PTX 커널 함수의 코드 크기(바이트) 반환
 *
 * @entry: 크기를 구할 커널의 function_info 포인터.
 * @return: 바이트 단위 코드 크기. gpgpusim.config 파싱 후 자원 추정에 사용.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc 또는 커널 launch 경로 → [get_kernel_code_size(entry)]
 */
size_t get_kernel_code_size(class function_info *entry);

/*
 * [한국어 설명] checkpoint - 시뮬레이션 전역 메모리 상태 저장/복원 클래스
 *
 * === 파일의 역할 ===
 * 시뮬레이션 중간 상태(특히 GPU 전역 메모리)를 파일에 직렬화하여 저장하고,
 * 나중에 동일 지점부터 시뮬레이션을 재개(resume)할 수 있도록 지원한다.
 * gpgpu_functional_sim_config의 checkpoint_option, checkpoint_pc 옵션으로 활성화.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 기능 시뮬레이션(cuda-sim)의 PTX 실행 흐름에서 checkpoint_pc에 도달하면
 * store_global_mem()을 호출하여 메모리 스냅샷을 파일에 저장한다.
 * 재개 시(checkpoint_option=2) load_global_mem()으로 스냅샷을 복원한 후 실행 재개.
 *
 * === 타 모듈과의 연결 ===
 * - gpgpu_functional_sim_config: checkpoint_option, checkpoint_pc, checkpoint_CTA_t 옵션 참조.
 * - cuda-sim/ptx_ir.cc: 실행 중 checkpoint PC 도달 시 checkpoint 생성/복원 호출.
 * - memory_space: 전역 메모리 데이터 소스/대상.
 *
 * === 주요 함수 요약 ===
 * load_global_mem():  파일에서 전역 메모리 상태 복원.
 * store_global_mem(): 현재 전역 메모리 상태를 파일에 저장.
 */
class checkpoint {
 public:
  checkpoint();
  /* [한국어] 생성자: checkpoint 객체 기본 초기화.
   * 체크포인트 관련 파일 이름이나 오프셋을 초기화할 수 있음. */

  ~checkpoint() { printf("clasfsfss destructed\n"); }
  /* [한국어] 소멸자: 삭제 시 디버그 메시지 출력.
   * 주의: "clasfsfss"는 오타가 있는 원본 코드 — "checkpoint destructed"를 의도.
   * 코드 수정 금지 원칙에 따라 오타 그대로 유지. */

  void load_global_mem(class memory_space *temp_mem, char *f1name);
  /* [한국어] 체크포인트 파일(f1name)에서 전역 메모리 상태를 temp_mem에 복원.
   * @temp_mem: 복원 대상 GPU 전역 메모리 공간.
   * @f1name:   체크포인트 파일 경로 문자열.
   * checkpoint_option=2 (재개 모드)에서 호출됨.
   * 호출 체인: cuda-sim 초기화 → [load_global_mem()] */

  void store_global_mem(class memory_space *mem, char *fname, char *format);
  /* [한국어] 현재 전역 메모리 상태(mem)를 파일(fname)에 저장.
   * @mem:    저장할 GPU 전역 메모리 공간.
   * @fname:  체크포인트 파일 경로.
   * @format: 출력 형식 문자열 (텍스트/바이너리 등).
   * checkpoint_pc에 도달했을 때 호출됨.
   * 호출 체인: PTX 실행 루프 → [store_global_mem()] */

  unsigned radnom;
  /* [한국어] 원본 코드의 미사용 필드 (오타: "random"의 오타로 추정).
   * 실제 랜덤값으로 사용되는지 불분명. 코드 수정 금지 원칙으로 원본 유지. */
};

/*
 * [한국어 설명] core_t - GPU SM(Shader Multiprocessor) 추상 기반 클래스
 *
 * === 파일의 역할 ===
 * 기능 시뮬레이션과 타이밍 시뮬레이션 양쪽의 공통 기반으로 사용하는 순수 추상 클래스.
 * GPU 코어(SM)의 기능 시뮬레이션에 필요한 기본 자료구조(스레드 배열, SIMT 스택,
 * 워프 카운트, 배리어 리덕션 저장소)와 워프 실행 인터페이스를 정의한다.
 * shader_core_ctx(타이밍 모델)와 cuda-sim의 기능 실행 경로 모두 이 클래스를 상속한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 상속 계층: core_t (추상 기반) → shader_core_ctx (타이밍 모델, shader.cc)
 *                                → functional_core_t (기능 전용, cuda-sim)
 * SIMT 스택(m_simt_stack[])은 각 워프마다 하나씩 존재하며 분기·재수렴을 관리.
 * m_thread[]는 각 스레드의 레지스터 파일과 실행 컨텍스트(ptx_thread_info).
 *
 * === 타 모듈과의 연결 ===
 * - gpgpu_sim: m_gpu로 참조 — 전체 GPU 레벨 시뮬레이션 상태 접근.
 * - kernel_info_t: m_kernel로 참조 — 실행 중인 커널의 파라미터/PTX 함수 접근.
 * - simt_stack: m_simt_stack[warp_id] — 워프별 SIMT 스택(분기/재수렴).
 * - ptx_thread_info: m_thread[thread_id] — 스레드별 레지스터 파일 및 실행 컨텍스트.
 * - warp_inst_t: execute_warp_inst_t()를 통해 워프 명령어 실행.
 * - dram_callback_t: 원자적 연산 완료 콜백 경로.
 *
 * === 주요 함수/필드 요약 ===
 * core_t():             GPU·커널·워프 크기·스레드 수로 초기화. 워프 카운트 계산.
 * execute_warp_inst_t(): 워프 내 활성 스레드에 PTX 명령어를 기능 시뮬레이션으로 실행.
 * warp_exit():          워프가 완료될 때 SIMT 스택 정리 및 완료 표시.
 * m_simt_stack[]:       워프별 SIMT 스택 배열. 분기 명령어 처리.
 * m_thread[]:           스레드별 실행 컨텍스트(ptx_thread_info) 배열.
 * m_reduction_storage[]:배리어 리덕션 결과 저장 배열.
 */
// This abstract class used as a base for functional and performance and
// simulation, it has basic functional simulation data structures and
// procedures.
class core_t {
 public:
  /*
   * [한국어]
   * core_t - SM 시뮬레이션 기반 클래스 생성자
   *
   * @gpu:               gpgpu_sim 포인터 — 전체 GPU 시뮬레이터 최상위 객체.
   * @kernel:            실행할 kernel_info_t 포인터 — PTX 함수, CTA 크기, 인자 포함.
   * @warp_size:         이 SM의 워프 크기 (보통 32).
   * @threads_per_shader: 이 SM이 지원하는 최대 스레드 수 (예: 1536, 2048).
   *
   * 초기화 리스트로 m_gpu, m_kernel, m_simt_stack=NULL, m_thread=NULL, m_warp_size를 설정.
   * 이후 워프 카운트 = ceil(threads_per_shader / warp_size) 계산.
   * m_simt_stack[], m_thread[]는 이 생성자에서 NULL로 두고 파생 클래스나
   * 별도 init 함수에서 실제 배열을 할당한다.
   *
   * 호출 체인:
   *   shader_core_ctx 생성자 → [core_t(...)] (super() 역할)
   */
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
         unsigned threads_per_shader)
      : m_gpu(gpu),            // [한국어] 전체 GPU 시뮬레이터 최상위 포인터 저장
        m_kernel(kernel),      // [한국어] 현재 실행 중인 커널 정보 포인터
        m_simt_stack(NULL),    // [한국어] SIMT 스택 배열 = NULL (파생 클래스에서 init)
        m_thread(NULL),        // [한국어] 스레드 컨텍스트 배열 = NULL (파생 클래스에서 init)
        m_warp_size(warp_size) {  // [한국어] 워프 크기 (보통 32) 설정
    m_warp_count = threads_per_shader / m_warp_size;  // [한국어] SM 내 워프 수 = 총 스레드 수 / 워프 크기
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    // [한국어] 총 스레드 수가 워프 크기의 정확한 배수가 아닌 경우 처리 (천장 나눗셈)
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;  // [한국어] 나머지 스레드가 있으면 부분적으로 찬 워프를 위해 +1
    }
    assert(m_warp_count * m_warp_size > 0);  // [한국어] 유효 스레드 수 검증 — 0이면 assert 실패

    // calloc: 메모리 할당 + 0 초기화 (모든 포인터를 NULL로 초기화)
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    /* [한국어] m_thread[warp_count * warp_size]: SM 내 모든 스레드 컨텍스트 포인터 배열.
     * calloc으로 NULL 초기화. 스레드가 실제로 할당될 때까지 NULL 유지.
     * 인덱스: hw_thread_id = warp_id * warp_size + lane_id. */

    initilizeSIMTStack(m_warp_count, m_warp_size);  // [한국어] 워프 수·크기로 SIMT 스택 배열 할당 및 초기화

    // [한국어] 배리어 리덕션 결과 저장소 초기화
    // reduction_storage[CTA_ID][BAR_ID]: 배리어 리덕션 누적값 (POPC/AND/OR)
    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {  // [한국어] SM 내 모든 CTA 슬롯 순회
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {  // [한국어] 각 CTA의 모든 배리어 슬롯
        reduction_storage[i][j] = 0;  // [한국어] 리덕션 결과 0으로 초기화
      }
    }
  }

  virtual ~core_t() { free(m_thread); }
  /* [한국어] 소멸자: calloc으로 할당한 m_thread 배열 해제.
   * m_simt_stack 해제는 deleteSIMTStack()로 별도 처리 (파생 클래스에서 호출). */

  // [한국어] 순수 가상 함수들 (= 0): 파생 클래스(shader_core_ctx)에서 반드시 구현해야 함.

  virtual void warp_exit(unsigned warp_id) = 0;
  /* [한국어] 워프(warp_id)가 마지막 스레드 완료 시 호출 — SM 자원 해제 및 완료 표시.
   * 구현: shader_core_ctx::warp_exit() — CTA 배리어 해제, 워프 슬롯 반환.
   * 호출자: checkExecutionStatusAndUpdate() 내 스레드 완료 감지 시. */

  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  /* [한국어] 워프(warp_id)가 배리어(bar.sync)에서 대기 중인지 반환.
   * true이면 이 워프는 배리어가 완료될 때까지 발행 불가.
   * warp scheduler가 워프 선택 시 이 함수로 차단 여부 확인.
   * 구현: shader_core_ctx::warp_waiting_at_barrier(). */

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  /* [한국어] 스레드 tid의 실행 상태를 확인하고 완료 여부를 업데이트.
   * @inst: 방금 실행된 워프 명령어.
   * @t:    워프 내 스레드 번호 (lane index).
   * @tid:  전체 SM에서의 하드웨어 스레드 ID.
   * 스레드 종료(ret) 명령어 감지 → warp_exit() 트리거 경로.
   * 구현: shader_core_ctx::checkExecutionStatusAndUpdate(). */

  class gpgpu_sim *get_gpu() { return m_gpu; }
  /* [한국어] gpgpu_sim 포인터 반환. 전체 GPU 시뮬레이터 최상위 객체 접근. */

  /*
   * [한국어]
   * execute_warp_inst_t - 워프 명령어를 기능 시뮬레이션으로 실행
   *
   * @inst:   실행할 warp_inst_t 참조. 활성 마스크에 따라 각 스레드를 실행.
   * @warpId: 이 명령어를 실행하는 워프 ID. (unsigned)-1이면 inst에서 추출.
   *
   * 활성 마스크(m_warp_active_mask)가 1인 각 스레드에 대해
   * m_thread[tid]->ptx_exec_inst(inst)를 호출하여 PTX 기능 실행.
   * 각 스레드 완료 후 checkExecutionStatusAndUpdate() 호출.
   *
   * 호출 체인:
   *   shader_core_ctx::issue_warp() 또는 functional_core_t → [execute_warp_inst_t()]
   *     → ptx_thread_info::ptx_exec_inst() [PTX 명령어 기능 실행]
   *       → checkExecutionStatusAndUpdate() [완료 상태 확인]
   */
  void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);

  bool ptx_thread_done(unsigned hw_thread_id) const;
  /* [한국어] hw_thread_id 스레드가 PTX 기능 실행을 완료했는지 반환.
   * m_thread[hw_thread_id] == NULL 또는 스레드가 종료 상태이면 true.
   * 호출자: 워프 완료 판정 코드. */

  // [한국어] SIMT 스택 관련 함수들 — 워프별 분기/재수렴 관리.

  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  /* [한국어] 분기 명령어(inst) 실행 후 warpId의 SIMT 스택을 업데이트.
   * reconvergence_pc를 스택에 push하거나, 재수렴 점에서 pop한다.
   * 가상 함수로 파생 클래스에서 특수화 가능. 기본 구현은 simt_stack::update() 호출.
   * 호출 체인: execute_warp_inst_t() 내 분기 처리 → [updateSIMTStack()] */

  void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
  /* [한국어] warp_count개의 simt_stack 객체를 할당하여 m_simt_stack[] 초기화.
   * @warp_count: SM 내 워프 수. @warps_size: 워프 당 스레드 수.
   * 생성자에서 호출. deleteSIMTStack()으로 해제. */

  void deleteSIMTStack();
  /* [한국어] m_simt_stack[] 배열 내 모든 simt_stack 객체를 delete하고 배열 해제.
   * 파생 클래스 소멸자 또는 커널 완료 시 호출. */

  warp_inst_t getExecuteWarp(unsigned warpId);
  /* [한국어] warpId의 SIMT 스택 상단(top)에서 실행할 워프 명령어를 구성하여 반환.
   * 스택 상단의 active_mask, PC를 적용한 warp_inst_t 사본 반환.
   * 기능 시뮬레이션 fetch 경로에서 어떤 스레드가 활성인지 결정하는 데 사용. */

  void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                               unsigned *rpc) const;
  /* [한국어] warpId의 SIMT 스택 상단의 PC와 재수렴 PC(rpc)를 가져옴.
   * @pc:  현재 실행 PC. @rpc: 재수렴 PC.
   * 분기 분석 및 디버그 출력에서 사용. */

  kernel_info_t *get_kernel_info() { return m_kernel; }
  /* [한국어] 현재 실행 중인 커널 정보 반환. PTX 함수, CTA/그리드 크기, 파라미터 접근. */

  class ptx_thread_info **get_thread_info() { return m_thread; }
  /* [한국어] 스레드 컨텍스트 배열 포인터 반환. 외부에서 개별 스레드 상태 접근에 사용. */

  unsigned get_warp_size() const { return m_warp_size; }
  /* [한국어] 이 코어의 워프 크기 반환 (보통 32). */

  // [한국어] 배리어 리덕션 연산 함수들 — bar.red 명령어 처리.
  // PTX bar.red (POPC/AND/OR): 모든 스레드의 값을 배리어에서 집계.

  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] &= value;
    /* [한국어] AND 리덕션: 배리어 [ctaid][barid]에 value를 AND 누적.
     * 모든 스레드가 true를 제출해야 결과가 true. bar.red.and에서 호출. */
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] |= value;
    /* [한국어] OR 리덕션: 배리어 [ctaid][barid]에 value를 OR 누적.
     * 하나라도 true를 제출하면 결과가 true. bar.red.or에서 호출. */
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] += value;
    /* [한국어] POPC(population count) 리덕션: true 제출 스레드 수 누적 합산.
     * bar.red.popc에서 호출. 결과 = 배리어에서 value=true인 스레드 수. */
  }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return reduction_storage[ctaid][barid];
    /* [한국어] 배리어 [ctaid][barid]의 현재 리덕션 누적 결과값 반환.
     * broadcast_barrier_reduction()에서 결과를 스레드에 전파하기 전에 읽음. */
  }

 protected:  // [한국어] 보호 영역: core_t와 파생 클래스에서만 직접 접근.

  class gpgpu_sim *m_gpu;
  /* [한국어] 전체 GPU 시뮬레이터 최상위 포인터. DRAM/NoC/L2 등 전역 자원 접근.
   * 설정자: 생성자. 읽는 자: get_gpu(), 파생 클래스 전역 통계 업데이트. */

  kernel_info_t *m_kernel;
  /* [한국어] 현재 이 SM에서 실행 중인 커널 정보.
   * PTX 함수 포인터, CTA/그리드 크기, 커널 파라미터 포함.
   * 설정자: 생성자. 읽는 자: get_kernel_info(), PTX 실행 경로. */

  simt_stack **m_simt_stack;
  // pdom based reconvergence context for each warp
  /* [한국어] 워프별 SIMT 스택 포인터 배열 (크기: m_warp_count).
   * m_simt_stack[warp_id]: 해당 워프의 SIMT 스택 (포스트 도미네이터 기반 재수렴).
   * initilizeSIMTStack()에서 할당. deleteSIMTStack()에서 해제.
   * 분기 시 현재 PC/mask/reconvergence_PC를 push, 재수렴 점에서 pop. */

  class ptx_thread_info **m_thread;
  /* [한국어] SM 내 모든 스레드 실행 컨텍스트(레지스터 파일, PC, 상태) 포인터 배열.
   * 크기: m_warp_count * m_warp_size.
   * 인덱스: hw_thread_id = warp_id * warp_size + lane_id.
   * calloc으로 초기화(NULL). 스레드 생성 시 ptx_thread_info 객체가 할당됨.
   * 설정자: 파생 클래스의 스레드 init 코드. 읽는 자: execute_warp_inst_t(). */

  unsigned m_warp_size;
  /* [한국어] 워프 크기 (통상 32). SM 내 warp_size = active_mask_t 비트 수.
   * 생성자에서 설정. get_warp_size()로 접근. */

  unsigned m_warp_count;
  /* [한국어] 이 SM이 수용하는 최대 워프 수 = ceil(threads_per_shader / warp_size).
   * m_simt_stack[], m_thread[] 배열 크기의 기준. */

  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
  /* [한국어] 배리어 리덕션 결과 저장 2차원 배열.
   * reduction_storage[cta_id][bar_id]: 해당 배리어의 POPC/AND/OR 누적값.
   * and_reduction(), or_reduction(), popc_reduction()에서 업데이트.
   * get_reduction_value()로 읽어 broadcast_barrier_reduction()에서 스레드에 전파.
   * 초기화: 생성자에서 전체 0으로 설정. 배리어 완료 후 재초기화 필요. */
};

/*
 * [한국어 설명] register_set - GPU 파이프라인 스테이지 간 명령어 레지스터 세트
 *
 * === 파일의 역할 ===
 * SM 파이프라인(fetch→decode→issue→execute→writeback) 단계 사이의 파이프라인 레지스터.
 * 한 스테이지 간 통로에 여러 슬롯을 두어 복수 워프의 명령어를 동시에 보유할 수 있다.
 * 슬롯 수는 SM 설정(scheduler 수, 실행 유닛 수)에 맞춰 결정된다.
 * 서브코어(sub-core) 모델에서는 각 스케줄러가 담당하는 슬롯이 고정되어 충돌 없이 접근.
 *
 * === 전체 아키텍처에서의 위치 ===
 * shader_core_ctx가 파이프라인 레지스터를 register_set 객체들로 선언.
 * 예: m_issuing_cta[scheduler_id] → ID_OC → EX_WB 사이 슬롯.
 * move_in()으로 명령어를 현재 스테이지에서 다음 스테이지로 이동.
 * move_out_to()로 다음 스테이지가 명령어를 꺼냄.
 *
 * === 타 모듈과의 연결 ===
 * - warp_inst_t: 각 슬롯에 warp_inst_t 포인터를 저장. move_warp()로 포인터 스왑.
 * - shader_core_ctx: issue, execute, writeback 단계에서 register_set 접근.
 * - 서브코어 모델: reg_id = scheduler_id로 슬롯을 고정 매핑.
 *
 * === 주요 함수 요약 ===
 * has_free():       빈 슬롯 있는지 확인. 이전 단계가 명령어 push 전에 호출.
 * has_ready():      유효한 명령어가 있는지 확인. 다음 단계가 pop 전에 호출.
 * move_in():        이전 단계에서 명령어를 빈 슬롯으로 이동.
 * move_out_to():    유효한(준비된) 명령어를 다음 단계로 꺼냄.
 * get_ready_reg_id: 가장 오래된(uid 최소) 준비된 슬롯 인덱스 반환.
 */
// register that can hold multiple instructions.
class register_set {
 public:
  /*
   * [한국어]
   * register_set - 슬롯 수와 이름으로 파이프라인 레지스터 세트 초기화
   *
   * @num:  이 레지스터 세트의 슬롯(warp_inst_t) 수.
   *        scheduler 수(sub-core) 또는 실행 유닛 수에 맞춰 설정.
   * @name: 디버그 출력용 이름 문자열. 예: "ID_OC_SP" = decode→SP 실행 유닛 사이.
   *
   * 각 슬롯을 빈(empty=true) warp_inst_t 객체로 초기화.
   * 이때 core_config 없는 기본 생성자 사용 — 크기 의존 함수 사용 금지.
   */
  register_set(unsigned num, const char *name) {
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());  // [한국어] 슬롯마다 빈 warp_inst_t 객체 할당
    }
    m_name = name;  // [한국어] 디버그용 이름 저장
  }
  const char *get_name() { return m_name; }
  /* [한국어] 이 레지스터 세트의 이름 문자열 반환 (디버그/로그 출력용). */

  bool has_free() {
    /* [한국어] 빈 슬롯(empty=true인 warp_inst_t)이 하나 이상 있는지 확인.
     * 이전 파이프라인 스테이지가 명령어를 push하기 전에 공간 여부 확인.
     * 모든 슬롯이 차있으면 false → 발행 스톨. */
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) return true;  // [한국어] 빈 슬롯 발견 시 즉시 true 반환
    }
    return false;  // [한국어] 모든 슬롯이 사용 중
  }

  bool has_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on sched id)
    /* [한국어] 서브코어(sub-core) 모델: 스케줄러 reg_id에 해당하는 고정 슬롯이 비어있는지.
     * 서브코어 모델에서는 스케줄러 i가 항상 regs[i] 슬롯을 독점 사용하여 경쟁을 없앤다.
     * sub_core_model=false이면 일반 has_free()로 위임. */
    if (!sub_core_model) return has_free();  // [한국어] 서브코어 모델 아님 → 일반 방식

    assert(reg_id < regs.size());        // [한국어] reg_id 유효 범위 검사
    return regs[reg_id]->empty();        // [한국어] 해당 슬롯이 비어있으면 true
  }

  bool has_ready() {
    /* [한국어] 유효 명령어(empty=false)가 있는 슬롯이 하나 이상인지 확인.
     * 다음 파이프라인 스테이지가 pop 전에 꺼낼 수 있는 명령어가 있는지 확인. */
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) return true;  // [한국어] 유효 명령어 발견 시 즉시 true
    }
    return false;  // [한국어] 모든 슬롯이 비어있음
  }

  bool has_ready(bool sub_core_model, unsigned reg_id) {
    /* [한국어] 서브코어 모델: 스케줄러 reg_id의 전담 슬롯에 유효 명령어가 있는지. */
    if (!sub_core_model) return has_ready();  // [한국어] 서브코어 아님 → 일반 방식
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());  // [한국어] 해당 슬롯에 명령어가 있으면 true
  }

  unsigned get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    /* [한국어] 유효 명령어 슬롯 중 가장 오래된(uid 최소) 슬롯의 인덱스 반환.
     * uid가 작을수록 먼저 발행된(오래된) 명령어. FIFO 우선 순위로 처리.
     * has_ready() = true인 상태에서만 호출해야 한다 (assert로 보장). */
    assert(has_ready());  // [한국어] 유효 명령어 없으면 호출 금지
    warp_inst_t **ready = NULL;
    unsigned reg_id = 0;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // [한국어] 현재 선택된 것이 더 오래됨(uid 작음) → 유지
        } else {
          ready = &regs[i];  // [한국어] 더 오래된(또는 첫 번째 유효) 슬롯으로 갱신
          reg_id = i;
        }
      }
    }
    return reg_id;  // [한국어] 가장 오래된 유효 명령어가 있는 슬롯 인덱스
  }

  unsigned get_schd_id(unsigned reg_id) {
    /* [한국어] reg_id 슬롯에 있는 명령어를 발행한 스케줄러 ID 반환.
     * 서브코어 모델 통계 수집에서 어느 스케줄러 명령어인지 식별. */
    assert(not regs[reg_id]->empty());  // [한국어] 빈 슬롯의 스케줄러 ID 조회 금지
    return regs[reg_id]->get_schd_id();
  }

  void move_in(warp_inst_t *&src) {
    /* [한국어] src 명령어를 이 레지스터 세트의 임의 빈 슬롯으로 이동.
     * move_warp()로 포인터 스왑 — src는 빈 슬롯 포인터로 교체.
     * 호출자: 이전 파이프라인 스테이지가 명령어를 다음 스테이지로 push. */
    warp_inst_t **free = get_free();  // [한국어] 비어있는 슬롯 탐색
    move_warp(*free, src);            // [한국어] 포인터 스왑: src → *free, *free → src
  }
  // void copy_in( warp_inst_t* src ){  // [한국어] 복사 방식 (사용 안 함, 포인터 스왑으로 대체)
  //   src->copy_contents_to(*get_free());
  //}

  void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    /* [한국어] 서브코어 모델: reg_id 슬롯에 src 명령어 이동. 일반 모델은 임의 빈 슬롯. */
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();                          // [한국어] 일반 모델: 임의 빈 슬롯 탐색
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);    // [한국어] 서브코어: reg_id 전담 슬롯
    }
    move_warp(*free, src);  // [한국어] 포인터 스왑
  }

  void move_out_to(warp_inst_t *&dest) {
    /* [한국어] 가장 오래된 유효 명령어를 dest로 이동 (다음 스테이지로 pop).
     * move_warp()로 포인터 스왑 — 슬롯은 다시 빈 상태 포인터를 갖게 됨.
     * 호출자: 다음 파이프라인 스테이지가 명령어를 꺼낼 때. */
    warp_inst_t **ready = get_ready();  // [한국어] 가장 오래된 유효 슬롯 탐색
    move_warp(dest, *ready);            // [한국어] 포인터 스왑: *ready → dest
  }

  void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    /* [한국어] 서브코어 모델: reg_id 슬롯의 명령어를 dest로 이동. */
    if (!sub_core_model) {
      return move_out_to(dest);  // [한국어] 일반 모델 위임
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);  // [한국어] 지정 슬롯 탐색
    assert(ready != NULL);  // [한국어] 해당 슬롯에 유효 명령어가 있어야 함
    move_warp(dest, *ready);
  }

  warp_inst_t **get_ready() {
    /* [한국어] 모든 슬롯 중 uid가 가장 작은(가장 오래된) 유효 명령어 슬롯 포인터 반환.
     * 유효 슬롯이 없으면 NULL 반환. move_out_to()에서 내부적으로 사용. */
    warp_inst_t **ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // [한국어] 현재 ready가 더 오래됨(uid 작음) → 교체 불필요
        } else {
          ready = &regs[i];  // [한국어] 더 오래된 슬롯으로 갱신
        }
      }
    }
    return ready;  // [한국어] 가장 오래된 유효 슬롯 포인터 (없으면 NULL)
  }

  warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {
    /* [한국어] 서브코어 모델: reg_id 전담 슬롯에 유효 명령어가 있으면 해당 슬롯 포인터 반환. */
    if (!sub_core_model) return get_ready();  // [한국어] 일반 모델 위임
    warp_inst_t **ready = NULL;
    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty()) ready = &regs[reg_id];  // [한국어] 유효하면 포인터 저장
    return ready;
  }

  void print(FILE *fp) const {
    /* [한국어] 레지스터 세트의 이름, 메모리 주소, 각 슬롯의 명령어 정보를 파일에 출력.
     * 파이프라인 상태 덤프 및 디버그 출력에서 사용. */
    fprintf(fp, "%s : @%p\n", m_name, this);  // [한국어] 세트 이름과 객체 주소 출력
    for (unsigned i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);  // [한국어] 각 슬롯의 warp_inst_t::print() 호출
      fprintf(fp, "\n");
    }
  }

  warp_inst_t **get_free() {
    /* [한국어] 빈(empty=true) 슬롯을 탐색하여 포인터 반환.
     * 빈 슬롯이 없으면 assert(0) — 프로그램 중단 (호출 전에 has_free() 확인 필수).
     * move_in()에서 내부적으로 사용. */
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) return &regs[i];  // [한국어] 첫 번째 빈 슬롯 반환
    }
    assert(0 && "No free registers found");  // [한국어] 빈 슬롯 없음 — 프로그램 중단
    return NULL;
  }

  warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on sched id)
    /* [한국어] 서브코어 모델: reg_id 전담 슬롯이 비어있으면 해당 슬롯 포인터 반환.
     * 비어있지 않으면 assert(0) — 발행 전에 has_free(sub_core_model, reg_id) 확인 필수. */
    if (!sub_core_model) return get_free();  // [한국어] 일반 모델 위임

    assert(reg_id < regs.size());  // [한국어] 유효 범위 검증
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];  // [한국어] 비어있음 → 해당 슬롯 포인터 반환
    }
    assert(0 && "No free register found");  // [한국어] 해당 슬롯이 차있음 — 프로그램 중단
    return NULL;
  }

  unsigned get_size() { return regs.size(); }
  /* [한국어] 이 레지스터 세트의 총 슬롯 수 반환. 생성 시 num 인자와 같음. */

 private:
  std::vector<warp_inst_t *> regs;
  /* [한국어] warp_inst_t 포인터 슬롯 벡터. 크기: 생성자의 num 인자.
   * 각 원소는 하나의 warp_inst_t 객체 포인터. 빈 슬롯은 empty()=true인 객체를 가리킴.
   * move_warp()가 포인터를 스왑하므로 객체 자체는 이동하지 않고 포인터만 교환. */

  const char *m_name;
  /* [한국어] 디버그/로그 출력용 이름 문자열 포인터.
   * 예: "ID_OC_SP" (decode→SP 파이프), "EX_WB" (execute→writeback).
   * 리터럴 문자열 포인터라 별도 해제 불필요. */
};

#endif  // #ifdef __cplusplus  // C++ 전용 코드 영역 끝

#endif  // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED  // 인클루드 가드 끝
