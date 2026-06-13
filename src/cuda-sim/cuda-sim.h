/*
 * [한국어 설명] CUDA PTX 기능 시뮬레이션 공개 인터페이스 (cuda-sim.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 PTX 기능 시뮬레이션(functional simulation) 레이어의
 * 공개 인터페이스를 정의하는 헤더 파일이다. 타이밍 모델(gpgpu-sim/)이 실제
 * GPU 마이크로아키텍처를 사이클 단위로 모델링하는 것과 달리, 이 모듈은 CUDA 커널이
 * 어떤 결과를 내는지를 시간 개념 없이 순수하게 명령어 의미론(semantics) 수준에서
 * 시뮬레이션한다. 기능 시뮬레이션은 정확성 검증, 체크포인트, OpenCL 지원에 활용된다.
 * cuda-sim.cc가 이 헤더를 구현하며, 나머지 시뮬레이터 모듈이 이 헤더를 include하여
 * PTX 실행 기능을 호출한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 실행 흐름 내 기능 시뮬레이션 계층에 위치한다.
 *   gpgpusim_entrypoint.cc (진입점)
 *     → cuda_sim::gpgpu_cuda_ptx_sim_main_func()  (이 파일의 핵심 진입 함수)
 *         → functionalCoreSim::execute()           (CTA 단위 실행 루프)
 *             → functionalCoreSim::executeWarp()   (워프 단위 실행)
 *                 → ptx_thread_info::ptx_exec_inst() (단일 PTX 명령어 실행)
 * 타이밍 모델(shader.cc)도 ptx_exec_inst()를 호출하여 각 스레드의 기능적 결과를 얻는다.
 * 실행 컨텍스트: 호스트 유저스페이스(CPU 스레드), 단일 스레드로 순차 실행.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: ptx_sim.h (ptx_thread_info, ptx_instruction 등 핵심 스레드/명령어 자료구조),
 *         abstract_hardware_model.h (core_t, warp_inst_t, kernel_info_t 등 HW 추상),
 *         gpgpu-sim/shader.h (core_t 상속 기반)
 * - 피의존: shader.cc (ptx_exec_inst 호출), gpgpusim_entrypoint.cc (커널 실행 진입),
 *            cuda-sim.cc (이 헤더의 구현 파일)
 * - 데이터 흐름: kernel_info_t(커널 메타+스레드 리스트) → ptx_sim_init_thread()로
 *   ptx_thread_info 생성 → ptx_exec_inst()로 PTX 명령어 실행 → 결과가 memory_space에 반영
 * - 공유 자료구조: cuda_sim 객체(gpgpu_context::func_sim)가 전역 상태를 보관하며
 *   모든 모듈이 gpgpu_ctx->func_sim->XXX 형태로 접근한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - functionalCoreSim : core_t 상속, 타이밍 없이 CTA 전체를 기능적으로 실행하는 코어 시뮬레이터
 * - cuda_sim          : PTX 시뮬레이션 전역 상태(레지스터 latency, 통계, 디버그 설정 등) 관리 클래스
 * - ptx_sim_init_thread() : 새 CTA에 속한 스레드(ptx_thread_info)를 할당·초기화
 * - gpgpu_cuda_ptx_sim_main_func() : 기능 시뮬레이션 전용 커널 실행 최상위 함수
 * - get_converge_point() : SIMT 스택에서 현재 PC의 재수렴(post-dominator) 주소를 반환
 */

// Copyright (c) 2009-2011, Tor M. Aamodt
// 저작권 표시: 이 코드는 2009~2011년에 Tor M. Aamodt라는 분이 만들었습니다.
// The University of British Columbia
// 캐나다 브리티시컬럼비아 대학교에서 만든 코드입니다.
// All rights reserved.
// 모든 권리가 보호됩니다 (저작권법에 의해 보호받는다는 뜻).
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 이 코드를 다시 배포하거나 사용할 때는 아래 조건을 지켜야 합니다:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 소스 코드를 배포할 때는 위의 저작권 표시와 이 조건들을 반드시 포함해야 합니다.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
// 컴파일된 형태로 배포할 때도 저작권 표시를 문서에 포함해야 하며,
// 대학교 이름을 허락 없이 홍보에 사용하면 안 됩니다.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
// 이 소프트웨어는 "있는 그대로" 제공되며, 사용으로 인한 어떤 손해에 대해서도
// 만든 사람이 책임지지 않는다는 뜻입니다. (법적 면책 조항)

/*
 * 인클루드 가드(Include Guard): CUDASIM_H_INCLUDED 매크로가 미정의인 경우에만 이 헤더를
 * 처리하도록 보호한다. 동일 헤더가 여러 번 include되어 발생하는 중복 정의 오류를 방지한다.
 * 설정자: 이 파일을 처음 include하는 번역 단위(translation unit).
 * 읽는 자: 이후 동일 헤더를 include하려는 모든 번역 단위.
 */
#ifndef CUDASIM_H_INCLUDED  /* [한국어] 이 매크로가 아직 정의되지 않았을 때만 헤더 본문을 처리 */
#define CUDASIM_H_INCLUDED  /* [한국어] 매크로를 정의하여 이후 중복 include를 방지 */

/* [한국어] C 표준 라이브러리 포함: malloc/free(동적 메모리 관리), abort(), NULL 등
 * 기능 시뮬레이션 도중 스레드 자료구조를 힙에 동적 할당하는 데 사용된다. */
#include <stdlib.h>
/* [한국어] std::map 포함: 정렬된 키-값 쌍 연관 컨테이너.
 * g_global_name_lookup, g_const_name_lookup, g_pc_to_finfo 등 다수의 룩업 테이블에 사용. */
#include <map>
/* [한국어] std::string 포함: 가변 길이 문자열 클래스.
 * 커널 함수명, 심볼 이름, 디버그 출력 문자열 등을 저장·비교하는 데 사용. */
#include <string>
/* [한국어] std::vector 포함: 동적 배열 컨테이너.
 * 커널 인자 목록(gpgpu_ptx_sim_arg_list_t) 등에 사용. */
#include <vector>
/* [한국어] GPU 하드웨어 추상 모델 헤더 포함.
 * core_t(SM 추상), warp_inst_t(워프 명령어), kernel_info_t(커널 메타),
 * gpgpu_t(GPU 전체), gpgpu_sim 등 기능·타이밍 공통 자료구조를 제공.
 * 기능 시뮬레이션과 타이밍 시뮬레이션이 이 추상 위에서 공통 인터페이스를 공유한다. */
#include "../abstract_hardware_model.h"
/* [한국어] 셰이더 코어(SM) 타이밍 모델 헤더 포함.
 * functionalCoreSim이 core_t를 상속받기 위해 필요하다.
 * core_t는 shader.h → abstract_hardware_model.h 체인을 통해 선언되며,
 * SIMT 스택(simt_stack), 스레드 배열(m_thread[]) 등을 포함한다. */
#include "../gpgpu-sim/shader.h"
/* [한국어] PTX 시뮬레이션 핵심 자료구조 헤더 포함.
 * ptx_thread_info(개별 CUDA 스레드 실행 상태), ptx_instruction(PTX 명령어 객체),
 * ptx_cta_info(CTA 메타), ptx_reg_t(레지스터 값 유니온) 등을 제공한다.
 * PTX(Parallel Thread eXecution): CUDA 코드를 NVIDIA 컴파일러가 변환하는 가상 ISA.
 * 실제 GPU에서 SASS로 추가 컴파일되지만, GPGPU-Sim은 PTX 수준에서 시뮬레이션한다. */
#include "ptx_sim.h"

/* [한국어] 전방 선언(forward declaration): 포인터·참조 매개변수 타입으로만 사용되므로
 * 완전한 정의 없이 이름만 선언하여 순환 포함(circular include)과 컴파일 시간을 줄인다. */
class gpgpu_context;
/* [한국어] gpgpu_context: 시뮬레이터 전역 싱글턴 컨텍스트.
 * cuda_sim(func_sim), ptx_parser, device_runtime, stream_manager 등을 보유한다.
 * 모든 모듈이 gpgpu_ctx->XXX 형태로 이 컨텍스트를 통해 전역 상태에 접근한다. */

class memory_space;
/* [한국어] memory_space: 시뮬레이션 메모리 공간 추상 인터페이스.
 * 전역(global), 공유(shared), 로컬(local), 상수(const) 등 여러 GPU 메모리 영역을
 * read()/write() API로 추상화한다. 실제 구현은 memory_space_impl<N> 템플릿이다. */

class function_info;
/* [한국어] function_info: PTX 커널 함수(또는 디바이스 함수)의 완전한 메타데이터.
 * 명령어 배열(m_instr_mem), 심볼 테이블(m_symtab), 파라미터 정보,
 * 재수렴(pdom) 분석 결과 등을 보관한다.
 * 커널 실행 시 kernel_info_t::entry()를 통해 얻는다. */

class symbol_table;
/* [한국어] symbol_table: PTX 함수 범위 내 모든 변수·레지스터·레이블의 이름→주소 매핑.
 * function_info가 하나의 symbol_table을 소유하며, ptx_thread_info가 레지스터 조회에 사용. */

/*
 * [한국어] g_gpgpusim_version_string - GPGPU-Sim 버전 문자열 전역 변수 (extern 선언)
 *
 * 실제 정의: src/version 파일에서 #include되어 cuda-sim.cc 번역 단위에 존재한다.
 * 설정자: GPGPU-Sim 빌드 시스템이 version 파일을 통해 초기화.
 * 읽는 자: print_splash()가 시뮬레이터 시작 배너 출력 시 사용.
 * 값 범위: NULL이 아닌 유효한 C 문자열 포인터 (예: "GPGPU-Sim version 4.0.0...").
 * 동기화: 읽기 전용 — 초기화 이후 변경되지 않으므로 동기화 불필요.
 */
extern const char *g_gpgpusim_version_string;

/*
 * [한국어] g_debug_execution - PTX 시뮬레이션 디버그 레벨 전역 변수 (extern 선언)
 *
 * 실제 정의: cuda-sim.cc 파일의 전역 변수 (int g_debug_execution = 0;).
 * 설정자: read_sim_environment_variables()가 환경변수 PTX_SIM_DEBUG 값으로 설정.
 * 읽는 자: ptx_exec_inst(), ptx_sim_init_thread() 등 PTX 실행 경로 전반.
 * 값 범위: 0 = 디버그 출력 없음; 1~5 = 점진적으로 상세한 출력;
 *          6 이상 = 레지스터 덤프; 40 이상 = PTX 파서 디버그까지 활성.
 *          -1 = 특별 CTA 할당 추적 모드.
 * 동기화: 단일 호스트 스레드에서 시뮬레이션 전에 설정되므로 별도 동기화 불필요.
 */
extern int g_debug_execution;

/*
 * [한국어]
 * print_splash - 시뮬레이터 시작 배너를 stdout에 출력하는 함수 (extern 선언)
 *
 * @return: void
 *
 * 실제 구현: cuda-sim.cc의 print_splash() 함수.
 * 시뮬레이터가 처음 커널을 실행할 때 gpgpusim_entrypoint.cc에서 호출되며,
 * static 변수를 이용해 배너가 단 한 번만 출력되도록 보장한다.
 * g_gpgpusim_version_string과 g_gpgpusim_build_string을 출력한다.
 *
 * 호출 체인: gpgpusim_entrypoint → [print_splash]
 */
extern void print_splash();

/*
 * [한국어]
 * ptxinfo_opencl_addinfo - PTX 정보 파서 결과를 OpenCL 커널 함수 정보에 연결하는 함수
 *
 * @kernels: 커널 이름(std::string) → function_info* 매핑. ptx_loader가 구축한 맵.
 *           참조 전달이므로 내부에서 각 function_info의 커널 정보(smem, regs 등)를 갱신.
 * @return: void
 *
 * ptxinfo 파서(ptx.tab.c 기반)가 .ptx 파일을 파싱한 결과(g_ptxinfo 정적 구조체)를
 * kernels 맵 내 해당 커널 function_info 객체에 set_kernel_info()로 주입한다.
 * OpenCL 경로에서 커널 초기화 완료 단계에 호출된다.
 * 실제 구현: cuda-sim.cc.
 *
 * 호출 체인: OpenCL 커널 로드 → [ptxinfo_opencl_addinfo] → function_info::set_kernel_info()
 */
extern void ptxinfo_opencl_addinfo(
    std::map<std::string, function_info *> &kernels);

/*
 * [한국어]
 * ptx_sim_init_thread - 새 PTX 스레드를 생성·초기화하거나 기존 스레드를 할당 해제 후 갱신
 *
 * @kernel            : 실행 중인 커널 정보 참조. kernel.active_threads() 리스트에서 스레드를 꺼낸다.
 * @thread_info       : 출력 파라미터. *thread_info가 NULL이 아니면 기존 완료 스레드를 먼저 삭제하고,
 *                      이후 새로 할당된 ptx_thread_info 포인터를 여기에 저장한다.
 * @sid               : 셰이더(SM) 코어 ID (0-based). shared/local 메모리 룩업 인덱스 계산에 사용.
 *                      SM(Streaming Multiprocessor): GPU 내 독립 실행 파이프라인 단위.
 * @tid               : 코어 내부 소프트웨어 스레드 ID (CTA 내 0-based 오프셋).
 *                      sm_idx = hw_cta_id * num_shaders + sid 계산에 사용.
 * @threads_left      : 현재 CTA에서 아직 초기화하지 않은 스레드 수. active_threads 크기 검증에 사용.
 * @num_threads       : 코어에서 실행 중인 총 스레드 수 (max_cta_per_sm * threads_per_cta).
 * @core              : 이 스레드가 실행될 core_t 객체. ptx_thread_info::init()에 전달된다.
 * @hw_cta_id         : 하드웨어 CTA 슬롯 ID (SM 내 0-based). shared 메모리 인덱스 구성에 사용.
 *                      CTA(Cooperative Thread Array): CUDA 블록에 해당하는 스레드 협력 단위.
 * @hw_warp_id        : 하드웨어 워프 ID. ptx_warp_info 룩업 키로 사용된다.
 *                      워프(warp): SIMT 실행 기본 단위, 보통 32개 스레드.
 * @gpu               : gpgpu_t GPU 객체 포인터. gpgpu_ctx 접근 및 기능 시뮬레이션 메모리 제공.
 * @functionalSimulationMode : true이면 기능 시뮬레이션 전용 모드(타이밍 정보 불필요),
 *                             false이면 타이밍+기능 혼합 모드. ptx_thread_info::init()에 전달.
 * @return: 1 = 스레드가 성공적으로 할당됨, 0 = 더 이상 실행할 CTA/스레드 없음.
 *
 * 이 함수는 CTA 단위 스레드 할당의 핵심이다. 호출될 때마다 하나의 스레드를 할당하며,
 * 새 CTA가 필요하면 해당 CTA에 속한 모든 스레드를 한꺼번에 생성하여 active_threads 리스트에 넣는다.
 * shared/local 메모리 객체는 SM 인덱스별로 static 맵에 캐싱되어 재사용된다.
 * 에러 경로: kernel.no_more_ctas_to_run() 또는 threads_left < threads_per_cta이면 0 반환.
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드.
 *
 * 호출 체인:
 *   functionalCoreSim::initializeCTA()  → [ptx_sim_init_thread] → ptx_thread_info::init()
 *   shader_core_ctx::init_warps()       → [ptx_sim_init_thread] → ptx_thread_info::init()
 */
unsigned ptx_sim_init_thread(
    kernel_info_t &kernel,
    /* [한국어] 실행할 커널 메타+스레드 리스트를 담은 kernel_info_t 참조.
     * kernel.active_threads()로 미할당 스레드를 꺼내고,
     * kernel.more_threads_in_cta()로 새 CTA 스레드 생성 여부를 판단한다. */
    class ptx_thread_info **thread_info,
    /* [한국어] 이중 포인터: 함수 내부에서 *thread_info에 새 ptx_thread_info 객체를 저장한다.
     * *thread_info != NULL이면 기존 완료 스레드를 delete 후 NULL로 설정한다. */
    int sid,
    /* [한국어] 셰이더 코어(SM) ID. sm_idx 계산과 shared/local 메모리 룩업 키에 사용.
     * 실제 NVIDIA GPU에서는 각 SM이 독립적인 shared memory를 보유한다. */
    unsigned tid,
    /* [한국어] 코어 내 스레드 오프셋 (CTA 내 0-based). new_tid = tid + cta_내_인덱스 로 계산. */
    unsigned threads_left,
    /* [한국어] 아직 할당되지 않은 스레드 수. active_threads.size() <= threads_left 단정문에 사용. */
    unsigned num_threads,
    /* [한국어] SM에서 동시에 실행하는 총 스레드 수. max_cta_per_sm 계산에 사용. */
    class core_t *core,
    /* [한국어] 이 스레드가 귀속될 코어 추상 객체. ptx_thread_info가 실행 중 코어 참조를 필요로 한다. */
    unsigned hw_cta_id,
    /* [한국어] SM 내 하드웨어 CTA 슬롯 번호. sm_idx = hw_cta_id * num_shaders + sid. */
    unsigned hw_warp_id,
    /* [한국어] 워프 ID. ptx_warp_info 캐시 맵의 키. 워프별 배리어/동기화 상태 추적에 사용. */
    gpgpu_t *gpu,
    /* [한국어] GPU 전체 객체. gpgpu_ctx 포인터 접근 및 global memory 객체 제공에 사용. */
    bool functionalSimulationMode = false
    /* [한국어] 기능 시뮬레이션 전용 모드 플래그.
     * true: gpgpu_cuda_ptx_sim_main_func()에서 호출 — 타이밍 통계 수집 불필요.
     * false: 타이밍 모델(shader_core_ctx)에서 호출 — 타이밍+기능 혼합. */
);

/*
 * [한국어]
 * ptx_sim_kernel_info - 커널 함수의 PTX 시뮬레이션 리소스 정보를 반환
 *
 * @kernel : 정보를 가져올 커널 함수 객체 포인터 (const: 내부 상태 변경 금지).
 * @return : gpgpu_ptx_sim_info 구조체 포인터 (const: 호출자가 변경 불가).
 *           regs(사용 레지스터 수), lmem(로컬 메모리), smem(공유 메모리),
 *           cmem(상수 메모리), gmem(전역 메모리) 등을 포함한다.
 *
 * 타이밍 모델이 CTA 배치(placement) 결정 시 이 정보를 활용한다.
 * 예: smem 사용량이 SM의 최대 공유 메모리를 초과하면 해당 SM에 CTA를 배치하지 않는다.
 * 실제 구현: cuda-sim.cc — kernel->get_kernel_info()를 그대로 반환하는 간단한 래퍼.
 *
 * 호출 체인:
 *   shader_core_ctx::can_issue_1block() → [ptx_sim_kernel_info] → kernel.get_kernel_info()
 *   gpgpu_cuda_ptx_sim_main_func()      → [ptx_sim_kernel_info] → kernel.get_kernel_info()
 */
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const class function_info *kernel);

/*!
 * This class functionally executes a kernel. It uses the basic data structures
 * and procedures in core_t
 */
/*
 * [한국어]
 * functionalCoreSim - 타이밍 없이 CUDA 커널을 기능적으로 실행하는 코어 시뮬레이터 클래스
 *
 * core_t를 public 상속받아 SM(Streaming Multiprocessor)의 공통 자료구조
 * (SIMT 스택 배열 m_simt_stack[], 스레드 배열 m_thread[], 워프 카운트 m_warp_count 등)를
 * 재사용한다. 타이밍 모델(shader_core_ctx)과 달리 클럭 사이클을 전혀 추적하지 않고,
 * CTA 내 모든 워프를 완료될 때까지 순차적으로 반복 실행한다.
 *
 * 주요 사용처:
 *   1) gpgpu_cuda_ptx_sim_main_func(): 기능 전용 시뮬레이션(PTX_SIM_MODE_FUNC=1)
 *   2) 체크포인트 저장 모드(checkpoint_option=1)에서 CTA별 레지스터/메모리 덤프
 *
 * 동기화 특성: 기능 시뮬레이션은 단일 호스트 스레드가 순차 실행하므로 락 불필요.
 *   단, BAR_OP(배리어 명령어)가 발생하면 m_warpAtBarrier[]로 모든 워프를 대기시키고,
 *   전체 워프가 배리어에 도달한 후에야 일괄 해제한다(__syncthreads() 에뮬레이션).
 */
class functionalCoreSim : public core_t {
 public:

  /*
   * [한국어]
   * functionalCoreSim 생성자 - 기능 시뮬레이션 코어를 초기화
   *
   * @kernel     : 실행할 커널 정보 포인터. 워프 수 계산에 threads_per_cta()를 사용.
   * @g          : gpgpu_sim 객체 포인터. core_t 생성자에 전달되어 GPU 설정(warp_size 등) 접근.
   * @warp_size  : 한 워프의 스레드 수 (NVIDIA GPU = 32). m_warp_size로 저장됨.
   *
   * 초기화 순서:
   *   1) core_t(g, kernel, warp_size, kernel->threads_per_cta()) 호출 →
   *      m_simt_stack[], m_thread[], m_warp_count 등 SM 공통 자원 초기화
   *   2) m_warpAtBarrier 배열 동적 할당 (크기: m_warp_count)
   *   3) m_liveThreadCount 배열 동적 할당 (크기: m_warp_count)
   *
   * 호출 체인:
   *   gpgpu_cuda_ptx_sim_main_func() → new functionalCoreSim(&kernel, g, warp_size)
   */
  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)
      : core_t(g, kernel, warp_size, kernel->threads_per_cta()) {
    /* [한국어] m_warp_count개 워프의 배리어 대기 여부를 저장할 bool 배열 동적 할당.
     * new bool[n]은 힙에 n개 bool 배열을 생성한다.
     * initializeCTA()에서 모두 false로 초기화, executeWarp()에서 BAR_OP 시 true로 설정. */
    m_warpAtBarrier = new bool[m_warp_count];
    /* [한국어] m_warp_count개 워프별 활성(live) 스레드 수를 저장할 배열 동적 할당.
     * createWarp()에서 초기화, checkExecutionStatusAndUpdate()에서 스레드 완료 시 감소.
     * 이 값이 0이 되면 해당 워프는 더 이상 실행할 스레드가 없음을 의미한다. */
    m_liveThreadCount = new unsigned[m_warp_count];
  }

  /*
   * [한국어]
   * ~functionalCoreSim 소멸자 - 동적 할당 메모리를 해제하여 메모리 누수를 방지
   *
   * virtual: 다형성을 위해 가상 소멸자로 선언. 상속 계층에서 올바른 소멸자가 호출된다.
   * 해제 순서:
   *   1) warp_exit(0): 모든 워프의 ptx_thread_info 객체 delete 및 CTA 등록 해제
   *   2) delete[] m_liveThreadCount: 활성 스레드 카운트 배열 해제
   *   3) delete[] m_warpAtBarrier: 배리어 대기 배열 해제
   * delete[]를 사용하는 이유: new[]로 배열 할당했으므로 반드시 delete[]로 해제해야 한다.
   *   delete (비배열)를 쓰면 배열의 첫 번째 원소만 소멸자가 호출되어 메모리 누수 발생.
   */
  virtual ~functionalCoreSim() {
    warp_exit(0);              /* [한국어] 모든 스레드 객체를 순회하며 delete 및 CTA 상태 등록 해제 */
    delete[] m_liveThreadCount; /* [한국어] 워프별 활성 스레드 수 배열 해제 (new unsigned[]에 대응) */
    delete[] m_warpAtBarrier;   /* [한국어] 워프별 배리어 대기 상태 배열 해제 (new bool[]에 대응) */
  }

  //! executes all warps till completion
  /*
   * [한국어]
   * execute - CTA 하나를 초기화하고 모든 워프가 완료될 때까지 반복 실행
   *
   * @inst_count : 체크포인트 명령어 한도. 0이면 무제한. 0보다 크면 count > inst_count
   *              조건이 충족될 때 조기 종료하여 중간 상태를 저장한다.
   * @ctaid_cp   : 실행할 CTA의 체크포인트 ID. 체크포인트 파일명에 포함됨.
   *
   * 실행 흐름:
   *   1) initializeCTA(ctaid_cp): 스레드 생성, shared/local 메모리 할당, SIMT 스택 초기화
   *   2) 루프: 모든 워프에 대해 executeWarp() 호출
   *   3) someOneLive == false이면 루프 종료
   *   4) allAtBarrier == true이면 배리어를 일괄 해제(m_warpAtBarrier[] = false)
   *   5) checkpoint_option == 1이면 레지스터/공유메모리/SIMT 스택을 파일로 덤프
   *
   * 사이클 모델링 없음: 실제 GPU와 달리 명령어 실행에 걸리는 클럭을 추적하지 않는다.
   *
   * 호출 체인:
   *   gpgpu_cuda_ptx_sim_main_func() → functionalCoreSim cta → [execute]
   *     → initializeCTA() → executeWarp() × warp_count
   */
  void execute(int inst_count, unsigned ctaid_cp);

  /*
   * [한국어]
   * warp_exit - 해당 코어에 속한 모든 스레드 객체를 소멸시키는 정리 함수
   *
   * @warp_id : 현재 구현에서는 실질적으로 사용되지 않음.
   *            소멸자에서 warp_exit(0)으로 호출하여 전체 m_thread[] 배열을 순회.
   *
   * m_thread[i]가 NULL이 아닌 모든 슬롯에 대해:
   *   - m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]): CTA 레퍼런스 카운트 감소
   *   - delete m_thread[i]: ptx_thread_info 힙 메모리 해제
   *
   * 호출 체인:
   *   ~functionalCoreSim() → [warp_exit(0)]
   */
  virtual void warp_exit(unsigned warp_id);

  /*
   * [한국어]
   * warp_waiting_at_barrier - 특정 워프가 배리어 대기 상태인지 (또는 완료 상태인지) 확인
   *
   * @warp_id : 확인할 워프 번호.
   * @return  : true이면 이 워프는 현재 실행 진행이 막혀있음(배리어 대기 or 모든 스레드 완료).
   *            false이면 실행 가능한 스레드가 남아있음.
   *
   * 판정 조건:
   *   - m_warpAtBarrier[warp_id] == true: BAR_OP 실행 후 배리어 대기 중
   *   - m_liveThreadCount[warp_id] == 0: 모든 스레드가 EXIT 명령어 실행 완료
   *
   * execute() 루프에서 allAtBarrier 플래그 계산에 사용된다.
   * core_t의 순수 가상 함수를 오버라이드한다.
   */
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const {
    /* [한국어] 배리어 대기 중이거나 활성 스레드가 없으면 true 반환.
     * ||: 논리 OR — 둘 중 하나라도 true이면 전체 표현식이 true.
     * !(m_liveThreadCount[warp_id] > 0): liveThreadCount가 0이면 !(false) = true. */
    return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id] > 0));
  }

 private:

  /*
   * [한국어]
   * executeWarp - 단일 워프의 한 스텝을 실행하고 배리어/활성 상태를 갱신
   *
   * @i            : 실행할 워프 번호.
   * @allAtBarrier : 참조 전달. 이 워프가 배리어 아님이 확인되면 false로 설정.
   *                 모든 워프가 배리어이면 execute()가 배리어를 일괄 해제.
   * @someOneLive  : 참조 전달. 이 워프에 활성 스레드가 있으면 true로 설정.
   *                 execute() 루프 종료 조건 결정에 사용.
   *
   * 실행 흐름:
   *   1) m_warpAtBarrier[i] == false && m_liveThreadCount[i] != 0인 경우에만 진행
   *   2) getExecuteWarp(i): SIMT 스택에서 다음 실행할 warp_inst_t를 결정
   *   3) execute_warp_inst_t(inst, i): 워프 내 각 레인(스레드)에 ptx_exec_inst() 호출
   *   4) inst.isatomic()이면 do_atomic(true)로 원자 연산 커밋
   *   5) inst.op == BARRIER_OP이면 m_warpAtBarrier[i] = true
   *   6) updateSIMTStack(i, &inst): SIMT 스택 top을 inst 결과에 따라 갱신
   *
   * 호출 체인:
   *   execute() → [executeWarp] → execute_warp_inst_t() → ptx_exec_inst()
   */
  void executeWarp(unsigned, bool &, bool &);

  // initializes threads in the CTA block which we are executing
  /*
   * [한국어]
   * initializeCTA - 새 CTA의 모든 스레드와 워프를 초기화
   *
   * @ctaid_cp : 체크포인트 CTA ID. cp_cta_resume == 1이면 이 ID의 체크포인트 파일에서
   *             레지스터 및 SIMT 스택 상태를 복원(resume)한다.
   *
   * 실행 흐름:
   *   1) m_warpAtBarrier[], m_liveThreadCount[], m_thread[] 배열 초기화 (0/false/NULL)
   *   2) threads_per_cta() 루프: ptx_sim_init_thread()를 호출하여 각 스레드 객체 할당
   *   3) 체크포인트 복원 모드이면 resume_reg_thread()로 레지스터 상태 파일에서 복원
   *   4) 모든 워프에 대해 createWarp() 호출하여 SIMT 스택 시작
   *
   * 호출 체인:
   *   execute() → [initializeCTA] → ptx_sim_init_thread(), createWarp()
   */
  void initializeCTA(unsigned ctaid_cp);

  /*
   * [한국어]
   * checkExecutionStatusAndUpdate - 스레드 완료 여부를 확인하고 m_liveThreadCount를 갱신
   *
   * @inst : 현재 실행 중인 warp_inst_t. 레인(lane) 유효성 확인에 사용 (현재 구현에서는 미사용).
   * @t    : 워프 내 레인(lane) 번호 (0 ~ warp_size-1).
   * @tid  : 전체 스레드 ID (= warp_id * warp_size + t).
   *
   * m_thread[tid]가 NULL이거나 is_done()이면 해당 워프(tid/m_warp_size)의
   * m_liveThreadCount를 1 감소시킨다.
   * execute_warp_inst_t()가 각 레인에 ptx_exec_inst() 실행 후 이 함수를 호출한다.
   *
   * core_t의 순수 가상 함수를 오버라이드한다.
   *
   * 호출 체인:
   *   execute_warp_inst_t() → [checkExecutionStatusAndUpdate]
   */
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) {
    /* [한국어] 스레드 슬롯이 비어있거나 해당 스레드가 EXIT 명령어 실행 후 완료 상태이면 */
    if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
      /* [한국어] tid를 warp_size로 정수 나눗셈하여 워프 인덱스를 구하고 활성 카운트 1 감소.
       * 예: tid=35, warp_size=32 → 워프 1번의 liveThreadCount 감소. */
      m_liveThreadCount[tid / m_warp_size]--;
    }
  }

  // lunches the stack and set the threads count
  /*
   * [한국어]
   * createWarp - 특정 워프의 SIMT 스택을 시작하고 활성 스레드 수를 설정
   *
   * @warpId : 초기화할 워프 번호.
   *
   * 실행 흐름:
   *   1) 해당 워프의 스레드 슬롯을 순회하여 NULL인 레인을 initialMask에서 제거
   *   2) m_simt_stack[warpId]->launch(first_thread_pc, initialMask):
   *      SIMT 스택에 초기 PC와 활성 마스크를 넣어 실행 시작
   *   3) 체크포인트 복원 모드(cp_cta_resume==1)이면 warp SIMT 스택 파일에서 상태 복원
   *   4) m_liveThreadCount[warpId] = liveThreadsCount 설정
   *
   * SIMT 스택(simt_stack): SIMT 분기 처리를 위한 (active_mask, PC, RPC) 스택.
   * 분기(if-else) 발생 시 스택에 경로 정보가 push되고 재수렴 시 pop된다.
   *
   * 호출 체인:
   *   initializeCTA() → [createWarp] → m_simt_stack[warpId]->launch()
   */
  void createWarp(unsigned warpId);

  // each warp live thread count and barrier indicator
  unsigned *m_liveThreadCount;
  /* [한국어] 각 워프에서 아직 실행 중인(살아있는) 스레드 수를 저장하는 동적 배열.
   * 설정자: createWarp()가 초기화, checkExecutionStatusAndUpdate()가 스레드 완료 시 감소.
   * 읽는 자: warp_waiting_at_barrier()가 모든 스레드 완료 여부 판단에 사용,
   *           executeWarp()가 실행 진행 여부 결정에 사용.
   * 값 범위: 0 ~ warp_size (32). 0이면 이 워프의 모든 스레드가 완료됨.
   * 동기화: 단일 호스트 스레드 순차 실행 — 별도 동기화 불필요. */

  bool *m_warpAtBarrier;
  /* [한국어] 각 워프가 배리어(BAR.SYNC 명령어) 대기 상태인지를 저장하는 동적 배열.
   * 설정자: executeWarp()가 BARRIER_OP 명령어 실행 후 true로 설정.
   *          execute()가 allAtBarrier == true 감지 시 모든 원소를 false로 일괄 해제.
   * 읽는 자: warp_waiting_at_barrier()가 배리어 대기 여부 반환,
   *           executeWarp()가 배리어 대기 중인 워프 실행 스킵에 사용.
   * 값 범위: true(배리어 대기 중) / false(실행 가능).
   * 동기화: 단일 호스트 스레드 순차 실행 — 별도 동기화 불필요.
   * CUDA __syncthreads() 에뮬레이션: 모든 워프가 BAR_OP에 도달했을 때 일괄 해제됨. */
};

/*
 * [한국어] RECONVERGE_RETURN_PC - 재수렴 복귀 PC 센티넬 값
 *
 * get_converge_point()가 분기 명령어의 즉시 후위 지배자(immediate post-dominator)가
 * 함수 외부에 있는 경우(다중 반환 경로)를 나타내기 위해 반환하는 특수 값.
 * (address_type)-2는 address_type이 unsigned 타입이므로 최대값 - 1에 해당하며,
 * 유효한 PTX 코드 주소 범위와 겹치지 않는 마커로 사용된다.
 * SIMT 스택 처리 시 이 값을 만나면 호출 스택의 리턴 PC를 재수렴 주소로 사용한다.
 */
#define RECONVERGE_RETURN_PC ((address_type)-2)

/*
 * [한국어] NO_BRANCH_DIVERGENCE - 분기 발산 없음을 나타내는 센티넬 값
 *
 * get_converge_point()가 현재 PC 주소가 분기 명령어가 아닌 경우(재수렴 불필요)를
 * 나타내기 위해 반환하는 특수 값.
 * (address_type)-1은 최대 address_type 값으로, 역시 유효한 코드 주소와 겹치지 않는다.
 * SIMT 스택의 pre_decode()에서 reconvergence_pc 필드에 저장되어,
 * 해당 명령어가 분기가 아님을 표시한다.
 * 분기 발산(branch divergence): 동일 워프의 스레드들이 조건부 분기에서 서로 다른
 * 경로를 취하는 현상 — GPU 성능에 심각한 영향(직렬화 실행)을 미친다.
 */
#define NO_BRANCH_DIVERGENCE ((address_type)-1)

/*
 * [한국어]
 * get_return_pc - 스레드의 함수 호출 복귀 PC를 반환
 *
 * @thd   : ptx_thread_info* 를 void* 로 전달받는다.
 *          SIMT 스택이 함수 호출 복귀 주소를 관리하는 콜백 인터페이스에서 타입 소거 사용.
 * @return: 호출 스택 최상단의 복귀 PC 주소 (address_type).
 *
 * ptx_thread_info::get_return_PC()를 호출하는 얇은 래퍼.
 * SIMT 스택이 RECONVERGE_RETURN_PC를 만나면 이 함수를 통해 실제 복귀 주소를 얻는다.
 *
 * 호출 체인:
 *   simt_stack (SIMT 스택 관리 코드) → [get_return_pc] → ptx_thread_info::get_return_PC()
 */
address_type get_return_pc(void *thd);

/*
 * [한국어]
 * get_ptxinfo_kname - ptxinfo 파서가 파싱 중인 커널의 이름을 반환
 *
 * @return: 현재 파싱 중인 커널 함수명 C 문자열 (NULL이면 바이너리 전역 섹션).
 *
 * ptxinfo 파서(ptx.tab.c 기반 Bison 파서)가 .ptxinfo 파일을 파싱할 때
 * 전역 변수 g_ptxinfo_kname에 커널 이름을 저장하며, 이 함수는 그 값을 반환한다.
 * ptxinfo_opencl_addinfo()가 커널 이름으로 function_info를 검색할 때 사용.
 */
const char *get_ptxinfo_kname();

/*
 * [한국어]
 * print_ptxinfo - 현재 g_ptxinfo 전역 구조체의 내용을 stdout에 출력 (디버그용)
 *
 * 커널 이름(g_ptxinfo_kname)과 regs/lmem/smem/cmem 값을 printf로 출력한다.
 * ptxinfo 파서가 .ptxinfo 파일을 파싱한 직후 로그 확인 목적으로 호출된다.
 */
void print_ptxinfo();

/*
 * [한국어]
 * clear_ptxinfo - g_ptxinfo 전역 구조체와 g_ptxinfo_kname을 초기화
 *
 * g_ptxinfo_kname을 free()하고 NULL로 설정, g_ptxinfo의 모든 수치 필드를 0으로 리셋.
 * 다음 커널의 ptxinfo 파싱을 시작하기 전에 이전 상태를 지우기 위해 호출된다.
 * ptxinfo_function(), ptxinfo_opencl_addinfo() 등에서 호출된다.
 */
void clear_ptxinfo();

/*
 * [한국어]
 * get_ptxinfo - 현재 g_ptxinfo 전역 구조체를 값으로 복사하여 반환
 *
 * @return: gpgpu_ptx_sim_info 구조체 복사본.
 *          regs(레지스터 수), lmem(로컬 메모리 바이트), smem(공유 메모리 바이트),
 *          cmem(상수 메모리 바이트), gmem(전역 변수 크기), ptx_version, sm_target 포함.
 *
 * 값 반환이므로 호출자는 독립적인 복사본을 얻는다.
 * function_info::set_kernel_info()로 전달되어 커널 리소스 정보가 저장된다.
 */
struct gpgpu_ptx_sim_info get_ptxinfo();

/*
 * [한국어] gpgpu_recon_t 전방 선언.
 * ptx_ir.h에서 정의된 재수렴 쌍(source 분기 PC → target 후위 지배자 PC) 정보를 담는 구조체.
 * 완전한 정의 없이 포인터만 사용하므로 전방 선언으로 충분하다.
 */
class gpgpu_recon_t;

/*
 * [한국어]
 * rec_pts - 특정 PTX 함수의 모든 재수렴 포인트(reconvergence points)를 보관하는 구조체
 *
 * 재수렴(reconvergence): SIMT 워프에서 분기(if-else, for 등) 후 갈라진 스레드들이
 * 다시 동일한 PC에서 재합류하는 지점. GPGPU-Sim은 post-dominator 분석(PDOM)을 통해
 * 각 분기 명령어의 즉시 후위 지배자(immediate post-dominator)를 미리 계산한다.
 * 계산 결과는 g_rpts 맵에 캐싱되어 동일 함수에 대한 중복 분석을 방지한다.
 */
struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;
  /* [한국어] 이 커널 함수의 재수렴 쌍 배열 포인터.
   * 설정자: find_reconvergence_points()가 calloc()으로 할당하고
   *         finfo->get_reconvergence_pairs()로 내용을 채운다.
   * 읽는 자: get_converge_point()가 source_pc 일치 항목을 선형 탐색하여 target_pc 반환.
   * 값 범위: 유효한 힙 포인터. s_num_recon == 0이면 빈 배열(그러나 NULL이 아닌 calloc 결과).
   * 동기화: 최초 접근 시 g_rpts 맵에 캐싱 후 읽기 전용 — 별도 락 불필요. */

  int s_num_recon;
  /* [한국어] s_kernel_recon_points 배열의 원소 수.
   * 설정자: find_reconvergence_points()가 finfo->get_num_reconvergence_pairs()로 설정.
   * 읽는 자: get_converge_point()의 for 루프 상한, print 루프.
   * 값 범위: 0 이상 정수. 분기 명령어가 없는 커널은 0. */
};

/*
 * [한국어]
 * cuda_sim - PTX 기능 시뮬레이션 전역 상태 및 핵심 API를 관리하는 클래스
 *
 * gpgpu_context가 유일한 인스턴스를 소유하며(gpgpu_ctx->func_sim), 모든 PTX 실행
 * 경로가 gpgpu_ctx->func_sim->XXX 형태로 이 클래스의 멤버에 접근한다.
 * 명령어 지연시간 설정, 통계 수집, 재수렴 분석, 메모리 심볼 등록, 디버그 조건 등
 * 기능 시뮬레이션에 필요한 모든 전역 상태를 하나의 클래스로 캡슐화한다.
 * 실행 컨텍스트: 시뮬레이터 초기화 후 단일 호스트 스레드 — 멀티스레드 동시 접근 없음.
 */
class cuda_sim {
 public:

  /*
   * [한국어]
   * cuda_sim 생성자 - 모든 멤버를 안전한 초기값으로 설정
   *
   * @ctx : 이 cuda_sim을 소유하는 gpgpu_context 포인터. gpgpu_ctx 역방향 포인터에 저장.
   *
   * 초기화가 필요한 이유: C++에서 POD 타입 멤버는 초기화하지 않으면 불확정 값을 가지며,
   * 이후 조건 분기에서 예측 불가능한 동작이 발생할 수 있다.
   * 주요 초기값 의미:
   *   g_ptx_kernel_count = -1: 첫 커널 실행 시 0으로 증가되어 통계 인덱스가 됨
   *   g_debug_pc = 0xBEEF1518: 유효한 PTX 코드 주소와 구분되는 마커 — "미설정" 표시
   *   g_ptx_thread_info_uid_next = 1: 0을 "없음/무효" 값으로 예약하기 위해 1부터 시작
   */
  cuda_sim(gpgpu_context *ctx) {
    /* [한국어] 지금까지 실행한 총 PTX 명령어 수 초기화 — 시뮬레이션 진행도 추적에 사용 */
    g_ptx_sim_num_insn = 0;
    g_ptx_kernel_count =
        -1;  // used for classification stat collection purposes
             /* [한국어] 커널 카운터 -1로 초기화. 최초 커널 실행 전 gpgpu_opencl_ptx_sim_init_grid()가
              * g_ptx_kernel_count++하여 0이 됨. 통계 배열 인덱스로 사용되므로 0-based. */
    /* [한국어] 셰이더 코어(SM) 수 초기화. set_param_gpgpu_num_shaders()가 실제 값을 설정 */
    gpgpu_param_num_shaders = 0;
    /* [한국어] 비동기 커널 실행 모드(false=비동기). CUDA_LAUNCH_BLOCKING=1이면 true로 변경 */
    g_cuda_launch_blocking = false;
    /* [한국어] 명령어 분류 통계 포인터 NULL 초기화. init_inst_classification_stat()에서 calloc으로 할당 */
    g_inst_classification_stat = NULL;
    /* [한국어] 명령어 연산 분류 통계 포인터 NULL 초기화. 위와 동일한 시점에 할당 */
    g_inst_op_classification_stat = NULL;
    /* [한국어] PTX 어셈블 다음 PC 초기화. function_info::ptx_assemble()이 이 값부터 PC 할당 시작 */
    g_assemble_code_next_pc = 0;
    /* [한국어] 디버그 대상 스레드 UID 초기화. 0이면 모든 스레드에 디버그 정보 출력 */
    g_debug_thread_uid = 0;
    /* [한국어] 내장 PTX 덮어쓰기 비활성화. PTX_SIM_USE_PTX_FILE 환경변수 설정 시 true로 변경 */
    g_override_embedded_ptx = false;
    /* [한국어] 텍스처 레지스터 포인터 NULL 초기화 */
    ptx_tex_regs = NULL;
    /* [한국어] 삭제된 ptx_thread_info 객체 수 초기화 — 메모리 누수 디버깅용 카운터 */
    g_ptx_thread_info_delete_count = 0;
    /* [한국어] 다음 스레드 UID를 1로 초기화. 0은 "없음"을 의미하므로 1부터 시작 */
    g_ptx_thread_info_uid_next = 1;
    /* [한국어] 디버그 PC를 0xBEEF1518 마커로 초기화.
     * ptx_debug_exec_dump_cond()가 이 값이면 PC 필터링을 건너뜀("미설정" 상태).
     * PTX_SIM_DEBUG_PC 환경변수로 실제 디버그 대상 PC를 지정하면 이 값이 교체됨 */
    g_debug_pc = 0xBEEF1518;
    /* [한국어] 역방향 포인터 저장. 이 객체를 통해 gpgpu_ctx의 다른 서브시스템에 접근 */
    gpgpu_ctx = ctx;
  }

  /* ===========================================================
   * [한국어] PTX 명령어 지연시간(latency) 및 개시 간격(initiation interval) 설정 문자열
   * gpgpusim.config 파일의 -ptx_opcode_latency_* 및 -ptx_opcode_initiation_* 옵션이
   * option_parser를 통해 이 포인터들에 쉼표 구분 문자열로 저장된다.
   * set_opcode_and_latency()가 sscanf로 파싱하여 각 ptx_instruction의 latency/initiation_interval 필드를 설정.
   * =========================================================== */

  char *opcode_latency_int;
  /* [한국어] 정수(INT) 연산 명령어 지연시간 문자열.
   * 설정자: ptx_opcocde_latency_options()가 option_parser_register() 기본값 "1,1,19,25,145,32" 적용.
   * 형식: "ADD/SUB,MAX/MIN,MUL,MAD,DIV,SHFL" 순서의 사이클 수.
   * 예: ADD=1사이클, MUL=19사이클, DIV=145사이클 (NVIDIA GPU 측정값 기반). */

  char *opcode_latency_fp;
  /* [한국어] 단정밀도 부동소수점(FP32) 연산 지연시간 문자열.
   * 기본값: "1,1,1,1,30" (ADD,MAX,MUL,MAD,DIV 순서).
   * 단정밀도(F32): 32비트 IEEE 754 부동소수점, 약 7자리 유효자릿수. */

  char *opcode_latency_dp;
  /* [한국어] 배정밀도 부동소수점(FP64) 연산 지연시간 문자열.
   * 기본값: "8,8,8,8,335" (ADD,MAX,MUL,MAD,DIV 순서).
   * 배정밀도(F64): 64비트, 약 15자리 유효자릿수. FP32 대비 낮은 처리량. */

  char *opcode_latency_sfu;
  /* [한국어] SFU(Special Function Unit) 명령어 지연시간 문자열.
   * 기본값: "8". SFU는 sin/cos/rcp/sqrt/rsqrt/ex2/lg2 등 초월함수를 담당하는 하드웨어 유닛. */

  char *opcode_latency_tensor;
  /* [한국어] 텐서 코어(Tensor Core) 명령어 지연시간 문자열.
   * 기본값: "64". 텐서 코어는 Volta 아키텍처 이후 행렬 곱셈(WMMA/MMA)에 사용되는 전용 유닛.
   * AI/딥러닝 워크로드에서 핵심 가속 요소. */

  char *opcode_initiation_int;
  /* [한국어] 정수 연산 개시 간격(initiation interval) 문자열.
   * 기본값: "1,1,4,4,32,4". 파이프라인에서 동일 유닛에 다음 명령어를 투입할 수 있을 때까지의 사이클 수.
   * 지연시간과 다를 수 있음: MUL 지연시간=19사이클이지만 개시간격=4사이클(파이프라인화). */

  char *opcode_initiation_fp;
  /* [한국어] FP32 개시 간격 문자열. 기본값: "1,1,1,1,5". */

  char *opcode_initiation_dp;
  /* [한국어] FP64 개시 간격 문자열. 기본값: "8,8,8,8,130". */

  char *opcode_initiation_sfu;
  /* [한국어] SFU 개시 간격 문자열. 기본값: "8". */

  char *opcode_initiation_tensor;
  /* [한국어] 텐서 코어 개시 간격 문자열. 기본값: "64". */

  int cp_count;
  /* [한국어] 기능 시뮬레이션 체크포인트 명령어 한도.
   * 설정자: gpgpu_cuda_ptx_sim_main_func()가 checkpoint_insn_Y 값으로 설정.
   * 읽는 자: functionalCoreSim::execute()가 이 명령어 수 이후 조기 종료 조건에 사용.
   * 값 범위: 0이면 체크포인트 없이 완전 실행; 양수이면 해당 명령어 수 이후 중단. */

  int cp_cta_resume;
  /* [한국어] 체크포인트 재개 시작 CTA 번호.
   * 설정자: gpgpu_cuda_ptx_sim_main_func()가 checkpoint_CTA_t 값으로 설정.
   * 읽는 자: initializeCTA()가 1이면 레지스터/SIMT 스택을 파일에서 복원(resume).
   * 값 범위: 0이면 재개 없이 처음부터 실행; 1이면 체크포인트 파일에서 복원. */

  int g_ptxinfo_error_detected;
  /* [한국어] ptxinfo 파서에서 감지된 오류 플래그.
   * 설정자: ptxinfo 파서(ptx.tab.c Bison 규칙)가 오류 발생 시 설정.
   * 읽는 자: PTX 로더가 파싱 완료 후 오류 여부 확인에 사용.
   * 값 범위: 0 = 정상; 비零 = 오류 감지됨. */

  unsigned g_ptx_sim_num_insn;
  /* [한국어] 지금까지 시뮬레이션된 총 PTX 명령어 실행 횟수.
   * 설정자: ptx_exec_inst()가 명령어 실행마다 1 증가.
   * 읽는 자: 진행도 로그(100000개마다 출력), 최종 시뮬레이션 속도 통계.
   * 값 범위: 0 이상. 복잡한 커널에서 수십억 단위까지 증가 가능.
   * 동기화: 단일 호스트 스레드 순차 실행 — 별도 동기화 불필요. */

  char *cdp_latency_str;
  /* [한국어] CDP(CUDA Dynamic Parallelism) API 지연시간 설정 문자열.
   * 기본값: "7200,8000,100,12000,1600".
   * 형식: "cudaStreamCreateWithFlags, cudaGetParameterBufferV2_init_perWarp,
   *         cudaGetParameterBufferV2_perKernel, cudaLaunchDeviceV2_init_perWarp,
   *         cudaLaunchDevicV2_perKernel" 순서의 사이클 수.
   * CDP: GPU 커널이 런타임에 새 커널을 동적으로 실행하는 CUDA 5.0 기능.
   * 이 값은 sscanf로 파싱되어 cdp_latency[5] 배열에 저장된다. */

  int g_ptx_kernel_count;  // used for classification stat collection purposes
  /* [한국어] 실행된 커널 번호 카운터 (통계 분류 목적).
   * 설정자: gpgpu_opencl_ptx_sim_init_grid()가 커널 그리드 생성 시 ++.
   * 읽는 자: init_inst_classification_stat()가 g_inst_classification_stat 배열 인덱스로 사용.
   * 값 범위: -1(초기) → 0(첫 커널) → 1, 2, ... (최대 MAX_CLASS_KER-1 = 1023). */

  std::map<const void *, std::string> g_global_name_lookup;
  /* [한국어] 호스트 전역 변수 주소 → PTX 전역 변수 이름 매핑 테이블.
   * 설정자: gpgpu_ptx_sim_register_global_variable()이 libcuda의 cudaRegisterVar() 인터셉트 시 등록.
   * 읽는 자: gpgpu_ptx_sim_memcpy_symbol()이 cudaMemcpyToSymbol() 처리 시 이름 조회.
   * 키: 호스트 측 __device__ 변수의 주소 (컴파일러가 생성한 전역 포인터).
   * 값: PTX 파일에서의 변수 이름 (예: "d_data"). */
  // indexed by hostVar

  std::map<const void *, std::string> g_const_name_lookup;
  /* [한국어] 호스트 상수 변수 주소 → PTX 상수 변수 이름 매핑 테이블.
   * 설정자: gpgpu_ptx_sim_register_const_variable()이 cudaRegisterVar() 인터셉트 시 등록.
   * 읽는 자: gpgpu_ptx_sim_memcpy_symbol()이 cudaMemcpyToSymbol() 처리 시 이름 조회.
   * 키: 호스트 측 __constant__ 변수의 주소.
   * 상수 메모리: GPU에서 모든 스레드가 읽기 전용으로 공유하는 캐시 가속 메모리 공간. */
  // indexed by hostVar

  int g_ptx_sim_mode;  // if non-zero run functional simulation only (i.e., no
                       // notion of a clock cycle)
  /* [한국어] PTX 시뮬레이션 모드 플래그.
   * 설정자: read_sim_environment_variables()가 환경변수 PTX_SIM_MODE_FUNC 값으로 설정.
   * 읽는 자: gpgpusim_entrypoint가 기능 전용 모드 여부 판단에 사용.
   * 값 범위: 0 = 타이밍+기능 혼합 시뮬레이션(기본); 1 이상 = 기능 시뮬레이션만 실행.
   * 기능 전용 모드: 클럭 사이클 없이 결과 정확성만 검증 — 훨씬 빠르게 실행됨. */

  unsigned gpgpu_param_num_shaders;
  /* [한국어] 시뮬레이션 중인 GPU의 SM(셰이더 코어) 수.
   * 설정자: set_param_gpgpu_num_shaders()가 gpgpusim.config의 셰이더 수 옵션을 읽어 설정.
   * 읽는 자: ptx_sim_init_thread()가 sm_idx = hw_cta_id * num_shaders + sid 계산에 사용.
   * 값 범위: 1 이상의 양수. 실제 GPU 모델에 따라 수십~수백 개. */

  class std::map<function_info *, rec_pts> g_rpts;
  /* [한국어] PTX 함수 → 재수렴 포인트 목록 캐시 맵.
   * 설정자: find_reconvergence_points()가 최초 접근 시 PDOM 분석 결과를 계산하여 삽입.
   * 읽는 자: find_reconvergence_points()가 재접근 시 캐시 결과를 반환 (중복 분석 방지).
   *           get_converge_point()가 분기 명령어의 목표 PC 조회 시 사용.
   * 키: function_info 포인터 (커널 또는 디바이스 함수 단위).
   * 동기화: 단일 호스트 스레드 순차 실행 — 별도 락 불필요. */

  bool g_cuda_launch_blocking;
  /* [한국어] CUDA 커널 동기 실행(blocking) 모드 플래그.
   * 설정자: read_sim_environment_variables()가 환경변수 CUDA_LAUNCH_BLOCKING=1이면 true 설정.
   *          CUDART_VERSION <= 1010이면 항상 true (구형 런타임은 동기 실행만 지원).
   * 읽는 자: libcuda의 cudaLaunch() 인터셉트 코드가 커널 완료 대기 여부 결정에 사용.
   * 값 범위: true(동기: CPU가 커널 완료 대기) / false(비동기: CPU가 즉시 반환).
   * 성능 영향: true이면 CPU-GPU 파이프라인 오버랩 불가 — 일반적으로 false를 권장. */

  void **g_inst_classification_stat;
  /* [한국어] 커널별 PTX 명령어 분류 통계 배열 포인터.
   * 설정자: init_inst_classification_stat()가 calloc(MAX_CLASS_KER=1024)으로 할당.
   *         각 원소는 StatCreate()로 생성된 통계 버킷(최대 20개 분류).
   * 읽는 자: ptx_exec_inst()가 StatAddSample()로 op_classification 값을 기록.
   *           gpgpu_cuda_ptx_sim_main_func() 완료 후 StatDisp()로 출력.
   * 인덱스: g_ptx_kernel_count (커널 번호). */

  void **g_inst_op_classification_stat;
  /* [한국어] 커널별 PTX 명령어 opcode 분류 통계 배열 포인터.
   * g_inst_classification_stat와 구조 동일하나 분류 기준이 opcode(최대 100가지).
   * ptx_exec_inst()가 pI->get_opcode()를 직접 기록하여 어떤 opcode가 많이 실행되는지 추적. */

  std::set<std::string> g_globals;
  /* [한국어] PTX 전역(__device__) 변수 이름 집합.
   * 설정자: PTX 파서가 .global 선언을 만날 때 추가.
   * 읽는 자: gpgpu_ptx_sim_memcpy_symbol()이 문자열 hostVar로 전역 변수를 찾을 때 fallback 조회.
   * CUDA 5.0 이후 문자열 hostVar 방식이 deprecated되었으므로 구형 코드 호환용. */

  std::set<std::string> g_constants;
  /* [한국어] PTX 상수(__constant__) 변수 이름 집합.
   * 설정자: PTX 파서가 .const 선언을 만날 때 추가.
   * 읽는 자: gpgpu_ptx_sim_memcpy_symbol()의 g_globals 조회와 동일한 fallback 경로. */

  std::map<unsigned, function_info *> g_pc_to_finfo;
  /* [한국어] PTX 명령어 주소(PC) → 해당 명령어가 속한 function_info 역방향 매핑 테이블.
   * 설정자: function_info::ptx_assemble()이 각 명령어에 PC를 할당할 때 등록.
   * 읽는 자: ptx_print_insn(), ptx_get_insn_str() — PC로 함수 정보를 찾아 명령어 텍스트 출력.
   *           get_converge_point() — PC 소속 함수를 찾아 재수렴 포인트 목록 조회.
   * 키: 전역 유일 PC 값 (모든 함수에 걸쳐 중복 없는 주소). */

  int gpgpu_ptx_instruction_classification;
  /* [한국어] PTX 명령어 분류 통계 수집 활성화 여부 플래그.
   * 설정자: gpgpusim.config의 옵션(또는 기본값 0).
   * 읽는 자: ptx_exec_inst()가 비零이면 StatAddSample()로 통계 기록.
   * 값 범위: 0 = 비활성(기본); 비零 = 통계 수집 활성화. */

  unsigned cdp_latency[5];
  /* [한국어] CDP(CUDA Dynamic Parallelism) API 단계별 지연시간 배열.
   * 설정자: set_opcode_and_latency()가 cdp_latency_str을 sscanf로 파싱하여 채움.
   * 인덱스 의미:
   *   [0]: cudaStreamCreateWithFlags (7200 사이클 기본)
   *   [1]: cudaGetParameterBufferV2 per-warp 초기화 (8000)
   *   [2]: cudaGetParameterBufferV2 per-kernel (100)
   *   [3]: cudaLaunchDeviceV2 per-warp 초기화 (12000)
   *   [4]: cudaLaunchDeviceV2 per-kernel (1600)
   * 읽는 자: cuda_device_runtime.cc의 CDP API 에뮬레이션 함수들. */

  unsigned g_assemble_code_next_pc;
  /* [한국어] 다음에 어셈블될 PTX 함수의 시작 PC.
   * 설정자: function_info::ptx_assemble()이 현재 함수 명령어들을 배치한 후 마지막 PC+1 저장.
   * 읽는 자: 다음 function_info::ptx_assemble()이 이 PC에서 시작하여 전역 유일 주소 할당.
   * 값 범위: 0부터 시작하여 모든 커널 함수 어셈블 완료 시점까지 단조 증가.
   * MAX_INST_SIZE(8바이트) 배수로 정렬되어 함수 간 주소가 겹치지 않도록 보장. */

  int g_debug_thread_uid;
  /* [한국어] 디버그 출력을 제한할 특정 스레드의 UID.
   * 설정자: read_sim_environment_variables()가 PTX_SIM_DEBUG_THREAD_UID 환경변수에서 파싱.
   * 읽는 자: ptx_debug_exec_dump_cond()가 0이 아닌 경우 이 UID와 일치하는 스레드만 출력.
   * 값 범위: 0 = 모든 스레드 디버그 출력; 양수 = 해당 UID 스레드만 출력. */

  bool g_override_embedded_ptx;
  /* [한국어] CUDA 바이너리에 내장된 PTX 코드 대신 외부 .ptx 파일을 사용할지 여부.
   * 설정자: read_sim_environment_variables()가 PTX_SIM_USE_PTX_FILE 환경변수 설정 시 true.
   *          CUDART_VERSION <= 1010이면 무조건 true (구형 런타임은 내장 PTX 미지원).
   * 읽는 자: PTX 로더가 CUBIN에서 PTX를 추출할지 파일을 읽을지 결정에 사용.
   * 의미: true이면 실행 디렉토리의 .ptx 파일을 로드하여 시뮬레이션 소스로 사용. */

  std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
  /* [한국어] 이미 CTA 정보 객체가 생성된 SM 인덱스들의 집합.
   * 설정자: ptx_sim_init_thread()가 새 SM 인덱스의 CTA를 할당할 때 삽입 (현재 미사용, 예비).
   * 값 범위: sm_idx = hw_cta_id * num_shaders + sid 형태의 조합 인덱스.
   * unsigned long long: 넓은 SM 인덱스 범위를 수용하기 위해 64비트 사용. */

  ptx_reg_t *ptx_tex_regs;
  /* [한국어] 텍스처 명령어 실행 시 임시 레지스터 저장 포인터.
   * 설정자: 현재 코드베이스에서 명시적 초기화 외 사용처 제한적 — 향후 확장을 위한 예비.
   * ptx_reg_t: PTX 레지스터 값 유니온 (u32, s32, f32, u64 등 여러 타입을 하나의 union으로). */

  unsigned g_ptx_thread_info_delete_count;
  /* [한국어] 지금까지 delete된 ptx_thread_info 객체 수 — 메모리 관리 디버깅용 카운터.
   * 설정자: ptx_thread_info의 소멸 경로에서 증가 (현재 코드에서는 직접 사용처 제한적).
   * 읽는 자: 메모리 누수 추적 디버그 출력. */

  unsigned g_ptx_thread_info_uid_next;
  /* [한국어] 다음에 생성될 ptx_thread_info 객체에 부여할 고유 식별자(UID).
   * 설정자: ptx_thread_info 생성자가 이 값을 UID로 채택한 후 1 증가.
   * 읽는 자: g_debug_thread_uid와 비교하여 특정 스레드 디버그 출력 결정.
   * 값 범위: 1부터 단조 증가. 0은 "없음/무효" 예약값. */

  addr_t g_debug_pc;
  /* [한국어] 디버그 출력을 제한할 특정 PTX 명령어 주소(PC).
   * 설정자: read_sim_environment_variables()가 PTX_SIM_DEBUG_PC 환경변수에서 파싱.
   *          초기값 0xBEEF1518은 "미설정" 마커 — 이 값이면 PC 필터링 없이 모두 출력.
   * 읽는 자: ptx_debug_exec_dump_cond()가 0xBEEF1518이 아닌 경우 이 PC 명령어만 디버그 출력.
   * addr_t: GPU 주소 공간의 주소 타입 (abstract_hardware_model.h에서 typedef). */

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] 이 cuda_sim 객체를 소유하는 gpgpu_context에 대한 역방향 포인터.
   * 설정자: 생성자에서 ctx 매개변수로 초기화 — 이후 변경 없음.
   * 읽는 자: 모든 멤버 함수가 gpgpu_ctx->the_gpgpusim, ->ptx_parser, ->device_runtime 등에
   *           접근하기 위해 사용.
   * 역방향 포인터가 필요한 이유: cuda_sim이 상위 컨텍스트의 다른 서브시스템
   *   (stream_manager, gpgpu_sim 등)과 협력해야 하기 때문.
   * 동기화: 읽기 전용 — 초기화 이후 변경 없음. */

  /* ==================================================================
   * [한국어] 주요 공개 멤버 함수들
   * ================================================================== */

  /*
   * [한국어]
   * ptx_opcocde_latency_options - PTX 명령어 지연시간/개시간격 옵션을 파서에 등록
   *
   * @opp : option_parser_t 핸들. gpgpusim.config 파일이나 명령줄에서 옵션을 읽는 도구.
   *
   * option_parser_register()를 11번 호출하여 -ptx_opcode_latency_int/fp/dp/sfu/tensor,
   * -ptx_opcode_initiation_int/fp/dp/sfu/tensor, -cdp_latency 옵션을 등록한다.
   * 각 옵션은 OPT_CSTR 타입으로 해당 char* 멤버 포인터에 문자열을 저장한다.
   * gpgpusim_entrypoint에서 시뮬레이터 초기화 단계에 호출된다.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint → [ptx_opcocde_latency_options] → option_parser_register() × 11
   */
  void ptx_opcocde_latency_options(option_parser_t opp);

  /*
   * [한국어]
   * gpgpu_cuda_ptx_sim_main_func - CUDA PTX 기능 시뮬레이션의 최상위 진입 함수
   *
   * @kernel : 실행할 커널의 메타데이터+스레드 리스트를 담은 kernel_info_t 참조.
   * @openCL : true이면 OpenCL 커널, false이면 CUDA 커널(기본값).
   *           OpenCL이면 완료 시 stream_manager에 등록하지 않음.
   * @return : void
   *
   * 실행 흐름:
   *   1) PDOM 분석(do_pdom()): 커널 내 분기 명령어의 후위 지배자 PC 계산 (최초 실행 시)
   *   2) max_cta() 호출: 이 커널이 SM에서 동시에 실행 가능한 최대 CTA 수 계산
   *   3) kernel.no_more_ctas_to_run() 루프: CTA 하나씩 functionalCoreSim 생성 후 execute()
   *   4) 체크포인트 옵션(cp_op==1)이면 전역 메모리를 파일로 덤프
   *   5) 완료 후 stream_manager::register_finished_kernel() 호출
   *   6) 시뮬레이션 시간과 명령어 처리율(inst/sec) 출력
   *
   * 호출 체인:
   *   gpgpusim_entrypoint::gpgpu_cuda_ptx_sim_main_func() →
   *   [gpgpu_cuda_ptx_sim_main_func] →
   *     functionalCoreSim::execute() × num_CTAs
   */
  void gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel, bool openCL = false);

  /*
   * [한국어]
   * gpgpu_opencl_ptx_sim_main_func - OpenCL PTX 기능 시뮬레이션 메인 함수
   *
   * @grid   : 실행할 OpenCL 커널 정보 포인터(kernel_info_t*).
   * @return : int. 정상 완료 = 0. (현재 구현에서는 내부에서 gpgpu_cuda_ptx_sim_main_func 호출)
   *
   * OpenCL 커널 실행 경로에서 호출되며, openCL=true 플래그로 CUDA 경로와 동일한
   * gpgpu_cuda_ptx_sim_main_func()를 재사용한다.
   *
   * 호출 체인:
   *   libopencl 인터셉트 → [gpgpu_opencl_ptx_sim_main_func] → gpgpu_cuda_ptx_sim_main_func()
   */
  int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid);

  /*
   * [한국어]
   * init_inst_classification_stat - 현재 커널의 명령어 분류 통계를 초기화
   *
   * g_ptx_kernel_count 번호의 통계 버킷이 아직 초기화되지 않았으면 StatCreate()로 생성.
   * static std::set<unsigned> init을 사용하여 동일 커널 번호의 중복 초기화를 방지.
   * 최대 MAX_CLASS_KER(1024)개 커널 지원.
   *
   * 호출 체인:
   *   ptx_exec_inst() (gpgpu_ptx_instruction_classification 비零 시) → [init_inst_classification_stat]
   */
  void init_inst_classification_stat();

  /*
   * [한국어]
   * gpgpu_opencl_ptx_sim_init_grid - OpenCL 커널 실행을 위한 kernel_info_t 그리드 생성
   *
   * @entry    : 실행할 커널의 PTX function_info 포인터.
   * @args     : 커널 인자 목록 (gpgpu_ptx_sim_arg_list_t = std::list<gpgpu_ptx_sim_arg>).
   *             각 원소는 (데이터 포인터, 크기, 오프셋) 튜플.
   * @gridDim  : 그리드 차원 (CTA 개수: x*y*z).
   * @blockDim : 블록 차원 (CTA 내 스레드 수: x*y*z).
   * @gpu      : gpgpu_t GPU 객체 포인터. 텍스처/배열 매핑 정보 제공.
   * @return   : 새로 할당된 kernel_info_t 포인터. 호출자가 소유권을 가진다.
   *
   * 실행 흐름:
   *   1) kernel_info_t(gridDim, blockDim, entry, ...) 생성
   *   2) args 역순 순회하며 entry->add_param_data() 호출
   *   3) entry->finalize(result->get_param_memory()): 파라미터를 파라미터 메모리 공간에 복사
   *   4) g_ptx_kernel_count++ 증가
   *
   * 호출 체인:
   *   libopencl 인터셉트 → [gpgpu_opencl_ptx_sim_init_grid] → kernel_info_t 생성
   */
  kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                                gpgpu_ptx_sim_arg_list_t args,
                                                struct dim3 gridDim,
                                                struct dim3 blockDim,
                                                gpgpu_t *gpu);

  /*
   * [한국어]
   * gpgpu_ptx_sim_register_global_variable - __device__ 전역 변수를 이름 테이블에 등록
   *
   * @hostVar    : 호스트 측 __device__ 변수 주소. CUDA 런타임이 생성한 포인터.
   * @deviceName : PTX 파일에서의 변수 이름 (예: "_ZN...d_data").
   * @size       : 변수 크기(바이트). 로그 출력용.
   *
   * g_global_name_lookup[hostVar] = deviceName 매핑을 등록한다.
   * cudaRegisterVar() 인터셉트 시 libcuda가 이 함수를 호출한다.
   *
   * 호출 체인:
   *   libcuda::cudaRegisterVar() → [gpgpu_ptx_sim_register_global_variable]
   */
  void gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                              const char *deviceName,
                                              size_t size);

  /*
   * [한국어]
   * gpgpu_ptx_sim_register_const_variable - __constant__ 상수 변수를 이름 테이블에 등록
   *
   * @hostVar    : 호스트 측 __constant__ 변수 주소.
   * @deviceName : PTX 파일에서의 변수 이름.
   * @size       : 변수 크기(바이트). 로그 출력용.
   *
   * g_const_name_lookup[hostVar] = deviceName 매핑을 등록한다.
   * gpgpu_ptx_sim_register_global_variable()과 동일한 패턴이지만 상수 메모리 전용.
   *
   * 호출 체인:
   *   libcuda::cudaRegisterVar() → [gpgpu_ptx_sim_register_const_variable]
   */
  void gpgpu_ptx_sim_register_const_variable(void *, const char *deviceName,
                                             size_t size);

  /*
   * [한국어]
   * read_sim_environment_variables - 환경 변수에서 시뮬레이션 설정을 읽어 적용
   *
   * 읽는 환경변수:
   *   PTX_SIM_MODE_FUNC    → g_ptx_sim_mode (기능 전용 vs 타이밍+기능)
   *   GPGPUSIM_DEBUG       → g_interactive_debugger_enabled (인터랙티브 디버거)
   *   PTX_SIM_DEBUG        → g_debug_execution (디버그 레벨)
   *   PTX_SIM_DEBUG_THREAD_UID → g_debug_thread_uid (특정 스레드 추적)
   *   PTX_SIM_DEBUG_PC     → g_debug_pc (특정 PC 추적)
   *   PTX_SIM_USE_PTX_FILE → g_override_embedded_ptx (외부 PTX 파일 사용)
   *   CUDA_LAUNCH_BLOCKING → g_cuda_launch_blocking (동기 실행 모드)
   *
   * gpgpusim_entrypoint 초기화 시 최초 1회 호출된다.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint::gpgpu_ptx_sim_init_perf() → [read_sim_environment_variables]
   */
  void read_sim_environment_variables();

  /*
   * [한국어]
   * set_param_gpgpu_num_shaders - SM(셰이더 코어) 수 설정
   *
   * @num_shaders : GPU 설정 파일에서 읽은 SM 수. gpgpu_param_num_shaders에 저장.
   *
   * ptx_sim_init_thread()가 sm_idx 계산에 이 값을 사용하므로 PTX 시뮬레이션 전에
   * 반드시 설정되어야 한다.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint 초기화 → [set_param_gpgpu_num_shaders]
   */
  void set_param_gpgpu_num_shaders(int num_shaders);

  /*
   * [한국어]
   * find_reconvergence_points - PTX 함수의 재수렴 포인트를 분석하고 캐싱
   *
   * @finfo  : 분석할 커널/디바이스 함수의 function_info 포인터.
   * @return : rec_pts 구조체 (값 반환). s_kernel_recon_points 배열과 s_num_recon 포함.
   *
   * g_rpts 맵에서 finfo 키를 먼저 찾고, 없으면 PDOM(Post-DOMinator) 분석을 수행한다:
   *   1) finfo->get_num_reconvergence_pairs()로 재수렴 쌍 수 획득
   *   2) calloc으로 gpgpu_recon_t 배열 할당
   *   3) finfo->get_reconvergence_pairs()로 배열 채움
   *   4) 결과를 g_rpts[finfo]에 저장 (이후 같은 함수 요청 시 캐시 반환)
   *
   * 호출 체인:
   *   get_converge_point() → [find_reconvergence_points]
   *   gpgpu_cuda_ptx_sim_main_func() → (do_pdom 직접 호출로 대체된 경우도 있음)
   */
  struct rec_pts find_reconvergence_points(function_info *finfo);

  /*
   * [한국어]
   * get_converge_point - 분기 명령어 PC에 대응하는 재수렴 목표 PC를 반환
   *
   * @pc     : 분기 명령어(BRA, CALLP 등)의 프로그램 카운터 값.
   * @return : 재수렴 목표 주소.
   *           RECONVERGE_RETURN_PC(-2이면): 함수 외부에서 재수렴 — 호출 스택 복귀 PC 사용
   *           NO_BRANCH_DIVERGENCE(-1이면): 분기 발산 없음 (단순 조건 분기이거나 비분기)
   *           그 외: 즉시 후위 지배자(immediate post-dominator)의 PC
   *
   * g_pc_to_finfo로 소속 함수를 찾고, find_reconvergence_points()로 재수렴 목록을 얻은 후
   * source_pc == pc인 항목을 선형 탐색하여 target_pc를 반환한다.
   * pre_decode()가 이 값을 reconvergence_pc 필드에 저장한다.
   *
   * 호출 체인:
   *   ptx_instruction::pre_decode() → [get_converge_point]
   */
  address_type get_converge_point(address_type pc);

  /*
   * [한국어]
   * gpgpu_ptx_sim_memcpy_symbol - PTX 심볼(전역/상수 변수)로 데이터를 복사
   *
   * @hostVar : 심볼을 식별하는 키. 호스트 측 변수 주소 또는 변수 이름 문자열.
   * @src     : 복사 원본 데이터 포인터 (호스트 메모리).
   * @count   : 복사할 바이트 수.
   * @offset  : 심볼 시작 주소로부터의 오프셋 (바이트 단위).
   * @to      : 복사 방향. 1이면 호스트→GPU(to device), 0이면 GPU→호스트(from device).
   * @gpu     : gpgpu_t 포인터. 시뮬레이션 전역 메모리(m_global_mem) 접근 제공.
   *
   * CUDA의 cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()을 에뮬레이션한다.
   * hostVar를 g_const_name_lookup → g_global_name_lookup → g_globals/g_constants 순으로
   * 탐색하여 디바이스 측 변수명 sym_name을 찾고, 심볼 테이블에서 실제 주소를 얻은 후
   * memory_space::write() 또는 read()로 바이트 단위 복사를 수행한다.
   *
   * 호출 체인:
   *   libcuda::cudaMemcpyToSymbol() → [gpgpu_ptx_sim_memcpy_symbol]
   *     → memory_space::write() / read()
   */
  void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                   size_t count, size_t offset, int to,
                                   gpgpu_t *gpu);

  /*
   * [한국어]
   * ptx_print_insn - 특정 PC의 PTX 명령어 텍스트를 파일 스트림에 출력
   *
   * @pc : 출력할 명령어의 프로그램 카운터.
   * @fp : 출력 대상 FILE 포인터 (stdout, stderr 또는 디버그 파일).
   *
   * g_pc_to_finfo로 소속 함수를 찾아 function_info::print_insn(pc, fp)을 호출한다.
   * PC가 등록되지 않은 경우 "<no instruction at address 0x...>" 메시지를 출력.
   * 디버그 및 에러 출력 경로에서 사용된다.
   *
   * 호출 체인:
   *   디버그 코드, 에러 핸들러 → [ptx_print_insn] → function_info::print_insn()
   */
  void ptx_print_insn(address_type pc, FILE *fp);

  /*
   * [한국어]
   * ptx_get_insn_str - 특정 PC의 PTX 명령어 텍스트를 std::string으로 반환
   *
   * @pc     : 텍스트를 얻을 명령어의 프로그램 카운터.
   * @return : 명령어 어셈블리 텍스트. PC가 유효하지 않으면 오류 메시지 문자열.
   *
   * ptx_print_insn()과 동일한 로직이나 FILE 대신 string을 반환한다.
   * 파이프라인 시각화, 이벤트 로그 등 문자열 기반 출력에 사용.
   *
   * 호출 체인:
   *   파이프라인 시각화, 로그 → [ptx_get_insn_str] → function_info::get_insn_str()
   */
  std::string ptx_get_insn_str(address_type pc);

  /*
   * [한국어]
   * ptx_debug_exec_dump_cond - 이 스레드/PC 조합에 대한 디버그 출력 여부를 판단
   *
   * 템플릿 파라미터 <int activate_level>: 컴파일 시간에 결정되는 최소 디버그 레벨.
   *   예: ptx_debug_exec_dump_cond<5>(uid, pc) → g_debug_execution >= 5일 때만 true 가능.
   *   예: ptx_debug_exec_dump_cond<10>(uid, pc) → g_debug_execution >= 10일 때만 true 가능.
   * @thd_uid : 확인할 스레드의 고유 ID.
   * @pc      : 확인할 명령어의 PC.
   * @return  : true이면 이 지점에서 디버그 출력 실행; false이면 스킵.
   *
   * 판정 로직:
   *   1) g_debug_execution < activate_level → false
   *   2) g_debug_thread_uid != 0 && thd_uid != g_debug_thread_uid → false (스레드 미일치)
   *   3) g_debug_pc != 0xBEEF1518 && pc != g_debug_pc → false (PC 미일치)
   *   4) 모든 조건 통과 → true
   *
   * 호출 체인:
   *   ptx_exec_inst() → [ptx_debug_exec_dump_cond<5>/<6>/<10>]
   */
  template <int activate_level>
  bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc);
};

#endif  // CUDASIM_H_INCLUDED의 끝: 인클루드 가드의 닫는 부분
        // #ifndef로 시작한 조건부 컴파일 블록을 여기서 닫습니다
