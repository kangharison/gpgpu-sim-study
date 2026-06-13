/*
 * [한국어 설명] GPGPU-Sim GPU 타이밍 시뮬레이터 핵심 헤더 (gpu-sim.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 최상위 GPU 타이밍 시뮬레이터를 정의하는 핵심 헤더이다.
 * GPU 전체(모든 SM 클러스터, L2 캐시, DRAM 파티션, ICNT)를 사이클-레벨로
 * 시뮬레이션하는 gpgpu_sim 클래스와 그에 필요한 설정 구조체들을 선언한다.
 * 여기서 선언된 gpgpu_sim::cycle()이 매 클럭 사이클마다 호출되며, 전체 GPU
 * 마이크로아키텍처의 상태 전이를 구동하는 시뮬레이션의 심장부 역할을 한다.
 * 또한 전력 설정(power_config), 메모리 설정(memory_config), 시뮬레이션 전체
 * 설정(gpgpu_sim_config) 클래스를 정의하여 gpgpusim.config 파일로부터 읽은
 * 파라미터를 구조화된 객체로 관리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 타이밍 시뮬레이션 계층(gpgpu-sim/)의 최상위 진입점이다.
 * 호출 체인:
 *   gpgpusim_entrypoint.cc (cuLaunchKernel 인터셉트)
 *     → stream_manager.cc (CUDA 스트림 관리)
 *       → gpgpu_sim::cycle() [이 파일]  ← 매 코어 클럭 사이클마다 반복
 *           → shader_core_cluster::core_cycle()  (SM 파이프라인: shader.cc)
 *           → memory_partition_unit::dram_cycle() (DRAM 타이밍: dram.cc)
 *           → memory_sub_partition::cache_cycle() (L2 캐시: l2cache.cc)
 *           → icnt_transfer() (NoC 라우팅: intersim2/)
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드 (시뮬레이션 루프).
 * 다중 SM/클러스터는 루프로 순차 처리하며, 실제 병렬성 없음.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - shader.h: shader_core_config, simt_core_cluster (SM 설정 및 클러스터)
 *   - gpu-cache.h: l2_cache_config, cache_stats (L2 캐시 설정)
 *   - addrdec.h: linear_to_raw_address_translation (주소→DRAM 매핑)
 *   - abstract_hardware_model.h: gpgpu_t 기반 클래스, kernel_info_t, mem_fetch
 *   - option_parser.h: gpgpusim.config 파라미터 등록/파싱
 *   - dram.h (gpu-sim.cc에서): memory_partition_unit DRAM 타이밍 모델
 *   - intersim2/ (icnt_wrapper.h): ICNT NoC 시뮬레이터 인터페이스
 * 이 파일에 의존하는 모듈:
 *   - gpgpusim_entrypoint.cc: exec_gpgpu_sim 생성 및 cycle() 구동
 *   - stream_manager.cc: gpgpu_sim::launch(), active(), cycle() 호출
 *   - libcuda/: CUDA 런타임 인터셉트에서 GPU 속성 조회
 *   - accelwattch/ (power_interface.h): AccelWattch 전력 모델 연동
 * 공유 자료구조:
 *   - kernel_info_t: 실행 중인 커널 메타데이터 (m_running_kernels[])
 *   - mem_fetch: 메모리 요청 패킷 (mem_fetch.h)
 *   - shader_core_stats, memory_stats_t, power_stat_t: 통계 집계 객체
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpu_sim::cycle()     - 매 사이클 전체 GPU 상태 전이; CORE/ICNT/DRAM/L2
 *                          클럭 도메인을 next_clock_domain()으로 선택해 진행
 * gpgpu_sim::init()      - 커널 실행 전 사이클 카운터·비주얼라이저·ICNT 초기화
 * gpgpu_sim::launch()    - 커널을 m_running_kernels[] 슬롯에 등록
 * gpgpu_sim::active()    - 미완료 warp·DRAM busy·ICNT traffic 여부로 종료 판정
 * gpgpu_sim::select_kernel() - 라운드로빈으로 실행할 커널 선택
 * gpgpu_sim_config       - power/shader/memory/clock 설정 통합; reg_options()로
 *                          gpgpusim.config 파라미터를 OptionParser에 등록
 * memory_config          - DRAM 타이밍(tRCD/tRAS/CL 등), L2 큐, 파티션 수 등
 *                          메모리 서브시스템 전체 설정을 담는 클래스
 * power_config           - AccelWattch XML, 전력 시뮬레이션 모드,
 *                          hybrid 카운터 설정을 담는 구조체
 * occupancy_stats        - warp 슬롯 채움률(점유율) 누적 집계 구조체
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah,
// Nikos Hardavellas Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// 위 내용은 저작권(Copyright) 표시입니다.
// 이 코드를 만든 사람들과 대학교 이름이 적혀 있습니다.
// 오픈소스 소프트웨어이므로 누구나 사용할 수 있지만, 아래 조건을 지켜야 합니다.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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
//
// 위 내용은 BSD 라이선스입니다.
// "이 소프트웨어를 사용해서 문제가 생겨도 만든 사람들은 책임지지 않습니다"라는 뜻입니다.
// 오픈소스 소프트웨어에서 흔히 볼 수 있는 법적 보호 문구입니다.

/*
 * #ifndef / #define / #endif 는 "인클루드 가드(Include Guard)"라고 합니다.
 * 같은 헤더 파일이 여러 번 포함(include)되는 것을 방지합니다.
 *
 * 예를 들어, A.cpp와 B.cpp 모두 이 파일을 #include 하면,
 * 컴파일러가 같은 코드를 두 번 읽게 되어 "이미 정의되었다"는 오류가 생깁니다.
 * 인클루드 가드는 처음 한 번만 내용을 읽고, 두 번째부터는 건너뛰게 합니다.
 */
#ifndef GPU_SIM_H  // GPU_SIM_H가 아직 정의되지 않았다면 아래 내용을 포함
#define GPU_SIM_H  // GPU_SIM_H를 정의하여, 다음에 이 파일이 다시 포함될 때 건너뛰게 함

/*
 * ============================================================================
 * 헤더 파일 포함(Include) 섹션
 * ============================================================================
 * #include는 다른 파일의 내용을 이 파일에 "복사해서 붙여넣기" 하는 것과 같습니다.
 * 마치 레고를 조립할 때 다른 상자에서 부품을 가져오는 것과 비슷합니다.
 */
#include <stdio.h>      // C언어의 표준 입출력 라이브러리 (printf 같은 화면 출력 함수)
#include <fstream>       // C++ 파일 입출력 (파일 읽기/쓰기를 위한 도구)
#include <iostream>      // C++ 표준 입출력 (cout, cin 등 화면 출력/키보드 입력)
#include <list>          // C++ 리스트 자료구조 (데이터를 순서대로 저장하는 연결 리스트)
#include <stdint.h>      // 정수 타입 정의 (uint32_t, uint64_t 등 크기가 정해진 정수형)

/*
 * 아래는 GPGPU-Sim 프로젝트 내부의 다른 헤더 파일들입니다.
 * "../"는 "한 단계 위 폴더"를 의미합니다.
 */
#include "../abstract_hardware_model.h"  // GPU 하드웨어의 추상적 모델 정의 (GPU의 기본 구조)
#include "../option_parser.h"            // 설정 옵션을 읽어오는 파서 (사용자 설정을 처리)
#include "../trace.h"                    // 트레이스(추적) 기능 - 시뮬레이션 과정을 기록
#include "addrdec.h"                     // 주소 디코딩 - 메모리 주소를 분해하여 어디에 접근할지 결정
#include "gpu-cache.h"                   // GPU 캐시 설정 (자주 쓰는 데이터를 빠르게 접근하기 위한 임시 저장소)
#include "shader.h"                      // 셰이더 코어 설정 (GPU의 실제 계산을 수행하는 핵심 유닛)

/*
 * ============================================================================
 * 통계(Statistics) 관련 상수 정의
 * ============================================================================
 * #define은 "매크로(macro)"라고 합니다.
 * 컴파일러가 코드를 번역하기 전에, 이름을 값으로 바꿔치기 합니다.
 *
 * 아래 상수들은 "비트 플래그(bit flag)"입니다.
 * 비트 플래그란? 하나의 숫자에 여러 개의 ON/OFF 스위치를 담는 기법입니다.
 * 예: 0x1 = 0001, 0x2 = 0010, 0x4 = 0100 (2진수)
 * 여러 플래그를 동시에 켜려면 OR(|) 연산을 합니다: 0x1 | 0x4 = 0101
 */

// GPU 런타임 통계 출력을 위한 상수들
#define GPU_RSTAT_SHD_INFO 0x1   // 셰이더(shader) 정보 통계 출력 플래그 (0x1 = 1, 2진수로 0000 0001)
#define GPU_RSTAT_BW_STAT 0x2    // 대역폭(bandwidth) 통계 플래그 (데이터가 얼마나 빨리 이동하는지)
#define GPU_RSTAT_WARP_DIS 0x4   // 워프(warp) 분배 통계 플래그 (워프 = GPU에서 동시에 실행되는 스레드 묶음, 보통 32개)
#define GPU_RSTAT_DWF_MAP 0x8    // DWF(Dynamic Warp Formation) 맵 통계 플래그
#define GPU_RSTAT_L1MISS 0x10    // L1 캐시 미스(miss) 통계 플래그 (캐시에서 원하는 데이터를 못 찾은 횟수)
#define GPU_RSTAT_PDOM 0x20      // PDOM(Post-Dominator) 통계 플래그 (분기(if/else) 처리 관련)
#define GPU_RSTAT_SCHED 0x40     // 스케줄러 통계 플래그 (어떤 작업을 언제 실행할지 결정하는 장치)
#define GPU_MEMLATSTAT_MC 0x2    // 메모리 컨트롤러 지연시간(latency) 통계 플래그

/*
 * ============================================================================
 * MSHR 병합(Merge) 관련 상수
 * ============================================================================
 * MSHR(Miss Status Holding Register)이란?
 * 캐시에서 데이터를 못 찾았을 때(캐시 미스), 그 요청을 기록해두는 장치입니다.
 * 같은 데이터를 여러 번 요청하면, 하나로 합쳐서(merge) 메모리에 한 번만 요청합니다.
 * 이렇게 하면 메모리 접근 횟수가 줄어서 성능이 좋아집니다.
 *
 * Scatter-Gather란? 흩어진(scatter) 데이터를 모으는(gather) 메모리 접근 패턴입니다.
 * GPU는 여러 스레드가 동시에 메모리를 접근하므로, 비슷한 주소의 요청을 합치는 것이 중요합니다.
 */
#define TEX_MSHR_MERGE 0x4      // 텍스처(texture) 메모리 요청 병합 플래그 (텍스처 = 이미지 데이터)
#define CONST_MSHR_MERGE 0x2    // 상수(constant) 메모리 요청 병합 플래그 (변하지 않는 데이터)
#define GLOBAL_MSHR_MERGE 0x1   // 전역(global) 메모리 요청 병합 플래그 (모든 스레드가 접근 가능한 메모리)

/*
 * 클럭(clock) 주파수 단위 변환 매크로
 * MhZ는 메가헤르츠(MHz)를 의미합니다.
 * 1 MHz = 1,000,000 Hz (1초에 1백만 번 진동)
 * 예: 1000 MhZ 라고 쓰면 1000 * 1000000 = 1,000,000,000 (1GHz)이 됩니다.
 *
 * 클럭이란? 컴퓨터의 심장 박동 같은 것으로,
 * 한 번 "틱"할 때마다 하나의 작업을 처리합니다.
 * 클럭이 빠를수록 컴퓨터가 더 빠르게 동작합니다.
 */
#define MhZ *1000000  // 숫자 뒤에 MhZ를 붙이면 백만을 곱하는 매크로

/*
 * 로깅(logging) 관련 상수
 * 로그(log)란 프로그램이 실행되면서 일어나는 일들을 기록한 것입니다.
 * 마치 일기장처럼, 나중에 무엇이 잘못되었는지 찾을 때 유용합니다.
 */
#define CREATELOG 111   // 로그 파일 생성 명령을 나타내는 상수
#define SAMPLELOG 222   // 로그에 샘플 데이터 기록 명령을 나타내는 상수
#define DUMPLOG 333     // 로그 데이터를 파일에 출력(dump) 명령을 나타내는 상수

/*
 * 전방 선언(Forward Declaration)
 * 클래스의 자세한 내용은 나중에 정의하겠다고 컴파일러에게 미리 알려주는 것입니다.
 * "이런 이름의 클래스가 존재한다"고만 말해두면,
 * 해당 클래스의 포인터나 참조를 먼저 사용할 수 있습니다.
 */
class gpgpu_context;  // GPGPU 시뮬레이터의 전체 문맥(context)을 담는 클래스 (나중에 정의됨)

/*
 * extern 키워드는 "이 변수는 다른 파일에서 정의되어 있다"는 뜻입니다.
 * 여러 파일에서 같은 변수를 공유할 때 사용합니다.
 *
 * tr1_hash_map은 해시맵(hash map) 자료구조입니다.
 * 해시맵이란? 열쇠(key)로 값(value)을 빠르게 찾을 수 있는 사전 같은 자료구조입니다.
 * 여기서는 메모리 주소(key)를 랜덤하게 섞어서 고르게 분배하기 위한 매핑 테이블입니다.
 *
 * 왜 주소를 랜덤하게 섞을까? 메모리 접근이 한 곳에 몰리면 병목이 생기므로,
 * 여러 메모리 채널에 고르게 분산시키기 위해서입니다.
 */
extern tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/*
 * ============================================================================
 * SST(Structural Simulation Toolkit) 통신 함수들
 * ============================================================================
 * SST란? 컴퓨터 시스템을 시뮬레이션하는 또 다른 도구입니다.
 * GPGPU-Sim과 SST를 연결하면, GPU와 메모리 시스템을 더 정확하게 시뮬레이션할 수 있습니다.
 *
 * __attribute__((weak))란?
 * "약한 심볼(weak symbol)"이라는 뜻으로, 다른 곳에서 같은 이름의 함수를 정의하면
 * 그 함수가 우선 사용됩니다. 여기에 있는 것은 "기본값(default)"입니다.
 * SST를 사용하지 않을 때는 이 기본(빈) 함수가 사용되고,
 * SST를 사용할 때는 SST에서 제공하는 진짜 함수로 대체됩니다.
 */

/**
 * @brief SST 요청 버퍼가 가득 찼는지 확인하는 함수
 *
 * @param core_id 어떤 GPU 코어(core)가 요청했는지 식별하는 번호
 * @return true 버퍼가 가득 참 (더 이상 요청을 보낼 수 없음)
 * @return false 버퍼에 여유가 있음 (요청을 보낼 수 있음)
 */
extern bool is_SST_buffer_full(unsigned core_id);  // 함수 선언 (다른 파일에 구현이 있을 수 있음)
// 아래는 약한(weak) 기본 구현 - SST를 안 쓰면 항상 false(버퍼가 안 찼다)를 반환
__attribute__((weak)) bool is_SST_buffer_full(unsigned core_id) {
  return false;  // 기본적으로 버퍼가 가득 차지 않았다고 반환
}

/**
 * @brief SST 메모리 백엔드(뒷단)로 읽기 요청을 보내는 함수
 *
 * 메모리에서 데이터를 "읽어오기(load)" 위한 요청을 SST 시스템에 전달합니다.
 *
 * @param core_id   요청을 보낸 GPU 코어 번호
 * @param address   읽고 싶은 메모리 주소 (uint64_t = 64비트 양의 정수, 매우 큰 주소 표현 가능)
 * @param size      읽고 싶은 데이터의 크기 (바이트 단위)
 * @param mem_req   메모리 요청 객체에 대한 포인터 (void* = 아무 타입이나 가리킬 수 있는 범용 포인터)
 */
extern void send_read_request_SST(unsigned core_id, uint64_t address,
                                  size_t size, void *mem_req);
// 약한 기본 구현 - SST를 안 쓰면 아무것도 하지 않음 (빈 함수)
__attribute__((weak)) void send_read_request_SST(unsigned core_id,
                                                 uint64_t address, size_t size,
                                                 void *mem_req) {}

/**
 * @brief SST 메모리 백엔드로 쓰기 요청을 보내는 함수
 *
 * 메모리에 데이터를 "저장하기(store)" 위한 요청을 SST 시스템에 전달합니다.
 *
 * @param core_id   요청을 보낸 GPU 코어 번호
 * @param address   쓰고 싶은 메모리 주소
 * @param size      쓰고 싶은 데이터의 크기 (바이트 단위)
 * @param mem_req   메모리 요청 객체에 대한 포인터
 */
extern void send_write_request_SST(unsigned core_id, uint64_t address,
                                   size_t size, void *mem_req);
// 약한 기본 구현 - SST를 안 쓰면 아무것도 하지 않음
__attribute__((weak)) void send_write_request_SST(unsigned core_id,
                                                  uint64_t address, size_t size,
                                                  void *mem_req) {}

/*
 * ============================================================================
 * DRAM 컨트롤러 타입 열거형 (enum)
 * ============================================================================
 * enum(열거형)이란? 관련된 상수들을 하나의 그룹으로 묶는 방법입니다.
 * 숫자 대신 이름을 사용할 수 있어서 코드를 읽기 쉽게 만듭니다.
 *
 * DRAM(Dynamic Random Access Memory)이란?
 * 컴퓨터의 주 메모리(RAM)로, 데이터를 임시로 저장하는 장치입니다.
 * GPU도 자체 DRAM(비디오 메모리, VRAM)을 가지고 있습니다.
 *
 * DRAM 컨트롤러(controller)란?
 * DRAM에 대한 읽기/쓰기 요청을 관리하는 장치입니다.
 * 여러 요청이 동시에 오면, 어떤 순서로 처리할지 결정합니다.
 */
enum dram_ctrl_t {
  DRAM_FIFO = 0,    // FIFO(First In, First Out) 방식 - 먼저 온 요청을 먼저 처리 (줄 서기와 같음)
  DRAM_FRFCFS = 1   // FR-FCFS(First Ready, First Come First Served) 방식
                     // 준비된 요청(같은 행에 있는)을 우선 처리하여 성능을 높임
                     // 이 방식이 실제 GPU에서 더 많이 사용됨
};

/*
 * ============================================================================
 * 하드웨어 성능 카운터 열거형
 * ============================================================================
 * 이 enum은 실제 GPU 하드웨어의 성능 카운터(performance counter)를 나타냅니다.
 * 성능 카운터란? GPU가 실행되면서 다양한 이벤트(캐시 히트, 미스 등)를
 * 세는 카운터입니다. 시뮬레이터가 실제 하드웨어와 비교할 때 사용합니다.
 *
 * 캐시 히트(hit)란? 캐시에서 원하는 데이터를 찾은 것 (빠름!)
 * 캐시 미스(miss)란? 캐시에서 데이터를 못 찾아서 느린 메모리에서 가져와야 하는 것 (느림!)
 */
enum hw_perf_t {
  HW_BENCH_NAME = 0,  // 벤치마크 이름 (테스트 프로그램의 이름)
  HW_KERNEL_NAME,     // 커널(kernel) 이름 (GPU에서 실행되는 함수의 이름)
  HW_L1_RH,           // L1 캐시 읽기 히트(Read Hit) 횟수 (L1 = CPU/GPU에서 가장 빠른 1차 캐시)
  HW_L1_RM,           // L1 캐시 읽기 미스(Read Miss) 횟수
  HW_L1_WH,           // L1 캐시 쓰기 히트(Write Hit) 횟수
  HW_L1_WM,           // L1 캐시 쓰기 미스(Write Miss) 횟수
  HW_CC_ACC,          // 상수 캐시(Constant Cache) 접근 횟수
  HW_SHRD_ACC,        // 공유 메모리(Shared Memory) 접근 횟수 (같은 블록의 스레드들이 공유하는 빠른 메모리)
  HW_DRAM_RD,         // DRAM 읽기 횟수
  HW_DRAM_WR,         // DRAM 쓰기 횟수
  HW_L2_RH,           // L2 캐시 읽기 히트 횟수 (L2 = L1보다 크지만 느린 2차 캐시)
  HW_L2_RM,           // L2 캐시 읽기 미스 횟수
  HW_L2_WH,           // L2 캐시 쓰기 히트 횟수
  HW_L2_WM,           // L2 캐시 쓰기 미스 횟수
  HW_NOC,             // NoC(Network on Chip) 트래픽 - 칩 내부의 통신 네트워크 사용량
  HW_PIPE_DUTY,       // 파이프라인 활용률 (파이프라인 = 여러 작업을 단계별로 겹쳐서 처리하는 구조)
  HW_NUM_SM_IDLE,     // 유휴(idle) SM 수 (SM = Streaming Multiprocessor, GPU의 계산 유닛)
  HW_CYCLES,          // 실행 사이클 수 (사이클 = 클럭의 한 박자, 모든 동작의 기본 시간 단위)
  HW_VOLTAGE,         // 전압 (GPU에 공급되는 전기의 세기)
  HW_TOTAL_STATS      // 전체 통계 항목의 수 (배열 크기를 정할 때 사용)
};

/*
 * ============================================================================
 * 전력(Power) 설정 구조체
 * ============================================================================
 * struct(구조체)란? 관련된 데이터를 하나로 묶는 방법입니다.
 * class와 비슷하지만, struct는 기본적으로 모든 멤버가 공개(public)입니다.
 *
 * 이 구조체는 GPU의 전력 소비를 시뮬레이션하기 위한 설정을 담고 있습니다.
 * 전력(Power)이란? GPU가 얼마나 많은 전기를 사용하는지를 나타냅니다.
 * GPU의 전력 소비를 줄이는 것은 배터리 수명과 발열에 직접적으로 영향을 줍니다.
 *
 * AccelWattch는 GPGPU-Sim에서 사용하는 전력 모델의 이름입니다.
 */
struct power_config {
  // 생성자(Constructor) - 구조체가 만들어질 때 자동으로 호출됩니다
  // m_valid를 true로 초기화하여 설정이 유효하다고 표시합니다
  power_config() { m_valid = true; }

  /*
   * init() 함수 - 전력 설정을 초기화(시작할 때 준비)하는 함수
   * 로그 파일 이름을 현재 날짜/시간으로 생성하고,
   * 정상 상태(steady state) 전력 추적 옵션을 설정합니다.
   */
  void init() {
    // initialize file name if it is not set
    // 현재 시간을 가져와서 로그 파일 이름에 사용합니다
    time_t curr_time;          // 시간을 저장할 변수
    time(&curr_time);          // 현재 시간을 가져옴
    char *date = ctime(&curr_time);  // 시간을 읽기 쉬운 문자열로 변환 (예: "Mon Jan 01 12:00:00 2024")
    char *s = date;            // 문자열을 순회하기 위한 포인터

    // 파일 이름에 쓸 수 없는 문자(공백, 탭, 콜론)를 하이픈(-)으로 바꿈
    // 파일 이름에 공백이나 콜론이 있으면 운영체제에서 문제가 생길 수 있기 때문
    while (*s) {  // 문자열 끝(null 문자)까지 반복
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';  // 공백, 탭, 콜론을 '-'로 교체
      if (*s == '\n' || *s == '\r') *s = 0;  // 줄바꿈 문자를 문자열 끝(null)으로 변환
      s++;  // 다음 문자로 이동
    }

    // 전력 보고서 파일 이름 생성
    char buf1[1024];  // 1024바이트 크기의 문자 배열(버퍼)
    // snprintf는 형식에 맞춰 문자열을 만드는 안전한 함수 (버퍼 크기를 넘지 않게 보장)
    // snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);  // 원래는 날짜 포함
    snprintf(buf1, 1024, "accelwattch_power_report.log");  // 지금은 고정 이름 사용
    g_power_filename = strdup(buf1);  // strdup = 문자열을 복사하여 새 메모리에 저장

    // 전력 트레이스(추적) 파일 이름 생성 (.gz = gzip 압축 파일)
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    g_power_trace_filename = strdup(buf2);

    // 메트릭(측정값) 트레이스 파일 이름 생성
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    g_metric_trace_filename = strdup(buf3);

    // 정상 상태 추적(steady state tracking) 파일 이름 생성
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    g_steady_state_tracking_filename = strdup(buf4);

    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }
    // 위 주석 처리된 코드는 하이브리드 설정 배열을 0으로 초기화하려던 코드입니다

    // 정상 상태 전력 레벨 기능이 켜져 있다면, 편차와 최소 기간을 파싱(분석)합니다
    // 정상 상태(steady state)란? 전력이 일정하게 유지되는 상태를 말합니다
    if (g_steady_power_levels_enabled) {
      // sscanf는 문자열에서 값을 읽어오는 함수
      // "%lf:%lf"는 "소수점 숫자:소수점 숫자" 형식을 의미
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation,  // 허용 편차 (얼마나 변동해도 "정상"으로 볼 것인지)
             &gpu_steady_min_period);       // 최소 기간 (얼마나 오래 유지되어야 "정상"인지)
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    // 비선형(nonlinear) 모델 관련 코드가 주석 처리되어 있습니다.
    // 비선형 모델이란? 전력이 코어 수에 비례하지 않고, 복잡한 관계를 가지는 모델입니다.
    // if (g_use_nonlinear_model)
    //   sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
    //          &gpu_min_inc_per_active_sm);
  }

  // 옵션 파서에 전력 관련 설정 옵션을 등록하는 함수
  // 사용자가 설정 파일에서 전력 관련 값을 지정할 수 있게 합니다
  void reg_options(class OptionParser *opp);

  // === 멤버 변수(member variables) - 전력 설정 값들을 저장 ===

  char *g_power_config_name;  // 전력 설정 파일의 이름

  bool m_valid;                          // 이 설정이 유효한지 여부 (true = 유효, false = 무효)
  bool g_power_simulation_enabled;       // 전력 시뮬레이션 활성화 여부
  bool g_power_trace_enabled;            // 전력 트레이스(추적) 기록 활성화 여부
  bool g_steady_power_levels_enabled;    // 정상 상태 전력 레벨 추적 활성화 여부
  bool g_power_per_cycle_dump;           // 매 사이클마다 전력 정보를 출력할지 여부
  bool g_power_simulator_debug;          // 전력 시뮬레이터 디버그 모드 활성화 여부
  char *g_power_filename;                // 전력 보고서 파일 이름
  char *g_power_trace_filename;          // 전력 트레이스 파일 이름
  char *g_metric_trace_filename;         // 메트릭 트레이스 파일 이름
  char *g_steady_state_tracking_filename;// 정상 상태 추적 파일 이름
  int g_power_trace_zlevel;              // 트레이스 파일 압축 수준 (0 = 압축 안함, 9 = 최대 압축)
  char *gpu_steady_state_definition;     // 정상 상태 정의 문자열 (편차:최소기간 형식)
  double gpu_steady_power_deviation;     // 정상 상태로 간주하는 전력 편차 허용 범위
  double gpu_steady_min_period;          // 정상 상태로 간주하는 최소 유지 기간

  char *g_hw_perf_file_name;             // 하드웨어 성능 파일 이름 (실제 GPU 측정 데이터)
  char *g_hw_perf_bench_name;            // 하드웨어 성능 벤치마크 이름
  int g_power_simulation_mode;           // 전력 시뮬레이션 모드 (어떤 방법으로 전력을 계산할지)
  bool g_dvfs_enabled;                   // DVFS(Dynamic Voltage and Frequency Scaling) 활성화 여부
                                         // DVFS란? 필요에 따라 전압과 주파수를 조절하여 전력을 절약하는 기술
  bool g_aggregate_power_stats;          // 전력 통계를 누적할지 여부

  // AccelWattch 하이브리드 설정 배열
  // 각 성능 카운터에 대해 시뮬레이션 값을 사용할지, 실제 하드웨어 값을 사용할지 결정
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

  // === 비선형 전력 모델 관련 변수들 ===
  // 비선형 모델은 GPU 코어가 많아질수록 전력이 단순히 비례하지 않는 현상을 모델링
  bool g_use_nonlinear_model;            // 비선형 전력 모델 사용 여부
  char *gpu_nonlinear_model_config;      // 비선형 모델 설정 문자열
  double gpu_idle_core_power;            // 유휴(idle) 코어의 전력 소비량 (아무 일도 안 할 때 쓰는 전기)
  double gpu_min_inc_per_active_sm;      // 활성 SM 하나당 최소 전력 증가량
};

/*
 * ============================================================================
 * 메모리 설정 클래스
 * ============================================================================
 * 이 클래스는 GPU의 메모리 시스템 전체를 설정합니다.
 * GPU 메모리 시스템은 매우 복잡하며, 여러 단계의 캐시와 DRAM으로 구성됩니다:
 *
 * [GPU 코어] → [L1 캐시] → [L2 캐시] → [DRAM(메인 메모리)]
 *    빠름          ↓           ↓              느림
 *    작음        중간         중간             큼
 *
 * DRAM은 여러 개의 파티션(partition, 구역)으로 나뉘어 있고,
 * 각 파티션은 다시 여러 개의 뱅크(bank, 저장 칸)로 나뉩니다.
 * 이렇게 나누는 이유는 여러 곳에서 동시에 데이터를 읽고 쓸 수 있게 하기 위해서입니다.
 */
class memory_config {
 public:
  /*
   * 생성자 - memory_config 객체가 만들어질 때 호출됩니다.
   * gpgpu_context 포인터를 받아 시뮬레이터 전체 문맥에 접근할 수 있게 합니다.
   * 처음에는 설정이 유효하지 않은(m_valid = false) 상태로 시작합니다.
   */
  memory_config(gpgpu_context *ctx) {
    m_valid = false;                   // 아직 초기화되지 않았으므로 무효
    gpgpu_dram_timing_opt = NULL;      // DRAM 타이밍 옵션 문자열을 NULL(없음)로 초기화
    gpgpu_L2_queue_config = NULL;      // L2 캐시 큐 설정을 NULL로 초기화
    gpgpu_ctx = ctx;                   // 시뮬레이터 전체 문맥에 대한 포인터 저장
  }

  /*
   * init() 함수 - 메모리 설정을 초기화하는 핵심 함수
   *
   * DRAM 타이밍(timing)이란?
   * DRAM이 데이터를 읽고 쓰는 데 걸리는 시간을 나타내는 여러 매개변수입니다.
   * 실제 DRAM 칩의 데이터시트(설명서)에 적혀 있는 값들입니다.
   * 이 값들을 정확하게 설정해야 시뮬레이션이 실제 하드웨어와 비슷하게 동작합니다.
   */
  void init() {
    // DRAM 타이밍 옵션이 반드시 설정되어 있어야 합니다
    assert(gpgpu_dram_timing_opt);  // assert = 조건이 false이면 프로그램을 멈추고 오류를 알림

    // '=' 문자가 없으면 레거시(옛날) 형식으로 파싱
    if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
      // dram timing option in ordered variables (legacy)
      // 레거시 형식: 값들이 콜론(:)으로 구분된 순서대로 나열됨
      // Disabling bank groups if their values are not specified
      // 뱅크 그룹이 지정되지 않으면 비활성화 (기본값 설정)
      nbkgrp = 1;    // 뱅크 그룹 수를 1로 (그룹 없음과 같은 효과)
      tCCDL = 0;     // 그룹 간 열-열 지연을 0으로
      tRTPL = 0;     // 그룹 간 읽기-프리차지 지연을 0으로

      // sscanf로 14개의 DRAM 타이밍 파라미터를 순서대로 읽어옴
      sscanf(gpgpu_dram_timing_opt, "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk,      // 뱅크(bank) 수
             &tCCD,     // CCD(Column to Column Delay) - 같은 뱅크에서 연속 읽기 간 지연
             &tRRD,     // RRD - 다른 뱅크의 행 활성화 사이 최소 지연
             &tRCD,     // RCD(Row to Column Delay) - 행 활성화 후 열 접근까지 걸리는 시간
             &tRAS,     // RAS - 행 활성화에 필요한 시간
             &tRP,      // RP(Row Precharge) - 행 비활성화(프리차지)에 필요한 시간
             &tRC,      // RC(Row Cycle) - 한 행 사이클 전체 시간
             &CL,       // CL(CAS Latency) - 읽기 명령 후 데이터가 나올 때까지의 지연
             &WL,       // WL(Write Latency) - 쓰기 명령 후 데이터를 받을 때까지의 지연
             &tCDLR,    // CDLR - 쓰기에서 읽기로 전환 시 지연
             &tWR,      // WR(Write Recovery) - 마지막 데이터 입력 후 프리차지까지 지연
             &nbkgrp,   // 뱅크 그룹 수
             &tCCDL,    // CCDL - 다른 뱅크 그룹 간 열-열 지연
             &tRTPL);   // RTPL - 다른 뱅크 그룹 간 읽기-프리차지 지연
    } else {
      // named dram timing options (unordered)
      // 새로운 형식: 이름=값 쌍으로 지정 (순서 무관)
      // 이 형식이 더 읽기 쉽고 유연합니다
      option_parser_t dram_opp = option_parser_create();  // 옵션 파서 생성

      // 각 DRAM 타이밍 파라미터를 옵션으로 등록
      option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
                             "number of banks", "");  // 뱅크 수
      option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
                             "column to column delay", "");  // 열-열 지연
      option_parser_register(
          dram_opp, "RRD", OPT_UINT32, &tRRD,
          "minimal delay between activation of rows in different banks", "");  // 다른 뱅크 행 활성화 간 최소 지연
      option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
                             "row to column delay", "");  // 행-열 지연
      option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
                             "time needed to activate row", "");  // 행 활성화 시간
      option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
                             "time needed to precharge (deactivate) row", "");  // 행 프리차지(비활성화) 시간
      option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle time",
                             "");  // 행 사이클 시간
      option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
                             "switching from write to read (changes tWTR)", "");  // 쓰기→읽기 전환 시간
      option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
                             "last data-in to row precharge", "");  // 마지막 데이터 입력 후 프리차지까지 시간

      option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
                             "");  // CAS 지연시간
      option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency",
                             "");  // 쓰기 지연시간

      // Disabling bank groups if their values are not specified
      // 뱅크 그룹 관련 - 기본값이 있어서 지정하지 않으면 비활성화됨
      option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
                             "number of bank groups", "1");  // 뱅크 그룹 수 (기본값: 1)
      option_parser_register(
          dram_opp, "CCDL", OPT_UINT32, &tCCDL,
          "column to column delay between accesses to different bank groups",
          "0");  // 다른 뱅크 그룹 간 열-열 지연 (기본값: 0)
      option_parser_register(
          dram_opp, "RTPL", OPT_UINT32, &tRTPL,
          "read to precharge delay between accesses to different bank groups",
          "0");  // 다른 뱅크 그룹 간 읽기-프리차지 지연 (기본값: 0)

      // 설정 문자열을 파싱 (=, :, ; 를 구분자로 사용)
      option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
      fprintf(stdout, "DRAM Timing Options:\n");  // 파싱 결과를 화면에 출력
      option_parser_print(dram_opp, stdout);       // 모든 옵션 값을 출력
      option_parser_destroy(dram_opp);             // 파서 메모리 해제
    }

    /*
     * 뱅크 태그 길이 계산
     * 뱅크 그룹 내에서 뱅크를 구분하기 위해 필요한 비트 수를 계산합니다.
     * 예: 뱅크 그룹당 4개의 뱅크 → 2비트 필요 (00, 01, 10, 11)
     * 비트 시프트(>> 1)를 반복하여 2의 로그(log2)를 구합니다.
     */
    int nbkt = nbk / nbkgrp;  // 뱅크 그룹당 뱅크 수
    unsigned i;
    for (i = 0; nbkt > 0; i++) {  // 값이 0이 될 때까지 오른쪽으로 1비트씩 이동
      nbkt = nbkt >> 1;           // >> 1은 2로 나누는 것과 같음
    }
    bk_tag_length = i - 1;  // 필요한 비트 수 저장

    // 뱅크 그룹 수가 0이면 안 됨 - 안전 검사
    assert(nbkgrp > 0 && "Number of bank groups cannot be zero");

    /*
     * 파생 타이밍 파라미터 계산
     * 기본 타이밍 값들로부터 추가적인 타이밍 값들을 계산합니다.
     */
    tRCDWR = tRCD - (WL + 1);  // 쓰기 명령을 위한 행-열 지연 계산

    // 읽기/쓰기 전환 턴어라운드(turnaround) 제거 옵션
    // 턴어라운드 타임이란? 읽기와 쓰기를 번갈아 할 때 필요한 대기 시간
    if (elimnate_rw_turnaround) {
      tRTW = 0;  // 읽기→쓰기 전환 시간을 0으로 (이상적인 경우)
      tWTR = 0;  // 쓰기→읽기 전환 시간을 0으로
    } else {
      // 실제 DRAM 사양에 따른 전환 시간 계산
      // BL = Burst Length(한 번에 전송하는 데이터 양)
      // data_command_freq_ratio = 데이터 버스와 명령 버스의 주파수 비율
      tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);  // 읽기→쓰기 전환 시간
      tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);    // 쓰기→읽기 전환 시간
    }
    tWTP = (WL + (BL / data_command_freq_ratio) + tWR);  // 쓰기→프리차지 전환 시간

    // DRAM 원자 크기(atom size) 계산
    // 한 번의 읽기/쓰기 명령으로 전송되는 총 바이트 수
    // = 버스트 길이 × 버스 폭 × 컨트롤러당 메모리 칩 수
    dram_atom_size =
        BL * busW * gpu_n_mem_per_ctrlr;  // burst length x bus width x # chips
                                          // per partition

    // 메모리 서브 파티션 수가 유효한지 검사
    assert(m_n_sub_partition_per_memory_channel > 0);
    // DRAM 뱅크 수가 서브 파티션 수의 배수인지 검사
    // (균등하게 나눌 수 있어야 하므로)
    assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
           "Number of DRAM banks must be a perfect multiple of memory sub "
           "partition");

    // 전체 메모리 서브 파티션 수 계산
    // = 메모리 채널 수 × 채널당 서브 파티션 수
    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    fprintf(stdout, "Total number of memory sub partition = %u\n",
            m_n_mem_sub_partition);  // 결과를 화면에 출력

    // 주소 매핑(address mapping) 초기화
    // 메모리 주소를 실제 DRAM의 채널/뱅크/행/열로 변환하는 방법을 설정
    m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);

    // L2 캐시 설정 초기화 (주소 매핑 정보를 사용)
    m_L2_config.init(&m_address_mapping);

    m_valid = true;  // 모든 초기화가 끝났으므로 설정이 유효하다고 표시

    // 쓰기 큐(queue) 크기 옵션 파싱
    // 형식: "큐크기:상한워터마크:하한워터마크"
    // 워터마크(watermark)란? 큐가 이 수준에 도달하면 특별한 동작을 하는 기준점
    // 상한(high): 이 이상이면 쓰기 우선 모드로 전환
    // 하한(low): 이 이하로 내려오면 다시 일반 모드로 복귀
    sscanf(write_queue_size_opt, "%d:%d:%d",
           &gpgpu_frfcfs_dram_write_queue_size,  // 쓰기 큐의 최대 크기
           &write_high_watermark,                 // 상한 워터마크
           &write_low_watermark);                 // 하한 워터마크
  }

  // 옵션 파서에 메모리 관련 설정 옵션을 등록하는 함수
  void reg_options(class OptionParser *opp);

  /**
   * @brief SST 모드인지 확인하는 함수
   * SST(Structural Simulation Toolkit) 모드에서는 외부 메모리 시뮬레이터를 사용합니다
   * const 키워드는 "이 함수가 객체의 상태를 변경하지 않는다"는 약속입니다
   */
  bool is_SST_mode() const { return SST_mode; }

  // === 멤버 변수들 - 메모리 시스템 설정값 저장 ===

  bool m_valid;                         // 이 설정이 유효한지 여부
  mutable l2_cache_config m_L2_config;  // L2 캐시 설정 (mutable = const 함수에서도 수정 가능)
  bool m_L2_texure_only;                // L2 캐시를 텍스처 전용으로 사용할지 여부

  char *gpgpu_dram_timing_opt;          // DRAM 타이밍 옵션 문자열 (사용자 입력)
  char *gpgpu_L2_queue_config;          // L2 캐시 큐 설정 문자열
  bool l2_ideal;                        // L2 캐시를 이상적(항상 히트)으로 시뮬레이션할지 여부
  unsigned gpgpu_frfcfs_dram_sched_queue_size;  // FR-FCFS 스케줄러의 큐 크기
  unsigned gpgpu_dram_return_queue_size;        // DRAM 반환 큐 크기 (처리된 요청이 돌아오는 대기열)
  enum dram_ctrl_t scheduler_type;              // DRAM 스케줄러 타입 (FIFO 또는 FR-FCFS)
  bool gpgpu_memlatency_stat;                   // 메모리 지연시간 통계 수집 여부
  unsigned m_n_mem;                             // 메모리 채널(파티션) 수
  unsigned m_n_sub_partition_per_memory_channel; // 채널당 서브 파티션 수
  unsigned m_n_mem_sub_partition;                // 전체 서브 파티션 수 (= 채널 수 × 채널당 서브 파티션)
  unsigned gpu_n_mem_per_ctrlr;                 // 컨트롤러당 메모리 칩 수

  unsigned rop_latency;   // ROP(Raster Operation Processor) 지연시간 - 그래픽 처리의 마지막 단계
  unsigned dram_latency;  // DRAM 접근 지연시간 (DRAM에서 데이터를 가져오는 데 걸리는 시간)

  // === DRAM 타이밍 파라미터들 ===
  // 이 값들은 실제 DRAM 칩의 사양서(datasheet)에서 가져온 것입니다.
  // 각 값의 단위는 DRAM 클럭 사이클입니다.

  unsigned tCCDL;  // 뱅크 그룹이 활성화되었을 때 열-열 지연
                   // (같은 뱅크 그룹 내 다른 뱅크에 연속 접근 시 필요한 대기 시간)
  unsigned tRTPL;  // 뱅크 그룹이 활성화되었을 때 읽기-프리차지 지연
                   // GDDR5에서는 RTPS와 동일하지만, 다른 DRAM에서는 다를 수 있음

  unsigned tCCD;    // 열-열 지연 (같은 뱅크 내 연속 열 접근 간 최소 시간)
  unsigned tRRD;    // 다른 뱅크의 행 활성화 사이 최소 시간
  unsigned tRCD;    // 행-열 지연 - 행을 활성화한 후 읽기를 시작하기까지 필요한 시간
  unsigned tRCDWR;  // 쓰기 명령을 위한 행-열 지연
  unsigned tRAS;    // 행 활성화에 필요한 시간
  unsigned tRP;     // 행 프리차지(비활성화)에 필요한 시간
  unsigned
      tRC;  // 행 사이클 시간 (현재 행 프리차지 후 다른 행 활성화까지의 전체 시간)
  unsigned tCDLR;  // 쓰기에서 읽기로 전환할 때 필요한 지연 시간
  unsigned tWR;    // 마지막 데이터 입력부터 행 프리차지까지의 시간

  unsigned CL;    // CAS 지연시간 (읽기 명령 후 데이터가 나올 때까지 걸리는 클럭 수)
  unsigned WL;    // 쓰기 지연시간 (쓰기 명령 후 데이터를 보낼 수 있을 때까지 걸리는 클럭 수)
  unsigned BL;    // 버스트 길이(Burst Length) - 한 번에 전송하는 데이터 바이트 수
                  // GDDR3은 4, GDDR5는 8
  unsigned tRTW;  // 읽기→쓰기 전환에 필요한 시간
  unsigned tWTR;  // 쓰기→읽기 전환에 필요한 시간
  unsigned tWTP;  // 같은 뱅크에서 쓰기→프리차지 전환에 필요한 시간
  unsigned busW;  // 버스 폭(bus width) - 데이터 버스의 너비 (바이트 단위)

  unsigned nbkgrp;  // 뱅크 그룹 수 (2의 거듭제곱이어야 함: 1, 2, 4, 8...)
  unsigned
      bk_tag_length;  // 뱅크 그룹 내에서 뱅크를 식별하는 데 필요한 비트 수

  unsigned nbk;  // 전체 뱅크(bank) 수

  bool elimnate_rw_turnaround;  // 읽기/쓰기 전환 지연을 제거할지 여부 (이상적 시뮬레이션용)

  unsigned
      data_command_freq_ratio;  // DRAM 데이터 버스와 명령 버스의 주파수 비율
                                // GDDR3은 2, GDDR5는 4
  unsigned
      dram_atom_size;  // 한 번의 읽기/쓰기 명령으로 전송되는 바이트 수

  // 주소 변환 객체 - 선형 주소를 DRAM의 채널/뱅크/행/열 주소로 변환
  linear_to_raw_address_translation m_address_mapping;

  unsigned icnt_flit_size;  // 인터커넥트(interconnect) 네트워크의 flit 크기
                            // flit = 네트워크에서 한 번에 전송하는 데이터 단위

  unsigned dram_bnk_indexing_policy;      // DRAM 뱅크 인덱싱 정책 (주소→뱅크 매핑 방법)
  unsigned dram_bnkgrp_indexing_policy;   // DRAM 뱅크 그룹 인덱싱 정책
  bool dual_bus_interface;                // 듀얼 버스 인터페이스 사용 여부 (읽기/쓰기 버스 분리)

  bool seperate_write_queue_enabled;      // 별도 쓰기 큐 활성화 여부
                                          // (읽기와 쓰기 요청을 분리하여 관리)
  char *write_queue_size_opt;             // 쓰기 큐 크기 옵션 문자열
  unsigned gpgpu_frfcfs_dram_write_queue_size;  // FR-FCFS 스케줄러의 쓰기 큐 크기
  unsigned write_high_watermark;          // 쓰기 큐 상한 워터마크
  unsigned write_low_watermark;           // 쓰기 큐 하한 워터마크
  bool m_perf_sim_memcpy;                 // 메모리 복사(memcpy) 성능 시뮬레이션 여부
  bool simple_dram_model;                 // 단순 DRAM 모델 사용 여부 (빠르지만 덜 정확)
  bool SST_mode;                          // SST 모드 여부 (외부 메모리 시뮬레이터 사용)
  gpgpu_context *gpgpu_ctx;              // 시뮬레이터 전체 문맥에 대한 포인터
};

/*
 * extern 변수 - 인터랙티브 디버거 활성화 여부
 * 디버거(debugger)란? 프로그램의 버그(오류)를 찾기 위해
 * 실행을 멈추고 상태를 확인할 수 있는 도구입니다.
 * 인터랙티브(interactive)는 사용자가 직접 조작할 수 있다는 뜻입니다.
 */
extern bool g_interactive_debugger_enabled;

/*
 * ============================================================================
 * GPU 시뮬레이션 설정 클래스
 * ============================================================================
 * 이 클래스는 GPU 시뮬레이터의 전체 설정을 하나로 모읍니다.
 *
 * 상속(Inheritance):
 * - power_config: 전력 설정을 물려받음
 * - gpgpu_functional_sim_config: 기능적(functional) 시뮬레이션 설정을 물려받음
 *
 * 상속이란? 부모 클래스의 모든 기능을 자식 클래스가 그대로 가져가는 것입니다.
 * 마치 부모님의 재산을 물려받는 것과 비슷합니다.
 * 이렇게 하면 코드를 중복 없이 재사용할 수 있습니다.
 *
 * 이 클래스 안에는 두 개의 중요한 설정 객체가 들어 있습니다:
 * - m_shader_config: 셰이더 코어(계산 유닛) 설정
 * - m_memory_config: 메모리 시스템 설정
 */
class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
 public:
  /*
   * 생성자 - 시뮬레이션 설정 객체를 초기화합니다.
   * 멤버 초기화 리스트(: m_shader_config(ctx), m_memory_config(ctx))를 사용하여
   * 셰이더 설정과 메모리 설정 객체를 먼저 초기화합니다.
   * 초기화 리스트는 객체 본문({}) 실행 전에 멤버들을 초기화하는 효율적인 방법입니다.
   */
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;    // 아직 초기화되지 않았으므로 무효
    gpgpu_ctx = ctx;    // 시뮬레이터 전체 문맥 포인터 저장
  }

  // 옵션 파서에 시뮬레이션 관련 설정 옵션을 등록하는 함수
  void reg_options(class OptionParser *opp);

  /*
   * init() 함수 - 시뮬레이션 설정 전체를 초기화합니다.
   * 셰이더, 메모리, 클럭, 전력, 트레이스 등 모든 하위 시스템을 순서대로 초기화합니다.
   */
  void init() {
    gpu_stat_sample_freq = 10000;   // 통계 샘플링 주기를 10000 사이클로 기본 설정
    gpu_runtime_stat_flag = 0;      // 런타임 통계 플래그를 0(꺼짐)으로 기본 설정

    // 런타임 통계 설정 문자열을 파싱 (형식: "주기:플래그(16진수)")
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);

    m_shader_config.init();  // 셰이더 코어 설정 초기화

    // 텍스처 캐시의 라인 크기를 PTX(GPU 어셈블리어) 시스템에 알려줌
    // 캐시 라인(cache line)이란? 캐시가 한 번에 가져오는 데이터 블록의 크기
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());

    m_memory_config.init();  // 메모리 시스템 설정 초기화
    init_clock_domains();    // 클럭 도메인 초기화 (코어, 메모리 등 각각의 클럭 속도 설정)
    power_config::init();    // 전력 설정 초기화
    Trace::init();           // 트레이스(실행 추적) 시스템 초기화

    // 비주얼라이저(시각화 도구) 파일 이름 생성
    // 비주얼라이저는 시뮬레이션 과정을 그래프나 차트로 보여주는 도구입니다
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    // 파일 이름에 사용할 수 없는 문자를 하이픈으로 교체 (위 power_config::init()과 동일한 로직)
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    g_visualizer_filename = strdup(buf);  // 비주얼라이저 로그 파일 이름 저장

    m_valid = true;  // 모든 초기화 완료, 설정 유효
  }

  // === 게터(getter) 함수들 ===
  // 게터란? private 멤버 변수의 값을 외부에서 읽을 수 있게 해주는 함수
  // const 키워드는 이 함수가 객체를 변경하지 않는다는 보장

  unsigned get_core_freq() const { return core_freq; }          // 코어 클럭 주파수 반환
  unsigned num_shader() const { return m_shader_config.num_shader(); }  // 셰이더 코어 총 수 반환
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }  // SIMT 클러스터 수 반환
                                                                            // 클러스터 = 셰이더 코어들의 그룹
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }  // 동시 실행 가능한 최대 커널 수 반환

  /**
   * @brief SST 모드인지 확인하는 함수
   * 메모리 설정의 SST_mode 값을 반환합니다.
   */
  bool is_SST_mode() const { return m_memory_config.SST_mode; }

  unsigned checkpoint_option;  // 체크포인트 옵션 (시뮬레이션 상태를 저장하는 시점 설정)

  // === 디바이스 리밋(Device Limits) 게터 함수들 ===
  // GPU 디바이스의 하드웨어 제한값들을 반환합니다

  size_t stack_limit() const { return stack_size_limit; }  // 스택 크기 제한 반환
                                                            // 스택 = 함수 호출 시 사용하는 메모리 영역
  size_t heap_limit() const { return heap_size_limit; }    // 힙 크기 제한 반환
                                                            // 힙 = 동적 메모리 할당에 사용하는 영역
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }  // 동기화 깊이 제한
                                                                         // 커널이 다른 커널을 호출하는 깊이 제한
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;  // 대기 중인 커널 실행 수 제한
  }

  // L1 캐시를 플러시(flush, 비우기)할지 여부를 반환
  // 캐시 플러시란? 캐시의 모든 데이터를 메모리에 쓰고 캐시를 비우는 것
  bool flush_l1() const { return gpgpu_flush_l1_cache; }

 private:
  // private 멤버들 - 이 클래스 내부에서만 접근 가능

  void init_clock_domains(void);  // 클럭 도메인 초기화 함수 (각 부분의 클럭 속도 설정)

  // backward pointer - 시뮬레이터 전체 문맥에 대한 역방향 포인터
  // 역방향 포인터란? 자식이 부모를 가리키는 포인터로, 부모의 정보에 접근할 때 사용
  class gpgpu_context *gpgpu_ctx;

  bool m_valid;                        // 설정 유효 여부
  shader_core_config m_shader_config;  // 셰이더 코어 설정 객체
  memory_config m_memory_config;       // 메모리 시스템 설정 객체

  /*
   * 클럭 도메인(Clock Domain) 설정
   *
   * GPU에는 여러 개의 독립적인 클럭이 있습니다. 각 부분이 다른 속도로 동작할 수 있습니다:
   * - core: 연산 코어의 클럭 (가장 빠름)
   * - icnt: 인터커넥트(칩 내부 네트워크)의 클럭
   * - dram: DRAM 메모리의 클럭
   * - l2: L2 캐시의 클럭
   *
   * freq(frequency, 주파수): 1초에 몇 번 동작하는지 (Hz 단위)
   * period(주기): 한 번 동작하는 데 걸리는 시간 (주파수의 역수)
   */
  double core_freq;     // 코어 클럭 주파수
  double icnt_freq;     // 인터커넥트 클럭 주파수
  double dram_freq;     // DRAM 클럭 주파수
  double l2_freq;       // L2 캐시 클럭 주파수
  double core_period;   // 코어 클럭 주기
  double icnt_period;   // 인터커넥트 클럭 주기
  double dram_period;   // DRAM 클럭 주기
  double l2_period;     // L2 캐시 클럭 주기

  // === GPGPU-Sim 타이밍 모델 옵션들 ===

  unsigned long long gpu_max_cycle_opt;  // 최대 시뮬레이션 사이클 수 (이 수에 도달하면 시뮬레이션 종료)
  unsigned long long gpu_max_insn_opt;   // 최대 시뮬레이션 명령어 수 (이 수에 도달하면 종료)
  unsigned gpu_max_cta_opt;              // 최대 CTA(Cooperative Thread Array) 발행 수
                                         // CTA = 함께 동작하는 스레드 그룹 (CUDA에서는 블록이라고도 함)
  unsigned gpu_max_completed_cta_opt;    // 최대 완료된 CTA 수
  char *gpgpu_runtime_stat;              // 런타임 통계 설정 문자열
  bool gpgpu_flush_l1_cache;             // 커널 실행 후 L1 캐시 플러시 여부
  bool gpgpu_flush_l2_cache;             // 커널 실행 후 L2 캐시 플러시 여부
  bool gpu_deadlock_detect;              // 데드락(deadlock) 감지 활성화 여부
                                         // 데드락 = 서로 상대방을 기다리며 영원히 멈추는 상태
  int gpgpu_frfcfs_dram_sched_queue_size;// FR-FCFS 스케줄러 큐 크기
  int gpgpu_cflog_interval;              // 제어 흐름(control flow) 로그 간격
  char *gpgpu_clock_domains;             // 클럭 도메인 설정 문자열
  unsigned max_concurrent_kernel;        // 동시 실행 가능한 최대 커널 수

  // === 비주얼라이저(Visualizer) 설정 ===
  // 시뮬레이션 결과를 시각적으로 보여주는 도구 관련 설정
  bool g_visualizer_enabled;       // 비주얼라이저 활성화 여부
  char *g_visualizer_filename;     // 비주얼라이저 출력 파일 이름
  int g_visualizer_zlevel;         // 비주얼라이저 파일 압축 수준

  // === 통계 수집 설정 ===
  int gpu_stat_sample_freq;   // 통계 샘플링 주기 (몇 사이클마다 통계를 수집할지)
  int gpu_runtime_stat_flag;  // 런타임 통계 플래그 (어떤 통계를 수집할지 비트 플래그로 지정)

  // === 디바이스 리밋(Device Limits) ===
  // GPU가 허용하는 최대 자원 크기
  size_t stack_size_limit;                     // 스레드당 스택 크기 제한
  size_t heap_size_limit;                      // 힙(동적 메모리) 크기 제한
  size_t runtime_sync_depth_limit;             // 런타임 동기화 깊이 제한
  size_t runtime_pending_launch_count_limit;   // 대기 중인 커널 실행 수 제한

  // === GPU 컴퓨트 캐퍼빌리티(Compute Capability) ===
  // 컴퓨트 캐퍼빌리티란? GPU의 기능 수준을 나타내는 버전 번호입니다.
  // 예: 7.5 = 주 버전 7, 부 버전 5 (NVIDIA Turing 아키텍처)
  // 버전이 높을수록 더 많은 기능을 지원합니다.
  unsigned int gpgpu_compute_capability_major;  // 주(major) 버전 번호
  unsigned int gpgpu_compute_capability_minor;  // 부(minor) 버전 번호
  unsigned long long liveness_message_freq;     // 생존 메시지 출력 주기
                                                 // (시뮬레이션이 아직 실행 중임을 알리는 메시지)

  /*
   * friend 선언
   * friend란? 이 클래스의 private 멤버에 접근할 수 있는 "친구" 클래스입니다.
   * 일반적으로 private 멤버는 외부에서 접근할 수 없지만,
   * friend로 선언된 클래스는 예외적으로 접근이 가능합니다.
   */
  friend class gpgpu_sim;      // gpgpu_sim 클래스가 이 클래스의 private 멤버에 접근 가능
  friend class sst_gpgpu_sim;  // sst_gpgpu_sim 클래스도 private 멤버에 접근 가능
};

/*
 * ============================================================================
 * 점유율(Occupancy) 통계 구조체
 * ============================================================================
 * 점유율(Occupancy)이란?
 * GPU의 SM(Streaming Multiprocessor)이 실제로 얼마나 활용되고 있는지를 나타냅니다.
 * GPU에는 워프(warp, 32개 스레드 묶음)를 실행할 수 있는 "슬롯"이 있는데,
 * 이 슬롯 중 실제로 채워진 비율이 점유율입니다.
 *
 * 예: SM이 64개의 워프 슬롯을 가지고 있고, 48개가 채워졌다면
 *     점유율 = 48/64 = 75%
 *
 * 점유율이 높을수록 GPU가 더 효율적으로 사용되고 있다는 뜻입니다.
 */
struct occupancy_stats {
  // 기본 생성자 - 모든 값을 0으로 초기화
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}

  // 매개변수가 있는 생성자 - 지정된 값으로 초기화
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),         // 실제 채워진 워프 슬롯 수
        aggregate_theoretical_warp_slots(tws) {}  // 이론적 총 워프 슬롯 수

  unsigned long long aggregate_warp_slot_filled;       // 누적 채워진 워프 슬롯 수
  unsigned long long aggregate_theoretical_warp_slots; // 누적 이론적 총 워프 슬롯 수

  /*
   * 점유율을 분수(비율)로 계산하는 함수
   * 채워진 슬롯 수를 총 슬롯 수로 나눕니다.
   * 반환값은 0.0 ~ 1.0 사이의 실수입니다 (예: 0.75 = 75%)
   */
  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  /*
   * += 연산자 오버로딩
   * 연산자 오버로딩이란? +, -, +=  같은 기호의 동작을 사용자가 직접 정의하는 것입니다.
   * 이 함수는 두 occupancy_stats를 더해서 누적할 때 사용합니다.
   * 예: stats1 += stats2; → stats1에 stats2의 값을 더함
   */
  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;              // 채워진 슬롯 누적
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;  // 총 슬롯 누적
    return *this;  // 자기 자신에 대한 참조를 반환 (연쇄 호출 가능하게)
  }

  /*
   * + 연산자 오버로딩
   * 두 occupancy_stats를 더해서 새로운 객체를 반환합니다.
   * 예: stats3 = stats1 + stats2;
   * const를 붙여 원본 객체가 변경되지 않음을 보장합니다.
   */
  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};

// 전방 선언 - 이 클래스들의 자세한 정의는 다른 파일에 있습니다
class gpgpu_context;      // GPGPU 시뮬레이터 전체 문맥
class ptx_instruction;    // PTX 명령어 (PTX = GPU의 중간 어셈블리 언어)

/*
 * ============================================================================
 * 워치포인트(Watchpoint) 이벤트 클래스
 * ============================================================================
 * 워치포인트(watchpoint)란?
 * 특정 메모리 주소에 접근할 때 시뮬레이션을 멈추는 디버깅 기능입니다.
 * "이 메모리 주소를 누가 읽거나 쓰면 알려줘!"라고 설정하는 것입니다.
 *
 * 이 클래스는 워치포인트가 발생했을 때의 정보를 저장합니다:
 * - 어떤 스레드(thread)가 접근했는지
 * - 어떤 명령어(instruction)로 접근했는지
 */
class watchpoint_event {
 public:
  // 기본 생성자 - 빈 이벤트 생성 (아무 정보 없음)
  watchpoint_event() {
    m_thread = NULL;  // 스레드 정보 없음 (NULL = 아무것도 가리키지 않는 포인터)
    m_inst = NULL;    // 명령어 정보 없음
  }

  // 매개변수가 있는 생성자 - 스레드와 명령어 정보를 저장
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;   // 워치포인트를 발생시킨 스레드 정보 저장
    m_inst = pI;      // 해당 명령어 정보 저장
  }

  // 게터 함수들 - 저장된 정보를 반환
  const ptx_thread_info *thread() const { return m_thread; }  // 스레드 정보 반환
  const ptx_instruction *inst() const { return m_inst; }      // 명령어 정보 반환

 private:
  const ptx_thread_info *m_thread;  // 워치포인트를 트리거한 스레드에 대한 포인터
  const ptx_instruction *m_inst;    // 워치포인트를 트리거한 명령어에 대한 포인터
};

/*
 * ============================================================================
 * gpgpu_sim 클래스 - GPU 시뮬레이터의 핵심 클래스
 * ============================================================================
 * 이 클래스가 바로 GPU 시뮬레이터의 "심장"입니다!
 *
 * gpgpu_t를 상속받습니다 (gpgpu_t는 GPU의 기본적인 기능적 동작을 정의)
 *
 * 이 클래스가 하는 일:
 * 1. GPU의 사이클별(cycle-by-cycle) 동작을 시뮬레이션합니다
 *    - 매 클럭 사이클마다 GPU의 모든 부분이 무엇을 하는지 계산합니다
 *    - 이것을 "사이클 정확(cycle-accurate) 시뮬레이션"이라고 합니다
 *
 * 2. 커널(kernel) 실행을 관리합니다
 *    - 커널 = GPU에서 실행하는 함수
 *    - 여러 커널을 동시에 실행할 수 있습니다
 *
 * 3. 메모리 시스템을 관리합니다
 *    - 캐시, DRAM 등
 *
 * 4. 성능 통계를 수집합니다
 *    - 실행 시간, 캐시 히트율, 메모리 대역폭 등
 *
 * 이 클래스는 순수 가상 함수(createSIMTCluster)를 포함하고 있어서
 * 직접 객체를 만들 수 없는 추상 클래스(abstract class)입니다.
 * 실제 사용하려면 exec_gpgpu_sim이나 sst_gpgpu_sim같은 자식 클래스를 통해 사용합니다.
 */
class gpgpu_sim : public gpgpu_t {
 public:
  // 생성자 - 설정과 문맥 포인터를 받아 시뮬레이터를 초기화합니다
  gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx);

  // CUDA 디바이스 속성을 설정하는 함수
  // cudaDeviceProp = GPU의 하드웨어 사양 (메모리 크기, 코어 수, 클럭 속도 등)
  void set_prop(struct cudaDeviceProp *prop);

  // === 커널 실행 관리 함수들 ===
  void launch(kernel_info_t *kinfo);        // 커널을 GPU에서 실행 시작 (launch = 발사, 시작)
  bool can_start_kernel();                  // 새 커널을 시작할 수 있는지 확인
  unsigned finished_kernel();               // 완료된 커널의 ID를 반환
  void set_kernel_done(kernel_info_t *kernel);  // 커널 실행 완료 표시
  void stop_all_running_kernels();          // 모든 실행 중인 커널을 중지

  // === 시뮬레이션 핵심 함수들 ===
  void init();    // 시뮬레이터 초기화 (모든 하위 시스템 준비)
  void cycle();   // 한 사이클 실행 (시뮬레이터의 심장 박동 - 매 호출마다 GPU가 한 클럭 진행)
  bool active();  // 시뮬레이터가 아직 실행 중인지 확인 (처리할 작업이 남았는지)

  /*
   * cycle_insn_cta_max_hit() - 시뮬레이션 종료 조건 확인 함수
   * 다음 조건 중 하나라도 만족하면 true를 반환합니다:
   * 1. 최대 사이클 수에 도달
   * 2. 최대 명령어 수에 도달
   * 3. 최대 CTA 발행 수에 도달
   * 4. 최대 완료된 CTA 수에 도달
   *
   * || 는 OR(또는) 연산자, && 는 AND(그리고) 연산자입니다.
   * gpu_tot_sim_cycle + gpu_sim_cycle = 이전 커널들의 총 사이클 + 현재 커널의 사이클
   */
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }

  // === 통계 및 디버깅 함수들 ===
  void print_stats(unsigned long long streamID);  // 통계를 화면에 출력
  void update_stats();                             // 통계 갱신
  void deadlock_check();                           // 데드락(교착 상태) 검사
  void inc_completed_cta() { gpu_completed_cta++; }  // 완료된 CTA 수 1 증가

  // PDOM(Post-Dominator) 스택의 맨 위 정보를 가져오는 함수
  // PDOM은 GPU에서 분기(if/else) 처리를 관리하는 방법입니다
  void get_pdom_stack_top_info(unsigned sid,    // 셰이더 ID
                               unsigned tid,    // 스레드 ID
                               unsigned *pc,    // 프로그램 카운터 (현재 실행 중인 명령어 주소)
                               unsigned *rpc);  // 재수렴 지점 (분기된 스레드들이 다시 만나는 곳)

  // === GPU 하드웨어 속성 조회 함수들 ===
  // 이 함수들은 시뮬레이션하는 GPU의 하드웨어 사양을 반환합니다

  int shared_mem_size() const;          // 공유 메모리 총 크기 (바이트)
  int shared_mem_per_block() const;     // 블록당 공유 메모리 크기
  int compute_capability_major() const; // 컴퓨트 캐퍼빌리티 주 버전
  int compute_capability_minor() const; // 컴퓨트 캐퍼빌리티 부 버전
  int num_registers_per_core() const;   // 코어당 레지스터(register) 수
                                        // 레지스터 = 가장 빠른 메모리, 계산에 직접 사용
  int num_registers_per_block() const;  // 블록당 레지스터 수
  int wrp_size() const;                 // 워프 크기 (보통 32 스레드)
  int shader_clock() const;             // 셰이더 클럭 속도
  int max_cta_per_core() const;         // 코어당 최대 CTA 수
  int get_max_cta(const kernel_info_t &k) const;  // 특정 커널에 대한 최대 CTA 수 계산
  const struct cudaDeviceProp *get_prop() const;   // CUDA 디바이스 속성 반환
  enum divergence_support_t simd_model() const;    // SIMD 분기(divergence) 처리 모델 반환
                                                   // SIMD = 하나의 명령으로 여러 데이터를 동시 처리

  // === 커널 스케줄링 관련 함수들 ===
  unsigned threads_per_core() const;     // 코어당 스레드 수
  bool get_more_cta_left() const;        // 실행할 CTA가 더 남았는지 확인
  bool kernel_more_cta_left(kernel_info_t *kernel) const;  // 특정 커널의 CTA가 남았는지 확인
  bool hit_max_cta_count() const;        // 최대 CTA 수에 도달했는지 확인
  kernel_info_t *select_kernel();        // 다음에 실행할 커널을 선택
  PowerscalingCoefficients *get_scaling_coeffs();  // 전력 스케일링 계수 반환
  void decrement_kernel_latency();       // 커널 지연시간 감소 (커널 실행 전 대기 시간)

  // 시뮬레이션 설정에 대한 참조를 반환하는 게터
  const gpgpu_sim_config &get_config() const { return m_config; }

  // GPU 성능 통계를 출력하는 함수
  void gpu_print_stat(unsigned long long streamID);

  // 파이프라인 상태를 덤프(출력)하는 함수 - 디버깅용
  // mask = 어떤 정보를 출력할지, s = 셰이더 ID, m = 메모리 파티션 ID
  void dump_pipeline(int mask, int s, int m) const;

  // GPU로 데이터 복사(memcpy) 시 성능을 시뮬레이션하는 함수
  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // === 기능적 시뮬레이션(Functional Simulation)을 위한 함수들 ===
  // 기능적 시뮬레이션이란? 타이밍(시간)은 무시하고,
  // 프로그램이 "올바르게" 동작하는지만 확인하는 시뮬레이션입니다.
  // 타이밍 시뮬레이션보다 훨씬 빠르지만, 성능(얼마나 빠른지)은 알 수 없습니다.

  //! 셰이더 코어 설정을 반환하는 함수
  const shader_core_config *getShaderCoreConfig();

  //! 셰이더 코어의 메모리 설정을 반환하는 함수
  const memory_config *getMemoryConfig();

  //! SIMT 클러스터(셰이더 코어들의 그룹)를 반환하는 함수
  simt_core_cluster *getSIMTCluster();

  // 워치포인트가 트리거되었을 때 호출되는 함수
  void hit_watchpoint(unsigned watchpoint_num,    // 워치포인트 번호
                      ptx_thread_info *thd,       // 트리거한 스레드
                      const ptx_instruction *pI); // 트리거한 명령어

  /**
   * @brief SST 모드인지 확인하는 함수
   */
  bool is_SST_mode() { return m_config.is_SST_mode(); }

  // backward pointer - 시뮬레이터 전체 문맥에 대한 역방향 포인터
  class gpgpu_context *gpgpu_ctx;

 protected:
  // protected 멤버 - 이 클래스와 자식 클래스에서만 접근 가능
  // (private보다 조금 더 열린 접근 수준)

  // === 클럭 관련 함수들 ===
  void reinit_clock_domains(void);    // 클럭 도메인을 다시 초기화
  int next_clock_domain(void);        // 다음에 실행할 클럭 도메인을 결정
                                      // (어느 부분이 다음 "틱"을 할 차례인지)
  void issue_block2core();            // CTA(블록)를 코어에 할당하는 함수
                                      // (어떤 작업을 어떤 코어에서 실행할지 결정)

  // === 통계 출력 함수들 ===
  void print_dram_stats(FILE *fout) const;         // DRAM 통계 출력
  void shader_print_runtime_stat(FILE *fout);      // 셰이더 런타임 통계 출력
  void shader_print_l1_miss_stat(FILE *fout) const;// L1 캐시 미스 통계 출력
  void shader_print_cache_stats(FILE *fout) const; // 캐시 통계 출력
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;  // 스케줄러 통계 출력
  void visualizer_printstat();                      // 비주얼라이저용 통계 출력
  void print_shader_cycle_distro(FILE *fout) const;// 셰이더 사이클 분포 출력

  void gpgpu_debug();  // GPU 디버그 함수 (디버그 정보 출력)

 protected:
  // === 데이터 멤버들 ===

  /*
   * m_cluster - SIMT 클러스터 배열에 대한 이중 포인터
   * 이중 포인터(**)란? 포인터의 포인터로, 동적으로 크기가 정해지는 배열을 가리킵니다.
   *
   * GPU 구조:
   * GPU → [클러스터0][클러스터1]...[클러스터N]
   *         ↓
   *       [코어0][코어1]...[코어M]
   *
   * 클러스터(cluster)는 여러 개의 셰이더 코어를 묶은 그룹입니다.
   * 같은 클러스터 안의 코어들은 L1 캐시나 인터커넥트 포트를 공유합니다.
   */
  class simt_core_cluster **m_cluster;

  // 메모리 파티션 유닛 배열 - DRAM 메모리를 여러 구역으로 나눈 것
  class memory_partition_unit **m_memory_partition_unit;

  // 메모리 서브 파티션 배열 - 파티션을 더 작은 단위로 나눈 것
  // (각 서브 파티션에 L2 캐시 슬라이스가 있음)
  class memory_sub_partition **m_memory_sub_partition;

  // 실행 중인 커널들의 벡터(vector, 동적 배열)
  // 여러 커널이 동시에 실행될 수 있으므로 벡터로 관리합니다
  std::vector<kernel_info_t *> m_running_kernels;

  unsigned m_last_issued_kernel;  // 마지막으로 발행(issue)된 커널의 인덱스

  // 완료된 커널들의 목록 (list = 연결 리스트 자료구조)
  std::list<unsigned> m_finished_kernel;

  // CTA 관련 카운터들
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long m_total_cta_launched;  // 현재 커널에서 발행된 총 CTA 수
  unsigned long long gpu_tot_issued_cta;    // 전체 시뮬레이션에서 발행된 총 CTA 수
  unsigned gpu_completed_cta;               // 완료된 CTA 수

  unsigned m_last_cluster_issue;  // 마지막으로 CTA를 발행한 클러스터 인덱스
                                  // (라운드 로빈 방식으로 클러스터에 CTA를 분배하기 위해 사용)

  float *average_pipeline_duty_cycle;  // 평균 파이프라인 활용률 포인터
                                       // (파이프라인이 실제로 일하는 비율)
  float *active_sms;                   // 활성 SM(Streaming Multiprocessor) 수 포인터

  /*
   * 각 클럭 도메인의 다음 상승 에지(rising edge) 시간
   *
   * 클럭 도메인이란? 같은 속도로 동작하는 회로 영역입니다.
   * GPU에는 여러 개의 독립적인 클럭이 있어서, 각각의 "다음 틱 시간"을 추적합니다.
   *
   * 시뮬레이터는 이 시간들 중 가장 빠른 것을 먼저 처리합니다.
   * 이것이 cycle() 함수에서 next_clock_domain()을 호출하는 이유입니다.
   */
  double core_time;  // 코어의 다음 상승 에지 시간
  double icnt_time;  // 인터커넥트의 다음 상승 에지 시간
  double dram_time;  // DRAM의 다음 상승 에지 시간
  double l2_time;    // L2 캐시의 다음 상승 에지 시간

  // 디버깅용 변수
  bool gpu_deadlock;  // 데드락이 감지되었는지 여부

  //// 설정 매개변수들 ////
  const gpgpu_sim_config &m_config;  // 시뮬레이션 설정에 대한 const 참조
                                     // const 참조 = 읽기만 가능하고 수정 불가

  const struct cudaDeviceProp *m_cuda_properties;  // CUDA 디바이스 속성 포인터
  const shader_core_config *m_shader_config;       // 셰이더 코어 설정 포인터
  const memory_config *m_memory_config;            // 메모리 설정 포인터

  // === 통계 관련 객체 포인터들 ===
  class shader_core_stats *m_shader_stats;        // 셰이더 코어 통계
  class memory_stats_t *m_memory_stats;           // 메모리 통계
  class power_stat_t *m_power_stats;              // 전력 통계
  class gpgpu_sim_wrapper *m_gpgpusim_wrapper;    // 시뮬레이터 래퍼 (전력 모델과의 인터페이스)
  unsigned long long last_gpu_sim_insn;           // 마지막으로 기록한 시뮬레이션 명령어 수

  unsigned long long last_liveness_message_time;  // 마지막 생존 메시지 출력 시각

  // 특수 캐시 설정 맵
  // 특정 커널에 대해 다른 캐시 설정을 사용하고 싶을 때 사용합니다
  // std::map = 키-값 쌍으로 데이터를 저장하는 자료구조 (사전과 유사)
  std::map<std::string, FuncCache> m_special_cache_config;

  // 실행된 커널의 이름과 UID 기록
  // 나중에 통계 출력 시 사용합니다
  std::vector<std::string>
      m_executed_kernel_names;  //< 실행된 커널들의 이름 목록 (통계 출력용)
  std::vector<unsigned>
      m_executed_kernel_uids;  //< 실행된 커널들의 고유 ID 목록 (통계 출력용)

  // 워치포인트 히트 기록 - 워치포인트 번호를 키로 이벤트 정보를 저장
  std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  // 커널 정보를 문자열로 만드는 함수 (통계 출력용)
  std::string executed_kernel_info_string();
  // 실행된 커널 이름을 반환하는 함수
  std::string executed_kernel_name();
  // 통계 출력 후 커널 정보를 초기화하는 함수
  void clear_executed_kernel_info();

  /*
   * 순수 가상 함수(pure virtual function)
   * "= 0"은 이 함수가 이 클래스에서 구현되지 않고,
   * 반드시 자식 클래스에서 구현해야 한다는 뜻입니다.
   *
   * 이 때문에 gpgpu_sim은 추상 클래스(abstract class)가 됩니다.
   * 추상 클래스는 직접 객체를 만들 수 없고,
   * 자식 클래스(exec_gpgpu_sim, sst_gpgpu_sim)를 통해서만 사용합니다.
   *
   * 이것은 "템플릿 메서드 패턴(Template Method Pattern)"의 일부입니다.
   * 공통 로직은 부모 클래스에, 구체적인 차이는 자식 클래스에 구현합니다.
   */
  virtual void createSIMTCluster() = 0;

 public:
  // === 공개 통계 변수들 ===
  // 시뮬레이션 진행 상황을 추적하는 카운터들

  unsigned long long gpu_sim_insn;              // 현재 커널에서 실행된 명령어 수
  unsigned long long gpu_tot_sim_insn;          // 전체 시뮬레이션에서 실행된 총 명령어 수
  unsigned long long gpu_sim_insn_last_update;  // 마지막 통계 갱신 시 명령어 수
  unsigned gpu_sim_insn_last_update_sid;        // 마지막으로 명령어를 갱신한 셰이더 ID

  occupancy_stats gpu_occupancy;      // 현재 커널의 점유율 통계
  occupancy_stats gpu_tot_occupancy;  // 전체 시뮬레이션의 누적 점유율 통계

  /*
   * 커널 실행 시간을 기록하는 구조체
   * 각 커널이 언제 시작하고 끝났는지를 사이클 단위로 기록합니다
   */
  typedef struct {
    unsigned long long start_cycle;  // 커널 시작 사이클
    unsigned long long end_cycle;    // 커널 종료 사이클
  } kernel_time_t;

  // 커널 실행 시간 기록 맵
  // 외부 맵의 키: 스트림 ID, 내부 맵의 키: 커널 UID, 값: 시작/종료 시간
  std::map<unsigned long long, std::map<unsigned, kernel_time_t>>
      gpu_kernel_time;
  unsigned long long last_streamID;  // 마지막으로 사용된 스트림 ID
  unsigned long long last_uid;       // 마지막으로 사용된 커널 UID

  cache_stats aggregated_l1_stats;  // 집계된 L1 캐시 통계 (모든 코어의 L1 합산)
  cache_stats aggregated_l2_stats;  // 집계된 L2 캐시 통계 (모든 파티션의 L2 합산)

  // === 정체(congestion)로 인한 스톨(stall) 카운터들 ===
  // 스톨(stall)이란? 어떤 이유로 진행이 멈추는 것입니다.
  // 마치 교통 체증으로 차가 멈추는 것과 같습니다.

  unsigned int gpu_stall_dramfull;  // DRAM 큐가 가득 차서 발생한 스톨 횟수
  unsigned int gpu_stall_icnt2sh;  // 인터커넥트→셰이더 경로가 막혀서 발생한 스톨 횟수

  // 파티션별 요청 병렬처리 관련 통계
  unsigned long long partiton_reqs_in_parallel;        // 현재 병렬 처리 중인 요청 수
  unsigned long long partiton_reqs_in_parallel_total;  // 누적 병렬 요청 수
  unsigned long long partiton_reqs_in_parallel_util;        // 현재 파티션 활용률
  unsigned long long partiton_reqs_in_parallel_util_total;  // 누적 파티션 활용률
  unsigned long long gpu_sim_cycle_parition_util;           // 현재 커널의 파티션 활용 사이클
  unsigned long long gpu_tot_sim_cycle_parition_util;       // 누적 파티션 활용 사이클
  unsigned long long partiton_replys_in_parallel;           // 현재 병렬 처리 중인 응답 수
  unsigned long long partiton_replys_in_parallel_total;     // 누적 병렬 응답 수

  // === 캐시 설정 관련 함수들 ===
  // 커널별로 다른 캐시 설정을 적용할 수 있습니다

  FuncCache get_cache_config(std::string kernel_name);    // 특정 커널의 캐시 설정 반환
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);  // 커널별 캐시 설정 지정
  bool has_special_cache_config(std::string kernel_name); // 특수 캐시 설정이 있는지 확인
  void change_cache_config(FuncCache cache_config);       // 캐시 설정 변경 적용
  void set_cache_config(std::string kernel_name);         // 캐시 설정 적용

  // === CDP(CUDA Dynamic Parallelism) 기능적 시뮬레이션 관련 ===
  // CDP란? GPU 커널이 다른 커널을 직접 실행할 수 있는 기능입니다.
  // 재귀(recursion)나 동적 병렬처리가 가능해집니다.
  // Jin이라는 개발자가 추가한 기능입니다.
 protected:
  // 스트림 연산이 기능적 시뮬레이션을 수행할 때마다 설정됩니다
  bool m_functional_sim;                    // 기능적 시뮬레이션 모드인지 여부
  kernel_info_t *m_functional_sim_kernel;   // 기능적 시뮬레이션 중인 커널

 public:
  // 현재 기능적 시뮬레이션 모드인지 반환
  bool is_functional_sim() { return m_functional_sim; }

  // 기능적 시뮬레이션 중인 커널을 반환
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }

  // 현재 실행 중인 모든 커널의 목록을 반환 (복사본)
  std::vector<kernel_info_t *> get_running_kernels() {
    return m_running_kernels;
  }

  // 기능적 시뮬레이션으로 커널 실행 시작
  void functional_launch(kernel_info_t *k) {
    m_functional_sim = true;          // 기능적 시뮬레이션 모드 활성화
    m_functional_sim_kernel = k;      // 실행할 커널 설정
  }

  // 기능적 시뮬레이션 완료 처리
  void finish_functional_sim(kernel_info_t *k) {
    assert(m_functional_sim);              // 기능적 시뮬레이션 모드여야 함
    assert(m_functional_sim_kernel == k);  // 완료하려는 커널이 현재 실행 중인 커널과 일치해야 함
    m_functional_sim = false;              // 기능적 시뮬레이션 모드 해제
    m_functional_sim_kernel = NULL;        // 커널 포인터 초기화
  }
};

/*
 * ============================================================================
 * exec_gpgpu_sim 클래스 - 실행 모드 GPU 시뮬레이터
 * ============================================================================
 * gpgpu_sim을 상속받아 일반적인 실행 모드의 시뮬레이터를 구현합니다.
 * SST 없이 독립적으로 실행되는 기본 시뮬레이터입니다.
 *
 * virtual로 선언된 createSIMTCluster()를 구현하여
 * 실제 SIMT 클러스터를 생성합니다.
 */
class exec_gpgpu_sim : public gpgpu_sim {
 public:
  // 생성자 - 부모 클래스(gpgpu_sim) 초기화 후 SIMT 클러스터를 생성합니다
  exec_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {  // 부모 클래스 생성자 호출
    createSIMTCluster();          // SIMT 클러스터 생성 (순수 가상 함수 구현)
  }

  // SIMT 클러스터를 생성하는 가상 함수 구현
  // virtual 키워드로 이 함수도 자식 클래스에서 다시 재정의(override)할 수 있습니다
  virtual void createSIMTCluster();
};

/**
 * ============================================================================
 * sst_gpgpu_sim 클래스 - SST 연동 GPU 시뮬레이터
 * ============================================================================
 * gpgpu_sim을 상속받아 SST(Structural Simulation Toolkit)와
 * 연동되는 시뮬레이터를 구현합니다.
 *
 * SST Balar란? SST에서 GPU를 시뮬레이션하는 컴포넌트의 이름입니다.
 *
 * exec_gpgpu_sim과의 차이점:
 * 1. 메모리 시스템을 SST가 관리함 (GPU 자체 메모리 시뮬레이션 사용 안 함)
 * 2. 사이클 실행 방식이 다름 (SST_cycle 사용)
 * 3. memcpy가 빈 함수 (SST가 메모리를 관리하므로)
 *
 * 이것은 "전략 패턴(Strategy Pattern)"의 예입니다.
 * 같은 인터페이스(gpgpu_sim)를 사용하면서,
 * 내부 구현만 다른(독립 실행 vs SST 연동) 두 가지 버전을 제공합니다.
 */
class sst_gpgpu_sim : public gpgpu_sim {
 public:
  // 생성자 - 부모 클래스 초기화 후 SIMT 클러스터를 생성합니다
  sst_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {  // 부모 클래스 생성자 호출
    createSIMTCluster();          // SIMT 클러스터 생성
  }

  // === SST 메모리 처리 ===

  /*
   * SST 메모리 응답 버퍼
   * SST에서 메모리 요청의 응답이 돌아오면 이 버퍼에 저장됩니다.
   *
   * std::vector<std::deque<mem_fetch *>> 구조:
   * - vector: 코어 수만큼의 배열 (각 코어마다 하나의 큐)
   * - deque: 양쪽 끝에서 넣고 뺄 수 있는 큐 (Double-Ended Queue)
   * - mem_fetch*: 메모리 요청/응답 객체에 대한 포인터
   *
   * 구조: [코어0의 큐][코어1의 큐]...[코어N의 큐]
   *        ↓
   *      [응답0, 응답1, 응답2, ...]
   */
  std::vector<std::deque<mem_fetch *>>
      SST_gpgpu_reply_buffer; /** SST 메모리 응답 큐 */

  /**
   * @brief SST로부터 메모리 요청에 대한 응답을 받아
   *        버퍼(SST_gpgpu_reply_buffer)에 저장하는 함수
   *
   * @param core_id  응답을 받을 GPU 코어 번호
   * @param mem_req  메모리 요청 객체 포인터
   */
  void SST_receive_mem_reply(unsigned core_id, void *mem_req);

  /**
   * @brief 버퍼 큐의 맨 앞(head)에서 메모리 응답을 꺼내는 함수
   *        FIFO 방식으로 가장 먼저 온 응답부터 처리합니다
   *
   * @param core_id  응답을 가져올 GPU 코어 번호
   * @return mem_fetch*  꺼낸 메모리 요청/응답 객체의 포인터
   */
  mem_fetch *SST_pop_mem_reply(unsigned core_id);

  // SIMT 클러스터를 생성하는 가상 함수 구현
  virtual void createSIMTCluster();

  // === SST Balar 인터페이싱 함수들 ===

  /**
   * @brief GPU 코어를 한 사이클 진행시키고 통계를 수집하는 함수
   *        메모리 시스템은 SST가 처리하므로, 여기서는 코어만 진행합니다
   */
  void SST_cycle();

  /**
   * @brief SST_cycle()의 래퍼(wrapper) 함수
   *        부모 클래스의 cycle() 함수를 재정의(override)합니다
   *        래퍼란? 기존 함수를 감싸서 추가 기능을 제공하는 함수
   */
  void cycle();

  /**
   * @brief GPU가 활성 상태인지 확인하는 함수
   *        메모리 시스템 검사를 제거했습니다 (SST가 메모리를 관리하므로)
   *        부모 클래스의 active() 함수를 재정의합니다
   *
   * @return true   GPU가 아직 처리할 작업이 있음
   * @return false  GPU가 모든 작업을 완료함
   */
  bool active();

  /**
   * @brief GPU로 데이터 복사 성능을 시뮬레이션하는 함수
   *        SST 모드에서는 메모리를 SST가 관리하므로 빈 함수입니다
   *        부모 클래스의 함수를 재정의하여 아무것도 하지 않습니다
   *
   * @param dst_start_addr  복사 대상 시작 주소
   * @param count           복사할 바이트 수
   */
  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count){};

  /**
   * @brief SST 설정과 gpgpusim.config의 코어 수가 일치하는지 확인하는 함수
   *        두 시뮬레이터의 설정이 다르면 시뮬레이션 결과가 잘못될 수 있으므로
   *        시작 시 반드시 확인해야 합니다
   *
   * @param sst_numcores  SST에서 설정한 GPU 코어 수
   */
  void SST_gpgpusim_numcores_equal_check(unsigned sst_numcores);
};

#endif  // GPU_SIM_H 인클루드 가드 끝 - 이 파일의 내용이 여기서 끝남
