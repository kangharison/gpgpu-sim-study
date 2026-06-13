/*
 * [한국어 설명] CUDA PTX 기능 시뮬레이션 구현 (cuda-sim.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 **PTX 기능 시뮬레이션(functional simulation)** 레이어를 구현한다.
 * 기능 시뮬레이션이란 GPU 하드웨어의 사이클 단위 타이밍을 모델링하지 않고, PTX 명령어를 정확하게
 * 실행하여 올바른 결과만 생성하는 것을 목표로 한다. 타이밍 시뮬레이션(gpgpu-sim/)이 의존하는
 * 기반 연산 의미론(semantics)과, 순수 기능 모드(functionalCoreSim)에서의 CTA 단위 실행을 모두 제공한다.
 * 또한 PTX 명령어에 레이턴시/개시간격(initiation interval) 정보를 부여해 타이밍 모델이 활용할 수 있도록 준비한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름 관점:
 *   CUDA 애플리케이션
 *     → libcuda(런타임 인터셉트) / libopencl
 *         → gpgpusim_entrypoint.cc (시뮬레이터 초기화)
 *             → cuda-sim.cc (PTX 기능 시뮬레이션) ← 이 파일
 *                 ↕ 양방향: 타이밍 모델(shader.cc)이 ptx_exec_inst()를 워프 단계마다 호출
 *                 → instructions.cc (개별 PTX 명령어 구현)
 *                 → ptx_ir.cc (PTX IR 자료구조)
 * 실행 컨텍스트: 호스트 유저스페이스. GPU 커널 코드가 아니라 GPU를 에뮬레이션하는 호스트 프로세스다.
 * 순수 기능 모드에서는 gpgpu_cuda_ptx_sim_main_func()가 CTA 단위로 직접 실행 루프를 돌린다.
 * 타이밍 시뮬레이션 모드에서는 ptx_exec_inst()만 호출되어 레인별 명령어 실행을 담당한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - instructions.cc: 각 PTX 옵코드(OP_DEF)의 실제 연산 구현. opcodes.def X-매크로로 연결.
 *   - ptx_ir.{cc,h}: ptx_instruction, function_info, operand_info 자료구조.
 *   - abstract_hardware_model.{cc,h}: warp_inst_t, core_t, simt_stack 등 HW 추상 모델.
 *   - gpu-sim.{cc,h}: gpgpu_sim 최상위 시뮬레이터. 타이밍 모델과의 인터페이스.
 *   - memory.h: memory_space_impl — 가상 메모리 공간 구현.
 *   - ptx_loader.cc: PTX 바이너리 로딩 후 ptx_assemble()을 호출하여 PC를 할당.
 * 이 파일에 의존하는 모듈:
 *   - shader.cc (타이밍 모델): ptx_exec_inst()로 워프의 각 레인을 실행.
 *   - libcuda: gpgpu_ptx_sim_bindNameToTexture() 등 텍스처 바인딩 API를 호출.
 *   - gpgpusim_entrypoint.cc: read_sim_environment_variables(), ptx_opcocde_latency_options() 호출.
 * 데이터 흐름:
 *   kernel_info_t(커널 메타데이터) → ptx_sim_init_thread()(스레드 초기화)
 *   → ptx_thread_info(스레드 상태) → ptx_exec_inst()(명령어 실행)
 *   → warp_inst_t(타이밍 모델에 반환할 메모리 주소/공간 정보).
 *
 * === 주요 함수/구조체 요약 ===
 * ptx_thread_info::ptx_exec_inst()   — 가장 핵심. 하나의 PTX 명령어를 하나의 SIMT 레인에서 실행.
 *                                      predicate 평가, X-매크로 dispatch, 메모리 주소 수집을 포함.
 * ptx_instruction::pre_decode()      — 타이밍 모델이 필요한 필드(in[]/out[], op, latency,
 *                                      reconvergence_pc)를 파싱된 PTX IR로부터 채운다.
 * ptx_instruction::set_opcode_and_latency() — config 파일의 레이턴시 문자열을 파싱해 각 명령어의
 *                                      latency/initiation_interval/op 타입을 결정한다.
 * ptx_sim_init_thread()              — CTA 첫 실행 시 ptx_thread_info를 생성하고 공유/로컬/파라미터
 *                                      메모리 공간을 초기화한다. 재실행 시 기존 객체를 재사용한다.
 * cuda_sim::gpgpu_cuda_ptx_sim_main_func() — 순수 기능 시뮬레이션 최상위 루프.
 *                                      PDOM 분석 후 CTA를 하나씩 functionalCoreSim으로 실행한다.
 * function_info::ptx_assemble()      — PTX 함수 내 모든 명령어에 전역 고유 PC 주소를 부여하고
 *                                      g_pc_to_finfo/s_g_pc_to_insn 맵을 구성한다.
 * cuda_sim::get_converge_point()     — BRA 명령어 PC를 입력받아 PDOM 분석 결과에서
 *                                      즉각 사후지배자(immediate postdominator) PC를 반환한다.
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Jimmy Kwa, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
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

#include "cuda-sim.h"               // [한국어] cuda_sim 클래스 및 기능 시뮬레이터 선언 (이 파일의 헤더)

#include "instructions.h"           // [한국어] PTX 명령어 구현 함수들 (add_impl, ld_impl 등). opcodes.def의 FUNC 대상.
#include "ptx_ir.h"                 // [한국어] PTX IR 자료구조 (ptx_instruction, function_info, operand_info, symbol_table)
class ptx_recognizer;               // [한국어] Flex/Bison 파서의 재진입 스캐너 타입 전방 선언. yyscan_t 정의 전 필요.
typedef void *yyscan_t;             // [한국어] Flex reentrant scanner handle. ptx.tab.h 포함 전 정의되어야 한다.
#include <stdio.h>                  // [한국어] printf, fprintf, fflush, snprintf, fopen/fclose 등 C 표준 I/O
#include <map>                      // [한국어] std::map — PC→함수 매핑, 심볼→텍스처 매핑 등 대부분의 룩업 테이블
#include <set>                      // [한국어] std::set — 텍스처 레퍼런스 집합(m_NameToTextureRef), 초기화 중복 방지 집합
#include <sstream>                  // [한국어] std::ostringstream — 문자열 조합에 사용 (일부 경로)
#include "../../libcuda/gpgpu_context.h"  // [한국어] gpgpu_context — 시뮬레이터 전역 컨텍스트 (func_sim, ptx_parser 등 소유)
#include "../abstract_hardware_model.h"   // [한국어] warp_inst_t, core_t, simt_stack, kernel_info_t 등 HW 추상 인터페이스
#include "../gpgpu-sim/gpu-sim.h"         // [한국어] gpgpu_sim (최상위 타이밍 시뮬레이터), gpgpu_sim_config, memory_config
#include "../gpgpusim_entrypoint.h"       // [한국어] GPGPUsim_ctx, g_the_gpu 접근, checkpoint 옵션
#include "../statwrapper.h"               // [한국어] StatCreate/StatAddSample/StatDisp — 명령어 분류 통계 래퍼
#include "../stream_manager.h"            // [한국어] stream_manager — 커널 완료 등록(register_finished_kernel)
#include "cuda_device_runtime.h"          // [한국어] CDP(CUDA Dynamic Parallelism) — GPU 커널에서 자식 커널 실행
#include "decuda_pred_table/decuda_pred_table.h"  // [한국어] decuda predicate 조회 테이블 — PTXPlus predicate 평가
#include "memory.h"                       // [한국어] memory_space_impl<N> — shared/local/global 가상 메모리 공간 구현
#include "opcodes.h"                      // [한국어] PTX 옵코드 enum 정의 (LD_OP, ST_OP, BRA_OP 등)
#include "ptx-stats.h"                    // [한국어] ptx_file_line_stats_add_exec_count 등 PTX 실행 통계
#include "ptx.tab.h"                      // [한국어] Bison 파서 생성 헤더 (token 정의, ptx_recognizer 타입)
#include "ptx_loader.h"                   // [한국어] PTX 바이너리 로더 — ptx_assemble() 호출 선행 조건
#include "ptx_parser.h"                   // [한국어] PTX 파서 진입점 및 g_sym_name_to_symbol_table
#include "ptx_sim.h"                      // [한국어] ptx_thread_info, ptx_cta_info, ptx_warp_info 선언

int g_debug_execution = 0;
// Output debug information to file options
// [한국어] 전역 디버그 실행 레벨. 환경변수 PTX_SIM_DEBUG로 설정(read_sim_environment_variables 참조).
// 0=디버그 없음, 1~5=단계별 상태 출력, 6=레지스터 덤프, 10=전체 레지스터 덤프.
// ptx_exec_inst() 내에서 이 값에 따라 dump_modifiedregs()/dump_regs() 호출 여부가 결정된다.

/*
 * [한국어]
 * cuda_sim::ptx_opcocde_latency_options - PTX 옵코드 레이턴시/개시간격 옵션 등록
 *
 * @opp: option_parser_t 핸들 — gpgpusim.config 파싱을 담당하는 옵션 파서 객체
 * @return: 없음 (void)
 *
 * gpgpusim.config 파일에서 -ptx_opcode_latency_* / -ptx_opcode_initiation_* 옵션을
 * 파서에 등록한다. 실제 레이턴시 값은 이후 set_opcode_and_latency()가 sscanf로 읽는다.
 * 각 옵션의 기본값은 Fermi/Kepler 실측 데이터를 참고한 근사치이다.
 * INT[6]: ADD/SUB, MAX/MIN, MUL, MAD, DIV, SHFL
 * FP32[5]: ADD/SUB, MAX/MIN, MUL, MAD, DIV
 * FP64[5]: ADD/SUB, MAX/MIN, MUL, MAD, DIV
 * SFU[1]:  SQRT, SIN, COS, EX2, LG2, RSQRT, RCP
 * TENSOR[1]: MMA (Tensor Core 연산)
 * CDP[5]:  cudaStreamCreateWithFlags, cudaGetParameterBufferV2_init_perWarp,
 *          cudaGetParameterBufferV2_perKernel, cudaLaunchDeviceV2_init_perWarp,
 *          cudaLaunchDevicV2_perKernel (CUDA Dynamic Parallelism API 레이턴시)
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 시점 1회 호출 (gpgpusim_entrypoint.cc).
 * 멀티스레드 비고: 초기화 단계에서만 호출되므로 별도 락 불필요.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc::GPGPUsim_ctx::init() → [ptx_opcocde_latency_options]
 *     → option_parser_register() (option_parser.cc)
 */
void cuda_sim::ptx_opcocde_latency_options(option_parser_t opp) {
  // [한국어] 정수 연산(ADD/SUB, MAX/MIN, MUL, MAD, DIV, SHFL) 레이턴시 — 기본값 1,1,19,25,145,32 사이클
  option_parser_register(
      opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
      "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,19,25,145,32",
      "1,1,19,25,145,32");
  // [한국어] 단정밀도 부동소수(F32) 연산 레이턴시 — 기본값 1,1,1,1,30 사이클 (DIV만 SFU 30사이클)
  option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR,
                         &opcode_latency_fp,
                         "Opcode latencies for single precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,30",
                         "1,1,1,1,30");
  // [한국어] 배정밀도 부동소수(F64) 연산 레이턴시 — 기본값 8,8,8,8,335 사이클 (DIV 매우 큼)
  option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR,
                         &opcode_latency_dp,
                         "Opcode latencies for double precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,335",
                         "8,8,8,8,335");
  // [한국어] SFU(Special Function Unit) 명령어 레이턴시 — 기본값 8사이클 (sin/cos/sqrt/rcp 등)
  option_parser_register(opp, "-ptx_opcode_latency_sfu", OPT_CSTR,
                         &opcode_latency_sfu,
                         "Opcode latencies for SFU instructions"
                         "Default 8",
                         "8");
  // [한국어] Tensor Core(MMA 명령어) 레이턴시 — 기본값 64사이클 (Volta 이상 wmma 연산)
  option_parser_register(opp, "-ptx_opcode_latency_tesnor", OPT_CSTR,
                         &opcode_latency_tensor,
                         "Opcode latencies for Tensor instructions"
                         "Default 64",
                         "64");
  // [한국어] 정수 연산 개시간격(initiation interval, II) — 기본값 1,1,4,4,32,4 사이클
  // II는 레이턴시와 달리 파이프라인이 다음 명령어를 발행할 수 있는 최소 간격을 의미한다.
  option_parser_register(
      opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
      "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,4,4,32,4",
      "1,1,4,4,32,4");
  // [한국어] F32 연산 개시간격 — 기본값 1,1,1,1,5 사이클 (DIV만 5사이클 II)
  option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR,
                         &opcode_initiation_fp,
                         "Opcode initiation intervals for single precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,5",
                         "1,1,1,1,5");
  // [한국어] F64 연산 개시간격 — 기본값 8,8,8,8,130 사이클 (배정밀도 파이프라인은 처리량이 낮음)
  option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR,
                         &opcode_initiation_dp,
                         "Opcode initiation intervals for double precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,130",
                         "8,8,8,8,130");
  // [한국어] SFU 개시간격 — 기본값 8사이클
  option_parser_register(opp, "-ptx_opcode_initiation_sfu", OPT_CSTR,
                         &opcode_initiation_sfu,
                         "Opcode initiation intervals for sfu instructions"
                         "Default 8",
                         "8");
  // [한국어] Tensor Core 개시간격 — 기본값 64사이클
  option_parser_register(opp, "-ptx_opcode_initiation_tensor", OPT_CSTR,
                         &opcode_initiation_tensor,
                         "Opcode initiation intervals for tensor instructions"
                         "Default 64",
                         "64");
  // [한국어] CDP(CUDA Dynamic Parallelism) API 호출 레이턴시 5종 — 기본값 7200,8000,100,12000,1600 사이클
  // GPU 커널에서 cudaStreamCreateWithFlags/cudaLaunchDevice 등을 호출할 때 적용되는 지연값.
  option_parser_register(opp, "-cdp_latency", OPT_CSTR, &cdp_latency_str,
                         "CDP API latency <cudaStreamCreateWithFlags, \
cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, \
cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>"
                         "Default 7200,8000,100,12000,1600",
                         "7200,8000,100,12000,1600");
}

/*
 * [한국어]
 * gpgpu_t::gpgpu_ptx_sim_bindNameToTexture - 텍스처 이름과 텍스처 레퍼런스를 바인딩
 *
 * @name:    텍스처 심볼 이름 (PTX .tex 지시자의 이름, 예: "texData")
 * @texref:  CUDA 런타임이 제공하는 textureReference 구조체 포인터.
 *           normalized 좌표, filterMode, addressMode, channelDesc 등을 포함한다.
 * @dim:     텍스처 차원 (1, 2, 3)
 * @readmode: cudaTextureReadMode — cudaReadModeElementType 또는 cudaReadModeNormalizedFloat
 * @ext:     확장 텍스처 여부 플래그
 * @return:  없음 (void)
 *
 * libcuda의 cudaBindTexture* 호환 경로에서 호출되어, 텍스처 이름-레퍼런스 매핑을
 * m_NameToTextureRef 및 m_TextureRefToName에 등록한다.
 * CUDART_VERSION <= 1200 조건: CUDA 1.2 이하에서만 이 경로를 사용한다.
 * 최신 CUDA에서는 텍스처 바인딩 방식이 변경되어 이 함수 본문이 컴파일되지 않는다.
 * 동일 텍스처 이름에 여러 textureReference가 바인딩될 경우 모든 레퍼런스의 속성이
 * 동일한지 assert로 검증한다 (normalized, filterMode, addressMode, channelDesc 등 비교).
 *
 * 실행 컨텍스트: 커널 실행 전 호스트 스레드에서 호출된다.
 * 동기화: m_NameToTextureRef/m_TextureRefToName 맵 접근은 커널 실행 전에만 이루어지므로 락 불필요.
 *
 * 호출 체인:
 *   libcuda::cudaBindTexture() → [gpgpu_ptx_sim_bindNameToTexture]
 *     → m_NameToTextureRef, m_TextureRefToName, m_NameToAttribute 갱신
 */
void gpgpu_t::gpgpu_ptx_sim_bindNameToTexture(
    const char *name, const struct textureReference *texref, int dim,
    int readmode, int ext) {
#if (CUDART_VERSION <= 1200)
  // [한국어] CUDA 1.2 이하에서만 이 텍스처 바인딩 경로를 활성화한다.
  std::string texname(name);                                            // [한국어] C 문자열 이름을 std::string으로 변환 (맵 키로 사용)
  if (m_NameToTextureRef.find(texname) == m_NameToTextureRef.end()) {  // [한국어] 이 텍스처 이름이 처음 등록되는 경우
    m_NameToTextureRef[texname] = std::set<const struct textureReference *>();  // [한국어] 빈 set 초기화 — 동일 이름에 여러 texref 허용
  } else {
    // [한국어] 이미 등록된 이름: 기존 레퍼런스와 속성이 동일한지 검증한다.
    // 같은 이름에 바인딩된 모든 texref는 동일한 포맷이어야 하므로 assert로 보장.
    const struct textureReference *tr = *m_NameToTextureRef[texname].begin();  // [한국어] 기존 첫 번째 texref 가져오기
    assert(tr != NULL);                                                         // [한국어] NULL이면 버그: 등록된 레퍼런스가 유효해야 함
    // asserts that all texrefs in set have same fields
    // [한국어] normalized: 텍스처 좌표가 [0,1] 정규화되는지 여부 (같아야 함)
    // [한국어] filterMode: 점 샘플링(cudaFilterModePoint) vs 선형 보간(cudaFilterModeLinear)
    // [한국어] addressMode[0..2]: UV/W 축 각각의 클램프/랩 모드
    // [한국어] channelDesc.x/y/z/w: 각 채널의 비트 폭, .f: 채널 포맷 (float/int/uint)
    assert(tr->normalized == texref->normalized &&
           tr->filterMode == texref->filterMode &&
           tr->addressMode[0] == texref->addressMode[0] &&
           tr->addressMode[1] == texref->addressMode[1] &&
           tr->addressMode[2] == texref->addressMode[2] &&
           tr->channelDesc.x == texref->channelDesc.x &&
           tr->channelDesc.y == texref->channelDesc.y &&
           tr->channelDesc.z == texref->channelDesc.z &&
           tr->channelDesc.w == texref->channelDesc.w &&
           tr->channelDesc.f == texref->channelDesc.f);
  }
  m_NameToTextureRef[texname].insert(texref);          // [한국어] 이름→레퍼런스 set에 추가 (동일 이름에 여러 texref 가능)
  m_TextureRefToName[texref] = texname;                // [한국어] 역방향 매핑: texref 포인터→이름 (unbind 및 address 조회용)
  const textureReferenceAttr *texAttr = new textureReferenceAttr(
      texref, dim, (enum cudaTextureReadMode)readmode, ext);  // [한국어] 텍스처 속성 객체 생성 (차원, 읽기 모드, 확장 플래그 포함)
  m_NameToAttribute[texname] = texAttr;                // [한국어] 이름→속성 매핑 저장 — TEX 명령어 실행 시 사용
#endif
}

/*
 * [한국어]
 * gpgpu_t::gpgpu_ptx_sim_findNamefromTexture - texref 포인터에서 텍스처 이름 역조회
 *
 * @texref: CUDA 텍스처 레퍼런스 포인터 (bindNameToTexture에서 등록된 것)
 * @return: 텍스처 이름 C 문자열 (m_TextureRefToName 맵의 내부 버퍼, 맵이 유효한 동안만 유효)
 *
 * m_TextureRefToName (texref→이름 역방향 맵)을 조회하여 이름을 반환한다.
 * 등록되지 않은 texref로 호출하면 assert 실패로 프로그램이 중단된다.
 * gpgpu_ptx_sim_bindTextureToArray() / gpgpu_ptx_sim_unbindTexture()에서
 * texref 포인터를 이름으로 변환할 때 사용한다.
 *
 * 호출 체인:
 *   gpgpu_ptx_sim_bindTextureToArray() → [gpgpu_ptx_sim_findNamefromTexture]
 *   gpgpu_ptx_sim_unbindTexture()      → [gpgpu_ptx_sim_findNamefromTexture]
 */
const char *gpgpu_t::gpgpu_ptx_sim_findNamefromTexture(
    const struct textureReference *texref) {
  std::map<const struct textureReference *, std::string>::const_iterator t =
      m_TextureRefToName.find(texref);                  // [한국어] texref 포인터로 이름 역방향 룩업
  assert(t != m_TextureRefToName.end());                // [한국어] 등록되지 않은 texref면 버그 — abort
  return t->second.c_str();                             // [한국어] std::string의 내부 C 문자열 반환 (맵 소멸 전까지 유효)
}

/*
 * [한국어]
 * intLOGB2 - 부호 없는 정수의 정수 부분 이진 로그(floor(log2(v))) 계산
 *
 * @v: 로그를 계산할 양의 정수 (2의 거듭제곱이어야 올바른 결과 보장)
 * @return: floor(log2(v)) — 예: v=8 → 3, v=16 → 4, v=1 → 0
 *
 * 5단계 이진 탐색(binary search)으로 O(1)에 MSB(최상위 비트) 위치를 찾는다.
 * 텍스처 타일 크기(Tx, Ty)와 텍셀 크기의 비트 수를 계산하는 데 사용된다.
 * (gpgpu_ptx_sim_bindTextureToArray 내부에서 intLOGB2(Tx), intLOGB2(texel_size) 호출)
 * 알고리즘: 상위 절반 비트에 1이 있으면 shift를 늘리고, v를 shift만큼 우시프트하여
 * 반복적으로 범위를 좁혀 나간다.
 *
 * 호출 체인:
 *   gpgpu_ptx_sim_bindTextureToArray() → [intLOGB2] → (반환값)
 */
unsigned int intLOGB2(unsigned int v) {
  unsigned int shift;   // [한국어] 각 단계에서 v의 비트를 얼마나 시프트할지 (0, 1, 2, 4, 8, 16 중 하나)
  unsigned int r;       // [한국어] 누적 결과 — 최종적으로 floor(log2(v))가 된다

  r = 0;                // [한국어] 결과 초기화 (v=1이면 log2=0이므로 0으로 시작)

  // [한국어] 1단계: 비트 16 이상인지 확인 → 그렇다면 16비트 우시프트 필요
  shift = ((v & 0xFFFF0000) != 0) << 4;  // [한국어] 상위 16비트에 1이 있으면 shift=16, 없으면 shift=0
  v >>= shift;                            // [한국어] 상위 16비트가 있었다면 v를 16비트 우시프트 (범위를 절반으로 축소)
  r |= shift;                             // [한국어] shift=16이면 결과에 16을 더함

  // [한국어] 2단계: 남은 v의 비트 8 이상인지 확인 → 8비트 우시프트
  shift = ((v & 0xFF00) != 0) << 3;      // [한국어] 상위 8비트에 1이 있으면 shift=8
  v >>= shift;                            // [한국어] 필요시 8비트 우시프트
  r |= shift;                             // [한국어] 결과에 8 추가

  // [한국어] 3단계: 비트 4 이상인지 확인 → 4비트 우시프트
  shift = ((v & 0xF0) != 0) << 2;        // [한국어] 상위 4비트에 1이 있으면 shift=4
  v >>= shift;                            // [한국어] 필요시 4비트 우시프트
  r |= shift;                             // [한국어] 결과에 4 추가

  // [한국어] 4단계: 비트 2 이상인지 확인 → 2비트 우시프트
  shift = ((v & 0xC) != 0) << 1;         // [한국어] 비트 2-3에 1이 있으면 shift=2 (0xC = 0b1100)
  v >>= shift;                            // [한국어] 필요시 2비트 우시프트
  r |= shift;                             // [한국어] 결과에 2 추가

  // [한국어] 5단계: 비트 1인지 확인 → 1비트 우시프트
  shift = ((v & 0x2) != 0) << 0;         // [한국어] 비트 1에 1이 있으면 shift=1 (0x2 = 0b10)
  v >>= shift;                            // [한국어] 필요시 1비트 우시프트 (v는 이제 0 또는 1)
  r |= shift;                             // [한국어] 결과에 1 추가 — r이 최종 floor(log2(원래v))

  return r;                               // [한국어] 5단계 합산된 비트 위치 반환
}

/*
 * [한국어]
 * gpgpu_t::gpgpu_ptx_sim_bindTextureToArray - 텍스처를 cudaArray에 바인딩하고 타일 정보 계산
 *
 * @texref: 이미 bindNameToTexture로 등록된 textureReference 포인터
 * @array:  cudaMallocArray로 할당된 GPU 배열. desc(채널 포맷)와 devPtr32(GPU 주소) 포함.
 * @return: 없음 (void)
 *
 * CUDA 텍스처 시스템에서 cudaBindTextureToArray()에 해당하는 시뮬레이터 내부 처리를 수행한다.
 * 주요 작업:
 *   1. texref→이름 역조회 후 m_NameToCudaArray에 배열 포인터 등록
 *   2. cudaArray의 채널 비트 폭(desc.w+x+y+z)으로 텍셀 크기(바이트) 계산
 *   3. 텍스처 캐시 라인 크기와 텍셀 크기로부터 2D 타일 크기(Tx, Ty) 결정
 *      - Tx, Ty는 텍스처 좌표→캐시 라인 매핑에 사용 (TEX 명령어 주소 계산)
 *      - 캐시 라인 16B→Tx=4, 32/64B→Tx=8, 128/256B→Tx=16 (기본값)
 *      - 텍셀이 4바이트보다 클수록 Tx를 절반씩 줄여 라인당 텍셀 수를 맞춤
 *   4. intLOGB2로 비트 수 계산 후 textureInfo 구조체에 저장
 * CUDART_VERSION <= 1200에서만 활성화 (구형 텍스처 API 경로).
 *
 * 호출 체인:
 *   libcuda::cudaBindTextureToArray() → [gpgpu_ptx_sim_bindTextureToArray]
 *     → gpgpu_ptx_sim_findNamefromTexture() → intLOGB2()
 */
void gpgpu_t::gpgpu_ptx_sim_bindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array) {
#if (CUDART_VERSION <= 1200)
  // [한국어] texref 포인터로 텍스처 이름 역조회 — 이름이 없으면 assert abort
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);

  std::map<std::string, const struct cudaArray *>::const_iterator t =
      m_NameToCudaArray.find(texname);                  // [한국어] 동일 이름에 이미 배열이 바인딩됐는지 확인
  // check that there's nothing there first
  if (t != m_NameToCudaArray.end()) {
    // [한국어] 기존 바인딩이 있으면 경고 출력 (암묵적 unbind 후 새로 바인딩함)
    printf(
        "GPGPU-Sim PTX:   Warning: binding to texref associated with %s, which "
        "was previously bound.\nImplicitly unbinding texref associated to %s "
        "first\n",
        texname.c_str(), texname.c_str());
  }
  m_NameToCudaArray[texname] = array;                   // [한국어] 이름→cudaArray 매핑 등록 (TEX 명령어에서 실제 데이터 읽기에 사용)

  // [한국어] 텍셀 크기 계산: cudaArray의 채널 포맷에서 비트 폭을 합산
  // desc.w=알파, desc.x=R, desc.y=G, desc.z=B 각 채널의 비트 수
  unsigned int texel_size_bits =
      array->desc.w + array->desc.x + array->desc.y + array->desc.z;
  unsigned int texel_size = texel_size_bits / 8;        // [한국어] 비트→바이트 변환 (예: RGBA32 = 128비트 → 16바이트)
  unsigned int Tx, Ty;                                  // [한국어] 2D 텍스처 타일 크기: Tx=수평 텍셀 수, Ty=수직 텍셀 수
  int r;                                                // [한국어] 텍셀 크기 보정 루프용 임시 변수

  printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);     // [한국어] 텍셀 크기 디버그 출력
  printf("GPGPU-Sim PTX:   texture cache linesize = %d\n",
         m_function_model_config.get_texcache_linesize());       // [한국어] 텍스처 캐시 라인 크기 출력 (gpgpusim.config 설정값)

  // first determine base Tx size for given linesize
  // [한국어] 캐시 라인 크기별 기본 Tx 결정: 라인에 맞는 텍셀 수의 제곱근에 가까운 값
  switch (m_function_model_config.get_texcache_linesize()) {
    case 16:                // [한국어] 16바이트 라인 → 기본 Tx=4 (4×4 타일이면 16바이트 1텍셀 기준)
      Tx = 4;
      break;
    case 32:                // [한국어] 32바이트 라인 → 기본 Tx=8
      Tx = 8;
      break;
    case 64:                // [한국어] 64바이트 라인 → 기본 Tx=8 (더 큰 타일은 Ty로 커버)
      Tx = 8;
      break;
    case 128:               // [한국어] 128바이트 라인 → 기본 Tx=16
      Tx = 16;
      break;
    case 256:               // [한국어] 256바이트 라인 → 기본 Tx=16
      Tx = 16;
      break;
    default:
      // [한국어] 지원하지 않는 라인 크기: 구성 오류로 assert abort
      printf(
          "GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n",
          m_function_model_config.get_texcache_linesize());
      assert(0);
      break;
  }
  r = texel_size >> 2;      // [한국어] 텍셀이 4바이트보다 큰 경우: r>0이면 Tx를 절반씩 줄인다
  // modify base Tx size to take into account size of each texel in bytes
  // [한국어] 텍셀 크기가 4배씩 증가할 때마다 Tx를 절반으로 줄여 라인당 텍셀 수 유지
  while (r != 0) {
    Tx = Tx >> 1;           // [한국어] Tx를 절반으로 줄임 (텍셀이 클수록 Tx가 작아짐)
    r = r >> 2;             // [한국어] r을 4로 나눔 (텍셀이 4배 더 크면 한 번 더 줄임)
  }
  // by now, got the correct Tx size, calculate correct Ty size
  // [한국어] Ty = 캐시 라인 바이트 수 / (Tx × 텍셀 크기) — 라인 하나를 Tx×Ty 타일로 채움
  Ty = m_function_model_config.get_texcache_linesize() / (Tx * texel_size);

  printf(
      "GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n",
      Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));             // [한국어] 타일 크기와 비트 수 디버그 출력
  printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n",
         texel_size, intLOGB2(texel_size));             // [한국어] 텍셀 크기와 비트 수 디버그 출력
  printf(
      "GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n",
      array->devPtr32);                                 // [한국어] GPU 배열의 32비트 디바이스 주소 출력
  printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);

  // [한국어] textureInfo 구조체 할당 및 타일 정보 저장 — TEX 명령어 실행 시 참조
  struct textureInfo *texInfo =
      (struct textureInfo *)malloc(sizeof(struct textureInfo));
  texInfo->Tx = Tx;                            // [한국어] 수평 타일 크기 (텍셀 단위)
  texInfo->Ty = Ty;                            // [한국어] 수직 타일 크기 (텍셀 단위)
  texInfo->Tx_numbits = intLOGB2(Tx);          // [한국어] Tx의 비트 수 — 텍스처 주소 비트 분리에 사용
  texInfo->Ty_numbits = intLOGB2(Ty);          // [한국어] Ty의 비트 수
  texInfo->texel_size = texel_size;            // [한국어] 텍셀 크기 (바이트) — 주소 오프셋 계산에 사용
  texInfo->texel_size_numbits = intLOGB2(texel_size);  // [한국어] 텍셀 크기의 비트 수
  m_NameToTextureInfo[texname] = texInfo;      // [한국어] 이름→타일 정보 매핑 저장 (TEX 명령어에서 참조)
#endif
}

/*
 * [한국어]
 * gpgpu_t::gpgpu_ptx_sim_unbindTexture - 텍스처-배열 바인딩 해제
 *
 * @texref: 해제할 textureReference 포인터
 * @return: 없음 (void)
 *
 * cudaUnbindTexture()에 해당하는 시뮬레이터 처리.
 * m_NameToCudaArray와 m_NameToTextureInfo에서 이 텍스처의 항목을 제거한다.
 * m_NameToTextureRef와 m_TextureRefToName은 유지 (이름 등록은 해제하지 않음).
 * bind-use-unbind-bind-use-unbind 패턴을 가정: unbind 이후 즉시 새 바인딩이 온다.
 *
 * 호출 체인:
 *   libcuda::cudaUnbindTexture() → [gpgpu_ptx_sim_unbindTexture]
 */
void gpgpu_t::gpgpu_ptx_sim_unbindTexture(
    const struct textureReference *texref) {
#if (CUDART_VERSION <= 1200)
  // assumes bind-use-unbind-bind-use-unbind pattern
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);  // [한국어] texref→이름 역조회
  m_NameToCudaArray.erase(texname);                                  // [한국어] 이름→cudaArray 매핑 제거 (배열 바인딩 해제)
  m_NameToTextureInfo.erase(texname);                                // [한국어] 이름→타일 정보 매핑 제거
#endif
}

#define MAX_INST_SIZE 8 /*bytes*/
// [한국어] MAX_INST_SIZE: 하나의 PTX 명령어가 차지하는 최대 가상 주소 크기(바이트).
// PTX는 가변 크기 명령어를 허용하며, 이 값은 전역 PC 주소 공간에서 명령어 간 간격을 결정한다.
// 실제 SASS(하드웨어 ISA)의 8바이트 명령어 크기와 일치하도록 설정됨.

/*
 * [한국어]
 * function_info::ptx_assemble - PTX 함수 내 명령어에 전역 PC 주소를 할당하고 어셈블
 *
 * @return: 없음 (void)
 *
 * PTX 로더가 파싱한 명령어 리스트(m_instructions)를 받아서:
 *   1. 전역 고유 PC 어드레스를 각 명령어에 할당 (g_assemble_code_next_pc로부터 시작)
 *   2. m_instr_mem[]: 함수 로컬 오프셋→명령어 포인터 배열 구성
 *   3. s_g_pc_to_insn[]: 전역 PC→명령어 포인터 벡터에 등록 (ptx_fetch_inst에서 사용)
 *   4. g_pc_to_finfo[]: 전역 PC→함수 정보 맵에 등록 (get_converge_point에서 사용)
 *   5. 레이블 오프셋 계산 및 BRA/BREAKADDR/CALLP 명령어의 target PC 심볼 테이블 등록
 * m_assembled 플래그로 중복 호출을 방지한다.
 * PC 정렬: 함수 시작은 MAX_INST_SIZE(8) 배수로 정렬하여 명령어 크기 변수 명령어 간 충돌 방지.
 * PDOM 분석은 이 함수에서 제거되어 런타임(do_pdom())에서 수행된다 (#if 0 블록 참조).
 *
 * 실행 컨텍스트: PTX 로딩 시점 1회 (kernel launch 전 준비 단계). 싱글스레드 컨텍스트.
 *
 * 호출 체인:
 *   ptx_loader.cc::load_ptx_from_string() → [ptx_assemble]
 *     → g_pc_to_finfo 맵 갱신, s_g_pc_to_insn 벡터 확장
 *     → m_symtab->set_label_address() (BRA 대상 PC 등록)
 */
void function_info::ptx_assemble() {
  if (m_assembled) {         // [한국어] 이미 어셈블된 함수는 재처리하지 않음 (중복 호출 방지)
    return;
  }

  // get the instructions into instruction memory...
  unsigned num_inst = m_instructions.size();              // [한국어] 파싱된 PTX 명령어 수 (레이블 포함)
  m_instr_mem_size = MAX_INST_SIZE * (num_inst + 1);      // [한국어] 명령어당 최대 8바이트 × (명령어 수+1) 크기로 배열 할당
  m_instr_mem = new ptx_instruction *[m_instr_mem_size];  // [한국어] 함수 로컬 명령어 포인터 배열 동적 할당

  printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ",
         m_name.c_str());
  fflush(stdout);
  std::list<ptx_instruction *>::iterator i;               // [한국어] 명령어 리스트 순회 이터레이터

  addr_t PC =
      gpgpu_ctx->func_sim->g_assemble_code_next_pc;  // globally unique address
                                                     // (across functions)
  // [한국어] g_assemble_code_next_pc: 전역 PC 카운터. 모든 함수에 걸쳐 유일한 주소 공간을 형성.
  // 함수가 로드될 때마다 이 값에서 시작하여 다음 함수는 이 값 이후에 배치된다.

  // start function on an aligned address
  // [한국어] 함수 시작 PC를 MAX_INST_SIZE(8) 배수로 정렬: PC % 8 바이트만큼 NULL 패딩
  for (unsigned i = 0; i < (PC % MAX_INST_SIZE); i++)
    gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);  // [한국어] 정렬 패딩 — NULL 슬롯 삽입
  PC += PC % MAX_INST_SIZE;     // [한국어] PC를 정렬된 경계로 전진
  m_start_PC = PC;              // [한국어] 이 함수의 시작 PC 저장 (ptx_thread_info 초기화 시 사용)

  addr_t n = 0;  // offset in m_instr_mem
  // Why s_g_pc_to_insn.size() is needed to reserve additional memory for insts?
  // reserve is cumulative. s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() +
  // MAX_INST_SIZE*m_instructions.size());
  // [한국어] 전역 벡터에 공간 사전 예약 — 재할당 비용 방지. reserve는 누적이므로 현재 크기+추가 필요량.
  gpgpu_ctx->s_g_pc_to_insn.reserve(MAX_INST_SIZE * m_instructions.size());

  // [한국어] 모든 명령어를 순회하며 PC 할당
  for (i = m_instructions.begin(); i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;   // [한국어] 현재 처리 중인 PTX 명령어 포인터
    if (pI->is_label()) {
      // [한국어] 레이블 명령어: 실행 가능 명령어가 아니라 분기 대상 이름.
      // 현재 오프셋 n을 labels 맵에 저장해 BRA 명령어가 참조할 수 있게 함.
      const symbol *l = pI->get_label();  // [한국어] 레이블 심볼 가져오기
      labels[l->name()] = n;              // [한국어] 이름→오프셋 매핑 저장 (BRA 해석 시 사용)
    } else {
      // [한국어] 실행 가능 명령어: PC 할당 및 전역 맵에 등록
      gpgpu_ctx->func_sim->g_pc_to_finfo[PC] = this;       // [한국어] PC→함수 정보 등록 (get_converge_point에서 함수 식별)
      m_instr_mem[n] = pI;                                  // [한국어] 함수 로컬 배열에 저장
      gpgpu_ctx->s_g_pc_to_insn.push_back(pI);             // [한국어] 전역 PC→명령어 벡터에 추가
      assert(pI == gpgpu_ctx->s_g_pc_to_insn[PC]);         // [한국어] 벡터 인덱스가 PC와 일치하는지 검증
      pI->set_m_instr_mem_index(n);                         // [한국어] 명령어에 함수 로컬 오프셋 저장
      pI->set_PC(PC);                                       // [한국어] 명령어에 전역 PC 저장 (ptx_exec_inst에서 next_instr()로 사용)
      assert(pI->inst_size() <= MAX_INST_SIZE);             // [한국어] 명령어 크기가 최대값 초과 방지

      // [한국어] 멀티 슬롯 명령어: inst_size() > 1이면 나머지 슬롯에 NULL 삽입 (간격 유지)
      for (unsigned i = 1; i < pI->inst_size(); i++) {
        gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);  // [한국어] 빈 슬롯 — 이 PC는 명령어 중간
        m_instr_mem[n + i] = NULL;                          // [한국어] 함수 로컬 배열에도 NULL 슬롯
      }
      n += pI->inst_size();    // [한국어] 함수 로컬 오프셋 전진
      PC += pI->inst_size();   // [한국어] 전역 PC 전진 (inst_size() 단위로 증가)
    }
  }
  gpgpu_ctx->func_sim->g_assemble_code_next_pc = PC;       // [한국어] 다음 함수가 시작할 전역 PC 갱신

  // [한국어] 2단계: BRA/BREAKADDR/CALLP 명령어의 대상 레이블을 PC로 해석하여 심볼 테이블에 등록
  for (unsigned ii = 0; ii < n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];   // [한국어] 현재 처리할 명령어
    if (pI->get_opcode() == BRA_OP || pI->get_opcode() == BREAKADDR_OP ||
        pI->get_opcode() == CALLP_OP) {
      // [한국어] 분기/호출 명령어: 대상 레이블 이름을 실제 PC로 변환
      operand_info &target = pI->dst();  // get operand, e.g. target name
      // [한국어] 대상 레이블이 이 함수의 레이블 맵에 없으면 오류 (잘못된 PTX)
      if (labels.find(target.name()) == labels.end()) {
        printf(
            "GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not "
            "appear in assembly code.",
            pI->source_file(), pI->source_line(), target.name().c_str());
        abort();
      }
      unsigned index = labels[target.name()];  // determine address from name
      // [한국어] 레이블 오프셋 index에 있는 명령어의 실제 전역 PC 조회
      unsigned PC = m_instr_mem[index]->get_PC();
      m_symtab->set_label_address(target.get_symbol(), PC);  // [한국어] 심볼 테이블에 레이블→PC 등록
      target.set_type(label_t);                              // [한국어] 피연산자를 label 타입으로 마크
    }
  }
  m_n = n;           // [한국어] 함수의 총 명령어 슬롯 수 저장 (pre_decode 루프 상한)
  printf("  done.\n");
  fflush(stdout);

  // disable pdom analysis  here and do it at runtime
  // [한국어] PDOM(Post-Dominator) 분석은 이 함수에서 제거됨 (#if 0).
  // 현재는 do_pdom()을 런타임에서 on-demand로 수행한다 (gpgpu_cuda_ptx_sim_main_func 참조).
  // 이렇게 하면 실행되지 않는 커널의 PDOM 분석 비용을 줄일 수 있다.
#if 0
   printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n", m_name.c_str() );
   create_basic_blocks();
   connect_basic_blocks();
   bool modified = false;
   do {
      find_dominators();
      find_idominators();
      modified = connect_break_targets();
   } while (modified == true);

   if ( g_debug_execution>=50 ) {
      print_basic_blocks();
      print_basic_block_links();
      print_basic_block_dot();
   }
   if ( g_debug_execution>=2 ) {
      print_dominators();
   }
   find_postdominators();
   find_ipostdominators();
   if ( g_debug_execution>=50 ) {
      print_postdominators();
      print_ipostdominators();
   }

   printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n", m_name.c_str() );
   for ( unsigned ii=0; ii < n; ii += m_instr_mem[ii]->inst_size() ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      pI->pre_decode();
   }
   printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n", m_name.c_str() );
   fflush(stdout);

   m_assembled = true;
#endif
}

/*
 * [한국어] 메모리 주소 공간 변환 함수들
 *
 * CUDA PTX는 여러 메모리 공간(shared/local/global)을 가진다. 기능 시뮬레이터는
 * 이 공간들을 단일 선형 가상 주소 공간(generic address space)에 매핑하여 관리한다.
 * 주소 범위 배치 (abstract_hardware_model.h에서 정의):
 *   GLOBAL_HEAP_START (≫0x80000000) 이상 또는 STATIC_ALLOC_LIMIT 미만: 글로벌 공간
 *   SHARED_GENERIC_START 이상 ~ LOCAL_GENERIC_START 미만: 공유 메모리 공간
 *   LOCAL_GENERIC_START 이상 ~ SHARED_GENERIC_START 미만: 로컬 메모리 공간
 * SM별로 SHARED_MEM_SIZE_MAX 크기의 공유 메모리 슬롯이 배치되고,
 * 스레드별로 LOCAL_MEM_SIZE_MAX 크기의 로컬 메모리 슬롯이 배치된다.
 */

/*
 * [한국어]
 * shared_to_generic - 공유 메모리 주소를 제네릭 주소로 변환
 *
 * @smid: SM(Streaming Multiprocessor) 인덱스 — 어느 SM의 공유 메모리인지 식별
 * @addr: 공유 메모리 내 오프셋 주소 (0 ~ SHARED_MEM_SIZE_MAX-1)
 * @return: 제네릭 주소 공간에서의 전역 주소
 *
 * 공식: SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + addr
 * 각 SM의 공유 메모리는 제네릭 공간에서 연속적으로 배치된다.
 * ptx_exec_inst()가 공유 메모리 접근 주소를 타이밍 모델에 보고할 때 사용.
 *
 * 호출 체인:
 *   instructions.cc::ld_impl()/st_impl() → [shared_to_generic]
 */
addr_t shared_to_generic(unsigned smid, addr_t addr) {
  assert(addr < SHARED_MEM_SIZE_MAX);                   // [한국어] 공유 메모리 범위 초과 방지 (버그 감지)
  return SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + addr;
  // [한국어] SM별 공유 메모리 슬롯 시작(SHARED_GENERIC_START + smid×슬롯크기)에 오프셋 추가
}

/*
 * [한국어]
 * global_to_generic - 글로벌 메모리 주소를 제네릭 주소로 변환 (항등 함수)
 *
 * @addr: 글로벌 메모리 주소
 * @return: 동일한 주소 (글로벌 공간은 제네릭 공간과 동일하게 매핑됨)
 *
 * 글로벌 메모리는 제네릭 주소와 동일한 번지를 사용하므로 변환이 불필요하다.
 * 대칭성을 위해 존재하는 항등 함수(identity function).
 */
addr_t global_to_generic(addr_t addr) { return addr; }  // [한국어] 글로벌 주소 = 제네릭 주소 (변환 없음)

/*
 * [한국어]
 * isspace_shared - 제네릭 주소가 특정 SM의 공유 메모리 범위에 속하는지 확인
 *
 * @smid: SM 인덱스
 * @addr: 확인할 제네릭 주소
 * @return: true이면 이 SM의 공유 메모리 범위 내, false이면 아님
 *
 * whichspace()에서 주소 공간 판별 시 사용. shared_to_generic 공식의 역방향 범위 검사.
 */
bool isspace_shared(unsigned smid, addr_t addr) {
  addr_t start = SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX;          // [한국어] 이 SM 공유 메모리의 제네릭 시작 주소
  addr_t end = SHARED_GENERIC_START + (smid + 1) * SHARED_MEM_SIZE_MAX;      // [한국어] 이 SM 공유 메모리의 제네릭 종료 주소 (exclusive)
  if ((addr >= end) || (addr < start)) return false;  // [한국어] 범위 밖이면 false
  return true;                                         // [한국어] 범위 내이면 이 SM의 공유 메모리
}

/*
 * [한국어]
 * isspace_global - 제네릭 주소가 글로벌 메모리 범위에 속하는지 확인
 *
 * @addr: 확인할 제네릭 주소
 * @return: true이면 글로벌 메모리 (GLOBAL_HEAP_START 이상 또는 정적 상수 영역)
 *
 * 글로벌 메모리는 두 범위로 구성: (1) GLOBAL_HEAP_START 이상의 힙 영역,
 * (2) STATIC_ALLOC_LIMIT 미만의 정적 전역/상수 변수 영역.
 */
bool isspace_global(addr_t addr) {
  return (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
  // [한국어] 동적 할당(힙) 또는 정적 전역/상수 영역 — 둘 다 global_space로 분류
}

/*
 * [한국어]
 * whichspace - 제네릭 주소가 어느 메모리 공간에 속하는지 판별
 *
 * @addr: 제네릭 주소
 * @return: memory_space_t enum — global_space, shared_space, local_space 중 하나
 *
 * PTX ld/st 명령어 실행 후 접근된 메모리 공간을 타이밍 모델에 보고하기 위해 사용.
 * 주소 범위로만 판별하므로 smid/hwtid 정보 없이 공간을 식별 가능하다.
 * (단, 어느 SM/스레드의 shared/local인지는 별도로 smid/hwtid 필요)
 */
memory_space_t whichspace(addr_t addr) {
  if ((addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT)) {  // [한국어] 글로벌 메모리 범위
    return global_space;
  } else if (addr >= SHARED_GENERIC_START) {                          // [한국어] 공유 메모리 범위 (SHARED_GENERIC_START ~ LOCAL_GENERIC_START)
    return shared_space;
  } else {
    return local_space;                                               // [한국어] 로컬 메모리 범위 (나머지)
  }
}

/*
 * [한국어]
 * generic_to_shared - 제네릭 주소를 공유 메모리 내 오프셋으로 역변환
 *
 * @smid: SM 인덱스
 * @addr: 제네릭 공간의 공유 메모리 주소
 * @return: 공유 메모리 내 오프셋 (0 ~ SHARED_MEM_SIZE_MAX-1)
 *
 * shared_to_generic의 역함수. 타이밍 모델이 제네릭 주소로 공유 메모리에 접근할 때 사용.
 */
addr_t generic_to_shared(unsigned smid, addr_t addr) {
  assert(isspace_shared(smid, addr));                                         // [한국어] 이 주소가 실제로 이 SM의 공유 메모리인지 검증
  return addr - (SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX);         // [한국어] 슬롯 시작 주소를 빼서 오프셋 계산
}

/*
 * [한국어]
 * local_to_generic - 로컬 메모리 주소를 제네릭 주소로 변환
 *
 * @smid:  SM 인덱스
 * @hwtid: SM 내 하드웨어 스레드 인덱스 (0 ~ n_threads_per_sm-1)
 * @addr:  로컬 메모리 내 오프셋 (0 ~ LOCAL_MEM_SIZE_MAX-1)
 * @return: 제네릭 공간에서의 전역 주소
 *
 * 공식: LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM × smid) + (LOCAL_MEM_SIZE_MAX × hwtid) + addr
 * 각 스레드는 LOCAL_MEM_SIZE_MAX 크기의 독립적인 로컬 메모리 슬롯을 가진다.
 * SM 전체 로컬 메모리는 TOTAL_LOCAL_MEM_PER_SM = LOCAL_MEM_SIZE_MAX × n_threads_per_sm.
 */
addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(addr < LOCAL_MEM_SIZE_MAX);                        // [한국어] 로컬 메모리 범위 초과 방지
  return LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
         (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
  // [한국어] SM 슬롯 시작 + 스레드 슬롯 시작 + 오프셋
}

/*
 * [한국어]
 * isspace_local - 제네릭 주소가 특정 SM/스레드의 로컬 메모리 범위에 속하는지 확인
 *
 * @smid:  SM 인덱스
 * @hwtid: 하드웨어 스레드 인덱스
 * @addr:  확인할 제네릭 주소
 * @return: true이면 해당 스레드의 로컬 메모리, false이면 아님
 */
bool isspace_local(unsigned smid, unsigned hwtid, addr_t addr) {
  addr_t start = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid);              // [한국어] 이 스레드 로컬 메모리의 제네릭 시작 주소
  addr_t end = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
               (LOCAL_MEM_SIZE_MAX * (hwtid + 1));          // [한국어] 이 스레드 로컬 메모리의 제네릭 종료 주소 (exclusive)
  if ((addr >= end) || (addr < start)) return false;        // [한국어] 범위 밖이면 false
  return true;
}

/*
 * [한국어]
 * generic_to_local - 제네릭 주소를 로컬 메모리 내 오프셋으로 역변환
 *
 * @smid:  SM 인덱스
 * @hwtid: 하드웨어 스레드 인덱스
 * @addr:  제네릭 공간의 로컬 메모리 주소
 * @return: 로컬 메모리 내 오프셋 (0 ~ LOCAL_MEM_SIZE_MAX-1)
 *
 * local_to_generic의 역함수.
 */
addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(isspace_local(smid, hwtid, addr));                 // [한국어] 이 주소가 해당 스레드 로컬 메모리인지 검증
  return addr - (LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid));             // [한국어] 스레드 로컬 슬롯 시작 주소를 빼서 오프셋 계산
}

/*
 * [한국어]
 * generic_to_global - 제네릭 주소를 글로벌 메모리 주소로 역변환 (항등 함수)
 *
 * @addr: 제네릭 주소
 * @return: 동일한 주소 (글로벌 공간은 제네릭 주소와 동일)
 */
addr_t generic_to_global(addr_t addr) { return addr; }  // [한국어] 글로벌 = 제네릭 (변환 없음)

/*
 * [한국어]
 * gpgpu_t::gpu_malloc - 시뮬레이션된 GPU 글로벌 메모리 동적 할당
 *
 * @size:   할당할 바이트 수
 * @return: 할당된 GPU 가상 주소 (void* — cudaMalloc 반환 타입과 호환)
 *
 * 실제 GPU DRAM을 할당하지 않고, 시뮬레이터 내부의 선형 주소 공간에서
 * m_dev_malloc 포인터를 전진시켜 가상 GPU 주소를 할당한다.
 * 256바이트 경계로 정렬하여 메모리 접근 패턴과 캐시 동작을 정확히 모델링한다.
 * (실제 NVIDIA GPU도 cudaMalloc 결과가 256바이트 정렬됨)
 * 설정자: libcuda::cudaMalloc() → 이 함수
 * 읽는 자: 커널이 ld/st 명령어로 이 주소 범위에 접근
 *
 * 호출 체인:
 *   libcuda::cudaMalloc() → [gpu_malloc] → (할당 주소 반환)
 */
void *gpgpu_t::gpu_malloc(size_t size) {
  unsigned long long result = m_dev_malloc;  // [한국어] 현재 할당 포인터를 반환값으로 저장 (할당 전 주소)
  if (g_debug_execution >= 3) {
    // [한국어] 디버그 레벨 3 이상: 할당 내역 출력 (환경변수 PTX_SIM_DEBUG=3 이상)
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;                      // [한국어] 할당 포인터를 요청 크기만큼 전진
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
    // [한국어] 256바이트 나머지가 있으면 다음 256바이트 경계로 패딩 (정렬 보장)
  return (void *)result;                     // [한국어] 할당된 GPU 가상 주소 반환 (할당 전 값)
}

/*
 * [한국어]
 * gpgpu_t::gpu_mallocarray - 시뮬레이션된 GPU 배열(cudaArray) 메모리 할당
 *
 * @size:   할당할 바이트 수
 * @return: 할당된 GPU 가상 주소 (cudaArray의 devPtr32로 사용)
 *
 * gpu_malloc과 동일한 로직으로 GPU 가상 주소를 할당한다.
 * cudaMallocArray()에 해당하며 텍스처/서피스 배열에 사용된다.
 * 반환된 주소는 gpgpu_ptx_sim_bindTextureToArray()에서 array->devPtr32로 저장된다.
 *
 * 호출 체인:
 *   libcuda::cudaMallocArray() → [gpu_mallocarray]
 */
void *gpgpu_t::gpu_mallocarray(size_t size) {
  unsigned long long result = m_dev_malloc;  // [한국어] 현재 할당 포인터 저장
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;                      // [한국어] 할당 포인터 전진
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
    // [한국어] 256바이트 경계 정렬 패딩
  return (void *)result;                     // [한국어] 할당된 주소 반환
}

/*
 * [한국어]
 * gpgpu_t::memcpy_to_gpu - 호스트(CPU) 메모리를 시뮬레이션된 GPU 글로벌 메모리로 복사
 *
 * @dst_start_addr: GPU 가상 주소 (gpu_malloc으로 할당된 주소)
 * @src:            호스트 메모리 포인터
 * @count:          복사할 바이트 수
 * @return:         없음 (void)
 *
 * cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice)에 해당한다.
 * 1바이트씩 m_global_mem->write()를 호출하여 시뮬레이터 가상 메모리에 기록한다.
 * 이후 perf_memcpy_to_gpu()를 호출하여 타이밍 모델의 메모리 상태도 동기화한다.
 * (타이밍 모델은 L2/DRAM 상태를 추적하므로 초기 데이터 존재를 알려야 함)
 * 동기화: 커널 실행 전 호스트 스레드에서만 호출되므로 락 불필요.
 *
 * 호출 체인:
 *   libcuda::cudaMemcpy(HostToDevice) → [memcpy_to_gpu]
 *     → m_global_mem->write() (1바이트씩)
 *     → g_the_gpu->perf_memcpy_to_gpu() (타이밍 모델 동기화)
 */
void gpgpu_t::memcpy_to_gpu(size_t dst_start_addr, const void *src,
                            size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: copying %zu bytes from CPU[0x%Lx] to GPU[0x%Lx] ... ",
        count, (unsigned long long)src, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  char *src_data = (char *)src;            // [한국어] void* → char* 캐스트 — 바이트 단위 포인터 산술을 위해 필요
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, src_data + n, NULL, NULL);
    // [한국어] GPU 가상 글로벌 메모리에 1바이트씩 기록. NULL, NULL: 이벤트/타이밍 콜백 없음 (기능 시뮬레이션 전용)

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  // [한국어] 타이밍 모델(gpgpu_sim)에도 복사 이벤트 통지 — L2 캐시/DRAM 초기 상태 반영
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(dst_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
 * [한국어]
 * gpgpu_t::memcpy_from_gpu - 시뮬레이션된 GPU 글로벌 메모리를 호스트로 복사
 *
 * @dst:            호스트 메모리 포인터 (결과를 받을 버퍼)
 * @src_start_addr: GPU 가상 주소 (읽을 GPU 메모리의 시작 주소)
 * @count:          복사할 바이트 수
 * @return:         없음 (void)
 *
 * cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost)에 해당한다.
 * m_global_mem->read()로 GPU 가상 메모리에서 1바이트씩 읽어 호스트 버퍼에 복사한다.
 * perf_memcpy_to_gpu() 호출은 타이밍 모델의 D2H 전송 추적을 위한 것이다.
 * (주: 함수 이름은 from_gpu지만 내부에서 perf_memcpy_to_gpu를 호출하는 것은
 *  타이밍 모델이 이 주소 범위의 데이터가 변경 가능함을 알도록 하는 안전장치)
 *
 * 호출 체인:
 *   libcuda::cudaMemcpy(DeviceToHost) → [memcpy_from_gpu]
 *     → m_global_mem->read() (1바이트씩)
 */
void gpgpu_t::memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to CPU[0x%Lx] ...",
           count, (unsigned long long)src_start_addr, (unsigned long long)dst);
    fflush(stdout);
  }
  unsigned char *dst_data = (unsigned char *)dst;  // [한국어] void* → uchar* 캐스트 (1바이트 단위 접근)
  for (unsigned n = 0; n < count; n++)
    m_global_mem->read(src_start_addr + n, 1, dst_data + n);
    // [한국어] GPU 가상 메모리에서 1바이트씩 읽어 호스트 버퍼에 저장

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  // [한국어] 타이밍 모델에 D2H 영역 통지 — GPU 메모리가 호스트로 전송됨을 알림
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(src_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
 * [한국어]
 * gpgpu_t::memcpy_gpu_to_gpu - 시뮬레이션된 GPU 메모리 내 복사 (D2D)
 *
 * @dst:   복사 대상 GPU 가상 주소
 * @src:   복사 원본 GPU 가상 주소
 * @count: 복사할 바이트 수
 * @return: 없음 (void)
 *
 * cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice)에 해당한다.
 * m_global_mem에서 read 후 write 패턴으로 1바이트씩 내부 복사를 수행한다.
 * 실제 GPU의 D2D 복사는 DMA 엔진이 처리하지만, 기능 시뮬레이터는 순차 복사로 대체한다.
 *
 * 호출 체인:
 *   libcuda::cudaMemcpy(DeviceToDevice) → [memcpy_gpu_to_gpu]
 */
void gpgpu_t::memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to GPU[0x%Lx] ...",
           count, (unsigned long long)src, (unsigned long long)dst);
    fflush(stdout);
  }
  for (unsigned n = 0; n < count; n++) {
    unsigned char tmp;                                  // [한국어] 1바이트 임시 버퍼 — src에서 읽은 값을 dst에 쓰기 전 보관
    m_global_mem->read(src + n, 1, &tmp);               // [한국어] GPU 가상 메모리 src+n에서 1바이트 읽기
    m_global_mem->write(dst + n, 1, &tmp, NULL, NULL);  // [한국어] GPU 가상 메모리 dst+n에 1바이트 쓰기
  }
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
 * [한국어]
 * gpgpu_t::gpu_memset - 시뮬레이션된 GPU 글로벌 메모리를 특정 바이트값으로 초기화
 *
 * @dst_start_addr: GPU 가상 주소 (초기화 시작 주소)
 * @c:              초기화할 바이트값 (int이지만 unsigned char로 잘라서 사용)
 * @count:          초기화할 바이트 수
 * @return:         없음 (void)
 *
 * cudaMemset()에 해당한다. m_global_mem->write()를 count번 반복 호출하여
 * 지정된 주소 범위를 c_value로 채운다. 실제 GPU의 cudaMemset은 하드웨어 최적화를
 * 사용하지만, 기능 시뮬레이터는 바이트 단위 루프로 처리한다.
 *
 * 호출 체인:
 *   libcuda::cudaMemset() → [gpu_memset]
 */
void gpgpu_t::gpu_memset(size_t dst_start_addr, int c, size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at "
        "0x%Lx... ",
        count, (unsigned char)c, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  unsigned char c_value = (unsigned char)c;  // [한국어] int → unsigned char 잘림: memset은 하위 1바이트만 사용
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, &c_value, NULL, NULL);
    // [한국어] 주소 범위 [dst_start_addr, dst_start_addr+count)를 c_value로 1바이트씩 채움
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
 * [한국어]
 * cuda_sim::ptx_print_insn - 특정 PC의 PTX 명령어를 파일에 출력 (디버그용)
 *
 * @pc: 출력할 PTX 명령어의 전역 PC 주소
 * @fp: 출력 대상 파일 스트림 (stdout 또는 디버그 파일)
 * @return: 없음 (void)
 *
 * g_pc_to_finfo 맵에서 PC에 해당하는 함수를 찾아 그 함수의 print_insn()을 호출한다.
 * PC가 등록되지 않은 주소면 오류 메시지를 출력한다.
 * 타이밍 시뮬레이터의 디버그 덤프나 파이프라인 추적 시 호출된다.
 *
 * 호출 체인:
 *   shader.cc::shader_core_ctx::cycle() (디버그 경로) → [ptx_print_insn]
 *     → function_info::print_insn()
 */
void cuda_sim::ptx_print_insn(address_type pc, FILE *fp) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);  // [한국어] PC→함수 맵에서 해당 함수 찾기
  if (f == g_pc_to_finfo.end()) {
    // [한국어] 등록되지 않은 PC: NULL 슬롯(다중 슬롯 명령어의 중간) 또는 잘못된 PC
    fprintf(fp, "<no instruction at address 0x%llx>", pc);
    return;
  }
  function_info *finfo = f->second;   // [한국어] 이 PC가 속한 함수 정보
  assert(finfo);                       // [한국어] NULL 함수 포인터는 버그
  finfo->print_insn(pc, fp);          // [한국어] 함수의 print_insn() 위임 — PTX 소스 행과 명령어 텍스트 출력
}

/*
 * [한국어]
 * cuda_sim::ptx_get_insn_str - 특정 PC의 PTX 명령어를 문자열로 반환 (디버그용)
 *
 * @pc:    조회할 PTX 명령어의 전역 PC 주소
 * @return: PTX 명령어 텍스트 문자열. 해당 PC가 없으면 오류 메시지 문자열.
 *
 * ptx_print_insn의 문자열 반환 버전. 프로파일러나 통계 모듈에서 명령어 텍스트가
 * 필요할 때 사용한다. STR_SIZE(255)로 버퍼 크기를 제한한다.
 *
 * 호출 체인:
 *   통계/프로파일 출력 경로 → [ptx_get_insn_str]
 *     → function_info::get_insn_str()
 */
std::string cuda_sim::ptx_get_insn_str(address_type pc) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);  // [한국어] PC→함수 맵 조회
  if (f == g_pc_to_finfo.end()) {
#define STR_SIZE 255       // [한국어] 오류 메시지 버퍼 크기 — 16진수 주소 표현에 충분한 크기
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';                                    // [한국어] 널 종료 보장 (snprintf 잘림 방어)
    snprintf(buff, STR_SIZE, "<no instruction at address 0x%llx>", pc);  // [한국어] 오류 메시지 포맷
    return std::string(buff);                                     // [한국어] 오류 문자열 반환
  }
  function_info *finfo = f->second;  // [한국어] 해당 함수 정보
  assert(finfo);                      // [한국어] 유효한 함수 포인터 검증
  return finfo->get_insn_str(pc);    // [한국어] 함수의 get_insn_str() 위임 — PTX 명령어 텍스트 반환
}

/*
 * [한국어]
 * ptx_instruction::set_fp_or_int_archop - 명령어를 FP 연산 또는 INT 연산으로 분류
 *
 * @return: 없음 (void). 결과는 멤버 oprnd_type에 저장됨.
 *
 * oprnd_type(operand type) 필드를 설정하여 이 명령어가 부동소수점(FP) 연산인지
 * 정수(INT) 연산인지를 분류한다.
 * 분류 규칙:
 *   - MEMBAR, SSY, BRA, BAR, RET, RETP, NOP, EXIT, CALLP, CALL: UN_OP (분류 없음)
 *   - CVT, SET, SLCT: 소스 타입(get_type2())이 F16/F32/F64면 FP_OP, 아니면 INT_OP
 *   - 그 외: 주 데이터 타입(get_type())이 F16/F32/F64면 FP_OP, 아니면 INT_OP
 * 이 정보는 AccelWattch 전력 계산 및 파이프라인 자원 분류에 사용된다.
 * set_opcode_and_latency() 내부에서 set_mul_div_or_other_archop()과 함께 호출된다.
 *
 * 호출 체인:
 *   set_opcode_and_latency() → [set_fp_or_int_archop]
 */
void ptx_instruction::set_fp_or_int_archop() {
  oprnd_type = UN_OP;  // [한국어] 기본값: 분류 없음 (제어 흐름/배리어 명령어)

  if ((m_opcode == MEMBAR_OP) || (m_opcode == SSY_OP) || (m_opcode == BRA_OP) ||
      (m_opcode == BAR_OP) || (m_opcode == RET_OP) || (m_opcode == RETP_OP) ||
      (m_opcode == NOP_OP) || (m_opcode == EXIT_OP) || (m_opcode == CALLP_OP) ||
      (m_opcode == CALL_OP)) {
    // do nothing
    // [한국어] 제어 흐름/동기화 명령어: FP/INT 분류 불필요 — UN_OP(미분류)로 유지
  } else if ((m_opcode == CVT_OP || m_opcode == SET_OP ||
              m_opcode == SLCT_OP)) {
    // [한국어] CVT(타입 변환)/SET(비교 후 설정)/SLCT(선택): 소스 타입2로 분류
    // 이 명령어들은 소스(입력)와 목적지(출력) 타입이 다를 수 있으므로 소스 타입 기준
    if (get_type2() == F16_TYPE || get_type2() == F32_TYPE ||
        get_type2() == F64_TYPE || get_type2() == FF64_TYPE) {
      oprnd_type = FP_OP;    // [한국어] 소스가 부동소수점 타입 → FP 연산 분류
    } else
      oprnd_type = INT_OP;   // [한국어] 소스가 정수 타입 → INT 연산 분류

  } else {
    // [한국어] 일반 연산: 주 데이터 타입(get_type())으로 분류
    if (get_type() == F16_TYPE || get_type() == F32_TYPE ||
        get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      oprnd_type = FP_OP;    // [한국어] F16/F32/F64/FF64(double-double) → FP 연산
    } else
      oprnd_type = INT_OP;   // [한국어] B8/B16/B32/B64/U/S 등 정수 타입 → INT 연산
  }
}

/*
 * [한국어]
 * ptx_instruction::set_mul_div_or_other_archop - 명령어를 SP 파이프라인 세부 타입으로 분류
 *
 * @return: 없음 (void). 결과는 멤버 sp_op에 저장됨.
 *
 * sp_op(special purpose operation) 필드를 설정하여 이 명령어가 어느 실행 유닛에서
 * 처리되어야 하는지 세부 분류한다. AccelWattch 전력 모델과 warp_inst_t의 자원 추적에 사용.
 * 분류 체계 (데이터 타입 × 옵코드):
 *   F64/FF64 타입:
 *     MUL/MAD/FMA → DP_MUL_OP (배정밀도 곱셈기)
 *     DIV/REM/RCP → DP_DIV_OP (배정밀도 나눗셈기)
 *     LG2        → FP_LG_OP   (로그 SFU)
 *     RSQRT/SQRT → FP_SQRT_OP (제곱근 SFU)
 *     SIN/COS    → FP_SIN_OP  (삼각함수 SFU)
 *     EX2        → FP_EXP_OP  (지수 SFU)
 *     MMA        → TENSOR__OP (Tensor Core)
 *     TEX        → TEX__OP    (텍스처 유닛)
 *   F16/F32 타입: FP_MUL_OP, FP_DIV_OP, FP_LG/SQRT/SIN/EXP_OP, TENSOR, TEX
 *   정수 타입:
 *     MUL24/MAD24 → INT_MUL24_OP (24비트 곱셈기)
 *     MUL/MAD/FMA U32/S32/B32 → INT_MUL32_OP (32비트 곱셈기)
 *     MUL/MAD/FMA 기타 → INT_MUL_OP (일반 정수 곱셈)
 *     DIV/REM     → INT_DIV_OP (정수 나눗셈)
 * 제어 흐름/배리어 명령어는 분류 대상 제외 (OTHER_OP 유지).
 *
 * 호출 체인:
 *   set_opcode_and_latency() → [set_mul_div_or_other_archop]
 */
void ptx_instruction::set_mul_div_or_other_archop() {
  sp_op = OTHER_OP;  // [한국어] 기본값: 특별 분류 없음 (일반 ALU 또는 제어 흐름)

  // [한국어] 제어 흐름/동기화/배리어 명령어 제외: 이들은 ALU 파이프라인을 사용하지 않음
  if ((m_opcode != MEMBAR_OP) && (m_opcode != SSY_OP) && (m_opcode != BRA_OP) &&
      (m_opcode != BAR_OP) && (m_opcode != EXIT_OP) && (m_opcode != NOP_OP) &&
      (m_opcode != RETP_OP) && (m_opcode != RET_OP) && (m_opcode != CALLP_OP) &&
      (m_opcode != CALL_OP)) {
    if (get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      // [한국어] 배정밀도(F64/FF64) 타입: DP 파이프라인 세부 분류
      switch (get_opcode()) {
        case MUL_OP:
        case MAD_OP:
        case FMA_OP:
          sp_op = DP_MUL_OP;    // [한국어] 배정밀도 곱셈 연산 (DP MUL 파이프라인)
          break;
        case DIV_OP:
        case REM_OP:
          sp_op = DP_DIV_OP;    // [한국어] 배정밀도 나눗셈/나머지 (DP DIV 파이프라인)
          break;
        case RCP_OP:
          sp_op = DP_DIV_OP;    // [한국어] 역수(reciprocal) — 나눗셈과 동일 파이프라인
          break;
        case LG2_OP:
          sp_op = FP_LG_OP;     // [한국어] 이진 로그(log2) — SFU 로그 파이프라인
          break;
        case RSQRT_OP:
        case SQRT_OP:
          sp_op = FP_SQRT_OP;   // [한국어] 제곱근 / 역제곱근 — SFU SQRT 파이프라인
          break;
        case SIN_OP:
        case COS_OP:
          sp_op = FP_SIN_OP;    // [한국어] 삼각함수(sin/cos) — SFU 삼각함수 파이프라인
          break;
        case EX2_OP:
          sp_op = FP_EXP_OP;    // [한국어] 지수(2^x) — SFU 지수 파이프라인
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;   // [한국어] Tensor Core MMA 연산 (wmma.mma / mma.sync)
          break;
        case TEX_OP:
          sp_op = TEX__OP;      // [한국어] 텍스처 샘플링 — 텍스처 유닛 파이프라인
          break;
        default:
          if ((op == DP_OP) || (op == ALU_OP)) sp_op = DP___OP;  // [한국어] 기타 DP 연산
          break;
      }
    } else if (get_type() == F16_TYPE || get_type() == F32_TYPE) {
      // [한국어] 단정밀도(F32) / 반정밀도(F16) 타입: SP 파이프라인 세부 분류
      switch (get_opcode()) {
        case MUL_OP:
        case MAD_OP:
        case FMA_OP:
          sp_op = FP_MUL_OP;    // [한국어] 단정밀도 곱셈/누산 (SP MUL 파이프라인)
          break;
        case DIV_OP:
        case REM_OP:
          sp_op = FP_DIV_OP;    // [한국어] 단정밀도 나눗셈 (SP DIV 파이프라인)
          break;
        case RCP_OP:
          sp_op = FP_DIV_OP;    // [한국어] 단정밀도 역수
          break;
        case LG2_OP:
          sp_op = FP_LG_OP;     // [한국어] 이진 로그 (SFU)
          break;
        case RSQRT_OP:
        case SQRT_OP:
          sp_op = FP_SQRT_OP;   // [한국어] 제곱근/역제곱근 (SFU)
          break;
        case SIN_OP:
        case COS_OP:
          sp_op = FP_SIN_OP;    // [한국어] 삼각함수 (SFU)
          break;
        case EX2_OP:
          sp_op = FP_EXP_OP;    // [한국어] 지수 (SFU)
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;   // [한국어] Tensor Core (F16 MMA)
          break;
        case TEX_OP:
          sp_op = TEX__OP;      // [한국어] 텍스처 샘플링
          break;
        default:
          if ((op == SP_OP) || (op == ALU_OP)) sp_op = FP__OP;  // [한국어] 기타 단정밀도 ALU 연산
          break;
      }
    } else {
      // [한국어] 정수 타입 (B8/B16/B32/S8/S16/S32/U8/U16/U32 등): INT 파이프라인 세부 분류
      switch (get_opcode()) {
        case MUL24_OP:
        case MAD24_OP:
          sp_op = INT_MUL24_OP;  // [한국어] 24비트 정수 곱셈 — 구형 GPU(compute 1.x)에서 빠름
          break;
        case MUL_OP:
        case MAD_OP:
        case FMA_OP:
          if (get_type() == U32_TYPE || get_type() == S32_TYPE ||
              get_type() == B32_TYPE)
            sp_op = INT_MUL32_OP;   // [한국어] 32비트 정수 곱셈 (INT MUL32 파이프라인)
          else
            sp_op = INT_MUL_OP;     // [한국어] 기타 정수 너비 곱셈 (16비트 이하 등)
          break;
        case DIV_OP:
        case REM_OP:
          sp_op = INT_DIV_OP;    // [한국어] 정수 나눗셈/나머지 (정수 DIV 파이프라인, 레이턴시 높음)
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;   // [한국어] 정수 Tensor Core (INT8 mma.sync)
          break;
        case TEX_OP:
          sp_op = TEX__OP;      // [한국어] 텍스처 샘플링 (정수 텍스처 포맷)
          break;
        default:
          if ((op == INTP_OP) || (op == ALU_OP)) sp_op = INT__OP;  // [한국어] 기타 정수 ALU 연산
          break;
      }
    }
  }
}

/*
 * [한국어]
 * ptx_instruction::set_bar_type - BAR 명령어의 배리어 타입을 설정
 *
 * @return: 없음 (void). 결과는 멤버 bar_type 및 red_type에 저장됨.
 *
 * PTX BAR(barrier) 명령어의 세부 타입을 결정한다:
 *   BAR.SYNC  → SYNC  : 모든 스레드가 이 배리어에 도달할 때까지 대기 (__syncthreads)
 *   BAR.ARRIVE→ ARRIVE: 도달 신호만 보내고 대기하지 않음 (비대칭 배리어)
 *   BAR.RED   → RED   : 배리어 + 원자적 리덕션 연산 동시 수행
 *     RED 세부 타입: POPC_RED(popcount), AND_RED(비트AND), OR_RED(비트OR)
 *   SST_OP    → SYNC  : warp-level split-phase barrier (PTXPlus)
 * 타이밍 모델(shader.cc)이 이 타입에 따라 배리어 완료 조건을 판단한다.
 * 알 수 없는 BAR 옵션은 abort()로 프로그램 종료.
 *
 * 호출 체인:
 *   set_opcode_and_latency() → [set_bar_type]
 */
void ptx_instruction::set_bar_type() {
  if (m_opcode == BAR_OP) {         // [한국어] BAR(barrier) 명령어인 경우 세부 타입 분류
    switch (m_barrier_op) {
      case SYNC_OPTION:
        bar_type = SYNC;             // [한국어] BAR.SYNC — __syncthreads() 구현. CTA 내 모든 스레드 대기
        break;
      case ARRIVE_OPTION:
        bar_type = ARRIVE;           // [한국어] BAR.ARRIVE — 도달 신호만 보내고 계속 실행 (비대칭 동기화)
        break;
      case RED_OPTION:
        bar_type = RED;              // [한국어] BAR.RED — 배리어 + 리덕션 연산 (popcount/and/or)
        switch (m_atomic_spec) {     // [한국어] 리덕션 연산 종류 결정
          case ATOMIC_POPC:
            red_type = POPC_RED;     // [한국어] BAR.RED.POPC — 활성 스레드 수 카운트
            break;
          case ATOMIC_AND:
            red_type = AND_RED;      // [한국어] BAR.RED.AND — 비트 AND 리덕션
            break;
          case ATOMIC_OR:
            red_type = OR_RED;       // [한국어] BAR.RED.OR — 비트 OR 리덕션
            break;
        }
        break;
      default:
        abort();                     // [한국어] 알 수 없는 BAR 옵션: 구현되지 않은 PTX 확장 → 종료
    }
  } else if (m_opcode == SST_OP) {  // [한국어] SST(split-phase synchronize-threads): PTXPlus 전용
    bar_type = SYNC;                 // [한국어] SST도 SYNC 배리어로 취급 (CTA 내 동기화)
  }
}

/*
 * [한국어]
 * ptx_instruction::set_opcode_and_latency - 명령어의 op 타입, latency, initiation_interval 결정
 *
 * @return: 없음 (void). 결과는 op, latency, initiation_interval, mem_op, num_operands,
 *          num_regs, oprnd_type, sp_op, bar_type, red_type 필드에 저장됨.
 *
 * 이 함수는 타이밍 모델이 명령어를 스케줄링할 때 필요한 모든 시간 관련 정보를 결정하는
 * **핵심 함수**이다. pre_decode()에서 호출된다.
 * 동작 단계:
 *   1. gpgpusim.config의 레이턴시 문자열을 sscanf로 파싱하여 int/fp/dp/sfu/tensor 배열 구성
 *   2. CDP 레이턴시 문자열도 파싱하여 cdp_latency[] 전역 배열에 저장
 *   3. 피연산자 수(num_operands) 및 레지스터 피연산자 수(num_regs) 집계
 *   4. 옵코드와 데이터 타입에 따라 op(실행 파이프라인), latency, initiation_interval 결정
 *   5. set_fp_or_int_archop() / set_mul_div_or_other_archop() / set_bar_type() 호출로 세부 분류
 *
 * latency(레이턴시): 명령어 실행 결과가 다음 명령어에서 사용 가능해질 때까지의 사이클 수.
 * initiation_interval(II): 동일 파이프라인에서 다음 명령어를 발행하기까지의 최소 사이클 간격.
 * op(파이프라인 타입): LOAD_OP, STORE_OP, BRANCH_OP, BARRIER_OP, ALU_OP, SP_OP, DP_OP,
 *                    SFU_OP, TENSOR_CORE_OP, CALL_OPS, RET_OPS 등.
 *
 * 실행 컨텍스트: PTX 로딩 시점(pre_decode 호출 시) 또는 런타임 디코딩 시. 싱글스레드.
 * 멀티스레드 비고: gpgpu_ctx->func_sim 멤버를 읽기만 하므로 락 불필요.
 *
 * 호출 체인:
 *   pre_decode() → [set_opcode_and_latency]
 *     → set_fp_or_int_archop()
 *     → set_mul_div_or_other_archop()
 *     (set_bar_type()은 pre_decode()에서 별도 호출)
 */
void ptx_instruction::set_opcode_and_latency() {
  // [한국어] 레이턴시/II 값을 담을 로컬 배열 — config 문자열을 파싱해서 채움
  unsigned int_latency[6];    // [한국어] 정수 연산 레이턴시 [ADD, MAX, MUL, MAD, DIV, SHFL]
  unsigned fp_latency[5];     // [한국어] F32 연산 레이턴시 [ADD, MAX, MUL, MAD, DIV]
  unsigned dp_latency[5];     // [한국어] F64 연산 레이턴시 [ADD, MAX, MUL, MAD, DIV]
  unsigned sfu_latency;       // [한국어] SFU 연산 레이턴시 (sin/cos/sqrt/log2/ex2/rcp/rsqrt)
  unsigned tensor_latency;    // [한국어] Tensor Core(MMA) 연산 레이턴시
  unsigned int_init[6];       // [한국어] 정수 연산 initiation interval
  unsigned fp_init[5];        // [한국어] F32 연산 initiation interval
  unsigned dp_init[5];        // [한국어] F64 연산 initiation interval
  unsigned sfu_init;          // [한국어] SFU 연산 initiation interval
  unsigned tensor_init;       // [한국어] Tensor Core 연산 initiation interval
  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  // [한국어] gpgpusim.config의 레이턴시 문자열을 파싱 — "1,1,19,25,145,32" 형식
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);   // [한국어] SFU 레이턴시 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency); // [한국어] Tensor Core 레이턴시 파싱
  // [한국어] initiation interval 문자열 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_int, "%u,%u,%u,%u,%u,%u",
         &int_init[0], &int_init[1], &int_init[2], &int_init[3], &int_init[4],
         &int_init[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_fp, "%u,%u,%u,%u,%u",
         &fp_init[0], &fp_init[1], &fp_init[2], &fp_init[3], &fp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_dp, "%u,%u,%u,%u,%u",
         &dp_init[0], &dp_init[1], &dp_init[2], &dp_init[3], &dp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_sfu, "%u", &sfu_init);   // [한국어] SFU II 파싱
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_tensor, "%u", &tensor_init); // [한국어] Tensor Core II 파싱
  // [한국어] CDP(CUDA Dynamic Parallelism) API 레이턴시 파싱 — GPU에서 cudaLaunchDevice 등 호출 시
  sscanf(gpgpu_ctx->func_sim->cdp_latency_str, "%u,%u,%u,%u,%u",
         &gpgpu_ctx->func_sim->cdp_latency[0],
         &gpgpu_ctx->func_sim->cdp_latency[1],
         &gpgpu_ctx->func_sim->cdp_latency[2],
         &gpgpu_ctx->func_sim->cdp_latency[3],
         &gpgpu_ctx->func_sim->cdp_latency[4]);

  // [한국어] 피연산자 집계: 첫 번째 피연산자(목적지)를 건너뛰고 나머지 소스 피연산자 수 카운트
  if (!m_operands.empty()) {
    std::vector<operand_info>::iterator it;
    for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      num_operands++;                          // [한국어] 소스 피연산자 수 증가
      if ((it->is_reg() || it->is_vector())) {
        num_regs++;                            // [한국어] 레지스터/벡터 피연산자 수 증가 (scoreboard 의존성 추적)
      }
    }
  }
  // [한국어] 기본값 설정: 분류되지 않은 명령어는 레이턴시 1사이클의 ALU 연산으로 처리
  op = ALU_OP;                     // [한국어] 기본 실행 파이프라인: 일반 ALU
  mem_op = NOT_TEX;                // [한국어] 기본 메모리 연산 타입: 텍스처 아님
  initiation_interval = latency = 1;  // [한국어] 기본 레이턴시/II: 1사이클

  // [한국어] 옵코드별 op 타입 및 레이턴시/II 결정
  switch (m_opcode) {
    case MOV_OP:
      // [한국어] MOV: 메모리 읽기가 있으면 로드, 쓰기가 있으면 스토어, 둘 다면 버그
      assert(!(has_memory_read() && has_memory_write()));
      if (has_memory_read()) op = LOAD_OP;    // [한국어] shared/global/local에서 레지스터로 이동
      if (has_memory_write()) op = STORE_OP;  // [한국어] 레지스터에서 메모리로 이동
      break;
    case LD_OP:
      op = LOAD_OP;                           // [한국어] LD: 명시적 메모리 로드 명령어
      break;
    case MMA_LD_OP:
      op = TENSOR_CORE_LOAD_OP;              // [한국어] MMA_LD: Tensor Core 레지스터 파일 로드
      break;
    case LDU_OP:
      op = LOAD_OP;                           // [한국어] LDU: uniform load (읽기 전용 캐시 경유)
      break;
    case ST_OP:
      op = STORE_OP;                          // [한국어] ST: 명시적 메모리 스토어 명령어
      break;
    case MMA_ST_OP:
      op = TENSOR_CORE_STORE_OP;             // [한국어] MMA_ST: Tensor Core 레지스터 파일 스토어
      break;
    case BRA_OP:
      op = BRANCH_OP;                         // [한국어] BRA: 조건부/무조건 분기 — SIMT 스택에서 처리
      break;
    case BREAKADDR_OP:
      op = BRANCH_OP;                         // [한국어] BREAKADDR: 루프 break 대상 주소 설정 (분기처럼 처리)
      break;
    case TEX_OP:
      op = LOAD_OP;                           // [한국어] TEX: 텍스처 샘플링 — 로드 연산 카테고리
      mem_op = TEX;                           // [한국어] 텍스처 메모리 연산으로 마크 (텍스처 캐시 경유)
      break;
    case ATOM_OP:
      op = LOAD_OP;                           // [한국어] ATOM: 원자적 연산 — 로드+modify+스토어 (로드로 분류)
      break;
    case BAR_OP:
      op = BARRIER_OP;                        // [한국어] BAR: 배리어 — __syncthreads() 등
      break;
    case SST_OP:
      op = BARRIER_OP;                        // [한국어] SST: split-phase 배리어 (PTXPlus)
      break;
    case MEMBAR_OP:
      op = MEMORY_BARRIER_OP;                 // [한국어] MEMBAR: 메모리 순서 보장 펜스 (membar.cta/.gl/.sys)
      break;
    case CALL_OP: {
      // [한국어] CALL: printf 또는 CDP API 호출이면 ALU로 처리, 일반 함수 호출이면 CALL_OPS
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;                          // [한국어] printf/CDP는 기능 시뮬레이터가 직접 처리 (타이밍 무관)
      } else
        op = CALL_OPS;                        // [한국어] 일반 함수 호출: 스택 관리 필요
      break;
    }
    case CALLP_OP: {
      // [한국어] CALLP: 간접 함수 호출 — printf/CDP면 ALU, 아니면 CALL_OPS
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;
      } else
        op = CALL_OPS;
      break;
    }
    case RET_OP:
    case RETP_OP:
      op = RET_OPS;                           // [한국어] RET/RETP: 함수 반환 — 스택 복원 필요
      break;
    case ADD_OP:
    case ADDP_OP:
    case ADDC_OP:
    case SUB_OP:
    case SUBC_OP:
      // ADD,SUB latency
      // [한국어] ADD/SUB 계열: 데이터 타입에 따라 파이프라인과 레이턴시 결정
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[0];            // [한국어] F32 ADD/SUB 레이턴시 (기본 1사이클)
          initiation_interval = fp_init[0];   // [한국어] F32 ADD/SUB II (기본 1사이클)
          op = SP_OP;                         // [한국어] 단정밀도 파이프라인 (SP=Single Precision)
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[0];            // [한국어] F64 ADD/SUB 레이턴시 (기본 8사이클)
          initiation_interval = dp_init[0];   // [한국어] F64 ADD/SUB II (기본 8사이클)
          op = DP_OP;                         // [한국어] 배정밀도 파이프라인 (DP=Double Precision)
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[0];           // [한국어] 정수 ADD/SUB 레이턴시 (기본 1사이클)
          initiation_interval = int_init[0];  // [한국어] 정수 ADD/SUB II (기본 1사이클)
          op = INTP_OP;                       // [한국어] 정수 파이프라인 (INTP=Integer Pipelined)
          break;
      }
      break;
    case MAX_OP:
    case MIN_OP:
      // MAX,MIN latency
      // [한국어] MAX/MIN: [1] 인덱스 사용 (ADD/SUB와 동일 파이프라인이지만 별도 레이턴시 설정 가능)
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[1];            // [한국어] F32 MAX/MIN 레이턴시
          initiation_interval = fp_init[1];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[1];            // [한국어] F64 MAX/MIN 레이턴시
          initiation_interval = dp_init[1];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[1];           // [한국어] 정수 MAX/MIN 레이턴시
          initiation_interval = int_init[1];
          op = INTP_OP;
          break;
      }
      break;
    case MUL_OP:
      // MUL latency
      // [한국어] MUL: [2] 인덱스 — 곱셈은 일반적으로 ADD보다 레이턴시가 높음 (기본 19사이클)
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[2];            // [한국어] F32 MUL 레이턴시 (기본 1사이클 — 파이프라인됨)
          initiation_interval = fp_init[2];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[2];            // [한국어] F64 MUL 레이턴시 (기본 8사이클)
          initiation_interval = dp_init[2];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[2];           // [한국어] 정수 MUL 레이턴시 (기본 19사이클)
          initiation_interval = int_init[2];  // [한국어] 정수 MUL II (기본 4사이클)
          op = INTP_OP;
          break;
      }
      break;
    case MAD_OP:
    case MADC_OP:
    case MADP_OP:
    case FMA_OP:
      // MAD latency
      // [한국어] MAD/FMA: [3] 인덱스 — multiply-accumulate, MUL보다 레이턴시가 높음 (기본 25사이클)
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[3];            // [한국어] F32 MAD/FMA 레이턴시 (기본 1사이클)
          initiation_interval = fp_init[3];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[3];            // [한국어] F64 MAD/FMA 레이턴시 (기본 8사이클)
          initiation_interval = dp_init[3];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[3];           // [한국어] 정수 MAD 레이턴시 (기본 25사이클)
          initiation_interval = int_init[3];  // [한국어] 정수 MAD II (기본 4사이클)
          op = INTP_OP;
          break;
      }
      break;
    case MUL24_OP:  // MUL24 is performed on mul32 units (with additional
                    // instructions for bitmasking) on devices with compute
                    // capability >1.x
      // [한국어] MUL24: compute 1.x에서는 전용 24비트 곱셈기 사용, 이후 버전에서는 mul32+마스킹
      // MUL 레이턴시에 +1 사이클 추가 (마스킹 연산 비용)
      latency = int_latency[2] + 1;           // [한국어] MUL24 레이턴시 = 정수 MUL + 1
      initiation_interval = int_init[2] + 1;  // [한국어] MUL24 II = 정수 MUL II + 1
      op = INTP_OP;
      break;
    case MAD24_OP:
      // [한국어] MAD24: 24비트 곱셈 후 누산. MAD 레이턴시에 +1
      latency = int_latency[3] + 1;           // [한국어] MAD24 레이턴시 = 정수 MAD + 1
      initiation_interval = int_init[3] + 1;  // [한국어] MAD24 II = 정수 MAD II + 1
      op = INTP_OP;
      break;
    case DIV_OP:
    case REM_OP:
      // Floating point only
      // [한국어] DIV/REM: [4] 인덱스 — 나눗셈은 SFU에서 처리 (레이턴시 매우 높음: 정수 145, FP32 30, FP64 335)
      op = SFU_OP;                            // [한국어] SFU(Special Function Unit) 파이프라인 사용
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[4];            // [한국어] F32 DIV 레이턴시 (기본 30사이클)
          initiation_interval = fp_init[4];   // [한국어] F32 DIV II (기본 5사이클)
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[4];            // [한국어] F64 DIV 레이턴시 (기본 335사이클 — 매우 높음)
          initiation_interval = dp_init[4];   // [한국어] F64 DIV II (기본 130사이클)
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[4];           // [한국어] 정수 DIV 레이턴시 (기본 145사이클)
          initiation_interval = int_init[4];  // [한국어] 정수 DIV II (기본 32사이클)
          break;
      }
      break;
    case SQRT_OP:
    case SIN_OP:
    case COS_OP:
    case EX2_OP:
    case LG2_OP:
    case RSQRT_OP:
    case RCP_OP:
      // [한국어] SFU 연산: SQRT, SIN, COS, EX2(2^x), LG2(log2), RSQRT(1/sqrt), RCP(1/x)
      // 모두 동일한 SFU 레이턴시/II 사용 (기본 8사이클/8사이클)
      latency = sfu_latency;                  // [한국어] SFU 공통 레이턴시
      initiation_interval = sfu_init;         // [한국어] SFU 공통 II
      op = SFU_OP;                            // [한국어] SFU 파이프라인
      break;
    case MMA_OP:
      // [한국어] MMA(Matrix Multiply-Accumulate): Tensor Core 연산 (wmma.mma / mma.sync)
      latency = tensor_latency;               // [한국어] Tensor Core 레이턴시 (기본 64사이클)
      initiation_interval = tensor_init;      // [한국어] Tensor Core II (기본 64사이클)
      op = TENSOR_CORE_OP;                    // [한국어] Tensor Core 파이프라인 (WMMA/TC)
      break;
    case SHFL_OP:
      // [한국어] SHFL: warp shuffle 연산 — [5] 인덱스 (기본 레이턴시 32사이클, II 4사이클)
      // warp 내 스레드 간 레지스터 값 교환 (shfl.sync, shfl.up, shfl.down, shfl.bfly)
      latency = int_latency[5];              // [한국어] SHFL 레이턴시 (기본 32사이클)
      initiation_interval = int_init[5];     // [한국어] SHFL II (기본 4사이클)
      // [한국어] op는 기본값 ALU_OP 유지 (SHFL은 레지스터 파일 내 연산이므로 ALU 분류)
      break;
    default:
      // [한국어] 그 외 명령어: 기본값(ALU_OP, 레이턴시 1, II 1) 유지
      break;
  }
  set_fp_or_int_archop();         // [한국어] oprnd_type 설정: FP_OP / INT_OP / UN_OP
  set_mul_div_or_other_archop();  // [한국어] sp_op 설정: 세부 실행 유닛 분류
}

/*
 * [한국어]
 * ptx_thread_info::ptx_fetch_inst - 현재 스레드의 PC에서 명령어를 가져옴
 *
 * @inst: (출력) 명령어를 복사할 inst_t 레퍼런스
 * @return: 없음 (void)
 *
 * 현재 스레드의 PC(get_pc())를 사용하여 해당 함수의 명령어 메모리에서
 * ptx_instruction을 가져와 inst_t 형태로 복사한다.
 * 순수 기능 시뮬레이션 모드(functionalCoreSim)에서 사용된다.
 * assert(inst.valid())로 가져온 명령어의 유효성을 검증한다.
 *
 * 호출 체인:
 *   functionalCoreSim::executeWarp() → getExecuteWarp() → [ptx_fetch_inst]
 */
void ptx_thread_info::ptx_fetch_inst(inst_t &inst) const {
  addr_t pc = get_pc();                                 // [한국어] 현재 스레드의 프로그램 카운터
  const ptx_instruction *pI = m_func_info->get_instruction(pc);  // [한국어] 함수 명령어 메모리에서 PC에 해당하는 명령어 조회
  inst = (const inst_t &)*pI;                           // [한국어] ptx_instruction을 inst_t로 슬라이스 복사 (기본 클래스 복사)
  assert(inst.valid());                                 // [한국어] 유효하지 않은 명령어(NULL 슬롯 등)이면 버그
}

/*
 * [한국어]
 * datatype2size - PTX 데이터 타입 enum을 바이트 크기로 변환 (static 내부 함수)
 *
 * @data_type: PTX 타입 enum (B8_TYPE, S16_TYPE, F32_TYPE, B64_TYPE 등)
 * @return:    해당 타입의 크기(바이트): 1, 2, 4, 8, 또는 16
 *
 * pre_decode()와 ptx_exec_inst()에서 메모리 접근 크기(data_size)를 결정할 때 사용한다.
 * 알 수 없는 타입이 입력되면 assert(0)으로 프로그램 종료.
 * static으로 선언되어 이 번역 단위 내에서만 사용됨.
 *
 * 타입 매핑:
 *   B8/S8/U8   → 1바이트
 *   B16/S16/U16/F16 → 2바이트
 *   B32/S32/U32/F32 → 4바이트
 *   B64/BB64/S64/U64/F64/FF64 → 8바이트
 *   BB128       → 16바이트
 *
 * 호출 체인:
 *   pre_decode() → [datatype2size] (data_size 필드 설정)
 *   ptx_exec_inst() → [datatype2size] (insn_data_size 계산)
 */
static unsigned datatype2size(unsigned data_type) {
  unsigned data_size;
  switch (data_type) {
    case B8_TYPE:
    case S8_TYPE:
    case U8_TYPE:
      data_size = 1;       // [한국어] 8비트 타입 → 1바이트
      break;
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
    case F16_TYPE:
      data_size = 2;       // [한국어] 16비트 타입 (반정밀도 F16 포함) → 2바이트
      break;
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
    case F32_TYPE:
      data_size = 4;       // [한국어] 32비트 타입 (단정밀도 F32 포함) → 4바이트
      break;
    case B64_TYPE:
    case BB64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      data_size = 8;       // [한국어] 64비트 타입 (배정밀도 F64, BB64 구조체 포함) → 8바이트
      break;
    case BB128_TYPE:
      data_size = 16;      // [한국어] 128비트 타입 (벡터 레지스터 쌍) → 16바이트
      break;
    default:
      assert(0);           // [한국어] 알 수 없는 타입: 구현 누락 → 즉시 종료
      break;
  }
  return data_size;        // [한국어] 타입에 해당하는 바이트 크기 반환
}

/*
 * [한국어]
 * ptx_instruction::pre_decode - 타이밍 모델을 위한 명령어 디코딩 (핵심 함수)
 *
 * @return: 없음 (void). 결과는 inst_t 기본 클래스 필드(pc, isize, in[], out[],
 *          incount, outcount, arch_reg, pred, ar1, ar2, space, memory_op,
 *          data_size, cache_op, reconvergence_pc, m_decoded 등)에 저장됨.
 *
 * PTX 명령어를 파싱된 IR(ptx_instruction 필드)에서 타이밍 모델이 사용하는
 * warp_inst_t/inst_t 필드로 변환하는 핵심 디코딩 함수이다.
 * 처리 단계:
 *   1. 기본 필드 초기화 (pc, isize, in/out 배열, arch_reg 등)
 *   2. 메모리 접근 여부에 따라 data_size와 memory_op 설정
 *   3. X-매크로(opcodes.def)로 has_dst(목적지 레지스터 여부) 결정
 *   4. 캐시 옵션(CA/NC/CG/CS/LU/CV/WB/WT) → cache_op 매핑
 *   5. set_opcode_and_latency()로 op 타입, latency, initiation_interval 결정
 *   6. set_bar_type()으로 배리어 세부 타입 결정
 *   7. 목적지/소스 레지스터 피연산자를 out[]/in[]/arch_reg 배열로 추출
 *      - 스칼라 레지스터: out[0], in[0..2], arch_reg.dst/src
 *      - 벡터 레지스터: 최대 8개 원소까지 out[]/in[] 배열로 확장
 *   8. outcount/incount 계산 (스코어보드 의존성 추적용)
 *   9. predicate 레지스터 번호 추출
 *   10. 메모리 피연산자 내 주소 레지스터(ar1, ar2) 추출 (PTX/PTXPlus 형식)
 *   11. get_converge_point(pc)로 PDOM 재합류 PC 조회 후 reconvergence_pc 설정
 *   12. m_decoded = true로 마크 (중복 디코딩 방지)
 *
 * 타이밍 모델(shader.cc)이 이 함수의 결과를 사용하여:
 *   - in[]/out[]: 스코어보드 RAW 해저드 감지
 *   - reconvergence_pc: SIMT 스택 분기 합류 지점
 *   - latency/initiation_interval: 파이프라인 발행 스케줄링
 *   - memory_op/data_size: 메모리 접근 패킷(mem_fetch) 생성
 *
 * 실행 컨텍스트: 처음 실행될 때 on-demand로 호출됨 (혹은 PTX 로딩 시 일괄 처리).
 * 멀티스레드 비고: 명령어 객체는 공유되지 않으므로 별도 락 불필요.
 *
 * 호출 체인:
 *   shader.cc::shader_core_ctx::fetch() / functionalCoreSim::execute()
 *     → [pre_decode]
 *         → set_opcode_and_latency() → set_fp_or_int_archop() + set_mul_div_or_other_archop()
 *         → set_bar_type()
 *         → get_converge_point() (PDOM 재합류 PC 조회)
 */
void ptx_instruction::pre_decode() {
  pc = m_PC;              // [한국어] 타이밍 모델용 PC 필드 설정 (ptx_instruction의 m_PC → inst_t의 pc)
  isize = m_inst_size;    // [한국어] 명령어 크기(슬롯 수) 복사 (기본 1, 멀티슬롯 명령어는 >1)

  // [한국어] 출력 레지스터 배열 초기화 (0 = 미사용)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  // [한국어] 입력 레지스터 배열 초기화 (0 = 미사용)
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }
  incount = 0;            // [한국어] 소스 레지스터 수 카운터 초기화
  outcount = 0;           // [한국어] 목적지 레지스터 수 카운터 초기화
  is_vectorin = 0;        // [한국어] 벡터 목적지(ld.v4 등) 플래그 초기화
  is_vectorout = 0;       // [한국어] 벡터 소스(st.v4/tex 등) 플래그 초기화
  std::fill_n(arch_reg.src, MAX_REG_OPERANDS, -1);  // [한국어] 아키텍처 소스 레지스터 번호 초기화 (-1 = 미사용)
  std::fill_n(arch_reg.dst, MAX_REG_OPERANDS, -1);  // [한국어] 아키텍처 목적지 레지스터 번호 초기화
  pred = 0;               // [한국어] predicate 레지스터 번호 초기화 (0 = 조건 없음)
  ar1 = 0;                // [한국어] 첫 번째 주소 레지스터 초기화 (메모리 피연산자의 base reg)
  ar2 = 0;                // [한국어] 두 번째 주소 레지스터 초기화 (PTXPlus 2-reg addressing용)
  space = m_space_spec;   // [한국어] 메모리 공간 명세(shared/global/local/const 등) 복사
  memory_op = no_memory_op;  // [한국어] 기본값: 메모리 연산 없음
  data_size = 0;          // [한국어] 메모리 접근 데이터 크기 초기화

  // [한국어] 메모리 접근이 있는 명령어: data_size와 memory_op 설정
  if (has_memory_read() || has_memory_write()) {
    unsigned to_type = get_type();                                    // [한국어] 명령어의 데이터 타입 (B32, F32, U64 등)
    data_size = datatype2size(to_type);                               // [한국어] 타입 → 바이트 크기 변환
    memory_op = has_memory_read() ? memory_load : memory_store;      // [한국어] 읽기 → memory_load, 쓰기 → memory_store
  }

  bool has_dst = false;   // [한국어] 이 명령어가 목적지 레지스터를 가지는지 여부

  // [한국어] X-매크로(opcodes.def)를 사용하여 옵코드별 has_dst 결정
  // OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) — DST가 0이면 목적지 없음 (ST, BR, BAR 등)
  switch (get_opcode()) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    has_dst = (DST != 0);                          \  // [한국어] DST != 0이면 목적지 레지스터를 기록하는 명령어
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    has_dst = (DST != 0);                            \  // [한국어] OP_W_DEF: warp_inst_t를 인수로 받는 명령어 (VOTE, ACTIVEMASK 등)
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
    default:
      printf("Execution error: Invalid opcode (0x%x)\n", get_opcode());  // [한국어] 알 수 없는 옵코드: 구현 누락
      break;
  }

  // [한국어] 캐시 옵션을 cache_op 필드로 매핑
  // PTX ld/st 명령어는 .ca/.nc/.cg/.cs/.lu/.cv/.wb/.wt 수식어로 캐시 정책을 지정한다.
  switch (m_cache_option) {
    case CA_OPTION:
      cache_op = CACHE_ALL;          // [한국어] .ca: 모든 레벨에서 캐시 (L1+L2)
      break;
    case NC_OPTION:
      cache_op = CACHE_L1;           // [한국어] .nc: L1 캐시(비일관성, non-coherent 읽기 전용)
      break;
    case CG_OPTION:
      cache_op = CACHE_GLOBAL;       // [한국어] .cg: 글로벌 레벨만 캐시 (L2만, L1 바이패스)
      break;
    case CS_OPTION:
      cache_op = CACHE_STREAMING;    // [한국어] .cs: 스트리밍 데이터 (최저 우선순위 캐시)
      break;
    case LU_OPTION:
      cache_op = CACHE_LAST_USE;     // [한국어] .lu: 마지막 사용 표시 (재사용 없음, 즉시 evict)
      break;
    case CV_OPTION:
      cache_op = CACHE_VOLATILE;     // [한국어] .cv: volatile 접근 (캐시 무효화, 매번 메모리에서 읽기)
      break;
    case WB_OPTION:
      cache_op = CACHE_WRITE_BACK;   // [한국어] .wb: write-back 쓰기 (L1에 쓰고 dirty 마크)
      break;
    case WT_OPTION:
      cache_op = CACHE_WRITE_THROUGH; // [한국어] .wt: write-through 쓰기 (즉시 L2까지 전달)
      break;
    default:
      // [한국어] 캐시 옵션이 명시되지 않은 경우 옵코드별 기본 정책 적용
      // if( m_opcode == LD_OP || m_opcode == LDU_OP )
      if (m_opcode == MMA_LD_OP || m_opcode == LD_OP || m_opcode == LDU_OP)
        cache_op = CACHE_ALL;          // [한국어] 기본 로드: 모든 레벨 캐시
      // else if( m_opcode == ST_OP )
      else if (m_opcode == MMA_ST_OP || m_opcode == ST_OP)
        cache_op = CACHE_WRITE_BACK;   // [한국어] 기본 스토어: write-back
      else if (m_opcode == ATOM_OP)
        cache_op = CACHE_GLOBAL;       // [한국어] 원자 연산: 글로벌 레벨 (일관성 보장)
      break;
  }

  set_opcode_and_latency();            // [한국어] op 타입, latency, initiation_interval, mem_op 등 설정
  set_bar_type();                      // [한국어] BAR/SST 명령어의 bar_type/red_type 설정

  // Get register operands
  // [한국어] 피연산자 목록을 순회하여 in[]/out[]/arch_reg 배열 구성
  int n = 0, m = 0;  // [한국어] n: 전체 피연산자 인덱스, m: 소스 레지스터 인덱스
  ptx_instruction::const_iterator opr = op_iter_begin();
  for (; opr != op_iter_end(); opr++, n++) {  // process operands
    const operand_info &o = *opr;             // [한국어] 현재 피연산자 정보

    if (has_dst && n == 0) {
      // [한국어] 첫 번째 피연산자 = 목적지 레지스터 (has_dst가 true일 때만)
      // Do not set the null register "_" as an architectural register
      if (o.is_reg() && !o.is_non_arch_reg()) {
        // [한국어] 스칼라 레지스터: 목적지 레지스터 번호와 아키텍처 레지스터 번호 저장
        out[0] = o.reg_num();              // [한국어] 목적지 레지스터 번호 (스코어보드에서 WAW 감지 등)
        arch_reg.dst[0] = o.arch_reg_num(); // [한국어] 아키텍처 레지스터 번호 (레지스터 파일 뱅크 식별)
      } else if (o.is_vector()) {
        // [한국어] 벡터 목적지 (ld.v4, mma_ld 등): 최대 8개 레지스터를 out[] 배열에 확장
        is_vectorin = 1;                   // [한국어] 벡터 목적지 마크 (is_vectorin: 로드 벡터 플래그)
        unsigned num_elem = o.get_vect_nelem();  // [한국어] 벡터 원소 수 (2, 4, 8 등)
        if (num_elem >= 1) out[0] = o.reg1_num();   // [한국어] 벡터 첫 번째 원소 레지스터 번호
        if (num_elem >= 2) out[1] = o.reg2_num();   // [한국어] 벡터 두 번째 원소
        if (num_elem >= 3) out[2] = o.reg3_num();
        if (num_elem >= 4) out[3] = o.reg4_num();
        if (num_elem >= 5) out[4] = o.reg5_num();
        if (num_elem >= 6) out[5] = o.reg6_num();
        if (num_elem >= 7) out[6] = o.reg7_num();
        if (num_elem >= 8) out[7] = o.reg8_num();   // [한국어] 최대 8개 원소까지 처리
        for (int i = 0; i < num_elem; i++) arch_reg.dst[i] = o.arch_reg_num(i);  // [한국어] 아키텍처 레지스터 번호도 배열로 저장
      }
    } else {
      // [한국어] 목적지 이외의 피연산자 = 소스 레지스터 또는 즉치값/메모리 피연산자
      if (o.is_reg() && !o.is_non_arch_reg()) {
        // [한국어] 스칼라 소스 레지스터: in[0..2]에 순서대로 저장 (최대 3개)
        int reg_num = o.reg_num();                  // [한국어] 소스 레지스터 번호
        arch_reg.src[m] = o.arch_reg_num();         // [한국어] 아키텍처 소스 레지스터 번호 (뱅크 충돌 감지용)
        switch (m) {
          case 0:
            in[0] = reg_num;   // [한국어] 첫 번째 소스 레지스터
            break;
          case 1:
            in[1] = reg_num;   // [한국어] 두 번째 소스 레지스터
            break;
          case 2:
            in[2] = reg_num;   // [한국어] 세 번째 소스 레지스터
            break;
          default:
            break;              // [한국어] 3개 초과: in[] 배열 범위 초과 — 기록하지 않음
        }
        m++;                   // [한국어] 소스 레지스터 인덱스 전진
      } else if (o.is_vector()) {
        // assert(m == 0); //only support 1 vector operand (for textures) right
        // now
        // [한국어] 벡터 소스 피연산자 (st.v4, tex 등): in[m..m+num_elem-1]에 확장
        is_vectorout = 1;                     // [한국어] 벡터 소스 마크 (is_vectorout: 스토어 벡터 플래그)
        unsigned num_elem = o.get_vect_nelem(); // [한국어] 벡터 원소 수
        if (num_elem >= 1) in[m + 0] = o.reg1_num();   // [한국어] 벡터 소스 원소들을 in[] 배열에 기록
        if (num_elem >= 2) in[m + 1] = o.reg2_num();
        if (num_elem >= 3) in[m + 2] = o.reg3_num();
        if (num_elem >= 4) in[m + 3] = o.reg4_num();
        if (num_elem >= 5) in[m + 4] = o.reg5_num();
        if (num_elem >= 6) in[m + 5] = o.reg6_num();
        if (num_elem >= 7) in[m + 6] = o.reg7_num();
        if (num_elem >= 8) in[m + 7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++)
          arch_reg.src[m + i] = o.arch_reg_num(i);  // [한국어] 아키텍처 소스 레지스터 번호 배열 저장
        m += num_elem;                        // [한국어] 소스 인덱스를 원소 수만큼 전진
      }
    }
  }

  // Setting number of input and output operands which is required for
  // scoreboard check
  // [한국어] 스코어보드 체크를 위한 incount/outcount 계산 (0 초과인 항목만 카운트)
  for (int i = 0; i < MAX_OUTPUT_VALUES; i++)
    if (out[i] > 0) outcount++;             // [한국어] 유효한 출력 레지스터 수 카운트

  for (int i = 0; i < MAX_INPUT_VALUES; i++)
    if (in[i] > 0) incount++;              // [한국어] 유효한 입력 레지스터 수 카운트

  // Get predicate
  // [한국어] predicate 레지스터 번호 추출 (@p 등 조건부 실행 레지스터)
  if (has_pred()) {
    const operand_info &p = get_pred();    // [한국어] predicate 피연산자 정보
    pred = p.reg_num();                    // [한국어] predicate 레지스터 번호 저장 (ptx_exec_inst에서 skip 여부 판단)
  }

  // Get address registers inside memory operands.
  // Assuming only one memory operand per instruction,
  //  and maximum of two address registers for one memory operand.
  // [한국어] 메모리 피연산자 내 주소 레지스터 추출 (ar1, ar2)
  // ar1: 베이스 주소 레지스터, ar2: PTXPlus 이중 주소 표현의 두 번째 레지스터
  // 스코어보드가 메모리 주소 계산에 필요한 레지스터의 RAW 해저드도 감지해야 하기 때문.
  if (has_memory_read() || has_memory_write()) {
    ptx_instruction::const_iterator op = op_iter_begin();
    for (; op != op_iter_end(); op++, n++) {  // process operands
      const operand_info &o = *op;

      if (o.is_memory_operand()) {
        // We do not support the null register as a memory operand
        assert(!o.is_non_arch_reg());      // [한국어] 메모리 피연산자에 null 레지스터("_") 불가

        // Check PTXPlus-type operand
        // memory operand with addressing (ex. s[0x4] or g[$r1])
        if (o.is_memory_operand2()) {
          // memory operand with one address register (ex. g[$r1+0x4] or
          // s[$r2+=0x4])
          if (o.get_double_operand_type() == 0 ||
              o.get_double_operand_type() == 3) {
            // [한국어] PTXPlus: 레지스터+오프셋 형식 (g[$r1+0x4]) — 하나의 주소 레지스터
            ar1 = o.reg_num();              // [한국어] 베이스 주소 레지스터 저장
            arch_reg.src[4] = o.arch_reg_num();  // [한국어] arch_reg.src[4]에 저장 (일반 소스[0..3] 이후)
            // TODO: address register in $r2+=0x4 should be an output register
            // as well
          }
          // memory operand with two address register (ex. s[$r1+$r1] or
          // g[$r1+=$r2])
          else if (o.get_double_operand_type() == 1 ||
                   o.get_double_operand_type() == 2) {
            // [한국어] PTXPlus: 레지스터+레지스터 형식 (s[$r1+$r2]) — 두 개의 주소 레지스터
            ar1 = o.reg1_num();             // [한국어] 첫 번째 주소 레지스터 (베이스)
            arch_reg.src[4] = o.arch_reg_num();  // [한국어] arch_reg.src[4]: 첫 번째 주소 레지스터
            ar2 = o.reg2_num();             // [한국어] 두 번째 주소 레지스터 (오프셋/인덱스)
            arch_reg.src[5] = o.arch_reg_num();  // [한국어] arch_reg.src[5]: 두 번째 주소 레지스터
            // TODO: first address register in $r1+=$r2 should be an output
            // register as well
          }
        } else if (o.is_immediate_address()) {
          // [한국어] 즉치 주소(상수 오프셋만 있는 경우): 주소 레지스터 없음 — 처리 불필요
        }
        // Regular PTX operand
        else if (o.get_symbol()
                     ->type()
                     ->get_key()
                     .is_reg()) {  // Memory operand contains a register
          // [한국어] 표준 PTX 메모리 피연산자: 레지스터가 포함된 경우 (ld.global [%rd0+4])
          ar1 = o.reg_num();              // [한국어] 주소 베이스 레지스터 저장
          arch_reg.src[4] = o.arch_reg_num();  // [한국어] arch_reg.src[4]에 저장
        }
      }
    }
  }

  // get reconvergence pc
  // [한국어] PDOM(Post-Dominator) 분석 결과에서 이 명령어의 재합류 PC 조회
  // BRA 명령어: 즉각 사후지배자(immediate postdominator) PC 반환
  // return이 재합류인 경우: RECONVERGE_RETURN_PC(-2) 반환
  // 분기가 아닌 경우: NO_BRANCH_DIVERGENCE(-1) 반환
  reconvergence_pc = gpgpu_ctx->func_sim->get_converge_point(pc);
  // [한국어] SIMT 스택이 이 값을 사용하여 워프 분기 시 재합류 지점을 스택에 push

  m_decoded = true;  // [한국어] 디코딩 완료 마크 — 중복 호출 방지 (pre_decode는 1회만 실행되어야 함)
}

/*
 * [한국어]
 * function_info::add_param_name_type_size - 커널 파라미터 메타데이터 등록
 *
 * @index:  파라미터 순서 인덱스 (0부터 시작)
 * @name:   PTX 파라미터 심볼 이름 (예: "kernelName_param_0")
 * @type:   PTX 타입 enum
 * @size:   파라미터 크기 (비트 단위)
 * @ptr:    포인터 파라미터 여부
 * @space:  메모리 공간 (param_space_kernel, shared_space 등)
 * @return: 없음 (void)
 *
 * PTX 파서가 .param 지시자를 파싱할 때 각 파라미터의 타입/크기 정보를
 * m_ptx_kernel_param_info 맵에 등록한다.
 * 파라미터 이름이 "kernelName_param_N" 형식이면 N을 인덱스로 사용,
 * 아니면 전달된 index 값을 그대로 사용한다.
 * assert: 동일 인덱스가 중복 등록되지 않도록 검증.
 *
 * 호출 체인:
 *   ptx_parser.cc (PTX 파싱 중) → [add_param_name_type_size]
 */
void function_info::add_param_name_type_size(unsigned index, std::string name,
                                             int type, size_t size, bool ptr,
                                             memory_space_t space) {
  unsigned parsed_index;                          // [한국어] 이름에서 추출한 파라미터 인덱스
  char buffer[2048];                              // [한국어] sscanf 패턴 문자열 버퍼
  snprintf(buffer, 2048, "%s_param_%%u", m_name.c_str());  // [한국어] 예: "vecAdd_param_%u" 패턴 생성
  int ntokens = sscanf(name.c_str(), buffer, &parsed_index);  // [한국어] 이름에서 인덱스 추출 시도
  if (ntokens == 1) {
    // [한국어] 표준 CUDA 이름 형식("kernelName_param_N"): 추출한 인덱스 사용
    assert(m_ptx_kernel_param_info.find(parsed_index) ==
           m_ptx_kernel_param_info.end());        // [한국어] 중복 등록 방지
    m_ptx_kernel_param_info[parsed_index] =
        param_info(name, type, size, ptr, space); // [한국어] 파라미터 정보 저장
  } else {
    // [한국어] 비표준 이름 형식(OpenCL 등): 전달된 index 값 사용
    assert(m_ptx_kernel_param_info.find(index) ==
           m_ptx_kernel_param_info.end());        // [한국어] 중복 등록 방지
    m_ptx_kernel_param_info[index] = param_info(name, type, size, ptr, space);
  }
}

/*
 * [한국어]
 * function_info::add_param_data - 실제 커널 파라미터 데이터를 파라미터 정보에 추가
 *
 * @argn: 파라미터 인덱스 (0부터 시작)
 * @args: gpgpu_ptx_sim_arg — 파라미터의 실제 데이터 포인터, 크기, 오프셋 포함
 * @return: 없음 (void)
 *
 * cuLaunchKernel() 또는 cudaLaunchKernel() 시점에 호출되어 실제 파라미터 값을
 * m_ptx_kernel_param_info에 등록된 파라미터 메타데이터에 연결한다.
 * 처리 분기:
 *   - 정규 파라미터: param_info::add_data()로 데이터 저장
 *   - OpenCL shared 포인터(is_ptr_shared): NULL 포인터 — 동적 공유 메모리 크기 처리
 *   - scratchpad 파라미터(심볼 없음): 심볼 테이블에서 심볼을 찾아 주소 설정
 *     NULL 데이터 + shared 심볼: 동적 공유 메모리 영역 할당
 *
 * 호출 체인:
 *   libcuda::cudaLaunchKernel() / cuda_sim::gpgpu_opencl_ptx_sim_init_grid()
 *     → [add_param_data]
 */
void function_info::add_param_data(unsigned argn,
                                   struct gpgpu_ptx_sim_arg *args) {
  const void *data = args->m_start;    // [한국어] 파라미터 실제 데이터 포인터 (NULL이면 동적 공유 메모리)
  if (g_debug_execution >= 3) {
    // [한국어] 디버그 출력: 4바이트이면 정수값, 아니면 포인터 주소 출력
    if (args->m_nbytes == 4)
      printf("ADD_PARAM_DATA %d\n", *((uint32_t *)data));
    else
      printf("ADD_PARAM_DATA %p\n", *((void **)data));
  }
  bool scratchpad_memory_param =
      false;  // Is this parameter in CUDA shared memory or OpenCL local memory
  // [한국어] scratchpad_memory_param: OpenCL local 메모리 또는 동적 공유 메모리 파라미터 여부

  std::map<unsigned, param_info>::iterator i =
      m_ptx_kernel_param_info.find(argn);
  if (i != m_ptx_kernel_param_info.end()) {
    if (i->second.is_ptr_shared()) {
      assert(
          args->m_start == NULL &&
          "OpenCL parameter pointer to local memory must have NULL as value");
      scratchpad_memory_param = true;
    } else {
      param_t tmp;
      tmp.pdata = args->m_start;
      tmp.size = args->m_nbytes;
      tmp.offset = args->m_offset;
      tmp.type = 0;
      i->second.add_data(tmp);
      i->second.add_offset((unsigned)args->m_offset);
    }
  } else {
    scratchpad_memory_param = true;
  }

  if (scratchpad_memory_param) {
    // This should only happen for OpenCL:
    //
    // The LLVM PTX compiler in NVIDIA's driver (version 190.29)
    // does not generate an argument in the function declaration
    // for __constant arguments.
    //
    // The associated constant memory space can be allocated in two
    // ways. It can be explicitly initialized in the .ptx file where
    // it is declared.  Or, it can be allocated using the clCreateBuffer
    // on the host. In this later case, the .ptx file will contain
    // a global declaration of the parameter, but it will have an unknown
    // array size.  Thus, the symbol's address will not be set and we need
    // to set it here before executing the PTX.

    char buffer[2048];
    snprintf(buffer, 2048, "%s_param_%u", m_name.c_str(), argn);

    symbol *p = m_symtab->lookup(buffer);
    if (p == NULL) {
      printf(
          "GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot "
          "bind buffer\n",
          buffer);
      abort();
    }
    if (data)
      p->set_address((addr_t) * (size_t *)data);
    else {
      // clSetKernelArg was passed NULL pointer for data...
      // this is used for dynamically sized shared memory on NVIDIA platforms
      bool is_ptr_shared = false;
      if (i != m_ptx_kernel_param_info.end()) {
        is_ptr_shared = i->second.is_ptr_shared();
      }

      if (!is_ptr_shared and !p->is_shared()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** clSetKernelArg passed NULL but arg not "
            "shared memory\n");
        abort();
      }
      unsigned num_bits = 8 * args->m_nbytes;
      printf(
          "GPGPU-Sim PTX: deferred allocation of shared region for \"%s\" from "
          "0x%llx to 0x%llx (shared memory space)\n",
          p->name().c_str(), m_symtab->get_shared_next(),
          m_symtab->get_shared_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr_t addr = m_symtab->get_shared_next();
      addr_t addr_pad =
          num_bits
              ? (((num_bits / 8) - (addr % (num_bits / 8))) % (num_bits / 8))
              : 0;
      p->set_address(addr + addr_pad);
      m_symtab->alloc_shared(num_bits / 8 + addr_pad);
    }
  }
}

/*
 * [한국어]
 * function_info::get_args_aligned_size - 커널 파라미터 블록의 정렬된 총 크기 계산
 *
 * @return: 파라미터 블록의 총 크기(바이트), 4바이트 워드 단위로 최종 정렬됨
 *
 * 모든 커널 파라미터를 각각의 크기에 맞게 정렬하면서 연속 배치했을 때의 총 크기를 계산한다.
 * 계산 결과는 m_args_aligned_size에 캐시되어 두 번째 호출부터는 즉시 반환한다.
 * 또한 각 파라미터의 오프셋을 계산하여 param_info에 저장하고, 심볼 테이블의 주소도 갱신한다.
 * 파라미터 정렬: 각 파라미터는 자신의 크기의 배수 주소에 배치됨 (자연 정렬).
 * 최종 크기는 4바이트(워드) 단위로 올림 정렬.
 *
 * 호출 체인:
 *   ptx_loader.cc / libcuda — 파라미터 메모리 할당 크기 결정 시
 */
unsigned function_info::get_args_aligned_size() {
  if (m_args_aligned_size >= 0) return m_args_aligned_size;  // [한국어] 캐시된 값 즉시 반환

  unsigned param_address = 0;                // [한국어] 현재 파라미터 주소 (심볼 테이블 주소 설정용)
  unsigned int total_size = 0;               // [한국어] 누적 파라미터 블록 크기
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;              // [한국어] 현재 파라미터 정보
    std::string name = p.get_name();        // [한국어] 파라미터 이름 (심볼 테이블 조회용)
    symbol *param = m_symtab->lookup(name.c_str());  // [한국어] 심볼 테이블에서 파라미터 심볼 조회

    size_t arg_size = p.get_size() / 8;  // size of param in bytes
    // [한국어] 파라미터 크기: get_size()는 비트 단위이므로 8로 나눠 바이트로 변환
    total_size = (total_size + arg_size - 1) / arg_size * arg_size;  // aligned
    // [한국어] 자연 정렬: 현재 total_size를 arg_size의 배수로 올림 (예: 3바이트 후 4바이트 파라미터 → 4로 정렬)
    p.add_offset(total_size);                   // [한국어] 파라미터 오프셋을 param_info에 저장
    param->set_address(param_address + total_size);  // [한국어] 심볼 테이블에 파라미터 주소 설정
    total_size += arg_size;                     // [한국어] 파라미터 크기만큼 총 크기 증가
  }

  m_args_aligned_size = (total_size + 3) / 4 * 4;  // final size aligned to word
  // [한국어] 최종 파라미터 블록 크기를 4바이트(워드) 단위로 올림 정렬 후 캐시

  return m_args_aligned_size;             // [한국어] 정렬된 파라미터 블록 총 크기 반환
}

/*
 * [한국어]
 * function_info::finalize - 파라미터 값을 시뮬레이션된 파라미터 메모리에 복사
 *
 * @param_mem: 커널의 파라미터 메모리 공간 (kernel_info_t::get_param_memory()로 얻음)
 * @return:    없음 (void)
 *
 * 커널 실행 직전에 호출되어, add_param_data()로 수집된 파라미터 값들을
 * 시뮬레이션된 파라미터 메모리 공간에 실제로 기록한다.
 * PTX 커널은 .param 공간에서 ld.param으로 파라미터를 읽으므로, 이 메모리가 채워져야 한다.
 * 처리 규칙:
 *   - is_ptr_shared() 파라미터(OpenCL local 메모리 포인터): 건너뜀
 *   - 실제 크기와 선언 크기가 다르면 경고 후 작은 값 사용
 *   - 정렬 명세(align_amount)에 따라 주소를 정렬
 *   - 4바이트(word_size) 단위로 param_mem->write() 반복 (페이지 경계 걱정 불필요)
 *   - 심볼 테이블의 파라미터 주소를 최종 주소로 갱신
 *
 * 호출 체인:
 *   libcuda::cudaLaunchKernel() / gpgpu_opencl_ptx_sim_init_grid()
 *     → [finalize] → param_mem->write() (4바이트씩)
 */
void function_info::finalize(memory_space *param_mem) {
  unsigned param_address = 0;    // [한국어] 현재 파라미터 배치 주소 (param_mem 내 오프셋)
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;   // [한국어] 현재 파라미터 정보
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    // [한국어] OpenCL local 메모리 포인터: 실제 데이터 없이 크기만 전달되므로 건너뜀
    std::string name = p.get_name();               // [한국어] 파라미터 이름
    int type = p.get_type();                       // [한국어] PTX 타입 enum
    param_t param_value = p.get_value();           // [한국어] 파라미터 실제 값 (데이터 포인터, 크기, 오프셋)
    param_value.type = type;                       // [한국어] 타입 정보도 param_t에 저장
    symbol *param = m_symtab->lookup(name.c_str()); // [한국어] 심볼 테이블에서 파라미터 심볼 조회
    unsigned xtype = param->type()->get_key().scalar_type();  // [한국어] 심볼의 스칼라 타입 코드
    assert(xtype == (unsigned)type);               // [한국어] 파라미터 선언 타입과 심볼 타입 일치 검증
    size_t size;
    size = param_value.size;  // size of param in bytes
    // [한국어] 파라미터 실제 크기 (add_param_data에서 기록된 args->m_nbytes)
    // assert(param_value.offset == param_address);
    if (size != p.get_size() / 8) {
      // [한국어] 실제 크기와 PTX 선언 크기가 다른 경우: 경고 후 작은 값 선택 (안전한 복사)
      printf(
          "GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. "
          "formal size = %zu (using smaller of two)\n",
          size, p.get_size() / 8);
      size = (size < (p.get_size() / 8)) ? size : (p.get_size() / 8);  // [한국어] 더 작은 크기 선택
    }
    // copy the parameter over word-by-word so that parameter that crosses a
    // memory page can be copied over
    // Jin: copy parameter using aligned rules
    // [한국어] 파라미터 정렬 규칙: .align 지시자가 있으면 그 값, 없으면(-1) 파라미터 크기로 자연 정렬
    const type_info *paramtype = param->type();    // [한국어] 파라미터의 타입 정보 객체
    int align_amount = paramtype->get_key().get_alignment_spec();  // [한국어] .align N 명세 (-1이면 미지정)
    align_amount = (align_amount == -1) ? size : align_amount;     // [한국어] 미지정이면 크기로 자연 정렬
    param_address = (param_address + align_amount - 1) / align_amount *
                    align_amount;  // aligned
    // [한국어] 현재 주소를 align_amount 배수로 올림 정렬

    const size_t word_size = 4;  // [한국어] 4바이트(32비트 워드) 단위로 복사
    // param_address = (param_address + size - 1) / size * size; //aligned with
    // size
    for (size_t idx = 0; idx < size; idx += word_size) {
      const char *pdata = reinterpret_cast<const char *>(param_value.pdata) +
                          idx;  // cast to char * for ptr arithmetic
      // [한국어] 파라미터 데이터를 4바이트씩 param_mem에 기록 (페이지 경계 문제 회피)
      param_mem->write(param_address + idx, word_size, pdata, NULL, NULL);
    }
    unsigned offset = p.get_offset();               // [한국어] get_args_aligned_size()에서 계산된 오프셋
    assert(offset == param_address);                // [한국어] 오프셋이 실제 배치 주소와 일치하는지 검증
    param->set_address(param_address);              // [한국어] 심볼 테이블에 최종 파라미터 주소 설정
    param_address += size;                          // [한국어] 다음 파라미터 주소로 전진
  }
}

/*
 * [한국어]
 * function_info::param_to_shared - 파라미터를 공유 메모리에 복사 (PTXPlus GT200 전용)
 *
 * @shared_mem: 공유 메모리 공간 (m_shared_mem 또는 m_sstarr_mem)
 * @symtab:     함수 심볼 테이블
 * @return:     없음 (void)
 *
 * GTX200(GT200) 아키텍처의 PTXPlus 모드에서는 파라미터가 공유 메모리의
 * 특정 오프셋(offset + 0x10)에 배치된다 (하드웨어 동작 에뮬레이션).
 * convert_to_ptxplus()가 false이면(일반 PTX 모드) 즉시 반환한다.
 * 0x10 오프셋: GT200 하드웨어에서 파라미터가 공유 메모리 시작점에서 16바이트 뒤에 놓이는 규약.
 * ptx_sim_init_thread()에서 각 스레드의 shared_mem과 sstarr_mem에 대해 2회 호출됨.
 *
 * 호출 체인:
 *   ptx_sim_init_thread() → thd->func_info()->param_to_shared()
 */
void function_info::param_to_shared(memory_space *shared_mem,
                                    symbol_table *symtab) {
  // TODO: call this only for PTXPlus with GT200 models
  // extern gpgpu_sim* g_the_gpu;
  // [한국어] PTXPlus GT200 모드가 아니면 파라미터를 공유 메모리에 복사하지 않음
  if (not gpgpu_ctx->the_gpgpusim->g_the_gpu->get_config().convert_to_ptxplus())
    return;

  // copies parameters into simulated shared memory
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;                    // [한국어] 현재 파라미터 정보
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    // [한국어] OpenCL local 메모리 포인터: 건너뜀
    std::string name = p.get_name();              // [한국어] 파라미터 이름
    int type = p.get_type();                      // [한국어] 파라미터 타입
    param_t value = p.get_value();                // [한국어] 파라미터 실제 값
    value.type = type;                            // [한국어] 타입 정보 설정
    symbol *param = symtab->lookup(name.c_str()); // [한국어] 심볼 테이블 조회
    unsigned xtype = param->type()->get_key().scalar_type();  // [한국어] 스칼라 타입 코드
    assert(xtype == (unsigned)type);              // [한국어] 타입 일치 검증

    int tmp;                                       // [한국어] type_decode 내부 임시 변수
    size_t size;                                   // [한국어] 타입 크기 (비트 단위)
    unsigned offset = p.get_offset();              // [한국어] 파라미터의 공유 메모리 오프셋
    type_info_key::type_decode(xtype, size, tmp);  // [한국어] 타입 코드에서 크기(비트) 추출

    // Write to shared memory - offset + 0x10
    // [한국어] GT200 하드웨어: 파라미터가 공유 메모리 시작에서 0x10(16) 바이트 오프셋에 배치
    shared_mem->write(offset + 0x10, size / 8, value.pdata, NULL, NULL);
    // [한국어] size/8: 비트→바이트 변환. value.pdata: 실제 파라미터 데이터.
  }
}

/*
 * [한국어]
 * function_info::list_param - 파라미터 이름과 주소를 파일에 출력 (디버그용)
 *
 * @fout: 출력 대상 파일 스트림
 * @return: 없음 (void)
 *
 * 모든 파라미터의 이름과 파라미터 메모리 주소를 "name: 0x12345678" 형식으로 출력한다.
 * 파라미터 주소가 올바르게 설정됐는지 디버그할 때 사용한다.
 *
 * 호출 체인:
 *   디버그 경로 → [list_param]
 */
void function_info::list_param(FILE *fout) const {
  for (std::map<unsigned, param_info>::const_iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    const param_info &p = i->second;              // [한국어] 현재 파라미터 정보
    std::string name = p.get_name();              // [한국어] 파라미터 이름
    symbol *param = m_symtab->lookup(name.c_str());  // [한국어] 심볼 테이블에서 주소 조회
    addr_t param_addr = param->get_address();     // [한국어] 파라미터의 파라미터 메모리 주소
    fprintf(fout, "%s: %#08llx\n", name.c_str(), param_addr);  // [한국어] "이름: 0x주소" 형식 출력
  }
  fflush(fout);                                   // [한국어] 버퍼 강제 출력
}

/*
 * [한국어]
 * function_info::ptx_jit_config - WatchYourStep 디버그 도구용 파라미터 설정 파일 생성
 *
 * @mallocPtr_Size: GPU cudaMalloc 포인터→배열 크기 맵 (포인터 추적용)
 * @param_mem:      파라미터 메모리 공간
 * @gpu:            gpgpu_t 인스턴스 (전역 메모리 접근용)
 * @gridDim:        그리드 차원
 * @blockDim:       블록(CTA) 차원
 * @return:         없음 (void)
 *
 * GPGPU-Sim의 WatchYourStep 디버그 도구를 위한 설정 파일을 생성한다.
 * WatchYourStep은 메모리 접근 패턴을 분석하여 메모리 안전성을 검사하는 도구이다.
 * 출력 파일: $GPGPUSIM_ROOT/debug_tools/WatchYourStep/data/params.config<N>
 *            $GPGPUSIM_ROOT/debug_tools/WatchYourStep/data/ptx.config<N>
 * 기능:
 *   - c++filt로 맹글링된 함수 이름을 디맹글링
 *   - 파라미터 목록에서 포인터인지 여부 판별 (c++filt 출력의 '*' 확인)
 *   - 포인터 파라미터: 실제 배열 데이터를 GPU 메모리에서 읽어 파일에 기록
 *   - 비포인터 파라미터: 파라미터 메모리에서 값을 읽어 기록
 *   - PTX 소스 파일에서 커널 함수 헤더를 복사하여 ptx.config 생성
 * 환경변수: GPGPUSIM_ROOT, WYS_EXEC_PATH가 반드시 설정되어 있어야 함.
 *
 * 호출 체인:
 *   [별도 디버그 모드에서 커널 실행 전] → [ptx_jit_config]
 */
void function_info::ptx_jit_config(
    std::map<unsigned long long, size_t> mallocPtr_Size,
    memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim, dim3 blockDim) {
  static unsigned long long counter = 0;  // [한국어] 실행 횟수 카운터 — 파일명 중복 방지용 접미사
  std::vector<std::pair<size_t, unsigned char *> > param_data;
  std::vector<unsigned> offsets;
  std::vector<bool> paramIsPointer;

  char *gpgpusim_path = getenv("GPGPUSIM_ROOT");
  assert(gpgpusim_path != NULL);
  char *wys_exec_path = getenv("WYS_EXEC_PATH");
  assert(wys_exec_path != NULL);
  std::string command =
      std::string("mkdir ") + gpgpusim_path + "/debug_tools/WatchYourStep/data";
  std::string filename(std::string(gpgpusim_path) +
                       "/debug_tools/WatchYourStep/data/params.config" +
                       std::to_string(counter));

  // initialize paramList
  char buff[1024];
  std::string filename_c(filename + "_c");
  snprintf(buff, 1024, "c++filt %s > %s", get_name().c_str(),
           filename_c.c_str());
  assert(system(buff) != 0);
  FILE *fp = fopen(filename_c.c_str(), "r");
  char *ptr = fgets(buff, 1024, fp);
  if (ptr == NULL) {
    printf("can't read file %s \n", filename_c.c_str());
    assert(0);
  }
  fclose(fp);
  std::string fn(buff);
  size_t pos1, pos2;
  pos1 = fn.find_last_of("(");
  pos2 = fn.find(")", pos1);
  assert(pos2 > pos1 && pos1 > 0);
  strcpy(buff, fn.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  char *tok;
  tok = strtok(buff, ",");
  std::string tmp;
  while (tok != NULL) {
    std::string param(tok);
    if (param.find("<") != std::string::npos) {
      assert(param.find(">") == std::string::npos);
      assert(param.find("*") == std::string::npos);
      tmp = param;
    } else {
      if (tmp.length() > 0) {
        tmp = "";
        assert(param.find(">") != std::string::npos);
        assert(param.find("<") == std::string::npos);
        assert(param.find("*") == std::string::npos);
      }
      printf("%s\n", param.c_str());
      if (param.find("*") != std::string::npos) {
        paramIsPointer.push_back(true);
      } else {
        paramIsPointer.push_back(false);
      }
    }
    tok = strtok(NULL, ",");
  }

  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    param_t param_value = p.get_value();
    offsets.push_back((unsigned)p.get_offset());

    if (paramIsPointer[i->first] &&
        (*(unsigned long long *)param_value.pdata != 0)) {
      // is pointer
      assert(param_value.size == sizeof(void *) &&
             "MisID'd this param as pointer");
      size_t array_size = 0;
      unsigned long long param_pointer =
          *(unsigned long long *)param_value.pdata;
      if (mallocPtr_Size.find(param_pointer) != mallocPtr_Size.end()) {
        array_size = mallocPtr_Size[param_pointer];
      } else {
        for (std::map<unsigned long long, size_t>::iterator j =
                 mallocPtr_Size.begin();
             j != mallocPtr_Size.end(); j++) {
          if (param_pointer > j->first &&
              param_pointer < j->first + j->second) {
            array_size = j->first + j->second - param_pointer;
            break;
          }
        }
        assert(array_size > 0 && "pointer was not previously malloc'd");
      }

      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      unsigned char *array_val = (unsigned char *)malloc(array_size);
      gpu->get_global_memory()->read(*(unsigned *)((void *)val), array_size,
                                     (void *)array_val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(array_size, array_val));
      paramIsPointer.push_back(true);
    } else {
      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(param_value.size, val));
      paramIsPointer.push_back(false);
    }
  }

  FILE *fout = fopen(filename.c_str(), "w");
  printf("Writing data to %s ...\n", filename.c_str());
  fprintf(fout, "%s\n", get_name().c_str());
  fprintf(fout, "%u,%u,%u %u,%u,%u\n", gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z);
  size_t index = 0;
  for (std::vector<std::pair<size_t, unsigned char *> >::const_iterator i =
           param_data.begin();
       i != param_data.end(); i++) {
    if (paramIsPointer[index]) {
      fprintf(fout, "*");
    }
    fprintf(fout, "%lu :", i->first);
    for (size_t j = 0; j < i->first; j++) {
      fprintf(fout, " %u", i->second[j]);
    }
    fprintf(fout, " : %u", offsets[index]);
    free(i->second);
    fprintf(fout, "\n");
    index++;
  }
  fflush(fout);
  fclose(fout);

  // ptx config
  std::string ptx_config_fn(std::string(gpgpusim_path) +
                            "/debug_tools/WatchYourStep/data/ptx.config" +
                            std::to_string(counter));
  snprintf(buff, 1024,
           "grep -rn \".entry %s\" %s/*.ptx | cut -d \":\" -f 1-2 > %s",
           get_name().c_str(), wys_exec_path, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx source \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  FILE *fin = fopen(ptx_config_fn.c_str(), "r");
  char ptx_source[256];
  unsigned line_number;
  int numscanned = fscanf(fin, "%[^:]:%u", ptx_source, &line_number);
  assert(numscanned == 2);
  fclose(fin);
  snprintf(buff, 1024,
           "grep -rn \".version\" %s | cut -d \":\" -f 1 | xargs -I \"{}\" awk "
           "\"NR>={}&&NR<={}+2\" %s > %s",
           ptx_source, ptx_source, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx header \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  fin = fopen(ptx_source, "r");
  assert(fin != NULL);
  printf("Writing data to %s ...\n", ptx_config_fn.c_str());
  fout = fopen(ptx_config_fn.c_str(), "a");
  assert(fout != NULL);
  for (unsigned i = 0; i < line_number; i++) {
    assert(fgets(buff, 1024, fin) != NULL);
    assert(!feof(fin));
  }
  fprintf(fout, "\n\n");
  do {
    fprintf(fout, "%s", buff);
    assert(fgets(buff, 1024, fin) != NULL);
    if (feof(fin)) {
      break;
    }
  } while (strstr(buff, "entry") == NULL);

  fclose(fin);
  fflush(fout);
  fclose(fout);
  counter++;
}

/*
 * [한국어]
 * cuda_sim::ptx_debug_exec_dump_cond<N> - 디버그 덤프 조건 필터 (템플릿 함수)
 *
 * @activate_level: 이 덤프를 활성화하는 최소 g_debug_execution 레벨 (템플릿 매개변수)
 * @thd_uid: 현재 스레드의 고유 ID
 * @pc:      현재 실행 중인 PC
 * @return:  true이면 이 스레드/PC에서 디버그 덤프 수행, false이면 건너뜀
 *
 * 디버그 덤프 출력 여부를 필터링하는 템플릿 함수.
 * g_debug_execution >= activate_level이어야 활성화되고,
 * g_debug_thread_uid != 0이면 해당 스레드만, g_debug_pc != 0xBEEF1518이면 해당 PC만 출력.
 * 0xBEEF1518: "sentinel" 값 — 특정 PC 필터링 비활성화 표시.
 *
 * 인스턴스화:
 *   ptx_debug_exec_dump_cond<5>(uid, pc)  — 레벨 5 이상에서 활성
 *   ptx_debug_exec_dump_cond<6>(uid, pc)  — 레벨 6 이상에서 활성 (레지스터 덤프)
 *   ptx_debug_exec_dump_cond<10>(uid, pc) — 레벨 10 이상에서 활성 (전체 레지스터)
 *
 * 호출 체인:
 *   ptx_exec_inst() → [ptx_debug_exec_dump_cond<5/6/10>]
 */
template <int activate_level>
bool cuda_sim::ptx_debug_exec_dump_cond(int thd_uid, addr_t pc) {
  if (g_debug_execution >= activate_level) {
    // check each type of debug dump constraint to filter out dumps
    // [한국어] 특정 스레드 UID 필터: g_debug_thread_uid가 0이면 모든 스레드 출력
    if ((g_debug_thread_uid != 0) &&
        (thd_uid != (unsigned)g_debug_thread_uid)) {
      return false;  // [한국어] 이 스레드는 덤프 대상 아님
    }
    // [한국어] 특정 PC 필터: 0xBEEF1518은 필터 비활성 표시 (모든 PC 출력)
    if ((g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc)) {
      return false;  // [한국어] 이 PC는 덤프 대상 아님
    }

    return true;     // [한국어] 모든 조건 통과: 덤프 수행
  }

  return false;      // [한국어] 디버그 레벨 미달: 덤프 건너뜀
}

/*
 * [한국어]
 * cuda_sim::init_inst_classification_stat - 명령어 분류 통계 초기화
 *
 * @return: 없음 (void)
 *
 * 현재 커널(g_ptx_kernel_count)에 대한 명령어 분류 통계 구조체를 초기화한다.
 * 두 종류의 통계:
 *   1. g_inst_classification_stat[kernel_id]: 공간 타입(글로벌/로컬/공유/etc) × 명령어 분류
 *   2. g_inst_op_classification_stat[kernel_id]: 옵코드별 실행 횟수
 * static std::set<unsigned> init으로 동일 커널에 대해 한 번만 초기화함.
 * MAX_CLASS_KER(1024): 지원하는 최대 커널 수. 초과 시 assert 실패.
 * gpgpu_ptx_instruction_classification 플래그가 켜져 있을 때만 의미 있음.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [init_inst_classification_stat]
 *     → StatCreate() (statwrapper.h)
 */
void cuda_sim::init_inst_classification_stat() {
  static std::set<unsigned> init;   // [한국어] 이미 초기화된 커널 ID 집합 (중복 초기화 방지)
  if (init.find(g_ptx_kernel_count) != init.end()) return;  // [한국어] 이미 초기화된 커널 → 즉시 반환
  init.insert(g_ptx_kernel_count);  // [한국어] 이 커널을 초기화 완료 집합에 추가

#define MAX_CLASS_KER 1024           // [한국어] 최대 커널 수 — 이를 초과하면 assert 실패
  char kernelname[MAX_CLASS_KER] = "";  // [한국어] 통계 이름 버퍼
  if (!g_inst_classification_stat)
    // [한국어] 첫 번째 커널 시: 통계 포인터 배열 동적 할당 (void* 배열, StatCreate 결과 저장)
    g_inst_classification_stat = (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",
           g_ptx_kernel_count);     // [한국어] 통계 이름: "Kernel N Classification"
  assert(g_ptx_kernel_count <
         MAX_CLASS_KER);  // a static limit on number of kernels increase it if
                          // it fails!
  // [한국어] 공간 타입 + 명령어 분류 통계 생성 (버킷 1~20)
  g_inst_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 20);  // [한국어] StatCreate(이름, 최솟값, 최댓값)
  if (!g_inst_op_classification_stat)
    // [한국어] 첫 번째 커널 시: 옵코드 분류 통계 포인터 배열 할당
    g_inst_op_classification_stat =
        (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",
           g_ptx_kernel_count);     // [한국어] 통계 이름: "Kernel N OP Classification"
  // [한국어] 옵코드별 실행 횟수 통계 생성 (버킷 1~100: 다양한 옵코드 코드 수용)
  g_inst_op_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 100);
}

/*
 * [한국어]
 * get_tex_datasize - TEX 명령어의 텍스처 데이터 크기(텍셀 크기) 조회 (static 내부 함수)
 *
 * @pI:     실행 중인 TEX 명령어 포인터
 * @thread: 현재 실행 스레드 — 레지스터 값 읽기와 커널 정보 접근에 사용
 * @return: 텍셀 크기(바이트) — 텍스처 샘플링 시 접근하는 데이터 단위
 *
 * TEX 명령어의 소스1 피연산자에서 텍스처 이름을 추출하고,
 * 커널의 텍스처 스냅샷에서 해당 텍스처의 텍셀 크기를 반환한다.
 * 직접 접근(texname 심볼): src1.name()으로 이름 추출
 * 간접 접근(레지스터로 주소 전달): src1 레지스터 값을 읽어 심볼 테이블에서 이름 조회
 * 커널 스냅샷 사용 이유: 멀티스트림 환경에서 텍스처가 비동기적으로 bind/unbind될 수 있으므로
 * 커널 실행 시점의 텍스처 상태를 사용해야 정확한 결과를 보장한다.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [get_tex_datasize] (TEX_OP 처리 후 insn_data_size 계산)
 */
static unsigned get_tex_datasize(const ptx_instruction *pI,
                                 ptx_thread_info *thread) {
  const operand_info &src1 = pI->src1();  // the name of the texture
  // [한국어] src1: TEX 명령어의 첫 번째 소스 피연산자 — 텍스처 이름 또는 레지스터
  std::string texname = src1.name();     // [한국어] 기본 텍스처 이름 추출 (직접 접근 경우)
  // If indirect access, use register's value as address
  // to find the symbol
  if (src1.is_reg()) {
    // [한국어] 간접 접근: src1이 레지스터 → 레지스터 값이 텍스처 심볼 주소
    const operand_info &dst = pI->dst();               // [한국어] 목적지 피연산자 (get_operand_value 인수로 필요)
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, pI->get_type(), thread, 1);
    // [한국어] 레지스터에서 실제 값 읽기 — 이 값이 텍스처 심볼의 주소
    addr_t sym_addr = src1_data.u64;                   // [한국어] 64비트 주소로 해석
    symbol *texRef = thread->get_symbol_table()->lookup_by_addr(sym_addr);
    // [한국어] 주소로 심볼 테이블에서 텍스처 레퍼런스 심볼 조회
    assert(texRef != NULL);                            // [한국어] 등록되지 않은 텍스처 주소 → 버그
    texname = texRef->name();                          // [한국어] 텍스처 심볼의 이름으로 갱신
  }

  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
   */
  // [한국어] 커널 실행 시점의 텍스처 정보 스냅샷에서 조회 (bind/unbind 경쟁 조건 방지)
  kernel_info_t &k = thread->get_kernel();             // [한국어] 이 스레드가 실행 중인 커널 정보
  const struct textureInfo *texInfo = k.get_texinfo(texname);
  // [한국어] 커널 실행 시 캡처된 텍스처 정보 (Tx, Ty, texel_size 등)

  unsigned data_size = texInfo->texel_size;            // [한국어] 텍셀 크기(바이트) — 텍스처 샘플링 접근 단위
  return data_size;                                    // [한국어] 텍스처 데이터 크기 반환
}

/*
 * [한국어]
 * tensorcore_op - 주어진 옵코드가 Tensor Core 연산인지 확인
 *
 * @inst_opcode: PTX 옵코드 enum 값
 * @return:      1이면 Tensor Core 연산(MMA_OP, MMA_LD_OP, MMA_ST_OP), 0이면 아님
 *
 * ptx_exec_inst()에서 Tensor Core 연산(MMA)인 경우 lane_id==0인 스레드만 실행하여
 * warp-synchronous 연산을 단 1회만 시뮬레이션하는 최적화에 사용된다.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [tensorcore_op]
 */
int tensorcore_op(int inst_opcode) {
  if ((inst_opcode == MMA_OP) || (inst_opcode == MMA_LD_OP) ||
      (inst_opcode == MMA_ST_OP))
    return 1;   // [한국어] Tensor Core 관련 옵코드 (MMA, MMA_LD, MMA_ST)
  else
    return 0;   // [한국어] 일반 명령어
}
/*
 * [한국어]
 * ptx_thread_info::ptx_exec_inst - 하나의 PTX 명령어를 하나의 SIMT 레인에서 실행 (핵심 함수)
 *
 * @inst:    실행 중인 warp_inst_t 참조 — active mask, 메모리 주소 등을 쓰기 위해 참조.
 *           이 함수 실행 후 inst.addr[lane_id], inst.space, inst.data_size가 채워짐.
 * @lane_id: warp 내 이 스레드의 레인 번호 (0 ~ warp_size-1)
 * @return:  없음 (void)
 *
 * 이 함수는 GPGPU-Sim PTX 기능 시뮬레이터의 **가장 핵심적인 함수**이다.
 * 타이밍 모델(shader.cc)이 각 warp의 각 스레드(레인)를 1 사이클 단위로 실행할 때 호출한다.
 *
 * 실행 단계:
 *   1. 현재 스레드 PC에서 ptx_instruction 가져오기 (m_func_info->get_instruction)
 *   2. NPC(Next PC) 설정: pc + inst_size()
 *   3. 디버그 모드 설정 (clear_modifiedregs, enable_debug_trace)
 *   4. Predicate 평가: @p pred_value를 읽어 skip 여부 결정
 *      - 표준 PTX predicate: pred_value.pred & 0x0001 ^ pred_neg
 *      - PTXPlus predicate: pred_lookup 테이블 참조 (decuda_pred_table)
 *   5. skip이면: inst.set_not_active(lane_id) — 이 레인은 마스크에서 제거
 *   6. skip이 아니면:
 *      a. VOTE/ACTIVEMASK: warp active mask 정보를 pI 복사본에 주입
 *      b. MMA 검증: Tensor Core는 모든 스레드 활성 상태 필요
 *      c. X-매크로 dispatch: opcodes.def의 OP_DEF/OP_W_DEF로 실제 명령어 함수 호출
 *         - OP_DEF: FUNC(pI, this) — 스레드 상태만 필요한 명령어
 *         - OP_W_DEF: FUNC(pI, get_core(), inst) — warp_inst_t도 필요한 명령어 (VOTE 등)
 *         - Tensor Core(MMA): lane_id==0인 스레드만 1회 실행 (warp 동기 연산 최적화)
 *      d. exit 처리: pI->is_exit()이면 exit_impl() 호출
 *   7. 메모리 접근 정보 수집:
 *      - last_eaddr(): 명령어 실행 후 기록된 마지막 유효 주소
 *      - last_space(): 마지막 메모리 공간
 *      - datatype2size(get_type()): 데이터 크기
 *   8. BAR.RED: add_callback()으로 리덕션 콜백 등록
 *   9. ATOM: add_callback()으로 원자 연산 콜백 등록 + 주소/크기 수집
 *   10. TEX: set_addr() + get_tex_datasize()로 텍스처 주소/크기 수집
 *   11. 디버그 출력 (레벨 5/6/10에 따라 명령어 정보, 수정 레지스터, 전체 레지스터)
 *   12. update_pc(): PC를 NPC로 전진
 *   13. g_ptx_sim_num_insn 증가: 전역 명령어 카운터
 *   14. ptx_file_line_stats_add_exec_count(): PTX 라인별 실행 횟수 통계 (비기능 모드만)
 *   15. 명령어 분류 통계 업데이트 (gpgpu_ptx_instruction_classification 켜진 경우)
 *   16. inst에 insn_space/insn_memaddr/insn_data_size 기록 (타이밍 모델에 반환)
 *   17. 예외 처리: int x 예외 발생 시 오류 메시지 출력 후 abort()
 *
 * 실행 컨텍스트: 타이밍 시뮬레이션 모드 — shader.cc::execute()에서 각 warp의 각 레인을 반복 호출.
 *               기능 시뮬레이션 모드 — functionalCoreSim::executeWarp()에서 호출.
 * 멀티스레드 비고: 각 스레드는 자체 ptx_thread_info 객체를 가지므로 스레드 상태 접근은 안전.
 *               warp_inst_t inst는 warp 내 모든 레인이 공유하므로, set_addr/set_not_active는
 *               lane_id를 인수로 받아 해당 레인 슬롯만 수정한다.
 *
 * 호출 체인:
 *   shader.cc::shader_core_ctx::execute() → warp의 각 active 레인에 대해 → [ptx_exec_inst]
 *     → instructions.cc::FUNC(pI, this) (X-매크로 dispatch)
 *     → last_eaddr() / last_space() / last_callback() (명령어 결과 수집)
 *     → update_pc() / g_ptx_sim_num_insn++
 */
void ptx_thread_info::ptx_exec_inst(warp_inst_t &inst, unsigned lane_id) {
  bool skip = false;          // [한국어] predicate 조건으로 이 레인 실행을 건너뛸지 여부
  int op_classification = 0;  // [한국어] 명령어 분류 코드 (통계 집계용, X-매크로에서 채워짐)
  addr_t pc = next_instr();   // [한국어] 현재 스레드가 실행할 명령어의 PC
  assert(pc ==
         inst.pc);  // make sure timing model and functional model are in sync
  // [한국어] 타이밍 모델이 fetch한 PC와 기능 시뮬레이터가 실행할 PC가 일치해야 함 — 불일치는 심각한 버그
  const ptx_instruction *pI = m_func_info->get_instruction(pc);
  // [한국어] PC에 해당하는 PTX 명령어 포인터 조회 (m_instr_mem 배열 인덱싱)

  set_npc(pc + pI->inst_size());
  // [한국어] NPC(Next PC) 설정: 현재 PC + 명령어 크기. 분기/호출 명령어는 실행 중 NPC를 덮어씀.

  try {
    clearRPC();                              // [한국어] RPC(Return PC) 초기화 — 이전 실행의 return 주소 정리
    m_last_set_operand_value.u64 = 0;        // [한국어] 마지막 쓰여진 피연산자 값 초기화 (디버그 출력용)

    if (is_done()) {
      // [한국어] 이미 완료된 스레드에 실행 시도 → 심각한 버그
      printf(
          "attempted to execute instruction on a thread that is already "
          "done.\n");
      assert(0);
    }

    // [한국어] 디버그 추적 활성화: g_debug_execution >= 6 또는 ptx_inst_debug_to_file이 켜진 경우
    if (g_debug_execution >= 6 ||
        m_gpu->get_config().get_ptx_inst_debug_to_file()) {
      if ((m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid == 0) ||
          (get_uid() ==
           (unsigned)(m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid))) {
        // [한국어] 대상 스레드(모든 스레드 또는 특정 UID)에 대해 레지스터 변경 추적 시작
        clear_modifiedregs();              // [한국어] 이전 명령어에서 수정된 레지스터 목록 초기화
        enable_debug_trace();             // [한국어] 이번 명령어의 레지스터 변경 추적 활성화
      }
    }

    // [한국어] Predicate 평가: @p0 (또는 @!p0) 등 조건부 실행 검사
    if (pI->has_pred()) {
      const operand_info &pred = pI->get_pred();   // [한국어] predicate 레지스터 피연산자
      ptx_reg_t pred_value = get_operand_value(pred, pred, PRED_TYPE, this, 0);
      // [한국어] predicate 레지스터의 현재 값 읽기 (PRED_TYPE: 1비트 불리언)
      if (pI->get_pred_mod() == -1) {
        // [한국어] 표준 PTX predicate: pred.pred의 최하위 비트 ^ get_pred_neg()
        // get_pred_neg(): @!p이면 true (반전), @p이면 false
        skip = (pred_value.pred & 0x0001) ^
               pI->get_pred_neg();  // ptxplus inverts the zero flag
      } else {
        // [한국어] PTXPlus predicate: decuda_pred_table을 사용한 복잡한 조건 평가
        skip = !pred_lookup(pI->get_pred_mod(), pred_value.pred & 0x000F);
      }
    }
    int inst_opcode = pI->get_opcode();  // [한국어] 옵코드 캐시 (여러 곳에서 사용)

    if (skip) {
      inst.set_not_active(lane_id);     // [한국어] predicate 실패: 이 레인을 active mask에서 제거
    } else {
      const ptx_instruction *pI_saved = pI;   // [한국어] 원본 pI 저장 (pJ 임시 복사 후 복원용)
      ptx_instruction *pJ = NULL;              // [한국어] VOTE/ACTIVEMASK용 임시 명령어 복사본
      if (pI->get_opcode() == VOTE_OP || pI->get_opcode() == ACTIVEMASK_OP) {
        // [한국어] VOTE/ACTIVEMASK: warp active mask 정보를 명령어 복사본에 주입해야 함
        // 이 명령어들은 warp 전체의 active mask를 operand로 사용하므로
        // pI에 inst의 active mask를 복사한 pJ를 만들어 실행
        pJ = new ptx_instruction(*pI);          // [한국어] ptx_instruction 복사 생성
        *((warp_inst_t *)pJ) = inst;  // copy active mask information
        // [한국어] warp_inst_t 기본 클래스 슬라이스 복사 — active mask가 pJ에 주입됨
        pI = pJ;                               // [한국어] 이후 실행에 pJ(active mask 포함)를 사용
      }

      if (((inst_opcode == MMA_OP || inst_opcode == MMA_LD_OP ||
            inst_opcode == MMA_ST_OP))) {
        // [한국어] Tensor Core 검증: MMA 연산은 warp의 모든 스레드가 활성 상태여야 함
        if (inst.active_count() != MAX_WARP_SIZE) {
          printf(
              "Tensor Core operation are warp synchronous operation. All the "
              "threads needs to be active.");
          assert(0);                             // [한국어] 일부 스레드 비활성 상태 → 시뮬레이션 오류
        }
      }

      // Tensorcore is warp synchronous operation. So these instructions needs
      // to be executed only once. To make the simulation faster removing the
      // redundant tensorcore operation
      // [한국어] Tensor Core 최적화: warp-synchronous이므로 lane_id==0에서만 1회 실행
      // 나머지 31개 레인은 건너뜀 — 기능적으로 동일하고 시뮬레이션 속도 향상
      if (!tensorcore_op(inst_opcode) ||
          ((tensorcore_op(inst_opcode)) && (lane_id == 0))) {
        // [한국어] X-매크로 dispatch: opcodes.def의 OP_DEF/OP_W_DEF 매크로 확장
        // OP_DEF(옵코드, 함수, 문자열, DST여부, 분류코드)
        //   → case 옵코드: 함수(pI, this); op_classification=분류코드; break;
        // OP_W_DEF(옵코드, 함수, 문자열, DST여부, 분류코드)
        //   → case 옵코드: 함수(pI, get_core(), inst); op_classification=분류코드; break;
        //   (warp_inst_t inst를 추가 인수로 받는 명령어: VOTE, ACTIVEMASK, BAR 등)
        switch (inst_opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    FUNC(pI, this);                                \  // [한국어] instructions.cc의 FUNC 호출 (예: add_impl(pI, this))
    op_classification = CLASSIFICATION;            \  // [한국어] 명령어 분류 코드 저장 (통계용)
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    FUNC(pI, get_core(), inst);                      \  // [한국어] 코어 정보와 warp_inst_t를 추가 인수로 전달
    op_classification = CLASSIFICATION;              \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
          default:
            printf("Execution error: Invalid opcode (0x%x)\n",
                   pI->get_opcode());   // [한국어] 알 수 없는 옵코드: PTX 버전 미지원 가능성
            break;
        }
      }
      delete pJ;                         // [한국어] VOTE/ACTIVEMASK용 임시 복사본 해제 (NULL이면 no-op)
      pI = pI_saved;                     // [한국어] 원본 pI 복원 (이후 pI 사용은 원본으로)

      // Run exit instruction if exit option included
      // [한국어] PTX exit 플래그: 명령어에 exit가 포함된 경우 (예: setp.eq.s32 %p, %r, 0 exit)
      if (pI->is_exit()) exit_impl(pI, this);  // [한국어] exit_impl: 스레드 종료 처리
    }

    const gpgpu_functional_sim_config &config = m_gpu->get_config();
    // [한국어] 기능 시뮬레이션 설정 참조 (디버그 파일 출력 여부 등)

    // Output instruction information to file and stdout
    // [한국어] PTX 명령어 디버그 파일 출력 (ptx_inst_debug_to_file 켜진 경우)
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      fprintf(m_gpu->get_ptx_inst_debug_file(), "[thd=%u] : (%s:%u - %s)\n",
              get_uid(), pI->source_file(), pI->source_line(),
              pI->get_source());
      // [한국어] 형식: "[thd=N] : (파일:줄번호 - PTX소스텍스트)"
      // fprintf(ptx_inst_debug_file, "has memory read=%d, has memory
      // write=%d\n", pI->has_memory_read(), pI->has_memory_write());
      fflush(m_gpu->get_ptx_inst_debug_file());
    }

    // [한국어] 디버그 레벨 5 이상: 명령어 실행 상세 정보 stdout 출력
    if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<5>(get_uid(),
                                                                pc)) {
      dim3 ctaid = get_ctaid();   // [한국어] 현재 CTA(스레드 블록) ID
      dim3 tid = get_tid();       // [한국어] 현재 스레드 ID (블록 내)
      printf(
          "%u [thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u "
          "[pc=%llu] (%s:%u - %s)  [0x%llx]\n",
          m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, get_uid(), pI->uid(),
          ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, get_icount(), pc,
          pI->source_file(), pI->source_line(), pI->get_source(),
          m_last_set_operand_value.u64);
      // [한국어] 전역 명령어 번호, 스레드UID, 명령어UID, CTA/스레드 좌표, 실행 횟수, PC, 소스위치, 마지막 값 출력
      fflush(stdout);
    }

    // [한국어] 메모리 접근 정보 수집 — 타이밍 모델에 반환할 데이터
    addr_t insn_memaddr = 0xFEEBDAED;           // [한국어] 초기값: 유효하지 않은 주소 (sentinel)
    memory_space_t insn_space = undefined_space; // [한국어] 초기값: 미정의 공간
    _memory_op_t insn_memory_op = no_memory_op;  // [한국어] 초기값: 메모리 연산 없음
    unsigned insn_data_size = 0;                 // [한국어] 초기값: 데이터 크기 0

    if ((pI->has_memory_read() || pI->has_memory_write())) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        // [한국어] MMA_LD/ST는 별도 처리 — Tensor Core 레지스터 파일 접근이므로 일반 메모리 주소 없음
        insn_memaddr = last_eaddr();             // [한국어] 명령어 실행 후 기록된 마지막 유효 주소
        insn_space = last_space();               // [한국어] 마지막으로 접근한 메모리 공간
        unsigned to_type = pI->get_type();       // [한국어] 명령어의 데이터 타입
        insn_data_size = datatype2size(to_type); // [한국어] 타입→바이트 크기 변환
        insn_memory_op = pI->has_memory_read() ? memory_load : memory_store;
        // [한국어] 로드 또는 스토어 구분
      }
    }

    // [한국어] BAR.RED: 리덕션 콜백 등록 — 모든 스레드가 배리어 도달 후 실행
    if (pI->get_opcode() == BAR_OP && pI->barrier_op() == RED_OPTION) {
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this,
                        false /*not atomic*/);
      // [한국어] last_callback(): BAR.RED 명령어 구현이 등록한 리덕션 함수 포인터
    }

    // [한국어] ATOM(원자 연산): 메모리 주소 수집 + 원자 콜백 등록
    if (pI->get_opcode() == ATOM_OP) {
      insn_memaddr = last_eaddr();              // [한국어] 원자 연산 주소 (ld+modify+st 대상)
      insn_space = last_space();                // [한국어] 원자 연산 메모리 공간 (global/shared)
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this, true /*atomic*/);
      // [한국어] 원자 연산: 타이밍 모델이 완료 시점에 호출할 콜백 등록 (메모리 일관성)
      unsigned to_type = pI->get_type();        // [한국어] 원자 데이터 타입
      insn_data_size = datatype2size(to_type);  // [한국어] 원자 접근 크기
    }

    // [한국어] TEX(텍스처 샘플링): 주소 설정 + 텍셀 크기 조회
    if (pI->get_opcode() == TEX_OP) {
      inst.set_addr(lane_id, last_eaddr());     // [한국어] 텍스처 샘플링 주소를 lane_id 슬롯에 저장
      assert(inst.space == last_space());       // [한국어] 텍스처 공간 일치 검증
      insn_data_size = get_tex_datasize(
          pI,
          this);  // texture obtain its data granularity from the texture info
      // [한국어] 텍스처 정보에서 텍셀 크기를 가져옴 (채널 포맷에 따라 1~16바이트)
    }

    // Output register information to file and stdout
    // [한국어] 디버그 파일: 수정된 레지스터와 전체 레지스터 상태 출력
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      dump_modifiedregs(m_gpu->get_ptx_inst_debug_file()); // [한국어] 이번 명령어에서 변경된 레지스터 출력
      dump_regs(m_gpu->get_ptx_inst_debug_file());         // [한국어] 전체 레지스터 파일 상태 출력
    }

    // [한국어] 디버그 레벨 6 이상: 수정된 레지스터만 stdout에 출력
    if (g_debug_execution >= 6) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<6>(get_uid(),
                                                                  pc))
        dump_modifiedregs(stdout);   // [한국어] 변경된 레지스터 목록 출력
    }
    // [한국어] 디버그 레벨 10 이상: 전체 레지스터 파일 stdout에 출력
    if (g_debug_execution >= 10) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<10>(get_uid(),
                                                                   pc))
        dump_regs(stdout);           // [한국어] 전체 레지스터 덤프 출력 (매우 많은 출력)
    }
    update_pc();                     // [한국어] PC를 NPC(Next PC)로 전진 (분기/호출 시 NPC가 변경됐을 수 있음)
    m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn++;
    // [한국어] 전역 명령어 실행 카운터 증가 (100000단위로 진행 상황 출력)

    // not using it with functional simulation mode
    // [한국어] 타이밍 시뮬레이션 모드에서만: PTX 라인별 실행 횟수 통계 업데이트
    if (!(this->m_functionalSimulationMode))
      ptx_file_line_stats_add_exec_count(pI);  // [한국어] ptx-stats.cc의 라인 통계 함수

    // [한국어] 명령어 분류 통계 업데이트 (gpgpu_ptx_instruction_classification 옵션 켜진 경우)
    if (m_gpu->gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
      m_gpu->gpgpu_ctx->func_sim->init_inst_classification_stat();  // [한국어] 통계 구조체 초기화 (1회)
      unsigned space_type = 0;   // [한국어] 메모리 공간 타입 코드 (통계 버킷)
      // [한국어] 메모리 공간을 통계 버킷 번호로 매핑
      switch (pI->get_space().get_type()) {
        case global_space:
          space_type = 10;       // [한국어] 글로벌 메모리 접근 명령어
          break;
        case local_space:
          space_type = 11;       // [한국어] 로컬 메모리 접근
          break;
        case tex_space:
          space_type = 12;       // [한국어] 텍스처 메모리 접근
          break;
        case surf_space:
          space_type = 13;       // [한국어] 서피스 메모리 접근
          break;
        case param_space_kernel:
        case param_space_local:
          space_type = 14;       // [한국어] 파라미터 메모리 접근 (ld.param)
          break;
        case shared_space:
          space_type = 15;       // [한국어] 공유 메모리 접근
          break;
        case const_space:
          space_type = 16;       // [한국어] 상수 메모리 접근 (ld.const)
          break;
        default:
          space_type = 0;        // [한국어] 공간 없음 (순수 레지스터 연산)
          break;
      }
      // [한국어] 명령어 분류 통계 샘플 추가 (op_classification은 X-매크로의 CLASSIFICATION 값)
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    op_classification);
      // [한국어] 메모리 공간 타입도 통계에 추가 (space_type > 0인 경우만)
      if (space_type)
        StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                          [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                      (int)space_type);
      // [한국어] 옵코드별 실행 횟수 통계 추가
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_op_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    (int)pI->get_opcode());
    }

    // [한국어] 10만 명령어마다 진행 상황 출력 (LIVENESS 디버그 메시지)
    if ((m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn % 100000) == 0) {
      dim3 ctaid = get_ctaid();  // [한국어] 현재 CTA ID (진행 상황 표시용)
      dim3 tid = get_tid();      // [한국어] 현재 스레드 ID
      DPRINTF(LIVENESS,
              "GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "
              "tid=(%u,%u,%u)\n",
              m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, ctaid.x, ctaid.y,
              ctaid.z, tid.x, tid.y, tid.z);
      fflush(stdout);
    }

    // "Return values"
    // [한국어] 타이밍 모델에 메모리 접근 정보 반환: skip하지 않은 레인의 결과만 inst에 기록
    if (!skip) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        // [한국어] MMA_LD/ST는 일반 메모리 주소가 없으므로 제외
        inst.space = insn_space;               // [한국어] 접근한 메모리 공간 (global/shared/local 등)
        inst.set_addr(lane_id, insn_memaddr);  // [한국어] lane_id 슬롯에 유효 주소 저장 (mem_fetch 생성에 사용)
        inst.data_size = insn_data_size;  // simpleAtomicIntrinsics
        // [한국어] 접근 데이터 크기 (바이트) — L1/L2 캐시 요청 크기 결정에 사용
        assert(inst.memory_op == insn_memory_op);  // [한국어] pre_decode에서 설정된 memory_op와 일치 검증
      }
    }

  } catch (int x) {
    // [한국어] PTX 명령어 구현 함수(instructions.cc)에서 int 예외 발생 — 시뮬레이션 불가 상황
    printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x,
           pI->source_file(), pI->source_line());
    printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source());
    abort();  // [한국어] 복구 불가: 강제 종료 (core dump 생성)
  }
}

/*
 * [한국어]
 * cuda_sim::set_param_gpgpu_num_shaders - 시뮬레이터 SM 수 파라미터 설정
 *
 * @num_shaders: SM(Streaming Multiprocessor) 수 — gpgpusim.config의 gpgpu_n_clusters
 *               × gpgpu_n_cores_per_cluster로 계산된 값.
 * @return: 없음 (void)
 *
 * gpgpu_param_num_shaders는 ptx_sim_init_thread()에서 sm_idx를 계산할 때 사용된다.
 * (sm_idx = hw_cta_id * gpgpu_param_num_shaders + sid)
 * 이 값은 공유/로컬 메모리 lookup 테이블의 키 생성에 영향을 미친다.
 *
 * 호출 체인:
 *   gpgpu_sim::init() → [set_param_gpgpu_num_shaders]
 */
void cuda_sim::set_param_gpgpu_num_shaders(int num_shaders) {
  gpgpu_param_num_shaders = num_shaders;  // [한국어] func_sim 컨텍스트에 SM 수 저장
}

/*
 * [한국어]
 * ptx_sim_kernel_info - 커널 함수의 리소스 사용량 정보 반환
 *
 * @kernel: 조회할 커널의 function_info 포인터
 * @return: gpgpu_ptx_sim_info 포인터 — regs/lmem/smem/cmem 필드 포함
 *
 * ptxinfo 파서가 .ptxinfo 파일에서 파싱한 커널별 리소스 정보를 반환한다.
 * max_cta() 계산 시 smem/regs 값을 사용해 SM당 최대 CTA 수를 결정한다.
 *
 * 호출 체인:
 *   gpgpu_cuda_ptx_sim_main_func() / shader.cc → [ptx_sim_kernel_info] → function_info::get_kernel_info()
 */
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const function_info *kernel) {
  return kernel->get_kernel_info();  // [한국어] function_info에 저장된 리소스 정보 구조체 반환
}

/*
 * [한국어]
 * gpgpu_context::ptx_fetch_inst - PC로 warp_inst_t 포인터 조회
 *
 * @pc: 조회할 명령어 PC (program counter)
 * @return: 해당 PC의 warp_inst_t 포인터 (ptx_instruction은 warp_inst_t를 상속함)
 *
 * 타이밍 모델(shader.cc)이 warp의 다음 명령어를 fetch할 때 호출한다.
 * pc_to_instruction()은 g_pc_to_insn 전역 맵에서 lookup한다.
 *
 * 호출 체인:
 *   shader.cc::fetch() → [ptx_fetch_inst] → pc_to_instruction(pc)
 */
const warp_inst_t *gpgpu_context::ptx_fetch_inst(address_type pc) {
  return pc_to_instruction(pc);  // [한국어] g_pc_to_insn 맵에서 PC → ptx_instruction* 조회
}

/*
 * [한국어]
 * ptx_sim_init_thread - 커널 CTA의 스레드를 초기화하고 thread_info를 채우는 핵심 함수
 *
 * @kernel:   실행 중인 커널의 kernel_info_t 참조 — active_threads 리스트와 CTA 반복 상태 보유
 * @thread_info: [in/out] 이전 스레드 포인터 또는 NULL. 종료된 경우 해제 후 새 스레드로 채움.
 * @sid:      이 스레드가 할당될 SM(Shader) ID
 * @tid:      SM 내 스레드 슬롯 번호 (하드웨어 스레드 인덱스, 0 ~ n_thread_per_shader-1)
 * @threads_left: SM에 남은 스레드 슬롯 수 (최소 threads_per_cta 이상 있어야 CTA 할당 가능)
 * @num_threads: SM의 최대 스레드 수 (n_thread_per_shader)
 * @core:     이 스레드가 속할 core_t 포인터 (SM 추상화 객체)
 * @hw_cta_id: SM 내 하드웨어 CTA 슬롯 번호
 * @hw_warp_id: SM 내 하드웨어 warp 슬롯 번호
 * @gpu:      시뮬레이터 GPU 상태 객체 (전역 메모리 등)
 * @isInFunctionalSimulationMode: true이면 기능 시뮬레이션 모드 (타이밍 정보 무시)
 * @return: 1 = 스레드 성공적으로 할당됨, 0 = 더 이상 실행할 CTA 없음 또는 슬롯 부족
 *
 * 이 함수는 두 가지 역할을 한다:
 *   1. 이전 스레드(*thread_info != NULL)가 완료된 경우: 해제 처리 후 active_threads에서 다음 스레드 반환
 *   2. active_threads가 비어 있고 CTA가 남아 있는 경우: 새 CTA의 모든 스레드를 생성하고 큐에 추가
 *
 * static 변수를 사용해 SM별 공유/로컬 메모리 공간을 영속적으로 관리한다:
 *   - shared_memory_lookup[sm_idx]: SM×CTA 슬롯의 공유 메모리 (memory_space_impl<16K>)
 *   - sstarr_memory_lookup[sm_idx]: GT200 PTXPlus 모드의 sstarr 공유 메모리
 *   - ptx_cta_lookup[sm_idx]: CTA 정보 객체 (스레드 완료 추적)
 *   - ptx_warp_lookup[hw_warp_id]: warp 정보 객체
 *   - local_memory_lookup[sid][new_tid]: 스레드별 로컬 메모리 (memory_space_impl<32>)
 *
 * sm_idx 계산 공식:
 *   sm_idx = hw_cta_id * gpgpu_param_num_shaders + sid
 *   (SM ID와 CTA 슬롯 번호를 조합해 고유 인덱스 생성)
 *
 * CTA 재사용: sm_idx가 이미 shared_memory_lookup에 존재하면 메모리를 재사용하고
 *             ptx_cta_info::check_cta_thread_status_and_reset()으로 완료 상태를 리셋한다.
 *
 * 실행 컨텍스트: 타이밍 모델(shader.cc)의 CTA 디스패치 루프에서 호출됨.
 *               기능 시뮬레이션(functionalCoreSim::initializeCTA)에서도 호출됨.
 *
 * 호출 체인:
 *   shader.cc::shader_core_ctx::issue_block2core() → [ptx_sim_init_thread] × threads_per_cta
 *   functionalCoreSim::initializeCTA() → [ptx_sim_init_thread]
 */
unsigned ptx_sim_init_thread(kernel_info_t &kernel,
                             ptx_thread_info **thread_info, int sid,
                             unsigned tid, unsigned threads_left,
                             unsigned num_threads, core_t *core,
                             unsigned hw_cta_id, unsigned hw_warp_id,
                             gpgpu_t *gpu, bool isInFunctionalSimulationMode) {
  std::list<ptx_thread_info *> &active_threads = kernel.active_threads();
  // [한국어] active_threads: 이미 생성된 스레드 객체 큐 (CTA 생성 시 채워지고 이 함수에서 하나씩 꺼냄)

  // [한국어] static 변수: SM별 메모리 공간을 프로그램 수명 동안 유지 (CTA 재사용 지원)
  static std::map<unsigned, memory_space *> shared_memory_lookup;
  // [한국어] sm_idx → 공유 메모리 공간 (SM×CTA 슬롯 단위)
  static std::map<unsigned, memory_space *> sstarr_memory_lookup;
  // [한국어] sm_idx → sstarr 메모리 공간 (PTXPlus GT200 모드 전용)
  static std::map<unsigned, ptx_cta_info *> ptx_cta_lookup;
  // [한국어] sm_idx → CTA 정보 객체 (스레드 완료 카운팅)
  static std::map<unsigned, ptx_warp_info *> ptx_warp_lookup;
  // [한국어] hw_warp_id → warp 정보 객체
  static std::map<unsigned, std::map<unsigned, memory_space *> >
      local_memory_lookup;
  // [한국어] sid → (스레드 전역 인덱스 → 로컬 메모리 공간) 2단계 맵

  if (*thread_info != NULL) {
    // [한국어] 이전 스레드가 이미 할당됨 → 완료 여부 확인 후 해제
    ptx_thread_info *thd = *thread_info;  // [한국어] 이전 스레드 포인터 저장
    assert(thd->is_done());               // [한국어] 완료되지 않은 스레드 재할당 시도는 버그
    if (g_debug_execution == -1) {
      // [한국어] g_debug_execution == -1: 스레드 생명주기 추적 모드
      dim3 ctaid = thd->get_ctaid();   // [한국어] 완료된 스레드의 CTA ID (디버그 출력)
      dim3 t = thd->get_tid();         // [한국어] 완료된 스레드의 내부 ID
      printf(
          "GPGPU-Sim PTX simulator:  thread exiting ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) uid=%u\n",
          ctaid.x, ctaid.y, ctaid.z, t.x, t.y, t.z, thd->get_uid());
      fflush(stdout);
    }
    thd->m_cta_info->register_deleted_thread(thd);  // [한국어] CTA 정보 객체에 스레드 완료 등록
    delete thd;             // [한국어] 스레드 객체 메모리 해제
    *thread_info = NULL;    // [한국어] 포인터 초기화 (dangling pointer 방지)
  }

  if (!active_threads.empty()) {
    // [한국어] 이미 생성된 스레드가 큐에 남아 있음 → 바로 반환 (CTA 재생성 불필요)
    assert(active_threads.size() <= threads_left);  // [한국어] 슬롯 초과 방지
    ptx_thread_info *thd = active_threads.front();  // [한국어] 큐의 첫 번째 스레드
    active_threads.pop_front();                      // [한국어] 큐에서 제거
    *thread_info = thd;                              // [한국어] 호출자에게 스레드 반환
    thd->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
              isInFunctionalSimulationMode);
    // [한국어] 스레드에 GPU, SM, CTA, warp 컨텍스트를 바인딩 (hw 슬롯 정보 설정)
    return 1;  // [한국어] 스레드 할당 성공
  }

  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
    // [한국어] 실행할 CTA가 더 없음 → SM 유휴 대기 (0 반환)
  }

  if (threads_left < kernel.threads_per_cta()) {
    return 0;
    // [한국어] SM에 남은 스레드 슬롯이 CTA 크기보다 작음 → 이 SM에 새 CTA 배치 불가
  }

  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  STARTING THREAD ALLOCATION --> \n");
    fflush(stdout);
  }

  // initializing new CTA
  ptx_cta_info *cta_info = NULL;   // [한국어] 새로 할당하거나 재사용할 CTA 정보 객체
  memory_space *shared_mem = NULL; // [한국어] 이 CTA의 공유 메모리 공간
  memory_space *sstarr_mem = NULL; // [한국어] 이 CTA의 sstarr 메모리 공간 (GTX200 PTXPlus용)

  unsigned cta_size = kernel.threads_per_cta();         // [한국어] CTA당 스레드 수 (예: 256)
  unsigned max_cta_per_sm = num_threads / cta_size;  // e.g., 256 / 48 = 5
  // [한국어] SM의 최대 CTA 수 = SM 스레드 수 / CTA 크기
  assert(max_cta_per_sm > 0);  // [한국어] SM에 최소 1개 이상의 CTA가 들어와야 함

  // unsigned sm_idx = (tid/cta_size)*gpgpu_param_num_shaders + sid;
  unsigned sm_idx =
      hw_cta_id * gpu->gpgpu_ctx->func_sim->gpgpu_param_num_shaders + sid;
  // [한국어] sm_idx: SM×CTA 슬롯의 고유 인덱스
  // 공식: hw_cta_id * 전체_SM_수 + sid
  // 예: CTA슬롯2, SM3, 총4SM → sm_idx = 2*4+3 = 11

  if (shared_memory_lookup.find(sm_idx) == shared_memory_lookup.end()) {
    // [한국어] 이 sm_idx에 처음 할당 → 새 메모리 공간 생성
    if (g_debug_execution >= 1) {
      printf("  <CTA alloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    char buf[512];  // [한국어] 메모리 공간 이름 버퍼
    snprintf(buf, 512, "shared_%u", sid);
    shared_mem = new memory_space_impl<16 * 1024>(buf, 4);
    // [한국어] 공유 메모리: 16KB 청크, 4바이트 정렬 (memory_space_impl 템플릿 인수)
    shared_memory_lookup[sm_idx] = shared_mem;         // [한국어] 새 공유 메모리 등록
    snprintf(buf, 512, "sstarr_%u", sid);
    sstarr_mem = new memory_space_impl<16 * 1024>(buf, 4);
    // [한국어] sstarr 메모리: PTXPlus에서 배열 참조에 사용하는 별도 공유 메모리 영역
    sstarr_memory_lookup[sm_idx] = sstarr_mem;         // [한국어] sstarr 메모리 등록
    cta_info = new ptx_cta_info(sm_idx, gpu->gpgpu_ctx);
    // [한국어] CTA 정보 객체 생성 (스레드 완료 추적 및 barrier 관리)
    ptx_cta_lookup[sm_idx] = cta_info;                 // [한국어] CTA 정보 등록
  } else {
    // [한국어] 이전에 이 슬롯을 사용한 적 있음 → 메모리 재사용 (새 CTA가 같은 슬롯 점유)
    if (g_debug_execution >= 1) {
      printf("  <CTA realloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    shared_mem = shared_memory_lookup[sm_idx];          // [한국어] 기존 공유 메모리 재사용
    sstarr_mem = sstarr_memory_lookup[sm_idx];          // [한국어] 기존 sstarr 메모리 재사용
    cta_info = ptx_cta_lookup[sm_idx];                  // [한국어] 기존 CTA 정보 재사용
    cta_info->check_cta_thread_status_and_reset();      // [한국어] 이전 CTA 스레드 상태 리셋
  }

  std::map<unsigned, memory_space *> &local_mem_lookup =
      local_memory_lookup[sid];
  // [한국어] 이 SM(sid)의 스레드별 로컬 메모리 맵 참조

  while (kernel.more_threads_in_cta()) {
    // [한국어] CTA 내 모든 스레드를 순서대로 생성하여 active_threads에 추가
    dim3 ctaid3d = kernel.get_next_cta_id();           // [한국어] 현재 CTA의 3D 인덱스 (gridDim 내 좌표)
    unsigned new_tid = kernel.get_next_thread_id();    // [한국어] CTA 내 스레드 선형 인덱스
    dim3 tid3d = kernel.get_next_thread_id_3d();       // [한국어] 블록 내 스레드 3D 인덱스 (threadIdx)
    kernel.increment_thread_id();                      // [한국어] 다음 스레드로 반복자 전진
    new_tid += tid;                                    // [한국어] SM 내 전역 스레드 인덱스 = CTA내인덱스 + SM시작tid

    ptx_thread_info *thd = new ptx_thread_info(kernel); // [한국어] 새 스레드 상태 객체 생성

    // [한국어] warp 정보 조회 또는 신규 생성 (같은 hw_warp_id는 warp 정보 공유)
    ptx_warp_info *warp_info = NULL;
    if (ptx_warp_lookup.find(hw_warp_id) == ptx_warp_lookup.end()) {
      warp_info = new ptx_warp_info();                  // [한국어] 새 warp 정보 객체 생성
      ptx_warp_lookup[hw_warp_id] = warp_info;          // [한국어] warp 정보 등록
    } else {
      warp_info = ptx_warp_lookup[hw_warp_id];          // [한국어] 기존 warp 정보 재사용
    }
    thd->m_warp_info = warp_info;  // [한국어] 스레드에 warp 정보 연결

    // [한국어] 로컬 메모리 조회 또는 신규 생성 (스레드별로 독립적인 로컬 메모리)
    memory_space *local_mem = NULL;
    std::map<unsigned, memory_space *>::iterator l =
        local_mem_lookup.find(new_tid);
    if (l != local_mem_lookup.end()) {
      local_mem = l->second;                            // [한국어] 기존 로컬 메모리 재사용
    } else {
      char buf[512];                                    // [한국어] 로컬 메모리 이름 버퍼
      snprintf(buf, 512, "local_%u_%u", sid, new_tid); // [한국어] 이름: "local_<sid>_<tid>"
      local_mem = new memory_space_impl<32>(buf, 32);   // [한국어] 32바이트 청크, 32바이트 정렬
      local_mem_lookup[new_tid] = local_mem;            // [한국어] 새 로컬 메모리 등록
    }

    thd->set_info(kernel.entry());     // [한국어] 스레드에 커널 function_info 설정 (PTX IR 접근용)
    thd->set_nctaid(kernel.get_grid_dim()); // [한국어] gridDim 설정 (nctaid.x/y/z = 그리드 크기)
    thd->set_ntid(kernel.get_cta_dim());    // [한국어] blockDim 설정 (ntid.x/y/z = 블록 크기)
    thd->set_ctaid(ctaid3d);               // [한국어] ctaid 설정 (blockIdx에 해당)
    thd->set_tid(tid3d);                   // [한국어] tid 설정 (threadIdx에 해당)

    if (kernel.entry()->get_ptx_version().extensions())
      thd->cpy_tid_to_reg(tid3d);
    // [한국어] PTX 확장 모드(PTXPlus): tid를 레지스터에 직접 복사 (특수 레지스터 초기화)

    thd->set_valid();                      // [한국어] 스레드 유효 상태로 전환 (실행 가능)
    thd->m_shared_mem = shared_mem;        // [한국어] 공유 메모리 바인딩 (CTA 내 모든 스레드 공유)
    thd->m_sstarr_mem = sstarr_mem;        // [한국어] sstarr 메모리 바인딩 (PTXPlus 전용)

    function_info *finfo = thd->func_info(); // [한국어] 커널의 function_info (파라미터 정보 포함)
    symbol_table *st = finfo->get_symtab();  // [한국어] 심볼 테이블 (파라미터 심볼 참조)
    thd->func_info()->param_to_shared(thd->m_shared_mem, st);
    // [한국어] 커널 파라미터를 공유 메모리에 복사 (ld.param 접근을 위해 실제 값 기록)
    thd->func_info()->param_to_shared(thd->m_sstarr_mem, st);
    // [한국어] sstarr 메모리에도 동일하게 파라미터 복사 (PTXPlus 호환성)

    thd->m_cta_info = cta_info;            // [한국어] CTA 정보 바인딩 (barrier/syncthreads 구현에 사용)
    cta_info->add_thread(thd);             // [한국어] CTA 정보에 이 스레드 등록
    thd->m_local_mem = local_mem;          // [한국어] 로컬 메모리 바인딩 (스레드 전용)

    if (g_debug_execution == -1) {
      printf(
          "GPGPU-Sim PTX simulator:  allocating thread ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) @ 0x%Lx\n",
          ctaid3d.x, ctaid3d.y, ctaid3d.z, tid3d.x, tid3d.y, tid3d.z,
          (unsigned long long)thd);
      fflush(stdout);
    }
    active_threads.push_back(thd);        // [한국어] 초기화 완료 스레드를 큐에 추가 (다음 호출 시 반환)
  }

  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  <-- FINISHING THREAD ALLOCATION\n");
    fflush(stdout);
  }

  kernel.increment_cta_id();  // [한국어] CTA 반복자를 다음 CTA로 전진

  assert(active_threads.size() <= threads_left);  // [한국어] 생성된 스레드 수가 슬롯을 초과하지 않음
  *thread_info = active_threads.front();           // [한국어] 첫 번째 스레드를 호출자에게 반환
  (*thread_info)
      ->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
             isInFunctionalSimulationMode);
  // [한국어] 반환할 스레드에 하드웨어 컨텍스트(SM/CTA/warp/tid) 바인딩
  active_threads.pop_front();  // [한국어] 반환한 스레드를 큐에서 제거
  return 1;                    // [한국어] 스레드 할당 성공
}

/*
 * [한국어]
 * get_kernel_code_size - 커널 PTX 코드의 명령어 바이트 크기 반환
 *
 * @entry: 커널 function_info 포인터
 * @return: PTX 명령어 공간 크기 (바이트) — ptx_assemble() 후 할당된 m_instr_mem 크기
 *
 * 타이밍 모델이 명령어 캐시 크기 계산 등에 사용한다.
 *
 * 호출 체인:
 *   shader.cc / gpu-sim.cc → [get_kernel_code_size] → function_info::get_function_size()
 */
size_t get_kernel_code_size(class function_info *entry) {
  return entry->get_function_size();  // [한국어] m_instr_mem 배열의 바이트 크기 반환
}

/*
 * [한국어]
 * cuda_sim::gpgpu_opencl_ptx_sim_init_grid - OpenCL 커널용 kernel_info_t 생성
 *
 * @entry:   OpenCL 커널에 해당하는 function_info 포인터 (PTX IR)
 * @args:    OpenCL 커널 인수 리스트 (gpgpu_ptx_sim_arg 구조체의 std::list)
 * @gridDim: 그리드 차원 (OpenCL NDRange에서 변환)
 * @blockDim: 블록 차원 (OpenCL workgroup 크기에서 변환)
 * @gpu:     GPU 상태 객체
 * @return:  새로 생성된 kernel_info_t 포인터 (cuda_sim::gpgpu_cuda_ptx_sim_main_func()에 전달)
 *
 * libopencl의 clEnqueueNDRangeKernel() 호출에 의해 트리거된다.
 * CUDA cuLaunchKernel() 경로의 함수와 대응하며, OpenCL 전용 초기화를 수행한다.
 * 인수는 역순으로 add_param_data()에 전달됨에 주의 (argcount - argn).
 *
 * 호출 체인:
 *   libopencl/CL/cl.cc::clEnqueueNDRangeKernel() → [gpgpu_opencl_ptx_sim_init_grid]
 *     → kernel_info_t 생성 → add_param_data() × argcount → finalize()
 */
kernel_info_t *cuda_sim::gpgpu_opencl_ptx_sim_init_grid(
    class function_info *entry, gpgpu_ptx_sim_arg_list_t args,
    struct dim3 gridDim, struct dim3 blockDim, gpgpu_t *gpu) {
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  // [한국어] 커널 정보 객체 생성: 그리드/블록 차원, 함수 정보, 텍스처/배열 이름 매핑 전달
  unsigned argcount = args.size();   // [한국어] 총 인수 개수
  unsigned argn = 1;                 // [한국어] 인수 역순 인덱스 (1부터 시작)
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    // [한국어] add_param_data에 역순 인덱스 사용 — OpenCL 인수 순서 변환
    argn++;  // [한국어] 다음 인수로 전진
  }
  entry->finalize(result->get_param_memory());
  // [한국어] 파라미터 데이터를 파라미터 메모리에 최종 기록
  g_ptx_kernel_count++;   // [한국어] 전역 커널 카운터 증가 (분류 통계 인덱싱에 사용)
  fflush(stdout);          // [한국어] 진행 출력 플러시

  return result;           // [한국어] 생성된 kernel_info_t 반환
}

#include "../../version"      // [한국어] g_gpgpusim_version_string 정의 (GPGPU-Sim 버전 문자열)
#include "detailed_version"   // [한국어] g_gpgpusim_build_string 정의 (빌드 날짜/커밋 문자열)

/*
 * [한국어]
 * print_splash - GPGPU-Sim 버전 정보를 stdout에 1회 출력
 *
 * @return: 없음 (void)
 *
 * static 변수 splash_printed로 중복 출력을 방지한다.
 * read_sim_environment_variables() 또는 시뮬레이터 초기화 시 호출되어
 * 시뮬레이터 이름과 빌드 정보를 사용자에게 표시한다.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc::gpgpu_ptx_sim_init_perf() → [print_splash]
 */
void print_splash() {
  static int splash_printed = 0;  // [한국어] 중복 출력 방지용 static 플래그
  if (!splash_printed) {
    fprintf(stdout, "\n\n        *** %s [build %s] ***\n\n\n",
            g_gpgpusim_version_string, g_gpgpusim_build_string);
    // [한국어] 예: "*** GPGPU-Sim version 4.0.0 (build Jun 13 2026) ***"
    splash_printed = 1;  // [한국어] 이후 호출에서 중복 출력 차단
  }
}

/*
 * [한국어]
 * cuda_sim::gpgpu_ptx_sim_register_const_variable - 상수 변수 주소-이름 매핑 등록
 *
 * @hostVar:    호스트 측 상수 변수의 포인터 (cudaMemcpyToSymbol에서 심볼 식별자로 사용)
 * @deviceName: PTX에서의 심볼 이름 (예: "__cuda_const_variable")
 * @size:       변수 크기 (바이트)
 * @return: 없음 (void)
 *
 * cudaMemcpyToSymbol() 호출 전 CUDA 런타임이 이 함수를 통해 심볼 주소→이름 매핑을 등록한다.
 * g_const_name_lookup: hostVar 포인터 → PTX 심볼 이름 맵.
 * 나중에 gpgpu_ptx_sim_memcpy_symbol()이 이 맵을 참조해 심볼 테이블에서 실제 GPU 주소를 찾는다.
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc::cudaMemcpyToSymbol() 경로의 초기화 단계
 *   → [gpgpu_ptx_sim_register_const_variable]
 */
void cuda_sim::gpgpu_ptx_sim_register_const_variable(void *hostVar,
                                                     const char *deviceName,
                                                     size_t size) {
  printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n",
         deviceName, size);
  g_const_name_lookup[hostVar] = deviceName;
  // [한국어] 호스트 포인터 → PTX 심볼 이름 등록 (상수 메모리 심볼용 맵)
}

/*
 * [한국어]
 * cuda_sim::gpgpu_ptx_sim_register_global_variable - 전역 변수 주소-이름 매핑 등록
 *
 * @hostVar:    호스트 측 전역 변수의 포인터
 * @deviceName: PTX에서의 전역 심볼 이름 (예: "__device__ 변수명")
 * @size:       변수 크기 (바이트)
 * @return: 없음 (void)
 *
 * gpgpu_ptx_sim_register_const_variable()과 동일한 역할이지만 전역 메모리 심볼 전용.
 * g_global_name_lookup 맵에 등록되며 gpgpu_ptx_sim_memcpy_symbol()에서 참조한다.
 *
 * 호출 체인:
 *   libcuda 초기화 → [gpgpu_ptx_sim_register_global_variable]
 */
void cuda_sim::gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                                      const char *deviceName,
                                                      size_t size) {
  printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n",
         deviceName);
  g_global_name_lookup[hostVar] = deviceName;
  // [한국어] 호스트 포인터 → PTX 심볼 이름 등록 (전역 메모리 심볼용 맵)
}

/*
 * [한국어]
 * cuda_sim::gpgpu_ptx_sim_memcpy_symbol - cudaMemcpyToSymbol/FromSymbol 에뮬레이션
 *
 * @hostVar: 심볼 식별자 (포인터 또는 심볼 이름 문자열).
 *           CUDA 4.1 이전: 문자열("심볼명"), CUDA 4.1 이후: 등록된 hostVar 포인터.
 * @src:     복사 원본 데이터 포인터 (to=1이면 GPU로 복사할 호스트 데이터)
 * @count:   복사할 바이트 수
 * @offset:  심볼 내 오프셋 (바이트)
 * @to:      1 = 호스트→GPU 방향 (cudaMemcpyToSymbol), 0 = GPU→호스트 방향 (cudaMemcpyFromSymbol)
 * @gpu:     GPU 상태 객체 (전역 메모리 접근)
 * @return:  없음 (void). 심볼을 찾지 못하면 abort().
 *
 * 이 함수는 cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()을 시뮬레이터 내에서 구현한다.
 * 심볼 탐색 순서:
 *   1. g_const_name_lookup에서 hostVar 포인터로 탐색 (const 변수)
 *   2. g_global_name_lookup에서 hostVar 포인터로 탐색 (global 변수)
 *   3. g_globals / g_constants에서 문자열(hostVar)로 탐색 (CUDA 4.1 이전 방식, deprecated)
 * 심볼명을 찾은 후 g_sym_name_to_symbol_table에서 심볼 테이블을 찾고,
 * symtab->lookup()으로 GPU 주소를 얻어 1바이트 단위로 메모리에 write/read한다.
 *
 * 실행 컨텍스트: 커널 실행 전 호스트 측에서 호출 (CPU 컨텍스트).
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc::cudaMemcpyToSymbol() → [gpgpu_ptx_sim_memcpy_symbol]
 *     → gpu->get_global_memory()->write(dst+n, 1, ...) 반복
 */
void cuda_sim::gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                           size_t count, size_t offset, int to,
                                           gpgpu_t *gpu) {
  printf(
      "GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n",
      hostVar);
  bool found_sym = false;                   // [한국어] 심볼 탐색 성공 여부
  memory_space_t mem_region = undefined_space; // [한국어] 심볼이 속한 메모리 공간
  std::string sym_name;                     // [한국어] 찾은 PTX 심볼 이름

  // [한국어] 1단계: g_const_name_lookup에서 hostVar 포인터로 상수 심볼 탐색
  std::map<const void *, std::string>::iterator c =
      gpu->gpgpu_ctx->func_sim->g_const_name_lookup.find(hostVar);
  if (c != gpu->gpgpu_ctx->func_sim->g_const_name_lookup.end()) {
    found_sym = true;             // [한국어] 상수 심볼로 발견
    sym_name = c->second;         // [한국어] PTX 심볼 이름 저장
    mem_region = const_space;     // [한국어] 상수 메모리 공간으로 마킹
  }

  // [한국어] 2단계: g_global_name_lookup에서 hostVar 포인터로 전역 심볼 탐색
  std::map<const void *, std::string>::iterator g =
      gpu->gpgpu_ctx->func_sim->g_global_name_lookup.find(hostVar);
  if (g != gpu->gpgpu_ctx->func_sim->g_global_name_lookup.end()) {
    if (found_sym) {
      // [한국어] 같은 심볼이 const와 global 모두에 등록 → 선언 오류
      printf(
          "Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared "
          "both const and global?\n",
          sym_name.c_str(), (unsigned long long)hostVar);
      abort();  // [한국어] 이중 등록 오류: 복구 불가
    }
    found_sym = true;             // [한국어] 전역 심볼로 발견
    sym_name = g->second;         // [한국어] PTX 심볼 이름 저장
    mem_region = global_space;    // [한국어] 전역 메모리 공간으로 마킹
  }

  // Weili: Only attempt to find symbol as it is a string
  // if we could not find it in previously registered variable.
  // This will avoid constructing std::string() from hostVar address
  // where it is not a string as
  // Use of a string naming a variable as the symbol parameter was deprecated in
  // CUDA 4.1 and removed in CUDA 5.0.
  // [한국어] 3단계: 포인터 맵에서 못 찾은 경우 — hostVar를 문자열로 해석하는 구형 방식 시도
  // (CUDA 4.1 이전 cudaMemcpyToSymbol(&var_name_string, ...) 호환)
  if (!found_sym) {
    if (g_globals.find(hostVar) != g_globals.end()) {
      // [한국어] g_globals: PTX 파일 파싱 시 등록된 전역 심볼 이름 집합
      found_sym = true;           // [한국어] 전역 심볼 이름으로 발견
      sym_name = hostVar;         // [한국어] hostVar 자체를 심볼 이름 문자열로 사용
      mem_region = global_space;  // [한국어] 전역 메모리
    }
    if (g_constants.find(hostVar) != g_constants.end()) {
      // [한국어] g_constants: PTX 파일 파싱 시 등록된 상수 심볼 이름 집합
      found_sym = true;           // [한국어] 상수 심볼 이름으로 발견
      sym_name = hostVar;         // [한국어] hostVar 자체를 심볼 이름 문자열로 사용
      mem_region = const_space;   // [한국어] 상수 메모리
    }
  }

  if (!found_sym) {
    // [한국어] 세 가지 방법 모두 실패 → 알 수 없는 심볼
    printf("Execution error: No information for PTX symbol w/ hostVar=0x%Lx\n",
           (unsigned long long)hostVar);
    abort();  // [한국어] 심볼 없음: 잘못된 cudaMemcpyToSymbol 호출
  } else
    printf(
        "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ "
        "hostVar=0x%Lx\n",
        (unsigned long long)hostVar);

  const char *mem_name = NULL;   // [한국어] 출력용 메모리 공간 이름 ("const" 또는 "global")
  memory_space *mem = NULL;      // [한국어] 실제 메모리 접근에 사용할 메모리 공간 객체

  // [한국어] 심볼 이름으로 심볼 테이블 탐색 → GPU 주소 획득
  std::map<std::string, symbol_table *>::iterator st =
      gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.find(sym_name.c_str());
  // [한국어] g_sym_name_to_symbol_table: PTX 모듈 이름 → 심볼 테이블 맵 (ptx_parser 관리)
  assert(st != gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.end());
  // [한국어] 심볼 테이블이 없으면 PTX 파싱 오류
  symbol_table *symtab = st->second;  // [한국어] 해당 PTX 모듈의 심볼 테이블

  symbol *sym = symtab->lookup(sym_name.c_str());  // [한국어] 심볼 테이블에서 이름으로 심볼 조회
  assert(sym);   // [한국어] 심볼이 없으면 PTX 선언 오류
  unsigned dst = sym->get_address() + offset;       // [한국어] GPU 주소 = 심볼 기본 주소 + offset

  // [한국어] 메모리 공간 선택: const/global 모두 GPU 전역 메모리 사용 (시뮬레이터에서는 동일)
  switch (mem_region.get_type()) {
    case const_space:
      mem = gpu->get_global_memory();  // [한국어] 상수 메모리 = GPU 전역 메모리 (시뮬레이터 구현)
      mem_name = "const";              // [한국어] 출력용 레이블
      break;
    case global_space:
      mem = gpu->get_global_memory();  // [한국어] 전역 메모리 = GPU 전역 메모리
      mem_name = "global";             // [한국어] 출력용 레이블
      break;
    default:
      abort();  // [한국어] 예상치 못한 메모리 공간: 버그
  }

  printf(
      "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu bytes "
      "%s symbol %s+%zu @0x%x ...\n",
      mem_name, count, (to ? " to " : "from"), sym_name.c_str(), offset, dst);

  // [한국어] 1바이트 단위로 메모리 복사 (시뮬레이터의 memory_space 인터페이스는 바이트 단위 접근)
  for (unsigned n = 0; n < count; n++) {
    if (to)
      mem->write(dst + n, 1, ((char *)src) + n, NULL, NULL);
      // [한국어] to=1: 호스트→GPU 복사 (cudaMemcpyToSymbol), src[n]을 dst+n에 1바이트 write
    else
      mem->read(dst + n, 1, ((char *)src) + n);
      // [한국어] to=0: GPU→호스트 복사 (cudaMemcpyFromSymbol), dst+n에서 1바이트 read하여 src[n]에 저장
  }
  fflush(stdout);  // [한국어] 진행 메시지 플러시
}

extern int ptx_debug;
// [한국어] ptx_debug: PTX 파서(ptx.y)의 디버그 출력 레벨 (g_debug_execution >= 40이면 1로 설정)
// 정의: ptx_ir.cc 또는 ptx.y 에서 `int ptx_debug = 0;`으로 선언

/*
 * [한국어]
 * cuda_sim::read_sim_environment_variables - 환경 변수에서 시뮬레이션 설정 읽기
 *
 * @return: 없음 (void)
 *
 * GPGPU-Sim의 동작을 제어하는 환경 변수들을 읽어 전역 변수에 설정한다.
 * gpgpu_ptx_sim_init_perf() (gpgpusim_entrypoint.cc)에서 시뮬레이터 초기화 시 호출된다.
 *
 * 처리하는 환경 변수:
 *   PTX_SIM_MODE_FUNC:      0=타이밍+기능 시뮬레이션, 1=기능 시뮬레이션만
 *   GPGPUSIM_DEBUG:         설정 시 인터랙티브 디버거 활성화
 *   PTX_SIM_DEBUG:          디버그 출력 레벨 (g_debug_execution, 0~40+)
 *   PTX_SIM_DEBUG_THREAD_UID: 특정 스레드 UID에 대해서만 디버그 출력
 *   PTX_SIM_DEBUG_PC:       특정 PC의 명령어에 대해서만 디버그 출력
 *   PTX_SIM_USE_PTX_FILE:   바이너리 내장 PTX 대신 .ptx 파일 직접 사용 (CUDA > 1010만)
 *   CUDA_LAUNCH_BLOCKING:   1이면 CUDA 커널 동기식 실행
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (CPU, 단일 스레드).
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc::gpgpu_ptx_sim_init_perf() → [read_sim_environment_variables]
 */
void cuda_sim::read_sim_environment_variables() {
  ptx_debug = 0;                          // [한국어] PTX 파서 디버그 출력 비활성화
  g_debug_execution = 0;                  // [한국어] 명령어 실행 디버그 레벨 초기화
  g_interactive_debugger_enabled = false; // [한국어] 인터랙티브 디버거 비활성화

  char *mode = getenv("PTX_SIM_MODE_FUNC");  // [한국어] 시뮬레이션 모드 환경 변수 읽기
  if (mode) sscanf(mode, "%u", &g_ptx_sim_mode);  // [한국어] "0" 또는 "1"을 unsigned로 파싱
  printf(
      "GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC "
      "environment variable:\n",
      g_ptx_sim_mode);
  printf(
      "               1=functional simulation only, 0=detailed performance "
      "simulator)\n");

  char *dbg_inter = getenv("GPGPUSIM_DEBUG");  // [한국어] 인터랙티브 디버거 활성화 환경 변수
  if (dbg_inter && strlen(dbg_inter)) {
    // [한국어] 비어 있지 않은 GPGPUSIM_DEBUG → 인터랙티브 디버거 활성화
    printf("GPGPU-Sim PTX: enabling interactive debugger\n");
    fflush(stdout);
    g_interactive_debugger_enabled = true;  // [한국어] 실행 중 gdb-like 디버거 진입 허용
  }

  char *dbg_level = getenv("PTX_SIM_DEBUG");  // [한국어] 디버그 출력 레벨 환경 변수
  if (dbg_level && strlen(dbg_level)) {
    printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level);
    fflush(stdout);
    sscanf(dbg_level, "%d", &g_debug_execution);  // [한국어] 정수 파싱 (6=레지스터, 10=전체 덤프, 40=파서)
  }

  char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");  // [한국어] 특정 스레드 UID 디버그
  if (dbg_thread && strlen(dbg_thread)) {
    printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n",
           dbg_thread);
    fflush(stdout);
    sscanf(dbg_thread, "%d", &g_debug_thread_uid);  // [한국어] 대상 스레드 UID 설정
  }

  char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");  // [한국어] 특정 PC 디버그 환경 변수
  if (dbg_pc && strlen(dbg_pc)) {
    printf(
        "GPGPU-Sim PTX: printing debug information for instruction with PC = "
        "%s\n",
        dbg_pc);
    fflush(stdout);
    sscanf(dbg_pc, "%llu", &g_debug_pc);  // [한국어] 64비트 PC 주소 파싱
  }

#if CUDART_VERSION > 1010
  // [한국어] CUDA 1010(1.1) 이후 버전: PTX 파일 오버라이드 및 blocking 모드 지원
  g_override_embedded_ptx = false;  // [한국어] 기본값: 바이너리 내장 PTX 사용
  char *usefile = getenv("PTX_SIM_USE_PTX_FILE");  // [한국어] .ptx 파일 직접 사용 여부
  if (usefile && strlen(usefile)) {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with ptx file "
        "(PTX_SIM_USE_PTX_FILE is set)\n");
    fflush(stdout);
    g_override_embedded_ptx = true;  // [한국어] 내장 PTX 대신 파일시스템의 .ptx 파일 파싱
  }
  char *blocking = getenv("CUDA_LAUNCH_BLOCKING");  // [한국어] CUDA_LAUNCH_BLOCKING 환경 변수
  if (blocking && !strcmp(blocking, "1")) {
    g_cuda_launch_blocking = true;  // [한국어] 1이면 커널 launch 후 동기적으로 완료 대기
  }
#else
  // [한국어] CUDA 1010 이하: 항상 blocking + .ptx 파일 사용 (구형 CUDA 호환 모드)
  g_cuda_launch_blocking = true;
  g_override_embedded_ptx = true;
#endif

  if (g_debug_execution >= 40) {
    ptx_debug = 1;  // [한국어] 디버그 레벨 40 이상이면 PTX 파서 verbose 출력 활성화
  }
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
// [한국어] 두 값 중 큰 값 반환 매크로 — elapsed_time 계산에서 최소 1초 보장에 사용

/*
 * [한국어]
 * max_cta - SM당 최대 동시 실행 CTA 수 계산 (리소스 제약 기반)
 *
 * @kernel_info:          커널 리소스 정보 (regs: 레지스터 수, smem: 공유 메모리 바이트)
 * @threads_per_cta:      CTA당 스레드 수 (blockDim.x * blockDim.y * blockDim.z)
 * @warp_size:            워프 크기 (보통 32)
 * @n_thread_per_shader:  SM의 최대 동시 스레드 수 (gpgpu_n_thread_per_shader)
 * @gpgpu_shmem_size:     SM의 총 공유 메모리 크기 (바이트)
 * @gpgpu_shader_registers: SM의 총 레지스터 수 (예: 65536)
 * @max_cta_per_core:     SM당 하드웨어 CTA 슬롯 수 (gpgpu_max_cta_per_core)
 * @return: SM당 최대 실행 가능 CTA 수 (4가지 제약 중 최소값)
 *
 * NVIDIA GPU의 occupancy 계산과 동일한 로직:
 *   1. 스레드 제약: n_thread_per_shader / padded_cta_size
 *      - padded_cta_size: CTA 크기를 warp_size 배수로 올림 (마지막 warp 패딩)
 *   2. 공유 메모리 제약: gpgpu_shmem_size / kernel_info->smem
 *      - smem == 0이면 제약 없음 ((unsigned)-1 = 최대값)
 *   3. 레지스터 제약: gpgpu_shader_registers / (padded_cta_size * 레지스터수_4정렬)
 *      - 레지스터는 4개 단위로 할당 (& ~3으로 반올림)
 *   4. CTA 슬롯 제약: max_cta_per_core (하드웨어 슬롯 수)
 *
 * 실행 컨텍스트: gpgpu_cuda_ptx_sim_main_func()에서 기능 시뮬레이션 전 occupancy 계산.
 *               shader.cc::issue_block2core()에서 타이밍 시뮬레이션 시 CTA 배치 제어.
 *
 * 호출 체인:
 *   gpgpu_cuda_ptx_sim_main_func() → [max_cta]
 *   shader.cc::shader_core_ctx::can_issue_1block() → [max_cta]
 */
unsigned max_cta(const struct gpgpu_ptx_sim_info *kernel_info,
                 unsigned threads_per_cta, unsigned int warp_size,
                 unsigned int n_thread_per_shader,
                 unsigned int gpgpu_shmem_size,
                 unsigned int gpgpu_shader_registers,
                 unsigned int max_cta_per_core) {
  unsigned int padded_cta_size = threads_per_cta;
  // [한국어] CTA 크기를 warp_size 배수로 올림 (마지막 warp의 빈 스레드 슬롯 포함)
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
  // [한국어] 예: threads_per_cta=48, warp_size=32 → padded_cta_size = 64

  unsigned int result_thread = n_thread_per_shader / padded_cta_size;
  // [한국어] 스레드 제약: SM의 총 스레드 슬롯을 패딩된 CTA 크기로 나눔
  // 예: 1024 / 64 = 16 CTA

  unsigned int result_shmem = (unsigned)-1;
  // [한국어] 초기값: 제약 없음 (UINT_MAX) — smem을 사용하지 않는 커널
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;
  // [한국어] 공유 메모리 제약: SM 공유 메모리 총량 / 커널당 공유 메모리 사용량
  // 예: 49152 / 12288 = 4 CTA

  unsigned int result_regs = (unsigned)-1;
  // [한국어] 초기값: 제약 없음 — 레지스터를 사용하지 않는 커널
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));
  // [한국어] 레지스터 제약: SM 레지스터 총수 / (패딩 CTA 크기 × 4정렬 레지스터 수)
  // (kernel_info->regs + 3) & ~3: 레지스터 수를 4의 배수로 올림 (HW 할당 단위)
  // 예: 65536 / (64 * 20) = 51 → 50 정도

  printf("padded cta size is %d and %d and %d", padded_cta_size,
         kernel_info->regs, ((kernel_info->regs + 3) & ~3));
  // [한국어] 디버그: 패딩 CTA 크기, 원본 레지스터 수, 4정렬 레지스터 수 출력

  // Limit by CTA
  unsigned int result_cta = max_cta_per_core;
  // [한국어] CTA 슬롯 제약: 하드웨어가 지원하는 최대 CTA 슬롯 수 (예: 8 or 16)

  // [한국어] 네 가지 제약 중 가장 작은 값이 실제 SM당 최대 CTA 수
  unsigned result = result_thread;              // [한국어] 스레드 제약부터 시작
  result = gs_min2(result, result_shmem);       // [한국어] 공유 메모리 제약 적용
  result = gs_min2(result, result_regs);        // [한국어] 레지스터 제약 적용
  result = gs_min2(result, result_cta);         // [한국어] CTA 슬롯 제약 적용

  // [한국어] 결과 및 병목 원인 출력
  printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
  if (result == result_thread) printf(" threads");  // [한국어] 스레드 수가 병목
  if (result == result_shmem) printf(" shmem");     // [한국어] 공유 메모리가 병목
  if (result == result_regs) printf(" regs");       // [한국어] 레지스터가 병목
  if (result == result_cta) printf(" cta_limit");   // [한국어] 하드웨어 CTA 슬롯이 병목
  printf("\n");

  return result;  // [한국어] SM당 동시 실행 가능한 최대 CTA 수
}
/*!
This function simulates the CUDA code functionally, it takes a kernel_info_t
parameter which holds the data for the CUDA kernel to be executed
!*/
/*
 * [한국어]
 * cuda_sim::gpgpu_cuda_ptx_sim_main_func - 기능 시뮬레이션 전용 커널 실행 함수 (최상위 루프)
 *
 * @kernel:  실행할 커널의 kernel_info_t 참조 (그리드 차원, CTA 반복 상태, 파라미터 메모리 포함)
 * @openCL:  true이면 OpenCL 커널 — 완료 시 stream_manager에 등록하지 않음
 * @return:  없음 (void)
 *
 * 이 함수는 GPGPU-Sim의 **순수 기능 시뮬레이션 최상위 루프**이다.
 * g_ptx_sim_mode == 1인 경우(PTX_SIM_MODE_FUNC=1) 또는 타이밍 모델 없이 결과를 얻을 때 사용한다.
 * 타이밍 시뮬레이션(gpgpu_sim::cycle())과 달리, CTA 단위로 순차 실행한다:
 *   각 CTA는 functionalCoreSim 객체를 통해 완전히 완료될 때까지 실행된 후 다음 CTA로 이동.
 *
 * 실행 순서:
 *   1. PDOM(Post-Dominator) 분석: kernel_func_info->do_pdom()으로 재합류 포인트 계산 (1회)
 *   2. max_cta() 호출로 SM당 최대 CTA 수 계산 (로그 출력용)
 *   3. 체크포인트 설정: cp_op/cp_kernel/cp_count/cp_cta_resume 읽기
 *   4. CTA 루프: kernel.no_more_ctas_to_run()이 true가 될 때까지 반복
 *      a. 체크포인트 조건 확인: cp_op==0 또는 이전 CTA면 실행, 아니면 건너뜀
 *      b. functionalCoreSim::execute()로 이 CTA의 모든 warp 완료까지 실행
 *      c. CDP(CUDA Dynamic Parallelism): launch_all_device_kernels() 호출
 *   5. 체크포인트 저장: cp_op==1이면 전역 메모리를 파일로 저장
 *   6. stream_manager에 커널 완료 등록 (CUDA 모드만)
 *   7. 시뮬레이션 통계 출력 (명령어 수, 경과 시간, 초당 명령어 수)
 *
 * 체크포인트 시스템:
 *   cp_op=1이면 특정 CTA 범위(cp_cta_resume ~ cp_cta_resume+1)의 스레드/warp/메모리 상태를
 *   checkpoint_files/ 디렉토리에 저장하여 나중에 재개할 수 있게 한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드 (타이밍 시뮬레이션 루프와 무관).
 *
 * 호출 체인:
 *   gpgpu_sim::functional_sim() → [gpgpu_cuda_ptx_sim_main_func]
 *     → kernel_func_info->do_pdom()
 *     → functionalCoreSim::execute() × (CTA 수)
 *     → stream_manager::register_finished_kernel()
 */
void cuda_sim::gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel,
                                            bool openCL) {
  printf(
      "GPGPU-Sim: Performing Functional Simulation, executing kernel %s...\n",
      kernel.name().c_str());

  // using a shader core object for book keeping, it is not needed but as most
  // function built for performance simulation need it we use it here
  // extern gpgpu_sim *g_the_gpu;
  // before we execute, we should do PDOM analysis for functional simulation
  // scenario.
  function_info *kernel_func_info = kernel.entry();
  // [한국어] 커널의 function_info: PTX IR, 파라미터 심볼 테이블, 재합류 정보 보유
  const struct gpgpu_ptx_sim_info *kernel_info =
      ptx_sim_kernel_info(kernel_func_info);
  // [한국어] 커널 리소스 정보 (regs/smem/lmem/cmem) — max_cta() 계산에 사용
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();  // [한국어] 체크포인트 저장/복원 객체 생성

  // [한국어] PDOM(Post-Dominator) 분석: warp divergence 재합류 포인트 계산
  if (kernel_func_info->is_pdom_set()) {
    // [한국어] 이미 분석됨 (이전 실행에서) → 재계산 불필요
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kernel.name().c_str());
  } else {
    // [한국어] 처음 실행 → PDOM 분석 수행
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kernel.name().c_str());
    kernel_func_info->do_pdom();   // [한국어] 즉시 후지배자(IPDOM) 기반 재합류 포인트 계산
    kernel_func_info->set_pdom();  // [한국어] 분석 완료 플래그 설정 (중복 분석 방지)
  }

  // [한국어] SM당 최대 CTA 수 계산 (기능 시뮬레이션에서는 실제로 제약을 강제하지 않지만 출력)
  unsigned max_cta_tot = max_cta(
      kernel_info, kernel.threads_per_cta(),
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->n_thread_per_shader,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shmem_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shader_registers,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->max_cta_per_core);
  printf("Max CTA : %d\n", max_cta_tot);  // [한국어] 계산된 SM당 최대 CTA 출력

  // [한국어] 체크포인트 설정 읽기
  int cp_op = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_option;
  // [한국어] 0=체크포인트 없음, 1=체크포인트 저장 모드
  int cp_kernel = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_kernel;
  // [한국어] 체크포인트를 저장할 커널 UID
  cp_count = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_insn_Y;
  // [한국어] 체크포인트 트리거 명령어 수 (이 수에 도달하면 상태 저장)
  cp_cta_resume = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_CTA_t;
  // [한국어] 재개할 CTA 번호 (cp_op==1의 재개 시작 지점)
  int cta_launched = 0;  // [한국어] 현재 커널에서 실행된 CTA 수 카운터

  // we excute the kernel one CTA (Block) at the time, as synchronization
  // functions work block wise
  // [한국어] CTA 단위 실행 루프: __syncthreads()는 블록 범위이므로 블록 단위로 완료해야 함
  while (!kernel.no_more_ctas_to_run()) {
    unsigned temp = kernel.get_next_cta_id_single();
    // [한국어] 다음에 실행할 CTA의 선형 ID (단일 정수 인덱스)

    // [한국어] 체크포인트 조건: 이 CTA를 실행할지 건너뛸지 결정
    if (cp_op == 0 ||
        (cp_op == 1 && cta_launched < cp_cta_resume &&
         kernel.get_uid() == cp_kernel) ||
        kernel.get_uid() < cp_kernel)  // just fro testing
    {
      // [한국어] 조건 설명:
      //   cp_op==0: 체크포인트 없음 → 항상 실행
      //   cp_op==1 && 이전 CTA && 대상 커널: 재개 지점까지의 CTA는 실행
      //   현재 커널이 체크포인트 커널보다 이전: 무조건 실행
      functionalCoreSim cta(
          &kernel, gpgpu_ctx->the_gpgpusim->g_the_gpu,
          gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size);
      // [한국어] functionalCoreSim: 단일 CTA를 기능 시뮬레이션하는 임시 코어 객체
      cta.execute(cp_count, temp);
      // [한국어] 이 CTA의 모든 warp를 완료까지 실행 (cp_count: 체크포인트 명령어 수 한도)

#if (CUDART_VERSION >= 5000)
      gpgpu_ctx->device_runtime->launch_all_device_kernels();
      // [한국어] CDP(CUDA Dynamic Parallelism, CUDA 5.0+): GPU가 CTA 실행 중 생성한 자식 커널 실행
#endif
    } else {
      kernel.increment_cta_id();
      // [한국어] 체크포인트 재개 지점 이후의 CTA → 건너뜀 (이미 checkpoint에서 복원됨)
    }
    cta_launched++;  // [한국어] 실행(또는 건너뜀) CTA 수 증가
  }

  // [한국어] 체크포인트 저장: cp_op==1이면 전역 메모리를 파일로 저장
  if (cp_op == 1) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             kernel.get_uid());
    // [한국어] 파일명: "checkpoint_files/global_mem_<kernel_uid>.txt"
    g_checkpoint->store_global_mem(
        gpgpu_ctx->the_gpgpusim->g_the_gpu->get_global_memory(), f1name,
        (char *)"%08x");
    // [한국어] 전역 메모리 전체를 16진수 형식으로 파일에 저장
  }

  // registering this kernel as done

  // openCL kernel simulation calls don't register the kernel so we don't
  // register its exit
  // [한국어] 커널 완료 등록: CUDA 모드만 (OpenCL은 스트림 관리자가 별도 처리)
  if (!openCL) {
    // extern stream_manager *g_stream_manager;
    gpgpu_ctx->the_gpgpusim->g_stream_manager->register_finished_kernel(
        kernel.get_uid());
    // [한국어] stream_manager에 커널 완료 통보 → cudaDeviceSynchronize() 해제
  }

  //******PRINTING*******
  printf("GPGPU-Sim: Done functional simulation (%u instructions simulated).\n",
         g_ptx_sim_num_insn);
  // [한국어] 전체 기능 시뮬레이션에서 실행된 총 PTX 명령어 수 출력
  if (gpgpu_ptx_instruction_classification) {
    StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
    // [한국어] 명령어 분류별 통계 출력 (op_classification 버킷)
    StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
    // [한국어] 옵코드별 실행 횟수 통계 출력
  }

  // time_t variables used to calculate the total simulation time
  // the start time of simulation is hold by the global variable
  // g_simulation_starttime g_simulation_starttime is initilized by
  // gpgpu_ptx_sim_init_perf() in gpgpusim_entrypoint.cc upon starting gpgpu-sim
  time_t end_time, elapsed_time, days, hrs, minutes, sec;
  end_time = time((time_t *)NULL);   // [한국어] 현재 시각 (UNIX timestamp, 초 단위)
  elapsed_time =
      MAX(end_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  // [한국어] 경과 시간 = 종료 시각 - 시작 시각 (최소 1초 보장, 0으로 나누기 방지)

  // calculating and printing simulation time in terms of days, hours, minutes
  // and seconds
  // [한국어] 경과 시간을 일/시/분/초로 분해하여 출력
  days = elapsed_time / (3600 * 24);   // [한국어] 전체 날 수
  hrs = elapsed_time / 3600 - 24 * days;  // [한국어] 시간 (날 제외)
  minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);  // [한국어] 분 (시 제외)
  sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));  // [한국어] 초 (분 제외)

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec,
      (unsigned)elapsed_time);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(g_ptx_sim_num_insn / elapsed_time));
  // [한국어] 초당 시뮬레이션된 명령어 수 (시뮬레이터 성능 지표)
  fflush(stdout);
}

/*
 * [한국어]
 * functionalCoreSim::initializeCTA - 기능 시뮬레이션용 CTA를 초기화
 *
 * @ctaid_cp: 현재 실행할 CTA의 선형 ID (체크포인트 파일명 생성에 사용)
 * @return: 없음 (void)
 *
 * 이 함수는 functionalCoreSim 객체가 하나의 CTA를 실행하기 전에 호출된다.
 * 다음을 수행한다:
 *   1. m_warpAtBarrier[], m_liveThreadCount[] 초기화
 *   2. m_thread[] 배열을 NULL로 초기화
 *   3. CTA의 모든 스레드를 ptx_sim_init_thread()로 생성하고 m_thread[]에 배치
 *   4. 체크포인트 재개 모드(cp_cta_resume==1)이면 각 스레드의 레지스터 상태를 파일에서 복원
 *   5. 각 warp를 createWarp()로 SIMT 스택 초기화
 *
 * 실행 컨텍스트: functionalCoreSim::execute()에서 CTA 루프 진입 시 호출.
 *
 * 호출 체인:
 *   functionalCoreSim::execute() → [initializeCTA] → ptx_sim_init_thread() × threads_per_cta
 *     → createWarp() × m_warp_count
 */
void functionalCoreSim::initializeCTA(unsigned ctaid_cp) {
  int ctaLiveThreads = 0;  // [한국어] 생성된 스레드 수 카운터 (디버그 용도)
  symbol_table *symtab = m_kernel->entry()->get_symtab();
  // [한국어] 커널의 심볼 테이블 — 체크포인트 복원 시 레지스터 이름 조회에 필요

  // [한국어] warp 상태 초기화
  for (int i = 0; i < m_warp_count; i++) {
    m_warpAtBarrier[i] = false;    // [한국어] 배리어 대기 상태 초기화
    m_liveThreadCount[i] = 0;      // [한국어] warp별 활성 스레드 수 초기화
  }
  for (int i = 0; i < m_warp_count * m_warp_size; i++) m_thread[i] = NULL;
  // [한국어] 스레드 포인터 배열 초기화 (모두 NULL)

  // get threads for a cta
  // [한국어] CTA의 모든 스레드를 순서대로 생성 및 초기화
  for (unsigned i = 0; i < m_kernel->threads_per_cta(); i++) {
    ptx_sim_init_thread(*m_kernel, &m_thread[i], 0, i,
                        m_kernel->threads_per_cta() - i,
                        m_kernel->threads_per_cta(), this, 0, i / m_warp_size,
                        (gpgpu_t *)m_gpu, true);
    // [한국어] 스레드 i를 생성: sid=0(기능 시뮬레이션은 SM 0), tid=i, hw_warp_id=i/warp_size
    // isInFunctionalSimulationMode=true → 타이밍 정보 무시
    assert(m_thread[i] != NULL && !m_thread[i]->is_done());
    // [한국어] 스레드 생성 및 유효 상태 확인
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/thread_%d_0_reg.txt", i);
    // [한국어] 체크포인트 레지스터 파일명: "checkpoint_files/thread_<i>_0_reg.txt"
    if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1)
      m_thread[i]->resume_reg_thread(fname, symtab);
    // [한국어] 체크포인트 재개 모드: 저장된 레지스터 상태를 파일에서 복원
    ctaLiveThreads++;  // [한국어] 생성된 스레드 수 증가
  }

  // [한국어] 각 warp의 SIMT 스택 초기화
  for (int k = 0; k < m_warp_count; k++) createWarp(k);
  // [한국어] createWarp(): SIMT 스택에 초기 active mask와 PC를 launch
}

/*
 * [한국어]
 * functionalCoreSim::createWarp - 하나의 warp의 SIMT 스택과 active mask 초기화
 *
 * @warpId: 초기화할 warp ID (0 ~ m_warp_count-1)
 * @return: 없음 (void)
 *
 * warpId번 warp의 SIMT 스택을 초기 상태로 설정한다:
 *   1. initialMask를 전체 활성(all 1)으로 설정한 후, NULL 스레드 슬롯은 비활성으로 reset
 *      (CTA의 마지막 warp가 warp_size보다 작은 스레드를 가질 때 발생)
 *   2. m_simt_stack[warpId]->launch(pc, initialMask): 스택에 초기 PC와 active mask 등록
 *   3. 체크포인트 재개 모드: SIMT 스택을 파일에서 복원하고 warp의 각 스레드 PC를 갱신
 *
 * 실행 컨텍스트: initializeCTA()에서 CTA의 모든 스레드 생성 후 호출.
 *
 * 호출 체인:
 *   initializeCTA() → [createWarp] × m_warp_count
 *     → m_simt_stack[warpId]->launch() 또는 resume()
 */
void functionalCoreSim::createWarp(unsigned warpId) {
  simt_mask_t initialMask;          // [한국어] warp의 초기 active mask (bitset)
  unsigned liveThreadsCount = 0;    // [한국어] 이 warp의 유효한 스레드 수
  initialMask.set();                // [한국어] 모든 레인을 활성으로 초기화 (all 1)

  // [한국어] NULL 스레드(패딩 슬롯)는 비활성으로 처리
  for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
       i++) {
    if (m_thread[i] == NULL)
      initialMask.reset(i - warpId * m_warp_size);
      // [한국어] 스레드가 없는 레인(마지막 warp의 빈 슬롯)을 비활성으로 설정
    else
      liveThreadsCount++;  // [한국어] 유효 스레드 수 카운트
  }

  assert(m_thread[warpId * m_warp_size] != NULL);
  // [한국어] warp의 첫 번째 스레드는 반드시 존재해야 함 (warp 0번 슬롯 필수)
  m_simt_stack[warpId]->launch(m_thread[warpId * m_warp_size]->get_pc(),
                               initialMask);
  // [한국어] SIMT 스택에 초기 항목 push: (PC=첫 스레드의 PC, active_mask=initialMask)
  // warp 내 모든 스레드는 같은 시작 PC를 가짐

  char fname[2048];
  snprintf(fname, 2048, "checkpoint_files/warp_%d_0_simt.txt", warpId);
  // [한국어] 체크포인트 SIMT 스택 파일명: "checkpoint_files/warp_<warpId>_0_simt.txt"

  if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1) {
    // [한국어] 체크포인트 재개 모드: SIMT 스택 상태를 파일에서 복원
    unsigned pc, rpc;       // [한국어] 복원된 현재 PC와 재합류 PC
    m_simt_stack[warpId]->resume(fname);  // [한국어] 저장된 SIMT 스택 상태 파일에서 읽기
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
    // [한국어] 복원된 스택 상단에서 현재 PC(pc)와 재합류 PC(rpc) 획득

    // [한국어] warp의 모든 스레드 PC를 복원된 값으로 설정
    for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
         i++) {
      m_thread[i]->set_npc(pc);   // [한국어] NPC를 복원된 PC로 설정
      m_thread[i]->update_pc();   // [한국어] PC를 NPC로 전진 (복원 적용)
    }
  }
  m_liveThreadCount[warpId] = liveThreadsCount;
  // [한국어] warp의 유효 스레드 수 저장 (executeWarp에서 완료 판단에 사용)
}

/*
 * [한국어]
 * functionalCoreSim::execute - 하나의 CTA를 기능 시뮬레이션으로 완전 실행
 *
 * @inst_count: 체크포인트 트리거 명령어 수 한도 (0이면 한도 없음)
 * @ctaid_cp:   현재 실행하는 CTA의 선형 ID (체크포인트 파일명과 범위 판단에 사용)
 * @return:     없음 (void)
 *
 * 이 함수는 하나의 CTA의 모든 warp를 순환하며 완료될 때까지 실행하는 루프이다.
 * 루프 종료 조건:
 *   - someOneLive == false: 모든 warp가 완료됨 (정상 종료)
 *   - inst_count > 0 && count > inst_count && 체크포인트 범위 내: 명령어 수 한도 초과 (체크포인트 저장)
 *
 * 배리어 처리:
 *   allAtBarrier: 모든 warp가 __syncthreads()에 도달한 상태
 *   → m_warpAtBarrier[] 전체 리셋: 모든 warp가 배리어를 동시에 통과
 *
 * 체크포인트 저장:
 *   체크포인트 조건 충족 시 CTA의 상태를 저장:
 *   - 공유 메모리 ("checkpoint_files/shared_mem_<ctaid>.txt")
 *   - 각 스레드의 레지스터 ("checkpoint_files/thread_<i>_<ctaid>_reg.txt")
 *   - 각 스레드의 로컬 메모리 ("checkpoint_files/local_mem_thread_<i>_<ctaid>_reg.txt")
 *   - 각 warp의 SIMT 스택 ("checkpoint_files/warp_<i>_<ctaid>_simt.txt")
 *
 * 실행 컨텍스트: gpgpu_cuda_ptx_sim_main_func()의 CTA 루프에서 호출.
 *
 * 호출 체인:
 *   gpgpu_cuda_ptx_sim_main_func() → [execute] → initializeCTA() → executeWarp() × (warp×step)
 */
void functionalCoreSim::execute(int inst_count, unsigned ctaid_cp) {
  m_gpu->gpgpu_ctx->func_sim->cp_count = m_gpu->checkpoint_insn_Y;
  // [한국어] 체크포인트 트리거 명령어 수를 GPU 설정에서 갱신
  m_gpu->gpgpu_ctx->func_sim->cp_cta_resume = m_gpu->checkpoint_CTA_t;
  // [한국어] 체크포인트 재개 CTA 번호를 GPU 설정에서 갱신
  initializeCTA(ctaid_cp);
  // [한국어] CTA의 모든 스레드와 warp SIMT 스택 초기화

  int count = 0;  // [한국어] 이번 CTA에서 실행한 warp 실행 횟수 카운터
  while (true) {
    bool someOneLive = false;   // [한국어] 하나라도 살아있는 warp가 있는지
    bool allAtBarrier = true;   // [한국어] 모든 warp가 배리어 대기 중인지

    // [한국어] 각 warp를 한 번씩 실행 (한 명령어 단계)
    for (unsigned i = 0; i < m_warp_count; i++) {
      executeWarp(i, allAtBarrier, someOneLive);
      // [한국어] warp i를 하나의 명령어 단계 실행 → someOneLive/allAtBarrier 업데이트
      count++;  // [한국어] warp 실행 횟수 증가
    }

    // [한국어] 체크포인트 한도 초과 판단: 지정 명령어 수 이상 실행 + 체크포인트 범위 내
    if (inst_count > 0 && count > inst_count &&
        (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
        (ctaid_cp >= m_gpu->checkpoint_CTA) &&
        (ctaid_cp < m_gpu->checkpoint_CTA_t) && m_gpu->checkpoint_option == 1) {
      someOneLive = false;  // [한국어] 강제로 루프 종료 조건 설정 (체크포인트 저장을 위해)
      break;                // [한국어] 체크포인트 저장 단계로 이동
    }

    if (!someOneLive) break;  // [한국어] 모든 warp 완료 → CTA 실행 완료, 루프 탈출

    // [한국어] 배리어 처리: 모든 warp가 __syncthreads()에 도달했으면 배리어 해제
    if (allAtBarrier) {
      for (unsigned i = 0; i < m_warp_count; i++) m_warpAtBarrier[i] = false;
      // [한국어] 모든 warp의 배리어 플래그 리셋 → 다음 사이클에서 재실행 가능
    }
  }

  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();  // [한국어] 체크포인트 저장 객체 생성

  ptx_reg_t regval;
  regval.u64 = 123;  // [한국어] 미사용 초기화 변수 (레거시 코드)

  unsigned ctaid = m_kernel->get_next_cta_id_single();
  // [한국어] 다음 CTA ID (현재 CTA가 완료된 후이므로 이미 증가된 상태 — ctaid-1이 현재 CTA)

  // [한국어] 체크포인트 저장 조건: 체크포인트 옵션 켜짐 + 대상 커널 + CTA 범위 내
  if (m_gpu->checkpoint_option == 1 &&
      (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
      (ctaid_cp >= m_gpu->checkpoint_CTA) &&
      (ctaid_cp < m_gpu->checkpoint_CTA_t)) {
    // [한국어] 공유 메모리 저장
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid - 1);
    // [한국어] 파일명: "checkpoint_files/shared_mem_<현재ctaid>.txt" (ctaid-1 = 현재 완료된 CTA)
    g_checkpoint->store_global_mem(m_thread[0]->m_shared_mem, fname,
                                   (char *)"%08x");
    // [한국어] 이 CTA의 공유 메모리 전체를 16진수 형식으로 저장

    // [한국어] 각 스레드의 레지스터 + 로컬 메모리 저장
    for (int i = 0; i < 32 * m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt", i,
               ctaid - 1);
      m_thread[i]->print_reg_thread(fname);
      // [한국어] 스레드 i의 레지스터 파일 전체를 파일로 저장
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i, ctaid - 1);
      g_checkpoint->store_global_mem(m_thread[i]->m_local_mem, f1name,
                                     (char *)"%08x");
      // [한국어] 스레드 i의 로컬 메모리 전체를 파일로 저장
      m_thread[i]->set_done();      // [한국어] 스레드를 완료 상태로 전환
      m_thread[i]->exitCore();      // [한국어] 코어 종료 처리 (활성 스레드 카운트 감소)
      m_thread[i]->registerExit();  // [한국어] CTA 정보에 스레드 종료 등록
    }

    // [한국어] 각 warp의 SIMT 스택 상태 저장
    for (int i = 0; i < m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt", i,
               ctaid - 1);
      // [한국어] 파일명: "checkpoint_files/warp_<warpId>_<ctaid>_simt.txt"
      FILE *fp = fopen(fname, "w");       // [한국어] 파일 열기 (쓰기 모드)
      assert(fp != NULL);                  // [한국어] 파일 열기 실패 시 중단
      m_simt_stack[i]->print_checkpoint(fp); // [한국어] SIMT 스택 전체 상태를 텍스트로 저장
      fclose(fp);                          // [한국어] 파일 닫기
    }
  }
}

/*
 * [한국어]
 * functionalCoreSim::executeWarp - 하나의 warp를 한 명령어 단계 실행
 *
 * @i:           실행할 warp ID
 * @allAtBarrier: [in/out] 모든 warp가 배리어 대기 중인지 (false로 갱신될 수 있음)
 * @someOneLive:  [in/out] 살아있는 warp가 하나라도 있는지 (true로 갱신될 수 있음)
 * @return:      없음 (void)
 *
 * warp i가 배리어 대기 중이지 않고 활성 스레드가 남아 있으면:
 *   1. getExecuteWarp(): SIMT 스택에서 현재 active mask와 PC를 가져와 warp_inst_t 생성
 *   2. execute_warp_inst_t(): warp의 모든 활성 레인에 ptx_exec_inst() 호출
 *   3. isatomic()이면 do_atomic(): 원자 연산 콜백 실행
 *   4. BARRIER_OP/MEMORY_BARRIER_OP이면: m_warpAtBarrier[i] = true (배리어 대기 상태)
 *   5. updateSIMTStack(): 명령어 실행 결과로 SIMT 스택 갱신 (분기/재합류 처리)
 *
 * 이후 m_liveThreadCount 및 m_warpAtBarrier를 기반으로 someOneLive/allAtBarrier 갱신.
 *
 * 실행 컨텍스트: execute()의 warp 루프에서 호출.
 *
 * 호출 체인:
 *   execute() → [executeWarp] → getExecuteWarp() → execute_warp_inst_t()
 *     → ptx_exec_inst() × (active 레인 수) → updateSIMTStack()
 */
void functionalCoreSim::executeWarp(unsigned i, bool &allAtBarrier,
                                    bool &someOneLive) {
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] != 0) {
    // [한국어] 배리어 대기 중이지 않고 활성 스레드가 있는 경우에만 실행
    warp_inst_t inst = getExecuteWarp(i);
    // [한국어] SIMT 스택 상단에서 active mask와 PC를 읽어 이번 사이클의 명령어 생성
    execute_warp_inst_t(inst, i);
    // [한국어] warp i의 모든 활성 레인에 대해 ptx_exec_inst()를 순서대로 호출
    if (inst.isatomic()) inst.do_atomic(true);
    // [한국어] 원자 명령어(ATOM_OP)이면 등록된 콜백 실행 (메모리 수정 적용)
    if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP)
      m_warpAtBarrier[i] = true;
    // [한국어] __syncthreads()(BARRIER_OP) 또는 membar(MEMORY_BARRIER_OP) 도달 → 배리어 대기
    updateSIMTStack(i, &inst);
    // [한국어] 명령어 결과(active mask 변화, 분기 등)로 SIMT 스택 갱신
    // 분기 명령어면 목적지 PC와 재합류 PC를 스택에 push/pop
  }
  if (m_liveThreadCount[i] > 0) someOneLive = true;
  // [한국어] 이 warp에 활성 스레드가 남아 있으면 someOneLive=true (루프 계속)
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] > 0) allAtBarrier = false;
  // [한국어] 배리어 대기 중이 아닌 살아있는 warp가 하나라도 있으면 allAtBarrier=false
}

/*
 * [한국어]
 * gpgpu_context::translate_pc_to_ptxlineno - PC를 PTX 소스 라인 번호로 변환
 *
 * @pc: 변환할 PTX 명령어 PC
 * @return: 해당 명령어의 PTX 소스 파일 라인 번호
 *
 * 디버그 출력이나 PTX 라인별 통계(ptx_file_line_stats)에서 사용한다.
 * 전제: 커널이 단일 PTX 파일 내에 있음 (멀티 파일 지원 제한).
 *
 * 호출 체인:
 *   cuda_sim / 디버거 → [translate_pc_to_ptxlineno] → pc_to_instruction(pc) → pInsn->source_line()
 */
unsigned gpgpu_context::translate_pc_to_ptxlineno(unsigned pc) {
  // this function assumes that the kernel fits inside a single PTX file
  // function_info *pFunc = g_func_info; // assume that the current kernel is
  // the one in query
  const ptx_instruction *pInsn = pc_to_instruction(pc);
  // [한국어] PC에 해당하는 ptx_instruction 포인터 조회 (g_pc_to_insn 맵)
  unsigned ptx_line_number = pInsn->source_line();
  // [한국어] ptx_instruction에 저장된 PTX 소스 파일 라인 번호 (파싱 시 기록)

  return ptx_line_number;  // [한국어] 호출자에게 PTX 라인 번호 반환
}

// ptxinfo parser
/*
 * [한국어] ptxinfo 파서 그룹 — .ptxinfo 파일에서 커널 리소스 사용량을 파싱하여 function_info에 등록
 *
 * .ptxinfo 파일은 ptxas(NVIDIA PTX 어셈블러)가 PTX 컴파일 후 생성하는 부가 정보 파일로,
 * 커널별 레지스터 수, 로컬/공유/상수/전역 메모리 사용량을 담고 있다.
 * 이 정보는 max_cta() 계산과 시뮬레이터 occupancy 분석에 필수적이다.
 *
 * 파서 동작:
 *   ptxinfo.l (Flex 렉서) + ptxinfo.y (Bison 파서)가 .ptxinfo 파일을 파싱하면서
 *   아래의 C 함수들을 콜백으로 호출해 g_ptxinfo에 값을 채운다.
 *   파싱 완료 후 ptxinfo_opencl_addinfo() 또는 CUDA 경로의 별도 함수가
 *   function_info::set_kernel_info()를 호출해 커널 객체에 리소스 정보를 등록한다.
 *
 * 전역 변수:
 *   g_ptxinfo_kname: 현재 파싱 중인 커널 이름 (동적 할당, strdup 사용)
 *   g_ptxinfo:       현재 파싱 중인 커널의 리소스 정보 집계 구조체
 *   g_duplicate:     PTX 라인 번호 → 중복 타입 이름 맵 (cuobjdump 정보)
 *   g_last_dup_type: 마지막으로 파싱된 중복 타입 이름 (ptxinfo_linenum에서 사용)
 */

extern std::map<unsigned, const char *> get_duplicate();
// [한국어] get_duplicate() 전방 선언 (하단에 정의, 파서가 호출)

static char *g_ptxinfo_kname = NULL;
// [한국어] 현재 파싱 중인 커널 이름 (strdup으로 할당, clear_ptxinfo에서 free)
// 설정자: ptxinfo_function() — 새 커널 블록 시작 시
// 읽는 자: get_ptxinfo_kname(), print_ptxinfo(), ptxinfo_opencl_addinfo()

static struct gpgpu_ptx_sim_info g_ptxinfo;
// [한국어] 현재 파싱 중인 커널의 리소스 사용량 집계 구조체
// 필드: regs(레지스터 수), lmem(로컬 메모리), smem(공유 메모리), cmem(상수 메모리), gmem(전역 메모리)
// 설정자: ptxinfo_regs/lmem/smem/cmem/gmem — 파서 콜백
// 읽는 자: get_ptxinfo(), ptxinfo_opencl_addinfo() → function_info::set_kernel_info()

static std::map<unsigned, const char *> g_duplicate;
// [한국어] PTX 소스 라인 번호 → 중복 명령어 타입 이름 맵
// cuobjdump 출력에서 중복 PTX 라인을 추적하는 데 사용

static const char *g_last_dup_type;
// [한국어] 마지막으로 파싱된 중복 타입 이름 (ptxinfo_dup_type에서 설정, ptxinfo_linenum에서 사용)

/*
 * [한국어]
 * get_ptxinfo_kname - 현재 파싱 중인 커널 이름 반환
 *
 * @return: g_ptxinfo_kname (NULL이면 커널 이름 없는 바이너리 정보 섹션)
 */
const char *get_ptxinfo_kname() { return g_ptxinfo_kname; }

/*
 * [한국어]
 * print_ptxinfo - 현재 파싱된 ptxinfo 내용을 stdout에 출력 (디버그)
 *
 * @return: 없음 (void)
 *
 * 커널 이름 유무에 따라 두 가지 형식으로 출력:
 *   이름 없음: "Binary info : gmem=N, cmem=N"
 *   이름 있음: "Kernel 'name' : regs=N, lmem=N, smem=N, cmem=N"
 */
void print_ptxinfo() {
  if (!get_ptxinfo_kname()) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
    // [한국어] 커널 이름 없는 바이너리 전역 메모리/상수 메모리 정보
  }
  if (get_ptxinfo_kname()) {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        get_ptxinfo_kname(), g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
    // [한국어] 커널별 리소스 사용량 출력 (레지스터, 로컬/공유/상수 메모리)
  }
}

/*
 * [한국어]
 * get_ptxinfo - 현재 파싱된 ptxinfo 구조체 복사본 반환
 *
 * @return: gpgpu_ptx_sim_info 구조체 (값 복사)
 */
struct gpgpu_ptx_sim_info get_ptxinfo() {
  return g_ptxinfo;  // [한국어] g_ptxinfo 복사본 반환 (파서 호출자가 사용)
}

/*
 * [한국어]
 * get_duplicate - PTX 라인 번호 → 중복 타입 맵 반환
 *
 * @return: g_duplicate 복사본 (값 복사)
 */
std::map<unsigned, const char *> get_duplicate() { return g_duplicate; }

/*
 * [한국어]
 * ptxinfo_linenum - 파서 콜백: 중복 라인 번호 등록
 *
 * @linenum: 중복 PTX 라인 번호
 *
 * 호출 체인: ptxinfo.y 파서 → [ptxinfo_linenum]
 */
void ptxinfo_linenum(unsigned linenum) {
  g_duplicate[linenum] = g_last_dup_type;  // [한국어] 라인 번호 → 마지막 중복 타입 이름 등록
}

/*
 * [한국어]
 * ptxinfo_dup_type - 파서 콜백: 중복 타입 이름 설정
 *
 * @dup_type: 중복 타입 이름 문자열 (정적 문자열, 해제 불필요)
 */
void ptxinfo_dup_type(const char *dup_type) { g_last_dup_type = dup_type; }
// [한국어] 다음 ptxinfo_linenum 호출 시 이 값이 맵에 저장됨

/*
 * [한국어]
 * ptxinfo_function - 파서 콜백: 새 커널 파싱 시작
 *
 * @fname: 커널 함수 이름 문자열
 *
 * 이전 g_ptxinfo 상태를 초기화하고 새 커널 이름을 설정한다.
 * 각 ".entry" 선언 앞에 호출됨.
 */
void ptxinfo_function(const char *fname) {
  clear_ptxinfo();                    // [한국어] 이전 커널 정보 초기화
  g_ptxinfo_kname = strdup(fname);    // [한국어] 커널 이름 동적 복사 (clear_ptxinfo에서 free)
}

/*
 * [한국어]
 * ptxinfo_regs - 파서 콜백: 커널의 레지스터 사용량 설정
 *
 * @nregs: 이 커널에서 사용하는 레지스터 수
 */
void ptxinfo_regs(unsigned nregs) { g_ptxinfo.regs = nregs; }
// [한국어] max_cta()의 result_regs 계산에 사용

/*
 * [한국어]
 * ptxinfo_lmem - 파서 콜백: 커널의 로컬 메모리 사용량 설정
 *
 * @declared: PTX에서 선언된 로컬 변수 크기
 * @system:   시스템이 암묵적으로 사용하는 로컬 메모리 크기
 */
void ptxinfo_lmem(unsigned declared, unsigned system) {
  g_ptxinfo.lmem = declared + system;  // [한국어] 선언 + 시스템 로컬 메모리 합산
}

/*
 * [한국어]
 * ptxinfo_gmem - 파서 콜백: 전역 메모리 사용량 설정
 *
 * @declared: 선언된 전역 메모리 크기
 * @system:   시스템 전역 메모리 크기
 */
void ptxinfo_gmem(unsigned declared, unsigned system) {
  g_ptxinfo.gmem = declared + system;  // [한국어] 선언 + 시스템 전역 메모리 합산
}

/*
 * [한국어]
 * ptxinfo_smem - 파서 콜백: 공유 메모리 사용량 설정
 *
 * @declared: 선언된 공유 메모리 크기 (바이트)
 * @system:   시스템 공유 메모리 크기
 */
void ptxinfo_smem(unsigned declared, unsigned system) {
  g_ptxinfo.smem = declared + system;  // [한국어] 선언 + 시스템 공유 메모리 합산
}

/*
 * [한국어]
 * ptxinfo_cmem - 파서 콜백: 상수 메모리 사용량 누적
 *
 * @nbytes: 이 상수 메모리 뱅크가 사용하는 바이트 수
 * @bank:   상수 메모리 뱅크 번호 (0~15, PTX 스펙)
 *
 * 상수 메모리는 여러 뱅크에 나뉘어 선언될 수 있으므로 += 누적
 */
void ptxinfo_cmem(unsigned nbytes, unsigned bank) { g_ptxinfo.cmem += nbytes; }
// [한국어] 각 상수 메모리 뱅크 크기를 누적 합산

/*
 * [한국어]
 * clear_ptxinfo - g_ptxinfo 구조체와 g_ptxinfo_kname 초기화
 *
 * @return: 없음 (void)
 *
 * 새 커널 파싱 시작 전 또는 파싱 완료 후 호출.
 * g_ptxinfo_kname을 free하여 메모리 누수를 방지한다.
 */
void clear_ptxinfo() {
  free(g_ptxinfo_kname);          // [한국어] strdup으로 할당된 커널 이름 해제
  g_ptxinfo_kname = NULL;         // [한국어] NULL 초기화 (dangling pointer 방지)
  g_ptxinfo.regs = 0;             // [한국어] 레지스터 수 초기화
  g_ptxinfo.lmem = 0;             // [한국어] 로컬 메모리 크기 초기화
  g_ptxinfo.smem = 0;             // [한국어] 공유 메모리 크기 초기화
  g_ptxinfo.cmem = 0;             // [한국어] 상수 메모리 크기 초기화
  g_ptxinfo.gmem = 0;             // [한국어] 전역 메모리 크기 초기화
  g_ptxinfo.ptx_version = 0;      // [한국어] PTX 버전 초기화
  g_ptxinfo.sm_target = 0;        // [한국어] 대상 SM 아키텍처 초기화
}

/*
 * [한국어]
 * ptxinfo_opencl_addinfo - OpenCL 커널의 ptxinfo를 function_info에 등록
 *
 * @kernels: OpenCL 커널 이름 → function_info* 맵 (libopencl/cl.cc에서 구성)
 * @return:  없음 (void). 커널을 찾지 못하면 abort().
 *
 * ptxinfo 파서가 하나의 커널 블록 파싱을 완료한 후 호출된다.
 * g_ptxinfo_kname으로 kernels 맵에서 function_info를 찾아 set_kernel_info()를 호출한다.
 * CUDA 경로의 동등 함수: cuda_sim.cc의 다른 경로 (PTX 파싱 완료 시 직접 설정)
 *
 * 특수 케이스:
 *   g_ptxinfo_kname == NULL: 바이너리 전역 정보 → 출력 후 스킵
 *   "__cuda_dummy_entry__": ptxas가 빈 PTX 파일에 생성하는 더미 → 스킵
 *
 * 호출 체인:
 *   ptxinfo.y 파서 완료 → [ptxinfo_opencl_addinfo] → function_info::set_kernel_info()
 */
void ptxinfo_opencl_addinfo(std::map<std::string, function_info *> &kernels) {
  if (!g_ptxinfo_kname) {
    // [한국어] 커널 이름 없음 → 바이너리 전역 정보 출력 후 초기화
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
    clear_ptxinfo();  // [한국어] 다음 파싱을 위해 초기화
    return;
  }

  if (!strcmp("__cuda_dummy_entry__", g_ptxinfo_kname)) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    // [한국어] ptxas가 빈 PTX 파일에 생성하는 더미 항목 → 무시하고 초기화
    clear_ptxinfo();  // [한국어] 더미 항목 초기화
    return;
  }

  // [한국어] kernels 맵에서 커널 이름으로 function_info 조회
  std::map<std::string, function_info *>::iterator k =
      kernels.find(g_ptxinfo_kname);
  if (k == kernels.end()) {
    // [한국어] 커널 구현을 찾지 못함 → PTX/ptxinfo 불일치 오류
    printf("GPGPU-Sim PTX: ERROR ** implementation for '%s' not found.\n",
           g_ptxinfo_kname);
    abort();  // [한국어] 복구 불가 오류
  } else {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        g_ptxinfo_kname, g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
    function_info *finfo = k->second;  // [한국어] 찾은 커널의 function_info 포인터
    assert(finfo != NULL);              // [한국어] NULL이면 kernels 맵 구성 오류
    finfo->set_kernel_info(g_ptxinfo);  // [한국어] 파싱된 리소스 정보를 커널에 등록
  }
  clear_ptxinfo();  // [한국어] 다음 커널 파싱을 위해 전역 상태 초기화
}

/*
 * [한국어]
 * cuda_sim::find_reconvergence_points - 커널의 재합류 포인트 테이블 반환 (캐시 포함)
 *
 * @finfo: 재합류 포인트를 조회할 function_info 포인터
 * @return: rec_pts 구조체 — {s_kernel_recon_points: 배열 포인터, s_num_recon: 쌍 수}
 *
 * PDOM(Post-Dominator) 분석 결과를 g_rpts 맵에 캐시한다.
 * 처음 호출 시 finfo->get_reconvergence_pairs()로 배열을 채우고 g_rpts에 저장.
 * 이후 호출 시 캐시된 결과를 반환.
 *
 * 재합류 포인트(reconvergence point):
 *   분기 명령어(BRA)와 그 즉시 후지배자(IPDOM)의 쌍.
 *   warp 분기 발산 시 SIMT 스택이 IPDOM PC에서 재합류한다.
 *   source_inst: 분기 명령어, target_inst: 재합류 명령어 (IPDOM)
 *   target_pc == -2이면 RECONVERGE_RETURN_PC (함수 반환 후 재합류)
 *
 * 호출 체인:
 *   get_converge_point() → [find_reconvergence_points] → finfo->get_reconvergence_pairs()
 */
struct rec_pts cuda_sim::find_reconvergence_points(function_info *finfo) {
  rec_pts tmp;   // [한국어] 반환용 재합류 포인트 테이블
  std::map<function_info *, rec_pts>::iterator r = g_rpts.find(finfo);
  // [한국어] g_rpts: function_info → rec_pts 캐시 맵

  if (r == g_rpts.end()) {
    // [한국어] 캐시 미스 → 처음 호출, 재합류 포인트 계산
    int num_recon = finfo->get_num_reconvergence_pairs();
    // [한국어] PDOM 분석에서 발견된 분기/재합류 쌍의 수

    gpgpu_recon_t *kernel_recon_points =
        (struct gpgpu_recon_t *)calloc(num_recon, sizeof(struct gpgpu_recon_t));
    // [한국어] 재합류 포인트 쌍 배열 할당 (calloc으로 0 초기화)
    finfo->get_reconvergence_pairs(kernel_recon_points);
    // [한국어] PDOM 분석 결과에서 분기/재합류 쌍을 배열에 채움

    // [한국어] 재합류 포인트 목록 출력 (디버그)
    printf("GPGPU-Sim PTX: reconvergence points for %s...\n",
           finfo->get_name().c_str());
    for (int i = 0; i < num_recon; i++) {
      printf("GPGPU-Sim PTX: %2u (potential) branch divergence @ ", i + 1);
      kernel_recon_points[i].source_inst->print_insn();  // [한국어] 분기 명령어 정보 출력
      printf("\n");
      printf("GPGPU-Sim PTX:    immediate post dominator      @ ");
      if (kernel_recon_points[i].target_inst)
        kernel_recon_points[i].target_inst->print_insn();  // [한국어] IPDOM 명령어 정보 출력
      printf("\n");
    }
    printf("GPGPU-Sim PTX: ... end of reconvergence points for %s\n",
           finfo->get_name().c_str());

    tmp.s_kernel_recon_points = kernel_recon_points;  // [한국어] 결과 배열 포인터 저장
    tmp.s_num_recon = num_recon;                       // [한국어] 재합류 포인트 수 저장
    g_rpts[finfo] = tmp;                               // [한국어] 캐시에 등록 (이후 호출 시 재사용)
  } else {
    tmp = r->second;  // [한국어] 캐시 히트 → 기존 결과 반환
  }
  return tmp;  // [한국어] rec_pts 구조체 반환 (값 복사)
}

/*
 * [한국어]
 * get_return_pc - void* 스레드 포인터에서 현재 반환 PC를 가져오는 C 인터페이스
 *
 * @thd: ptx_thread_info 포인터 (void*로 전달 — C 링키지 호환)
 * @return: 스레드의 현재 반환 PC (함수 호출 스택 상단의 반환 주소)
 *
 * SIMT 스택의 함수 호출 재합류(call-return) 처리에 사용된다.
 * 분기의 RECONVERGE_RETURN_PC인 경우 타이밍 모델이 이 함수를 통해 실제 반환 PC를 얻는다.
 *
 * 호출 체인:
 *   shader.cc의 SIMT 스택 처리 → [get_return_pc] → ptx_thread_info::get_return_PC()
 */
address_type get_return_pc(void *thd) {
  // function call return
  ptx_thread_info *the_thread = (ptx_thread_info *)thd;  // [한국어] void* → ptx_thread_info* 캐스트
  assert(the_thread != NULL);                             // [한국어] NULL 스레드 접근 방지
  return the_thread->get_return_PC();                     // [한국어] 호출 스택 상단의 반환 PC 반환
}

/*
 * [한국어]
 * cuda_sim::get_converge_point - 분기 명령어 PC에서 재합류 PC를 반환
 *
 * @pc: 분기 명령어(BRA)의 PC
 * @return:
 *   - RECONVERGE_RETURN_PC(-2): 이 분기는 함수 내에 즉시 후지배자가 없음 → 호출 스택 반환 주소에서 재합류
 *   - NO_BRANCH_DIVERGENCE(-1): 이 PC는 분기 발산 지점이 아님 (재합류 불필요)
 *   - 양수 PC: 즉시 후지배자(IPDOM)의 PC (재합류 포인트)
 *
 * 이 함수는 SIMT 스택에서 분기 처리 시 재합류 PC를 결정하는 핵심 함수다.
 * shader.cc::simt_stack::update()에서 BRA 명령어 실행 후 호출된다.
 *
 * 구현:
 *   1. g_pc_to_finfo[pc]로 이 PC가 속한 function_info 찾기
 *   2. find_reconvergence_points()로 이 커널의 재합류 테이블 조회 (캐시)
 *   3. source_pc == pc인 항목 탐색
 *   4. target_pc == -2이면 RECONVERGE_RETURN_PC 반환 (함수 반환 재합류)
 *      그 외이면 target_pc 반환 (IPDOM PC)
 *   5. 테이블에 없으면 NO_BRANCH_DIVERGENCE 반환
 *
 * 호출 체인:
 *   shader.cc::simt_stack::update() → [get_converge_point]
 *     → find_reconvergence_points() → g_rpts 캐시 또는 finfo->get_reconvergence_pairs()
 */
address_type cuda_sim::get_converge_point(address_type pc) {
  // the branch could encode the reconvergence point and/or a bit that indicates
  // the reconvergence point is the return PC on the call stack in the case the
  // branch has no immediate postdominator in the function (i.e., due to
  // multiple return points).

  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  // [한국어] g_pc_to_finfo: PC → 이 명령어가 속한 function_info 맵 (ptx_assemble에서 채워짐)
  assert(f != g_pc_to_finfo.end());  // [한국어] 알 수 없는 PC → 심각한 오류
  function_info *finfo = f->second;  // [한국어] 이 PC의 function_info

  rec_pts tmp = find_reconvergence_points(finfo);
  // [한국어] 이 커널의 재합류 포인트 테이블 (캐시에서 가져오거나 새로 계산)

  // [한국어] 재합류 테이블에서 source_pc == pc 인 항목 탐색
  int i = 0;
  for (; i < tmp.s_num_recon; ++i) {
    if (tmp.s_kernel_recon_points[i].source_pc == pc) {
      // [한국어] 이 PC의 재합류 포인트 항목 발견
      if (tmp.s_kernel_recon_points[i].target_pc == (unsigned)-2) {
        return RECONVERGE_RETURN_PC;
        // [한국어] target_pc == -2: 함수 내 IPDOM 없음 → 호출 스택의 반환 PC에서 재합류
      } else {
        return tmp.s_kernel_recon_points[i].target_pc;
        // [한국어] IPDOM PC 반환 (warp 발산 후 이 PC에서 재합류)
      }
    }
  }
  return NO_BRANCH_DIVERGENCE;
  // [한국어] 테이블에 없음 → 이 분기는 warp 발산이 없거나 무조건 분기
}

/*
 * [한국어]
 * functionalCoreSim::warp_exit - CTA의 모든 스레드 객체를 해제
 *
 * @warp_id: 종료할 warp ID (현재 구현에서는 사용되지 않음 — CTA 전체 해제)
 * @return:  없음 (void)
 *
 * 기능 시뮬레이션에서 하나의 CTA 실행이 완료된 후 호출된다.
 * CTA에 속한 모든 스레드 객체를 해제하고 CTA 정보에 삭제를 등록한다.
 * warp_id 파라미터는 현재 미사용 (전체 CTA 해제 방식으로 구현됨).
 *
 * 실행 컨텍스트: functionalCoreSim 소멸 시 또는 CTA 완료 후 호출 (CPU, 단일 스레드).
 *
 * 호출 체인:
 *   (CTA 완료 후) → [warp_exit] → m_thread[i]->m_cta_info->register_deleted_thread()
 */
void functionalCoreSim::warp_exit(unsigned warp_id) {
  // [한국어] CTA의 모든 스레드를 순회하여 비NULL인 스레드 해제
  for (int i = 0; i < m_warp_count * m_warp_size; i++) {
    if (m_thread[i] != NULL) {
      m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]);
      // [한국어] CTA 정보에 스레드 삭제 등록 (완료된 스레드 카운트 업데이트)
      delete m_thread[i];  // [한국어] 스레드 객체 메모리 해제 (소멸자에서 리소스 정리)
    }
  }
}
