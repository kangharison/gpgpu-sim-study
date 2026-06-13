/*
 * [한국어 설명] PTX 소스/ELF 로딩 및 PTXPlus 변환 인터페이스 (ptx_loader.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 CUDA 바이너리(ELF/CUBIN)에 내장된 PTX 소스를 파싱하고,
 * ptxas를 외부 프로세스로 실행하여 레지스터 사용량 등 자원 정보(ptxinfo)를 추출하며,
 * PTX를 PTXPlus(SASS 수준의 확장 PTX)로 변환하는 인터페이스를 선언한다.
 * 핵심 클래스인 ptxinfo_data는 ptxas와의 렉서/파서(ptxinfo.l, ptxinfo.y)가
 * 파싱 결과를 저장하는 컨텍스트 구조체 역할을 한다.
 * GPGPU-Sim의 execution-driven 시뮬레이션을 가능하게 하는 로딩 파이프라인의 입구이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: libcuda (cuModuleLoad/cuLaunchKernel 인터셉트) → gpgpu_context::
 *   gpgpu_ptx_sim_load_ptx_from_string → PTX 파서(ptx.l/ptx.y) → symbol_table 반환
 *   → gpgpu_ptxinfo_load_from_string → ptxas 실행 → ptxinfo 파싱
 * 이 파일은 CUDA 런타임 인터셉트(libcuda/) 바로 다음 단계에 위치.
 * 실행 컨텍스트: 호스트 유저스페이스 (시뮬레이션 시작 시 커널 로딩 단계).
 *
 * === 타 모듈과의 연결 ===
 * 의존: option_parser.h (커맨드라인 옵션 등록), libcuda/gpgpu_context.h
 * 의존받음: libcuda/cuda_runtime_api.cc (cuModuleLoad에서 ptx_loader 함수 호출)
 * 데이터 흐름: ELF 내 PTX 문자열 → ptxinfo_data 파서 컨텍스트 → gpgpu_context의 심볼 테이블
 * 공유 구조: yyscan_t(Flex 스캐너 핸들), ptxinfo_data(파서 상태)
 *
 * === 주요 함수/구조체 요약 ===
 * ptxinfo_data: ptxas 출력 파싱 컨텍스트 — 스캐너, 라인 버퍼, 옵션 플래그 보관
 * ptxinfo_data::ptxinfo_addinfo(): 파싱된 자원 정보를 gpgpu_context에 등록
 * ptxinfo_data::keep_intermediate_files(): -keep 옵션 반환
 * ptxinfo_data::gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus():
 *   cuobjdump_to_ptxplus 외부 도구를 호출하여 PTX+SASS → PTXPlus 변환
 */
// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#ifndef PTX_LOADER_H_INCLUDED /* [한국어] 헤더 중복 포함 방지 가드 */
#define PTX_LOADER_H_INCLUDED
#include <string> /* [한국어] std::string — PTX/SASS/ELF 파일 경로 문자열 전달 */

#define PTXINFO_LINEBUF_SIZE 1024 /* [한국어] ptxas 출력 파싱용 라인 버퍼 최대 크기(바이트) — ptxas 한 줄 출력은 이 길이를 넘지 않음 */
class gpgpu_context;              /* [한국어] 순환 의존 방지 전방 선언 — gpgpu_context는 ptxinfo_data를 멤버로 보유 */
typedef void* yyscan_t;           /* [한국어] Flex 렉서 스캐너 핸들 타입 — ptxinfo.l 파서가 void* 핸들로 상태 관리 */

/*
 * [한국어]
 * ptxinfo_data - ptxas 출력 파싱 컨텍스트 및 PTXPlus 변환 인터페이스
 *
 * ptxas(NVIDIA PTX 어셈블러)를 외부 프로세스로 실행한 결과(레지스터 수, 스택 크기 등
 * 커널 자원 사용량)를 Flex/Bison 렉서/파서로 읽어 gpgpu_context에 등록하는 역할.
 * gpgpu_context가 이 객체를 멤버로 보유하며(gpgpu_context::ptxinfo),
 * ptxinfo 파싱 함수들이 yyscan_t 스캐너 핸들을 이 구조체를 통해 전달받는다.
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 로딩 단계 (단일 스레드).
 */
class ptxinfo_data {
 public:
  ptxinfo_data(gpgpu_context* ctx) { gpgpu_ctx = ctx; } /* [한국어] 생성자 — gpgpu_ctx 백포인터 초기화 */
  yyscan_t scanner;                    /* [한국어] Flex 렉서 스캐너 상태 핸들.
                                         * 설정자: ptxinfo_lex_init()으로 초기화, ptxinfo_set_in()으로 입력 파일 연결.
                                         * 읽는 자: ptxinfo_parse() — Flex/Bison 파싱 함수에 직접 전달.
                                         * 값 범위: 유효한 Flex 스캐너 포인터 (NULL이면 파싱 불가).
                                         * 동기화: 단일 스레드 로딩 단계에서만 사용. */
  char linebuf[PTXINFO_LINEBUF_SIZE];  /* [한국어] ptxas 출력 한 줄을 임시 저장하는 버퍼.
                                         * 설정자: ptxinfo 렉서(ptxinfo.l)가 토큰 읽기 시 채움.
                                         * 읽는 자: ptxinfo_addinfo() — 파싱된 자원 정보 추출.
                                         * 값 범위: 0~PTXINFO_LINEBUF_SIZE-1 바이트 문자열.
                                         * 동기화: 단일 스레드, 락 불필요. */
  unsigned col;                        /* [한국어] 현재 파싱 중인 열(column) 번호 — 오류 위치 추적용.
                                         * 설정자: ptxinfo 렉서가 문자 처리 시 증가.
                                         * 읽는 자: 오류 보고 시 현재 위치 표시.
                                         * 값 범위: 0 ~ PTXINFO_LINEBUF_SIZE.
                                         * 동기화: 단일 스레드. */
  const char* g_ptxinfo_filename;      /* [한국어] ptxas가 생성한 .ptxinfo 임시 파일 경로.
                                         * 설정자: gpgpu_ptxinfo_load_from_string()에서 임시 파일 생성 후 설정.
                                         * 읽는 자: ptxinfo 파서가 이 파일을 fopen으로 열어 파싱.
                                         * 값 범위: 유효한 파일 경로 문자열 또는 NULL.
                                         * 동기화: 로딩 단계 단일 스레드. */
  class gpgpu_context* gpgpu_ctx;      /* [한국어] 상위 gpgpu_context 역방향 포인터.
                                         * 설정자: 생성자에서 ctx로 초기화.
                                         * 읽는 자: ptxinfo_addinfo() — 파싱 결과를 gpgpu_ctx에 등록.
                                         * 값 범위: 유효한 gpgpu_context 포인터 (NULL 불가).
                                         * 동기화: 생성 후 불변. */
  bool g_keep_intermediate_files;      /* [한국어] 임시 PTX/ptxinfo 파일 유지 여부 (-keep 옵션).
                                         * 설정자: option_parser를 통해 -keep 플래그로 설정.
                                         * 읽는 자: keep_intermediate_files() → 임시 파일 삭제 여부 결정.
                                         * 값 범위: true=유지, false=삭제(기본값).
                                         * 동기화: 옵션 파싱 후 불변. */
  bool m_ptx_save_converted_ptxplus;   /* [한국어] PTXPlus 변환 결과 파일 보존 여부 (-gpgpu_ptx_save_converted_ptxplus).
                                         * 설정자: option_parser를 통해 설정.
                                         * 읽는 자: gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus() — false이면 변환 후 임시 파일 삭제.
                                         * 값 범위: true=보존, false=삭제(기본값).
                                         * 동기화: 옵션 파싱 후 불변. */
  /*
   * [한국어]
   * ptxinfo_addinfo - 파싱된 ptxas 자원 정보를 gpgpu_context에 등록
   *
   * ptxinfo 렉서/파서가 ptxas 출력에서 커널 자원(레지스터 수, lmem, smem 등)을
   * 추출하면 이 함수를 호출하여 gpgpu_context의 kernel_info 구조에 저장.
   * 구현: ptx_loader.cc (ptxinfo_data::ptxinfo_addinfo)
   */
  void ptxinfo_addinfo();
  /*
   * [한국어]
   * keep_intermediate_files - 임시 파일 유지 여부 반환
   *
   * @return: g_keep_intermediate_files 플래그 값 — true이면 임시 파일 유지
   *
   * 구현: ptx_loader.cc
   */
  bool keep_intermediate_files();
  /*
   * [한국어]
   * gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus - PTX + SASS → PTXPlus 변환
   *
   * @ptx_str: PTX 소스 파일 경로
   * @sass_str: SASS 바이너리 파일 경로
   * @elf_str: ELF 바이너리 파일 경로
   * @return: 변환된 PTXPlus 문자열 (heap 할당 — 호출자가 delete[] 해야 함)
   *
   * cuobjdump_to_ptxplus 외부 도구를 system() 호출로 실행하여
   * PTX와 SASS를 합성한 PTXPlus 포맷으로 변환. 임시 파일을 통해 결과 전달.
   * m_ptx_save_converted_ptxplus가 false이면 변환 후 임시 파일 삭제.
   * 구현: ptx_loader.cc
   */
  char* gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(
      const std::string ptx_str, const std::string sass_str,
      const std::string elf_str);
};

#endif
