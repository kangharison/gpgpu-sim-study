/*
 * [한국어 설명] PTX/ptxinfo 파일 로딩 및 PTXPlus 변환 구현 (ptx_loader.cc)
 *
 * === 파일의 역할 ===
 * CUDA 바이너리(ELF/CUBIN)에 내장된 PTX 소스를 파싱하고, ptxas를 외부 프로세스로
 * 호출하여 커널 자원 정보(ptxinfo: 레지스터 수, 스택 크기 등)를 얻으며,
 * PTX를 PTXPlus로 변환하는 전체 로딩 파이프라인을 구현한다.
 * 시뮬레이터 시작 시 CUDA 런타임 인터셉트 직후 호출되며, 이 단계에서 생성된
 * symbol_table이 이후 기능 시뮬레이션(instructions.cc)의 기반이 된다.
 * 임시 파일 생성/삭제, PTX 중복 제거(fix_duplicate_errors) 등 부가 처리도 포함.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: libcuda/cuda_runtime_api.cc (cuModuleLoad) →
 *   gpgpu_context::gpgpu_ptx_sim_load_ptx_from_string (PTX 파싱)
 *   → ptx.l / ptx.y (Flex/Bison 파서) → symbol_table 생성
 *   → gpgpu_context::gpgpu_ptxinfo_load_from_string (ptxas 호출)
 *   → ptxinfo.l / ptxinfo.y (Flex/Bison 파서) → 자원 정보 등록
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 로딩 단계 (단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존: ptx_loader.h, libcuda/gpgpu_context.h, cuda-sim.h, ptx_ir.h, ptx_parser.h
 * 의존받음: libcuda/cuda_runtime_api.cc (cuModuleLoad 인터셉트)
 * 외부 프로세스: $CUDA_INSTALL_PATH/bin/ptxas (ptxinfo 생성),
 *   $GPGPUSIM_ROOT/build/.../cuobjdump_to_ptxplus (PTXPlus 변환)
 * 데이터 흐름: ELF 내 PTX 문자열 → 임시 파일 → ptxas 실행 → ptxinfo 파일 파싱 → gpgpu_context
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpu_context::ptx_reg_options(): -save_embedded_ptx, -keep 등 로더 관련 옵션 등록
 * gpgpu_context::gpgpu_ptx_sim_load_ptx_from_string(): PTX 문자열 → symbol_table 파싱
 * gpgpu_context::gpgpu_ptxinfo_load_from_string(): ptxas 호출 → ptxinfo 파싱 → 자원 등록
 * ptxinfo_data::gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(): cuobjdump_to_ptxplus 호출
 * fix_duplicate_errors(): PTX 중복 정의 제거 (ptxas 오류 코드 65280 처리)
 * get_app_binary_name(): /proc/self/exe로 실행 바이너리 이름 추출
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

#include "ptx_loader.h"              /* [한국어] ptxinfo_data, 로더 인터페이스 선언 */
#include <dirent.h>                  /* [한국어] 디렉토리 탐색 (현재 직접 사용하지 않으나 포함됨) */
#include <unistd.h>                  /* [한국어] readlink(/proc/self/exe), close(fd) 등 POSIX 시스템 호출 */
#include <fstream>                   /* [한국어] std::ifstream — PTXPlus 변환 결과 파일 읽기 */
#include <sstream>                   /* [한국어] std::stringstream — /proc/self/exe 경로 구성 */
#include "../../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context — ptxinfo/ptx_parser 멤버 접근 */
#include "cuda-sim.h"                /* [한국어] init_parser() 선언 — PTX 파서 초기화 및 symbol_table 생성 */
#include "ptx_ir.h"                  /* [한국어] symbol_table, ptx_instruction 등 PTX IR 자료구조 */
#include "ptx_parser.h"              /* [한국어] ptx_recognizer — PTX Bison 파서 컨텍스트 */

/// extern prototypes
/* [한국어] PTX 렉서/파서 외부 심볼 선언 — ptx.l과 ptx.y에서 생성된 Flex/Bison 함수들 */

extern int ptx_error(yyscan_t yyscanner, ptx_recognizer *recognizer,
                     const char *s); /* [한국어] PTX 파싱 오류 처리 함수 (ptx.y 생성) */
extern int ptx_lex_init(yyscan_t *scanner); /* [한국어] PTX Flex 스캐너 초기화 */
extern void ptx_set_in(FILE *_in_str, yyscan_t yyscanner); /* [한국어] PTX 스캐너 입력 파일 설정 */
extern int ptx_parse(yyscan_t scanner, ptx_recognizer *recognizer); /* [한국어] PTX Bison 파서 실행 */
extern int ptx_lex_destroy(yyscan_t scanner); /* [한국어] PTX Flex 스캐너 자원 해제 */
extern int ptx__scan_string(const char *, yyscan_t scanner); /* [한국어] 문자열에서 직접 PTX 파싱 */

extern std::map<unsigned, const char *> get_duplicate(); /* [한국어] PTX 중복 정의된 함수/변수 목록 반환 (ptx.y에서 관리) */

typedef void *yyscan_t;              /* [한국어] ptxinfo 렉서용 스캐너 타입 재선언 (ptx_loader.h에도 동일하게 있음) */
extern int ptxinfo_lex_init(yyscan_t *scanner); /* [한국어] ptxinfo Flex 스캐너 초기화 */
extern void ptxinfo_set_in(FILE *_in_str, yyscan_t yyscanner); /* [한국어] ptxinfo 스캐너 입력 파일 설정 */
extern int ptxinfo_parse(yyscan_t scanner, ptxinfo_data *ptxinfo); /* [한국어] ptxinfo Bison 파서 실행 */
extern int ptxinfo_lex_destroy(yyscan_t scanner); /* [한국어] ptxinfo Flex 스캐너 자원 해제 */

static bool g_save_embedded_ptx;     /* [한국어] -save_embedded_ptx 옵션 — true이면 ELF에서 추출한 PTX를 파일로 저장
                                       * 설정자: ptx_reg_options()에서 option_parser를 통해 설정.
                                       * 읽는 자: gpgpu_ptx_sim_load_ptx_from_string, gpgpu_ptxinfo_load_from_string.
                                       * 기본값: false (0). */
static int g_occupancy_sm_number;    /* [한국어] ptxas에 전달할 SM 버전 번호 (-gpgpu_occupancy_sm_number).
                                       * 레지스터 사용량 계산 시 SM 아키텍처 특성이 달라 정확한 버전 지정 필요.
                                       * 설정자: ptx_reg_options()에서 option_parser를 통해 설정.
                                       * 읽는 자: gpgpu_ptxinfo_load_from_string — --gpu-name=sm_XX 플래그 생성.
                                       * 기본값: 0 (설정하지 않으면 오류로 종료). */

/*
 * [한국어]
 * ptxinfo_data::keep_intermediate_files - 임시 파일 유지 여부 반환
 *
 * @return: g_keep_intermediate_files 플래그 — true이면 임시 PTX/ptxinfo 파일 유지
 *
 * -keep 옵션 활성화 시 임시 파일을 삭제하지 않아 디버그에 유용.
 * ptx_loader.cc 내 임시 파일 삭제 코드에서 이 값을 확인.
 * 실행 컨텍스트: 커널 로딩 단계.
 *
 * 호출 체인:
 *   gpgpu_ptxinfo_load_from_string 내부 → [이 함수]
 */
bool ptxinfo_data::keep_intermediate_files() {
  return g_keep_intermediate_files; /* [한국어] option_parser가 설정한 -keep 플래그 반환 */
}

/*
 * [한국어]
 * gpgpu_context::ptx_reg_options - PTX 로더 관련 커맨드라인/설정 옵션 등록
 *
 * @opp: option_parser 핸들 — gpgpusim.config 파싱 컨텍스트
 *
 * PTX 로딩/변환에 영향을 미치는 네 가지 옵션을 option_parser에 등록:
 *   -save_embedded_ptx: ELF에서 추출한 PTX를 _<n>.ptx 파일로 저장
 *   -keep: 임시 파일(ptxas 출력 등) 유지
 *   -gpgpu_ptx_save_converted_ptxplus: PTXPlus 변환 결과 파일 보존
 *   -gpgpu_occupancy_sm_number: ptxas 호출 시 사용할 SM 아키텍처 번호
 * 실행 컨텍스트: 시뮬레이터 옵션 파싱 단계 (main 진입 직후).
 *
 * 호출 체인:
 *   gpgpusim_entrypoint → gpgpu_context::ptx_reg_options → [이 함수]
 */
void gpgpu_context::ptx_reg_options(option_parser_t opp) {
  option_parser_register(opp, "-save_embedded_ptx", OPT_BOOL,
                         &g_save_embedded_ptx,
                         "saves ptx files embedded in binary as <n>.ptx", "0"); /* [한국어] ELF 내 PTX 저장 옵션 등록 — 기본 비활성 */
  option_parser_register(opp, "-keep", OPT_BOOL,
                         &(ptxinfo->g_keep_intermediate_files),
                         "keep intermediate files created by GPGPU-Sim when "
                         "interfacing with external programs",
                         "0"); /* [한국어] 임시 파일 유지 옵션 등록 — ptxinfo->g_keep_intermediate_files에 바인딩 */
  option_parser_register(opp, "-gpgpu_ptx_save_converted_ptxplus", OPT_BOOL,
                         &(ptxinfo->m_ptx_save_converted_ptxplus),
                         "Saved converted ptxplus to a file", "0"); /* [한국어] PTXPlus 변환 결과 보존 옵션 등록 */
  option_parser_register(opp, "-gpgpu_occupancy_sm_number", OPT_INT32,
                         &g_occupancy_sm_number,
                         "The SM number to pass to ptxas when getting register "
                         "usage for computing GPU occupancy. "
                         "This parameter is required in the config.",
                         "0"); /* [한국어] SM 버전 번호 옵션 등록 — 0이면 로딩 시 오류로 종료 */
}

/*
 * [한국어]
 * gpgpu_context::print_ptx_file - PTX 소스를 줄 번호와 PC와 함께 출력 (디버그용)
 *
 * @p: 출력할 PTX 소스 문자열
 * @source_num: PTX 소스 번호 (여러 PTX 파일 중 몇 번째인지)
 * @filename: PTX 파일명 (심볼 테이블 조회 키)
 *
 * g_debug_execution >= 100 일 때 gpgpu_ptx_sim_load_ptx_from_string에서 호출.
 * PTX 소스를 줄 단위로 분리하고, 각 줄에 대응하는 ptx_instruction의 PC 값을
 * 심볼 테이블에서 조회하여 함께 출력한다.
 * strdup으로 소스 복사 후 '\n' 위치를 '\0'으로 교체하는 방식으로 줄 분리.
 * 실행 컨텍스트: 커널 로딩 단계 (디버그 모드).
 *
 * 호출 체인:
 *   gpgpu_ptx_sim_load_ptx_from_string → [이 함수]
 */
void gpgpu_context::print_ptx_file(const char *p, unsigned source_num,
                                   const char *filename) {
  printf("\nGPGPU-Sim PTX: file _%u.ptx contents:\n\n", source_num); /* [한국어] PTX 파일 내용 출력 시작 헤더 */
  char *s = strdup(p);  /* [한국어] 원본 PTX 문자열 복사 — '\n' 교체로 원본 수정을 피하기 위함 */
  char *t = s;          /* [한국어] 현재 줄 시작 포인터 */
  unsigned n = 1;       /* [한국어] 현재 줄 번호 (1부터 시작) */
  while (*t != '\0') {  /* [한국어] 문자열 끝까지 줄 단위로 반복 */
    char *u = t;        /* [한국어] 현재 줄 끝 탐색용 포인터 */
    while ((*u != '\n') && (*u != '\0')) u++; /* [한국어] 줄 끝('\n' 또는 '\0') 위치 탐색 */
    unsigned last = (*u == '\0'); /* [한국어] 마지막 줄 여부 — '\0'이면 마지막 줄 */
    *u = '\0';          /* [한국어] '\n'을 '\0'으로 교체하여 현재 줄을 null-종료 문자열로 분리 */
    const ptx_instruction *pI = ptx_parser->ptx_instruction_lookup(filename, n); /* [한국어] 해당 파일/줄에 대응하는 PTX 명령어 객체 조회 */
    char pc[64];        /* [한국어] PC 값을 문자열로 저장하는 버퍼 */
    if (pI && pI->get_PC())
      snprintf(pc, 64, "%4llu", pI->get_PC()); /* [한국어] 유효한 PC이면 4자리 10진수로 포맷 */
    else
      snprintf(pc, 64, "    "); /* [한국어] PC 없으면 공백 4칸 — 정렬 유지 */
    printf("    _%u.ptx  %4u (pc=%s):  %s\n", source_num, n, pc, t); /* [한국어] "<파일번호>.ptx  <줄번호> (pc=<PC>): <줄내용>" 형식 출력 */
    if (last) break;    /* [한국어] 마지막 줄이면 반복 종료 */
    t = u + 1;          /* [한국어] 다음 줄 시작으로 포인터 이동 (이전 '\0' 위치 다음) */
    n++;                /* [한국어] 줄 번호 증가 */
  }
  free(s);              /* [한국어] strdup으로 할당한 복사본 해제 */
  fflush(stdout);       /* [한국어] 출력 버퍼 플러시 — 시뮬레이터 크래시 전 출력 보장 */
}

/*
 * [한국어]
 * ptxinfo_data::gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus - PTX+SASS → PTXPlus 변환
 *
 * @ptxfilename: PTX 소스 임시 파일 경로
 * @elffilename: ELF 바이너리 임시 파일 경로
 * @sassfilename: SASS(실제 GPU 어셈블리) 임시 파일 경로
 * @return: 변환된 PTXPlus 문자열 포인터 (new[]로 할당 — 호출자가 delete[] 해야 함)
 *
 * cuobjdump_to_ptxplus 외부 도구(GPGPU-Sim 빌드 산출물)를 system() 호출로 실행하여
 * PTX 가상 ISA와 SASS 실제 ISA를 결합한 PTXPlus 포맷으로 변환한다.
 * PTXPlus는 PTX보다 하드웨어에 가까운 레지스터 할당 정보를 포함하여
 * 더 정확한 타이밍 시뮬레이션을 가능하게 한다.
 * 변환 과정: 임시 파일 생성(mkstemp) → cuobjdump_to_ptxplus 실행 → 결과 파일 읽기
 *   → (옵션에 따라) 임시 파일 삭제.
 * 실행 컨텍스트: 커널 로딩 단계, 호스트 유저스페이스.
 *
 * 호출 체인:
 *   libcuda (PTXPlus 모드 활성화 시) → [이 함수] → system(cuobjdump_to_ptxplus)
 */
char *ptxinfo_data::gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(
    const std::string ptxfilename, const std::string elffilename,
    const std::string sassfilename) {
  printf("GPGPU-Sim PTX: converting EMBEDDED .ptx file to ptxplus \n"); /* [한국어] PTXPlus 변환 시작 알림 */

  char fname_ptxplus[1024];
  snprintf(fname_ptxplus, 1024, "_ptxplus_XXXXXX"); /* [한국어] mkstemp용 임시 파일명 템플릿 — XXXXXX는 고유 문자열로 교체됨 */
  int fd4 = mkstemp(fname_ptxplus);                 /* [한국어] 고유한 임시 파일 생성 (레이스 컨디션 방지) */
  close(fd4);                                        /* [한국어] cuobjdump_to_ptxplus가 파일에 쓸 수 있도록 파일 디스크립터 닫기 */

  // Run cuobjdump_to_ptxplus
  char commandline[1024];
  int result;
  snprintf(commandline, 1024,
           "$GPGPUSIM_ROOT/build/$GPGPUSIM_CONFIG/cuobjdump_to_ptxplus/"
           "cuobjdump_to_ptxplus %s %s %s %s",
           ptxfilename.c_str(), sassfilename.c_str(), elffilename.c_str(),
           fname_ptxplus); /* [한국어] cuobjdump_to_ptxplus 실행 커맨드라인 구성: ptx sass elf → 출력파일 */
  fflush(stdout);          /* [한국어] system() 호출 전 버퍼 플러시 — 자식 프로세스와 출력 순서 보장 */
  printf("GPGPU-Sim PTX: calling cuobjdump_to_ptxplus\ncommandline: %s\n",
         commandline);     /* [한국어] 실행할 커맨드라인 출력 — 디버그용 */
  result = system(commandline); /* [한국어] cuobjdump_to_ptxplus 외부 프로세스 실행 — 반환값 0이면 성공 */
  if (result) {            /* [한국어] 비정상 종료 — 환경 변수 미설정이나 도구 미존재 가능성 */
    fprintf(stderr, "GPGPU-Sim PTX: ERROR ** could not execute %s\n",
            commandline);  /* [한국어] 실행 실패 오류 메시지 */
    exit(1);               /* [한국어] 복구 불가 오류로 시뮬레이터 종료 */
  }

  // Get ptxplus from file
  /* [한국어] cuobjdump_to_ptxplus가 생성한 임시 파일에서 PTXPlus 문자열 읽기 */
  std::ifstream fileStream(fname_ptxplus, std::ios::in); /* [한국어] 변환 결과 임시 파일 열기 */
  std::string text, line;
  while (getline(fileStream, line)) { /* [한국어] 한 줄씩 읽어 text에 누적 */
    text += (line + "\n");            /* [한국어] 줄바꿈을 다시 추가하며 전체 텍스트 구성 */
  }
  fileStream.close(); /* [한국어] 파일 스트림 닫기 */

  char *ptxplus_str = new char[strlen(text.c_str()) + 1]; /* [한국어] PTXPlus 문자열을 저장할 힙 버퍼 할당 (+1: null 종료) */
  strcpy(ptxplus_str, text.c_str()); /* [한국어] std::string에서 C 문자열로 복사 */

  if (!m_ptx_save_converted_ptxplus) { /* [한국어] 변환 결과 보존 옵션이 꺼진 경우 임시 파일 삭제 */
    char rm_commandline[1024];

    snprintf(rm_commandline, 1024, "rm -f %s", fname_ptxplus); /* [한국어] 임시 PTXPlus 파일 삭제 커맨드 구성 */

    printf("GPGPU-Sim PTX: removing temporary files using \"%s\"\n",
           rm_commandline); /* [한국어] 삭제 커맨드 출력 */
    int rm_result = system(rm_commandline); /* [한국어] 임시 파일 삭제 실행 */
    if (rm_result != 0) {   /* [한국어] 삭제 실패 — 파일시스템 권한 문제 등 */
      fprintf(stderr,
              "GPGPU-Sim PTX: ERROR ** while removing temporary files %d\n",
              rm_result);   /* [한국어] 오류 코드와 함께 실패 메시지 출력 */
      exit(1);              /* [한국어] 복구 불가 오류로 종료 */
    }
  }
  printf("GPGPU-Sim PTX: DONE converting EMBEDDED .ptx file to ptxplus \n"); /* [한국어] PTXPlus 변환 완료 알림 */

  return ptxplus_str; /* [한국어] 변환된 PTXPlus 문자열 반환 — 호출자가 delete[] 책임 */
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_ptx_sim_load_ptx_from_string - PTX 문자열 파싱 → symbol_table 생성
 *
 * @p: ELF에서 추출한 PTX 소스 문자열
 * @source_num: PTX 소스 번호 (여러 embedded PTX 중 식별자)
 * @return: 파싱 완료된 symbol_table 포인터 — 커널 함수와 전역 변수가 등록된 심볼 테이블
 *
 * PTX Flex/Bison 파서를 통해 PTX 소스 문자열을 파싱하고 심볼 테이블을 생성한다.
 * 파싱 전 g_save_embedded_ptx가 설정된 경우 PTX를 _<n>.ptx 파일로 저장.
 * 파싱 오류 시 PTX를 임시 파일로 추출 후 abort() — 복구 불가 오류로 간주.
 * 성공 시 g_debug_execution >= 100이면 줄 번호+PC 형태로 PTX 덤프.
 * 실행 컨텍스트: 커널 로딩 단계, 호스트 유저스페이스 (단일 스레드).
 *
 * 호출 체인:
 *   libcuda (cuModuleLoad 인터셉트) → [이 함수] → init_parser → ptx_parse → symbol_table
 */
symbol_table *gpgpu_context::gpgpu_ptx_sim_load_ptx_from_string(
    const char *p, unsigned source_num) {
  char buf[1024];
  snprintf(buf, 1024, "_%u.ptx", source_num); /* [한국어] "<source_num>.ptx" 형식의 가상 파일명 생성 — 심볼 테이블 조회 키로 사용 */
  if (g_save_embedded_ptx) {                  /* [한국어] -save_embedded_ptx 옵션 활성화 시 PTX 파일로 저장 */
    FILE *fp = fopen(buf, "w");               /* [한국어] 저장할 .ptx 파일 생성 */
    fprintf(fp, "%s", p);                     /* [한국어] PTX 소스 전체를 파일에 기록 */
    fclose(fp);                               /* [한국어] 파일 닫기 */
  }
  symbol_table *symtab = init_parser(buf); /* [한국어] 파서 초기화 및 빈 심볼 테이블 생성 — buf는 소스 파일 식별자 */
  ptx_lex_init(&(ptx_parser->scanner));    /* [한국어] PTX Flex 스캐너 초기화 — 내부 상태 할당 */
  ptx__scan_string(p, ptx_parser->scanner); /* [한국어] PTX 문자열을 스캐너 입력으로 설정 (파일 대신 메모리 버퍼 스캔) */
  int errors = ptx_parse(ptx_parser->scanner, ptx_parser); /* [한국어] PTX Bison 파서 실행 — 오류 수 반환 */
  if (errors) { /* [한국어] 파싱 오류 발생 — PTX 문법 오류 또는 지원하지 않는 구문 */
    char fname[1024];
    snprintf(fname, 1024, "_ptx_errors_XXXXXX"); /* [한국어] 오류 원인 분석용 임시 파일명 템플릿 */
    int fd = mkstemp(fname);                     /* [한국어] 오류 PTX를 저장할 임시 파일 생성 */
    close(fd);                                   /* [한국어] 파일 디스크립터 닫기 (fopen으로 재열기 예정) */
    printf(
        "GPGPU-Sim PTX: parser error detected, exiting... but first extracting "
        ".ptx to \"%s\"\n",
        fname); /* [한국어] 오류 파일 경로 출력 — 사용자가 PTX를 직접 확인 가능 */
    FILE *ptxfile = fopen(fname, "w"); /* [한국어] 오류 PTX 저장 파일 열기 */
    fprintf(ptxfile, "%s", p);         /* [한국어] 오류가 있는 PTX 전체를 파일에 기록 */
    fclose(ptxfile);                   /* [한국어] 파일 닫기 */
    abort();                           /* [한국어] 코어 덤프와 함께 즉시 종료 — 스택 트레이스 생성 */
    exit(40);                          /* [한국어] abort() 이후 도달 불가 — 컴파일러 경고 방지용 */
  }
  ptx_lex_destroy(ptx_parser->scanner); /* [한국어] PTX Flex 스캐너 자원 해제 */

  if (g_debug_execution >= 100) print_ptx_file(p, source_num, buf); /* [한국어] 최고 디버그 레벨에서 파싱된 PTX를 줄 번호+PC와 함께 출력 */

  printf("GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file %s\n", buf); /* [한국어] 파싱 완료 알림 */
  return symtab; /* [한국어] 파싱으로 생성된 심볼 테이블 반환 — 커널 함수 포인터 조회에 사용 */
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_ptx_sim_load_ptx_from_filename - 파일에서 PTX 파싱
 *
 * @filename: 파싱할 PTX 파일 경로
 * @return: 파싱 완료된 symbol_table 포인터
 *
 * 파일에서 직접 PTX를 파싱하는 단순 버전 — init_parser가 파일을 열어 파싱.
 * 문자열 버전(gpgpu_ptx_sim_load_ptx_from_string)과 달리 파일 경로를 직접 받음.
 * 실행 컨텍스트: 커널 로딩 단계.
 *
 * 호출 체인:
 *   libcuda (파일 기반 PTX 로딩 경로) → [이 함수] → init_parser
 */
symbol_table *gpgpu_context::gpgpu_ptx_sim_load_ptx_from_filename(
    const char *filename) {
  symbol_table *symtab = init_parser(filename); /* [한국어] 파서 초기화 및 파일에서 PTX 파싱 */
  printf("GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file %s\n", filename); /* [한국어] 파싱 완료 알림 */
  return symtab; /* [한국어] 생성된 심볼 테이블 반환 */
}

/*
 * [한국어]
 * fix_duplicate_errors - PTX 파일에서 중복 정의된 함수/변수 제거
 *
 * @fname2: 중복 오류가 있는 PTX 파일 경로 (입출력 — 함수 완료 후 중복 제거된 새 파일로 교체)
 *
 * ptxas가 반환 코드 65280(중복 정의 오류)으로 실패했을 때 호출된다.
 * get_duplicate()에서 중복 항목(함수 또는 변수) 목록을 얻어, PTX 파일에서
 * 해당 정의를 삭제하고 새 파일을 생성한다.
 * 처리 과정: 원본 파일을 _temp_ptx로 이름 변경 → 전체 내용을 메모리로 로드
 *   → 중복 항목 위치를 찾아 해당 부분을 건너뜀 → 새 파일 생성 → 임시 파일 삭제.
 * 함수/변수 타입별 처리:
 *   "function": .func 키워드 이전부터 함수 본체(중괄호 쌍) 끝까지 삭제
 *   "variable": 해당 줄 삭제
 * 실행 컨텍스트: 커널 로딩 단계, ptxas 오류 복구 경로.
 *
 * 호출 체인:
 *   gpgpu_ptxinfo_load_from_string (ptxas result == 65280) → [이 함수]
 */
void fix_duplicate_errors(char fname2[1024]) {
  char tempfile[1024] = "_temp_ptx"; /* [한국어] 원본 파일을 임시로 이름 변경할 고정 임시 파일명 */
  char commandline[1024];

  // change the name of the ptx file to _temp_ptx
  /* [한국어] 원본 PTX 파일을 임시 이름으로 변경 — 이후 fname2에 새 파일 생성을 위함 */
  snprintf(commandline, 1024, "mv %s %s", fname2, tempfile); /* [한국어] mv <fname2> _temp_ptx 커맨드 구성 */
  printf("Running: %s\n", commandline);                       /* [한국어] 실행 커맨드 출력 */
  int result = system(commandline);                           /* [한국어] 파일 이름 변경 실행 */
  if (result != 0) {                                          /* [한국어] 이름 변경 실패 — 권한 문제 등 */
    fprintf(stderr,
            "GPGPU-Sim PTX: ERROR ** while changing filename from %s to %s",
            fname2, tempfile); /* [한국어] 오류 메시지 출력 */
    exit(1);                   /* [한국어] 복구 불가 오류로 종료 */
  }

  // store all of the ptx into a char array
  /* [한국어] 임시 파일 전체 내용을 메모리 버퍼로 로드 */
  FILE *ptxsource = fopen(tempfile, "r");           /* [한국어] 임시 파일 읽기 모드로 열기 */
  fseek(ptxsource, 0, SEEK_END);                    /* [한국어] 파일 끝으로 이동 — 크기 측정용 */
  long filesize = ftell(ptxsource);                 /* [한국어] 현재 위치 = 파일 크기(바이트) */
  rewind(ptxsource);                                /* [한국어] 파일 포인터를 처음으로 되돌림 */
  char *ptxdata = (char *)malloc((filesize + 1) * sizeof(char)); /* [한국어] 파일 전체를 담을 버퍼 할당 (+1: null 종료) */
  // Fail if we do not read the file
  assert(fread(ptxdata, filesize, 1, ptxsource) == 1); /* [한국어] 파일 전체를 한 번에 읽기 — 실패 시 assert abort */
  fclose(ptxsource); /* [한국어] 소스 파일 닫기 */

  FILE *ptxdest = fopen(fname2, "w"); /* [한국어] 중복 제거된 새 PTX 파일 생성 (원본과 같은 이름 fname2) */
  std::map<unsigned, const char *> duplicate = get_duplicate(); /* [한국어] PTX 파서가 감지한 중복 항목(줄번호 → "function"/"variable") 맵 가져오기 */
  unsigned offset;                    /* [한국어] 다목적 오프셋 변수 — 줄 오프셋, 중괄호 탐색 등에 사용 */
  unsigned oldlinenum = 1;            /* [한국어] 이전 중복 항목의 줄 번호 — 다음 항목까지 전진하기 위한 기준 */
  unsigned linenum;                   /* [한국어] 현재 처리 중인 중복 항목의 줄 번호 */
  char *startptr = ptxdata;           /* [한국어] 다음 fwrite 시작 위치 — 이 위치부터 현재 중복 항목 직전까지 복사 */
  char *funcptr = NULL;               /* [한국어] 가장 최근 .func 키워드 위치 — 함수 시작점 탐색용 */
  char *tempptr = ptxdata - 1;        /* [한국어] .func 탐색용 임시 포인터 (ptxdata-1로 초기화하여 첫 strstr이 ptxdata부터 시작) */
  char *lineptr = ptxdata - 1;        /* [한국어] 현재 탐색 중인 줄 끝('\n') 위치 (ptxdata-1로 초기화 — 1번째 줄부터 정상 탐색) */

  // recreate the ptx file without duplications
  /* [한국어] 중복 항목을 순서대로 처리하며 PTX 파일 재구성 */
  for (std::map<unsigned, const char *>::iterator iter = duplicate.begin();
       iter != duplicate.end(); iter++) { /* [한국어] 줄 번호 오름차순으로 중복 항목 순회 */
    // find the line of the next error
    /* [한국어] 이전 처리 위치(oldlinenum)에서 현재 중복 항목 줄(linenum)까지 '\n' 찾기 */
    linenum = iter->first; /* [한국어] 현재 중복 항목의 줄 번호 */
    for (int i = oldlinenum; i < linenum; i++) { /* [한국어] oldlinenum에서 linenum-1까지 '\n' 위치 전진 */
      lineptr = strchr(lineptr + 1, '\n');        /* [한국어] 다음 줄바꿈 위치로 포인터 이동 */
    }

    // find the end of the current section to be copied over
    // then find the start of the next section that will be copied
    /* [한국어] 중복 항목 타입별로 삭제 범위 결정 및 복사 */
    if (strcmp("function", iter->second) == 0) { /* [한국어] 함수 중복 — .func 선언부터 본체 끝까지 삭제 */
      // get location of most recent .func
      /* [한국어] lineptr 이전의 마지막 .func 키워드 위치 탐색 */
      while (tempptr < lineptr && tempptr != NULL) { /* [한국어] lineptr 이전까지 .func 위치 계속 전진 */
        funcptr = tempptr;                           /* [한국어] 현재까지 가장 최근 .func 위치 저장 */
        tempptr = strstr(funcptr + 1, ".func");      /* [한국어] 다음 .func 위치 탐색 */
      }

      // get the start of the previous line
      /* [한국어] funcptr이 가리키는 .func가 시작되는 줄의 앞 줄바꿈 위치 찾기 */
      offset = 0;
      while (*(funcptr - offset) != '\n') offset++; /* [한국어] funcptr 앞으로 '\n' 위치 탐색 */

      fwrite(startptr, sizeof(char), funcptr - offset + 1 - startptr, ptxdest); /* [한국어] startptr부터 .func 직전 줄 끝까지 출력 파일에 기록 */

      // find next location of startptr
      /* [한국어] 중복 함수 본체 끝 이후를 새 startptr로 설정 */
      if (*(lineptr + 3) == ';') {
        // for function definitions
        /* [한국어] 함수 선언(본체 없음, 세미콜론으로 끝): "..." + ";\n" → +5 위치 */
        startptr = lineptr + 5; /* [한국어] 세미콜론 라인 다음 위치로 startptr 이동 */
      } else if (*(lineptr + 3) == '{') {
        // for functions enclosed with curly brackets
        /* [한국어] 함수 정의(중괄호 본체): 중괄호 쌍을 세어 본체 끝 위치 탐색 */
        offset = 5;
        unsigned bracket = 1; /* [한국어] 중괄호 깊이 카운터 — 1은 여는 중괄호 '{' */
        while (bracket != 0) { /* [한국어] 열린 중괄호가 모두 닫힐 때까지 반복 */
          if (*(lineptr + offset) == '{')
            bracket++;         /* [한국어] 중첩 중괄호 열기 */
          else if (*(lineptr + offset) == '}')
            bracket--;         /* [한국어] 중괄호 닫기 */
          offset++;            /* [한국어] 다음 문자로 전진 */
        }
        startptr = lineptr + offset + 1; /* [한국어] 닫힌 중괄호 다음 위치로 startptr 이동 */
      } else {
        printf("GPGPU-Sim PTX: ERROR ** Unrecognized function format\n"); /* [한국어] 알 수 없는 함수 형식 오류 */
        abort(); /* [한국어] 복구 불가 오류로 즉시 종료 */
      }
    } else if (strcmp("variable", iter->second) == 0) { /* [한국어] 변수 중복 — 해당 줄 전체 삭제 */
      fwrite(startptr, sizeof(char), (int)(lineptr + 1 - startptr), ptxdest); /* [한국어] startptr부터 중복 변수 줄 끝('\n')까지 기록 */

      // find next location of startptr
      /* [한국어] 중복 변수 다음 줄로 startptr 이동 */
      offset = 1;
      while (*(lineptr + offset) != '\n') offset++; /* [한국어] 같은 줄의 끝('\n') 탐색 (lineptr은 이 줄 앞의 '\n') */
      startptr = lineptr + offset + 1; /* [한국어] 중복 줄 다음 줄 시작으로 startptr 설정 */
    } else {
      printf("GPGPU-Sim PTX: ERROR ** Unsupported duplicate type: %s\n",
             iter->second); /* [한국어] "function"/"variable" 이외의 알 수 없는 중복 타입 */
    }

    oldlinenum = linenum; /* [한국어] 다음 반복에서 이전 위치 기준으로 줄 전진하기 위해 현재 줄 번호 저장 */
  }
  // copy over the rest of the file
  /* [한국어] 마지막 중복 항목 이후 남은 PTX 내용을 출력 파일에 기록 */
  fwrite(startptr, sizeof(char), ptxdata + filesize - startptr, ptxdest); /* [한국어] startptr부터 파일 끝까지 전부 복사 */

  // cleanup
  free(ptxdata);     /* [한국어] malloc으로 할당한 PTX 데이터 버퍼 해제 */
  fclose(ptxdest);   /* [한국어] 새 PTX 파일 닫기 */
  snprintf(commandline, 1024, "rm -f %s", tempfile); /* [한국어] 임시 파일 삭제 커맨드 구성 */
  printf("Running: %s\n", commandline);               /* [한국어] 삭제 커맨드 출력 */
  result = system(commandline);                       /* [한국어] 임시 파일(_temp_ptx) 삭제 실행 */
  if (result != 0) {                                  /* [한국어] 삭제 실패 */
    fprintf(stderr, "GPGPU-Sim PTX: ERROR ** while deleting %s", tempfile); /* [한국어] 오류 메시지 */
    exit(1);                                          /* [한국어] 복구 불가 오류로 종료 */
  }
}

// we need the application name here too.
/*
 * [한국어]
 * get_app_binary_name - 현재 실행 중인 바이너리의 이름(확장자 제외) 반환
 *
 * @return: 실행 바이너리 파일명 포인터 (strtok으로 exe_path 내부를 가리킴 — 함수 반환 후 유효하지 않음 주의)
 *
 * Linux에서는 /proc/self/exe 심볼릭 링크를 readlink()로 읽어 실행 파일 경로를 얻고,
 * '/'로 분리하여 마지막 컴포넌트(파일명)만 추출, '.'으로 분리하여 확장자를 제거한다.
 * 이 이름은 gpgpu_ptxinfo_load_from_string에서 PTX 파일명 패턴
 * (<name>.<index>.sm_<ver>.ptx)을 구성하는 데 사용된다.
 * macOS는 지원하지 않으며 abort()로 즉시 종료.
 * 주의: strtok은 정적 버퍼(exe_path)를 수정하며 반환 포인터의 수명이 짧음.
 * 실행 컨텍스트: 커널 로딩 단계.
 *
 * 호출 체인:
 *   gpgpu_ptxinfo_load_from_string → [이 함수]
 */
char *get_app_binary_name() {
  char exe_path[1025];        /* [한국어] readlink가 채울 실행 파일 전체 경로 버퍼 */
  char *self_exe_path = NULL; /* [한국어] 최종 반환할 파일명 포인터 — strtok 결과 저장 */
#ifdef __APPLE__
  // AMRUTH:  get apple device and check the result.
  /* [한국어] macOS 지원 미구현 — readlink 대신 proc_pidpath 등 macOS API 필요 */
  printf("WARNING: not tested for Apple-mac devices \n");
  abort(); /* [한국어] macOS에서는 즉시 종료 */
#else
  std::stringstream exec_link;
  exec_link << "/proc/self/exe";          /* [한국어] Linux /proc/self/exe: 현재 프로세스의 실행 파일 심볼릭 링크 */
  ssize_t path_length = readlink(exec_link.str().c_str(), exe_path, 1024); /* [한국어] 심볼릭 링크가 가리키는 실제 경로 읽기 */
  assert(path_length != -1);              /* [한국어] readlink 실패(링크 없음 등) 시 abort — Linux 환경에서 항상 성공해야 함 */
  exe_path[path_length] = '\0';           /* [한국어] readlink는 null 종료를 보장하지 않으므로 수동으로 추가 */

  char *token = strtok(exe_path, "/");    /* [한국어] 경로를 '/'로 분리하여 첫 컴포넌트 반환 */
  while (token != NULL) {                 /* [한국어] 마지막 컴포넌트(파일명)까지 순회 */
    self_exe_path = token;                /* [한국어] 매 반복마다 최신 컴포넌트 저장 — 루프 끝에 파일명이 남음 */
    token = strtok(NULL, "/");            /* [한국어] 다음 '/' 분리 컴포넌트 탐색 */
  }
#endif
  self_exe_path = strtok(self_exe_path, "."); /* [한국어] 파일명에서 '.' 이전 부분(확장자 제거) 추출 */
  printf("self exe links to: %s\n", self_exe_path); /* [한국어] 추출된 바이너리 이름 출력 — 확인용 */
  return self_exe_path; /* [한국어] 확장자 제거된 바이너리 이름 반환 (exe_path 스택 버퍼 내부 포인터 — 사용 후 즉시 사용해야 함) */
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_ptx_info_load_from_filename - 파일 기반 PTX의 자원 정보(ptxinfo) 로딩
 *
 * @filename: PTX 파일 경로 — 이 파일에 대해 ptxas를 실행하여 자원 사용량을 얻음
 * @sm_version: ptxas에 전달할 SM 아키텍처 버전 (예: 20, 30, 52)
 *
 * 이미 디스크에 존재하는 PTX 파일에 대해 ptxas를 호출하고, 표준에러를 파일명+"as"
 * 형태의 임시 파일로 리다이렉트한다. 그 후 ptxinfo 렉서/파서를 통해 레지스터 수,
 * lmem/smem 사용량 등을 파싱하여 gpgpu_context에 등록한다.
 * CDP가 활성화되면 --compile-only 플래그를 추가하여 링크 단계를 생략한다.
 * 이 함수는 파일 기반 PTX 로딩 경로에서 사용된다.
 * 실행 컨텍스트: 커널 로딩 단계, 호스트 유저스페이스 (단일 스레드).
 *
 * 호출 체인:
 *   libcuda (파일 기반 PTX 로딩) → [이 함수]
 *     → system(ptxas) → ptxinfo_lex_init → ptxinfo_parse → ptxinfo_addinfo
 */
void gpgpu_context::gpgpu_ptx_info_load_from_filename(const char *filename,
                                                      unsigned sm_version) {
  std::string ptxas_filename(std::string(filename) + "as");
  char buff[1024], extra_flags[1024];
  extra_flags[0] = 0;
  if (!device_runtime->g_cdp_enabled)
    snprintf(extra_flags, 1024, "--gpu-name=sm_%u", sm_version);
  else
    snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u", sm_version);
  snprintf(
      buff, 1024,
      "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  /dev/null 2> %s",
      extra_flags, filename, ptxas_filename.c_str());
  int result = system(buff);
  if (result != 0) {
    printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
    printf("               Ensure ptxas is in your path.\n");
    exit(1);
  }

  FILE *ptxinfo_in;
  ptxinfo->g_ptxinfo_filename = strdup(ptxas_filename.c_str());
  ptxinfo_in = fopen(ptxinfo->g_ptxinfo_filename, "r");
  ptxinfo_lex_init(&(ptxinfo->scanner));
  ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
  ptxinfo_parse(ptxinfo->scanner, ptxinfo);
  ptxinfo_lex_destroy(ptxinfo->scanner);
  fclose(ptxinfo_in);
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_ptxinfo_load_from_string - 문자열(ELF 임베디드 PTX)로부터
 *   ptxinfo 자원 정보 로딩
 *
 * @p_for_info: ELF에서 추출한 PTX 소스 문자열 (CDP 경로에서만 사용, no_of_ptx==0)
 * @source_num: PTX 소스 번호 (현재는 이 함수 난 낮게 직접 사용하지 않고 파일명 패턴에 사용)
 * @sm_version: ptxas에 전달할 SM 아키텍처 버전 (no_of_ptx==0 CDP 경로에서 사용)
 * @no_of_ptx: 임베디드 PTX 파일 개수 — 0보다 크면 개별 파일별로 ptxas 실행, 0이면 CDP dump 전체 처리
 *
 * 두 가지 경로를 처리한다:
 *   1) no_of_ptx > 0: "<binary_name>.<index>.sm_<ver>.ptx" 파일들을 개별적으로
 *      임시 파일로 복사한 뒤 ptxas 실행 → 각각 ptxinfo 수집 → 마지막에 cat으로 합침.
 *      중복 정의 오류(result == 65280) 발생 시 fix_duplicate_errors()로 자동 복구 후 재시도.
 *   2) no_of_ptx == 0 (CDP): p_for_info 문자열 전체를 임시 파일로 dump한 뒤
 *      동일한 ptxas 파이프라인으로 처리.
 * CUDART_VERSION >= 3000 환경에서는 -gpgpu_occupancy_sm_number 옵션이 설정되어 있어야
 * 레지스터 사용량/점유율 계산용 SM 버전을 결정할 수 있다. 설정되지 않으면 오류로 종료.
 * 임시 파일들은 g_keep_intermediate_files(-keep) 옵션이 꺼진 경우 삭제된다.
 * 실행 컨텍스트: 커널 로딩 단계, 호스트 유저스페이스 (단일 스레드).
 *
 * 관련 gpgpusim.config 옵션:
 *   -gpgpu_occupancy_sm_number: 레지스터 사용량/점유율 계산용 SM 버전 (필수)
 *   -keep: 임시 ptxinfo 파일 유지 여부
 *   -gpgpu_cuda_cdp_enabled: CDP 모드 여부에 따라 --compile-only 플래그 추가
 *
 * 호출 체인:
 *   libcuda (cuModuleLoad 인터셉트) → [이 함수]
 *     → get_app_binary_name → system(ptxas/sed/cat) → fix_duplicate_errors(중복 오류 시)
 *     → ptxinfo_parse → ptxinfo_addinfo
 */
void gpgpu_context::gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                                   unsigned source_num,
                                                   unsigned sm_version,
                                                   int no_of_ptx) {
  // do ptxas for individual files instead of one big embedded ptx. This
  // prevents the duplicate defs and declarations.
  char ptx_file[1000];
  char *name = get_app_binary_name();
  char commandline[4096], fname[1024], fname2[1024],
      final_tempfile_ptxinfo[1024], tempfile_ptxinfo[1024];
  for (int index = 1; index <= no_of_ptx; index++) {
    snprintf(ptx_file, 1000, "%s.%d.sm_%u.ptx", name, index, sm_version);
    snprintf(fname, 1024, "_ptx_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);

    printf("GPGPU-Sim PTX: extracting embedded .ptx to temporary file \"%s\"\n",
           fname);
    snprintf(commandline, 4096, "cat %s > %s", ptx_file, fname);
    if (system(commandline) != 0) {
      printf("ERROR: %s command failed\n", commandline);
      exit(0);
    }

    snprintf(fname2, 1024, "_ptx2_XXXXXX");
    fd = mkstemp(fname2);
    close(fd);
    char commandline2[4096];
    snprintf(commandline2, 4096,
             "cat %s | sed 's/.version 1.5/.version 1.4/' | sed 's/, "
             "texmode_independent//' | sed 's/\\(\\.extern \\.const\\[1\\] .b8 "
             "\\w\\+\\)\\[\\]/\\1\\[1\\]/' | sed "
             "'s/const\\[.\\]/const\\[0\\]/g' > %s",
             fname, fname2);
    printf("Running: %s\n", commandline2);
    int result = system(commandline2);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (a) %d\n", result);
      printf(
          "               Ensure you have write access to simulation "
          "directory\n");
      printf("               and have \'cat\' and \'sed\' in your path.\n");
      exit(1);
    }

    snprintf(tempfile_ptxinfo, 1024, "%sinfo", fname);
    char extra_flags[1024];
    extra_flags[0] = 0;

#if CUDART_VERSION >= 3000
    if (g_occupancy_sm_number == 0) {
      fprintf(
          stderr,
          "gpgpusim.config must specify the sm version for the GPU that you "
          "use to compute occupancy \"-gpgpu_occupancy_sm_number XX\".\n"
          "The register file size is specifically tied to the sm version used "
          "to querry ptxas for register usage.\n"
          "A register size/SM mismatch may result in occupancy differences.");
      exit(1);
    }
    if (!device_runtime->g_cdp_enabled)
      snprintf(extra_flags, 1024, "--gpu-name=sm_%u", g_occupancy_sm_number);
    else
      snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u",
               g_occupancy_sm_number);
#endif

    snprintf(commandline, 1024,
             "$PTXAS_CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  "
             "/dev/null 2> %s",
             extra_flags, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: generating ptxinfo using \"%s\"\n", commandline);
    result = system(commandline);
    if (result != 0) {
      // 65280 = duplicate errors
      if (result == 65280) {
        FILE *ptxinfo_in;
        ptxinfo_in = fopen(tempfile_ptxinfo, "r");
        ptxinfo->g_ptxinfo_filename = tempfile_ptxinfo;
        ptxinfo_lex_init(&(ptxinfo->scanner));
        ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
        ptxinfo_parse(ptxinfo->scanner, ptxinfo);
        ptxinfo_lex_destroy(ptxinfo->scanner);
        fclose(ptxinfo_in);

        fix_duplicate_errors(fname2);
        snprintf(commandline, 1024,
                 "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  "
                 "/dev/null 2> %s",
                 extra_flags, fname2, tempfile_ptxinfo);
        printf("GPGPU-Sim PTX: regenerating ptxinfo using \"%s\"\n",
               commandline);
        result = system(commandline);
      }
      if (result != 0) {
        printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
        printf("               Ensure ptxas is in your path.\n");
        exit(1);
      }
    }
  }

  // TODO: duplicate code! move it into a function so that it can be reused!
  if (no_of_ptx == 0) {
    // For CDP, we dump everything. So no_of_ptx will be 0.
    snprintf(fname, 1024, "_ptx_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);

    printf("GPGPU-Sim PTX: extracting embedded .ptx to temporary file \"%s\"\n",
           fname);
    FILE *ptxfile = fopen(fname, "w");
    fprintf(ptxfile, "%s", p_for_info);
    fclose(ptxfile);

    snprintf(fname2, 1024, "_ptx2_XXXXXX");
    fd = mkstemp(fname2);
    close(fd);
    char commandline2[4096];
    snprintf(commandline2, 4096,
             "cat %s | sed 's/.version 1.5/.version 1.4/' | sed 's/, "
             "texmode_independent//' | sed 's/\\(\\.extern \\.const\\[1\\] .b8 "
             "\\w\\+\\)\\[\\]/\\1\\[1\\]/' | sed "
             "'s/const\\[.\\]/const\\[0\\]/g' > %s",
             fname, fname2);
    printf("Running: %s\n", commandline2);
    int result = system(commandline2);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (a) %d\n", result);
      printf(
          "               Ensure you have write access to simulation "
          "directory\n");
      printf("               and have \'cat\' and \'sed\' in your path.\n");
      exit(1);
    }
    // char tempfile_ptxinfo[1024];
    snprintf(tempfile_ptxinfo, 1024, "%sinfo", fname);
    char extra_flags[1024];
    extra_flags[0] = 0;

#if CUDART_VERSION >= 3000
    if (sm_version == 0) sm_version = 20;
    if (!device_runtime->g_cdp_enabled)
      snprintf(extra_flags, 1024, "--gpu-name=sm_%u", sm_version);
    else
      snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u",
               sm_version);
#endif

    snprintf(
        commandline, 1024,
        "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  /dev/null 2> %s",
        extra_flags, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: generating ptxinfo using \"%s\"\n", commandline);
    fflush(stdout);
    result = system(commandline);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
      printf("               Ensure ptxas is in your path.\n");
      exit(1);
    }
  }

  // Now that we got resource usage per kernel in a ptx file, we dump all into
  // one file and pass it to rest of the code as usual.
  if (no_of_ptx > 0) {
    char commandline3[4096];
    snprintf(final_tempfile_ptxinfo, 1024, "f_tempfile_ptx");
    snprintf(commandline3, 4096, "cat *info > %s", final_tempfile_ptxinfo);
    if (system(commandline3) != 0) {
      printf("ERROR: Either we dont have info files or cat is not working \n");
      printf("ERROR: %s command failed\n", commandline3);
      exit(1);
    }
  }

  if (no_of_ptx > 0)
    ptxinfo->g_ptxinfo_filename = final_tempfile_ptxinfo;
  else
    ptxinfo->g_ptxinfo_filename = tempfile_ptxinfo;
  FILE *ptxinfo_in;
  ptxinfo_in = fopen(ptxinfo->g_ptxinfo_filename, "r");

  ptxinfo_lex_init(&(ptxinfo->scanner));
  ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
  ptxinfo_parse(ptxinfo->scanner, ptxinfo);
  ptxinfo_lex_destroy(ptxinfo->scanner);
  fclose(ptxinfo_in);

  snprintf(commandline, 1024, "rm -f *info");
  if (system(commandline) != 0) {
    printf("GPGPU-Sim PTX: ERROR ** while removing temporary info files\n");
    exit(1);
  }
  if (!g_save_embedded_ptx) {
    if (no_of_ptx > 0)
      snprintf(commandline, 1024, "rm -f %s %s %s", fname, fname2,
               final_tempfile_ptxinfo);
    else
      snprintf(commandline, 1024, "rm -f %s %s %s", fname, fname2,
               tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: removing ptxinfo using \"%s\"\n", commandline);
    if (system(commandline) != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while removing temporary files\n");
      exit(1);
    }
  }
}
