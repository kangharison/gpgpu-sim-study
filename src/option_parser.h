// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

/*
 * [한국어 설명] GPGPU-Sim 설정 파서 공개 인터페이스 (option_parser.h)
 *
 * === 파일의 역할 ===
 * 이 헤더 파일은 GPGPU-Sim 시뮬레이터 전체의 동작 파라미터를 설정 파일
 * (gpgpusim.config) 또는 커맨드라인 인자로부터 파싱하여 각 모듈의 C 변수에
 * 자동으로 바인딩하는 옵션 파서 모듈의 공개 C 인터페이스를 정의한다.
 * SM(Streaming Multiprocessor) 수, L1/L2 캐시 크기, DRAM 타이밍 파라미터,
 * 워프(warp) 스케줄러 종류, NoC(네트워크온칩) 라우팅 설정 등 시뮬레이터의
 * 모든 하드웨어 파라미터가 이 파서를 통해 런타임에 각 모듈 변수로 주입된다.
 * 실제 파서 구현은 option_parser.cc에 있으며, OptionParser C++ 클래스로
 * 캡슐화되어 있다. 이 헤더는 C 코드에서도 사용할 수 있도록 불투명 포인터
 * (opaque pointer) 패턴으로 C++ 클래스를 숨긴다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 시뮬레이터 초기화 파이프라인에서 가장 이른 단계에 위치한다. GPGPU-Sim은
 * 사이클-레벨(cycle-level) 타이밍 모델을 갖는 시뮬레이터로, 시뮬레이션을
 * 시작하기 전에 모든 하드웨어 파라미터가 확정되어야 한다. 이 옵션 파서는
 * 그 확정 과정을 담당하며, 이후 gpu-sim, shader, dram 등 모든 타이밍 모델
 * 모듈들이 파싱된 값을 읽어 자신의 내부 상태를 초기화한다.
 *
 * 호출 체인 (초기화 순서):
 *   gpgpusim_entrypoint.cc (시뮬레이터 진입점)
 *     → option_parser_create()         -- 파서 객체 생성
 *     → option_parser_register() × N   -- 각 모듈이 자신의 옵션 등록
 *     → option_parser_cfgfile()         -- gpgpusim.config 파일 파싱
 *     → option_parser_print()           -- 파싱 결과를 stdout/로그에 출력
 *     → 각 모듈 초기화 (gpu-sim, shader, dram, intersim2 등)
 *
 * 실행 컨텍스트: 호스트 유저스페이스(CPU 쪽), 시뮬레이션 시작 전
 * 초기화 단계에서만 호출된다. 시뮬레이션 사이클 루프가 시작된 이후에는
 * 이 파서 함수들이 호출되지 않는다 (설정은 불변).
 *
 * === 타 모듈과의 연결 ===
 * [이 파일에 의존하는 모듈 — include 측]
 *   - src/gpgpusim_entrypoint.cc : 파서 생성/파일 파싱/출력을 담당하는 진입점
 *   - src/gpgpu-sim/gpu-sim.cc   : SM 수, L2 캐시 크기, 클럭 주파수 등 등록
 *   - src/gpgpu-sim/shader.cc    : 워프 스케줄러 종류, 파이프라인 폭, 레지스터
 *                                  파일 크기, scoreboard 크기 등 등록
 *   - src/gpgpu-sim/gpu-cache.cc : L1D/L1T/L1C/L2 캐시 파라미터 등록
 *   - src/gpgpu-sim/dram.cc      : DRAM 타이밍(tRCD, tCAS, tRP 등) 파라미터 등록
 *   - src/intersim2/             : NoC 토폴로지, 라우팅, 버퍼 크기 등 등록
 *   - src/accelwattch/           : 전력 모델 파라미터 등록
 *
 * [이 파일이 의존하는 모듈 — 구현 측]
 *   - option_parser.cc           : OptionParser 클래스의 실제 구현
 *
 * 데이터 흐름: gpgpusim.config 텍스트 파일 → option_parser_cfgfile() 파싱
 * → OptionParser 내부 옵션 테이블에서 이름 조회 → 등록된 void* variable 주소에
 * 타입에 맞게 값을 기록 → 각 모듈의 전역/멤버 변수가 설정값으로 채워짐.
 *
 * 공유 자료구조: OptionParser 클래스 (option_parser.cc 내부 정의). 외부에서는
 * option_parser_t 불투명 포인터로만 접근하며, 내부 구조는 공개되지 않는다.
 *
 * === 주요 함수/구조체 요약 ===
 * option_parser_create()           -- OptionParser 객체를 힙에 생성하고 포인터 반환
 * option_parser_destroy()          -- 파서 객체와 내부 자원을 해제
 * option_parser_register()         -- 이름/타입/변수 주소/기본값으로 옵션 하나 등록
 * option_parser_cfgfile()          -- gpgpusim.config 파일을 읽어 등록된 옵션에 값 주입
 * option_parser_cmdline()          -- argc/argv 커맨드라인 인자를 파싱하여 값 주입
 * option_parser_delimited_string() -- 구분자로 분리된 문자열을 파싱하여 값 주입
 * option_parser_print()            -- 현재 등록된 모든 옵션의 이름/값/설명을 출력
 *
 * 핵심 자료구조:
 *   option_parser_t  -- OptionParser* 불투명 포인터. C 인터페이스에서 파서 핸들로 사용.
 *   option_dtype     -- 각 옵션의 C 데이터 타입을 식별하는 열거형.
 *                       파서가 문자열 값을 올바른 이진 표현으로 변환하는 데 사용.
 */

#pragma once
/* [한국어] 헤더 중복 include 방지 가드.
 * 전통적인 #ifndef/#define/#endif 패턴 대신 #pragma once를 사용하여
 * 컴파일러 수준에서 중복 포함을 차단한다. GPGPU-Sim의 다수 모듈이
 * 이 헤더를 include하므로 중복 방지가 필수이다. */

#include <stdio.h>
/* [한국어] 표준 C 입출력 라이브러리.
 * option_parser_print() 함수의 fout 파라미터 타입인 FILE*를 위해 필요하다.
 * 파싱 결과를 stdout 또는 로그 파일로 출력하는 데 사용된다. */

#include <stdlib.h>
/* [한국어] 표준 C 유틸리티 라이브러리.
 * option_parser.cc 구현에서 atoi(), atof(), strtol() 등 문자열-숫자 변환
 * 함수와 malloc()/free() 메모리 관리 함수를 사용하기 위해 필요하다.
 * 헤더 수준에서는 직접 사용되지 않지만, 구현 파일과의 일관성을 위해 포함. */

// pointer to C++ class
typedef class OptionParser *option_parser_t;
/* [한국어] OptionParser C++ 클래스에 대한 불투명 포인터(opaque pointer) 타입 정의.
 * C 코드에서도 이 헤더를 include할 수 있도록, C++ 클래스를 직접 노출하지
 * 않고 전방 선언(forward declaration) 형태의 불투명 포인터로 감싼다.
 * 설정자: option_parser_create()가 new OptionParser()로 생성하여 반환.
 * 읽는 자: option_parser_register/cfgfile/print 등 모든 API 함수.
 * 값 범위: 유효한 힙 포인터(NULL 불가). option_parser_destroy() 이후 무효.
 * 동기화: 시뮬레이션 초기화 단계(단일 스레드)에서만 사용되므로 별도 락 불필요.
 * 이 패턴은 C에서 C++ 객체를 핸들로 전달하는 표준 관용구로,
 * 내부 구현 변경 시 이 헤더를 include하는 모든 파일을 재컴파일하지
 * 않아도 되는 이점(컴파일 방화벽, compilation firewall)도 제공한다. */

// data type of the option
enum option_dtype {
  /* [한국어] 옵션 값의 C 데이터 타입을 식별하는 열거형.
   * option_parser_register() 호출 시 등록된 void* variable 포인터를
   * 어떤 타입으로 캐스팅하여 값을 기록할지를 파서에게 알려준다.
   * 파서는 gpgpusim.config의 문자열 값을 이 타입에 따라 적절한
   * 변환 함수(atoi, atof, strdup 등)로 해석하여 변수에 저장한다. */

  OPT_INT32,
  /* [한국어] 부호 있는 32비트 정수 (int / int32_t).
   * 사용 예: SM(Streaming Multiprocessor) 수(-gpgpu_n_clusters),
   * warp 스케줄러의 타임슬라이스 크기, 파이프라인 스테이지 레이턴시 등
   * 음수가 허용되는 정수 설정값에 사용된다.
   * 파서 내부에서 strtol() 또는 atoi()로 문자열을 변환한다. */

  OPT_UINT32,
  /* [한국어] 부호 없는 32비트 정수 (unsigned int / uint32_t).
   * 사용 예: 캐시 라인 크기(-gpgpu_cache:dl1 등의 크기 필드),
   * DRAM 채널 수(-gpgpu_n_mem), 레지스터 파일 크기 등
   * 음수가 의미 없는 하드웨어 크기/개수 파라미터에 사용된다.
   * 파서 내부에서 strtoul()로 문자열을 변환한다. */

  OPT_INT64,
  /* [한국어] 부호 있는 64비트 정수 (long long / int64_t).
   * 사용 예: 시뮬레이션 최대 사이클 수(-gpgpu_max_cycle)처럼
   * 32비트 범위(약 42억)를 초과할 수 있는 큰 정수 파라미터에 사용된다.
   * 파서 내부에서 strtoll()로 문자열을 변환한다. */

  OPT_UINT64,
  /* [한국어] 부호 없는 64비트 정수 (unsigned long long / uint64_t).
   * 사용 예: GPU 전역 메모리(DRAM) 총 크기처럼 매우 큰 바이트 단위 용량,
   * 또는 64비트 주소 공간 관련 마스크/오프셋 값에 사용된다.
   * 파서 내부에서 strtoull()로 문자열을 변환한다. */

  OPT_BOOL,
  /* [한국어] 불리언 플래그 (bool 또는 int, 0/1).
   * 사용 예: 특정 기능의 활성화 여부(-gpgpu_perfect_mem,
   * -gpgpu_cache_wt_through 등). config 파일에서 "1"/"0" 또는
   * "true"/"false"로 지정하며, 파서가 bool/int 변수에 0 또는 1을 기록.
   * GPGPU-Sim 내부에서는 주로 int로 선언되어 사용된다. */

  OPT_FLOAT,
  /* [한국어] 단정밀도 부동소수점 (float).
   * 사용 예: 클럭 주파수 비율, 전력 모델 스케일링 계수 등
   * 소수점이 필요하지만 높은 정밀도가 불필요한 파라미터에 사용된다.
   * 파서 내부에서 atof()로 문자열을 변환한 뒤 float로 캐스팅한다. */

  OPT_DOUBLE,
  /* [한국어] 배정밀도 부동소수점 (double).
   * 사용 예: AccelWattch(전력 모델)의 고정밀 전력 계수, 통계 집계용
   * 누적 값 등 높은 정밀도가 요구되는 파라미터에 사용된다.
   * 파서 내부에서 atof() (또는 strtod())로 문자열을 변환한다. */

  OPT_CHAR,
  /* [한국어] 단일 문자 (char).
   * 사용 예: 구분자 문자, 모드 식별자처럼 한 글자로 표현되는 옵션에 사용.
   * config 파일에서 단일 문자열 "X"를 읽어 char 변수에 첫 번째 문자를 저장.
   * 비교적 드물게 사용되는 타입이다. */

  OPT_CSTR
  /* [한국어] C 문자열 포인터 (const char* 또는 char*).
   * 사용 예: 캐시 설정 문자열(-gpgpu_cache:dl1 "S:4:128:256,..."),
   * warp 스케줄러 이름(-gpgpu_scheduler "lrr"), DRAM 모델 파라미터 문자열 등
   * 복잡한 복합 설정을 하나의 문자열 토큰으로 전달할 때 사용된다.
   * 파서 내부에서 strdup()로 문자열을 복사하여 char* 변수에 포인터를 저장.
   * 호출자는 이 문자열을 직접 free()해서는 안 되며, 파서가 소유권을 갖는다. */
};

/*
 * [한국어]
 * option_parser_create - OptionParser 객체를 생성하고 핸들을 반환한다.
 *
 * @return: 새로 할당된 OptionParser 객체에 대한 불투명 포인터(option_parser_t).
 *          실패 시 NULL을 반환하거나 내부에서 abort()할 수 있다.
 *          호출자는 반환된 핸들을 사용 후 반드시 option_parser_destroy()로 해제해야 한다.
 *
 * 이 함수는 시뮬레이터 초기화 파이프라인의 첫 번째 단계로 호출된다.
 * 내부적으로 new OptionParser()를 호출하여 옵션 테이블(이름→타입/변수/기본값 맵)을
 * 초기화한 뒤, 해당 포인터를 option_parser_t로 캐스팅하여 반환한다.
 * 반환된 핸들은 이후 option_parser_register() 호출 시 첫 번째 인자로 전달된다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 시작 전 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [option_parser_create] → (OptionParser 생성자)
 */
// create and destroy option parser
option_parser_t option_parser_create();

/*
 * [한국어]
 * option_parser_destroy - OptionParser 객체와 내부 자원을 해제한다.
 *
 * @opp: option_parser_create()가 반환한 유효한 파서 핸들.
 *       NULL 전달 시 동작은 정의되지 않는다(구현에 따라 crash 가능).
 *
 * 내부적으로 delete opp를 호출하여 OptionParser 객체와 그것이 소유하는
 * 모든 내부 자원(OPT_CSTR 타입으로 strdup된 문자열 등)을 해제한다.
 * 이 함수 호출 이후 opp 핸들은 무효화되므로 다시 사용해서는 안 된다.
 * 실제 시뮬레이터 코드에서는 프로세스 종료 시점까지 파서를 유지하는
 * 경우가 많아 명시적 destroy가 생략되기도 한다(OS가 정리).
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 종료 후 정리 단계.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [option_parser_destroy] → (OptionParser 소멸자)
 */
void option_parser_destroy(option_parser_t opp);

/*
 * [한국어]
 * option_parser_register - 파서에 새로운 옵션 하나를 등록한다.
 *
 * @opp:          option_parser_create()가 반환한 파서 핸들.
 * @name:         config 파일 또는 커맨드라인에서 인식할 옵션 이름 문자열.
 *                예: "-gpgpu_n_clusters", "-gpgpu_cache:dl1".
 *                이 이름이 파일에 등장하면 variable 주소에 값을 기록한다.
 * @type:         옵션 값의 C 데이터 타입 (option_dtype 열거형).
 *                파서가 문자열을 올바른 이진 타입으로 변환하는 데 사용.
 * @variable:     파싱된 값을 저장할 C 변수의 주소 (void* 포인터).
 *                type에 따라 int*, unsigned int*, float*, char** 등으로 해석.
 *                이 포인터는 파서보다 오래 살아야 한다 (dangling pointer 위험).
 * @desc:         이 옵션에 대한 영문 설명 문자열. option_parser_print() 출력에 사용.
 * @defaultvalue: 파일/커맨드라인에 해당 옵션이 없을 때 사용할 기본값 문자열.
 *                NULL이 아닌 경우, 등록 시점에 즉시 variable에 기본값을 기록한다.
 *
 * 각 모듈(gpu-sim, shader, dram, intersim2 등)은 자신의 하드웨어 파라미터를
 * 이 함수로 파서에 등록한다. 등록은 실제 파일 파싱(option_parser_cfgfile)보다
 * 먼저 이루어져야 한다. 파서는 내부적으로 name → {type, variable, desc, default}
 * 테이블을 유지하며, 파싱 시 이름 매칭으로 해당 변수에 값을 기록한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 초기화 단계 단일 스레드.
 *
 * 호출 체인:
 *   gpu_sim_t::reg_options()   → [option_parser_register] (SM 수, L2 크기 등)
 *   shader_core_config::init() → [option_parser_register] (warp 스케줄러 등)
 *   dram_t::reg_options()      → [option_parser_register] (DRAM 타이밍 등)
 *   intersim2 설정 함수        → [option_parser_register] (NoC 파라미터 등)
 */
// register new option
void option_parser_register(option_parser_t opp, const char *name,
                            enum option_dtype type, void *variable,
                            const char *desc, const char *defaultvalue);

/*
 * [한국어]
 * option_parser_cmdline - 커맨드라인 인자(argc/argv)를 파싱하여 등록된 옵션에 값을 주입한다.
 *
 * @opp:  option_parser_create()가 반환한 파서 핸들.
 * @argc: main()에서 전달된 인자 개수.
 * @argv: main()에서 전달된 인자 문자열 배열.
 *        argv[i]가 등록된 옵션 이름과 일치하면 argv[i+1]을 값으로 파싱한다.
 *
 * GPGPU-Sim을 직접 실행할 때 커맨드라인으로 설정을 오버라이드하는 경우에
 * 사용된다. gpgpusim.config 파일 파싱(option_parser_cfgfile) 이후에 호출하면
 * 커맨드라인 옵션이 파일 설정을 덮어쓰는 우선순위를 구현할 수 있다.
 * execution-driven 모드에서는 CUDA 애플리케이션의 argv가 그대로 전달될 수
 * 있으므로, 파서는 인식하지 못한 인자를 무시하거나 에러 처리한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이터 초기화 단계 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [option_parser_cmdline]
 *     → (OptionParser 내부 argv 파싱 루프)
 */
// parse command line
void option_parser_cmdline(option_parser_t opp, int argc, const char *argv[]);

/*
 * [한국어]
 * option_parser_cfgfile - gpgpusim.config 설정 파일을 파싱하여 등록된 옵션에 값을 주입한다.
 *
 * @opp:      option_parser_create()가 반환한 파서 핸들.
 * @filename: 파싱할 설정 파일 경로 문자열 (예: "gpgpusim.config").
 *            파일이 존재하지 않으면 에러를 출력하고 abort()하거나 무시한다.
 *
 * 이 함수는 GPGPU-Sim 설정 파이프라인의 핵심 단계이다. gpgpusim.config는
 * 시뮬레이션할 GPU 아키텍처(예: GTX480, Volta, Turing)의 모든 하드웨어
 * 파라미터를 "-옵션이름 값" 형태로 기술한 텍스트 파일이다.
 * 파서는 파일을 한 줄씩 읽어 "#"으로 시작하는 주석을 건너뛰고,
 * 각 옵션 이름을 내부 테이블에서 검색하여 대응하는 변수 주소에 값을 기록한다.
 * 등록되지 않은 옵션 이름이 파일에 있으면 경고를 출력하거나 무시한다.
 * 이 함수 호출 후 모든 하드웨어 파라미터가 확정되므로, 이후 각 모듈의
 * 초기화(예: shader_core_config::init())가 의미 있는 값을 읽을 수 있다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 시작 전 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [option_parser_cfgfile]
 *     → (fopen → 줄 단위 파싱 → option 이름 조회 → 타입별 변환 → 변수 기록)
 */
// parse config file
void option_parser_cfgfile(option_parser_t opp, const char *filename);

/*
 * [한국어]
 * option_parser_delimited_string - 구분자로 분리된 문자열을 파싱하여 등록된 옵션에 값을 주입한다.
 *
 * @opp:         option_parser_create()가 반환한 파서 핸들.
 * @inputstring: 파싱할 옵션 내용이 담긴 문자열.
 *               예: "-gpgpu_n_clusters 15 -gpgpu_n_sp_units 4".
 *               파일이나 커맨드라인 대신 프로그램 내부에서 동적으로 생성된
 *               설정 문자열을 파싱할 때 사용한다.
 * @delimiters:  inputstring을 토큰으로 분리할 구분자 문자들의 문자열.
 *               예: " \t\n" (공백, 탭, 개행). strtok() 계열 함수에 전달된다.
 *
 * 이 함수는 option_parser_cfgfile()의 문자열 버전으로, 파일 I/O 없이
 * 메모리 상의 설정 문자열을 직접 파싱한다. trace-driven 시뮬레이션 모드에서
 * 트레이스 파일 헤더에 내장된 설정 문자열을 파싱하거나, 테스트 코드에서
 * 프로그래매틱하게 설정을 주입할 때 유용하다. 내부적으로 inputstring을
 * 복사한 뒤 구분자로 토큰 분리하여 option_parser_cmdline()과 유사하게 처리한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 초기화 단계 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc 또는 트레이스 로더
 *     → [option_parser_delimited_string]
 *         → (strtok 계열로 토큰 분리 → 옵션 이름 조회 → 변수 기록)
 */
// parse a delimited string
void option_parser_delimited_string(option_parser_t opp,
                                    const char *inputstring,
                                    const char *delimiters);

/*
 * [한국어]
 * option_parser_print - 현재 등록된 모든 옵션의 이름, 현재 값, 설명을 출력한다.
 *
 * @opp:  option_parser_create()가 반환한 파서 핸들.
 * @fout: 출력 대상 파일 스트림. stdout(표준 출력) 또는 열린 파일 포인터.
 *        NULL 전달 시 동작은 구현에 따라 다를 수 있다.
 *
 * 시뮬레이터 초기화 완료 후 현재 적용된 설정 전체를 로그로 남기기 위해
 * 호출된다. 출력 형식은 일반적으로 "옵션이름 = 현재값  # 설명" 형태이며,
 * 이를 통해 재현 가능한 시뮬레이션 설정을 기록할 수 있다. GPGPU-Sim 실행
 * 로그에서 "GPGPU-Sim: Configuration..." 섹션이 이 함수의 출력이다.
 * 연구자가 실험 설정을 검증하고 재현성을 확보하는 데 중요한 역할을 한다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 시작 직전 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [option_parser_print]
 *     → (OptionParser 내부 옵션 테이블 순회 → fprintf(fout, ...) 반복)
 */
// print options
void option_parser_print(option_parser_t opp, FILE *fout);
