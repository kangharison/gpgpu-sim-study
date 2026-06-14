/*
 * [한국어 설명] GPGPU-Sim 전역 시뮬레이션 컨텍스트 선언 (gpgpu_context.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 시뮬레이터 전체의 최상위 루트 객체인 gpgpu_context 클래스를 선언한다.
 * gpgpu_context는 시뮬레이터의 모든 서브시스템(CUDA 런타임 API, PTX 파서, 기능 시뮬레이터,
 * 타이밍 시뮬레이터, 디바이스 런타임, 전력 통계)을 소유하고 생성하는 루트 컨테이너 역할을 한다.
 * 또한 PTX 명령어 UID, 워프 인스트럭션 UID, 심볼 UID 등 시뮬레이터 전체에서 고유해야 하는
 * 카운터들을 중앙 집중 관리하여 여러 모듈 간의 일관된 식별자를 보장한다.
 * GPGPU_Context() 전역 함수를 통해 싱글턴 방식으로 접근되며, 시뮬레이션 전 기간 동안 유지된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일은 libcuda/ 레이어에 속하며, CUDA 런타임 인터셉트 계층의 최상단에 위치한다.
 * 실행 흐름:
 *   CUDA Application (cuInit, cuLaunchKernel 등)
 *       → libcuda 런타임 인터셉트 (cuda_runtime_api.cc)
 *           → GPGPU_Context() 싱글턴 획득 → gpgpu_context 멤버 함수 호출
 *               ├── PTX 로드: gpgpu_ptx_sim_load_ptx_from_string/filename()
 *               ├── 타이밍 시뮬레이션 초기화: gpgpu_ptx_sim_init_perf()
 *               ├── OpenCL 성능 시뮬레이션: gpgpu_opencl_ptx_sim_main_perf()
 *               ├── SST 동기화: synchronize() / synchronize_check()
 *               └── 시뮬레이션 종료: exit_simulation()
 * 실행 컨텍스트: CPU 호스트 유저스페이스, CUDA 런타임 초기화 및 커널 런치 시.
 * gpgpu_context는 시뮬레이션 전 기간 동안 힙에 유지되며, 프로그램 종료 시 소멸한다.
 *
 * === 타 모듈과의 연결 ===
 * 이 파일이 의존하는 모듈 (include):
 *   - cuda-sim/cuda-sim.h: cuda_sim 클래스 (PTX 기능 시뮬레이션 엔진)
 *   - cuda-sim/cuda_device_runtime.h: cuda_device_runtime (디바이스 측 CUDA 런타임 서비스)
 *   - cuda-sim/ptx-stats.h: ptx_stats (PTX 명령어 통계 수집)
 *   - cuda-sim/ptx_loader.h: PTX 로드 유틸리티 함수들
 *   - cuda-sim/ptx_parser.h: ptx_recognizer (Flex/Bison PTX 파서), ptxinfo_data
 *   - gpgpusim_entrypoint.h: GPGPUsim_ctx (타이밍 시뮬레이션 진입점 컨텍스트)
 *   - cuda_api_object.h: cuda_runtime_api (CUDA API 인터셉트 객체)
 * 이 파일에 의존하는 모듈:
 *   - libcuda/cuda_runtime_api.cc: GPGPU_Context()를 통해 gpgpu_context에 접근
 *   - libcuda/cuobjdump.cc: cuobjdumpParseBinary() 호출
 *   - libcuda/libcudart.cc: CUDA 런타임 API 구현에서 컨텍스트 접근
 * 데이터 흐름:
 *   CUDA 바이너리 → cuobjdumpParseBinary() → PTX 로드 → s_g_pc_to_insn 구성
 *       → cuda-sim 기능 시뮬레이션 (pc_to_instruction) → 타이밍 모델 (ptx_fetch_inst)
 *
 * === 주요 함수/구조체 요약 ===
 * - gpgpu_context(): 생성자 — 모든 UID 카운터 초기화 및 7개 서브시스템 객체 생성
 * - GPGPU_Context(): 전역 싱글턴 접근 함수 — gpgpu_context 포인터 반환
 * - synchronize() / synchronize_check(): SST(코-시뮬레이션) 동기화 포인트
 * - gpgpu_ptx_sim_load_ptx_from_string/filename(): PTX 소스를 파싱하여 IR 구성
 * - pc_to_instruction() / ptx_fetch_inst(): PC 주소 → PTX 명령어 역방향 매핑
 * - s_g_pc_to_insn: PC → ptx_instruction* 직접 배열 매핑 (기능 시뮬레이션의 핵심 테이블)
 */

#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include "../src/cuda-sim/cuda-sim.h"              // [한국어] cuda_sim 클래스 선언 — PTX 기능 시뮬레이션 엔진 (fetch/decode/execute 사이클 처리)
#include "../src/cuda-sim/cuda_device_runtime.h"   // [한국어] cuda_device_runtime 클래스 선언 — GPU 디바이스 측에서 호출하는 CUDA 런타임 서비스 에뮬레이션 (동적 병렬성 등)
#include "../src/cuda-sim/ptx-stats.h"             // [한국어] ptx_stats 클래스 선언 — PTX 명령어별 실행 횟수, 분기 통계 등 수집
#include "../src/cuda-sim/ptx_loader.h"            // [한국어] PTX 파일 로드 유틸리티 함수들 — 파일/문자열에서 PTX를 읽어 파서에 전달
#include "../src/cuda-sim/ptx_parser.h"            // [한국어] ptx_recognizer (Flex/Bison PTX 파서), ptxinfo_data 선언 — PTX 소스를 파싱하여 ptx_instruction IR 생성
#include "../src/gpgpusim_entrypoint.h"            // [한국어] GPGPUsim_ctx 클래스 선언 — 타이밍 시뮬레이션(shader/cache/dram 사이클 루프) 진입점
#include "cuda_api_object.h"                       // [한국어] cuda_runtime_api 클래스 선언 — cuLaunchKernel 등 CUDA 런타임 API를 인터셉트하여 시뮬레이터로 리다이렉트

/*
 * [한국어]
 * gpgpu_context - GPGPU-Sim 전체 시뮬레이션 컨텍스트의 루트 객체
 *
 * 이 클래스는 GPGPU-Sim의 모든 서브시스템을 소유하고 조율하는 최상위 컨테이너이다.
 * CUDA 런타임 인터셉트 레이어(libcuda)부터 PTX 파서, 기능 시뮬레이터, 타이밍 시뮬레이터까지
 * 모든 핵심 객체를 생성자에서 한 번에 생성하고, 시뮬레이션 전 기간 동안 이 객체들의 수명을 관리한다.
 * 전역 UID 카운터들을 중앙 집중 관리하여 mem_fetch, warp_inst_t, ptx_instruction 등
 * 시뮬레이터 전체에서 생성되는 객체들의 식별자가 충돌 없이 고유함을 보장한다.
 * 실행 컨텍스트: CPU 호스트 유저스페이스, 단일 스레드(시뮬레이션 초기화 및 커널 런치 스레드).
 * GPGPU_Context() 전역 함수를 통해 싱글턴으로 사용되며, 첫 접근 시 생성된다.
 */
class gpgpu_context {
 public:
  /*
   * [한국어]
   * gpgpu_context() - GPGPU-Sim 전체 시뮬레이션 컨텍스트 생성자
   *
   * @return: 없음 (생성자)
   *
   * 시뮬레이터의 모든 전역 UID 카운터를 0 또는 1로 초기화하고,
   * 7개의 서브시스템 객체를 힙에 동적 할당한다. 각 서브시스템은 gpgpu_context* (this)를
   * 받아 역참조를 통해 다른 서브시스템에 접근할 수 있다.
   * 모든 카운터와 객체가 이 생성자에서 초기화되므로, 이 객체가 완전히 구성된 후에야
   * 시뮬레이션 관련 함수들을 안전하게 호출할 수 있다.
   * 실행 컨텍스트: CPU 호스트, GPGPU_Context() 싱글턴 초기화 시 단 한 번 실행됨.
   *
   * 호출 체인:
   *   GPGPU_Context() → [new gpgpu_context()] → 7개 서브시스템 new 호출
   *       → cuda_runtime_api / ptxinfo_data / ptx_recognizer / GPGPUsim_ctx
   *           / cuda_sim / cuda_device_runtime / ptx_stats 생성
   */
  gpgpu_context() {
    g_global_allfiles_symbol_table = NULL; // [한국어] 모든 PTX 파일에 걸친 전역 심볼 테이블 포인터를 NULL로 초기화 — PTX 로드 전에는 심볼 테이블이 없으므로 NULL이 올바른 초기 상태
    sm_next_access_uid = 0;                // [한국어] 메모리 접근(mem_fetch) 객체의 고유 ID 카운터를 0으로 초기화 — 첫 번째 mem_fetch는 uid=0을 받음
    warp_inst_sm_next_uid = 0;             // [한국어] warp_inst_t 객체(인스트럭션 슬롯)의 고유 ID 카운터를 0으로 초기화 — 워프별 실행 인스트럭션 추적에 사용
    operand_info_sm_next_uid = 1;          // [한국어] operand_info(피연산자 정보) 객체의 고유 ID 카운터를 1로 초기화 — 0을 "미할당/무효"로 예약하기 위해 1부터 시작
    kernel_info_m_next_uid = 1;            // [한국어] kernel_info_t(커널 실행 정보) 객체의 고유 ID 카운터를 1로 초기화 — 0을 "미할당"으로 예약
    g_num_ptx_inst_uid = 0;               // [한국어] ptx_instruction 객체(PTX 명령어 IR)의 전역 고유 ID 카운터를 0으로 초기화 — PTX 파싱 시 각 명령어에 고유 ID 부여
    g_ptx_cta_info_uid = 1;               // [한국어] CTA(Cooperative Thread Array, 스레드 블록) 정보 객체의 고유 ID 카운터를 1로 초기화 — 0을 "미할당"으로 예약
    symbol_sm_next_uid = 1;               // [한국어] PTX 심볼(변수, 레이블) 객체의 고유 ID 카운터를 1로 초기화 — 0을 "미할당"으로 예약하여 심볼 유효성 체크에 활용
    function_info_sm_next_uid = 1;        // [한국어] function_info(PTX 커널/함수 정보) 객체의 고유 ID 카운터를 1로 초기화 — 0을 "미할당"으로 예약
    debug_tensorcore = 0;                 // [한국어] Tensor Core 디버그 모드 플래그를 false(0)로 초기화 — gpgpusim.config에서 활성화 가능
    api = new cuda_runtime_api(this);             // [한국어] CUDA 런타임 API 인터셉트 객체 생성 — cuLaunchKernel, cuMemAlloc 등의 호출을 시뮬레이터로 리다이렉트함; this를 전달해 api가 컨텍스트 내 다른 서브시스템에 접근 가능
    ptxinfo = new ptxinfo_data(this);             // [한국어] PTX 정보 데이터 객체 생성 — PTX 파일의 메타데이터(레지스터 수, shared memory 크기 등 .ptxinfo 정보) 관리
    ptx_parser = new ptx_recognizer(this);        // [한국어] PTX 파서(Flex/Bison) 객체 생성 — PTX 소스 텍스트를 파싱하여 ptx_instruction IR 트리 구성; this를 통해 s_g_pc_to_insn 등에 결과를 기록
    the_gpgpusim = new GPGPUsim_ctx(this);        // [한국어] 타이밍 시뮬레이션 진입점 컨텍스트 생성 — gpgpu_sim(사이클-레벨 타이밍 모델)을 소유하며 시뮬레이션 루프 관리
    func_sim = new cuda_sim(this);                // [한국어] PTX 기능 시뮬레이터 객체 생성 — 각 워프의 PTX 명령어를 fetch/decode/execute하여 레지스터/메모리 상태 변경
    device_runtime = new cuda_device_runtime(this); // [한국어] GPU 디바이스 측 CUDA 런타임 서비스 에뮬레이터 생성 — 디바이스 함수 내 cudaLaunchDevice() 등 동적 병렬성 지원
    stats = new ptx_stats(this);                  // [한국어] PTX 명령어 실행 통계 수집 객체 생성 — 명령어 타입별 실행 횟수, 분기 통계를 시뮬레이션 후 출력
  }
  // global list
  symbol_table *g_global_allfiles_symbol_table;
  /* [한국어] 모든 PTX 파일에 걸쳐 공유되는 전역 심볼 테이블 포인터.
   * 여러 PTX 파일(여러 CUDA 번역 단위)이 로드될 때, 각 파일의 지역 심볼 테이블이
   * 이 전역 테이블에 병합되어 서로의 심볼을 참조할 수 있게 된다.
   * 설정자: gpgpu_ptx_sim_load_ptx_from_string/filename() 내 init_parser()가 첫 로드 시 생성,
   *         이후 PTX 파일 로드마다 심볼이 추가됨.
   * 읽는 자: PTX IR 생성 중 심볼 참조 해석(resolve), 커널 함수 탐색 시.
   * 값 범위: NULL(PTX 로드 전) 또는 유효한 symbol_table* (로드 후).
   * 동기화: PTX 로드는 시뮬레이션 초기화 단계에서 단일 스레드로 수행되므로 락 불필요. */

  const char *g_filename;
  /* [한국어] 현재 파싱 중인 PTX 파일의 경로(파일명).
   * PTX 파싱 오류 메시지에서 파일명을 출력하거나, 디버그 정보에 소스 파일을 표시할 때 사용.
   * 설정자: init_parser() 또는 gpgpu_ptx_sim_load_ptx_from_filename()에서 파싱 시작 시 설정.
   * 읽는 자: PTX 파서 오류 핸들러(yyerror), 디버그 출력 코드.
   * 값 범위: NULL 또는 유효한 C 문자열 포인터 (파일 로드 중에만 유효).
   * 동기화: 단일 파싱 스레드에서만 접근되므로 락 불필요. */

  unsigned sm_next_access_uid;
  /* [한국어] 메모리 접근(mem_fetch) 객체에 할당되는 전역 순차 고유 ID 카운터.
   * mem_fetch는 SM에서 발생하는 각각의 메모리 요청(로드/스토어/원자 연산)을 추상화한 객체이다.
   * 각 mem_fetch 생성 시 이 카운터를 후위 증가(post-increment)하여 고유한 uid를 부여한다.
   * uid는 디버그 추적, 메모리 요청 식별, MSHR(Miss Status Holding Register) 매칭에 사용된다.
   * 설정자: mem_fetch 생성자에서 gpgpu_context::sm_next_access_uid++로 획득.
   * 읽는 자: mem_fetch::get_uid()를 통해 캐시/DRAM/NoC 디버그 추적 로직.
   * 값 범위: 0부터 시뮬레이션 중 생성된 총 메모리 접근 수까지 단조 증가.
   * 동기화: 현재 단일 시뮬레이션 스레드에서만 접근되므로 별도 락 없음; SST 코-시뮬레이션 시 주의 필요. */

  unsigned warp_inst_sm_next_uid;
  /* [한국어] warp_inst_t(워프 인스트럭션 슬롯) 객체에 할당되는 전역 순차 고유 ID 카운터.
   * warp_inst_t는 SM 파이프라인을 흐르는 하나의 워프 인스트럭션 슬롯을 표현하며,
   * 각 슬롯에 고유한 uid를 부여하여 파이프라인 스테이지 간 추적 및 디버그에 사용한다.
   * 설정자: warp_inst_t 생성 시 gpgpu_context::warp_inst_sm_next_uid++로 획득.
   * 읽는 자: shader.cc의 파이프라인 디버그, 스코어보드 디버그.
   * 값 범위: 0부터 시뮬레이션 중 발행된 총 워프 인스트럭션 수까지 단조 증가.
   * 동기화: 사이클 루프 내 단일 코어 스케줄러에서 접근되므로 별도 락 없음. */

  unsigned operand_info_sm_next_uid;  // uid for operand_info
  /* [한국어] operand_info(PTX 피연산자 정보) 객체에 할당되는 전역 순차 고유 ID 카운터.
   * operand_info는 PTX 명령어의 각 피연산자(레지스터, 상수, 주소 등)를 표현하는 객체이다.
   * 1부터 시작하여 0을 "미할당/무효" 상태의 예약값으로 사용한다.
   * 설정자: operand_info 생성 시 gpgpu_context::operand_info_sm_next_uid++로 획득.
   * 읽는 자: PTX IR 빌드 과정에서 피연산자 참조 디버그.
   * 값 범위: 1 이상 (0은 미할당 예약).
   * 동기화: PTX 파싱 단계(단일 스레드)에서만 사용되므로 락 불필요. */

  unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
  /* [한국어] kernel_info_t(커널 실행 메타데이터) 객체에 할당되는 전역 순차 고유 ID 카운터.
   * kernel_info_t는 하나의 CUDA 커널 런치(그리드 크기, 블록 크기, 커널 함수 포인터 등)를
   * 표현하는 객체이다. 동시에 여러 커널이 큐에 대기할 수 있으므로 uid로 구별한다.
   * 1부터 시작하여 0을 "미할당" 상태의 예약값으로 사용한다.
   * 설정자: kernel_info_t 생성자에서 gpgpu_context::kernel_info_m_next_uid++로 획득.
   * 읽는 자: gpu-sim.cc의 커널 스케줄링, 통계 출력.
   * 값 범위: 1 이상 (0은 미할당 예약).
   * 동기화: 커널 런치는 단일 호스트 스레드에서 순차적으로 발생하므로 락 불필요. */

  unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
  /* [한국어] ptx_instruction(PTX 명령어 IR 노드) 객체에 할당되는 전역 순차 고유 ID 카운터.
   * PTX 파싱 시 각 명령어마다 고유한 uid를 부여하여 ptx_instruction 객체를 식별한다.
   * PC(Program Counter) 값과 별개의 식별자이므로, 동일 코드가 여러 번 로드되어도 구별 가능.
   * 0부터 시작하여 파싱된 모든 PTX 명령어에 순서대로 할당된다.
   * 설정자: ptx_instruction 생성자에서 gpgpu_context::g_num_ptx_inst_uid++로 획득.
   * 읽는 자: PTX IR 디버그, ptx_stats 통계 인덱싱.
   * 값 범위: 0 이상 (파싱된 총 PTX 명령어 수까지 단조 증가).
   * 동기화: PTX 파싱 단계(단일 스레드)에서만 사용되므로 락 불필요. */

  unsigned long long g_ptx_cta_info_uid;
  /* [한국어] ptx_cta_info(CTA = Cooperative Thread Array, 스레드 블록 정보) 객체의 전역 고유 ID.
   * CTA는 CUDA의 스레드 블록(thread block)에 해당하며, 같은 SM에서 함께 실행되는 워프들의 묶음이다.
   * __syncthreads() 등 블록 내 동기화 구현에서 어떤 워프들이 같은 블록인지 추적하기 위해 uid를 사용한다.
   * unsigned long long을 사용하는 이유: 대규모 시뮬레이션에서 수백만 개의 CTA가 생성될 수 있음.
   * 설정자: ptx_cta_info 생성자에서 gpgpu_context::g_ptx_cta_info_uid++로 획득.
   * 읽는 자: __syncthreads() 에뮬레이션 코드, CTA 완료 감지 로직.
   * 값 범위: 1 이상 (0은 미할당 예약; long long이므로 오버플로우 실질적 불가).
   * 동기화: 기능 시뮬레이션 단계에서 단일 호스트 스레드로 처리되므로 락 불필요. */

  unsigned symbol_sm_next_uid;  // uid for symbol
  /* [한국어] PTX 심볼(symbol — 변수, 레이블, 함수명 등) 객체의 전역 순차 고유 ID 카운터.
   * PTX 파서가 심볼을 인식하면 symbol_table에 등록하고 이 카운터로 uid를 부여한다.
   * uid=0을 "미등록/무효" 예약값으로 두기 위해 1부터 시작한다.
   * 설정자: symbol 생성자에서 gpgpu_context::symbol_sm_next_uid++로 획득.
   * 읽는 자: PTX IR 참조 해석(resolve), 심볼 테이블 룩업 디버그.
   * 값 범위: 1 이상 (0은 무효 예약).
   * 동기화: PTX 파싱 단계(단일 스레드)에서만 사용되므로 락 불필요. */

  unsigned function_info_sm_next_uid;
  /* [한국어] function_info(PTX 커널/디바이스 함수 정보) 객체의 전역 순차 고유 ID 카운터.
   * function_info는 PTX의 .entry(커널) 또는 .func(디바이스 함수) 정의를 표현하며,
   * 함수별 레지스터 수, 공유 메모리 크기, 지역 심볼 테이블을 포함한다.
   * uid=0을 "미할당" 예약값으로 두기 위해 1부터 시작한다.
   * 설정자: function_info 생성자에서 gpgpu_context::function_info_sm_next_uid++로 획득.
   * 읽는 자: 커널 함수 탐색(symbol_table::lookup), 커널 런치 경로.
   * 값 범위: 1 이상 (0은 미할당 예약).
   * 동기화: PTX 파싱 단계(단일 스레드)에서만 사용되므로 락 불필요. */

  std::vector<ptx_instruction *>
      s_g_pc_to_insn;  // a direct mapping from PC to instruction
  /* [한국어] PC(Program Counter) 값을 인덱스로 사용하여 ptx_instruction 포인터를 직접 조회하는 배열.
   * PTX가 로드되면 각 명령어에 순차적 PC 값이 할당되고, 이 벡터의 해당 인덱스에 포인터가 저장된다.
   * 기능 시뮬레이션에서 워프가 한 사이클에 인스트럭션을 fetch할 때 O(1)으로 명령어에 접근 가능하다.
   * 설정자: PTX 파서(ptx_recognizer)가 각 명령어 파싱 후 push_back()으로 추가.
   * 읽는 자: pc_to_instruction(unsigned pc) — 타이밍 모델의 fetch 단계에서 매 사이클 호출.
   *          ptx_fetch_inst(address_type pc) — 워프 스케줄러가 next PC의 명령어를 가져올 때.
   * 값 범위: 인덱스 = PC 값 (0부터 총 PTX 명령어 수-1까지), 포인터 = 유효한 ptx_instruction*.
   * 동기화: PTX 로드 단계(쓰기)와 시뮬레이션 단계(읽기)가 분리되어 있어 락 불필요. */

  bool debug_tensorcore;
  /* [한국어] Tensor Core 연산 디버그 출력 활성화 플래그.
   * true이면 Tensor Core(행렬 곱셈 가속 유닛) 관련 PTX 명령어 처리 시 상세 디버그 정보를 출력한다.
   * gpgpusim.config의 설정 옵션으로 제어 가능하며, 기본값은 false(0).
   * 설정자: 생성자에서 0으로 초기화, ptx_reg_options()를 통해 config 파싱 후 설정 가능.
   * 읽는 자: cuda-sim/instructions.cc의 Tensor Core 명령어 실행 핸들러.
   * 값 범위: false(0, 디버그 비활성) 또는 true(1, 디버그 활성).
   * 동기화: 시뮬레이션 시작 전 설정되고 이후 읽기 전용이므로 락 불필요. */

  // SST related
  bool requested_synchronize = false;
  /* [한국어] SST(Structural Simulation Toolkit) 코-시뮬레이션 환경에서 동기화 요청 플래그.
   * GPGPU-Sim이 SST 프레임워크와 연동하여 코-시뮬레이션할 때, 외부 SST 컴포넌트가
   * 시뮬레이션 동기화를 요청하면 이 플래그가 true로 설정된다.
   * synchronize_check()가 매 사이클 이 플래그를 검사하여 동기화 포인트에서 대기한다.
   * 설정자: SST 컴포넌트 또는 synchronize() 호출 시 true로 설정.
   * 읽는 자: synchronize_check()가 매 사이클 폴링하여 동기화 필요 여부 판단.
   * 값 범위: false(동기화 불필요) 또는 true(동기화 요청됨).
   * 동기화: SST와 GPGPU-Sim 스레드가 접근할 수 있으나, 현재 구현에서는 단순 bool 사용;
   *         멀티스레드 SST 환경에서는 atomic 또는 락이 필요할 수 있음. */

  // objects pointers for each file
  cuda_runtime_api *api;
  /* [한국어] CUDA 런타임 API 인터셉트 객체 포인터.
   * cuLaunchKernel, cuMemAlloc, cuMemcpy 등 CUDA Runtime API 호출을 실제 GPU 드라이버 대신
   * GPGPU-Sim 시뮬레이터로 리다이렉트하는 객체이다.
   * 설정자: gpgpu_context 생성자에서 new cuda_runtime_api(this)로 생성.
   * 읽는 자: libcuda/cuda_runtime_api.cc의 각 API 구현 함수, 커널 런치 경로.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: CUDA 애플리케이션 메인 스레드에서 호출되므로 단일 스레드 접근. */

  ptxinfo_data *ptxinfo;
  /* [한국어] PTX 파일의 .ptxinfo 메타데이터를 저장하는 객체 포인터.
   * nvcc는 PTX 컴파일 시 레지스터 수, 공유 메모리 크기 등의 메타데이터를 .ptxinfo 파일에 기록한다.
   * 이 객체는 그 .ptxinfo 파일을 파싱하여 각 커널의 자원 사용량 정보를 저장한다.
   * 이 정보는 SM 점유율(occupancy) 계산 및 자원 할당에 사용된다.
   * 설정자: 생성자에서 new ptxinfo_data(this)로 생성; gpgpu_ptx_info_load_from_filename()으로 채워짐.
   * 읽는 자: 커널 런치 경로에서 레지스터/공유 메모리 사용량 조회.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: PTX 로드(단일 스레드)에서만 쓰이고 이후 읽기 전용이므로 락 불필요. */

  ptx_recognizer *ptx_parser;
  /* [한국어] PTX 소스 텍스트를 파싱하는 Flex/Bison 파서 객체 포인터.
   * PTX(Parallel Thread eXecution) 가상 ISA를 파싱하여 ptx_instruction IR 노드를 생성하고,
   * s_g_pc_to_insn 테이블을 채우며, g_global_allfiles_symbol_table에 심볼을 등록한다.
   * 설정자: 생성자에서 new ptx_recognizer(this)로 생성.
   * 읽는 자: init_parser(), gpgpu_ptx_sim_load_ptx_from_string/filename()에서 파싱 호출.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: PTX 로드 단계(단일 스레드)에서만 사용되므로 락 불필요. */

  GPGPUsim_ctx *the_gpgpusim;
  /* [한국어] 타이밍 시뮬레이션 진입점 컨텍스트 객체 포인터.
   * gpgpu_sim 인스턴스(사이클-레벨 SM/캐시/DRAM/NoC 타이밍 모델)를 소유하며,
   * 시뮬레이션 스레드 시작(start_sim_thread), SST 동기화, 설정 파라미터 전달 등을 담당한다.
   * 설정자: 생성자에서 new GPGPUsim_ctx(this)로 생성.
   * 읽는 자: gpgpu_ptx_sim_init_perf(), start_sim_thread(), synchronize() 등.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: 시뮬레이션 스레드와 메인 스레드가 모두 접근할 수 있으며,
   *         GPGPUsim_ctx 내부에서 필요한 동기화(mutex/condition variable)를 처리. */

  cuda_sim *func_sim;
  /* [한국어] PTX 기능 시뮬레이터 객체 포인터.
   * 각 워프의 PTX 명령어를 fetch/decode/execute하여 레지스터 파일, 공유 메모리,
   * 전역 메모리 상태를 변경하는 기능(functional) 시뮬레이션을 담당한다.
   * 타이밍 모델(gpgpu_sim)이 기능 시뮬레이터에 명령어 실행을 위임하는 구조.
   * 설정자: 생성자에서 new cuda_sim(this)로 생성.
   * 읽는 자: shader.cc의 execute 스테이지, ptx_fetch_inst() 등.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: 사이클 루프 내 단일 시뮬레이션 스레드에서 호출되므로 락 불필요. */

  cuda_device_runtime *device_runtime;
  /* [한국어] GPU 디바이스 측 CUDA 런타임 서비스 에뮬레이터 포인터.
   * GPU 디바이스 함수 내에서 호출하는 cudaLaunchDevice() 등 동적 병렬성(Dynamic Parallelism)
   * 관련 런타임 서비스를 에뮬레이션한다. 호스트에서 커널이 또 다른 커널을 런치하는
   * 중첩 커널 실행(nested kernel launch)이 이 객체를 통해 처리된다.
   * 설정자: 생성자에서 new cuda_device_runtime(this)로 생성.
   * 읽는 자: cuda-sim/instructions.cc의 cudaLaunchDevice 명령어 핸들러.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: 기능 시뮬레이션 단계에서 호출되므로 단일 스레드 접근. */

  ptx_stats *stats;
  /* [한국어] PTX 명령어 실행 통계 수집 객체 포인터.
   * 명령어 타입별(산술, 메모리, 분기 등) 실행 횟수, 분기 발산(divergence) 통계,
   * 워프별 명령어 분포 등을 수집하여 시뮬레이션 종료 시 리포트한다.
   * AccelWattch 전력 모델의 동적 전력 추정 입력으로도 사용된다.
   * 설정자: 생성자에서 new ptx_stats(this)로 생성; 각 명령어 실행 시 업데이트됨.
   * 읽는 자: exit_simulation(), print_simulation_time()에서 통계 출력.
   * 값 범위: NULL 불가 (생성자에서 반드시 생성됨).
   * 동기화: 기능 시뮬레이션 단계에서 단일 스레드로 업데이트되므로 락 불필요. */

  // member function list
  /*
   * [한국어]
   * synchronize() - SST 코-시뮬레이션 동기화 포인트 진입
   *
   * @return: 없음
   *
   * GPGPU-Sim이 SST(Structural Simulation Toolkit) 프레임워크와 연동될 때,
   * 호스트 CUDA 스레드가 시뮬레이션 동기화를 기다리는 블로킹 포인트이다.
   * requested_synchronize 플래그를 true로 설정하고 SST 이벤트 처리를 기다린다.
   * 단독 실행 모드에서는 이 함수가 즉시 반환될 수 있다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, 커널 런치 또는 동기화 API 호출 시.
   *
   * 호출 체인:
   *   cudaDeviceSynchronize() → cuda_runtime_api → [synchronize()]
   *       → SST 이벤트 루프 대기 (코-시뮬레이션 모드)
   */
  void synchronize();

  /*
   * [한국어]
   * synchronize_check() - 동기화 요청 여부 폴링 확인
   *
   * @return: bool — true이면 동기화가 요청된 상태, false이면 계속 진행 가능
   *
   * 시뮬레이션 사이클 루프에서 매 사이클(또는 일정 주기로) 호출하여
   * SST로부터 동기화 요청이 들어왔는지 확인한다.
   * true 반환 시 시뮬레이션 루프는 synchronize()를 호출하여 대기한다.
   * 실행 컨텍스트: CPU 시뮬레이션 스레드, 사이클 루프 내부.
   *
   * 호출 체인:
   *   gpu-sim.cc cycle() 루프 → [synchronize_check()] → 조건부 synchronize()
   */
  bool synchronize_check();

  /*
   * [한국어]
   * exit_simulation() - 시뮬레이션 정상 종료 처리
   *
   * @return: 없음
   *
   * 모든 커널이 완료된 후 시뮬레이션을 정상 종료하는 함수이다.
   * 통계 출력(ptx_stats, 캐시/DRAM 통계), 로그 파일 플러시, 시뮬레이션 스레드 종료 등
   * 종료에 필요한 정리 작업을 순서대로 수행한다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, 마지막 커널 완료 후.
   *
   * 호출 체인:
   *   cuda_runtime_api → [exit_simulation()] → print_simulation_time()
   *       → 각 서브시스템 통계 출력 → 프로세스 종료
   */
  void exit_simulation();

  /*
   * [한국어]
   * print_simulation_time() - 시뮬레이션 소요 시간 출력
   *
   * @return: 없음
   *
   * 시뮬레이션 시작부터 종료까지의 벽시계 시간(wall-clock time)과
   * 시뮬레이션된 GPU 사이클 수를 출력한다. 성능 벤치마킹 및 시뮬레이션 속도 측정에 사용.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, exit_simulation() 내부에서 호출.
   *
   * 호출 체인:
   *   exit_simulation() → [print_simulation_time()] → 시간 정보 출력
   */
  void print_simulation_time();

  /*
   * [한국어]
   * gpgpu_opencl_ptx_sim_main_perf() - OpenCL 커널의 타이밍 시뮬레이션 메인 루프
   *
   * @param grid: kernel_info_t* — 런치된 OpenCL 커널의 그리드/블록 크기 및 함수 정보를 담은 객체
   * @return: int — 0(성공) 또는 오류 코드
   *
   * OpenCL 커널을 GPGPU-Sim의 타이밍 모델로 시뮬레이션하는 메인 진입점이다.
   * CUDA 런타임과 달리 OpenCL은 별도의 커널 런치 경로를 사용하며, 이 함수가 그 역할을 담당한다.
   * 내부적으로 gpgpu_sim의 사이클 루프를 구동하여 커널이 완료될 때까지 반복한다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, OpenCL clEnqueueNDRangeKernel() 호출 후.
   *
   * 호출 체인:
   *   libopencl clEnqueueNDRangeKernel() → [gpgpu_opencl_ptx_sim_main_perf(grid)]
   *       → gpgpu_sim::cycle() 루프 → 커널 완료
   */
  int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid);

  /*
   * [한국어]
   * cuobjdumpParseBinary() - CUDA fat binary를 cuobjdump로 분해하여 PTX/ELF 추출
   *
   * @param handle: unsigned int — CUDA fat binary 핸들 (cuModuleLoad 등으로 등록된 식별자)
   * @return: 없음
   *
   * 지정된 핸들에 해당하는 CUDA fat binary를 cuobjdump 도구로 분해한다.
   * cuobjdump를 서브프로세스로 실행하여 출력을 Flex/Bison 파서로 파싱하고,
   * cuobjdumpPTXSection / cuobjdumpELFSection 객체들을 생성하여 리스트로 관리한다.
   * 이후 gpgpu_ptx_sim_load_ptx_from_filename() 등을 통해 PTX를 기능 시뮬레이터에 로드한다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, cuModuleLoad / cuLaunchKernel 초기화 시.
   *
   * 호출 체인:
   *   cuda_runtime_api::cuModuleLoad() → [cuobjdumpParseBinary(handle)]
   *       → Flex/Bison 파서 → cuobjdumpPTXSection 생성 → gpgpu_ptx_sim_load_ptx_from_filename()
   */
  void cuobjdumpParseBinary(unsigned int handle);

  /*
   * [한국어]
   * gpgpu_ptx_sim_load_ptx_from_string() - 메모리 내 PTX 문자열을 파싱하여 IR 구성
   *
   * @param p: const char* — 파싱할 PTX 소스 텍스트 (null-terminated C 문자열)
   * @param source_num: unsigned — 소스 식별 번호 (여러 PTX 소스 구별용)
   * @return: class symbol_table* — 파싱된 PTX의 심볼 테이블 (함수, 변수 심볼 포함)
   *
   * 문자열로 전달된 PTX 소스를 직접 Flex/Bison 파서에 입력하여 IR을 구성한다.
   * 파일 경로 없이 메모리에서 직접 파싱하므로, 런타임에 PTX가 생성되는 경우에 사용된다.
   * 반환된 symbol_table을 통해 커널 함수(function_info)를 조회하여 시뮬레이션에 사용한다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, CUDA 초기화 중 단 한 번.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() 또는 cuModuleLoadData() → [gpgpu_ptx_sim_load_ptx_from_string()]
   *       → init_parser() → ptx_recognizer → symbol_table 반환
   */
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                         unsigned source_num);

  /*
   * [한국어]
   * gpgpu_ptx_sim_load_ptx_from_filename() - 파일 경로의 PTX를 파싱하여 IR 구성
   *
   * @param filename: const char* — 파싱할 PTX 파일의 경로 (null-terminated C 문자열)
   * @return: class symbol_table* — 파싱된 PTX의 심볼 테이블
   *
   * 지정된 파일 경로의 PTX 텍스트 파일을 열어 Flex/Bison 파서에 입력하여 IR을 구성한다.
   * cuobjdumpParseBinary()가 추출한 PTX 임시 파일을 입력으로 사용하는 것이 주요 경로이다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, CUDA 모듈 로드 초기화 중.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → cuobjdumpPTXSection::getPTXfilename()
   *       → [gpgpu_ptx_sim_load_ptx_from_filename()] → init_parser() → symbol_table 반환
   */
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename(
      const char *filename);

  /*
   * [한국어]
   * gpgpu_ptx_info_load_from_filename() - PTX info 파일 로드 및 파싱
   *
   * @param filename: const char* — .ptxinfo 파일의 경로
   * @param sm_version: unsigned — 대상 SM 아키텍처 버전 (예: 60 = sm_60)
   * @return: 없음
   *
   * nvcc가 생성한 .ptxinfo 파일을 파싱하여 각 커널의 레지스터 수, 공유 메모리 크기,
   * local 메모리 크기 등의 자원 메타데이터를 ptxinfo_data 객체에 저장한다.
   * 이 정보는 SM 점유율 계산 및 CTA 스케줄링에 사용된다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, PTX 로드 직후 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [gpgpu_ptx_info_load_from_filename(filename, sm_version)]
   *       → ptxinfo_data 파싱 → 커널 자원 정보 저장
   */
  void gpgpu_ptx_info_load_from_filename(const char *filename,
                                         unsigned sm_version);

  /*
   * [한국어]
   * gpgpu_ptxinfo_load_from_string() - 메모리 내 PTX info 문자열 파싱
   *
   * @param p_for_info: const char* — .ptxinfo 내용을 담은 문자열
   * @param source_num: unsigned — 소스 식별 번호
   * @param sm_version: unsigned — 대상 SM 아키텍처 버전 (기본값: 20 = sm_20 = Fermi)
   * @param no_of_ptx: int — 처리할 PTX 소스 수 (기본값: 0)
   * @return: 없음
   *
   * 파일 경로 대신 메모리 문자열로 .ptxinfo 내용을 직접 파싱한다.
   * cuModuleLoadData() 등 파일 없이 인메모리 PTX를 로드하는 경로에서 사용된다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, 인메모리 PTX 로드 시.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_load_ptx_from_string() → [gpgpu_ptxinfo_load_from_string()]
   *       → ptxinfo_data 파싱 → 커널 자원 정보 저장
   */
  void gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                      unsigned source_num,
                                      unsigned sm_version = 20,
                                      int no_of_ptx = 0);

  /*
   * [한국어]
   * print_ptx_file() - PTX 소스 내용을 번호를 붙여 출력
   *
   * @param p: const char* — 출력할 PTX 소스 텍스트
   * @param source_num: unsigned — 소스 식별 번호 (여러 PTX 파일 구별)
   * @param filename: const char* — 출력에 포함할 원본 파일명 (표시 목적)
   * @return: 없음
   *
   * 디버그 목적으로 파싱 전 PTX 소스를 줄 번호와 함께 표준 출력으로 덤프한다.
   * GPGPU-Sim의 PTX 파싱 오류 진단 시 소스 위치 추적에 도움을 준다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, PTX 로드 디버그 경로에서 조건부 호출.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_load_ptx_from_string() (디버그 모드) → [print_ptx_file()]
   */
  void print_ptx_file(const char *p, unsigned source_num, const char *filename);

  /*
   * [한국어]
   * init_parser() - PTX 파서 초기화 및 파싱 실행
   *
   * @param filename_or_string: const char* — 파싱할 PTX 파일 경로 또는 소스 텍스트
   * @return: class symbol_table* — 파싱 결과로 구성된 심볼 테이블
   *
   * ptx_recognizer를 초기화하고 PTX 소스를 파싱하여 ptx_instruction IR 노드를 생성한다.
   * 파싱 결과는 s_g_pc_to_insn 테이블에 기록되고, 반환된 symbol_table로 커널 함수를 조회한다.
   * 내부적으로 Flex 스캐너와 Bison 파서를 구동하며, g_global_allfiles_symbol_table을 갱신한다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, gpgpu_ptx_sim_load_ptx_from_*() 내부에서 호출.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_load_ptx_from_string/filename() → [init_parser()]
   *       → Flex/Bison PTX 파서 → s_g_pc_to_insn 구성 → symbol_table 반환
   */
  class symbol_table *init_parser(const char *);

  /*
   * [한국어]
   * gpgpu_ptx_sim_init_perf() - 타이밍 시뮬레이터(gpgpu_sim) 초기화
   *
   * @return: class gpgpu_sim* — 초기화된 타이밍 시뮬레이터 객체 포인터
   *
   * gpgpusim.config를 파싱하여 타이밍 모델의 설정 파라미터(SM 수, 캐시 크기,
   * DRAM 타이밍 등)를 로드하고, gpgpu_sim 인스턴스를 생성 및 초기화한다.
   * 이 함수 호출 후에야 타이밍 시뮬레이션 루프를 구동할 수 있다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, 첫 번째 커널 런치 직전.
   *
   * 호출 체인:
   *   cuda_runtime_api::cuLaunchKernel() (첫 번째 런치) → [gpgpu_ptx_sim_init_perf()]
   *       → gpgpusim.config 파싱 → new gpgpu_sim() → gpgpu_sim* 반환
   */
  class gpgpu_sim *gpgpu_ptx_sim_init_perf();

  /*
   * [한국어]
   * start_sim_thread() - 타이밍 시뮬레이션 스레드 시작
   *
   * @param api: int — 사용할 API 종류 (0: CUDA, 1: OpenCL)
   * @return: 없음
   *
   * 별도의 POSIX 스레드를 생성하여 gpgpu_sim의 사이클-레벨 타이밍 루프를 비동기적으로 실행한다.
   * 이후 CUDA 애플리케이션 스레드와 시뮬레이션 스레드가 동기화하며 커널을 실행한다.
   * SST 코-시뮬레이션 모드에서는 SST 이벤트 루프가 시뮬레이션 스레드 역할을 대신할 수 있다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드에서 호출, pthread_create()로 시뮬레이션 스레드 생성.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_init_perf() 이후 → [start_sim_thread(api)]
   *       → pthread_create() → gpgpusim_simulation_thread() 비동기 실행
   */
  void start_sim_thread(int api);

  /*
   * [한국어]
   * GPGPUSim_Init() - GPGPU-Sim 전체 초기화 및 가상 GPU 디바이스 등록
   *
   * @return: struct _cuda_device_id* — 시뮬레이터가 나타내는 가상 GPU 디바이스 식별자 포인터
   *
   * GPGPU-Sim의 최초 초기화를 수행한다. gpgpusim.config 파싱, 가상 GPU 디바이스 생성,
   * CUDA 드라이버 초기화 완료 등 시뮬레이션 시작에 필요한 모든 준비를 한 번에 수행한다.
   * 반환된 _cuda_device_id는 이후 cuDeviceGet() 등 디바이스 쿼리 API에서 사용된다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, cuInit(0) 또는 첫 번째 CUDA API 호출 시.
   *
   * 호출 체인:
   *   cuda_runtime_api::cuInit() → [GPGPUSim_Init()]
   *       → gpgpusim.config 파싱 → 가상 디바이스 생성 → _cuda_device_id* 반환
   */
  struct _cuda_device_id *GPGPUSim_Init();

  /*
   * [한국어]
   * ptx_reg_options() - PTX 관련 옵션을 옵션 파서에 등록
   *
   * @param opp: option_parser_t — gpgpusim.config 파싱에 사용하는 옵션 파서 객체
   * @return: 없음
   *
   * PTX 시뮬레이션 관련 설정 옵션(debug_tensorcore 등)을 옵션 파서에 등록하여
   * gpgpusim.config 파일에서 해당 옵션 값을 파싱할 수 있도록 한다.
   * option_parser_t는 option_parser.h에 정의된 C 인터페이스 타입이다.
   * 실행 컨텍스트: CPU 호스트 메인 스레드, GPGPUSim_Init() 내부 설정 파싱 단계.
   *
   * 호출 체인:
   *   GPGPUSim_Init() → option_parser_t 생성 → [ptx_reg_options(opp)]
   *       → option_parser_register() 호출들 → config 파싱 → debug_tensorcore 등 설정
   */
  void ptx_reg_options(option_parser_t opp);

  /*
   * [한국어]
   * pc_to_instruction() - PC 값을 ptx_instruction 포인터로 변환
   *
   * @param pc: unsigned — 조회할 PTX 프로그램 카운터 값
   * @return: const ptx_instruction* — pc에 해당하는 PTX 명령어 IR 노드 포인터,
   *          pc가 범위를 벗어나면 NULL
   *
   * s_g_pc_to_insn 배열을 직접 인덱싱하여 O(1)으로 PTX 명령어를 조회한다.
   * 타이밍 모델의 fetch 스테이지, ptx_fetch_inst(), 디버그 추적 등에서 매 사이클 호출된다.
   * 실행 컨텍스트: CPU 호스트 시뮬레이션 스레드, 사이클 루프 내 매 인스트럭션 fetch 시.
   *
   * 호출 체인:
   *   shader.cc fetch 스테이지 / ptx_fetch_inst() → [pc_to_instruction(pc)]
   *       → s_g_pc_to_insn[pc] → ptx_instruction* 반환
   */
  const ptx_instruction *pc_to_instruction(unsigned pc);

  /*
   * [한국어]
   * ptx_fetch_inst() - address_type PC로 warp_inst_t 형식의 명령어 조회
   *
   * @param pc: address_type — 조회할 PTX 프로그램 카운터 값 (address_type = unsigned long long)
   * @return: const warp_inst_t* — 해당 PC의 명령어를 warp_inst_t 형식으로 반환,
   *          없으면 NULL
   *
   * 타이밍 모델(shader.cc)의 워프 스케줄러가 다음 실행할 인스트럭션을 fetch할 때 호출한다.
   * 내부적으로 pc_to_instruction()을 사용하여 ptx_instruction을 조회하고,
   * 이를 타이밍 모델이 이해하는 warp_inst_t 형식으로 변환하여 반환한다.
   * 실행 컨텍스트: CPU 호스트 시뮬레이션 스레드, 사이클 루프 내 워프 fetch 단계.
   *
   * 호출 체인:
   *   shader.cc warp_scheduler::cycle() → [ptx_fetch_inst(pc)]
   *       → pc_to_instruction(pc) → ptx_instruction → warp_inst_t* 반환
   */
  const warp_inst_t *ptx_fetch_inst(address_type pc);

  /*
   * [한국어]
   * translate_pc_to_ptxlineno() - PC 값을 원본 PTX 소스 라인 번호로 변환
   *
   * @param pc: unsigned — 변환할 PTX 프로그램 카운터 값
   * @return: unsigned — 해당 PC에 대응하는 원본 PTX 소스 파일의 라인 번호,
   *          매핑 불가 시 0
   *
   * 시뮬레이션 오류 보고, 디버그 추적, ptxas 정보 출력 시 PC 값을 사람이 읽을 수 있는
   * 소스 라인 번호로 변환하는 데 사용된다.
   * ptx_instruction이 파싱 시 저장한 소스 위치 정보(line number)를 역방향 조회한다.
   * 실행 컨텍스트: CPU 호스트 시뮬레이션 스레드, 오류 보고 또는 디버그 덤프 시.
   *
   * 호출 체인:
   *   디버그/오류 핸들러 → [translate_pc_to_ptxlineno(pc)]
   *       → pc_to_instruction(pc) → ptx_instruction::source_line() → 라인 번호 반환
   */
  unsigned translate_pc_to_ptxlineno(unsigned pc);
};

/*
 * [한국어]
 * GPGPU_Context() - gpgpu_context 싱글턴 접근 함수
 *
 * @return: gpgpu_context* — 전역 단일 gpgpu_context 인스턴스 포인터
 *
 * GPGPU-Sim 전체에서 단 하나의 gpgpu_context 인스턴스가 존재하며,
 * 이 함수를 통해 어디서든 접근한다. 첫 번째 호출 시 gpgpu_context 객체를 힙에 생성하고
 * 이후 호출에서는 동일한 포인터를 반환한다. 정적 지역 변수 패턴이나 전역 포인터로 구현된다.
 * 모든 libcuda 함수, cuda-sim 함수, gpu-sim 함수가 이 함수를 통해 시뮬레이터 상태에 접근한다.
 * 실행 컨텍스트: CPU 호스트 유저스페이스, 시뮬레이션 전 기간 동안 임의 시점에 호출 가능.
 *
 * 호출 체인:
 *   모든 CUDA API 인터셉트 함수 → [GPGPU_Context()] → gpgpu_context 인스턴스 반환
 *       → api / func_sim / the_gpgpusim 등 서브시스템 접근
 */
gpgpu_context *GPGPU_Context();

#endif /* __gpgpu_context_h__ */
