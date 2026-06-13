/*
 * [한국어 설명] PTX opcode 및 특수 레지스터 열거형 정의 (opcodes.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이션 레이어(cuda-sim/)에서 사용하는
 * PTX 명령어 opcode 목록(`opcode_t`), PTX 특수 레지스터 식별자(`special_regs`),
 * Tensor Core WMMA 연산 타입(`wmma_type`)을 열거형으로 정의한다.
 * 직접 열거값을 나열하는 대신 X-매크로(X-macro) 패턴을 사용하여
 * `opcodes.def` 파일에 선언된 opcode 목록을 `#include`로 끌어들임으로써
 * enum 정의와 문자열 테이블 등 여러 용도에 동일 목록을 재사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim은 크게 기능 시뮬레이션(cuda-sim/)과 타이밍 시뮬레이션(gpgpu-sim/)
 * 두 레이어로 구성된다. 이 파일은 기능 시뮬레이션 레이어에 속하며,
 * PTX 파서(ptx.l / ptx.y)가 소스를 파싱한 뒤 생성하는 PTX IR(`ptx_ir`)에서
 * 각 명령어의 opcode를 `opcode_t` 값으로 저장하는 데 사용된다.
 * 타이밍 모델(shader.cc)은 이 opcode를 바탕으로 명령어 지연·자원 요구를 결정한다.
 * 실행 컨텍스트: 호스트 유저스페이스 시뮬레이터 (GPU 디바이스가 아님).
 *
 * 호출 체인:
 *   PTX 파서(ptx.y) → ptx_ir(명령어 객체 생성 시 opcode_t 저장)
 *   → instructions.cc(ptx_instruction::execute) → shader.cc(타이밍 모델)
 *
 * === 타 모듈과의 연결 ===
 * 의존 (이 파일이 사용하는):
 *   - opcodes.def: X-매크로 방식으로 opcode 목록을 제공하는 데이터 파일.
 *     각 행은 `OP_DEF(이름, 실행함수, 문자열, DST플래그, 분류)` 형태이며,
 *     여기서는 OP 이름만 enum 값으로 추출한다.
 *
 * 의존받음 (이 파일에 의존하는):
 *   - cuda-sim/ptx_ir.h : ptx_instruction이 opcode_t 필드를 보유함.
 *   - cuda-sim/instructions.cc : execute() 분기에서 opcode_t를 switch/if로 사용.
 *   - cuda-sim/ptx.y : 파서 액션에서 opcode 문자열을 opcode_t 값으로 변환.
 *   - gpgpu-sim/shader.cc : 타이밍 모델이 opcode_t로 명령어 클래스(정수/부동소수/메모리)를 판단.
 *   - special_regs는 ptx.y / ptx_ir.cc에서 %tid, %ctaid 등 특수 레지스터를
 *     파싱할 때 식별자로 사용된다.
 *   - wmma_type은 Tensor Core wmma 명령어(wmma.load / wmma.store / wmma.mma)를
 *     처리하는 instructions.cc와 ptx_ir.cc에서 사용된다.
 *
 * 데이터 흐름:
 *   CUDA 소스 → NVCC → PTX 텍스트 → ptx 파서 → ptx_instruction(opcode_t 저장)
 *   → instructions.cc execute() → 결과(레지스터 파일/메모리 갱신)
 *
 * === 주요 함수/구조체 요약 ===
 * enum opcode_t   : opcodes.def를 X-매크로로 확장한 PTX 전체 opcode 열거형.
 *                   NUM_OPCODES는 마지막 센티널 값으로 opcode 수 계산에 사용.
 * enum special_regs: PTX ISA가 정의하는 읽기 전용 특수 레지스터(%tid, %smid 등)
 *                   를 시뮬레이터 내부 정수 ID로 매핑한 열거형.
 * enum wmma_type  : Tensor Core WMMA 명령어의 세부 동작 종류(행렬 로드/저장/연산,
 *                   레이아웃, 타일 크기)를 나타내는 열거형.
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda
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

/* [한국어] 인클루드 가드 시작 — 이 헤더가 여러 번 포함되어도 중복 정의를 방지한다.
 * cuda-sim/, gpgpu-sim/ 양쪽에서 모두 include되므로 가드가 필수적이다. */
#ifndef opcodes_h_included
#define opcodes_h_included /* [한국어] 가드 매크로 정의 — 이후 재포함 시 파일 내용을 건너뜀 */

/* [한국어] opcode_t — PTX 전체 명령어 opcode를 정수 열거형으로 정의한다.
 *
 * X-매크로(X-macro) 패턴을 사용한다:
 *   1. OP_DEF / OP_W_DEF 매크로를 "OP 이름만 꺼내는 형태"로 #define한다.
 *   2. opcodes.def를 #include하면 파일 안의 OP_DEF(...) 행들이 모두 "OP,"로 치환된다.
 *   3. #undef로 매크로를 해제하여 다른 헤더에서 다른 목적으로 같은 .def를 재사용할 수 있게 한다.
 *
 * 이 패턴 덕분에 opcodes.def 하나를 수정하면 enum, 문자열 배열, 함수 포인터 테이블이
 * 자동으로 동기화된다 (DRY 원칙).
 *
 * 사용처:
 *   - ptx_ir.h의 ptx_instruction::m_opcode 필드 타입으로 사용됨.
 *   - instructions.cc의 execute() switch 분기 레이블로 사용됨.
 *   - shader.cc에서 명령어를 정수/부동소수/메모리/제어 클래스로 분류하는 데 사용됨.
 *
 * 값 범위: 0부터 NUM_OPCODES-1까지. NUM_OPCODES는 opcodes.def의 행 수와 같다. */
enum opcode_t {
/* [한국어] OP_DEF 매크로 재정의: 5개 인자 중 첫 번째(OP 이름)만 enum 값으로 추출하고 나머지는 버린다.
 * opcodes.def 안의 각 OP_DEF(ADD, execute_add, "add", DST, INT_OP) 행이
 * 전처리 후 "ADD,"로 치환되어 enum 값이 된다. */
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
/* [한국어] OP_W_DEF 매크로 재정의: OP_DEF와 동일한 방식으로 처리한다.
 * W는 "wide" 또는 "write-destination" 변형 명령어를 의미하며,
 * FUNC·STR·DST·CLASSIFICATION 인자는 enum 생성 시 불필요하므로 버린다. */
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
/* [한국어] opcodes.def 파일을 현재 위치에 텍스트로 삽입한다.
 * 이 시점에 OP_DEF / OP_W_DEF가 위에서 재정의된 상태이므로,
 * 파일 안의 모든 OP_DEF(...), OP_W_DEF(...) 호출이 "OP이름,"으로 펼쳐져
 * enum 값 목록을 자동으로 생성한다. */
#include "opcodes.def"
  NUM_OPCODES
  /* [한국어] NUM_OPCODES — opcode 목록의 끝을 나타내는 센티널(sentinel) 값.
   * opcodes.def의 마지막 OP 뒤에 자동으로 배치되어 전체 opcode 수를 나타낸다.
   * 설정자: 직접 설정되지 않음 — 컴파일러가 enum 규칙에 따라 자동 계산.
   * 읽는 자: 배열/테이블 크기 선언(예: opcode_str[NUM_OPCODES])이나
   *           경계 검사(op < NUM_OPCODES)에서 사용된다.
   * 값 범위: opcodes.def에 정의된 opcode 수와 동일한 양의 정수. */
#undef OP_DEF
/* [한국어] OP_DEF 매크로 해제 — enum 생성 목적의 정의를 제거한다.
 * 이후 다른 헤더(예: instructions.cc)에서 OP_DEF를 함수 포인터 테이블 등
 * 다른 목적으로 재정의해 opcodes.def를 재사용할 수 있도록 깨끗하게 비운다. */
#undef OP_W_DEF
/* [한국어] OP_W_DEF 매크로 해제 — OP_DEF와 같은 이유로 정리한다. */
};

/* [한국어] special_regs — PTX ISA가 정의하는 읽기 전용 특수 레지스터를
 * 시뮬레이터 내부 정수 식별자로 매핑한 열거형이다.
 *
 * PTX 소스에서 %tid, %smid 등의 이름으로 등장하는 레지스터들은 실제 범용
 * 레지스터 파일에 없고, 하드웨어가 런타임에 스레드/블록/SM 정보를 자동으로
 * 제공한다. GPGPU-Sim에서는 ptx 파서가 이 이름들을 special_regs 값으로
 * 변환하고, instructions.cc의 read_operand()에서 이 열거형을 switch/case로
 * 분기하여 시뮬레이터의 대응 상태 변수(tid, ctaid, sm_id 등)를 반환한다.
 *
 * 설정자: ptx.y 파서 액션 — 특수 레지스터 이름 토큰을 이 enum 값으로 변환.
 * 읽는 자: instructions.cc의 get_special_register_value() 또는 동등 함수 —
 *           operand fetch 시 대응 시뮬레이터 상태를 레지스터값으로 주입.
 * 동기화: 각 warp/스레드가 독립적으로 자신의 ID를 읽으므로 공유 상태 없음. */
enum special_regs {
  CLOCK_REG,
  /* [한국어] %clock (PTX ISA §9.2): 32비트 SM 클럭 사이클 카운터.
   * 설정자: ptx.y가 "%clock" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc가 mov/cvt 실행 시 현재 시뮬레이션 사이클 수(하위 32비트)를 반환.
   * 값 범위: 0 ~ 2^32-1 (32비트 랩어라운드).
   * 사용 목적: 커널 내 micro-benchmark 타이밍, PTX 레이턴시 측정. */

  HALFCLOCK_ID,
  /* [한국어] %halfclock: SM 클럭의 절반 주파수(1/2 클럭)로 증가하는 카운터.
   * 일부 NVIDIA 아키텍처에서 명령어 처리율을 반주파수로 측정하기 위해 존재한다.
   * 설정자: ptx.y가 "%halfclock" 또는 관련 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 사이클 값을 2로 나누어 반환.
   * 값 범위: 0 ~ 2^31 (CLOCK_REG의 절반 속도로 증가).
   * 동기화: 읽기 전용 — 여러 스레드가 동시에 읽어도 일관된 값 제공. */

  CLOCK64_REG,
  /* [한국어] %clock64 (PTX ISA §9.2): 64비트 SM 클럭 사이클 카운터.
   * CLOCK_REG의 64비트 확장 버전으로 랩어라운드 없이 긴 커널 실행을 측정한다.
   * 설정자: ptx.y가 "%clock64" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 전체 시뮬레이션 사이클 수를 64비트로 반환.
   * 값 범위: 0 ~ 2^64-1.
   * 사용 목적: 수백만 사이클 이상의 장시간 커널 정밀 타이밍 측정. */

  CTAID_REG,
  /* [한국어] %ctaid (PTX ISA §9.2): 그리드 내 현재 블록(CTA)의 3차원 ID (x/y/z).
   * CTA(Cooperative Thread Array)는 CUDA의 thread block에 해당한다.
   * 설정자: ptx.y가 "%ctaid.x" / "%ctaid.y" / "%ctaid.z" 토큰을 이 값으로 변환;
   *         x/y/z 차원은 operand suffix로 구분.
   * 읽는 자: instructions.cc — 현재 warp가 속한 thread block의 grid 좌표를 반환.
   * 값 범위: 0 ~ nctaid.{x,y,z}-1 (blockIdx.{x,y,z}에 해당).
   * 동기화: 블록 단위 상수 — 커널 실행 중 변하지 않음. */

  ENVREG_REG,
  /* [한국어] %envreg<n> (PTX ISA §9.2): 환경 레지스터 0~31번.
   * 드라이버 또는 런타임이 커널 실행 전에 설정하는 읽기 전용 32비트 레지스터.
   * 커널이 실행 환경 파라미터(예: 공유 메모리 크기, 동적 파라미터)를 읽는 데 사용.
   * 설정자: ptx.y가 "%envreg<n>" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 시뮬레이터 환경 레지스터 배열에서 해당 값 반환.
   * 값 범위: n은 0~31; 각 레지스터는 32비트 정수.
   * 동기화: 커널 실행 중 읽기 전용이므로 별도 락 불필요. */

  GRIDID_REG,
  /* [한국어] %gridid (PTX ISA §9.2): 현재 커널 그리드(kernel launch)의 고유 ID.
   * 여러 커널이 순차 또는 동시 실행될 때 각 launch를 구별하기 위한 정수 ID.
   * 설정자: ptx.y가 "%gridid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 실행 중인 kernel_info_t의 grid ID를 반환.
   * 값 범위: 1 이상의 양의 정수; 동시 kernel 지원 시 서로 다른 값.
   * 사용 목적: dynamic parallelism이나 multi-kernel 환경에서 커널 추적. */

  LANEID_REG,
  /* [한국어] %laneid (PTX ISA §9.2): warp 내 현재 스레드의 lane 번호 (0~31).
   * warp는 32개 스레드(lane)로 구성되며, laneid는 warp 내 스레드의 위치를 나타낸다.
   * 설정자: ptx.y가 "%laneid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 실행 중인 스레드의 warp 내 인덱스를 반환.
   * 값 범위: 0 ~ 31 (warp 크기 = 32 고정).
   * 사용 목적: warp-level primitive(vote, ballot, shfl 등)에서 스레드 위치 파악. */

  LANEMASK_EQ_REG,
  /* [한국어] %lanemask_eq (PTX ISA §9.2): 현재 laneid와 같은 비트만 1인 32비트 마스크.
   * 예: laneid=5 이면 lanemask_eq = 0b00000000000000000000000000100000 (비트 5만 1).
   * 설정자: ptx.y가 "%lanemask_eq" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — (1 << laneid) 값을 32비트 정수로 반환.
   * 사용 목적: warp-level 선거/투표 연산에서 자신의 lane 비트를 고립시키는 마스크. */

  LANEMASK_LE_REG,
  /* [한국어] %lanemask_le (PTX ISA §9.2): laneid 이하의 모든 lane이 1인 32비트 마스크.
   * 예: laneid=5 이면 bits 0~5가 1 → 0b00000000000000000000000000111111.
   * 설정자: ptx.y가 "%lanemask_le" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — (1 << (laneid+1)) - 1 값을 반환.
   * 사용 목적: prefix-sum, reduce, exclusive scan 등 warp 내 집합 연산. */

  LANEMASK_LT_REG,
  /* [한국어] %lanemask_lt (PTX ISA §9.2): laneid 미만(strictly less)의 lane이 1인 마스크.
   * 예: laneid=5 이면 bits 0~4가 1 → 0b00000000000000000000000000011111.
   * 설정자: ptx.y가 "%lanemask_lt" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — (1 << laneid) - 1 값을 반환.
   * 사용 목적: warp-level 배타적 스캔(exclusive prefix) 구현에서 자신보다 앞선 lane 집합 파악. */

  LANEMASK_GE_REG,
  /* [한국어] %lanemask_ge (PTX ISA §9.2): laneid 이상의 lane이 1인 32비트 마스크.
   * 예: laneid=5 이면 bits 5~31이 1 → 0b11111111111111111111111111100000.
   * 설정자: ptx.y가 "%lanemask_ge" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — ~((1 << laneid) - 1) 값을 32비트 범위에서 반환.
   * 사용 목적: 자신과 그 이후 lane들을 대상으로 하는 warp-level 연산. */

  LANEMASK_GT_REG,
  /* [한국어] %lanemask_gt (PTX ISA §9.2): laneid 초과(strictly greater)의 lane이 1인 마스크.
   * 예: laneid=5 이면 bits 6~31이 1 → 0b11111111111111111111111111000000.
   * 설정자: ptx.y가 "%lanemask_gt" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — ~((1 << (laneid+1)) - 1) 값을 32비트 범위에서 반환.
   * 사용 목적: 자신 이후의 lane만 선택하는 warp-level 연산 또는 reduction에서 다음 단계 파악. */

  NCTAID_REG,
  /* [한국어] %nctaid (PTX ISA §9.2): 그리드의 3차원 블록(CTA) 수 (x/y/z).
   * CUDA의 gridDim.{x,y,z}에 해당하며, 그리드 전체에서 블록의 총 개수를 나타낸다.
   * 설정자: ptx.y가 "%nctaid.x" / "%nctaid.y" / "%nctaid.z" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 실행 중인 kernel_info_t의 grid 차원 값을 반환.
   * 값 범위: 1 이상의 양의 정수; CUDA에서 x차원 최대 2^31-1, y/z 최대 65535.
   * 동기화: 커널 실행 중 변하지 않는 상수 — 락 불필요. */

  NTID_REG,
  /* [한국어] %ntid (PTX ISA §9.2): 블록 내 스레드 수 (x/y/z 차원).
   * CUDA의 blockDim.{x,y,z}에 해당하며, 블록 크기를 나타낸다.
   * 설정자: ptx.y가 "%ntid.x" / "%ntid.y" / "%ntid.z" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 블록의 스레드 차원 값을 반환.
   * 값 범위: x*y*z ≤ 1024 (Fermi 이후 GPU 제약); x 최대 1024, y/z 최대 1024.
   * 동기화: 커널 실행 중 변하지 않는 상수. */

  NSMID_REG,
  /* [한국어] %nsmid (PTX ISA §9.2): 디바이스의 SM 총 수.
   * 커널이 하드웨어에 존재하는 SM 개수를 런타임에 조회할 때 사용한다.
   * 설정자: ptx.y가 "%nsmid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 시뮬레이터 gpgpu_t의 SM 수 설정값을 반환.
   * 값 범위: 1 이상의 양의 정수; `gpgpusim.config`의 -gpgpu_n_cores 파라미터와 대응.
   * 사용 목적: work distribution 알고리즘이나 SW-managed load balancing에서 SM 수 파악. */

  NWARPID_REG,
  /* [한국어] %nwarpid (PTX ISA §9.2): SM이 동시에 보유할 수 있는 최대 warp 수.
   * 하드웨어의 warp 슬롯 수로, SM의 다중 warp 자원을 나타낸다.
   * 설정자: ptx.y가 "%nwarpid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — gpgpu_t의 max_warp_per_shader 설정값을 반환.
   * 값 범위: 아키텍처에 따라 다름 (Fermi: 48, Kepler: 64 warp/SM).
   * 사용 목적: 런타임 점유율(occupancy) 계산 또는 동기화 전략 결정. */

  PM_REG,
  /* [한국어] %pm<n> (PTX ISA §9.2): 하드웨어 성능 모니터 카운터 레지스터 (n=0~7).
   * 실제 GPU에서는 특정 이벤트(캐시 미스, 명령어 수 등)를 누적하는 하드웨어 카운터이다.
   * 설정자: ptx.y가 "%pm0" ~ "%pm7" 토큰을 이 값으로 변환; 실제 n값은 operand에서 구분.
   * 읽는 자: instructions.cc — 시뮬레이터 성능 카운터 배열(pm_counters)에서 해당 값 반환.
   * 값 범위: 32비트 부호 없는 정수 (0 ~ 2^32-1).
   * 사용 목적: GPU 프로파일링 도구(nvprof 등)의 소프트웨어 에뮬레이션 지원. */

  SMID_REG,
  /* [한국어] %smid (PTX ISA §9.2): 현재 블록이 실행 중인 SM(Streaming Multiprocessor)의 ID.
   * 같은 그리드의 블록들이 서로 다른 SM에 분배되므로 각 블록마다 다른 값을 갖는다.
   * 설정자: ptx.y가 "%smid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 shader_core_ctx(SM)의 m_sid 필드를 반환.
   * 값 범위: 0 ~ (SM 수 - 1); `gpgpusim.config`의 -gpgpu_n_cores - 1이 최대값.
   * 사용 목적: 블록의 물리적 위치 파악; per-SM 통계 수집; 캐시/메모리 지역성 분석. */

  TID_REG,
  /* [한국어] %tid (PTX ISA §9.2): 블록 내 현재 스레드의 3차원 ID (x/y/z).
   * CUDA의 threadIdx.{x,y,z}에 해당하며, 블록 내 스레드의 위치를 나타낸다.
   * 설정자: ptx.y가 "%tid.x" / "%tid.y" / "%tid.z" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 실행 중인 ptx_thread_info의 스레드 ID를 반환.
   * 값 범위: x: 0~ntid.x-1, y: 0~ntid.y-1, z: 0~ntid.z-1.
   * 동기화: 스레드마다 고유한 값 — 공유 상태 없음. */

  WARPID_REG,
  /* [한국어] %warpid (PTX ISA §9.2): SM 내 현재 warp의 ID.
   * 같은 SM에 분배된 여러 warp를 구별하는 정수로, SM 내부의 warp 슬롯 번호이다.
   * 주의: warpid는 커널 실행 중 재스케줄링으로 바뀔 수 있는 하드웨어 종속 값이다
   *       (블록 내 일관된 warp 번호인 %laneid와 달리 HW 의존적).
   * 설정자: ptx.y가 "%warpid" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 현재 warp의 m_warp_id 필드를 반환.
   * 값 범위: 0 ~ nwarpid-1.
   * 동기화: 각 warp가 독립적으로 자신의 ID를 읽으므로 경쟁 조건 없음. */

  WARPSZ_REG
  /* [한국어] %WARP_SZ (PTX ISA §9.2): warp 크기 상수 (항상 32).
   * NVIDIA GPU의 SIMT 실행 폭이 32로 고정되어 있음을 커널 코드에서 읽을 수 있게 한다.
   * 설정자: ptx.y가 "%WARP_SZ" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc — 상수 32를 반환 (시뮬레이터에서도 warp 크기는 32 고정).
   * 값 범위: 항상 32 (CUDA HW 불변 속성).
   * 사용 목적: 이식성 있는 warp-level 코드 작성 — 하드코딩 대신 이 레지스터를 사용하면
   *            미래 아키텍처에서 warp 크기가 바뀌어도 코드가 유지된다 (현재는 항상 32). */
};

/* [한국어] wmma_type — Tensor Core의 WMMA(Warp Matrix Multiply Accumulate) 명령어 세부 타입.
 *
 * PTX ISA는 wmma.load, wmma.store, wmma.mma 명령어를 통해 Tensor Core 행렬 연산을 지원한다.
 * 이 열거형은 wmma 명령어의 다양한 세부 동작(행렬 로드/저장, 실제 MMA 연산,
 * 행렬 메모리 레이아웃, 타일 크기)을 하나의 타입으로 통합하여 구분한다.
 *
 * Tensor Core는 fp16 행렬 곱(A×B + C = D)을 warp 단위로 한 번에 실행하며,
 * 각 warp의 32개 스레드가 행렬 타일의 서로 다른 원소를 분담하여 보유한다.
 *
 * 설정자: ptx.y 파서 — wmma 명령어의 opcode variant를 이 enum 값으로 변환.
 * 읽는 자: instructions.cc의 wmma_impl() 또는 동등 함수 — wmma_type에 따라
 *           올바른 행렬 로드/저장/곱셈 시뮬레이션 경로를 선택.
 * 동기화: warp 단위 연산으로 단일 warp 내에서만 사용 — 별도 락 불필요. */
enum wmma_type {
  LOAD_A,
  /* [한국어] wmma.load 명령어로 행렬 A를 전역/공유 메모리에서 warp 레지스터 파일로 로드한다.
   * 행렬 A는 M×K 타일로 fp16 타입이며, 각 스레드가 타일의 일부 원소를 레지스터에 저장한다.
   * 설정자: ptx.y가 "wmma.load.a.*" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 행렬 A에 해당하는 레지스터 파티션을 채운다.
   * 레이아웃: row-major 또는 col-major (ROW/COL 열거값으로 별도 지정).
   * 하드웨어 배경: Tensor Core는 A 행렬을 특수 레지스터 배열에 저장; 스레드당 할당 원소 수는
   *               타일 크기와 데이터 타입에 따라 결정 (M16N16K16 fp16 기준: 스레드당 8원소). */

  LOAD_B,
  /* [한국어] wmma.load 명령어로 행렬 B를 warp 레지스터 파일로 로드한다.
   * 행렬 B는 K×N 타일로 fp16 타입이며, A와 함께 MMA 연산의 입력을 구성한다.
   * 설정자: ptx.y가 "wmma.load.b.*" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 행렬 B에 해당하는 레지스터 파티션을 채운다.
   * 레이아웃: A와 마찬가지로 ROW/COL로 지정.
   * 하드웨어 배경: B 행렬도 스레드 간 분산 저장; MMA 실행 전에 A/B/C 모두 로드 완료 필요. */

  LOAD_C,
  /* [한국어] wmma.load 명령어로 누산 행렬 C를 warp 레지스터 파일로 로드한다.
   * 행렬 C는 M×N 타일로 fp16 또는 fp32 타입이며, MMA 결과(D = A×B + C)의 초기값이다.
   * 설정자: ptx.y가 "wmma.load.c.*" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — accumulator 레지스터 파티션을 채운다.
   * 값 범위: MMA 연산의 누산기로 사용; fp32 accumulator가 정밀도 손실을 줄인다.
   * 하드웨어 배경: Tensor Core는 입력 fp16 × fp16을 내부 fp32로 누산하므로
   *               C를 fp32로 제공하면 전체 연산이 fp32 정밀도를 유지한다. */

  STORE_D,
  /* [한국어] wmma.store 명령어로 MMA 결과 행렬 D를 레지스터에서 메모리로 저장한다.
   * MMA 연산(D = A×B + C) 완료 후 누산기 레지스터 내용을 전역/공유 메모리로 기록한다.
   * 설정자: ptx.y가 "wmma.store.d.*" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — accumulator 레지스터 파티션을 메모리에 기록.
   * 레이아웃: ROW/COL로 지정; warp의 32개 스레드가 협력하여 M×N 타일을 분산 저장.
   * 하드웨어 배경: STORE_D 전에 wmma.mma가 완료되어야 하며, 순서를 보장하기 위해
   *               컴파일러가 wmma.mma 후 wmma.store.d 순으로 코드를 생성한다. */

  MMA,
  /* [한국어] wmma.mma 명령어: D = A × B + C 행렬 곱셈 + 누산 연산을 Tensor Core로 수행한다.
   * LOAD_A/B/C로 레지스터에 올라온 행렬을 입력으로 받아 Tensor Core 하드웨어를 구동한다.
   * 설정자: ptx.y가 "wmma.mma.*" 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 행렬 곱셈 시뮬레이션을 수행하고
   *           결과를 accumulator(D) 레지스터 파티션에 저장.
   * 값 범위/정밀도: 입력 A/B fp16, 입력 C fp16/fp32, 출력 D fp16/fp32.
   * 하드웨어 배경: Tensor Core는 warp의 32스레드가 협력하는 SIMT 행렬 단위 연산이며,
   *               단일 사이클에 4×4×4 fp16 행렬 곱을 처리하는 특수 함수 유닛이다.
   *               GPGPU-Sim에서는 소프트웨어로 스레드별 원소 곱셈·누산을 에뮬레이션한다. */

  ROW,
  /* [한국어] 행렬의 메모리 레이아웃이 행 우선(row-major, C-order)임을 나타낸다.
   * wmma.load / wmma.store 명령어의 레이아웃 qualifier로 사용된다.
   * 설정자: ptx.y가 "row" qualifier 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 메모리 주소 계산 시 row stride를 사용.
   * 의미: 한 행(row)의 원소들이 메모리에서 연속 배치됨 (행 인덱스가 느리게 변함).
   * 하드웨어 배경: Tensor Core는 ROW/COL 레이아웃을 모두 지원하며, 컴파일러가
   *               cuBLAS 등 최적 레이아웃을 선택하여 메모리 접근 패턴을 최적화한다. */

  COL,
  /* [한국어] 행렬의 메모리 레이아웃이 열 우선(col-major, Fortran-order)임을 나타낸다.
   * wmma.load / wmma.store 명령어의 레이아웃 qualifier로 사용된다.
   * 설정자: ptx.y가 "col" qualifier 토큰을 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 메모리 주소 계산 시 col stride를 사용.
   * 의미: 한 열(column)의 원소들이 메모리에서 연속 배치됨 (열 인덱스가 느리게 변함).
   * 사용 목적: cuBLAS / 수학 라이브러리는 전통적으로 col-major를 기본으로 사용한다. */

  M16N16K16,
  /* [한국어] 행렬 타일 크기 16×16×16을 나타낸다 (M=16, N=16, K=16).
   * wmma 명령어의 shape qualifier로, 행렬 A(16×16), B(16×16), C/D(16×16) 타일을 지정한다.
   * 설정자: ptx.y가 "m16n16k16" shape qualifier를 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — 행렬 원소 수(256)와 스레드당 원소 분배를 결정.
   * 스레드당 원소 수: A(fp16) 8원소, B(fp16) 8원소, C/D(fp32) 8원소 (warp 32스레드 기준).
   * 하드웨어 배경: Volta(V100)부터 지원되는 기본 wmma 타일 크기; cuBLAS 기본 타일이기도 함. */

  M32N8K16,
  /* [한국어] 행렬 타일 크기 32×8×16을 나타낸다 (M=32, N=8, K=16).
   * 직사각형 타일로, M 방향으로 긴 행렬 연산에 적합하다.
   * 설정자: ptx.y가 "m32n8k16" shape qualifier를 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — M32N8K16 타일에 맞는 원소 분배 계산.
   * 스레드당 원소 수: M16N16K16과 동일한 총 256원소이나 M/N 비율이 다름.
   * 사용 목적: 배치 행렬 곱에서 M이 N보다 크거나 tall-and-skinny 행렬 형태에 최적. */

  M8N32K16
  /* [한국어] 행렬 타일 크기 8×32×16을 나타낸다 (M=8, N=32, K=16).
   * M32N8K16의 전치(transpose) 형태로 N 방향으로 긴 행렬 연산에 적합하다.
   * 설정자: ptx.y가 "m8n32k16" shape qualifier를 이 값으로 변환.
   * 읽는 자: instructions.cc wmma_impl() — M8N32K16 타일에 맞는 원소 분배 계산.
   * 스레드당 원소 수: M16N16K16, M32N8K16과 동일한 총 원소 수(256)이나 M/N 비율 다름.
   * 사용 목적: wide-and-short 행렬(N이 M보다 큰 경우) 연산 최적화;
   *            cuBLAS/cuDNN 내부 gemm 호출에서 커널 tile 선택 시 사용될 수 있음. */

};

/* [한국어] 인클루드 가드 닫기 — opcodes_h_included 블록을 종료한다.
 * 이 이하의 코드는 이 헤더가 중복 포함될 경우 무시된다. */
#endif
