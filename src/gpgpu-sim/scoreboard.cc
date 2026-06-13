// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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
 * [한국어 설명] GPU 파이프라인 레지스터 의존성 추적 — 스코어보드 구현 (scoreboard.cc)
 *
 * === 파일의 역할 ===
 * scoreboard.h에서 선언된 Scoreboard 클래스의 메서드를 구현한다.
 * Scoreboard는 SM(Streaming Multiprocessor) 파이프라인에서 RAW(Read-After-Write)
 * 및 WAW(Write-After-Write) 해저드를 탐지하여 warp 스케줄러가 안전하게 명령어를
 * 발행(issue)할 수 있도록 돕는 하드웨어 유닛의 소프트웨어 모델이다.
 * 구체적으로는: (1) 명령어 발행 시 출력 레지스터를 예약(reserve), (2) writeback 완료 시
 * 예약을 해제(release), (3) 발행 후보 명령어의 레지스터가 예약 집합과 겹치는지 충돌 검사를
 * 수행하는 세 가지 핵심 기능을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 타이밍 시뮬레이션 레이어(gpgpu-sim/)에 속한다.
 * SM 사이클 시뮬레이션의 issue 단계와 writeback 단계에서 각각 호출된다.
 * 호출 체인 (issue 방향):
 *   gpu-sim.cc: gpgpu_sim::cycle()
 *   → shader.cc: shader_core_ctx::cycle()
 *     → shader.cc: scheduler_unit::cycle()
 *       → checkCollision() [발행 가능 여부 판별]
 *       → shader_core_ctx::issue_warp()
 *         → reserveRegisters() [출력 레지스터 예약]
 * 호출 체인 (writeback 방향):
 *   shader.cc: ldst_unit::writeback() 또는 exec_unit::writeback()
 *   → releaseRegisters() 또는 releaseRegister() [예약 해제]
 * 실행 컨텍스트: 호스트 유저스페이스 CPU 스레드. GPU 디바이스 코드가 아님.
 * 사이클 단위: 매 SM 사이클마다 checkCollision/reserve/release 중 해당하는 것이 호출되나,
 *   Scoreboard 자체는 사이클 카운터를 갖지 않고 순수 상태 테이블로만 동작한다.
 *
 * === 타 모듈과의 연결 ===
 * 포함하는 헤더:
 *   - scoreboard.h: Scoreboard 클래스 선언 (reg_table, longopregs, 멤버 함수 시그니처)
 *   - cuda-sim/ptx_sim.h: PTX 시뮬레이션 관련 타입 (warp_inst_t 등의 완전한 정의)
 *   - shader.h: shader_core_ctx, scheduler_unit 등 SM 시뮬레이션 컨텍스트 타입
 *   - shader_trace.h: SHADER_DPRINTF 디버그 트레이스 매크로 (SCOREBOARD 채널)
 * 데이터 흐름:
 *   warp_inst_t (명령어 객체) → inst->out[r], inst->warp_id() 추출
 *   → reg_table[wid] 집합에 삽입/제거/조회
 *   → 결과(bool)를 scheduler_unit에게 반환하여 발행/stall 결정.
 * 공유 자료구조:
 *   reg_table: warp별 pending write 레지스터 집합 (핵심 상태)
 *   longopregs: warp별 long-latency 메모리 연산 목적 레지스터 집합.
 *
 * === 주요 함수/구조체 요약 ===
 * Scoreboard(): reg_table과 longopregs를 n_warps 크기로 초기화하는 생성자.
 * reserveRegisters(): 발행 명령어의 출력 레지스터를 reg_table에 삽입. 메모리 로드면 longopregs에도 삽입.
 * releaseRegisters(): writeback 완료 명령어의 출력 레지스터를 reg_table과 longopregs에서 제거.
 * releaseRegister(): 개별 레지스터 단위 해제 (메모리 응답 처리 시 사용).
 * checkCollision(): 발행 후보 명령어의 전체 레지스터 집합과 reg_table의 교집합 확인.
 * pendingWrites(): reg_table[wid].empty()로 warp의 in-flight 완료 여부 반환.
 * islongop(): longopregs에서 특정 레지스터가 long-latency 연산 중인지 확인.
 * printContents(): 디버그용 스코어보드 전체 상태 출력.
 */

#include "scoreboard.h"         /* [한국어] Scoreboard 클래스 선언 — reg_table, longopregs 멤버, 멤버 함수 시그니처 */
#include "../cuda-sim/ptx_sim.h" /* [한국어] PTX 시뮬레이션 헤더 — warp_inst_t 등 명령어 관련 타입의 완전한 정의 포함 */
#include "shader.h"              /* [한국어] SM 시뮬레이션 컨텍스트 헤더 — shader_core_ctx, scheduler_unit 등 정의.
                                  *          Scoreboard가 속한 SM 구조를 이해하는 데 필요 */
#include "shader_trace.h"        /* [한국어] SHADER_DPRINTF 매크로 정의 — SCOREBOARD 채널의 디버그 트레이스 출력에 사용.
                                  *          -gpgpu_shader_core_pipeline_opt 설정에 따라 출력 여부 결정 */

// Constructor
/*
 * [한국어]
 * Scoreboard::Scoreboard - SM의 스코어보드를 초기화하는 생성자
 *
 * @sid: 이 스코어보드가 속한 SM(shader core)의 정수 ID.
 *        shader_core_ctx 생성 시 전달되는 m_sid와 동일.
 *        디버그 출력("sid=%d")에만 사용되며, 동작 로직에는 영향 없음.
 * @n_warps: 이 SM이 관리하는 warp의 총 개수.
 *            gpgpusim.config의 -gpgpu_num_sched_per_core 및 warp 수 설정에서 결정됨.
 *            reg_table과 longopregs 벡터의 크기(= 인덱스 최대값 + 1)가 된다.
 * @gpu: GPGPU-Sim 전체 시뮬레이터 컨텍스트 포인터.
 *        m_gpu 멤버에 저장되며, SHADER_DPRINTF 매크로의 디버그 출력 레벨 확인에 사용.
 *
 * 생성자 초기화 목록(initializer list)에서 longopregs()를 명시적으로 기본 초기화한다.
 * reg_table과 longopregs는 resize() 이후 각 원소가 빈 std::set<unsigned>로 시작한다.
 * 실행 컨텍스트: shader_core_ctx 생성자에서 단 한 번 호출. 시뮬레이션 시작 전 초기화 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::shader_core_ctx() → new Scoreboard(sid, n_warps, gpu)
 *   → [Scoreboard::Scoreboard()]
 */
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {   /* [한국어] longopregs를 기본 생성자로 명시 초기화 — std::vector의 기본 초기화와 동일하나
                        *          코드 의도를 명확히 하기 위해 명시. reg_table은 초기화 목록에 없어 자동 기본 초기화 */
  m_sid = sid;         /* [한국어] SM ID 저장 — 이후 printContents()의 "scoreboard contents (sid=%d)" 출력에 사용 */
  // Initialize size of table
  reg_table.resize(n_warps);   /* [한국어] warp 수만큼 pending-write 레지스터 집합을 준비.
                                 *          resize() 이후 reg_table[0] ~ reg_table[n_warps-1]이 각각 빈 set으로 생성됨.
                                 *          이 SM에서 실행 가능한 모든 warp에 대한 슬롯을 사전 할당 */
  longopregs.resize(n_warps);  /* [한국어] warp 수만큼 long-latency 연산 목적 레지스터 집합을 준비.
                                 *          reg_table과 동일한 인덱스 구조 — longopregs[wid]는
                                 *          warp wid의 전역/로컬/텍스처 메모리 로드 목적 레지스터를 추적 */

  m_gpu = gpu;   /* [한국어] GPU 시뮬레이터 컨텍스트 포인터 저장 — SHADER_DPRINTF 매크로가
                  *          내부적으로 gpgpu_t를 통해 디버그 플래그와 출력 스트림에 접근 */
}

// Print scoreboard contents
/*
 * [한국어]
 * Scoreboard::printContents - 스코어보드 전체 상태를 표준 출력으로 덤프
 *
 * 반환값: 없음.
 *
 * 시뮬레이터 디버깅, 파이프라인 행 감지(hang detection), 또는 display_pipeline() 호출 시
 * SM 내부 상태를 사람이 읽을 수 있는 형태로 출력한다.
 * 각 warp에 대해 pending write 레지스터가 있는 경우만 출력하고, 빈 warp는 건너뛴다.
 * 출력 형식: "scoreboard contents (sid=N): \n  wid = X: reg1 reg2 ..."
 * 실행 컨텍스트: 디버그/덤프 용도로 호출되며, 시뮬레이션 루프 외부에서도 안전하게 호출 가능.
 *
 * 호출 체인:
 *   shader_core_ctx::display_pipeline() → [Scoreboard::printContents()]
 */
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);   /* [한국어] 이 스코어보드가 속한 SM의 ID를 먼저 출력.
                                                          *          어떤 SM의 스코어보드인지 식별 가능 */
  for (unsigned i = 0; i < reg_table.size(); i++) {     /* [한국어] 모든 warp 슬롯을 순서대로 순회.
                                                          *          i는 warp ID (0 ~ n_warps-1) */
    if (reg_table[i].size() == 0) continue;             /* [한국어] pending write가 없는 warp는 출력 생략 —
                                                          *          관심 있는 warp만 표시하여 출력량 축소 */
    printf("  wid = %2d: ", i);                         /* [한국어] pending write가 있는 warp의 ID를 출력.
                                                          *          2자리 정렬(%2d)로 정렬된 출력 제공 */
    std::set<unsigned>::const_iterator it;              /* [한국어] set 순회를 위한 const 반복자 선언.
                                                          *          printContents()는 const 함수이므로
                                                          *          const_iterator를 사용해야 컴파일 가능 */
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      /* [한국어] warp i의 pending write 레지스터 번호를 오름차순으로 순회.
       *          std::set은 자동 정렬(오름차순)이므로 별도 정렬 불필요 */
      printf("%u ", *it);   /* [한국어] 각 레지스터 번호를 공백으로 구분하여 출력.
                              *          예: "3 7 12 " — warp가 레지스터 3, 7, 12에 아직 쓰기 중 */
    printf("\n");            /* [한국어] 각 warp 줄의 끝에 개행 추가 — 다음 warp 출력과 구분 */
  }
}

/*
 * [한국어]
 * Scoreboard::reserveRegister - 단일 레지스터를 "쓰기 보류" 집합에 삽입 (private 헬퍼)
 *
 * @wid: 예약할 레지스터가 속한 warp의 ID. reg_table의 인덱스로 직접 사용.
 * @regnum: 예약할 레지스터 번호. reg_table[wid]에 삽입될 값.
 *
 * 이미 예약된 레지스터를 다시 예약하려 할 경우(이중 예약) 오류 메시지를 출력하고
 * abort()로 프로세스를 강제 종료한다. 이는 정상 시뮬레이션에서는 발생해서는 안 되는
 * 상황이므로 즉각적인 종료로 버그를 명확히 드러낸다.
 * 성공 시 SHADER_DPRINTF(SCOREBOARD, ...) 매크로로 예약 이벤트를 트레이스한다.
 * 실행 컨텍스트: reserveRegisters()에서만 호출. 단일 시뮬레이션 스레드.
 * 에러 처리: 이중 예약 시 printf + abort() — 복구 불가 치명 오류로 처리.
 *
 * 호출 체인:
 *   Scoreboard::reserveRegisters() → [Scoreboard::reserveRegister(wid, regnum)]
 *   → reg_table[wid].insert(regnum)
 */
void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end())) {
    /* [한국어] reg_table[wid]에서 regnum을 검색한 결과가 end()와 같지 않으면 —
     *          즉, 이미 해당 레지스터가 집합에 존재하면 — 이중 예약 오류 조건.
     *          정상 파이프라인이라면 같은 레지스터를 두 번 예약할 일이 없으므로
     *          이 분기 진입은 시뮬레이터 버그를 의미한다 */
    printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);   /* [한국어] 어떤 SM(sid), warp(wid), 레지스터(regnum)에서
                                 *          이중 예약이 발생했는지 정확히 출력 — 디버깅 단서 제공 */
    abort();   /* [한국어] 즉각적인 프로세스 종료 — core dump를 생성하여 스택 트레이스 분석 가능.
                *          assert()와 달리 NDEBUG 여부와 무관하게 항상 종료 */
  }
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  /* [한국어] SHADER_DPRINTF 매크로: gpgpusim.config에서 SCOREBOARD 채널이 활성화된 경우에만
   *          출력. "Reserved Register - warp:W, reg:R" 형식으로 예약 이벤트 기록.
   *          이 트레이스를 통해 시뮬레이션 중 어떤 순서로 레지스터가 예약되는지 추적 가능 */
  reg_table[wid].insert(regnum);   /* [한국어] warp wid의 pending-write 집합에 regnum 삽입.
                                    *          std::set::insert()는 이미 존재하면 무시하지만,
                                    *          위에서 이중 예약을 이미 차단했으므로 여기선 항상 새 원소 삽입 */
}

// Unmark register as write-pending
/*
 * [한국어]
 * Scoreboard::releaseRegister - 단일 레지스터의 "쓰기 보류" 표시를 해제 (public)
 *
 * @wid: 해제할 레지스터가 속한 warp의 ID.
 * @regnum: 해제할 레지스터 번호. reg_table[wid]에서 제거될 값.
 *
 * 레지스터가 집합에 없는 경우(이미 해제되었거나 예약된 적 없음)는 조용히 반환한다
 * (오류로 처리하지 않음). 이는 일부 실행 경로에서 같은 레지스터를 중복 해제할 수
 * 있는 상황을 허용하기 위함이다.
 * longopregs는 이 함수에서 건드리지 않는다 — longopregs 해제가 필요하면
 * releaseRegisters()를 사용해야 한다.
 * 실행 컨텍스트: writeback 단계에서 호출. 단일 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ldst_unit::L1_latency_queue_cycle() 또는 ldst_unit::writeback()
 *   → [Scoreboard::releaseRegister()] → reg_table[wid].erase(regnum)
 */
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  /* [한국어] reg_table[wid]에서 regnum을 찾지 못한 경우(find() == end())이면 즉시 반환.
   *          "!(find() != end())" = "find() == end()" 즉, 집합에 없음을 의미.
   *          이미 해제된 레지스터를 다시 해제하려는 경우를 오류 없이 무시 */
  SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  /* [한국어] SCOREBOARD 채널 트레이스 출력 — "Release register - warp:W, reg:R".
   *          예약과 해제 이벤트를 짝 지어 추적하면 레지스터 의존성 흐름 전체를 파악 가능 */
  reg_table[wid].erase(regnum);   /* [한국어] warp wid의 pending-write 집합에서 regnum을 제거.
                                   *          std::set::erase(value)는 원소가 없으면 아무 일도 하지 않지만,
                                   *          위에서 이미 존재 여부를 확인했으므로 항상 1개 원소를 삭제 */
}

/*
 * [한국어]
 * Scoreboard::islongop - 특정 레지스터가 long-latency 메모리 연산의 목적 레지스터인지 확인
 *
 * @warp_id: 확인할 warp의 ID. longopregs 인덱스로 사용.
 * @regnum: 확인할 레지스터 번호. longopregs[warp_id] 집합에서 탐색.
 * @return (const bool): true  → warp_id의 warp에서 regnum이 long-latency 메모리 로드의
 *                               목적 레지스터로 등록되어 있음.
 *                        false → longopregs에 없음. 일반 산술 연산이거나 미예약 레지스터.
 *
 * warp 스케줄러(scheduler_unit::cycle())가 특정 warp가 stall된 원인을 분류할 때 사용한다.
 * 입력 피연산자가 longopregs에 포함된 경우, 해당 stall은 메모리 레이턴시(전역/텍스처 메모리
 * 미스)에 의한 것으로 분류되며, 통계 카운터(예: MIO_DATA_DEPENDENCE_STALL)에 별도 기록된다.
 * longopregs에 없는 경우 산술 실행 지연 또는 구조적 해저드에 의한 stall로 분류된다.
 * 실행 컨텍스트: scheduler_unit::cycle() — 매 SM 사이클 issue 단계. 단일 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   scheduler_unit::cycle() (stall 원인 분류) → [Scoreboard::islongop()]
 *   → longopregs[warp_id].find(regnum) != longopregs[warp_id].end()
 */
const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
  /* [한국어] longopregs[warp_id]에서 regnum을 검색한 반복자가 end()와 다르면
   *          원소가 존재함(= long-latency 연산 중)을 의미.
   *          std::set::find()는 O(log n) 탐색. longopregs 집합은 reg_table보다
   *          항상 작거나 같으므로(부분 집합) 실제 탐색 비용은 매우 작다 */
}

/*
 * [한국어]
 * Scoreboard::reserveRegisters - 발행 명령어의 출력 레지스터 전체를 예약
 *
 * @inst: 이번 사이클에 파이프라인에 발행(issue)된 warp 명령어.
 *         inst->out[0..MAX_OUTPUT_VALUES-1]에 목적 레지스터 번호,
 *         inst->warp_id()에 warp ID가 담겨 있다.
 *         MAX_OUTPUT_VALUES = 8 (abstract_hardware_model.h 정의).
 *
 * 명령어 발행 직후 호출되어, 해당 명령어가 쓸 모든 출력 레지스터를 reg_table에 삽입한다.
 * 출력 레지스터가 없거나(out[r] == 0) 레지스터 번호가 0인 경우(GPGPU-Sim 관례상 "없음")는
 * 건너뛴다. 전역/로컬/텍스처 메모리 로드(long-latency 연산)의 경우 추가로 longopregs에도
 * 삽입하여 long-latency 레이턴시로 인한 stall임을 표시한다.
 * 실행 컨텍스트: shader_core_ctx::issue_warp() 내부 — 매 SM 사이클 issue 단계.
 * 에러 처리: 이중 예약은 reserveRegister()가 abort()로 처리.
 *
 * 호출 체인:
 *   scheduler_unit::cycle() → shader_core_ctx::issue_warp()
 *   → [Scoreboard::reserveRegisters(inst)]
 *   → reserveRegister(wid, out[r]) [출력 레지스터별 반복]
 *   → (is_load() && long-latency space) longopregs[wid].insert(out[r])
 */
void Scoreboard::reserveRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    /* [한국어] 명령어의 출력 레지스터 슬롯을 0번부터 MAX_OUTPUT_VALUES-1(=7)번까지 순회.
     *          대부분의 명령어는 출력 레지스터가 1개이지만, 벡터 연산(v2/v4 LD 등)은
     *          여러 출력 레지스터를 사용하므로 배열 전체를 확인해야 한다 */
    if (inst->out[r] > 0) {
      /* [한국어] out[r] == 0은 "출력 없음"을 의미하는 GPGPU-Sim 관례.
       *          실제 레지스터 번호는 1 이상이어야 유효하다.
       *          0인 슬롯은 이미 출력 레지스터가 없는 것이므로 예약할 필요 없음 */
      reserveRegister(inst->warp_id(), inst->out[r]);
      /* [한국어] private 헬퍼 함수를 통해 reg_table[warp_id]에 out[r]을 삽입.
       *          이후 같은 warp의 후속 명령어가 이 레지스터를 읽으려 하면
       *          checkCollision()에서 충돌로 탐지되어 발행이 거부된다 */
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      /* [한국어] SCOREBOARD 채널 디버그 트레이스 — reserveRegisters() 레벨 예약 이벤트 기록.
       *          private reserveRegister()(scoreboard.cc:197)에서도 같은 레지스터에 대해
       *          SHADER_DPRINTF를 호출하므로, SCOREBOARD 채널 활성화 시 레지스터 1개당
       *          2줄이 출력된다 — 이는 정상 동작이며 버그가 아니다:
       *            1) 이 줄: "Reserved register - warp:W, reg:R"  (소문자 r)
       *            2) reserveRegister() 내부: "Reserved Register - warp:W, reg:R"  (대문자 R)
       *          두 메시지의 대소문자 차이로 어느 레벨에서 출력된 것인지 구분할 수 있다 */
    }
  }

  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    /* [한국어] long-latency 연산 판별 조건:
     *   (1) is_load(): 메모리에서 값을 읽어 레지스터에 저장하는 로드 명령어여야 한다.
     *       store는 레지스터를 읽기만 하고 쓰지 않으므로 scoreboard 예약과 무관.
     *   (2) 메모리 공간이 다음 중 하나:
     *       - global_space: 전역 메모리 — GPU 전체가 공유하는 GDDR/HBM. 미스 시 수백 사이클.
     *       - local_space: 로컬 메모리 — 스레드 전용이지만 전역 메모리에 매핑됨. 역시 느림.
     *       - param_space_kernel: 커널 파라미터 공간 — 전역 메모리에 위치.
     *       - param_space_local: 로컬 파라미터 공간 — 로컬 메모리에 위치.
     *       - param_space_unclassified: 아직 분류되지 않은 파라미터 공간 — 안전하게 long-latency로 처리.
     *       - tex_space: 텍스처 메모리 — 전역 메모리 기반이지만 텍스처 캐시를 경유.
     *   반면 shared_space(공유 메모리), const_space(상수 메모리 캐시 히트)는 빠르므로
     *   longopregs에 등록하지 않는다 */
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      /* [한국어] long-latency 로드 명령어의 출력 레지스터를 다시 순회.
       *          위의 reg_table 예약 루프와 별개로, 동일한 레지스터들을 longopregs에도 삽입 */
      if (inst->out[r] > 0) {
        /* [한국어] 유효한 출력 레지스터(번호 > 0)에 대해서만 longopregs에 삽입 */
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       inst->warp_id(), inst->out[r]);
        /* [한국어] longopregs 등록 이벤트를 SCOREBOARD 채널에 트레이스.
         *          "New longopreg marked" 로그로 long-latency stall의 시작점을 추적 가능 */
        longopregs[inst->warp_id()].insert(inst->out[r]);
        /* [한국어] warp의 long-latency 레지스터 집합에 삽입.
         *          이후 scheduler_unit이 islongop()으로 이 레지스터를 조회하면 true를 반환하여
         *          stall 원인이 메모리 레이턴시임을 식별할 수 있다 */
      }
    }
  }
}

// Release registers for an instruction
/*
 * [한국어]
 * Scoreboard::releaseRegisters - writeback 완료 명령어의 출력 레지스터 전체를 해제
 *
 * @inst: writeback이 완료된 warp 명령어.
 *         inst->out[0..MAX_OUTPUT_VALUES-1]에 목적 레지스터 번호,
 *         inst->warp_id()에 warp ID가 담겨 있다.
 *
 * 실행 유닛(산술 유닛 또는 LD/ST 유닛)이 결과를 레지스터 파일에 쓴 직후 호출된다.
 * reg_table[warp_id]에서 해당 레지스터를 제거(erase)하여 해당 레지스터의 RAW 해저드를 해소한다.
 * 동시에 longopregs[warp_id]에서도 제거(erase)하여 long-latency stall 표시도 해제한다.
 * 이 함수가 호출된 후 다음 사이클부터 같은 warp에서 이 레지스터를 읽는 명령어가 발행 가능해진다.
 * 실행 컨텍스트: writeback 단계 (ldst_unit::writeback() 또는 exec_unit 계열의 writeback).
 *                단일 시뮬레이션 스레드.
 * 에러 처리: 이미 해제된 레지스터는 releaseRegister()에서 조용히 무시.
 *
 * 호출 체인:
 *   ldst_unit::writeback() 또는 simd_function_unit::writeback()
 *   → [Scoreboard::releaseRegisters(inst)]
 *   → releaseRegister(wid, out[r])   [reg_table에서 제거]
 *   → longopregs[wid].erase(out[r]) [longopregs에서 제거]
 */
void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    /* [한국어] 명령어의 출력 레지스터 슬롯 0번부터 MAX_OUTPUT_VALUES-1번까지 순회.
     *          reserveRegisters()와 동일한 배열 범위를 검사하여 예약/해제의 대칭성 보장 */
    if (inst->out[r] > 0) {
      /* [한국어] 유효한 출력 레지스터(번호 > 0)에 대해서만 해제 수행.
       *          out[r] == 0인 슬롯은 예약된 적 없으므로 건너뜀 */
      SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      /* [한국어] SCOREBOARD 채널 트레이스 — "Register Released - warp:W, reg:R".
       *          예약(reserveRegisters)과 해제(releaseRegisters) 이벤트를 쌍으로 추적하면
       *          레지스터별 in-flight 기간을 분석할 수 있다 */
      releaseRegister(inst->warp_id(), inst->out[r]);
      /* [한국어] private 헬퍼를 통해 reg_table[warp_id]에서 out[r]을 제거.
       *          이 호출 이후 같은 warp에서 이 레지스터를 읽는 명령어가
       *          checkCollision()에서 충돌을 발생시키지 않게 된다 */
      longopregs[inst->warp_id()].erase(inst->out[r]);
      /* [한국어] longopregs[warp_id]에서도 out[r]을 제거.
       *          std::set::erase(value)는 원소가 없으면 아무 일도 하지 않으므로
       *          long-latency 연산이 아니었던 경우(longopregs에 없는 레지스터)도 안전하게 처리 */
    }
  }
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
/*
 * [한국어]
 * Scoreboard::checkCollision - 발행 후보 명령어의 레지스터가 보류 집합과 충돌하는지 검사
 *
 * @wid: 발행 후보 명령어를 실행할 warp의 ID. reg_table의 인덱스로 사용.
 * @inst: 발행 후보 명령어(inst_t* 타입). warp_inst_t의 기반 클래스.
 *         inst->out[], inst->in[], inst->pred, inst->ar1, inst->ar2의
 *         레지스터 번호를 모두 모아 reg_table[wid]과 교집합을 구한다.
 * @return: true  → 충돌 존재 (RAW 또는 WAW 해저드).
 *                   이 warp의 명령어 발행을 이번 사이클에 거부해야 한다.
 *           false → 충돌 없음. 이 명령어는 피연산자가 모두 준비됐으므로 발행 가능.
 *
 * 동작 원리:
 *   1. 명령어의 모든 관련 레지스터(출력, 입력, 조건, 주소)를 inst_regs 집합에 수집.
 *   2. inst_regs의 각 원소를 reg_table[wid]에서 검색.
 *   3. 하나라도 발견되면 즉시 true 반환(조기 종료 — 최악 O(N·log M)).
 *   4. 모두 검색 후 없으면 false 반환.
 *
 * WAR(Write-After-Read) 해저드를 검사하지 않는 이유:
 *   GPGPU-Sim은 in-order issue 모델을 사용하므로 읽기 명령어가 항상 쓰기 명령어보다
 *   먼저 발행된다. 따라서 읽기 명령어가 실행 중이면 쓰기 명령어는 아직 발행 대기 중이므로
 *   WAR은 구조적으로 발생할 수 없다.
 *
 * 실행 컨텍스트: scheduler_unit::cycle() — 매 SM 사이클 issue 단계. 단일 시뮬레이션 스레드.
 * 성능: std::set 삽입/검색은 O(log n). 레지스터 수는 최대 MAX_INPUT_VALUES + MAX_OUTPUT_VALUES = 32개
 *        수준으로 제한되어 실용적으로 빠르다.
 *
 * 호출 체인:
 *   scheduler_unit::cycle() → [Scoreboard::checkCollision(wid, inst)]
 *   → inst_regs 집합 구성 → reg_table[wid] 교집합 검사
 */
bool Scoreboard::checkCollision(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;   /* [한국어] 이 명령어가 참조하는 모든 레지스터 번호를 담는 임시 집합.
                               *          std::set을 사용하므로 중복 레지스터 번호는 자동으로 제거된다.
                               *          int 타입으로 선언된 이유: pred, ar1, ar2가 int 타입이기 때문
                               *          (부호 없는 값과 혼재하지만 레지스터 번호는 항상 양수) */

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    /* [한국어] 출력 레지스터 배열(out[])을 outcount 개수만큼 순회하여 inst_regs에 삽입.
     *          outcount: 실제로 사용되는 출력 레지스터 수 (MAX_OUTPUT_VALUES 이하).
     *          WAW 해저드: 이미 예약된 레지스터에 또 쓰려 하는 경우 — 이 루프에서 수집한
     *          출력 레지스터가 reg_table에 이미 있으면 WAW로 탐지된다 */
    inst_regs.insert(inst->out[iii]);   /* [한국어] 출력 레지스터 번호를 집합에 삽입. out[iii]은 unsigned이나
                                         *          int 집합에 넣어도 레지스터 번호 범위에서는 안전 */

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    /* [한국어] 입력 레지스터 배열(in[])을 incount 개수만큼 순회하여 inst_regs에 삽입.
     *          RAW 해저드: 이전 명령어가 아직 레지스터에 쓰기를 완료하지 않았는데
     *          이 명령어가 같은 레지스터를 읽으려 하는 경우 — 이 루프에서 수집한
     *          입력 레지스터가 reg_table에 있으면 RAW로 탐지된다 */
    inst_regs.insert(inst->in[jjj]);    /* [한국어] 입력 레지스터 번호를 집합에 삽입.
                                         *          in[jjj]도 unsigned이나 int 집합에 안전하게 삽입 가능 */

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  /* [한국어] 조건 레지스터(predicate register) 처리.
   *          pred(프레디케이트): PTX 명령어에서 @p 형태로 지정되는 조건 레지스터.
   *          이 레지스터의 값에 따라 명령어를 실행할지 건너뛸지 결정한다.
   *          pred > 0인 경우만 추가 (0은 "조건 레지스터 없음"을 의미) */

  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  /* [한국어] 주소 레지스터 1(address register 1) 처리.
   *          ar1: 메모리 주소 계산에 사용되는 베이스 레지스터 번호.
   *          LD/ST 명령어에서 [%ar1 + offset] 형태의 주소 계산에 참여.
   *          ar1 > 0인 경우만 추가 (0은 사용하지 않음을 의미) */

  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);
  /* [한국어] 주소 레지스터 2(address register 2) 처리.
   *          ar2: 스케일드 인덱스 어드레싱 등에서 두 번째 주소 레지스터로 사용.
   *          일부 복잡한 주소 계산 모드에서만 필요하며, 대부분의 경우 0.
   *          ar2 > 0인 경우만 추가 */

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;   /* [한국어] inst_regs 순회를 위한 const 반복자.
                                         *          checkCollision()이 const 멤버 함수이므로
                                         *          const_iterator를 사용해야 컴파일 가능 */
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
    /* [한국어] inst_regs의 모든 레지스터 번호를 순회하며 reg_table[wid]에서 검색.
     *          std::set은 정렬된 구조이므로 begin()부터 end()까지가 오름차순 순회 */
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      /* [한국어] reg_table[wid]에서 *it2(현재 레지스터 번호)를 찾았다면 —
       *          즉, 이 레지스터가 현재 파이프라인의 "쓰기 보류" 목록에 있다면 —
       *          해저드가 존재하는 것이다.
       *          find()가 end()와 다르면 원소가 집합에 존재함을 의미 */
      return true;   /* [한국어] 충돌 발견 — 즉시 true 반환(조기 종료).
                      *          하나라도 충돌하면 발행 불가이므로 나머지 레지스터는 검사하지 않음.
                      *          호출자(scheduler_unit)는 이 warp의 발행을 이번 사이클에 건너뜀 */
    }
  return false;   /* [한국어] 모든 레지스터를 검사했지만 reg_table과 교집합이 없음 —
                   *          충돌 없음. 이 명령어는 모든 피연산자가 준비되었으므로 발행 가능.
                   *          호출자는 이 warp의 명령어를 파이프라인에 발행하고,
                   *          이어서 reserveRegisters()를 호출하여 출력 레지스터를 예약한다 */
}

/*
 * [한국어]
 * Scoreboard::pendingWrites - 특정 warp에 미완료 쓰기가 남아 있는지 확인
 *
 * @wid: 확인할 warp의 ID. reg_table의 인덱스로 사용.
 * @return: true  → reg_table[wid]가 비어 있지 않음.
 *                   이 warp에는 아직 writeback이 완료되지 않은 레지스터가 1개 이상 존재.
 *           false → reg_table[wid]가 비어 있음.
 *                   이 warp의 모든 in-flight 명령어가 완료되어 레지스터 쓰기 보류 없음.
 *
 * CTA(블록)의 실행 완료 여부를 판별할 때, 또는 warp가 메모리 배리어(__syncthreads에 해당하는
 * 시뮬레이션 내부 배리어)를 통과할 수 있는지 확인할 때 사용된다.
 * 구체적으로 shader_core_ctx::warp_waiting_at_mem_barrier()에서 pendingWrites()를 통해
 * 해당 warp의 모든 메모리 연산이 완료되었는지 판별하고, CTA 종료 조건 확인에도 활용된다.
 * 단순히 set의 empty() 여부를 반환하므로 O(1) 연산이다.
 * 실행 컨텍스트: 다양한 SM 상태 확인 로직에서 호출. 단일 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   shader_core_ctx::warp_waiting_at_mem_barrier() 또는
 *   shader_core_ctx::can_issue_1block()
 *   → [Scoreboard::pendingWrites(wid)]
 *   → reg_table[wid].empty() 반환
 */
bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();   /* [한국어] reg_table[wid]가 비어 있지 않으면(쓰기 보류 레지스터 존재)
                                     *          true를 반환하고, 비어 있으면(모든 쓰기 완료) false를 반환.
                                     *          std::set::empty()는 O(1) 연산 — 집합 크기와 무관하게 즉시 반환.
                                     *          !empty()로 부정하여 "보류가 있으면 true" 의미론을 충족 */
}
