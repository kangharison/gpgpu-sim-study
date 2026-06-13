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
 * [한국어 설명] GPU 파이프라인 레지스터 의존성 추적 — 스코어보드 (scoreboard.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 SM(Streaming Multiprocessor, 스트리밍 멀티프로세서) 파이프라인에서
 * RAW(Read-After-Write, 쓰기 후 읽기) 해저드와 WAW(Write-After-Write, 쓰기 후 쓰기)
 * 해저드를 감지하고 방지하는 Scoreboard 클래스를 선언한다.
 * 하드웨어 GPU의 scoreboard 유닛을 소프트웨어로 모델링한 것으로, 발행(issue)된 명령어가
 * 아직 쓰기(writeback)를 완료하지 않은 레지스터들의 집합을 warp 단위로 관리한다.
 * 이를 통해 warp 스케줄러는 파이프라인에 넣을 다음 명령어가 미완료 쓰기 레지스터에
 * 의존하는지 판단할 수 있으며, 의존성이 있으면 해당 warp의 명령어 발행을 지연(stall)시킨다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 타이밍 시뮬레이션(gpgpu-sim/) 레이어에 속하며, SM 파이프라인의
 * 명령어 발행(issue) 단계에서 핵심적인 역할을 담당한다.
 * 실행 흐름: shader_core_ctx(SM 시뮬레이션 컨텍스트) → warp 스케줄러(scheduler_unit)
 *   → Scoreboard::checkCollision() 조회 → 충돌 없으면 reserveRegisters() 호출 후 발행
 *   → 실행 유닛이 writeback 완료 시 releaseRegisters() 호출.
 * 사이클(cycle) 구동: SM의 매 사이클 issue 단계에서 호출된다. 스코어보드 자체는
 * 사이클을 진행시키지 않으며, 상태 테이블 조회/갱신만 수행한다.
 * 실행 컨텍스트: 호스트 유저스페이스(CPU 스레드)에서 실행된다. GPU 디바이스 코드가 아님.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - abstract_hardware_model.h: warp_inst_t(명령어+warp 정보), inst_t(명령어 IR),
 *     MAX_OUTPUT_VALUES(최대 출력 레지스터 수 = 8) 등 핵심 타입 정의.
 *   - gpgpu_t: GPU 시뮬레이터 전체 컨텍스트 (디버그 출력 등에 사용).
 * 이 모듈에 의존하는 모듈:
 *   - gpgpu-sim/shader.cc: shader_core_ctx가 Scoreboard를 소유하며, 워프 스케줄러
 *     (scheduler_unit)가 매 사이클 checkCollision()을 호출하고, 발행 직후
 *     reserveRegisters()를, writeback 시 releaseRegisters()를 호출한다.
 *   - gpgpu-sim/shader.h: Scoreboard*를 멤버로 갖는 scheduler_unit 클래스 선언.
 * 데이터 흐름:
 *   warp_inst_t(발행될 명령어) → reserveRegisters()로 출력 레지스터 예약
 *   → 실행 파이프라인(EX/MEM/WB 단계) → releaseRegisters()로 예약 해제
 *   → 다음 명령어의 checkCollision()이 빈 집합을 확인하고 통과.
 *
 * === 주요 함수/구조체 요약 ===
 * Scoreboard(sid, n_warps, gpu): SM ID와 warp 수로 테이블을 초기화하는 생성자.
 * reserveRegisters(inst): 명령어 발행 시 출력 레지스터를 "쓰기 보류(pending write)" 집합에 추가.
 * releaseRegisters(inst): writeback 완료 시 출력 레지스터를 집합에서 제거.
 * checkCollision(wid, inst): 명령어의 입/출력 레지스터가 보류 집합과 겹치는지 확인 (RAW/WAW 감지).
 * pendingWrites(wid): 특정 warp에 미완료 쓰기가 남아 있는지 확인 (CTA 완료 판별에 사용).
 * islongop(warp_id, regnum): 해당 레지스터가 long-latency 연산(전역/로컬/텍스처 메모리 로드)에 속하는지 확인.
 * reg_table: warp별 "쓰기 보류 중인 레지스터 번호" 집합 (핵심 상태 테이블).
 * longopregs: warp별 "long-latency 연산의 목적 레지스터" 집합 (MIO 파이프라인 판별용).
 */

#include <stdio.h>    /* [한국어] printf() 등 표준 입출력 함수 — printContents()의 디버그 출력에 사용 */
#include <stdlib.h>   /* [한국어] abort() 함수 — 이중 예약(double-reserve) 오류 발생 시 즉시 프로세스 종료에 사용 */
#include <set>        /* [한국어] std::set<unsigned> — 레지스터 번호 집합 자료구조. 중복 없이 O(log n) 검색/삽입/삭제를 보장 */
#include <vector>     /* [한국어] std::vector — warp 인덱스로 접근하는 동적 배열. reg_table[wid], longopregs[wid] 형태로 사용 */
#include "assert.h"   /* [한국어] assert() 매크로 — 디버그 모드에서 불변 조건(invariant) 위반 시 프로그램을 멈춤 */

#ifndef SCOREBOARD_H_   /* [한국어] 인클루드 가드(include guard) 시작 — 이 헤더가 여러 소스 파일에서 중복 포함될 때 재정의 오류를 방지 */
#define SCOREBOARD_H_   /* [한국어] SCOREBOARD_H_ 매크로 정의 — 이후 동일 헤더 포함 시 아래 내용을 건너뜀 */

#include "../abstract_hardware_model.h"
/* [한국어] 상위 디렉토리의 GPU 하드웨어 추상 모델 헤더를 포함.
 * 이 헤더를 통해 다음 타입들이 사용 가능해진다:
 *   - inst_t: PTX/SASS 명령어 하나를 표현하는 기반 클래스 (in[], out[], incount, outcount, pred, ar1, ar2 등 포함)
 *   - warp_inst_t: inst_t를 상속하며 warp_id() 등 warp 단위 정보를 추가한 명령어 클래스
 *   - MAX_OUTPUT_VALUES: 명령어당 최대 출력 레지스터 수 (= 8, 벡터 연산 확장 후 기준)
 *   - _memory_space_t 열거형: global_space, local_space, tex_space 등 메모리 공간 구분 */

/*
 * [한국어]
 * Scoreboard — warp 단위 레지스터 의존성 추적 클래스
 *
 * 하나의 SM(shader core)에 대응하며, 그 SM 내의 모든 warp가 현재 "쓰기 보류 중"인
 * 레지스터 번호들을 집합(set)으로 관리한다.
 *
 * 동작 원리:
 *   1. 명령어가 발행(issue)될 때 → reserveRegisters(): 출력 레지스터를 reg_table에 추가.
 *   2. 다음 명령어 발행 시도 → checkCollision(): 입출력 레지스터와 reg_table의 교집합 확인.
 *      교집합이 비어 있지 않으면 RAW 또는 WAW 해저드가 존재하므로 발행을 거부한다.
 *   3. writeback 완료 → releaseRegisters(): 해당 레지스터를 reg_table에서 제거.
 *
 * WAR(Write-After-Read) 해저드를 검사하지 않는 이유:
 *   GPGPU-Sim은 in-order issue(순서대로 발행)를 가정하므로, 읽기가 먼저 발행된 후
 *   쓰기가 발행된다. 따라서 읽기 명령어의 피연산자 레지스터는 쓰기 전에 이미 사용이
 *   완료되어 WAR은 문제가 되지 않는다.
 *
 * long-latency 연산(longopregs):
 *   전역(global)/로컬(local)/텍스처(tex) 메모리 로드는 수백 사이클이 걸리는 장거리 연산이다.
 *   이들의 목적 레지스터는 longopregs에도 별도로 추적되며, warp 스케줄러는 이를 통해
 *   해당 warp의 stall이 메모리 레이턴시 때문임을 판별하고 MIO 파이프 통계를 집계한다.
 *
 * 실행 컨텍스트: 호스트 CPU 스레드 (GPGPU-Sim 시뮬레이션 루프) 내에서 단일 스레드로 실행.
 * 동시성: 한 SM의 Scoreboard는 단일 시뮬레이션 스레드만 접근하므로 별도 락 불필요.
 */
class Scoreboard {
 public:
  /*
   * [한국어]
   * Scoreboard - SM의 스코어보드 초기화 생성자
   *
   * @sid: 이 스코어보드가 속한 SM(shader core)의 ID.
   *        shader_core_ctx::m_sid와 동일한 값. 디버그 출력 식별에 사용.
   * @n_warps: 이 SM이 관리하는 warp의 총 수.
   *            gpgpusim.config의 max_warps_per_shader 값에 해당.
   *            reg_table과 longopregs 벡터의 크기를 결정한다.
   * @gpu: GPGPU-Sim 전체 시뮬레이터 컨텍스트 포인터.
   *        디버그 플래그(SHADER_DPRINTF) 접근에 사용.
   *
   * SM이 생성될 때(shader_core_ctx 생성자) 단 한 번 호출된다.
   * reg_table과 longopregs를 n_warps 크기로 초기화하며, 각 warp의 집합은 비어 있는 상태로 시작.
   *
   * 호출 체인:
   *   shader_core_ctx::shader_core_ctx() → [Scoreboard::Scoreboard()] (scoreboard.cc)
   */
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);

  /*
   * [한국어]
   * reserveRegisters - 명령어 발행 직후 출력 레지스터를 "쓰기 보류" 집합에 등록
   *
   * @inst: 방금 발행된 warp 명령어. inst->out[0..MAX_OUTPUT_VALUES-1]에
   *         쓸 레지스터 번호가, inst->warp_id()에 warp ID가 들어 있다.
   *
   * 명령어가 파이프라인에 발행(issue)된 직후 호출되어, 그 명령어가 쓸 목적 레지스터를
   * reg_table[warp_id]에 삽입한다. 이후 같은 warp의 다른 명령어가 이 레지스터를
   * checkCollision()에서 조회하면 충돌(hazard)로 탐지된다.
   * 전역/로컬/텍스처 메모리 로드 명령어의 경우에는 longopregs에도 추가로 등록하여
   * long-latency stall임을 표시한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → shader_core_ctx::issue_warp() →
   *   [Scoreboard::reserveRegisters()] → reserveRegister(wid, regnum) (private)
   */
  void reserveRegisters(const warp_inst_t *inst);

  /*
   * [한국어]
   * releaseRegisters - writeback 완료 시 출력 레지스터를 "쓰기 보류" 집합에서 제거
   *
   * @inst: writeback이 완료된 warp 명령어. inst->out[]과 inst->warp_id()로
   *         어떤 warp의 어떤 레지스터를 해제할지 결정한다.
   *
   * 파이프라인의 writeback(WB) 단계에서 실행 유닛(SP, SFU, LD/ST unit 등)이
   * 결과를 레지스터 파일에 쓴 후 호출된다. reg_table[warp_id]에서 해당 레지스터를 제거하고,
   * longopregs[warp_id]에서도 제거하여 long-latency 플래그를 해제한다.
   * 이 함수가 호출된 후에야 같은 warp에서 해당 레지스터를 읽는 명령어가 발행 가능해진다.
   *
   * 호출 체인:
   *   ldst_unit::writeback() 또는 exec_unit::writeback()
   *   → [Scoreboard::releaseRegisters()] → releaseRegister(wid, regnum) (private)
   */
  void releaseRegisters(const warp_inst_t *inst);

  /*
   * [한국어]
   * releaseRegister - 단일 레지스터를 "쓰기 보류" 집합에서 제거
   *
   * @wid: 해제할 레지스터를 소유한 warp의 ID.
   * @regnum: 해제할 레지스터 번호.
   *
   * 메모리 load 명령어처럼 출력 레지스터가 여러 사이클에 걸쳐 하나씩 완료되는 경우,
   * 또는 메모리 응답(mem_fetch) 처리 시 개별 레지스터를 해제해야 할 때 직접 호출된다.
   * scoreboard.cc의 구현이 직접 reg_table[wid].erase(regnum)을 수행한다.
   * longopregs는 건드리지 않으므로 long-latency 스톨 표시는 해제되지 않는다.
   * longopregs까지 함께 해제해야 하는 경우에는
   * releaseRegisters(const warp_inst_t*)를 사용해야 한다.
   *
   * 호출 체인:
   *   ldst_unit::L1_latency_queue_cycle() 또는 ldst_unit::writeback()
   *   → [Scoreboard::releaseRegister()] → reg_table[wid].erase(regnum)
   */
  void releaseRegister(unsigned wid, unsigned regnum);

  /*
   * [한국어]
   * checkCollision - 새 명령어의 레지스터가 보류 집합과 충돌(hazard)하는지 검사
   *
   * @wid: 발행 후보 명령어를 실행할 warp의 ID.
   * @inst: 발행 후보 명령어. inst->out[], inst->in[], inst->pred, inst->ar1, inst->ar2
   *         등의 레지스터 번호를 모두 모아 reg_table[wid]과 교집합을 구한다.
   * @return: true  → 충돌 존재 (RAW 또는 WAW 해저드). 이 warp의 발행을 이번 사이클에 거부.
   *           false → 충돌 없음. 이 명령어는 발행 가능.
   *
   * warp 스케줄러(scheduler_unit::cycle())가 매 사이클 발행 가능 warp를 탐색할 때 호출한다.
   * 명령어의 모든 입력/출력/조건/주소 레지스터를 하나의 집합으로 합쳐서,
   * 이미 예약된 레지스터(reg_table[wid])와 공통 원소가 있는지 확인한다.
   * 하나라도 겹치면 즉시 true를 반환하고, 완전히 겹치지 않으면 false를 반환한다.
   * WAR은 in-order issue 가정 하에 발생하지 않으므로 별도로 검사하지 않는다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [Scoreboard::checkCollision()] (scoreboard.cc)
   */
  bool checkCollision(unsigned wid, const inst_t *inst) const;

  /*
   * [한국어]
   * pendingWrites - 특정 warp에 미완료 쓰기가 남아 있는지 확인
   *
   * @wid: 확인할 warp의 ID.
   * @return: true  → reg_table[wid]이 비어 있지 않음. 아직 writeback 미완료 레지스터 존재.
   *           false → reg_table[wid]이 빔. 모든 in-flight 명령어가 완료.
   *
   * CTA(블록)의 실행 완료 여부를 판별하거나, warp가 파이프라인에서 완전히 빠져나갔는지
   * 확인할 때 사용한다. shader_core_ctx::warp_waiting_at_barrier()에서 동기화 판별,
   * shader_core_ctx::can_issue_1block()에서 CTA 할당 가능 여부 판별에 활용된다.
   *
   * 호출 체인:
   *   shader_core_ctx::warp_waiting_at_mem_barrier() 또는
   *   shader_core_ctx::can_issue_1block()
   *   → [Scoreboard::pendingWrites()]
   */
  bool pendingWrites(unsigned wid) const;

  /*
   * [한국어]
   * printContents - 스코어보드 상태를 표준 출력으로 덤프 (디버그용)
   *
   * 현재 SM ID와 함께 warp별 "쓰기 보류 중인 레지스터 번호" 목록을 출력한다.
   * 비어 있는 warp(pending write 없음)는 출력하지 않고 건너뛴다.
   * 시뮬레이션 디버깅, 행 감지(hang detection), 시뮬레이터 상태 덤프 시 사용된다.
   *
   * 호출 체인:
   *   shader_core_ctx::display_pipeline() → [Scoreboard::printContents()]
   */
  void printContents() const;

  /*
   * [한국어]
   * islongop - 특정 레지스터가 long-latency 연산의 목적 레지스터인지 확인
   *
   * @warp_id: 확인할 warp의 ID.
   * @regnum: 확인할 레지스터 번호.
   * @return: true  → longopregs[warp_id]에 regnum이 포함됨.
   *                   전역/로컬/텍스처 메모리 로드 결과를 기다리는 레지스터.
   *           false → 일반(단거리) 연산의 보류 레지스터이거나, 미보류 레지스터.
   *
   * warp 스케줄러가 어떤 실행 유닛(SP/SFU vs MIO/메모리 파이프)에서 stall이 발생했는지
   * 판별할 때 사용한다. long-latency stall을 별도로 집계하여 성능 분석(profiling) 통계에
   * 반영하는 역할을 한다. reserveRegisters()에서 로드 명령어일 때만 longopregs에 삽입된다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() (stall 원인 분류 통계) → [Scoreboard::islongop()]
   */
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  /*
   * [한국어]
   * reserveRegister - 단일 레지스터를 "쓰기 보류" 집합에 삽입 (내부 전용)
   *
   * @wid: 예약할 레지스터가 속한 warp의 ID.
   * @regnum: 예약할 레지스터 번호.
   *
   * reserveRegisters()에서 출력 레지스터 루프 내에서 호출되는 내부 헬퍼 함수.
   * 동일 레지스터가 이미 예약된 경우 오류 메시지를 출력하고 abort()로 프로세스를 종료한다
   * (정상 시뮬레이션에서는 이중 예약이 발생해서는 안 된다).
   * 외부에서는 reserveRegisters(const warp_inst_t*)를 통해서만 호출해야 한다.
   *
   * 호출 체인:
   *   Scoreboard::reserveRegisters() → [Scoreboard::reserveRegister()] (private)
   */
  void reserveRegister(unsigned wid, unsigned regnum);

  /*
   * [한국어]
   * get_sid - 이 스코어보드가 속한 SM의 ID 반환 (내부 전용 접근자)
   *
   * @return: m_sid — 이 객체가 속한 SM(shader core)의 정수 ID.
   *
   * 주로 내부 디버그/로깅 목적으로 사용되며, 외부에서는 직접 호출할 필요가 없다.
   * inline 함수로 구현되어 별도의 함수 호출 오버헤드 없이 m_sid를 반환한다.
   */
  int get_sid() const { return m_sid; }

  unsigned m_sid;
  /* [한국어] 이 Scoreboard가 속한 SM(Streaming Multiprocessor)의 정수 ID.
   * 설정자: 생성자(Scoreboard::Scoreboard())에서 sid 인자를 통해 단 한 번 초기화.
   * 읽는 자: get_sid() (내부), printContents() (디버그 출력의 "sid=%d" 포맷에 사용).
   * 값 범위: 0 이상, SM 개수 미만 (gpgpusim.config의 num_shader 값 미만).
   * 동기화: 초기화 이후 변경되지 않는 읽기 전용 필드 — 별도 동기화 불필요. */

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  std::vector<std::set<unsigned> > reg_table;
  /* [한국어] warp별 "쓰기 보류 중인 레지스터 번호" 집합 테이블 — 스코어보드의 핵심 상태.
   * 인덱싱: reg_table[wid]는 warp wid가 현재 파이프라인에서 실행 중이며 아직 writeback이
   *         완료되지 않은 레지스터 번호들의 집합이다.
   * 설정자: reserveRegister(wid, regnum)이 명령어 발행 시 wid 집합에 regnum을 삽입.
   * 읽는 자: checkCollision()이 발행 후보 명령어의 레지스터 집합과 교집합 확인.
   *           pendingWrites()가 warp의 모든 in-flight 완료 여부를 reg_table[wid].empty()로 판별.
   *           printContents()가 디버그 출력.
   * 값 범위: 각 집합의 원소는 레지스터 번호(unsigned). GPGPU-Sim에서 레지스터 0은
   *           "출력 없음"을 의미하므로 regnum > 0인 경우만 삽입된다.
   * 크기: 생성자에서 resize(n_warps)로 초기화. SM의 warp 수와 1:1 대응.
   * 동기화: 단일 시뮬레이션 스레드(호스트 CPU)가 단독 접근 — 별도 락 불필요. */

  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;
  /* [한국어] warp별 "long-latency 메모리 연산의 목적 레지스터" 집합.
   * long-latency 연산: 전역(global_space), 로컬(local_space), 커널 파라미터
   *   (param_space_kernel/local/unclassified), 텍스처(tex_space) 메모리 로드.
   *   이들은 L2 캐시 미스 시 수십~수백 사이클 동안 결과가 나오지 않는다.
   * 설정자: reserveRegisters()가 is_load() && 해당 메모리 공간 조건 충족 시 삽입.
   *          일반 산술/공유 메모리 연산은 삽입하지 않는다.
   * 읽는 자: islongop()이 인자로 받은 (warp_id, regnum) 쌍이 이 집합에 있는지 확인.
   *           warp 스케줄러가 stall 원인을 "메모리 레이턴시"와 "기타"로 구분하는 데 사용.
   * 값 범위: reg_table의 부분 집합 — longopregs의 원소는 항상 reg_table에도 존재.
   *           releaseRegisters()에서 reg_table과 동시에 제거된다.
   * 크기: 생성자에서 resize(n_warps)로 초기화. longopregs()로 초기화 (기본 생성자 호출).
   * 동기화: 단일 시뮬레이션 스레드 단독 접근 — 별도 락 불필요. */

  class gpgpu_t *m_gpu;
  /* [한국어] GPGPU-Sim 전체 시뮬레이터 컨텍스트 포인터.
   * 설정자: 생성자(Scoreboard::Scoreboard())에서 gpu 인자를 통해 단 한 번 초기화.
   * 읽는 자: SHADER_DPRINTF 매크로(scoreboard.cc)가 디버그 레벨/출력 스트림을 얻기 위해
   *           gpgpu_t의 설정 정보를 간접 참조한다.
   * 값 범위: 유효한 gpgpu_t 포인터 (NULL 불가). 시뮬레이터 생명주기 동안 항상 유효.
   * 동기화: 포인터 자체는 초기화 후 불변(read-only). 가리키는 객체는 시뮬레이터 루프에서
   *          별도로 관리하며, Scoreboard는 해당 객체를 수정하지 않는다. */
};

#endif /* SCOREBOARD_H_ */
/* [한국어] 인클루드 가드 종료 — #ifndef SCOREBOARD_H_와 짝을 이룬다.
 * 이 이후의 코드는 SCOREBOARD_H_가 정의된 후 다시 이 헤더를 포함해도 실행되지 않는다. */
