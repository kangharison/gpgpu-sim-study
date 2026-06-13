// Copyright (c) 2009-2011, Tor M. Aamodt,
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
 * [한국어 설명] PTX 디버거 정의 (debug.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 인터랙티브 PTX 디버거에서 사용하는 브레이크포인트(brk_pt) 클래스를
 * 정의한다. brk_pt는 두 가지 모드(라인 기반 브레이크포인트 / 메모리 워치포인트)를
 * 하나의 클래스로 통합한다. debug.cc의 gpgpu_sim::gpgpu_debug()가 이 클래스를
 * 사용하여 시뮬레이션 중단 조건을 관리한다. 이 디버거는 기능 시뮬레이션(cuda-sim/)
 * 레이어에서 실행되는 PTX 스레드를 대상으로 하며, 타이밍 모델과 직접 연관되지 않는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: CUDA 앱 → libcuda → gpgpusim_entrypoint → cuda-sim(PTX 실행)
 *   → gpgpu_sim::gpgpu_debug() → brk_pt(이 파일)
 * 인터랙티브 디버거는 매 시뮬레이션 사이클마다 gpgpu_sim::gpgpu_debug()에서
 * 호출되며, 단일-스텝 모드 또는 브레이크포인트 조건 충족 시 stdin 루프에 진입한다.
 * 실행 컨텍스트: 호스트 유저스페이스(단일 스레드, 시뮬레이션 메인 루프 내).
 *
 * === 타 모듈과의 연결 ===
 * 의존: abstract_hardware_model.h (addr_t 타입), cuda-sim/ptx_ir.h (ptx_thread_info),
 *       cuda-sim/ptx_sim.h (ptx_instruction)
 * 사용처: debug.cc (gpgpu_sim::gpgpu_debug, thread_at_brkpt)
 * 데이터 흐름: brk_pt 객체는 debug.cc 내 정적 map<unsigned, brk_pt> breakpoints에
 *   저장되며, 사이클마다 전체 워치포인트/브레이크포인트를 순회하여 조건 검사.
 *
 * === 주요 함수/구조체 요약 ===
 * brk_pt        - 브레이크포인트(파일:라인 + 스레드UID) 또는 워치포인트(주소+값) 표현
 * is_equal()    - 현재 스레드 위치가 이 브레이크포인트 조건을 만족하는지 검사
 * thread_at_brkpt() - ptx_thread_info가 브레이크포인트 조건과 일치하는지 확인하는 전역 함수
 */

#ifndef PTX_DEBUG_INCLUDED
#define PTX_DEBUG_INCLUDED

#include "abstract_hardware_model.h" /* [한국어] addr_t 등 하드웨어 추상 타입 정의 */

#include <string> /* [한국어] m_fileline 문자열 저장에 사용 */

/*
 * [한국어]
 * brk_pt - PTX 디버거용 브레이크포인트/워치포인트 통합 클래스
 *
 * @생성자 1 (기본): m_valid=false인 빈 객체 생성. map의 기본값 초기화용.
 * @생성자 2 (fileline, uid): 파일:라인 기반 브레이크포인트. m_watch=false.
 * @생성자 3 (addr, value): 메모리 주소 기반 워치포인트. m_watch=true.
 *
 * 브레이크포인트 모드: PTX 소스 파일명:줄번호와 스레드 UID를 기준으로 실행 중단.
 * 워치포인트 모드: 지정된 글로벌 메모리 주소의 값이 변경될 때 실행 중단.
 * m_watch 플래그로 두 모드를 구분하며, 모드에 따라 사용하는 필드가 다르다.
 * 실행 컨텍스트: 시뮬레이션 메인 루프(단일 스레드 호스트 유저스페이스)에서만 접근.
 *
 * 호출 체인:
 *   gpgpu_sim::gpgpu_debug() → [brk_pt 생성/is_equal/get_addr/get_value]
 *   thread_at_brkpt() → brk_pt::is_equal()
 */
class brk_pt {
 public:
  /*
   * [한국어]
   * brk_pt() - 빈 브레이크포인트 생성자 (무효 상태)
   *
   * debug.cc의 breakpoints map에 기본 초기화용으로 사용.
   * m_valid=false로 설정해 아직 설정되지 않은 슬롯임을 표시.
   */
  brk_pt() { m_valid = false; } /* [한국어] 미사용 슬롯 — m_valid=false로 비활성 표시 */
  /*
   * [한국어]
   * brk_pt(fileline, uid) - 파일:라인 기반 브레이크포인트 생성자
   *
   * @fileline: "파일명:줄번호" 형식의 문자열 — PTX 소스 위치 식별자
   * @uid: 특정 스레드 UID. (unsigned)-1이면 모든 스레드에 적용.
   *
   * debug.cc의 'b' 명령어로 생성. m_watch=false로 설정해 브레이크포인트 모드 표시.
   */
  brk_pt(const char *fileline, unsigned uid) {
    m_valid = true;        /* [한국어] 유효한 브레이크포인트로 활성화 */
    m_watch = false;       /* [한국어] 브레이크포인트 모드 (워치포인트 아님) */
    m_fileline = std::string(fileline); /* [한국어] "파일명:줄번호" 문자열 저장 */
    m_thread_uid = uid;    /* [한국어] 대상 스레드 UID 저장; (unsigned)-1이면 전체 대상 */
  }
  /*
   * [한국어]
   * brk_pt(addr, value) - 메모리 워치포인트 생성자
   *
   * @addr: 감시할 글로벌 메모리 주소 (물리/가상 모두 가능, 4바이트 정렬 권장)
   * @value: 워치포인트 설정 시점의 현재 메모리 값 (변경 감지 기준값)
   *
   * debug.cc의 'w' 명령어로 생성. m_watch=true로 설정해 워치포인트 모드 표시.
   * gpgpu_sim::gpgpu_debug()가 매 사이클 m_global_mem->read()로 현재 값과 비교.
   */
  brk_pt(unsigned addr, unsigned value) {
    m_valid = true;   /* [한국어] 유효한 워치포인트로 활성화 */
    m_watch = true;   /* [한국어] 워치포인트 모드로 설정 */
    m_addr = addr;    /* [한국어] 감시할 글로벌 메모리 주소 */
    m_value = value;  /* [한국어] 변경 감지 기준이 되는 초기 메모리 값 */
  }

  /*
   * [한국어]
   * get_value() - 워치포인트의 현재 기준값 반환
   *
   * @return: m_value — 마지막으로 확인된 메모리 값 (변경 전 값)
   *
   * gpgpu_debug()에서 m_global_mem->read() 결과와 비교하여 워치포인트 트리거 여부 결정.
   */
  unsigned get_value() const { return m_value; }
  /*
   * [한국어]
   * get_addr() - 워치포인트의 감시 대상 메모리 주소 반환
   *
   * @return: m_addr — gpgpu_sim::m_global_mem에서 read할 주소
   */
  addr_t get_addr() const { return m_addr; }
  /*
   * [한국어]
   * is_valid() - 이 brk_pt 객체가 유효한 브레이크포인트/워치포인트인지 확인
   *
   * @return: true이면 설정된 브레이크포인트, false이면 기본 생성된 빈 객체
   */
  bool is_valid() const { return m_valid; }
  /*
   * [한국어]
   * is_watchpoint() - 이 brk_pt가 워치포인트 모드인지 확인
   *
   * @return: true이면 메모리 주소 감시 모드, false이면 파일:라인 브레이크포인트 모드
   */
  bool is_watchpoint() const { return m_watch; }
  /*
   * [한국어]
   * is_equal() - 현재 스레드 위치가 이 브레이크포인트와 일치하는지 검사
   *
   * @fileline: 현재 스레드의 PTX 소스 위치 ("파일명:줄번호")
   * @uid: 현재 스레드의 고유 UID
   * @return: true이면 브레이크포인트 조건 충족 (중단 필요)
   *
   * 워치포인트 모드에서는 항상 false 반환 (파일:라인 비교 불가).
   * m_thread_uid가 (unsigned)-1이면 모든 스레드에 대해 파일:라인만 비교.
   * thread_at_brkpt() 전역 함수가 이 메서드를 호출.
   *
   * 호출 체인:
   *   gpgpu_sim::gpgpu_debug() → thread_at_brkpt() → [is_equal()]
   */
  bool is_equal(const std::string &fileline, unsigned uid) const {
    if (m_watch) return false; /* [한국어] 워치포인트 모드에서는 파일:라인 비교 불가 */
    if ((m_thread_uid != (unsigned)-1) && (uid != m_thread_uid)) return false;
    /* [한국어] 특정 스레드 UID가 지정된 경우, UID가 다르면 미일치 */
    return m_fileline == fileline; /* [한국어] 파일:라인 문자열 비교 */
  }
  /*
   * [한국어]
   * location() - 브레이크포인트 위치를 사람이 읽을 수 있는 문자열로 반환
   *
   * @return: "파일명:줄번호 thread uid = <uid>" 형식 문자열
   *
   * gpgpu_debug()에서 브레이크포인트 도달 시 콘솔 출력용.
   */
  std::string location() const {
    char buffer[1024]; /* [한국어] 출력 문자열 임시 버퍼 (1024바이트로 충분) */
    sprintf(buffer, "%s thread uid = %u", m_fileline.c_str(), m_thread_uid);
    /* [한국어] "파일:라인 thread uid = UID" 형식으로 포맷 */
    return buffer;
  }

  /*
   * [한국어]
   * set_value() - 워치포인트의 기준값 갱신
   *
   * @val: 새로운 기준값 (메모리에서 읽은 현재 값)
   * @return: 설정된 새 값
   *
   * 워치포인트 트리거 후 gpgpu_debug()가 m_value를 현재 값으로 업데이트.
   * 다음 사이클부터 갱신된 값을 기준으로 변경 감지.
   */
  unsigned set_value(unsigned val) { return m_value = val; }

 private:
  bool m_valid;
  /* [한국어] 이 brk_pt가 실제로 설정된 브레이크포인트인지 여부.
   * 설정자: 생성자(true) 또는 기본 생성자(false).
   * 읽는 자: gpgpu_debug()가 브레이크포인트 맵 순회 시 유효성 확인용.
   * 값 범위: true(설정됨) / false(미설정, 기본값).
   * 동기화: 단일 스레드에서만 접근하므로 락 불필요. */

  bool m_watch;
  /* [한국어] 브레이크포인트 모드 vs. 워치포인트 모드 구분 플래그.
   * 설정자: 생성자(brk_pt(fileline,uid)=false, brk_pt(addr,value)=true).
   * 읽는 자: is_watchpoint(), is_equal(), gpgpu_debug()의 분기 조건.
   * 값 범위: false=파일:라인 브레이크포인트, true=메모리 워치포인트.
   * 동기화: 단일 스레드 접근, 락 불필요. */

  // break point
  std::string m_fileline;
  /* [한국어] 파일:라인 브레이크포인트의 위치 식별자 ("파일명:줄번호" 형식).
   * 설정자: brk_pt(fileline, uid) 생성자.
   * 읽는 자: is_equal()에서 현재 스레드 위치와 문자열 비교.
   *          location()에서 콘솔 출력용으로 사용.
   * 값 범위: 유효한 PTX 소스 파일명과 줄번호 (예: "vecAdd.ptx:42").
   * 동기화: 단일 스레드 접근, 락 불필요. */
  unsigned m_thread_uid;
  /* [한국어] 이 브레이크포인트가 적용될 특정 스레드의 UID.
   * 설정자: brk_pt(fileline, uid) 생성자.
   * 읽는 자: is_equal()에서 특정 스레드 필터링에 사용.
   * 값 범위: 유효한 스레드 UID, 또는 (unsigned)-1이면 모든 스레드에 적용.
   * 동기화: 단일 스레드 접근, 락 불필요. */

  // watch point
  unsigned m_addr;
  /* [한국어] 워치포인트 감시 대상 글로벌 메모리 주소.
   * 설정자: brk_pt(addr, value) 생성자.
   * 읽는 자: gpgpu_debug()에서 get_addr()로 읽어 m_global_mem->read() 호출.
   * 값 범위: GPGPU-Sim 글로벌 메모리 주소 공간 내 유효한 주소.
   * 동기화: 단일 스레드 접근, 락 불필요. */
  unsigned m_value;
  /* [한국어] 워치포인트에서 변경 감지 기준이 되는 마지막으로 확인된 메모리 값(4바이트).
   * 설정자: brk_pt(addr, value) 생성자에서 초기 설정;
   *          이후 set_value()로 트리거 때마다 현재 값으로 갱신.
   * 읽는 자: gpgpu_debug()에서 get_value()로 읽어 새 값과 비교.
   * 값 범위: 임의의 32비트 값.
   * 동기화: 단일 스레드 접근, 락 불필요. */
};

class ptx_thread_info;  /* [한국어] PTX 스레드 상태 클래스 전방 선언 (cuda-sim/ptx_ir.h) */
class ptx_instruction;  /* [한국어] PTX 명령어 클래스 전방 선언 (cuda-sim/ptx_ir.h) */
/*
 * [한국어]
 * thread_at_brkpt() - PTX 스레드가 브레이크포인트 조건에 해당하는지 확인
 *
 * @thd_info: 검사할 PTX 스레드 정보 포인터 (현재 PC/위치/UID 보유)
 * @b: 비교할 브레이크포인트 객체
 * @return: true이면 스레드가 브레이크포인트 조건 충족 (중단 필요)
 *
 * debug.cc에서 구현. b.is_equal(thread->get_location(), thread->get_uid())를 호출.
 * 워치포인트 모드의 brk_pt에는 false를 반환(is_equal 내부에서 처리).
 *
 * 호출 체인:
 *   gpgpu_sim::gpgpu_debug() → [thread_at_brkpt()] → brk_pt::is_equal()
 */
bool thread_at_brkpt(ptx_thread_info *thd_info, const class brk_pt &b);

#endif
