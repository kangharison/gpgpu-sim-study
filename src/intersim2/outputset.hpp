// $Id: outputset.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * [한국어 설명] BookSim NoC 라우터 출력 포트/VC 집합 헤더 (outputset.hpp)
 *
 * === 파일의 역할 ===
 * NoC(Network-on-Chip) 라우터에서 플릿(flit)이 다음 홉으로 전달될 수 있는
 * 출력 포트(output port)와 가상 채널(VC: Virtual Channel) 후보 집합을
 * 나타내는 OutputSet 클래스를 선언한다.
 * 라우팅 알고리즘이 플릿의 가능한 출력 후보들을 이 집합에 등록하면,
 * VC 할당기(VC allocator)가 집합에서 실제로 사용할 포트/VC를 선택한다.
 * 각 후보 항목(sSetElement)은 출력 포트, VC 범위, 우선순위(pri)를 가지며
 * std::set을 사용하여 우선순위 내림차순으로 정렬 저장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 라우터 파이프라인의 라우팅 단계(Route Computation)와 VC 할당 단계 사이에 위치한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — 매 시뮬레이션 사이클마다 라우터가 호출.
 * 호출 체인:
 *   Router::_Route() (라우팅 알고리즘) → OutputSet::Add/AddRange()
 *   → Router::_VCAlloc() (VC 할당) → OutputSet::GetSet() / GetVC()
 * 각 VC 객체는 자신의 OutputSet을 소유하며, 라우팅 단계마다 Clear() 후 재사용된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: <set> (std::set으로 _outputs 저장), booksim.hpp (공통 타입).
 * 피의존: VC 클래스 (각 VC가 OutputSet을 멤버로 보유),
 *         라우팅 함수 routefunc.cc (Add/AddRange 호출),
 *         VC 할당기 allocator.cc (GetSet/GetVC 호출).
 * 데이터 흐름: 라우팅 알고리즘 → [OutputSet에 후보 추가] → VC 할당기 → 출력 포트/VC 결정.
 * 공유 자료구조: sSetElement 구조체, _outputs std::set.
 *
 * === 주요 함수/구조체 요약 ===
 * sSetElement          - 출력 후보 항목: output_port, vc_start, vc_end, pri
 * Clear()              - 출력 집합 초기화 (매 라우팅 단계 시작 시 호출)
 * Add(port, vc, pri)   - 단일 VC를 출력 후보로 추가
 * AddRange(port, s, e) - VC 범위를 출력 후보로 추가
 * GetSet()             - 내부 _outputs set의 const 참조 반환 (고성능 직접 접근)
 * GetVC(port, index)   - 레거시: 인덱스로 VC 번호 조회
 * GetPortVC(p, vc)     - 레거시: 단일 포트/VC 쌍인지 확인하고 반환
 * NumVCs(port)         - 레거시: 특정 포트의 VC 후보 수 반환
 * OutputEmpty(port)    - 특정 포트에 대한 후보가 없는지 확인
 */

#ifndef _OUTPUTSET_HPP_ // [한국어] 헤더 중복 포함 방지 가드 시작
#define _OUTPUTSET_HPP_ // [한국어] 가드 매크로 정의

#include <set> // [한국어] std::set: 우선순위 정렬된 출력 후보 집합(_outputs) 저장에 사용

/*
 * [한국어]
 * OutputSet - NoC 라우터에서 플릿의 가능한 출력 포트/VC 후보 집합 클래스.
 *
 * 라우팅 알고리즘이 플릿에 대한 가능한 다음 홉(출력 포트 + VC 범위)을
 * 이 클래스에 등록하면, VC 할당기가 실제로 사용할 포트/VC를 결정한다.
 * 내부적으로 std::set<sSetElement>을 사용하며, operator<가 우선순위 내림차순으로
 * 정렬하므로 높은 우선순위 항목이 집합의 앞에 위치한다.
 * 각 VC 객체가 OutputSet 인스턴스를 하나씩 보유하며, 매 라우팅 사이클마다
 * Clear() → Add/AddRange()로 갱신된다.
 *
 * 동기화: 단일 라우터 인스턴스는 단일 시뮬레이션 스레드에서만 접근되므로
 *         별도 락이 없다.
 */
class OutputSet {

public:
  /*
   * [한국어]
   * sSetElement - OutputSet에 저장되는 출력 후보 항목 구조체.
   *
   * 한 항목은 특정 output_port로 가는 [vc_start, vc_end] 범위의 VC와
   * 그 우선순위(pri)를 나타낸다. std::set에 의해 operator<(pri 내림차순)로 정렬된다.
   * 라우팅 알고리즘이 여러 후보를 등록할 때 각각 하나의 sSetElement가 생성된다.
   */
  struct sSetElement {
    int vc_start;
    /* [한국어] 이 후보에서 사용 가능한 VC 번호 범위의 시작값.
     * 설정자: AddRange() 호출 시 파라미터로 설정.
     * 읽는 자: GetVC(), NumVCs(), GetPortVC(), VC 할당기.
     * 값 범위: 0 이상의 정수; vc_end 이하여야 함.
     * 동기화: 한 시뮬레이션 스레드에서만 접근, 락 불필요. */

    int vc_end;
    /* [한국어] 이 후보에서 사용 가능한 VC 번호 범위의 끝값 (포함).
     * [vc_start, vc_end] 범위가 이 항목이 커버하는 VC 집합을 나타낸다.
     * 설정자: AddRange() 호출 시 파라미터로 설정.
     * 읽는 자: GetVC(), NumVCs() (vc_end - vc_start + 1로 범위 계산), GetPortVC().
     * 값 범위: vc_start 이상의 정수. Add() 호출 시 vc_start == vc_end (단일 VC).
     * 동기화: 한 시뮬레이션 스레드에서만 접근, 락 불필요. */

    int pri;
    /* [한국어] 이 후보 항목의 우선순위값. 높을수록 std::set 내 앞에 위치한다.
     * operator<에서 se1.pri > se2.pri로 비교하여 높은 우선순위가 먼저 오도록 한다.
     * 설정자: Add() 또는 AddRange() 호출 시 pri 파라미터로 설정 (기본값 0).
     * 읽는 자: operator<(std::set 정렬에 사용), GetVC()(pri 포인터 파라미터로 반환).
     * 값 범위: 임의 정수; 기본값 0. 적응형 라우팅에서 경로 품질을 나타낼 수 있음.
     * 동기화: 한 시뮬레이션 스레드에서만 접근, 락 불필요. */

    int output_port;
    /* [한국어] 이 후보의 출력 포트 번호 (라우터의 물리적 출력 포트 인덱스).
     * 라우터의 출력 포트는 인접 라우터 또는 종단 노드(injection/ejection)로의 연결에 해당한다.
     * 설정자: Add() 또는 AddRange() 호출 시 output_port 파라미터로 설정.
     * 읽는 자: NumVCs(), OutputEmpty(), GetVC(), GetPortVC() — 포트로 필터링 시 사용.
     * 값 범위: 0 이상의 정수; 라우터의 실제 포트 수 미만이어야 함.
     * 동기화: 한 시뮬레이션 스레드에서만 접근, 락 불필요. */
  };

  /*
   * [한국어]
   * Clear - 출력 후보 집합을 모두 비운다.
   *
   * @return: 없음 (void).
   *
   * 매 라우팅 단계 시작 시 호출하여 이전 사이클의 후보를 제거한다.
   * 이후 Add() / AddRange()로 새 후보를 등록한다.
   * 실행 컨텍스트: 호스트 CPU, 매 시뮬레이션 사이클의 라우팅 단계.
   *
   * 호출 체인:
   *   Router::_Route() → vc->outputs.Clear() → [Clear()]
   */
  void Clear( );

  /*
   * [한국어]
   * Add - 단일 VC를 출력 후보로 등록한다.
   *
   * @output_port: 출력 포트 번호.
   * @vc:          등록할 VC 번호 (단일 값).
   * @pri:         우선순위 (기본값 0; 높을수록 std::set에서 앞에 위치).
   * @return:      없음 (void).
   *
   * AddRange(output_port, vc, vc, pri)를 호출하는 래퍼 함수이다.
   * 결정론적 라우팅처럼 정확히 하나의 포트/VC를 지정할 때 사용한다.
   * 실행 컨텍스트: 호스트 CPU, 매 시뮬레이션 사이클의 라우팅 단계.
   *
   * 호출 체인:
   *   라우팅 함수(routefunc.cc) → [Add(port, vc, pri)] → AddRange()
   */
  void Add( int output_port, int vc, int pri = 0 );

  /*
   * [한국어]
   * AddRange - VC 범위를 출력 후보로 등록한다.
   *
   * @output_port: 출력 포트 번호.
   * @vc_start:    VC 범위 시작값 (포함).
   * @vc_end:      VC 범위 끝값 (포함). vc_start == vc_end이면 단일 VC.
   * @pri:         우선순위 (기본값 0).
   * @return:      없음 (void).
   *
   * sSetElement를 생성하여 _outputs set에 삽입한다.
   * 적응형 라우팅에서 여러 출력 포트 또는 VC 범위를 한 번에 등록할 때 사용한다.
   * 실행 컨텍스트: 호스트 CPU, 매 시뮬레이션 사이클의 라우팅 단계.
   *
   * 호출 체인:
   *   라우팅 함수(routefunc.cc) / Add() → [AddRange(port, s, e, pri)] → _outputs.insert()
   */
  void AddRange( int output_port, int vc_start, int vc_end, int pri = 0 );

  /*
   * [한국어]
   * OutputEmpty - 특정 출력 포트에 대한 후보 항목이 없는지 확인한다.
   *
   * @output_port: 확인할 출력 포트 번호.
   * @return:      해당 포트의 후보가 없으면 true, 하나 이상 있으면 false.
   *
   * VC 할당기가 특정 포트로의 경로가 라우팅 결과에 포함되었는지 확인할 때 사용한다.
   * 실행 컨텍스트: 호스트 CPU, 매 사이클 VC 할당 단계.
   *
   * 호출 체인:
   *   VC 할당기 → [OutputEmpty(port)] → true/false 반환
   */
  bool OutputEmpty( int output_port ) const;

  /*
   * [한국어]
   * NumVCs - 특정 출력 포트에 대해 등록된 VC의 총 수를 반환한다.
   *
   * @output_port: 확인할 출력 포트 번호.
   * @return:      해당 포트에 등록된 VC 후보의 총 수 (범위 합산).
   *
   * 레거시 지원 함수. 성능을 위해서는 GetSet()을 직접 사용하는 것이 권장된다.
   * _outputs를 선형 순회하므로 O(n) 시간 복잡도를 가진다.
   * 실행 컨텍스트: 호스트 CPU.
   *
   * 호출 체인:
   *   할당기 / 라우팅 검증 코드 → [NumVCs(port)] → 정수 반환
   */
  int NumVCs( int output_port ) const;

  /*
   * [한국어]
   * GetSet - 내부 _outputs set의 const 참조를 반환한다.
   *
   * @return: set<sSetElement>의 const 참조. 복사 없이 고성능으로 순회 가능.
   *
   * VC 할당기와 라우팅 코드가 후보 집합을 순회할 때 직접 사용하는 권장 인터페이스.
   * 레거시 함수(GetVC, NumVCs, GetPortVC)보다 성능이 우수하다.
   * 실행 컨텍스트: 호스트 CPU, 매 사이클 VC 할당 단계.
   *
   * 호출 체인:
   *   VC 할당기 / Router → [GetSet()] → set 순회
   */
  const set<sSetElement> & GetSet() const;

  /*
   * [한국어]
   * GetVC - 특정 출력 포트에서 인덱스로 VC 번호를 조회한다 (레거시).
   *
   * @output_port: 조회할 출력 포트 번호.
   * @vc_index:    0-based 인덱스 (해당 포트의 모든 VC 후보 중 vc_index번째).
   * @pri:         NULL이 아니면 해당 항목의 우선순위를 저장 (출력 파라미터).
   * @return:      해당 인덱스의 VC 번호. 범위 초과 시 -1.
   *
   * 레거시 지원 함수. 성능을 위해서는 GetSet()을 직접 사용하는 것이 권장된다.
   * _outputs를 선형 순회하여 output_port 일치 항목의 범위를 누적 계산한다.
   * 실행 컨텍스트: 호스트 CPU.
   *
   * 호출 체인:
   *   레거시 할당기 코드 → [GetVC(port, index, &pri)] → VC 번호 반환
   */
  int  GetVC( int output_port,  int vc_index, int *pri = 0 ) const;

  /*
   * [한국어]
   * GetPortVC - 집합이 단일 포트/VC 쌍으로만 구성되었는지 확인하고 그 값을 반환한다 (레거시).
   *
   * @out_port: (출력) 단일 포트 번호가 저장될 포인터.
   * @out_vc:   (출력) 단일 VC 번호가 저장될 포인터.
   * @return:   집합 내 모든 항목이 동일 포트의 단일 VC(vc_start==vc_end)이면 true,
   *            여러 포트 또는 VC 범위가 있으면 false.
   *
   * 레거시 지원 함수. 성능을 위해서는 GetSet()을 직접 사용하는 것이 권장된다.
   * 결정론적 라우팅에서 단일 출력이 확정된 경우를 빠르게 판별할 때 사용된다.
   * 실행 컨텍스트: 호스트 CPU.
   *
   * 호출 체인:
   *   레거시 할당기 코드 → [GetPortVC(&port, &vc)] → true/false 반환
   */
  bool GetPortVC( int *out_port, int *out_vc ) const;

private:
  set<sSetElement> _outputs;
  /* [한국어] 라우팅 알고리즘이 등록한 출력 포트/VC 후보 항목들의 정렬 집합.
   * operator<(pri 내림차순)로 정렬되어 높은 우선순위 항목이 begin()에 위치한다.
   * 설정자: AddRange()가 insert()로 항목 추가; Clear()가 전체 삭제.
   * 읽는 자: GetSet(), GetVC(), NumVCs(), OutputEmpty(), GetPortVC().
   * 값 범위: 0개 이상의 sSetElement 항목.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근, 별도 락 불필요. */
};

/*
 * [한국어]
 * operator< - sSetElement 두 항목의 우선순위 비교 연산자 (std::set 정렬 기준).
 *
 * @se1: 비교 대상 첫 번째 sSetElement.
 * @se2: 비교 대상 두 번째 sSetElement.
 * @return: se1.pri > se2.pri이면 true; 즉, 높은 pri가 "작은" 것으로 간주되어
 *          std::set의 앞에 배치된다 (우선순위 내림차순 정렬).
 *
 * std::set은 기본적으로 오름차순 정렬하므로, pri가 높은 항목이 앞에 오려면
 * "pri가 높을수록 작다"는 역방향 비교가 필요하다.
 * 이 operator<는 set 자체의 정렬에만 사용되며 실제 대소 관계를 의미하지 않는다.
 * 실행 컨텍스트: set::insert() 호출 시 자동으로 사용.
 *
 * 호출 체인:
 *   AddRange() → _outputs.insert() → [operator<] (std::set이 내부적으로 호출)
 */
inline bool operator<(const OutputSet::sSetElement & se1,
	       const OutputSet::sSetElement & se2) {
  return se1.pri > se2.pri; // higher priorities first!
  // [한국어] se1.pri > se2.pri이면 se1이 se2보다 "작다"고 판정하여
  // [한국어] std::set에서 se1이 se2 앞에 배치됨 → 높은 우선순위가 집합 앞에 위치
}

#endif // [한국어] _OUTPUTSET_HPP_ 헤더 가드 끝
