// $Id: injection.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 트래픽 주입 프로세스 선언 (injection.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 NoC 시뮬레이터에서 합성 트래픽(synthetic traffic)을 생성하는
 * 주입 프로세스(Injection Process) 클래스 계층 구조를 선언한다. 주입 프로세스란
 * "이번 사이클에 소스 노드 n에서 패킷을 새로 생성할지 여부"를 결정하는 모델이다.
 * Booksim2의 단독(standalone) 벤치마크 모드에서 사용되며, GPGPU-Sim 통합 모드에서는
 * 실제 GPU 워크로드 패킷이 icnt_push()를 통해 직접 주입되므로 이 클래스는 사용되지
 * 않는다. 그럼에도 Booksim2 라이브러리 코드가 이를 포함하므로 함께 컴파일된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Booksim2 단독 시뮬레이션 루프(main.cpp)에서 트래픽 주입 단계에 해당한다.
 *
 * 실행 흐름 (단독 모드):
 *   main() → 매 사이클 → TrafficManager::_Step() →
 *   InjectionProcess::test(source) → true이면 Flit 생성 후 네트워크에 주입
 *
 * GPGPU-Sim 통합 시:
 *   이 파일의 클래스들은 사용되지 않음. 대신 icnt_push()가 패킷을 직접 주입.
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — Booksim2 합성 벤치마크 단일 스레드
 *
 * === 타 모듈과의 연결 ===
 * - config_utils.hpp: Configuration 클래스 — burst_alpha, burst_beta 등 파라미터 읽기
 * - traffic_manager.cpp: TrafficManager::_Step()에서 InjectionProcess::test()를 호출하여
 *   패킷 생성 여부를 결정한다.
 * - random_utils.hpp: RandomFloat(), RandomInt() — 확률적 패킷 생성에 사용
 * - injection.cpp: 이 헤더의 모든 클래스를 구현
 *
 * === 주요 함수/구조체 요약 ===
 * - InjectionProcess           : 추상 기반 클래스 — nodes, rate, test()/reset()/New() 제공
 * - BernoulliInjectionProcess  : 베르누이 과정 — 매 사이클 rate 확률로 독립 패킷 생성
 * - OnOffInjectionProcess      : ON/OFF 버스트 과정 — 상태 기반(ON/OFF) 클러스터 트래픽
 * - InjectionProcess::New()    : 설정 문자열을 파싱하여 적절한 서브클래스 인스턴스 생성
 */

#ifndef _INJECTION_HPP_
#define _INJECTION_HPP_

#include "config_utils.hpp" // [한국어] Configuration 클래스 포함 — burst_alpha 등 파라미터 읽기용

using namespace std; // [한국어] string, vector 등 STL 컨테이너를 네임스페이스 없이 사용

/*
 * [한국어] InjectionProcess — NoC 트래픽 주입 프로세스 추상 기반 클래스
 *
 * "이번 사이클에 소스 노드에서 패킷을 생성할지 여부"를 결정하는 인터페이스를 정의한다.
 * 순수 가상 함수 test(source)를 서브클래스가 구현하여 각각의 트래픽 모델을 제공한다.
 *
 * 설계 패턴: Strategy 패턴 — 트래픽 모델을 런타임에 교체 가능.
 * New() 팩토리 메서드가 설정 문자열을 파싱하여 적절한 서브클래스를 생성한다.
 */
class InjectionProcess {
protected:
  int _nodes;
  /* [한국어] 이 주입 프로세스가 관리하는 소스 노드 총 수.
   * 설정자: 생성자에서 nodes 파라미터로 초기화됨.
   * 읽는 자: test() 구현에서 source 범위 검증 및 노드별 상태 배열 크기 결정에 사용.
   * 값 범위: 1 이상의 양의 정수. 일반적으로 gNodes와 동일.
   * 동기화: 생성 후 읽기 전용 — 별도 락 불필요. */

  double _rate;
  /* [한국어] 목표 패킷 주입률 (단위: 패킷/사이클/노드, 0.0~1.0).
   * 예: _rate=0.5이면 각 노드가 평균적으로 사이클의 50%에서 패킷을 생성.
   * 설정자: 생성자에서 rate 파라미터로 초기화됨. 유효 범위는 [0.0, 1.0].
   * 읽는 자: BernoulliInjectionProcess::test()에서 RandomFloat()와 비교,
   *          OnOffInjectionProcess 생성자에서 alpha/beta/r1 도출 공식에 사용.
   * 값 범위: 0.0(주입 없음) ~ 1.0(매 사이클 항상 주입).
   * 동기화: 생성 후 읽기 전용 — 별도 락 불필요. */

  /*
   * [한국어]
   * InjectionProcess - 기반 클래스 생성자.
   *
   * @param nodes: 관리할 소스 노드 수 (양의 정수여야 함, 아니면 exit(-1))
   * @param rate:  목표 주입률 [0.0, 1.0] (범위 벗어나면 exit(-1))
   *
   * 공통 파라미터를 초기화하고 유효성을 검증한다.
   * protected로 선언되어 직접 인스턴스화 불가 — 서브클래스 생성자에서만 호출.
   *
   * 호출 체인:
   *   BernoulliInjectionProcess() / OnOffInjectionProcess() →
   *   InjectionProcess(nodes, rate)
   */
  InjectionProcess(int nodes, double rate);

public:

  /*
   * [한국어]
   * ~InjectionProcess - 가상 소멸자.
   *
   * 서브클래스 객체를 기반 클래스 포인터로 delete할 때 서브클래스 소멸자가
   * 올바르게 호출되도록 virtual로 선언한다.
   */
  virtual ~InjectionProcess() {}

  /*
   * [한국어]
   * test - 이번 사이클에 지정 소스 노드에서 패킷을 생성할지 결정하는 순수 가상 함수.
   *
   * @param source: 쿼리 대상 소스 노드 번호 (0 이상 _nodes 미만)
   * @return: bool — true이면 이번 사이클에 패킷 생성, false이면 생성하지 않음.
   *
   * 매 사이클 TrafficManager::_Step()이 각 소스 노드에 대해 호출한다.
   * 서브클래스가 각 트래픽 모델에 맞게 구현한다.
   *
   * 호출 체인:
   *   TrafficManager::_Step() → InjectionProcess::test(source) →
   *   true이면 Flit 생성 → 네트워크 주입
   */
  virtual bool test(int source) = 0;

  /*
   * [한국어]
   * reset - 주입 프로세스를 초기 상태로 재설정한다 (기본 구현은 빈 함수).
   *
   * @return: void
   *
   * OnOffInjectionProcess처럼 내부 상태(state)를 가지는 서브클래스는
   * 이 함수를 override하여 상태를 초기값(_initial)으로 되돌린다.
   * 기반 클래스 구현은 아무 것도 하지 않는다.
   *
   * 호출 체인:
   *   TrafficManager (시뮬레이션 재시작 시) → InjectionProcess::reset()
   */
  virtual void reset();

  /*
   * [한국어]
   * New - 설정 문자열을 파싱하여 적절한 InjectionProcess 서브클래스 인스턴스를 생성한다.
   *
   * @param inject: 주입 프로세스 유형과 파라미터를 담은 문자열
   *                (예: "bernoulli", "on_off(0.1,0.9)", "on_off(0.1,0.9,0.5)")
   * @param nodes:  소스 노드 수
   * @param load:   목표 주입 부하율 (0.0~1.0)
   * @param config: Configuration 포인터 — 파라미터를 config 파일에서 읽을 때 사용
   *                (NULL이면 inject 문자열의 인라인 파라미터만 사용)
   * @return: InjectionProcess* — 새로 할당된 서브클래스 인스턴스. 호출자가 delete 책임.
   *
   * 팩토리 메서드(Factory Method) 패턴. inject 문자열에서 프로세스 이름과
   * 파라미터를 분리하여 "bernoulli" → BernoulliInjectionProcess,
   * "on_off" → OnOffInjectionProcess를 생성한다.
   * 알 수 없는 프로세스 이름이면 에러 메시지 출력 후 exit(-1).
   *
   * 호출 체인:
   *   TrafficManager 초기화 → InjectionProcess::New("bernoulli", nodes, load) →
   *   new BernoulliInjectionProcess(nodes, load)
   */
  static InjectionProcess * New(string const & inject, int nodes, double load,
				Configuration const * const config = NULL);
};

/*
 * [한국어] BernoulliInjectionProcess — 베르누이 과정 기반 패킷 주입 클래스
 *
 * 베르누이(Bernoulli) 과정은 매 사이클 독립적으로 rate 확률로 패킷을 생성하는
 * 가장 단순한 트래픽 모델이다. 각 소스 노드는 서로 독립적이며, 이전 사이클의
 * 패킷 생성 여부가 다음 사이클에 영향을 주지 않는다(무기억성/memoryless).
 * 메모리 액세스 패턴이 균일하게 분포할 때의 이상적인 모델에 해당한다.
 *
 * 수학적 모델: P(패킷 생성) = rate (매 사이클, 각 노드 독립)
 */
class BernoulliInjectionProcess : public InjectionProcess {
public:
  /*
   * [한국어]
   * BernoulliInjectionProcess - 베르누이 주입 프로세스 생성자.
   *
   * @param nodes: 소스 노드 수
   * @param rate:  패킷 생성 확률 [0.0, 1.0]
   *
   * 기반 클래스 InjectionProcess(nodes, rate)를 초기화하는 것 외에
   * 추가 상태가 없으므로 생성자 본문이 비어 있다.
   *
   * 호출 체인:
   *   InjectionProcess::New("bernoulli", nodes, load) →
   *   new BernoulliInjectionProcess(nodes, load) → InjectionProcess(nodes, rate)
   */
  BernoulliInjectionProcess(int nodes, double rate);

  /*
   * [한국어]
   * test - 이번 사이클에 source 노드에서 베르누이 과정으로 패킷 생성 여부를 결정한다.
   *
   * @param source: 소스 노드 번호 (0 이상 _nodes 미만)
   * @return: bool — RandomFloat() < _rate 이면 true (이번 사이클에 패킷 생성)
   *
   * 매 호출마다 독립적인 균일 난수 [0, 1)을 생성하여 _rate와 비교한다.
   * 이전 상태 없이 순수하게 확률적으로 결정되는 가장 단순한 구현이다.
   *
   * 호출 체인:
   *   TrafficManager::_Step() → BernoulliInjectionProcess::test(source)
   */
  virtual bool test(int source);
};

/*
 * [한국어] OnOffInjectionProcess — ON/OFF 버스트 과정 기반 패킷 주입 클래스
 *
 * ON/OFF 버스트(bursty) 트래픽 모델이다. 각 소스 노드는 ON 상태(활성, 패킷 생성 가능)와
 * OFF 상태(비활성, 패킷 생성 안 함) 사이를 확률적으로 전이한다.
 * ON 상태에서 r1 확률로 실제 패킷을 생성하며, 매 사이클 상태 전이가 발생한다.
 *
 * 상태 전이 확률:
 *   OFF → ON: alpha (사이클당 ON 상태로 전이할 확률)
 *   ON  → OFF: beta  (사이클당 OFF 상태로 전이할 확률)
 *   ON 상태에서 패킷 생성 확률: r1
 *
 * 목표 주입률 _rate는 alpha, beta, r1의 조합으로 결정된다:
 *   _rate = r1 * alpha / (alpha + beta)
 * 세 파라미터 중 두 개가 주어지면 나머지 하나는 위 공식에서 자동 계산된다.
 *
 * GPU 워크로드에서 메모리 버스트(캐시 미스 폭발 등) 특성을 모델링하기에 적합하다.
 */
class OnOffInjectionProcess : public InjectionProcess {
private:
  double _alpha;
  /* [한국어] OFF→ON 전이 확률 (사이클당, 각 노드 독립).
   * 설정자: 생성자에서 직접 파라미터로 받거나, beta와 r1로부터 역산.
   *         역산 공식: _alpha = beta * rate / (r1 - rate)
   * 읽는 자: test()의 상태 전이 계산 — OFF→ON 전이 결정에 사용.
   * 값 범위: [0.0, 1.0]. alpha가 클수록 OFF 상태에서 빨리 ON으로 전환.
   * 동기화: 생성 후 읽기 전용 — 별도 락 불필요. */

  double _beta;
  /* [한국어] ON→OFF 전이 확률 (사이클당, 각 노드 독립).
   * 설정자: 생성자에서 직접 파라미터로 받거나, alpha와 r1로부터 역산.
   *         역산 공식: _beta = alpha * (r1 - rate) / rate
   * 읽는 자: test()의 상태 전이 계산 — ON→OFF 전이 결정에 사용.
   * 값 범위: [0.0, 1.0]. beta가 클수록 ON 상태에서 빨리 OFF로 전환.
   * 동기화: 생성 후 읽기 전용 — 별도 락 불필요. */

  double _r1;
  /* [한국어] ON 상태에서 실제 패킷을 생성할 확률 (사이클당, 각 노드 독립).
   * 설정자: 생성자에서 직접 파라미터로 받거나, alpha와 beta로부터 역산.
   *         역산 공식: _r1 = rate * (alpha + beta) / alpha
   * 읽는 자: test()에서 ON 상태일 때 RandomFloat() < _r1 비교로 패킷 생성 결정.
   * 값 범위: [0.0, 1.0]. r1=1.0이면 ON 상태에서 매 사이클 패킷 생성.
   * 동기화: 생성 후 읽기 전용 — 별도 락 불필요. */

  vector<int> _initial;
  /* [한국어] 각 노드의 초기 ON/OFF 상태 벡터 (reset() 시 복원 기준값).
   * 설정자: 생성자 파라미터에 초기 상태 리스트가 제공되면 그 값으로, 없으면
   *         RandomInt(1)로 무작위 0 또는 1로 초기화.
   * 읽는 자: reset()에서 _state = _initial로 복사하여 재현성 있는 재시작을 지원.
   * 값 범위: 각 원소는 0(OFF) 또는 1(ON). 크기 = _nodes.
   * 동기화: 생성 후 읽기 전용(reset()에서만 읽음) — 별도 락 불필요. */

  vector<int> _state;
  /* [한국어] 각 노드의 현재 ON/OFF 상태 벡터 (매 사이클 test()에서 갱신).
   * 설정자: reset()에서 _initial로 초기화, test(source)에서 해당 노드의 상태를 갱신.
   * 읽는 자: test(source)에서 _state[source] 읽어 ON/OFF 여부 판단 후 전이 결정.
   * 값 범위: 각 원소는 0(OFF) 또는 1(ON). 크기 = _nodes.
   * 동기화: 단일 시뮬레이션 스레드에서 test()가 순차 호출되므로 별도 락 불필요. */

public:
  /*
   * [한국어]
   * OnOffInjectionProcess - ON/OFF 버스트 주입 프로세스 생성자.
   *
   * @param nodes:   소스 노드 수
   * @param rate:    목표 주입률 [0.0, 1.0]
   * @param alpha:   OFF→ON 전이 확률 (음수이면 beta와 r1로 역산)
   * @param beta:    ON→OFF 전이 확률 (음수이면 alpha와 r1로 역산)
   * @param r1:      ON 상태에서 패킷 생성 확률 (음수이면 alpha와 beta로 역산)
   * @param initial: 각 노드의 초기 ON/OFF 상태 벡터
   *
   * alpha, beta, r1 세 파라미터 중 정확히 두 개가 양수여야 한다.
   * 음수인 파라미터는 자동으로 역산된다. 세 개 모두 양수이거나 두 개 이상 음수이면
   * 에러 없이 생성되지만 InjectionProcess::New()에서 사전 검증이 이루어진다.
   * 생성자 마지막에 reset()을 호출하여 _state를 _initial로 초기화한다.
   *
   * 호출 체인:
   *   InjectionProcess::New("on_off", ...) → new OnOffInjectionProcess(...)
   */
  OnOffInjectionProcess(int nodes, double rate, double alpha, double beta,
			double r1, vector<int> initial);

  /*
   * [한국어]
   * reset - ON/OFF 상태 벡터를 초기값(_initial)으로 복원한다.
   *
   * @return: void
   *
   * 시뮬레이션을 재시작할 때 각 노드의 상태를 생성 시 지정한 초기 상태로 되돌린다.
   * 재현성 있는(reproducible) 시뮬레이션 실행을 지원한다.
   *
   * 호출 체인:
   *   OnOffInjectionProcess 생성자 마지막 → reset()
   *   TrafficManager 재시작 → OnOffInjectionProcess::reset()
   */
  virtual void reset();

  /*
   * [한국어]
   * test - ON/OFF 상태를 한 사이클 전진시키고, 이번 사이클에 패킷 생성 여부를 반환한다.
   *
   * @param source: 소스 노드 번호 (0 이상 _nodes 미만)
   * @return: bool — 현재 상태가 ON이고 RandomFloat() < _r1이면 true (패킷 생성)
   *
   * 동작 순서:
   *   1. 현재 _state[source] 상태를 읽는다.
   *   2. ON(1)이면: RandomFloat() >= _beta일 때 ON 유지, 아니면 OFF로 전이.
   *      OFF(0)이면: RandomFloat() < _alpha일 때 ON으로 전이, 아니면 OFF 유지.
   *   3. 전이 후 상태가 ON(1)이고 RandomFloat() < _r1이면 패킷 생성(true 반환).
   *   상태 전이와 패킷 생성 결정이 같은 사이클에 한 번에 이루어진다.
   *
   * 호출 체인:
   *   TrafficManager::_Step() → OnOffInjectionProcess::test(source)
   */
  virtual bool test(int source);
};

#endif
