// $Id: injection.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 트래픽 주입 프로세스 구현 (injection.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 NoC 시뮬레이터의 합성 트래픽 주입 프로세스를 구현한다.
 * InjectionProcess 추상 기반 클래스, BernoulliInjectionProcess(베르누이 과정),
 * OnOffInjectionProcess(ON/OFF 버스트 과정)의 멤버 함수 구현을 포함한다.
 * GPGPU-Sim 통합 모드에서는 실제 GPU 트래픽이 icnt_push()로 직접 주입되므로
 * 이 파일의 클래스들은 활성화되지 않는다. Booksim2 standalone 벤치마크 실행 시
 * TrafficManager가 이 클래스들을 이용해 합성 트래픽을 생성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Booksim2 합성 트래픽 생성 계층:
 *   main.cpp → TrafficManager 초기화 → InjectionProcess::New() →
 *   매 사이클 TrafficManager::_Step() → InjectionProcess::test(source) →
 *   true이면 Flit 생성 → 네트워크 주입
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — Booksim2 합성 벤치마크 단일 스레드
 *
 * === 타 모듈과의 연결 ===
 * - random_utils.hpp: RandomFloat(), RandomInt() — 확률적 패킷 생성 난수 제공
 * - injection.hpp: InjectionProcess, BernoulliInjectionProcess,
 *                 OnOffInjectionProcess 클래스 선언
 * - config_utils.hpp: tokenize_str(), tokenize_int() — 문자열 파싱 유틸리티,
 *                     Configuration — burst_alpha 등 파라미터 읽기
 * - traffic_manager.cpp: InjectionProcess::test()를 매 사이클 호출하는 사용자
 *
 * === 주요 함수/구조체 요약 ===
 * - InjectionProcess::InjectionProcess()  : 기반 클래스 생성자, 파라미터 유효성 검증
 * - InjectionProcess::reset()             : 기본 구현 — 빈 함수
 * - InjectionProcess::New()               : 팩토리 메서드 — 문자열 파싱 후 서브클래스 생성
 * - BernoulliInjectionProcess::test()     : RandomFloat() < _rate 이면 패킷 생성
 * - OnOffInjectionProcess::OnOffInjectionProcess() : alpha/beta/r1 역산 및 상태 초기화
 * - OnOffInjectionProcess::reset()        : _state를 _initial로 복원
 * - OnOffInjectionProcess::test()         : 상태 전이 + 패킷 생성 결정
 */

#include <iostream>  // [한국어] cout — 에러 메시지 출력에 사용
#include <vector>    // [한국어] vector<int> initial, _state, params에 사용
#include <cassert>   // [한국어] assert() — 파라미터 범위 검증에 사용
#include <limits>    // [한국어] numeric_limits<double>::quiet_NaN() — 초기화 전 NaN 마커
#include "random_utils.hpp"  // [한국어] RandomFloat(), RandomInt() — 확률 난수 생성
#include "injection.hpp"     // [한국어] InjectionProcess 클래스 계층 선언 포함

using namespace std; // [한국어] cout, vector, string 등 STL 타입을 네임스페이스 없이 사용

/*
 * [한국어]
 * InjectionProcess::InjectionProcess - 기반 클래스 생성자.
 *
 * @param nodes: 관리할 소스 노드 수 (1 이상이어야 함)
 * @param rate:  목표 주입률 [0.0, 1.0]
 *
 * 공통 멤버 _nodes, _rate를 초기화하고 유효성을 검증한다.
 * nodes <= 0이거나 rate가 [0.0, 1.0] 범위를 벗어나면 에러 출력 후 exit(-1).
 * protected 생성자이므로 직접 인스턴스화 불가, 서브클래스 생성자에서만 호출.
 *
 * 호출 체인:
 *   BernoulliInjectionProcess(nodes, rate) → InjectionProcess(nodes, rate)
 *   OnOffInjectionProcess(nodes, rate, ...) → InjectionProcess(nodes, rate)
 */
InjectionProcess::InjectionProcess(int nodes, double rate)
  : _nodes(nodes), _rate(rate) // [한국어] 멤버 초기화 리스트로 _nodes와 _rate를 직접 초기화
{
  if(nodes <= 0) { // [한국어] 노드 수가 0 이하면 유효하지 않은 설정 — 즉시 종료
    cout << "Error: Number of nodes must be greater than zero." << endl;
    exit(-1); // [한국어] 비정상 종료 (-1) — 설정 오류로 시뮬레이션 계속 진행 불가
  }
  if((rate < 0.0) || (rate > 1.0)) { // [한국어] 주입률이 확률 범위[0,1]을 벗어난 경우 오류
    cout << "Error: Injection process must have load between 0.0 and 1.0."
	 << endl;
    exit(-1); // [한국어] 비정상 종료 (-1) — 유효하지 않은 부하율로 시뮬레이션 불가
  }
}

/*
 * [한국어]
 * InjectionProcess::reset - 기반 클래스의 기본 reset 구현 (빈 함수).
 *
 * @return: void
 *
 * 상태가 없는 베르누이 프로세스 등 서브클래스를 위한 기본 구현.
 * 상태를 가지는 OnOffInjectionProcess는 이 함수를 override한다.
 *
 * 호출 체인:
 *   TrafficManager 재시작 → InjectionProcess::reset() (BernoulliInjectionProcess의 경우)
 */
void InjectionProcess::reset()
{
  // [한국어] 기반 클래스에는 재설정할 상태가 없으므로 아무 것도 하지 않는다
}

/*
 * [한국어]
 * InjectionProcess::New - 설정 문자열을 파싱하여 적절한 서브클래스를 생성하는 팩토리.
 *
 * @param inject: 주입 프로세스 유형+파라미터 문자열 (예: "bernoulli", "on_off(0.1,0.9)")
 * @param nodes:  소스 노드 수
 * @param load:   목표 주입률 [0.0, 1.0]
 * @param config: Configuration 포인터 — burst_alpha 등 파라미터를 config에서 읽을 때 사용
 *                (NULL이면 inject 문자열에서 인라인 파라미터만 사용)
 * @return: InjectionProcess* — 새로 할당된 서브클래스 인스턴스. 호출자가 delete 책임.
 *
 * 문자열 파싱 순서:
 *   1. inject에서 '(' 이전 부분을 process_name으로, '(' ~ ')' 사이를 param_str로 분리.
 *   2. param_str을 쉼표 기준으로 tokenize하여 params 벡터 생성.
 *   3. process_name에 따라 분기:
 *      "bernoulli" → BernoulliInjectionProcess(nodes, load) 생성
 *      "on_off"    → alpha/beta/r1 파싱(또는 config에서 읽기) 후 OnOffInjectionProcess 생성
 *   4. 알 수 없는 process_name이면 에러 출력 후 exit(-1).
 *
 * on_off 파라미터 역산 규칙:
 *   alpha, beta, r1 중 정확히 두 개가 지정되고 나머지 하나는 -1 또는 NaN이어야 함.
 *   (모두 지정하거나 두 개 이상 미지정이면 "Invalid parameters" 에러)
 *
 * 호출 체인:
 *   TrafficManager 초기화 → InjectionProcess::New("bernoulli", nodes, load, config)
 */
InjectionProcess * InjectionProcess::New(string const & inject, int nodes,
					 double load,
					 Configuration const * const config)
{
  string process_name; // [한국어] inject 문자열에서 추출한 프로세스 이름 ("bernoulli" 또는 "on_off")
  string param_str;    // [한국어] inject 문자열의 괄호 안 파라미터 부분 ("0.1,0.9" 등)

  size_t left = inject.find_first_of('('); // [한국어] '(' 의 첫 위치를 찾아 이름/파라미터 분리 시작
  if(left == string::npos) { // [한국어] '('가 없으면 파라미터 없는 단순 이름 형태 (예: "bernoulli")
    process_name = inject;   // [한국어] inject 문자열 전체가 프로세스 이름
  } else {                   // [한국어] '('가 있으면 이름과 파라미터 문자열을 분리
    process_name = inject.substr(0, left); // [한국어] '(' 이전 부분이 프로세스 이름
    size_t right = inject.find_last_of(')'); // [한국어] 닫는 ')' 위치 탐색
    if(right == string::npos) {              // [한국어] ')'가 없으면 '(' 이후 전체가 파라미터
      param_str = inject.substr(left+1);     // [한국어] '(' 다음부터 끝까지를 파라미터 문자열로 사용
    } else {                                 // [한국어] ')'가 있으면 괄호 사이만 파라미터 문자열
      param_str = inject.substr(left+1, right-left-1); // [한국어] '('와 ')' 사이의 부분문자열 추출
    }
  }
  vector<string> params = tokenize_str(param_str); // [한국어] 파라미터 문자열을 쉼표로 분리하여 벡터 생성

  InjectionProcess * result = NULL; // [한국어] 생성할 서브클래스 포인터 초기화 (NULL = 미결정)

  if(process_name == "bernoulli") { // [한국어] 베르누이 과정: 매 사이클 rate 확률로 독립 패킷 생성
    result = new BernoulliInjectionProcess(nodes, load); // [한국어] 파라미터 없이 노드 수와 부하율만으로 생성
  } else if(process_name == "on_off") { // [한국어] ON/OFF 버스트 과정: 상태 기반 클러스터 트래픽
    bool missing_params = false; // [한국어] 필수 파라미터 누락 여부 추적 플래그

    double alpha = numeric_limits<double>::quiet_NaN(); // [한국어] alpha 초기값 NaN — 아직 미설정 상태 표시
    if(params.size() < 1) { // [한국어] 인라인 파라미터에 alpha가 없는 경우
      if(config) { // [한국어] Configuration 객체가 있으면 설정 파일에서 burst_alpha를 읽는다
	alpha = config->GetFloat("burst_alpha"); // [한국어] gpgpusim.config의 burst_alpha 값 읽기
      } else { // [한국어] config도 없고 인라인 파라미터도 없으면 파라미터 누락
	missing_params = true; // [한국어] 누락 플래그 설정 — 나중에 에러 처리
      }
    } else {                                  // [한국어] 인라인 파라미터에 alpha가 있는 경우
      alpha = atof(params[0].c_str()); // [한국어] params[0] 문자열을 double로 변환하여 alpha 설정
    }

    double beta = numeric_limits<double>::quiet_NaN(); // [한국어] beta 초기값 NaN — 아직 미설정 상태
    if(params.size() < 2) { // [한국어] 인라인 파라미터에 beta가 없는 경우
      if(config) { // [한국어] Configuration 객체가 있으면 burst_beta를 설정 파일에서 읽는다
	beta = config->GetFloat("burst_beta"); // [한국어] gpgpusim.config의 burst_beta 값 읽기
      } else { // [한국어] config도 없고 인라인도 없으면 파라미터 누락
	missing_params = true; // [한국어] 누락 플래그 추가 설정
      }
    } else {                                 // [한국어] 인라인 파라미터에 beta가 있는 경우
      beta = atof(params[1].c_str()); // [한국어] params[1] 문자열을 double로 변환하여 beta 설정
    }

    double r1 = numeric_limits<double>::quiet_NaN(); // [한국어] r1 초기값 NaN — 아직 미설정 상태
    if(params.size() < 3) { // [한국어] 인라인 파라미터에 r1이 없는 경우
      r1 = config ? config->GetFloat("burst_r1") : -1.0; // [한국어] config 있으면 burst_r1, 없으면 -1.0(미지정)
    } else {                                              // [한국어] 인라인 파라미터에 r1이 있는 경우
      r1 = atof(params[2].c_str()); // [한국어] params[2] 문자열을 double로 변환하여 r1 설정
    }

    if(missing_params) { // [한국어] 필수 파라미터가 누락된 경우 에러 출력 후 종료
      cout << "Missing parameters for injection process: " << inject << endl;
      exit(-1); // [한국어] 비정상 종료 — on_off 파라미터 부족으로 시뮬레이션 불가
    }

    // [한국어] alpha, beta, r1 중 정확히 하나만 미지정(음수 또는 NaN)이어야 유효한 설정
    // 조건: (둘 다 음수)인 쌍이 존재하거나, 셋 모두 양수이면 유효하지 않음
    if((alpha < 0.0 && beta < 0.0) ||   // [한국어] alpha와 beta가 모두 미지정 — 유효하지 않음
       (alpha < 0.0 && r1 < 0.0) ||     // [한국어] alpha와 r1이 모두 미지정 — 유효하지 않음
       (beta < 0.0 && r1 < 0.0) ||      // [한국어] beta와 r1이 모두 미지정 — 유효하지 않음
       (alpha >= 0.0 && beta >= 0.0 && r1 >= 0.0)) { // [한국어] 셋 모두 지정 — 역산 불필요하지만 일관성 오류
      cout << "Invalid parameters for injection process: " << inject << endl;
      exit(-1); // [한국어] 비정상 종료 — on_off 파라미터 조합이 유효하지 않음
    }

    vector<int> initial(nodes); // [한국어] 각 노드의 초기 ON/OFF 상태를 담을 벡터 (크기=nodes)
    if(params.size() > 3) {     // [한국어] 인라인에 초기 상태 리스트가 있는 경우 (4번째 파라미터)
      initial = tokenize_int(params[2]);       // [한국어] 초기 상태 문자열을 정수 벡터로 변환
      initial.resize(nodes, initial.back());   // [한국어] 노드 수보다 짧으면 마지막 값으로 패딩
    } else {                                   // [한국어] 초기 상태가 지정되지 않은 경우 무작위 초기화
      for(int n = 0; n < nodes; ++n) {         // [한국어] 모든 노드에 대해 초기 상태를 무작위 결정
	initial[n] = RandomInt(1);             // [한국어] 0 또는 1을 균일 확률로 선택하여 초기 ON/OFF 상태 설정
      }
    }
    result = new OnOffInjectionProcess(nodes, load, alpha, beta, r1, initial); // [한국어] ON/OFF 프로세스 인스턴스 생성
  } else { // [한국어] 알 수 없는 프로세스 이름인 경우 에러 출력 후 종료
    cout << "Invalid injection process: " << inject << endl;
    exit(-1); // [한국어] 비정상 종료 — 지원하지 않는 주입 프로세스 유형
  }
  return result; // [한국어] 생성된 서브클래스 인스턴스 포인터 반환 — 호출자가 delete 책임
}

//=============================================================

/*
 * [한국어]
 * BernoulliInjectionProcess::BernoulliInjectionProcess - 베르누이 주입 프로세스 생성자.
 *
 * @param nodes: 소스 노드 수
 * @param rate:  패킷 생성 확률 [0.0, 1.0]
 *
 * 기반 클래스 InjectionProcess(nodes, rate)만 초기화한다.
 * 베르누이 과정은 추가 상태가 없으므로 생성자 본문이 비어 있다.
 *
 * 호출 체인:
 *   InjectionProcess::New("bernoulli", nodes, load) →
 *   new BernoulliInjectionProcess(nodes, load) → InjectionProcess(nodes, rate)
 */
BernoulliInjectionProcess::BernoulliInjectionProcess(int nodes, double rate)
  : InjectionProcess(nodes, rate) // [한국어] 기반 클래스 생성자에 nodes와 rate를 위임하여 초기화
{
  // [한국어] 추가 초기화 불필요 — 베르누이 과정은 _nodes, _rate 외 내부 상태 없음
}

/*
 * [한국어]
 * BernoulliInjectionProcess::test - 이번 사이클에 source 노드에서 패킷을 생성할지 결정한다.
 *
 * @param source: 소스 노드 번호 (0 이상 _nodes 미만)
 * @return: bool — RandomFloat() < _rate 이면 true (패킷 생성), 아니면 false
 *
 * 베르누이 과정의 핵심 동작: 매 호출마다 [0, 1) 균일 난수를 생성하고 _rate와 비교한다.
 * 이전 사이클이나 다른 노드의 상태와 완전히 독립적(memoryless) 이다.
 * source 번호가 유효 범위인지 assert로 검증한다.
 *
 * 호출 체인:
 *   TrafficManager::_Step() → BernoulliInjectionProcess::test(source)
 */
bool BernoulliInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] source가 유효 범위 [0, _nodes) 인지 검증
  return (RandomFloat() < _rate); // [한국어] [0,1) 균일 난수가 _rate 미만이면 패킷 생성 — rate가 클수록 생성 빈도 증가
}

//=============================================================

/*
 * [한국어]
 * OnOffInjectionProcess::OnOffInjectionProcess - ON/OFF 버스트 주입 프로세스 생성자.
 *
 * @param nodes:   소스 노드 수
 * @param rate:    목표 주입률 [0.0, 1.0]
 * @param alpha:   OFF→ON 전이 확률 (음수이면 beta와 r1로 역산)
 * @param beta:    ON→OFF 전이 확률 (음수이면 alpha와 r1로 역산)
 * @param r1:      ON 상태에서 패킷 생성 확률 (음수이면 alpha와 beta로 역산)
 * @param initial: 각 노드의 초기 ON/OFF 상태 벡터
 *
 * 동작 순서:
 *   1. 기반 클래스 InjectionProcess와 _alpha, _beta, _r1, _initial 초기화.
 *   2. 파라미터 범위 assert (모두 1.0 이하).
 *   3. 음수인 파라미터를 역산 공식으로 계산:
 *      - alpha < 0: _alpha = beta * rate / (r1 - rate)
 *      - beta  < 0: _beta  = alpha * (r1 - rate) / rate
 *      - r1    < 0: _r1    = rate * (alpha + beta) / alpha
 *   4. reset()으로 _state를 _initial로 초기화.
 *
 * 수학적 배경: ON/OFF 마르코프 체인에서 안정 상태(steady state) 주입률:
 *   E[rate] = r1 * P(ON) = r1 * alpha / (alpha + beta)
 *   이를 _rate에 맞추기 위해 미지 파라미터를 역산한다.
 *
 * 호출 체인:
 *   InjectionProcess::New("on_off", ...) → new OnOffInjectionProcess(...)
 */
OnOffInjectionProcess::OnOffInjectionProcess(int nodes, double rate,
					     double alpha, double beta,
					     double r1, vector<int> initial)
  : InjectionProcess(nodes, rate),   // [한국어] 기반 클래스 초기화 — _nodes, _rate 설정 및 유효성 검증
    _alpha(alpha), _beta(beta), _r1(r1), _initial(initial) // [한국어] 멤버 초기화 리스트로 alpha/beta/r1/initial 설정
{
  assert(alpha <= 1.0); // [한국어] alpha는 확률이므로 1.0 이하여야 함 (음수인 경우는 역산 대상)
  assert(beta <= 1.0);  // [한국어] beta는 확률이므로 1.0 이하여야 함 (음수인 경우는 역산 대상)
  assert(r1 <= 1.0);    // [한국어] r1은 확률이므로 1.0 이하여야 함 (음수인 경우는 역산 대상)

  if(alpha < 0.0) {            // [한국어] alpha가 미지정(-1 또는 NaN < 0)인 경우 beta와 r1로 역산
    assert(beta >= 0.0);       // [한국어] 역산을 위해 beta는 지정되어 있어야 함
    assert(r1 >= 0.0);         // [한국어] 역산을 위해 r1은 지정되어 있어야 함
    _alpha = beta * rate / (r1 - rate); // [한국어] 역산 공식: E[rate]=r1*alpha/(alpha+beta)를 alpha에 대해 풂
  } else if(beta < 0.0) {      // [한국어] beta가 미지정인 경우 alpha와 r1로 역산
    assert(alpha >= 0.0);      // [한국어] 역산을 위해 alpha는 지정되어 있어야 함
    assert(r1 >= 0.0);         // [한국어] 역산을 위해 r1은 지정되어 있어야 함
    _beta = alpha * (r1 - rate) / rate; // [한국어] 역산 공식: E[rate]=r1*alpha/(alpha+beta)를 beta에 대해 풂
  } else {                     // [한국어] r1이 미지정인 경우 (r1 < 0) alpha와 beta로 역산
    assert(r1 < 0.0);          // [한국어] 이 분기에서는 r1이 반드시 미지정 상태여야 함
    _r1 = rate * (alpha + beta) / alpha; // [한국어] 역산 공식: E[rate]=r1*alpha/(alpha+beta)를 r1에 대해 풂
  }
  reset(); // [한국어] _state를 _initial로 초기화하여 첫 번째 test() 호출을 위해 상태를 준비
}

/*
 * [한국어]
 * OnOffInjectionProcess::reset - ON/OFF 상태 벡터를 초기값(_initial)으로 복원한다.
 *
 * @return: void
 *
 * 시뮬레이션 재시작이나 생성자 마지막 단계에서 호출된다.
 * _state = _initial 대입으로 각 노드의 ON/OFF 상태를 생성 시 지정한 초기값으로 되돌린다.
 * 이를 통해 동일한 초기 상태에서 재현성 있는(reproducible) 시뮬레이션 재실행이 가능하다.
 *
 * 호출 체인:
 *   OnOffInjectionProcess::OnOffInjectionProcess() → reset() (생성자 마지막)
 *   TrafficManager 재시작 → OnOffInjectionProcess::reset()
 */
void OnOffInjectionProcess::reset()
{
  _state = _initial; // [한국어] 현재 ON/OFF 상태를 초기 상태 벡터로 완전 복사하여 재설정
}

/*
 * [한국어]
 * OnOffInjectionProcess::test - ON/OFF 상태를 전진시키고 이번 사이클 패킷 생성 여부를 반환한다.
 *
 * @param source: 소스 노드 번호 (0 이상 _nodes 미만)
 * @return: bool — 상태가 ON(1)이고 RandomFloat() < _r1이면 true (패킷 생성)
 *
 * 동작 순서 (매 사이클, 각 노드별):
 *   1. 현재 _state[source] 확인.
 *   2. 상태 전이:
 *      ON(1)  이면: RandomFloat() >= _beta → ON 유지 (1), < _beta → OFF(0)
 *      OFF(0) 이면: RandomFloat() <  _alpha → ON 전이 (1), >= _alpha → OFF 유지(0)
 *   3. 전이 후 상태가 ON(1)이고 RandomFloat() < _r1이면 패킷 생성(true 반환).
 *   상태 전이와 패킷 생성은 같은 사이클에 한 번의 함수 호출로 처리된다.
 *
 * 수학적 특성:
 *   장기 평균 주입률 = _r1 * _alpha / (_alpha + _beta) = _rate (목표 주입률과 일치)
 *
 * 호출 체인:
 *   TrafficManager::_Step() → OnOffInjectionProcess::test(source)
 */
bool OnOffInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] source가 유효 범위 [0, _nodes) 인지 검증

  // advance state
  // [한국어] 현재 ON 상태(1)이면 _beta 확률로 OFF 전이, OFF 상태(0)이면 _alpha 확률로 ON 전이
  _state[source] =
    _state[source] ? (RandomFloat() >= _beta) : (RandomFloat() < _alpha);
    // [한국어] _state[source]==1(ON): RandomFloat() >= _beta 이면 1(ON 유지), 아니면 0(OFF 전이)
    // [한국어] _state[source]==0(OFF): RandomFloat() < _alpha 이면 1(ON 전이), 아니면 0(OFF 유지)

  // generate packet
  return _state[source] && (RandomFloat() < _r1);
  // [한국어] 전이 후 상태가 ON(1)이고 추가 확률 _r1을 통과해야 실제 패킷 생성
  // [한국어] _state[source]==0(OFF)이면 단락 평가(short-circuit)로 패킷 생성 없음
}
