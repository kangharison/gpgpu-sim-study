// $Id: traffic.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 트래픽 패턴 클래스 계층 선언 (traffic.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim NoC(Network-on-Chip) 시뮬레이터에서 사용하는 모든 합성
 * 트래픽 패턴(synthetic traffic pattern)의 클래스 계층 구조를 선언한다.
 * 트래픽 패턴은 시뮬레이션 중 각 소스 노드가 어느 목적지 노드로 패킷을
 * 보낼지를 결정하는 규칙을 추상화한다.
 * 팩토리 메서드 TrafficPattern::New()를 통해 설정 파일의 문자열 이름으로
 * 원하는 패턴 객체를 생성할 수 있으며, TrafficManager가 이를 사용한다.
 * GPGPU-Sim에서는 GPUTrafficManager가 실제 GPU 메모리 요청을 직접 주입하므로
 * 이 패턴들은 standalone BookSim 벤치마크 또는 합성 부하 평가에서 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 계층 구조:
 *   TrafficManager (trafficmanager.cpp) → TrafficPattern::dest() 호출
 *     → 각 사이클마다 소스 노드별로 목적지 노드를 결정
 *   InjectionProcess (injection.hpp)와 조합 → 실제 패킷 생성 여부 + 목적지 결정
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 싱글 스레드 시뮬레이션 루프
 *
 * === 타 모듈과의 연결 ===
 * - trafficmanager.cpp: _injection_process[c]->test() 후 _traffic_pattern[c]->dest() 호출
 * - config_utils.hpp: TrafficPattern::New()가 Configuration 객체로 파라미터 읽기
 * - random_utils.hpp (traffic.cpp 내): 확률적 패턴(uniform, hotspot 등)에서 RandomInt() 사용
 * - GPUTrafficManager (gputrafficmanager.cpp): GPU 모드에서는 TrafficPattern 사용 안 함
 *   (실제 mem_fetch에서 src/dest 결정)
 *
 * === 주요 함수/구조체 요약 ===
 * - TrafficPattern::New()        : 문자열 → 구체 패턴 객체 팩토리 (traffic.cpp 구현)
 * - TrafficPattern::dest()       : 소스 노드 → 목적지 노드 매핑 (순수 가상)
 * - BitCompTrafficPattern        : 비트 보수 패턴 (~source & mask)
 * - TransposeTrafficPattern      : 2D 메시 행/열 교환 패턴
 * - BitRevTrafficPattern         : 비트 역전 패턴
 * - ShuffleTrafficPattern        : 사이클릭 왼쪽 시프트 패턴
 * - TornadoTrafficPattern        : 각 차원에서 절반-거리+1 이동 패턴 (최악 부하)
 * - NeighborTrafficPattern       : 각 차원에서 +1 이동 패턴
 * - RandomPermutationTrafficPattern : 시드 기반 랜덤 전단사 매핑
 * - UniformRandomTrafficPattern  : 완전 균등 랜덤 목적지
 * - UniformBackgroundTrafficPattern : 특정 노드 제외 균등 랜덤
 * - DiagonalTrafficPattern       : 50% 확률로 인접 노드, 나머지는 자기 자신
 * - AsymmetricTrafficPattern     : 상하반 교차 트래픽 (비대칭 부하)
 * - Taper64TrafficPattern        : 64노드 전용 국소+랜덤 혼합 패턴
 * - BadPermDFlyTrafficPattern    : Dragonfly 토폴로지 최악 순열 패턴
 * - BadPermYarcTrafficPattern    : YARC 토폴로지 최악 순열 패턴
 * - HotSpotTrafficPattern        : 복수 핫스팟 노드에 가중 확률로 집중
 */

#ifndef _TRAFFIC_HPP_ // [한국어] 중복 포함 방지 가드 시작
#define _TRAFFIC_HPP_ // [한국어] 헤더 심볼 정의

#include <vector> // [한국어] std::vector: 핫스팟 목록, 목적지 배열 등에 사용
#include <set>    // [한국어] std::set: UniformBackgroundTrafficPattern의 제외 노드 집합에 사용
#include "config_utils.hpp" // [한국어] Configuration 클래스: New()에서 파라미터 읽기용

using namespace std; // [한국어] std:: 접두사 생략을 위한 네임스페이스 임포트

/*
 * [한국어]
 * TrafficPattern - 모든 트래픽 패턴의 추상 기반 클래스
 *
 * @_nodes: 네트워크의 총 노드 수 (소스/목적지의 유효 범위)
 *
 * 이 클래스는 순수 가상 함수 dest()를 통해 트래픽 패턴의 인터페이스를 정의한다.
 * 각 하위 클래스는 고유한 수학적 매핑 규칙으로 dest()를 구현한다.
 * New() 팩토리를 통해 설정 파일의 "traffic" 키 값에 따라 동적으로 객체가 생성되며,
 * TrafficManager 생성자에서 각 트래픽 클래스(c)마다 하나씩 인스턴스화된다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [_traffic_pattern[c]->dest(source)]
 */
class TrafficPattern {
protected:
  int _nodes;
  /* [한국어] 네트워크의 총 노드 수.
   * 설정자: 생성자(TrafficPattern(int nodes))에서 초기화.
   * 읽는 자: 각 하위 클래스의 dest()가 유효 범위 검사 및 모듈로 연산에 사용.
   * 값 범위: 1 이상의 양의 정수 (0 이하면 생성자에서 오류 종료).
   * 동기화: 싱글 스레드 시뮬레이션 루프에서만 읽히므로 락 불필요. */

  TrafficPattern(int nodes);
  /* [한국어] 보호 생성자: 노드 수 유효성 검사 후 _nodes 초기화.
   * nodes <= 0이면 오류 메시지 출력 후 exit(-1). */

public:
  virtual ~TrafficPattern() {}
  /* [한국어] 가상 소멸자: 하위 클래스 소멸자가 올바르게 호출되도록 보장. */

  virtual void reset();
  /* [한국어] reset - 패턴 상태 초기화 (시뮬레이션 재실행 시 호출).
   * 기본 구현은 아무것도 하지 않음.
   * 상태를 가진 패턴(RandomPermutation 등)은 재정의 가능하나 현재는 기본 구현만 사용.
   *
   * 호출 체인:
   *   TrafficManager::Run() → [_traffic_pattern[c]->reset()] */

  virtual int dest(int source) = 0;
  /* [한국어] dest - 소스 노드에서 패킷을 보낼 목적지 노드를 반환 (순수 가상).
   * @source: 패킷을 생성하는 소스 노드 번호 (0 이상 _nodes 미만).
   * @return: 목적지 노드 번호 (0 이상 _nodes 미만). 패턴에 따라 결정론적/확률적.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [패턴 객체->dest(source)] */

  static TrafficPattern * New(string const & pattern, int nodes,
			      Configuration const * const config = NULL);
  /* [한국어] New - 문자열 패턴 이름으로 구체 TrafficPattern 객체를 생성하는 팩토리.
   * @pattern: "bitcomp", "uniform", "hotspot(0,3)" 형식의 패턴 이름+파라미터 문자열.
   * @nodes: 네트워크 노드 수.
   * @config: 파라미터가 문자열에 없을 때 대체로 읽을 설정 객체 (NULL 허용).
   * @return: 힙에 생성된 TrafficPattern 파생 객체 포인터 (호출자가 delete 책임).
   *
   * 지원 패턴: bitcomp / transpose / bitrev / shuffle / randperm /
   *            uniform / background / diagonal / asymmetric / taper64 /
   *            bad_dragon / tornado / neighbor / badperm_yarc / hotspot
   * 알 수 없는 패턴이면 오류 출력 후 exit(-1).
   *
   * 호출 체인:
   *   TrafficManager 생성자 → [TrafficPattern::New()] */
};

/*
 * [한국어]
 * PermutationTrafficPattern - 전단사(bijective) 순열 기반 패턴의 중간 기반 클래스
 *
 * 모든 순열 패턴(Bit*, Digit*, Random*)의 공통 조상.
 * 추가 필드 없음 — 계층 구조 명시 목적으로만 존재.
 *
 * 호출 체인: TrafficPattern → [PermutationTrafficPattern] → 구체 순열 클래스
 */
class PermutationTrafficPattern : public TrafficPattern {
protected:
  PermutationTrafficPattern(int nodes);
  /* [한국어] 보호 생성자: 상위 TrafficPattern(nodes) 위임. */
};

/*
 * [한국어]
 * BitPermutationTrafficPattern - 비트 조작 기반 순열 패턴의 중간 기반 클래스
 *
 * BitComp / Transpose / BitRev / Shuffle의 공통 조상.
 * 생성자에서 nodes가 2의 거듭제곱인지 검사한다 — 비트 연산 패턴은
 * 노드 수가 2의 거듭제곱이어야 올바르게 동작하기 때문.
 *
 * 호출 체인: PermutationTrafficPattern → [BitPermutationTrafficPattern] → 구체 클래스
 */
class BitPermutationTrafficPattern : public PermutationTrafficPattern {
protected:
  BitPermutationTrafficPattern(int nodes);
  /* [한국어] 보호 생성자: (nodes & -nodes) != nodes이면 오류 종료 (2의 거듭제곱 검사). */
};

/*
 * [한국어]
 * BitCompTrafficPattern - 비트 보수(Bit Complement) 트래픽 패턴
 *
 * source의 모든 비트를 반전하여 목적지를 계산: dest = ~source & (nodes-1).
 * nodes=8이면 0→7, 1→6, 2→5, 3→4, 4→3, 5→2, 6→1, 7→0.
 * 이는 균형 잡힌 최악 부하를 생성하며 fat-tree 등의 대역폭 분석에 사용된다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BitCompTrafficPattern::dest()]
 */
class BitCompTrafficPattern : public BitPermutationTrafficPattern {
public:
  BitCompTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 BitPermutationTrafficPattern(nodes) 위임 (2의 거듭제곱 검사). */

  virtual int dest(int source);
  /* [한국어] dest - 비트 보수 목적지 계산.
   * @source: 소스 노드 번호 (0 이상 _nodes 미만).
   * @return: ~source & (_nodes-1) — 모든 유효 비트 반전.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [BitCompTrafficPattern::dest()] */
};

/*
 * [한국어]
 * TransposeTrafficPattern - 전치(Transpose) 트래픽 패턴
 *
 * 소스 노드 번호를 2D 격자의 (행, 열) 인덱스로 해석하고 행과 열을 교환한다.
 * nodes=N^2일 때 source=(row*N+col) → dest=(col*N+row).
 * 이를 비트 조작으로 구현하기 위해 _shift = log2(N)을 사용한다.
 * nodes가 짝수 제곱의 2의 거듭제곱이어야 한다 (예: 4, 16, 64, 256).
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [TransposeTrafficPattern::dest()]
 */
class TransposeTrafficPattern : public BitPermutationTrafficPattern {
protected:
  int _shift;
  /* [한국어] 비트 시프트 양 = log2(sqrt(nodes)) = log2(nodes)/2.
   * 설정자: 생성자에서 nodes의 비트 폭을 세어 절반으로 나눈 값으로 초기화.
   * 읽는 자: dest()에서 상위/하위 절반 비트를 추출하고 교환할 때 사용.
   * 값 범위: 양의 정수 (nodes가 짝수 제곱 2^(2k)이면 _shift = k).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

public:
  TransposeTrafficPattern(int nodes);
  /* [한국어] 생성자: _shift 계산, nodes가 짝수 제곱이 아니면 오류 종료. */

  virtual int dest(int source);
  /* [한국어] dest - 상위/하위 절반 비트 교환으로 전치 목적지 계산.
   * @source: 소스 노드 번호.
   * @return: 하위 _shift 비트와 상위 _shift 비트를 교환한 결과.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [TransposeTrafficPattern::dest()] */
};

/*
 * [한국어]
 * BitRevTrafficPattern - 비트 역전(Bit Reverse) 트래픽 패턴
 *
 * 소스 번호의 비트 순서를 완전히 역전하여 목적지를 계산한다.
 * nodes=8(3비트)이면 0b001(1) → 0b100(4), 0b110(6) → 0b011(3).
 * 네트워크에서 최장 경로를 강제하는 스트레스 테스트에 활용된다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BitRevTrafficPattern::dest()]
 */
class BitRevTrafficPattern : public BitPermutationTrafficPattern {
public:
  BitRevTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 BitPermutationTrafficPattern(nodes) 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 비트 역전 루프로 목적지 계산.
   * @source: 소스 노드 번호.
   * @return: source의 이진 표현을 log2(_nodes) 비트 기준으로 역전한 값.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [BitRevTrafficPattern::dest()] */
};

/*
 * [한국어]
 * ShuffleTrafficPattern - 셔플(Shuffle) 트래픽 패턴
 *
 * 소스 번호를 사이클릭 왼쪽 1비트 시프트로 목적지를 계산한다.
 * source의 최상위 비트가 최하위 비트로 순환된다.
 * Omega 네트워크의 스위칭 패턴과 대응되며 패턴이 규칙적이다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [ShuffleTrafficPattern::dest()]
 */
class ShuffleTrafficPattern : public BitPermutationTrafficPattern {
public:
  ShuffleTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 BitPermutationTrafficPattern(nodes) 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 사이클릭 왼쪽 시프트로 목적지 계산.
   * @source: 소스 노드 번호.
   * @return: (source << 1) 의 하위 log2(nodes) 비트 + 오버플로 비트를 LSB로.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [ShuffleTrafficPattern::dest()] */
};

/*
 * [한국어]
 * DigitPermutationTrafficPattern - 진수(Digit) 기반 순열 패턴의 중간 기반 클래스
 *
 * k-ary n-cube(k진수 n차원 큐브) 토폴로지를 위한 패턴의 공통 조상.
 * 각 노드 번호를 n개의 k진수 digit으로 분해하여 각 차원별로 목적지를 계산한다.
 * _xr(expansion ratio)은 라우터당 노드 수를 나타낸다.
 *
 * 호출 체인: PermutationTrafficPattern → [DigitPermutationTrafficPattern] → 구체 클래스
 */
class DigitPermutationTrafficPattern : public PermutationTrafficPattern {
protected:
  int _k;
  /* [한국어] 기수(radix) — k-ary n-cube에서 k (각 차원의 라우터 수).
   * 설정자: 생성자에서 설정 파일의 "k" 또는 패턴 파라미터로 초기화.
   * 읽는 자: dest()에서 각 차원의 digit 추출 및 모듈로 계산에 사용.
   * 값 범위: 2 이상의 양의 정수.
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  int _n;
  /* [한국어] 차원 수(dimension) — k-ary n-cube에서 n.
   * 설정자: 생성자에서 설정 파일의 "n" 또는 패턴 파라미터로 초기화.
   * 읽는 자: dest()의 차원 순환 루프 반복 횟수.
   * 값 범위: 1 이상의 양의 정수. (_k^_n == _nodes 관계)
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  int _xr;
  /* [한국어] 라우터당 노드 확장 비율(expansion ratio/concentration).
   * 설정자: 생성자에서 설정 파일의 "xr" 또는 패턴 파라미터로 초기화.
   * 읽는 자: dest()에서 실질 digit 범위(_xr*_k)를 계산할 때 사용.
   * 값 범위: 1 이상 (일반적으로 1~4). xr=1이면 라우터=노드.
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  DigitPermutationTrafficPattern(int nodes, int k, int n, int xr = 1);
  /* [한국어] 보호 생성자: _k, _n, _xr 초기화. */
};

/*
 * [한국어]
 * TornadoTrafficPattern - 토네이도(Tornado) 트래픽 패턴
 *
 * 각 차원에서 현재 digit에 floor(k/2)-1을 더한 값을 목적지 digit으로 사용.
 * 이는 각 차원에서 절반 거리 바로 전까지 보내는 패턴으로,
 * k-ary n-cube에서 링크 부하를 최대화하는 스트레스 트래픽이다.
 * 각 링크가 최대 트래픽을 부담하게 하여 실제 네트워크의 포화 지점을 측정한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [TornadoTrafficPattern::dest()]
 */
class TornadoTrafficPattern : public DigitPermutationTrafficPattern {
public:
  TornadoTrafficPattern(int nodes, int k, int n, int xr = 1);
  /* [한국어] 생성자: 상위 DigitPermutationTrafficPattern 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 각 차원 digit에 (xr*k+1)/2-1을 더해 목적지 계산.
   * @source: 소스 노드 번호.
   * @return: 각 차원에서 절반 거리 이동한 목적지 노드 번호.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [TornadoTrafficPattern::dest()] */
};

/*
 * [한국어]
 * NeighborTrafficPattern - 이웃(Neighbor) 트래픽 패턴
 *
 * 각 차원에서 현재 digit에 +1(모듈로 xr*k)을 더한 값을 목적지 digit으로 사용.
 * 모든 노드가 각 차원에서 바로 인접한 노드로 패킷을 보내는 패턴이다.
 * 지역 트래픽(nearest-neighbor)의 네트워크 부하를 측정하는 데 사용한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [NeighborTrafficPattern::dest()]
 */
class NeighborTrafficPattern : public DigitPermutationTrafficPattern {
public:
  NeighborTrafficPattern(int nodes, int k, int n, int xr = 1);
  /* [한국어] 생성자: 상위 DigitPermutationTrafficPattern 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 각 차원 digit에 +1 이동 목적지 계산.
   * @source: 소스 노드 번호.
   * @return: 각 차원에서 인접 노드인 목적지 번호.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [NeighborTrafficPattern::dest()] */
};

/*
 * [한국어]
 * RandomPermutationTrafficPattern - 시드 기반 랜덤 순열 트래픽 패턴
 *
 * 주어진 시드(seed)로 무작위 전단사 순열 테이블을 미리 생성하고,
 * dest(source)는 테이블 조회로 O(1)에 반환한다.
 * 실험 재현성을 위해 동일 시드면 동일 순열이 보장된다.
 * reset() 없이 시뮬레이션 전체에서 동일 순열 사용.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [RandomPermutationTrafficPattern::dest()]
 */
class RandomPermutationTrafficPattern : public TrafficPattern {
private:
  vector<int> _dest;
  /* [한국어] 노드 번호 → 목적지 번호 순열 테이블 (크기 = _nodes).
   * 설정자: 생성자에서 randomize(seed)가 Fisher-Yates 변형으로 전단사 순열 생성.
   * 읽는 자: dest(source)가 _dest[source]를 반환.
   * 값 범위: 각 원소는 0 이상 _nodes 미만, 전체가 전단사(중복 없음).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  inline void randomize(int seed);
  /* [한국어] randomize - seed로 _dest 순열 테이블 생성 (내부 유틸).
   * @seed: 순열 생성 시드.
   * 이전 랜덤 시드를 저장/복원하여 시뮬레이터 전역 RNG 상태를 오염하지 않음.
   * Fisher-Yates 변형으로 완전 랜덤 전단사 순열 생성.
   *
   * 호출 체인:
   *   RandomPermutationTrafficPattern() → [randomize()] */

public:
  RandomPermutationTrafficPattern(int nodes, int seed);
  /* [한국어] 생성자: _dest 벡터 크기 설정 후 randomize(seed) 호출. */

  virtual int dest(int source);
  /* [한국어] dest - 순열 테이블에서 목적지 반환.
   * @source: 소스 노드 번호.
   * @return: _dest[source] (전단사 순열 값).
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [RandomPermutationTrafficPattern::dest()] */
};

/*
 * [한국어]
 * RandomTrafficPattern - 확률적 랜덤 패턴의 중간 기반 클래스
 *
 * Uniform / Background / Diagonal / Asymmetric / Taper64의 공통 조상.
 * 추가 필드 없음 — 계층 구조 명시 목적.
 *
 * 호출 체인: TrafficPattern → [RandomTrafficPattern] → 구체 랜덤 클래스
 */
class RandomTrafficPattern : public TrafficPattern {
protected:
  RandomTrafficPattern(int nodes);
  /* [한국어] 보호 생성자: 상위 TrafficPattern(nodes) 위임. */
};

/*
 * [한국어]
 * UniformRandomTrafficPattern - 완전 균등 랜덤(Uniform Random) 트래픽 패턴
 *
 * 매 호출마다 [0, _nodes-1] 범위에서 균등 분포 랜덤 목적지를 반환한다.
 * 가장 단순한 합성 벤치마크로, 이론적 최대 처리량 측정에 사용된다.
 * source와 무관하게 임의 노드(자기 자신 포함)가 목적지가 될 수 있다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [UniformRandomTrafficPattern::dest()]
 */
class UniformRandomTrafficPattern : public RandomTrafficPattern {
public:
  UniformRandomTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 RandomTrafficPattern(nodes) 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 균등 랜덤 목적지 반환.
   * @source: 사용되지 않음 (균등 패턴은 출발지 무관).
   * @return: RandomInt(_nodes - 1) — [0, _nodes-1] 균등 분포 난수.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [UniformRandomTrafficPattern::dest()] */
};

/*
 * [한국어]
 * UniformBackgroundTrafficPattern - 특정 노드 제외 균등 랜덤 배경 트래픽 패턴
 *
 * 지정된 제외 노드 집합(_excluded)을 피해 균등 랜덤 목적지를 생성한다.
 * 특정 노드(예: 핫스팟)를 배경 트래픽에서 제외하고 별도로 테스트할 때 사용.
 * 반복-거부(rejection sampling) 방식으로 제외 노드를 피한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [UniformBackgroundTrafficPattern::dest()]
 */
class UniformBackgroundTrafficPattern : public RandomTrafficPattern {
private:
  set<int> _excluded;
  /* [한국어] 목적지 후보에서 제외할 노드 번호 집합.
   * 설정자: 생성자에서 excluded_nodes 벡터의 각 원소를 삽입.
   * 읽는 자: dest()의 rejection sampling 루프에서 결과 제외 여부 판정.
   * 값 범위: 각 원소는 0 이상 _nodes 미만. 빈 집합 가능(전체 랜덤과 동일).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

public:
  UniformBackgroundTrafficPattern(int nodes, vector<int> excluded_nodes);
  /* [한국어] 생성자: excluded_nodes를 _excluded에 삽입 (유효성 검사 포함). */

  virtual int dest(int source);
  /* [한국어] dest - 제외 노드를 피하며 균등 랜덤 목적지 반환.
   * @source: 사용되지 않음.
   * @return: _excluded에 없는 랜덤 노드 번호 (rejection sampling).
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [UniformBackgroundTrafficPattern::dest()] */
};

/*
 * [한국어]
 * DiagonalTrafficPattern - 대각선(Diagonal) 트래픽 패턴
 *
 * 50% 확률로 인접 다음 노드((source+1)%nodes)로, 나머지 50%는 자기 자신(source)으로 보낸다.
 * 매우 지역적인 트래픽과 자기-통신이 혼재하는 패턴을 만들어낸다.
 * 낮은 홉 카운트를 가지는 트래픽 시나리오를 모의한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [DiagonalTrafficPattern::dest()]
 */
class DiagonalTrafficPattern : public RandomTrafficPattern {
public:
  DiagonalTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 RandomTrafficPattern(nodes) 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 50% 확률로 다음 노드, 50%로 자기 자신 반환.
   * @source: 소스 노드 번호.
   * @return: (source+1)%_nodes 또는 source.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [DiagonalTrafficPattern::dest()] */
};

/*
 * [한국어]
 * AsymmetricTrafficPattern - 비대칭(Asymmetric) 트래픽 패턴
 *
 * 소스 노드의 위치(하반/상반)에 따라 목적지를 결정한다.
 * source를 상반/하반 중 하나로 매핑: dest = (source % half) + (random ? half : 0).
 * 네트워크의 상하반 간 링크에 비대칭 부하를 생성하여 hotspot과 유사한 효과.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [AsymmetricTrafficPattern::dest()]
 */
class AsymmetricTrafficPattern : public RandomTrafficPattern {
public:
  AsymmetricTrafficPattern(int nodes);
  /* [한국어] 생성자: 상위 RandomTrafficPattern(nodes) 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 상반/하반 중 하나로 랜덤 목적지 반환.
   * @source: 소스 노드 번호.
   * @return: (source % half) + (RandomInt(1) ? half : 0).
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [AsymmetricTrafficPattern::dest()] */
};

/*
 * [한국어]
 * Taper64TrafficPattern - Taper-64 특수 트래픽 패턴 (64노드 전용)
 *
 * 정확히 64개 노드에서만 동작하는 특수 혼합 패턴.
 * 50% 확률로 8x8 격자 기준 ±1행/열 내 근방 노드(지역 트래픽),
 * 나머지 50%는 전체 랜덤 목적지.
 * HPC 워크로드의 지역성과 글로벌 통신을 혼합 모의한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [Taper64TrafficPattern::dest()]
 */
class Taper64TrafficPattern : public RandomTrafficPattern {
public:
  Taper64TrafficPattern(int nodes);
  /* [한국어] 생성자: nodes != 64면 오류 종료. */

  virtual int dest(int source);
  /* [한국어] dest - 50% 지역 근방 / 50% 전체 랜덤 목적지 반환.
   * @source: 소스 노드 번호 (0-63).
   * @return: 근방 노드(±8행, ±1열 범위 모듈로) 또는 전체 랜덤.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [Taper64TrafficPattern::dest()] */
};

/*
 * [한국어]
 * BadPermDFlyTrafficPattern - Dragonfly 토폴로지 최악 순열 트래픽 패턴
 *
 * Dragonfly 네트워크에서 적응형 라우팅을 무력화하는 최악 케이스 트래픽 패턴.
 * 각 소스는 자신이 속한 그룹(group) 외부의 랜덤 노드로 패킷을 보낸다.
 * 그룹 간 링크를 집중 포화시켜 드래곤플라이 네트워크의 취약점을 드러낸다.
 * 그룹 크기 = 2*k 라우터, 그룹당 노드 수 = 2*k*k.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BadPermDFlyTrafficPattern::dest()]
 */
class BadPermDFlyTrafficPattern : public DigitPermutationTrafficPattern {
public:
  BadPermDFlyTrafficPattern(int nodes, int k, int n);
  /* [한국어] 생성자: xr=1로 고정하여 상위 DigitPermutationTrafficPattern 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 소스의 그룹 외부 랜덤 목적지 반환.
   * @source: 소스 노드 번호.
   * @return: 현재 그룹 다음 그룹부터 시작하는 랜덤 노드 번호.
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [BadPermDFlyTrafficPattern::dest()] */
};

/*
 * [한국어]
 * BadPermYarcTrafficPattern - YARC 토폴로지 최악 순열 트래픽 패턴
 *
 * YARC(Yet Another Regular Cluster) 네트워크에서 최악 케이스 트래픽 패턴.
 * 소스의 열(column)을 유지하되 행(row)을 랜덤으로 선택하는 패턴.
 * 열 간 링크에 집중 부하를 주어 YARC 네트워크의 병목을 드러낸다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BadPermYarcTrafficPattern::dest()]
 */
class BadPermYarcTrafficPattern : public DigitPermutationTrafficPattern {
public:
  BadPermYarcTrafficPattern(int nodes, int k, int n, int xr = 1);
  /* [한국어] 생성자: 상위 DigitPermutationTrafficPattern 위임. */

  virtual int dest(int source);
  /* [한국어] dest - 소스의 행을 랜덤 행으로 교체한 목적지 반환.
   * @source: 소스 노드 번호.
   * @return: 랜덤 행 * (_xr*_k) + source의 열 번호(row).
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [BadPermYarcTrafficPattern::dest()] */
};

/*
 * [한국어]
 * HotSpotTrafficPattern - 핫스팟(Hot Spot) 트래픽 패턴
 *
 * 하나 이상의 핫스팟 노드에 가중 확률로 트래픽을 집중시키는 패턴.
 * 각 핫스팟에 연관된 rate 값이 클수록 해당 노드로의 트래픽 비중이 높다.
 * 가중 랜덤 선택(weighted random selection)으로 핫스팟 노드를 결정한다.
 * GPU 메모리 컨트롤러 부하 불균형 시뮬레이션에 유용하다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [HotSpotTrafficPattern::dest()]
 */
class HotSpotTrafficPattern : public TrafficPattern {
private:
  vector<int> _hotspots;
  /* [한국어] 핫스팟 노드 번호 목록.
   * 설정자: 생성자에서 hotspots 인수로 초기화 (음수면 랜덤 노드 할당).
   * 읽는 자: dest()의 가중 랜덤 선택에서 최종 목적지 노드 번호로 참조.
   * 값 범위: 각 원소는 0 이상 _nodes 미만 (생성자에서 assert 검사).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  vector<int> _rates;
  /* [한국어] 각 핫스팟의 가중치(상대적 트래픽 비율).
   * 설정자: 생성자에서 rates 인수로 초기화; 없으면 모두 1로 설정.
   * 읽는 자: dest()에서 _max_val 범위의 랜덤값과 누적 비교하여 핫스팟 선택.
   * 값 범위: 각 원소는 0 이상 (assert 검사). 합이 _max_val+1이 됨.
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

  int _max_val;
  /* [한국어] 가중 랜덤 선택의 최대값 = sum(_rates) - 1.
   * 설정자: 생성자에서 _rates 합산으로 계산 (_max_val = -1로 시작 후 각 rate 누적).
   * 읽는 자: dest()에서 RandomInt(_max_val)의 상한으로 사용.
   * 값 범위: 0 이상 (최소 하나의 핫스팟, rate >= 0).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */

public:
  HotSpotTrafficPattern(int nodes, vector<int> hotspots,
			vector<int> rates = vector<int>());
  /* [한국어] 생성자: 핫스팟/가중치 초기화 및 _max_val 계산.
   * @nodes: 네트워크 노드 수.
   * @hotspots: 핫스팟 노드 번호 목록 (비어 있으면 assert 실패).
   * @rates: 가중치 목록 (생략 시 모두 1). */

  virtual int dest(int source);
  /* [한국어] dest - 가중 랜덤 선택으로 핫스팟 목적지 반환.
   * @source: 사용되지 않음 (핫스팟은 소스 무관).
   * @return: _rates 가중치에 비례한 랜덤 핫스팟 노드 번호.
   *
   * 단일 핫스팟이면 직접 반환(O(1)), 다중이면 선형 탐색(O(|_hotspots|)).
   *
   * 호출 체인:
   *   TrafficManager::_GeneratePacket() → [HotSpotTrafficPattern::dest()] */
};

#endif // [한국어] _TRAFFIC_HPP_ 중복 포함 방지 가드 끝
