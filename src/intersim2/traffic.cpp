// $Id: traffic.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 트래픽 패턴 클래스 구현 (traffic.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 traffic.hpp에 선언된 모든 트래픽 패턴 클래스들의 구현부이다.
 * 각 클래스의 생성자, dest() 함수(소스 → 목적지 계산 로직), 그리고
 * TrafficPattern::New() 팩토리 함수를 구현한다.
 * 팩토리는 설정 파일 문자열(예: "tornado(4,2,1)")을 파싱하여 적절한
 * 패턴 객체를 동적 생성하고, TrafficManager가 이를 사용하여 매 사이클
 * 각 소스 노드의 패킷 목적지를 결정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   TrafficManager 생성자
 *     → TrafficPattern::New("traffic_pattern_str", nodes, &config)
 *         → 구체 패턴 객체 생성 (힙 할당)
 *   매 사이클 TrafficManager::_GeneratePacket()
 *     → _traffic_pattern[c]->dest(source)
 *         → 구체 클래스의 dest() 실행 → 목적지 노드 번호 반환
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 싱글 스레드 BookSim 시뮬레이션 루프.
 * GPGPU-Sim 통합 모드(GPUTrafficManager)에서는 이 패턴들이 사용되지 않음 —
 * 실제 GPU 커널의 mem_fetch 객체가 src/dest를 직접 지정한다.
 *
 * === 타 모듈과의 연결 ===
 * - traffic.hpp: 모든 클래스의 선언부
 * - random_utils.hpp: RandomInt(), RandomFloat(), RandomSeed(), RandomIntLong() 제공
 *   (확률적 패턴 — uniform, hotspot, diagonal, asymmetric, taper64에서 사용)
 * - config_utils.hpp (Configuration): New()에서 설정 파일의 "k", "n", "xr",
 *   "perm_seed" 값 읽기
 * - trafficmanager.cpp: TrafficManager::_traffic_pattern[c] 멤버로 소유
 *   (TrafficManager 소멸자에서 delete _traffic_pattern[c] 호출)
 *
 * === 주요 함수/구조체 요약 ===
 * - TrafficPattern::New()          : 문자열 파싱 후 구체 패턴 객체 생성하는 팩토리
 * - BitCompTrafficPattern::dest()  : ~source & (nodes-1) — 비트 보수
 * - TransposeTrafficPattern::dest(): 상하위 절반 비트 교환 — 2D 전치
 * - BitRevTrafficPattern::dest()   : 이진 비트 역순 루프
 * - ShuffleTrafficPattern::dest()  : 사이클릭 왼쪽 1비트 시프트
 * - TornadoTrafficPattern::dest()  : 각 차원 digit에 floor(k/2) 더하기
 * - NeighborTrafficPattern::dest() : 각 차원 digit에 +1 이동
 * - RandomPermutationTrafficPattern::randomize(): Fisher-Yates 변형 순열 생성
 * - HotSpotTrafficPattern::dest()  : 가중 랜덤 핫스팟 선택
 */

#include <iostream>  // [한국어] cout, endl: 오류 메시지 출력
#include <sstream>   // [한국어] (현재 직접 사용 안 하나 관련 헤더로 포함)
#include "random_utils.hpp" // [한국어] RandomInt, RandomSeed, RandomIntLong 등 RNG 유틸
#include "traffic.hpp"      // [한국어] 모든 트래픽 패턴 클래스 선언

/*
 * [한국어]
 * TrafficPattern::TrafficPattern - 기반 클래스 생성자
 *
 * @nodes: 네트워크의 총 노드 수. 0 이하면 오류 종료.
 *
 * _nodes를 초기화하고 노드 수의 유효성을 검사한다.
 * 모든 하위 클래스 생성자의 초기화 목록에서 이 생성자를 위임 호출한다.
 *
 * 호출 체인:
 *   각 TrafficPattern 서브클래스 생성자 → [TrafficPattern(nodes)]
 */
TrafficPattern::TrafficPattern(int nodes)
: _nodes(nodes) // [한국어] _nodes 초기화: 이후 dest()에서 유효 범위 검사에 사용
{
  if(nodes <= 0) { // [한국어] 노드 수가 0 이하면 시뮬레이션 불가 — 즉시 종료
    cout << "Error: Traffic patterns require at least one node." << endl;
    exit(-1); // [한국어] 비정상 종료: 패턴 없이 시뮬레이션 진행 불가
  }
}

/*
 * [한국어]
 * TrafficPattern::reset - 패턴 상태 초기화 (기본 구현)
 *
 * 기본 구현은 아무것도 하지 않는다.
 * TrafficManager::Run()이 각 시뮬레이션 반복(sim) 시작 시 호출한다.
 * RandomPermutationTrafficPattern처럼 상태를 가진 패턴은 이 함수를 재정의할 수 있다.
 *
 * 호출 체인:
 *   TrafficManager::Run() → _traffic_pattern[c]->reset() → [TrafficPattern::reset()]
 */
void TrafficPattern::reset()
{
  // [한국어] 기본 구현: 상태 없음 — 아무 동작도 하지 않는다.
}

/*
 * [한국어]
 * TrafficPattern::New - 문자열 이름으로 구체 트래픽 패턴 객체를 생성하는 팩토리
 *
 * @pattern: "bitcomp", "tornado(4,2,1)", "hotspot(0,3)" 형식의 패턴 이름+파라미터 문자열.
 *           괄호 안의 파라미터는 콤마로 구분된 정수 목록.
 * @nodes: 네트워크의 총 노드 수.
 * @config: 패턴 파라미터가 문자열에 없을 때 설정 파일에서 읽을 Configuration 포인터.
 *          NULL이면 파라미터가 부족한 경우 오류 종료.
 * @return: 힙에 생성된 TrafficPattern 파생 객체 포인터.
 *          호출자(TrafficManager 생성자)가 소멸자에서 delete해야 함.
 *
 * 동작 단계:
 *   1. 문자열에서 패턴 이름과 괄호 내 파라미터 문자열을 분리
 *   2. 파라미터 문자열을 정수 벡터로 토큰화
 *   3. 패턴 이름에 따라 적절한 하위 클래스 객체를 new로 생성
 *   4. 파라미터가 부족하면 config에서 읽거나 오류 종료
 *
 * 호출 체인:
 *   TrafficManager 생성자 → [TrafficPattern::New()] → 구체 패턴 생성자
 */
TrafficPattern * TrafficPattern::New(string const & pattern, int nodes,
				     Configuration const * const config)
{
  string pattern_name; // [한국어] 괄호 이전의 패턴 이름 부분 (예: "tornado")
  string param_str;    // [한국어] 괄호 안의 파라미터 문자열 (예: "4,2,1")

  size_t left = pattern.find_first_of('('); // [한국어] 여는 괄호 위치 탐색
  if(left == string::npos) { // [한국어] 괄호 없음 — 파라미터 없는 패턴 (예: "bitcomp")
    pattern_name = pattern; // [한국어] 전체 문자열이 패턴 이름
  } else {
    pattern_name = pattern.substr(0, left); // [한국어] 괄호 이전까지가 패턴 이름
    size_t right = pattern.find_last_of(')'); // [한국어] 닫는 괄호 위치 탐색
    if(right == string::npos) { // [한국어] 닫는 괄호 없음 — 괄호 이후 전체를 파라미터로
      param_str = pattern.substr(left+1); // [한국어] '(' 다음부터 끝까지
    } else {
      param_str = pattern.substr(left+1, right-left-1); // [한국어] '(' 다음부터 ')' 이전까지
    }
  }

  vector<string> params = tokenize_str(param_str); // [한국어] 파라미터 문자열을 ','로 토큰화하여 벡터로

  TrafficPattern * result = NULL; // [한국어] 반환할 패턴 객체 포인터 초기화

  if(pattern_name == "bitcomp") { // [한국어] 비트 보수 패턴 — 파라미터 불필요
    result = new BitCompTrafficPattern(nodes);
  } else if(pattern_name == "transpose") { // [한국어] 2D 전치 패턴 — 파라미터 불필요
    result = new TransposeTrafficPattern(nodes);
  } else if(pattern_name == "bitrev") { // [한국어] 비트 역순 패턴 — 파라미터 불필요
    result = new BitRevTrafficPattern(nodes);
  } else if(pattern_name == "shuffle") { // [한국어] 셔플(사이클릭 시프트) 패턴 — 파라미터 불필요
    result = new ShuffleTrafficPattern(nodes);
  } else if(pattern_name == "randperm") { // [한국어] 랜덤 순열 패턴 — 시드 필요
    int perm_seed = -1; // [한국어] 순열 생성 시드 초기화
    if(params.empty()) { // [한국어] 문자열에 시드 파라미터 없음
      if(config) {
	perm_seed = config->GetInt("perm_seed"); // [한국어] 설정 파일의 "perm_seed" 값 사용
      } else {
	cout << "Error: Missing parameter for random permutation traffic pattern: " << pattern << endl;
	exit(-1); // [한국어] 시드 없이 랜덤 순열 생성 불가 — 오류 종료
      }
    } else {
      perm_seed = atoi(params[0].c_str()); // [한국어] 문자열 파라미터를 정수 시드로 변환
    }
    result = new RandomPermutationTrafficPattern(nodes, perm_seed);
  } else if(pattern_name == "uniform") { // [한국어] 완전 균등 랜덤 패턴 — 파라미터 불필요
    result = new UniformRandomTrafficPattern(nodes);
  } else if(pattern_name == "background") { // [한국어] 배경 트래픽 패턴 — 제외 노드 목록 필요
    vector<int> excludes = tokenize_int(params[0]); // [한국어] 첫 파라미터를 정수 목록으로 변환
    result = new UniformBackgroundTrafficPattern(nodes, excludes);
  } else if(pattern_name == "diagonal") { // [한국어] 대각선 패턴 — 파라미터 불필요
    result = new DiagonalTrafficPattern(nodes);
  } else if(pattern_name == "asymmetric") { // [한국어] 비대칭 패턴 — 파라미터 불필요
    result = new AsymmetricTrafficPattern(nodes);
  } else if(pattern_name == "taper64") { // [한국어] Taper64 패턴 — nodes==64 필요
    result = new Taper64TrafficPattern(nodes);
  } else if(pattern_name == "bad_dragon") { // [한국어] Dragonfly 최악 순열 — k, n 필요
    bool missing_params = false; // [한국어] 파라미터 부족 여부 추적 플래그
    int k = -1; // [한국어] Dragonfly 기수(radix) 초기화
    if(params.size() < 1) { // [한국어] k 파라미터 문자열에 없음
      if(config) {
	k = config->GetInt("k"); // [한국어] 설정 파일에서 k 읽기
      } else {
	missing_params = true; // [한국어] 설정도 없으면 파라미터 부족으로 표시
      }
    } else {
      k = atoi(params[0].c_str()); // [한국어] 첫 파라미터를 k로 변환
    }
    int n = -1; // [한국어] Dragonfly 차원 수 초기화
    if(params.size() < 2) { // [한국어] n 파라미터 문자열에 없음
      if(config) {
	n = config->GetInt("n"); // [한국어] 설정 파일에서 n 읽기
      } else {
	missing_params = true; // [한국어] 설정도 없으면 파라미터 부족으로 표시
      }
    } else {
      n = atoi(params[1].c_str()); // [한국어] 두 번째 파라미터를 n으로 변환
    }
    if(missing_params) { // [한국어] k 또는 n 파라미터를 얻지 못한 경우
      cout << "Error: Missing parameters for dragonfly bad permutation traffic pattern: " << pattern << endl;
      exit(-1); // [한국어] 파라미터 없이 Dragonfly 패턴 생성 불가 — 오류 종료
    }
    result = new BadPermDFlyTrafficPattern(nodes, k, n);
  } else if((pattern_name == "tornado") || (pattern_name == "neighbor") ||
	    (pattern_name == "badperm_yarc")) { // [한국어] k-ary n-cube 기반 패턴들 — k, n, xr 필요
    bool missing_params = false; // [한국어] 파라미터 부족 여부 추적 플래그
    int k = -1; // [한국어] 기수(radix) 초기화
    if(params.size() < 1) { // [한국어] k 파라미터 없음
      if(config) {
	k = config->GetInt("k"); // [한국어] 설정 파일에서 k 읽기
      } else {
	missing_params = true;
      }
    } else {
      k = atoi(params[0].c_str()); // [한국어] 첫 파라미터를 k로 변환
    }
    int n = -1; // [한국어] 차원 수 초기화
    if(params.size() < 2) { // [한국어] n 파라미터 없음
      if(config) {
	n = config->GetInt("n"); // [한국어] 설정 파일에서 n 읽기
      } else {
	missing_params = true;
      }
    } else {
      n = atoi(params[1].c_str()); // [한국어] 두 번째 파라미터를 n으로 변환
    }
    int xr = -1; // [한국어] 라우터당 노드 수(expansion ratio) 초기화
    if(params.size() < 3) { // [한국어] xr 파라미터 없음
      if(config) {
	xr = config->GetInt("xr"); // [한국어] 설정 파일에서 xr 읽기
      } else {
	missing_params = true;
      }
    } else {
      xr = atoi(params[2].c_str()); // [한국어] 세 번째 파라미터를 xr로 변환
    }
    if(missing_params) { // [한국어] k, n, xr 중 하나라도 누락된 경우
      cout << "Error: Missing parameters for digit permutation traffic pattern: " << pattern << endl;
      exit(-1); // [한국어] 파라미터 없이 차원 기반 패턴 생성 불가 — 오류 종료
    }
    if(pattern_name == "tornado") { // [한국어] 토네이도 패턴: 각 차원 floor(k/2) 이동
      result = new TornadoTrafficPattern(nodes, k, n, xr);
    } else if(pattern_name == "neighbor") { // [한국어] 이웃 패턴: 각 차원 +1 이동
      result = new NeighborTrafficPattern(nodes, k, n, xr);
    } else if(pattern_name == "badperm_yarc") { // [한국어] YARC 최악 패턴: 행 교체
      result = new BadPermYarcTrafficPattern(nodes, k, n, xr);
    }
  } else if(pattern_name == "hotspot") { // [한국어] 핫스팟 패턴 — 핫스팟 목록 + 가중치
    if(params.empty()) { // [한국어] 파라미터 없으면 "-1" 삽입 → 생성자에서 랜덤 핫스팟 결정
      params.push_back("-1");
    }
    vector<int> hotspots = tokenize_int(params[0]); // [한국어] 첫 파라미터를 핫스팟 번호 목록으로 변환
    for(size_t i = 0; i < hotspots.size(); ++i) { // [한국어] 음수 핫스팟 → 랜덤 노드로 교체
      if(hotspots[i] < 0) {
	hotspots[i] = RandomInt(nodes - 1); // [한국어] 음수 값은 랜덤 노드 번호로 대체
      }
    }
    vector<int> rates; // [한국어] 핫스팟별 가중치 벡터 초기화
    if(params.size() >= 2) { // [한국어] 두 번째 파라미터에 가중치 목록이 있는 경우
      rates = tokenize_int(params[1]); // [한국어] 두 번째 파라미터를 가중치 목록으로 변환
      rates.resize(hotspots.size(), rates.back()); // [한국어] 핫스팟 수보다 짧으면 마지막 값으로 채움
    } else { // [한국어] 가중치 없으면 모두 1로 균등 설정
      rates.resize(hotspots.size(), 1);
    }
    result = new HotSpotTrafficPattern(nodes, hotspots, rates);
  } else { // [한국어] 알 수 없는 패턴 이름 — 오류 종료
    cout << "Error: Unknown traffic pattern: " << pattern << endl;
    exit(-1); // [한국어] 알 수 없는 패턴으로는 시뮬레이션 진행 불가
  }
  return result; // [한국어] 힙에 생성된 패턴 객체 반환 (호출자가 delete 책임)
}

/*
 * [한국어]
 * PermutationTrafficPattern::PermutationTrafficPattern - 순열 패턴 중간 생성자
 *
 * @nodes: 네트워크 노드 수.
 *
 * 상위 TrafficPattern(nodes)에 위임하는 것 외에 추가 초기화 없음.
 * 순열 패턴 계층의 공통 조상으로서 인터페이스 명시 목적.
 *
 * 호출 체인:
 *   BitPermutationTrafficPattern 또는 DigitPermutationTrafficPattern 생성자
 *     → [PermutationTrafficPattern(nodes)] → TrafficPattern(nodes)
 */
PermutationTrafficPattern::PermutationTrafficPattern(int nodes)
  : TrafficPattern(nodes) // [한국어] 상위 기반 클래스 생성자에 노드 수 위임
{

}

/*
 * [한국어]
 * BitPermutationTrafficPattern::BitPermutationTrafficPattern - 비트 순열 패턴 중간 생성자
 *
 * @nodes: 네트워크 노드 수. 2의 거듭제곱이어야 함.
 *
 * nodes가 2의 거듭제곱인지 검사한다.
 * 비트 연산(보수, 역전, 시프트)은 노드 수가 2의 거듭제곱일 때만 올바른 결과를 냄.
 * 검사 방법: (nodes & -nodes) == nodes — 2의 거듭제곱이면 단 하나의 비트만 설정됨.
 *
 * 호출 체인:
 *   BitComp/Transpose/BitRev/Shuffle 생성자
 *     → [BitPermutationTrafficPattern(nodes)] → PermutationTrafficPattern(nodes)
 */
BitPermutationTrafficPattern::BitPermutationTrafficPattern(int nodes)
  : PermutationTrafficPattern(nodes) // [한국어] 상위 순열 패턴 생성자에 위임
{
  if((nodes & -nodes) != nodes) { // [한국어] 2의 거듭제곱 검사: 하나의 비트만 있으면 통과
    cout << "Error: Bit permutation traffic patterns require the number of "
	 << "nodes to be a power of two." << endl;
    exit(-1); // [한국어] 2의 거듭제곱 아니면 비트 연산 결과가 노드 범위를 벗어남 — 오류 종료
  }
}

/*
 * [한국어]
 * BitCompTrafficPattern::BitCompTrafficPattern - 비트 보수 패턴 생성자
 *
 * @nodes: 네트워크 노드 수 (2의 거듭제곱 필요).
 *
 * 추가 초기화 없음 — BitPermutationTrafficPattern의 2의 거듭제곱 검사에 의존.
 *
 * 호출 체인:
 *   TrafficPattern::New("bitcomp", nodes) → [BitCompTrafficPattern(nodes)]
 */
BitCompTrafficPattern::BitCompTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes) // [한국어] 상위 클래스의 2의 거듭제곱 검사 수행
{

}

/*
 * [한국어]
 * BitCompTrafficPattern::dest - 비트 보수 목적지 계산
 *
 * @source: 소스 노드 번호 (0 이상 _nodes 미만).
 * @return: source의 모든 유효 비트를 반전한 목적지 번호.
 *
 * 계산식: ~source & (_nodes-1)
 * _nodes-1은 유효 비트의 마스크 역할을 한다.
 * 예) _nodes=8, source=3(0b011) → mask=7(0b111) → ~3&7=4(0b100)
 *
 * 이 패턴은 완전히 결정론적이며 상태가 없다.
 * 비트 보수 패턴은 fat-tree 같은 대칭 네트워크에서
 * 가장 많은 홉을 요구하는 최악 케이스를 생성한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BitCompTrafficPattern::dest(source)]
 */
int BitCompTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int const mask = _nodes - 1; // [한국어] 유효 비트 마스크: log2(_nodes)개 비트가 모두 1 (예: 8노드 → 0b111)
  return ~source & mask; // [한국어] source의 모든 유효 비트를 반전하여 목적지 계산
}

/*
 * [한국어]
 * TransposeTrafficPattern::TransposeTrafficPattern - 전치 패턴 생성자
 *
 * @nodes: 네트워크 노드 수 (짝수 제곱의 2의 거듭제곱, 예: 4, 16, 64, 256).
 *
 * _shift = log2(sqrt(nodes)) = log2(nodes)/2를 계산한다.
 * nodes의 비트 폭을 세어(_shift 증가) 절반으로 나눈다(_shift >>= 1).
 * 비트 폭이 홀수이면(짝수 제곱이 아니면) 오류 종료.
 *
 * 호출 체인:
 *   TrafficPattern::New("transpose", nodes) → [TransposeTrafficPattern(nodes)]
 */
TransposeTrafficPattern::TransposeTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes), _shift(0) // [한국어] 2의 거듭제곱 검사 + _shift=0 초기화
{
  while(nodes >>= 1) { // [한국어] nodes를 1비트씩 오른쪽 시프트하며 비트 폭을 셈
    ++_shift; // [한국어] 비트 폭 카운트 증가 (log2(nodes)가 됨)
  }
  if(_shift % 2) { // [한국어] log2(nodes)가 홀수이면 2D 격자(sqrt가 정수)가 아님
    cout << "Error: Transpose traffic pattern requires the number of nodes to "
	 << "be an even power of two." << endl;
    exit(-1); // [한국어] 예: 8노드(2^3)는 2D 격자 불가 — 오류 종료
  }
  _shift >>= 1; // [한국어] _shift = log2(nodes)/2 = log2(sqrt(nodes)) (예: 16노드 → _shift=2)
}

/*
 * [한국어]
 * TransposeTrafficPattern::dest - 2D 전치 목적지 계산
 *
 * @source: 소스 노드 번호.
 * @return: source의 상위 _shift 비트와 하위 _shift 비트를 교환한 값.
 *
 * 2D 격자(sqrt(nodes) × sqrt(nodes))에서 source를 (row, col)로 해석하고
 * (col, row)로 전치하는 효과이다.
 * 예) _nodes=16(_shift=2), source=6(0b0110):
 *   mask_lo=0b0011, mask_hi=0b1100
 *   하위=source>>2 & mask_lo=1, 상위=source<<2 & mask_hi=8
 *   dest=1|8=9
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [TransposeTrafficPattern::dest(source)]
 */
int TransposeTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int const mask_lo = (1 << _shift) - 1;      // [한국어] 하위 _shift 비트 마스크 (예: _shift=2 → 0b0011)
  int const mask_hi = mask_lo << _shift;       // [한국어] 상위 _shift 비트 마스크 (예: _shift=2 → 0b1100)
  // [한국어] 하위 절반을 상위로, 상위 절반을 하위로 교환:
  // (source >> _shift) & mask_lo: 상위 _shift 비트를 하위로 이동
  // (source << _shift) & mask_hi: 하위 _shift 비트를 상위로 이동
  return (((source >> _shift) & mask_lo) | ((source << _shift) & mask_hi));
}

/*
 * [한국어]
 * BitRevTrafficPattern::BitRevTrafficPattern - 비트 역순 패턴 생성자
 *
 * @nodes: 네트워크 노드 수 (2의 거듭제곱 필요).
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("bitrev", nodes) → [BitRevTrafficPattern(nodes)]
 */
BitRevTrafficPattern::BitRevTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes) // [한국어] 2의 거듭제곱 검사 수행
{

}

/*
 * [한국어]
 * BitRevTrafficPattern::dest - 비트 역순 목적지 계산
 *
 * @source: 소스 노드 번호.
 * @return: source의 이진 표현을 log2(_nodes) 비트 기준으로 역순으로 뒤집은 값.
 *
 * 루프에서 매 반복마다 source의 최하위 비트(source%2)를 result의 최하위 비트로,
 * source를 1비트 오른쪽 시프트, result를 1비트 왼쪽 시프트하여 비트 역순 구현.
 * 예) _nodes=8(3비트), source=6(0b110):
 *   i=1: result=0<<1|0=0, source=3
 *   i=2: result=0<<1|1=1, source=1
 *   i=4: result=1<<1|1=3(0b011), source=0
 *   결과: 3
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BitRevTrafficPattern::dest(source)]
 */
int BitRevTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int result = 0; // [한국어] 역순 비트를 누적할 결과 초기화
  for(int n = _nodes; n > 1; n >>= 1) { // [한국어] log2(_nodes)번 반복 (n은 절반씩 줄어듦)
    result = (result << 1) | (source % 2); // [한국어] result를 왼쪽 시프트하고 source의 현재 LSB를 추가
    source >>= 1; // [한국어] source의 다음 비트를 LSB 위치로 이동
  }
  return result; // [한국어] 비트 역순으로 뒤집힌 목적지 반환
}

/*
 * [한국어]
 * ShuffleTrafficPattern::ShuffleTrafficPattern - 셔플 패턴 생성자
 *
 * @nodes: 네트워크 노드 수 (2의 거듭제곱 필요).
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("shuffle", nodes) → [ShuffleTrafficPattern(nodes)]
 */
ShuffleTrafficPattern::ShuffleTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes) // [한국어] 2의 거듭제곱 검사 수행
{

}

/*
 * [한국어]
 * ShuffleTrafficPattern::dest - 사이클릭 왼쪽 1비트 시프트 목적지 계산
 *
 * @source: 소스 노드 번호.
 * @return: source를 사이클릭 왼쪽 1비트 시프트한 값.
 *
 * 계산식: ((source << 1) & (_nodes-1)) | bool(shifted & _nodes)
 * - source를 왼쪽으로 1비트 시프트
 * - 하위 log2(_nodes) 비트 유지 (&(_nodes-1))
 * - 시프트 오버플로 비트(_nodes 위치)가 있으면 최하위 비트(LSB)로 순환
 * 예) _nodes=8, source=5(0b101):
 *   shifted=0b1010, 하위3비트=0b010=2, 오버플로=0b1000&0b1000=1
 *   dest=2|1=3(0b011)
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [ShuffleTrafficPattern::dest(source)]
 */
int ShuffleTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int const shifted = source << 1; // [한국어] source를 왼쪽으로 1비트 시프트 (상위 비트 오버플로 가능)
  // [한국어] (shifted & (_nodes-1)): 하위 log2(_nodes) 비트 유지 (범위 내 부분)
  // [한국어] bool(shifted & _nodes): 오버플로 비트가 있으면 1, 없으면 0 → LSB로 순환
  return ((shifted & (_nodes - 1)) | bool(shifted & _nodes));
}

/*
 * [한국어]
 * DigitPermutationTrafficPattern::DigitPermutationTrafficPattern - 진수 순열 패턴 중간 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @k: 기수(radix) — 각 차원의 노드/라우터 수.
 * @n: 차원 수(dimension).
 * @xr: 라우터당 노드 수(expansion ratio, 기본값=1).
 *
 * _k, _n, _xr을 초기화한다.
 * k-ary n-cube에서 nodes = (k*xr)^n 관계가 성립해야 하나
 * 생성자에서 직접 검사하지는 않음(호출자가 보장).
 *
 * 호출 체인:
 *   Tornado/Neighbor/BadPermYarc/BadPermDFly 생성자
 *     → [DigitPermutationTrafficPattern(nodes, k, n, xr)]
 */
DigitPermutationTrafficPattern::DigitPermutationTrafficPattern(int nodes, int k,
							       int n, int xr)
  : PermutationTrafficPattern(nodes), _k(k), _n(n), _xr(xr) // [한국어] 기반 초기화 + k/n/xr 저장
{

}

/*
 * [한국어]
 * TornadoTrafficPattern::TornadoTrafficPattern - 토네이도 패턴 생성자
 *
 * @nodes, @k, @n, @xr: DigitPermutationTrafficPattern으로 위임.
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("tornado", nodes, config) → [TornadoTrafficPattern(nodes, k, n, xr)]
 */
TornadoTrafficPattern::TornadoTrafficPattern(int nodes, int k, int n, int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr) // [한국어] 진수 순열 기반 초기화
{

}

/*
 * [한국어]
 * TornadoTrafficPattern::dest - 토네이도 목적지 계산
 *
 * @source: 소스 노드 번호.
 * @return: 각 차원에서 floor((_xr*_k+1)/2)-1 이동한 목적지 노드 번호.
 *
 * k-ary n-cube에서 노드 번호를 n개의 (_xr*_k)진수 digit으로 분해하고,
 * 각 digit에 ((_xr*_k+1)/2 - 1)을 더하여(모듈로) 재합성한다.
 * 이 이동량은 각 차원에서 이동 거리가 최대가 되는 바로 전 값(floor(k/2)-1)이다.
 * 결과적으로 모든 링크에 균등한 최대 부하를 만들어 네트워크 포화를 유도한다.
 *
 * 예) _k=4, _n=2, _xr=1, _nodes=16:
 *   source=0(d0=0, d1=0):
 *     d0 이동량=(4+1)/2-1=1, d1 이동량=1
 *     dest=1*1 + 1*4=5
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [TornadoTrafficPattern::dest(source)]
 */
int TornadoTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사

  int offset = 1;  // [한국어] 현재 차원의 위치값 (처음 차원은 1, 다음은 _xr*_k, ...)
  int result = 0;  // [한국어] 누적 목적지 노드 번호 초기화

  for(int n = 0; n < _n; ++n) { // [한국어] 각 차원(0 ~ _n-1)에 대해 반복
    // [한국어] (source / offset) % (_xr * _k): 현재 차원의 digit 추출
    // [한국어] + ((_xr*_k+1)/2 - 1): 이 차원에서의 이동량 (floor(k*xr/2)-1 방향)
    // [한국어] % (_xr * _k): 순환(모듈로)하여 유효 digit 범위로 제한
    result += offset *
      (((source / offset) % (_xr * _k) + ((_xr * _k + 1) / 2 - 1)) % (_xr * _k));
    offset *= (_xr * _k); // [한국어] 다음 차원의 위치값 = 현재 차원 크기 배수로 증가
  }
  return result; // [한국어] 모든 차원의 digit을 재합성한 목적지 노드 번호
}

/*
 * [한국어]
 * NeighborTrafficPattern::NeighborTrafficPattern - 이웃 패턴 생성자
 *
 * @nodes, @k, @n, @xr: DigitPermutationTrafficPattern으로 위임.
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("neighbor", nodes, config) → [NeighborTrafficPattern(nodes, k, n, xr)]
 */
NeighborTrafficPattern::NeighborTrafficPattern(int nodes, int k, int n, int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr) // [한국어] 진수 순열 기반 초기화
{

}

/*
 * [한국어]
 * NeighborTrafficPattern::dest - 이웃 목적지 계산
 *
 * @source: 소스 노드 번호.
 * @return: 각 차원에서 +1 이동한 목적지 노드 번호.
 *
 * 토네이도와 구조가 동일하나, 이동량이 ((_xr*_k+1)/2 - 1) 대신 1이다.
 * 각 차원에서 인접 노드로만 이동하는 로컬 트래픽 패턴.
 * 링크 가중치가 낮고 홉 카운트가 1~n으로 제한되는 시나리오를 모의한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [NeighborTrafficPattern::dest(source)]
 */
int NeighborTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사

  int offset = 1; // [한국어] 현재 차원의 위치값 초기화
  int result = 0; // [한국어] 누적 목적지 노드 번호 초기화

  for(int n = 0; n < _n; ++n) { // [한국어] 각 차원(0 ~ _n-1)에 대해 반복
    // [한국어] 현재 차원 digit에 +1을 더하고 모듈로(_xr*_k)로 순환
    result += offset *
      (((source / offset) % (_xr * _k) + 1) % (_xr * _k));
    offset *= (_xr * _k); // [한국어] 다음 차원의 위치값으로 갱신
  }
  return result; // [한국어] 모든 차원 digit을 +1 이동하여 재합성한 목적지 노드 번호
}

/*
 * [한국어]
 * RandomPermutationTrafficPattern::RandomPermutationTrafficPattern - 랜덤 순열 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @seed: 순열 생성 시드.
 *
 * _dest 벡터를 nodes 크기로 할당하고 randomize(seed)를 호출하여
 * 전단사(bijective) 순열 테이블을 구성한다.
 * 동일 시드면 항상 동일 순열이 생성되어 실험 재현성을 보장한다.
 *
 * 호출 체인:
 *   TrafficPattern::New("randperm(seed)", nodes) → [RandomPermutationTrafficPattern(nodes, seed)]
 */
RandomPermutationTrafficPattern::RandomPermutationTrafficPattern(int nodes,
								 int seed)
  : TrafficPattern(nodes) // [한국어] 기반 클래스 초기화 (유효 노드 수 검사)
{
  _dest.resize(nodes); // [한국어] 노드 수만큼 목적지 테이블 공간 할당
  randomize(seed); // [한국어] 시드로 전단사 순열 테이블 생성
}

/*
 * [한국어]
 * RandomPermutationTrafficPattern::randomize - 시드 기반 전단사 순열 테이블 생성
 *
 * @seed: 순열 생성 시드.
 *
 * 전역 RNG 상태를 보존하고 별도의 시드로 순열을 생성하여 복원한다.
 * Fisher-Yates 변형 알고리즘으로 완전 랜덤 전단사 순열을 생성한다:
 *   - _dest를 -1로 초기화 (-1은 아직 할당 안 된 목적지 표시)
 *   - i=0부터 _nodes-1까지: 남은 미할당 슬롯 중 ind번째에 i를 할당
 * 결과: _dest[j]는 source=j의 목적지가 됨 (전단사 보장)
 *
 * 호출 체인:
 *   RandomPermutationTrafficPattern 생성자 → [randomize(seed)]
 */
void RandomPermutationTrafficPattern::randomize(int seed)
{
  unsigned long prev_seed = RandomIntLong( ); // [한국어] 전역 RNG 현재 시드 저장 (복원을 위해)
  RandomSeed(seed); // [한국어] 지정된 시드로 RNG 재설정 — 재현 가능한 순열 생성

  _dest.assign(_nodes, -1); // [한국어] 모든 목적지를 -1(미할당)로 초기화

  for(int i = 0; i < _nodes; ++i) { // [한국어] 각 소스 i에 목적지를 할당
    int ind = RandomInt(_nodes - 1 - i); // [한국어] 남은 미할당 슬롯 수에서 랜덤 인덱스 선택

    int j = 0;   // [한국어] _dest 배열 탐색 인덱스
    int cnt = 0; // [한국어] 발견한 미할당 슬롯 수 카운터

    // [한국어] _dest[j]==-1인 슬롯을 cnt가 ind에 도달할 때까지 탐색
    while((cnt < ind) || (_dest[j] != -1)) { // [한국어] ind번째 미할당 슬롯을 찾을 때까지
      if(_dest[j] == -1) { // [한국어] 미할당 슬롯을 발견하면 카운터 증가
	++cnt;
      }
      ++j; // [한국어] 다음 슬롯으로 이동
      assert(j < _nodes); // [한국어] 배열 범위 초과 검사 (전단사 보장에 의해 항상 통과)
    }

    _dest[j] = i; // [한국어] j번째 슬롯에 소스 i를 목적지로 할당 (j가 i의 목적지가 됨)
  }

  RandomSeed(prev_seed); // [한국어] 전역 RNG 상태 복원 — 다른 모듈의 RNG 시퀀스 보존
}

/*
 * [한국어]
 * RandomPermutationTrafficPattern::dest - 순열 테이블 조회
 *
 * @source: 소스 노드 번호.
 * @return: _dest[source] — 전단사 순열에 의한 목적지 노드 번호.
 *
 * O(1) 테이블 조회. 순열은 생성자에서 미리 계산되어 있다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [RandomPermutationTrafficPattern::dest(source)]
 */
int RandomPermutationTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  assert((_dest[source] >= 0) && (_dest[source] < _nodes)); // [한국어] 순열 값 유효 범위 검사
  return _dest[source]; // [한국어] 전단사 순열 테이블에서 목적지 직접 반환
}

/*
 * [한국어]
 * RandomTrafficPattern::RandomTrafficPattern - 랜덤 패턴 중간 생성자
 *
 * @nodes: 네트워크 노드 수.
 * 상위 TrafficPattern(nodes)에 위임하는 것 외에 추가 초기화 없음.
 *
 * 호출 체인:
 *   Uniform/Background/Diagonal/Asymmetric/Taper64 생성자
 *     → [RandomTrafficPattern(nodes)] → TrafficPattern(nodes)
 */
RandomTrafficPattern::RandomTrafficPattern(int nodes)
  : TrafficPattern(nodes) // [한국어] 기반 클래스에 노드 수 유효성 검사 위임
{

}

/*
 * [한국어]
 * UniformRandomTrafficPattern::UniformRandomTrafficPattern - 균등 랜덤 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("uniform", nodes) → [UniformRandomTrafficPattern(nodes)]
 */
UniformRandomTrafficPattern::UniformRandomTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes) // [한국어] 랜덤 패턴 중간 생성자에 위임
{

}

/*
 * [한국어]
 * UniformRandomTrafficPattern::dest - 균등 랜덤 목적지 반환
 *
 * @source: 사용되지 않음 (출발지 무관 균등 분포).
 * @return: [0, _nodes-1] 범위에서 균등 랜덤으로 선택된 목적지 번호.
 *
 * 매 호출마다 독립적인 랜덤값 생성. 자기 자신(source==dest)도 가능.
 * 이론적 최대 처리량은 1/(1-1/N)로 N이 크면 1에 근접한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [UniformRandomTrafficPattern::dest(source)]
 */
int UniformRandomTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  return RandomInt(_nodes - 1); // [한국어] [0, _nodes-1] 균등 분포 랜덤 목적지 반환
}

/*
 * [한국어]
 * UniformBackgroundTrafficPattern::UniformBackgroundTrafficPattern - 배경 트래픽 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @excluded_nodes: 목적지 후보에서 제외할 노드 번호 목록.
 *
 * excluded_nodes의 각 원소를 _excluded 집합에 삽입한다.
 * 유효 범위(0 이상 _nodes 미만) 검사 포함.
 *
 * 호출 체인:
 *   TrafficPattern::New("background(excluded_list)", nodes)
 *     → [UniformBackgroundTrafficPattern(nodes, excludes)]
 */
UniformBackgroundTrafficPattern::UniformBackgroundTrafficPattern(int nodes, vector<int> excluded_nodes)
  : RandomTrafficPattern(nodes) // [한국어] 랜덤 패턴 기반 초기화
{
  for(size_t i = 0; i < excluded_nodes.size(); ++i) { // [한국어] 제외 노드 목록을 집합에 삽입
    int const node = excluded_nodes[i]; // [한국어] 현재 제외할 노드 번호
    assert((node >= 0) && (node < _nodes)); // [한국어] 노드 번호 유효 범위 검사
    _excluded.insert(node); // [한국어] _excluded 집합에 삽입 (중복 자동 무시)
  }
}

/*
 * [한국어]
 * UniformBackgroundTrafficPattern::dest - 제외 노드를 피하는 균등 랜덤 목적지 반환
 *
 * @source: 사용되지 않음.
 * @return: _excluded에 없는 균등 랜덤 목적지 노드 번호.
 *
 * 반복-거부(rejection sampling) 방식: 무한 루프로 랜덤값을 뽑아
 * 제외 집합에 없는 값이 나올 때까지 반복한다.
 * _excluded가 거의 모든 노드를 포함하면 루프 횟수가 매우 많아질 수 있음.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [UniformBackgroundTrafficPattern::dest(source)]
 */
int UniformBackgroundTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사

  int result; // [한국어] 후보 목적지 번호

  do {
    result = RandomInt(_nodes - 1); // [한국어] [0, _nodes-1] 균등 랜덤 목적지 후보 생성
  } while(_excluded.count(result) > 0); // [한국어] 제외 집합에 있으면 재시도 (rejection sampling)

  return result; // [한국어] 제외 집합에 없는 유효 목적지 반환
}

/*
 * [한국어]
 * DiagonalTrafficPattern::DiagonalTrafficPattern - 대각선 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("diagonal", nodes) → [DiagonalTrafficPattern(nodes)]
 */
DiagonalTrafficPattern::DiagonalTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes) // [한국어] 랜덤 패턴 기반 초기화
{

}

/*
 * [한국어]
 * DiagonalTrafficPattern::dest - 대각선 목적지 반환
 *
 * @source: 소스 노드 번호.
 * @return: 50% 확률로 (source+1)%_nodes, 나머지 50%로 source 자신.
 *
 * RandomInt(2) == 0이면 다음 노드로, 아니면 자기 자신으로 보낸다.
 * 이 패턴은 링 형상(ring topology)에서 인접 통신 위주의 부하를 만들어낸다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [DiagonalTrafficPattern::dest(source)]
 */
int DiagonalTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  // [한국어] RandomInt(2): [0,2] 중 랜덤값. ==0이면 인접 노드, 아니면 자기 자신
  return ((RandomInt(2) == 0) ? ((source + 1) % _nodes) : source);
}

/*
 * [한국어]
 * AsymmetricTrafficPattern::AsymmetricTrafficPattern - 비대칭 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * 추가 초기화 없음.
 *
 * 호출 체인:
 *   TrafficPattern::New("asymmetric", nodes) → [AsymmetricTrafficPattern(nodes)]
 */
AsymmetricTrafficPattern::AsymmetricTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes) // [한국어] 랜덤 패턴 기반 초기화
{

}

/*
 * [한국어]
 * AsymmetricTrafficPattern::dest - 비대칭 목적지 반환
 *
 * @source: 소스 노드 번호.
 * @return: (source % half) + (랜덤으로 0 또는 half).
 *
 * _nodes를 upper half / lower half로 분리.
 * source의 절반 내 위치(source % half)는 유지하되 upper/lower는 랜덤 선택.
 * 예) _nodes=8, half=4, source=5(5%4=1):
 *   dest = 1 + (RandomInt(1) ? 4 : 0) → 1 또는 5
 * 모든 소스가 같은 열 인덱스의 상반 또는 하반 노드로만 보내어
 * 상하반 연결 링크에 집중 부하를 만든다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [AsymmetricTrafficPattern::dest(source)]
 */
int AsymmetricTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int const half = _nodes / 2; // [한국어] 노드 수의 절반 — 상반/하반 경계
  // [한국어] source % half: 소스의 절반 내 열 인덱스 (상반/하반 무관)
  // [한국어] RandomInt(1) ? half : 0: 50%로 상반(+half), 50%로 하반(+0)
  return (source % half) + (RandomInt(1) ? half : 0);
}

/*
 * [한국어]
 * Taper64TrafficPattern::Taper64TrafficPattern - Taper-64 패턴 생성자
 *
 * @nodes: 반드시 64이어야 함. 다른 값이면 오류 종료.
 *
 * 호출 체인:
 *   TrafficPattern::New("taper64", nodes) → [Taper64TrafficPattern(nodes)]
 */
Taper64TrafficPattern::Taper64TrafficPattern(int nodes)
  : RandomTrafficPattern(nodes) // [한국어] 랜덤 패턴 기반 초기화
{
  if(nodes != 64) { // [한국어] Taper64는 정확히 64노드 8x8 격자 가정
    cout << "Error: Tthe Taper64 traffic pattern requires the number of nodes "
	 << "to be exactly 64." << endl;
    exit(-1); // [한국어] 64노드가 아니면 동작 보장 불가 — 오류 종료
  }
}

/*
 * [한국어]
 * Taper64TrafficPattern::dest - Taper-64 혼합 목적지 반환
 *
 * @source: 소스 노드 번호 (0-63).
 * @return: 50% 확률로 8x8 격자 내 근방 노드, 나머지 50%로 전체 랜덤 노드.
 *
 * 근방 노드 계산: (64 + source + 8*([-1,0,1] 중 랜덤) + ([-1,0,1] 중 랜덤)) % 64
 * - 8 배수 부분: 격자의 행 이동 (+8, 0, -8)
 * - 1 배수 부분: 격자의 열 이동 (+1, 0, -1)
 * - + 64 후 모듈로 64: 음수 인덱스 방지
 * HPC 벤치마크의 지역성(인접 PE간 통신) + 글로벌 통신 혼합 모의.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [Taper64TrafficPattern::dest(source)]
 */
int Taper64TrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사 (0~63)
  if(RandomInt(1)) { // [한국어] 50% 확률로 근방 노드 선택
    // [한국어] 8*(RandomInt(2)-1): -8, 0, +8 중 랜덤 행 오프셋 (8x8 격자에서 행 이동)
    // [한국어] (RandomInt(2)-1): -1, 0, +1 중 랜덤 열 오프셋
    // [한국어] +64 후 %64: 음수 결과를 양수로 정규화
    return ((64 + source + 8 * (RandomInt(2) - 1) + (RandomInt(2) - 1)) % 64);
  } else { // [한국어] 50% 확률로 전체 랜덤 목적지 선택
    return RandomInt(_nodes - 1); // [한국어] [0, 63] 전체 균등 랜덤
  }
}

/*
 * [한국어]
 * BadPermDFlyTrafficPattern::BadPermDFlyTrafficPattern - Dragonfly 최악 순열 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @k: 기수(radix) — Dragonfly에서 그룹 내 라우터 수 = k.
 * @n: 차원 수(여기서는 계층 레벨 수, 일반적으로 2).
 * xr=1로 고정 (Dragonfly는 라우터당 노드 확장 없음).
 *
 * 호출 체인:
 *   TrafficPattern::New("bad_dragon(k,n)", nodes, config)
 *     → [BadPermDFlyTrafficPattern(nodes, k, n)]
 */
BadPermDFlyTrafficPattern::BadPermDFlyTrafficPattern(int nodes, int k, int n)
  : DigitPermutationTrafficPattern(nodes, k, n, 1) // [한국어] xr=1로 고정하여 진수 순열 기반 초기화
{

}

/*
 * [한국어]
 * BadPermDFlyTrafficPattern::dest - Dragonfly 최악 순열 목적지 반환
 *
 * @source: 소스 노드 번호.
 * @return: 소스 그룹 외부의 랜덤 노드 번호.
 *
 * Dragonfly 네트워크 구조:
 *   그룹당 라우터 수 = 2*k, 그룹당 노드 수 = grp_size_nodes = 2*k*k
 *
 * 소스가 속한 그룹(source / grp_size_nodes)을 제외한 다른 그룹의 랜덤 노드를 선택.
 * 계산식: (rand % grp_size_nodes + (current_group+1)*grp_size_nodes) % _nodes
 * 이는 모든 그룹 간 링크(글로벌 링크)를 동시에 포화시켜
 * Dragonfly의 적응형 라우팅(valiant routing)을 무력화한다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BadPermDFlyTrafficPattern::dest(source)]
 */
int BadPermDFlyTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사

  int const grp_size_routers = 2 * _k;              // [한국어] 그룹당 라우터 수 = 2*k
  int const grp_size_nodes = grp_size_routers * _k; // [한국어] 그룹당 노드 수 = 2*k*k

  // [한국어] RandomInt(grp_size_nodes-1): 그룹 내 랜덤 위치 오프셋
  // [한국어] ((source/grp_size_nodes)+1)*grp_size_nodes: 다음 그룹의 시작 노드 번호
  // [한국어] % _nodes: 전체 노드 수로 모듈로하여 순환 (마지막 그룹에서 첫 그룹으로)
  return ((RandomInt(grp_size_nodes - 1) + ((source / grp_size_nodes) + 1) * grp_size_nodes) % _nodes);
}

/*
 * [한국어]
 * BadPermYarcTrafficPattern::BadPermYarcTrafficPattern - YARC 최악 순열 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @k: 기수(각 열의 라우터 수).
 * @n: 차원 수.
 * @xr: 라우터당 노드 수(expansion ratio, 기본값=1).
 *
 * 호출 체인:
 *   TrafficPattern::New("badperm_yarc(k,n,xr)", nodes, config)
 *     → [BadPermYarcTrafficPattern(nodes, k, n, xr)]
 */
BadPermYarcTrafficPattern::BadPermYarcTrafficPattern(int nodes, int k, int n,
						     int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr) // [한국어] 진수 순열 기반 초기화
{

}

/*
 * [한국어]
 * BadPermYarcTrafficPattern::dest - YARC 최악 순열 목적지 반환
 *
 * @source: 소스 노드 번호.
 * @return: 소스의 행(row)을 랜덤 열(column)에 매핑한 목적지.
 *
 * YARC 네트워크 구조:
 *   행(row) = source / (_xr*_k): 소스가 속한 행
 *   열(column) = source % (_xr*_k): 소스가 속한 열
 *
 * dest = 랜덤 열 * (_xr*_k) + row
 * 즉, 소스의 행을 유지하되 열을 완전히 랜덤으로 변경한다.
 * 이는 열 간 링크를 최대 포화시키는 최악 케이스를 만든다.
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [BadPermYarcTrafficPattern::dest(source)]
 */
int BadPermYarcTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사
  int const row = source / (_xr * _k); // [한국어] 소스의 행 번호 추출 (전체 열 수로 나눔)
  // [한국어] RandomInt((_xr*_k)-1): 랜덤 열 선택 (0 ~ _xr*_k-1)
  // [한국어] * (_xr*_k): 열 번호를 노드 번호로 변환 (열 시작 노드)
  // [한국어] + row: 해당 열의 row번째 노드가 목적지
  return RandomInt((_xr * _k) - 1) * (_xr * _k) + row;
}

/*
 * [한국어]
 * HotSpotTrafficPattern::HotSpotTrafficPattern - 핫스팟 패턴 생성자
 *
 * @nodes: 네트워크 노드 수.
 * @hotspots: 핫스팟 노드 번호 목록 (비어 있으면 assert 실패).
 * @rates: 각 핫스팟의 가중치 목록 (비어 있으면 모두 1, 짧으면 마지막 값으로 채움).
 *
 * _max_val = sum(_rates) - 1을 계산한다 (-1로 시작 후 각 rate 누적).
 * dest()에서 RandomInt(_max_val)의 상한으로 사용되어 가중 확률 분포를 구현한다.
 *
 * 호출 체인:
 *   TrafficPattern::New("hotspot(nodes,rates)", nodes)
 *     → [HotSpotTrafficPattern(nodes, hotspots, rates)]
 */
HotSpotTrafficPattern::HotSpotTrafficPattern(int nodes, vector<int> hotspots,
					     vector<int> rates)
  : TrafficPattern(nodes), _hotspots(hotspots), _rates(rates), _max_val(-1) // [한국어] 기반 초기화 + 필드 초기화
{
  assert(!_hotspots.empty()); // [한국어] 핫스팟이 하나도 없으면 패턴 동작 불가
  size_t const size = _hotspots.size(); // [한국어] 핫스팟 수 (이후 반복에 사용)
  // [한국어] _rates가 핫스팟 수보다 짧으면 마지막 rate로 채움; _rates가 비어 있으면 1로 채움
  _rates.resize(size, _rates.empty() ? 1 : _rates.back());
  for(size_t i = 0; i < size; ++i) { // [한국어] 각 핫스팟의 유효성 검사 및 _max_val 계산
    int const hotspot = _hotspots[i]; // [한국어] i번째 핫스팟 노드 번호
    assert((hotspot >= 0) && (hotspot < _nodes)); // [한국어] 핫스팟 노드 번호 유효 범위 검사
    int const rate = _rates[i]; // [한국어] i번째 핫스팟의 가중치
    assert(rate >= 0); // [한국어] 가중치는 0 이상이어야 함 (음수 가중치 불가)
    _max_val += rate; // [한국어] _max_val 누적: 초기값 -1 + 모든 rate 합 = sum(rates) - 1
  }
}

/*
 * [한국어]
 * HotSpotTrafficPattern::dest - 가중 랜덤 핫스팟 목적지 반환
 *
 * @source: 사용되지 않음 (핫스팟 목적지는 소스 무관).
 * @return: _rates 가중치에 비례하여 선택된 핫스팟 노드 번호.
 *
 * 단일 핫스팟이면 항상 그 핫스팟을 반환 (O(1)).
 * 다중 핫스팟이면:
 *   1. RandomInt(_max_val)으로 [0, sum(rates)-1] 범위 랜덤값 pct 획득
 *   2. _rates[0], _rates[1], ... 를 순차 누적하며 pct가 어느 핫스팟 범위에 속하는지 결정
 *   3. 선형 탐색 O(|_hotspots|)로 해당 핫스팟 반환
 *
 * 가중치가 클수록 해당 핫스팟이 선택될 확률이 높다.
 * 마지막 핫스팟은 별도 처리 (루프 이후 assert로 범위 확인 후 반환).
 *
 * 호출 체인:
 *   TrafficManager::_GeneratePacket() → [HotSpotTrafficPattern::dest(source)]
 */
int HotSpotTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes)); // [한국어] 소스 번호 유효 범위 검사

  if(_hotspots.size() == 1) { // [한국어] 핫스팟이 하나뿐이면 항상 그 노드 반환
    return _hotspots[0];
  }

  int pct = RandomInt(_max_val); // [한국어] [0, sum(rates)-1] 범위의 랜덤값 — 가중 선택에 사용

  for(size_t i = 0; i < (_hotspots.size() - 1); ++i) { // [한국어] 마지막 핫스팟 제외하고 순차 탐색
    int const limit = _rates[i]; // [한국어] i번째 핫스팟의 가중치 범위
    if(limit > pct) { // [한국어] pct가 이 핫스팟의 가중치 범위 내에 있으면 선택
      return _hotspots[i]; // [한국어] i번째 핫스팟 노드 반환
    } else {
      pct -= limit; // [한국어] i번째 핫스팟 범위를 제외하고 나머지 pct로 계속 탐색
    }
  }
  assert(_rates.back() > pct); // [한국어] 마지막 핫스팟의 가중치가 남은 pct보다 커야 함 (정합성 검사)
  return _hotspots.back(); // [한국어] 마지막 핫스팟 노드 반환 (루프에서 걸리지 않은 경우)
}
