// Copyright (c) 2009-2013, Tor M. Aamodt, Dongdong Li, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * [한국어 설명] GPGPU-Sim 전용 NoC 설정 파라미터 기본값 등록 (intersim_config.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 IntersimConfig 생성자를 구현하며, GPGPU-Sim의 GPU NoC 시뮬레이션에
 * 필요한 추가 파라미터들의 기본값을 Configuration의 내부 맵(_int_map, _str_map)에
 * 등록한다. 이 기본값들은 이후 gpgpusim.config 파일 파싱 시 사용자 설정으로
 * 덮어씌워진다. Booksim2의 BookSimConfig 생성자가 먼저 표준 Booksim2 파라미터를
 * 등록하고, IntersimConfig 생성자가 그 위에 GPU 특화 파라미터를 추가한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이터 초기화 흐름:
 *   gpgpu_sim_init() → icnt_wrapper_init() →
 *   new IntersimConfig() [이 파일의 생성자] →
 *   config->ParseFile("gpgpusim.config") [기본값 덮어쓰기] →
 *   Booksim2 Network 생성 [config 값으로 토폴로지/라우팅 결정]
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — 시뮬레이터 초기화 단계 (단일 스레드, 1회 실행)
 *
 * === 타 모듈과의 연결 ===
 * - intersim_config.hpp: IntersimConfig 클래스 선언 포함
 * - booksim_config.hpp: BookSimConfig 부모 클래스 — BookSimConfig() 생성자가
 *   먼저 호출되어 topology, routing, num_vcs, vc_buf_size 등 표준 파라미터 등록
 * - config_utils.hpp: Configuration 기반 클래스 — _int_map, _str_map 저장소와
 *   AddStrField() 메서드 제공
 * - icnt_wrapper.cc: IntersimConfig 인스턴스를 생성하고 파싱하는 소비자
 * - local_interconnect.cc: GetInt("input_buffer_size"), GetInt("flit_size") 등을
 *   읽어 버퍼/플릿 크기를 결정하는 소비자
 *
 * === 주요 함수/구조체 요약 ===
 * - IntersimConfig::IntersimConfig() : 유일한 함수. GPU 특화 파라미터 기본값 등록.
 *   등록 파라미터 목록:
 *     perfect_icnt        (0) — 완벽 인터커넥트 모드 비활성화
 *     fixed_lat_per_hop   (0) — 고정 지연 모드 비활성화
 *     use_map             (1) — SM-메모리 노드 매핑 사용
 *     memory_node_map    ("") — SM-메모리 매핑 문자열 (빈 값=자동)
 *     flit_size          (32) — 플릿 크기 32바이트
 *     input_buffer_size   (0) — 0이면 vc_buf_size 사용
 *     ejection_buffer_size(0) — 0이면 vc_buf_size 사용
 *     boundary_buffer_size(16)— 경계 버퍼 16슬롯
 *     network_count       (2) — sh→mem, mem→sh 2개 독립 NoC
 *     enable_link_stats   (0) — 링크/VC 통계 비활성화
 *     MATLAB_OUTPUT       (0) — MATLAB 형식 출력 비활성화
 *     DISPLAY_LAT_DIST    (0) — 레이턴시 분포 출력 비활성화
 *     DISPLAY_HOP_DIST    (0) — 홉 수 분포 출력 비활성화
 *     DISPLAY_PAIR_LATENCY(0) — 소스-목적지 쌍 평균 레이턴시 출력 비활성화
 */

#include "intersim_config.hpp" // [한국어] IntersimConfig 클래스 선언 포함

/*
 * [한국어]
 * IntersimConfig::IntersimConfig - GPGPU-Sim 전용 NoC 파라미터 기본값을 등록하는 생성자.
 *
 * @return: (생성자 반환값 없음)
 *
 * BookSimConfig 부모 생성자가 묵시적으로 먼저 호출되어 Booksim2의 표준 파라미터
 * (topology, routing, num_vcs, vc_buf_size, packet_size 등)가 _int_map 등에 등록된다.
 * 이후 이 생성자 본문에서 GPGPU-Sim GPU NoC에 특화된 파라미터들의 기본값을 추가한다.
 * 등록된 기본값들은 icnt_wrapper.cc에서 gpgpusim.config 파일을 파싱할 때 사용자 설정으로
 * 덮어씌워질 수 있다. 사용자가 설정 파일에서 명시하지 않으면 여기서 등록한 기본값이 사용된다.
 *
 * 호출 체인:
 *   icnt_wrapper_init() → new IntersimConfig() → (묵시적) BookSimConfig() →
 *   IntersimConfig::IntersimConfig() 본문 실행
 */
IntersimConfig::IntersimConfig()
{
  // Add options for intersim

  _int_map["perfect_icnt"] = 0;
  // [한국어] perfect_icnt: 완벽 인터커넥트 모드 비활성화 (기본값 0 = 실제 NoC 시뮬레이션 수행).
  // 1로 설정하면 NoC를 시뮬레이션하지 않고 모든 패킷이 지연 없이 즉시 도달하는 이상적 모드.
  // fixed_lat_per_hop 설정을 override하는 상위 플래그이다.
  // gpgpusim.config에서: -icnt_perfect_icnt 1 처럼 설정 가능.

  _int_map["fixed_lat_per_hop"] = 0;
  // [한국어] fixed_lat_per_hop: 홉당 고정 지연 모드 (기본값 0 = 완전 NoC 시뮬레이션).
  // 0이 아닌 값으로 설정하면 NoC를 사이클 레벨로 시뮬레이션하지 않고,
  // 대신 패킷이 (홉 수 × fixed_lat_per_hop) 사이클 후 목적지에 도달하는 단순 모델 사용.
  // perfect_icnt가 0일 때만 이 설정이 효과를 발휘한다.
  // gpgpusim.config에서: -icnt_fixed_lat_per_hop 2 처럼 설정 가능.

  _int_map["use_map"] = 1;
  // [한국어] use_map: SM(Shader Core)과 메모리 노드 번호 매핑 사용 여부 (기본값 1 = 사용).
  // 1이면 memory_node_map 문자열에 따라 SM과 메모리 파티션의 노드 번호를 재매핑한다.
  // GPU의 SM 수와 메모리 파티션 수에 맞게 NoC 노드 번호를 논리적으로 배치할 때 필요하다.

  // config SMs and memory nodes map
  AddStrField("memory_node_map", "");
  // [한국어] memory_node_map: SM→메모리 파티션 노드 번호 매핑 문자열 (기본값 빈 문자열 = 자동 매핑).
  // 형식 예: "0,1,2,3" — NoC에서 메모리 파티션에 해당하는 노드 번호 목록.
  // 비어 있으면 icnt_wrapper_init()이 SM 수와 메모리 파티션 수를 기반으로 자동 계산한다.
  // AddStrField()로 등록하는 이유: 문자열 필드는 _str_map에 별도 등록이 필요하기 때문.
  // gpgpusim.config에서: -icnt_memory_node_map "0 1 2 3" 처럼 설정 가능.

  _int_map["flit_size"] = 32;
  // [한국어] flit_size: NoC 플릿(flit) 크기(바이트, 기본값 32).
  // 플릿(flit: flow control unit)은 NoC에서 흐름 제어의 기본 단위이다.
  // GPU 메모리 패킷(캐시 라인 크기 128바이트 등)을 이 크기로 분할하여 여러 플릿으로 전송한다.
  // local_interconnect.cc와 icnt_wrapper.cc에서 읽어 패킷을 플릿으로 쪼갤 때 사용한다.
  // gpgpusim.config에서: -icnt_flit_size 32 처럼 설정 가능.

  _int_map["input_buffer_size"] = 0;
  // [한국어] input_buffer_size: 라우터 입력 포트 버퍼 크기(슬롯 수, 기본값 0).
  // 0이면 Booksim2 기본값인 vc_buf_size를 그대로 사용한다.
  // 0이 아닌 값으로 설정하면 각 VC의 입력 버퍼를 이 크기로 명시적으로 설정한다.
  // 버퍼 크기는 백프레셔(back-pressure) 발생 임계점과 레이턴시에 직접 영향을 준다.
  // gpgpusim.config에서: -icnt_input_buffer_size 8 처럼 설정 가능.

  _int_map["ejection_buffer_size"] = 0;
  // [한국어] ejection_buffer_size: 이젝션(목적지 단말) 버퍼 크기(슬롯 수, 기본값 0).
  // 0이면 vc_buf_size를 대신 사용한다. NoC에서 패킷이 목적지 단말(SM 또는 메모리)에
  // 도달할 때 최종적으로 들어가는 이젝션 버퍼의 크기를 결정한다.
  // 이 버퍼가 가득 차면 NoC로의 역압력(back-pressure)이 발생한다.
  // gpgpusim.config에서: -icnt_ejection_buffer_size 8 처럼 설정 가능.

  _int_map["boundary_buffer_size"] = 16;
  // [한국어] boundary_buffer_size: SM-NoC 경계 버퍼 크기(슬롯 수, 기본값 16).
  // SM(Shader Core)과 NoC 사이의 경계에 위치하는 버퍼로, SM이 생성한 메모리 요청
  // 패킷을 NoC로 전달하기 전에 임시로 보관한다. SM과 NoC 사이의 속도 불일치를 흡수한다.
  // local_interconnect.cc가 이 값을 읽어 단말-라우터 사이 버퍼를 생성한다.
  // gpgpusim.config에서: -icnt_boundary_buffer_size 16 처럼 설정 가능.

  // FIXME: obsolete, unsupport configs
  _int_map["output_extra_latency"] = 0;
  // [한국어] output_extra_latency: 출력 포트에 추가할 여분 지연 사이클 (기본값 0).
  // 현재는 deprecated(폐기)되어 실제로 사용되지 않는다. 과거 호환성을 위해 등록만 한다.

  _int_map["network_count"] = 2;
  // [한국어] network_count: 독립적으로 생성할 NoC 네트워크 수 (기본값 2).
  // 2로 설정하면 sh→mem (SM에서 메모리로의 요청) 방향과 mem→sh (메모리에서 SM으로의
  // 응답) 방향에 대해 각각 독립된 NoC 인스턴스를 생성한다. 이는 GPU 내부 NoC의
  // 요청/응답 트래픽을 분리하여 간섭을 줄이는 설계 선택이다.
  // icnt_wrapper.cc가 이 값을 읽어 network 배열 크기를 결정한다.

  _int_map["enable_link_stats"] = 0;
  // [한국어] enable_link_stats: 링크별/VC별 활용도 통계 출력 활성화 (기본값 0 = 비활성).
  // 1로 설정하면 시뮬레이션 종료 시 각 NoC 링크와 VC의 평균 활용률을 출력한다.
  // 성능 분석에 유용하지만 시뮬레이션 오버헤드가 증가한다.

  //stats
  _int_map["MATLAB_OUTPUT"] = 0;
  // [한국어] MATLAB_OUTPUT: 통계 데이터를 MATLAB 친화적 형식으로 출력할지 여부 (기본값 0 = 비활성).
  // 1로 설정하면 MATLAB에서 직접 로드 가능한 행렬/배열 형식으로 통계를 출력한다.
  // 오프라인 분석에 MATLAB을 사용하는 연구 환경에서 활성화한다.

  _int_map["DISPLAY_LAT_DIST"] = 0;
  // [한국어] DISPLAY_LAT_DIST: 패킷 레이턴시 분포(히스토그램) 출력 활성화 (기본값 0 = 비활성).
  // 1로 설정하면 시뮬레이션 종료 시 패킷별 레이턴시 빈도 분포를 출력한다.
  // 평균 레이턴시 외에 꼬리 레이턴시(tail latency) 분석이 필요할 때 활성화한다.

  _int_map["DISPLAY_HOP_DIST"] = 0;
  // [한국어] DISPLAY_HOP_DIST: 패킷 홉 수 분포(히스토그램) 출력 활성화 (기본값 0 = 비활성).
  // 1로 설정하면 시뮬레이션 종료 시 각 패킷이 몇 홉을 거쳐 목적지에 도달했는지
  // 분포를 출력한다. 라우팅 알고리즘이나 토폴로지 선택의 효율을 평가할 때 사용한다.

  _int_map["DISPLAY_PAIR_LATENCY"] = 0;
  // [한국어] DISPLAY_PAIR_LATENCY: 소스-목적지 쌍별 평균 레이턴시 출력 활성화 (기본값 0 = 비활성).
  // 1로 설정하면 시뮬레이션 종료 시 모든 (소스, 목적지) 노드 쌍에 대한 평균 레이턴시를
  // 행렬 형태로 출력한다. 트래픽 불균형이나 핫스팟(hot-spot) 분석에 유용하다.
  // 노드 수가 많을수록 출력 크기가 quadratic하게 증가하므로 주의해서 활성화한다.
}
