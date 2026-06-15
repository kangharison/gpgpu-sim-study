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
 * [한국어 설명] GPGPU-Sim 전용 NoC 설정 클래스 선언 (intersim_config.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim이 Booksim2 NoC 시뮬레이터를 사용할 때 필요한 GPU 특화
 * 설정 항목들을 추가하는 IntersimConfig 클래스를 선언한다. Booksim2의 기본
 * 설정 클래스인 BookSimConfig를 상속하고, 생성자(intersim_config.cpp)에서
 * GPU NoC에 특화된 추가 파라미터 기본값을 등록한다.
 * 이 클래스의 인스턴스는 icnt_wrapper_init()에서 생성되어 Booksim2 네트워크와
 * intersim 전용 기능(flit_size, buffer 크기, 매핑 등)을 설정하는 데 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 시뮬레이터 → icnt_wrapper.cc(icnt_wrapper_init) →
 * IntersimConfig 생성 → gpgpusim.config 파싱 →
 * Booksim2 네트워크 및 라우터 초기화
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — 시뮬레이터 초기화 단계 (단일 스레드)
 *
 * === 타 모듈과의 연결 ===
 * - config_utils.hpp: Configuration 기반 클래스 — _int_map, _str_map, _float_map
 *   저장소와 GetInt(), GetStr(), AddStrField() 등 파라미터 읽기/쓰기 인터페이스 제공
 * - booksim_config.hpp: BookSimConfig — Booksim2 표준 파라미터(topology, routing,
 *   vc_buf_size, num_vcs 등)를 선언하는 부모 클래스
 * - intersim_config.cpp: IntersimConfig 생성자 구현 — GPU 특화 파라미터 기본값 등록
 * - icnt_wrapper.cc: IntersimConfig 인스턴스를 생성하고 gpgpusim.config를 파싱하여
 *   Booksim2 초기화에 전달한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - IntersimConfig()              : 생성자 — GPU 특화 파라미터 기본값을 _int_map/_str_map에 등록
 * - (상속) GetInt("flit_size")   : NoC 플릿 크기(바이트) 조회 — 기본값 32
 * - (상속) GetInt("network_count"): 독립 NoC 개수 조회 — 기본값 2 (sh→mem, mem→sh)
 * - (상속) GetStr("memory_node_map"): SM-메모리 노드 매핑 문자열 조회
 * - (상속) GetInt("input_buffer_size"): 입력 버퍼 크기 조회 (0이면 vc_buf_size 사용)
 * - (상속) GetInt("perfect_icnt"): 완벽 인터커넥트(지연 없음) 모드 여부 조회
 */

#ifndef _INTERSIM_CONFIG_HPP_
#define _INTERSIM_CONFIG_HPP_

#include "config_utils.hpp"    // [한국어] Configuration 기반 클래스 포함 — _int_map, AddStrField() 등
#include "booksim_config.hpp"  // [한국어] BookSimConfig 포함 — Booksim2 표준 NoC 파라미터를 상속받기 위해

/*
 * [한국어] IntersimConfig — GPGPU-Sim 전용 NoC 설정 클래스
 *
 * BookSimConfig를 상속하여 Booksim2 기본 파라미터(topology, routing, num_vcs 등)를
 * 모두 물려받고, 추가로 GPGPU-Sim GPU NoC에 특화된 파라미터들을 생성자에서 등록한다.
 *
 * 주요 추가 파라미터 (intersim_config.cpp 참조):
 *   - perfect_icnt          : 완벽 인터커넥트 모드 (0=시뮬, 1=지연 없음)
 *   - fixed_lat_per_hop     : 홉당 고정 지연 (0=시뮬, >0=고정 지연 사용)
 *   - use_map               : SM-메모리 노드 매핑 사용 여부 (기본 1)
 *   - memory_node_map       : SM-메모리 노드 번호 매핑 문자열
 *   - flit_size             : 플릿 크기(바이트, 기본 32)
 *   - input_buffer_size     : 입력 버퍼 크기 (0이면 vc_buf_size 사용)
 *   - ejection_buffer_size  : 이젝션 버퍼 크기 (0이면 vc_buf_size 사용)
 *   - boundary_buffer_size  : 경계 버퍼 크기 (기본 16)
 *   - network_count         : 독립 NoC 개수 (기본 2: sh→mem, mem→sh)
 *   - DISPLAY_LAT_DIST 등   : 통계 출력 플래그들
 *
 * 사용 방법:
 *   IntersimConfig* config = new IntersimConfig();
 *   config->ParseFile("gpgpusim.config"); // 파일의 설정이 기본값을 덮어씀
 *   network = new Network(config, ...);
 */
class IntersimConfig : public BookSimConfig {
public:
  /*
   * [한국어]
   * IntersimConfig - GPU NoC 특화 설정 파라미터를 등록하는 생성자.
   *
   * @return: (생성자 반환값 없음)
   *
   * 부모 클래스 BookSimConfig()를 먼저 호출하여 Booksim2 기본 파라미터를 등록하고,
   * 이후 GPU-특화 파라미터들의 기본값을 _int_map과 _str_map에 등록한다.
   * gpgpusim.config 파싱 시 여기서 등록한 기본값들이 파일의 값으로 덮어씌워진다.
   * intersim_config.cpp에 구현되어 있다.
   *
   * 호출 체인:
   *   icnt_wrapper_init() → new IntersimConfig() → BookSimConfig() → Configuration()
   */
  IntersimConfig();
};

#endif
