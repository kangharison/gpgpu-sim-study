// $Id: booksim_config.hpp 5487 2013-02-27 08:16:18Z qtedq $

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
 * [한국어 설명] BookSim NoC 시뮬레이터 설정 클래스 헤더 (booksim_config.hpp)
 *
 * === 파일의 역할 ===
 * BookSim2 NoC 시뮬레이터의 모든 설정 파라미터를 관리하는 두 클래스를 선언한다.
 * - BookSimConfig: 네트워크 토폴로지, 라우터, 트래픽, 시뮬레이션 파라미터를 포괄하는
 *   메인 설정 클래스이다. gpgpusim.config 파일에서 읽어 온 intersim 설정이 여기에 저장된다.
 * - PowerConfig: NoC 전력 모델(링크/버퍼 소비 전력 추정)에 필요한 공정 파라미터를 저장한다.
 * 두 클래스 모두 Configuration(config_utils.hpp) 기반 키-값 저장소 방식으로
 * 파라미터를 관리하며, 생성자에서 모든 기본값을 등록한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 컨텍스트: 시뮬레이터 호스트 유저스페이스.
 * 호출 체인:
 *   icnt_wrapper_init() → intersim2_create()
 *     → BookSimConfig 생성자 (기본값 일괄 등록)
 *     → Configuration::ParseFile() (설정 파일 파싱, 덮어쓰기)
 *     → Network/TrafficManager 생성자에 config 전달
 * PowerConfig는 전력 모델 활성화 시 별도로 인스턴스화된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: config_utils.hpp (Configuration 기반 클래스, AddStrField, _int_map, _float_map)
 * - 피의존: trafficmanager.cc, network.cc, router/*.cc, batchtrafficmanager.cpp 등
 *   시뮬레이터의 거의 모든 모듈이 BookSimConfig 레퍼런스를 인자로 받는다.
 * - GPGPU-Sim 연결: icnt_wrapper.cc가 intersim2_create() 내에서 이 클래스를 생성한다.
 * - 데이터 흐름: .booksim 또는 gpgpusim.config → BookSimConfig → 각 시뮬레이터 모듈
 *
 * === 주요 함수/구조체 요약 ===
 * BookSimConfig::BookSimConfig() — 모든 NoC 파라미터의 기본값을 등록하는 생성자
 *   (구현은 booksim_config.cpp에 있으며, 약 300줄 분량의 기본값 등록 코드를 포함)
 * PowerConfig::PowerConfig() — 전력 모델 공정 파라미터 기본값(모두 0) 등록
 * 상속 구조: BookSimConfig → Configuration → 키-값 저장소 (int/float/string 맵)
 */

#ifndef _BOOKSIM_CONFIG_HPP_
#define _BOOKSIM_CONFIG_HPP_

#include "config_utils.hpp" // [한국어] Configuration 기반 클래스 — AddStrField, _int_map, _float_map, GetInt 등 제공

/*
 * [한국어]
 * BookSimConfig - BookSim2 NoC 시뮬레이터 전체 설정 클래스
 *
 * 이 클래스는 Configuration(config_utils.hpp)을 상속하여,
 * NoC 시뮬레이션에 필요한 모든 파라미터의 기본값을 생성자에서 일괄 등록한다.
 * 파라미터 범주는 크게 다음과 같다:
 *   1) 네트워크 토폴로지 (topology, k, n, c, x, y 등)
 *   2) 라우터 설정 (num_vcs, vc_buf_size, routing_delay, allocator 종류 등)
 *   3) 트래픽 패턴 (traffic, injection_rate, packet_size, read/write VC 범위 등)
 *   4) 시뮬레이션 제어 (sim_type, warmup_periods, sample_period, seed 등)
 *   5) 전력 모델 연동 (sim_power, tech_file, channel_width 등)
 *
 * Configuration::ParseFile()이 호출되면 .booksim 설정 파일의 값으로 덮어쓴다.
 * 이후 GetInt()/GetFloat()/GetStr()로 각 모듈이 파라미터를 조회한다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계에서 1회 생성됨 (싱글스레드).
 *
 * 호출 체인:
 *   icnt_wrapper_init() → [BookSimConfig 생성자] → TrafficManager/Network 생성자
 */
class BookSimConfig : public Configuration {
protected:
  /* [한국어] 이 클래스는 Configuration이 제공하는 _int_map, _float_map, _str_map을
   * 그대로 상속하므로 추가 필드가 없다.
   * 기본값 등록은 전적으로 생성자(booksim_config.cpp)에서 수행된다. */

public:
  BookSimConfig( );
  /* [한국어] 생성자 — 모든 NoC 파라미터의 기본값을 _int_map/_float_map/_str_map에 등록한다.
   * 구현은 booksim_config.cpp에 있으며, 약 70여 개의 파라미터를 순서대로 등록한다.
   * 이후 Configuration::ParseFile()이 필요한 항목만 덮어쓰는 방식으로 동작한다.
   * 호출자: icnt_wrapper_init() 또는 BookSim 독립 실행 시 main() */
};

#endif

#ifndef _POWER_CONFIG_HPP_
#define _POWER_CONFIG_HPP_

#include "config_utils.hpp" // [한국어] Configuration 기반 클래스 재포함 (두 번째 include guard로 보호됨)

/*
 * [한국어]
 * PowerConfig - NoC 전력 모델용 공정(process technology) 파라미터 설정 클래스
 *
 * BookSim 전력 모델이 활성화될 때(sim_power=1) 사용된다.
 * 트랜지스터 크기(H_INVD2, W_INVD2 등), 전압(Vdd), 누설 전류(IoffSRAM, IoffP, IoffN),
 * 게이트/드레인 커패시턴스(Cg, Cd), 금속 배선 파라미터(LAMBDA, MetalPitch, Rw, Cw_gnd 등)를
 * 저장한다. 모든 기본값은 0이며, 실제 공정 파라미터 파일(tech_file)에서 읽혀 덮어쓰인다.
 *
 * 실행 컨텍스트: 전력 계산 전용 — 시뮬레이터 초기화 시 1회 생성.
 * 호출 체인: power_module → [PowerConfig 생성자] → ParseFile(tech_file)
 */
class PowerConfig : public Configuration {
public:
  PowerConfig( );
  /* [한국어] 생성자 — 전력 모델에 필요한 공정 파라미터 기본값(모두 0)을 등록한다.
   * 실제 값은 tech_file 파싱 후 덮어씌워진다.
   * 파라미터 목록: H/W_INVD2, H/W_DFQD1, H/W_ND2D1, H/W_SRAM (게이트 크기),
   *   Vdd(공급 전압), R(저항), IoffSRAM/IoffP/IoffN(누설 전류),
   *   Cg_pwr/Cd_pwr/Cgdl/Cg/Cd(커패시턴스), LAMBDA/MetalPitch(공정 스케일),
   *   Rw/Cw_gnd/Cw_cpl/wire_length(배선 파라미터). */
};

#endif
