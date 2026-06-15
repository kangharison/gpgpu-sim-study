// $Id: power_module.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim2 NoC 전력/면적 추정 모듈 헤더 (power_module.hpp)
 *
 * === 파일의 역할 ===
 * BookSim2가 시뮬레이션한 NoC(Network-on-Chip)의 채널, 입력 버퍼, 크로스바 스위치,
 * 출력 포트를 대상으로 동적/정적(누설) 전력 및 면적을 추정하는 Power_Module 클래스를
 * 선언한다. AccelWattch와 별개로, BookSim2 자체의 NoC 전력 모델을 제공하며,
 * wire repeater 최적화, SRAM 버퍼, 크로스바 커패시턴스 모델 등을 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/power_module.hpp
 *     → intersim2/power/power_module.cpp: 전력/면적 계산 수행
 *     → intersim2/networks/network.hpp: Network* 를 통해 채널/라우터 목록 획득
 *     → intersim2/routers/iq_router.hpp: 각 라우터의 BufferMonitor/SwitchMonitor 접근
 *     → intersim2/booksim_config.hpp: PowerConfig를 통해 공정 파라미터 로드
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - module.hpp: Module 기반 클래스 (이름, 시뮬레이션 시간 조회)
 *   - network.hpp: Network, Router, FlitChannel 클래스 정의
 *   - config_utils.hpp: Configuration 기반 옵션 읽기
 *   - flitchannel.hpp: FlitChannel 클래스 (GetLatency, GetActivity)
 *   - switch_monitor.hpp / buffer_monitor.hpp: 활동량 카운터
 * 피의존:
 *   - interconnect_interface.cpp / main.cpp 등: sim_power 옵션 활성화 시
 *     Power_Module::run() 호출하여 전력/면적 보고서 출력
 * 설정 연동 (gpgpusim.config / BookSim config):
 *   - "sim_power"      : 전력 모델 활성화 여부
 *   - "tech_file"      : 공정 파라미터 파일 경로 (PowerConfig 파싱에 사용)
 *   - "power_output_file" : 결과 파일 경로 (현재는 표준 출력으로도 출력)
 *   - "classes"        : 트래픽 클래스 수
 *   - "channel_width"  : 채널 폭 (bit); 모든 전력/면적 공식의 기본 폭
 *   - "channel_sweep"  : 채널 폭 스윕 감소 폭 (현재 코드에서는 저장만 하고 추가 루프 없음)
 *   - "num_vcs"        : 가상 채널(VC) 수 → 버퍼 깊이 계산
 *   - "vc_buf_size"    : VC당 버퍼 크기(flit 단위) → 버퍼 깊이 계산
 *
 * === 주요 함수/구조체 요약 ===
 * struct wire        — 반복기(repeater) 최적화 결과: L(길이), K/M/N(구조 파라미터)
 * Power_Module       — NoC 전체 전력/면적 계산기
 *   calcChannel()    — FlitChannel 하나의 동적/정적 전력 및 면적 계산
 *   wireOptimize()   — 길이 L에 대해 반복기 수/크기/단계를 최적화하여 wire_map 캐싱
 *   calcBuffer()     — BufferMonitor 기반 입력 버퍼 전력/면적 계산
 *   calcSwitch()     — SwitchMonitor 기반 크로스바 및 출력 포트 전력/면적 계산
 *   area*()          — 채널/크로스바/입력/출력 모듈 면적 공식
 *   run()            — Network 전체를 순회하며 전력/면적 합산 및 보고서 출력
 */

#ifndef _POWER_MODULE_HPP_
#define _POWER_MODULE_HPP_

#include <map>                // [한국어] 길이별 wire 최적화 결과를 캐싱할 wire_map용

#include "module.hpp"         // [한국어] Module 기반 클래스 (Power_Module의 이름 및 시뮬레이션 시간)
#include "network.hpp"        // [한국어] Network, Router, FlitChannel 클래스 정의
#include "config_utils.hpp"   // [한국어] Configuration 기반 설정 읽기
#include "flitchannel.hpp"    // [한국어] FlitChannel: GetLatency(), GetActivity() 사용
#include "switch_monitor.hpp" // [한국어] 크로스바 활동량 집계 객체
#include "buffer_monitor.hpp" // [한국어] 입력 버퍼 활동량 집계 객체

/*
 * [한국어]
 * struct wire — 반복기(repeater) 기반 금속 배선의 최적화 결과를 저장
 *
 * Power_Module::wireOptimize()에서 길이 L을 입력으로 받아
 * K(드라이버 크기 배율), M(수평 단계 수), N(세그먼트 수)를 최적화한 뒤
 * wire_map에 (L → wire) 형태로 캐싱한다.
 */
struct wire{
  double L; // [한국어] 배선 총 길이 (mm)
  double K; // [한국어] 반복기 드라이버 크기 배율
  double M; // [한국어] 한 방향(수평/수직) 반복기 단계 수
  double N; // [한국어] 전체 세그먼트 수 (M*N 형태로 사용)
};

/*
 * [한국어]
 * Power_Module — BookSim2 NoC의 채널/버퍼/스위치/출력 포트 전력 및 면적을 추정하는 클래스
 *
 * Module을 상속받아 시뮬레이션 시간(GetSimTime)을 활용하고,
 * 생성자에서 PowerConfig(tech_file)로부터 공정 파라미터를 읽어온다.
 * run()이 호출되면 Network의 모든 주입/배출/채널과 모든 라우터의
 * BufferMonitor/SwitchMonitor를 순회하며 전력/면적을 합산한다.
 */
class Power_Module : public Module {

protected:
  //network undersimulation
  Network * net; // [한국어] 전력을 계산할 대상 Network 객체 포인터
  int classes;   // [한국어] 트래픽 클래스 수; BookSimConfig "classes"에서 읽어옴
  //all channels are this width
  double channel_width; // [한국어] 채널 폭 (bit); "channel_width" 설정값
  //resimulate all with channel_width decremented by channel_sweep until 0
  double  channel_sweep; // [한국어] 채널 폭 스윕 간격; 현재는 값만 저장됨
  //write result to a tabbed format to file
  string output_file_name; // [한국어] 결과 파일 이름; "power_output_file" 설정값

  //buffer depth
  double depthVC; // [한국어] VC당 버퍼 깊이 (flit 수); "vc_buf_size" 설정값
  //vcs
  double numVC;   // [한국어] 가상 채널(VC) 수; "num_vcs" 설정값

  //store the property of wires based on length
  map<double, wire> wire_map; // [한국어] 배선 길이별 반복기 최적화 결과 캐시

  //////////////////////////////////Constants/////////////////////////////
  //wire length in (mm)
  double wire_length; // [한국어] 기준 배선 길이 (mm); "wire_length" 공정 파라미터
  //////////Metal Parameters////////////
  // Wire left/right coupling capacitance [ F/mm ]
  double Cw_cpl ; // [한국어] 좌우 커플링 커패시턴스 [F/mm]
  // Wire up/down groudn capacitance      [ F/mm ]
  double Cw_gnd  ; // [한국어] 상하 접지 커패시턴스 [F/mm]
  double Cw ;      // [한국어] 총 배선 커패시턴스; Cw = 2*Cw_cpl + 2*Cw_gnd
  double Rw ;      // [한국어] 배선 단위 저항 [Ohm/mm]
  // metal pitch [mm]
  double MetalPitch ; // [한국어] 금속 피치 [mm]


  //////////Device Parameters////////////
  
  double LAMBDA  ;       // [um/LAMBDA] [한국어] 공정 스케일링 팩터
  double Cd   ;           // [F/um] (for Delay) [한국어] 드레인 커패시턴스 (지연용)
  double Cg  ;           // [F/um] (for Delay) [한국어] 게이트 커패시턴스 (지연용)
  double Cgdl  ;           // [F/um] (for Delay) [한국어] 게이트-드레인 커플링 커패시턴스 (지연용)
  
  double Cd_pwr;           // [F/um] (for Power) [한국어] 드레인 커패시턴스 (전력용)
  double Cg_pwr  ;           // [F/um] (for Power) [한국어] 게이트 커패시턴스 (전력용)
				       
  double IoffN  ;            // [A/um] [한국어] NMOS 오프 상태 누설 전류
  double IoffP  ;            // [A/um] [한국어] PMOS 오프 상태 누설 전류
  // Leakage from bitlines, two-port cell  [A]
  double IoffSRAM;  // [한국어] 2포트 SRAM 비트라인 누설 전류 [A]
  // [Ohm] ( D1=1um Inverter)
  double R     ;                         // [한국어] 1um 인버터 등가 저항 [Ohm]
  // [F]   ( D1=1um Inverter - for Power )
  double Ci_delay;   // [한국어] 지연용 입력 커패시턴스 [F]
  // [F]   ( D1=1um Inverter - for Power )
  double Co_delay ;              // [한국어] 지연용 출력 커패시턴스 [F]

  double Ci ; // [한국어] 전력용 입력 커패시턴스 [F]
  double Co ; // [한국어] 전력용 출력 커패시턴스 [F]
  double Vdd  ; // [한국어] 공급 전압 [V]
  double FO4   ;		     // [한국어] 4배 부하 인버터 지연 [s]
  double tCLK ; // [한국어] 클록 주기 [s]
  double fCLK ; // [한국어] 클록 주파수 [Hz]
              
  double H_INVD2; // [한국어] 인버터 셀 높이 (단위: LAMBDA)
  double W_INVD2; // [한국어] 인버터 셀 폭 (단위: LAMBDA)
  double H_DFQD1; // [한국어] DFF 셀 높이 (단위: LAMBDA)
  double W_DFQD1; // [한국어] DFF 셀 폭 (단위: LAMBDA)
  double H_ND2D1; // [한국어] 2입 NAND 셀 높이 (단위: LAMBDA)
  double W_ND2D1; // [한국어] 2입 NAND 셀 폭 (단위: LAMBDA)
  double H_SRAM;  // [한국어] SRAM 셀 높이 (단위: LAMBDA)
  double W_SRAM;  // [한국어] SRAM 셀 폭 (단위: LAMBDA)
  double  ChannelPitch ; // [한국어] 채널 피치 [mm]
  double   CrossbarPitch; // [한국어] 크로스바 피치 [mm]
  ////////////////////////////////End of Constants/////////////////////////////

  /////////////results///////////////////
  double totalTime;           // [한국어] 시뮬레이션 총 사이클 수 (GetSimTime())
  double channelWirePower;    // [한국어] 채널 배선 동적 전력
  double channelClkPower;     // [한국어] 채널 클록 분배 동적 전력
  double channelDFFPower;     // [한국어] 채널 리타이밍 DFF 동적 전력
  double channelLeakPower;    // [한국어] 채널 누설 전력
  double inputReadPower;      // [한국어] 입력 버퍼 읽기 동적 전력
  double inputWritePower;     // [한국어] 입력 버퍼 쓰기 동적 전력
  double inputLeakagePower;   // [한국어] 입력 버퍼 누설 전력
  double switchPower;         // [한국어] 크로스바 데이터 경로 동적 전력
  double switchPowerCtrl;     // [한국어] 크로스바 제어 신호 동적 전력
  double switchPowerLeak;     // [한국어] 크로스바 누설 전력
  double outputPower;         // [한국어] 출력 포트 DFF 동적 전력
  double outputPowerClk;      // [한국어] 출력 포트 클록 동적 전력
  double outputCtrlPower;     // [한국어] 출력 포트 제어 신호 동적 전력
  double channelArea;         // [한국어] 채널 면적
  double switchArea;          // [한국어] 크로스바 면적
  double inputArea;           // [한국어] 입력 버퍼 면적
  double outputArea;          // [한국어] 출력 포트 면적
  double maxInputPort;        // [한국어] (현재 사용되지 않음) 최대 입력 포트 수 관련 값
  double maxOutputPort;       // [한국어] (현재 사용되지 않음) 최대 출력 포트 수 관련 값


  ////////////////////////

  //channels
  /* [한국어] FlitChannel 하나의 전력/면적을 계산 (주입/배출/중간 채널 모두 대상) */
  void calcChannel(const FlitChannel * f);
  /* [한국어] 길이 L에 대한 반복기 최적화; 최초 호출 시 탐색 후 wire_map에 캐싱 */
  wire const & wireOptimize(double l);
  /* [한국어] L/K/M/N 기반 반복 배선의 동적 전력 계산 */
  double powerRepeatedWire(double L, double K, double M, double N);
  /* [한국어] 반복 배선의 누설 전력 계산 */
  double powerRepeatedWireLeak (double K, double M, double N);
  /* [한국어] 채널 클록 분배 전력 계산 (M: 반복기 단계, W: 채널 폭) */
  double powerWireClk (double M, double W);
  /* [한국어] 채널 DFF(리타이밍) 동적 전력 계산 (alpha: 활동률) */
  double powerWireDFF(double M, double W, double alpha);
  
  //memory
  /* [한국어] BufferMonitor의 read/write 카운터를 바탕으로 입력 버퍼 전력/면적 계산 */
  void calcBuffer(const BufferMonitor *bm);
  /* [한국어] 메모리 워드라인 구동 전력 계산 */
  double powerWordLine(double memoryWidth, double memoryDepth);
  /* [한국어] 비트 한 개를 읽을 때의 동적 전력 */
  double powerMemoryBitRead(double memoryDepth);
  /* [한국어] 비트 한 개를 쓸 때의 동적 전력 */
  double powerMemoryBitWrite(double memoryDepth);
  /* [한국어] 비트 한 개의 누설 전력 */
  double powerMemoryBitLeak(double memoryDepth );

  //switch
  /* [한국어] SwitchMonitor의 스위칭 카운터를 바탕으로 크로스바/출력 전력/면적 계산 */
  void calcSwitch(const SwitchMonitor *sm);
  /* [한국어] from → to 경로의 크로스바 데이터 경로 동적 전력 */
  double powerCrossbar(double width, double inputs, double outputs, double from, double to);
  /* [한국어] 크로스바 제어 신호 동적 전력 */
  double powerCrossbarCtrl(double width, double inputs, double outputs);
  /* [한국어] 크로스바 누설 전력 */
  double powerCrossbarLeak (double width, double inputs, double outputs);
  
  //output
  /* [한국어] 출력 포트 제어 신호(Enable) 동적 전력 */
  double powerOutputCtrl(double width);

  //area
  /* [한국어] K/N/M 기반 채널(반복 배선+DFF) 면적 */
  double areaChannel (double K, double N, double M);
  /* [한국어] 입력x출력 크기의 크로스바 면적 */
  double areaCrossbar(double Inputs, double Outputs) ;
  /* [한국어] 깊이 Words를 가지는 입력 버퍼(SRAM) 면적 */
  double areaInputModule(double Words) ;
  /* [한국어] 출력 포트 DFF 모듈 면적 */
  double areaOutputModule(double Outputs);

public:
  /*
   * [한국어]
   * 생성자 — Network 객체와 BookSimConfig를 받아 공정 파라미터를 초기화
   * @n      : 전력 계산 대상 Network 포인터
   * @config : BookSimConfig (또는 PowerConfig 포함 Configuration)
   */
  Power_Module(Network * net, const Configuration &config);
  /* [한국어] 소멸자 — 현재는 별도 해제 작업 없음 */
  ~Power_Module();

  /*
   * [한국어]
   * run() — Network 전체를 순회하며 채널/버퍼/스위치/출력 포트의 전력과 면적을 합산하고
   * 표준 출력(또는 power_output_file)에 보고서를 출력한다.
   */
  void run();


};
#endif
