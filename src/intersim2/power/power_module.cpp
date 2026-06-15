// $Id: power_module.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim2 NoC 전력/면적 추정 모듈 구현 (power_module.cpp)
 *
 * === 파일의 역할 ===
 * Power_Module 클래스의 생성자와 전력/면적 계산 함수들을 구현한다.
 * 시뮬레이션이 끝난 후 Network에 연결된 모든 FlitChannel(주입/배출/중간 채널)과
 * 모든 IQRouter의 BufferMonitor/SwitchMonitor를 순회하면서
 * 채널 전력, 입력 버퍼 전력, 크로스바 스위치 전력, 출력 포트 전력 및 면적을 추정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   intersim2/power/power_module.cpp
 *     → BookSim main() / interconnect_interface.cpp: sim_power 옵션 활성화 시
 *       Power_Module::run() 호출
 *     → networks/network.cpp: Network::GetInject/GetEject/GetChannels/GetRouters
 *     → routers/iq_router.cpp: IQRouter::GetBufferMonitor/GetSwitchMonitor
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - power_module.hpp: 클래스 선언
 *   - booksim_config.hpp: PowerConfig 공정 파라미터 클래스
 *   - buffer_monitor.hpp / switch_monitor.hpp: 활동량 카운터
 *   - iq_router.hpp: IQRouter* dynamic_cast 및 모니터 접근
 * 피의존:
 *   - main.cpp / interconnect_interface.cpp: 시뮬레이션 종료 후 run() 호출
 * 설정 연동 (gpgpusim.config / BookSim config):
 *   - "sim_power"        : 1일 때 전력 모델이 실행됨
 *   - "tech_file"        : 공정 파라미터 파일 (PowerConfig 파싱)
 *   - "power_output_file": 결과 파일 이름
 *   - "classes"          : 트래픽 클래스 수
 *   - "channel_width"    : 채널 폭 (bit), 모든 전력/면적 공식의 기본값
 *   - "channel_sweep"    : 채널 폭 스윕 간격 (값만 저장, 추가 루프는 미구현)
 *   - "num_vcs"          : VC 수 → 버퍼 깊이 계산
 *   - "vc_buf_size"      : VC당 버퍼 크기 → 버퍼 깊이 계산
 *
 * === 주요 함수/구조체 요약 ===
 * Power_Module()        — 공정 파라미터(tech_file) 및 NoC 설정 로드, 상수 초기화
 * calcChannel()         — FlitChannel 하나의 동적/정적 채널 전력 및 면적 계산
 * wireOptimize()        — 배선 길이별 반복기(repeater) 최적화 (K/M/N 탐색)
 * powerRepeatedWire*()  — 반복 배선의 동적/누설/클록/DFF 전력 공식
 * calcBuffer()          — BufferMonitor 기반 입력 버퍼 전력/면적
 * powerMemory*()        — SRAM 비트 단위 읽기/쓰기/누설 전력 및 워드라인 전력
 * calcSwitch()          — SwitchMonitor 기반 크로스바 및 출력 포트 전력/면적
 * powerCrossbar*()      — 크로스바 데이터/제어/누설 전력
 * powerOutputCtrl()     — 출력 포트 제어 신호 전력
 * area*()               — 채널/크로스바/입력/출력 모듈 면적 공식
 * run()                 — Network 전체 순회 및 최종 전력/면적 보고서 출력
 */

#include "power_module.hpp"   // [한국어] Power_Module 클래스 선언
#include "booksim_config.hpp" // [한국어] PowerConfig 공정 파라미터 클래스
#include "buffer_monitor.hpp" // [한국어] 입력 버퍼 활동량
#include "switch_monitor.hpp" // [한국어] 크로스바 활동량
#include "iq_router.hpp"      // [한국어] IQRouter* dynamic_cast 및 모니터 접근자

/*
 * [한국어]
 * 생성자 — BookSimConfig에서 NoC 전력 모델 관련 설정을 읽고,
 * PowerConfig(tech_file)에서 공정 파라미터를 읽어 각종 전력/지연 상수를 초기화한다.
 *
 * @n      : 전력 계산 대상 Network 포인터
 * @config : BookSimConfig (또는 Configuration)
 */
Power_Module::Power_Module(Network * n , const Configuration &config)
  : Module( 0, "power_module" ){ // [한국어] Module 기반 클래스 초기화: ID=0, 이름="power_module"

  
  string pfile = config.GetStr("tech_file"); // [한국어] 공정 파라미터 파일 경로 읽기
  PowerConfig pconfig;                       // [한국어] 공정 파라미터를 담을 PowerConfig 객체 생성
  pconfig.ParseFile(pfile);                  // [한국어] tech_file 파싱하여 공정 파라미터 로드

  net = n;                                          // [한국어] 대상 Network 포인터 저장
  output_file_name = config.GetStr("power_output_file"); // [한국어] 결과 파일 이름
  classes = config.GetInt("classes");               // [한국어] 트래픽 클래스 수
  channel_width = (double)config.GetInt("channel_width"); // [한국어] 채널 폭 (bit)
  channel_sweep = (double)config.GetInt("channel_sweep"); // [한국어] 채널 폭 스윕 간격

  numVC = (double)config.GetInt("num_vcs");       // [한국어] 가상 채널 수
  depthVC  = (double)config.GetInt("vc_buf_size"); // [한국어] VC당 버퍼 깊이 (flit 수)

  //////////////////////////////////Constants/////////////////////////////
  //wire length in (mm)
  wire_length = pconfig.GetFloat("wire_length"); // [한국어] 기준 배선 길이 [mm]
  //////////Metal Parameters////////////
  // Wire left/right coupling capacitance [ F/mm ]
  Cw_cpl = pconfig.GetFloat("Cw_cpl");  // [한국어] 좌우 커플링 커패시턴스 [F/mm]
  // Wire up/down groudn capacitance      [ F/mm ]
  Cw_gnd = pconfig.GetFloat("Cw_gnd");  // [한국어] 상하 접지 커패시턴스 [F/mm]
  Cw = 2.0 * Cw_cpl + 2.0 * Cw_gnd ;     // [한국어] 총 배선 커패시턴스: 양쪽 커플링 + 양쪽 접지
  Rw = pconfig.GetFloat("Rw");          // [한국어] 배선 단위 저항 [Ohm/mm]
  // metal pitch [mm]
  MetalPitch = pconfig.GetFloat("MetalPitch"); // [한국어] 금속 피치 [mm]
  
  //////////Device Parameters////////////
  
  LAMBDA =  pconfig.GetFloat("LAMBDA")  ;       // [um/LAMBDA] [한국어] 공정 스케일링 팩터
  Cd     =  pconfig.GetFloat("Cd");           // [F/um] (for Delay) [한국어] 드레인 커패시턴스(지연)
  Cg     =  pconfig.GetFloat("Cg");           // [F/um] (for Delay) [한국어] 게이트 커패시턴스(지연)
  Cgdl   =  pconfig.GetFloat("Cgdl");           // [F/um] (for Delay) [한국어] 게이트-드레인 커플링(지연)
  
  Cd_pwr =  pconfig.GetFloat("Cd_pwr") ;           // [F/um] (for Power) [한국어] 드레인 커패시턴스(전력)
  Cg_pwr =  pconfig.GetFloat("Cg_pwr") ;           // [F/um] (for Power) [한국어] 게이트 커패시턴스(전력)
				       
  IoffN  = pconfig.GetFloat("IoffN");            // [A/um] [한국어] NMOS 오프 누설 전류
  IoffP  = pconfig.GetFloat("IoffP");            // [A/um] [한국어] PMOS 오프 누설 전류
  // Leakage from bitlines, two-port cell  [A]
  IoffSRAM = pconfig.GetFloat("IoffSRAM");  // [한국어] 2포트 SRAM 비트라인 누설 전류
  // [Ohm] ( D1=1um Inverter)
  R        = pconfig.GetFloat("R");                  // [한국어] 1um 인버터 등가 저항
  // [F]   ( D1=1um Inverter - for Power )
  Ci_delay = (1.0 + 2.0) * ( Cg + Cgdl );   // [한국어] 지연용 입력 커패시턴스 [F]
  // [F]   ( D1=1um Inverter - for Power )
  Co_delay = (1.0 + 2.0) * Cd ;              // [한국어] 지연용 출력 커패시턴스 [F]


  Ci = (1.0 + 2.0) * Cg_pwr ; // [한국어] 전력용 입력 커패시턴스 [F]
  Co = (1.0 + 2.0) * Cd_pwr ; // [한국어] 전력용 출력 커패시턴스 [F]

  Vdd    = pconfig.GetFloat("Vdd");                                  // [한국어] 공급 전압 [V]
  FO4    = R * ( 3.0 * Cd + 12 * Cg + 12 * Cgdl);		     // [한국어] 4배 부하 인버터 지연 [s]
  tCLK   = 20 * FO4;                                                 // [한국어] 클록 주기 [s]
  fCLK   = 1.0 / tCLK;                                               // [한국어] 클록 주파수 [Hz]

  H_INVD2=(double)pconfig.GetInt("H_INVD2");                         // [한국어] 인버터 셀 높이
  W_INVD2=(double)pconfig.GetInt("W_INVD2") ;                        // [한국어] 인버터 셀 폭
  H_DFQD1=(double)pconfig.GetInt("H_DFQD1");                         // [한국어] DFF 셀 높이
  W_DFQD1= (double)pconfig.GetInt("W_DFQD1");                        // [한국어] DFF 셀 폭
  H_ND2D1= (double)pconfig.GetInt("H_ND2D1");                        // [한국어] 2입 NAND 셀 높이
  W_ND2D1=(double)pconfig.GetInt("W_ND2D1");                         // [한국어] 2입 NAND 셀 폭
  H_SRAM=(double)pconfig.GetInt("H_SRAM");                           // [한국어] SRAM 셀 높이
  W_SRAM=(double)pconfig.GetInt("W_SRAM");                           // [한국어] SRAM 셀 폭

  ChannelPitch = 2.0 * MetalPitch ;  // [한국어] 채널 피치 = 2 × 금속 피치
  CrossbarPitch = 2.0 * MetalPitch ; // [한국어] 크로스바 피치 = 2 × 금속 피치
}

/* [한국어] 소멸자 — 현재는 동적 할당된 자원이 없어 빈 본문 */
Power_Module::~Power_Module(){


}


//////////////////////////////////////////////
//Channels
//////////////////////////////////////////////

/*
 * [한국어]
 * calcChannel() — 하나의 FlitChannel에 대한 채널 전력과 면적을 계산
 *
 * 채널 지연(GetLatency)을 기준 배선 길이(wire_length)와 곱해 실제 물리 길이를 구하고,
 * wireOptimize()로 반복기 구조(K/M/N)를 얻는다.
 * FlitChannel::GetActivity()로 클래스별 활동량을 읽어 activity factor를 계산한 뒤
 * 동적 배선 전력, 클록 분배 전력, DFF 전력, 누설 전력을 누적한다.
 */
void Power_Module::calcChannel(const FlitChannel* f){
  double channelLength = f->GetLatency()* wire_length; // [한국어] 채널 지연 × 단위 길이 = 물리 길이 [mm]
  wire const this_wire = wireOptimize(channelLength);  // [한국어] 길이에 대한 최적 반복기 구조 획득
  double const & K = this_wire.K; // [한국어] 드라이버 크기 배율
  double const & N = this_wire.N; // [한국어] 세그먼트 수
  double const & M = this_wire.M; // [한국어] 반복기 단계 수
  //area
  channelArea += areaChannel(K,N,M); // [한국어] 채널 면적 누적

  //activity factor;
  const vector<int> temp = f->GetActivity(); // [한국어] 클래스별 플릿 활동량(이벤트 수) 획득
  vector<double> a(classes);                 // [한국어] 클래스별 activity factor 저장
  for(int i = 0; i< classes; i++){

    a[i] = ((double)temp[i])/totalTime;      // [한국어] 이벤트 수를 총 사이클로 나눠 활동률 산출
  }

  //power calculation
  double const bitPower = powerRepeatedWire(channelLength, K,M,N); // [한국어] 1bit 반복 배선 동적 전력

  channelClkPower += powerWireClk(M,channel_width); // [한국어] 채널 클록 분배 전력 누적
  for(int i = 0; i< classes; i++){
    channelWirePower += bitPower * a[i]*channel_width;          // [한국어] 활동률 × 채널 폭만큼 동적 전력 누적
    channelDFFPower += powerWireDFF(M, channel_width, a[i]);    // [한국어] 채널 DFF(리타이밍) 동적 전력 누적
  }
  channelLeakPower+= powerRepeatedWireLeak(K,M,N)*channel_width; // [한국어] 채널 누설 전력 누적
}

/*
 * [한국어]
 * wireOptimize() — 주어진 배선 길이 L에 대해 반복기 최적화를 수행하고 결과를 캐싱
 *
 * K(드라이버 크기), M(단계 수), N(세그먼트 수)를 3중 루프로 탐색하여
 * 전력 × M^4 지표(metric)가 최소가 되는 조합을 선택한다.
 * 제약 조건: N*Tw < 0.8 * tCLK (한 클록 주기의 80% 이내에 신호가 도달해야 함)
 * 동일한 L에 대한 재호출은 wire_map에서 캐싱된 결과를 반환한다.
 */
wire const & Power_Module::wireOptimize(double L){
  map<double, wire>::iterator iter = wire_map.find(L); // [한국어] L에 대한 기존 최적화 결과 탐색
  if(iter == wire_map.end()){
    
    double W = 64;                // [한국어] 탐색 시 사용하는 임의의 기준 폭
    double bestMetric =  100000000 ; // [한국어] 최소 metric 초기값 (충분히 큰 수)
    double bestK = -1;            // [한국어] 최적 K
    double bestM = -1;            // [한국어] 최적 M
    double bestN = -1;            // [한국어] 최적 N
    for (double K = 1.0 ; K < 10 ; K+=0.1 ) {      // [한국어] 드라이버 크기 배율 탐색
      for (double N = 1.0 ; N < 40 ; N += 1.0 ) {  // [한국어] 세그먼트 수 탐색
	for (double M = 1.0 ; M < 40.0 ; M +=1.0 ) { // [한국어] 반복기 단계 수 탐색
	  double l = 1.0 * L/( N * M) ;              // [한국어] 각 세그먼트의 길이
	  
	  double k0 = R * (Co_delay + Ci_delay) ;    // [한국어] 고정 지연 성분
	  double k1 = R/K * Cw + K * Rw * Ci_delay ; // [한국어] 선형 지연 성분
	  double k2 = 0.5 * Rw * Cw ;                // [한국어] 2차 지연 성분
	  double Tw = k0 + (k1 * l) + k2 * (l * l) ; // [한국어] 한 세그먼트 지연
	  double alpha = 0.2 ;                       // [한국어] 활동률 가정값
	  double power = alpha * W * powerRepeatedWire( L, K, M, N) + powerWireDFF( M, W, alpha ) ; // [한국어] 예상 전력
	  double metric = M * M * M * M * power ;    // [한국어] 최소화할 metric (M^4 × power)
	  if ( (N*Tw) < (0.8 * tCLK) ) {             // [한국어] 타이밍 제약: 전체 지연이 0.8*tCLK 이내
	    if ( metric < bestMetric ) {             // [한국어] 더 좋은 metric이면 최적값 갱신
	      bestMetric = metric ;
	      bestK = K ;
	      bestM = M ;
	      bestN = N ;
	    }
	  }
	}
      }
    }
    cout<<"L = "<<L<<" K = "<<bestK<<" M = "<<bestM<<" N = "<<bestN<<endl; // [한국어] 최적 파라미터 출력
    
    wire const temp = {L, bestK, bestM, bestN};            // [한국어] 최적 결과를 wire 구조체로 구성
    iter = wire_map.insert(make_pair(L, temp)).first;      // [한국어] wire_map에 (L, wire) 쌍 삽입
  }
  return iter->second; // [한국어] 캐싱된(또는 새로 계산된) wire 결과 반환
}

/*
 * [한국어]
 * powerRepeatedWire() — 길이 L의 반복 배선(repeated wire) 동적 전력 계산
 *
 * M×N개의 세그먼트로 나누어진 배선을 모델링하며,
 * 각 반복기 단계의 커패시턴스 Ca를 이용해 0.5*Ca*Vdd^2*fCLK 형태로 동적 전력을 계산한다.
 */
double Power_Module::powerRepeatedWire(double L, double K, double M, double N){
  
  double segments = 1.0 * M * N ;                    // [한국어] 전체 세그먼트 수
  double Ca = K * (Ci + Co) + Cw * (L/segments) ;    // [한국어] 한 단계의 등가 커패시턴스
  double Pa = 0.5 * Ca * Vdd * Vdd * fCLK;           // [한국어] 한 단계의 동적 전력
  return Pa * M * N  ;                               // [한국어] 전체 단계 수만큼 곱해 반환

}

/*
 * [한국어]
 * powerRepeatedWireLeak() — 반복 배선의 누설 전력 계산
 *
 * NMOS/PMOS 오프 상태 누설 전류를 이용해 한 단계의 누설 전력을 구하고,
 * 전체 M×N 단계를 곱한다.
 */
double Power_Module::powerRepeatedWireLeak (double K, double M, double N){
  double Pl = K * 0.5 * ( IoffN + 2.0 * IoffP ) * Vdd  ; // [한국어] 한 단계 누설 전력
  return Pl * M * N ;                                      // [한국어] 전체 단계 수만큼 곱함

}

/*
 * [한국어]
 * powerWireClk() — 채널 클록 분배 네트워크의 동적 전력 계산
 *
 * M개 반복기 뱅크를 따라 실행되는 클럭 와이어의 커패시턴스를 추정하고,
 * 0.5*C*Vdd^2*fCLK 형태로 전력을 계산한다.
 */
double Power_Module:: powerWireClk (double M, double W){
  // number of clock wires running down one repeater bank
  double columns = H_DFQD1 * MetalPitch /  ChannelPitch ; // [한국어] 한 반복기 뱅크당 클럭 와이어 수

  // length of clock wire
  double clockLength = W * ChannelPitch ;                 // [한국어] 클럭 와이어 길이
  double Cclk = (1 + 5.0/16.0 * (1+Co_delay/Ci_delay)) * (clockLength * Cw * columns +W * Ci_delay); // [한국어] 클럭 커패시턴스

  return M * Cclk * (Vdd * Vdd) * fCLK ;                  // [한국어] M개 뱅크에 대한 동적 전력

}

/*
 * [한국어]
 * powerWireDFF() — 채널 리타이밍 DFF의 동적 전력 계산
 *
 * M개 뱅크, W bit 폭, alpha 활동률을 가정하여 DFF 입력/클록 커패시턴스를 계산한다.
 */
double Power_Module::powerWireDFF(double M, double W, double alpha){
  double Cdin = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Co )  ;  // [한국어] DFF 데이터 입력 커패시턴스
  double Cclk = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Cg_pwr) ; // [한국어] DFF 클록 입력 커패시턴스
  double Cint = (alpha * 0.5) * Cdin + alpha * Cclk ;                 // [한국어] 활동률 가중 등가 커패시턴스
  
  return Cint * M * W * (Vdd*Vdd) * fCLK ;                            // [한국어] 동적 전력
}


///////////////////////////////////////////////////////////////
//Memory
//////////////////////////////////////////////////////////////
/*
 * [한국어]
 * calcBuffer() — BufferMonitor가 집계한 read/write 카운터를 바탕으로
 * 입력 버퍼 SRAM의 동적 읽기/쓰기 전력, 누설 전력, 면적을 계산
 *
 * numVC × vc_buf_size로 버퍼 깊이(depth)를 구하고,
 * 포트별/클줄별 activity factor를 구해 read/write 전력을 누적한다.
 */
void Power_Module::calcBuffer(const BufferMonitor *bm){
  double depth = numVC * depthVC  ;                         // [한국어] 입력 버퍼 총 깊이(flit 수)
  double Pleak = powerMemoryBitLeak( depth ) * channel_width ; // [한국어] 채널 폭만큼의 비트 누설 전력
  //area

  const vector<int> reads = bm->GetReads();                  // [한국어] 포트×클줄별 읽기 카운터
  const vector<int> writes = bm->GetWrites();                // [한국어] 포트×클줄별 쓰기 카운터
  for(int i = 0; i<bm->NumInputs(); i++){                    // [한국어] 모든 입력 포트 순회
    inputArea += areaInputModule( depth );                   // [한국어] 포트당 입력 버퍼 면적 누적
    inputLeakagePower += Pleak ;                             // [한국어] 포트당 누설 전력 누적
    for(int j = 0; j< classes; j++){                         // [한국어] 모든 트래픽 클래스 순회
      double ar = ((double)reads[i* classes+j])/totalTime;   // [한국어] 읽기 활동률
      double aw = ((double)writes[i* classes+j])/totalTime;  // [한국어] 쓰기 활동률
      if(ar>1 ||aw >1){                                      // [한국어] 활동률은 1을 초과할 수 없음 (오류 검사)
	cout<<"activity factor is greater than one, soemthing is stomping memory\n"; exit(-1);
      }
      double Pwl =  powerWordLine( channel_width, depth) ;   // [한국어] 워드라인 구동 전력
      double Prd = powerMemoryBitRead( depth ) * channel_width ; // [한국어] 읽기 동적 전력
      double Pwr = powerMemoryBitWrite( depth ) * channel_width ; // [한국어] 쓰기 동적 전력
      inputReadPower    += ar * ( Pwl + Prd ) ;              // [한국어] 읽기 활동률 × (워드라인+읽기) 전력
      inputWritePower   += aw * ( Pwl + Pwr ) ;              // [한국어] 쓰기 활동률 × (워드라인+쓰기) 전력
    }
  }
}


/*
 * [한국어]
 * powerWordLine() — SRAM 워드라인(wordline) 및 주변 회로의 동적 전력 계산
 *
 * 셀 커패시턴스, 프리디코딩/디코딩 회로, 프리차지/쓰기 인에이블 회로의 커패시턴스를 합산하여
 * 0.5*C*Vdd^2*fCLK 형태로 전력을 계산한다.
 */
double Power_Module::powerWordLine(double memoryWidth, double memoryDepth){
  // wordline capacitance
  double Ccell = 2 * ( 4.0 * LAMBDA ) * Cg_pwr +  6 * MetalPitch * Cw ;     // [한국어] 한 SRAM 셀의 워드라인 커패시턴스
  double Cwl = memoryWidth * Ccell ;                                        // [한국어] 전체 워드라인 커패시턴스

  // wordline circuits
  double Warray = 8 * MetalPitch + memoryDepth ;                            // [한국어] 워드라인 회로 폭
  double x = 1.0 + (5.0/16.0) * (1 + Co/Ci)  ;                             // [한국어] 드라이버 커패시턴스 계수
  double Cpredecode = x * (Cw * Warray  * Ci) ;                           // [한국어] 프리디코더 커패시턴스
  double Cdecode    = x * Cwl ;                                            // [한국어] 디코더 커패시턴스

  // bitline circuits
  double Harray =  6 * memoryWidth * MetalPitch ;                          // [한국어] 비트라인 회로 높이
  double y = (1 + 0.25) * (1 + Co/Ci) ;                                    // [한국어] 비트라인 회로 계수
  double Cprecharge = y * ( Cw * Harray + 3 * channel_width * Ci ) ;       // [한국어] 프리차지 회로 커패시턴스
  double Cwren      = y * ( Cw * Harray + 2 * channel_width * Ci ) ;       // [한국어] 쓰기 인에이블 회로 커패시턴스

  double Cbd = Cprecharge + Cwren ;                                        // [한국어] 비트라인 측 총 커패시턴스
  double Cwd = 2 * Cpredecode + Cdecode ;                                  // [한국어] 워드라인 측 총 커패시턴스

  return ( Cbd + Cwd ) * Vdd * Vdd * fCLK ;                                // [한국어] 워드라인 동적 전력
  
}

/*
 * [한국어]
 * powerMemoryBitRead() — SRAM 비트 한 개를 읽을 때의 동적 전력
 *
 * 비트라인 커패시턴스(Cbl)와 전압 스윙(Vswing)을 이용해 계산한다.
 */
double Power_Module::powerMemoryBitRead(double memoryDepth){
  // bitline capacitance
  double Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw ; // [한국어] 한 셀의 비트라인 커패시턴스
  double Cbl    = memoryDepth * Ccell ;                         // [한국어] 전체 비트라인 커패시턴스
  double Vswing = Vdd  ;                                        // [한국어] 전압 스윙 = Vdd
  return ( Cbl ) * ( Vdd * Vswing ) * fCLK ;                    // [한국어] 비트 읽기 동적 전력
}

/*
 * [한국어]
 * powerMemoryBitWrite() — SRAM 비트 한 개를 쓸 때의 동적 전력
 *
 * 비트라인 충방전과 낶은 인버터 교차 결합 셀의 낶은 커패시턴스(Ccc)를 모두 고려한다.
 */
double Power_Module:: powerMemoryBitWrite(double memoryDepth){
  // bitline capacitance
  double Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw ; // [한국어] 한 셀의 비트라인 커패시턴스
  double Cbl    = memoryDepth * Ccell ;                         // [한국어] 전체 비트라인 커패시턴스

  // internal capacitance
  double Ccc    = 2 * (Co + Ci) ;                               // [한국어] 셀 낶은 노드 커패시턴스

  return (0.5 * Ccc * (Vdd*Vdd)) + ( Cbl ) * ( Vdd * Vdd ) * fCLK ; // [한국어] 쓰기 동적 전력
}

/*
 * [한국어]
 * powerMemoryBitLeak() — SRAM 비트 한 개의 누설 전력
 *
 * 메모리 깊이(memoryDepth)와 SRAM 비트라인 누설 전류(IoffSRAM)를 곱한다.
 */
double Power_Module::powerMemoryBitLeak(double memoryDepth ){
  
  return memoryDepth * IoffSRAM * Vdd ; // [한국어] 비트 누설 전력
}

///////////////////////////////////////////////////////////////
//switch
//////////////////////////////////////////////////////////////

/*
 * [한국어]
 * calcSwitch() — SwitchMonitor가 집계한 스위칭 이벤트를 바탕으로
 * 크로스바 및 출력 포트의 동적/누설 전력과 면적을 계산
 *
 * 출력 포트(i)마다 모든 입력 포트(j)와 클래스(k)를 순회하며
 * activity factor(a)를 구하고, powerCrossbar() 등으로 전력을 누적한다.
 */
void Power_Module::calcSwitch(const SwitchMonitor* sm){

  switchArea += areaCrossbar(sm->NumInputs(), sm->NumOutputs());            // [한국어] 크로스바 면적 누적
  outputArea += areaOutputModule(sm->NumOutputs());                         // [한국어] 출력 포트 면적 누적
  switchPowerLeak += powerCrossbarLeak(channel_width, sm->NumInputs(), sm->NumOutputs()); // [한국어] 크로스바 누설 전력 누적

  const vector<int> activity = sm->GetActivity();                           // [한국어] 입출력×클줄별 스위칭 카운터
  vector<double> type_activity(classes);                                    // [한국어] 출력 포트별 클래스 활동률 합계

  for(int i = 0; i<sm->NumOutputs(); i++){                                  // [한국어] 모든 출력 포트 순회
    for(int k = 0; k<classes; k++){
      type_activity[k] = 0;                                                 // [한국어] 현재 출력 포트의 클래스 활동률 누적값 초기화
    }
    for(int j = 0; j<sm->NumInputs(); j++){                                 // [한국어] 모든 입력 포트 순회
      for(int k  = 0; k<classes; k++){                                      // [한국어] 모든 트래픽 클래스 순회
	double a = activity[k+classes*(i+sm->NumOutputs()*j)];              // [한국어] (j→i, k) 스위칭 이벤트 수
	a = a/totalTime;                                                    // [한국어] 활동률로 변환
	if(a>1){                                                            // [한국어] 활동률은 1을 초과할 수 없음
	  cout<<"Switcht activity factor is greater than 1!!!\n";exit(-1);
	}
	double Px = powerCrossbar(channel_width, sm->NumInputs(),sm->NumOutputs(),j,i); // [한국어] j→i 경로 크로스바 전력
	switchPower += a*channel_width*Px;                                  // [한국어] 동적 크로스바 전력 누적
	switchPowerCtrl += a *powerCrossbarCtrl(channel_width,  sm->NumInputs(),sm->NumOutputs()); // [한국어] 제어 신호 전력 누적
	type_activity[k]+=a;                                                // [한국어] 출력 포트 i의 클래스 k 활동률 누적
      }
    }
    outputPowerClk += powerWireClk( 1, channel_width ) ;                     // [한국어] 출력 포트 클록 전력 누적
    for(int k = 0; k<classes; k++){
      outputPower += type_activity[k] * powerWireDFF( 1, channel_width, 1.0 ) ; // [한국어] 출력 포트 DFF 동적 전력 누적
      outputCtrlPower += type_activity[k] * powerOutputCtrl(channel_width ) ;  // [한국어] 출력 포트 제어 전력 누적
    }
  }

}

/*
 * [한국어]
 * powerCrossbar() — 특정 입력(from) → 출력(to) 경로의 크로스바 데이터 경로 동적 전력
 *
 * 크로스바의 수평/수직 와이어 커패시턴스, 크로스포인트, 입력/출력 드라이버 커패시턴스를 합산하고
 * from/to 위치에 따라 불필요한 부분을 제거하여 0.5*C*Vdd^2*fCLK로 계산한다.
 */
double Power_Module::powerCrossbar(double width, double inputs, double outputs, double from, double to){
  // datapath traversal power
  double Wxbar = width * outputs * CrossbarPitch ; // [한국어] 크로스바 수평 길이
  double Hxbar = width * inputs  * CrossbarPitch ; // [한국어] 크로스바 수직 길이

  // wires
  double CwIn  = Wxbar * Cw ; // [한국어] 입력 측(수평) 와이어 총 커패시턴스
  double CwOut = Hxbar * Cw ; // [한국어] 출력 측(수직) 와이어 총 커패시턴스

  // cross-points
  double Cxi = (1.0/16.0) * CwOut ;                // [한국어] 출력 측 크로스포인트 커패시턴스
  double Cxo = 4.0 * Cxi * (Co_delay/Ci_delay) ;   // [한국어] 입력 측 크로스포인트 커패시턴스

  // drivers
  double Cti = (1.0/16.0) * CwIn ;                 // [한국어] 입력 트라이스테이트 드라이버 커패시턴스
  double Cto = 4.0 * Cti * (Co_delay/Ci_delay) ;   // [한국어] 출력 트라이스테이트 드라이버 커패시턴스

  double CinputDriver = 5.0/16.0 * (1 + Co_delay/Ci_delay) * (0.5 * Cw * Wxbar + Cti) ; // [한국어] 입력 드라이버 커패시턴스

  // total switched capacitance
  
  //this maybe missing +Cto
  double Cin  = CinputDriver + CwIn + Cti + (outputs * Cxi) ; // [한국어] 입력 측 총 커패시턴스
  if ( to < outputs/2 ) {                                      // [한국어] 중앙 기준 반대편 제거
    Cin -= ( 0.5 * CwIn + outputs/2 * Cxi) ;
  }
  //this maybe missing +cti
  double Cout = CwOut + Cto + (inputs * Cxo) ;                // [한국어] 출력 측 총 커패시턴스
  if ( from < inputs/2) {                                      // [한국어] 중앙 기준 반대편 제거
    Cout -= ( 0.5 * CwOut + (inputs/2 * Cxo)) ;
  }
  return 0.5 * (Cin + Cout) * (Vdd * Vdd * fCLK) ;            // [한국어] 크로스바 데이터 경로 동적 전력
}


/*
 * [한국어]
 * powerCrossbarCtrl() — 크로스바 제어(셀렉트) 신호의 동적 전력 계산
 *
 * 입력 와이어, 제어 와이어, 드라이버 커패시턴스를 합산하여 제어 신호 전력을 구한다.
 */
double Power_Module::powerCrossbarCtrl(double width, double inputs, double outputs){
 
  // datapath traversal power
  double Wxbar = width * outputs * CrossbarPitch ; // [한국어] 크로스바 수평 길이
  double Hxbar = width * inputs  * CrossbarPitch ; // [한국어] 크로스바 수직 길이

  // wires
  double CwIn  = Wxbar * Cw ; // [한국어] 입력 와이어 총 커패시턴스

  // drivers
  double Cti  = (5.0/16.0) * CwIn ; // [한국어] 제어 드라이버 입력 커패시턴스

  // need some estimate of how many control wires are required
  double Cctrl  = width * Cti + (Wxbar + Hxbar) * Cw  ; // [한국어] 제어 와이어 총 커패시턴스
  double Cdrive = (5.0/16.0) * (1 + Co_delay/Ci_delay) * Cctrl ; // [한국어] 제어 드라이버 커패시턴스

  return (Cdrive + Cctrl) * (Vdd*Vdd) * fCLK ; // [한국어] 제어 신호 동적 전력
  
}

/*
 * [한국어]
 * powerCrossbarLeak() — 크로스바의 누설 전력 계산
 *
 * 크로스포인트 및 드라이버 트랜지스터의 오프 상태 누설 전류를 이용해
 * 전체 입력×출력 크기에 대한 누설 전력을 추정한다.
 */
double Power_Module::powerCrossbarLeak (double width, double inputs, double outputs){
  // datapath traversal power
    double Wxbar = width * outputs * CrossbarPitch ; // [한국어] 크로스바 수평 길이
    double Hxbar = width * inputs  * CrossbarPitch ; // [한국어] 크로스바 수직 길이

    // wires
    double CwIn  = Wxbar * Cw ; // [한국어] 입력 와이어 총 커패시턴스
    double CwOut = Hxbar * Cw ; // [한국어] 출력 와이어 총 커패시턴스
    // cross-points
    double Cxi = (1.0/16.0) * CwOut ; // [한국어] 출력 측 크로스포인트 커패시턴스
    // drivers
    double Cti  = (1.0/16.0) * CwIn ; // [한국어] 입력 드라이버 커패시턴스

    return 0.5 * (IoffN + 2 * IoffP)*width*(inputs*outputs*Cxi+inputs*Cti+outputs*Cti)/Ci; // [한국어] 누설 전력
}

//////////////////////////////////////////////////////////////////
//output module
//////////////////////////////////////////////////////////////////
/*
 * [한국어]
 * powerOutputCtrl() — 출력 포트 제어(Enable) 신호의 동적 전력 계산
 *
 * 출력 모듈 폭과 인에이블 커패시턴스를 이용해 제어 신호 전력을 구한다.
 */
double Power_Module:: powerOutputCtrl(double width) {

    double Woutmod = channel_width * ChannelPitch ; // [한국어] 출력 모듈의 수평 길이
    double Cen     = Ci ;                           // [한국어] 인에이블 커패시턴스

    double Cenable = (1 + 5.0/16.0)*(1.0+Co/Ci)*(Woutmod* Cw + width* Cen) ; // [한국어] 제어 신호 총 커패시턴스

    return Cenable * (Vdd*Vdd) * fCLK ; // [한국어] 출력 제어 동적 전력
    
}

//////////////////////////////////////////////////////////////////
//area
//////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * areaChannel() — 반복 배선 기반 채널의 면적 계산
 *
 * DFF 면적과 인버터 뱅크 면적을 합산 후 채널 폭(channel_width)과 금속 피치를 곱한다.
 */
double Power_Module:: areaChannel (double K, double N, double M){

    double Adff = M * W_DFQD1 * H_DFQD1 ;              // [한국어] 채널 DFF 면적
    double Ainv = M * N * ( W_INVD2 + 3 * K) * H_INVD2 ; // [한국어] 반복기 인버터 뱅크 면적

    return channel_width * (Adff + Ainv) * MetalPitch * MetalPitch ; // [한국어] 채널 총 면적
}

/*
 * [한국어]
 * areaCrossbar() — 입력×출력 크기의 크로스바 면적 계산
 */
double Power_Module:: areaCrossbar(double Inputs, double Outputs) {
    return (Inputs * channel_width * CrossbarPitch) * (Outputs * channel_width * CrossbarPitch) ; // [한국어] 크로스바 면적
}

/*
 * [한국어]
 * areaInputModule() — 깊이 Words를 가지는 입력 버퍼(SRAM) 면적 계산
 */
double Power_Module:: areaInputModule(double Words) {
    double Asram =  ( channel_width * H_SRAM ) * (Words * W_SRAM) ; // [한국어] SRAM 어레이 면적
    return Asram * (MetalPitch * MetalPitch) ;                      // [한국어] 실제 면적
}

/*
 * [한국어]
 * areaOutputModule() — 출력 포트 DFF 모듈 면적 계산
 */
double Power_Module:: areaOutputModule(double Outputs) {
    double Adff = Outputs * W_DFQD1 * H_DFQD1 ;                  // [한국어] 출력 DFF 면적
    return channel_width * Adff * MetalPitch * MetalPitch ;      // [한국어] 출력 모듈 총 면적
}

/*
 * [한국어]
 * run() — Network 전체를 순회하여 채널/버퍼/스위치/출력 포트의 전력과 면적을 합산하고
 * 표준 출력(및 power_output_file)에 OCN Power/Area Summary 보고서를 출력한다.
 *
 * 호출 시점: 시뮬레이션이 완료된 후 (GetSimTime()이 최종 사이클을 반환)
 */
void Power_Module::run(){
  totalTime = GetSimTime(); // [한국어] 시뮬레이션 총 사이클 수 획득
  // [한국어] 보고서 전 누적 변수들을 0으로 초기화
  channelWirePower=0;
  channelClkPower=0;
  channelDFFPower=0;
  channelLeakPower=0;
  inputReadPower=0;
  inputWritePower=0;
  inputLeakagePower=0;
  switchPower=0;
  switchPowerCtrl=0;
  switchPowerLeak=0;
  outputPower=0;
  outputPowerClk=0;
  outputCtrlPower=0;
  channelArea=0;
  switchArea=0;
  inputArea=0;
  outputArea=0;
  maxInputPort = 0;
  maxOutputPort = 0;

  vector<FlitChannel *> inject = net->GetInject();   // [한국어] 주입(injection) 채널 목록
  vector<FlitChannel *> eject = net->GetEject();     // [한국어] 배출(ejection) 채널 목록
  vector<FlitChannel *> chan = net->GetChannels();   // [한국어] 라우터 간 채널 목록
  
  for(int i = 0; i<net->NumNodes(); i++){
    calcChannel(inject[i]); // [한국어] 각 노드의 주입 채널 전력/면적 계산
  }

  for(int i = 0; i<net->NumNodes(); i++){
    calcChannel(eject[i]);  // [한국어] 각 노드의 배출 채널 전력/면적 계산
  }

  for(int i = 0; i<net->NumChannels();i++){
    calcChannel(chan[i]);   // [한국어] 라우터 간 채널 전력/면적 계산
  }

  vector<Router*> routers = net->GetRouters();       // [한국어] 네트워크 내 모든 라우터 목록
  for(size_t i = 0; i < routers.size(); i++){
    IQRouter* temp = dynamic_cast<IQRouter*>(routers[i]); // [한국어] Router*를 IQRouter*로 다운캐스트
    const BufferMonitor * bm = temp->GetBufferMonitor();  // [한국어] 라우터의 입력 버퍼 모니터 획득
    calcBuffer(bm);                                       // [한국어] 입력 버퍼 전력/면적 계산
    const SwitchMonitor * sm = temp->GetSwitchMonitor();  // [한국어] 라우터의 크로스바 모니터 획득
    calcSwitch(sm);                                       // [한국어] 크로스바/출력 전력/면적 계산
  }
  
  // [한국어] 모든 구성 요소의 전력을 합산
  double totalpower =  channelWirePower+channelClkPower+channelDFFPower+channelLeakPower+ inputReadPower+inputWritePower+inputLeakagePower+ switchPower+switchPowerCtrl+switchPowerLeak+outputPower+outputPowerClk+outputCtrlPower;
  // [한국어] 모든 구성 요소의 면적을 합산
  double totalarea =  channelArea+switchArea+inputArea+outputArea;
  // [한국어] OCN 전력 보고서 출력
  cout<< "-----------------------------------------\n" ;
  cout<< "- OCN Power Summary\n" ;
  cout<< "- Completion Time:         "<<totalTime <<"\n" ;
  cout<< "- Flit Widths:            "<<channel_width<<"\n" ;
  cout<< "- Channel Wire Power:      "<<channelWirePower <<"\n" ;
  cout<< "- Channel Clock Power:     "<<channelClkPower <<"\n" ;
  cout<< "- Channel Retiming Power:  "<<channelDFFPower <<"\n" ;
  cout<< "- Channel Leakage Power:   "<<channelLeakPower <<"\n" ;
  
  cout<< "- Input Read Power:        "<<inputReadPower <<"\n" ;
  cout<< "- Input Write Power:       "<<inputWritePower <<"\n" ;
  cout<< "- Input Leakage Power:     "<<inputLeakagePower <<"\n" ;
  
  cout<< "- Switch Power:            "<<switchPower <<"\n" ;
  cout<< "- Switch Control Power:    "<<switchPowerCtrl <<"\n" ;
  cout<< "- Switch Leakage Power:    "<<switchPowerLeak <<"\n" ;
  
  cout<< "- Output DFF Power:        "<<outputPower <<"\n" ;
  cout<< "- Output Clk Power:        "<<outputPowerClk <<"\n" ;
  cout<< "- Output Control Power:    "<<outputCtrlPower <<"\n" ;
  cout<< "- Total Power:             "<<totalpower <<"\n";
  cout<< "-----------------------------------------\n" ;
  cout<< "\n" ;
  // [한국어] OCN 면적 보고서 출력
  cout<< "-----------------------------------------\n" ;
  cout<< "- OCN Area Summary\n" ;
  cout<< "- Channel Area:  "<<channelArea<<"\n" ;
  cout<< "- Switch  Area:  "<<switchArea<<"\n" ;
  cout<< "- Input  Area:   "<<inputArea<<"\n" ;
  cout<< "- Output  Area:  "<<outputArea<<"\n" ;
  cout<< "- Total Area:    "<<totalarea<<endl;
  cout<< "-----------------------------------------\n" ;




}
