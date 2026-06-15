// $Id: booksim_config.cpp 5506 2013-05-07 21:22:23Z qtedq $

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

/*booksim_config.cpp
 *
 *Contains all the configurable parameters in a network
 *
 */

/*
 * [한국어 설명] BookSim NoC 시뮬레이터 설정 파라미터 구현 (booksim_config.cpp)
 *
 * === 파일의 역할 ===
 * BookSimConfig 및 PowerConfig 생성자를 구현한다.
 * 두 생성자는 각각 NoC 시뮬레이션 파라미터와 전력 모델 공정 파라미터의
 * 기본값을 Configuration 내부 맵(_int_map, _float_map, _str_map)에 등록한다.
 * 이 기본값들은 이후 ParseFile()로 .booksim 설정 파일에서 읽은 값으로 선택적으로 덮어쓰인다.
 * 즉, 이 파일은 전체 시스템의 "파라미터 설명서 겸 기본값 저장소" 역할을 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이터 초기화 단계 (싱글스레드).
 * 호출 체인:
 *   GPGPU-Sim icnt_wrapper_init()
 *     → intersim2_create() / BookSim main()
 *         → BookSimConfig::BookSimConfig() ← [이 파일]
 *         → Configuration::ParseFile()
 *         → Network::New() / TrafficManager::New()
 * PowerConfig는 sim_power=1 설정 시 별도로 생성된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: booksim.hpp (using namespace std 및 표준 헤더), booksim_config.hpp (클래스 선언)
 * - 피의존: 사실상 intersim2/ 내 모든 모듈이 BookSimConfig&를 수신해 GetInt()/GetStr()로 조회
 * - GPGPU-Sim 연결: intersim_simparams는 BookSimConfig 인스턴스를 통해 icnt_wrapper에 전달됨
 * - 데이터 흐름: 하드코딩 기본값 → BookSimConfig 맵 → ParseFile() 덮어쓰기 → 각 모듈 조회
 *
 * === 주요 함수/구조체 요약 ===
 * BookSimConfig::BookSimConfig() — 약 70여 개의 파라미터를 6개 범주로 나누어 기본값 등록:
 *   (1) 네트워크 옵션: channel_file, subnets, topology, k, n, c, x, y, xr, yr
 *   (2) 라우터 옵션: router 종류, VC 수, 버퍼 크기, 투기적 VC 할당, 파이프라인 딜레이
 *   (3) 트래픽 옵션: traffic 패턴, injection_rate, packet_size, read/write VC 범위
 *   (4) 시뮬레이션 파라미터: sim_type, warmup, sample_period, seed, latency_thres
 *   (5) 출력/디버그: stats_out, watch_file, viewer_trace, TRACK_FLOWS/TRACK_CREDITS
 *   (6) 전력 모델: sim_power, tech_file, channel_width
 * PowerConfig::PowerConfig() — 공정 파라미터 전체를 0으로 초기화
 */

#include "booksim.hpp"         // [한국어] using namespace std 및 표준 C++ 헤더 포함
#include "booksim_config.hpp"  // [한국어] BookSimConfig, PowerConfig 클래스 선언

/*
 * [한국어]
 * BookSimConfig::BookSimConfig - BookSim NoC 시뮬레이터 전체 파라미터 기본값 등록
 *
 * @return: (생성자, 반환값 없음)
 *
 * 이 생성자는 NoC 시뮬레이션에 필요한 모든 파라미터의 기본값을
 * Configuration 내부의 _int_map, _float_map, _str_map에 등록한다.
 * AddStrField()는 string 타입 파라미터를 등록하고,
 * _int_map[key] = val / _float_map[key] = val 은 정수/실수 파라미터를 직접 설정한다.
 * 이후 ParseFile()이 호출되면 설정 파일에 명시된 항목만 덮어쓰이고,
 * 나머지는 이 생성자에서 설정한 기본값이 유지된다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 싱글스레드.
 * 호출 체인:
 *   icnt_wrapper_init() → [BookSimConfig::BookSimConfig()] → ParseFile() → 각 모듈 생성
 */
BookSimConfig::BookSimConfig( )
{
  //========================================================
  // Network options
  //========================================================

  // Channel length listing file
  AddStrField( "channel_file", "" ) ;
  /* [한국어] 채널(링크) 길이를 정의하는 파일 경로.
   * 기본값 ""는 균일 채널 길이를 의미한다.
   * 불규칙 토폴로지에서 링크별 지연/전력 계산에 사용된다. */

  // Physical sub-networks
  _int_map["subnets"] = 1;
  /* [한국어] 물리적 서브네트워크 수.
   * read/write 요청-응답을 별도 서브넷으로 분리할 때 2 이상으로 설정한다.
   * GPGPU-Sim에서는 기본적으로 1개 서브넷을 사용한다. */

  //==== Topology options =======================
  AddStrField( "topology", "torus" );
  /* [한국어] 네트워크 토폴로지 종류 (문자열 식별자).
   * 지원 값: "torus", "mesh", "fly"(butterfly), "fat_tree", "single" 등.
   * GPGPU-Sim에서는 주로 "single" (크로스바) 또는 "mesh"를 사용한다. */

  _int_map["k"] = 8;
  /* [한국어] 네트워크 래딕스(radix) — 각 차원의 노드 수.
   * k-ary n-cube 토폴로지에서 k=8, n=2이면 8×8 2D 메시/토러스를 의미한다.
   * 총 노드 수 = k^n. GPGPU-Sim은 SM 수와 메모리 파티션 수에 따라 실제 k를 결정한다. */

  _int_map["n"] = 2;
  /* [한국어] 네트워크 차원 수 — k-ary n-cube에서의 n.
   * n=1이면 링(ring), n=2이면 2D 메시/토러스 등. */

  _int_map["c"] = 1;
  /* [한국어] 집중(concentration) 계수 — 하나의 라우터 포트에 연결되는 단말 노드 수.
   * c=1이면 라우터와 단말이 1:1 대응. c>1이면 여러 단말이 하나의 라우터 포트를 공유한다. */

  AddStrField( "routing_function", "none" );
  /* [한국어] 라우팅 알고리즘 식별자.
   * 지원 값: "none", "dim_order"(DOR), "valiant", "ugal", "min_adapt" 등.
   * "none"은 기본 라우팅(토폴로지 기본값)을 의미한다. */

  //simulator tries to correctly adjust latency for node/router placement
  _int_map["use_noc_latency"] = 1;
  /* [한국어] NoC 지연 보정 활성화 플래그.
   * 1이면 시뮬레이터가 노드-라우터 배치에 따른 홉(hop) 카운트를 기반으로
   * 지연을 자동 조정한다. 0이면 보정 없이 고정 지연 사용. */

  //used for noc latency calculation for network with concentration
  _int_map["x"] = 8;
  /* [한국어] X축 방향 라우터 수 — 집중(concentration) 네트워크에서 NoC 지연 계산에 사용.
   * c>1일 때 실제 배치 그리드의 가로 크기를 나타낸다. */

  _int_map["y"] = 8;
  /* [한국어] Y축 방향 라우터 수 — x와 함께 2D 배치 그리드 크기를 정의한다. */

  _int_map["xr"] = 1;
  /* [한국어] X 방향에서 하나의 라우터에 연결되는 단말 노드 수.
   * c>1 설정 시에만 의미가 있다. xr*yr = c가 되어야 한다. */

  _int_map["yr"] = 1;
  /* [한국어] Y 방향에서 하나의 라우터에 연결되는 단말 노드 수.
   * xr과 함께 집중 비율을 결정한다. */

  _int_map["link_failures"] = 0;
  /* [한국어] 링크 고장 수 — 레거시 옵션, 현재 사용되지 않음.
   * 0이면 모든 링크가 정상 동작한다. */

  _int_map["fail_seed"] = 0;
  /* [한국어] 링크 고장 시뮬레이션용 랜덤 시드 — 레거시 옵션, 현재 사용되지 않음. */

  //==== Single-node options ===============================

  _int_map["in_ports"]  = 5;
  /* [한국어] 단일 노드(single) 토폴로지에서의 입력 포트 수.
   * "single" 토폴로지(크로스바)를 사용할 때만 의미가 있다.
   * GPGPU-Sim의 SM-메모리 파티션 연결 모델에서는 SM 수에 따라 결정된다. */

  _int_map["out_ports"] = 5;
  /* [한국어] 단일 노드 토폴로지에서의 출력 포트 수.
   * in_ports와 함께 크로스바 크기를 정의한다. */

  //========================================================
  // Router options
  //========================================================

  //==== General options ===================================

  AddStrField( "router", "iq" );
  /* [한국어] 라우터 마이크로아키텍처 종류.
   * "iq" = Input-Queued 라우터 (가장 일반적).
   * "event" = 이벤트 드리븐 라우터.
   * BookSim은 IQ 라우터가 기본이며, GPGPU-Sim도 "iq"를 사용한다. */

  _int_map["output_delay"] = 0;
  /* [한국어] 출력 포트에서 추가되는 사이클 지연.
   * 물리적 링크 전파 지연을 추가로 모델링할 때 사용한다. */

  _int_map["credit_delay"] = 0;
  /* [한국어] 크레딧(credit) 신호 전달에 걸리는 사이클 지연.
   * 크레딧은 다운스트림 버퍼의 가용성을 업스트림에 알리는 흐름 제어 신호이다.
   * 물리적 배선 지연이 있을 때 0보다 큰 값으로 설정한다. */

  _float_map["internal_speedup"] = 1.0;
  /* [한국어] 라우터 내부 스위치 패브릭의 속도 배율.
   * 1.0이면 입력 포트 속도와 동일. 2.0이면 크로스바가 2배 빠르게 동작.
   * 스위치 스피드업이 있으면 출력 버퍼가 필요하다(output_buffer_size 참조). */

  //with switch speedup flits require output buffering
  //full output buffer will cancel switch allocation requests
  //default setting is unlimited
  _int_map["output_buffer_size"] = -1;
  /* [한국어] 출력 버퍼 크기 (플릿 단위).
   * -1이면 무제한. internal_speedup > 1.0일 때 필수적으로 필요하다.
   * 출력 버퍼가 가득 차면 스위치 할당 요청이 취소된다. */

  // enable next-hop-output queueing
  _int_map["noq"] = 0;
  /* [한국어] NOQ(Next-hop Output Queueing) 활성화 플래그.
   * 1이면 헤드-오브-라인 블로킹을 완화하기 위해 다음 홉의 출력 큐를 미리 예약한다.
   * 0이면 비활성화 (표준 IQ 라우터 동작). */

  //==== Input-queued ======================================

  // Control of virtual channel speculation
  _int_map["speculative"] = 0 ;
  /* [한국어] 투기적(speculative) VC 할당 활성화 플래그.
   * 1이면 VC 할당 결과를 기다리지 않고 스위치 할당을 동시에 시도한다.
   * 지연을 줄일 수 있지만 충돌 처리가 복잡해진다. */

  _int_map["spec_check_elig"] = 1 ;
  /* [한국어] 투기적 할당 시 VC 적격성(eligibility) 검사 여부.
   * 1이면 스위치 할당 전에 VC 적격성을 확인하여 불필요한 투기를 줄인다. */

  _int_map["spec_check_cred"] = 1 ;
  /* [한국어] 투기적 할당 시 크레딧 가용성 검사 여부.
   * 1이면 다운스트림 버퍼 크레딧이 있는 경우에만 투기적 할당을 진행한다. */

  _int_map["spec_mask_by_reqs"] = 0 ;
  /* [한국어] 요청 마스킹을 통한 투기적 할당 제어.
   * 1이면 이미 정규 VC 할당 요청이 있는 포트는 투기 대상에서 제외한다. */

  AddStrField("spec_sw_allocator", "prio");
  /* [한국어] 투기적 스위치 할당기 종류.
   * "prio" = 우선순위 기반 할당기. 투기적 요청보다 정규 요청을 우선한다. */

  _int_map["num_vcs"] = 16;
  /* [한국어] 라우터 포트당 가상 채널(VC) 수.
   * VC는 동일 물리 링크에서 서로 다른 논리적 흐름을 분리하여 데드락을 방지한다.
   * read/write 요청과 응답이 각각 다른 VC를 사용하도록 범위를 분할한다.
   * 기본값 16은 read/write 요청-응답 각 4개 구간으로 분할하기에 충분하다. */

  _int_map["vc_buf_size"] = 8;
  /* [한국어] VC당 버퍼 크기 (플릿 단위).
   * 각 가상 채널이 독립적으로 보유할 수 있는 최대 플릿 수.
   * buffer_policy가 "private"일 때 각 VC가 이 크기의 전용 버퍼를 가진다. */

  _int_map["buf_size"] = -1;
  /* [한국어] 공유 버퍼 크기 (플릿 단위). -1이면 비활성화(전용 버퍼 사용).
   * buffer_policy가 "shared"일 때 포트 내 모든 VC가 이 크기를 공유한다. */

  AddStrField("buffer_policy", "private");
  /* [한국어] 버퍼 공유 정책.
   * "private" = 각 VC 전용 버퍼 (vc_buf_size 사용).
   * "shared" = 포트 내 VC들이 공유 버퍼 풀 사용 (buf_size 필요). */

  _int_map["private_bufs"] = -1;
  /* [한국어] 전용 버퍼 구간 수. -1이면 기본값(모든 VC에 단일 전용 버퍼). */

  _int_map["private_buf_size"] = 1;
  /* [한국어] 개별 전용 버퍼 크기 (플릿). AddStrField 버전과 함께 사용 가능. */

  AddStrField("private_buf_size", "");
  /* [한국어] VC 범위별 전용 버퍼 크기 벡터 지정용 문자열 오버라이드.
   * 예: "4 4 8 8"처럼 VC 그룹별로 다른 크기를 지정할 때 사용한다. */

  _int_map["private_buf_start_vc"] = -1;
  /* [한국어] 전용 버퍼 구간의 시작 VC 번호. -1이면 기본(0번부터). */

  AddStrField("private_buf_start_vc", "");
  /* [한국어] 여러 구간의 시작 VC를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["private_buf_end_vc"] = -1;
  /* [한국어] 전용 버퍼 구간의 종료 VC 번호. -1이면 기본(마지막 VC까지). */

  AddStrField("private_buf_end_vc", "");
  /* [한국어] 여러 구간의 종료 VC를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["max_held_slots"] = -1;
  /* [한국어] 한 VC가 스위치 트래버설 단계에서 동시에 보유할 수 있는 최대 슬롯 수.
   * -1이면 무제한. 특정 공정 정책(QoS) 적용 시 제한한다. */

  _int_map["feedback_aging_scale"] = 1;
  /* [한국어] 피드백 기반 VC 우선순위 에이징(aging) 스케일.
   * 오래된 패킷의 우선순위를 높이는 속도를 조절한다. 1이면 기본 속도. */

  _int_map["feedback_offset"] = 0;
  /* [한국어] 피드백 우선순위 오프셋. 기본 우선순위에 더해지는 편향값. */

  _int_map["wait_for_tail_credit"] = 0;
  /* [한국어] 꼬리 플릿(tail flit) 크레딧을 받은 후에야 VC를 재할당할지 여부.
   * 0이면 꼬리 크레딧 없이도 VC 재할당 가능 (더 공격적인 할당).
   * 1이면 안전하지만 보수적인 VC 재사용. */

  _int_map["vc_busy_when_full"] = 0;
  /* [한국어] 크레딧이 없을 때 VC를 "사용 중(busy)"으로 표시할지 여부.
   * 1이면 버퍼가 가득 찬 VC에 새 패킷이 할당되지 않도록 한다. */

  _int_map["vc_prioritize_empty"] = 0;
  /* [한국어] 빈 VC를 비어 있지 않은 VC보다 우선 할당할지 여부.
   * 1이면 레이턴시를 줄이는 데 유리하지만 공정성에 영향을 줄 수 있다. */

  _int_map["vc_priority_donation"] = 0;
  /* [한국어] 높은 우선순위 플릿이 앞에 대기 중인 낮은 우선순위 플릿에 우선순위를 "기증"할지 여부.
   * 1이면 HoL(Head-of-Line) 블로킹으로 인한 우선순위 역전을 완화한다. */

  _int_map["vc_shuffle_requests"] = 0;
  /* [한국어] VC 할당기 요청 순서를 셔플하여 불공정을 방지할지 여부.
   * 1이면 항상 동일한 순서로 처리되는 편향을 제거한다. */

  _int_map["hold_switch_for_packet"] = 0;
  /* [한국어] 패킷 전체가 통과할 때까지 스위치 설정을 유지할지 여부.
   * 1이면 wormhole 라우팅처럼 패킷 단위로 스위치를 점유한다.
   * 0이면 플릿 단위로 스위치를 재할당하는 Virtual Cut-Through(VCT) 방식. */

  _int_map["input_speedup"] = 1;
  /* [한국어] 크로스바 입력 포트 확장 배수.
   * 1보다 크면 입력 포트를 복제하여 스위치 경합을 줄인다. */

  _int_map["output_speedup"] = 1;
  /* [한국어] 크로스바 출력 포트 확장 배수.
   * 1보다 크면 출력 포트를 복제하여 스위치 경합을 줄인다. */

  _int_map["routing_delay"] = 1;
  /* [한국어] 라우팅 계산(RC) 단계에 소요되는 파이프라인 사이클 수. */

  _int_map["vc_alloc_delay"] = 1;
  /* [한국어] VC 할당(VA) 단계에 소요되는 파이프라인 사이클 수. */

  _int_map["sw_alloc_delay"] = 1;
  /* [한국어] 스위치 할당(SA) 단계에 소요되는 파이프라인 사이클 수. */

  _int_map["st_prepare_delay"] = 0;
  /* [한국어] 스위치 트래버설(ST) 준비 단계 지연 사이클. 0이면 즉시. */

  _int_map["st_final_delay"] = 1;
  /* [한국어] 스위치 트래버설 완료 후 최종 링크 전달 지연 사이클. */

  //==== Event-driven =====================================

  _int_map["vct"] = 0;
  /* [한국어] VCT(Virtual Cut-Through) 이벤트 드리븐 라우터 활성화 플래그.
   * 1이면 이벤트 드리븐 VCT 라우터를 사용한다. 0이면 IQ 라우터 사용. */

  //==== Allocators ========================================

  AddStrField( "vc_allocator", "islip" );
  /* [한국어] VC 할당기 알고리즘 종류.
   * "islip" = iSLIP(Iterative Sliding Lookup Iterative Priority) — 공정한 라운드로빈 기반.
   * "prio" = 우선순위 기반. "loa" = Least Outstanding Allocation 등.
   * iSLIP은 BookSim 기본이며 GPGPU-Sim에도 그대로 사용된다. */

  AddStrField( "sw_allocator", "islip" );
  /* [한국어] 스위치 할당기 알고리즘 종류.
   * VC 할당기와 동일한 옵션을 지원한다. iSLIP이 기본값. */

  AddStrField( "arb_type", "round_robin" );
  /* [한국어] 할당기 내부 중재(arbitration) 방식.
   * "round_robin" = 라운드로빈 (기본, 공정성 보장).
   * "matrix" = 행렬 중재기. */

  _int_map["alloc_iters"] = 1;
  /* [한국어] 할당기 반복(iteration) 횟수.
   * iSLIP 같은 반복 알고리즘에서 1회 사이클당 수행하는 반복 횟수.
   * 반복 횟수가 많을수록 처리율이 높아지지만 하드웨어 비용도 증가한다. */

  //==== Traffic ========================================

  _int_map["classes"] = 1;
  /* [한국어] 트래픽 클래스 수 — QoS 차별화를 위한 논리적 트래픽 분류.
   * 1이면 단일 클래스 (GPGPU-Sim 기본). 2이면 예를 들어 요청/응답을 별도 클래스로 처리. */

  AddStrField( "traffic", "uniform" );
  /* [한국어] 합성 트래픽 패턴 종류 (독립 실행 시 사용).
   * "uniform" = 균일 랜덤 (모든 소스-목적지 쌍이 동일 확률).
   * "hotspot", "transpose", "bit_reverse" 등 다양한 패턴 지원.
   * GPGPU-Sim 연동 시에는 실제 GPU 메모리 트래픽이 주입되므로 이 설정은 무시된다. */

  _int_map["class_priority"] = 0;
  /* [한국어] 클래스별 우선순위 기본값. 0이면 최저 우선순위. */

  AddStrField("class_priority", "");
  /* [한국어] 클래스별 우선순위를 벡터로 지정하는 문자열 오버라이드.
   * 예: "0 1"은 클래스 0에 우선순위 0, 클래스 1에 우선순위 1 부여. */

  _int_map["perm_seed"] = 0;
  /* [한국어] 순열(permutation) 트래픽 패턴 생성용 랜덤 시드.
   * "transpose" 등 순열 기반 패턴에서 재현 가능한 패턴을 위해 사용한다. */

  _float_map["injection_rate"] = 0.1;
  /* [한국어] 패킷 주입 속도 (포화 처리율의 비율).
   * 0.1이면 이론적 최대 처리율의 10%로 패킷을 주입한다.
   * 합성 트래픽 시뮬레이션에서만 사용; GPGPU-Sim 연동 시 무시됨. */

  AddStrField("injection_rate", "");
  /* [한국어] 클래스별 주입 속도를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["injection_rate_uses_flits"] = 0;
  /* [한국어] 주입 속도의 단위를 플릿으로 할지 패킷으로 할지 결정.
   * 0이면 패킷 단위, 1이면 플릿 단위로 injection_rate를 해석한다. */

  // number of flits per packet
  _int_map["packet_size"] = 1;
  /* [한국어] 패킷당 플릿 수 기본값.
   * 1이면 패킷 = 1 플릿(헤더만). 멀티-플릿 패킷의 경우 더 크게 설정한다.
   * GPGPU-Sim에서는 실제 메모리 요청 크기에 따라 동적으로 결정한다. */

  AddStrField("packet_size", "");
  /* [한국어] 클래스별 패킷 크기를 벡터로 지정하는 문자열 오버라이드. */

  // if multiple values are specified per class, set probabilities for each
  _int_map["packet_size_rate"] = 1;
  /* [한국어] 클래스별 패킷 크기 확률 분포 기본 가중치.
   * 여러 packet_size 값이 지정된 경우 각 크기의 선택 확률 비율을 나타낸다. */

  AddStrField("packet_size_rate", "");
  /* [한국어] 패킷 크기 확률 분포를 벡터로 지정하는 문자열 오버라이드. */

  AddStrField( "injection_process", "bernoulli" );
  /* [한국어] 패킷 주입 프로세스 모델.
   * "bernoulli" = 베르누이(독립 확률) 주입 — 각 사이클에 injection_rate 확률로 주입.
   * "on_off" = On-Off 버스트 모델 (burst_alpha, burst_beta로 파라미터화). */

  _float_map["burst_alpha"] = 0.5;
  /* [한국어] On-Off 버스트 모델에서 On→Off 전이 확률 (버스트 간격 조절).
   * 값이 작을수록 On 구간(버스트)이 길어진다. */

  _float_map["burst_beta"]  = 0.5;
  /* [한국어] On-Off 버스트 모델에서 Off→On 전이 확률 (버스트 길이 조절).
   * 값이 클수록 Off 구간이 짧아져 버스트가 잦아진다. */

  _float_map["burst_r1"] = -1.0;
  /* [한국어] 버스트 On 구간에서의 주입 속도 (-1.0 = 미사용/기본).
   * 유효 값이 설정되면 On 구간의 주입 속도를 별도로 지정할 수 있다. */

  AddStrField( "priority", "none" );
  /* [한국어] 메시지 우선순위 부여 방식.
   * "none" = 우선순위 없음 (모든 패킷 동등 처리).
   * "age", "network_age", "local_age" 등 에이징 기반 우선순위 지원. */

  _int_map["batch_size"] = 1000;
  /* [한국어] 배치(batch) 시뮬레이션에서 한 배치당 주입할 패킷 수.
   * BatchTrafficManager::_SingleSim()이 이 값을 사용한다.
   * 각 노드에서 이 수만큼 패킷을 주입한 후 드레인 단계로 진입한다. */

  _int_map["batch_count"] = 1;
  /* [한국어] 실행할 배치 수. _SingleSim()의 외부 루프 반복 횟수.
   * batch_count=3이면 3개의 독립적인 배치를 순서대로 실행한다. */

  _int_map["max_outstanding_requests"] = 0;
  /* [한국어] 배치 모드에서 노드당 동시에 허용되는 최대 미완료(outstanding) 요청 수.
   * 0이면 제한 없음. 양의 정수이면 흐름 제어 역할을 한다.
   * _IssuePacket()에서 _requestsOutstanding[source] < max_outstanding를 검사한다. */

  // Use read/write request reply scheme
  _int_map["use_read_write"] = 0;
  /* [한국어] read/write 요청-응답 쌍 트래픽 모드 활성화 플래그.
   * 1이면 패킷이 요청(request)과 응답(reply)으로 구분되어 별도 VC를 사용한다.
   * GPGPU-Sim에서 메모리 읽기/쓰기를 별도 채널로 모델링할 때 활성화한다. */

  AddStrField("use_read_write", "");
  /* [한국어] 클래스별 read/write 모드를 벡터로 지정하는 문자열 오버라이드. */

  _float_map["write_fraction"] = 0.5;
  /* [한국어] use_read_write 활성화 시 write 요청의 비율 (0.0~1.0).
   * 0.5이면 read와 write가 50:50으로 발생한다. */

  AddStrField("write_fraction", "");
  /* [한국어] 클래스별 write 비율을 벡터로 지정하는 문자열 오버라이드. */

  // Control assignment of packets to VCs
  _int_map["read_request_begin_vc"] = 0;
  /* [한국어] read 요청 패킷이 사용 가능한 VC 범위의 시작 번호.
   * VC 0~5가 read 요청에 할당됨 (기본값 기준). */

  _int_map["read_request_end_vc"] = 5;
  /* [한국어] read 요청 패킷이 사용 가능한 VC 범위의 끝 번호 (포함). */

  _int_map["write_request_begin_vc"] = 2;
  /* [한국어] write 요청 패킷이 사용 가능한 VC 범위의 시작 번호.
   * read 요청과 VC 구간이 겹칠 수 있다 (0~5 vs 2~7). */

  _int_map["write_request_end_vc"] = 7;
  /* [한국어] write 요청 패킷이 사용 가능한 VC 범위의 끝 번호 (포함). */

  _int_map["read_reply_begin_vc"] = 8;
  /* [한국어] read 응답 패킷이 사용 가능한 VC 범위의 시작 번호.
   * 응답(reply)은 요청(request)과 분리된 VC 구간(8~15)을 사용해 데드락을 방지한다. */

  _int_map["read_reply_end_vc"] = 13;
  /* [한국어] read 응답 패킷이 사용 가능한 VC 범위의 끝 번호 (포함). */

  _int_map["write_reply_begin_vc"] = 10;
  /* [한국어] write 응답 패킷이 사용 가능한 VC 범위의 시작 번호. */

  _int_map["write_reply_end_vc"] = 15;
  /* [한국어] write 응답 패킷이 사용 가능한 VC 범위의 끝 번호 (포함). */

  // Control Injection of Packets into Replicated Networks
  _int_map["read_request_subnet"] = 0;
  /* [한국어] read 요청 패킷이 주입될 서브넷 인덱스 (subnets > 1일 때 유효).
   * 서브넷 분리를 통해 요청과 응답 트래픽 간 간섭을 줄인다. */

  _int_map["read_reply_subnet"] = 0;
  /* [한국어] read 응답 패킷이 주입될 서브넷 인덱스. */

  _int_map["write_request_subnet"] = 0;
  /* [한국어] write 요청 패킷이 주입될 서브넷 인덱스. */

  _int_map["write_reply_subnet"] = 0;
  /* [한국어] write 응답 패킷이 주입될 서브넷 인덱스. */

  // Set packet length in flits
  _int_map["read_request_size"]  = 1;
  /* [한국어] read 요청 패킷의 크기 (플릿 수). 기본값 1 = 헤더 플릿만.
   * GPU 메모리 읽기 요청은 주소만 담으므로 보통 작다. */

  AddStrField("read_request_size", "");
  /* [한국어] 클래스별 read 요청 크기를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["write_request_size"] = 1;
  /* [한국어] write 요청 패킷의 크기 (플릿 수).
   * 실제 데이터를 포함하므로 read 요청보다 클 수 있다 (캐시라인 크기 의존). */

  AddStrField("write_request_size", "");
  /* [한국어] 클래스별 write 요청 크기를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["read_reply_size"]    = 1;
  /* [한국어] read 응답 패킷의 크기 (플릿 수).
   * 실제 데이터를 포함하므로 일반적으로 read 요청보다 훨씬 크다. */

  AddStrField("read_reply_size", "");
  /* [한국어] 클래스별 read 응답 크기를 벡터로 지정하는 문자열 오버라이드. */

  _int_map["write_reply_size"]   = 1;
  /* [한국어] write 응답(ACK) 패킷의 크기 (플릿 수). 보통 1 플릿으로 작다. */

  AddStrField("write_reply_size", "");
  /* [한국어] 클래스별 write 응답 크기를 벡터로 지정하는 문자열 오버라이드. */

  //==== Simulation parameters ==========================

  // types:
  //   latency    - average + latency distribution for a particular injection rate
  //   throughput - sustained throughput for a particular injection rate

  AddStrField( "sim_type", "latency" );
  /* [한국어] 시뮬레이션 목표 유형.
   * "latency" = 주어진 주입 속도에서 평균 지연과 지연 분포를 측정.
   * "throughput" = 지속 처리율을 측정 (주입 속도를 높여가며 포화 지점 탐색).
   * "batch" = BatchTrafficManager가 활성화될 때 사용하는 배치 모드. */

  _int_map["warmup_periods"] = 3;
  /* [한국어] 워밍업(warm-up) 기간 수.
   * 통계 수집 전에 시뮬레이터를 안정 상태로 만들기 위해
   * sample_period * warmup_periods 사이클 동안 통계를 수집하지 않는다. */

  _int_map["sample_period"] = 1000;
  /* [한국어] 측정 주기 (사이클 단위).
   * 1000 사이클마다 지연/처리율 통계를 갱신하고 수렴 여부를 판단한다. */

  _int_map["max_samples"] = 10;
  /* [한국어] 최대 샘플 기간 수. warmup 이후 최대 10번의 sample_period를 실행한다.
   * 수렴 전이라도 이 횟수에 달하면 시뮬레이션을 종료한다. */

  // whether or not to measure statistics for a given traffic class
  _int_map["measure_stats"] = 1;
  /* [한국어] 트래픽 클래스별 통계 측정 활성화 플래그.
   * 1이면 해당 클래스의 지연/처리율 통계를 수집한다. */

  AddStrField("measure_stats", "");
  /* [한국어] 클래스별 통계 측정 여부를 벡터로 지정하는 문자열 오버라이드. */

  //whether to enable per pair statistics, caution N^2 memory usage
  _int_map["pair_stats"] = 0;
  /* [한국어] 소스-목적지 쌍별 통계 활성화 플래그.
   * 1이면 N×N 행렬 크기의 통계를 수집한다 (N=노드 수, 메모리 사용량 주의).
   * 0이면 전체 집계 통계만 수집한다. */

  // if avg. latency exceeds the threshold, assume unstable
  _float_map["latency_thres"] = 500.0;
  /* [한국어] 불안정 판정 지연 임계값 (사이클 단위).
   * 평균 지연이 이 값을 초과하면 네트워크가 포화(saturated)되었다고 판단한다. */

  AddStrField("latency_thres", "");
  /* [한국어] 클래스별 지연 임계값을 벡터로 지정하는 문자열 오버라이드. */

  // consider warmed up once relative change in latency / throughput between
  // successive iterations is smaller than this
  _float_map["warmup_thres"] = 0.05;
  /* [한국어] 워밍업 수렴 판정 임계값.
   * 연속 두 sample_period 간 지연/처리율의 상대 변화가 5% 미만이면 워밍업 완료로 간주. */

  AddStrField("warmup_thres", "");
  /* [한국어] 클래스별 워밍업 임계값을 벡터로 지정하는 문자열 오버라이드. */

  _float_map["acc_warmup_thres"] = 0.05;
  /* [한국어] 누적 워밍업 수렴 임계값 (누적 평균 기준 비교). */

  AddStrField("acc_warmup_thres", "");
  /* [한국어] 클래스별 누적 워밍업 임계값을 벡터로 지정하는 문자열 오버라이드. */

  // consider converged once relative change in latency / throughput between
  // successive iterations is smaller than this
  _float_map["stopping_thres"] = 0.05;
  /* [한국어] 측정 수렴 판정 임계값.
   * 워밍업 완료 후 연속 sample_period 간 변화가 5% 미만이면 시뮬레이션을 종료한다. */

  AddStrField("stopping_thres", "");
  /* [한국어] 클래스별 종료 임계값을 벡터로 지정하는 문자열 오버라이드. */

  _float_map["acc_stopping_thres"] = 0.05;
  /* [한국어] 누적 측정 수렴 판정 임계값 (누적 평균 기준 비교). */

  AddStrField("acc_stopping_thres", "");
  /* [한국어] 클래스별 누적 종료 임계값을 벡터로 지정하는 문자열 오버라이드. */

  _int_map["sim_count"] = 1;
  /* [한국어] 수행할 독립 시뮬레이션 실행 수.
   * 1보다 크면 동일 조건으로 여러 번 실행하여 평균 통계를 낸다. */

  _int_map["include_queuing"] = 1;
  /* [한국어] 소스 큐잉(source queuing) 지연을 측정에 포함할지 여부.
   * 1이면 패킷이 주입 대기 큐에서 대기한 시간도 지연에 포함된다. */

  _int_map["seed"] = 0;
  /* [한국어] 시뮬레이션 랜덤 시드 (트래픽 패턴 재현성을 위해 사용).
   * 동일한 시드로 실행하면 동일한 트래픽 패턴이 생성된다. */

  _int_map["print_activity"] = 0;
  /* [한국어] 사이클별 활동 정보 출력 여부.
   * 1이면 각 사이클의 플릿 이동을 stdout으로 출력한다 (디버그용, 매우 상세함). */

  _int_map["print_csv_results"] = 0;
  /* [한국어] CSV 형식으로 결과를 출력할지 여부.
   * 1이면 통계 결과를 CSV 형식으로 출력하여 스크립트 처리에 용이하다. */

  _int_map["deadlock_warn_timeout"] = 256;
  /* [한국어] 데드락 경고 타임아웃 (사이클 단위).
   * 네트워크에 패킷이 남아 있음에도 이 사이클 동안 진행이 없으면
   * 데드락 가능성을 경고 메시지로 출력한다. */

  _int_map["viewer_trace"] = 0;
  /* [한국어] 비주얼 트레이스 출력 활성화 (AerialVision 연동용).
   * 1이면 플릿 이동 트레이스를 파일로 기록하여 시각화 도구와 연동한다. */

  AddStrField("watch_file", "");
  /* [한국어] 감시할 특정 패킷/플릿 ID를 나열한 파일 경로.
   * 지정된 패킷의 이동을 상세 추적하는 디버그 기능. */

  AddStrField("watch_flits", "");
  /* [한국어] 감시할 플릿 ID 목록 (직접 지정, 공백 구분).
   * watch_file과 같은 기능이지만 파일 대신 설정에서 직접 ID를 나열한다. */

  AddStrField("watch_packets", "");
  /* [한국어] 감시할 패킷 ID 목록 (직접 지정). */

  AddStrField("watch_transactions", "");
  /* [한국어] 감시할 트랜잭션 ID 목록 (직접 지정).
   * 트랜잭션은 요청-응답 쌍을 하나의 단위로 묶은 것이다. */

  AddStrField("watch_out", "");
  /* [한국어] 감시 출력 파일 경로. 비어 있으면 stdout으로 출력한다. */

  AddStrField("stats_out", "");
  /* [한국어] 최종 통계 출력 파일 경로. 비어 있으면 stdout으로 출력한다. */

#ifdef TRACK_FLOWS
  /* [한국어] TRACK_FLOWS 컴파일 옵션 활성화 시에만 포함되는 플로우 추적 출력 파일 설정.
   * 각 출력 파일은 사이클별로 해당 통계를 기록하며, 네트워크 분석에 사용된다. */
  AddStrField("injected_flits_out", "");     // [한국어] 주입된 플릿 수 기록 파일
  AddStrField("received_flits_out", "");     // [한국어] 수신된 플릿 수 기록 파일
  AddStrField("stored_flits_out", "");       // [한국어] 버퍼에 저장된 플릿 수 기록 파일
  AddStrField("sent_flits_out", "");         // [한국어] 전송된 플릿 수 기록 파일
  AddStrField("outstanding_credits_out", "");// [한국어] 미사용 크레딧 수 기록 파일
  AddStrField("ejected_flits_out", "");      // [한국어] 네트워크에서 배출된 플릿 수 기록 파일
  AddStrField("active_packets_out", "");     // [한국어] 네트워크 내 활성 패킷 수 기록 파일
#endif

#ifdef TRACK_CREDITS
  /* [한국어] TRACK_CREDITS 컴파일 옵션 활성화 시에만 포함되는 크레딧 추적 출력 파일 설정.
   * 라우터 버퍼 상태를 사이클 단위로 기록하여 병목 지점을 분석한다. */
  AddStrField("used_credits_out", "");   // [한국어] 사용 중인 크레딧 수 기록 파일
  AddStrField("free_credits_out", "");   // [한국어] 여유 크레딧 수 기록 파일
  AddStrField("max_credits_out", "");    // [한국어] 최대 크레딧 수 기록 파일
#endif

  // batch only -- packet sequence numbers
  AddStrField("sent_packets_out", "");
  /* [한국어] 배치 모드 전용: 각 사이클에서 노드별 전송 패킷 시퀀스 번호를 기록하는 파일.
   * BatchTrafficManager::_SingleSim()에서 _packet_seq_no를 이 파일에 출력한다.
   * 배치 주입 진행 상황을 외부에서 모니터링할 때 사용한다. */

  //==================Power model params=====================
  _int_map["sim_power"] = 0;
  /* [한국어] NoC 전력 모델 활성화 플래그.
   * 1이면 각 사이클에서 라우터/링크 전력 소비를 추정한다.
   * BookSim 내장 전력 모델을 사용하며, AccelWattch와는 별도이다. */

  AddStrField("power_output_file", "pwr_tmp");
  /* [한국어] 전력 모델 결과 출력 파일 경로 (기본값: "pwr_tmp"). */

  AddStrField("tech_file", "");
  /* [한국어] 전력 모델 공정 파라미터 파일 경로.
   * PowerConfig가 이 파일을 파싱하여 Vdd, 커패시턴스 등을 읽는다. */

  _int_map["channel_width"] = 128;
  /* [한국어] 물리 링크(채널) 폭 (비트 단위).
   * 전력 계산에서 링크 스위칭 에너지를 추정할 때 사용한다.
   * 128비트 = 16바이트 링크가 기본값이다. */

  _int_map["channel_sweep"] = 0;
  /* [한국어] 채널 폭 스윕(sweep) 분석 활성화 플래그.
   * 1이면 여러 채널 폭에 대해 전력을 반복 계산한다. */

  //==================Network file===========================
  AddStrField("network_file", "");
  /* [한국어] 네트워크 토폴로지를 정의하는 외부 파일 경로.
   * 비어 있으면 topology 파라미터로 네트워크를 자동 생성한다.
   * 파일이 지정되면 해당 파일에서 노드/링크 구성을 직접 읽는다. */
}



/*
 * [한국어]
 * PowerConfig::PowerConfig - NoC 전력 모델 공정 파라미터 기본값 초기화
 *
 * @return: (생성자, 반환값 없음)
 *
 * 트랜지스터 크기, 전압, 누설 전류, 커패시턴스, 배선 파라미터 등
 * NoC 전력 계산에 필요한 모든 공정 파라미터를 0으로 초기화한다.
 * 실제 값은 tech_file을 ParseFile()로 파싱한 후 덮어쓰인다.
 * 모든 값이 0인 이유는 파일에서 명시적으로 지정되지 않은 파라미터가
 * 잘못된 기본값으로 전력 계산에 영향을 미치지 않도록 하기 위함이다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, sim_power=1일 때만 생성됨.
 * 호출 체인:
 *   power_module_init() → [PowerConfig::PowerConfig()] → ParseFile(tech_file)
 */
PowerConfig::PowerConfig( )
{
  _int_map["H_INVD2"] = 0;
  /* [한국어] INVD2(인버터) 게이트의 높이(Height) — 공정 그리드 단위.
   * 트랜지스터 물리 크기 → 게이트 커패시턴스 계산에 사용. */

  _int_map["W_INVD2"] = 0;
  /* [한국어] INVD2 인버터 게이트의 폭(Width) — 공정 그리드 단위. */

  _int_map["H_DFQD1"] = 0;
  /* [한국어] DFQD1(D 플립플롭) 게이트의 높이 — 순차 로직 면적 추정에 사용. */

  _int_map["W_DFQD1"] = 0;
  /* [한국어] DFQD1 D 플립플롭 게이트의 폭. */

  _int_map["H_ND2D1"] = 0;
  /* [한국어] ND2D1(NAND 게이트) 높이 — 조합 논리 면적 추정에 사용. */

  _int_map["W_ND2D1"] = 0;
  /* [한국어] ND2D1 NAND 게이트 폭. */

  _int_map["H_SRAM"] = 0;
  /* [한국어] SRAM 셀 높이 — VC 버퍼/FIFO SRAM 면적 추정에 사용. */

  _int_map["W_SRAM"] = 0;
  /* [한국어] SRAM 셀 폭. */

  _float_map["Vdd"] = 0;
  /* [한국어] 공급 전압 (Volt) — 동적 전력 = α·C·Vdd²·f 계산의 핵심 파라미터.
   * 전압의 제곱에 비례하므로 공정 스케일링의 주요 영향 인자. */

  _float_map["R"] = 0;
  /* [한국어] 등가 트랜지스터 온 저항 (Ohm) — RC 지연 및 전력 계산에 사용. */

  _float_map["IoffSRAM"] = 0;
  /* [한국어] SRAM 셀의 누설 전류 (Ampere) — 정적 전력 = Ioff·Vdd 계산에 사용.
   * 공정 스케일링에 따라 누설 전류가 동적 전력에 근접하거나 초과할 수 있다. */

  _float_map["IoffP"] = 0;
  /* [한국어] PMOS 트랜지스터 오프 상태 누설 전류 (Ampere). */

  _float_map["IoffN"] = 0;
  /* [한국어] NMOS 트랜지스터 오프 상태 누설 전류 (Ampere). */

  _float_map["Cg_pwr"] = 0;
  /* [한국어] 게이트 커패시턴스 (전력 계산용, Farad) — 스위칭 에너지 = ½·Cg·Vdd²·α에 사용. */

  _float_map["Cd_pwr"] = 0;
  /* [한국어] 드레인 커패시턴스 (전력 계산용, Farad). */

  _float_map["Cgdl"] = 0;
  /* [한국어] 게이트-드레인 중첩 커패시턴스 (Gate-Drain overlap, Farad).
   * 밀러 효과를 통해 입력 커패시턴스를 증가시키는 기생 성분. */

  _float_map["Cg"] = 0;
  /* [한국어] 게이트 커패시턴스 (지연 계산용, Farad) — 전파 지연 tpd = 0.69·RC 계산에 사용. */

  _float_map["Cd"] = 0;
  /* [한국어] 드레인 커패시턴스 (지연 계산용, Farad). */

  _float_map["LAMBDA"] = 0;
  /* [한국어] 공정 피처 크기의 절반 (LAMBDA = L_min/2, 미터 단위).
   * 모든 물리 치수를 LAMBDA 단위로 정규화하는 기준값. */

  _float_map["MetalPitch"] = 0;
  /* [한국어] 금속 배선 피치 (배선 간격, 미터 단위).
   * 배선 저항/커패시턴스 계산에서 단위 길이당 값을 스케일링하는 데 사용. */

  _float_map["Rw"] = 0;
  /* [한국어] 단위 길이당 배선 저항 (Ohm/미터) — 링크 RC 지연 계산에 사용. */

  _float_map["Cw_gnd"] = 0;
  /* [한국어] 배선과 접지(GND) 사이의 단위 길이당 커패시턴스 (Farad/미터).
   * 배선의 기생 커패시턴스 중 접지면에 대한 성분. */

  _float_map["Cw_cpl"] = 0;
  /* [한국어] 인접 배선 간 커플링 커패시턴스 (Farad/미터).
   * 고밀도 배선에서 인접 신호 간 용량성 커플링에 의한 기생 성분. */

  _float_map["wire_length"] = 0;
  /* [한국어] 링크(채널) 배선 길이 (미터 단위) — 배선 RC 전력을 계산할 때 Rw, Cw와 함께 사용.
   * 설정 파일에서 읽어 특정 토폴로지의 링크 길이를 반영한다. */
}
