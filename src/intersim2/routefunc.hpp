// $Id: routefunc.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 라우팅 함수 헤더 (routefunc.hpp)
 *
 * === 파일의 역할 ===
 * NoC 라우팅 알고리즘의 함수 포인터 타입, 글로벌 라우팅 맵, VC 범위 전역 변수를 선언한다.
 * 실제 라우팅 함수 구현은 routefunc.cpp에 있으며, InitializeRoutingMap()이 호출되면
 * 사용 가능한 모든 라우팅 알고리즘이 gRoutingFunctionMap에 등록된다.
 * 라우터는 설정 파일에서 지정된 알고리즘 이름으로 함수를 조회하여 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   CreateInterconnect() → InitializeRoutingMap() (함수 맵 초기화, VC 범위 설정)
 *   Router 생성자 → gRoutingFunctionMap["routing_function"] → tRoutingFunction 획득
 *   VC::Route(rf, router, flit, in_channel) → rf() (실제 라우팅 함수 호출)
 *   GPUTrafficManager::_Step() → _rf(NULL, cf, -1, &route_set, true) (주입 VC 결정)
 *
 * === 타 모듈과의 연결 ===
 * - vc.cpp: Route()에서 tRoutingFunction 함수 포인터로 라우팅 계산
 * - interconnect_interface.cpp: InitializeRoutingMap() 호출
 * - routefunc.cpp: 실제 라우팅 알고리즘 구현 및 gRoutingFunctionMap 등록
 * - flit.hpp: 라우팅 함수가 Flit의 dest, type, ph, intm 필드 참조
 * - router.hpp: 라우팅 함수가 Router의 GetID(), GetUsedCredit() 참조
 *
 * === 주요 함수/구조체 요약 ===
 * - tRoutingFunction: 라우팅 함수 포인터 타입 typedef
 * - InitializeRoutingMap(): 모든 라우팅 함수를 gRoutingFunctionMap에 등록
 * - gRoutingFunctionMap: 이름 → 함수 포인터 맵
 * - gNumVCs / gRead/WriteReq/ReplyBegin/EndVC: VC 범위 전역 변수 (요청/응답 분리)
 */

#ifndef _ROUTEFUNC_HPP_
#define _ROUTEFUNC_HPP_

#include "flit.hpp"         // [한국어] Flit 타입 (라우팅 함수의 파라미터)
#include "router.hpp"       // [한국어] Router 타입 (현재 라우터 정보 참조)
#include "outputset.hpp"    // [한국어] OutputSet: 라우팅 결과 출구 포트 집합
#include "config_utils.hpp" // [한국어] Configuration: 설정 파일 파싱 객체

/*
 * [한국어]
 * tRoutingFunction - NoC 라우팅 함수의 함수 포인터 타입
 *
 * @param1 (const Router*): 현재 라우터 (위치 정보 제공). 주입 시에는 NULL.
 * @param2 (const Flit*): 라우팅 대상 Flit (dest, type, ph, intm 등 참조)
 * @param3 (int in_channel): 입구 채널 번호 (DOR에서 방향 전환 감지)
 * @param4 (OutputSet*): 라우팅 결과 출구 포트 집합 (AddRange()로 채움)
 * @param5 (bool inject): true이면 주입(injection) 단계 (out_port = -1 설정)
 *
 * 라우팅 함수는 OutputSet에 가능한 (출구 포트, VC 범위) 쌍을 추가한다.
 * inject=true이면 out_port=-1을 설정하여 주입 시 VC 선택만 수행.
 */
typedef void (*tRoutingFunction)( const Router *, const Flit *, int in_channel, OutputSet *, bool );

/*
 * [한국어]
 * InitializeRoutingMap() - 모든 라우팅 알고리즘을 gRoutingFunctionMap에 등록한다
 *
 * @config: BookSim 설정 파일 파싱 결과 (num_vcs, read/write request/reply VC 범위)
 *
 * 동작 순서:
 *   1. gNumVCs = config의 num_vcs 값
 *   2. VC 범위 전역 변수 설정 (gReadReqBeginVC 등, 음수이면 기본값 사용)
 *   3. 모든 라우팅 함수를 gRoutingFunctionMap에 이름 → 포인터로 등록
 *      예: "dor_mesh" → dim_order_mesh, "valiant_mesh" → valiant_mesh
 *
 * 요청/응답 VC 분리의 목적:
 *   READ_REQUEST/WRITE_REQUEST는 하위 VC (gReadReqBeginVC .. gReadReqEndVC)
 *   READ_REPLY/WRITE_REPLY는 상위 VC (gReadReplyBeginVC .. gReadReplyEndVC)
 *   이 분리로 요청-응답 순환 의존성에 의한 데드락을 방지한다.
 *
 * 호출 체인: InterconnectInterface::CreateInterconnect() → [이 함수]
 */
void InitializeRoutingMap( const Configuration & config );

// [한국어] 이름 → 라우팅 함수 포인터 맵 (routefunc.cpp에서 정의)
// 설정 파일의 "routing_function" 값으로 라우터가 사용할 함수를 조회
extern map<string, tRoutingFunction> gRoutingFunctionMap;

// [한국어] 전체 VC 수 (설정 파일의 num_vcs, 라우팅 함수에서 VC 범위 계산에 사용)
extern int gNumVCs;

// [한국어] 읽기 요청 패킷이 사용하는 VC 범위 (Flit::READ_REQUEST 타입)
// 설정 파일의 read_request_begin/end_vc; 기본값: 0 .. gNumVCs/2-1
extern int gReadReqBeginVC, gReadReqEndVC;

// [한국어] 쓰기 요청 패킷이 사용하는 VC 범위 (Flit::WRITE_REQUEST 타입)
// 설정 파일의 write_request_begin/end_vc; 기본값: 0 .. gNumVCs/2-1 (read와 동일)
extern int gWriteReqBeginVC, gWriteReqEndVC;

// [한국어] 읽기 응답 패킷이 사용하는 VC 범위 (Flit::READ_REPLY 타입)
// 설정 파일의 read_reply_begin/end_vc; 기본값: gNumVCs/2 .. gNumVCs-1
extern int gReadReplyBeginVC, gReadReplyEndVC;

// [한국어] 쓰기 응답(ACK) 패킷이 사용하는 VC 범위 (Flit::WRITE_REPLY 타입)
// 설정 파일의 write_reply_begin/end_vc; 기본값: gNumVCs/2 .. gNumVCs-1
extern int gWriteReplyBeginVC, gWriteReplyEndVC;

#endif
