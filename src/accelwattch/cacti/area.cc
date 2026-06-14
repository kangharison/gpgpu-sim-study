/*****************************************************************************
 *                                McPAT/CACTI
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/

/*
 * [한국어 설명] CACTI 면적 모델 구현 파일 (area.cc)
 *
 * === 파일의 역할 ===
 * Area 클래스의 구현 파일이나 현재 버전에서는 실제 구현 함수 본문이 없다.
 * Area 클래스의 모든 멤버 함수(get_w, get_h, get_area, set_w, set_h, set_area)는
 * area.h 헤더에 인라인으로 정의되어 있어 별도 .cc 구현이 필요 없다. 이 파일은
 * 과거 구현이 있었거나 향후 확장을 위한 자리 표시자(placeholder)로 남아 있다.
 * 빌드 시스템이 이 .cc 파일을 컴파일하여 area.o를 생성하지만 실질적인 코드는 없다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Area 클래스는 CACTI 면적 모델의 최하위 기본 타입이다. Bank, Mat, Subarray, Decoder,
 * Wire 등 모든 컴포넌트가 Area를 멤버로 포함하여 자신의 물리적 레이아웃 크기를 표현한다.
 * AccelWattch → CACTI → Bank → Mat → ... → Area 순으로 면적이 하향 집계되고,
 * 최종적으로 Bank::area가 GPU 캐시 면적 추정의 결과로 AccelWattch에 반환된다.
 *
 * === 타 모듈과의 연결 ===
 * - area.h: Area 클래스 전체 선언 및 인라인 함수 정의를 포함.
 * - component.h: Component 클래스가 Area area 멤버를 선언하며, 이 파일이 component.h를 포함.
 * - decoder.h: Decoder 컴포넌트도 Area 인스턴스를 보유 (decoder.h 포함).
 * - parameter.h: g_ip, g_tp 전역 파라미터 (이 파일에서는 직접 사용 안 함).
 * - basic_circuit.h: CMOS 회로 계산 함수 (이 파일에서는 직접 사용 안 함).
 *
 * === 주요 함수/구조체 요약 ===
 * (이 파일에는 구현된 함수가 없음. Area 클래스의 함수는 area.h에 인라인 정의.)
 * - Area::get_area(): w×h 또는 직접 지정 area 반환 (area.h 인라인).
 * - Area::set_w/set_h/set_area(): 폭·높이·면적 설정 (area.h 인라인).
 */

#include "area.h"          // [한국어] Area 클래스 선언 및 인라인 함수 정의
#include "component.h"     // [한국어] Component 기반 클래스 (Area를 멤버로 보유)
#include "decoder.h"       // [한국어] Decoder 컴포넌트 (Area 인스턴스 보유)
#include "parameter.h"     // [한국어] g_ip, g_tp 전역 파라미터 (이 파일에서 직접 미사용)
#include "basic_circuit.h" // [한국어] CMOS 기초 회로 계산 함수 (이 파일에서 직접 미사용)
#include <iostream>        // [한국어] cout/cerr 출력 (이 파일에서 직접 미사용)
#include <math.h>          // [한국어] 수학 함수 (이 파일에서 직접 미사용)
#include <assert.h>        // [한국어] 불변식 검증 (이 파일에서 직접 미사용)

using namespace std; // [한국어] std 네임스페이스 전역 using (CACTI 코드베이스 관례)



