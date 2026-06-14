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
 * [한국어 설명] CACTI 결과 출력 함수 헤더 (io.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 모델링 결과(uca_org_t)를 표준 출력으로 내보내는 두 가지 함수를 선언한다.
 * output_UCA()는 인간이 읽기 좋은 형태로 캐시 면적/지연/전력 결과를 출력하며,
 * output_data_csv()는 동일 결과를 CSV(쉼표 구분 값) 형식으로 출력하여 자동 파싱에 적합하다.
 * AccelWattch에서는 GPU 캐시 추정 결과를 검증하거나 로깅할 때 이 함수들을 사용한다.
 * 이 헤더는 CACTI 내부에서 널리 include되므로 의존성이 최소화되어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 계산 흐름의 최종 단계: UCA 결과 집계 완료 → output_UCA() / output_data_csv() 호출
 * 호출자: cacti_interface.cc의 최상위 진입 함수(cacti()) 또는 AccelWattch의 결과 보고 코드
 * 이 파일 자체는 계산 로직 없이 인터페이스만 선언하므로 헤더 포함 의존성이 없음.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이션 결과 보고 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈: const.h (기본 상수/타입), cacti_interface.h (uca_org_t 결과 구조체 정의)
 * 상위 모듈: cacti_interface.cc — 캐시 파라미터 탐색 완료 후 최적 결과를 이 함수로 출력
 * 데이터 흐름: uca_org_t (면적, access_time, power, cycle_time 등) → stdout (텍스트 또는 CSV)
 * parameter.h도 io.h를 include하므로 순환 의존 없이 기술 파라미터 출력도 지원.
 *
 * === 주요 함수/구조체 요약 ===
 * output_data_csv() : uca_org_t 결과를 CSV 한 줄로 출력. 자동화 스크립트나 스프레드시트 파싱에 사용.
 * output_UCA()      : uca_org_t 결과를 사람이 읽기 좋은 형태(레이블+값)로 여러 줄 출력.
 *                     캐시 면적, 접근 시간, 사이클 타임, 읽기/쓰기 동적 전력, 누설 전력을 포함.
 */

#ifndef __IO_H__
#define __IO_H__


#include "const.h"          // [한국어] CACTI 공통 상수 및 기본 타입 정의
#include "cacti_interface.h" // [한국어] uca_org_t 결과 구조체 정의 (면적, 지연, 전력 등 출력 필드)


/*
 * [한국어]
 * output_data_csv - CACTI 캐시 결과를 CSV 형식으로 출력
 *
 * @fin_res : 출력할 UCA 결과 구조체 (const 참조 — 수정 없음)
 *            uca_org_t에는 access_time, cycle_time, area, power 등이 포함됨
 * @return  : (void) — 결과를 stdout에 CSV 한 줄로 직접 출력
 *
 * 자동화 파이프라인(Python 스크립트, 스프레드시트 등)이 파싱할 수 있도록
 * 모든 결과 필드를 쉼표로 구분하여 한 줄로 출력한다.
 * AccelWattch가 GPU 캐시 에너지 추정 결과를 로그 파일에 기록할 때 사용.
 *
 * 호출 체인:
 *   cacti() 또는 AccelWattch 보고 코드 → [output_data_csv()]
 */
void output_data_csv(const uca_org_t & fin_res);

/*
 * [한국어]
 * output_UCA - CACTI UCA 캐시 결과를 사람이 읽기 좋은 형태로 출력
 *
 * @fin_res : 출력할 UCA 결과 구조체 포인터 (비const — 일부 필드를 정규화하여 출력할 수 있음)
 *            uca_org_t에는 면적(mm²), 접근 시간(ns), 사이클 타임(ns),
 *            읽기/쓰기 동적 전력(mW), 누설 전력(mW) 등이 포함됨
 * @return  : (void) — 결과를 stdout에 여러 줄 레이블+값 형태로 출력
 *
 * 개발자 또는 사용자가 캐시 특성을 한눈에 확인할 수 있도록 포맷된 출력을 제공한다.
 * AccelWattch에서 GPU 캐시 모델 결과를 검증하거나 디버그할 때 사용.
 *
 * 호출 체인:
 *   cacti() 또는 AccelWattch 보고 코드 → [output_UCA()]
 */
void output_UCA(uca_org_t * fin_res);


#endif
