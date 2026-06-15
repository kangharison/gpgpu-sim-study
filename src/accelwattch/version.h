/*****************************************************************************
 *                                McPAT
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
 * [한국어 설명] McPAT 버전 번호 매크로 정의 (version.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT(Multi-core Power, Area, and Timing) 도구의 버전 번호를
 * 전처리기 매크로로 정의한다. 버전은 주(VER_MAJOR).부(VER_MINOR).갱신일
 * (VER_UPDATE) 세 부분으로 구성된다. AccelWattch는 McPAT 0.8 베타(2010년 8월)를
 * 기반으로 GPU 전력 모델링 기능을 추가·확장한 프레임워크이므로, 이 버전 정보는
 * AccelWattch가 파생된 McPAT 원본 버전을 식별하는 역할을 한다. 버전 매크로는
 * 도구 출력 배너, 로그, 오류 메시지에서 버전 정보를 문자열로 출력할 때 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 서브시스템은 GPGPU-Sim 타이밍 시뮬레이터
 * (gpgpu-sim/gpu-sim.cc의 사이클 루프)에서 수집된 하드웨어 카운터를 받아
 * McPAT 기반 전력 추정값을 반환한다. 이 파일은 그 McPAT 엔진의 최상위
 * 식별자 계층에 위치하며, 초기화/출력 루틴에서 참조된다. 실행 컨텍스트:
 * 호스트 유저스페이스, 시뮬레이터 시작 시 버전 출력 경로에서 단 한 번 참조됨.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: 없음 (순수 매크로 상수 헤더)
 * - 이 파일에 의존하는 모듈: McPAT 메인 엔트리 포인트(mcpat.cc) 및
 *   AccelWattch 초기화 코드 — 버전 번호 문자열 출력 시 참조
 * - 데이터 흐름: 전처리 단계에서 VER_MAJOR/VER_MINOR/VER_UPDATE가 컴파일 시 상수로
 *   치환되며, 런타임 데이터 흐름에는 관여하지 않음
 * - 공유 상태: 없음 (매크로 상수는 컴파일 타임에 인라인 치환됨)
 *
 * === 주요 함수/구조체 요약 ===
 * - VER_MAJOR=0: 주 버전 번호 — 0은 베타(beta) 릴리스를 의미
 * - VER_MINOR=8: 부 버전 번호 — McPAT 0.8 릴리스
 * - VER_UPDATE="Aug, 2010": 갱신일 문자열 — 2010년 8월 HP Labs 릴리스
 *   (AccelWattch MICRO 2021 논문은 이 버전의 McPAT을 GPU용으로 확장함)
 */

#ifndef VERSION_H_
#define VERSION_H_

#define VER_MAJOR 0 /* beta release */
/* [한국어] 주 버전 번호: 0 = 베타(beta) 릴리스 상태.
 * McPAT은 HP Labs에서 2010년 공개 당시 아직 베타 단계였으며,
 * AccelWattch(MICRO 2021)가 이 0.8 베타를 GPU 전력 모델용으로 확장함.
 * 이 매크로는 버전 출력 시 "Version: 0.8" 형태로 사용됨. */

#define VER_MINOR 8
/* [한국어] 부 버전 번호: 8 → McPAT 0.8 릴리스.
 * VER_MAJOR와 조합하여 "0.8"이 되며, McPAT 공개 버전 계보에서
 * CPU 중심 다중코어 전력 모델의 최초 공개 릴리스에 해당한다.
 * AccelWattch는 이 버전을 분기(fork)하여 GPU SM, SIMT, 대규모 레지스터 파일,
 * 공유 메모리 등 GPU 특화 컴포넌트 전력 모델을 추가하였다. */

#define VER_UPDATE "Aug, 2010"
/* [한국어] 갱신일 문자열: "Aug, 2010" — 2010년 8월 HP Labs McPAT 공개 시점.
 * 이 날짜는 HP Labs가 McPAT 0.8 소스코드를 공개한 시점이며, 이후
 * AccelWattch 팀(MICRO 2021)이 GPU 확장을 추가한 날짜와는 다름.
 * 문자열 형식이므로 버전 비교 등 프로그램 로직에는 사용되지 않으며,
 * 출력 배너(printf/cout)에서 사용자 정보 출력용으로만 활용됨. */

#endif /* VERSION_H_ */
