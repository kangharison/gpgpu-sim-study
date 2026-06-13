// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*
 * [한국어 설명] AerialVision 시각화 지원 및 시간 벡터 추적 헤더 (visualizer.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 AerialVision 성능 시각화 도구 지원에 필요한 인터페이스를 선언한다.
 * 크게 두 가지 기능을 제공한다:
 * (1) gpgpu_sim::visualizer_printstat(): 매 인터벌 통계를 AerialVision gzip 파일에 출력
 * (2) time_vector: 각 메모리 요청(mem_fetch)이 시스템의 각 단계를 통과하는
 *     타임스탬프를 추적하여 레이턴시 분포를 계산하고 시각화 파일에 출력
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpu-sim.cc::cycle() → 인터벌마다 visualizer_printstat() 호출
 *   → AerialVision gzip 파일(g_visualizer_filename)에 누적 통계 출력
 *   mem_fetch 생성/완료 시점 → time_vector_update() → 각 단계 타임스탬프 기록
 *   → 완료 시 calculate_ld/st_dist() → AerialVision 파일에 레이턴시 분포 출력
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: stdio.h (FILE*), zlib.h (gzFile)
 * 사용처: visualizer.cc (모든 함수 구현), mem_fetch.cc (time_vector_update 호출),
 *          gpu-sim.cc (visualizer_printstat 호출)
 * 공유 자료구조: mem_fetch의 uid와 mem_access_type(READ_REQUEST 등),
 *               mem_latency_stat.h의 NUM_MEM_REQ_STAT, IN_SHADER_FETCHED 등
 *
 * === 주요 함수/구조체 요약 ===
 * time_vector_create()    - my_time_vector 전역 객체 생성 (size: 메모리 요청 단계 수)
 * time_vector_update()    - 특정 mem_fetch uid의 slot(단계)에 사이클 타임스탬프 기록
 * check_time_vector_update() - 레이턴시 계산 일관성 검증 (디버그용)
 * time_vector_print()     - 전체 LD/ST 레이턴시 분포 stdout에 출력
 */

#ifndef VISUALIZER_H_INCLUDED
#define VISUALIZER_H_INCLUDED

#include <stdio.h> /* [한국어] FILE* 타입 (gpgpu_sim::visualizer_printstat 등에서 사용) */
#include <zlib.h>  /* [한국어] gzFile — AerialVision 압축 출력 파일 스트림 */

/*
 * [한국어]
 * time_vector_create() - my_time_vector 전역 객체 생성
 *
 * @size: 추적할 메모리 요청 단계 수 (NUM_MEM_REQ_STAT에 해당)
 *
 * ld(읽기)와 st(쓰기) 요청 타임스탬프 추적 자료구조를 size 슬롯으로 초기화.
 * 시뮬레이터 초기화 시 1회 호출.
 */
void time_vector_create(int size);

/*
 * [한국어]
 * time_vector_print() - 전체 LD/ST 메모리 레이턴시 분포를 stdout에 출력
 *
 * calculate_dist() 후 LD/ST 레이턴시 분포를 "LD_mem_lat_dist" / "ST_mem_lat_dist" 형식으로 출력.
 * 시뮬레이션 종료 시 결과 요약에 사용.
 */
void time_vector_print(void);

/*
 * [한국어]
 * time_vector_update() - 특정 메모리 요청의 단계별 타임스탬프 기록
 *
 * @uid: 메모리 요청(mem_fetch)의 고유 UID
 * @slot: 기록할 단계 인덱스 (IN_ICNT_TO_MEM, IN_SHADER_FETCHED 등)
 * @cycle: 현재 시뮬레이션 사이클 (타임스탬프)
 * @type: 요청 타입 (READ_REQUEST, READ_REPLY, WRITE_REQUEST, WRITE_ACK)
 *
 * READ_REQUEST/READ_REPLY이면 ld_time_map에, WRITE_REQUEST/WRITE_ACK이면 st_time_map에 기록.
 * mem_fetch.cc에서 각 파이프라인 단계 진입 시 호출.
 */
void time_vector_update(unsigned int uid, int slot, long int cycle, int type);

/*
 * [한국어]
 * check_time_vector_update() - 레이턴시 계산 일관성 검증 (디버그용)
 *
 * @uid: 메모리 요청 UID
 * @slot: 검증할 단계 인덱스
 * @latency: 검증할 레이턴시 값
 * @type: 요청 타입
 *
 * slot 타임스탬프 - IN_ICNT_TO_MEM 타임스탬프가 latency와 일치하는지 assert로 검증.
 * 타임스탬프 계산 오류를 조기에 감지하기 위한 진단 함수.
 */
void check_time_vector_update(unsigned int uid, int slot, long int latency,
                              int type);

#endif
