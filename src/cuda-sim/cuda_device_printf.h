/*
 * [한국어 설명] GPU 디바이스 printf() 에뮬레이션 인터페이스 (cuda_device_printf.h)
 *
 * === 파일의 역할 ===
 * CUDA 커널 코드 내에서 호출된 printf()를 호스트 측에서 에뮬레이션하는 함수를 선언한다.
 * 실제 GPU에서는 printf가 디바이스 측 버퍼에 출력을 쌓고 나중에 호스트가 수집하지만,
 * GPGPU-Sim의 기능 시뮬레이션에서는 PTX 명령어 실행 시 이 함수가 직접 호출되어
 * 즉시 stdout으로 출력한다. 현재 %u, %f, %d 형식만 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: instructions.cc (PTX call 명령어) → 내장 함수 디스패치
 *   → gpgpusim_cuda_vprintf() → my_cuda_printf()
 * 기능 시뮬레이션(cuda-sim/) 레이어에 속하며, GPU 스레드가 printf()를 호출할 때
 * 타이밍 모델과 무관하게 즉시 실행된다.
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존: cuda-sim/ptx_ir.h (ptx_instruction, ptx_thread_info, function_info)
 * 의존받음: cuda-sim/instructions.cc (call 명령어 내 vprintf 디스패치)
 * 데이터 흐름: PTX 스레드의 로컬/파라미터 메모리 → 형식 문자열 + 인자 읽기 → stdout 출력
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpusim_cuda_vprintf(): PTX call 명령어에서 호출 — 인자를 메모리에서 읽어 printf 에뮬
 * my_cuda_printf(): 형식 문자열을 파싱하여 실제 출력 수행 (cuda_device_printf.cc에 구현)
 */
// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef CUDA_DEVICE_PRINTF_INCLUDED /* [한국어] 헤더 중복 포함 방지 가드 */
#define CUDA_DEVICE_PRINTF_INCLUDED

/*
 * [한국어]
 * gpgpusim_cuda_vprintf - CUDA 디바이스 printf() PTX call 명령어 핸들러
 *
 * @pI: 현재 실행 중인 PTX call 명령어 (피호출 함수 파라미터 정보 포함)
 * @thread: printf를 호출한 PTX 스레드 — 로컬 메모리에서 인자 읽기
 * @target_func: vprintf 함수 정보 — 파라미터 수, 크기, 리턴 여부 기술
 *
 * CUDA PTX의 call 명령어가 vprintf/printf 내장 함수를 호출할 때 실행되는 에뮬레이션 함수.
 * 두 파라미터(형식 문자열 포인터, 인자 목록 포인터)를 스레드 로컬 메모리에서 읽고,
 * 각 포인터가 가리키는 실제 데이터를 메모리 공간에서 바이트 단위로 읽어 my_cuda_printf에 전달.
 * 현재 %u, %f, %d 형식만 지원; 그 외 형식자 발견 시 abort().
 * 구현: cuda_device_printf.cc
 *
 * 호출 체인:
 *   instructions.cc (call 명령어) → [이 함수] → my_cuda_printf → fprintf(stdout)
 */
void gpgpusim_cuda_vprintf(const class ptx_instruction* pI,
                           class ptx_thread_info* thread,
                           const class function_info* target_func);

#endif
