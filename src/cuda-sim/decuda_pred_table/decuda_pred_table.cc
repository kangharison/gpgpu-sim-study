/*
 * [한국어 설명] decuda predicate 조회 테이블 구현 (decuda_pred_table.cc)
 *
 * === 파일의 역할 ===
 * decuda_pred_table.h에서 선언한 pred_lookup() 함수를 구현한다.
 * NVIDIA G80 GPU에서 실제 하드웨어로 측정한 predicate 평가 진리표를
 * 32×16 정적 2차원 배열(pred_table)로 저장하고, condition code와
 * Z/S/C/O 플래그 조합을 인덱스로 하여 결과를 반환한다.
 * PTXPlus 모드에서 조건 분기/명령어의 predicate 평가 정확성을 보장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: PTXPlus predicate 처리 (ptx_ir.cc / instructions.cc)
 *   → pred_lookup(condition, flags) → pred_table[condition][flags] 반환
 * 기능 시뮬레이션(cuda-sim/) 보조 모듈.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: decuda_pred_table.h
 * 의존_받음: cuda-sim/ptx_ir.cc, cuda-sim/instructions.cc
 * 데이터 흐름: condition(0~31) × flags(0~15) → 정적 pred_table → bool
 *
 * === 주요 함수/구조체 요약 ===
 * pred_lookup(): condition과 flags로 pred_table을 조회하여 predicate 결과 반환
 * pred_table[32][16]: G80 하드웨어 측정 기반 정적 predicate 진리표
 */

/*Copyright (c) 2007, Wladimir J. van der Laan

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the organization nor the names of its
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#include "decuda_pred_table.h"

/*
 * [한국어]
 * pred_lookup - G80 predicate 조건/플래그를 정적 진리표에서 조회
 *
 * @condition: predicate 조건 코드 (0~31). 예: 0=fl, 1=lt, 2=eq, 5=ne, 15=tr 등.
 * @flags: Z/S/C/O 플래그의 4비트 조합 (0~15). 1=Z, 2=S, 4=C, 8=O.
 * @return: predicate 평가 결과 — true면 실행, false면 실행 안 함.
 *
 * G80 아키텍처에서 실제 하드웨어로 측정된 pred_table[32][16]을 직접 인덱싱.
 * 별도의 계산 없이 O(1) 조회로 predicate 결과를 반환한다.
 * condition이나 flags가 범위를 벗어나면 정의되지 않은 동작(undefined behavior)이
 * 발생할 수 있으나, 상위 호출자가 유효한 값을 전달한다고 가정한다.
 *
 * 호출 체인:
 *   PTXPlus predicate 평가 코드 → [이 함수] → pred_table[condition][flags]
 */
bool pred_lookup(int condition, int flags)
{

	// Logic table for G80 architecture, all condition codes against all
	// flag combinations. This was evaluated on actual hardware.
	// The flags are assigned to values like this:
	// 1 Z zero flag
	// 2 S sign flag
	// 4 C carry flag
	// 8 O overflow flag

	//
	// fl 0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
	static bool const pred_table[32][16] =
	    {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 00 fl
	     {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1}, // 01 lt
	     {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}, // 02 eq
	     {0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0}, // 03 le
	     {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}, // 04 gt
	     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, // 05 ne (also nz)
	     {1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1}, // 06 ge
	     {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0}, // 07 leg
	     {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, // 08 nan
	     {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, // 09 ltu
	     {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, // 0a equ (also zf)
	     {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1}, // 0b leu
	     {1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1}, // 0c gtu
	     {1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1}, // 0d neu
	     {1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0}, // 0e geu
	     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 0f tr
	     {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, // 10 of
	     {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}, // 11 cf
	     {0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0}, // 12 ab
	     {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, // 13 sf
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	     {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0}, // 1c nsf
	     {1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1}, // 1d ble
	     {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}, // 1e ncf
	     {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}}; // 1f nof

	return pred_table[condition][flags];
}
