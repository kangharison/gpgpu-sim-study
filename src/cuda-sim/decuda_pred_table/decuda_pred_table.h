/*
 * [한국어 설명] decuda predicate 조회 테이블 헤더 (decuda_pred_table.h)
 *
 * === 파일의 역할 ===
 * PTXPlus(SASS 수준의 확장 PTX)를 사용할 때, NVIDIA G80 아키텍처의 predicate
 * 평가 결과를 실제 하드웨어 측정값으로부터 구축한 정적 조회 테이블(pred_table)을
 * 통해 제공한다. condition code(행)와 Z/S/C/O 플래그 조합(열)을 입력으로 받아
 * 해당 분기/명령어가 실행되어야 하는지 true/false를 반환한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: PTXPlus 모드에서 predicate 명령어 평가
 *   → cuda-sim/instructions.cc 또는 ptx_ir.cc 난이도 분기 처리
 *   → pred_lookup(condition, flags) 호출
 * 기능 시뮬레이션(cuda-sim/)의 보조 모듈이며, 타이밍 모델과 무관하게 즉시 조회된다.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: 없음 (독립적인 순수 C 함수)
 * 의존_받음: cuda-sim/ptx_ir.cc (PTXPlus predicate 처리), cuda-sim/instructions.cc
 * 데이터 흐름: condition(0~31) × flags(0~15) → pred_table[32][16] → bool
 *
 * === 주요 함수/구조체 요약 ===
 * pred_lookup(): G80 하드웨어 기준 predicate 평가 결과를 정적 테이블에서 조회
 */

#ifndef DECUDA_PRED_TABLE_H
#define DECUDA_PRED_TABLE_H

/*
 * [한국어]
 * pred_lookup - G80 predicate 조건/플래그에 따른 평가 결과 조회
 *
 * @condition: predicate 조건 코드 (0~31). 예: 0=fl, 1=lt, 2=eq, 5=ne, 15=tr 등.
 * @flags: 현재 연산 결과 상태 플래그 비트 조합 (0~15).
 *         1=Z(zero), 2=S(sign), 4=C(carry), 8=O(overflow).
 * @return: predicate 평가 결과 — true면 명령어/분기 실행, false면 실행 안 함.
 *
 * PTXPlus 모드에서 NVIDIA G80 GPU의 실제 predicate 평가 동작을 재현하기 위해
 * 하드웨어에서 측정된 32×16 정적 진리표를 조회한다.
 * 구현은 decuda_pred_table.cc에 있다.
 */
bool pred_lookup(int condition, int flags);

#endif
