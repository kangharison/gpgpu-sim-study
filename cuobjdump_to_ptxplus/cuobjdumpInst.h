/*
 * [한국어 설명] SASS 단일 명령어 표현 클래스 (cuobjdumpInst.h)
 *
 * === 파일의 역할 ===
 * cuobjdumpInst 클래스는 cuobjdump가 출력한 SASS 명령어 하나를 추상화한다.
 * 레이블, 프레디케이트, 니모닉(base), 베이스 수정자, 타입 수정자,
 * 피연산자, 프레디케이트 수정자를 저장하고, 이를 PTXPlus 문법으로
 * 출력(printCuobjdumpPtxPlus)하는 기능을 제공한다. SASS의 GPU 고유
 * 명령어를 GPGPU-Sim이 해석 가능한 PTX/PTXPlus 표현으로 변환하는
 * 핵심 데이터 컨테이너이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   sass.l/sass.y (SASS 파서)
 *     → cuobjdumpInstList::add() 로 cuobjdumpInst 객체 추가
 *     → cuobjdumpInstList::printCuobjdumpPtxPlusList()
 *         → cuobjdumpInst::printCuobjdumpPtxPlus() [이 파일]
 *             → output()를 통해 .ptxplus 파일 기록
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - 표준 라이브러리: <string>, <list>, <cstdlib>
 * 이 파일에 의존하는 모듈:
 *   - cuobjdumpInstList.h/.cc : cuobjdumpInst 객체들의 엔트리별 리스트 관리
 *   - sass.y                  : 파싱 중 cuobjdumpInst 생성 및 필드 설정
 *   - cuobjdump_to_ptxplus.cc : 최종 변환 출력 조율
 *
 * === 주요 함수/구조체 요약 ===
 * cuobjdumpInst          - SASS 명령어 하나를 표현하는 클래스
 *   m_label              - 분기 목적지 등에 사용되는 레이블
 *   m_predicate          - 명령어 실행 조건(predicate) 목록
 *   m_base               - SASS 니모닉(예: IADD, GLD, FADD 등)
 *   m_baseModifiers      - EQ/GE/반올림 모드 등 베이스 수정자
 *   m_typeModifiers      - .F32/.S32/.U64 등 피연산자 타입 수정자
 *   m_operands           - 변환된 피연산자 문자열 목록
 *   m_predicateModifiers - .not_sign/.sign/.carry 등 predicate 수정자
 * printCuobjdumpPtxPlus() - m_base에 따라 PTXPlus 명령어로 변환/출력
 * printCuobjdumpOperand() - 레지스터/메모리/즉시값/특수 레지스터 출력
 */

// Copyright (c) 2009-2012, Jimmy Kwa, Andrew Boktor
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _CUOBJDUMPINST_H_
#define _CUOBJDUMPINST_H_

// External includes
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>

// Local includes
//#include "cuobjdumpInstList.h"

/*
 * [한국어]
 * cuobjdumpInst — SASS 명령어 하나를 표현하는 클래스.
 *
 * SASS 디스어셈블리 한 줄을 구성 요소별로 분해하여 저장하고,
 * PTXPlus 출력 시 각 구성 요소를 적절히 조합한다.
 */
class cuobjdumpInst
{
protected:
	//instruction data
	std::string m_label; //instruction label
	// [한국어] 명령어에 붙은 레이블. SASS에서는 /* hex */ 형태의 인스트럭션 헥스값이
	// 변환되어 l0x... 형태의 PTXPlus 레이블로 사용된다.
	std::list<std::string>* m_predicate; //instruction predicate
	/* [한국어] 명령어 실행을 제어하는 predicate(C0/C1 등) 문자열 목록.
	 * 출력 시 @pN 형태로 변환된다. */
	std::string m_base; //instruction mnemonic
	/* [한국어] SASS 명령어의 니모닉. 예: IADD, GLD, FADD, MOV 등.
	 * printCuobjdumpPtxPlus()에서 이 값으로 변환 로직을 분기한다. */
	std::list<std::string>* m_baseModifiers; //base modifiers
	/* [한국어] 베이스 명령어에 붙는 수정자 목록(EQ, GE, .rz, .abs 등). */
	std::list<std::string>* m_typeModifiers; //operand types
	/* [한국어] 피연산자 타입 수정자 목록(.F32, .S32, .U64, .HI 등). */
	std::list<std::string>* m_operands; //operands
	/* [한국어] 변환 중간/최종 피연산자 문자열 목록.
	 * sass_parse()에서 추가되고 출력 시 순서대로 기록된다. */
	std::list<std::string>* m_predicateModifiers; //predicate modifiers
	/* [한국어] predicate에 붙는 수정자 목록(.sign, .not_sign, .carry 등). */

public:
	//Constructor
	cuobjdumpInst();
	~cuobjdumpInst();

	//accessors
	const std::string getBase();
	std::list<std::string>* getTypeModifiers();

	//Mutators
	void setLabel(const char* setLabelValue);
	void setPredicate(const char* setPredicateValue);
	void addPredicateModifier(const char* addPredicateMod);
	void setBase(const char* setBaseValue);
	void addBaseModifier(const char* addBaseMod);
	void addTypeModifier(const char* addTypeMod);
	void addOperand(const char* addOp);

	bool checkCubojdumpLabel(std::list<std::string> labelList, std::string label);

	void printCuobjdumpLabel(std::list<std::string> labelList);
	void printCuobjdumpPredicate();
	void printCuobjdumpTypeModifiers();
	void printCuobjdumpOutputModifiers(const char* defaultMod);
	void printCuobjdumpBaseModifiers();
	void printCuobjdumpOperand(std::string currentPiece, std::string operandDelimiter, std::string base);
	void printCuobjdumpOperandlohi(std::string op);
	void printCuobjdumpOperands();

	/*
	 * [한국어]
	 * printCuobjdumpPtxPlus - 이 SASS 명령어를 PTXPlus 형식으로 변환하여 출력
	 *
	 * @labelList: 현재 엔트리에서 사용된 레이블 목록(레이블 출력 여부 결정)
	 * @texList:   원본 PTX에서 추출한 실제 텍스처 이름 목록(TEX 명령어 변환용)
	 *
	 * m_base 값에 따라 거대한 if-else 분기를 통해 적절한 PTXPlus 명령어를
	 * 생성한다. 각 분기는 공통 패턴 printCuobjdumpPredicate(), output(),
	 * printCuobjdumpBaseModifiers(), printCuobjdumpTypeModifiers(),
	 * printCuobjdumpOperands()를 사용한다.
	 */
	void printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList);

	//print representation
	bool printHeaderInst();
	void printCuobjdumpInst();
	void printHeaderPtx();

	static void printStringList(std::list<std::string>* strlist);
};

#endif //_CUOBJDUMPINST_H_
