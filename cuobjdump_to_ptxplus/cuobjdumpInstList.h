/*
 * [한국어 설명] SASS → PTXPlus 변환 IR 컨테이너 (cuobjdumpInstList.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 cuobjdumpInst 객체들을 엔트리(커널/함수) 단위로 묶고,
 * 변환 과정에서 필요한 상수 메모리, 전역 메모리, 로컬 메모리, 텍스처
 * 정보를 함께 관리하는 cuobjdumpInstList 클래스와 관련 구조체들을
 * 선언한다. SASS 파서(sass.y)와 ELF 파서(elf.y)가 이 구조체들을
 * 채우며, 최종 출력 시 PTXPlus의 .entry 헤더, 메모리 선언, 레지스터
 * 선언, 그리고 변환된 명령어들을 순서대로 출력한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   sass.y / elf.y / ptx_parser.h (파서)
 *     → cuobjdumpInstList 멤버 함수 호출로 IR 구축
 *     → cuobjdump_to_ptxplus.cc
 *         → printCuobjdumpPtxPlusList(g_headerList) 호출
 *             → PTXPlus 파일 출력
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInst.h : 단일 SASS 명령어 표현 클래스
 *   - <list>, <map>, <string> : IR 컨테이너 자료구조
 * 이 파일에 의존하는 모듈:
 *   - cuobjdumpInstList.cc : 이 헤더의 구현
 *   - sass.y, elf.y, ptx_parser.h : 파싱 중 IR 조작
 *   - cuobjdump_to_ptxplus.cc : 최종 출력 조율
 *
 * === 주요 함수/구조체 요약 ===
 * constMemory        - 상수 메모리 뱅크 하나(.const)를 표현
 * constMemory2       - 엔트리별 상수 메모리 뱅크(constant1<kernel>)
 * constMemoryPtr     - 상수 메모리 포인터(.constptr)
 * globalMemory       - 전역 메모리 심볼 하나(.global)
 * localMemory        - 로컬 메모리 슬롯 하나(.local)
 * cuobjdumpEntry     - 하나의 GPU 커널/함수 엔트리
 *   m_instList       - 이 엔트리의 SASS 명령어 리스트
 *   m_largestRegIndex- 선언해야 할 최대 일반 레지스터 인덱스
 *   m_largestOfsRegIndex - 선언해야 할 최대 offset 레지스터 인덱스
 *   m_largestPredIndex- 선언해야 할 최대 predicate 인덱스
 *   m_reg124/m_oreg127 - 특수 레지스터 사용 여부
 *   m_lMemSize       - 이 엔트리의 로컬 메모리 크기
 *   m_labelList      - 엔트리 내 사용된 레이블 목록
 * cuobjdumpInstList  - 모든 엔트리와 메모리 선언을 담는 최상위 컨테이너
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

#ifndef _CUOBJDUMPINSTLIST_H_
#define _CUOBJDUMPINSTLIST_H_

// External includes
#include <list>
#include <map>
#include <string>

// Local includes
#include "cuobjdumpInst.h"

/*
 * [한국어]
 * constMemory — 전역 또는 엔트리 공유 상수 메모리 뱅크 하나를 표현.
 *
 * index:       상수 메모리 뱅크 인덱스(예: constant0의 경우 0)
 * entryIndex:  이 상수 메모리가 속한 엔트리 인덱스(0이면 전역)
 * type:        .u32 등 원소 타입 문자열
 * m_constMemory: 16진수 상수값 문자열들의 리스트
 */
// Used for entry specific constant memory segments (c1)
struct constMemory
{
	int index;
	int entryIndex;
	const char* type;
	std::list<std::string> m_constMemory;
};

/*
 * [한국어]
 * constMemory2 — 엔트리별 상수 메모리 뱅크를 표현.
 *
 * kernel:      이 상수 메모리가 속한 커널 이름
 * type:        원소 타입
 * m_constMemory: 상수값 리스트
 */
struct constMemory2
{
	const char* kernel;
	const char* type;
	std::list<std::string> m_constMemory;
};

/*
 * [한국어]
 * constMemoryPtr — 상수 메모리 포인터(.constptr) 선언을 표현.
 *
 * bytes:       포인터가 가리키는 상수 메모리 영역의 바이트 수
 * name:        심볼 이름
 * destination: 목표 상수 메모리 뱅크 이름(주로 "constant0")
 * offset:      destination 내 바이트 오프셋
 */
// Used for uninitialized constant memory (globally defined)
struct constMemoryPtr
{
	int bytes;
	std::string name;

	std::string destination;
	int offset;
};

/*
 * [한국어]
 * globalMemory — ELF SYMTAB에서 추출한 전역 메모리 심볼 하나를 표현.
 *
 * offset:      .elf 상에서의 오프셋(워드 단위로 변환되어 저장)
 * bytes:       전역 변수의 크기(바이트)
 * name:        전역 변수 이름
 */
// Used for global memory segments
struct globalMemory
{
	int offset;
	int bytes;
	std::string name;
};

/*
 * [한국어]
 * cuobjdumpEntry — 하나의 GPU 커널/함수(엔트리)에 해당하는 IR 노드.
 *
 * SASS의 "Function : <이름>" 블록 하나를 표현하며, 해당 엔트리의
	 * 명령어 목록과 레지스터/predicate/로컬 메모리/레이블 정보를 포함한다.
 */
struct cuobjdumpEntry
{
	//char* m_entryName;
	std::string m_entryName;
	std::list<cuobjdumpInst> m_instList;	// List of cuobjdump instructions

	// Register list
	int m_largestRegIndex;
	/* [한국어] 이 엔트리에서 사용된 최대 일반 레지스터(R#) 번호.
	 * 출력 시 .reg .u32 $r<(최대+1)>; 형태로 선언한다. */
	int m_largestOfsRegIndex;
	/* [한국어] 이 엔트리에서 사용된 최대 offset 레지스터(A#) 번호. */
	bool m_reg124;
	/* [한국어] 레지스터 R124가 사용되었는지 여부(특수 취급). */
	bool m_oreg127;
	/* [한국어] 출력 레지스터 o[0x7f]가 사용되었는지 여부. */

	// Predicate list
	int m_largestPredIndex;
	/* [한국어] 이 엔트리에서 사용된 최대 predicate 번호. */

	// Local memory size
	int m_lMemSize;
	/* [한국어] 이 엔트리의 로컬 메모리 크기(바이트). 0보다 크면
	 * .local .b8 l<entryIndex>[크기]; 로 선언한다. */

	//use for recording used labels
	std::list<std::string> m_labelList;
	/* [한국어] 이 엔트리 내에서 참조된 레이블 목록. */

   // Histogram for operation per cycle count
   std::map<std::string, int> m_opPerCycleHistogram;
   /* [한국어] 사이클당 명령어 수 통계용 히스토그램(현재 변환 출력에는 사용되지 않음). */
};

// Used for local memory segments
struct localMemory
{
	int value;
	int entryIndex;
};

/*
 * [한국어]
 * cuobjdumpInstList — SASS → PTXPlus 변환을 위한 최상위 IR 컨테이너.
 *
 * 여러 개의 cuobjdumpEntry(엔트리)와 상수/전역/로컬 메모리 선언,
 * 텍스처 목록을 포함한다. 파서가 정보를 수집하면 printCuobjdumpPtxPlusList()
 * 에서 이를 순서대로 PTXPlus 텍스트로 출력한다.
 */
class cuobjdumpInstList
{
protected:
	std::list<cuobjdumpEntry> m_entryList;
	/* [한국어] 파싱된 모든 엔트리(커널/함수)의 리스트. */
	std::list<constMemory> m_constMemoryList;
	/* [한국어] 전역 상수 메모리 뱅크(constant0 등)의 리스트. */
	std::list<constMemory2> m_constMemoryList2;
	/* [한국어] 엔트리별 상수 메모리 뱅크(constant1<kernel> 등)의 리스트. */
	std::list<globalMemory> m_globalMemoryList;
	/* [한국어] 전역 메모리 심볼의 리스트. */

	int m_kernelCount;
	/* [한국어] 파싱된 커널(엔트리)의 수. ELF 파싱 후 설정된다. */
	std::map<std::string,int>kernelcmemmap;
	/* [한국어] 커널 이름 → 상수 메모리 뱅크 인덱스 매핑(현재 역변환용). */
	std::map<std::string,int>kernellmemmap;
	/* [한국어] 커널 이름 → 로컬 메모리 인덱스 매핑(주석 처리된 경로). */
	std::list<localMemory> m_localMemoryList;
	/* [한국어] 로컬 메모리 슬롯의 리스트. */
	std::list<std::string>  m_realTexList;	// Stores the real names of tex variables
	/* [한국어] 원본 PTX에서 추출한 실제 텍스처 변수 이름 목록.
	 * SASS의 $texN 형태를 실제 이름으로 대체할 때 사용한다. */
	std::list<constMemoryPtr> m_constMemoryPtrList;
	/* [한국어] .constptr 선언 목록. */
	int m_globalVarShndx; //records shndx value of global variables in the elf file SYMTAB
	/* [한국어] ELF SYMTAB에서 전역 변수가 속한 섹션 인덱스(shndx). */

	// Functions:
	std::string parseCuobjdumpPredicate(std::string pred);
	void printMemory();// Print const memory directives
	// Print register names
	void printRegNames(cuobjdumpEntry entry);
	void printOutOfBoundRegisters(cuobjdumpEntry entry);

	// Print predicate names
	void printPredNames(cuobjdumpEntry entry);
public:
	//Constructor
	cuobjdumpInstList();

	cuobjdumpInst getListEnd();

	// Functions used by the parser
	int addEntry(std::string entryName); // creates a new entry
	int add(cuobjdumpInst* newInst); //add cuobjdumpInst to list
	void addConstMemory(int index); // add global const memory
	void addTex(std::string tex);	// add tex operand
	bool findEntry(std::string entryName, cuobjdumpEntry& entry); // find and return entry

	void setKernelCount(int k);
	void readConstMemoryFromElfFile(std::string elf);
	void setLastEntryName(std::string entryName); // sets name of last entry
	void addCuobjdumpRegister(std::string reg, bool lo=false); //add register
	void addCuobjdumpMemoryOperand(std::string mem, int memType);
	std::string parseCuobjdumpRegister(std::string reg, bool lo, int vectorFlag);
	void addCuobjdumpDoublePredReg(std::string pred, std::string reg, bool lo=false);

	void addCubojdumpLabel(std::string label);

	void addEntryConstMemory(int index, int entryIndex);
	void addEntryConstMemory2(char* kernel);
	void setConstMemoryType(const char* type);
	void setConstMemoryType2(const char* type); // set type of constant memory
	void addConstMemoryValue(std::string constMemoryValue); // add const memory
	void addConstMemoryValue2(std::string constMemoryValue); // add const memory
	void addConstMemoryPtr(const char* bytes, const char* offset, const char* name);
	void setConstMemoryMap(const char* kernelname, int index);
	void setLocalMemoryMap(const char* kernelname, int index);
	void setglobalVarShndx(const char* shndx);
	int getglobalVarShndx();
	void addGlobalMemoryID(const char* bytes, const char* name);
	void updateGlobalMemoryID(const char* offset, const char* name);
	void reverseConstMemory();
	void addEntryLocalMemory(int value, int entryIndex);
	void readOtherConstMemoryFromBinFile(std::string binString); // read in constant memory from bin file
	std::list<std::string> getRealTexList(); // get the list of real tex names
	void setRealTexList(std::list<std::string> realTexList); // set the list of real tex names
	void printHeaderInstList();
	void printCuobjdumpLocalMemory();
	void printCuobjdumpInstList();
	/*
	 * [한국어]
	 * printCuobjdumpPtxPlusList - 수집된 IR을 PTXPlus 파일로 출력
	 *
	 * @headerInfo: 원본 PTX에서 추출한 헤더/엔트리 선언 정보를 담은 리스트.
	 *
	 *   1) 상수/로컬/전역 메모리 선언 출력
	 *   2) 각 엔트리를 역순으로 순회하며 .entry 헤더(원본 PTX 기반) 출력
	 *   3) 레지스터/predicate/특수 레지스터 선언 출력
	 *   4) 각 SASS 명령어를 PTXPlus로 변환하여 출력
	 *   5) l_exit: exit; 및 } 종료
	 */
	void printCuobjdumpPtxPlusList(cuobjdumpInstList* headerInfo);
};

#endif //_CUOBJDUMPINSTLIST_H_
