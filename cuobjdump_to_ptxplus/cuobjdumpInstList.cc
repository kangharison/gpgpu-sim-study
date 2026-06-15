/*
 * [한국어 설명] cuobjdump SASS/IR 명령어 리스트 구현 (cuobjdumpInstList.cc)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 ELF/SASS/PTX를 파싱한 후 생성되는 중간 표현(IR)의
 * 컨테이너 및 출력 변환을 담당하는 클래스들의 구현 파일이다.
 *   - constMemory/constMemory2 : 전역 및 엔트리별 상수 메모리 뱅크
 *   - constMemoryPtr           : 초기화되지 않은 상수 메모리 포인터
 *   - globalMemory             : ELF SYMTAB에서 추출한 전역 메모리 심볼
 *   - localMemory              : 엔트리별 로컬 메모리 슬롯
 *   - cuobjdumpEntry           : CUDA 커널(함수) 엔트리 하나
 *   - cuobjdumpInstList        : 엔트리/메모리/텍스처 등을 관리하고
 *                                최종 PTXPlus 출력
 * SASS→PTXPlus 변환 시 레지스터 인덱스 집계, 메모리 피연산자 문자열 처리,
 * predicate 인덱스 집계, 상수 메모리 선언 출력, .entry 헤더 매칭 등
 * 핵심 로직의 상당 부분이 이 파일에 들어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → elf_parse() / ptx_parse() / sass_parse()
 *         → cuobjdumpInstList (이 파일)에 IR 누적
 *             → printCuobjdumpPtxPlusList()
 *                 → cuobjdumpInst::printCuobjdumpPtxPlus() 호출
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h : 클래스/구조체 선언
 *   - cuobjdumpInst.h/.cc : 개별 명령어 IR 및 출력
 *   - sass.y, elf.y, header.y, ptx_parser.h : 파서가 이 파일의 메서드 호출
 * 이 파일에 의존하는 모듈:
 *   - cuobjdump_to_ptxplus.cc : 전역 g_instList/g_headerList 사용
 *
 * === 주요 함수/구조체 요약 ===
 * 메모리 관련:
 *   addConstMemory(), addConstMemoryValue(), setConstMemoryType(),
 *   addEntryConstMemory(), addConstMemoryPtr(), printMemory()
 *   addGlobalMemoryID(), updateGlobalMemoryID()
 *   addEntryLocalMemory(), printCuobjdumpLocalMemory()
 * 레지스터/predicate 관련:
 *   addCuobjdumpRegister(), parseCuobjdumpRegister(),
 *   addCuobjdumpDoublePredReg(), parseCuobjdumpPredicate(),
 *   printRegNames(), printPredNames(), printOutOfBoundRegisters()
 * 텍스처/엔트리 관련:
 *   addTex(), setRealTexList(), getRealTexList(), addEntry(), setLastEntryName()
 * 최종 출력:
 *   printHeaderInstList(), printCuobjdumpInstList(), printCuobjdumpPtxPlusList()
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

#include <sstream>
#include <iostream>
#include <cassert>
#include "cuobjdumpInstList.h"

/* [한국어] 디버그 출력용 매크로. P_DEBUG가 0이면 출력되지 않는다. */
#define P_DEBUG 0
#define DPRINTF(...) \
   if(P_DEBUG) { \
      printf("(%s:%u) ", __FILE__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
      fflush(stdout); \
   }

extern void output(const char * text);

//Constructor
/*
 * [한국어]
 * cuobjdumpInstList 생성자
 *
 * 별도의 초기화 코드는 없으며, 멤버 변수들은 클래스 내에서 빈 컨테이너로
 * 자동 초기화된다.
 */
cuobjdumpInstList::cuobjdumpInstList()
{
        //initilize everything to empty
}



// add to tex list
/*
 * [한국어]
 * addTex - 텍스처 이름을 실제 텍스처 리스트에 추가/매핑
 *
 * @tex: SASS에서 등장한 텍스처 피연산자(예: "$tex0" 또는 실제 이름)
 *
 * SASS는 종종 $texN 형태로 텍스처를 참조한다. 이 경우 원본 PTX에서
 * 추출한 m_realTexList의 N번째 항목으로 치환한다. 이미 실제 이름인
 * 경우에는 m_realTexList에 직접 추가한다.
 */
void cuobjdumpInstList::addTex(std::string tex)
{
	std::string origTex = tex;
	DPRINTF("cuobjdumpInstList::addTex tex=%s", tex.c_str());
	// If $tex# tex from cuobjdump, then use index to get real tex name
	if(tex.substr(0, 4) == "$tex") {
		tex = tex.substr(4, tex.size()-4);
		unsigned texNum = atoi(tex.c_str());
		if(texNum >= m_realTexList.size()) {
			output("ERROR: tex does not exist in real tex list from ptx.\n.");
			assert(0);
		}

		std::list<std::string>::iterator itex = m_realTexList.begin();
		for(unsigned i=0; i<texNum; i++) itex++;
		origTex = *itex;
	}
	// Otherwise, tex from original ptx
	else {
		m_realTexList.push_back(tex);
	}

	// Add the tex to instruction operand list
	//char* texName = new char [strlen(origTex.c_str())+1];
	//strcpy(texName, origTex.c_str());
	//getListEnd().addOperand(texName);
}

/*
 * [한국어]
 * setLastEntryName - 가장 최근에 추가된 엔트리의 이름을 변경
 *
 * @entryName: 새 엔트리 이름
 */
void cuobjdumpInstList::setLastEntryName(std::string entryName)
{
	m_entryList.back().m_entryName = entryName;
}

// create new global constant memory "bank"
/*
 * [한국어]
 * addConstMemory - 전역 상수 메모리 뱅크 하나를 추가
 *
 * @index: 상수 메모리 뱅크 인덱스
 *
 * entryIndex=0으로 설정하여 전역 상수 메모리(constant0 등)를 표현한다.
 */
void cuobjdumpInstList::addConstMemory(int index)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = 0;
	m_constMemoryList.push_back(newConstMem);
}

//add cuobjdumpInst to the last entry in entry list
/*
 * [한국어]
 * add - 새로 파싱한 SASS 명령어를 현재 엔트리에 추가
 *
 * @newCuobjdumpInst: 추가할 명령어 IR 객체 포인터
 * @return: 현재 엔트리 리스트의 크기
 *
 * 엔트리가 아직 없으면 이름이 ""인 더미 엔트리를 먼저 생성한 뒤 추가한다.
 */
int cuobjdumpInstList::add(cuobjdumpInst* newCuobjdumpInst)
{
	if(m_entryList.size() == 0) {
		//output("ERROR: Adding an instruction before entry.\n");
		addEntry("");
		//assert(0);
	}

	m_entryList.back().m_instList.push_back(*newCuobjdumpInst);

	return m_entryList.size();
}

// add a new entry
/*
 * [한국어]
 * addEntry - 새 CUDA 커널/함수(엔트리)를 생성
 *
 * @entryName: 엔트리 이름
 * @return: 생성 후 엔트리 리스트의 크기
 *
 * 레지스터/predicate/로컬 메모리 등의 집계값을 초기화하고,
 * m_opPerCycleHistogram에 기본 키를 미리 삽입한다.
 */
int cuobjdumpInstList::addEntry(std::string entryName)
{
	cuobjdumpEntry newEntry;
	newEntry.m_largestRegIndex = -1;
	newEntry.m_largestOfsRegIndex = -1;
	newEntry.m_largestPredIndex = -1;
	newEntry.m_reg124 = false;
	newEntry.m_oreg127 = false;
	newEntry.m_lMemSize = -1;

	newEntry.m_entryName = entryName;


   // Fill opPerCycle histogram with values
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_1", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_2", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_8", 0) );


	m_entryList.push_back(newEntry);
	return m_entryList.size();
}

// print out .version and .target headers
/*
 * [한국어]
 * printHeaderInstList - 원본 PTX 헤더(.version/.target/.tex)를 출력
 *
 * 첫 번째 엔트리에 저장된 헤더 관련 cuobjdumpInst들을 순회하며
 * printHeaderInst()를 호출하고, 실제 텍스처 목록을 .tex .u64 선언으로
 * 출력한다.
 */
void cuobjdumpInstList::printHeaderInstList()
{
	// These should be in the first entry
	cuobjdumpEntry e_first = m_entryList.front();

	std::list<cuobjdumpInst>::iterator currentInst;
	for(currentInst=e_first.m_instList.begin(); currentInst!=e_first.m_instList.end(); ++currentInst)
	{
		if(!(currentInst->printHeaderInst()))
		{
			break;
		}
	}
	for (	std::list<std::string>::iterator iter = m_realTexList.begin();
			iter != m_realTexList.end();
			iter ++) {
		output(".tex .u64 ");
		output((*iter).c_str());
		output(";\n");
	}
}

/*
 * [한국어]
 * findEntry - 주어진 이름의 엔트리를 검색
 *
 * @entryName: 검색할 엔트리 이름
 * @entry:     찾은 경우 복사본을 저장할 참조
 * @return:    찾으면 true, 아니면 false
 */
bool cuobjdumpInstList::findEntry(std::string entryName, cuobjdumpEntry& entry) {
	std::list<cuobjdumpEntry>::iterator e;

	std::string entryNameS = entryName;

	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if( e->m_entryName == entryNameS) {
			entry = *e;
			return true;
		}
	}

	return false;
}

// get the list of real tex names
/*
 * [한국어]
 * getRealTexList - 원본 PTX에서 추출한 텍스처 이름 목록을 반환
 */
std::list<std::string> cuobjdumpInstList::getRealTexList() {
	return m_realTexList;
}

// set the list of real tex names
/*
 * [한국어]
 * setRealTexList - 원본 PTX에서 추출한 텍스처 이름 목록을 설정
 *
 * @realTexList: 설정할 텍스처 이름 목록
 */
void cuobjdumpInstList::setRealTexList(std::list<std::string> realTexList) {
	m_realTexList = realTexList;
}

// add value to const memory
/*
 * [한국어]
 * addConstMemoryValue - 가장 최근 전역 상수 메모리 뱅크에 값 추가
 *
 * @constMemoryValue: 16진수 상수값 문자열
 */
void cuobjdumpInstList::addConstMemoryValue(std::string constMemoryValue)
{
	m_constMemoryList.back().m_constMemory.push_back(constMemoryValue);
}

/*
 * [한국어]
 * addConstMemoryValue2 - 가장 최근 엔트리별 상수 메모리 뱅크에 값 추가
 */
void cuobjdumpInstList::addConstMemoryValue2(std::string constMemoryValue)
{
	m_constMemoryList2.back().m_constMemory.push_back(constMemoryValue);
}

// set type of constant memory
/*
 * [한국어]
 * setConstMemoryType - 가장 최근 전역 상수 메모리 뱅크의 타입 설정
 */
void cuobjdumpInstList::setConstMemoryType(const char* type)
{
	m_constMemoryList.back().type = type;
}

/*
 * [한국어]
 * setConstMemoryType2 - 가장 최근 엔트리별 상수 메모리 뱅크의 타입 설정
 */
void cuobjdumpInstList::setConstMemoryType2(const char* type)
{
	m_constMemoryList2.back().type = type;
}

//retrieve point to list end
/*
 * [한국어]
 * getListEnd - 현재 엔트리의 마지막 명령어를 값으로 반환
 *
 * Bison action에서 마지막 명령어의 modifier/operand를 수정할 때 사용.
 * 주의: 값 반환이므로 반환된 객체의 변경이 원본에 영향을 주지 않는다.
 */
cuobjdumpInst cuobjdumpInstList::getListEnd()
{
	return m_entryList.back().m_instList.back();
}

// print out predicate names
/*
 * [한국어]
 * printPredNames - 현재 엔트리에 필요한 predicate 레지스터 선언 출력
 *
 * @entry: 출력할 엔트리
 *
 * 사용된 predicate 중 최대 인덱스를 기반으로 .reg .pred $p<N>; 를 출력.
 * GT200/Fermi에서는 최소 4개의 predicate를 선언한다.
 */
void cuobjdumpInstList::printPredNames(cuobjdumpEntry entry)
{
	if( entry.m_largestPredIndex >= 0) {
		char out[30];
		// there is at least 4 predicates for GT200, possibly more in Fermi 
		sprintf(out, "\t.reg .pred $p<%d>;", std::max(entry.m_largestPredIndex+1, 4)); 
		output(out);
		output("\n");
	}

}

// print reg124 and set its value to 0
/*
 * [한국어]
 * printOutOfBoundRegisters - 특수 레지스터 R124 및 o127 선언 출력
 *
 * @entry: 출력할 엔트리
 *
 * SASS에서 R124(범위를 벗어난 레지스터)나 o[0x7f]가 사용된 경우
 * 별도로 .reg .u32 $r124; / .reg .u32 $o127; 를 선언한다.
 */
void cuobjdumpInstList::printOutOfBoundRegisters(cuobjdumpEntry entry)
{
	if( entry.m_reg124 == true ) {
		output("\n");
		output("\t.reg .u32 $r124;\n");
	//	output("\tmov.u32 $r124, 0x00000000;\n");
	}
	if( entry.m_oreg127 == true) {
		output("\n");
		output("\t.reg .u32 $o127;\n");
	}
}

// print out register names
/*
 * [한국어]
 * printRegNames - 현재 엔트리에 필요한 일반/offset 레지스터 선언 출력
 *
 * @entry: 출력할 엔트리
 *
 * 사용된 최대 R# 인덱스와 A# 인덱스를 기반으로
 * .reg .u32 $r<(max+1)>; 및 .reg .u32 $ofs<(max+1)>; 를 출력한다.
 */
void cuobjdumpInstList::printRegNames(cuobjdumpEntry entry)
{
	if( entry.m_largestRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $r<%d>;", entry.m_largestRegIndex+1);
		output(out);
		output("\n");
	}

	if( entry.m_largestOfsRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $ofs<%d>;", entry.m_largestOfsRegIndex+1);
		output(out);
		output("\n");
	}
}

// print const memory directive
/*
 * [한국어]
 * printMemory - 상수 메모리/로컬 메모리/전역 메모리/constptr 선언 출력
 *
 * IR에 누적된 메모리 관련 정보를 PTXPlus 지시어 형태로 출력한다.
 *   - constMemory / constMemory2 → .const ... = { ... };
 *   - cuobjdumpEntry::m_lMemSize → .local .b8 l<index>[size];
 *   - globalMemory              → .global .b8 <name>[bytes];
 *   - constMemoryPtr            → .const .b8 <name>[bytes]; .constptr ...;
 */
void cuobjdumpInstList::printMemory()
{

	// Constant memory

	for(std::list<constMemory>::iterator i=m_constMemoryList.begin(); i!=m_constMemoryList.end(); ++i) {
		char line[40];

		// Global or entry specific
		if(i->entryIndex == 0)
			sprintf(line, ".const %s constant0[%d] = {", i->type, (int)i->m_constMemory.size());
		else
			sprintf(line, ".const %s ce%dc%d[%d] = {", i->type, i->entryIndex, i->index, (int)i->m_constMemory.size());

		output(line);

		std::list<std::string>::iterator j;
		int l=0;
		for(j=i->m_constMemory.begin(); j!=i->m_constMemory.end(); ++j) {
			if(j!=i->m_constMemory.begin())
				output(", ");
			if( (l++ % 4) == 0) output("\n          ");
			output(j->c_str());
		}
		output("\n};\n\n");
	}


	for(std::list<constMemory2>::iterator i=m_constMemoryList2.begin(); i!=m_constMemoryList2.end(); ++i) {
		char line[1024];

		// Global or entry specific
		sprintf(line, ".const %s constant1%s[%d] = {", i->type, i->kernel, (int)i->m_constMemory.size());

		output(line);

		std::list<std::string>::iterator j;
		int l=0;
		for(j=i->m_constMemory.begin(); j!=i->m_constMemory.end(); ++j) {
			if(j!=i->m_constMemory.begin())
				output(", ");
			if( (l++ % 4) == 0) output("\n          ");
			output(j->c_str());
		}
		output("\n};\n\n");
	}

	// Next, print out the local memory declaration
	std::list<cuobjdumpEntry>::iterator e;
	int eIndex=1; // entry index starts from 1 from the first blank entry is missing here (only in header entry list)
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if(e->m_lMemSize > 0) {
			std::stringstream ssout;
			ssout << ".local .b8 l" << eIndex << "[" << e->m_lMemSize << "];" << std::endl;
			output(ssout.str().c_str());
		}
		eIndex++;
	}
	output("\n");

	// Next, print out the global memory declaration
	std::list<globalMemory>::iterator g;
	for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
		std::stringstream out;
		out << ".global .b8 " << g->name << "[" << g->bytes << "];" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

	// Next, print out constant memory pointers
	std::list<constMemoryPtr>::iterator cp;
	for(cp=m_constMemoryPtrList.begin(); cp!=m_constMemoryPtrList.end(); ++cp) {
		std::stringstream out;
		out << ".const .b8 " << cp->name << "[" << cp->bytes << "];" << std::endl;
		out << ".constptr " << cp->name << ", " << cp->destination << ", " << cp->offset << ";" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

}


//TODO: Some register processing work is supposed to be done here.
/*
 * [한국어]
 * addCuobjdumpRegister - SASS 레지스터를 현재 명령어의 피연산자로 추가
 *
 * @reg: 레지스터 문자열 (예: "R0", "R1L", "A0", "SR_TidX" 등)
 * @lo:  true이면 lo/hi 레지스터(R#L/R#H)
 *
 * 벡터 연산(64bit/128bit) 여부를 판단한 뒤 parseCuobjdumpRegister()로
 * PTXPlus 형식의 문자열을 얻어 현재 명령어에 추가한다.
 */
void cuobjdumpInstList::addCuobjdumpRegister(std::string reg, bool lo)
{
	int vectorFlag = 0;
	char * regString;
	regString = new char [reg.size()+1];

	std::list<std::string>* typeModifiers = getListEnd().getTypeModifiers();
	std::string baseInst = getListEnd().getBase();

	//TODO: support for 64bit vectors and 128bit vectors
	if((baseInst == "DADD") || (baseInst == "DMUL") || (baseInst == "DFMA") ||
		((typeModifiers->size()==1) &&
		(typeModifiers->front() == ".S64") &&
		((baseInst == "G2R")||(baseInst == "R2G")||
		(baseInst == "GLD")||(baseInst == "GST")||
		(baseInst == "LST")|| (baseInst == "LLD"))))
	{
		vectorFlag = 64;
	}
	else if((typeModifiers->size()==1) && (typeModifiers->front() == ".S128"))
	{
		vectorFlag = 128;
	}

	//TODO: does the vector flag ever need to be set?
	std::string parsedReg = parseCuobjdumpRegister(reg, lo, vectorFlag);
	
	strcpy(regString, parsedReg.c_str());

	getListEnd().addOperand(regString);
}

// add memory operand
// memType: 0=constant, 1=shared, 2=global, 3=local
/*
 * [한국어]
 * addCuobjdumpMemoryOperand - SASS 메모리 피연산자를 현재 명령어에 추가
 *
 * @mem:     메모리 문자열(예: "g [0x10]", "global14 [...]", "c [0x0] [...]")
 * @memType: 메모리 공간 종류 (0=constant, 1=shared, 2=global, 3=local)
 *
 * constant 메모리의 경우 c [0xe]는 전역 메모리 심볼로, c [0x0]은
 * constant0, 그 외 c [0xN]은 constant1<entryName>으로 변환한다.
 * 공유/전역/로컬 메모리는 별도의 접두어 변환 없이 그대로 피연산자로
 * 추가되며, cuobjdumpInst::printCuobjdumpOperand()에서 최종 PTXPlus
 * 메모리 표현으로 해석된다.
 */
void cuobjdumpInstList::addCuobjdumpMemoryOperand(std::string mem, int memType) {
	std::string origMem = mem;
	bool neg = false;

	// If constant memory type, add prefix for entry specific constant memory
	if(memType == 0) {
		// Global memory c14
		// Replace this with the actual global memory name
		if(mem.substr(0,1) == "-") {
			//Remove minus sign if exists
			mem = mem.substr(1, mem.size()-1);
			neg = true;
		}

		if(mem.substr(0, 7) == "c [0xe]") {
			// Find the global memory identifier based on the offset provided
			int offset;
			sscanf(mem.substr(9,mem.size()-10).c_str(), "%x", &offset);
			// Find memory
			bool found = false;
			std::list<globalMemory>::iterator g;
			for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
				if(g->offset == offset) {
					mem = "varglobal" + g->name;
					found = true;
					break;
				}
			}

			if(!found) {
				printf("Could not find a global memory with this offset in: %s\n", mem.c_str());
				output("Could not find a global memory with this offset.\n");
				assert(0);
			}

		}
		else if(mem.substr(0, 7) == "c [0x0]"){
			mem = "constant0" + mem.substr(7, mem.length());
		}
		else if(mem.substr(0, 5) == "c [0x"){
			std::string out;
			out = "constant1" + m_entryList.back().m_entryName + mem.substr(8);
			mem = out.c_str();
		}
		else {
			output("Unrecognized memory type:");
			output(mem.c_str());
			output("\n");
			assert(0);
		}

		if (neg) {
			mem = "-"+mem;
		}
	}

	// Local memory
	/*
	if(memType == 3) {
		std::stringstream out;
		printf("Trying to find lmem for: %s\n", m_entryList.back().m_entryName.c_str());
		printf("Original memory: %s\n", mem.c_str());
		assert(kernellmemmap[m_entryList.back().m_entryName] !=0 );
		out << "l" << kernellmemmap[m_entryList.back().m_entryName];// << mem;
		mem = out.str();
	}
	*/
	// Add the memory operand to instruction operand list
	char* memName = new char [strlen(mem.c_str())+1];
	strcpy(memName, mem.c_str());
	getListEnd().addOperand(memName);
}

// increment register list and parse register
/*
 * [한국어]
 * parseCuobjdumpRegister - SASS 레지스터 문자열을 PTXPlus 형식으로 변환
 *
 * @reg:       SASS 레지스터 문자열
 * @lo:        lo/hi 레지스터 여부
 * @vectorFlag: 64 또는 128bit 벡터 연산 여부(사용된 최대 인덱스 보정용)
 * @return:    변환된 레지스터 문자열
 *
 * R# → 그대로 (cuobjdumpInst에서 "$r" 접두로 출력)
 * A# → offset 레지스터로 인덱스 집계
 * "o [0x7f]" → o127 사용 플래그 설정
 * SR_Tid* → %%tid(.x) 등 특수 레지스터로 변환
 * 동시에 엔트리의 m_largestRegIndex, m_largestOfsRegIndex 등을 갱신한다.
 */
std::string cuobjdumpInstList::parseCuobjdumpRegister(std::string reg, bool lo, int vectorFlag)
{
	std::string origReg = reg;
	// Make sure entry list is not empty
	if(m_entryList.size() == 0) {
		output("ERROR: Adding a register before adding an entry.\n");
		assert(0);
	}

	// remove minus sign if exists
	if(reg.substr(0,1) == "-")
		reg = reg.substr(1, reg.size()-1);

	// if lo or hi register, get register name only (remove 'H' or 'L')
	if(lo)
		reg = reg.substr(0, reg.size()-1);

	// Increase register number if needed
	// Two types of registers, R# or A#
	if(reg.substr(0, 1) == "R") {
		reg = reg.substr(1, reg.size()-1);
		int regNum = atoi(reg.c_str());

		// Remove register overlap at 64
		// TODO: is this still needed?
		/*if(regNum > 63 && regNum < 124) {
			regNum -= 64;
			// Fix the origReg string
			std::stringstream out;
			out << ((origReg.substr(0,1)=="-") ? "-" : "")
			    << "$r" << regNum
			    << (lo ? origReg.substr(origReg.size()-3, 3) : "");
			origReg = out.str();
		}*/

		if(vectorFlag==64)
			regNum += 1;
		if(vectorFlag==128)
			regNum += 3;

		if( m_entryList.back().m_largestRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestRegIndex = regNum;
		else if( regNum == 124 )
			m_entryList.back().m_reg124 = true;
	} else if(reg.substr(0, 1) == "A") {
		reg = reg.substr(1, reg.size()-1);
		int regNum = atoi(reg.c_str());

		if( m_entryList.back().m_largestOfsRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestOfsRegIndex = regNum;
	} else if(reg == "o [0x7f]") {
		m_entryList.back().m_oreg127 = true;
	} else if (reg.substr(0,3) == "SR_") {
		if(reg.substr(3,3)=="Tid") {
			origReg = "%%tid";
			if(reg.substr(7,1)=="X") {
				origReg += ".x";
			}
		}
	} else {
		output("ERROR: unknown register type.\n");
		printf("\nERROR: unknown register type: ");
		printf("%s",reg.c_str());
		printf("\n");
		assert(0);
	}
	return origReg;
}

// pred|reg double operand
/*
 * [한국어]
 * addCuobjdumpDoublePredReg - predicate와 레지스터를 하나의 피연산자로 추가
 *
 * @pred: predicate 문자열(예: ".C0")
 * @reg:  레지스터 문자열
 * @lo:   lo/hi 레지스터 여부
 *
 * DSET/FSET/ISET 연산의 경우 "pred/reg" 형태로, 그 외에는 "pred|reg"
 * 형태로 합쳐서 피연산자로 추가한다.
 */
void cuobjdumpInstList::addCuobjdumpDoublePredReg(std::string pred, std::string reg, bool lo)
{
	std::string parsedPred = parseCuobjdumpPredicate(pred);
	std::string parsedReg = parseCuobjdumpRegister(reg, lo, 0);

	std::string doublePredReg;
	if(
		getListEnd().getBase() == "DSET" ||
		getListEnd().getBase() == "FSET" ||
		getListEnd().getBase() == "ISET"
	)
		doublePredReg = parsedPred + "/" + parsedReg;
	else
		doublePredReg = parsedPred + "|" + parsedReg;

	char* doublePredRegName = new char [strlen(doublePredReg.c_str())+1];
	strcpy(doublePredRegName, doublePredReg.c_str());
	doublePredRegName[strlen(doublePredReg.c_str())] = '\0';
	getListEnd().addOperand(doublePredRegName);
}

/*
 * [한국어]
 * parseCuobjdumpPredicate - SASS predicate 문자열을 파싱하고 인덱스 집계
 *
 * @pred: predicate 문자열(예: "C0")
 * @return: 원본 predicate 문자열
 *
 * "C" 접두어를 제외한 숫자를 읽어 m_largestPredIndex를 갱신한다.
 */
std::string cuobjdumpInstList::parseCuobjdumpPredicate(std::string pred)
{
	std::string origPred = pred;

	// Make sure entry list is not empty
	if(m_entryList.size() == 0) {
		output("ERROR: Adding a predicate before adding an entry.\n");
		assert(0);
	}

	// increase predicate numbers if needed
	pred = pred.substr(2, pred.size()-2);
	int predNum = atoi(pred.c_str());
	if( m_entryList.back().m_largestPredIndex < predNum )
		m_entryList.back().m_largestPredIndex = predNum;

	return origPred;
}

/*
 * [한국어]
 * addCubojdumpLabel - 현재 엔트리의 레이블 목록에 레이블 추가
 *
 * @label: 추가할 레이블
 *
 * 중복 추가를 방지하기 위해 이미 목록에 있는 경우 무시한다.
 */
void cuobjdumpInstList::addCubojdumpLabel(std::string label)
{

	if(!(m_entryList.back().m_labelList.empty()))
	{
		std::list<std::string>::iterator labelIterator;

		for( labelIterator=m_entryList.back().m_labelList.begin(); labelIterator!=m_entryList.back().m_labelList.end(); labelIterator++ )
		{
			if(label.compare(*labelIterator) == 0)
				return;
		}
	}

	m_entryList.back().m_labelList.push_back(label);
}

/*
 * [한국어]
 * setConstMemoryMap - 커널 이름과 상수 메모리 뱅크 인덱스 매핑
 *
 * @kernelname: cuobjdump 출력에서 추출한 커널 이름 문자열
 * @index:      상수 메모리 뱅크 인덱스
 *
 * 문자열에서 탭 이전의 실제 커널 이름을 추출하여 kernelcmemmap에 저장.
 */
void cuobjdumpInstList::setConstMemoryMap(const char* kernelname, int index){
	std::string kernel = kernelname;
	kernel = kernel.substr(14, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	kernelcmemmap[kernel] = index;
}

/*
 * [한국어]
 * setLocalMemoryMap - 커널 이름과 로컬 메모리 인덱스 매핑(현재 미사용)
 */
void cuobjdumpInstList::setLocalMemoryMap(const char* kernelname, int index){
	std::string kernel = kernelname;
	kernel = kernel.substr(10, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	kernellmemmap[kernel] = index;
}

/*
 * [한국어]
 * setglobalVarShndx - 전역 변수가 속한 ELF 섹션 인덱스(shndx) 설정
 */
void cuobjdumpInstList::setglobalVarShndx(const char* shndx){
	m_globalVarShndx = atoi(shndx);
}

/*
 * [한국어]
 * getglobalVarShndx - 설정된 전역 변수 섹션 인덱스 반환
 */
int cuobjdumpInstList::getglobalVarShndx(){
	return m_globalVarShndx;
}

/*
 * [한국어]
 * addGlobalMemoryID - 전역 메모리 심볼 초기 정보 추가
 *
 * @bytes: 심볼의 바이트 크기
 * @name:  심볼 이름
 *
 * 오프셋은 이후 updateGlobalMemoryID()에서 설정된다.
 */
void cuobjdumpInstList::addGlobalMemoryID(const char* bytes, const char* name){
	globalMemory globalMemID;
	//globalMemID.offset = atoi(index)/4;
	globalMemID.bytes = atoi(bytes);
	globalMemID.name = name;

	m_globalMemoryList.push_back(globalMemID);
}

/*
 * [한국어]
 * updateGlobalMemoryID - ELF 재배치 정보로 전역 메모리 오프셋 갱신
 *
 * @offset: ELF 재배치 오프셋(바이트)
 * @name:   갱신할 심볼 이름
 *
 * 이름이 일치하는 globalMemory의 offset을 offset/4(워드 단위)로 설정.
 */
void cuobjdumpInstList::updateGlobalMemoryID(const char* offset, const char* name){
	bool found = false;
	std::list<globalMemory>::iterator g;
	for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
		if(g->name.compare(name) == 0) {
			g->offset = atoi(offset)/4;
			found = true;
			break;
		}
	}

	if(!found) {
		printf("Could not find a global memory with this offset in: %s\n", name);
		output("Could not find a global memory with this offset.\n");
		assert(0);
	}
}

//NOT USED
/*
 * [한국어]
 * reverseConstMemory - 상수 메모리 매핑의 순서를 뒤집는다(현재 미사용)
 */
void cuobjdumpInstList::reverseConstMemory() {
	int total = kernelcmemmap.size();
	for (	std::map<std::string,int>::iterator iter = kernelcmemmap.begin();
			iter != kernelcmemmap.end();
			iter++){
		(*iter).second = total - (*iter).second;
	}
}


// create new entry specific constant memory "bank"
/*
 * [한국어]
 * addEntryConstMemory - 엔트리별 상수 메모리 뱅크 추가
 *
 * @index:      뱅크 인덱스
 * @entryIndex: 엔트리 인덱스
 */
void cuobjdumpInstList::addEntryConstMemory(int index, int entryIndex)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = entryIndex;
	m_constMemoryList.push_back(newConstMem);
}

/*
 * [한국어]
 * addEntryConstMemory2 - 엔트리별 두 번째 상수 메모리 뱅크 추가
 *
 * @kernelname: "c [0xN]<kernelname>" 형태의 커널 이름 문자열
 */
void cuobjdumpInstList::addEntryConstMemory2(char* kernelname)
{
	std::string kernel = kernelname;
	kernel = kernel.substr(14, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	constMemory2 newConstMem2;
	newConstMem2.kernel = strdup(kernel.c_str());
	m_constMemoryList2.push_back(newConstMem2);
}

/*
 * [한국어]
 * addEntryLocalMemory - 엔트리별 로컬 메모리 슬롯 추가
 *
 * @value:      로컬 메모리 크기
 * @entryIndex: 엔트리 인덱스
 */
void cuobjdumpInstList::addEntryLocalMemory(int value, int entryIndex)
{
	localMemory newLocalMem;
	newLocalMem.value = value;
	newLocalMem.entryIndex = entryIndex;
	m_localMemoryList.push_back(newLocalMem);
}

/*
 * [한국어]
 * setKernelCount - 파싱된 커널(엔트리) 수 설정
 */
void cuobjdumpInstList::setKernelCount(int k){
	m_kernelCount = k;
}

/*
 * [한국어]
 * printCuobjdumpInstList - 디버그용: 모든 명령어를 원형 SASS 형태로 출력
 */
void cuobjdumpInstList::printCuobjdumpInstList()
{
	// Each entry
	std::list<cuobjdumpEntry>::iterator e;
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {




		for(	std::list<cuobjdumpInst>::iterator currentInst=e->m_instList.begin();
				currentInst!=e->m_instList.end();
				++currentInst) {
			// Output the instruction
			output("\t");
			currentInst->printCuobjdumpInst();
			output("\n");
		}
	}
}

/*
 * [한국어]
 * printCuobjdumpLocalMemory - 별도 로컬 메모리 슬롯 선언 출력
 */
void cuobjdumpInstList::printCuobjdumpLocalMemory()
{
	for(	std::list<localMemory>::iterator i=m_localMemoryList.begin();
			i!=m_localMemoryList.end();
			++i) {
		char line[40];
		//if(i->value > 0)
		{
			sprintf(line, ".local .b8 l%d[%d];\n", i->entryIndex, i->value);
			output(line);
		}
	}
}

/*
 * [한국어]
 * printCuobjdumpPtxPlusList - 수집된 IR을 PTXPlus 파일로 출력
 *
 * @headerInfo: 원본 PTX에서 추출한 헤더/엔트리 선언 정보를 담은 리스트.
 *
 *   1) 상수/로컬/전역 메모리 선언 출력(printMemory, printCuobjdumpLocalMemory)
 *   2) 각 엔트리를 역순으로 순회
 *   3) headerInfo에서 동일한 이름의 .entry/.func 선언을 찾아 출력
 *      없으면 __cuda_dummy_entry__ 특수 처리
 *   4) 레지스터/predicate/특수 레지스터 선언 출력
 *   5) 각 SASS 명령어를 cuobjdumpInst::printCuobjdumpPtxPlus()로 변환 출력
 *   6) l_exit: exit; 및 } 종료
 */
void cuobjdumpInstList::printCuobjdumpPtxPlusList(cuobjdumpInstList* headerInfo)
{
	output("\n");
	printMemory();
	printCuobjdumpLocalMemory();
	// Each entry
	std::list<cuobjdumpEntry>::reverse_iterator e;
	for(e=m_entryList.rbegin(); e!=m_entryList.rend(); ++e) {

		output("\n");

		// Output the header information for this entry using headerInfo
		// First, find the matching entry in headerInfo
		cuobjdumpEntry headerEntry;

		if( headerInfo->findEntry(e->m_entryName, headerEntry) ) {
			// Entry for current header found, print it out
			std::list<cuobjdumpInst>::iterator headerInstIter;
			for(headerInstIter=headerEntry.m_instList.begin();
				headerInstIter!=headerEntry.m_instList.end();
				++headerInstIter) {
				if(headerInstIter!=headerEntry.m_instList.begin()) {
					output("\t");
				}
				headerInstIter->printHeaderPtx();
				output("\n");
			}
			output("{\n");
		} else {
			// Couldn't find this entry in ptx file
			// Check if it is a dummy entry
			if(e->m_entryName == "__cuda_dummy_entry__") {
				output(".entry ");
				output("__cuda_dummy_entry__");
				output("\n");
				output("{\n");
			} else {
				output("Mismatch in entry names between cuobjdump output and original ptx file.\n");
				assert(0);
			}
		}
		assert( &*e != NULL);
		printRegNames(*e);
		printPredNames(*e);
		printOutOfBoundRegisters(*e);
		output("\n");

		for(std::list<cuobjdumpInst>::iterator currentInst=e->m_instList.begin(); currentInst!=e->m_instList.end(); ++currentInst){
			// Output the instruction
			//cuobjdumpInst* outputInst = &*currentInst;
			output("\t");
			//outputInst->printCuobjdumpPtxPlus(m_entryList.back().m_labelList);
			currentInst->printCuobjdumpPtxPlus(e->m_labelList, this->m_realTexList);
			output("\n");
		}
		output("\n\tl_exit: exit;\n");
		output("}\n");
	}
}

/*
 * [한국어]
 * addConstMemoryPtr - 초기화되지 않은 상수 메모리 포인터 추가
 *
 * @offset:     destination 내 바이트 오프셋
 * @size:       포인터 심볼의 바이트 크기
 * @name:       심볼 이름
 *
 * destination은 항상 "constant0"으로 설정된다.
 */
void cuobjdumpInstList::addConstMemoryPtr(const char* offset, const char* size, const char* name){
	constMemoryPtr ptr;
	ptr.offset = atoi(offset);
	ptr.bytes = atoi(size);
	ptr.name = name;
	ptr.destination = "constant0";
	m_constMemoryPtrList.push_back(ptr);
}
