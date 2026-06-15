/*
 * [한국어 설명] SASS → PTXPlus 단일 명령어 변환 구현 (cuobjdumpInst.cc)
 *
 * === 파일의 역할 ===
 * cuobjdumpInst 클래스의 구현 파일로, SASS 명령어 하나를 PTXPlus 형식으로
 * 변환하는 핵심 로직이 들어 있다. m_base(니모닉)에 따라 거대한 분기를 통해
 * 산술/논리/메모리/분기/변환/원子/특수 명령어들을 PTXPlus 문법으로 출력한다.
 * 레지스터, 메모리 위치, 즉시값, predicate, 레이블, 특수 레지스터 등의
 * 출력 형식을 조정하며, 타입/베이스 수정자도 함께 처리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   sass.y (SASS 파서)
 *     → cuobjdumpInstList::add()로 cuobjdumpInst 객체 추가
 *         → cuobjdumpInstList::printCuobjdumpPtxPlusList()
 *             → cuobjdumpInst::printCuobjdumpPtxPlus() [이 파일]
 *                 → output()로 .ptxplus 파일 기록
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInst.h : 클래스 선언
 *   - cuobjdumpInstList.h/.cc : IR 컨테이너, 엔트리/레지스터/메모리 관리
 * 이 파일에 의존하는 모듈:
 *   - cuobjdumpInstList.cc : printCuobjdumpPtxPlusList()에서 호출
 *
 * === 주요 함수/구조체 요약 ===
 * printCuobjdumpPtxPlus()      - m_base에 따른 PTXPlus 출력 분기
 * printCuobjdumpOperand()      - 레지스터/메모리/즉시값/특수 레지스터 출력
 * printCuobjdumpTypeModifiers() - .F32/.S32/.U64 등 타입 수정자 출력
 * printCuobjdumpBaseModifiers() - EQ/GE/.rz/.abs 등 베이스 수정자 출력
 * printCuobjdumpPredicate()    - @pN 형태의 predicate 출력
 * printCuobjdumpLabel()        - l0x... 레이블 출력
 * printCuobjdumpOperands()     - 피연산자 목록 순회 출력
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

// External includes
#include <sstream>
#include <iostream>
#include <cassert>
#include <string>

// Local includes
#include "cuobjdumpInst.h"

extern void output(const char * text);
extern void output(const std::string text);

//Constructor
/*
 * [한국어]
 * cuobjdumpInst 생성자
 *
 * 레이블, predicate, base, 수정자, 피연산자, predicate 수정자를 저장할
 * 동적 list 객체들을 생성하고 초기화한다.
 */
cuobjdumpInst::cuobjdumpInst() {
	//initilize everything to empty
	m_label = "";
	m_predicate = new std::list<std::string>();
	m_base = "";
	m_baseModifiers = new std::list<std::string>();
	m_typeModifiers = new std::list<std::string>();
	m_operands = new std::list<std::string>();
	m_predicateModifiers = new std::list<std::string>();
}

/*
 * [한국어]
 * cuobjdumpInst 소멸자
 *
 * 현재는 동적 할당된 list들을 해제하지 않는다(주석 처리됨).
 */
cuobjdumpInst::~cuobjdumpInst() {
	/*
	delete m_predicate;
	delete m_baseModifiers;
	delete m_typeModifiers;
	delete m_operands;
	delete m_predicateModifiers;
	*/
}

/*
 * [한국어]
 * printCuobjdumpInst - 디버그용: 명령어의 기본 정보를 stdout에 출력
 */
void cuobjdumpInst::printCuobjdumpInst()
{
	/*TODO: print label here*/
	/*TODO: print predicate here*/
	/*std::cout << "Instruction Base: " << m_base << "\n";

	std::cout << "Instruction Modifiers: ";
	m_baseModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operand types: ";
	m_typeModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operands: ";
	m_operands->printStringList();
	std::cout << "\n\n";*/

	std::cout << m_base << " ";
	cuobjdumpInst::printStringList(m_baseModifiers);
	std::cout << " ";
	cuobjdumpInst::printStringList(m_typeModifiers);
	std::cout << " ";
	cuobjdumpInst::printStringList(m_operands);
	std::cout << "\n";
}

//static
/*
 * [한국어]
 * printStringList - 디버그용: 문자열 리스트를 공백으로 구분해 출력
 */
void cuobjdumpInst::printStringList(std::list<std::string>* strlist) {
	for (	std::list<std::string>::iterator iter = strlist->begin();
			iter != strlist->end();
			iter++) {
		std::cout << *iter << " ";
	}
}

// Just prints the base and operands
/*
 * [한국어]
 * printHeaderPtx - 원본 PTX 헤더 항목(.version/.target/.tex 등)을 그대로 출력
 */
void cuobjdumpInst::printHeaderPtx()
{
	output(m_base);
	output(" ");
	for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
			basemod != m_baseModifiers->end();
			basemod++) {
		output(" ");
		output(*basemod);
	}

	for (	std::list<std::string>::iterator operand = m_operands->begin();
			operand != m_operands->end();
			operand++) {
		output(" ");
		output(*operand);
	}
}

//retreive instruction mnemonic
//retreive instruction mnemonic
/*
 * [한국어]
 * getBase - 이 명령어의 SASS 니모닉(base)을 반환
 */
const std::string cuobjdumpInst::getBase()
{
	return m_base;
}

/*
 * [한국어]
 * getTypeModifiers - 이 명령어의 타입 수정자 리스트 포인터를 반환
 */
std::list<std::string>* cuobjdumpInst::getTypeModifiers()
{
	return m_typeModifiers;
}

//print out .version and .target header lines
//print out .version and .target header lines
/*
 * [한국어]
 * printHeaderInst - .version/.target/.tex 헤더 지시어를 PTXPlus 형식으로 출력
 * @return true면 헤더 항목을 출력했음, false면 더 이상 헤더 항목이 아님
 */
bool cuobjdumpInst::printHeaderInst()
{
	if(m_base == ".version")
	{
		output(m_base);
		output(" ");

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand ++;

		if(operand != m_operands->end()) {
			output(".");
			output(*operand);
		}
      output("+");
		output("\n");
	}
	else if(m_base ==  ".target")
	{
		output(m_base);
		output(" ");

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand ++;

		while(operand != m_operands->end()) {
			output(", ");
			output(*operand);
			operand++;
		}
		output("\n");
	}
	else if(m_base == ".tex")
	{
		output(m_base); output(" ");

		std::list<std::string>::iterator curr = m_baseModifiers->begin();
		output(*curr);
		output(" ");
		curr++;

		while(curr != m_baseModifiers->end())
		{
			output(" ");
			output(*curr);
		}

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand++;

		while(operand != m_operands->end()) {
			output(", ");
			output(*operand);
			operand++;
		}
		output(";\n");
	}
	else
	{
		return false;
	}
	return true;
}

/*
 * [한국어]
 * setBase - SASS 명령어 니모닉(base)을 설정
 */
void cuobjdumpInst::setBase(const char* setBaseValue)
{
	m_base = setBaseValue;
}

/*
 * [한국어]
 * addBaseModifier - 베이스 수정자(EQ, GE, .rz, .abs 등)를 추가
 */
void cuobjdumpInst::addBaseModifier(const char* addBaseMod)
{
	m_baseModifiers->push_back(addBaseMod);
}

/*
 * [한국어]
 * addTypeModifier - 타입 수정자(.F32, .S32, .U64 등)를 추가
 *
 * 메모리 피연산자에서 두 개 이상의 타입 수정자가 나올 경우
 * 마지막 수정자를 대체하여 최대 2개를 유지한다.
 */
void cuobjdumpInst::addTypeModifier(const char* addTypeMod)
{
	//We cannot have more than two modifiers, replace the last
	//This will be the case if we have memory operand modifiers
	if (m_typeModifiers->size() == 2){
		m_typeModifiers->pop_back();
	}
	m_typeModifiers->push_back(addTypeMod);
}

/*
 * [한국어]
 * addOperand - 변환된 피연산자 문자열을 추가
 */
void cuobjdumpInst::addOperand(const char* addOp)
{
	m_operands->push_back(addOp);
}

/*
 * [한국어]
 * setPredicate - 이 명령어의 실행 predicate(C0/C1 등)를 설정
 */
void cuobjdumpInst::setPredicate(const char* setPredicateValue)
{
	m_predicate->push_back(setPredicateValue);
}

/*
 * [한국어]
 * addPredicateModifier - predicate 수정자(.sign/.not_sign/.carry 등)를 추가
 */
void cuobjdumpInst::addPredicateModifier(const char* addPredicateMod)
{
	m_predicateModifiers->push_back(addPredicateMod);
}

/*
 * [한국어]
 * setLabel - 이 명령어에 붙은 레이블(l0x...)을 설정
 */
void cuobjdumpInst::setLabel(const char* setLabelValue)
{
	m_label = setLabelValue;
}

/*
 * [한국어]
 * checkCubojdumpLabel - 주어진 레이블이 현재 엔트리의 레이블 목록에 있는지 확인
 */
bool cuobjdumpInst::checkCubojdumpLabel(std::list<std::string> labelList, std::string label)
{
	if(labelList.empty())
		return false;

	std::list<std::string>::iterator labelIterator;

	for( labelIterator=labelList.begin(); labelIterator!=labelList.end(); labelIterator++ )
	{
		if(label.compare(*labelIterator) == 0)
			return true;
	}

	return false;
}

/*
 * [한국어]
 * printCuobjdumpLabel - 유효한 레이블이 있으면 "l0x...: " 형태로 출력
 */
void cuobjdumpInst::printCuobjdumpLabel(std::list<std::string> labelList)
{
	if((m_label != "")&&(checkCubojdumpLabel(labelList, m_label))) {
		output(m_label);
		output(": ");
	}
}

/*
 * [한국어]
 * printCuobjdumpPredicate - predicate를 "@pN<mod> " 형태로 출력
 *
 * .not_sign, .sign, .carry, .false 등의 수정자는 각각 .nsf, .sf, .cf, .false로
 * 변환되어 출력된다.
 */
void cuobjdumpInst::printCuobjdumpPredicate()
{
	std::list<std::string>::iterator pred = m_predicate->begin();
	if(pred != m_predicate->end())
	{
		output("@$p");
		output((*pred).substr(1,1));
		for (	std::list<std::string>::iterator predmod = m_predicateModifiers->begin();
				predmod != m_predicateModifiers->end();
				predmod++) {
			std::string modString3 = *predmod;

			for(unsigned i=0; i<modString3.length(); i++)
			{
				modString3[i] = tolower(modString3[i]);
			}
			if(modString3 ==".not_sign") {
				output(".nsf");
			} else if(modString3 == ".sign") {
				output(".sf");
			} else if(modString3 == ".carry") {
				output(".cf");
			} else if(modString3 == ".false") {
				output(".false"); //TODO: Need to find out what this is.
			} else {
				output(modString3);
			}
		}
		output(" ");
	}
}

/*
 * [한국어]
 * printCuobjdumpTypeModifiers - SASS 타입 수정자를 PTXPlus 타입 접미사로 출력
 *
 * .F16/.F32/.F64/.S8/.S16/.S32/.S64/.S128/.U8/.U16/.U32/.U64/.HI 등을
 * PTXPlus에서 사용하는 소문자 형태로 변환한다. 64bit/128bit 정수는
 * 현재 .bb64/.bb128로 매핑된다.
 */
void cuobjdumpInst::printCuobjdumpTypeModifiers()
{
	for (	std::list<std::string>::iterator typemod = m_typeModifiers->begin();
			typemod != m_typeModifiers->end();
			typemod++) {
		if (*typemod ==  ".F16")
			output(".f16");
		else if(*typemod == ".F32")
			output(".f32");
		else if(*typemod == ".F64"){
			if(		m_base == "F2I"||
					m_base == "F2F")
				output(".f64");
			else
				output(".ff64");
		}
		else if(*typemod == ".S8")
			output(".s8");
		else if(*typemod == ".S16")
			output(".s16");
		else if(*typemod == ".S32")
			output(".s32");
		else if(*typemod == ".S64")
			output(".bb64"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".S128")
			output(".bb128"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".U8")
			output(".u8");
		else if(*typemod == ".U16")
			output(".u16");
		else if(*typemod == ".U32")
			output(".u32");
		else if(*typemod == ".U64")
			output(".bb64"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".HI")
			output(".hi");
		else
		{
			printf("Unknown Type: ");
			printf("%s",(*typemod).c_str());
			printf("\n");
			output("Unknown Type: ");
			output(*typemod);
			assert(0);
		}
	}
}

/*
 * [한국어]
 * printCuobjdumpBaseModifiers - SASS 베이스 수정자를 PTXPlus 수정자로 출력
 *
 * 비교 연산자(EQ/EQU/GE/...), .abs, 반올림 모드(.rz/.rp/.rm), 
 * .any/.all, GRED/GATOM의 납부 연산 등을 처리한다.
 */
void cuobjdumpInst::printCuobjdumpBaseModifiers()
{
	for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
			basemod != m_baseModifiers->end();
			basemod++)
	{
		if( *basemod ==  "EQ")
			output(".eq");
		else if( *basemod == "EQU")
			output(".equ");
		else if( *basemod == "GE")
			output(".ge");
		else if( *basemod == "GEU")
			output(".geu");
		else if( *basemod == "GT")
			output(".gt");
		else if( *basemod == "GTU")
			output(".gtu");
		else if( *basemod == "LE")
			output(".le");
		else if( *basemod == "LEU")
			output(".leu");
		else if( *basemod == "LT")
			output(".lt");
		else if( *basemod == "LTU")
			output(".ltu");
		else if( *basemod == "NE")
			output(".ne");
		else if( *basemod == "NEU")
			output(".neu");
		else if( *basemod == ".abs")
		{
			if((m_base != "F2F") && (m_base != "I2I"))
			{
				output(*basemod);
			}
		}
		else if(	(*basemod == "ex2") ||
					(*basemod == ".exit") ||
					(*basemod == "sin") ||
					(*basemod == "cos") ||
					(*basemod == ".rz") ||
					(*basemod == ".rp") ||
					(*basemod == ".rm") ||
					(*basemod == ".any") ||
					(*basemod == ".all") )
				output(*basemod);
		else if( *basemod == ".bext")
		{
			//".bext" is a modifier that indicated u8 to u16 type conversion, I think
		}
		else if( *basemod == ".s")
		{
			//".s" is the same as ".join" in cuobjdump.
		}
		else if( *basemod == ".sfu")
		{
			//".sfu" is an unknown base modifier, TODO: find out what it is
		}
		else if( *basemod == ".x")
		{
			//".x" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if( *basemod == ".e")
		{
			//".e" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if( *basemod == ".ir")
		{
			//".ir" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if((*basemod == "IADD") ||
				(*basemod == "IMIN") ||
				(*basemod == "IMAX"))
		{
			/*
			 * This is the case of a GRED or GATOM operation
			 */
			output(".");
			std::string modstr = *basemod;
			modstr = modstr.substr(1);
			for (unsigned i=0; i<modstr.length(); i++){
				modstr[i] = tolower(modstr[i]);
			}
			output(modstr.c_str());
		}
		else
		{
			printf("Unknown Base Mod: ");
			printf("%s",(*basemod).c_str());
			printf("\n");
			output("Unknown Base Mod: ");
			output(*basemod);
			assert(0);
		}
	}
}

/*
 * Remove the trailing 'l' or 'h' and output the operand followed by ".lo" or ".hi" respectively
 */
/*
 * [한국어]
 * printCuobjdumpOperandlohi - 끝이 'l' 또는 'h'인 레지스터를 .lo/.hi 접미사로 출력
 *
 * 예: "R0l" → "R0.lo", "R0h" → "R0.hi"
 */
void cuobjdumpInst::printCuobjdumpOperandlohi(std::string op) {
	if (op.substr(op.length()-1) == "l") {
		output(op.substr(0, op.length()-1).c_str());
		output(".lo");
	} else if (op.substr(op.length()-1) == "h"){
		output(op.substr(0, op.length()-1).c_str());
		output(".hi");
	} else {
		output(op.c_str());
	}
}

/*
 * [한국어]
 * printCuobjdumpOperand - 하나의 피연산자를 PTXPlus 형식으로 출력
 *
 * @currentPiece:       출력할 피연산자 원본 문자열
 * @operandDelimiter:   앞에 붙을 구분자(주로 ",")
 * @base:               현재 명령어의 base(메모리 피연산자 처리 시 사용)
 *
 * 다음 종류를 처리한다:
 *   - 음수 즉시값: 2의 보수 형태로 변환
 *   - 특수 레지스터: %%ntid.x/y/z, %%nctaid.x/y, %%ctaid.x/y, %%clock
 *   - 일반/벡터 레지스터: $rN, {$rN,$rN+1} 등
 *   - offset 레지스터: $ofsN
 *   - 메모리 피연산자: g[...], s[...], l[...], constant0[...], constant1...,
 *                      varglobal<name>
 *   - 즉시값/레이블/전역 변수
 */
void cuobjdumpInst::printCuobjdumpOperand(std::string currentPiece, std::string operandDelimiter, std::string base)
{

	output(operandDelimiter);
	output(" ");

	//Current piece
	std::string currp = currentPiece;
	std::string mod;

	if(currp[0] == '-') {
		mod = currp.substr(1);

		if(mod.substr(0,2) == "0x") {
			unsigned immValue;
			std::stringstream hexStringConvert2;
			hexStringConvert2 << std::hex << mod;
			hexStringConvert2 >> immValue;

			if(immValue){
				immValue = ~immValue + 1;
			}

			std::stringstream outputhexstream;
			outputhexstream << std::hex << immValue;
			std::string outputhex;
			outputhex = outputhexstream.str();
			output("0x");
			for(unsigned i=8; i > outputhex.length(); i--)
			{
				output("0");
			}
			output(outputhex.c_str());
			return;
		}

		output("-");
	}
	else
	{
		mod = std::string(currp);
	}

	// Make it lower case
	if(mod.substr(0,9)!= "constant1" && mod.substr(0,9) != "varglobal")
	for(unsigned i=0; i<mod.length(); i++)
	{
		mod[i] = tolower(mod[i]);
	}

	//double destination
	if(mod[0]=='.') //double destination
	{
		std::string temp = mod.substr(2,2);
		output("$p");
		output(mod.substr(2,2).c_str());

		mod = mod.substr(4);
	}


	if(mod == "g [0x1].u16") { //handling special register case: %ntid.x
		output("%%ntid.x");
	} else if(mod == "g [0x2].u16") { //handling special register case: %ntid.y
		output("%%ntid.y");
	} else if(mod == "g [0x3].u16") { //handling special register case: %ntid.x
		output("%%ntid.z");
	} else if(mod =="g [0x4].u16") { //handling special register case: %nctaid.x
		output("%%nctaid.x");
	} else if(mod =="g [0x5].u16") { //handling special register case: %nctaid.y
		output("%%nctaid.y");
	} else if(mod == "g [0x6].u16") {//handling special register case: %ctaid.x
		output("%%ctaid.x");
	} else if(mod == "g [0x7].u16") {//handling special register case: %ctaid.y
		output("%%ctaid.y");
	} else if(mod == "sr1") {//handling special register case: %clock
		output("%%clock");
	} else if(mod[0]=='r') { //basic register
		if(	(m_base == "DADD") ||
			(m_base == "DMUL") ||
			(m_base == "DFMA") ||
			(	(m_typeModifiers->size()==1) &&
				(m_typeModifiers->front() == ".S64") &&
				(	(m_base == "G2R") ||
					(m_base == "R2G") ||
					(m_base == "GLD") ||
					(m_base == "GST") ||
					(m_base == "LST") ||
					(m_base == "LLD")))) {
			std::string modsub = mod.substr(1);
			int regNumInt = atoi(modsub.c_str());
			std::stringstream temp;
			temp << "{$r" << (regNumInt) << ",$r"<< (regNumInt+1) << "}";
			output(temp.str().c_str());
		} else if(	(m_typeModifiers->size()==1) &&
					(m_typeModifiers->front() == ".S128")) {
			std::string modsub = mod.substr(1);
			int regNumInt = atoi(modsub.c_str());
			std::stringstream temp;
			temp << "{$r" << (regNumInt);
			temp << ",$r" << (regNumInt+1);
			temp << ",$r" << (regNumInt+2);
			temp << ",$r" << (regNumInt+3) << "}";
			output(temp.str().c_str());
		} else {
			output("$");
			printCuobjdumpOperandlohi(mod);
		}
	} else if(mod[0] == 'c' && mod.length() == 2) { //predicate register (conditional code)
		output("$p");
		output(mod.substr(1,1).c_str());
	} else if(mod[0]=='a') {//offset register
		output("$ofs");
		mod = mod.substr(1);
		printCuobjdumpOperandlohi(mod);
	} else if(mod[0]=='o') {//output register
		output("$o127");
	} else if (	mod[0]=='g' ||
				mod.substr(0,2) == "lo" ||
				mod.substr(0,8) == "constant") {
		//memory operands, global14 = global, g = shared, c = constant
		std::string modsub;
		std::string modsub2;
		std::string modsub3;
		modsub = mod.c_str();
		int const_sharedFlag =0;
		if(mod.find("global14") != std::string::npos) {
			//Those instructions don't need the dereferencing done by g [*]
			if(	base == "GRED" ||
				base == "GATOM" ||
				base == "GST" ||
				base == "GLD")
				output("[");
			else
				output("g[");
		} else if(mod[0]=='g') {
			//Shared memory
			output("s[");
			const_sharedFlag=1;
		} else if(mod.find("local") !=  std::string::npos) {
			if((base=="LST")||
				(base=="LLD"))
			output("[");
		else
			output("l[");
			//modsub3 = modsub.substr(4, modsub.length()-10);
			//output(modsub3.c_str());
			//output("[");
			//localFlag = 1;
		} else if(mod.substr(0,9) == "constant1") {
			output(modsub.substr(0, modsub.find_first_of("[]")+1).c_str());
			const_sharedFlag=1;
		} else if(mod.substr(0,9)=="constant0"){
			output("constant0[");
			const_sharedFlag=1;
		} else {
			printf("Unidentified modifier: %s\n", mod.c_str());
			assert(0);
		}

		modsub = modsub.substr(modsub.find_first_of("[]")+1);
		modsub = modsub.substr(0, modsub.length()-1);
		//Here we handle whatever was inside the []
		int plusequalFlag = 0;
		if(modsub.find("+++") !=  std::string::npos) {
			plusequalFlag = 1;
		}

		if(modsub.find("+") !=  std::string::npos)
		{
			//Handle a1+++0x1 or a1+0x1
			modsub2 = modsub.substr(modsub.find("+")+1);

			output("$ofs");
			output(modsub.substr(1,1).c_str());

			if(plusequalFlag == 1) {
				output("+=");
			} else {
				output("+");
			}


			if(modsub2[0]=='r') {
				output("$");
				printCuobjdumpOperandlohi(modsub2);
			} else {
				modsub2 = modsub2.substr(2);
				unsigned int addrValue;
				std::stringstream hexStringConvert;
				hexStringConvert << std::hex << modsub2;
				hexStringConvert >> addrValue;
				if(const_sharedFlag == 1)
				{
					unsigned chunksize = 4;
					if (	this->m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".S16" ||
								m_typeModifiers->back() == ".U16")) chunksize = 2;
					if (	this->m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".U8" ||
								m_typeModifiers->back() == ".S8")) chunksize = 1;
					addrValue = addrValue*chunksize;
				}
				char outputHex[10];
				sprintf(outputHex, "%x", addrValue);
				std::stringstream outputhex;
				outputhex << std::hex << addrValue;
				output("0x");
				for(unsigned i=4; i > outputhex.str().length(); i--)
				{
					output("0");
				}
				output(outputhex.str().c_str());
			}
		} else {
			if(modsub[0]=='r') {
				//Register
				output("$");
				printCuobjdumpOperandlohi(modsub);
			} else {
				// Immediate offset
				mod = mod.substr(2);

				unsigned int addrValue;
				std::stringstream hexStringConvert;
				hexStringConvert << std::hex << modsub;
				hexStringConvert >> addrValue;

				if(const_sharedFlag == 1)
				{
					unsigned chunksize = 4;
					if ( m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".S16" ||
								m_typeModifiers->back() == ".U16")) chunksize = 2;
					if (	this->m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".U8" ||
								m_typeModifiers->back() == ".S8")) chunksize = 1;
					addrValue = addrValue*chunksize;
				}
				std::stringstream outputhex;
				outputhex << std::hex << addrValue;
				output("0x");
				for(unsigned i=4; i > outputhex.str().length(); i--) {
					output("0");
				}
				output(outputhex.str().c_str());
			}
		}
		output("]");
	} else if(mod.substr(0,2) == "0x") { //immediate value
		output("0x");
		std::string outputhex;
		outputhex = mod.substr(2);

		for(unsigned i=8; i > outputhex.length(); i--) {
			output("0");
		}
		output(outputhex.c_str());
	} else if(mod.substr(0,3) == "l0x") { //label
		output(mod.c_str());
	} else if(mod.substr(0,9) == "varglobal") { //global variable
		output(mod.substr(9).c_str());
	} else {//variable name
		printf("Unrecognized Operand: %s\n", mod.c_str());
		assert(0);
		output(mod.c_str());
	}
}

/*
 * [한국어]
 * printCuobjdumpOperands - 이 명령어의 모든 피연산자를 순회하며 출력
 *
 * LOP.PASS_B/LOP.S.PASS_B의 두 번째 피연산자는 NOT 변환에서 생략된다.
 */
void cuobjdumpInst::printCuobjdumpOperands()
{
	std::string delimiter = "";
	unsigned i=0;
	for (	std::list<std::string>::iterator operand = m_operands->begin();
			operand != m_operands->end();
			operand++, i++) {
		if(((m_base == "LOP.PASS_B") || (m_base == "LOP.S.PASS_B")) && (i==1)) {
			continue;
		}
		printCuobjdumpOperand(*operand, delimiter, m_base);
		delimiter = ",";
	}
}

/*
 * [한국어]
 * printCuobjdumpOutputModifiers - 곱셈/IMUL 계열 명령어의 출력 폭 수정자 결정
 *
 * 16bit 타입이고 대상 레지스터가 lo/hi인 경우 .lo/.hi, 그 외에는 .wide,
 * 그 밖의 경우 defaultMod를 출력한다.
 */
void cuobjdumpInst::printCuobjdumpOutputModifiers(const char* defaultMod)
{
	std::list<std::string>::iterator typemod = m_typeModifiers->begin();
	if (*typemod == ".U16" or *typemod == ".S16") {
		std::list<std::string>::iterator dest_op = m_operands->begin(); 
		std::string& destination = *dest_op; 
		if (destination[destination.length()-1] == 'l') {
			output(".lo");  // write to the lower 16-bits 
		} else if (destination[destination.length()-1] == 'h') {
			output(".hi");  // write to the upper 16-bits 
		} else {
			output(".wide");  // write to the whole 32-bits 
		}
		return; 
	}
	output(defaultMod);  // default output modifier for mul 
}

/*
 * [한국어]
 * int_default_mod - 정수형 명령어의 기본 타입 수정자(.u32)를 반환
 */
std::string int_default_mod () { return ".u32" ;}


std::string breaktarget;

/*
 * [한국어]
 * printCuobjdumpPtxPlus - 이 SASS 명령어를 PTXPlus 형식으로 변환/출력
 *
 * @labelList: 현재 엔트리에서 사용된 레이블 목록(레이블 출력 여부 결정)
 * @texList:   원본 PTX에서 추출한 실제 텍스처 이름 목록(TEX 변환용)
 *
 * m_base 값에 따라 if-else 분기를 통해 적절한 PTXPlus 명령어를 생성한다.
 * 각 분기는 공통 패턴 printCuobjdumpPredicate(), output(),
 * printCuobjdumpBaseModifiers(), printCuobjdumpTypeModifiers(),
 * printCuobjdumpOperands()를 사용한다.
 */
void cuobjdumpInst::printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList)
{
	printCuobjdumpLabel(labelList);

	/* [한국어] 빈 base: 아무것도 출력하지 않는다. */
	if(m_base == "")
	{
	}
	/* [한국어] .entry 헤더: 별도 출력 없이 cuobjdumpInstList에서 처리. */
	else if(m_base == ".entry")
	{
		/*do nothing here*/
	}
	/* [한국어] 동기화/배리어 명령어. */
	else if(m_base == "BAR.ARV.WAIT b0, 0xfff")
	{
		printCuobjdumpPredicate();
		output("bar.sync 0x00000000;");
	}
	/* [한국어] 정수/부동소수점 덧셈/변환 계열. */
	else if(m_base == "ADA")
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 분기/호출/반복 제어 명령어. */
	else if(m_base == "BRA")
	{
		printCuobjdumpPredicate();
		output("bra");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "BRX")
	{
		printCuobjdumpPredicate();
		output("brx");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "CAL")
	{
		printCuobjdumpPredicate();
		output("callp");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 수학 함수(삼각/지수/로그/역수/제곱근) 명령어. */
	else if(m_base == "COS")
	{
		printCuobjdumpPredicate();
		//output("nop;");
		//output(" //cos");
		output("cos");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "DADD")
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "DMIN")
	{
		printCuobjdumpPredicate();
		output("min");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "DMAX")
	{
		printCuobjdumpPredicate();
		output("max");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}


	/*
	else if(strcmp(m_base, "DFMA")==0)
	{
		printCuobjdumpPredicate();
		output("mad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	*/
	else if(m_base == "DMUL")
	{
		printCuobjdumpPredicate();
		output("mul");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "EX2")
	{
		printCuobjdumpPredicate();
		//output("nop;");
		//output(" //ex2");
		output("ex2");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "F2F")
	{
		printCuobjdumpPredicate();

		int absFlag = 0;

		for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
				basemod != m_baseModifiers->end();
				basemod++){
			if( *basemod ==  ".abs")
			{
				output("abs");
				absFlag = 1;
				break;
			}
		}
		if(absFlag == 0)
		{
			output("cvt");
		}

		printCuobjdumpBaseModifiers();

		if(absFlag == 0)
		{
			printCuobjdumpTypeModifiers();
		}
		else
		{
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "F2I")
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		//output(".f64.s32");
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FADD" || m_base == "FADD32I")
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FADD32")
	{
		printCuobjdumpPredicate();
		output("add.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FMAD" || m_base == "FMAD32I")
	{
		printCuobjdumpPredicate();
		output("mad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FMUL" || m_base == "FMUL32I")
	{
		printCuobjdumpPredicate();
		output("mul");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FMUL32")
	{
		printCuobjdumpPredicate();
		output("mul.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "FSET")
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".f32.f32");
		}
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "DSET")
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".f64.f64");
		}
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 레지스터 이동/변환(G2R/R2G/MOV) 및 메모리 접근(GLD/GST/LLD/LST) 명령어. */
	else if(m_base == "G2R")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if( m_typeModifiers->size() == 2 ) {
			std::string type1, type2, type;
			int type1Size, type2Size;
			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size < type2Size) ? type1 : type2;

			if( strcmp(type.c_str(), ".F16")==0 )
				output(".f16");
			else if( strcmp(type.c_str(), ".F32")==0 )
				output(".f32");
			else if( strcmp(type.c_str(), ".F64")==0 )
				output(".ff64");
			else if( strcmp(type.c_str(), ".S8")==0 )
				output(".s8");
			else if( strcmp(type.c_str(), ".S16")==0 )
				output(".s16");
			else if( strcmp(type.c_str(), ".S32")==0 )
				output(".s32");
			else if( strcmp(type.c_str(), ".S64")==0 )
				output(".bb64");
			else if( strcmp(type.c_str(), ".S128")==0 )
				output(".bb128");
			else if( strcmp(type.c_str(), ".U8")==0 )
				output(".u8");
			else if( strcmp(type.c_str(), ".U16")==0 )
				output(".u16");
			else if( strcmp(type.c_str(), ".U32")==0 )
				output(".u32");
			else if( strcmp(type.c_str(), ".U64")==0 )
				output(".bb64");
			else
				output(type.c_str());
		} else if( m_typeModifiers->size() == 1 ) {
			printCuobjdumpTypeModifiers();
		} else {
			output("Error: unsupported number of type modifiers. ");
                }
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "GLD")
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("ld.global");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "GST")
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("st.global");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 정수↔부동소수점 및 정수↔정수 형변환 명령어. */
	else if(m_base == "I2F")
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "I2I")
	{
		printCuobjdumpPredicate();

		int absFlag = 0;

		for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
				basemod != m_baseModifiers->end();
				basemod++) {
			if(*basemod == ".abs")
			{
				output("abs");
				absFlag = 1;
				break;
			}
		}
		if(absFlag == 0)
		{
			output("cvt");
		}

		printCuobjdumpBaseModifiers();


		if(m_typeModifiers->size() == 0 || absFlag)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();


		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 정수 산술/곱셈/비교(IADD/IMAD/IMUL/ISAD/ISET) 명령어. */
	else if( m_base.find("IADD.CARRY") == 0){ //searches for IADD.CARRY at the start to match IADD.CARRY{numeric}
		std::string pred = "C0";
		pred[1] = m_base[10];
		this->addOperand(pred.c_str());
		printCuobjdumpPredicate();
		output("addp");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IADD" || m_base == "IADD32I")
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IADD32")
	{
		printCuobjdumpPredicate();
		output("add.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMAD32I" || m_base == "IMAD32")
	{
		printCuobjdumpPredicate();
		output("mad.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMAD")
	{
		//Patching the C3 problem
		if(m_predicate->size() > 0 &&
				m_predicate->front() == "C3" &&
				m_operands->back()[0] == '-'){
			m_predicate->clear();
			std::string op = m_operands->back();
			m_operands->pop_back();
			m_operands->push_back(op.substr(1));
			m_operands->push_back("C1");
			output("madp.wide");
		} else {
			printCuobjdumpPredicate();
			output("mad.wide");
		}
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "ISAD")
	{
		printCuobjdumpPredicate();
		output("sad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}

	else if(m_base == "IMAD.U24")
	{
		printCuobjdumpPredicate();
		output("mad24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
        }
	else if(m_base == "IMAD.S24")
	{
		printCuobjdumpPredicate();
		output("mad24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMUL32I.U24.U24" ||
			m_base == "IMUL32I.S24.S24" ||
			m_base == "IMUL32.U24.U24" ||
			m_base == "IMUL32.S24.S24" ||
			m_base == "IMUL.U24.U24" ||
			m_base == "IMUL.S24.S24" )
	{
		printCuobjdumpPredicate();
		output("mul24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);
		}	
		else
		{
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMUL.HI.U24.U24")
	{
		printCuobjdumpPredicate();
		output("mul24.hi");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);
		}	
		else
		{
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMUL" || m_base == "IMUL32I")
	{
		printCuobjdumpPredicate();
		output("mul");
      printCuobjdumpOutputModifiers(".lo"); 
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;

			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();


			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
					output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);
		}	
		else
		{
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMUL32")
	{
		printCuobjdumpPredicate();
		output("mul.half.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);

		}	
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "IMUL32.S24.S24")
	{
		printCuobjdumpPredicate();
		output("mul24.half.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);

		}	
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "ISET")
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32.u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else if(m_typeModifiers->size() == 1)
		{
			printCuobjdumpTypeModifiers(); printCuobjdumpTypeModifiers();
		}
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 로컬 메모리 및 비트/시프트/이동 명령어. */
	else if(m_base == "LG2")
	{
		printCuobjdumpPredicate();
		output("lg2");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LST")
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("st.local");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "MVC"){
		printCuobjdumpPredicate();
		//Use cvt if there is conversion involved (2 modifiers) otherwise mov
		if(m_typeModifiers->size() < 2)
			output("mov");
		else
			output("cvt");

		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "MOV" || m_base == "MVI")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "MOV32")
	{
		printCuobjdumpPredicate();
		output("mov.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] NOP 및 공유/로컬 데이터 이동(STS/LDS/LD/ST) 명령어. */
	else if(m_base == "NOP")
	{
		printCuobjdumpPredicate();
		output("nop");
		printCuobjdumpBaseModifiers();
		output(";");
	}
	else if(m_base == "LLD")
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("ld.local");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LOP.AND" || m_base == "LOP.S.AND")
	{
		printCuobjdumpPredicate();
		output("and");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LOP.OR" || m_base == "LOP.S.OR")
	{
		printCuobjdumpPredicate();
		output("or");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LOP.PASS_B" || m_base == "LOP.S.PASS_B")
	{
		printCuobjdumpPredicate();
		output("not");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LOP.XOR" || m_base == "LOP.S.XOR")
	{
		printCuobjdumpPredicate();
		output("xor");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "R2A")
	{
		printCuobjdumpPredicate();
		output("shl");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		if(m_operands->size() == 2)
			output(", 0x0");
		output(";");
	}
	else if(m_base == "R2G.U16.U8"){
		/*
		 * This code handles a cuobjdump bug that causes the wrong register number to be printed
		 */
		printCuobjdumpPredicate();
		output("mov.u8");
		std::list<std::string>::iterator operand = m_operands->begin();
		std::string delimiter = "";
		printCuobjdumpOperand(*operand, delimiter, m_base);
		operand++;
		output(", ");
		std::string curr = *operand;
		curr = curr.substr(1);
		int regnum;
		std::istringstream(curr)>>regnum;
		output("$r");
		std::stringstream finalregnum;
		finalregnum << (regnum/2);
		output(finalregnum.str().c_str());
		output( regnum%2==0? ".lo": ".hi");
		output(";");
	}
	else if(m_base == "R2G")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if( m_typeModifiers->size() == 2 ) {
			std::string type1, type2, type;
			int type1Size, type2Size;
			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();


			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size < type2Size) ? type1 : type2;

			if( strcmp(type.c_str(), ".F16")==0 )
				output(".f16");
			else if( strcmp(type.c_str(), ".F32")==0 )
				output(".f32");
			else if( strcmp(type.c_str(), ".F64")==0 )
				output(".ff64");
			else if( strcmp(type.c_str(), ".S8")==0 )
				output(".s8");
			else if( strcmp(type.c_str(), ".S16")==0 )
				output(".s16");
			else if( strcmp(type.c_str(), ".S32")==0 )
				output(".s32");
			else if( strcmp(type.c_str(), ".S64")==0 )
				output(".bb64");
			else if( strcmp(type.c_str(), ".S128")==0 )
				output(".bb128");
			else if( strcmp(type.c_str(), ".U8")==0 )
				output(".u8");
			else if( strcmp(type.c_str(), ".U16")==0 )
				output(".u16");
			else if( strcmp(type.c_str(), ".U32")==0 )
				output(".u32");
			else if( strcmp(type.c_str(), ".U64")==0 )
				output(".bb64");
			else
				output(type.c_str());
		} else if( m_typeModifiers->size() == 1 ) {
			printCuobjdumpTypeModifiers();
		} else {
			output("Error: unsupported number of type modifiers. ");
                }

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "RCP")
	{
		printCuobjdumpPredicate();
		output("rcp");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "RCP32")
	{
		printCuobjdumpPredicate();
		output("rcp.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32");
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "RET")
	{
		printCuobjdumpPredicate();
		output("retp;");
	}
	else if(m_base == "RRO")
	{
		output("nop; //");
		printCuobjdumpPredicate();
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "RSQ")
	{
		printCuobjdumpPredicate();
		output("rsqrt");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "SHL")
	{
		printCuobjdumpPredicate();
		output("shl");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "SHR")
	{
		printCuobjdumpPredicate();
		output("shr");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "SIN")
	{
		printCuobjdumpPredicate();
		output("sin");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "S2R")
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		output(".u32.u32");
		printCuobjdumpOperands();
		output(";");
	}
	else if( m_base == "SSY")
	{
		printCuobjdumpPredicate();
		output("ssy");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "LD")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "STS")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "BAR")
	{
		printCuobjdumpPredicate();
		output("bar.sync 0;");
	}
	else if(m_base == "LDS")
	{
		// If there is not global address space that includes shared memory, then fix this.
		// Same for STS
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(m_base == "ST")
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
		/* [한국어] 최소/최대/원子(activemask) 명령어. */
	} else if(m_base == "IMIN") {
		printCuobjdumpPredicate();
		output("min");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "IMAX") {
		printCuobjdumpPredicate();
		output("max");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "FMIN") {
			printCuobjdumpPredicate();
			output("min");
			printCuobjdumpBaseModifiers();
			if(m_typeModifiers->size() == 0)
				output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
			else
				printCuobjdumpTypeModifiers();
			printCuobjdumpOperands();
			output(";");
	}
	else if(m_base == "FMAX") {
			printCuobjdumpPredicate();
			output("max");
			printCuobjdumpBaseModifiers();
			if(m_typeModifiers->size() == 0)
				output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
			else
				printCuobjdumpTypeModifiers();
			printCuobjdumpOperands();
			output(";");
	}
	else if(m_base == "A2R") {
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 텍스처 샘플링 및 원子/exit/atomic/vote 명령어. */
	else if((m_base == "TEX") ||
			(m_base == "TEX32")) {
		printCuobjdumpPredicate();
		output("tex.1d.v4.f32.s32 ");
		std::string addrReg, tex_id;
		std::list<std::string>::iterator operand = m_operands->begin();
		output("{");
		printCuobjdumpOperand(*operand, "", "");
		output(",_,_,_} , ");
		std::string reg = *operand;
		operand++;
		tex_id = *operand;
		unsigned int tex_id_int;
		std::stringstream ss;
		ss << std::hex << tex_id;
		ss >> tex_id_int;
		std::list<std::string>::iterator texIter = texList.begin();
		for (unsigned i=0; i<tex_id_int; i++) {
			assert (texIter!=texList.end());
			texIter++;
		}
		output((*texIter).c_str());
		output(",{");
		printCuobjdumpOperand(reg, "", "");
		output(",_,_,_};");
	}
	else if(m_base == "EXIT") {
		printCuobjdumpPredicate();
		output("exit");
		output(";");
	}
	else if(m_base == "GRED") {
		printCuobjdumpPredicate();
		// ptx instruction atom can be used to perform reduction using destination register '_'
		output("atom.global");
		printCuobjdumpBaseModifiers();
			printCuobjdumpTypeModifiers();
		output(" _, ");
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "GATOM") {
		printCuobjdumpPredicate();
		output("atom.global");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}else if(m_base == "PBK") {
		//PKB specifies the target of later BRK (break) instructions
		// Here we convert it to nop and store the target
		output("nop;");
		breaktarget = m_operands->front();
	} else if(m_base == "BRK") {
		printCuobjdumpPredicate();
		/*
		 * Convert it **at compile time** to a branch to the break target saved earlier
		 * Correct operation not guaranteed
		 */

		output("bra ");
		output(breaktarget.c_str());
		output(";");
	} else if(m_base == "C2R") {
		printCuobjdumpPredicate();
		output("mov.u32");
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "R2C") {
		printCuobjdumpPredicate();
		output("mov.pred");
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "VOTE") {
		printCuobjdumpPredicate();
		output("vote");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(m_base == "ACTIVEMASK") {
		printCuobjdumpPredicate();
		output("activemask.b32");
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] DFMA (이중정밀도 fma) 별도 처리. */
	else if(m_base == "DFMA")
	{
		printCuobjdumpPredicate();
		output("fma.rz.ff64");
		printCuobjdumpBaseModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	/* [한국어] 알 수 없는 명령어: 오류 출력 후 종료. */
	else
	{
		printf("Unknown Instruction: ");
		printf("%s",m_base.c_str());
		printf("\n");
		output("Unknown Instruction: ");
		output(m_base);
		assert(0);
	}

}

