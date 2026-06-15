/*
 * [한국어 설명] 원본 PTX 헤더 파서용 시맨틱 액션 (ptx_parser.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 cuobjdump_to_ptxplus의 PTX 파서(ptx.l/ptx.y)가 원본 PTX
 * 파일을 파싱할 때 호출하는 시맨틱 액션(콜백) 함수들을 정의한다.
 * GPGPU-Sim의 메인 PTX 파서(src/cuda-sim/ptx_parser.h/cc)와 달리,
 * 이 파일은 완전한 PTX IR을 구축하는 것이 아니라 .version, .target,
 * .entry/.func 선언, 파라미터, 텍스처 선언 등 **헤더 부분만** 추출하여
 * g_headerList(cuobjdumpInstList)에 저장하는 것이 목적이다.
 * 대부분의 함수는 더미(dummmy)로 구현되어 있고, 헤더 관련 액션만
 * 실제로 g_headerList를 조작한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → ptx_parse() 호출
 *         → ptx.y (Bison 문법)
 *             → ptx.l (Flex 어휘 분석)
 *                 → ptx_parser.h의 add_version_info(), func_header(),
 *                   add_space_spec(), add_scalar_type_spec() 등 호출
 *                     → g_headerList 채움
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h : g_headerList 타입 및 cuobjdumpInst 조작 API
 *   - ptx.tab.h           : S8_TYPE, F32_TYPE 등 Bison 토큰/타입 정의
 * 이 파일에 의존하는 모듈:
 *   - ptx.y               : 문법 규칙의 시맨틱 액션으로 이 파일의 함수들 참조
 *   - cuobjdump_to_ptxplus.cc : g_headerList를 g_instList에 텍스처 목록 전달
 *
 * === 주요 함수/구조체 요약 ===
 * _memory_space_t       - PTX 메모리 공간 enum(PTXPlus 용 단순화 버전)
 * PTX_PARSE_DPRINTF     - 파서 디버그 출력 매크로(P_DEBUG로 켜고 끔)
 * add_version_info()    - .version 지시어 처리
 * target_header*()      - .target 지시어 처리
 * func_header()         - .entry/.func 시작 처리
 * add_function_name()   - 엔트리/함수 이름 추가
 * func_header_info*()   - .entry/.func 파라미터 및 기타 정보 추가
 * add_space_spec()      - .param/.tex/.const 공간 지정 처리
 * add_scalar_type_spec()- scalar 타입 지정자(.s32/.f32 등) 처리
 * add_identifier()      - 식별자 추가(더미)
 */

// Copyright (c) 2009-2011, Jimmy Kwa,
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

#ifndef _PTX_PARSER_H_
#define _PTX_PARSER_H_

#include <cstdlib>
#include <cstring>
#include <string>
#include <list>
#include <map>
#include <vector>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include "cuobjdumpInstList.h"

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3
#define P_DEBUG 0
/* [한국어] 파서 난 디버그 출력 매크로. P_DEBUG가 0이면 아무것도 출력하지 않는다. */
#define PTX_PARSE_DPRINTF(...) \
   if(P_DEBUG) { \
      printf("(%s:%s:%u) ", __FILE__, __FUNCTION__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
      fflush(stdout); \
   }

/*
 * [한국어]
 * _memory_space_t — PTX 메모리 공간을 나타내는 열거형.
 *
 * 이 파일은 cuobjdump_to_ptxplus 납부에서만 사용되는 단순화된 버전으로,
 * src/cuda-sim/ptx_ir.h의 memory_space_t와는 별개이다.
 */
enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   sstarr_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space
};

/* [한국어] PTX 파싱 중 오류 발생 여부를 나타내는 전역 플래그. */
int g_error_detected;
/* [한국어] 현재 파싱 중인 파일명(오류 메시지용). */
const char *g_filename = "";
/* [한국어] 함수 선언 중임을 나타내는 전역 플래그. */
int g_func_decl;

/* [한국어] 아래 함수들은 GPGPU-Sim 메인 PTX 파서의 인터페이스와 맞추기 위한
 * 더미(dummmy) 구현들이다. cuobjdump_to_ptxplus에서는 헤더 정볧만 추출하므로
 * 대부분 아무 동작도 하지 않는다. */
void set_symtab( void* a ) {PTX_PARSE_DPRINTF(" ");}
void end_function() {PTX_PARSE_DPRINTF(" ");}
void add_directive() {PTX_PARSE_DPRINTF(" ");}
void add_function_arg() {PTX_PARSE_DPRINTF(" ");}
void add_instruction() {PTX_PARSE_DPRINTF(" ");}
void add_file( unsigned a, const char *b ) {PTX_PARSE_DPRINTF(" ");}
void add_variables() {PTX_PARSE_DPRINTF(" ");}
void set_variable_type() {PTX_PARSE_DPRINTF(" ");}
void add_option(int a ) {PTX_PARSE_DPRINTF(" ");}
void add_wmma_option(int a ) {PTX_PARSE_DPRINTF(" ");}
void add_array_initializer() {PTX_PARSE_DPRINTF(" ");}
void add_label( const char *a ) {PTX_PARSE_DPRINTF(" ");}
void set_return() {PTX_PARSE_DPRINTF(" ");}
void add_opcode( int a ) {PTX_PARSE_DPRINTF(" ");}
void add_pred( const char *a, int b, int c ) {PTX_PARSE_DPRINTF(" ");}
void add_scalar_operand( const char *a ) {PTX_PARSE_DPRINTF("%s", a);}
void add_neg_pred_operand( const char *a ) {PTX_PARSE_DPRINTF(" ");}
void add_address_operand( const char *a, int b ) {PTX_PARSE_DPRINTF("%s", a);}
void add_address_operand2( int b ) {PTX_PARSE_DPRINTF(" ");}
void change_operand_lohi( int a ) {PTX_PARSE_DPRINTF(" ");}
void change_double_operand_type( int a ) {PTX_PARSE_DPRINTF(" ");}
void change_operand_neg( ) {PTX_PARSE_DPRINTF(" ");}
void add_double_operand( const char *a, const char *b ) {PTX_PARSE_DPRINTF(" ");}
void add_1vector_operand( const char *a ) {PTX_PARSE_DPRINTF(" ");}
void add_2vector_operand( const char *a, const char *b ) {PTX_PARSE_DPRINTF(" ");}
void add_3vector_operand( const char *a, const char *b, const char *c ) {PTX_PARSE_DPRINTF(" ");}
void add_4vector_operand( const char *a, const char *b, const char *c, const char *d ) {PTX_PARSE_DPRINTF(" ");}
void add_8vector_operand( const char *a, const char *b, const char *c, const char *d ,const char *e,const char *f,const char *g,const char *h) {PTX_PARSE_DPRINTF(" ");}
void add_builtin_operand( int a, int b ) {PTX_PARSE_DPRINTF(" ");}
void add_memory_operand() {PTX_PARSE_DPRINTF(" ");}
void change_memory_addr_space( const char *a ) {PTX_PARSE_DPRINTF(" ");}
void add_literal_int( int a ) {PTX_PARSE_DPRINTF(" ");}
void add_literal_float( float a ) {PTX_PARSE_DPRINTF(" ");}
void add_literal_double( double a ) {PTX_PARSE_DPRINTF(" ");}
void add_ptr_spec( enum _memory_space_t spec ) {PTX_PARSE_DPRINTF(" ");}
void add_extern_spec() {PTX_PARSE_DPRINTF(" ");}
void add_alignment_spec( int ) {PTX_PARSE_DPRINTF(" ");}
void add_pragma( const char *a ) {PTX_PARSE_DPRINTF(" ");}
void add_constptr(const char* identifier1, const char* identifier2, int offset) {PTX_PARSE_DPRINTF(" ");}

//Jin: handle instructino group for cdp
void start_inst_group(){PTX_PARSE_DPRINTF(" ");};
void end_inst_group(){PTX_PARSE_DPRINTF(" ");};


/*non-dummy stuff below this point*/

/* [한국어] 원본 PTX 헤더 정보를 저장하는 전역 cuobjdumpInstList.
 * 이 파일의 실제 동작을 수행하는 함수들이 이 리스트를 조작한다. */
extern cuobjdumpInstList *g_headerList;

// Global variable to track if we are currently inside a entry directive
/* [한국어] 현재 .entry/.func 지시어 난부에 있는지 추적. */
bool inEntryDirective = false;
// Global variable to track is we are currently inside the parameter definitions for an entry
/* [한국어] 현재 .entry/.func의 파라미터 선언 난부에 있는지 추적. */
bool inParamDirective = false;

/* [한국어] 현재 .const 지시어 난부에 있는지 추적(대부분 비활성화됨). */
bool inConstDirective = false;

// Global variable to track if we are currently inside a tex directive
/* [한국어] 현재 .tex 지시어 난부에 있는지 추적. */
bool inTexDirective = false;


/*
 * [한국어]
 * add_identifier - PTX 식별자 추가(더미)
 *
 * @a: 식별자 이름
 * @b, @c: 추가 인덱스/플래그(사용되지 않음)
 *
 * 메인 PTX 파서와의 인터페이스 호환용. cuobjdump_to_ptxplus에서는
 * 헤더 외 식별자는 무시한다.
 */
void add_identifier( const char *a, int b, unsigned c ) {
	PTX_PARSE_DPRINTF("name=%s", a);
	if(inConstDirective){
		//g_headerList->getListEnd()
	}
}

/*
 * [한국어]
 * add_function_name - .entry/.func 선언에서 함수/커널 이름 추가
 *
 * @headerInput: 함수/커널 이름 문자열
 *
 * 현재 마지막으로 추가된 inst의 base가 ".entry" 또는 ".func"인 경우에만
 * g_headerList의 마지막 엔트리 이름을 설정하고 피연산자로 추가한다.
 */
void add_function_name( const char* headerInput )
{
	PTX_PARSE_DPRINTF("name=%s", headerInput);
	char* headerInfo = (char*) headerInput;
	std::string compareString = g_headerList->getListEnd().getBase();

	if((compareString == ".entry")||(compareString == ".func"))
	{
		g_headerList->setLastEntryName(headerInfo);
		g_headerList->getListEnd().addOperand(headerInfo);
	}
}

/*
 * [한국어]
 * add_space_spec - 메모리 공간 지시어 처리
 *
 * @spec: 메모리 공간 enum
 * @value: 상수 메모리 뱅크 번호 등 부가 값
 *
 * param_space_unclassified이면서 .entry/.func 파라미터 선언 중일 때
 * .param inst를 추가하고, tex_space이면 텍스처 선언 중임을 표시한다.
 * const_space는 현재 비활성화되어 처리하지 않는다.
 */
//void add_space_spec(int headerInput)
void add_space_spec( enum _memory_space_t spec, int value )
{
	PTX_PARSE_DPRINTF("spec=%u", spec);
	cuobjdumpInst *instEntry;
	//static int constmemindex=1;
	switch(spec)
	{
		case param_space_unclassified:
			if(inEntryDirective && inParamDirective) {
				instEntry = new cuobjdumpInst();
				instEntry->setBase(".param");
				g_headerList->add(instEntry);
			}
			break;
		case tex_space:
			inTexDirective = true;
			/*
			instEntry = new cuobjdumpInst();
			instEntry->setBase(".tex");
			g_headerList->add(instEntry);
			*/
			break;
		case const_space:
			if(!inEntryDirective) {
				/*
				inConstDirective = true;
				instEntry = new cuobjdumpInst();
				instEntry->setBase(".const");
				g_headerList->add(instEntry);
				*/
				//g_headerList->addConstMemory(constmemindex++);
			}
			break;
		default:
			break;
	}
}

/*
 * [한국어]
 * add_scalar_type_spec - scalar 타입 지정자(.s32, .f32 등)를 헤더에 추가
 *
 * @headerInput: S8_TYPE, F32_TYPE 등 ptx.tab.h에 정의된 타입 토큰
 *
 * .entry/.func 파라미터 선언 중이거나 .tex/.const 선언 중일 때,
 * g_headerList의 마지막 inst에 타입 수정자(base modifier)로 추가한다.
 */
void add_scalar_type_spec( int headerInput )
{
	PTX_PARSE_DPRINTF(" ");
	//const char* compareString = g_headerList->getListEnd().getBase();

	if( (inEntryDirective && inParamDirective) || inTexDirective || inConstDirective)
	{
		switch(headerInput)
		{
			case S8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s8");
				break;
			case S16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s16");
				break;
			case S32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s32");
				break;
			case S64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s64");
				break;
			case U8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u8");
				break;
			case U16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u16");
				break;
			case U32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u32");
				break;
			case U64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u64");
				break;
			case F16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f16");
				break;
			case F32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f32");
				break;
			case F64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f64");
				break;
			case B8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b8");
				break;
			case B16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b16");
				break;
			case B32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b32");
				break;
			case B64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b64");
				break;
			case PRED_TYPE:
				g_headerList->getListEnd().addBaseModifier(".pred");
				break;
			default:
				std::cout << "Unknown type spec" << "\n";
				break;
		}
	}
}

/*
 * [한국어]
 * add_version_info - .version 지시어 처리
 *
 * @versionNumber: PTX 버전 번호(예: 1.4)
 * @ext:           확장 플래그(사용되지 않음)
 *
 * g_headerList에 .version inst를 추가하고 버전 번호 문자열을 피연산자로
 * 추가한다. 출력 시 "1.4+" 형태로 변환된다.
 */
//void version_header(double versionNumber)
void add_version_info( float versionNumber, unsigned ext)
{
	PTX_PARSE_DPRINTF(" ");
	cuobjdumpInst *instEntry = new cuobjdumpInst();
	instEntry->setBase(".version");
	g_headerList->add(instEntry);


	//convert double to char*
	std::ostringstream strs;
	strs << versionNumber;
	char *versionNumber2 = strdup(strs.str().c_str());

	g_headerList->getListEnd().addOperand(versionNumber2);
	//g_headerList->getListEnd().addOperand("1.4");
}

/*
 * [한국어]
 * target_header - .target 지시어 처리(단일 타겟)
 *
 * @firstTarget: 타겟 아키텍처 문자열(예: sm_20)
 */
void target_header(char* firstTarget)
{
	PTX_PARSE_DPRINTF("%s", firstTarget);
	cuobjdumpInst *instEntry = new cuobjdumpInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);
}

/*
 * [한국어]
 * target_header2 - .target 지시어 처리(두 개 타겟)
 *
 * @firstTarget, @secondTarget: 타겟 아키텍처 문자열들
 */
void target_header2(char* firstTarget, char* secondTarget)
{
	PTX_PARSE_DPRINTF("%s, %s", firstTarget, secondTarget);
	cuobjdumpInst *instEntry = new cuobjdumpInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);

	g_headerList->getListEnd().addOperand(secondTarget);
}

/*
 * [한국어]
 * target_header3 - .target 지시어 처리(세 개 타겟)
 *
 * @firstTarget, @secondTarget, @thirdTarget: 타겟 아키텍처 문자열들
 */
void target_header3(char* firstTarget, char* secondTarget, char* thirdTarget)
{
	PTX_PARSE_DPRINTF("%s, %s, %s", firstTarget, secondTarget, thirdTarget);
	cuobjdumpInst *instEntry = new cuobjdumpInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);

	g_headerList->getListEnd().addOperand(firstTarget);

	g_headerList->getListEnd().addOperand(secondTarget);

	g_headerList->getListEnd().addOperand(thirdTarget);
}

/*
 * [한국어]
 * start_function - .entry/.func 블록 시작
 *
 * @a: 미사용 인자
 *
 * inEntryDirective 플래그를 true로 설정하여 이후 파라미터 선언을
 * 헤더에 추가하도록 한다.
 */
void start_function( int a )
{
	PTX_PARSE_DPRINTF(" ");
	inEntryDirective = true;
}

/*
 * [한국어]
 * reset_symtab - 함수/엔트리 파싱 종료 시 상태 리셋
 *
 * @return: 항상 NULL
 *
 * inEntryDirective 플래그를 false로 되돌린다.
 */
void* reset_symtab()
{
	PTX_PARSE_DPRINTF(" ");
	inEntryDirective = false;
	return (void*) NULL;
}

/*
 * [한국어]
 * func_header - .entry/.func 지시어 처리
 *
 * @headerBase: ".entry" 또는 ".func" 문자열
 *
 * g_headerList에 새 .entry/.func inst를 추가하고 inEntryDirective를 true로
 * 설정한다. 이 inst는 이후 add_function_name()으로 이름이 채워진다.
 */
void func_header(const char* headerBase)
{
	PTX_PARSE_DPRINTF("%s", headerBase);
	// If start of an entry
	if((strcmp(headerBase, ".entry")==0)||(strcmp(headerBase, ".func")==0)) {
		inEntryDirective = true;
		g_headerList->addEntry("");
		cuobjdumpInst *instEntry = new cuobjdumpInst();
		instEntry->setBase(headerBase);
		g_headerList->add(instEntry);

	}
}

/*
 * [한국어]
 * func_header_info - .entry/.func 헤더의 추가 정보 처리
 *
 * @headerInfo: 헤더에 추가할 토큰 문자열(예: "(", ")", 파라미터 이름 등)
 *
 * inEntryDirective가 true이고 .tex 선언이 아닌 경우,
 * g_headerList의 마지막 inst에 피연산자로 추가한다.
 * "("가 들어오면 inParamDirective를 true로, ")"가 들어오면 false로 설정.
 * .tex 선언 중이면 텍스처 이름을 m_realTexList에 추가한다.
 */
void func_header_info(const char* headerInfo)
{
	PTX_PARSE_DPRINTF("%s", headerInfo);
	//const char* compareString = g_headerList->getListEnd().getBase();

	if(inEntryDirective && !inTexDirective) {
		g_headerList->getListEnd().addOperand(headerInfo);
		// If start of parameters
		if(strcmp(headerInfo,"(")==0)
			inParamDirective = true;

		// If end of parameters
		if(strcmp(headerInfo,")")==0) {
			inParamDirective = false;
		}
	} else if(inTexDirective) {
		g_headerList->addTex(headerInfo);
		inTexDirective = false;
	} else if(inConstDirective){

	} else {
		// This information is supposed to be not needed.
		// Suppressing printing it
		// printf("Unkown header info: #%s#\n", headerInfo);
	}

}

/*
 * [한국어]
 * func_header_info_int - 정수가 포함된 헤더 정보 처리
 *
 * @s:   토큰 문자열
 * @i:   정수 값
 *
 * .entry/.func 파라미터 선언 중일 때, 토큰과 정수 값을 문자열로 변환하여
 * g_headerList 마지막 inst에 추가한다.
 */
void func_header_info_int(const char* s, int i)
{
	PTX_PARSE_DPRINTF("%s %d", s, i);
	if(inEntryDirective && !inTexDirective) {
		g_headerList->getListEnd().addOperand(s);
		char *buff = (char*) malloc(30*sizeof(char));
		sprintf(buff, "%d", i);
		assert (i>=0);
		g_headerList->getListEnd().addOperand(buff);
	}
}

/*
 * [한국어]
 * maxnt_id - .maxntid 지시어 처리(더미)
 *
 * @x, @y, @z: 스레드 블록 차원
 *
 * 현재 cuobjdump_to_ptxplus에서는 별도 처리 없이 func_header_info_int로
 * 출력되며, 이 함수는 호환성을 위해 존재한다.
 */
void maxnt_id(int x, int y, int z) {

}
#endif //_PTX_PARSER_H_
