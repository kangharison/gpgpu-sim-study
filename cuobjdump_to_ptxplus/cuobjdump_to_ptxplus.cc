/*
 * [한국어 설명] cuobjdump_to_ptxplus 메인 진입점 (cuobjdump_to_ptxplus.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 cuobjdump_to_ptxplus 도구의 독립 실행형 main() 함수를 제공한다.
 * NVIDIA cuobjdump가 출력한 SASS(`.sass`), ELF(`.elf`), 그리고 원본 PTX(`.ptx`)
 * 파일을 입력으로 받아, GPGPU-Sim이 기능 시뮬레이션할 수 있는 PTXPlus 형식의
 * 출력 파일(`.ptxplus`)을 생성한다. PTXPlus는 SASS의 저수준 제어 흐름과
 * 메모리 접근을 PTX 문법에 가깝게 변환한 중간 표현으로, 구형 GPU(GT200/Fermi 등)
 * 바이너리를 GPGPU-Sim에서 실행할 때 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 변환 파이프라인:
 *   CUDA 바이너리 → cuobjdump → .ptx(원본), .sass(SASS 디스어셈블리), .elf(상수/전역 메타데이터)
 *                              → [cuobjdump_to_ptxplus]
 *                                → elf_parse()   : 상수 메모리/로컬 메모리/전역 심볼 추출
 *                                → ptx_parse()   : 원본 PTX 헤더(.version/.target/.entry 등) 추출
 *                                → sass_parse()  : SASS 명령어 파싱 및 납은 IR(cuobjdumpInstList) 구축
 *                                → printCuobjdumpPtxPlusList() : PTXPlus 텍스트 출력
 *                              → .ptxplus
 *   이후 GPGPU-Sim은 convert_ptx_and_sass_to_ptxplus() (ptx_loader.cc)를 통해
 *   해당 .ptxplus를 PTX 파서로 다시 로드하여 기능 시뮬레이션에 사용한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h/.cc : 변환된 명령어/엔트리/메모리 선언을 담는 IR 컨테이너
 *   - elf.l / elf.y           : cuobjdump ELF 출력 파서 (상수/전역/로컬 메모리 추출)
 *   - ptx.l / ptx.y / ptx_parser.h : 원본 PTX 헤더 파서 (.version/.target/.entry/파라미터)
 *   - sass.l / sass.y         : SASS 디스어셈블리 파서 (실제 명령어 추출)
 * 이 파일에 의존하는 모듈:
 *   - 빌드 시스템(Makefile/CMake)이 이 main()을 기준으로 cuobjdump_to_ptxplus
 *     실행 파일을 생성한다.
 *   - src/cuda-sim/ptx_loader.cc 가 생성된 .ptxplus 파일을 읽어들인다.
 *
 * === 주요 함수/구조체 요약 ===
 * main()              : 명령행 인수 4개(ptx, sass, elf, 출력)를 받아 3단계 파싱 후 출력
 * output()            : 생성된 PTXPlus 텍스트를 출력 파일(ptxplus_out)에 기록
 * fileToString()      : 파일 전체를 std::string으로 읽어오는 헬퍼
 * extractFilename()   : 전체 경로에서 파일명만 추출(현재 사용되지 않음)
 * g_instList          : SASS 변환 결과를 담는 전역 cuobjdumpInstList
 * g_headerList        : PTX 헤더 정보를 담는 전역 cuobjdumpInstList
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

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cassert>

#include "cuobjdumpInstList.h"

using namespace std;

/* [한국어] SASS 명령어 IR 및 변환 결과를 저장하는 전역 리스트.
 * sass_parse()가 이 리스트에 엔트리와 명령어를 추가하고,
 * printCuobjdumpPtxPlusList()가 이 리스트를 PTXPlus 텍스트로 출력한다. */
cuobjdumpInstList *g_instList = new cuobjdumpInstList();

/* [한국어] 원본 PTX 파일의 헤더 정보(.version/.target/.entry/파라미터/텍스처)를
 * 저장하는 전역 리스트. ptx_parse()가 채우고, 최종 출력 시 g_instList와
 * 결합되어 완전한 PTXPlus 파일을 구성한다. */
cuobjdumpInstList *g_headerList = new cuobjdumpInstList();

int sass_parse();
extern FILE *sass_in;

int ptx_parse();
extern FILE *ptx_in;

int elf_parse();
extern FILE *elf_in;

extern int g_error_detected;

FILE *bin_in;
FILE *ptxplus_out;

/*
 * [한국어]
 * output - 생성된 PTXPlus 텍스트를 출력 파일에 기록
 *
 * @text: 출력할 C 문자열
 *
 * 모든 변환 출력이 이 함수를 통해 ptxplus_out 파일 스트림에 기록된다.
 * stdout 대신 파일로 출력함으로써 후속 PTX 파서가 .ptxplus 파일을
 * 읽어들일 수 있게 한다.
 */
void output(const char * text)
{
	//printf(text);
	fprintf(ptxplus_out,"%s", text);
}

/*
 * [한국어]
 * output(std::string) - std::string 버전의 output 래퍼
 *
 * @text: 출력할 std::string
 *
 * C++ 문자열을 C 문자열로 변환하여 파일 출력 함수로 위임한다.
 */
void output(const std::string text) {
	output(text.c_str());
}

/*
 * [한국어]
 * fileToString - 지정한 파일의 전체 내용을 문자열로 읽어온다
 *
 * @fileName: 읽을 파일 경로
 * @return:   파일 내용을 담은 std::string (줄바꿈 보존)
 *
 * 현재 main()에서 ELF 파일 전체를 문자열로 읽어 elf_parse()의 입력으로
 * 사용하기 위해 호출된다. getline을 사용하여 마지막 줄까지 읽는다.
 */
std::string fileToString(const char * fileName) {
	ifstream fileStream(fileName, ios::in);
	string text, line;
	while(getline(fileStream,line)) {
		text += (line + "\n");
	}
	fileStream.close();
	return text;
}

/*
 * [한국어]
 * extractFilename - 전체 경로 문자열에서 파일명 부분만 추출
 *
 * @path: 전체 파일 경로
 * @return: 마지막 '/' 이후의 파일명
 *
 * 현재 main()에서는 직접 사용되지 않지만, 경로 처리 유틸리티로 남아 있다.
 */
std::string extractFilename( const std::string& path )
{
	return path.substr( path.find_last_of( '/' ) +1 );
}

/*
 * [한국어]
 * main - cuobjdump_to_ptxplus 도구의 진입점
 *
 * @argc: 명령행 인수 개수 (반드시 5여야 함: 프로그램명 + 4개 파일)
 * @argv: [0] 프로그램명, [1] .ptx, [2] .sass, [3] .elf, [4] 출력 .ptxplus
 * @return: 0(정상), 1(인수 오류)
 *
 * 실행 흐름:
 *   1) 인수 개수 검증
 *   2) sass_in, ptx_in, elf_in, ptxplus_out 파일 포인터 개방
 *   3) elf_parse():   ELF에서 상수/전역/로컬 메모리 메타데이터 추출
 *   4) ptx_parse():   원본 PTX에서 헤더(.version/.target/.entry/파라미터/텍스처) 추출
 *   5) g_instList에 g_headerList의 실제 텍스처 이름 목록 복사
 *   6) sass_parse():  SASS 명령어를 cuobjdumpInstList로 변환
 *   7) //HEADER 섹션 출력: g_headerList->printHeaderInstList()
 *   8) //INSTRUCTIONS 섹션 출력: g_instList->printCuobjdumpPtxPlusList(g_headerList)
 *   9) 파일 스트림 종료
 *
 * 참고: 원래 readGlobalMemoryFromBinFile()는 주석 처리되어 사용되지 않는다.
 */
int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << "Usage: " << argv[0] << " ptxfile sassfile elffile ptxplusfile(output)\n";
		return 0;
	}

	string ptxfile = argv[1];
	string sassfile = argv[2];
	string elffile = argv[3];
	string ptxplusfile = argv[4];

	sass_in = fopen(sassfile.c_str(), "r" );
	ptx_in = fopen(ptxfile.c_str(), "r" );
	elf_in = fopen(elffile.c_str(), "r");
	ptxplus_out = fopen(ptxplusfile.c_str(), "w" );


	std::string elf = fileToString(elffile.c_str());

	printf("RUNNING cuobjdump_to_ptxplus ...\n");


	printf("Parsing .elf file %s\n", elffile.c_str());
	elf_parse();
	printf("Finished parsing .elf file %s\n", elffile.c_str());

	//Parse original ptx
	printf("Parsing .ptx file %s\n", ptxfile.c_str());
	ptx_parse();
	if (g_error_detected){
		assert(0 && "ptx parsing failed");
	}
	printf("Finished parsing .ptx file %s\n", ptxfile.c_str());

	// Copy real tex list from ptx to ptxplus instruction list
	g_instList->setRealTexList(g_headerList->getRealTexList());

	// Insert global memory from bin file
//	g_instList->readGlobalMemoryFromBinFile(fileToString(binFilename));

	// Parse cuobjdump output
	printf("Parsing .sass file %s\n", sassfile.c_str());
	sass_parse();
	printf("Finished parsing .sass file %s\n", sassfile.c_str());

	// Print ptxplus
	output("//HEADER\n");
	g_headerList->printHeaderInstList();
	output("//END HEADER\n\n\n");
	output("//INSTRUCTIONS\n");
	g_instList->printCuobjdumpPtxPlusList(g_headerList);
	output("//END INSTRUCTIONS\n");

	fclose(sass_in);
	fclose(ptx_in);

	fclose(ptxplus_out);

	printf("DONE. \n");

	return 0;
}
