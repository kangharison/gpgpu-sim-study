/*
 * [한국어 설명] OpenCL 소스→PTX 변환 래퍼 실행 파일 (nvopencl_wrapper.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 OpenCL 커널 소스(.cl)를 NVIDIA GPU 드라이버를 이용해 컴파일하고,
 * 컴파일된 바이너리에서 PTX(Parallel Thread eXecution) 어셈블리를 추출하는
 * 독립 실행 가능한 헬퍼 유틸리티이다. GPGPU-Sim 시뮬레이터 자체가 아니라
 * 시뮬레이터가 system() 또는 ssh 명령으로 자식 프로세스로 호출하는 보조 바이너리이다.
 * 이 바이너리가 없으면 GPGPU-Sim은 OpenCL 워크로드를 시뮬레이션할 수 없다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   OpenCL 앱 → libopencl.so(GPGPU-Sim) → opencl_runtime_api.cc::_cl_program::Build()
 *     → system("nvopencl_wrapper input.cl output.ptx [options]")
 *         → [이 파일: nvopencl_wrapper main()] → NVIDIA libOpenCL.so(실 드라이버)
 *             → clCreateProgramWithSource/clBuildProgram → clGetProgramInfo(BINARIES)
 *                 → output.ptx 파일 기록 → 종료
 *         → opencl_runtime_api.cc가 output.ptx를 읽어 PTX 시뮬레이션에 사용
 * 실행 컨텍스트: GPGPU-Sim 호스트 프로세스가 fork/exec한 별도 자식 프로세스.
 * 반드시 NVIDIA GPU와 실제 NVIDIA 드라이버(libOpenCL.so)가 설치된 머신에서만 실행된다.
 * GPGPU_LIBDIR 환경변수가 가리키는 NVIDIA의 libOpenCL.so를 LD_LIBRARY_PATH로 로드한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: NVIDIA libOpenCL.so(실 GPU 드라이버) - CL/cl.h 헤더를 통해 사용.
 * 의존: 표준 C 런타임(stdio, stdlib, string, stdarg).
 * 호출자: opencl_runtime_api.cc의 _cl_program::Build()가 GPGPUSIM_ROOT 경로 아래
 *   libopencl/bin/nvopencl_wrapper 바이너리를 찾아 system()으로 호출한다.
 * 입력 데이터: argv[1]=OpenCL 소스 파일, argv[2]=출력 PTX 파일, argv[3..]=컴파일 옵션.
 * 출력 데이터: argv[2] 경로에 OpenCL 바이너리(PTX 포함)를 기록.
 *   GPGPU-Sim은 이 파일을 읽어 PTX 시뮬레이션 입력으로 사용한다.
 *
 * === 주요 함수/구조체 요약 ===
 * vmyexit(code, str, ap) - 에러 메시지를 va_list로 포맷하고 code!=0이면 exit()
 * myexit(code, str, ...) - vmyexit의 가변인자 래퍼
 * main(argc, argv)       - 핵심 로직: OpenCL 플랫폼 탐색→컨텍스트 생성→컴파일→PTX 추출→파일 기록
 */
/*
 * Copyright © 2009 by Tor M. Aamodt and the University of British Columbia,
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <CL/cl.h>   // [한국어] NVIDIA 실 드라이버의 OpenCL C API 헤더 — clCreateContext/clBuildProgram 등 사용
#include <stdio.h>   // [한국어] fopen/fread/fwrite/printf — 파일 I/O 및 진단 출력
#include <stdlib.h>  // [한국어] malloc/calloc/free/exit — 동적 메모리 및 프로세스 종료
#include <string.h>  // [한국어] strlen/strncmp/strstr — 문자열 비교 및 플랫폼 이름 탐색
#include <stdarg.h>  // [한국어] va_list/va_start/va_end — vmyexit의 가변인자 처리

/* [한국어] 로그 메시지 접두어 상수.
 * printf 출력 앞에 붙어 이 프로세스가 GPGPU-Sim 래퍼임을 식별하게 한다. */
#define PREAMBLE "GPGPU-Sim nvopencl_wrapper"

/*
 * [한국어]
 * vmyexit - 에러 메시지를 포맷하여 출력하고 필요시 프로세스를 종료한다.
 *
 * @code: 종료 코드. 0이면 종료하지 않고 메시지만 출력. 0이 아니면 exit(code) 호출.
 * @str:  printf 형식 에러 메시지 문자열.
 * @ap:   가변인자 리스트(va_list). 호출자 myexit()가 전달한다.
 * @return: void (종료 시 반환 없음)
 *
 * PREAMBLE 접두어를 붙여 표준 출력에 에러를 기록하고, code != 0이면 즉시 exit한다.
 * 에러 복구 없이 단순히 진단 후 종료하는 패턴이며, 실행 컨텍스트는 자식 프로세스이므로
 * abort 대신 exit를 사용해 부모 프로세스가 종료 코드를 확인할 수 있도록 한다.
 *
 * 호출 체인:
 *   myexit() → [vmyexit] → vprintf, exit()
 */
void vmyexit(int code, const char *str,va_list ap)
{
   char buffer[1024]; // [한국어] 포맷된 에러 메시지를 담을 임시 버퍼
   snprintf(buffer,1024,"%s: ERROR ** %s\n", PREAMBLE, str); // [한국어] PREAMBLE 접두어와 에러 문자열을 조합
   vprintf(buffer,ap); // [한국어] va_list를 사용해 printf 형식으로 출력
   fflush(stdout); // [한국어] 버퍼링된 출력을 즉시 플러시하여 부모 프로세스가 메시지를 볼 수 있게 함
   if( code ) // [한국어] code가 0이 아닌 경우에만 프로세스 종료
      exit(code); // [한국어] 지정된 종료 코드로 즉시 종료 — 부모(opencl_runtime_api.cc)가 system()의 반환값으로 감지
}

/*
 * [한국어]
 * myexit - vmyexit의 가변인자 래퍼. 에러 발생 지점에서 직접 호출된다.
 *
 * @code: 종료 코드(vmyexit 참조)
 * @str:  printf 형식 에러 메시지
 * @...:  포맷 인자들
 * @return: void
 *
 * 가변인자를 va_list로 변환한 뒤 vmyexit에 위임한다. main() 전체에서
 * OpenCL API 에러 체크 후 즉시 호출된다.
 *
 * 호출 체인:
 *   main() 내 에러 체크 → [myexit] → vmyexit → exit()
 */
void myexit(int code, const char *str, ... )
{
   va_list ap; // [한국어] 가변인자 리스트 선언
   va_start(ap,str); // [한국어] str 다음부터 가변인자 파싱 시작
   vmyexit(code,str,ap); // [한국어] 실제 출력 및 종료 로직 위임
   va_end(ap); // [한국어] va_list 정리 (exit 이후에 실행될 수도 있으므로 안전하게 포함)
}

/*
 * [한국어]
 * main - OpenCL 소스 파일을 컴파일하여 PTX를 포함하는 바이너리를 추출하는 핵심 로직.
 *
 * @argc: 인자 수. 최소 3개 필요: [0]=실행파일명, [1]=입력.cl, [2]=출력.ptx
 * @argv: 인자 배열. argv[1]=OpenCL 소스, argv[2]=PTX 출력 경로, argv[3..]=컴파일 옵션
 *        argv[1]=="-d"이면 디버그 모드 진입 후 argv를 한 칸 뒤로 이동.
 * @return: 0(성공) 또는 exit()로 비정상 종료(에러 코드)
 *
 * 이 함수는 GPGPU-Sim의 opencl_runtime_api.cc가 system() 호출로 실행시킨다.
 * NVIDIA 실 GPU 드라이버를 통해 OpenCL 소스를 컴파일하고, 컴파일된 바이너리에서
 * PTX(또는 CUBIN+PTX 혼합)를 추출하여 파일에 기록한다.
 * 이후 GPGPU-Sim은 이 파일을 읽어 PTX 기능 시뮬레이션에 사용한다.
 * 반드시 NVIDIA GPU와 드라이버(libOpenCL.so)가 있는 환경에서만 정상 동작한다.
 *
 * 실행 단계:
 *   1. 입력 .cl 파일을 읽어 소스 문자열로 로드
 *   2. 사용 가능한 OpenCL 플랫폼 중 NVIDIA 플랫폼 선택
 *   3. GPU 디바이스 목록 조회
 *   4. OpenCL 컨텍스트 생성
 *   5. clCreateProgramWithSource로 소스 등록
 *   6. clBuildProgram으로 NVIDIA 드라이버에 컴파일 요청
 *   7. clGetProgramInfo(BINARIES)로 컴파일된 바이너리 추출
 *   8. 첫 번째 디바이스 바이너리를 argv[2] 경로에 기록
 *
 * 호출 체인:
 *   opencl_runtime_api.cc::_cl_program::Build() [system()] → [main] → NVIDIA libOpenCL.so
 */
int main(int argc, const char **argv)
{
   cl_context context;   // [한국어] NVIDIA 드라이버에서 생성한 OpenCL 컨텍스트 핸들
   cl_program pgm;       // [한국어] 컴파일할 OpenCL 프로그램 객체
   cl_int errcode;       // [한국어] OpenCL API 호출 결과 코드 (CL_SUCCESS=0)
   cl_uint num_devices;  // [한국어] 발견된 GPU 디바이스 수

   bool debug=false; // [한국어] 디버그 모드 플래그. -d 인자로 활성화하면 상세 진단 출력

   printf("%s: command line = \'",PREAMBLE); // [한국어] 호출된 명령줄 전체를 로그에 기록 (재현성 확보)
   for( int i=0; i < argc; i++ ) { // [한국어] 모든 인자를 순서대로 출력
      printf("%s", argv[i]); // [한국어] i번째 인자 출력
      if( (i+1) < argc ) printf(" "); // [한국어] 마지막 인자가 아니면 공백 구분자 추가
   }
   printf("'\n"); // [한국어] 명령줄 로그 종료

   if( !strncmp(argv[1],"-d",2) ) { // [한국어] 첫 번째 인자가 "-d"로 시작하면 디버그 모드
      printf("nvopencl_wrapper started\n"); // [한국어] 디버그 시작 메시지
      fflush(stdout); // [한국어] 즉시 출력
      debug=true; // [한국어] 디버그 플래그 설정
      argv = argv+1; // [한국어] "-d" 인자를 소비하고 포인터를 한 칸 앞으로 이동
      argc--; // [한국어] 인자 수 감소 (입력 파일이 argv[1]이 되도록)
   }

   FILE *fp = fopen(argv[1],"r"); // [한국어] argv[1]로 지정된 OpenCL 소스 파일(.cl)을 읽기 모드로 열기
   if ( fp == NULL ) myexit(1,"Could not open file \'%s\'",argv[1]); // [한국어] 파일 열기 실패 시 종료
   if ( debug ) { printf("opened \'%s\'\n", argv[1]); fflush(stdout); } // [한국어] 디버그: 파일 열기 성공 확인
   fseek(fp,0,SEEK_END); // [한국어] 파일 끝으로 이동하여 크기 측정 준비
   size_t source_length = ftell(fp); // [한국어] 현재 위치(끝) = 파일 전체 바이트 수
   if ( source_length == 0 ) myexit(2,"OpenCL file is empty"); // [한국어] 빈 파일이면 컴파일 불가
   if ( debug ) { printf("file \'%s\' has length %zu bytes\n", argv[1], source_length); fflush(stdout); }
   char *source = (char*)calloc(source_length+1,1); // [한국어] 소스 문자열용 버퍼 할당 (+1은 null terminator)
   if ( source == 0 ) myexit(2,"Memory allocation failed"); // [한국어] 메모리 부족 체크
   fseek(fp,0,SEEK_SET); // [한국어] 파일 시작으로 되돌아가 실제 읽기 준비
   fread(source,1,source_length,fp); // [한국어] 소스 전체를 버퍼에 읽어들임 (null terminator는 calloc이 이미 0으로 초기화)
   if ( debug ) { printf( "read in file \'%s\'\n", argv[1] ); fflush(stdout); } // [한국어] 디버그: 읽기 완료 확인

   char buffer[1024]; // [한국어] 플랫폼 이름 등 문자열 조회 결과를 담을 임시 버퍼
   cl_uint num_platforms; // [한국어] 시스템에서 발견된 OpenCL 플랫폼 수 (NVIDIA, Intel, AMD 등이 각각 하나씩)
   cl_platform_id* platforms; // [한국어] 플랫폼 핸들 배열

   errcode = clGetPlatformIDs(0, NULL, &num_platforms); // [한국어] 1단계: 플랫폼 수만 먼저 조회
   if ( errcode != CL_SUCCESS ) myexit(1,"clGetPlatformaIDs returned %d",errcode); // [한국어] 플랫폼 조회 실패
   if ( num_platforms == 0 ) myexit(2,"No OpenCL platforms found"); // [한국어] OpenCL 플랫폼이 없으면 NVIDIA 드라이버 미설치
   platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id)); // [한국어] 플랫폼 핸들 배열 할당
   errcode = clGetPlatformIDs(num_platforms, platforms, NULL); // [한국어] 2단계: 실제 플랫폼 핸들 목록 가져오기
   if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformIDs returned %d",errcode); // [한국어] 목록 조회 실패
   unsigned use_platform = 0; // [한국어] 사용할 플랫폼 인덱스. 기본값 0 (플랫폼이 하나면 자동 선택)
   if (num_platforms > 1) { // [한국어] 플랫폼이 여러 개이면 NVIDIA 플랫폼을 명시적으로 탐색
      char platformName[1024]; // [한국어] 각 플랫폼 이름 조회용 버퍼
      printf("%s: Multiple OpenCL platforms found.  Searching for compatible platform...\n",PREAMBLE);
      for (unsigned p = 0; p < num_platforms; p++) { // [한국어] 모든 플랫폼 순회
         errcode = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, &platformName, NULL); // [한국어] p번 플랫폼 이름 조회
         if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformInfo returned %d",errcode);
         printf("%s:     OpenCL platform \'%s\'\n",PREAMBLE, platformName); // [한국어] 발견된 플랫폼 목록 출력
         if (strstr(platformName, "NVIDIA") != NULL) { // [한국어] 이름에 "NVIDIA"가 포함된 플랫폼 선택
            use_platform = p; // [한국어] NVIDIA 플랫폼 인덱스 기록 (마지막 NVIDIA가 선택됨)
         }
      }
   }
   errcode = clGetPlatformInfo(platforms[use_platform], CL_PLATFORM_NAME, 1024, &buffer, NULL); // [한국어] 선택된 플랫폼 이름 재확인
   if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformInfo returned %d",errcode);
   printf("%s: Generating PTX using OpenCL platform \'%s\'\n",PREAMBLE,buffer); // [한국어] 사용 중인 플랫폼 기록

   errcode = clGetDeviceIDs(platforms[use_platform], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices); // [한국어] GPU 디바이스 수 먼저 조회
   if ( errcode != CL_SUCCESS ) myexit(4,"clGetDeviceIDs returned %d",errcode);
   printf("%s: found %u native OpenCL devices\n",PREAMBLE,num_devices); // [한국어] 발견된 GPU 수 출력

   cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id) ); // [한국어] 디바이스 핸들 배열 할당
   errcode = clGetDeviceIDs(platforms[use_platform], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL); // [한국어] GPU 디바이스 핸들 목록 가져오기
   if ( errcode != CL_SUCCESS ) myexit(5,"clGetDeviceIDs returned %d",errcode);
   context = clCreateContext(0, num_devices, devices, NULL, NULL, &errcode); // [한국어] 모든 GPU 디바이스를 포함하는 OpenCL 컨텍스트 생성
   if ( errcode != CL_SUCCESS ) myexit(6,"clCreateContext returned %d",errcode);
   pgm = clCreateProgramWithSource(context, 1, (const char **)&source, &source_length, &errcode); // [한국어] 읽어온 소스 문자열로 프로그램 객체 생성 (아직 컴파일 전)
   if( errcode != CL_SUCCESS ) myexit(7,"clCreateProgramWithSource returned %d",errcode);

   char options[4096]; // [한국어] 컴파일 옵션 문자열 버퍼 (argc[3..] 인자들을 공백 구분으로 조합)
   unsigned n=0; // [한국어] options 버퍼에 쓴 바이트 수 추적
   options[0]=0; // [한국어] 옵션 문자열 초기화 (옵션이 없으면 빈 문자열)
   for ( int i=3; i < argc; i++ ) { // [한국어] argv[3]부터 컴파일 옵션으로 처리
      snprintf(options+n,4096-n," %s ", argv[i] ); // [한국어] 각 옵션을 공백으로 구분하여 추가
      n+= strlen(argv[i]); // [한국어] 옵션 문자열 길이만큼 오프셋 전진
      n+= 2; // [한국어] 앞뒤 공백 2개 반영
   }
   errcode = clBuildProgram(pgm, 0, NULL, options, NULL, NULL); // [한국어] NVIDIA 드라이버에 컴파일 요청. 0 devices=모든 디바이스. 내부적으로 PTX/SASS 생성
   if ( errcode != CL_SUCCESS ) { // [한국어] 컴파일 오류 처리
      printf("%s: clBuildProgram returned %d (error) -- build log:\n\n",PREAMBLE,errcode);
      size_t build_log_length=0; // [한국어] 빌드 로그 크기 조회를 위한 변수
      errcode = clGetProgramBuildInfo(pgm,devices[0],CL_PROGRAM_BUILD_LOG,0,NULL,&build_log_length); // [한국어] 1단계: 빌드 로그 크기만 조회
      if( errcode != CL_SUCCESS ) myexit(8,"clGetProgramBuildInfo returned %d",errcode);
      char *build_log = (char*)calloc(1,build_log_length); // [한국어] 빌드 로그 저장 버퍼 할당
      errcode = clGetProgramBuildInfo(pgm,devices[0],CL_PROGRAM_BUILD_LOG,build_log_length,
                                      build_log,&build_log_length); // [한국어] 2단계: 실제 빌드 로그 내용 가져오기
      printf("%s",build_log); // [한국어] 컴파일 에러 메시지를 그대로 출력
      printf("\n\n%s: end of build log\n", PREAMBLE);
      printf("%s: exiting early because the OpenCL code had errors (see above).\n", PREAMBLE);
      exit(8); // [한국어] 컴파일 실패 종료 코드 8로 종료 — 부모 프로세스가 감지
   }

   size_t nbytes1=0; // [한국어] clGetProgramInfo 반환 바이트 수 저장용 (사용하지 않으나 API 요구사항)
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_NUM_DEVICES,sizeof(cl_uint),&num_devices,&nbytes1); // [한국어] 프로그램에 연결된 디바이스 수 재확인
   if ( errcode != CL_SUCCESS ) myexit(9,"clGetProgramInfo returned %d",errcode);

   size_t nbytes2=0; // [한국어] 반환 바이트 수 저장용
   size_t *binary_sizes = (size_t*)calloc(num_devices,sizeof(size_t)); // [한국어] 각 디바이스별 컴파일된 바이너리 크기 배열
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARY_SIZES,sizeof(size_t)*num_devices,binary_sizes,&nbytes2); // [한국어] 각 디바이스의 바이너리 크기 조회 (디바이스마다 다른 SM arch)
   if ( errcode != CL_SUCCESS ) myexit(10,"clGetProgramInfo returned %d",errcode);

   unsigned char **binaries = (unsigned char**)calloc(num_devices,sizeof(unsigned char*)); // [한국어] 디바이스별 바이너리 포인터 배열
   size_t bytes_to_read = 0; // [한국어] 전체 읽을 바이트 수 합산용

   for (unsigned int i=0; i < num_devices; i++ ) { // [한국어] 디바이스마다 바이너리 버퍼 할당
      binaries[i] = (unsigned char*) calloc(binary_sizes[i],1); // [한국어] i번 디바이스의 바이너리 크기만큼 버퍼 할당
      bytes_to_read += binary_sizes[i]; // [한국어] 전체 읽기 바이트 누적
   }

   size_t nbytes3=0; // [한국어] 반환 바이트 수 저장용
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARIES,bytes_to_read,binaries,&nbytes3); // [한국어] 모든 디바이스의 컴파일된 바이너리 가져오기. binaries[i]에 PTX/CUBIN 포함
   if ( errcode != CL_SUCCESS ) myexit(11,"clGetProgramInfo returned %d",errcode);

   fp = fopen(argv[2],"w"); // [한국어] 출력 파일(argv[2])을 쓰기 모드로 열기
   fprintf(fp,"%s",binaries[0]); // [한국어] 첫 번째 디바이스(devices[0])의 바이너리를 문자열로 기록. PTX가 포함됨
   fclose(fp); // [한국어] 출력 파일 닫기 및 flush
   return 0; // [한국어] 정상 종료. 부모 opencl_runtime_api.cc가 system()의 0 반환을 성공으로 해석
}
