/*
 * [한국어 설명] PTX 어휘 분석기(Lexer) 정의 (ptx.l)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim이 CUDA PTX(Parallel Thread eXecution) 어셈블리 텍스트를
 * 읽어 들일 때 가장 먼저 동작하는 Flex(Lex) 어휘 분석기(lexer/scanner)를 정의한다.
 * PTX 소스 파일의 문자열 스트림을 OPCODE, IDENTIFIER, INT_OPERAND, REGISTER 등의
 * 토큰(token) 단위로 쪼개어, 이후 Bison(Yacc) 파서(ptx.y)가 문법 규칙에 따라
 * 처리할 수 있도록 준비한다. 각 규칙은 정규표현식과 매칭되면 해당 토큰 번호를
 * 반환하며, 일부 규칙은 yylval에 의미 값(정수/실수/문자열 포인터)을 함께 담는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA Application → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *   → gpgpu_context::init_parser()
 *       → ptx_lex_init() → [이 파일: ptx_lex()] → 토큰 스트림 생성
 *           → ptx.y (Bison 문법 파서) → ptx_recognizer (ptx_parser.cc)
 *               → function_info / symbol_table / ptx_instruction (ptx_ir)
 *                   → cuda-sim/ 기능 시뮬레이션 → gpgpu-sim/ 타이밍 시뮬레이션
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 초기화 단계에서 1회 수행.
 * 사이클-레벨 시뮬레이션과 무관한 전처리 단계이다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptx.tab.h (Bison이 생성한 토큰 번호 enum, YYSTYPE 정의),
 *   opcodes.h (ADD_OP, LD_OP 등 opcode enum), ptx_parser.h (ptx_recognizer 클래스),
 *   gpgpu_context.h (gpgpu_context 전역 상태)
 * - 이 파일에 의존하는 모듈: ptx.y (Bison 파서가 ptx_lex를 호출하여 토큰 소비),
 *   ptx_parser.cc (토큰을 받아 semantic action 수행)
 * - Flex/Bison 생성 산출물: ptx_lexer.c (또는 유사 이름)이 빌드 시 생성되며
 *   ptx_lex(), ptx_error(), ptx_get_lineno() 등의 구현이 들어간다.
 *
 * === 주요 함수/구조체 요약 ===
 * - ptx_lex(): Flex가 생성한 토큰화 루프 진입점. yylval에 토큰 값을 채우고
 *              토큰 번호를 반환한다.
 * - ptx_error(): Lexer/Bison에서 문법 오류 발생 시 소스 위치와 함께 에러 메시지를
 *                출력하고 파싱을 중단한다.
 * - %option prefix="ptx_": 모든 Flex 심볼에 ptx_ 접두사를 붙여 ptxinfo_lex 등
 *                          다른 lexer와 충돌하지 않게 한다.
 * - %option reentrant/bison-bridge: 재진입 가능한 스캐너를 생성하여 여러 파일을
 *                                   순차 파싱할 수 있게 한다.
 * - START 조건(IN_STRING, IN_COMMENT, IN_FUNC_DECL, NOT_OPCODE): 문자열/주석/
 *   함수 선언/오피코드 회피 등 컨텍스트에 따른 토큰화 상태 전이.
 */

/*
Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas, 
Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
The University of British Columbia, Northwestern University, Purdue University
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer;
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution;
3. Neither the names of The University of British Columbia, Northwestern 
   University nor the names of their contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


%option nounput
%option noyywrap
%option yylineno
%option prefix="ptx_"

%option bison-bridge
%option reentrant

%{
#include "opcodes.h"               // [한국어] ADD_OP, LD_OP, BRA_OP 등 PTX opcode enum 정의
#include "ptx_parser.h"            // [한국어] ptx_recognizer 클래스 및 파서 상태 선언
#include "ptx.tab.h"               // [한국어] Bison 생성 토큰 번호(RETURN_TOKEN)와 YYSTYPE 정의
#include <string.h>                // [한국어] strdup, strlen 등 문자열 함수
#include "../../libcuda/gpgpu_context.h"  // [한국어] 시뮬레이터 전역 컨텍스트

#define LINEBUF_SIZE (4*1024)      // [한국어] 현재 파싱 중인 소스 라인을 저장할 버퍼 크기 (4KB)
#define TC recognizer->col+=strlen(yytext);
                                   // [한국어] TC(Token Column): 현재 토큰 길이만큼 컬럼 번호 증가
#define CHECK_UNSIGNED \
	if( yytext[strlen(yytext)-1]=='U' ) { \
		printf("GPGPU-Sim: ERROR ** U modifier not implemented\n"); \
		abort(); \
	}
                                   // [한국어] U 접미사가 붙은 정수 리터럴은 아직 미지원; 발견 시 abort
#define YY_DECL int ptx_lex \
	       (YYSTYPE * yylval_param , yyscan_t yyscanner, ptx_recognizer* recognizer)
                                   // [한국어] ptx_lex 함수 선언: Bison-bridge + reentrant + recognizer 인자
int ptx_error( yyscan_t yyscanner, ptx_recognizer* recognizer, const char *s );
                                   // [한국어] Bison/Flex 에러 핸들러 전방 선언
%}

%s IN_STRING                      // [한국어] 문자열 리터럴("...") 처리용 시작 조건
%s IN_COMMENT                     // [한국어] /* ... */ 블록 주석 처리용 시작 조건
%x IN_FUNC_DECL                   // [한국어] .func 선언 중 함수 이름이 opcode와 겹칠 때 회피용
%x NOT_OPCODE                     // [한국어] opcode로 해석되면 안 되는 컨텍스트(예: call 뒤 함수명)
%%

abs	TC; yylval->int_value = ABS_OP; return OPCODE;
add	TC; yylval->int_value = ADD_OP; return OPCODE;
addp	TC; yylval->int_value = ADDP_OP; return OPCODE;
addc    TC; yylval->int_value = ADDC_OP; return OPCODE;
and	TC; yylval->int_value = AND_OP; return OPCODE;
andn	TC; yylval->int_value = ANDN_OP; return OPCODE;
atom	TC; yylval->int_value = ATOM_OP; return OPCODE;
bar.warp 	TC; yylval->int_value = NOP_OP; return OPCODE;  // [한국어] bar.warp은 warp 내 동기화로, 현재 nop로 매핑
bar 	TC; yylval->int_value = BAR_OP; return OPCODE;       // [한국어] bar: CTA 단위 배리어 동기화
barrier	TC; yylval->int_value = BAR_OP; return OPCODE;     // [한국어] barrier는 bar의 별칭
bfe     TC; yylval->int_value = BFE_OP; return OPCODE;      // [한국어] 비트 필드 추출
bfi     TC; yylval->int_value = BFI_OP; return OPCODE;      // [한국어] 비트 필드 삽입
bfind   TC; yylval->int_value = BFIND_OP; return OPCODE;    // [한국어] 선행 1 비트 찾기
bra     TC; yylval->int_value = BRA_OP; return OPCODE;      // [한국어] 분기(branch)
brx     TC; yylval->int_value = BRX_OP; return OPCODE;      // [한국어] 인덱스 분기
brev    TC; yylval->int_value = BREV_OP; return OPCODE;     // [한국어] 비트 역전
brkpt   TC; yylval->int_value = BRKPT_OP; return OPCODE;    // [한국어] 중단점

wmma	TC; yylval->int_value = MMA_OP; return OPCODE;        // [한국어] WMMA(Tensor Core) 명령어
wmma\.load	TC; yylval->int_value = MMA_LD_OP; return OPCODE;  // [한국어] WMMA 행렬 로드
wmma\.store 	TC; yylval->int_value = MMA_ST_OP; return OPCODE; // [한국어] WMMA 행렬 스토어

call	TC; BEGIN(NOT_OPCODE); yylval->int_value = CALL_OP; return OPCODE; // [한국어] call 뒤의 함수 이름이 opcode와 겹치지 않게 NOT_OPCODE 상태로 전환
callp   TC; BEGIN(NOT_OPCODE); yylval->int_value = CALLP_OP; return OPCODE; // [한국어] 조걶 부 call
clz	TC; yylval->int_value = CLZ_OP; return OPCODE;        // [한국어] 선행 0 개수 세기
cnot	TC; yylval->int_value = CNOT_OP; return OPCODE;      // [한국어] 비트 NOT의 논리 보수
cos	TC; yylval->int_value = COS_OP; return OPCODE;        // [한국어] 코사인(SFU)
cvt	TC; yylval->int_value = CVT_OP; return OPCODE;        // [한국어] 타입 변환
cvta	TC; yylval->int_value = CVTA_OP; return OPCODE;      // [한국어] 제네릭 주소 변환
div	TC; yylval->int_value = DIV_OP; return OPCODE;        // [한국어] 나눗셈
dp4a	TC; yylval->int_value = DP4A_OP; return OPCODE;      // [한국어] 4바이트 점적 덧셈-누적
ex2	TC; yylval->int_value = EX2_OP; return OPCODE;        // [한국어] 2의 거듭제곱(SFU)
exit	TC; yylval->int_value = EXIT_OP; return OPCODE;      // [한국어] 커널 종료
fma     TC; yylval->int_value = FMA_OP; return OPCODE;      // [한국어] 융합 곱셈-덧셈
isspacep TC; yylval->int_value = ISSPACEP_OP; return OPCODE; // [한국어] 포인터 주소 공간 판별
ld      TC; yylval->int_value = LD_OP; return OPCODE;       // [한국어] 메모리 로드
ld.volatile TC; yylval->int_value = LD_OP; return OPCODE;   // [한국어] volatile 로드 (휘발성 힌트)
ldu     TC; yylval->int_value = LDU_OP; return OPCODE;      // [한국어] uniform 로드
lg2	TC; yylval->int_value = LG2_OP; return OPCODE;        // [한국어] log2(SFU)
mad24   TC; yylval->int_value = MAD24_OP; return OPCODE;    // [한국어] 24비트 곱셈-덧셈
mad     TC; yylval->int_value = MAD_OP; return OPCODE;      // [한국어] 곱셈-덧셈
madc	TC; yylval->int_value = MADC_OP; return OPCODE;      // [한국어] 캐리 포함 mad
madp	TC; yylval->int_value = MADP_OP; return OPCODE;      // [한국어] predicated mad
max     TC; yylval->int_value = MAX_OP; return OPCODE;      // [한국어] 최댓값
membar  TC; yylval->int_value = MEMBAR_OP; return OPCODE;   // [한국어] 메모리 배리어
min     TC; yylval->int_value = MIN_OP; return OPCODE;      // [한국어] 최솟값
mov     TC; yylval->int_value = MOV_OP; return OPCODE;      // [한국어] 레지스터 이동
mul24   TC; yylval->int_value = MUL24_OP; return OPCODE;    // [한국어] 24비트 곱셈
mul     TC; yylval->int_value = MUL_OP; return OPCODE;      // [한국어] 곱셈
neg     TC; yylval->int_value = NEG_OP; return OPCODE;      // [한국어] 부정
nandn   TC; yylval->int_value = NANDN_OP; return OPCODE;    // [한국어] NAND NOT
norn    TC; yylval->int_value = NORN_OP; return OPCODE;     // [한국어] NOR NOT
not     TC; yylval->int_value = NOT_OP; return OPCODE;      // [한국어] 비트 NOT
or      TC; yylval->int_value = OR_OP; return OPCODE;       // [한국어] 비트 OR
orn     TC; yylval->int_value = ORN_OP; return OPCODE;      // [한국어] OR NOT
pmevent TC; yylval->int_value = PMEVENT_OP; return OPCODE;  // [한국어] 성능 모니터링 이벤트
popc    TC; yylval->int_value = POPC_OP; return OPCODE;     // [한국어] 1 비트 개수 세기
prefetch TC; yylval->int_value = PREFETCH_OP; return OPCODE; // [한국어] 프리페치
prefetchu TC; yylval->int_value = PREFETCHU_OP; return OPCODE; // [한국어] uniform 프리페치
prmt    TC; yylval->int_value = PRMT_OP; return OPCODE;     // [한국어] 퍼뮤트(바이트 재배열)
rcp	TC; yylval->int_value = RCP_OP; return OPCODE;        // [한국어] 역수(SFU)
red     TC; yylval->int_value = RED_OP; return OPCODE;      // [한국어] 메모리 리덕션
rem	TC; yylval->int_value = REM_OP; return OPCODE;        // [한국어] 나머지
ret	TC; yylval->int_value = RET_OP; return OPCODE;        // [한국어] 함수 복귀
retp     TC; yylval->int_value = RETP_OP; return OPCODE;    // [한국어] predicated return
rsqrt	TC; yylval->int_value = RSQRT_OP; return OPCODE;    // [한국어] 역제곱근(SFU)
sad     TC; yylval->int_value = SAD_OP; return OPCODE;      // [한국어] 절대차 누적
selp	TC; yylval->int_value = SELP_OP; return OPCODE;      // [한국어] 프레디케이트 선택
setp    TC; yylval->int_value = SETP_OP; return OPCODE;     // [한국어] 프레디케이트 설정
set	TC; yylval->int_value = SET_OP; return OPCODE;         // [한국어] 조걶 결과 설정
shfl	TC; yylval->int_value = SHFL_OP; return OPCODE;      // [한국어] warp 내 레지스터 셔플
shf	TC; yylval->int_value = SHF_OP; return OPCODE;         // [한국어] 비트 필드 시프트
shl     TC; yylval->int_value = SHL_OP; return OPCODE;      // [한국어] 왼쪽 시프트
shr     TC; yylval->int_value = SHR_OP; return OPCODE;      // [한국어] 오른쪽 시프트
sin	TC; yylval->int_value = SIN_OP; return OPCODE;        // [한국어] 사인(SFU)
slct	TC; yylval->int_value = SLCT_OP; return OPCODE;      // [한국어] 선택
sqrt	TC; yylval->int_value = SQRT_OP; return OPCODE;      // [한국어] 제곱근(SFU)
sst	TC; yylval->int_value = SST_OP; return OPCODE;        // [한국어] shared state?
ssy     TC; yylval->int_value = SSY_OP; return OPCODE;      // [한국어] 분기 동기화 설정
st      TC; yylval->int_value = ST_OP; return OPCODE;       // [한국어] 메모리 스토어
st.volatile TC; yylval->int_value = ST_OP; return OPCODE;   // [한국어] volatile 스토어
sub	TC; yylval->int_value = SUB_OP; return OPCODE;        // [한국어] 뺄셈
subc	TC; yylval->int_value = SUBC_OP; return OPCODE;      // [한국어] 캐리 포함 뺄셈
suld	TC; yylval->int_value = SULD_OP; return OPCODE;      // [한국어] surface 로드
sured	TC; yylval->int_value = SURED_OP; return OPCODE;    // [한국어] surface 리덕션
surst	TC; yylval->int_value = SUST_OP; return OPCODE;     // [한국어] surface 스토어
suq	TC; yylval->int_value = SUQ_OP; return OPCODE;        // [한국어] surface 쿼리
tex	TC; BEGIN(NOT_OPCODE); yylval->int_value = TEX_OP; return OPCODE; // [한국어] tex 뒤 텍스처 심볼이 opcode와 충돌하지 않게 전환
txq	TC; yylval->int_value = TEX_OP; return OPCODE;        // [한국어] 텍스처 쿼리
trap	TC; yylval->int_value = TRAP_OP; return OPCODE;      // [한국어] 트랩
vabsdiff TC; yylval->int_value = VABSDIFF_OP; return OPCODE; // [한국어] 벡터 절대차
vadd    TC; yylval->int_value = VADD_OP; return OPCODE;     // [한국어] 벡터 덧셈
vmad    TC; yylval->int_value = VMAD_OP; return OPCODE;     // [한국어] 벡터 곱셈-덧셈
vmax    TC; yylval->int_value = VMAX_OP; return OPCODE;     // [한국어] 벡터 최댓값
vmin    TC; yylval->int_value = VMIN_OP; return OPCODE;     // [한국어] 벡터 최솟값
vset    TC; yylval->int_value = VSET_OP; return OPCODE;     // [한국어] 벡터 비교 설정
vshl    TC; yylval->int_value = VSHL_OP; return OPCODE;     // [한국어] 벡터 왼쪽 시프트
vshr    TC; yylval->int_value = VSHR_OP; return OPCODE;     // [한국어] 벡터 오른쪽 시프트
vsub    TC; yylval->int_value = VSUB_OP; return OPCODE;     // [한국어] 벡터 뺄셈
vote	TC; yylval->int_value = VOTE_OP; return OPCODE;      // [한국어] warp 투표
activemask TC; yylval->int_value = ACTIVEMASK_OP; return OPCODE; // [한국어] 활성 마스크
xor     TC; yylval->int_value = XOR_OP; return OPCODE;      // [한국어] 비트 XOR
nop     TC; yylval->int_value = NOP_OP; return OPCODE;      // [한국어] no-op
break  TC; yylval->int_value = BREAK_OP; return OPCODE;     // [한국어] 루프 탈출
breakaddr  TC; yylval->int_value = BREAKADDR_OP; return OPCODE; // [한국어] break 주소

"CPTX_END"	printf("ENDING CUSTOM PTX.\n"); BEGIN(IN_COMMENT); // [한국어] 사용자 정의 PTX 끝 표시; 이후 주석 상태로 전환

<INITIAL,NOT_OPCODE,IN_FUNC_DECL>{
\.a\.sync\.aligned TC; yylval->int_value = LOAD_A; return WMMA_DIRECTIVE;   // [한국어] WMMA A 행렬 로드
\.b\.sync\.aligned TC; yylval->int_value = LOAD_B; return WMMA_DIRECTIVE;   // [한국어] WMMA B 행렬 로드
\.c\.sync\.aligned TC; yylval->int_value = LOAD_C; return WMMA_DIRECTIVE;   // [한국어] WMMA C 행렬 로드
\.d\.sync\.aligned TC; yylval->int_value = STORE_D; return WMMA_DIRECTIVE; // [한국어] WMMA D 행렬 스토어
\.mma\.sync\.aligned TC;yylval->int_value=MMA; return WMMA_DIRECTIVE;       // [한국어] WMMA 행렬 곱셈-누적

\.row TC; yylval->int_value = ROW; return LAYOUT;           // [한국어] WMMA 행 우선 레이아웃
\.col TC; yylval->int_value = COL; return LAYOUT;           // [한국어] WMMA 열 우선 레이아웃
\.m16n16k16\.global TC; yylval->int_value = M16N16K16; return CONFIGURATION; // [한국어] WMMA 16x16x16 전역 메모리
\.m32n8k16\.global TC; yylval->int_value = M32N8K16; return CONFIGURATION;   // [한국어] WMMA 32x8x16 전역 메모리
\.m8n32k16\.global TC;  yylval->int_value = M8N32K16; return CONFIGURATION;  // [한국어] WMMA 8x32x16 전역 메모리

\.m16n16k16\.shared TC; yylval->int_value = M16N16K16; return CONFIGURATION; // [한국어] WMMA 16x16x16 공유 메모리
\.m32n8k16\.shared TC; yylval->int_value = M32N8K16; return CONFIGURATION;   // [한국어] WMMA 32x8x16 공유 메모리
\.m8n32k16\.shared TC;  yylval->int_value = M8N32K16; return CONFIGURATION;  // [한국어] WMMA 8x32x16 공유 메모리

\.m16n16k16 TC; yylval->int_value = M16N16K16; return CONFIGURATION;        // [한국어] WMMA 16x16x16
\.m32n8k16 TC; yylval->int_value = M32N8K16; return CONFIGURATION;          // [한국어] WMMA 32x8x16
\.m8n32k16 TC;  yylval->int_value = M8N32K16; return CONFIGURATION;         // [한국어] WMMA 8x32x16

\.f4e TC;  return PRMT_F4E_MODE;   // [한국어] PRMT Forward 4-extract 모드
\.b4e TC;  return PRMT_B4E_MODE;   // [한국어] PRMT Backward 4-extract 모드
\.rc8 TC;  return PRMT_RC8_MODE;   // [한국어] PRMT Replicate 8 모드
\.ecl TC;  return PRMT_ECL_MODE;   // [한국어] PRMT Edge clamp left 모드
\.ecr TC;  return PRMT_ECR_MODE;   // [한국어] PRMT Edge clamp right 모드
\.rc16 TC; return PRMT_RC16_MODE;  // [한국어] PRMT Replicate 16 모드

\.align TC; return ALIGN_DIRECTIVE;            // [한국어] 정렬 지시어
\.branchtargets TC; return BRANCHTARGETS_DIRECTIVE; // [한국어] 분기 대상 지시어
\.byte	TC; return BYTE_DIRECTIVE; /* not in PTX 2.1 */ // [한국어] 바이트 데이터 지시어(PTX 2.1 미포함)
\.callprototype TC; return CALLPROTOTYPE_DIRECTIVE;   // [한국어] 호출 프로토타입
\.calltargets TC; return CALLTARGETS_DIRECTIVE;       // [한국어] 호출 대상 지시어
\.const\[[0-9]+\] TC; yylval->int_value = atoi(yytext+7); return CONST_DIRECTIVE; // [한국어] 상수 메모리 뱅크 지정(.const[N])
\.const TC; yylval->int_value = 0; return CONST_DIRECTIVE; // [한국어] 기본 상수 메모리
\.entry TC; return ENTRY_DIRECTIVE;            // [한국어] 커널 엔트리 지시어
\.extern TC; return EXTERN_DIRECTIVE;          // [한국어] 외부 선언 지시어
\.file	 TC; BEGIN(INITIAL); return FILE_DIRECTIVE; // [한국어] 소스 파일 지시어; INITIAL 상태로 복귀
\.func   TC; BEGIN(IN_FUNC_DECL); return FUNC_DIRECTIVE; // [한국어] 함수 선언 시작; 함수 이름이 opcode와 겹치는 경우 회피
\.global TC; return GLOBAL_DIRECTIVE;          // [한국어] 전역 메모리 지시어
\.global.volatile TC; return GLOBAL_DIRECTIVE; //TODO: fix this! // [한국어] volatile 전역(임시 처리)
\.local  TC; return LOCAL_DIRECTIVE;           // [한국어] 로컬 메모리 지시어
\.loc    TC; return LOC_DIRECTIVE;             // [한국어] 소스 위치 지시어
\.maxnctapersm TC; return MAXNCTAPERSM_DIRECTIVE; // [한국어] SM당 최대 CTA 수
\.maxnreg TC; return MAXNNREG_DIRECTIVE;       // [한국어] 최대 레지스터 수
\.maxntid TC; return MAXNTID_DIRECTIVE;        // [한국어] CTA당 최대 스레드 수
\.minnctapersm TC; return MINNCTAPERSM_DIRECTIVE; // [한국어] SM당 최소 CTA 수
\.param  TC; return PARAM_DIRECTIVE;           // [한국어] 파라미터 지시어
\.pragma TC; return PRAGMA_DIRECTIVE;          // [한국어] pragma 지시어
\.reg    TC; return REG_DIRECTIVE;             // [한국어] 레지스터 지시어
\.reqntid TC; return REQNTID_DIRECTIVE;        // [한국어] 요구 스레드 수
\.section TC; return SECTION_DIRECTIVE;        // [한국어] 섹션 지시어
\.shared  TC; return SHARED_DIRECTIVE;         // [한국어] 공유 메모리 지시어
\.sreg   TC; return SREG_DIRECTIVE;            // [한국어] 특수 레지스터 지시어
\.sstarr TC; return SSTARR_DIRECTIVE;          // [한국어] sstarr 메모리 지시어
\.struct TC; return STRUCT_DIRECTIVE;          // [한국어] 구조체 지시어
\.surf   TC; return SURF_DIRECTIVE;   /* not in PTX 2.1 */ // [한국어] surface 메모리
\.target  TC; return TARGET_DIRECTIVE;         // [한국어] 타겟 아키텍처 지시어
\.tex   TC; BEGIN(NOT_OPCODE); return TEX_DIRECTIVE; // [한국어] tex 지시어; 뒤 텍스처 이름이 opcode와 겹칠 수 있음
\.union  TC; return UNION_DIRECTIVE; /* not in PTX 2.1 */ // [한국어] 공용체 지시어
\.version TC; return VERSION_DIRECTIVE;        // [한국어] PTX 버전 지시어
\.visible TC; return VISIBLE_DIRECTIVE;        // [한국어] visible 링크 속성
\.weak TC; return WEAK_DIRECTIVE;              // [한국어] weak 링크 속성
\.address_size TC; return ADDRESS_SIZE_DIRECTIVE; // [한국어] 주소 크기 지시어
\.weak	TC; return WEAK_DIRECTIVE;             // [한국어] weak 지시어(탭 변형)

\.constptr TC; return CONSTPTR_DIRECTIVE; /* Ptx plus directive for pointer to constant memory */
                                          // [한국어] 상수 메모리 포인터 지시어(PTX Plus)
\.ptr TC; return PTR_DIRECTIVE; /* Added for new OpenCL genrated code */
                                // [한국어] 포인터 지시어(OpenCL 생성 코드용)

"%clock" TC; yylval->int_value = CLOCK_REG; return SPECIAL_REGISTER;              // [한국어] 클럭 레지스터
"%halfclock" TC; yylval->int_value = HALFCLOCK_ID; return SPECIAL_REGISTER;       // [한국어] 반 클럭 레지스터
"%clock64" TC; yylval->int_value = CLOCK64_REG; return SPECIAL_REGISTER;          // [한국어] 64비트 클럭
"%ctaid" TC; yylval->int_value = CTAID_REG; return SPECIAL_REGISTER;              // [한국어] CTA ID
"%envreg"[0-9]+ TC; sscanf(yytext+7,"%u",&yylval->int_value); yylval->int_value<<=16; yylval->int_value += ENVREG_REG; return SPECIAL_REGISTER; // [한국어] 환경 레지스터
"%gridid" TC; yylval->int_value = GRIDID_REG; return SPECIAL_REGISTER;            // [한국어] 그리드 ID
"%laneid"  TC; yylval->int_value = LANEID_REG; return SPECIAL_REGISTER;           // [한국어] warp 내 레인 ID
"%lanemask_eq"  TC; yylval->int_value = LANEMASK_EQ_REG; return SPECIAL_REGISTER; // [한국어] 레인 마스크 equal
"%lanemask_le"  TC; yylval->int_value = LANEMASK_LE_REG; return SPECIAL_REGISTER; // [한국어] 레인 마스크 less-equal
"%lanemask_lt"  TC; yylval->int_value = LANEMASK_LT_REG; return SPECIAL_REGISTER; // [한국어] 레인 마스크 less-than
"%lanemask_ge"  TC; yylval->int_value = LANEMASK_GE_REG; return SPECIAL_REGISTER; // [한국어] 레인 마스크 greater-equal
"%lanemask_gt"  TC; yylval->int_value = LANEMASK_GT_REG; return SPECIAL_REGISTER; // [한국어] 레인 마스크 greater-than
"%nctaid" TC; yylval->int_value = NCTAID_REG; return SPECIAL_REGISTER;            // [한국어] 전체 CTA 수
"%ntid"  TC; yylval->int_value = NTID_REG; return SPECIAL_REGISTER;               // [한국어] CTA당 스레드 수
"%nsmid"  TC; yylval->int_value = NSMID_REG; return SPECIAL_REGISTER;             // [한국어] SM 수
"%nwarpid"  TC; yylval->int_value = NWARPID_REG; return SPECIAL_REGISTER;         // [한국어] warp ID 수
"%pm"[0-3]  TC; sscanf(yytext+3,"%u",&yylval->int_value); yylval->int_value<<=16; yylval->int_value += PM_REG; return SPECIAL_REGISTER; // [한국어] 성능 모니터링 레지스터
"%smid"  TC; yylval->int_value = SMID_REG; return SPECIAL_REGISTER;               // [한국어] SM ID
"%tid"  TC; yylval->int_value = TID_REG; return SPECIAL_REGISTER;                 // [한국어] 스레드 ID
"%warpid"  TC; yylval->int_value = WARPID_REG; return SPECIAL_REGISTER;           // [한국어] warp ID
"WARP_SZ"  TC; yylval->int_value = WARPSZ_REG; return SPECIAL_REGISTER;           // [한국어] warp 크기 상수

[a-zA-Z_][a-zA-Z0-9_$]*  TC; yylval->string_value = strdup(yytext); return IDENTIFIER; // [한국어] 일반 식별자(변수/레이블/함수명)
[$%][a-zA-Z0-9_$]+  TC; yylval->string_value = strdup(yytext); return IDENTIFIER;     // [한국어] %r0, %p1 같은 레지스터 식별자

[0-9]+\.[0-9]+ 	 TC; sscanf(yytext,"%lf", &yylval->double_value); return DOUBLE_OPERAND; // [한국어] 10진수 double 리터럴
		
0[xX][0-9a-fA-F]+U? TC; CHECK_UNSIGNED; sscanf(yytext,"%x", &yylval->int_value); return INT_OPERAND; // [한국어] 16진수 정수 리터럴
0[0-7]+U?   	TC; printf("GPGPU-Sim: ERROR ** parsing octal not (yet) implemented\n"); abort(); return INT_OPERAND; // [한국어] 8진수 미지원
0[bB][01]+U?  	TC; printf("GPGPU-Sim: ERROR ** parsing binary not (yet) implemented\n"); abort(); return INT_OPERAND; // [한국어] 2진수 미지원
[-]?[0-9]+U?    TC; CHECK_UNSIGNED; yylval->int_value =  atoi(yytext); return INT_OPERAND; // [한국어] 10진수 부호 있는 정수 리터럴

0[fF][0-9a-fA-F]{8}  TC; sscanf(yytext+2,"%x", (unsigned*)(void*)&yylval->float_value); return FLOAT_OPERAND;    // [한국어] 16진수 표현 float 리터럴
0[dD][0-9a-fA-F]{16}  TC; sscanf(yytext+2,"%Lx", (unsigned long long*)(void*)&yylval->double_value); return DOUBLE_OPERAND; // [한국어] 16진수 표현 double 리터럴

\.s8   TC;  return S8_TYPE;    // [한국어] 8비트 부호 있는 정수 타입
\.s16  TC;  return S16_TYPE;   // [한국어] 16비트 부호 있는 정수 타입
\.s32  TC;  return S32_TYPE;   // [한국어] 32비트 부호 있는 정수 타입
\.s64  TC;  return S64_TYPE;   // [한국어] 64비트 부호 있는 정수 타입
\.u8   TC;  return U8_TYPE;    // [한국어] 8비트 부호 없는 정수 타입
\.u16  TC;  return U16_TYPE;   // [한국어] 16비트 부호 없는 정수 타입
\.u32  TC;  return U32_TYPE;   // [한국어] 32비트 부호 없는 정수 타입
\.u64  TC;  return U64_TYPE;   // [한국어] 64비트 부호 없는 정수 타입
\.f16  TC;  return F16_TYPE;   // [한국어] 16비트 반정밀도 부동소수점
\.f16x2  TC;  return F16_TYPE; /* TODO: figure out what this should really be */
                               // [한국어] f16x2는 f16으로 처리(임시)
\.f32  TC;  return F32_TYPE;   // [한국어] 32비트 단정밀도 부동소수점
\.f64  TC;  return F64_TYPE;   // [한국어] 64비트 배정밀도 부동소수점
\.ff64 TC;  return FF64_TYPE;  // [한국어] 대체 f64 타입
\.b8   TC;  return B8_TYPE;    // [한국어] 8비트 비트 타입
\.b16  TC;  return B16_TYPE;   // [한국어] 16비트 비트 타입
\.b32  TC;  return B32_TYPE;   // [한국어] 32비트 비트 타입
\.b64  TC;  return B64_TYPE;   // [한국어] 64비트 비트 타입
\.bb64  TC;  return BB64_TYPE; // [한국어] 64비트 블록 비트 타입
\.bb128  TC;  return BB128_TYPE; // [한국어] 128비트 블록 비트 타입
\.pred TC;  return PRED_TYPE;  // [한국어] 프레디케이트 타입

\.texref TC; BEGIN(NOT_OPCODE); return TEXREF_TYPE;   // [한국어] 텍스처 참조 타입
\.samplerref TC;  return SAMPLERREF_TYPE;             // [한국어] 샘플러 참조 타입
\.surfref TC;  return SURFREF_TYPE;                   // [한국어] surface 참조 타입

\.v2	TC; return V2_TYPE;      // [한국어] 2원소 벡터 타입
\.v3	TC; return V3_TYPE;      // [한국어] 3원소 벡터 타입
\.v4	TC; return V4_TYPE;      // [한국어] 4원소 벡터 타입

\.half  TC; return HALF_OPTION; /* ptxplus */         // [한국어] half 옵션(PTX Plus)
\.cc	TC; return EXTP_OPTION; /* extended precision option */ // [한국어] 확장 정밀도 옵션

\.equ	TC; return EQU_OPTION;   // [한국어] compare: equal unordered
\.neu	TC; return NEU_OPTION;   // [한국어] compare: not-equal unordered
\.ltu	TC; return LTU_OPTION;   // [한국어] compare: less-than unordered
\.leu	TC; return LEU_OPTION;   // [한국어] compare: less-equal unordered
\.gtu	TC; return GTU_OPTION;   // [한국어] compare: greater-than unordered
\.geu	TC; return GEU_OPTION;   // [한국어] compare: greater-equal unordered
\.num	TC; return NUM_OPTION;   // [한국어] compare: numeric
\.nan	TC; return NAN_OPTION;   // [한국어] compare: NaN

\.sat	TC; return SAT_OPTION;   // [한국어] saturate 옵션

\.l		TC; return LEFT_OPTION;  // [한국어] 왼쪽 시프트 옵션
\.r		TC; return RIGHT_OPTION; // [한국어] 오른쪽 시프트 옵션

\.eq    TC; return EQ_OPTION;   // [한국어] compare: equal
\.ne    TC; return NE_OPTION;   // [한국어] compare: not equal
\.lt    TC; return LT_OPTION;   // [한국어] compare: less than
\.le    TC; return LE_OPTION;   // [한국어] compare: less equal
\.gt    TC; return GT_OPTION;   // [한국어] compare: greater than
\.ge    TC; return GE_OPTION;   // [한국어] compare: greater equal
\.cf    TC; return CF_OPTION;   // [한국어] carry flag 옵션
\.sf    TC; return SF_OPTION;   // [한국어] sign flag 옵션
\.nsf    TC; return NSF_OPTION; // [한국어] no sign flag 옵션

\.lo    TC; return LO_OPTION;   // [한국어] low 16비트 옵션
\.ls    TC; return LS_OPTION;   // [한국어] lower-or-same 옵션
\.hi    TC; return HI_OPTION;   // [한국어] high 16비트 옵션
\.hs    TC; return HS_OPTION;   // [한국어] higher-or-same 옵션


\.rni	TC; return RNI_OPTION;   // [한국어] 정수 반올림: nearest even
\.rzi	TC; return RZI_OPTION;   // [한국어] 정수 반올림: zero
\.rmi	TC; return RMI_OPTION;   // [한국어] 정수 반올림: minus infinity
\.rpi	TC; return RPI_OPTION;   // [한국어] 정수 반올림: plus infinity

\.rn	TC; return RN_OPTION;    // [한국어] 부동소수점 반올림: nearest even
\.rz	TC; return RZ_OPTION;    // [한국어] 부동소수점 반올림: zero
\.rm	TC; return RM_OPTION;    // [한국어] 부동소수점 반올림: minus infinity
\.rp	TC; return RP_OPTION;    // [한국어] 부동소수점 반올림: plus infinity

\.ftz	TC; return FTZ_OPTION;   // [한국어] flush-to-zero 옵션

\.neg	TC; return NEG_OPTION;   // [한국어] negate 옵션

\.wide  TC; return WIDE_OPTION; // [한국어] wide 결과 옵션
\.uni	TC; return UNI_OPTION;   // [한국어] uniform 분기 옵션

\.sync TC; return SYNC_OPTION;  // [한국어] 동기화 옵션
\.arrive TC; return ARRIVE_OPTION; // [한국어] arrive 옵션
\.red TC; return RED_OPTION;    // [한국어] reduction 옵션

\.clamp	TC; return CLAMP_OPTION; // [한국어] clamp 주소 모드
\.wrap	TC; return WRAP_OPTION;  // [한국어] wrap 주소 모드

\.approx TC; return APPROX_OPTION; // [한국어] 근사 연산 옵션
\.full  TC; return FULL_OPTION;    // [한국어] full range 옵션

\.any	TC; return ANY_OPTION;   // [한국어] warp vote any
\.all	TC; return ALL_OPTION;   // [한국어] warp vote all
\.ballot	TC; return BALLOT_OPTION; // [한국어] warp ballot
\.gl    TC; return GLOBAL_OPTION;   // [한국어] global scope 메모리 배리어
\.cta   TC; return CTA_OPTION;      // [한국어] CTA scope 메모리 배리어
\.sys   TC; return SYS_OPTION;      // [한국어] system scope 메모리 배리어

\.exit   TC; return EXIT_OPTION;    // [한국어] exit 수식어

\.abs   TC; return ABS_OPTION;      // [한국어] absolute value 옵션

\.to	TC; return TO_OPTION;         // [한국어] cvta .to 옵션

\.ca    TC; return CA_OPTION;       // [한국어] 캐시 all levels
\.cg    TC; return CG_OPTION;       // [한국어] 캐시 global
\.cs    TC; return CS_OPTION;       // [한국어] 캐시 streaming
\.lu    TC; return LU_OPTION;       // [한국어] 캐시 last use
\.cv    TC; return CV_OPTION;       // [한국어] 캐시 volatile

\.wb	TC; return WB_OPTION;        // [한국어] write-back 캐시 정책
\.wt	TC; return WT_OPTION;        // [한국어] write-through 캐시 정책

\.nc	TC; return NC_OPTION;        // [한국어] non-coherent 캐시

\.up	TC; return UP_OPTION;        // [한국어] shfl up
\.down	TC; return DOWN_OPTION;    // [한국어] shfl down
\.bfly	TC; return BFLY_OPTION;    // [한국어] shfl butterfly
\.idx	TC; return IDX_OPTION;      // [한국어] shfl indexed

\.popc	TC; return ATOMIC_POPC;    // [한국어] atomic population count
\.and	TC; return ATOMIC_AND;      // [한국어] atomic AND
\.or	TC; return ATOMIC_OR;        // [한국어] atomic OR
\.xor	TC; return ATOMIC_XOR;      // [한국어] atomic XOR
\.cas	TC; return ATOMIC_CAS;      // [한국어] atomic compare-and-swap
\.exch	TC; return ATOMIC_EXCH;    // [한국어] atomic exchange
\.add	TC; return ATOMIC_ADD;      // [한국어] atomic add
\.inc	TC; return ATOMIC_INC;      // [한국어] atomic increment
\.dec	TC; return ATOMIC_DEC;      // [한국어] atomic decrement
\.min	TC; return ATOMIC_MIN;      // [한국어] atomic min
\.max	TC; return ATOMIC_MAX;      // [한국어] atomic max



\.1d	TC; return GEOM_MODIFIER_1D; // [한국어] 1D 텍스처/서피스 지오메트리
\.2d	TC; return GEOM_MODIFIER_2D; // [한국어] 2D 텍스처/서피스 지오메트리
\.3d	TC; return GEOM_MODIFIER_3D; // [한국어] 3D 텍스처/서피스 지오메트리

\.0	TC; yylval->int_value = 0; return DIMENSION_MODIFIER; // [한국어] 차원 0
\.1	TC; yylval->int_value = 1; return DIMENSION_MODIFIER; // [한국어] 차원 1
\.2	TC; yylval->int_value = 2; return DIMENSION_MODIFIER; // [한국어] 차원 2
\.x	TC; yylval->int_value = 0; return DIMENSION_MODIFIER; // [한국어] x 차원
\.y	TC; yylval->int_value = 1; return DIMENSION_MODIFIER; // [한국어] y 차원
\.z	TC; yylval->int_value = 2; return DIMENSION_MODIFIER; // [한국어] z 차원

"-"	TC; return MINUS;          // [한국어] 마이너스 기호
"+"	TC; return PLUS;           // [한국어] 플러스 기호
","     TC; return COMMA;        // [한국어] 쉼표
"@" 	TC; return PRED;           // [한국어] 프레디케이트 접두사
"|" 	TC; return PIPE;          // [한국어] 파이프(벡터 구분자)
"["     TC; return LEFT_SQUARE_BRACKET;  // [한국어] 왼쪽 대괄호(주소)
"]"	TC; return RIGHT_SQUARE_BRACKET; // [한국어] 오른쪽 대괄호
"<"     TC; return LEFT_ANGLE_BRACKET;   // [한국어] 왼쪽 꺽쇠(벡터/제네릭)
">"	TC; return RIGHT_ANGLE_BRACKET;  // [한국어] 오른쪽 꺽쇠
"("	TC; return LEFT_PAREN;     // [한국어] 왼쪽 소괄호
")"	TC; return RIGHT_PAREN;    // [한국어] 오른쪽 소괄호
":"	TC; BEGIN(INITIAL); return COLON;  // [한국어] 콜론; INITIAL 상태로 복귀
";"	TC; BEGIN(INITIAL); return SEMI_COLON; // [한국어] 세미콜론; 명령어 종료 후 INITIAL 복귀
"!"	TC; return EXCLAMATION;    // [한국어] 느낌표(부정)
"=" 	TC; return EQUALS;        // [한국어] 등호
"}"	TC; return RIGHT_BRACE;    // [한국어] 오른쪽 중괄호
\.  	TC; return PERIOD;        // [한국어] 마침표
"/"	TC; return BACKSLASH;     // [한국어] 슬래시

"//"[^\n]* TC;	// eat single
                // [한국어] 한 줄 주석("//")은 무시

\n.*  recognizer->col=0; strncpy(recognizer->linebuf, yytext + 1, LINEBUF_SIZE); yyless( 1 );
                // [한국어] 줄바꿈 시 컬럼 초기화 및 다음 줄을 linebuf에 복사; yyless(1)로 '\n'만 소비

" " TC;          // [한국어] 공백 무시
"\t" TC;         // [한국어] 탭 무시


}

<IN_FUNC_DECL>"{"	TC; BEGIN(INITIAL); return LEFT_BRACE; // [한국어] 함수 본문 시작 중괄호; 이후부터는 opcode 해석 허용
<INITIAL,NOT_OPCODE>"{"	TC; return LEFT_BRACE; // [한국어] 벡터 피연산자 시작; 다음 토큰은 opcode로 해석되면 안 됨

<INITIAL>{
"/*"  BEGIN(IN_COMMENT); // [한국어] 블록 주석 시작
}
<IN_COMMENT>{
"*/"      BEGIN(INITIAL); // [한국어] 블록 주석 종료
"CPTX_BEGIN" printf("BEGINNING CUSTOM PTX.\n"); BEGIN(INITIAL); // [한국어] 커스텀 PTX 시작 마커
[^C*\n]+	// eat comment in chunks
                // [한국어] 주석 덩어리 무시
"C"			// eat the lone C 
                // [한국어] 단독 'C' 무시
"*"			// eat the lone star
                // [한국어] 단독 '*' 무시
\n        TC; 
                // [한국어] 주석 내 줄바꿈 처리
}

<INITIAL>{
"\""  BEGIN(IN_STRING); // [한국어] 문자열 리터럴 시작
}
<IN_STRING>{
"\"" 	TC; BEGIN(INITIAL); return STRING; // [한국어] 문자열 리터럴 종료
[^\"]*	TC; yylval->string_value = strdup(yytext); // [한국어] 문자열 내용 저장
}

<*>\t@@DWARF.*\n // [한국어] DWARF 디버그 정보 라인 무시

<INITIAL,NOT_OPCODE,IN_FUNC_DECL>.  TC; ptx_error(yyscanner, recognizer, (const char*)NULL);
                                   // [한국어] 위 규칙에 매칭되지 않는 문자는 문법 오류로 처리
%%

int ptx_error( yyscan_t yyscanner, ptx_recognizer* recognizer, const char *s )
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	int i;
	recognizer->g_error_detected = 1;
	fflush(stdout);
	if( s != NULL )
		printf("%s:%u Syntax error:\n\n", recognizer->gpgpu_ctx->g_filename, yylineno );
	printf("   %s\n", recognizer->linebuf );
	printf("   ");
	for( i=0; i < recognizer->col-1; i++ ) {
		if( recognizer->linebuf[i] == '\t' ) printf("\t");
		else printf(" ");
	}
			
	printf("^\n\n");
	fflush(stdout);
	//exit(1);
	return 0;
}
