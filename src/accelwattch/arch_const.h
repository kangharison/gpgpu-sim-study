/*****************************************************************************
 *                                McPAT
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/

/*
 * [한국어 설명] McPAT 물리 아키텍처 기본 상수 정의 (arch_const.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT 전력 모델이 사용하는 프로세서 물리 아키텍처의 기본(default) 상수
 * 전체를 정의한다. 프로세스 노드 크기(90nm), 클럭 속도, 캐시 용량, 레지스터 파일
 * 크기, 분기 예측기 파라미터, 크로스바 포트 수, CMOS 출력 드라이버 트랜지스터 폭
 * 등이 포함된다. 이 값들은 XML 설정 파일(mcpat.xml 등)이 로드될 때 XML에 명시된
 * 값으로 덮어써지므로, 실제 시뮬레이션에서는 이 상수들이 직접 사용되는 일은 드물다.
 * AccelWattch가 GPU 전력 추정에 McPAT를 활용할 때 이 상수들이 초기값 기반을 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch는 GPGPU-Sim의 전력 모델 서브시스템으로, 타이밍 시뮬레이터(gpgpu-sim/)가
 * 매 사이클마다 수집한 하드웨어 카운터(activity counter)를 McPAT에 전달해 동적/누설
 * 전력을 추정한다. 이 파일은 McPAT의 최하위 물리 상수 레이어로, McPAT 내부의 CACTI
 * SRAM 모델 및 파이프라인 면적/전력 계산 전체가 여기에 정의된 상수를 출발점으로 사용한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 시뮬레이션 초기화 단계(parse_cfg → McPAT init)
 * 에서 한 번 참조되며, 사이클-레벨 루프에서는 직접 접근하지 않는다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: 없음 (이 파일 자체가 최하위 상수 레이어)
 * - 이 파일에 의존하는 모듈: accelwattch/ 내 McPAT 컴포넌트 전체
 *   (processor.cc, core.cc, array.cc, cacti/*.cc 등)가 이 헤더를 include한다.
 * - 데이터 흐름: XML 파싱 전 기본값으로 사용 → XML 파서(xmlParser.*)가 덮어쓴 뒤
 *   McPAT ParseXML::parse()가 각 컴포넌트 초기화 시 전달.
 * - 공유 구조체: array_inputs — CACTI SRAM 모델에 캐시 지오메트리를 넘기는 간단한 POD.
 *
 * === 주요 함수/구조체 요약 ===
 * - array_inputs: SRAM 지오메트리 파라미터(capacity/assoc/blocksize)를 CACTI에 전달하는 POD 구조체
 * - archi_F_sz_nm=90.0: McPAT 기본 프로세스 노드 크기 (90nm HP CMOS)
 * - CLOCKRATE=1.2GHz: McPAT 기본 클럭 주파수 (XML로 덮어씌움)
 * - AF=0.5: 동적 전력 계산 시 사용되는 기본 활동 인자(Activity Factor, 스위칭 확률)
 * - Woutdrv*: CMOS 출력 드라이버 트랜지스터 폭 상수 — 90nm 노드 스케일 적용값(단위: µm)
 * - 분기 예측기 파라미터 (BTBEntries, globalHistoryBits 등): 기본 CPU 타겟(Niagara/O3) 기준값
 */

#ifndef ARCH_CONST_H_
#define ARCH_CONST_H_

/*
 * [한국어] array_inputs — CACTI SRAM 지오메트리 파라미터 전달용 POD 구조체.
 * CACTI는 캐시/레지스터 파일 등 SRAM 배열의 면적·전력·레이턴시를 모델링하는 툴이다.
 * 이 구조체를 통해 capacity/assoc/blocksize 세 가지 핵심 지오메트리 파라미터가 전달된다.
 */
typedef struct {
  unsigned int capacity;
  /* [한국어] SRAM 배열의 총 용량 (단위: 바이트).
   * 설정자: McPAT 초기화 시 각 컴포넌트(캐시, 레지스터 파일 등)가 직접 채워 넣음.
   * 읽는 자: CACTI 모델 함수가 면적·전력 추정 시 참조.
   * 값 범위: 0 초과 정수; 실제 캐시 크기에 따라 수백~수백만 바이트.
   * 동기화: 초기화 단계에서 단일 스레드가 설정하므로 별도 락 불필요. */

  unsigned int assoc;  // fully
  /* [한국어] 캐시 연관도(associativity). 0이면 완전 연관(fully associative)을 의미.
   * 설정자: McPAT 컴포넌트 초기화 경로(core.cc 등)에서 설정.
   * 읽는 자: CACTI가 태그 배열 크기 및 비교기 회로 전력 계산에 사용.
   * 값 범위: 0(완전 연관), 1(직접 매핑), 2/4/8/16(셋 연관).
   * 동기화: 초기화 후 읽기 전용으로 사용; 멀티스레드 접근 없음. */

  unsigned int blocksize;
  /* [한국어] 캐시 블록(라인) 크기 (단위: 바이트).
   * 설정자: 각 캐시 레벨 파라미터 초기화 시 설정 (예: L1=32B, L2=64B).
   * 읽는 자: CACTI가 데이터 배열 행 너비 및 감지 증폭기 수 계산에 사용.
   * 값 범위: 일반적으로 4~256 바이트 (2의 거듭제곱).
   * 동기화: 초기화 후 읽기 전용. */
} array_inputs;

// Do Not change, unless you want to bypass the XML interface and do not care
// about the default values. Global parameters
// [한국어] 아래 전역 파라미터들은 XML 설정 파일로 덮어씌워지는 기본값이다.
// XML 없이 직접 빌드/테스트할 때만 이 값들이 실제로 사용된다.
const int number_of_cores = 8;     // [한국어] 기본 CPU 코어 수 (Niagara T1 기준 8코어)
const int number_of_L2s = 1;       // [한국어] 공유 L2 캐시 뱅크 수 (기본값 1개)
const int number_of_L3s = 1;       // [한국어] L3 캐시 뱅크 수 (기본값 1개; 없으면 0으로 덮어씀)
const int number_of_NoCs = 1;      // [한국어] 온칩 네트워크(NoC) 수 (기본값 1개)

const double archi_F_sz_nm = 90.0; // [한국어] 기본 프로세스 노드 크기 (nm). 90nm HP CMOS 파라미터 세트 선택에 사용됨
const unsigned int dev_type = 0;   // [한국어] 디바이스 타입: 0=고성능(HP), 1=저전력(LSTP), 2=LOP; CMOS 문턱전압·누설 파라미터 결정
const double CLOCKRATE = 1.2 * 1e9; // [한국어] 기본 클럭 속도 1.2GHz; 동적 전력(P = α·C·V²·f)의 f 항에 해당
const double AF = 0.5;             // [한국어] 활동 인자(Activity Factor): 논리 게이트의 평균 스위칭 확률 (0~1). 0.5는 랜덤 데이터 가정
// const bool 			inorder			=	true;
const bool embedded = false;  // NEW  // [한국어] 임베디드 프로세서 모드 여부; false=데스크탑/서버급 CMOS 파라미터 사용

const bool homogeneous_cores = true;   // [한국어] 모든 코어가 동일한 구성인지 여부 (true=동종 코어 → 단일 코어 계산 후 곱셈으로 확장)
const bool temperature = 360;         // [한국어] 동작 온도 (켈빈, K). 360K≈87°C; 누설 전류 온도 의존성 계산에 사용
const int number_cache_levels = 3;    // [한국어] 캐시 계층 수 (L1/L2/L3 = 3)
const int L1_property = 0;  // private 0; coherent 1, shared 2.
                              // [한국어] L1 캐시 속성: 0=private(코어 전용), 1=coherent, 2=shared
const int L2_property = 2;  // [한국어] L2 캐시 속성: 2=shared (모든 코어가 공유)
const bool homogeneous_L2s = true;  // [한국어] 모든 L2 뱅크가 동일 구성인지 여부
const bool L3_property = 2;         // [한국어] L3 캐시 속성: 2=shared
const bool homogeneous_L3s = true;  // [한국어] 모든 L3 뱅크가 동일 구성인지 여부
const double Max_area_deviation = 50;     // [한국어] 면적 최적화 목표 대비 최대 허용 편차(%) — McPAT 최적화 루프 수렴 기준
const double Max_dynamic_deviation = 50;  // New // [한국어] 동적 전력 최적화 목표 대비 최대 허용 편차(%) — 추가된 수렴 기준
const int opt_dynamic_power = 1;  // [한국어] 동적 전력 최적화 활성화 플래그 (1=활성; McPAT 최적화 루프 대상)
const int opt_lakage_power = 0;   // [한국어] 누설(leakage) 전력 최적화 활성화 플래그 (0=비활성; 오탈자: "lakage"→"leakage")
const int opt_area = 0;           // [한국어] 면적 최적화 활성화 플래그 (0=비활성)
const int interconnect_projection_type = 0; // [한국어] 인터커넥트 RC 지연 모델 타입: 0=국제반도체기술로드맵(ITRS) 기반, 1=보수적 예측

//******************************Core Parameters
// [한국어] 코어 파이프라인 파라미터 — inorder/OoO(비순서 실행) 모드에 따라 다른 값 사용.
// inorder=true이면 Niagara T1 스타일 순서 실행 코어 파라미터,
// inorder=false이면 OoO(Out-of-Order) 코어 파라미터(레지스터 재명명 폭이 더 넓음).
#if (inorder)
const int opcode_length = 8;        // Niagara // [한국어] 순서 실행: 연산 코드(opcode) 비트 폭 — Niagara T1은 8비트
const int reg_length = 5;           // Niagara // [한국어] 순서 실행: 레지스터 파일 인덱스 비트 수 — 2^5=32개 논리 레지스터
const int instruction_length = 32;  // Niagara // [한국어] 명령어 길이 (비트) — SPARC/Niagara는 고정 32비트 명령어
const int data_width = 64;                     // [한국어] 데이터 버스 폭 (비트) — 64비트 주소/데이터 경로
#else
const int opcode_length = 8;        // 16;//Niagara // [한국어] OoO 실행: opcode 길이 8비트 (원래 x86은 더 길지만 McPAT 모델에서 8 사용)
const int reg_length = 7;           // Niagara // [한국어] OoO 실행: 물리 레지스터 인덱스 비트 수 — 2^7=128개 물리 레지스터 (재명명 후)
const int instruction_length = 32;  // Niagara // [한국어] 명령어 길이 32비트 (OoO에서도 동일)
const int data_width = 64;                     // [한국어] 데이터 경로 폭 64비트
#endif

// Caches
// itlb
// [한국어] ITLB(Instruction Translation Lookaside Buffer): 명령어 가상→물리 주소 변환 캐시
const int itlbsize = 512;      // [한국어] ITLB 전체 용량 (바이트 단위; 실제로는 항목(entry) 수×항목 크기)
const int itlbassoc = 0;  // fully // [한국어] ITLB 연관도: 0=완전 연관(fully associative)
const int itlbblocksize = 8;   // [한국어] ITLB 항목(블록) 크기 (바이트)

// icache
// [한국어] 명령어 캐시(I-Cache) 파라미터
const int icachesize = 32768;  // [한국어] I-Cache 크기: 32KB (32768 바이트)
const int icacheassoc = 4;     // [한국어] I-Cache 4-way 셋 연관
const int icacheblocksize = 32; // [한국어] I-Cache 블록 크기: 32바이트 (8개 SPARC 명령어)

// dtlb
// [한국어] DTLB(Data Translation Lookaside Buffer): 데이터 가상→물리 주소 변환 캐시
const int dtlbsize = 512;      // [한국어] DTLB 용량 (바이트)
const int dtlbassoc = 0;  // fully // [한국어] DTLB 연관도: 0=완전 연관
const int dtlbblocksize = 8;   // [한국어] DTLB 항목 크기 (바이트)

// dcache
// [한국어] 데이터 캐시(D-Cache) 파라미터
const int dcachesize = 32768;        // [한국어] D-Cache 크기: 32KB
const int dcacheassoc = 4;           // [한국어] D-Cache 4-way 셋 연관
const int dcacheblocksize = 32;      // [한국어] D-Cache 블록 크기: 32바이트
const int dcache_write_buffers = 8;  // [한국어] D-Cache 라이트 버퍼 수: 8개 — 스토어 미스 시 MSHR 없이 임시 보관

// cache controllers
// IB,
// [한국어] IB(Instruction Buffer): 프론트엔드가 패치한 명령어를 디코드 전에 보관하는 버퍼
const int numIBEntries = 64;  // [한국어] IB 항목 수: 64개 명령어 슬롯
const int IBsize = 64;  // 2*4*instruction_length/8*2;
                         // [한국어] IB 총 용량 (바이트); 실제 계산값과 주석이 다름 — 단순 기본값으로 사용
const int IBassoc = 0;  // In Niagara it is still fully associ
                         // [한국어] IB 연관도: 0=완전 연관 (Niagara는 완전 연관 IB 사용)
const int IBblocksize = 4; // [한국어] IB 블록 크기: 4바이트 (1 SPARC 명령어)

// IFB and MIL should have the same parameters CAM
// [한국어] IFB(Instruction Fill Buffer): 캐시 미스 시 채움 요청을 추적하는 CAM 기반 버퍼.
// MIL(Miss In Progress List)과 동일한 파라미터를 사용해야 한다.
const int IFBsize = 128;  //  // [한국어] IFB 용량: 128바이트 (CAM 항목들의 집합)
const int IFBassoc = 0;   // In Niagara it is still fully associ
                           // [한국어] IFB 연관도: 0=완전 연관 CAM
const int IFBblocksize = 4; // [한국어] IFB 블록(항목) 크기: 4바이트

const int icache_write_buffers = 8; // [한국어] I-Cache 라이트 버퍼 수 (명령어 패치 미스 처리용)

// register file RAM
// [한국어] 레지스터 파일 SRAM 파라미터 — GPU의 대규모 레지스터 파일과 달리 CPU용 소규모 RF
const int regfilesize = 5760;    // [한국어] 레지스터 파일 크기: 5760바이트 (SPARC Niagara 윈도우 레지스터 포함)
const int regfileassoc = 1;      // [한국어] 레지스터 파일 연관도: 1=직접 매핑 (레지스터 인덱스로 직접 접근)
const int regfileblocksize = 18; // [한국어] 레지스터 파일 블록 크기: 18바이트 (레지스터 1개 항목 크기)

// regwin  RAM
// [한국어] SPARC 레지스터 윈도우(Register Window) RAM: 함수 호출 시 빠른 레지스터 스위칭을 위한 구조
const int regwinsize = 256;    // [한국어] 레지스터 윈도우 크기: 256바이트
const int regwinassoc = 1;     // [한국어] 레지스터 윈도우 연관도: 1=직접 매핑
const int regwinblocksize = 8; // [한국어] 레지스터 윈도우 블록 크기: 8바이트

// store buffer, lsq
// [한국어] LSQ(Load-Store Queue): OoO 실행에서 메모리 순서 보장을 위한 로드/스토어 큐
const int lsqsize = 512;    // [한국어] LSQ 용량: 512바이트
const int lsqassoc = 0;     // [한국어] LSQ 연관도: 0=완전 연관 (모든 항목 동시 검색 필요)
const int lsqblocksize = 8; // [한국어] LSQ 항목 크기: 8바이트 (주소+데이터+플래그)

// data fill queue RAM
// [한국어] DFQ(Data Fill Queue): 캐시 미스 응답 데이터를 캐시로 채워 넣기 전 임시 보관
const int dfqsize = 1024;     // [한국어] DFQ 용량: 1024바이트
const int dfqassoc = 1;       // [한국어] DFQ 연관도: 1=직접 매핑
const int dfqblocksize = 16;  // [한국어] DFQ 블록 크기: 16바이트

// outside the cores
// L2 cache bank
// [한국어] L2 캐시 뱅크 파라미터 — 코어 외부의 공유 2차 캐시
const int l2cachesize = 262144;  // [한국어] L2 캐시 크기: 262144바이트 = 256KB
const int l2cacheassoc = 16;     // [한국어] L2 캐시 16-way 셋 연관
const int l2cacheblocksize = 64; // [한국어] L2 캐시 블록 크기: 64바이트 (현대 CPU 표준 캐시라인 크기)

// L2 directory
// [한국어] L2 디렉토리: 멀티코어 캐시 코히런스를 위한 캐시 소유권 추적 디렉토리
const int l2dirsize = 1024;    // [한국어] L2 디렉토리 크기: 1024바이트
const int l2dirassoc = 0;      // [한국어] L2 디렉토리 연관도: 0=완전 연관 (모든 항목 동시 조회 필요)
const int l2dirblocksize = 2;  // [한국어] L2 디렉토리 항목 크기: 2바이트

// crossbar
// PCX
// [한국어] PCX(Processor-Cache Crossbar): 코어에서 L2 캐시로 가는 방향의 크로스바 스위치 (Sun Niagara 용어)
const int PCX_NUMBER_INPUT_PORTS_CROSSBAR = 8;          // [한국어] PCX 입력 포트 수: 8개 (코어 수와 동일)
const int PCX_NUMBER_OUTPUT_PORTS_CROSSBAR = 9;         // [한국어] PCX 출력 포트 수: 9개 (L2 뱅크 8 + I/O 1)
const int PCX_NUMBER_SIGNALS_PER_PORT_CROSSBAR = 144;   // [한국어] PCX 포트당 신호 수: 144비트 폭 버스

// PCX buffer RAM
// [한국어] PCX 버퍼: 크로스바 입력 중재 전 요청을 임시 보관하는 버퍼
const int pcx_buffersize = 1024;    // [한국어] PCX 버퍼 크기: 1024바이트
const int pcx_bufferassoc = 1;      // [한국어] PCX 버퍼 연관도: 1=직접 매핑
const int pcx_bufferblocksize = 32; // [한국어] PCX 버퍼 블록 크기: 32바이트
const int pcx_numbuffer = 5;        // [한국어] PCX 버퍼 수: 5개 (입력 채널별)

// pcx arbiter
// [한국어] PCX 중재기(Arbiter): 여러 코어의 동시 캐시 요청을 중재하는 로직
const int pcx_arbsize = 128;    // [한국어] PCX 중재기 버퍼 크기: 128바이트
const int pcx_arbassoc = 1;     // [한국어] PCX 중재기 연관도: 1=직접 매핑
const int pcx_arbblocksize = 2; // [한국어] PCX 중재기 블록 크기: 2바이트 (중재 상태 1항목)
const int pcx_numarb = 5;       // [한국어] PCX 중재기 수: 5개

// CPX
// [한국어] CPX(Cache-Processor Crossbar): L2 캐시에서 코어로 응답이 돌아오는 방향의 크로스바 (PCX 반대 방향)
const int CPX_NUMBER_INPUT_PORTS_CROSSBAR = 5;          // [한국어] CPX 입력 포트 수: 5개 (L2 뱅크 수)
const int CPX_NUMBER_OUTPUT_PORTS_CROSSBAR = 8;         // [한국어] CPX 출력 포트 수: 8개 (코어 수와 동일)
const int CPX_NUMBER_SIGNALS_PER_PORT_CROSSBAR = 150;   // [한국어] CPX 포트당 신호 수: 150비트 폭 버스

// CPX buffer RAM
// [한국어] CPX 버퍼: L2 응답 데이터를 코어로 전달하기 전 임시 보관
const int cpx_buffersize = 1024;    // [한국어] CPX 버퍼 크기: 1024바이트
const int cpx_bufferassoc = 1;      // [한국어] CPX 버퍼 연관도: 1=직접 매핑
const int cpx_bufferblocksize = 32; // [한국어] CPX 버퍼 블록 크기: 32바이트
const int cpx_numbuffer = 8;        // [한국어] CPX 버퍼 수: 8개 (출력 채널별)

// cpx arbiter
// [한국어] CPX 중재기: L2에서 여러 코어로의 동시 응답 전달을 중재
const int cpx_arbsize = 128;    // [한국어] CPX 중재기 버퍼 크기: 128바이트
const int cpx_arbassoc = 1;     // [한국어] CPX 중재기 연관도: 1=직접 매핑
const int cpx_arbblocksize = 2; // [한국어] CPX 중재기 블록 크기: 2바이트
const int cpx_numarb = 8;       // [한국어] CPX 중재기 수: 8개

// [한국어] 물리 레지스터 파일 및 재순서화 버퍼(ROB) 파라미터 — OoO 실행 코어용
const int numPhysFloatRegs = 256; // [한국어] 물리 부동소수점 레지스터 수: 256개 (재명명 풀)
const int numPhysIntRegs = 32;    // [한국어] 물리 정수 레지스터 수: 32개
const int numROBEntries = 192;    // [한국어] ROB(Re-Order Buffer) 항목 수: 192개 — OoO 실행 깊이를 결정
const int umRobs = 1;             // [한국어] ROB 수: 1개 (단일 스레드 기준; SMT에서는 분할됨)

// [한국어] 분기 예측기(Branch Predictor) 파라미터 — TAGE/하이브리드 예측기 구성
const int BTBEntries = 4096;           // [한국어] BTB(Branch Target Buffer) 항목 수: 4096개 — 분기 목적지 주소 캐시
const int BTBTagSize = 16;             // [한국어] BTB 태그 크기: 16비트 (태그 비교로 BTB 히트 판별)
const int LFSTSize = 1024;             // [한국어] LFST(Load Forwarding Status Table) 크기: 1024항목 — 메모리 의존성 예측
const int LQEntries = 32;              // [한국어] 로드 큐(Load Queue) 항목 수: 32개
const int RASSize = 16;                // [한국어] RAS(Return Address Stack) 크기: 16개 — 함수 반환 주소 예측
const int SQEntries = 32;              // [한국어] 스토어 큐(Store Queue) 항목 수: 32개
const int SSITSize = 1024;             // [한국어] SSIT(Store Set Identification Table) 크기: 1024항목 — 메모리 순서 위반 예측
const int activity = 0;                // [한국어] 활동(activity) 카운터 기본값: 0 (시뮬레이션 시작 전 초기화)
const int backComSize = 5;             // [한국어] 백 커뮤니케이션(완료 → 프론트엔드 통지) 버스 폭: 5
const int cachePorts = 200;            // [한국어] 캐시 포트 수: 200 (병렬 접근 허용 수; 사실상 무제한에 가까운 기본값)
const int choiceCtrBits = 2;           // [한국어] 선택 예측기 카운터 비트 수: 2비트 포화 카운터 (로컬/전역 예측기 중 하나 선택)
const int choicePredictorSize = 8192;  // [한국어] 선택 예측기 테이블 크기: 8192항목 (하이브리드 예측기의 메타 예측기)

// [한국어] 파이프라인 폭 파라미터 — 슈퍼스칼라 코어의 각 단계 발행/처리 폭
const int commitWidth = 8;   // [한국어] 커밋 폭: 사이클당 최대 8개 명령어 완료(정수 순서 커밋)
const int decodeWidth = 8;   // [한국어] 디코드 폭: 사이클당 최대 8개 명령어 디코딩
const int dispatchWidth = 8; // [한국어] 디스패치 폭: 사이클당 최대 8개 명령어를 실행 유닛으로 디스패치
const int fetchWidth = 8;    // [한국어] 패치 폭: 사이클당 최대 8개 명령어를 I-Cache에서 패치
const int issueWidth = 1;    // [한국어] 발행 폭: 사이클당 최대 1개 명령어 발행 (순서 실행 기준; OoO에서는 더 큼)
const int renameWidth = 8;   // [한국어] 재명명 폭: 사이클당 최대 8개 명령어의 레지스터 재명명 처리
// what is this forwardComSize=5??

// [한국어] 전역 분기 예측기 파라미터 (gshare/tournament 전역 히스토리 컴포넌트)
const int globalCtrBits = 2;           // [한국어] 전역 예측기 포화 카운터 비트 수: 2비트 (strongly taken/weakly taken/weakly not taken/strongly not taken)
const int globalHistoryBits = 13;      // [한국어] 전역 분기 히스토리 레지스터(GHR) 길이: 13비트 → 2^13=8192 항목 인덱싱
const int globalPredictorSize = 8192;  // [한국어] 전역 예측기 PHT(Pattern History Table) 크기: 8192항목

// [한국어] 로컬 분기 예측기 파라미터 (per-PC 히스토리 기반)
const int localCtrBits = 2;              // [한국어] 로컬 예측기 포화 카운터 비트 수: 2비트
const int localHistoryBits = 11;         // [한국어] 로컬 히스토리 레지스터(LHR) 길이: 11비트 → 2048 패턴 구분
const int localHistoryTableSize = 2048;  // [한국어] 로컬 히스토리 테이블 크기: 2048항목 (각 PC별 과거 11비트 히스토리 저장)
const int localPredictorSize = 2048;     // [한국어] 로컬 예측기 PHT 크기: 2048항목

/*
 * [한국어] CMOS 출력 드라이버 트랜지스터 폭 상수 (Woutdrv*)
 *
 * McPAT은 출력 드라이버(output driver)의 상승/하강 시간을 계산하기 위해
 * NAND/NOR 게이트 기반 드라이버 트랜지스터 폭(W)을 필요로 한다.
 * 이 값들은 90nm 노드 기준의 이상적인 트랜지스터 폭(단위: µm)이며,
 * 계수 0.09는 90nm 스케일링 팩터(LSCALE)이다 — 괄호 안의 원래 값은
 * 1µm 기준 상대값(LSCALE 미적용)이며, 0.09를 곱해 절대 µm 단위로 변환된다.
 *
 * 회로 구성: NAND 기반 드라이버 → Woutdrvnand{n,p}
 *            NOR 기반 드라이버  → Woutdrvnor{n,p}
 *            최종 출력 인버터  → Woutdriver{n,p}
 * n/p 접미사: n=NMOS 트랜지스터, p=PMOS 트랜지스터
 */
const double Woutdrvnandn = 30 * 0.09;    //(24.0 * LSCALE)
// [한국어] NAND 드라이버의 NMOS 폭: 30×0.09=2.7µm (90nm 스케일 적용; 원래 24.0×LSCALE)
const double Woutdrvnandp = 12.5 * 0.09;  //(10.0 * LSCALE)
// [한국어] NAND 드라이버의 PMOS 폭: 12.5×0.09=1.125µm (PMOS는 전자 이동도가 낮아 NMOS보다 더 넓어야 하나, 드라이버 단계별 비율이 다름)
const double Woutdrvnorn = 7.5 * 0.09;    //(6.0 * LSCALE)
// [한국어] NOR 드라이버의 NMOS 폭: 7.5×0.09=0.675µm (NOR 게이트는 NMOS 직렬 구조로 NAND보다 NMOS 폭이 작아도 됨)
const double Woutdrvnorp = 50 * 0.09;     //	(40.0 * LSCALE)
// [한국어] NOR 드라이버의 PMOS 폭: 50×0.09=4.5µm (NOR의 PMOS는 병렬 구조로 드라이빙 강도를 맞추기 위해 매우 넓어야 함)
const double Woutdrivern = 60 * 0.09;     //(48.0 * LSCALE)
// [한국어] 최종 출력 인버터의 NMOS 폭: 60×0.09=5.4µm (최종 단은 큰 부하를 구동하므로 트랜지스터 폭이 가장 큼)
const double Woutdriverp = 100 * 0.09;    //(80.0 * LSCALE)
// [한국어] 최종 출력 인버터의 PMOS 폭: 100×0.09=9.0µm (PMOS는 전자 이동도 약 절반이므로 NMOS보다 ~1.67× 더 넓게 설계)

/*
smtCommitPolicy=RoundRobin
smtFetchPolicy=SingleThread
smtIQPolicy=Partitioned
smtIQThreshold=100
smtLSQPolicy=Partitioned
smtLSQThreshold=100
smtNumFetchingThreads=1
smtROBPolicy=Partitioned
smtROBThreshold=100
squashWidth=8
*/

/*
prefetch_access=false
prefetch_cache_check_push=true
prefetch_data_accesses_only=false
prefetch_degree=1
prefetch_latency=10000
prefetch_miss=false
prefetch_past_page=false
prefetch_policy=none
prefetch_serial_squash=false
prefetch_use_cpu_id=true
prefetcher_size=100
prioritizeRequests=false
repl=Null


split=false
split_size=0
subblock_size=0
tgts_per_mshr=20
trace_addr=0
two_queue=false

cpu_side=system.cpu0.dcache_port
mem_side=system.tol2bus.port[2]
*/

//[system.cpu0.dtb]
// type=AlphaDT

#endif /* ARCH_CONST_H_ */
