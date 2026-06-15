/*****************************************************************************
 *                                McPAT/CACTI
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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/

/*
 * [한국어 설명] CACTI 독립 실행형 커맨드라인 진입점 (main.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 CACTI(Cache Access and Cycle Time Information) 도구를 독립 실행형
 * 커맨드라인 프로그램으로 사용할 때의 진입점을 제공한다. CACTI는 주어진 캐시
 * 설계 파라미터(용량, 연관도, 블록 크기, 기술 노드 등)로부터 접근 지연(access
 * time), 동적/정적 전력 소모(power), 면적(area)을 추정하는 전력·면적·타이밍
 * 분석 도구이다. 이 파일은 `main()` 함수 단 하나만 포함하며, 커맨드라인 인수를
 * 파싱하여 `cacti_interface()`를 적절한 오버로드로 호출하는 역할을 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim/AccelWattch 통합 시뮬레이션 환경에서 이 파일은 **실행되지 않는다**.
 * AccelWattch가 캐시 전력을 추정할 때는 io.cc 내의 `cacti_interface(InputParameter*)`
 * 를 직접 호출하므로, 이 main.cc는 GPGPU-Sim 빌드·실행 흐름과 완전히 독립적이다.
 * 이 파일은 오직 CACTI를 단독 바이너리로 빌드하여 스탠드얼론(standalone) 도구로
 * 사용할 때만 의미가 있다. 실행 컨텍스트는 호스트 유저스페이스 단일 프로세스이며,
 * 멀티스레드·GPU 디바이스 코드와 무관하다.
 * 호출 체인: 사용자 셸 → [main()] → cacti_interface() → CACTI 내부 탐색/계산 엔진
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈: io.h / io.cc — `cacti_interface()`의 세 가지 오버로드 선언을
 *   포함한다. (1) `cacti_interface(string)`: 설정 파일 경로를 받아 파일을 파싱한
 *   뒤 계산 수행, (2) `cacti_interface(int, int, ..., 52개 인수)`: CACTI 6.5
 *   형식 커맨드라인 파라미터를 직접 받는 레거시 인터페이스, (3)
 *   `cacti_interface(int, int, ..., 54개 인수)`: McPAT 형식으로 `search_ports`
 *   파라미터가 8번째 위치에 추가된 확장 인터페이스.
 * 이 파일에 의존하는 모듈: 없음 (진입점이므로 상위 호출자 없음).
 * 데이터 흐름: 커맨드라인 인수(문자열) → 정수/실수 변환 → `cacti_interface()` 입력
 *   → `uca_org_t` 결과 반환 (access_time, power, area 포함) → cleanup() 해제.
 * 공유 자료구조: `uca_org_t` (io.h 정의) — data_array2, tag_array2 포인터를
 *   내부적으로 동적 할당하므로, 사용 후 반드시 cleanup()으로 해제해야 한다.
 *
 * === 주요 함수/구조체 요약 ===
 * main(argc, argv) — 유일한 함수. 커맨드라인 인수 수(argc)에 따라 세 가지
 *   호출 경로(설정 파일 / CACTI 6.5 레거시 52인수 / McPAT 54인수) 중 하나를
 *   선택하여 cacti_interface()를 호출하고, 결과 메모리를 해제한 뒤 종료한다.
 * uca_org_t (io.h) — cacti_interface()가 반환하는 결과 구조체. access_time(ns),
 *   power(W), area(mm²) 필드와 동적 할당된 data_array2/tag_array2 포인터를 갖는다.
 *   cleanup() 멤버 함수가 이 두 포인터를 delete하여 메모리 누수를 방지한다.
 *
 * === AccelWattch XML / gpgpusim.config 연동 ===
 * main.cc는 AccelWattch/GPGPU-Sim 런타임에서 직접 실행되지 않는다.
 * 그러나 main()이 호출하는 `cacti_interface(string)`은 XML/구성 파일 형태로
 * 작성된 CACTI 입력 파일(.cfg)을 파싱한다. AccelWattch는 gpgpusim.config의
 * --power_config_name <xml> 옵션을 통해 XML을 지정하며, 그 XML의 파라미터들이
 * cacti_interface(InputParameter*) 호출 경로로 매핑되어 GPU 캐시 전력 추정에
 * 사용된다.
 */

#include "io.h"      // [한국어] cacti_interface() 오버로드 선언 및 uca_org_t 정의를 포함하는 CACTI 핵심 헤더
#include <iostream>  // [한국어] cerr(표준 에러 출력)을 사용하기 위한 C++ 표준 I/O 헤더

using namespace std; // [한국어] std::cerr, std::string, std::endl 등을 네임스페이스 한정자 없이 사용하기 위한 선언


/*
 * [한국어]
 * main - CACTI 독립 실행형 커맨드라인 진입점
 *
 * @argc: 커맨드라인 인수의 개수 (프로그램 이름 포함). 53(52개 파라미터, CACTI 6.5
 *        형식), 55(54개 파라미터, McPAT 형식), 또는 그 외의 값(설정 파일 모드)을
 *        가질 수 있다.
 * @argv: 커맨드라인 인수 문자열 배열. argv[0]은 프로그램 이름, argv[1]부터
 *        실제 파라미터가 시작된다.
 * @return: 정상 종료 시 0. 인수가 부적절하고 설정 파일도 지정되지 않은 경우
 *          exit(1)로 비정상 종료하므로 호출자에게 반환되지 않는다.
 *
 * 이 함수는 CACTI 캐시 설계 파라미터를 커맨드라인으로 전달받아 전력·면적·타이밍
 * 분석을 수행하는 독립 실행형 진입점이다. GPGPU-Sim/AccelWattch 통합 환경에서는
 * 이 함수가 절대 호출되지 않으며, AccelWattch는 io.cc의 cacti_interface()를
 * 라이브러리 형태로 직접 호출한다.
 *
 * 동작 과정:
 *   1단계 — 인수 개수(argc) 확인: 53도 55도 아니면 설정 파일(-infile) 모드로 진입.
 *   2단계 — 설정 파일 모드: argv를 순회하여 "-infile" 플래그와 그 다음 인수(파일명)를
 *            탐색한다. 발견하면 cacti_interface(string) 오버로드를 호출한다.
 *            발견 못 하면 사용법을 cerr에 출력하고 exit(1)로 종료한다.
 *   3단계 — CACTI 6.5 레거시 모드(argc==53, 파라미터 52개): argv[1]~argv[52]를
 *            atoi/atof로 변환하여 cacti_interface의 52인수 오버로드를 호출한다.
 *            argv[10]만 atof(실수: 기술 노드 크기 등)이고 나머지는 모두 atoi(정수).
 *   4단계 — McPAT 모드(argc==55, 파라미터 54개): argv[1]~argv[54]를 변환하여
 *            54인수 오버로드를 호출한다. 6.5 형식과의 차이점은 8번째 위치에
 *            search_ports 파라미터가 삽입되어, 부동소수점 인수가 argv[9]로 밀린다.
 *   5단계 — 결과 정리: result.cleanup()으로 data_array2/tag_array2 동적 메모리 해제.
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 싱글스레드, GPU·커널 코드와 무관.
 * 호출자: 사용자 셸(독립 실행형 CACTI 바이너리).
 * 호출 대상: cacti_interface() (io.cc), result.cleanup() (io.cc 내 uca_org_t 멤버).
 * 에러 처리: 인수 불일치 시 cerr에 사용법을 출력하고 exit(1)로 즉시 종료.
 *
 * 호출 체인:
 *   사용자 셸 → [main(argc, argv)] → cacti_interface(string 또는 52/54개 인수)
 *                                  → result.cleanup()
 *                                  → return 0
 */
int main(int argc,char *argv[])
{

  uca_org_t result; // [한국어] cacti_interface()의 반환값을 담을 결과 구조체 — access_time(접근 지연), power(전력), area(면적), data_array2/tag_array2(동적 할당 포인터) 포함
  if (argc != 53 && argc != 55) // [한국어] 인수 개수가 53(CACTI 6.5 형식)도 55(McPAT 형식)도 아니면 설정 파일 모드로 진입 — 다른 개수는 모두 파일 기반 호출로 처리
  {
    bool infile_specified = false; // [한국어] "-infile" 플래그가 argv에 존재하는지 추적하는 플래그 — 초기값 false(미발견)
    string infile_name("");        // [한국어] "-infile" 다음 인수로 지정된 설정 파일 경로 — 초기값은 빈 문자열

    for (int32_t i = 0; i < argc; i++) // [한국어] argv[0](프로그램 이름)부터 마지막 인수까지 순차 탐색하여 "-infile" 플래그를 찾는 루프
    {
      if (argv[i] == string("-infile")) // [한국어] 현재 인수가 문자열 "-infile"과 일치하는지 확인 — C 스타일 포인터 비교가 아닌 std::string 값 비교이므로 안전
      {
        infile_specified = true;   // [한국어] "-infile" 플래그 발견 — 설정 파일 모드로 진행할 것을 표시
        i++;                       // [한국어] 인덱스를 하나 앞으로 이동하여 "-infile" 다음 인수(파일명)를 읽을 준비
        infile_name = argv[i];     // [한국어] "-infile" 바로 뒤의 인수를 설정 파일 경로로 저장 — 이후 cacti_interface(string)에 전달됨
      }
    }

    if (infile_specified == false) // [한국어] argv 전체를 탐색했지만 "-infile"을 찾지 못한 경우 — 유효한 호출 방식이 아님
    {
      cerr << " Invalid arguments -- how to use CACTI:" << endl;                                  // [한국어] 표준 에러(cerr)에 오류 메시지 출력 — 잘못된 인수 전달 안내
      cerr << "  1) cacti -infile <input file name>" << endl;                                     // [한국어] 사용법 1: 설정 파일(-infile) 지정 방식 안내
      cerr << "  2) cacti arg1 ... arg52 -- please refer to the README file" << endl;             // [한국어] 사용법 2: 52개 위치 인수 방식(CACTI 6.5 레거시) 안내 — 상세 파라미터는 README 참조
      cerr << " No. of arguments input - " << argc << endl;                                       // [한국어] 실제로 전달된 인수 개수를 출력하여 사용자가 문제를 진단할 수 있도록 정보 제공
      exit(1);                                                                                     // [한국어] 유효하지 않은 인수로 인한 비정상 종료 — 종료 코드 1(오류)로 셸에 실패를 알림
    }
    else // [한국어] "-infile" 플래그와 파일명이 모두 정상적으로 발견된 경우 — 설정 파일 기반 호출로 진행
    {
      result = cacti_interface(infile_name); // [한국어] 설정 파일 경로를 인수로 cacti_interface(string) 오버로드 호출 — io.cc가 파일을 파싱하여 InputParameter를 생성하고 분석을 수행한 뒤 uca_org_t를 반환
    }
  }
  else if (argc == 53) // [한국어] 인수 52개(argc==53, 프로그램명 포함) — CACTI 6.5 레거시 커맨드라인 형식; search_ports 파라미터 없음
  {
	  result = cacti_interface(atoi(argv[ 1]),  // [한국어] argv[1]: 캐시 용량(바이트) — atoi로 정수 변환, 예: 65536(64KB)
			  atoi(argv[ 2]),                   // [한국어] argv[2]: 캐시 라인(블록) 크기(바이트) — 일반적으로 64 또는 128
			  atoi(argv[ 3]),                   // [한국어] argv[3]: 연관도(associativity) — 1(직접 매핑), N(N-way 집합 연관), 0(완전 연관)
			  atoi(argv[ 4]),                   // [한국어] argv[4]: UCA(Uniform Cache Architecture) 뱅크 수 — 1이면 단일 뱅크
			  atoi(argv[ 5]),                   // [한국어] argv[5]: 기술 노드(technology node, nm) — 예: 32(32nm 공정)
			  atoi(argv[ 6]),                   // [한국어] argv[6]: 출력 버스 비트 폭(output bus width) — 캐시에서 CPU로 한 번에 전달하는 데이터 비트 수
			  atoi(argv[ 7]),                   // [한국어] argv[7]: 접근 모드(access mode) — 0: 일반(normal), 1: 시퀀셜(sequential), 2: 빠른(fast) 접근 모드
			  atoi(argv[ 8]),                   // [한국어] argv[8]: 캐시 동작 온도(operating temperature, K) — 예: 360(360K)
			  atoi(argv[ 9]),                   // [한국어] argv[9]: 캐시 타입 — 0: 데이터 캐시, 1: I/O, 2: 카운터 등
			  atof(argv[10]),                   // [한국어] argv[10]: 목표 사이클 시간(target cycle time, ns) — 유일하게 atof(실수 변환)를 사용; 클럭 주파수로부터 계산된 ns 단위 사이클 길이
			  atoi(argv[11]),                   // [한국어] argv[11]: 읽기/쓰기 포트 수 (read-write ports)
			  atoi(argv[12]),                   // [한국어] argv[12]: 전용 읽기 포트 수 (exclusive read ports)
			  atoi(argv[13]),                   // [한국어] argv[13]: 전용 쓰기 포트 수 (exclusive write ports)
			  atoi(argv[14]),                   // [한국어] argv[14]: 단일 종단(single ended) 읽기 포트 수
			  atoi(argv[15]),                   // [한국어] argv[15]: UCA 캐시 뱅크 수
			  atoi(argv[16]),                   // [한국어] argv[16]: 캐시 계층(hierarchy) 레벨 — L1=1, L2=2 등
			  atoi(argv[17]),                   // [한국어] argv[17]: 수평 와이어 분포(horizontal wire distribution) 방식
			  atoi(argv[18]),                   // [한국어] argv[18]: 수직 와이어 분포(vertical wire distribution) 방식
			  atoi(argv[19]),                   // [한국어] argv[19]: 전력 최적화 타입 — 0: 면적/딜레이 최적화, 1: 리키지(leakage) 전력 최소화
			  atoi(argv[20]),                   // [한국어] argv[20]: ECC(Error Correction Code) 사용 여부 — 0: 비사용, 1: 사용
			  atoi(argv[21]),                   // [한국어] argv[21]: 캐시 모델 타입 — 0: 일반 캐시, 1: NUCA(Non-Uniform Cache Architecture)
			  atoi(argv[22]),                   // [한국어] argv[22]: NVA(캐시 설계 변수) 주요 파라미터 — 내부 서브어레이 조직 구성
			  atoi(argv[23]),                   // [한국어] argv[23]: 뱅크 수 탐색 플래그 — 1이면 최적 뱅크 수를 탐색
			  atoi(argv[24]),                   // [한국어] argv[24]: 연관도 탐색 플래그 — 1이면 최적 연관도를 탐색
			  atoi(argv[25]),                   // [한국어] argv[25]: 출력 버스 폭 탐색 플래그 — 1이면 최적 폭을 탐색
			  atoi(argv[26]),                   // [한국어] argv[26]: NUCA 뱅크 수 탐색 플래그
			  atoi(argv[27]),                   // [한국어] argv[27]: 캐시 용량 탐색 플래그
			  atoi(argv[28]),                   // [한국어] argv[28]: NUCA 뱅크 크기 탐색 플래그
			  atoi(argv[29]),                   // [한국어] argv[29]: 목표 딜레이 가중치 — 다목적 최적화에서 딜레이에 부여하는 가중치
			  atoi(argv[30]),                   // [한국어] argv[30]: 목표 동적 전력 가중치
			  atoi(argv[31]),                   // [한국어] argv[31]: 목표 정적(리키지) 전력 가중치
			  atoi(argv[32]),                   // [한국어] argv[32]: 목표 사이클 시간 가중치
			  atoi(argv[33]),                   // [한국어] argv[33]: 목표 면적 가중치
			  atoi(argv[34]),                   // [한국어] argv[34]: 읽기 에너지 가중치
			  atoi(argv[35]),                   // [한국어] argv[35]: 쓰기 에너지 가중치
			  atoi(argv[36]),                   // [한국어] argv[36]: 읽기-쓰기 에너지 가중치
			  atoi(argv[37]),                   // [한국어] argv[37]: 리키지 전력 가중치(PST: Processor Static Temperature)
			  atoi(argv[38]),                   // [한국어] argv[38]: 면적 가중치(두 번째)
			  atoi(argv[39]),                   // [한국어] argv[39]: 캐시 설계 최적화 목표 타입
			  atoi(argv[40]),                   // [한국어] argv[40]: NUCA 뱅크 딜레이 가중치
			  atoi(argv[41]),                   // [한국어] argv[41]: NUCA 뱅크 동적 에너지 가중치
			  atoi(argv[42]),                   // [한국어] argv[42]: NUCA 뱅크 리키지 전력 가중치
			  atoi(argv[43]),                   // [한국어] argv[43]: NUCA 뱅크 사이클 시간 가중치
			  atoi(argv[44]),                   // [한국어] argv[44]: NUCA 뱅크 면적 가중치
			  atoi(argv[45]),                   // [한국어] argv[45]: 캐시 데이터 어레이 람다 테크놀로지 비율 파라미터
			  atoi(argv[46]),                   // [한국어] argv[46]: 캐시 태그 어레이 람다 테크놀로지 비율 파라미터
			  atoi(argv[47]),                   // [한국어] argv[47]: 인터커넥트 프로젝션(interconnect projection) 타입 — 0: 보수적, 1: 공격적
			  atoi(argv[48]),                   // [한국어] argv[48]: 와이어 타입(wire type) — 0: 최소 드라이버, 1: 중간 드라이버, 2: 반복기 없음
			  atoi(argv[49]),                   // [한국어] argv[49]: 데이터 어레이 셀 타입 — 0: SRAM, 1: CAM, 2: SCM 등
			  atoi(argv[50]),                   // [한국어] argv[50]: 태그 어레이 셀 타입
			  atoi(argv[51]),                   // [한국어] argv[51]: 데이터 어레이 와이어 타입
			  atoi(argv[52]));                  // [한국어] argv[52]: 태그 어레이 와이어 타입 — 52번째이자 마지막 파라미터; CACTI 6.5 형식에서는 search_ports 없음
  }
  else // [한국어] argc==55인 경우 — McPAT 형식(파라미터 54개); CACTI 6.5 형식 대비 8번째 위치에 search_ports 파라미터가 추가되어 이후 인수 위치가 한 칸씩 밀림
  {
	  result = cacti_interface(atoi(argv[ 1]),  // [한국어] argv[1]: 캐시 용량(바이트) — CACTI 6.5 형식과 동일
			  atoi(argv[ 2]),                   // [한국어] argv[2]: 캐시 라인(블록) 크기(바이트) — CACTI 6.5 형식과 동일
			  atoi(argv[ 3]),                   // [한국어] argv[3]: 연관도(associativity) — CACTI 6.5 형식과 동일
			  atoi(argv[ 4]),                   // [한국어] argv[4]: UCA 뱅크 수 — CACTI 6.5 형식과 동일
			  atoi(argv[ 5]),                   // [한국어] argv[5]: 기술 노드(nm) — CACTI 6.5 형식과 동일
			  atoi(argv[ 6]),                   // [한국어] argv[6]: 출력 버스 비트 폭 — CACTI 6.5 형식과 동일
			  atoi(argv[ 7]),                   // [한국어] argv[7]: 접근 모드 — CACTI 6.5 형식과 동일
			  atoi(argv[ 8]),                   // [한국어] argv[8]: search_ports 파라미터 — McPAT 형식에서만 추가된 인수; 포트 탐색 여부(0/1)를 지정하며, 이 위치가 CACTI 6.5의 argv[8](온도)과 다름에 주의
			  atof(argv[ 9]),                   // [한국어] argv[9]: 목표 사이클 시간(ns, 실수) — CACTI 6.5 형식에서는 argv[10]이었으나 search_ports 삽입으로 인해 한 칸 밀림; atof로 부동소수점 변환
			  atoi(argv[10]),                   // [한국어] argv[10]: 캐시 동작 온도(K) — CACTI 6.5의 argv[9]에서 이동
			  atoi(argv[11]),                   // [한국어] argv[11]: 캐시 타입 — CACTI 6.5의 argv[10] 위치에서 이동 (이하 모든 인수 동일하게 한 칸씩 밀림)
			  atoi(argv[12]),                   // [한국어] argv[12]: 읽기/쓰기 포트 수
			  atoi(argv[13]),                   // [한국어] argv[13]: 전용 읽기 포트 수
			  atoi(argv[14]),                   // [한국어] argv[14]: 전용 쓰기 포트 수
			  atoi(argv[15]),                   // [한국어] argv[15]: 단일 종단 읽기 포트 수
			  atoi(argv[16]),                   // [한국어] argv[16]: UCA 캐시 뱅크 수
			  atoi(argv[17]),                   // [한국어] argv[17]: 캐시 계층 레벨
			  atoi(argv[18]),                   // [한국어] argv[18]: 수평 와이어 분포 방식
			  atoi(argv[19]),                   // [한국어] argv[19]: 수직 와이어 분포 방식
			  atoi(argv[20]),                   // [한국어] argv[20]: 전력 최적화 타입
			  atoi(argv[21]),                   // [한국어] argv[21]: ECC 사용 여부
			  atoi(argv[22]),                   // [한국어] argv[22]: 캐시 모델 타입(일반/NUCA)
			  atoi(argv[23]),                   // [한국어] argv[23]: NVA 주요 파라미터
			  atoi(argv[24]),                   // [한국어] argv[24]: 뱅크 수 탐색 플래그
			  atoi(argv[25]),                   // [한국어] argv[25]: 연관도 탐색 플래그
			  atoi(argv[26]),                   // [한국어] argv[26]: 출력 버스 폭 탐색 플래그
			  atoi(argv[27]),                   // [한국어] argv[27]: NUCA 뱅크 수 탐색 플래그
			  atoi(argv[28]),                   // [한국어] argv[28]: 캐시 용량 탐색 플래그
			  atoi(argv[29]),                   // [한국어] argv[29]: NUCA 뱅크 크기 탐색 플래그
			  atoi(argv[30]),                   // [한국어] argv[30]: 목표 딜레이 가중치
			  atoi(argv[31]),                   // [한국어] argv[31]: 목표 동적 전력 가중치
			  atoi(argv[32]),                   // [한국어] argv[32]: 목표 정적 전력 가중치
			  atoi(argv[33]),                   // [한국어] argv[33]: 목표 사이클 시간 가중치
			  atoi(argv[34]),                   // [한국어] argv[34]: 목표 면적 가중치
			  atoi(argv[35]),                   // [한국어] argv[35]: 읽기 에너지 가중치
			  atoi(argv[36]),                   // [한국어] argv[36]: 쓰기 에너지 가중치
			  atoi(argv[37]),                   // [한국어] argv[37]: 읽기-쓰기 에너지 가중치
			  atoi(argv[38]),                   // [한국어] argv[38]: 리키지 전력 가중치
			  atoi(argv[39]),                   // [한국어] argv[39]: 면적 가중치(두 번째)
			  atoi(argv[40]),                   // [한국어] argv[40]: 캐시 설계 최적화 목표 타입
			  atoi(argv[41]),                   // [한국어] argv[41]: NUCA 뱅크 딜레이 가중치
			  atoi(argv[42]),                   // [한국어] argv[42]: NUCA 뱅크 동적 에너지 가중치
			  atoi(argv[43]),                   // [한국어] argv[43]: NUCA 뱅크 리키지 전력 가중치
			  atoi(argv[44]),                   // [한국어] argv[44]: NUCA 뱅크 사이클 시간 가중치
			  atoi(argv[45]),                   // [한국어] argv[45]: NUCA 뱅크 면적 가중치
			  atoi(argv[46]),                   // [한국어] argv[46]: 캐시 데이터 어레이 람다 비율 파라미터
			  atoi(argv[47]),                   // [한국어] argv[47]: 캐시 태그 어레이 람다 비율 파라미터
			  atoi(argv[48]),                   // [한국어] argv[48]: 인터커넥트 프로젝션 타입
			  atoi(argv[49]),                   // [한국어] argv[49]: 와이어 타입
			  atoi(argv[50]),                   // [한국어] argv[50]: 데이터 어레이 셀 타입
			  atoi(argv[51]),                   // [한국어] argv[51]: 태그 어레이 셀 타입
			  atoi(argv[52]),                   // [한국어] argv[52]: 데이터 어레이 와이어 타입
			  atoi(argv[53]),                   // [한국어] argv[53]: 태그 어레이 와이어 타입
			  atoi(argv[54]));                  // [한국어] argv[54]: McPAT 형식 전용 추가 파라미터(54번째이자 마지막) — io.cc의 54인수 오버로드에 그대로 전달
  }

  result.cleanup(); // [한국어] cacti_interface()가 내부적으로 동적 할당한 data_array2 및 tag_array2 포인터를 delete하여 메모리 누수를 방지 — 이 호출 없이 종료하면 힙 메모리가 해제되지 않음
//  delete result.data_array2;
//  if (result.tag_array2!=NULL)
//	  delete result.tag_array2;
// [한국어] 위 주석 처리된 두 줄은 cleanup() 멤버 함수로 대체된 과거 방식의 메모리 해제 코드이다.
//         현재는 uca_org_t::cleanup()이 이 두 포인터를 일괄 해제하므로, 직접 delete를 호출할 필요 없음.

  return 0; // [한국어] 정상 종료 — 종료 코드 0을 셸에 반환하여 성공을 알림
}

