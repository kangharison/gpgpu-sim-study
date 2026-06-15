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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/

/*
 * [한국어 설명] McPAT/AccelWattch 독립 실행 진입점 (main.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT의 독립 실행형(standalone) 진입점 main()과 사용법 출력 함수
 * print_usage()를 정의한다. XML 설정 파일을 읽어 ParseXML 트리를 구성하고,
 * Processor 객체를 생성하여 전체 프로세서/GPU의 전력·면적 결과를 계산한 뒤
 * displayEnergy()로 터미널에 출력하는 워크플로를 담는다. AccelWattch가
 * GPGPU-Sim에 통합되어 동작할 때에는 이 main()이 호출되지 않는다 — 대신
 * gpgpu_sim_wrapper.cc가 ParseXML과 Processor를 직접 생성한다. 그러나 이
 * 파일은 AccelWattch 빌드의 일부로 컴파일되어 독립 전력 추정 테스트를 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 독립 실행(standalone) 경로:
 *   $ mcpat -infile config.xml -print_level 2
 *     → [이 파일] main()
 *         → ParseXML::parse(fb)     : XML → 파라미터 트리 변환
 *         → Processor::Processor(p1): 전체 전력/면적 모델 구축
 *             ├── SharedCache, CCdir, Core, NoC, MemoryController ...
 *         → Processor::displayEnergy(indent=2, plevel): 결과 출력
 * GPGPU-Sim 통합 경로(이 main() 미사용):
 *   gpgpu_sim::cycle() → gpgpu_sim_wrapper::compute()
 *     → ParseXML + Processor 직접 생성 (gpgpu_sim_wrapper.cc)
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - XML_Parse.h / ParseXML   : XML 파일을 파싱하여 sys.* 트리를 채움
 *   - processor.h / Processor  : 전체 칩 전력/면적 모델 최상위 클래스
 *   - globalvar.h              : opt_for_clk 전역 변수 (SRAM 클럭 최적화 여부)
 *   - version.h                : VER_MAJOR, VER_MINOR, VER_UPDATE 버전 상수
 *   - io.h                     : McPAT I/O 유틸리티 (출력 포맷 도우미)
 *   - xmlParser.h              : 저수준 XML DOM 파서 (ParseXML이 내부 사용)
 * 데이터 흐름:
 *   CLI 인자(파일명, plevel, opt_for_clk) → ParseXML::parse() → Processor 생성
 *   → 전력/면적 결과(W, mm^2) → displayEnergy() → stdout
 *
 * === 주요 함수/구조체 요약 ===
 * main()        : CLI 인자 파싱 → ParseXML 구성 → Processor 전력 계산 → 결과 출력 후 종료.
 * print_usage() : 잘못된 인자 또는 -h 요청 시 사용법을 stderr에 출력하고 exit(1) 호출.
 * opt_for_clk   : 전역 bool (globalvar.h). true=클럭 주파수 제약 최적화,
 *                 false=ED^2P(에너지×지연^2 적) 최소화 전용 최적화.
 * plevel        : 출력 상세도(0~5). displayEnergy()에 전달되어 하위 컴포넌트 세부 수준 제어.
 */

#include <iostream>       // [한국어] cout/cerr 표준 입출력 스트림 — 버전 출력 및 사용법 출력에 사용
#include "XML_Parse.h"    // [한국어] ParseXML 클래스 — XML 설정 파일을 파라미터 트리로 변환
#include "globalvar.h"    // [한국어] opt_for_clk 전역 변수 선언 — SRAM 배열 클럭 최적화 여부
#include "io.h"           // [한국어] McPAT 출력 포맷 유틸리티 함수
#include "processor.h"    // [한국어] Processor 클래스 — 칩 전체 전력/면적 모델 최상위 진입점
#include "version.h"      // [한국어] VER_MAJOR, VER_MINOR, VER_UPDATE 버전 상수
#include "xmlParser.h"    // [한국어] 저수준 XML DOM 파서 — ParseXML이 내부적으로 사용

using namespace std;      // [한국어] std:: 접두사 생략 — cout, cerr, string 등 표준 심볼 직접 사용

void print_usage(char *argv0); // [한국어] 사용법 출력 함수 전방 선언 — main() 내에서 조기 호출 가능하도록

/*
 * [한국어]
 * main - McPAT/AccelWattch 독립 실행형 전력 추정 진입점
 *
 * @param argc: CLI 인자 개수. 1 이하이면 print_usage()를 호출하고 종료.
 * @param argv: CLI 인자 배열.
 *   argv[?] == "-infile"     : 다음 토큰을 XML 설정 파일 경로로 사용 (필수).
 *   argv[?] == "-print_level": 다음 토큰을 출력 상세도(0~5)로 파싱.
 *   argv[?] == "-opt_for_clk": 다음 토큰을 bool(0/1)로 파싱하여 SRAM 최적화 모드 결정.
 * @return: 0 (정상 종료). 에러 시 print_usage()→exit(1)로 프로세스 종료.
 *
 * 이 함수는 McPAT를 독립 실행 모드로 동작시키기 위한 최상위 진입점이다.
 * GPGPU-Sim + AccelWattch 통합 경로에서는 이 함수가 호출되지 않으며, 대신
 * gpgpu_sim_wrapper.cc가 ParseXML과 Processor를 직접 생성한다.
 * 동작 단계:
 *   1) CLI 인자를 순회하며 -infile/-print_level/-opt_for_clk 옵션 파싱.
 *   2) -infile이 지정되지 않으면 print_usage() 호출로 즉시 종료.
 *   3) ParseXML::parse(fb)로 XML 설정을 파라미터 트리로 변환.
 *   4) Processor(p1) 생성으로 칩 전체 전력/면적 모델 구축.
 *   5) proc.displayEnergy(2, plevel)로 결과 출력.
 *   6) ParseXML 동적 객체를 delete로 해제 후 0 반환.
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드 (재진입 불필요).
 * 에러 경로: 잘못된 인자 또는 -infile 누락 → print_usage() → exit(1).
 *
 * 호출 체인:
 *   OS shell → [main()] → ParseXML::parse() → Processor::Processor()
 *                       → Processor::displayEnergy() → exit
 */
int main(int argc, char *argv[]) {
  char *fb;                    // [한국어] XML 설정 파일 경로 포인터 — -infile 인자로 채워짐 (argv 배열 내 포인터, 별도 복사 없음)
  bool infile_specified = false; // [한국어] -infile 옵션이 파싱되었는지 추적하는 플래그 — false이면 main() 말미에서 print_usage() 호출
  int plevel = 2;              // [한국어] 출력 상세도 기본값 2 — displayEnergy()에 전달되어 컴포넌트 계층별 출력 수준 제어 (0=최소, 5=최대)
  opt_for_clk = true;          // [한국어] 전역 opt_for_clk를 true로 초기화 — CACTI SRAM 배열을 클럭 주파수 제약 기준으로 최적화 (기본값)
  // cout.precision(10);
  if (argc <= 1 || argv[1] == string("-h") || argv[1] == string("--help")) {
    // [한국어] 인자가 없거나 도움말 요청(-h/--help)인 경우 즉시 사용법 출력 후 종료
    print_usage(argv[0]);
  }

  for (int32_t i = 0; i < argc; i++) {
    // [한국어] argc개의 인자를 순서대로 순회하며 알려진 옵션 토큰을 탐색
    if (argv[i] == string("-infile")) {
      // [한국어] -infile 옵션 감지 — 다음 토큰(i+1)이 XML 설정 파일 경로
      infile_specified = true;  // [한국어] -infile이 제공되었음을 기록 — 말미의 누락 검사를 통과
      i++;                      // [한국어] 인덱스를 다음으로 전진하여 파일 경로 토큰을 가리킴
      fb = argv[i];             // [한국어] argv[i]를 fb에 저장 — ParseXML::parse()에 직접 전달될 파일 경로
    }

    if (argv[i] == string("-print_level")) {
      // [한국어] -print_level 옵션 감지 — 다음 토큰을 출력 상세도 정수로 파싱
      i++;                      // [한국어] 인덱스를 다음으로 전진하여 상세도 값 토큰을 가리킴
      plevel = atoi(argv[i]);   // [한국어] 문자열을 정수로 변환하여 plevel에 저장 (유효 범위: 0~5)
    }

    if (argv[i] == string("-opt_for_clk")) {
      // [한국어] -opt_for_clk 옵션 감지 — SRAM 배열 최적화 전략 선택
      i++;                               // [한국어] 인덱스를 다음으로 전진하여 0/1 값 토큰을 가리킴
      opt_for_clk = (bool)atoi(argv[i]); // [한국어] "0"→false(ED^2P 최적화), "1"→true(클럭 주파수 최적화)
                                         //          globalvar.h의 전역 변수에 직접 기록 — Processor 전체에 전파됨
    }
  }
  if (infile_specified == false) {
    // [한국어] 루프 종료 후에도 -infile이 지정되지 않았으면 사용법 출력 후 종료
    // [한국어] ParseXML::parse()에 유효한 파일 경로가 없으면 null 역참조 등의 미정의 동작 발생 가능
    print_usage(argv[0]);
  }

  cout << "McPAT (version " << VER_MAJOR << "." << VER_MINOR << " of "
       << VER_UPDATE << ") is computing the target processor...\n " << endl;
  // [한국어] McPAT 버전 정보를 stdout에 출력 — 버전 상수는 version.h에서 제공
  // [한국어] 이 메시지가 출력되면 XML 파싱과 Processor 생성이 시작됨을 사용자에게 알림

  // parse XML-based interface
  ParseXML *p1 = new ParseXML(); // [한국어] XML 파라미터 트리 객체 동적 생성 — 힙에 할당 (말미에서 delete로 해제)
  p1->parse(fb);                 // [한국어] fb 경로의 XML 파일을 읽어 p1->sys.* 트리를 채움
                                 //          실패 시 내부에서 abort/exit 처리됨 (에러 코드 반환 없음)
  Processor proc(p1);            // [한국어] p1 파라미터 트리를 기반으로 전체 칩 전력/면적 모델 구축
                                 //          Core, SharedCache, NoC, MemoryController 등의 서브컴포넌트를 생성하고
                                 //          computeEnergy()를 호출하여 TDP 전력과 면적을 산출
  proc.displayEnergy(2, plevel); // [한국어] indent=2, plevel=CLI 지정값으로 결과 출력
                                 //          plevel이 높을수록 하위 컴포넌트(캐시, 실행 유닛 등)의 세부 수치도 출력
  delete p1;                     // [한국어] 힙에 할당된 ParseXML 객체 해제 — Processor는 스택 객체이므로 자동 소멸
  return 0;                      // [한국어] 정상 종료 코드 0 반환 — OS에 성공을 알림
}

/*
 * [한국어]
 * print_usage - McPAT CLI 사용법 출력 및 비정상 종료
 *
 * @param argv0: 실행 파일 이름 (argv[0]). 사용법 메시지에 프로그램 이름을 표시할 때 사용.
 *               현재 구현에서는 메시지에 직접 포함되지 않지만, 확장 가능성을 위해 인자로 수용.
 * @return: 반환하지 않음 (exit(1) 호출로 프로세스 종료).
 *
 * 이 함수는 두 가지 상황에서 호출된다:
 *   1) CLI 인자가 없거나 -h/--help가 지정된 경우 (main() 앞부분 조건 분기).
 *   2) -infile이 지정되지 않은 채 인자 파싱 루프를 통과한 경우 (main() 말미 조건).
 * 메시지는 stdout이 아닌 stderr(cerr)에 출력된다 — 파이프라인 처리 시 결과 출력과 분리 가능.
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드. exit(1)로 즉시 종료하므로 정리 코드 없음.
 * 에러 경로: 이 함수 자체가 에러 경로의 끝 — exit(1) 호출.
 *
 * 호출 체인:
 *   main() → [print_usage()] → exit(1)
 */
void print_usage(char *argv0) {
  cerr << "How to use McPAT:" << endl;
  // [한국어] 사용법 첫 줄을 stderr에 출력 — stdout이 아닌 cerr로 출력하여 결과 파이핑 시 혼입 방지
  cerr << "  mcpat -infile <input file name>  -print_level < level of details "
          "0~5 >  -opt_for_clk < 0 (optimize for ED^2P only)/1 (optimzed for "
          "target clock rate)>"
       << endl;
  // [한국어] 지원하는 세 가지 CLI 옵션을 한 줄로 안내:
  //   -infile <파일>      : XML 설정 파일 경로 (필수)
  //   -print_level <0~5>  : 출력 상세도 (0=칩 수준 합계만, 5=모든 서브컴포넌트 세부 수치)
  //   -opt_for_clk <0/1>  : 0=ED^2P(에너지×지연^2 적) 최소화 전용, 1=클럭 주파수 제약 최적화(기본)
  // cerr << "    Note:default print level is at processor level, please
  // increase it to see the details" << endl;
  exit(1); // [한국어] 비정상 종료 코드 1로 프로세스 즉시 종료 — 이후 코드는 실행되지 않음
}
