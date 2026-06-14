/*
 * [한국어 설명] cuobjdump 파싱 자료구조 선언 (cuobjdump.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 NVIDIA의 cuobjdump 도구가 출력하는 바이너리 덤프(fat binary)를
 * 파싱하는 데 필요한 자료구조와 클래스 계층을 선언한다.
 * cuobjdump는 컴파일된 CUDA fat binary (.cubin) 안에 내포된 PTX(가상 ISA)와
 * SASS(실제 GPU ISA, ELF 형식) 섹션을 추출하는 NVIDIA 제공 커맨드라인 도구이다.
 * GPGPU-Sim은 실제 CUDA 애플리케이션을 실행할 때 이 도구를 호출하여 PTX/ELF를
 * 임시 파일로 추출한 뒤, 자체 파서(Flex/Bison 기반)로 읽어 기능 시뮬레이션에 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일은 libcuda/ 레이어에 속하며, CUDA 런타임 인터셉트 초기화 경로에서 사용된다.
 * 실행 흐름:
 *   CUDA Application → libcuda (cuobjdumpParseBinary) → [이 파일의 구조체 사용]
 *       → Flex/Bison 파서 (cuobjdump_lexer.l / cuobjdump_parser.y)
 *           → cuobjdumpELFSection / cuobjdumpPTXSection 객체 생성
 *               → cuda-sim/ 기능 시뮬레이션 (PTX 로드)
 * 실행 컨텍스트: CPU 호스트 유저스페이스, CUDA 런타임 초기화 중 단 한 번 호출된다.
 * 이 파일의 클래스들은 파싱 결과를 담는 순수한 데이터 컨테이너 역할만 한다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: <iostream>, <list>, <string> (C++ 표준 라이브러리)
 * - 이 파일을 사용하는 모듈:
 *   - libcuda/cuda_runtime_api.cc: cuobjdumpParseBinary()가 파싱 결과를 리스트로 관리
 *   - libcuda/cuobjdump.cc: 이 헤더의 클래스를 인스턴스화하고 필드를 채움
 *   - cuobjdump_lexer.l / cuobjdump_parser.y: cuobjdump_parser 구조체의 scanner를 사용
 * - 데이터 흐름:
 *   cuobjdump 도구 출력(텍스트) → Flex 스캐너(yyscan_t) → Bison 액션에서
 *   cuobjdumpELFSection / cuobjdumpPTXSection 생성 및 필드 설정 →
 *   리스트로 수집 → cuda-sim 기능 시뮬레이션에 PTX 파일 경로 전달
 *
 * === 주요 함수/구조체 요약 ===
 * - yyscan_t: reentrant Flex 스캐너의 불투명 핸들 타입 (void* 별칭)
 * - cuobjdump_parser: Flex/Bison 파서 전체 상태를 담는 C 구조체
 *   (스캐너 핸들, 섹션 시리얼 번호, 임시 파일 포인터들)
 * - cuobjdumpSection: PTX/ELF 섹션의 공통 기반 클래스 (arch, identifier 보유)
 * - cuobjdumpELFSection: ELF 바이너리 + SASS 어셈블리 파일명을 보유하는 파생 클래스
 * - cuobjdumpPTXSection: PTX 어셈블리 파일명을 보유하는 파생 클래스
 */

#ifndef __cuobjdump_h__
#define __cuobjdump_h__
#include <iostream> // [한국어] print() 메서드에서 std::cout 출력에 필요 — 디버그/덤프 목적
#include <list>     // [한국어] 파싱된 섹션 객체들을 순서 있는 리스트로 관리하기 위해 필요 (cuobjdump.cc에서 사용)
#include <string>   // [한국어] elffilename, sassfilename, ptxfilename 등 파일명을 std::string으로 저장하기 위해 필요

/*
 * [한국어] yyscan_t — reentrant Flex 스캐너의 불투명 핸들 타입
 *
 * Flex의 재진입(reentrant) 모드에서는 스캐너 상태 전체가 단일 힙 객체에 캡슐화되며,
 * 그 포인터를 yyscan_t라는 void* 타입으로 추상화한다.
 * GPGPU-Sim에서는 여러 cuobjdump 파싱 호출이 동시에 일어나지 않지만,
 * reentrant 모드를 사용함으로써 전역 스캐너 상태 오염 없이 안전하게 파싱할 수 있다.
 * 실제 타입은 Flex 내부 struct yyguts_t*이나, 외부에 노출할 필요 없으므로 void*로 숨긴다.
 */
typedef void *yyscan_t;

/*
 * [한국어] cuobjdump_parser — cuobjdump 출력 파싱 중 유지되는 전체 파서 상태
 *
 * cuobjdump 도구가 fat binary를 덤프한 텍스트 출력을 Flex/Bison으로 파싱할 때
 * 필요한 모든 상태(스캐너 핸들, 시리얼 카운터, 임시 파일 포인터)를 하나로 묶는 구조체이다.
 * Bison 액션 함수들이 이 구조체의 포인터를 통해 파서 전체 상태에 접근하므로,
 * 전역 변수 없이 재진입 가능한 파싱이 가능해진다.
 * 이 구조체는 cuobjdumpParseBinary()에서 스택에 선언되어 파싱 수명 동안 유지된다.
 */
struct cuobjdump_parser {
  yyscan_t scanner;
  /* [한국어] reentrant Flex 스캐너의 불투명 핸들.
   * 설정자: yylex_init_extra()로 초기화되며, 파싱 시작 시 Flex 내부 상태를 할당받는다.
   * 읽는 자: Flex가 생성한 yylex()가 내부적으로 이 핸들을 통해 입력 버퍼 상태를 관리한다.
   * 값 범위: NULL이면 미초기화 상태 (파싱 전에 반드시 yylex_init()으로 설정해야 함).
   * 동기화: 단일 파싱 세션 내에서만 사용되므로 별도 락 불필요. */

  int elfserial;
  /* [한국어] ELF 섹션에 할당되는 순차 시리얼 번호 카운터.
   * cuobjdump 출력에는 여러 GPU 아키텍처(sm_60, sm_70 등)에 대한 ELF 섹션이
   * 순서대로 나타날 수 있다. 각 ELF 섹션을 고유하게 구별하기 위해 파싱 중
   * 새로운 ELF 섹션이 인식될 때마다 이 카운터를 증가시키며, 임시 파일명에 삽입한다.
   * 설정자: cuobjdump_parser 초기화 시 0으로 시작, ELF 섹션 시작 규칙에서 증가.
   * 읽는 자: cuobjdumpSection 임시 파일명 생성 로직.
   * 값 범위: 0 이상 (fat binary 안의 ELF 섹션 수 이하).
   * 동기화: 단일 파싱 스레드에서만 접근되므로 락 불필요. */

  int ptxserial;
  /* [한국어] PTX 섹션에 할당되는 순차 시리얼 번호 카운터.
   * elfserial과 대응하는 PTX 버전. fat binary 안에 여러 아키텍처 버전의 PTX가
   * 포함될 수 있으며 (예: compute_60, compute_70), 각 PTX 섹션을 구별하는 번호.
   * 설정자: cuobjdump_parser 초기화 시 0으로 시작, PTX 섹션 시작 규칙에서 증가.
   * 읽는 자: cuobjdumpSection 임시 파일명 생성 로직.
   * 값 범위: 0 이상 (fat binary 안의 PTX 섹션 수 이하).
   * 동기화: 단일 파싱 스레드에서만 접근되므로 락 불필요. */

  FILE *ptxfile;
  /* [한국어] PTX 텍스트를 쓸 임시 파일의 스트림 포인터.
   * cuobjdump 출력에서 PTX 코드 부분을 발견하면 이 파일에 write하여 임시 저장한다.
   * 이후 PTX 파서(Bison)가 이 파일을 입력으로 읽어 PTX IR을 생성한다.
   * 설정자: Bison 액션에서 fopen()으로 생성, 파싱 완료 후 fclose().
   * 읽는 자: Bison PTX 내용 복사 규칙에서 fprintf()로 내용 기록.
   * 값 범위: NULL(파일 미열림) 또는 유효한 FILE* (PTX 섹션 파싱 중).
   * 동기화: 단일 파싱 스레드에서만 사용되므로 락 불필요. */

  FILE *elffile;
  /* [한국어] ELF 바이너리를 쓸 임시 파일의 스트림 포인터.
   * cuobjdump 출력에서 ELF(SASS) 바이너리 부분을 발견하면 이 파일에 write한다.
   * 추출된 ELF는 이후 cuobjdump --dump-sass 로 다시 분해되어 SASS 텍스트를 얻는다.
   * 설정자: Bison 액션에서 fopen()으로 바이너리 모드 생성, 파싱 완료 후 fclose().
   * 읽는 자: Bison ELF 내용 복사 규칙에서 fwrite()로 바이너리 데이터 기록.
   * 값 범위: NULL(파일 미열림) 또는 유효한 FILE* (ELF 섹션 파싱 중).
   * 동기화: 단일 파싱 스레드에서만 사용되므로 락 불필요. */

  FILE *sassfile;
  /* [한국어] SASS 어셈블리 텍스트를 쓸 임시 파일의 스트림 포인터.
   * ELF를 elffile에 저장한 뒤, cuobjdump --dump-sass를 재호출하여 얻은 SASS 텍스트를
   * 이 파일에 저장한다. PTXPlus 변환(cuobjdump_to_ptxplus)의 입력으로 사용된다.
   * 설정자: Bison 액션에서 fopen()으로 생성, SASS 덤프 완료 후 fclose().
   * 읽는 자: SASS → PTXPlus 변환 툴(cuobjdump_to_ptxplus)이 이 파일을 읽음.
   * 값 범위: NULL(파일 미열림) 또는 유효한 FILE* (SASS 덤프 중).
   * 동기화: 단일 파싱 스레드에서만 사용되므로 락 불필요. */

  char filename[1024];
  /* [한국어] 현재 처리 중인 cuobjdump 출력 파일 또는 임시 파일의 경로.
   * 파서가 입력 파일 이름을 추적하기 위해 사용하며, 에러 메시지나 임시 파일명
   * 생성에 활용된다. 1024바이트 고정 배열이므로 경로 길이가 이를 초과하지 않아야 한다.
   * 설정자: cuobjdumpParseBinary() 또는 관련 초기화 코드에서 snprintf()로 설정.
   * 읽는 자: Bison 에러 보고 및 임시 파일명 생성 코드.
   * 값 범위: null-terminated 문자열, 최대 1023자 (null 포함 1024).
   * 동기화: 단일 파싱 스레드에서만 접근되므로 락 불필요. */
};

/*
 * [한국어]
 * cuobjdumpSection - PTX 섹션과 ELF 섹션의 공통 기반 클래스
 *
 * @return: 해당 없음 (추상 클래스)
 *
 * fat binary 안에는 여러 GPU 아키텍처(예: sm_60, sm_70)를 위한 코드 섹션이
 * 함께 담겨 있다. cuobjdump 도구는 이를 PTX 섹션(가상 ISA)과 ELF 섹션(실제 ISA)으로
 * 분리해 출력한다. 이 클래스는 두 종류의 섹션이 공통으로 가지는 GPU 아키텍처 번호(arch)와
 * 식별자(identifier)를 캡슐화하고, 다형적 출력(print())을 위한 가상 인터페이스를 정의한다.
 * 실행 컨텍스트: CPU 호스트, CUDA fat binary 로드 시 단 한 번 인스턴스화됨.
 * 이 클래스는 직접 인스턴스화되지 않으며, cuobjdumpELFSection/cuobjdumpPTXSection의
 * 기반으로만 사용된다.
 *
 * 호출 체인:
 *   cuobjdumpParseBinary() → Bison 파서 액션 → [cuobjdumpSection 파생 클래스 생성]
 *       → getArch() / getIdentifier() → cuda-sim 로드 경로
 */
class cuobjdumpSection {
 public:
  // Constructor
  /*
   * [한국어]
   * cuobjdumpSection() - 기반 클래스 기본 생성자
   *
   * @return: 없음 (생성자)
   *
   * arch와 identifier를 "아직 파싱되지 않은" 초기 상태로 설정한다.
   * arch=0은 유효하지 않은 SM 버전을 의미하며, 파싱 중 setArch()로 실제 값이 채워진다.
   * 실행 컨텍스트: CPU 호스트, Bison 파서 액션에서 파생 클래스 생성자가 암묵적으로 호출.
   *
   * 호출 체인:
   *   Bison 액션 → new cuobjdumpELFSection() / new cuobjdumpPTXSection()
   *       → [이 생성자 호출] (기반 클래스 초기화)
   */
  cuobjdumpSection() {
    arch = 0;           // [한국어] GPU 아키텍처 번호를 0으로 초기화 — 0은 "미설정" 상태를 의미하며, 이후 setArch()로 실제 sm_XX 번호가 주입됨
    identifier = "";    // [한국어] 섹션 식별자를 빈 문자열로 초기화 — cuobjdump 출력의 "identifier:" 필드 값으로 이후 채워짐
  }
  /*
   * [한국어]
   * ~cuobjdumpSection() - 가상 소멸자
   *
   * @return: 없음 (소멸자)
   *
   * 파생 클래스(cuobjdumpELFSection, cuobjdumpPTXSection)를 기반 클래스 포인터로
   * delete할 때 파생 클래스의 소멸자가 올바르게 호출되도록 virtual로 선언한다.
   * 기반 클래스에 virtual 소멸자가 없으면 std::string 멤버의 소멸자가 호출되지 않아
   * 메모리 누수가 발생하므로 필수적이다.
   * 실행 컨텍스트: CPU 호스트, 파싱 완료 후 섹션 리스트 정리 시.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() 종료 → delete section → [이 소멸자 호출]
   *       → 파생 클래스 소멸자 → std::string 필드 해제
   */
  virtual ~cuobjdumpSection() {}

  /*
   * [한국어]
   * getArch() - GPU 아키텍처 버전 번호 반환
   *
   * @return: unsigned — SM 아키텍처 버전 번호 (예: 60은 sm_60 = Pascal, 70은 sm_70 = Volta)
   *
   * 이 섹션이 컴파일된 GPU SM 아키텍처 번호를 반환한다.
   * GPGPU-Sim이 시뮬레이션할 SM 버전과 일치하는 섹션을 선택하기 위해 사용된다.
   * 실행 컨텍스트: CPU 호스트, 파싱 완료 후 섹션 선택 로직에서 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [getArch()] → 아키텍처 매칭 → PTX/ELF 로드 결정
   */
  unsigned getArch() { return arch; } // [한국어] private 멤버 arch를 외부에 읽기 전용으로 노출 — 파생 클래스 및 외부 파싱 코드가 SM 버전 확인에 사용

  /*
   * [한국어]
   * setArch() - GPU 아키텍처 버전 번호 설정
   *
   * @param a: unsigned — cuobjdump 출력에서 파싱한 SM 아키텍처 번호 (예: 60, 70, 80)
   * @return: 없음
   *
   * Bison 파서 액션이 "sm_XX" 패턴을 인식하면 이 함수를 호출하여 arch를 설정한다.
   * 한 fat binary에 여러 아키텍처의 코드가 있을 수 있으므로, 각 섹션 객체에 개별적으로 설정한다.
   * 실행 컨텍스트: CPU 호스트, Bison 파서 액션에서 호출됨.
   *
   * 호출 체인:
   *   Bison arch 규칙 → [setArch(a)] → arch 저장
   */
  void setArch(unsigned a) { arch = a; } // [한국어] 파서가 인식한 SM 버전 번호를 저장 — 0은 미설정, 60/70/80 등이 유효한 값

  /*
   * [한국어]
   * getIdentifier() - 섹션 식별자 문자열 반환
   *
   * @return: std::string — cuobjdump 출력의 "identifier:" 필드 값
   *   (보통 커널 이름이나 모듈 식별자, 예: "_Z9vectorAddPfS_S_i")
   *
   * GPGPU-Sim이 특정 커널이나 모듈에 해당하는 섹션을 찾을 때 이 식별자를 키로 사용한다.
   * 실행 컨텍스트: CPU 호스트, 파싱 완료 후 섹션 검색 로직에서 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [getIdentifier()] → 섹션 매칭
   */
  std::string getIdentifier() { return identifier; } // [한국어] 섹션 식별자(커널/모듈명 등)를 값 복사로 반환 — 원본 보호를 위해 복사본 반환

  /*
   * [한국어]
   * setIdentifier() - 섹션 식별자 문자열 설정
   *
   * @param i: std::string — cuobjdump 출력에서 파싱한 "identifier:" 필드 값
   * @return: 없음
   *
   * Bison 파서 액션이 "identifier:" 토큰 다음의 값을 인식하면 이 함수로 저장한다.
   * 실행 컨텍스트: CPU 호스트, Bison 파서 액션에서 호출됨.
   *
   * 호출 체인:
   *   Bison identifier 규칙 → [setIdentifier(i)] → identifier 저장
   */
  void setIdentifier(std::string i) { identifier = i; } // [한국어] 파서가 인식한 섹션 식별자를 저장 — std::string 복사로 저장하므로 원본 문자열 수명에 무관함

  /*
   * [한국어]
   * print() - 섹션 정보를 표준 출력으로 덤프 (가상 함수)
   *
   * @return: 없음
   *
   * 디버그/진단 목적으로 이 섹션의 내용을 std::cout으로 출력한다.
   * 기반 클래스 버전은 "unknown type" 메시지를 출력하며, 파생 클래스에서 override하여
   * 실제 섹션 종류에 맞는 정보(arch, identifier, 파일명)를 출력한다.
   * virtual로 선언되어 있어 기반 클래스 포인터를 통해 호출해도 올바른 버전이 실행된다.
   * 실행 컨텍스트: CPU 호스트, 파싱 결과 디버그 시 수동 호출.
   *
   * 호출 체인:
   *   디버그 코드 → section->print() → [이 함수 또는 파생 클래스 override]
   */
  virtual void print() {
    std::cout << "cuobjdump Section: unknown type" << std::endl; // [한국어] 파생 클래스가 override하지 않은 경우의 폴백 메시지 출력 — 실제 사용 시 이 메시지가 나오면 파생 클래스 누락을 의미
  }

 private:
  unsigned arch;
  /* [한국어] 이 섹션이 컴파일된 GPU SM 아키텍처 버전 번호.
   * 설정자: setArch()를 통해 Bison 파서 액션이 "sm_XX" 토큰 파싱 후 설정.
   * 읽는 자: getArch()를 통해 파싱 완료 후 섹션 선택 로직이 읽음.
   * 값 범위: 0(미설정), 또는 실제 SM 버전 (30=Kepler, 50=Maxwell, 60=Pascal, 70=Volta, 80=Ampere 등).
   * 동기화: 파싱 중에만 쓰이고 이후 읽기 전용이므로 별도 락 불필요. */

  std::string identifier;
  /* [한국어] cuobjdump 출력의 "identifier:" 필드에 해당하는 섹션 식별자.
   * 보통 커널 함수의 mangled 이름(C++ name mangling 결과) 또는 모듈 경로이다.
   * 예: "_Z9vectorAddPfS_S_i" (vectorAdd(float*, float*, float*, int)의 mangled 이름)
   * 설정자: setIdentifier()를 통해 Bison 파서 액션이 identifier 토큰 파싱 후 설정.
   * 읽는 자: getIdentifier()를 통해 파싱 완료 후 섹션-커널 매칭 로직이 읽음.
   * 값 범위: 빈 문자열(미설정) 또는 cuobjdump가 출력한 임의의 식별자 문자열.
   * 동기화: 파싱 중에만 쓰이고 이후 읽기 전용이므로 별도 락 불필요. */
};

/*
 * [한국어]
 * cuobjdumpELFSection - ELF 바이너리 + SASS 어셈블리 섹션 파생 클래스
 *
 * fat binary 안의 ELF 섹션 하나를 표현하는 파생 클래스이다.
 * ELF(Executable and Linkable Format)는 실제 GPU에서 실행되는 SASS(Shader ASSembly) 코드를
 * 담고 있는 바이너리 형식이다. cuobjdump가 ELF를 추출하면 임시 파일에 저장하고,
 * 추가로 cuobjdump --dump-sass를 재호출하여 사람이 읽을 수 있는 SASS 텍스트도 저장한다.
 * GPGPU-Sim에서는 ELF를 PTXPlus(SASS-level 시뮬레이션)의 입력으로 사용하거나,
 * 단순히 아키텍처 판별용으로 참조한다.
 * 실행 컨텍스트: CPU 호스트, fat binary 로드 시 Bison 파서 액션에서 생성됨.
 *
 * 호출 체인:
 *   Bison ELF 섹션 규칙 → new cuobjdumpELFSection() → setELFfilename() / setSASSfilename()
 *       → cuobjdumpParseBinary() 섹션 리스트 추가 → PTXPlus 로드 또는 아키텍처 매칭
 */
class cuobjdumpELFSection : public cuobjdumpSection {
 public:
  /*
   * [한국어]
   * cuobjdumpELFSection() - ELF 섹션 기본 생성자
   *
   * @return: 없음 (생성자)
   *
   * 기반 클래스(cuobjdumpSection)의 생성자를 암묵적으로 호출하여 arch=0, identifier=""로
   * 초기화한다. elffilename과 sassfilename은 std::string 기본 생성자에 의해 빈 문자열로
   * 초기화되므로 본체에서 별도 초기화가 필요하지 않다.
   * 실행 컨텍스트: CPU 호스트, Bison 파서 ELF 섹션 인식 시.
   *
   * 호출 체인:
   *   Bison ELF 시작 규칙 → [new cuobjdumpELFSection()] → 기반 클래스 생성자 호출
   */
  cuobjdumpELFSection() {} // [한국어] elffilename/sassfilename은 std::string 기본 생성자가 빈 문자열로 초기화하므로 추가 코드 불필요

  /*
   * [한국어]
   * ~cuobjdumpELFSection() - ELF 섹션 소멸자
   *
   * @return: 없음 (소멸자)
   *
   * elffilename과 sassfilename을 빈 문자열로 명시적으로 초기화한다.
   * 사실 std::string 소멸자가 자동으로 메모리를 해제하므로 이 대입은 기능상 불필요하지만,
   * 디버그 빌드에서 use-after-free 탐지를 돕기 위한 방어적 코드로 해석할 수 있다.
   * 기반 클래스의 virtual 소멸자 덕분에 기반 포인터로 delete해도 이 소멸자가 호출된다.
   * 실행 컨텍스트: CPU 호스트, 파싱 완료 후 섹션 리스트 정리 시.
   *
   * 호출 체인:
   *   delete section (기반 클래스 포인터) → virtual ~cuobjdumpSection()
   *       → [~cuobjdumpELFSection()] 호출 → std::string 소멸자 연쇄
   */
  virtual ~cuobjdumpELFSection() {
    elffilename = "";   // [한국어] elffilename을 빈 문자열로 명시 리셋 — std::string 소멸자가 처리하나, 방어적 초기화로 포함됨
    sassfilename = "";  // [한국어] sassfilename을 빈 문자열로 명시 리셋 — 동일 이유
  }

  /*
   * [한국어]
   * getELFfilename() - ELF 바이너리 임시 파일 경로 반환
   *
   * @return: std::string — 디스크에 추출된 ELF 바이너리 임시 파일의 절대/상대 경로
   *
   * Bison 파서가 ELF 바이너리를 임시 파일로 저장한 뒤 이 경로를 기록한다.
   * 이후 PTXPlus 변환이나 SASS 덤프 재호출 시 입력 파일로 사용된다.
   * 실행 컨텍스트: CPU 호스트, 파싱 완료 후 ELF 파일 처리 로직에서 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [getELFfilename()] → cuobjdump --dump-sass <elffilename>
   */
  std::string getELFfilename() { return elffilename; } // [한국어] private elffilename을 복사본으로 반환 — 원본 수정 방지를 위해 값 복사

  /*
   * [한국어]
   * setELFfilename() - ELF 바이너리 임시 파일 경로 설정
   *
   * @param f: std::string — cuobjdump 파싱 중 생성된 ELF 바이너리 임시 파일 경로
   * @return: 없음
   *
   * Bison 파서 액션이 ELF 바이너리를 임시 파일로 추출한 뒤 그 경로를 이 함수로 저장한다.
   * 실행 컨텍스트: CPU 호스트, Bison ELF 내용 저장 완료 후 호출됨.
   *
   * 호출 체인:
   *   Bison ELF 저장 완료 → [setELFfilename(f)] → elffilename 저장
   */
  void setELFfilename(std::string f) { elffilename = f; } // [한국어] ELF 임시 파일 경로를 저장 — 이후 SASS 덤프 또는 PTXPlus 로드의 입력 경로로 사용됨

  /*
   * [한국어]
   * getSASSfilename() - SASS 어셈블리 텍스트 임시 파일 경로 반환
   *
   * @return: std::string — SASS 어셈블리가 저장된 임시 파일의 경로
   *
   * ELF에서 재추출된 SASS 텍스트 파일 경로를 반환한다.
   * cuobjdump_to_ptxplus 변환 툴이 이 파일을 읽어 PTXPlus ISA로 변환한다.
   * 실행 컨텍스트: CPU 호스트, PTXPlus 변환 단계에서 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [getSASSfilename()] → cuobjdump_to_ptxplus 입력
   */
  std::string getSASSfilename() { return sassfilename; } // [한국어] SASS 임시 파일 경로를 복사본으로 반환 — PTXPlus 변환 툴의 입력 파일 경로로 활용됨

  /*
   * [한국어]
   * setSASSfilename() - SASS 어셈블리 텍스트 임시 파일 경로 설정
   *
   * @param f: std::string — SASS 텍스트가 저장된 임시 파일 경로
   * @return: 없음
   *
   * cuobjdump --dump-sass 재호출로 생성된 SASS 텍스트 파일 경로를 저장한다.
   * 실행 컨텍스트: CPU 호스트, SASS 덤프 완료 후 Bison 액션에서 호출됨.
   *
   * 호출 체인:
   *   SASS 덤프 완료 → [setSASSfilename(f)] → sassfilename 저장
   */
  void setSASSfilename(std::string f) { sassfilename = f; } // [한국어] SASS 임시 파일 경로를 저장 — 이후 PTXPlus 변환의 입력 경로로 사용됨

  /*
   * [한국어]
   * print() - ELF 섹션의 모든 정보를 표준 출력으로 덤프
   *
   * @return: 없음
   *
   * 디버그/진단 목적으로 이 ELF 섹션의 아키텍처, 식별자, ELF 파일명, SASS 파일명을
   * 순서대로 std::cout으로 출력한다. 기반 클래스의 virtual print()를 override한다.
   * 실행 컨텍스트: CPU 호스트, 파싱 결과 검증 또는 디버그 시 수동 호출.
   *
   * 호출 체인:
   *   디버그 코드 → section->print() → [cuobjdumpELFSection::print()] (다형성)
   */
  virtual void print() {
    std::cout << "ELF Section:" << std::endl;                          // [한국어] 섹션 종류가 ELF임을 명시하는 헤더 라인 출력
    std::cout << "arch: sm_" << getArch() << std::endl;               // [한국어] SM 아키텍처 번호를 "sm_XX" 형식으로 출력 — getArch()로 기반 클래스 private 필드에 접근
    std::cout << "identifier: " << getIdentifier() << std::endl;      // [한국어] 섹션 식별자(커널명 등)를 출력 — getIdentifier()로 기반 클래스 private 필드에 접근
    std::cout << "elf filename: " << getELFfilename() << std::endl;   // [한국어] ELF 바이너리 임시 파일 경로 출력 — 파일이 실제 존재하는지는 이 메서드에서 검증하지 않음
    std::cout << "sass filename: " << getSASSfilename() << std::endl; // [한국어] SASS 어셈블리 임시 파일 경로 출력
    std::cout << std::endl;                                            // [한국어] 섹션 구분을 위한 빈 줄 출력 — 여러 섹션을 연속 출력할 때 가독성 향상
  }

 private:
  std::string elffilename;
  /* [한국어] 디스크에 추출된 ELF 바이너리 임시 파일의 경로.
   * cuobjdump 파싱 중 fat binary에서 추출된 ELF 데이터를 임시 파일로 저장한 경로이다.
   * 설정자: setELFfilename()을 통해 Bison ELF 내용 저장 완료 시 설정.
   * 읽는 자: getELFfilename()을 통해 SASS 재덤프 또는 PTXPlus 변환 시 읽음.
   * 값 범위: 빈 문자열(미설정) 또는 유효한 파일 시스템 경로.
   * 동기화: 파싱 중에만 쓰이고 이후 읽기 전용이므로 별도 락 불필요. */

  std::string sassfilename;
  /* [한국어] ELF에서 재추출된 SASS 어셈블리 텍스트 임시 파일의 경로.
   * cuobjdump --dump-sass <elffilename>을 재호출하여 얻은 텍스트 덤프 파일 경로이다.
   * cuobjdump_to_ptxplus 변환 툴이 이 파일을 읽어 PTXPlus ISA로 변환한다.
   * 설정자: setSASSfilename()을 통해 SASS 덤프 완료 후 설정.
   * 읽는 자: getSASSfilename()을 통해 PTXPlus 변환 입력으로 사용.
   * 값 범위: 빈 문자열(미설정) 또는 유효한 파일 시스템 경로.
   * 동기화: 파싱 중에만 쓰이고 이후 읽기 전용이므로 별도 락 불필요. */
};

/*
 * [한국어]
 * cuobjdumpPTXSection - PTX 가상 ISA 섹션 파생 클래스
 *
 * fat binary 안의 PTX(Parallel Thread eXecution) 섹션 하나를 표현하는 파생 클래스이다.
 * PTX는 NVIDIA의 가상 ISA(Instruction Set Architecture)로, CUDA 컴파일러(nvcc)가
 * CUDA C/C++ 코드를 컴파일할 때 중간 표현으로 생성한다. PTX는 실제 GPU 실행 시
 * Just-In-Time(JIT) 컴파일되어 SASS로 변환되지만, GPGPU-Sim은 PTX 레벨에서 직접
 * 기능 시뮬레이션을 수행한다. 따라서 fat binary에서 PTX를 추출하는 것이 핵심 경로이다.
 * 실행 컨텍스트: CPU 호스트, fat binary 로드 시 Bison 파서 액션에서 생성됨.
 *
 * 호출 체인:
 *   Bison PTX 섹션 규칙 → new cuobjdumpPTXSection() → setPTXfilename()
 *       → cuobjdumpParseBinary() 섹션 리스트 추가
 *           → gpgpu_ptx_sim_load_ptx_from_filename() → cuda-sim PTX 로드
 */
class cuobjdumpPTXSection : public cuobjdumpSection {
 public:
  /*
   * [한국어]
   * cuobjdumpPTXSection() - PTX 섹션 기본 생성자
   *
   * @return: 없음 (생성자)
   *
   * 기반 클래스(cuobjdumpSection)의 생성자를 암묵적으로 호출하여 arch=0, identifier=""로
   * 초기화하고, ptxfilename을 명시적으로 빈 문자열로 설정한다.
   * ptxfilename은 std::string 기본 생성자로도 빈 문자열이 되지만,
   * 의도를 명확히 하기 위해 명시적으로 초기화한다.
   * 실행 컨텍스트: CPU 호스트, Bison 파서 PTX 섹션 인식 시.
   *
   * 호출 체인:
   *   Bison PTX 시작 규칙 → [new cuobjdumpPTXSection()] → 기반 클래스 생성자 호출
   */
  cuobjdumpPTXSection() { ptxfilename = ""; } // [한국어] ptxfilename을 빈 문자열로 명시 초기화 — 이후 setPTXfilename()으로 실제 임시 파일 경로가 채워짐

  /*
   * [한국어]
   * getPTXfilename() - PTX 어셈블리 임시 파일 경로 반환
   *
   * @return: std::string — cuobjdump 파싱 중 추출된 PTX 텍스트 임시 파일의 경로
   *
   * GPGPU-Sim의 기능 시뮬레이션 경로(cuda-sim)가 이 파일을 읽어 PTX IR을 구성한다.
   * gpgpu_ptx_sim_load_ptx_from_filename()이 이 경로를 인자로 받아 PTX 파서를 구동한다.
   * 실행 컨텍스트: CPU 호스트, PTX 로드 단계에서 호출.
   *
   * 호출 체인:
   *   cuobjdumpParseBinary() → [getPTXfilename()] → gpgpu_ptx_sim_load_ptx_from_filename()
   *       → cuda-sim PTX 파서 → PTX IR 생성
   */
  std::string getPTXfilename() { return ptxfilename; } // [한국어] PTX 임시 파일 경로를 복사본으로 반환 — cuda-sim PTX 로더가 이 경로를 파일 열기에 사용함

  /*
   * [한국어]
   * setPTXfilename() - PTX 어셈블리 임시 파일 경로 설정
   *
   * @param f: std::string — cuobjdump 파싱 중 PTX 텍스트를 저장한 임시 파일 경로
   * @return: 없음
   *
   * Bison 파서 액션이 PTX 텍스트를 임시 파일로 저장한 뒤 그 경로를 이 함수로 기록한다.
   * 실행 컨텍스트: CPU 호스트, Bison PTX 내용 저장 완료 후 호출됨.
   *
   * 호출 체인:
   *   Bison PTX 저장 완료 → [setPTXfilename(f)] → ptxfilename 저장
   */
  void setPTXfilename(std::string f) { ptxfilename = f; } // [한국어] PTX 임시 파일 경로를 저장 — 이후 cuda-sim PTX 로드의 입력 경로로 사용됨

  /*
   * [한국어]
   * print() - PTX 섹션의 모든 정보를 표준 출력으로 덤프
   *
   * @return: 없음
   *
   * 디버그/진단 목적으로 이 PTX 섹션의 아키텍처, 식별자, PTX 파일명을
   * 순서대로 std::cout으로 출력한다. 기반 클래스의 virtual print()를 override한다.
   * 실행 컨텍스트: CPU 호스트, 파싱 결과 검증 또는 디버그 시 수동 호출.
   *
   * 호출 체인:
   *   디버그 코드 → section->print() → [cuobjdumpPTXSection::print()] (다형성)
   */
  virtual void print() {
    std::cout << "PTX Section:" << std::endl;                        // [한국어] 섹션 종류가 PTX임을 명시하는 헤더 라인 출력
    std::cout << "arch: sm_" << getArch() << std::endl;             // [한국어] SM 아키텍처 번호를 "sm_XX" 형식으로 출력 — getArch()로 기반 클래스 private 필드에 접근
    std::cout << "identifier: " << getIdentifier() << std::endl;    // [한국어] 섹션 식별자(커널명 등)를 출력 — getIdentifier()로 기반 클래스 private 필드에 접근
    std::cout << "ptx filename: " << getPTXfilename() << std::endl; // [한국어] PTX 텍스트 임시 파일 경로 출력 — cuda-sim PTX 로더가 사용할 파일 경로를 확인하는 데 유용
    std::cout << std::endl;                                          // [한국어] 섹션 구분을 위한 빈 줄 출력 — 여러 섹션을 연속 출력할 때 가독성 향상
  }

 private:
  std::string ptxfilename;
  /* [한국어] cuobjdump가 추출한 PTX 텍스트가 저장된 임시 파일의 경로.
   * GPGPU-Sim의 기능 시뮬레이션(cuda-sim)이 PTX 파싱의 입력으로 이 파일을 사용한다.
   * PTX(Parallel Thread eXecution)는 NVIDIA 가상 ISA로, GPGPU-Sim은 이를 해석하여
   * 각 스레드의 레지스터 상태, 메모리 접근, 분기를 사이클-레벨로 시뮬레이션한다.
   * 설정자: setPTXfilename()을 통해 Bison PTX 내용 저장 완료 시 설정.
   * 읽는 자: getPTXfilename()을 통해 gpgpu_ptx_sim_load_ptx_from_filename()에 전달.
   * 값 범위: 빈 문자열(미설정) 또는 유효한 임시 파일 경로 (예: "/tmp/gpgpusim_ptx_0.ptx").
   * 동기화: 파싱 중에만 쓰이고 이후 읽기 전용이므로 별도 락 불필요. */
};

#endif /* __cuobjdump_h__ */
