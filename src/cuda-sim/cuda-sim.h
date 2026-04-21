/*
 * ============================================================================
 * cuda-sim.h - CUDA 시뮬레이션 헤더 파일
 * ============================================================================
 *
 * CUDA 시뮬레이션이란?
 * 실제 GPU(그래픽 카드) 없이도 CUDA 프로그램이 어떻게 실행되는지
 * 소프트웨어(프로그램)로 흉내내는 것입니다.
 * 마치 비행기 조종사가 실제 비행기 대신 시뮬레이터로 연습하는 것처럼,
 * GPU 프로그래머가 실제 GPU 없이 프로그램을 테스트할 수 있게 해줍니다.
 *
 * 이 파일은 "헤더 파일"입니다.
 * 헤더 파일이란? 다른 코드 파일(.cc 또는 .cpp)에서 사용할 함수, 클래스 등의
 * "설계도"를 모아놓은 파일입니다. 실제 동작 코드는 .cc 파일에 있고,
 * 여기에는 "이런 것들이 있다"라는 선언(약속)만 적혀 있습니다.
 * ============================================================================
 */

// Copyright (c) 2009-2011, Tor M. Aamodt
// 저작권 표시: 이 코드는 2009~2011년에 Tor M. Aamodt라는 분이 만들었습니다.
// The University of British Columbia
// 캐나다 브리티시컬럼비아 대학교에서 만든 코드입니다.
// All rights reserved.
// 모든 권리가 보호됩니다 (저작권법에 의해 보호받는다는 뜻).
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 이 코드를 다시 배포하거나 사용할 때는 아래 조건을 지켜야 합니다:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 소스 코드를 배포할 때는 위의 저작권 표시와 이 조건들을 반드시 포함해야 합니다.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
// 컴파일된 형태로 배포할 때도 저작권 표시를 문서에 포함해야 하며,
// 대학교 이름을 허락 없이 홍보에 사용하면 안 됩니다.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 이 소프트웨어는 "있는 그대로" 제공되며, 사용으로 인한 어떤 손해에 대해서도
// 만든 사람이 책임지지 않는다는 뜻입니다. (법적 면책 조항)

/*
 * 아래 두 줄은 "인클루드 가드(Include Guard)"라고 합니다.
 * 왜 필요할까요?
 * 같은 헤더 파일이 여러 번 포함(include)되면 "이미 정의했다"는 오류가 납니다.
 * 그래서 #ifndef (만약 정의되지 않았다면) ~ #endif 로 감싸서
 * 처음 한 번만 이 코드가 포함되도록 보호합니다.
 * 마치 "이미 읽은 책은 다시 읽지 않기"와 같은 원리입니다.
 */
#ifndef CUDASIM_H_INCLUDED  // CUDASIM_H_INCLUDED가 아직 정의되지 않았다면 아래 코드를 포함해라
#define CUDASIM_H_INCLUDED  // CUDASIM_H_INCLUDED를 정의해서, 다음에 또 포함하려 하면 건너뛰게 함

/*
 * 아래는 #include 문들입니다.
 * #include란? 다른 파일에 있는 코드를 "여기에 가져다 쓰겠다"는 뜻입니다.
 * 마치 레고 블록을 가져와서 조립하는 것처럼,
 * 이미 만들어진 기능들을 가져와서 사용합니다.
 */
#include <stdlib.h>  // C 표준 라이브러리: 메모리 할당(malloc, free), 난수 생성 등 기본 기능 제공
#include <map>       // C++ STL의 map: "사전(dictionary)"처럼 키-값 쌍을 저장하는 자료구조 (예: 이름→전화번호)
#include <string>    // C++ 문자열(string) 클래스: 텍스트를 쉽게 다루기 위한 도구
#include <vector>    // C++ STL의 vector: 크기가 자동으로 변하는 배열 (리스트처럼 사용)
#include "../abstract_hardware_model.h"  // 하드웨어 모델의 추상적인(공통적인) 부분을 정의한 헤더 파일
                                         // GPU의 기본 구조(워프, 스레드 등)를 정의합니다
#include "../gpgpu-sim/shader.h"         // 셰이더(shader) 코어 관련 헤더 파일
                                         // 셰이더란? GPU에서 실제 계산을 수행하는 처리 장치입니다
#include "ptx_sim.h"                     // PTX 시뮬레이션 관련 헤더 파일
                                         // PTX란? CUDA 코드가 컴파일되면 생성되는 중간 언어(어셈블리)입니다
                                         // 사람이 쓴 CUDA 코드 → PTX → 실제 GPU 기계어 순서로 변환됩니다

/*
 * 아래는 "전방 선언(forward declaration)"입니다.
 * 전방 선언이란? "이런 클래스가 있을 거야, 자세한 내용은 나중에 알려줄게"라고
 * 컴파일러(코드를 기계어로 번역하는 프로그램)에게 미리 알려주는 것입니다.
 * 왜 필요할까요? 지금 당장은 클래스의 세부 내용이 필요 없고,
 * "이런 이름의 클래스가 존재한다"는 것만 알면 되기 때문입니다.
 * 이렇게 하면 불필요한 #include를 줄여서 컴파일 속도가 빨라집니다.
 */
class gpgpu_context;   // GPGPU 시뮬레이터의 전체적인 환경/상태를 담는 클래스 (전방 선언)
class memory_space;    // 메모리 공간을 나타내는 클래스 (전방 선언) - GPU에는 여러 종류의 메모리가 있습니다
class function_info;   // CUDA 커널 함수의 정보를 담는 클래스 (전방 선언) - 함수 이름, 매개변수 등
class symbol_table;    // 심볼 테이블 클래스 (전방 선언) - 변수 이름과 그 값/위치를 매핑하는 표

/*
 * 아래는 "extern" 전역 변수 선언입니다.
 * extern이란? "이 변수는 다른 파일에서 실제로 만들어졌고,
 * 여기서는 그것을 가져다 쓰겠다"는 뜻입니다.
 * 마치 "냉장고는 부엌에 있고, 나는 거실에서 그 냉장고를 사용하겠다"와 같습니다.
 */
extern const char *g_gpgpusim_version_string;  // GPGPU-Sim 시뮬레이터의 버전 문자열 (예: "4.0.0")
                                                // const는 "이 값은 변경할 수 없다"는 뜻
                                                // char*는 문자열을 가리키는 포인터(주소)
extern int g_debug_execution;  // 디버그 실행 모드 플래그: 0이 아니면 디버그 정보를 출력
                               // 디버그란? 프로그램의 문제를 찾기 위해 상세 정보를 보여주는 것

/*
 * 아래는 "extern" 함수 선언입니다.
 * 함수의 실제 코드(구현)는 다른 .cc 파일에 있고,
 * 여기서는 "이런 함수가 있다"고만 알려줍니다.
 */
extern void print_splash();  // 시뮬레이터 시작 시 환영 메시지(스플래시 화면)를 출력하는 함수
                              // void는 "이 함수는 아무 값도 돌려주지 않는다"는 뜻

/*
 * PTX 정보에 OpenCL 관련 정보를 추가하는 함수 선언
 * OpenCL이란? CUDA와 비슷하지만, NVIDIA뿐 아니라 AMD 등 다양한 GPU에서 쓸 수 있는 표준
 * std::map<std::string, function_info*>는 "함수 이름(문자열) → 함수 정보 포인터"를 매핑하는 사전
 * &kernels는 "참조(reference)"로 전달: 원본 데이터를 직접 수정할 수 있게 넘기는 방식
 */
extern void ptxinfo_opencl_addinfo(
    std::map<std::string, function_info *> &kernels);  // 커널 함수들의 이름→정보 매핑을 받아서 OpenCL 정보를 추가

/*
 * PTX 시뮬레이션에서 스레드를 초기화하는 함수
 *
 * GPU에서 "스레드(thread)"란? 가장 작은 실행 단위입니다.
 * GPU는 수천 개의 스레드를 동시에 실행하여 빠른 계산을 수행합니다.
 * 이 함수는 시뮬레이션에서 각 스레드를 준비(초기화)하는 역할을 합니다.
 *
 * 반환값: unsigned (부호 없는 정수) - 초기화된 스레드 관련 정보를 반환
 */
unsigned ptx_sim_init_thread(
    kernel_info_t &kernel,                // 커널 정보 참조: 실행할 CUDA 커널(GPU에서 돌릴 함수)의 정보
    class ptx_thread_info **thread_info,  // 스레드 정보 포인터의 포인터: 새로 만든 스레드 정보를 여기에 저장
                                          // 포인터의 포인터(**)를 쓰는 이유: 함수 안에서 포인터 자체를 변경하기 위해
    int sid,                              // 셰이더 코어 ID: 이 스레드가 어떤 셰이더 코어에서 실행되는지
                                          // 셰이더 코어는 GPU 안의 작은 프로세서(처리 장치)입니다
    unsigned tid,                         // 스레드 ID: 이 스레드의 고유 번호 (0부터 시작)
    unsigned threads_left,                // 남은 스레드 수: 아직 초기화하지 않은 스레드가 몇 개인지
    unsigned num_threads,                 // 총 스레드 수: 이 커널에서 실행할 전체 스레드 개수
    class core_t *core,                   // 코어 포인터: 이 스레드가 실행될 GPU 코어(처리 장치) 객체
    unsigned hw_cta_id,                   // 하드웨어 CTA ID: CTA(Cooperative Thread Array)의 하드웨어 식별 번호
                                          // CTA는 함께 협력하는 스레드들의 그룹 (= CUDA의 "블록")
    unsigned hw_warp_id,                  // 하드웨어 워프 ID: 워프의 하드웨어 식별 번호
                                          // 워프(warp)란? 동시에 같은 명령어를 실행하는 32개 스레드 묶음
    gpgpu_t *gpu,                         // GPU 객체 포인터: 시뮬레이션 중인 전체 GPU를 나타내는 객체
    bool functionalSimulationMode = false // 기능적 시뮬레이션 모드 여부 (기본값: false)
                                          // true면 타이밍(시간) 없이 결과만 시뮬레이션
                                          // false면 실제 GPU처럼 시간도 계산하며 시뮬레이션
);

/*
 * 커널 함수의 PTX 시뮬레이션 정보를 가져오는 함수
 *
 * const가 두 번 나오는 이유:
 * 1) 앞의 const: 반환되는 구조체 정보를 변경할 수 없다
 * 2) 매개변수의 const: 전달받은 커널 정보를 변경하지 않겠다는 약속
 *
 * 반환값: gpgpu_ptx_sim_info 구조체의 포인터 - 커널의 레지스터 수, 메모리 사용량 등의 정보
 */
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const class function_info *kernel);  // 정보를 얻고 싶은 커널 함수의 포인터

/*!
 * This class functionally executes a kernel. It uses the basic data structures
 * and procedures in core_t
 */
/*
 * functionalCoreSim 클래스 (기능적 코어 시뮬레이터)
 *
 * 이 클래스는 CUDA 커널을 "기능적(functional)"으로 실행합니다.
 * "기능적 실행"이란? 시간(클럭 사이클) 계산 없이 프로그램의 결과만 계산하는 것입니다.
 * 마치 수학 문제의 답만 구하고, 풀이 시간은 재지 않는 것과 같습니다.
 *
 * core_t를 상속(extends)받습니다.
 * 상속이란? 부모 클래스(core_t)의 기능을 물려받아서 사용하는 것입니다.
 * public 상속: 부모의 공개된 기능을 그대로 공개적으로 사용할 수 있습니다.
 */
class functionalCoreSim : public core_t {  // core_t 클래스를 상속받는 functionalCoreSim 클래스 선언
 public:  // public: 외부에서 자유롭게 접근할 수 있는 영역 (누구나 사용 가능)

  /*
   * 생성자(constructor): 객체가 만들어질 때 자동으로 호출되는 특별한 함수
   * 마치 새 컴퓨터를 사면 처음에 설정(setup)하는 것과 같습니다.
   *
   * 매개변수:
   * - kernel: 실행할 CUDA 커널 정보 포인터
   * - g: GPGPU 시뮬레이터 객체 포인터
   * - warp_size: 워프 크기 (보통 32, 즉 한 워프에 32개 스레드)
   *
   * 초기화 리스트(: core_t(...)): 부모 클래스 core_t의 생성자를 먼저 호출
   * kernel->threads_per_cta()는 하나의 CTA(블록)에 있는 스레드 수를 반환
   */
  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)
      : core_t(g, kernel, warp_size, kernel->threads_per_cta()) {  // 부모 클래스 생성자 호출
    m_warpAtBarrier = new bool[m_warp_count];       // 각 워프가 배리어에서 멈춰있는지 저장하는 배열을 동적 생성
                                                     // 배리어(barrier)란? "모든 스레드가 여기까지 도착할 때까지 기다려!"라는 동기화 지점
                                                     // new bool[m_warp_count]는 워프 개수만큼 bool(참/거짓) 배열을 만듦
    m_liveThreadCount = new unsigned[m_warp_count];  // 각 워프에서 아직 살아있는(실행 중인) 스레드 수를 저장하는 배열
                                                     // unsigned는 0 이상의 정수만 저장하는 타입
  }

  /*
   * 소멸자(destructor): 객체가 없어질 때 자동으로 호출되는 특별한 함수
   * virtual: 자식 클래스에서 재정의(오버라이드)할 수 있도록 허용
   * ~는 소멸자를 나타내는 기호
   * 마치 집을 철거할 때 안에 있는 가구를 먼저 치우는 것과 같습니다.
   * 메모리 누수(leak)를 방지하기 위해 new로 할당한 메모리를 delete로 해제합니다.
   */
  virtual ~functionalCoreSim() {
    warp_exit(0);                     // 워프 0번의 종료 처리를 수행 (정리 작업)
    delete[] m_liveThreadCount;       // m_liveThreadCount 배열의 메모리를 해제 (반환)
                                      // delete[]는 배열을 해제할 때 사용 (delete와 다름!)
    delete[] m_warpAtBarrier;         // m_warpAtBarrier 배열의 메모리를 해제 (반환)
  }

  //! executes all warps till completion
  /*
   * execute 함수: 모든 워프를 완료될 때까지 실행하는 함수
   *
   * 매개변수:
   * - inst_count: 실행할 명령어(instruction) 수
   * - ctaid_cp: 체크포인트용 CTA ID
   *   체크포인트란? 실행 상태를 저장해두어 나중에 이어서 할 수 있게 하는 것
   */
  void execute(int inst_count, unsigned ctaid_cp);

  /*
   * warp_exit 함수: 특정 워프의 실행이 끝났을 때 호출되는 함수
   * virtual: 자식 클래스에서 다른 동작으로 바꿀 수 있음 (다형성)
   *
   * 매개변수:
   * - warp_id: 종료할 워프의 ID 번호
   */
  virtual void warp_exit(unsigned warp_id);

  /*
   * warp_waiting_at_barrier 함수: 특정 워프가 배리어에서 기다리고 있는지 확인
   * virtual: 자식 클래스에서 재정의 가능
   * const: 이 함수는 객체의 상태를 변경하지 않겠다는 약속
   *
   * 반환값: true면 워프가 기다리는 중, false면 아직 실행 중
   *
   * 조건: m_warpAtBarrier[warp_id]가 true이거나,
   *       살아있는 스레드가 0개이면 (즉 다 끝났으면) true를 반환
   * ||는 "또는(OR)" 연산자, !는 "아니다(NOT)" 연산자
   */
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const {
    return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id] > 0));
  }

 private:  // private: 이 클래스 내부에서만 접근 가능한 영역 (외부에서 직접 사용 불가)
           // 왜 private을 쓸까? 내부 구현을 숨겨서 잘못된 사용을 방지하기 위해 (캡슐화)

  /*
   * executeWarp 함수: 하나의 워프를 실행하는 내부 함수
   *
   * 매개변수:
   * - unsigned: 워프 ID (이름이 생략되어 있음 - 헤더에서는 타입만 적어도 됨)
   * - bool&: 참조로 전달되는 bool 값들 (함수 안에서 값을 바꾸면 밖에서도 바뀜)
   */
  void executeWarp(unsigned, bool &, bool &);

  // initializes threads in the CTA block which we are executing
  /*
   * initializeCTA 함수: 실행할 CTA(블록) 안의 스레드들을 초기화하는 함수
   * CTA(Cooperative Thread Array)는 함께 협력하여 일하는 스레드들의 그룹입니다.
   * CUDA에서는 "블록(block)"이라고도 부릅니다.
   *
   * 매개변수:
   * - ctaid_cp: 체크포인트용 CTA ID
   */
  void initializeCTA(unsigned ctaid_cp);

  /*
   * checkExecutionStatusAndUpdate 함수: 스레드의 실행 상태를 확인하고 업데이트
   * virtual: 자식 클래스에서 재정의 가능
   *
   * 매개변수:
   * - inst: 현재 실행 중인 워프 명령어(warp_inst_t)의 참조
   * - t: 스레드 번호 (워프 내에서의 위치)
   * - tid: 전체 스레드 ID
   *
   * 동작: 스레드가 NULL(존재하지 않음)이거나 이미 실행이 끝났으면(is_done()),
   *       해당 워프의 살아있는 스레드 수를 1 줄입니다.
   * tid / m_warp_size: 스레드 ID를 워프 크기로 나누면 워프 ID가 됩니다
   *   예: 스레드 35번, 워프 크기 32 → 35/32 = 1 → 워프 1번에 속함
   */
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) {
    if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {  // 스레드가 없거나 끝났으면
      m_liveThreadCount[tid / m_warp_size]--;  // 해당 워프의 살아있는 스레드 수를 1 감소 (--)
    }
  }

  // lunches the stack and set the threads count
  /*
   * createWarp 함수: 워프를 생성하고 스택을 시작하며 스레드 수를 설정
   * "스택(stack)"이란? 함수 호출 순서를 기억하는 자료구조
   * GPU에서는 분기(if-else)가 발생할 때 어디로 돌아갈지 기억하는 데 사용
   *
   * 매개변수:
   * - warpId: 생성할 워프의 ID 번호
   */
  void createWarp(unsigned warpId);

  // each warp live thread count and barrier indicator
  // 각 워프별 살아있는 스레드 수와 배리어 상태 표시기

  unsigned *m_liveThreadCount;  // 각 워프의 살아있는(실행 중인) 스레드 수를 저장하는 동적 배열
                                // 포인터(*)를 쓴 이유: 프로그램 실행 중에 배열 크기를 정할 수 있도록
  bool *m_warpAtBarrier;        // 각 워프가 배리어(동기화 지점)에서 기다리고 있는지 저장하는 동적 배열
                                // true: 기다리는 중, false: 실행 중
};  // 클래스 정의 끝 (C++에서 클래스 정의 뒤에는 반드시 세미콜론(;)이 와야 함)

/*
 * 아래 두 줄은 매크로(macro) 정의입니다.
 * #define은 컴파일러에게 "이 이름을 만날 때마다 이 값으로 바꿔라"라고 지시합니다.
 * 마치 "단축키"를 만드는 것과 같습니다.
 *
 * address_type은 메모리 주소를 나타내는 타입입니다.
 * (address_type)-2와 (address_type)-1은 특별한 의미를 가진 "센티넬 값"입니다.
 * 센티넬 값이란? "이것은 일반적인 값이 아니라 특별한 상태를 나타낸다"는 표시입니다.
 */
#define RECONVERGE_RETURN_PC ((address_type)-2)  // 재수렴 리턴 PC: 분기(if-else) 후 다시 합쳐지는 지점으로 돌아갈 때 사용
                                                  // PC = Program Counter(프로그램 카운터): 현재 실행 중인 명령어의 주소
                                                  // -2를 address_type으로 변환 → 매우 큰 수가 됨 (특별한 표시용)
#define NO_BRANCH_DIVERGENCE ((address_type)-1)  // 분기 발산 없음을 나타내는 값
                                                  // 분기 발산(branch divergence)이란?
                                                  // 같은 워프의 스레드들이 if-else에서 서로 다른 길로 갈라지는 현상
                                                  // GPU 성능에 매우 나쁜 영향을 주는 현상입니다

/*
 * get_return_pc 함수: 스레드의 리턴(돌아갈) PC 값을 가져오는 함수
 *
 * 매개변수:
 * - thd: 스레드 객체의 포인터 (void*는 어떤 타입이든 가리킬 수 있는 범용 포인터)
 *
 * 반환값: address_type - 돌아갈 명령어의 주소
 */
address_type get_return_pc(void *thd);

/*
 * get_ptxinfo_kname 함수: 현재 PTX 정보의 커널 이름을 가져오는 함수
 * 반환값: const char* - 변경 불가능한 문자열 포인터 (커널 함수의 이름)
 */
const char *get_ptxinfo_kname();

/*
 * print_ptxinfo 함수: PTX 정보를 화면에 출력하는 함수 (디버깅용)
 * void: 아무것도 반환하지 않음
 */
void print_ptxinfo();

/*
 * clear_ptxinfo 함수: PTX 정보를 초기화(지우기)하는 함수
 * 마치 칠판을 지우는 것과 같습니다
 */
void clear_ptxinfo();

/*
 * get_ptxinfo 함수: PTX 시뮬레이션 정보 구조체를 반환하는 함수
 * 반환값: gpgpu_ptx_sim_info 구조체 - 커널의 레지스터 수, 메모리 사용량 등의 정보
 * 구조체(struct)란? 여러 종류의 데이터를 하나로 묶은 것 (예: 학생 = 이름 + 나이 + 성적)
 */
struct gpgpu_ptx_sim_info get_ptxinfo();

/*
 * gpgpu_recon_t 클래스의 전방 선언
 * "재수렴(reconvergence)" 관련 클래스: GPU에서 분기 후 스레드들이 다시 합쳐지는 것을 관리
 */
class gpgpu_recon_t;

/*
 * rec_pts 구조체: 재수렴 포인트(reconvergence points) 정보를 담는 구조체
 *
 * 재수렴 포인트란? if-else 같은 분기문 이후에
 * 갈라졌던 스레드들이 다시 모이는 지점입니다.
 * GPU에서 성능을 위해 이 지점을 미리 찾아두는 것이 중요합니다.
 */
struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;  // 커널의 재수렴 포인트들을 가리키는 포인터
                                          // 배열처럼 여러 재수렴 포인트를 저장
  int s_num_recon;                        // 재수렴 포인트의 개수
};

/*
 * ============================================================================
 * cuda_sim 클래스: CUDA 시뮬레이션의 핵심 클래스
 * ============================================================================
 *
 * 이 클래스는 CUDA 시뮬레이션에 필요한 모든 전역 상태(변수)와
 * 주요 기능(함수)을 하나로 모아놓은 "중앙 관리소"입니다.
 *
 * 마치 학교에서 교무실이 모든 학생 정보와 학교 운영 기능을 모아놓은 것처럼,
 * 이 클래스가 시뮬레이션의 모든 것을 관리합니다.
 */
class cuda_sim {
 public:  // 외부에서 접근 가능한 영역

  /*
   * 생성자: cuda_sim 객체가 만들어질 때 모든 변수를 초기값으로 설정
   *
   * 매개변수:
   * - ctx: GPGPU 컨텍스트 포인터 - 시뮬레이터의 전체 환경 정보
   *
   * 각 변수를 안전한 초기값으로 설정하는 이유:
   * 초기화하지 않은 변수는 쓰레기 값(이상한 값)이 들어있어서 버그의 원인이 됩니다.
   */
  cuda_sim(gpgpu_context *ctx) {
    g_ptx_sim_num_insn = 0;    // PTX 시뮬레이션에서 실행한 명령어 수를 0으로 초기화
    g_ptx_kernel_count =
        -1;  // used for classification stat collection purposes
             // 커널 실행 횟수를 -1로 초기화 (아직 커널이 실행되지 않았다는 뜻)
             // 통계 수집을 위한 분류 목적으로 사용됨
    gpgpu_param_num_shaders = 0;           // 셰이더 코어 수를 0으로 초기화
    g_cuda_launch_blocking = false;        // CUDA 런치 블로킹 모드를 꺼놓음 (false)
                                            // 블로킹 모드: 커널이 끝날 때까지 다음 작업을 기다리는 모드
    g_inst_classification_stat = NULL;     // 명령어 분류 통계 포인터를 NULL(없음)로 초기화
    g_inst_op_classification_stat = NULL;  // 명령어 연산 분류 통계 포인터를 NULL로 초기화
    g_assemble_code_next_pc = 0;           // 다음에 조립(assemble)할 코드의 PC를 0으로 초기화
    g_debug_thread_uid = 0;                // 디버그할 스레드의 고유 ID를 0으로 초기화
    g_override_embedded_ptx = false;       // 내장된 PTX를 덮어쓸지 여부를 false로 초기화
                                            // (외부 PTX 파일을 대신 사용할지 결정)
    ptx_tex_regs = NULL;                   // PTX 텍스처 레지스터 포인터를 NULL로 초기화
                                            // 텍스처: GPU에서 이미지 데이터를 빠르게 읽는 기능
    g_ptx_thread_info_delete_count = 0;    // 삭제된 스레드 정보 수를 0으로 초기화 (메모리 관리용)
    g_ptx_thread_info_uid_next = 1;        // 다음 스레드 정보의 고유 ID를 1부터 시작
                                            // 0이 아닌 1부터 시작하는 이유: 0은 "없음"을 나타낼 수 있으므로
    g_debug_pc = 0xBEEF1518;               // 디버그 PC를 특별한 16진수 값으로 설정
                                            // 0xBEEF1518은 "BEEF"가 포함된 기억하기 쉬운 마커 값
                                            // 16진수(hex): 0~9, A~F를 사용하는 숫자 표기법
    gpgpu_ctx = ctx;                       // 전달받은 컨텍스트 포인터를 멤버 변수에 저장
  }

  /*
   * ======================================================================
   * 전역 변수들 (global variables)
   * ======================================================================
   * 아래 변수들은 CUDA 시뮬레이션 전체에서 사용되는 설정 값과 상태 정보입니다.
   */

  // --- 명령어 지연 시간(latency) 관련 변수들 ---
  // 지연 시간(latency)이란? 하나의 명령어가 실행되는 데 걸리는 클럭 사이클 수
  // 클럭 사이클: GPU 안의 시계가 한 번 "똑딱"하는 시간 단위
  char *opcode_latency_int;     // 정수(integer) 연산 명령어의 지연 시간 설정 문자열
                                 // 예: 정수 덧셈, 뺄셈, 곱셈 등이 몇 사이클 걸리는지
  char *opcode_latency_fp;      // 단정밀도 부동소수점(float) 연산의 지연 시간 설정 문자열
                                 // 부동소수점: 소수점이 있는 숫자 (예: 3.14)
                                 // 단정밀도(32비트): 약 7자리 정밀도
  char *opcode_latency_dp;      // 배정밀도(double) 부동소수점 연산의 지연 시간 설정 문자열
                                 // 배정밀도(64비트): 약 15자리 정밀도, 더 정확하지만 더 느림
  char *opcode_latency_sfu;     // SFU(Special Function Unit) 연산의 지연 시간 설정 문자열
                                 // SFU: sin, cos, 제곱근 등 특수 수학 함수를 계산하는 장치
  char *opcode_latency_tensor;  // 텐서(tensor) 연산의 지연 시간 설정 문자열
                                 // 텐서 코어: AI/딥러닝에서 행렬 곱셈을 초고속으로 처리하는 장치

  // --- 명령어 개시 간격(initiation interval) 관련 변수들 ---
  // 개시 간격이란? 같은 종류의 다음 명령어를 시작할 수 있을 때까지 기다려야 하는 사이클 수
  // 파이프라인에서 명령어를 얼마나 빠르게 연속으로 넣을 수 있는지를 나타냄
  char *opcode_initiation_int;     // 정수 연산의 개시 간격 설정 문자열
  char *opcode_initiation_fp;      // 단정밀도 부동소수점 연산의 개시 간격 설정 문자열
  char *opcode_initiation_dp;      // 배정밀도 부동소수점 연산의 개시 간격 설정 문자열
  char *opcode_initiation_sfu;     // SFU 연산의 개시 간격 설정 문자열
  char *opcode_initiation_tensor;  // 텐서 연산의 개시 간격 설정 문자열

  // --- 체크포인트 관련 변수들 ---
  // 체크포인트(checkpoint): 시뮬레이션 중간 상태를 저장하는 기능
  // 게임의 "세이브 포인트"와 같은 개념
  int cp_count;       // 체크포인트 횟수: 몇 번째 체크포인트인지
  int cp_cta_resume;  // 체크포인트에서 재개할 CTA 번호: 어떤 CTA부터 이어서 실행할지

  int g_ptxinfo_error_detected;  // PTX 정보 처리 중 오류가 발견되었는지 (0: 정상, 그 외: 오류)

  unsigned g_ptx_sim_num_insn;  // PTX 시뮬레이션에서 지금까지 실행한 총 명령어 수
                                 // unsigned: 0 이상의 정수만 저장 (음수 불가)

  char *cdp_latency_str;  // CDP(CUDA Dynamic Parallelism) 지연 시간 설정 문자열
                           // CDP란? GPU 커널 안에서 또 다른 커널을 실행하는 기능
                           // 마치 함수 안에서 다른 함수를 호출하는 것처럼

  int g_ptx_kernel_count;  // used for classification stat collection purposes
                           // 커널 실행 횟수 카운터: 통계 분류 수집 목적으로 사용
                           // 어떤 커널이 몇 번 실행되었는지 추적

  /*
   * g_global_name_lookup: 전역(global) 변수 이름 조회 테이블
   * map 자료구조: "호스트 변수 포인터 → 디바이스 변수 이름" 매핑
   *
   * 호스트(host): CPU 쪽을 의미
   * 디바이스(device): GPU 쪽을 의미
   * CUDA에서는 CPU와 GPU가 각각 다른 메모리를 사용하므로,
   * CPU 쪽 변수와 GPU 쪽 변수의 연결 관계를 기록해둬야 합니다.
   */
  std::map<const void *, std::string>
      g_global_name_lookup;  // indexed by hostVar (호스트 변수 주소로 검색)

  /*
   * g_const_name_lookup: 상수(constant) 변수 이름 조회 테이블
   * GPU의 상수 메모리에 저장된 변수의 이름을 찾는 데 사용
   * 상수 메모리: 읽기만 가능한 특별한 GPU 메모리 (매우 빠름)
   */
  std::map<const void *, std::string>
      g_const_name_lookup;  // indexed by hostVar (호스트 변수 주소로 검색)

  int g_ptx_sim_mode;  // if non-zero run functional simulation only (i.e., no
                       // notion of a clock cycle)
                       // PTX 시뮬레이션 모드:
                       // 0이 아닌 값이면 기능적 시뮬레이션만 실행 (시간 개념 없이 결과만 계산)
                       // 0이면 타이밍 시뮬레이션 (실제 GPU처럼 클럭 사이클도 계산)

  unsigned gpgpu_param_num_shaders;  // GPU에 있는 셰이더 코어(처리 장치)의 수
                                      // 실제 GPU에서는 수십~수백 개의 코어가 있음

  /*
   * g_rpts: 각 함수별 재수렴 포인트(reconvergence points) 정보를 저장하는 맵
   * function_info*(함수 정보) → rec_pts(재수렴 포인트들) 매핑
   * 분기문(if-else)이 있는 곳에서 스레드들이 다시 합쳐지는 위치를 미리 계산해둠
   */
  class std::map<function_info *, rec_pts> g_rpts;

  bool g_cuda_launch_blocking;  // CUDA 커널 런치 블로킹 모드
                                 // true: 커널 실행이 끝날 때까지 CPU가 기다림
                                 // false: CPU가 기다리지 않고 다른 일을 계속함 (비동기)

  void **g_inst_classification_stat;     // 명령어 분류 통계 데이터를 가리키는 포인터의 포인터
                                          // 어떤 종류의 명령어가 얼마나 실행되었는지 통계
  void **g_inst_op_classification_stat;  // 명령어 연산 종류별 분류 통계 데이터 포인터의 포인터

  std::set<std::string> g_globals;    // 전역 변수 이름들의 집합(set)
                                       // set: 중복 없이 고유한 값들만 저장하는 자료구조
  std::set<std::string> g_constants;  // 상수 변수 이름들의 집합

  /*
   * g_pc_to_finfo: PC(프로그램 카운터) → 함수 정보 매핑
   * 어떤 명령어 주소가 어떤 함수에 속하는지 빠르게 찾기 위한 테이블
   * 예: PC=100번지 → "matrixMul" 함수에 속함
   */
  std::map<unsigned, function_info *> g_pc_to_finfo;

  int gpgpu_ptx_instruction_classification;  // PTX 명령어 분류 기능의 활성화 여부
                                              // 0: 비활성, 그 외: 활성 (통계 수집)

  unsigned cdp_latency[5];  // CDP(CUDA Dynamic Parallelism) 지연 시간 배열
                             // 5개 항목: CDP의 각 단계별 지연 시간을 저장
                             // [5]는 배열 크기가 5라는 뜻 (인덱스 0~4)

  unsigned g_assemble_code_next_pc;  // 다음에 조립(어셈블)할 코드의 프로그램 카운터 값
                                      // PTX 코드를 한 줄씩 조립할 때 다음 줄의 주소

  int g_debug_thread_uid;  // 디버그 대상 스레드의 고유 ID
                            // 이 ID를 가진 스레드만 상세 디버그 정보를 출력
                            // 수천 개 스레드 중 특정 스레드만 추적할 때 유용

  bool g_override_embedded_ptx;  // 바이너리에 내장된 PTX 코드를 외부 파일로 덮어쓸지 여부
                                  // true: 외부 PTX 파일 사용, false: 내장 PTX 사용

  /*
   * g_ptx_cta_info_sm_idx_used: 사용된 SM(Streaming Multiprocessor) 인덱스 집합
   * SM이란? GPU의 핵심 처리 유닛으로, 여러 개의 코어를 포함하는 큰 단위
   * unsigned long long: 매우 큰 정수를 저장할 수 있는 타입 (64비트)
   * set으로 관리하여 어떤 SM이 이미 사용되었는지 빠르게 확인
   */
  std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;

  ptx_reg_t *ptx_tex_regs;  // PTX 텍스처 레지스터 포인터
                              // 레지스터: CPU/GPU에서 가장 빠른 저장 공간 (아주 작지만 초고속)
                              // 텍스처 연산에 사용되는 특별한 레지스터들

  unsigned g_ptx_thread_info_delete_count;  // 삭제된 PTX 스레드 정보 객체의 수
                                             // 메모리 관리와 디버깅을 위해 추적

  unsigned g_ptx_thread_info_uid_next;  // 다음에 생성될 스레드 정보 객체에 부여할 고유 ID
                                         // 새 스레드가 생길 때마다 이 값을 1씩 증가시켜 부여

  addr_t g_debug_pc;  // 디버그 대상 프로그램 카운터 값
                       // 이 PC 주소의 명령어가 실행될 때 디버그 정보를 출력
                       // 초기값 0xBEEF1518은 "아직 설정되지 않음"을 나타내는 마커

  // backward pointer
  // 역방향 포인터: 이 객체를 포함하고 있는 상위 객체(gpgpu_context)를 가리킴
  // 마치 "내가 속한 학교"를 가리키는 것과 같음
  class gpgpu_context *gpgpu_ctx;  // GPGPU 컨텍스트(전체 시뮬레이터 환경)를 가리키는 포인터

  /*
   * ======================================================================
   * 전역 함수들 (global functions)
   * ======================================================================
   * 아래는 CUDA 시뮬레이션에서 사용하는 주요 함수들의 선언입니다.
   */

  /*
   * ptx_opcocde_latency_options: 명령어 지연 시간 옵션을 설정하는 함수
   * 시뮬레이션에서 각 명령어가 얼마나 걸리는지 사용자가 설정할 수 있게 해줌
   *
   * 매개변수:
   * - opp: 옵션 파서(option_parser_t) - 설정 파일이나 명령줄 옵션을 읽는 도구
   */
  void ptx_opcocde_latency_options(option_parser_t opp);

  /*
   * gpgpu_cuda_ptx_sim_main_func: CUDA PTX 시뮬레이션의 메인(주요) 실행 함수
   * 커널을 받아서 시뮬레이션을 실행하는 핵심 함수입니다.
   *
   * 매개변수:
   * - kernel: 실행할 커널 정보 (참조로 전달)
   * - openCL: OpenCL 모드 여부 (기본값: false)
   *           true면 OpenCL 방식으로 실행, false면 CUDA 방식으로 실행
   */
  void gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel, bool openCL = false);

  /*
   * gpgpu_opencl_ptx_sim_main_func: OpenCL용 PTX 시뮬레이션 메인 함수
   * OpenCL 커널을 시뮬레이션하는 함수
   *
   * 매개변수:
   * - grid: 실행할 그리드(커널) 정보 포인터
   *         그리드(grid): 블록들의 모음, 전체 작업 크기를 나타냄
   *
   * 반환값: int - 실행 결과 (0이면 성공, 그 외면 오류)
   */
  int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid);

  /*
   * init_inst_classification_stat: 명령어 분류 통계를 초기화하는 함수
   * 시뮬레이션 시작 전에 통계 수집을 위한 자료구조를 준비합니다
   */
  void init_inst_classification_stat();

  /*
   * gpgpu_opencl_ptx_sim_init_grid: OpenCL용 그리드를 초기화하는 함수
   * 커널 실행을 위한 그리드 구조를 만들고 설정합니다.
   *
   * 매개변수:
   * - entry: 실행할 커널 함수의 정보 포인터
   * - args: 커널에 전달할 인자(argument) 목록
   *         인자란? 함수에 넣어주는 입력값들
   * - gridDim: 그리드의 차원(크기) - 블록이 몇 개인지 (가로 x 세로 x 깊이)
   *            dim3는 3차원 크기를 나타내는 구조체 (x, y, z 값)
   * - blockDim: 블록의 차원(크기) - 블록 안에 스레드가 몇 개인지
   * - gpu: GPU 객체 포인터
   *
   * 반환값: kernel_info_t* - 새로 만들어진 그리드(커널) 정보 포인터
   */
  kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                                gpgpu_ptx_sim_arg_list_t args,
                                                struct dim3 gridDim,
                                                struct dim3 blockDim,
                                                gpgpu_t *gpu);

  /*
   * gpgpu_ptx_sim_register_global_variable: 전역 변수를 시뮬레이터에 등록하는 함수
   * CPU(호스트)에서 선언한 전역 변수를 GPU(디바이스) 시뮬레이션에서도 사용할 수 있게 등록
   *
   * 매개변수:
   * - hostVar: 호스트(CPU) 쪽 변수의 주소
   * - deviceName: 디바이스(GPU) 쪽에서 사용할 변수 이름
   * - size: 변수의 크기 (바이트 단위)
   */
  void gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                              const char *deviceName,
                                              size_t size);

  /*
   * gpgpu_ptx_sim_register_const_variable: 상수 변수를 시뮬레이터에 등록하는 함수
   * GPU의 상수 메모리에 저장될 변수를 등록
   * 상수 메모리는 모든 스레드가 같은 값을 읽을 때 매우 빠릅니다
   *
   * 매개변수:
   * - void*: 호스트 쪽 변수의 주소 (이름 생략됨)
   * - deviceName: 디바이스 쪽 상수 변수 이름
   * - size: 변수의 크기 (바이트 단위)
   */
  void gpgpu_ptx_sim_register_const_variable(void *, const char *deviceName,
                                             size_t size);

  /*
   * read_sim_environment_variables: 시뮬레이션 환경 변수를 읽는 함수
   * 운영체제의 환경 변수에서 시뮬레이션 설정을 가져옵니다
   * 환경 변수란? 운영체제에 저장된 설정 값 (예: PATH, HOME 등)
   */
  void read_sim_environment_variables();

  /*
   * set_param_gpgpu_num_shaders: 셰이더 코어 수를 설정하는 함수
   *
   * 매개변수:
   * - num_shaders: 설정할 셰이더 코어의 수
   */
  void set_param_gpgpu_num_shaders(int num_shaders);

  /*
   * find_reconvergence_points: 재수렴 포인트를 찾는 함수
   * 함수 내의 분기문(if-else, for, while 등) 후에
   * 스레드들이 다시 합쳐지는 지점을 분석하여 찾습니다
   *
   * 매개변수:
   * - finfo: 분석할 함수의 정보 포인터
   *
   * 반환값: rec_pts 구조체 - 찾아낸 재수렴 포인트들의 정보
   */
  struct rec_pts find_reconvergence_points(function_info *finfo);

  /*
   * get_converge_point: 특정 PC 주소에 대응하는 수렴 지점을 가져오는 함수
   *
   * 매개변수:
   * - pc: 현재 프로그램 카운터 값 (현재 명령어 주소)
   *
   * 반환값: address_type - 수렴 지점의 주소
   */
  address_type get_converge_point(address_type pc);

  /*
   * gpgpu_ptx_sim_memcpy_symbol: 심볼(변수)에 대한 메모리 복사를 수행하는 함수
   * CPU와 GPU 사이에서 데이터를 복사합니다 (CUDA의 cudaMemcpyToSymbol/cudaMemcpyFromSymbol)
   *
   * 매개변수:
   * - hostVar: 호스트 쪽 변수 이름/주소
   * - src: 복사할 원본 데이터의 주소
   * - count: 복사할 바이트 수
   * - offset: 대상 변수에서 시작할 위치 (바이트 단위 오프셋)
   * - to: 복사 방향 (호스트→디바이스 또는 디바이스→호스트)
   * - gpu: GPU 객체 포인터
   */
  void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                   size_t count, size_t offset, int to,
                                   gpgpu_t *gpu);

  /*
   * ptx_print_insn: 특정 PC 주소의 PTX 명령어를 파일에 출력하는 함수
   * 디버깅 시 현재 어떤 명령어가 실행 중인지 확인하는 데 사용
   *
   * 매개변수:
   * - pc: 출력할 명령어의 프로그램 카운터 값
   * - fp: 출력할 대상 파일 포인터 (FILE*: C언어의 파일 핸들)
   */
  void ptx_print_insn(address_type pc, FILE *fp);

  /*
   * ptx_get_insn_str: 특정 PC 주소의 PTX 명령어를 문자열로 가져오는 함수
   *
   * 매개변수:
   * - pc: 가져올 명령어의 프로그램 카운터 값
   *
   * 반환값: std::string - 명령어를 나타내는 문자열
   */
  std::string ptx_get_insn_str(address_type pc);

  /*
   * ptx_debug_exec_dump_cond: 디버그 실행 덤프 조건을 확인하는 템플릿 함수
   *
   * 템플릿(template)이란? 여러 타입이나 값에 대해 동일한 코드를 재사용하는 기능
   * <int activate_level>은 "활성화 레벨"을 컴파일 시간에 결정하겠다는 뜻
   * 레벨에 따라 더 자세하거나 간단한 디버그 정보를 출력할지 결정
   *
   * 매개변수:
   * - thd_uid: 확인할 스레드의 고유 ID
   * - pc: 확인할 프로그램 카운터 값
   *
   * 반환값: bool - true면 이 스레드/PC에 대해 디버그 정보를 출력해야 함
   */
  template <int activate_level>
  bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc);
};  // cuda_sim 클래스 정의 끝

#endif  // CUDASIM_H_INCLUDED의 끝: 인클루드 가드의 닫는 부분
        // #ifndef로 시작한 조건부 컴파일 블록을 여기서 닫습니다
