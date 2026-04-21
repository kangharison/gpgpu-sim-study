/*
 * ============================================================================
 * 파일 이름: gpgpusim_entrypoint.h
 * ============================================================================
 * 이 파일은 GPGPU-Sim 시뮬레이터의 "진입점(entrypoint)" 헤더 파일입니다.
 *
 * "진입점"이란?
 *   프로그램이 시작되는 곳을 말합니다. 이 파일은 GPU 시뮬레이션을
 *   시작하고 관리하는 데 필요한 핵심 데이터(변수)들을 모아놓은 클래스를
 *   정의합니다.
 *
 * "헤더 파일(.h)"이란?
 *   C++에서 클래스나 함수의 "설계도"를 담는 파일입니다.
 *   실제 동작하는 코드는 .cc 파일에 있고, 여기서는 "이런 것들이 있다"고
 *   선언(declaration)만 합니다.
 *
 * GPGPU-Sim이란?
 *   실제 GPU(그래픽 카드) 하드웨어 없이도 GPU가 어떻게 동작하는지
 *   컴퓨터 소프트웨어로 흉내 내는(시뮬레이션하는) 프로그램입니다.
 *   연구자들이 GPU 설계를 테스트할 때 사용합니다.
 * ============================================================================
 */

// Copyright (c) 2009-2011, Tor M. Aamodt
// 저작권: 2009~2011년, Tor M. Aamodt라는 사람이 만들었습니다.

// The University of British Columbia
// 캐나다 브리티시 컬럼비아 대학교에서 개발되었습니다.

// All rights reserved.
// 모든 권리가 보호됩니다. (허락 없이 마음대로 쓸 수 없다는 뜻)

//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 소스 코드나 실행 파일 형태로 다시 배포하거나 사용할 수 있지만,
// 아래의 조건들을 반드시 지켜야 합니다:

//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 소스 코드를 배포할 때는 위의 저작권 표시와 이 조건 목록,
// 그리고 아래의 면책 조항을 반드시 포함해야 합니다.

// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
// 실행 파일(바이너리) 형태로 배포할 때도 문서에 저작권 표시를 넣어야 하고,
// 대학 이름이나 개발자 이름을 허락 없이 광고에 사용할 수 없습니다.

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
/*
 * 위 내용은 "면책 조항(disclaimer)"입니다.
 * 쉽게 말하면: "이 소프트웨어를 있는 그대로 제공하며,
 * 이걸 사용해서 문제가 생겨도 개발자에게 책임을 물을 수 없습니다."
 * 라는 뜻입니다. 대부분의 오픈소스 소프트웨어에 이런 문구가 있습니다.
 * 이것을 "BSD 라이선스"라고 부릅니다.
 */

/*
 * ============================================================================
 * 여기서부터 실제 코드가 시작됩니다!
 * ============================================================================
 */

#ifndef GPGPUSIM_ENTRYPOINT_H_INCLUDED  // "만약 GPGPUSIM_ENTRYPOINT_H_INCLUDED가 아직 정의되지 않았다면"
#define GPGPUSIM_ENTRYPOINT_H_INCLUDED  // "GPGPUSIM_ENTRYPOINT_H_INCLUDED를 정의해라"
/*
 * 위 두 줄은 "인클루드 가드(include guard)"라고 부릅니다.
 *
 * 왜 필요한가요?
 *   여러 파일에서 이 헤더 파일을 #include(포함)할 수 있는데,
 *   같은 내용이 두 번 이상 포함되면 "이미 정의했는데 또 정의했다!"라는
 *   컴파일 에러가 발생합니다.
 *   인클루드 가드를 사용하면 처음 한 번만 포함되고,
 *   두 번째부터는 건너뛰게 됩니다.
 *
 * 작동 원리:
 *   1) 처음 이 파일을 만나면 GPGPUSIM_ENTRYPOINT_H_INCLUDED가 없으므로
 *      #ifndef(if not defined) 조건이 참(true)이 되어 안의 코드가 실행됩니다.
 *   2) #define으로 GPGPUSIM_ENTRYPOINT_H_INCLUDED를 정의합니다.
 *   3) 두 번째로 이 파일을 만나면 이미 정의되어 있으므로
 *      #ifndef 조건이 거짓(false)이 되어 코드를 건너뜁니다.
 */

#include <pthread.h>  // POSIX 쓰레드(pthread) 라이브러리를 포함합니다.
/*
 * pthread란?
 *   "쓰레드(thread)"는 프로그램 안에서 동시에 여러 작업을 하는 것을 말합니다.
 *   예를 들어, 음악을 들으면서 게임을 하는 것처럼요.
 *   pthread는 리눅스/유닉스에서 쓰레드를 만들고 관리하는 도구입니다.
 *   여기서는 GPU 시뮬레이션을 별도의 쓰레드에서 실행하기 위해 사용합니다.
 *   (즉, 시뮬레이션이 돌아가는 동안 다른 작업도 할 수 있게 합니다.)
 */

#include <semaphore.h>  // 세마포어(semaphore) 라이브러리를 포함합니다.
/*
 * 세마포어(semaphore)란?
 *   여러 쓰레드가 동시에 실행될 때, 서로 "신호"를 보내는 도구입니다.
 *   마치 교통 신호등처럼, 한 쓰레드가 "이제 시작해도 돼!"라고
 *   다른 쓰레드에게 알려주는 역할을 합니다.
 *   예: 시뮬레이션 쓰레드가 준비되면 메인 쓰레드에게 "준비 완료!" 신호를 보냅니다.
 */

#include <time.h>  // 시간 관련 라이브러리를 포함합니다.
/*
 * time.h는 현재 시간을 알아내거나, 시간을 측정하는 기능을 제공합니다.
 * 여기서는 시뮬레이션이 시작된 시간을 기록하는 데 사용합니다.
 * (시뮬레이션이 얼마나 오래 걸렸는지 계산할 수 있습니다.)
 */

#include "abstract_hardware_model.h"  // 추상 하드웨어 모델 헤더 파일을 포함합니다.
/*
 * "abstract_hardware_model.h"란?
 *   GPU 하드웨어를 소프트웨어로 표현한 "추상 모델"이 정의된 파일입니다.
 *   "추상(abstract)"이란 세부사항은 숨기고 중요한 것만 보여준다는 뜻입니다.
 *   예: 실제 GPU에는 수십억 개의 트랜지스터가 있지만,
 *       시뮬레이터에서는 "명령어를 받아서 실행하는 장치"로만 표현합니다.
 *
 * #include "파일이름" (큰따옴표):
 *   같은 프로젝트 안에 있는 우리가 만든 파일을 포함할 때 사용합니다.
 * #include <파일이름> (꺾쇠괄호):
 *   시스템에 이미 설치된 라이브러리를 포함할 때 사용합니다.
 */

// extern time_t g_simulation_starttime;
/*
 * 위 줄은 주석 처리(비활성화)되어 있습니다.
 * 원래는 "시뮬레이션 시작 시간"을 전역 변수로 선언하려 했지만,
 * 나중에 클래스 안으로 옮겨졌기 때문에 더 이상 필요 없어서 주석 처리한 것입니다.
 *
 * "extern"이란?
 *   "이 변수는 다른 파일에 이미 만들어져 있으니, 여기서는 그걸 쓰겠다"는 의미입니다.
 * "time_t"란?
 *   시간을 저장하는 데이터 타입(자료형)입니다.
 */

class gpgpu_context;  // gpgpu_context 클래스가 있다고 미리 알려줍니다. (전방 선언)
/*
 * "전방 선언(forward declaration)"이란?
 *   클래스의 자세한 내용은 아직 모르지만, "이런 이름의 클래스가 있다"고
 *   컴파일러에게 미리 알려주는 것입니다.
 *   왜 필요한가요? 아래 GPGPUsim_ctx 클래스 안에서 gpgpu_context의
 *   포인터(*)를 사용하는데, 포인터만 사용할 때는 클래스의 자세한
 *   내용(크기, 멤버 변수 등)을 몰라도 되기 때문입니다.
 *   (포인터는 그냥 "주소"만 저장하니까 크기가 항상 같습니다.)
 */

/*
 * ============================================================================
 * GPGPUsim_ctx 클래스 정의
 * ============================================================================
 * 이 클래스는 GPU 시뮬레이션의 "상태(context)"를 저장합니다.
 * "ctx"는 "context(문맥, 상태)"의 줄임말입니다.
 *
 * 시뮬레이션이 실행 중인지, 완료되었는지, GPU 설정은 무엇인지 등
 * 시뮬레이션에 필요한 모든 정보를 이 클래스 하나에 담고 있습니다.
 *
 * 왜 이런 클래스가 필요한가요?
 *   시뮬레이션에 필요한 변수들이 매우 많은데, 이것들을 하나의 클래스로
 *   묶어두면 관리하기 편하고, 여러 곳에서 쉽게 접근할 수 있기 때문입니다.
 * ============================================================================
 */
class GPGPUsim_ctx {  // GPGPUsim_ctx라는 이름의 클래스를 정의합니다.
 public:  // "public"은 "공개"라는 뜻입니다. 아래 내용은 클래스 바깥에서도 접근할 수 있습니다.
  /*
   * public vs private:
   *   public: 누구나 접근할 수 있음 (교실의 게시판처럼)
   *   private: 클래스 내부에서만 접근할 수 있음 (개인 사물함처럼)
   *   이 클래스는 모든 멤버가 public이므로 어디서든 자유롭게 사용할 수 있습니다.
   */

  GPGPUsim_ctx(gpgpu_context *ctx) {  // 생성자(constructor) - 객체가 만들어질 때 자동으로 실행되는 함수
  /*
   * "생성자(constructor)"란?
   *   클래스로부터 객체(실체)를 만들 때 자동으로 호출되는 특별한 함수입니다.
   *   클래스 이름과 같은 이름을 가집니다.
   *   예: GPGPUsim_ctx myObj(some_ctx); 라고 쓰면 이 생성자가 자동 실행됩니다.
   *
   * 매개변수(parameter):
   *   gpgpu_context *ctx : gpgpu_context 객체의 "포인터(주소)"를 받습니다.
   *   포인터(*)란? 어떤 데이터가 메모리 어디에 있는지 "주소"를 저장하는 변수입니다.
   *   마치 친구 집 주소를 적어둔 메모지 같은 것입니다.
   */

    g_sim_active = false;  // 시뮬레이션이 현재 "활성(실행 중)" 상태인지 표시. 처음에는 false(아니오).
    /*
     * bool 타입은 true(참/예) 또는 false(거짓/아니오) 두 가지 값만 가집니다.
     * 시뮬레이션이 아직 시작하지 않았으므로 false로 초기화합니다.
     */

    g_sim_done = true;  // 시뮬레이션이 "완료"되었는지 표시. 처음에는 true(예).
    /*
     * 왜 처음에 true(완료)인가요?
     *   아직 시뮬레이션을 시작하지 않았으므로, "할 일이 없는 상태" = "완료"로 봅니다.
     *   나중에 시뮬레이션이 시작되면 false로 바뀌고, 끝나면 다시 true가 됩니다.
     */

    break_limit = false;  // 시뮬레이션 실행 한도(제한)를 넘었는지 표시. 처음에는 false(아니오).
    /*
     * 시뮬레이션은 무한히 실행될 수 있으므로, 특정 한도를 정해두고
     * 그 한도를 넘으면 시뮬레이션을 멈추는 기능이 있습니다.
     * break_limit가 true가 되면 "한도를 넘었으니 멈춰라"는 신호입니다.
     */

    g_sim_lock = PTHREAD_MUTEX_INITIALIZER;  // 뮤텍스 잠금장치를 초기화합니다.
    /*
     * "뮤텍스(mutex)"란?
     *   "mutual exclusion(상호 배제)"의 줄임말입니다.
     *   여러 쓰레드가 동시에 같은 데이터를 수정하면 문제가 생길 수 있습니다.
     *   예: 두 사람이 동시에 같은 문서를 편집하면 내용이 엉망이 되는 것처럼요.
     *   뮤텍스는 "열쇠가 하나뿐인 화장실 자물쇠" 같은 것입니다.
     *   한 쓰레드가 잠금(lock)을 걸면, 다른 쓰레드는 그 쓰레드가
     *   잠금을 풀(unlock)때까지 기다려야 합니다.
     *   PTHREAD_MUTEX_INITIALIZER는 이 잠금장치를 기본 설정으로 만들어줍니다.
     */

    g_the_gpu_config = NULL;  // GPU 설정 정보 포인터를 NULL(비어있음)으로 초기화
    /*
     * NULL이란?
     *   "아직 아무것도 가리키지 않는다"는 뜻입니다.
     *   마치 주소록에 주소를 아직 적지 않은 상태입니다.
     *   나중에 실제 GPU 설정 객체가 만들어지면 그 주소가 여기에 저장됩니다.
     */

    g_the_gpu = NULL;  // GPU 시뮬레이터 객체 포인터를 NULL로 초기화
    // 실제 GPU를 소프트웨어로 흉내 내는 gpgpu_sim 객체를 나중에 여기에 연결합니다.

    g_stream_manager = NULL;  // 스트림 관리자 포인터를 NULL로 초기화
    /*
     * "스트림(stream)"이란?
     *   GPU에서 작업들을 순서대로 처리하는 "작업 줄(대기열)"입니다.
     *   마치 놀이공원의 줄서기처럼, 작업들이 줄을 서서 차례를 기다립니다.
     *   스트림 관리자는 이 줄들을 관리하는 역할을 합니다.
     */

    the_cude_device = NULL;  // CUDA 디바이스(장치) 포인터를 NULL로 초기화
    /*
     * "CUDA 디바이스"란?
     *   NVIDIA GPU 장치를 의미합니다. CUDA는 NVIDIA가 만든
     *   GPU 프로그래밍 기술 이름입니다.
     *   참고: "cude"는 "cuda"의 오타(typo)로 보이지만, 원래 코드에 이렇게 되어 있습니다.
     */

    the_context = NULL;  // CUDA 컨텍스트 포인터를 NULL로 초기화
    /*
     * "CUDA 컨텍스트(context)"란?
     *   GPU를 사용하기 위한 "작업 환경"입니다.
     *   마치 그림을 그리려면 도화지, 물감, 붓 등 환경이 필요하듯이,
     *   GPU 프로그래밍에도 메모리, 설정 등의 환경이 필요합니다.
     *   CUctx_st 구조체가 이 환경 정보를 담고 있습니다.
     */

    gpgpu_ctx = ctx;  // 매개변수로 받은 gpgpu_context 포인터를 멤버 변수에 저장합니다.
    /*
     * 생성자의 매개변수 ctx를 클래스 멤버 변수 gpgpu_ctx에 저장합니다.
     * 이렇게 하면 나중에 클래스 안의 다른 곳에서도 gpgpu_context에 접근할 수 있습니다.
     * 이것은 "의존성 주입(dependency injection)"이라는 설계 패턴입니다.
     * 쉽게 말하면 "밖에서 만든 것을 안으로 전달해주는 것"입니다.
     */
  }  // 생성자 끝

  // struct gpgpu_ptx_sim_arg *grid_params;
  /*
   * 위 줄은 주석 처리되어 있습니다. (사용하지 않는 코드)
   * 원래는 GPU 커널(프로그램) 실행에 필요한 매개변수들을 저장하려 했던 것 같습니다.
   * PTX: NVIDIA GPU의 중간 기계어(어셈블리 언어 비슷한 것)
   * grid_params: GPU에서 작업을 나누는 단위인 "그리드"의 매개변수
   */

  /*
   * ========================================================================
   * 세마포어(semaphore) 변수들
   * ========================================================================
   * 아래 3개의 세마포어는 시뮬레이션 쓰레드와 메인 쓰레드 사이에서
   * "신호"를 주고받는 데 사용됩니다.
   * sem_t는 세마포어의 데이터 타입(자료형)입니다.
   */
  sem_t g_sim_signal_start;  // 시뮬레이션 "시작" 신호를 보내는 세마포어
  // 메인 쓰레드가 "시뮬레이션을 시작해라!"라고 시뮬레이션 쓰레드에게 알려줄 때 사용합니다.

  sem_t g_sim_signal_finish;  // 시뮬레이션 "완료" 신호를 보내는 세마포어
  // 시뮬레이션 쓰레드가 "한 단계가 끝났어!"라고 메인 쓰레드에게 알려줄 때 사용합니다.

  sem_t g_sim_signal_exit;  // 시뮬레이션 "종료" 신호를 보내는 세마포어
  // 시뮬레이션이 완전히 끝나서 "이제 프로그램을 종료해도 돼!"라고 알려줄 때 사용합니다.

  time_t g_simulation_starttime;  // 시뮬레이션이 시작된 시간을 저장하는 변수
  /*
   * time_t 타입은 1970년 1월 1일 0시 0분 0초부터 지금까지 흐른 "초(seconds)"를
   * 숫자로 저장합니다. 이것을 "유닉스 타임스탬프(Unix timestamp)"라고 부릅니다.
   * 시뮬레이션이 얼마나 오래 걸렸는지 계산하는 데 사용합니다.
   * 예: 끝난 시간 - 시작 시간 = 소요 시간
   */

  pthread_t g_simulation_thread;  // 시뮬레이션을 실행하는 쓰레드(thread)의 식별자(ID)
  /*
   * pthread_t는 쓰레드를 구별하는 "이름표" 같은 것입니다.
   * GPU 시뮬레이션은 별도의 쓰레드에서 실행되는데,
   * 이 변수가 그 쓰레드를 가리킵니다.
   * 나중에 이 쓰레드에게 명령을 내리거나 상태를 확인할 때 이 식별자를 사용합니다.
   */

  /*
   * ========================================================================
   * 시뮬레이션의 핵심 객체 포인터들
   * ========================================================================
   * 아래 3개의 포인터는 시뮬레이션의 가장 중요한 구성 요소들입니다.
   * 각각 GPU 설정, GPU 자체, 그리고 작업 스트림을 관리합니다.
   */

  class gpgpu_sim_config *g_the_gpu_config;  // GPU 설정(configuration) 객체를 가리키는 포인터
  /*
   * gpgpu_sim_config 클래스는 시뮬레이션할 GPU의 설정 정보를 담고 있습니다.
   * 예: GPU 코어(SM)가 몇 개인지, 메모리 크기는 얼마인지,
   *     클럭 속도는 얼마인지 등의 설정값들입니다.
   * "class" 키워드는 이것이 클래스 타입이라고 알려줍니다. (전방 선언의 한 형태)
   * "*"는 이것이 포인터(주소를 저장하는 변수)라는 뜻입니다.
   */

  class gpgpu_sim *g_the_gpu;  // GPU 시뮬레이터 핵심 객체를 가리키는 포인터
  /*
   * gpgpu_sim은 GPU 전체를 소프트웨어로 표현한 클래스입니다.
   * 이것이 바로 시뮬레이션의 "심장"입니다!
   * GPU의 코어, 메모리, 캐시 등 모든 하드웨어 부품이 이 안에 들어있습니다.
   * "g_the_gpu"라는 이름은 "전역(global)의, 그(the), GPU"라는 뜻입니다.
   */

  class stream_manager *g_stream_manager;  // 스트림 관리자 객체를 가리키는 포인터
  /*
   * stream_manager는 GPU로 보내는 작업(커널 실행, 메모리 복사 등)의
   * 순서를 관리하는 클래스입니다.
   * 실제 CUDA 프로그래밍에서도 "스트림"을 사용하여 작업 순서를 제어합니다.
   */

  /*
   * ========================================================================
   * CUDA 런타임 관련 포인터들
   * ========================================================================
   * 아래 2개는 CUDA 런타임(실행 환경)을 시뮬레이션하기 위한 구조체 포인터입니다.
   */

  struct _cuda_device_id *the_cude_device;  // CUDA 디바이스(GPU 장치) 정보를 가리키는 포인터
  /*
   * _cuda_device_id 구조체는 시뮬레이션하는 GPU 장치의 정보를 담고 있습니다.
   * "struct"는 여러 변수를 하나로 묶은 데이터 구조입니다.
   * (클래스와 비슷하지만, C 언어에서 온 더 간단한 형태입니다.)
   * 참고: "cude"는 "cuda"의 오타입니다. 원본 코드가 이렇게 작성되어 있습니다.
   */

  struct CUctx_st *the_context;  // CUDA 컨텍스트(실행 환경) 구조체를 가리키는 포인터
  /*
   * CUctx_st는 "CUDA Context State(CUDA 컨텍스트 상태)"의 줄임말입니다.
   * GPU를 사용하려면 먼저 "컨텍스트"를 만들어야 합니다.
   * 이 컨텍스트에는 GPU 메모리 할당 정보, 로드된 프로그램 정보 등이 들어있습니다.
   */

  gpgpu_context *gpgpu_ctx;  // GPGPU-Sim의 전체 컨텍스트를 가리키는 포인터
  /*
   * gpgpu_context는 GPGPU-Sim 시뮬레이터 전체의 "큰 그림" 컨텍스트입니다.
   * 시뮬레이터의 모든 구성 요소들을 하나로 연결하는 "중심부" 역할을 합니다.
   * 생성자에서 매개변수로 받아서 저장한 바로 그 포인터입니다.
   */

  /*
   * ========================================================================
   * 쓰레드 동기화 및 상태 관리 변수들
   * ========================================================================
   */

  pthread_mutex_t g_sim_lock;  // 시뮬레이션 데이터를 보호하는 뮤텍스(잠금장치)
  /*
   * 이 뮤텍스는 여러 쓰레드가 동시에 시뮬레이션 데이터를 수정하는 것을 방지합니다.
   * 예를 들어, 시뮬레이션 쓰레드가 g_sim_active를 바꾸는 동안
   * 메인 쓰레드가 동시에 읽으면 잘못된 값을 읽을 수 있습니다.
   * 뮤텍스로 잠그면 한 번에 하나의 쓰레드만 접근할 수 있습니다.
   *
   * pthread_mutex_t는 뮤텍스의 데이터 타입입니다.
   */

  bool g_sim_active;  // 시뮬레이션이 현재 실행 중(활성 상태)인지 나타내는 변수
  /*
   * true = 시뮬레이션이 지금 돌아가고 있습니다.
   * false = 시뮬레이션이 멈춰있거나 아직 시작하지 않았습니다.
   * "g_"는 "global(전역)"을 의미하는 접두사입니다. (프로그래밍 관례)
   */

  bool g_sim_done;  // 시뮬레이션이 완전히 끝났는지 나타내는 변수
  /*
   * true = 시뮬레이션이 완료되었습니다. 더 이상 할 일이 없습니다.
   * false = 시뮬레이션이 아직 진행 중이거나, 해야 할 일이 남아있습니다.
   *
   * g_sim_active와 g_sim_done의 차이:
   *   g_sim_active: "지금 이 순간" 실행 중인지 (일시 정지일 수도 있음)
   *   g_sim_done: 모든 작업이 "완전히" 끝났는지
   */

  bool break_limit;  // 시뮬레이션 실행 제한에 도달했는지 나타내는 변수
  /*
   * true = 설정된 실행 한도(예: 최대 사이클 수)에 도달하여 시뮬레이션을 멈춰야 합니다.
   * false = 아직 한도에 도달하지 않았으므로 계속 실행해도 됩니다.
   *
   * 왜 실행 한도가 필요한가요?
   *   시뮬레이션은 실제 GPU보다 수백~수천 배 느리므로,
   *   너무 오래 걸리는 것을 방지하기 위해 한도를 설정합니다.
   *   예: "1억 사이클까지만 시뮬레이션하겠다"
   */
};  // GPGPUsim_ctx 클래스 정의 끝
/*
 * C++에서 클래스 정의의 닫는 중괄호(}) 뒤에는 반드시 세미콜론(;)을 붙여야 합니다.
 * 이것은 C++ 문법 규칙입니다. 빼먹으면 컴파일 에러가 발생합니다!
 */

#endif  // GPGPUSIM_ENTRYPOINT_H_INCLUDED
/*
 * 인클루드 가드의 끝입니다.
 * 맨 위의 #ifndef와 짝을 이루는 #endif입니다.
 * 주석으로 어떤 #ifndef에 대응하는지 적어두는 것이 좋은 습관입니다.
 *
 * ============================================================================
 * 파일 요약 (Summary)
 * ============================================================================
 * 이 헤더 파일은 GPGPUsim_ctx 클래스를 정의합니다.
 * 이 클래스가 하는 일:
 *   1. GPU 시뮬레이션의 상태(시작/실행중/완료)를 추적합니다.
 *   2. 쓰레드 간 동기화(세마포어, 뮤텍스)를 관리합니다.
 *   3. GPU 설정, GPU 시뮬레이터, 스트림 관리자 등 핵심 객체들의 포인터를 보관합니다.
 *   4. CUDA 런타임 환경(디바이스, 컨텍스트)의 포인터를 보관합니다.
 *
 * 이 클래스는 시뮬레이션의 "관제탑" 같은 역할을 합니다!
 * ============================================================================
 */
