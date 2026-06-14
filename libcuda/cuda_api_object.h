/*
 * [한국어 설명] CUDA API 객체 정의 헤더 (cuda_api_object.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 libcuda 계층에서 사용하는 핵심 C++ 래퍼 객체들을 정의한다.
 * CUDA 런타임 API가 노출하는 불투명 핸들(opaque handle) — CUdevice, CUcontext,
 * 커널 실행 설정 등 — 과 시뮬레이터 내부 자료구조(gpgpu_sim, symbol_table,
 * function_info) 사이의 브리지 역할을 수행한다.
 * 실제 NVIDIA GPU 드라이버에서 동일한 역할을 하는 커널 내부 객체를 사이클 정확
 * 시뮬레이션 환경에서 소프트웨어로 재현한 것이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 실행 흐름: CUDA 애플리케이션 → libcuda (이 헤더 포함) → 타이밍 모델
 * 이 파일에 정의된 구조체/클래스들은 libcuda/cuda_runtime_api.cc 에서 생성되고
 * CUDA 애플리케이션이 cudaLaunchKernel/cudaMalloc 등을 호출할 때마다 참조된다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스 (시뮬레이터 자체, GPU 코드가 아님).
 * 커널 실행 전까지의 설정 단계(configure → setArg → launch)가 이 헤더의 객체들로
 * 표현된다. 실제 사이클 시뮬레이션은 gpgpu-sim/gpu-sim.cc(cycle())에서 일어난다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 방향 (이 파일 → 아래):
 *   - abstract_hardware_model.h : kernel_info_t, gpgpu_t, gpgpu_ptx_sim_arg 등
 *     시뮬레이터 전반에 걸친 추상 하드웨어 타입
 *   - cuda-sim/ptx_ir.h        : symbol_table, function_info, symbol — PTX IR 객체
 *   - gpgpu-sim/gpu-sim.h      : gpgpu_sim (타이밍 모델 최상위), gpgpu_context
 *   - cuobjdump.h              : cuobjdumpSection 계층 구조 (ELF/PTX 섹션 파싱)
 *   - builtin_types.h          : dim3, cudaDeviceProp 등 CUDA 기본 타입
 * 역방향 (이 파일을 사용하는 쪽):
 *   - libcuda/cuda_runtime_api.cc : 이 헤더의 모든 클래스를 직접 인스턴스화
 *   - libcuda/ 아래 개별 API 구현 파일들
 * 데이터 흐름:
 *   CUDA 애플리케이션이 cudaConfigureCall()로 grid/block dim 설정
 *   → kernel_config 객체에 저장 → g_cuda_launch_stack에 push
 *   → cudaLaunchKernel()에서 pop → gpgpu_cuda_ptx_sim_init_grid()로 전달
 *   → kernel_info_t 생성 → gpgpu_sim::launch()로 시뮬레이션 시작
 *
 * === 주요 함수/구조체 요약 ===
 * glbmap_entry       : OpenGL 버퍼 객체 ↔ GPU 디바이스 포인터 매핑 노드 (연결 리스트)
 * _cuda_device_id    : gpgpu_sim 인스턴스 1개를 감싸는 디바이스 래퍼; 멀티-GPU를
 *                      위한 연결 리스트 구조로 이어진다
 * CUctx_st           : CUDA 컨텍스트; fat binary → symbol_table 맵 및
 *                      hostFun 포인터 → function_info 룩업 테이블을 관리
 * kernel_config      : 하나의 커널 실행에 필요한 설정 (grid/block dims, shared mem,
 *                      스트림, 인수 리스트); g_cuda_launch_stack의 원소
 * cuda_runtime_api   : CUDA 런타임 API 전체를 구현하는 최상위 클래스; 전역 상태
 *                      (malloc 추적, PTX 섹션 목록, 핀드 메모리 맵 등)를 소유
 */

#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__

#include <functional> // [한국어] std::function — cuobjdump 초기화 콜백 타입에 사용
#include <list>       // [한국어] std::list — 커널 인수 리스트, PTX 섹션 리스트, 실행 스택
#include <map>        // [한국어] std::map — fat binary 핸들→심볼 테이블, hostFun→function_info 등
#include <set>        // [한국어] std::set — SM 버전별 파일명 집합 (version_filename)
#include <string>     // [한국어] std::string — 바이너리 이름, 커널 심볼 문자열 처리

#include "builtin_types.h" // [한국어] CUDA 기본 타입 (dim3, cudaDeviceProp, size_t 등)
                           //          GPGPU-Sim 자체 구현체; 실제 CUDA SDK 헤더 대체

#include "../src/abstract_hardware_model.h" // [한국어] kernel_info_t, gpgpu_t, gpgpu_ptx_sim_arg,
                                            //          warp/스레드 추상 모델 — 시뮬레이터 공통 타입
#include "../src/cuda-sim/ptx_ir.h"         // [한국어] symbol_table, function_info, symbol —
                                            //          PTX IR 파싱 결과 객체; 커널 코드 표현
#include "../src/gpgpu-sim/gpu-sim.h"       // [한국어] gpgpu_sim (타이밍 모델 최상위),
                                            //          gpgpu_context — 전역 시뮬레이터 문맥
#include "cuobjdump.h"                      // [한국어] cuobjdumpSection 계층 (ELF/PTX 섹션 파싱);
                                            //          fat binary에서 PTX 텍스트를 추출하는 파이프라인

/* [한국어] 커널 실행 인수 하나를 담는 gpgpu_ptx_sim_arg 객체의 연결 리스트 타입.
 * 설정자: kernel_config::set_arg()가 push_front()로 인수를 앞에 삽입.
 * 읽는 자: gpgpu_cuda_ptx_sim_init_grid()가 이 리스트를 순회하여 PTX 인수를 세팅.
 * 값 범위: 커널 파라미터 수만큼의 원소; CUDA 호스트 코드에서 넘기는 포인터/값들. */
typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

#ifndef OPENGL_SUPPORT
/* [한국어] OpenGL 지원이 비활성화된 빌드에서 GLuint 타입을 직접 정의한다.
 * 이유: OpenGL 헤더 없이도 glbmap_entry 구조체를 컴파일할 수 있도록 하기 위함.
 * 실제 OpenGL 빌드(OPENGL_SUPPORT 정의 시)에는 GL/gl.h의 GLuint가 사용된다. */
typedef unsigned long GLuint;
#endif

/*
 * [한국어] glbmap_entry — OpenGL 버퍼 객체 ↔ GPU 디바이스 포인터 매핑 노드
 *
 * OpenGL-CUDA 상호운용(interop)을 위한 단방향 연결 리스트의 노드.
 * cudaGLRegisterBufferObject() 등이 호출되면 OpenGL VBO(Vertex Buffer Object)를
 * CUDA가 직접 접근할 수 있는 디바이스 포인터로 매핑해야 한다. 이 구조체가 그
 * 매핑 정보를 하나의 노드로 표현하며, cuda_runtime_api::g_glbmap 연결 리스트로
 * 전체 등록된 버퍼들을 추적한다.
 */
struct glbmap_entry {
  GLuint m_bufferObj;
  /* OpenGL에서 생성된 버퍼 객체 이름(정수 식별자).
   * 설정자: cudaGLRegisterBufferObject() 래퍼가 이 노드를 생성할 때 할당.
   * 읽는 자: cudaGLMapBufferObject()가 이 ID로 노드를 검색하여 m_devPtr 반환.
   * 값 범위: 0이 아닌 양의 정수 (GL 규약상 0은 예약됨).
   * 동기화: 호스트 단일 스레드에서만 접근하므로 별도 락 불필요. */

  void *m_devPtr;
  /* OpenGL 버퍼와 매핑된 CUDA 디바이스 메모리 포인터.
   * 설정자: cudaGLMapBufferObject()가 실제 매핑을 수행한 후 저장.
   * 읽는 자: CUDA 커널이 이 포인터를 통해 GL 버퍼 데이터를 직접 읽고 씀.
   * 값 범위: NULL(맵 미수행 상태) 또는 유효한 시뮬레이터 디바이스 주소.
   * 동기화: 맵/언맵 쌍 사이에서만 유효; 언맵 후 접근은 정의되지 않은 동작. */

  size_t m_size;
  /* 매핑된 버퍼의 바이트 크기.
   * 설정자: 버퍼 등록 시 GL 쿼리로 얻은 크기를 저장.
   * 읽는 자: 경계 검사나 매핑 크기 전달 시 참조.
   * 값 범위: 1 이상의 양수; GL 버퍼의 실제 할당 크기.
   * 동기화: 등록 이후 불변 값이므로 별도 동기화 불필요. */

  struct glbmap_entry *m_next;
  /* 다음 등록된 GL 버퍼 매핑 노드를 가리키는 포인터 (단방향 연결 리스트).
   * 설정자: 새 노드를 g_glbmap 리스트의 선두에 삽입할 때 기존 헤드를 연결.
   * 읽는 자: 리스트 순회 시 다음 노드로 이동하는 포인터.
   * 값 범위: 유효한 glbmap_entry 포인터 또는 NULL(리스트 끝).
   * 동기화: 리스트 삽입/삭제는 단일 호스트 스레드에서 수행. */
};

/* [한국어] glbmap_entry 구조체의 C 스타일 typedef.
 * C/C++ 혼용 코드에서 struct 키워드 없이 glbmap_entry_t로 참조할 수 있도록
 * 편의 별칭을 제공한다. */
typedef struct glbmap_entry glbmap_entry_t;

/*
 * [한국어] _cuda_device_id — 시뮬레이션 대상 GPU 디바이스 래퍼
 *
 * gpgpu_sim 인스턴스 하나를 감싸며 CUDA API 레벨의 디바이스 추상화를 제공한다.
 * 멀티-GPU 시뮬레이션을 지원하기 위해 연결 리스트 형태로 여러 디바이스를 연결한다.
 * cudaGetDeviceCount(), cudaSetDevice() 등 디바이스 관리 API가 이 객체를 통해
 * 시뮬레이터 내부로 접근한다.
 * 실제 NVIDIA 드라이버에서 CUdevice 핸들이 가리키는 커널 객체에 상응한다.
 */
struct _cuda_device_id {
  /*
   * [한국어]
   * _cuda_device_id 생성자 — 단일 GPU 디바이스 노드 초기화
   *
   * @gpu: 이 디바이스 노드가 래핑할 gpgpu_sim 인스턴스 포인터.
   *        gpgpusim_entrypoint.cc의 gpgpu_sim_init()에서 전달됨.
   * @return: (생성자, 반환값 없음)
   *
   * 새 디바이스 노드를 리스트의 첫 번째 원소(ID 0)로 초기화한다.
   * m_next를 NULL로 설정하여 단일 디바이스 상태를 표현하고,
   * m_id를 0으로 고정하여 CUDA 디바이스 인덱스의 시작값으로 설정한다.
   * 실행 컨텍스트: 호스트 초기화 단계 (시뮬레이터 기동 시 1회).
   *
   * 호출 체인:
   *   gpgpusim_entrypoint.cc → [_cuda_device_id(gpgpu_sim*)] → (필드 초기화)
   */
  _cuda_device_id(gpgpu_sim *gpu) {
    m_id = 0;     // [한국어] 디바이스 인덱스를 0으로 초기화 (CUDA 디바이스 번호는 0-based)
    m_next = NULL; // [한국어] 연결 리스트 끝 표시 — 현재 단일 디바이스 상태
    m_gpgpu = gpu; // [한국어] 타이밍 모델 최상위 포인터 저장 (이 노드의 실제 시뮬레이터)
  }

  /*
   * [한국어]
   * next — 연결 리스트에서 다음 디바이스 노드 반환
   *
   * @return: 다음 _cuda_device_id 노드 포인터; 마지막 디바이스이면 NULL.
   *
   * 멀티-GPU 시뮬레이션에서 디바이스 목록을 순회할 때 사용한다.
   * GPGPU-Sim은 현재 단일 GPU가 일반적이나, 확장성을 위해 연결 리스트를 유지한다.
   *
   * 호출 체인:
   *   cuda_runtime_api 내 디바이스 열거 로직 → [next()] → m_next 반환
   */
  struct _cuda_device_id *next() {
    return m_next; // [한국어] 다음 디바이스 노드를 반환 (없으면 NULL)
  }

  /*
   * [한국어]
   * num_shader — 이 디바이스에 구성된 shader core(SM) 수 반환
   *
   * @return: gpgpusim.config에서 설정된 SM(Streaming Multiprocessor) 개수.
   *
   * cudaGetDeviceProperties()의 multiProcessorCount 필드 계산에 사용된다.
   * 내부적으로 gpgpu_sim_config::num_shader()를 위임 호출하여 설정값을 읽는다.
   * 실행 컨텍스트: 호스트, 읽기 전용 쿼리.
   *
   * 호출 체인:
   *   cudaGetDeviceProperties() → [num_shader()] → gpgpu_sim_config::num_shader()
   */
  unsigned num_shader() const { return m_gpgpu->get_config().num_shader(); }
  // [한국어] gpgpu_sim의 설정 객체에서 SM 수를 읽어 반환 — gpgpusim.config의 -gpgpu_n_shader 옵션

  /*
   * [한국어]
   * num_devices — 이 노드부터 끝까지 연결 리스트에 있는 디바이스 총 수 반환
   *
   * @return: 현재 노드 포함 이후 모든 디바이스의 개수 (재귀 합산).
   *
   * cudaGetDeviceCount()가 이 함수를 통해 시뮬레이션 대상 GPU 개수를 파악한다.
   * 재귀적으로 리스트 끝까지 탐색하므로 디바이스 수가 많을 때 O(n) 시간이 걸리지만,
   * 초기화 단계 1회 호출이므로 성능 문제는 없다.
   *
   * 호출 체인:
   *   cudaGetDeviceCount() → [num_devices()] → m_next->num_devices() (재귀)
   */
  int num_devices() const {
    if (m_next == NULL) // [한국어] 리스트의 마지막 노드이면 자기 자신 1개만 반환
      return 1;
    else
      return 1 + m_next->num_devices(); // [한국어] 다음 노드부터 재귀적으로 개수를 더함
  }

  /*
   * [한국어]
   * get_device — n번째 디바이스 노드 반환
   *
   * @n: 0-based 디바이스 인덱스 (cudaSetDevice()에서 전달되는 값).
   * @return: n번째 _cuda_device_id 노드 포인터.
   *
   * cudaSetDevice(n) 호출 시 해당 인덱스의 디바이스 노드를 찾는 데 사용된다.
   * n이 전체 디바이스 수 이상이면 assert로 즉시 종료한다.
   * 순차 링크 탐색(O(n))이나 시뮬레이션에서 디바이스 수는 수 개 이하이므로 무방하다.
   * 실행 컨텍스트: 호스트, cudaSetDevice() 경로에서 1회 호출.
   *
   * 호출 체인:
   *   cudaSetDevice(n) → [get_device(n)] → 연결 리스트 순회 → 해당 노드 반환
   */
  struct _cuda_device_id *get_device(unsigned n) {
    assert(n < (unsigned)num_devices()); // [한국어] 범위 초과 접근 방지 — 유효하지 않은 디바이스 인덱스 즉시 오류
    struct _cuda_device_id *p = this;    // [한국어] 현재 노드(헤드)에서 탐색 시작
    for (unsigned i = 0; i < n; i++) p = p->m_next; // [한국어] n번만큼 다음 노드로 전진
    return p; // [한국어] n번째 노드 반환
  }

  /*
   * [한국어]
   * get_prop — 이 디바이스의 CUDA 속성 구조체 포인터 반환
   *
   * @return: cudaDeviceProp 구조체 상수 포인터 (SM 수, 메모리 크기, 클럭 등).
   *
   * cudaGetDeviceProperties() API가 반환할 속성 정보를 제공한다.
   * gpgpu_sim::get_prop()에 위임하며, 내부적으로 gpgpusim.config 설정값으로
   * 채워진 cudaDeviceProp 구조체를 가리킨다.
   *
   * 호출 체인:
   *   cudaGetDeviceProperties() → [get_prop()] → gpgpu_sim::get_prop()
   */
  const struct cudaDeviceProp *get_prop() const { return m_gpgpu->get_prop(); }
  // [한국어] 타이밍 모델에서 시뮬레이션 설정 기반으로 생성된 CUDA 디바이스 속성 반환

  /*
   * [한국어]
   * get_id — 이 디바이스의 CUDA 디바이스 번호(인덱스) 반환
   *
   * @return: 0-based 디바이스 ID (m_id 필드 값).
   *
   * 현재 구현에서는 항상 0; 멀티-GPU 확장 시 각 노드마다 다른 ID 부여 예정.
   *
   * 호출 체인:
   *   각종 디바이스 쿼리 API → [get_id()] → m_id 반환
   */
  unsigned get_id() const { return m_id; }
  // [한국어] 현재는 생성자에서 0으로 고정; 멀티-GPU 시 노드별로 다른 값을 가져야 함

  /*
   * [한국어]
   * get_gpgpu — 이 디바이스를 시뮬레이션하는 gpgpu_sim 포인터 반환
   *
   * @return: 타이밍 모델 최상위 gpgpu_sim 인스턴스 포인터 (비-const).
   *
   * 커널 실행, 메모리 할당 등 시뮬레이터 내부 동작이 필요할 때 타이밍 모델에
   * 직접 접근하기 위해 사용한다. 이 포인터로 gpgpu_sim::cycle() 구동 가능.
   * 실행 컨텍스트: 호스트 스레드에서 호출; 반환된 포인터를 통한 접근도 호스트 전용.
   *
   * 호출 체인:
   *   cuda_runtime_api 각 함수 → [get_gpgpu()] → gpgpu_sim 메서드 직접 호출
   */
  gpgpu_sim *get_gpgpu() { return m_gpgpu; }
  // [한국어] 타이밍 시뮬레이터 최상위 객체 반환 — 메모리 접근, 커널 실행 등에 사용

 private:
  unsigned m_id;
  /* 이 디바이스의 CUDA 디바이스 인덱스 (0-based).
   * 설정자: 생성자에서 0으로 초기화; 현재 멀티-GPU 지원 시 수동 설정 필요.
   * 읽는 자: get_id()가 cudaGetDevice() 응답에 사용.
   * 값 범위: 0 이상 정수; 단일 GPU 환경에서는 항상 0.
   * 동기화: 초기화 후 불변이므로 별도 동기화 불필요. */

  class gpgpu_sim *m_gpgpu;
  /* 이 디바이스 노드가 래핑하는 타이밍 모델 최상위 인스턴스.
   * 설정자: 생성자에서 매개변수로 받은 포인터를 저장.
   * 읽는 자: get_gpgpu(), num_shader(), get_prop() 등 전달 함수들.
   * 값 범위: 유효한 gpgpu_sim 포인터 (NULL 불가; 생성자 보장).
   * 동기화: 포인터 자체는 불변; gpgpu_sim 내부 상태 변경은 gpgpu_sim 자체가 관리. */

  struct _cuda_device_id *m_next;
  /* 연결 리스트에서 다음 디바이스 노드를 가리키는 포인터.
   * 설정자: 멀티-GPU 구성 시 두 번째 디바이스 노드를 연결할 때 설정.
   *          단일 GPU 환경에서는 생성자가 NULL로 초기화.
   * 읽는 자: next(), num_devices(), get_device() 순회 로직.
   * 값 범위: 유효한 _cuda_device_id 포인터 또는 NULL (리스트 끝).
   * 동기화: 초기화 단계 이후 불변으로 취급; 런타임에 변경 없음. */
};

/*
 * [한국어] CUctx_st — CUDA 컨텍스트 (Context) 구현체
 *
 * CUDA 컨텍스트는 GPU와 호스트 애플리케이션 사이의 연결 상태를 나타낸다.
 * 이 구조체는 시뮬레이터에서 컨텍스트의 역할 — fat binary 심볼 테이블 등록,
 * 호스트 함수 포인터 → PTX function_info 매핑 — 을 담당한다.
 * CUDA 애플리케이션이 __cudaRegisterFatBinary()와 __cudaRegisterFunction()을
 * 호출하면 그 결과가 이 구조체의 m_code와 m_kernel_lookup에 기록된다.
 * 이후 cudaLaunchKernel()이 호출되면 m_kernel_lookup으로 function_info를 찾아
 * PTX 시뮬레이션을 시작한다.
 */
struct CUctx_st {
  /*
   * [한국어]
   * CUctx_st 생성자 — CUDA 컨텍스트를 특정 디바이스에 바인딩하여 초기화
   *
   * @gpu: 이 컨텍스트가 연결될 디바이스 래퍼 포인터.
   * @return: (생성자, 반환값 없음)
   *
   * 컨텍스트를 디바이스에 바인딩하고, binary info(상수/전역 메모리 크기) 및
   * PTX 모듈 카운터를 초기 상태(0)로 설정한다.
   * cuCreateContext() 또는 초기화 경로에서 단 1회 생성된다.
   * 실행 컨텍스트: 호스트, 시뮬레이터 기동 초기화 단계.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint.cc 초기화 → [CUctx_st(gpu)] → 멤버 초기화
   */
  CUctx_st(_cuda_device_id *gpu) {
    m_gpu = gpu;              // [한국어] 이 컨텍스트가 실행될 디바이스 노드 저장
    m_binary_info.cmem = 0;   // [한국어] 상수 메모리 사용량 초기화 (아직 binary 로드 전)
    m_binary_info.gmem = 0;   // [한국어] 전역 메모리 사용량 초기화
    no_of_ptx = 0;            // [한국어] 로드된 PTX 모듈 수를 0으로 초기화
  }

  /*
   * [한국어]
   * get_device — 이 컨텍스트가 바인딩된 디바이스 노드 반환
   *
   * @return: 이 컨텍스트와 연결된 _cuda_device_id 포인터.
   *
   * 컨텍스트에서 디바이스 정보(SM 수, 속성 등)가 필요할 때 사용한다.
   *
   * 호출 체인:
   *   각 CUDA API 구현 → [get_device()] → _cuda_device_id 메서드 호출
   */
  _cuda_device_id *get_device() { return m_gpu; }
  // [한국어] private m_gpu 필드에 대한 접근자; 디바이스 속성 조회에 사용

  /*
   * [한국어]
   * add_binary — fat binary의 PTX 심볼 테이블을 컨텍스트에 등록
   *
   * @symtab: cuobjdump로 파싱된 PTX의 심볼 테이블 포인터.
   * @fat_cubin_handle: __cudaRegisterFatBinary()가 반환한 fat binary 핸들 정수.
   * @return: 없음 (void).
   *
   * __cudaRegisterFatBinary() 콜백에서 호출되며, fat binary ID와 심볼 테이블을
   * m_code 맵에 삽입한다. 이후 register_function()이나 add_ptxinfo()가 이
   * 핸들을 키로 심볼을 찾는다.
   * m_last_fat_cubin_handle을 갱신하여 직후의 add_ptxinfo() 호출이 올바른
   * 심볼 테이블을 참조하도록 한다.
   * 실행 컨텍스트: 호스트, 정적 초기화 단계 (__cudaRegisterFatBinary 경로).
   *
   * 호출 체인:
   *   __cudaRegisterFatBinary() → [add_binary(symtab, handle)] → m_code[handle] = symtab
   */
  void add_binary(symbol_table *symtab, unsigned fat_cubin_handle) {
    m_code[fat_cubin_handle] = symtab;            // [한국어] fat binary 핸들을 키로 심볼 테이블 등록
    m_last_fat_cubin_handle = fat_cubin_handle;   // [한국어] 마지막 등록 핸들을 기억하여 add_ptxinfo가 참조 가능하게 함
  }

  /*
   * [한국어]
   * add_ptxinfo (커널별 오버로드) — 특정 디바이스 함수에 PTX 시뮬레이션 메타데이터 연결
   *
   * @deviceFun: PTX 커널 함수의 이름 문자열 (예: "_Z6kernelPf").
   * @info: 해당 커널의 레지스터 수, 공유 메모리 크기 등 시뮬레이션 파라미터.
   * @return: 없음 (void).
   *
   * __cudaRegisterFunction() 이후 호출되며, PTX 파싱 단계에서 얻은 커널별 정보를
   * function_info 객체에 연결한다. 마지막으로 등록된 fat binary의 심볼 테이블에서
   * deviceFun을 조회하여 function_info::set_kernel_info()로 메타데이터를 저장한다.
   * 실행 컨텍스트: 호스트, 정적 초기화 단계.
   *
   * 호출 체인:
   *   PTX 파싱 후 → [add_ptxinfo(deviceFun, info)] → function_info::set_kernel_info()
   */
  void add_ptxinfo(const char *deviceFun,
                   const struct gpgpu_ptx_sim_info &info) {
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun); // [한국어] 마지막 등록 fat binary의 심볼 테이블에서 커널 심볼 검색
    assert(s != NULL);          // [한국어] 커널 이름이 심볼 테이블에 없으면 프로그램 오류 — PTX 파싱 실패
    function_info *f = s->get_pc(); // [한국어] 심볼로부터 PTX IR function_info 객체 획득
    assert(f != NULL);          // [한국어] function_info가 없으면 PTX 파싱이 완료되지 않은 상태
    f->set_kernel_info(info);   // [한국어] 레지스터 수, 공유 메모리 요구량 등 시뮬 메타데이터를 function_info에 저장
  }

  /*
   * [한국어]
   * add_ptxinfo (바이너리 전체 오버로드) — fat binary 수준의 메모리 정보 저장
   *
   * @info: 전체 fat binary의 상수/전역 메모리 사용량 정보.
   * @return: 없음 (void).
   *
   * 개별 커널이 아닌 fat binary 전체의 메모리 통계(상수/전역 메모리 총 사용량)를
   * m_binary_info에 저장한다. load_static_globals()와 load_constants()가
   * 이 값을 참조하여 디바이스 메모리 초기화를 수행한다.
   * 실행 컨텍스트: 호스트, 정적 초기화 단계.
   *
   * 호출 체인:
   *   cuobjdump 파싱 완료 후 → [add_ptxinfo(info)] → m_binary_info 갱신
   */
  void add_ptxinfo(const struct gpgpu_ptx_sim_info &info) {
    m_binary_info = info; // [한국어] 바이너리 전체의 메모리 크기 정보를 컨텍스트 레벨에 저장
  }

  /*
   * [한국어]
   * register_function — 호스트 함수 포인터와 디바이스 커널을 연결하여 룩업 테이블에 등록
   *
   * @fat_cubin_handle: 이 커널이 속한 fat binary의 핸들.
   * @hostFun: CUDA 런타임이 사용하는 호스트 측 스텁 함수 포인터 (void*로 취급).
   *            __cudaRegisterFunction()의 첫 번째 인수로 전달됨.
   * @deviceFun: PTX 커널의 맹글링된 이름 문자열 (예: "_Z6kernelPf").
   * @return: 없음 (void).
   *
   * CUDA 런타임의 __cudaRegisterFunction() 메커니즘을 시뮬레이터에서 구현한다.
   * 호스트 코드에서 커널을 호출할 때 사용하는 호스트 함수 포인터(hostFun)를 키로
   * 해당 PTX 커널의 function_info 포인터를 m_kernel_lookup에 등록한다.
   * fat_cubin_handle이 m_code에 없거나 deviceFun을 찾지 못하면 NULL을 등록하고
   * 경고를 출력하여 런타임 assert 대신 지연된 오류를 허용한다.
   * 실행 컨텍스트: 호스트, 정적 초기화 단계 (CUDA 모듈 로드 시).
   *
   * 호출 체인:
   *   __cudaRegisterFunction() → [register_function(handle, hostFun, deviceFun)]
   *   → m_kernel_lookup[hostFun] = function_info*
   */
  void register_function(unsigned fat_cubin_handle, const char *hostFun,
                         const char *deviceFun) {
    if (m_code.find(fat_cubin_handle) != m_code.end()) { // [한국어] 해당 fat binary 핸들이 이미 등록되어 있는지 확인
      symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun); // [한국어] fat binary의 심볼 테이블에서 deviceFun 이름으로 PTX 심볼 검색
      if (s != NULL) {                    // [한국어] 심볼 테이블에서 커널을 찾은 경우
        function_info *f = s->get_pc();   // [한국어] PTX 심볼로부터 function_info 객체 획득
        assert(f != NULL);                // [한국어] function_info가 없으면 PTX IR 구성 오류
        m_kernel_lookup[hostFun] = f;     // [한국어] 호스트 스텁 포인터 → PTX function_info 매핑 등록
      } else {
        printf("Warning: cannot find deviceFun %s\n", deviceFun); // [한국어] 커널 이름을 심볼 테이블에서 찾지 못함 — PTX 파싱 불완전 가능성
        m_kernel_lookup[hostFun] = NULL;  // [한국어] 찾지 못한 경우 NULL로 등록; get_kernel() 호출 시 오류 발생 예정
      }
      //		assert( s != NULL );
      //		function_info *f = s->get_pc();
      //		assert( f != NULL );
      //		m_kernel_lookup[hostFun] = f;
    } else {
      m_kernel_lookup[hostFun] = NULL; // [한국어] fat binary 자체가 없으면 NULL 등록 (후속 호출에서 오류 감지)
    }
  }

  /*
   * [한국어]
   * register_hostFun_function — 호스트 함수 포인터를 function_info에 직접 연결
   *
   * @hostFun: 호스트 측 커널 스텁 포인터 (CUDA 런타임 내부 식별자).
   * @f: 연결할 PTX function_info 포인터.
   * @return: 없음 (void).
   *
   * register_function()의 단순화 버전으로, 심볼 테이블 조회 없이 function_info를
   * 직접 등록한다. Accel-Sim 트레이스 기반 실행이나 외부 주입 경로에서 사용된다.
   * 실행 컨텍스트: 호스트, 초기화 단계.
   *
   * 호출 체인:
   *   트레이스 기반 초기화 경로 → [register_hostFun_function(hostFun, f)]
   *   → m_kernel_lookup[hostFun] = f
   */
  void register_hostFun_function(const char *hostFun, function_info *f) {
    m_kernel_lookup[hostFun] = f; // [한국어] 심볼 테이블 탐색 없이 hostFun → function_info 직접 등록
  }

  /*
   * [한국어]
   * get_kernel — 호스트 함수 포인터로 PTX function_info 룩업
   *
   * @hostFun: CUDA 애플리케이션에서 커널 호출에 사용한 호스트 스텁 포인터.
   * @return: 대응하는 function_info 포인터 (PTX IR + 커널 메타데이터).
   *           등록되지 않은 hostFun이면 assert로 즉시 종료.
   *
   * cudaLaunchKernel()이 커널 실행 전에 이 함수를 호출하여 PTX 표현을 얻는다.
   * 반환된 function_info는 gpgpu_cuda_ptx_sim_init_grid()로 전달되어
   * kernel_info_t를 생성하고 시뮬레이션이 시작된다.
   * assert로 인해 미등록 커널은 시뮬레이터를 즉시 종료시킨다.
   * 실행 컨텍스트: 호스트, cudaLaunchKernel() 호출 경로.
   *
   * 호출 체인:
   *   cudaLaunchKernel() → [get_kernel(hostFun)] → m_kernel_lookup 조회
   *   → function_info* → gpgpu_cuda_ptx_sim_init_grid()
   */
  function_info *get_kernel(const char *hostFun) {
    std::map<const void *, function_info *>::iterator i =
        m_kernel_lookup.find(hostFun); // [한국어] 호스트 함수 포인터를 키로 룩업 테이블 탐색
    assert(i != m_kernel_lookup.end()); // [한국어] 등록되지 않은 커널이면 즉시 중단 — register_function()이 먼저 호출되었어야 함
    return i->second; // [한국어] 대응하는 PTX function_info 반환 (NULL일 수 있음 — 등록 시 경고 출력된 경우)
  }

  int no_of_ptx;
  /* 이 컨텍스트에 로드된 PTX 모듈의 수.
   * 설정자: PTX 파일 로드 시 증가시키는 카운터.
   * 읽는 자: 초기화 완료 여부 판단 또는 디버그 출력에 사용.
   * 값 범위: 0 이상 정수; fat binary 수에 비례.
   * 동기화: 단일 호스트 스레드에서만 접근하므로 별도 동기화 불필요. */

 private:
  _cuda_device_id *m_gpu;  // selected gpu
  /* 이 컨텍스트가 연결된 GPU 디바이스 노드 포인터.
   * 설정자: 생성자에서 매개변수로 받은 _cuda_device_id 포인터를 저장.
   * 읽는 자: get_device()가 외부에 노출; 내부적으로 gpgpu_sim에 접근할 때 사용.
   * 값 범위: 유효한 _cuda_device_id 포인터 (NULL 불가; 컨텍스트 생명주기 동안 보장).
   * 동기화: 초기화 후 불변; 읽기 전용으로 취급. */

  std::map<unsigned, symbol_table *>
      m_code;  // fat binary handle => global symbol table
  /* fat binary 핸들 정수 → 해당 fat binary에서 파싱된 PTX 심볼 테이블 매핑.
   * 설정자: add_binary()가 __cudaRegisterFatBinary()마다 한 항목씩 삽입.
   * 읽는 자: add_ptxinfo(), register_function(), get_kernel() 등 모든 심볼 조회 함수.
   * 값 범위: 키는 cuobjdump가 부여한 핸들 정수; 값은 파싱 완료된 symbol_table 포인터.
   * 동기화: 정적 초기화 단계에서 순차적으로 채워지고 이후 읽기 전용으로 사용. */

  unsigned m_last_fat_cubin_handle;
  /* 가장 최근에 add_binary()로 등록된 fat binary의 핸들.
   * 설정자: add_binary()가 매번 갱신.
   * 읽는 자: add_ptxinfo(deviceFun, info) 오버로드가 심볼 조회 시 키로 사용.
   * 값 범위: m_code에 존재하는 유효한 핸들 정수.
   * 동기화: 정적 초기화 단계 단일 스레드에서만 접근. */

  std::map<const void *, function_info *>
      m_kernel_lookup;  // unique id (CUDA app function address) => kernel entry
                        // point
  /* 호스트 커널 스텁 포인터(const void*) → PTX function_info 매핑 테이블.
   * CUDA 런타임은 각 커널 함수에 대해 고유한 호스트 측 주소(스텁)를 생성하며,
   * 이 주소가 커널 식별자로 사용된다.
   * 설정자: register_function() 및 register_hostFun_function()이 항목 삽입.
   * 읽는 자: get_kernel()이 cudaLaunchKernel() 경로에서 function_info를 조회.
   * 값 범위: 키는 CUDA 앱 링크 시 결정된 함수 주소; 값은 function_info* 또는 NULL.
   * 동기화: 초기화 단계에서 쓰기, 실행 단계에서 읽기 전용으로 분리되므로 락 불필요. */

  struct gpgpu_ptx_sim_info m_binary_info;
  /* fat binary 전체 수준의 메모리 사용량 요약 (상수 메모리, 전역 메모리 크기).
   * 설정자: add_ptxinfo(info) 오버로드 (바이너리 전체 버전)에서 저장.
   *          생성자에서 cmem=0, gmem=0으로 초기화.
   * 읽는 자: load_static_globals(), load_constants()가 디바이스 메모리 초기화 시 참조.
   * 값 범위: cuobjdump 파싱 결과로 얻은 바이트 크기.
   * 동기화: 정적 초기화 후 불변; 이후 읽기 전용. */
};

/*
 * [한국어] kernel_config — 하나의 커널 실행에 필요한 설정 캡슐화
 *
 * cudaConfigureCall() / cudaSetupArgument() / cudaLaunchKernel()로 이어지는
 * CUDA 커널 실행 설정을 하나의 객체에 담는다.
 * 설정이 완료된 kernel_config는 g_cuda_launch_stack에 push되고,
 * cudaLaunchKernel() 시점에 pop되어 gpgpu_cuda_ptx_sim_init_grid()로 전달된다.
 * 이 클래스는 기능 시뮬레이션(cuda-sim)으로 넘어가기 전 마지막 설정 보관소다.
 */
class kernel_config {
 public:
  /*
   * [한국어]
   * kernel_config 생성자 (완전 초기화) — 커널 실행 설정을 명시적 값으로 초기화
   *
   * @GridDim: 실행할 커널의 그리드 차원 (블록 수 × 3차원).
   * @BlockDim: 각 블록의 스레드 차원 (스레드 수 × 3차원).
   * @sharedMem: 이 커널 실행에서 동적으로 할당할 공유 메모리 바이트 수.
   * @stream: 이 커널을 실행할 CUDA 스트림 포인터 (NULL이면 기본 스트림).
   * @return: (생성자, 반환값 없음)
   *
   * cudaConfigureCall()이 이 생성자를 통해 kernel_config를 생성하여
   * g_cuda_launch_stack에 push한다.
   * 실행 컨텍스트: 호스트, cudaConfigureCall() 경로.
   *
   * 호출 체인:
   *   cudaConfigureCall() → [kernel_config(GridDim, BlockDim, sharedMem, stream)]
   *   → g_cuda_launch_stack.push_back()
   */
  kernel_config(dim3 GridDim, dim3 BlockDim, size_t sharedMem,
                struct CUstream_st *stream) {
    m_GridDim = GridDim;     // [한국어] 그리드 차원 (x,y,z 블록 수) 저장
    m_BlockDim = BlockDim;   // [한국어] 블록 차원 (x,y,z 스레드 수) 저장
    m_sharedMem = sharedMem; // [한국어] 동적 공유 메모리 요청량 (바이트) 저장
    m_stream = stream;       // [한국어] 실행 스트림 핸들 저장 (NULL이면 기본 스트림)
  }

  /*
   * [한국어]
   * kernel_config 기본 생성자 — 유효하지 않은 센티넬 값으로 초기화
   *
   * @return: (생성자, 반환값 없음)
   *
   * GridDim과 BlockDim을 (-1,-1,-1)로 설정하여 '미설정' 상태임을 표시한다.
   * 실제 실행 전 반드시 set_grid_dim()/set_block_dim()으로 유효한 값을 설정해야 한다.
   * 일부 코드 경로에서 기본 생성 후 필드별로 값을 채우는 방식을 지원하기 위함.
   *
   * 호출 체인:
   *   일부 초기화 경로 → [kernel_config()] → 이후 set_grid_dim/set_block_dim 호출
   */
  kernel_config() {
    m_GridDim = dim3(-1, -1, -1);  // [한국어] 미초기화 센티넬 값 — dim3 필드를 -1로 설정하여 무효 상태 표시
    m_BlockDim = dim3(-1, -1, -1); // [한국어] 블록 차원도 동일하게 센티넬 값으로 초기화
    m_sharedMem = 0;               // [한국어] 공유 메모리 요구량 0 (동적 shared mem 미사용)
    m_stream = NULL;               // [한국어] 스트림 미설정 (기본 스트림으로 해석됨)
  }

  /*
   * [한국어]
   * set_arg — 커널 인수를 인수 리스트 앞에 추가
   *
   * @arg: 인수 값의 호스트 메모리 포인터 (포인터나 스칼라 등).
   * @size: 인수의 바이트 크기.
   * @offset: 커널 파라미터 블록 내 이 인수의 바이트 오프셋.
   * @return: 없음 (void).
   *
   * cudaSetupArgument()가 각 커널 인수에 대해 이 함수를 호출한다.
   * push_front()를 사용하므로 인수가 역순으로 리스트에 추가된다는 점에 주의.
   * 실제 시뮬레이션 시 gpgpu_ptx_sim_arg_list_t를 처리하는 측에서 offset을 기준으로
   * 파라미터 블록에 배치하므로 순서 역전은 문제가 되지 않는다.
   * 실행 컨텍스트: 호스트, cudaSetupArgument() 경로.
   *
   * 호출 체인:
   *   cudaSetupArgument() → [set_arg(arg, size, offset)] → m_args.push_front()
   */
  void set_arg(const void *arg, size_t size, size_t offset) {
    m_args.push_front(gpgpu_ptx_sim_arg(arg, size, offset));
    // [한국어] 인수를 gpgpu_ptx_sim_arg로 래핑하여 리스트 앞에 삽입
    // push_front 사용으로 인해 cudaSetupArgument 호출 역순으로 저장됨
  }

  /*
   * [한국어]
   * grid_dim — 저장된 그리드 차원 반환
   *
   * @return: dim3 형태의 그리드 차원 (블록 수 x/y/z).
   *
   * 호출 체인:
   *   gpgpu_cuda_ptx_sim_init_grid() → [grid_dim()] → kernel_info_t 생성 파라미터로 전달
   */
  dim3 grid_dim() const { return m_GridDim; }
  // [한국어] 그리드 차원 값을 복사하여 반환 (const이므로 수정 불가)

  /*
   * [한국어]
   * block_dim — 저장된 블록 차원 반환
   *
   * @return: dim3 형태의 블록 차원 (스레드 수 x/y/z).
   *
   * 호출 체인:
   *   gpgpu_cuda_ptx_sim_init_grid() → [block_dim()] → kernel_info_t 생성 파라미터로 전달
   */
  dim3 block_dim() const { return m_BlockDim; }
  // [한국어] 블록 차원 값을 복사하여 반환

  /*
   * [한국어]
   * set_grid_dim — 그리드 차원을 포인터로 전달받아 설정
   *
   * @d: 새로운 그리드 차원을 담은 dim3 포인터.
   * @return: 없음 (void).
   *
   * 기본 생성 후 별도로 그리드 차원을 설정해야 하는 경로에서 사용.
   *
   * 호출 체인:
   *   cudaLaunchKernel 변형 경로 → [set_grid_dim(&d)]
   */
  void set_grid_dim(dim3 *d) { m_GridDim = *d; }
  // [한국어] 포인터 역참조로 dim3 값을 복사하여 m_GridDim에 저장

  /*
   * [한국어]
   * set_block_dim — 블록 차원을 포인터로 전달받아 설정
   *
   * @d: 새로운 블록 차원을 담은 dim3 포인터.
   * @return: 없음 (void).
   *
   * 호출 체인:
   *   cudaLaunchKernel 변형 경로 → [set_block_dim(&d)]
   */
  void set_block_dim(dim3 *d) { m_BlockDim = *d; }
  // [한국어] 포인터 역참조로 dim3 값을 복사하여 m_BlockDim에 저장

  /*
   * [한국어]
   * get_args — 커널 인수 리스트 반환
   *
   * @return: gpgpu_ptx_sim_arg 객체의 리스트 (값 복사 반환).
   *
   * gpgpu_cuda_ptx_sim_init_grid()가 이 리스트로 PTX 커널에 인수를 설정한다.
   * 값 복사 반환이므로 원본 m_args는 변경되지 않는다.
   *
   * 호출 체인:
   *   gpgpu_cuda_ptx_sim_init_grid() → [get_args()] → arg 리스트 순회 → PTX 파라미터 설정
   */
  gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
  // [한국어] 커널 인수 리스트 전체를 값 복사로 반환

  /*
   * [한국어]
   * get_stream — 이 커널을 실행할 CUDA 스트림 핸들 반환
   *
   * @return: CUstream_st 포인터 (NULL이면 기본 스트림 0).
   *
   * 커널을 스트림에 삽입할 때 stream_manager가 이 값으로 대상 스트림을 결정한다.
   * NULL은 CUDA 기본 스트림(동기식)을 의미한다.
   *
   * 호출 체인:
   *   cudaLaunchKernel() → [get_stream()] → stream_manager에 스트림 지정
   */
  struct CUstream_st *get_stream() {
    return m_stream; // [한국어] 커널 실행에 사용할 스트림 핸들 반환 (NULL == 기본 스트림)
  }

 private:
  dim3 m_GridDim;
  /* 커널 그리드의 차원 (x/y/z 방향 블록 수).
   * 설정자: 완전 초기화 생성자 또는 set_grid_dim()으로 설정.
   * 읽는 자: grid_dim()을 통해 gpgpu_cuda_ptx_sim_init_grid()에 전달.
   * 값 범위: 각 축 1 이상의 양수; 기본 생성자에서는 -1 (센티넬).
   * 동기화: kernel_config는 단일 호스트 스레드에서만 사용. */

  dim3 m_BlockDim;
  /* 각 블록의 스레드 차원 (x/y/z 방향 스레드 수).
   * 설정자: 완전 초기화 생성자 또는 set_block_dim()으로 설정.
   * 읽는 자: block_dim()을 통해 gpgpu_cuda_ptx_sim_init_grid()에 전달.
   *           warp 수 계산 (블록 전체 스레드 수 / 32)에도 사용됨.
   * 값 범위: x × y × z ≤ GPU 최대 스레드/블록; 기본 생성자에서는 -1 (센티넬).
   * 동기화: 단일 호스트 스레드 전용. */

  size_t m_sharedMem;
  /* 이 커널 실행에서 동적으로 할당할 공유 메모리 바이트 수.
   * 설정자: 완전 초기화 생성자로 설정; 기본 생성자에서는 0.
   * 읽는 자: gpgpu_cuda_ptx_sim_init_grid()가 kernel_info_t 생성 시 사용.
   *           SM의 공유 메모리 자원 배분 계산에 영향.
   * 값 범위: 0 이상 정수; SM당 공유 메모리 상한(gpgpusim.config 설정) 이하.
   * 동기화: 불변값; 설정 후 읽기 전용. */

  struct CUstream_st *m_stream;
  /* 이 커널이 실행될 CUDA 스트림의 핸들 포인터.
   * 설정자: 완전 초기화 생성자로 설정; 기본 생성자에서는 NULL.
   * 읽는 자: get_stream()을 통해 stream_manager에 전달되어 실행 순서 결정에 사용.
   * 값 범위: 유효한 CUstream_st 포인터 또는 NULL (기본 스트림).
   * 동기화: 스트림 객체 자체의 동기화는 stream_manager가 관리. */

  gpgpu_ptx_sim_arg_list_t m_args;
  /* 이 커널 실행에 전달할 인수 객체들의 리스트.
   * 설정자: set_arg()가 각 인수를 push_front()로 삽입 (역순 누적).
   * 읽는 자: get_args()를 통해 gpgpu_cuda_ptx_sim_init_grid()에 전달.
   * 값 범위: 커널 파라미터 수만큼의 gpgpu_ptx_sim_arg 원소.
   * 동기화: 설정 완료 후 읽기 전용으로 사용. */
};

/*
 * [한국어] cuda_runtime_api — CUDA 런타임 API 전체 구현 최상위 클래스
 *
 * libcuda 계층에서 CUDA 애플리케이션이 호출하는 모든 CUDA 런타임 API
 * (cudaMalloc, cudaMemcpy, cudaLaunchKernel 등)의 실제 구현을 담는다.
 * 전역 시뮬레이터 상태 — 할당된 메모리 추적(g_mallocPtr_Size), 커널 실행 스택
 * (g_cuda_launch_stack), PTX 섹션 목록, 핀드 메모리 맵 등 — 을 소유하고 관리한다.
 * gpgpu_context를 통해 타이밍 모델(gpgpu_sim)에 접근하며,
 * cuobjdump 파이프라인을 통해 CUDA 바이너리에서 PTX를 추출하고 로드한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스; GPU 사이클 시뮬레이션은 별도 스레드
 * (gpgpusim_entrypoint.cc의 시뮬레이션 스레드)에서 수행된다.
 */
class cuda_runtime_api {
 public:
  /*
   * [한국어]
   * cuda_runtime_api 생성자 — 전역 API 상태 초기화
   *
   * @ctx: 시뮬레이터 전역 문맥(gpgpu_context) 포인터; gpgpu_sim 접근 경로.
   * @return: (생성자, 반환값 없음)
   *
   * libcuda 초기화 시 1회 생성되며, OpenGL 맵 포인터와 활성 디바이스 인덱스를
   * 초기 상태로 설정한다. g_cuda_launch_stack, g_mallocPtr_Size 등 STL 컨테이너는
   * 기본 생성자로 자동 초기화된다.
   * 실행 컨텍스트: 호스트, 시뮬레이터 기동 최초 1회.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint::init() → [cuda_runtime_api(ctx)] → 멤버 초기화
   */
  cuda_runtime_api(gpgpu_context *ctx) {
    g_glbmap = NULL;         // [한국어] OpenGL 버퍼 맵 연결 리스트 헤드를 NULL로 초기화 (등록된 GL 버퍼 없음)
    g_active_device = 0;     // active gpu that runs the code
    // [한국어] 활성 디바이스 인덱스를 0으로 초기화 — 단일 GPU 환경에서 유일한 디바이스
    gpgpu_ctx = ctx;         // [한국어] 시뮬레이터 전역 문맥 포인터 저장 — gpgpu_sim 접근 경로
  }

  // global list

  std::list<cuobjdumpSection *> cuobjdumpSectionList;
  /* cuobjdump로 파싱한 애플리케이션 fat binary의 섹션(PTX/ELF) 객체 포인터 리스트.
   * 설정자: extract_code_using_cuobjdump_internal()이 cuobjdump 실행 후 섹션 파싱 결과를 삽입.
   * 읽는 자: pruneSectionList(), mergeMatchingSections() 등 섹션 필터링 함수.
   *           findPTXSection()이 SM 버전에 맞는 PTX 섹션을 찾는 데 사용.
   * 값 범위: fat binary 내 섹션 수만큼의 포인터; ELF 섹션과 PTX 섹션이 혼재.
   * 동기화: 초기화 단계에서 채워지고 이후 읽기 전용. */

  std::list<cuobjdumpSection *> libSectionList;
  /* 링크된 라이브러리 fat binary에서 추출한 섹션 리스트.
   * 설정자: 라이브러리(cuBLAS, cuDNN 등) fat binary 파싱 경로에서 삽입.
   * 읽는 자: mergeSections()가 애플리케이션 섹션과 라이브러리 섹션을 합칠 때 참조.
   * 값 범위: 라이브러리 fat binary 섹션 수만큼; 없으면 빈 리스트.
   * 동기화: 초기화 단계 이후 불변. */

  std::list<kernel_config> g_cuda_launch_stack;
  /* CUDA 커널 실행 설정(kernel_config)의 스택 역할을 하는 리스트.
   * cudaConfigureCall() → cudaSetupArgument() → cudaLaunchKernel() 순서에서
   * 앞 두 단계가 kernel_config를 누적하고 마지막 단계가 pop하여 실행한다.
   * 설정자: cudaConfigureCall()이 새 kernel_config를 push_back.
   *          cudaSetupArgument()가 맨 뒤 원소의 set_arg()를 호출.
   * 읽는 자: cudaLaunchKernel()이 back()으로 최신 설정을 꺼내 시뮬레이션 시작.
   * 값 범위: 동시에 설정 중인 커널 수만큼의 원소 (보통 1개).
   * 동기화: 단일 호스트 스레드에서만 접근. */

  std::map<int, bool> fatbin_registered;
  /* fat binary 핸들 → 등록 완료 여부 맵.
   * 설정자: cuobjdumpRegisterFatBinary()가 등록 완료 시 true로 설정.
   * 읽는 자: 중복 등록 방지 체크에 사용.
   * 값 범위: 키는 fat binary 핸들 정수; 값은 true/false.
   * 동기화: 초기화 단계 단일 스레드 접근. */

  std::map<int, std::string> fatbinmap;
  /* fat binary 핸들 → fat binary 파일 경로 문자열 맵.
   * 설정자: cuobjdumpRegisterFatBinary()가 파일명과 핸들을 연결하여 저장.
   * 읽는 자: extract_code_using_cuobjdump_internal()이 cuobjdump 실행 시 파일 경로 조회.
   * 값 범위: 키는 핸들 정수; 값은 절대 또는 상대 파일 경로.
   * 동기화: 초기화 단계 이후 불변. */

  std::map<std::string, symbol_table *> name_symtab;
  /* PTX 파일 이름 → 파싱된 symbol_table 포인터 맵.
   * 설정자: PTX 파일 파싱 완료 후 파일명을 키로 삽입.
   * 읽는 자: 동일 PTX 파일의 재파싱을 피하기 위한 캐시로 사용.
   * 값 범위: PTX 파일 수만큼의 항목; 값은 heap 할당된 symbol_table 포인터.
   * 동기화: 초기화 단계 순차 접근. */

  std::map<unsigned long long, size_t> g_mallocPtr_Size;
  /* cudaMalloc()으로 할당된 디바이스 메모리 추적 맵.
   * 키: 할당된 디바이스 포인터(주소); 값: 해당 할당의 바이트 크기.
   * 설정자: cudaMalloc() 성공 시 새 항목 삽입; cudaFree() 시 항목 삭제.
   * 읽는 자: cudaFree() 유효성 검사, 메모리 사용량 통계 출력.
   * 값 범위: 키는 시뮬레이터 디바이스 주소 공간 내 포인터; 값은 1 이상 바이트.
   * 동기화: 호스트-시뮬레이터 스레드 동기화 후 단일 스레드 접근. */

  // maps sm version number to set of filenames
  std::map<unsigned, std::set<std::string> > version_filename;
  /* GPU SM 버전 번호 → 해당 버전용 PTX 파일명 집합 맵.
   * cuobjdump가 fat binary에서 여러 SM 아키텍처(sm_70, sm_75 등)의 PTX를
   * 추출할 때 각 버전별로 파일명을 분류하여 저장한다.
   * 설정자: extract_ptx_files_using_cuobjdump_internal()이 PTX 파일 추출 시 삽입.
   * 읽는 자: 시뮬레이션 대상 SM 버전에 맞는 PTX 파일 선택 시 참조.
   * 값 범위: 키는 SM 버전 정수 (예: 70=sm_70); 값은 파일 경로 문자열 집합.
   * 동기화: 초기화 단계 이후 불변. */

  std::map<void *, void **> pinned_memory;  // support for pinned memories added
  /* cudaMallocHost()로 할당된 호스트 핀드 메모리 포인터 → 내부 포인터 맵.
   * 핀드 메모리(페이지 잠금 메모리)는 DMA 전송 시 물리 주소가 고정되어야 하므로
   * 시뮬레이터 내부에서 별도 추적이 필요하다.
   * 설정자: cudaMallocHost() 성공 시 항목 삽입; cudaFreeHost() 시 항목 삭제.
   * 읽는 자: cudaMemcpy() 등이 소스/대상이 핀드 메모리인지 판별하는 데 사용.
   * 값 범위: 키는 호스트 가상 주소; 값은 내부 포인터 배열.
   * 동기화: 호스트 단일 스레드 접근. */

  std::map<void *, size_t> pinned_memory_size;
  /* 핀드 메모리 포인터 → 할당 바이트 크기 맵.
   * 설정자: cudaMallocHost()가 할당 시 크기와 함께 삽입.
   * 읽는 자: cudaFreeHost() 시 올바른 크기로 해제하기 위해 참조.
   * 값 범위: 키는 핀드 메모리 호스트 주소; 값은 1 이상 바이트.
   * 동기화: 호스트 단일 스레드 접근. */

  glbmap_entry_t *g_glbmap;
  /* OpenGL 버퍼 ↔ CUDA 디바이스 포인터 매핑 연결 리스트의 헤드 포인터.
   * 설정자: cudaGLRegisterBufferObject() 래퍼가 새 노드를 앞에 삽입;
   *          cudaGLUnregisterBufferObject()가 해당 노드를 삭제.
   * 읽는 자: cudaGLMapBufferObject()가 버퍼 ID로 리스트를 순회하여 노드를 찾음.
   * 값 범위: NULL(GL 버퍼 없음) 또는 유효한 glbmap_entry_t 포인터.
   * 동기화: 호스트 단일 스레드에서 관리. */

  int g_active_device;  // active gpu that runs the code
  /* 현재 활성화된 GPU 디바이스의 인덱스 (0-based).
   * cudaSetDevice(n)이 이 값을 변경하며, 이후 모든 CUDA 작업이 이 디바이스를 대상으로 한다.
   * 설정자: cudaSetDevice()가 n을 저장; 생성자에서 0으로 초기화.
   * 읽는 자: 디바이스 종속적인 모든 CUDA API가 g_active_device로 _cuda_device_id 노드를 조회.
   * 값 범위: 0 이상 정수; 시뮬레이션 중인 총 GPU 수 미만.
   * 동기화: 단일 호스트 스레드에서만 변경되므로 별도 락 불필요. */

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* 시뮬레이터 전역 문맥에 대한 역참조 포인터.
   * gpgpu_context는 gpgpu_sim 인스턴스와 기타 전역 설정을 소유한다.
   * 이 포인터를 통해 cuda_runtime_api가 타이밍 모델(gpgpu_sim)에 접근한다.
   * 설정자: 생성자 매개변수로 전달받아 저장.
   * 읽는 자: cuobjdump 초기화, 커널 실행, 메모리 관리 등 대부분의 API 구현 함수.
   * 값 범위: 유효한 gpgpu_context 포인터 (NULL 불가).
   * 동기화: 초기화 후 불변; 내부 상태 변경은 gpgpu_context/gpgpu_sim이 담당. */

  // member function list

  // For SST and other potential simulator interface

  /*
   * [한국어]
   * cuobjdumpInit(fn) — 외부 바이너리 파일을 지정하여 cuobjdump 파이프라인 초기화
   *
   * @fn: PTX를 추출할 CUDA 바이너리 파일 경로 문자열.
   * @return: 없음 (void).
   *
   * SST(Structural Simulation Toolkit) 등 외부 시뮬레이터 인터페이스에서 사용하는
   * 오버로드. 특정 파일 경로를 받아 cuobjdump를 실행하고 PTX 섹션을 파싱한다.
   * 내부적으로 cuobjdumpInit_internal()을 호출하여 공통 로직을 공유한다.
   *
   * 호출 체인:
   *   SST 인터페이스 → [cuobjdumpInit(fn)] → cuobjdumpInit_internal()
   *   → extract_code_using_cuobjdump(fn)
   */
  void cuobjdumpInit(const char *fn);

  /*
   * [한국어]
   * extract_code_using_cuobjdump(fn) — 지정된 CUDA 바이너리에서 PTX/ELF 추출
   *
   * @fn: cuobjdump로 분석할 CUDA 바이너리 파일 경로.
   * @return: 없음 (void).
   *
   * cuobjdump 도구를 실행하여 CUDA 바이너리 내 fat binary 섹션들을 추출하고
   * cuobjdumpSectionList에 파싱 결과를 저장한다.
   * SST 경로 전용 오버로드; 바이너리 경로를 명시적으로 받는다.
   *
   * 호출 체인:
   *   cuobjdumpInit(fn) → [extract_code_using_cuobjdump(fn)]
   *   → cuobjdumpSectionList 채움
   */
  void extract_code_using_cuobjdump(const char *fn);

  /*
   * [한국어]
   * extract_ptx_files_using_cuobjdump(context, fn) — PTX 파일을 추출하여 컨텍스트에 로드
   *
   * @context: 추출한 PTX 심볼 테이블을 등록할 CUDA 컨텍스트.
   * @fn: 원본 CUDA 바이너리 파일 경로.
   * @return: 없음 (void).
   *
   * 추출된 PTX 섹션에서 실제 .ptx 파일을 생성하거나 기존 파일을 찾아
   * PTX 파서를 실행하고, 파싱된 symbol_table을 context의 add_binary()로 등록한다.
   * SST 경로 전용 오버로드.
   *
   * 호출 체인:
   *   cuobjdumpInit(fn) → [extract_ptx_files_using_cuobjdump(context, fn)]
   *   → PTX 파서 → context->add_binary()
   */
  void extract_ptx_files_using_cuobjdump(CUctx_st *context, const char *fn);

  // For running GPGPUSim alone

  /*
   * [한국어]
   * cuobjdumpInit() — GPGPU-Sim 단독 실행 시 cuobjdump 파이프라인 초기화
   *
   * @return: 없음 (void).
   *
   * SST 없이 GPGPU-Sim만 단독 실행할 때 호출되는 오버로드.
   * 실행 중인 CUDA 바이너리 경로를 /proc/self/exe 등으로 자동 감지하여
   * cuobjdumpInit_internal()에 전달한다.
   *
   * 호출 체인:
   *   gpgpusim_entrypoint::init() → [cuobjdumpInit()] → cuobjdumpInit_internal()
   */
  void cuobjdumpInit();

  /*
   * [한국어]
   * extract_code_using_cuobjdump() — 현재 실행 바이너리에서 PTX/ELF 추출
   *
   * @return: 없음 (void).
   *
   * 단독 실행 경로의 오버로드; 현재 프로세스의 바이너리 파일을 자동으로 대상으로 삼아
   * cuobjdump를 실행하고 섹션을 파싱한다.
   *
   * 호출 체인:
   *   cuobjdumpInit() → [extract_code_using_cuobjdump()] → cuobjdumpSectionList 채움
   */
  void extract_code_using_cuobjdump();

  /*
   * [한국어]
   * extract_ptx_files_using_cuobjdump(context) — 단독 실행 경로의 PTX 로드
   *
   * @context: PTX 심볼 테이블을 등록할 CUDA 컨텍스트.
   * @return: 없음 (void).
   *
   * 단독 실행 경로의 오버로드; 바이너리 경로를 자동으로 판단하여 PTX 파일을 추출하고
   * 파싱 후 context에 등록한다.
   *
   * 호출 체인:
   *   cuobjdumpInit() → [extract_ptx_files_using_cuobjdump(context)]
   *   → PTX 파서 → context->add_binary()
   */
  void extract_ptx_files_using_cuobjdump(CUctx_st *context);

  // Internal functions for the above public methods

  /*
   * [한국어]
   * cuobjdumpInit_internal — cuobjdump 초기화의 공통 내부 구현
   *
   * @ctx_extract_code_func: cuobjdump 실행 후 섹션 추출을 수행할 콜백 함수.
   *                          SST/단독 경로별로 다른 바이너리 경로 처리 로직을 람다로 전달.
   * @return: 없음 (void).
   *
   * 외부 공개 cuobjdumpInit() 오버로드들의 공통 로직을 구현한다.
   * 템플릿 메서드 패턴처럼 콜백으로 경로별 차이를 주입받는다.
   * 이 함수가 PTX 로드 시퀀스의 실질적 진입점이다.
   *
   * 호출 체인:
   *   cuobjdumpInit() 또는 cuobjdumpInit(fn) → [cuobjdumpInit_internal(callback)]
   *   → callback() → extract_ptx_files_using_cuobjdump_internal()
   */
  void cuobjdumpInit_internal(std::function<void()> ctx_extract_code_func);

  /*
   * [한국어]
   * extract_code_using_cuobjdump_internal — PTX/ELF 추출의 공통 내부 구현
   *
   * @context: PTX를 등록할 CUDA 컨텍스트.
   * @app_binary: 분석 대상 CUDA 바이너리 파일 경로 참조.
   * @ctx_extract_ptx_func: PTX 파일 추출 후 context에 로드하는 콜백.
   * @return: 없음 (void).
   *
   * cuobjdump 도구를 실제로 실행하고 stdout을 파싱하여 ELF/PTX 섹션을
   * cuobjdumpSectionList에 채운다. 경로별 다른 처리는 콜백으로 주입된다.
   *
   * 호출 체인:
   *   cuobjdumpInit_internal() → [extract_code_using_cuobjdump_internal()]
   *   → cuobjdump 실행 → 섹션 파싱 → ctx_extract_ptx_func(context)
   */
  void extract_code_using_cuobjdump_internal(
      CUctx_st *context, std::string &app_binary,
      std::function<void(CUctx_st *)> ctx_extract_ptx_func);

  /*
   * [한국어]
   * extract_ptx_files_using_cuobjdump_internal — PTX 파일 추출 및 로드 공통 내부 구현
   *
   * @context: 파싱된 PTX의 symbol_table을 등록할 CUDA 컨텍스트.
   * @app_binary: 원본 CUDA 바이너리 파일 경로 참조.
   * @return: 없음 (void).
   *
   * cuobjdumpSectionList에서 시뮬레이션 대상 SM 버전과 일치하는 PTX 섹션을 선택하고,
   * .ptx 파일을 디스크에 기록하거나 기존 파일을 재사용한 뒤 PTX 파서를 실행한다.
   * 파싱 결과 symbol_table을 context->add_binary()로 등록하고, 정적 글로벌/
   * 상수 변수를 디바이스 메모리에 초기화한다.
   *
   * 호출 체인:
   *   extract_code_using_cuobjdump_internal() → [extract_ptx_files_using_cuobjdump_internal()]
   *   → PTX 파서(cuda-sim) → context->add_binary() → load_static_globals()
   */
  void extract_ptx_files_using_cuobjdump_internal(CUctx_st *context,
                                                  std::string &app_binary);

  /*
   * [한국어]
   * pruneSectionList — 현재 컨텍스트에 필요한 섹션만 추려낸 리스트 반환
   *
   * @context: 현재 실행 컨텍스트 (SM 버전 등 필터 기준 제공).
   * @return: 필터링된 cuobjdumpSection 포인터 리스트.
   *
   * cuobjdumpSectionList에서 시뮬레이션 대상 SM 아키텍처 버전에 맞는 섹션만 선택한다.
   * 하나의 fat binary에 여러 SM 버전의 코드가 포함될 수 있으므로 필터링이 필요하다.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [pruneSectionList(context)]
   *   → SM 버전 필터링 → 결과 리스트 반환
   */
  std::list<cuobjdumpSection *> pruneSectionList(CUctx_st *context);

  /*
   * [한국어]
   * mergeMatchingSections — 동일 식별자를 가진 섹션들을 병합한 리스트 반환
   *
   * @identifier: 병합 대상 섹션을 식별하는 문자열 (커널 이름 또는 모듈 식별자).
   * @return: 해당 식별자에 해당하는 섹션들을 병합한 리스트.
   *
   * 애플리케이션과 라이브러리의 fat binary에 동일한 커널이 중복 포함된 경우
   * 하나로 병합하여 중복 로드를 방지한다.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [mergeMatchingSections(id)]
   */
  std::list<cuobjdumpSection *> mergeMatchingSections(std::string identifier);

  /*
   * [한국어]
   * mergeSections — 모든 섹션(앱 + 라이브러리)을 병합한 최종 리스트 반환
   *
   * @return: cuobjdumpSectionList와 libSectionList를 합친 전체 섹션 리스트.
   *
   * PTX 로드 전 최종 단계에서 모든 소스의 섹션을 하나의 리스트로 통합한다.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [mergeSections()]
   *   → 통합 섹션 리스트 → pruneSectionList()
   */
  std::list<cuobjdumpSection *> mergeSections();

  /*
   * [한국어]
   * findELFSection — 식별자로 ELF 섹션 검색
   *
   * @identifier: ELF 섹션 식별 문자열.
   * @return: 매칭되는 cuobjdumpELFSection 포인터; 없으면 NULL.
   *
   * cuobjdumpSectionList에서 ELF 타입이며 주어진 식별자와 일치하는 섹션을 찾는다.
   * ELF 섹션은 GPU 커널의 기계어 코드를 포함하지만, 시뮬레이터는 PTX를 사용한다.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [findELFSection(id)]
   */
  cuobjdumpELFSection *findELFSection(const std::string identifier);

  /*
   * [한국어]
   * findPTXSection — 식별자로 PTX 섹션 검색 (cuobjdumpSectionList 전체 대상)
   *
   * @identifier: PTX 섹션 식별 문자열 (커널 이름 또는 SM 버전 식별자).
   * @return: 매칭되는 cuobjdumpPTXSection 포인터; 없으면 NULL.
   *
   * cuobjdumpSectionList에서 주어진 식별자와 일치하는 PTX 섹션을 찾는다.
   * 시뮬레이션 대상 SM 버전에 맞는 PTX를 선택하는 데 핵심 함수다.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [findPTXSection(id)]
   */
  cuobjdumpPTXSection *findPTXSection(const std::string identifier);

  /*
   * [한국어]
   * findPTXSectionInList — 지정된 섹션 리스트에서 PTX 섹션 검색
   *
   * @sectionlist: 검색 대상 섹션 리스트 참조.
   * @identifier: 검색할 PTX 섹션 식별 문자열.
   * @return: 매칭되는 cuobjdumpPTXSection 포인터; 없으면 NULL.
   *
   * findPTXSection()의 리스트 일반화 버전으로, cuobjdumpSectionList 이외의
   * 임의 섹션 리스트(병합 결과 등)에서도 PTX 섹션을 찾을 수 있다.
   *
   * 호출 체인:
   *   findPTXSection() 또는 pruneSectionList() → [findPTXSectionInList(list, id)]
   */
  cuobjdumpPTXSection *findPTXSectionInList(
      std::list<cuobjdumpSection *> &sectionlist, const std::string identifier);

  /*
   * [한국어]
   * cuobjdumpRegisterFatBinary — fat binary를 컨텍스트에 등록하고 PTX 로드
   *
   * @handle: __cudaRegisterFatBinary()가 부여한 fat binary 핸들.
   * @filename: 이 fat binary에 해당하는 파일 경로.
   * @context: PTX 심볼 테이블을 등록할 CUDA 컨텍스트.
   * @return: 없음 (void).
   *
   * CUDA 런타임의 __cudaRegisterFatBinary() 콜백에서 호출되는 핵심 함수.
   * fat binary를 fatbinmap에 등록하고, cuobjdump를 통해 PTX를 추출하여
   * context에 add_binary()로 등록한다.
   * 이미 등록된 핸들이면 fatbin_registered 맵으로 중복 작업을 건너뛴다.
   * 실행 컨텍스트: 호스트, CUDA 정적 초기화 단계.
   *
   * 호출 체인:
   *   __cudaRegisterFatBinary() → [cuobjdumpRegisterFatBinary(handle, fn, ctx)]
   *   → extract_ptx_files_using_cuobjdump_internal() → context->add_binary()
   */
  void cuobjdumpRegisterFatBinary(unsigned int handle, const char *filename,
                                  CUctx_st *context);

  /*
   * [한국어]
   * gpgpu_cuda_ptx_sim_init_grid — 커널 실행 구성을 kernel_info_t로 변환하여 시뮬레이션 준비
   *
   * @kernel_key: 호스트 커널 함수 포인터 (문자열로 캐스팅된 void 포인터).
   *              m_kernel_lookup의 키로 사용하여 function_info를 조회.
   * @args: 커널에 전달할 인수 리스트 (kernel_config::get_args() 결과).
   * @gridDim: 커널 그리드 차원 (블록 수).
   * @blockDim: 커널 블록 차원 (스레드 수).
   * @context: 현재 CUDA 컨텍스트 (function_info 조회 및 디바이스 정보 제공).
   * @return: 생성된 kernel_info_t 포인터; 시뮬레이션 시작 후 gpgpu_sim::launch()로 전달됨.
   *
   * cudaLaunchKernel()의 핵심 내부 함수. context->get_kernel()로 function_info를
   * 조회하고, grid/block 차원과 인수를 묶어 kernel_info_t를 생성한다.
   * 생성된 kernel_info_t는 gpgpu_sim::launch()로 전달되어 실제 사이클 시뮬레이션이 시작된다.
   * 실행 컨텍스트: 호스트, cudaLaunchKernel() 경로.
   *
   * 호출 체인:
   *   cudaLaunchKernel() → [gpgpu_cuda_ptx_sim_init_grid(key, args, grid, block, ctx)]
   *   → context->get_kernel() → kernel_info_t 생성 → gpgpu_sim::launch()
   */
  kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(const char *kernel_key,
                                              gpgpu_ptx_sim_arg_list_t args,
                                              struct dim3 gridDim,
                                              struct dim3 blockDim,
                                              struct CUctx_st *context);

  /*
   * [한국어]
   * load_static_globals — PTX 심볼 테이블의 정적 전역 변수를 디바이스 메모리에 초기화
   *
   * @symtab: 정적 전역 변수 심볼을 포함하는 PTX 심볼 테이블.
   * @min_gaddr: 전역 메모리 영역의 시작 주소 (할당 가능한 최소 주소).
   * @max_gaddr: 전역 메모리 영역의 끝 주소 (할당 가능한 최대 주소).
   * @gpu: 디바이스 메모리에 접근하기 위한 gpgpu_t 포인터.
   * @return: 성공 시 0; 오류 시 음수.
   *
   * CUDA 모듈의 __device__ 전역 변수들을 디바이스 메모리의 지정된 범위에 배치하고
   * 초기값이 있는 경우 GPU 메모리에 복사한다. PTX 로드 완료 직후 호출된다.
   * 실행 컨텍스트: 호스트, cuobjdumpRegisterFatBinary() 경로.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [load_static_globals()]
   *   → symtab 순회 → gpgpu_t::memcpy_to_gpu()
   */
  int load_static_globals(symbol_table *symtab, unsigned min_gaddr,
                          unsigned max_gaddr, gpgpu_t *gpu);

  /*
   * [한국어]
   * load_constants — PTX 심볼 테이블의 상수 변수를 디바이스 상수 메모리에 초기화
   *
   * @symtab: 상수 변수 심볼을 포함하는 PTX 심볼 테이블.
   * @min_gaddr: 상수 메모리 영역의 시작 주소.
   * @gpu: 상수 메모리에 접근하기 위한 gpgpu_t 포인터.
   * @return: 성공 시 0; 오류 시 음수.
   *
   * CUDA 모듈의 __constant__ 변수들을 GPU 상수 메모리 영역에 복사한다.
   * 상수 메모리는 모든 스레드가 공유하며 캐시 최적화된 읽기 전용 메모리이다.
   * load_static_globals()와 함께 PTX 로드 직후 호출된다.
   * 실행 컨텍스트: 호스트, cuobjdumpRegisterFatBinary() 경로.
   *
   * 호출 체인:
   *   extract_ptx_files_using_cuobjdump_internal() → [load_constants()]
   *   → symtab 순회 → gpgpu_t::get_global_memory()::write()
   */
  int load_constants(symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu);
};
#endif /* __cuda_api_object_h__ */
