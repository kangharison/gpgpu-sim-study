// a Wraper function for stats class

/*
 * [한국어 설명] intersim2 Stats 클래스 C 래퍼 구현 (statwrapper.cc)
 *
 * === 파일의 역할 ===
 * intersim2/stats.hpp의 C++ Stats 클래스를 C 스타일 void* 인터페이스로 감싸는
 * 함수들을 구현한다. void* 캐스팅을 통해 Stats 객체를 불투명 포인터(opaque pointer)로
 * 처리하며, 호출자가 C++ 헤더를 직접 포함하지 않아도 통계 기능을 사용할 수 있게 한다.
 * 파일 하단에 #if 0으로 비활성화된 테스트 main()이 남아 있어 개발 의도를 확인할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: 시뮬레이션 모듈(gpgpu-sim/) → StatCreate/AddSample → Stats 객체 내부 누적
 *   → 시뮬레이션 종료 후 StatDisp()로 결과 출력
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내 임의 시점.
 *
 * === 타 모듈과의 연결 ===
 * 의존: intersim2/stats.hpp (Stats 클래스 정의), stdio.h (printf)
 * 인터페이스: statwrapper.h 함수 선언을 구현
 * 데이터 흐름: Stats 객체는 힙에 동적 할당되며 void*로 관리.
 *   메모리 해제 없이 사용 — Stats 객체는 시뮬레이션 종료 시까지 유지.
 *
 * === 주요 함수/구조체 요약 ===
 * StatCreate()     - Stats(NULL, name, bin_size, num_bins) 생성 후 Clear()
 * StatDisp()       - DisplayHierarchy() + Min/Max/Average + Display() 출력
 * StatAddSample()  - Stats::AddSample(val) 위임
 * StatAverage/Max/Min() - Stats의 대응 메서드에 위임
 */

#include <stdio.h>               /* [한국어] printf 표준 출력용 */
#include "intersim2/stats.hpp"   /* [한국어] Stats C++ 클래스 정의 — NoC 시뮬레이터 통계 */

/*
 * [한국어]
 * StatCreate() - Stats 통계 객체 동적 생성 및 초기화
 *
 * @name: 통계 객체 이름 (DisplayHierarchy() 출력 시 사용)
 * @bin_size: 히스토그램 bin 하나의 값 범위 크기
 * @num_bins: 히스토그램 bin 개수
 * @return: 초기화된 Stats 객체의 void* 포인터
 *
 * Stats(NULL, name, bin_size, num_bins): 첫 인자 NULL은 부모 Stats 없음(독립 객체).
 * Clear()로 bin 카운터를 0으로 초기화 후 반환.
 * 반환된 포인터는 호출자가 관리. 명시적 해제 없이 사용.
 * 실행 컨텍스트: 시뮬레이터 초기화 시 1회 호출.
 *
 * 호출 체인:
 *   시뮬레이션 모듈 → [StatCreate()] → new Stats() → Stats::Clear()
 */
Stats *StatCreate(const char *name, double bin_size, int num_bins) {
  Stats *newstat = new Stats(NULL, name, bin_size, num_bins);
  /* [한국어] Stats 객체를 힙에 동적 할당. NULL=부모없음, 이름/bin크기/bin수 설정. */
  newstat->Clear(); /* [한국어] 모든 bin 카운터를 0으로 초기화 */
  return newstat;   /* [한국어] void*로 반환 — 호출자는 타입 캐스팅 없이 opaque 포인터로 관리 */
}

/*
 * [한국어]
 * StatClear() - 통계 데이터 초기화
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 *
 * Stats::Clear()를 호출하여 누적된 모든 샘플을 제거하고 초기 상태로 복원.
 * 인터벌 기반 통계 수집 시 각 인터벌 시작 시 호출.
 */
void StatClear(void *st) { ((Stats *)st)->Clear(); }
/* [한국어] void*를 Stats*로 캐스팅 후 Clear() 호출 */

/*
 * [한국어]
 * StatAddSample() - 통계에 샘플값 추가
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @val: 추가할 정수 샘플 값
 *
 * Stats::AddSample(val)에 위임. bin_size에 따라 val이 적절한 bin에 분류되어 누적.
 * 시뮬레이션 사이클마다 호출될 수 있어 빈번한 경로에 위치.
 */
void StatAddSample(void *st, int val) { ((Stats *)st)->AddSample(val); }
/* [한국어] void*를 Stats*로 캐스팅 후 AddSample() 호출 */

/*
 * [한국어]
 * StatAverage() - 수집된 샘플의 평균값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플의 평균값 (double). 샘플이 없으면 0.
 */
double StatAverage(void *st) { return ((Stats *)st)->Average(); }
/* [한국어] void*를 Stats*로 캐스팅 후 Average() 호출 */

/*
 * [한국어]
 * StatMax() - 수집된 샘플의 최대값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플 중 최대값 (double)
 */
double StatMax(void *st) { return ((Stats *)st)->Max(); }
/* [한국어] void*를 Stats*로 캐스팅 후 Max() 호출 */

/*
 * [한국어]
 * StatMin() - 수집된 샘플의 최소값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플 중 최소값 (double)
 */
double StatMin(void *st) { return ((Stats *)st)->Min(); }
/* [한국어] void*를 Stats*로 캐스팅 후 Min() 호출 */

/*
 * [한국어]
 * StatDisp() - 통계 내용을 stdout에 출력
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 *
 * 세 가지 출력을 순서대로 수행:
 *   1. DisplayHierarchy(): 통계 이름(계층 구조 포함) 출력
 *   2. printf(): Min/Max/Average 수치 한 줄로 출력
 *   3. Display(): bin별 분포 히스토그램 출력
 * 시뮬레이션 종료 후 결과 리포팅 또는 진단 목적으로 사용.
 * 실행 컨텍스트: 시뮬레이션 완료 후 결과 출력 단계.
 *
 * 호출 체인:
 *   시뮬레이션 종료 후 결과 출력 → [StatDisp()] → Stats::DisplayHierarchy/Display
 */
void StatDisp(void *st) {
  printf("Stats for "); /* [한국어] 통계 이름 출력 전 접두어 */
  ((Stats *)st)->DisplayHierarchy();
  /* [한국어] Stats 객체의 이름과 계층 구조를 stdout에 출력 */
  //   if (((Stats *)st)->NeverUsed()) {
  //      printf (" was never updated!\n");
  //   } else {
  printf("Min %f Max %f Average %f \n", ((Stats *)st)->Min(),
         ((Stats *)st)->Max(), StatAverage(st));
  /* [한국어] Min/Max/Average 값을 부동소수점으로 한 줄에 출력 */
  ((Stats *)st)->Display();
  /* [한국어] bin별 샘플 분포 히스토그램을 stdout에 출력 */
  //   }
}

#if 0
/* [한국어] 단위 테스트용 main 함수 — 비활성화(#if 0). 개발 시 StatCreate/AddSample/Disp
 * 동작 확인용으로 작성되었으며, 실제 빌드에서는 컴파일되지 않음. */
int main ()
{
   void * mytest = StatCreate("Test",1,5);
   StatAddSample(mytest,4);
   StatAddSample(mytest,4);StatAddSample(mytest,4);
   StatAddSample(mytest,2);
   StatDisp(mytest);
}
#endif
