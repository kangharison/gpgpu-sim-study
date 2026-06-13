/*
 * [한국어 설명] intersim2 Stats 클래스 C 래퍼 헤더 (statwrapper.h)
 *
 * === 파일의 역할 ===
 * intersim2/stats.hpp의 C++ Stats 클래스를 void* 기반의 C 스타일 함수로 감싸는
 * 래퍼 인터페이스를 선언한다. C++ 클래스를 직접 포함하기 어려운 모듈에서 통계 수집
 * 기능을 사용할 수 있도록 한다. bin 크기와 bin 수를 지정하여 히스토그램 형태의
 * 통계를 수집하며, 평균/최대/최소값을 조회하고 표준 출력으로 내용을 출력할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: 타이밍 시뮬레이션 모듈 → StatCreate() → StatAddSample() → StatDisp()
 * intersim2(NoC 시뮬레이터)의 Stats 클래스를 래핑하므로 NoC 계층 위의 통계 수집에 사용.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내 임의 시점.
 *
 * === 타 모듈과의 연결 ===
 * 의존: intersim2/stats.hpp (Stats 클래스 구현)
 * 사용처: GPGPU-Sim의 다양한 시뮬레이션 모듈에서 통계 수집용으로 사용
 * 데이터 흐름: StatCreate() → Stats 객체 생성(힙) → StatAddSample() 누적
 *   → StatAverage/Max/Min 조회 또는 StatDisp() 출력
 *
 * === 주요 함수/구조체 요약 ===
 * StatCreate()     - Stats 객체를 힙에 생성하고 초기화, void* 포인터 반환
 * StatAddSample()  - 정수 샘플값을 통계에 추가
 * StatDisp()       - Min/Max/Average와 분포 히스토그램을 stdout에 출력
 * StatAverage/Max/Min() - 지금까지 수집된 통계의 통계값 반환
 */

#ifndef STAT_WRAPER_H
#define STAT_WRAPER_H

/*
 * [한국어]
 * StatCreate() - Stats 통계 객체 생성
 *
 * @name: 통계 이름 문자열 (출력 시 식별자로 사용)
 * @bin_size: 히스토그램 한 bin의 크기 (값 범위 단위)
 * @num_bins: 히스토그램 bin의 총 개수
 * @return: 생성된 Stats 객체의 void* 포인터 (이후 모든 Stat* 함수에 전달)
 *
 * intersim2/stats.hpp의 Stats 클래스를 힙에 동적 생성하고 Clear()로 초기화.
 * 반환된 포인터는 StatClear/AddSample/Average/Max/Min/Disp에 첫 인자로 전달.
 */
class Stats* StatCreate(const char* name, double bin_size, int num_bins);

/*
 * [한국어]
 * StatClear() - 통계 데이터 초기화
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 *
 * 누적된 모든 샘플 데이터를 지우고 통계를 초기 상태로 되돌림.
 */
void StatClear(void* st);

/*
 * [한국어]
 * StatAddSample() - 통계에 샘플값 추가
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @val: 추가할 정수 샘플 값 (bin 크기에 따라 적절한 bin에 분류됨)
 */
void StatAddSample(void* st, int val);

/*
 * [한국어]
 * StatAverage() - 수집된 샘플의 평균값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플의 평균값 (double)
 */
double StatAverage(void* st);

/*
 * [한국어]
 * StatMax() - 수집된 샘플의 최대값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플 중 최대값 (double)
 */
double StatMax(void* st);

/*
 * [한국어]
 * StatMin() - 수집된 샘플의 최소값 반환
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 * @return: 전체 샘플 중 최소값 (double)
 */
double StatMin(void* st);

/*
 * [한국어]
 * StatDisp() - 통계 내용을 stdout에 출력
 *
 * @st: StatCreate()가 반환한 Stats 객체 void* 포인터
 *
 * 통계 이름(DisplayHierarchy), Min/Max/Average값, 분포 히스토그램(Display)을
 * printf로 표준 출력에 출력한다. 시뮬레이션 종료 후 결과 리포팅에 사용.
 */
void StatDisp(void* st);

#endif
