/*
 * [한국어 설명] ICNT 네트워크 트래픽 분류 및 통계 헤더 (traffic_breakdown.h)
 *
 * === 파일의 역할 ===
 * GPU 내부 네트워크(ICNT/intersim2)를 통해 전달되는 메모리 요청(mem_fetch)을
 * 트래픽 유형(Global load, L1 write-back 등)과 패킷 크기(byte)별로 분류하고
 * 통계를 집계하는 traffic_breakdown 클래스를 선언한다.
 * AerialVision 시각화 파일 또는 표준 출력에 트래픽 분포를 출력하는 데 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: SM → ICNT(intersim2) → 메모리 파티션 과정에서
 *   mem_fetch가 ICNT에 주입/추출될 때 record_traffic()이 호출됨.
 * mem_fetch의 mem_access_type을 classify_memfetch()로 문자열 분류하고,
 * packet_size()로 바이트 크기를 구한 뒤 m_stats에 누적.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: mem_fetch.h (mem_access_type, READ_REQUEST, WRITE_REQUEST 등),
 *       stdio.h (FILE*), map, string
 * 사용처: traffic_breakdown.cc (구현), icnt_wrapper.cc 또는 shader.cc (record_traffic 호출)
 * 데이터 흐름: mem_fetch → classify_memfetch() → 트래픽 유형 문자열
 *              → m_stats[type][size] += 1 누적 → print()로 출력
 *
 * === 주요 함수/구조체 요약 ===
 * traffic_breakdown()  - 네트워크 이름을 받아 통계 컨테이너 초기화
 * record_traffic()     - mem_fetch 1건 주입 시 유형·크기별 카운터 증가
 * classify_memfetch()  - mem_access_type → 트래픽 유형 문자열 변환
 * packet_size()        - mem_fetch → 패킷 바이트 크기 계산
 * print()              - m_stats 전체를 파일에 출력 (유형별 총 바이트 + 크기 분포)
 */

#pragma once

#include <stdio.h>  /* [한국어] FILE* — print()에서 텍스트 파일 출력용 */
#include <map>      /* [한국어] std::map — 트래픽 유형/크기별 카운터 중첩 맵 */
#include <string>   /* [한국어] std::string — 트래픽 유형 이름 키 */

// Breakdown traffic through the network according to category
/*
 * [한국어]
 * class traffic_breakdown - ICNT 네트워크 트래픽 유형·크기별 통계 수집기
 *
 * 각 network(SM→메모리 또는 메모리→SM 방향)에 대해 1개 인스턴스가 생성된다.
 * mem_fetch가 ICNT에 주입될 때마다 record_traffic()으로 유형과 크기를 분류하여
 * m_stats 이중 맵에 카운터를 누적한다.
 * 시뮬레이션 종료 시 print()로 트래픽 분포를 출력한다.
 */
class traffic_breakdown {
 public:
  /*
   * [한국어]
   * traffic_breakdown() - 트래픽 통계 컨테이너 초기화
   *
   * @network_name: ICNT 네트워크 이름 (예: "Interconnect0")
   *
   * m_network_name을 초기화. m_stats(이중 맵)는 기본 생성자로 빈 상태로 초기화됨.
   * 시뮬레이터 초기화 시 각 ICNT 네트워크 인스턴스마다 1회 호출.
   *
   * 호출 체인:
   *   ICNT 초기화 코드 → [traffic_breakdown(network_name)]
   */
  traffic_breakdown(const std::string& network_name)
      : m_network_name(network_name) {}

  // print the stats
  /*
   * [한국어]
   * print() - 트래픽 통계를 FILE*에 출력
   *
   * @fout: 출력 대상 FILE 포인터 (stdout 또는 리포트 파일)
   *
   * m_stats를 순회하여 각 트래픽 유형의 총 바이트 전송량과
   * 패킷 크기 분포(크기별 패킷 수)를 출력한다.
   * 시뮬레이션 종료 후 gpgpu_sim::print_stats()에서 호출.
   *
   * 호출 체인:
   *   gpgpu_sim::print_stats() → [print(fout)]
   */
  void print(FILE* fout);

  // record the amount and type of traffic introduced by this mem_fetch object
  /*
   * [한국어]
   * record_traffic() - mem_fetch 1건 전송 시 트래픽 통계 기록
   *
   * @mf:   전송되는 메모리 요청 객체 (유형 분류에 사용)
   * @size: 실제 전송 패킷 크기 (바이트, packet_size()와 별개로 외부에서 전달)
   *
   * classify_memfetch(mf)로 트래픽 유형 문자열 결정,
   * m_stats[type][size] += 1로 카운터 누적.
   * ICNT 주입 지점에서 메모리 요청마다 호출.
   *
   * 호출 체인:
   *   ICNT 주입 코드 (icnt_push/shader.cc) → [record_traffic(mf, size)]
   */
  void record_traffic(class mem_fetch* mf, unsigned int size);

 protected:
  std::string m_network_name;
  /* [한국어] 이 통계가 속한 ICNT 네트워크 이름 (식별용).
   * 설정자: 생성자에서 1회 초기화.
   * 읽는 자: print()에서 출력 헤더에 사용.
   * 값 범위: 빈 문자열 불가, 네트워크 식별 문자열.
   * 동기화: 생성 후 변경 없음. */

  /// helper functions to identify the type of traffic sent
  /*
   * [한국어]
   * classify_memfetch() - mem_fetch의 메모리 접근 유형을 문자열로 변환
   *
   * @mf:     분류할 메모리 요청 객체
   * @return: 트래픽 유형 문자열 (예: "GlobalLD", "L1WB", "CONST", "TEXTURE" 등)
   *
   * mem_fetch의 mem_access_type을 switch/if로 분류.
   * GLOBAL_ATOMIC의 경우 READ_REQUEST/WRITE_REQUEST에 따라 구분.
   * record_traffic()에서 m_stats 키 생성에 사용.
   *
   * 호출 체인:
   *   record_traffic() → [classify_memfetch(mf)]
   */
  std::string classify_memfetch(class mem_fetch* mf);

  /// helper functions to identify the size of traffic sent
  /*
   * [한국어]
   * packet_size() - mem_fetch의 패킷 크기 계산
   *
   * @mf:     크기를 계산할 메모리 요청 객체
   * @return: 전송 패킷 크기 (바이트)
   *
   * mem_fetch의 요청 크기(get_data_size() 등)를 기반으로 ICNT 패킷 크기 계산.
   * record_traffic()에서 size 인자 결정에 활용 가능.
   *
   * 호출 체인:
   *   record_traffic() 또는 ICNT 주입 코드 → [packet_size(mf)]
   */
  unsigned int packet_size(class mem_fetch* mf);

  typedef std::string
      mf_packet_type;  // use string so that it remains extensible
  /* [한국어] 트래픽 유형 이름 타입 별칭. std::string을 사용하여 새 유형 추가가 용이.
   * 예: "GlobalLD", "GlobalST", "L1WB", "CONST", "TEXTURE" 등 */

  typedef unsigned int mf_packet_size;
  /* [한국어] 패킷 크기 타입 별칭 (바이트 단위 unsigned int). m_stats의 내부 맵 키. */

  typedef std::map<mf_packet_size, unsigned int> traffic_class_t;
  /* [한국어] 특정 트래픽 유형의 패킷 크기 분포 맵: 크기(바이트) → 패킷 수.
   * 예: { 32: 1500, 128: 300 } → 32바이트 1500건, 128바이트 300건. */

  typedef std::map<mf_packet_type, traffic_class_t> traffic_stat_t;
  /* [한국어] 트래픽 통계 최상위 맵: 유형 문자열 → 크기 분포 맵.
   * 예: { "GlobalLD": {32: 1500}, "L1WB": {128: 300} }. */

  traffic_stat_t m_stats;
  /* [한국어] 트래픽 유형·크기별 패킷 카운터 이중 맵 (핵심 통계 자료구조).
   * 설정자: record_traffic()에서 m_stats[type][size] += 1로 누적.
   * 읽는 자: print()에서 전체 순회하여 출력.
   * 값 범위: 유형별 크기별 카운터, [0, ∞) unsigned int.
   * 동기화: 단일 시뮬레이션 스레드 전용 — 별도 락 불필요. */
};
